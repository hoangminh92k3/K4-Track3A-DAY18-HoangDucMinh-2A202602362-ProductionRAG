from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def _split_by_chars(text: str, max_chars: int) -> list[str]:
    """Chia text theo độ dài ký tự, giữ nguyên khoảng trắng và không để chunk rỗng."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    parts = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            split = text.rfind(" ", start, end)
            if split > start:
                end = split
        piece = text[start:end].strip()
        if piece:
            parts.append(piece)
        start = end
        while start < len(text) and text[start].isspace():
            start += 1
    return parts


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    metadata = metadata or {}
    text = text.strip()
    if not text:
        return []

    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n\n+', text) if s.strip()]
    if not sentences:
        return []

    chunks: list[Chunk] = []
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(sentences, convert_to_numpy=True)

        current: list[str] = [sentences[0]]
        for i in range(1, len(sentences)):
            prev = embeddings[i - 1]
            curr = embeddings[i]
            prev_norm = np.linalg.norm(prev)
            curr_norm = np.linalg.norm(curr)
            sim = float(np.dot(prev, curr) / (prev_norm * curr_norm + 1e-9))

            if sim < threshold:
                chunks.append(Chunk(
                    text=" ".join(current).strip(),
                    metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"},
                ))
                current = [sentences[i]]
            else:
                current.append(sentences[i])

        chunks.append(Chunk(
            text=" ".join(current).strip(),
            metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"},
        ))
        return [chunk for chunk in chunks if chunk.text.strip()]
    except Exception:
        # Fallback: nếu embedding model không sẵn, dùng split theo câu với cứng logic rời rạc.
        current = [sentences[0]]
        for sentence in sentences[1:]:
            joined = " ".join(current + [sentence])
            if len(joined) > 500 and current:
                chunks.append(Chunk(
                    text=" ".join(current).strip(),
                    metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"},
                ))
                current = [sentence]
            else:
                current.append(sentence)
        if current:
            chunks.append(Chunk(
                text=" ".join(current).strip(),
                metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"},
            ))
        return [chunk for chunk in chunks if chunk.text.strip()]


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    text = text.strip()
    if not text:
        return ([], [])

    paragraphs = [p.strip() for p in re.split(r'\n\s*\n+', text) if p.strip()]
    if not paragraphs:
        return ([], [])

    def _pack_by_size(items: list[str], max_chars: int) -> list[str]:
        packed: list[str] = []
        current = ""
        for item in items:
            if current and len(current) + len(item) + 2 > max_chars:
                packed.append(current.strip())
                current = item
            else:
                current = f"{current}\n\n{item}".strip() if current else item
        if current.strip():
            packed.append(current.strip())
        return packed

    parents: list[Chunk] = []
    for idx, parent_text in enumerate(_pack_by_size(paragraphs, parent_size)):
        pid = f"parent_{idx}"
        parents.append(Chunk(
            text=parent_text,
            metadata={**metadata, "chunk_type": "parent", "parent_id": pid},
            parent_id=pid,
        ))

    children: list[Chunk] = []
    for parent in parents:
        pid = parent.metadata.get("parent_id") or parent.parent_id
        for child_index, child_text in enumerate(_split_by_chars(parent.text, child_size)):
            children.append(Chunk(
                text=child_text,
                metadata={**metadata, "chunk_type": "child", "parent_id": pid, "chunk_index": child_index},
                parent_id=pid,
            ))

    return (parents, children)


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = metadata or {}
    text = text.strip()
    if not text:
        return []

    lines = text.splitlines()
    header_pattern = re.compile(r'^(#{1,3})\s+(.+?)\s*$')
    chunks: list[Chunk] = []
    current_header: str | None = None
    current_lines: list[str] = []

    def flush_section():
        nonlocal current_header, current_lines
        if current_header is None:
            return
        section_text = current_header
        if current_lines:
            section_body = "\n".join(current_lines)
            section_text = f"{section_text}\n{section_body}".strip()
        if section_text.strip():
            header_name = current_header.lstrip('#').strip()
            chunks.append(Chunk(
                text=section_text.strip(),
                metadata={**metadata, "section": header_name, "strategy": "structure"},
            ))
        current_header = None
        current_lines = []

    for line in lines:
        matched = header_pattern.match(line.strip())
        if matched:
            if current_header is not None:
                flush_section()
            current_header = line.strip()
        else:
            if current_header is None:
                current_header = "Document"
            current_lines.append(line.rstrip())

    if current_header is not None:
        flush_section()

    return [chunk for chunk in chunks if chunk.text.strip()]


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
