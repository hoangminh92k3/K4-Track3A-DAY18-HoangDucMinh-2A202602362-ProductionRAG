# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Hoàng Đức Minh  
**Khóa:** K4 - Track 3A  
**Ngày hoàn thành:** 04/10/2026

## Phần 1: Mapping bài giảng (Lecture Mapping)

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| Semantic và hierarchical chunking | M1 | `chunk_semantic()`, `chunk_hierarchical()` | Baseline tạo 57 paragraph chunks; production hierarchical tạo 100 child chunks từ 26 tài liệu. Đây là hai cấu hình pipeline khác nhau, không phải phép đo riêng về semantic chunking. |
| BM25 + Dense fusion | M2 | `BM25Search.search()`, `DenseSearch.search()`, `reciprocal_rank_fusion()` | Production dùng hybrid retrieval. Context precision đạt 0.9375, nhưng recall chỉ 0.8167; câu hỏi mua thiết bị 55 triệu có context recall 0.0, nên cần kiểm tra từng nhánh retrieval và coverage của candidate set. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Reranker được load thành công và dùng trước answer generation. Log không có latency riêng hoặc benchmark đối chứng, nên chưa thể kết luận mức cải thiện do reranking. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()`, `failure_analysis()` | Trên 20 câu, faithfulness/relevancy/precision/recall lần lượt là 0.7158/0.7831/0.9375/0.8167. Precision cao nhưng faithfulness và recall giảm so với baseline, cho thấy cần kiểm tra grounding và evidence coverage theo từng câu. |
| Contextual embeddings và enrichment | M5 | `contextual_prepend()`, `_enrich_single_call()`, `enrich_chunks()` | Pipeline xử lý 100 chunks bằng combined mode. Enrichment mất 376.8 giây; một số phản hồi gây lỗi parse JSON và fallback được dùng. Cần kiểm tra định dạng output và lưu version/status tài liệu để tránh lẫn chính sách cũ. |

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

- **Lỗi kỹ thuật gặp phải (Exact error message):** `Expecting value: line 1 column 1 (char 0)` trong các bước enrichment/metadata.
- **Nguyên nhân gốc rễ & Cách debug:** Lỗi xuất hiện khi parse phản hồi enrichment thành JSON. Log không ghi nội dung phản hồi thô, vì vậy chưa thể xác định phản hồi rỗng hay JSON không hợp lệ. Pipeline tiếp tục nhờ fallback và hoàn tất việc indexing/evaluation. Bước debug tiếp theo là log an toàn phần phản hồi lỗi, kiểm tra chế độ JSON của model/API, và thêm test cho phản hồi rỗng/không hợp lệ.
- **Vấn đề dữ liệu:** Hai PDF scan không có text layer nên bị bỏ qua; muốn đưa nội dung vào retrieval cần OCR. Hugging Face Hub cũng cảnh báo request không có token, dù model weights vẫn tải được.
- **Kiến thức còn thiếu & Cách khắc phục:** Cần hiểu rõ cách kiểm soát structured output, logging per-question trong RAG evaluation và cách đo riêng retrieval recall trước reranking. Bổ sung regression tests cho version policy, các ngưỡng số và câu hỏi cần tổng hợp nhiều nguồn.

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: RAG tra cứu chính sách nhân sự nội bộ

#### 1. Hiện trạng
- **Pipeline hiện tại:** Chunk tài liệu → enrichment → BM25 + dense hybrid search → cross-encoder reranking → sinh câu trả lời → đánh giá bằng RAGAS.
- **Vấn đề / Bottlenecks đang gặp:** Production có context precision tốt (0.9375) nhưng context recall (0.8167) và faithfulness (0.7158) thấp hơn baseline; một số enrichment output không parse được JSON; tài liệu chính sách có phiên bản cũ và mới.

#### 2. Kế hoạch cải tiến
1. **Chunking strategy:** Duy trì hierarchical chunking cho retrieval theo đoạn nhỏ, đồng thời thử structure-aware chunking để giữ tiêu đề và phạm vi từng quy định.
2. **Search retrieval:** Giữ BM25 + dense + RRF; đo recall@k riêng cho từng nhánh và cho các truy vấn có số tiền, ngày hoặc ngưỡng phê duyệt.
3. **Reranking:** Giữ cross-encoder cho top candidates; đo chất lượng trước/sau reranking và ghi latency thay vì suy luận từ tổng thời gian chạy.
4. **Evaluation:** Tiếp tục dùng bốn metric RAGAS trên cùng test set; bổ sung log answer/context/metric theo từng câu để phân tích lỗi có bằng chứng.
5. **Enrichment:** Dùng combined mode khi structured output ổn định; kiểm tra JSON và fallback có kiểm soát. Đưa version, trạng thái hiệu lực và ngày hiệu lực vào metadata để lọc chính sách đã bị thay thế.

#### 3. Timeline triển khai
- **Tuần 1:** Thêm logging per-question, metadata phiên bản tài liệu và regression tests cho các câu hỏi bottom-5.
- **Tuần 2:** So sánh retrieval recall@k và reranking trên cùng test set; xử lý tài liệu scan bằng OCR nếu cần thiết.
- **Tuần 3:** Rerun RAGAS, so sánh với baseline và cập nhật failure analysis dựa trên answer/context đã lưu.