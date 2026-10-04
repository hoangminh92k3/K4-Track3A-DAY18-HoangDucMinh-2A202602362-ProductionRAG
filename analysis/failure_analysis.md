# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Hoàng Đức Minh  
**Khóa:** K4 - Track 3A

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------:|-----------:|--:|
| Faithfulness | 0.8417 | 0.7158 | -0.1258 |
| Answer Relevancy | 0.7717 | 0.7831 | +0.0114 |
| Context Precision | 0.9250 | 0.9375 | +0.0125 |
| Context Recall | 0.9250 | 0.8167 | -0.1083 |

Evaluation covered 20 questions. Production improved answer relevancy and context precision slightly, but reduced faithfulness and context recall. This suggests that selecting cleaner top results did not ensure the required facts were retrieved or faithfully used.

## Bottom-5 Failures

The saved JSON contains each question, aggregate score, worst metric, and diagnostic, but not the generated answer or retrieved contexts. Therefore **Got** and context-level branches below are explicitly marked unavailable; they should not be treated as observed facts. Expected answers are taken from `test_set.json` and the corresponding policy documents.

### #1 — Password rotation
- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected:** Chính sách v2.0 hiện hành yêu cầu đổi mật khẩu mỗi 120 ngày; v1.0 yêu cầu 90 ngày nhưng đã bị thay thế.
- **Got:** Không được lưu trong report; không thể đối chiếu nội dung trả lời.
- **Worst metric:** Faithfulness = 0.0000; overall failure score = 0.3958.
- **Error Tree:** Output grounded? Không xác minh được. → Context có đúng chính sách v2.0? Không được lưu. → Query rõ ràng? Có. → RAGAS đánh dấu câu trả lời không faithful.
- **Root cause:** Bằng chứng hiện có chỉ xác nhận vấn đề grounding ở answer. Hai phiên bản chính sách cùng tồn tại, nên việc lấy nhầm quy định v1.0 là một giả thuyết cần kiểm tra, chưa thể kết luận nếu thiếu retrieved contexts.
- **Suggested fix:** Gắn version/status/effective date vào metadata; ưu tiên tài liệu hiện hành hoặc lọc tài liệu đã bị thay thế. Log top-k contexts và câu trả lời để xác nhận giả thuyết bằng một test cho chu kỳ 120 ngày.

### #2 — Approval for a 55-million-VND purchase
- **Question:** Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?
- **Expected:** Đơn hàng trên 50.000.000 VNĐ cần Tổng Giám đốc (CEO) phê duyệt.
- **Got:** Không được lưu trong report; không thể đối chiếu nội dung trả lời.
- **Worst metric:** Context recall = 0.0000; overall failure score = 0.6611.
- **Error Tree:** Output đúng? Không xác minh được. → Context có quy tắc đơn hàng trên 50 triệu? RAGAS cho thấy recall bằng 0, tức evidence cần thiết không được thu hồi theo đánh giá. → Query rõ ràng? Có, chứa giá trị và yêu cầu phê duyệt. → Ưu tiên sửa retrieval/evidence coverage.
- **Root cause:** Retrieval không đưa được evidence cần thiết vào context được đánh giá; answer generation không thể bù cho context thiếu.
- **Suggested fix:** Kiểm tra BM25 và dense top-k riêng cho query này, giữ nguyên các mốc tiền trong chunk, rồi đo recall@k trước/sau reranking. Thêm regression test cho ngưỡng >50 triệu và CEO.

### #3 — Senior employee with nine years of tenure
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** Theo v2024: 15 ngày cơ bản + 3 ngày thâm niên = 18 ngày phép; lương Senior (P3-P4) là 20-35 triệu VNĐ/tháng.
- **Got:** Không được lưu trong report; không thể đối chiếu nội dung trả lời.
- **Worst metric:** Faithfulness = 0.4000; overall failure score = 0.6908.
- **Error Tree:** Output grounded? Chỉ đạt faithfulness 0.4. → Context đủ cả phép năm/thâm niên và khung lương? Không được lưu. → Query rõ nhưng cần tổng hợp nhiều facts từ chính sách/bảng lương. → Kiểm tra retrieval coverage cho từng phần trước khi đánh giá cách tính.
- **Root cause:** Câu hỏi nhiều phần, cần kết hợp chính sách phép v2024, phép cộng thâm niên và bảng lương Senior; điểm thấp cho thấy một phần câu trả lời không được evidence hỗ trợ.
- **Suggested fix:** Tách truy vấn thành subquestions (số ngày cơ bản, phép thâm niên, lương Senior), xác nhận mỗi fact có context riêng, và yêu cầu LLM chỉ trả lời kèm evidence cho từng phần.

### #4 — Twenty days of unpaid leave
- **Question:** Nghỉ phép không lương 20 ngày cần ai phê duyệt?
- **Expected:** Nghỉ từ 16-30 ngày cần CEO phê duyệt; nghỉ trên 14 ngày thì nhân viên tự đóng phần bảo hiểm của mình.
- **Got:** Không được lưu trong report; không thể đối chiếu nội dung trả lời.
- **Worst metric:** Faithfulness = 0.5000; overall failure score = 0.6959.
- **Error Tree:** Output grounded? Chỉ đạt faithfulness 0.5. → Context chứa đúng khoảng 16-30 ngày và cấp phê duyệt? Không được lưu. → Query rõ ràng? Có. → Xác định lỗi ở context hay generation cần log từng bước.
- **Root cause:** Câu hỏi có ngưỡng ngày cụ thể; câu trả lời bị đánh giá thiếu grounding nhưng report không lưu context để phân biệt nhầm ngưỡng với lỗi diễn đạt.
- **Suggested fix:** Bảo toàn các khoảng số trong chunking, kiểm tra context top-k có đoạn 16-30 ngày, và thêm test biên cho 15/16/30 ngày.

### #5 — Maximum Junior probation salary
- **Question:** Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?
- **Expected:** Junior tối đa 20.000.000 VNĐ/tháng; lương thử việc là 85%, tức 17.000.000 VNĐ/tháng.
- **Got:** Không được lưu trong report; không thể đối chiếu nội dung trả lời.
- **Worst metric:** Faithfulness = 0.0000; overall failure score = 0.7060.
- **Error Tree:** Output grounded? RAGAS cho faithfulness 0. → Context có cả mức tối đa Junior và tỷ lệ thử việc 85%? Không được lưu. → Query rõ ràng nhưng cần phép tính từ hai facts. → Kiểm tra evidence rồi kiểm tra phép nhân 0.85 × 20 triệu.
- **Root cause:** Câu hỏi cần hợp nhất bảng lương và chính sách thử việc; có thể thiếu một trong hai facts hoặc câu trả lời không dựa trên context. Không thể phân biệt hai khả năng từ report hiện tại.
- **Suggested fix:** Retrieve cả bảng lương 2024 và chính sách thử việc; đưa phép tính vào answer prompt hoặc tính deterministic trong code, rồi thêm test kỳ vọng 17.000.000 VNĐ.

## Case Study (cho presentation)

**Question chọn phân tích:** Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?

**Error Tree walkthrough:**
1. Output đúng? Không thể xác minh vì answer không nằm trong report JSON.
2. Context đúng? Context recall = 0.0000; evidence cần thiết không được thu hồi theo kết quả RAGAS.
3. Query rewrite OK? Query gốc rõ ràng, có số tiền và ý định. Không có log query rewrite để kiểm tra thêm.
4. Fix ở bước retrieval: kiểm tra riêng BM25/dense candidates, giữ quy tắc ngưỡng trong cùng chunk, rồi xác nhận quy tắc CEO xuất hiện trong top-k trước reranking.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Thêm log per-question cho answer, top-k contexts trước/sau reranking và metric từng câu để Error Tree có bằng chứng trực tiếp.
- Chạy lại các câu bottom-5 sau khi thêm metadata cho phiên bản tài liệu và test retrieval cho các ngưỡng số.
- So sánh recall@k trước/sau thay đổi; không thay nhiều module cùng lúc để biết nguyên nhân của metric change.
