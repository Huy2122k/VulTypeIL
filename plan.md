1. Adaptive EWC Regularization (Điều chỉnh EWC động)
Mô tả: Thay vì sử dụng lambda cố định cho EWC, điều chỉnh lambda động dựa trên độ tương đồng giữa các tác vụ (task similarity) được tính từ embeddings của data. Điều này giúp cân bằng giữa plasticity (học mới) và stability (giữ kiến thức cũ) tốt hơn, tránh overfitting trên tasks dễ hoặc underfitting trên tasks khó.
Cài đặt:
Tính similarity matrix giữa tasks bằng cosine similarity trên mean embeddings của từng task (sử dụng CodeT5 encoder).
Cập nhật lambda = base_lambda * (1 - similarity_score) trong loss function.
Thêm vào class OnlineEWCWithFocalLabelSmoothLoss để tính toán động.
Đánh giá: So sánh forgetting rate (FR) và average accuracy (ACC) trên 5 tasks so với baseline (lambda cố định). Sử dụng metrics như Backward Transfer (BWT) và Forward Transfer (FWT) để đo lường khả năng học mới mà không quên cũ.
2. Generative Replay với Diffusion Models (Replay tổng hợp)
Mô tả: Thay thế experience replay bằng generative models (như Stable Diffusion hoặc text-to-code models như CodeGen) để tạo synthetic code snippets và descriptions cho các tasks cũ, giảm memory usage và tránh bias từ data thực. Điều này đặc biệt hữu ích cho continual learning trên code, nơi data có thể khan hiếm.
Cài đặt:
Huấn luyện một diffusion model nhỏ (như DiT) trên data của task hiện tại để generate samples (code + description).
Trong replay phase, generate 200 samples và mix với data mới.
Tích hợp vào hàm hybrid_replay bằng cách thay thế selection bằng generation.
Đánh giá: Đo lường FID (Fréchet Inception Distance) cho quality của generated data, và so sánh ACC/F1 trên tasks cũ sau 5 tasks. Kiểm tra robustness với data imbalance bằng cách đo macro-F1.
3. Prompt Tuning kết hợp LoRA (Parameter-Efficient Prompt Learning)
Mô tả: Kết hợp soft prompts với Low-Rank Adaptation (LoRA) để fine-tune chỉ một phần nhỏ parameters của CodeT5, giảm overfitting và cải thiện generalization trong continual learning. Điều này giúp prompts học được patterns chung hơn, không bị overfit trên từng task.
Cài đặt:
Sử dụng OpenPrompt với SoftTemplate và thêm LoRA layers (thư viện PEFT của Hugging Face) vào CodeT5.
Fine-tune chỉ LoRA parameters (rank=8) cùng với prompts trong mỗi task.
Merge LoRA weights sau mỗi task để tích lũy kiến thức.
Đánh giá: So sánh training time và memory usage với baseline (full fine-tune). Đánh giá ACC trên all tasks và forgetting matrix; sử dụng Area Under Curve (AUC) cho stability metrics.
4. Curriculum Learning với Task Ordering (Học theo chương trình giảng dạy)
Mô tả: Sắp xếp tasks theo độ khó tăng dần (dựa trên entropy của labels hoặc complexity của code) để mô hình học từ dễ đến khó, giảm catastrophic forgetting. Điều này tận dụng progressive learning để xây dựng kiến thức nền tảng trước khi học tasks phức tạp.
Cài đặt:
Tính difficulty score cho mỗi task (e.g., average code length + label entropy).
Sắp xếp data_paths theo score tăng dần.
Thêm curriculum scheduling: tăng replay size hoặc lambda theo task index.
Đánh giá: So sánh ACC progression qua tasks với random ordering. Sử dụng Learning Curve Analysis (đồ thị ACC vs. task) và final ACC/F1 trên all tasks. Đo lường improvement trong BWT.
5. Multi-Modal Continual Learning (Đầu vào đa phương thức)
Mô tả: Kết hợp code text với AST (Abstract Syntax Tree) hoặc graph representations (như từ GraphCodeBERT) để cung cấp context phong phú hơn, giúp mô hình hiểu cấu trúc code tốt hơn và giảm forgetting khi tasks có syntax khác nhau.
Cài đặt:
Sử dụng transformers như GraphCodeBERT để encode code thành embeddings đa chiều.
Concat hoặc fuse embeddings (text + graph) trước input vào prompt model.
Thêm modality fusion layer (e.g., cross-attention) trong PromptForClassification.
Đánh giá: So sánh ACC/F1 trên tasks với code phức tạp (e.g., task5) vs. baseline (chỉ text). Sử dụng modality ablation study (loại bỏ graph) và đo lường robustness với code obfuscation bằng macro-F1 và MCC.