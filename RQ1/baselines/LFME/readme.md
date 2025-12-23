teacher_main.py: File chính để train và test model. Nó sử dụng RoBERTa tokenizer và model, kết hợp với TextCNN để xử lý code vulnerability detection. Model được train trên dữ liệu CSV với các cột như func_before (code function) và cwe_ids (labels). Nó hỗ trợ logit adjustment, focal loss, và evaluation metrics (accuracy, F1, MCC, etc.).
teacher_model.py: Định nghĩa class CNNTeacherModel, là model chính với RoBERTa backbone + CNN layers + classification head. Sử dụng CrossEntropyLoss hoặc KLDivLoss cho distillation.
textcnn_model.py: Định nghĩa class TextCNN, sử dụng RoBERTa embeddings, convolutional layers với kernel sizes [3,4,5], max pooling, và fully connected layer. Hỗ trợ dropout và focal loss.
focal_loss.py: Implement focal loss để xử lý imbalanced data.
sup_contrastive_loss.py: Implement supervised contrastive loss (có thể dùng cho distillation hoặc regularization).
student_main.py và combined_teacher_model.py: Các file bổ sung cho student model và combined training (có thể liên quan đến continual learning, nhưng tập trung vào teacher_main.py cho baseline).
Model hoạt động như sau:

Input: Code snippets từ cột func_before trong CSV.
Tokenization: Sử dụng RoBERTa tokenizer (block_size=512 mặc định).
Embedding: RoBERTa embeddings.
CNN: Áp dụng conv layers với kernel [3,4,5], max pooling.
Classification: FC layer với num_class dựa trên cwe_label_map (số CWE labels).
Loss: CrossEntropyLoss, với tùy chọn focal loss hoặc logit adjustment cho imbalanced labels.