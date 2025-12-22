Tóm tắt cấu trúc và chức năng chính:
teacher_main.py: Script chính để train và evaluate. Nó loop qua 5 tasks, train trên mỗi task, sau đó evaluate trên tất cả tasks.
teacher_model.py: Mô hình CNNTeacherModel với shared TextCNN và các head riêng cho từng group.
textcnn_model.py: Mô hình TextCNN sử dụng CodeBERT để tạo embeddings.
focal_loss.py: Triển khai Focal Loss.
sup_contrastive_loss.py: Triển khai Supervised Contrastive Loss (không được sử dụng trong code chính).
deni.py: Tương tự teacher_main.py nhưng có thể là phiên bản khác (DENI baseline).