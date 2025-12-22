📊 Đánh giá checkpoint: task_1_final.ckpt
Loaded checkpoint: model/checkpoints/task_1_final.ckpt
  - Task 1... Acc: 0.5219
  - Task 2... Acc: 0.7807
  - Task 3... Acc: 0.6239
  - Task 4... Acc: 0.6720
  - Task 5... Acc: 0.6458

📊 Đánh giá checkpoint: task_2_final.ckpt
Loaded checkpoint: model/checkpoints/task_2_final.ckpt
  - Task 1... Acc: 0.4781
  - Task 2... Acc: 0.8327
  - Task 3... Acc: 0.8394
  - Task 4... Acc: 0.7524
  - Task 5... Acc: 0.6736

📊 Đánh giá checkpoint: task_3_final.ckpt
Loaded checkpoint: model/checkpoints/task_3_final.ckpt
  - Task 1... Acc: 0.4956
  - Task 2... Acc: 0.8364
  - Task 3... Acc: 0.8394
  - Task 4... Acc: 0.7653
  - Task 5... Acc: 0.5938

📊 Đánh giá checkpoint: task_4_final.ckpt
Loaded checkpoint: model/checkpoints/task_4_final.ckpt
  - Task 1... Acc: 0.5526
  - Task 2... Acc: 0.8439
  - Task 3... Acc: 0.8303
  - Task 4... Acc: 0.7010
  - Task 5... Acc: 0.7500

============================================================
📈 PHÂN TÍCH SO SÁNH PHASE 1 vs PHASE 2
============================================================

============================================================
🧠 PHÂN TÍCH CATASTROPHIC FORGETTING
============================================================

📊 Ma trận kết quả (Accuracy):
Hàng: Model sau khi học task i
Cột: Đánh giá trên task j
              Task 1  Task 2  Task 3  Task 4
After Task 1  0.5219  0.7807  0.6239  0.6720
After Task 2  0.4781  0.8327  0.8394  0.7524
After Task 3  0.4956  0.8364  0.8394  0.7653
After Task 4  0.5526  0.8439  0.8303  0.7010

🔥 Ma trận Catastrophic Forgetting:
              Task 1  Task 2  Task 3  Task 4
After Task 1  0.0000  0.0000  0.0000  0.0000
After Task 2  0.0439  0.0000  0.0000  0.0000
After Task 3  0.0263  0.0000  0.0000  0.0000
After Task 4  0.0000  0.0000  0.0092  0.0000

📈 Trung bình Catastrophic Forgetting theo task:
  Task 2: 0.0439
  Task 3: 0.0263
  Task 4: 0.0092

============================================================
🔄 PHÂN TÍCH HIỆU QUẢ REPLAY STRATEGY
============================================================

📊 Hiệu quả Replay Strategy:
 Task  Avg_Previous_Task_Performance  Num_Previous_Tasks
    2                         0.4781                   1
    3                         0.6660                   2
    4                         0.7423                   3

============================================================
📊 TẠO BIỂU ĐỒ TRỰC QUAN
============================================================
✅ Đã tạo biểu đồ:
  - evaluation_results/plots/performance_heatmap.png
  - evaluation_results/plots/learning_curves.png

============================================================
📋 TẠO BÁO CÁO TỔNG KẾT
============================================================
📊 Thống kê tổng quan:
  - Tổng số checkpoint: 4
  - Số task đánh giá: 5
  - Accuracy trung bình: 0.7014 ± 0.1214
  - Accuracy cao nhất: 0.8439
  - Accuracy thấp nhất: 0.4781

