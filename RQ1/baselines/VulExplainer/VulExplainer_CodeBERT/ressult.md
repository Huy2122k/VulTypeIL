01/02/2026 17:13:39 - INFO - __main__ -   ***** Running training *****
01/02/2026 17:13:39 - INFO - __main__ -     Num examples = 11508
01/02/2026 17:13:39 - INFO - __main__ -     Num Epochs = 50
01/02/2026 17:13:39 - INFO - __main__ -     Instantaneous batch size per GPU = 128
01/02/2026 17:13:39 - INFO - __main__ -     Total train batch size = 128
01/02/2026 17:13:39 - INFO - __main__ -     Gradient Accumulation steps = 1
01/02/2026 17:13:39 - INFO - __main__ -     Total optimization steps = 4500
epoch 0 loss 3.80664:  99%|████████████████████▊| 89/90 [00:13<00:00,  6.87it/s]01/02/2026 17:13:53 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 17:13:53 - INFO - __main__ -     Num examples = 1413
01/02/2026 17:13:53 - INFO - __main__ -     Batch size = 128
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
01/02/2026 17:13:54 - INFO - __main__ -   ***** Eval results *****
01/02/2026 17:13:54 - INFO - __main__ -     eval_acc = 0.2371
01/02/2026 17:13:54 - INFO - __main__ -     eval_f1 = 0.0365
01/02/2026 17:13:54 - INFO - __main__ -     eval_mcc = 0.2246
01/02/2026 17:13:54 - INFO - __main__ -     eval_precision = 0.0375
01/02/2026 17:13:54 - INFO - __main__ -     eval_recall = 0.0451
01/02/2026 17:13:54 - INFO - __main__ -     ********************
01/02/2026 17:13:54 - INFO - __main__ -     Best Acc:0.2371
01/02/2026 17:13:54 - INFO - __main__ -     ********************
01/02/2026 17:13:55 - INFO - __main__ -   Saving model checkpoint to ./saved_models/checkpoint-best-acc/cnnteacher.bin
epoch 0 loss 3.80664: 100%|█████████████████████| 90/90 [00:15<00:00,  5.90it/s]
epoch 1 loss 2.7449:  99%|█████████████████████▊| 89/90 [00:13<00:00,  6.80it/s]01/02/2026 17:14:08 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 17:14:08 - INFO - __main__ -     Num examples = 1413
01/02/2026 17:14:08 - INFO - __main__ -     Batch size = 128
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
01/02/2026 17:14:08 - INFO - __main__ -   ***** Eval results *****
01/02/2026 17:14:08 - INFO - __main__ -     eval_acc = 0.2994
01/02/2026 17:14:08 - INFO - __main__ -     eval_f1 = 0.0528
01/02/2026 17:14:08 - INFO - __main__ -     eval_mcc = 0.2579
01/02/2026 17:14:08 - INFO - __main__ -     eval_precision = 0.0525
01/02/2026 17:14:08 - INFO - __main__ -     eval_recall = 0.0607
01/02/2026 17:14:08 - INFO - __main__ -     ********************
01/02/2026 17:14:08 - INFO - __main__ -     Best Acc:0.2994
01/02/2026 17:14:08 - INFO - __main__ -     ********************
01/02/2026 17:14:09 - INFO - __main__ -   Saving model checkpoint to ./saved_models/checkpoint-best-acc/cnnteacher.bin
epoch 1 loss 2.7449: 100%|██████████████████████| 90/90 [00:14<00:00,  6.10it/s]
epoch 2 loss 2.22878:  99%|████████████████████▊| 89/90 [00:13<00:00,  6.94it/s]01/02/2026 17:14:22 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 17:14:22 - INFO - __main__ -     Num examples = 1413
01/02/2026 17:14:22 - INFO - __main__ -     Batch size = 128
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
01/02/2026 17:14:23 - INFO - __main__ -   ***** Eval results *****
01/02/2026 17:14:23 - INFO - __main__ -     eval_acc = 0.2895
01/02/2026 17:14:23 - INFO - __main__ -     eval_f1 = 0.0771
01/02/2026 17:14:23 - INFO - __main__ -     eval_mcc = 0.2446
01/02/2026 17:14:23 - INFO - __main__ -     eval_precision = 0.0942
01/02/2026 17:14:23 - INFO - __main__ -     eval_recall = 0.0801
epoch 2 loss 2.22878: 100%|█████████████████████| 90/90 [00:13<00:00,  6.58it/s]
epoch 3 loss 1.46938:  99%|████████████████████▊| 89/90 [00:13<00:00,  6.88it/s]01/02/2026 17:14:36 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 17:14:36 - INFO - __main__ -     Num examples = 1413
01/02/2026 17:14:36 - INFO - __main__ -     Batch size = 128
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
01/02/2026 17:14:37 - INFO - __main__ -   ***** Eval results *****
01/02/2026 17:14:37 - INFO - __main__ -     eval_acc = 0.3008
01/02/2026 17:14:37 - INFO - __main__ -     eval_f1 = 0.1026
01/02/2026 17:14:37 - INFO - __main__ -     eval_mcc = 0.2599
01/02/2026 17:14:37 - INFO - __main__ -     eval_precision = 0.1221
01/02/2026 17:14:37 - INFO - __main__ -     eval_recall = 0.1048
01/02/2026 17:14:37 - INFO - __main__ -     ********************
01/02/2026 17:14:37 - INFO - __main__ -     Best Acc:0.3008
01/02/2026 17:14:37 - INFO - __main__ -     ********************
01/02/2026 17:14:38 - INFO - __main__ -   Saving model checkpoint to ./saved_models/checkpoint-best-acc/cnnteacher.bin
epoch 3 loss 1.46938: 100%|█████████████████████| 90/90 [00:14<00:00,  6.07it/s]
epoch 4 loss 0.8762:  99%|█████████████████████▊| 89/90 [00:13<00:00,  6.88it/s]01/02/2026 17:14:51 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 17:14:51 - INFO - __main__ -     Num examples = 1413
01/02/2026 17:14:51 - INFO - __main__ -     Batch size = 128
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
01/02/2026 17:14:52 - INFO - __main__ -   ***** Eval results *****
01/02/2026 17:14:52 - INFO - __main__ -     eval_acc = 0.2895
01/02/2026 17:14:52 - INFO - __main__ -     eval_f1 = 0.1134
01/02/2026 17:14:52 - INFO - __main__ -     eval_mcc = 0.25
01/02/2026 17:14:52 - INFO - __main__ -     eval_precision = 0.1395
01/02/2026 17:14:52 - INFO - __main__ -     eval_recall = 0.1139
epoch 4 loss 0.8762: 100%|██████████████████████| 90/90 [00:13<00:00,  6.57it/s]
epoch 5 loss 0.67073:  99%|████████████████████▊| 89/90 [00:13<00:00,  6.92it/s]01/02/2026 17:15:05 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 17:15:05 - INFO - __main__ -     Num examples = 1413
01/02/2026 17:15:05 - INFO - __main__ -     Batch size = 128
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
01/02/2026 17:15:05 - INFO - __main__ -   ***** Eval results *****
01/02/2026 17:15:05 - INFO - __main__ -     eval_acc = 0.276
01/02/2026 17:15:05 - INFO - __main__ -     eval_f1 = 0.0982
01/02/2026 17:15:05 - INFO - __main__ -     eval_mcc = 0.237
01/02/2026 17:15:05 - INFO - __main__ -     eval_precision = 0.1154
01/02/2026 17:15:05 - INFO - __main__ -     eval_recall = 0.104
epoch 5 loss 0.67073: 100%|█████████████████████| 90/90 [00:13<00:00,  6.57it/s]
epoch 6 loss 0.62627:  99%|████████████████████▊| 89/90 [00:13<00:00,  6.86it/s]01/02/2026 17:15:18 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 17:15:18 - INFO - __main__ -     Num examples = 1413
01/02/2026 17:15:18 - INFO - __main__ -     Batch size = 128
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
01/02/2026 17:15:19 - INFO - __main__ -   ***** Eval results *****
01/02/2026 17:15:19 - INFO - __main__ -     eval_acc = 0.2682
01/02/2026 17:15:19 - INFO - __main__ -     eval_f1 = 0.0935
01/02/2026 17:15:19 - INFO - __main__ -     eval_mcc = 0.2281
01/02/2026 17:15:19 - INFO - __main__ -     eval_precision = 0.1137
01/02/2026 17:15:19 - INFO - __main__ -     eval_recall = 0.0922
epoch 6 loss 0.62627: 100%|█████████████████████| 90/90 [00:13<00:00,  6.55it/s]
epoch 7 loss 0.60078:  99%|████████████████████▊| 89/90 [00:13<00:00,  6.88it/s]01/02/2026 17:15:32 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 17:15:32 - INFO - __main__ -     Num examples = 1413
01/02/2026 17:15:32 - INFO - __main__ -     Batch size = 128
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
01/02/2026 17:15:33 - INFO - __main__ -   ***** Eval results *****
01/02/2026 17:15:33 - INFO - __main__ -     eval_acc = 0.2746
01/02/2026 17:15:33 - INFO - __main__ -     eval_f1 = 0.0793
01/02/2026 17:15:33 - INFO - __main__ -     eval_mcc = 0.2283
01/02/2026 17:15:33 - INFO - __main__ -     eval_precision = 0.0911
01/02/2026 17:15:33 - INFO - __main__ -     eval_recall = 0.0916
epoch 7 loss 0.60078: 100%|█████████████████████| 90/90 [00:13<00:00,  6.59it/s]
epoch 8 loss 0.61009:  99%|████████████████████▊| 89/90 [00:13<00:00,  6.79it/s]01/02/2026 17:15:46 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 17:15:46 - INFO - __main__ -     Num examples = 1413
01/02/2026 17:15:46 - INFO - __main__ -     Batch size = 128
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
01/02/2026 17:15:46 - INFO - __main__ -   ***** Eval results *****
01/02/2026 17:15:46 - INFO - __main__ -     eval_acc = 0.2711
01/02/2026 17:15:46 - INFO - __main__ -     eval_f1 = 0.0923
01/02/2026 17:15:46 - INFO - __main__ -     eval_mcc = 0.2294
01/02/2026 17:15:46 - INFO - __main__ -     eval_precision = 0.1126
01/02/2026 17:15:46 - INFO - __main__ -     eval_recall = 0.0895
epoch 8 loss 0.61009: 100%|█████████████████████| 90/90 [00:13<00:00,  6.57it/s]
epoch 9 loss 0.5481:  99%|█████████████████████▊| 89/90 [00:13<00:00,  6.88it/s]01/02/2026 17:15:59 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 17:15:59 - INFO - __main__ -     Num examples = 1413
01/02/2026 17:15:59 - INFO - __main__ -     Batch size = 128
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
01/02/2026 17:16:00 - INFO - __main__ -   ***** Eval results *****
01/02/2026 17:16:00 - INFO - __main__ -     eval_acc = 0.2767
01/02/2026 17:16:00 - INFO - __main__ -     eval_f1 = 0.0972
01/02/2026 17:16:00 - INFO - __main__ -     eval_mcc = 0.2356
01/02/2026 17:16:00 - INFO - __main__ -     eval_precision = 0.111
01/02/2026 17:16:00 - INFO - __main__ -     eval_recall = 0.0968
epoch 9 loss 0.5481: 100%|██████████████████████| 90/90 [00:13<00:00,  6.56it/s]
epoch 10 loss 0.50258:  99%|███████████████████▊| 89/90 [00:13<00:00,  6.67it/s]01/02/2026 17:16:14 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 17:16:14 - INFO - __main__ -     Num examples = 1413
01/02/2026 17:16:14 - INFO - __main__ -     Batch size = 128
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
01/02/2026 17:16:14 - INFO - __main__ -   ***** Eval results *****
01/02/2026 17:16:14 - INFO - __main__ -     eval_acc = 0.2619
01/02/2026 17:16:14 - INFO - __main__ -     eval_f1 = 0.0691
01/02/2026 17:16:14 - INFO - __main__ -     eval_mcc = 0.2212
01/02/2026 17:16:14 - INFO - __main__ -     eval_precision = 0.0731
01/02/2026 17:16:14 - INFO - __main__ -     eval_recall = 0.0777
epoch 10 loss 0.50258: 100%|████████████████████| 90/90 [00:14<00:00,  6.41it/s]
epoch 11 loss 0.40623:  99%|███████████████████▊| 89/90 [00:13<00:00,  6.90it/s]01/02/2026 17:16:27 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 17:16:27 - INFO - __main__ -     Num examples = 1413
01/02/2026 17:16:27 - INFO - __main__ -     Batch size = 128
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
01/02/2026 17:16:28 - INFO - __main__ -   ***** Eval results *****
01/02/2026 17:16:28 - INFO - __main__ -     eval_acc = 0.2654
01/02/2026 17:16:28 - INFO - __main__ -     eval_f1 = 0.0888
01/02/2026 17:16:28 - INFO - __main__ -     eval_mcc = 0.2185
01/02/2026 17:16:28 - INFO - __main__ -     eval_precision = 0.1203
01/02/2026 17:16:28 - INFO - __main__ -     eval_recall = 0.0817
epoch 11 loss 0.40623: 100%|████████████████████| 90/90 [00:13<00:00,  6.57it/s]
epoch 12 loss 0.34615:  99%|███████████████████▊| 89/90 [00:13<00:00,  6.80it/s]01/02/2026 17:16:41 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 17:16:41 - INFO - __main__ -     Num examples = 1413
01/02/2026 17:16:41 - INFO - __main__ -     Batch size = 128
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
01/02/2026 17:16:42 - INFO - __main__ -   ***** Eval results *****
01/02/2026 17:16:42 - INFO - __main__ -     eval_acc = 0.2569
01/02/2026 17:16:42 - INFO - __main__ -     eval_f1 = 0.0755
01/02/2026 17:16:42 - INFO - __main__ -     eval_mcc = 0.2155
01/02/2026 17:16:42 - INFO - __main__ -     eval_precision = 0.0753
01/02/2026 17:16:42 - INFO - __main__ -     eval_recall = 0.0821
epoch 12 loss 0.34615: 100%|████████████████████| 90/90 [00:13<00:00,  6.56it/s]
epoch 13 loss 0.2844:  99%|████████████████████▊| 89/90 [00:13<00:00,  6.90it/s]01/02/2026 17:16:55 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 17:16:55 - INFO - __main__ -     Num examples = 1413
01/02/2026 17:16:55 - INFO - __main__ -     Batch size = 128
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
01/02/2026 17:16:55 - INFO - __main__ -   ***** Eval results *****
01/02/2026 17:16:55 - INFO - __main__ -     eval_acc = 0.2689
01/02/2026 17:16:55 - INFO - __main__ -     eval_f1 = 0.0887
01/02/2026 17:16:55 - INFO - __main__ -     eval_mcc = 0.2252
01/02/2026 17:16:55 - INFO - __main__ -     eval_precision = 0.1123
01/02/2026 17:16:55 - INFO - __main__ -     eval_recall = 0.0875
epoch 13 loss 0.2844: 100%|█████████████████████| 90/90 [00:13<00:00,  6.58it/s]
epoch 14 loss 0.24055:  99%|███████████████████▊| 89/90 [00:13<00:00,  6.91it/s]01/02/2026 17:17:08 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 17:17:08 - INFO - __main__ -     Num examples = 1413
01/02/2026 17:17:08 - INFO - __main__ -     Batch size = 128
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
01/02/2026 17:17:09 - INFO - __main__ -   ***** Eval results *****
01/02/2026 17:17:09 - INFO - __main__ -     eval_acc = 0.2746
01/02/2026 17:17:09 - INFO - __main__ -     eval_f1 = 0.0854
01/02/2026 17:17:09 - INFO - __main__ -     eval_mcc = 0.2286
01/02/2026 17:17:09 - INFO - __main__ -     eval_precision = 0.0946
01/02/2026 17:17:09 - INFO - __main__ -     eval_recall = 0.0863
epoch 14 loss 0.24055: 100%|████████████████████| 90/90 [00:13<00:00,  6.58it/s]
epoch 15 loss 0.20989:  99%|███████████████████▊| 89/90 [00:13<00:00,  6.92it/s]01/02/2026 17:17:22 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 17:17:22 - INFO - __main__ -     Num examples = 1413
01/02/2026 17:17:22 - INFO - __main__ -     Batch size = 128
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
01/02/2026 17:17:23 - INFO - __main__ -   ***** Eval results *****
01/02/2026 17:17:23 - INFO - __main__ -     eval_acc = 0.2689
01/02/2026 17:17:23 - INFO - __main__ -     eval_f1 = 0.0849
01/02/2026 17:17:23 - INFO - __main__ -     eval_mcc = 0.2291
01/02/2026 17:17:23 - INFO - __main__ -     eval_precision = 0.091
01/02/2026 17:17:23 - INFO - __main__ -     eval_recall = 0.0873
epoch 15 loss 0.20989: 100%|████████████████████| 90/90 [00:13<00:00,  6.59it/s]
epoch 16 loss 0.16327:  99%|███████████████████▊| 89/90 [00:13<00:00,  6.92it/s]01/02/2026 17:17:36 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 17:17:36 - INFO - __main__ -     Num examples = 1413
01/02/2026 17:17:36 - INFO - __main__ -     Batch size = 128
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
01/02/2026 17:17:36 - INFO - __main__ -   ***** Eval results *****
01/02/2026 17:17:36 - INFO - __main__ -     eval_acc = 0.2994
01/02/2026 17:17:36 - INFO - __main__ -     eval_f1 = 0.0961
01/02/2026 17:17:36 - INFO - __main__ -     eval_mcc = 0.2506
01/02/2026 17:17:36 - INFO - __main__ -     eval_precision = 0.1152
01/02/2026 17:17:36 - INFO - __main__ -     eval_recall = 0.0929
epoch 16 loss 0.16327: 100%|████████████████████| 90/90 [00:13<00:00,  6.59it/s]
epoch 17 loss 0.15232:  99%|███████████████████▊| 89/90 [00:13<00:00,  6.89it/s]01/02/2026 17:17:49 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 17:17:49 - INFO - __main__ -     Num examples = 1413
01/02/2026 17:17:49 - INFO - __main__ -     Batch size = 128
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
01/02/2026 17:17:50 - INFO - __main__ -   ***** Eval results *****
01/02/2026 17:17:50 - INFO - __main__ -     eval_acc = 0.2916
01/02/2026 17:17:50 - INFO - __main__ -     eval_f1 = 0.0854
01/02/2026 17:17:50 - INFO - __main__ -     eval_mcc = 0.2452
01/02/2026 17:17:50 - INFO - __main__ -     eval_precision = 0.0877
01/02/2026 17:17:50 - INFO - __main__ -     eval_recall = 0.0898
epoch 17 loss 0.15232: 100%|████████████████████| 90/90 [00:13<00:00,  6.57it/s]
epoch 18 loss 0.13325:  99%|███████████████████▊| 89/90 [00:13<00:00,  6.91it/s]01/02/2026 17:18:03 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 17:18:03 - INFO - __main__ -     Num examples = 1413
01/02/2026 17:18:03 - INFO - __main__ -     Batch size = 128
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
01/02/2026 17:18:04 - INFO - __main__ -   ***** Eval results *****
01/02/2026 17:18:04 - INFO - __main__ -     eval_acc = 0.2923
01/02/2026 17:18:04 - INFO - __main__ -     eval_f1 = 0.086
01/02/2026 17:18:04 - INFO - __main__ -     eval_mcc = 0.2462
01/02/2026 17:18:04 - INFO - __main__ -     eval_precision = 0.0914
01/02/2026 17:18:04 - INFO - __main__ -     eval_recall = 0.099
epoch 18 loss 0.13325: 100%|████████████████████| 90/90 [00:13<00:00,  6.59it/s]
epoch 19 loss 0.12476:  99%|███████████████████▊| 89/90 [00:13<00:00,  6.76it/s]01/02/2026 17:18:17 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 17:18:17 - INFO - __main__ -     Num examples = 1413
01/02/2026 17:18:17 - INFO - __main__ -     Batch size = 128
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
01/02/2026 17:18:17 - INFO - __main__ -   ***** Eval results *****
01/02/2026 17:18:17 - INFO - __main__ -     eval_acc = 0.2909
01/02/2026 17:18:17 - INFO - __main__ -     eval_f1 = 0.0976
01/02/2026 17:18:17 - INFO - __main__ -     eval_mcc = 0.2441
01/02/2026 17:18:17 - INFO - __main__ -     eval_precision = 0.102
01/02/2026 17:18:17 - INFO - __main__ -     eval_recall = 0.1004
epoch 19 loss 0.12476: 100%|████████████████████| 90/90 [00:13<00:00,  6.59it/s]
epoch 20 loss 0.1199:  99%|████████████████████▊| 89/90 [00:13<00:00,  6.94it/s]01/02/2026 17:18:30 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 17:18:30 - INFO - __main__ -     Num examples = 1413
01/02/2026 17:18:30 - INFO - __main__ -     Batch size = 128
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
01/02/2026 17:18:31 - INFO - __main__ -   ***** Eval results *****
01/02/2026 17:18:31 - INFO - __main__ -     eval_acc = 0.288
01/02/2026 17:18:31 - INFO - __main__ -     eval_f1 = 0.0928
01/02/2026 17:18:31 - INFO - __main__ -     eval_mcc = 0.242
01/02/2026 17:18:31 - INFO - __main__ -     eval_precision = 0.1015
01/02/2026 17:18:31 - INFO - __main__ -     eval_recall = 0.095
epoch 20 loss 0.1199: 100%|█████████████████████| 90/90 [00:13<00:00,  6.59it/s]
epoch 21 loss 0.11098:  99%|███████████████████▊| 89/90 [00:13<00:00,  6.92it/s]01/02/2026 17:18:44 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 17:18:44 - INFO - __main__ -     Num examples = 1413
01/02/2026 17:18:44 - INFO - __main__ -     Batch size = 128
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
01/02/2026 17:18:45 - INFO - __main__ -   ***** Eval results *****
01/02/2026 17:18:45 - INFO - __main__ -     eval_acc = 0.2923
01/02/2026 17:18:45 - INFO - __main__ -     eval_f1 = 0.0919
01/02/2026 17:18:45 - INFO - __main__ -     eval_mcc = 0.2452
01/02/2026 17:18:45 - INFO - __main__ -     eval_precision = 0.0996
01/02/2026 17:18:45 - INFO - __main__ -     eval_recall = 0.0958
epoch 21 loss 0.11098: 100%|████████████████████| 90/90 [00:13<00:00,  6.59it/s]
epoch 22 loss 0.11013:  99%|███████████████████▊| 89/90 [00:13<00:00,  6.80it/s]01/02/2026 17:18:58 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 17:18:58 - INFO - __main__ -     Num examples = 1413
01/02/2026 17:18:58 - INFO - __main__ -     Batch size = 128
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
01/02/2026 17:18:58 - INFO - __main__ -   ***** Eval results *****
01/02/2026 17:18:58 - INFO - __main__ -     eval_acc = 0.2887
01/02/2026 17:18:58 - INFO - __main__ -     eval_f1 = 0.0862
01/02/2026 17:18:58 - INFO - __main__ -     eval_mcc = 0.2412
01/02/2026 17:18:58 - INFO - __main__ -     eval_precision = 0.1059
01/02/2026 17:18:58 - INFO - __main__ -     eval_recall = 0.0823
epoch 22 loss 0.11013: 100%|████████████████████| 90/90 [00:13<00:00,  6.57it/s]
epoch 23 loss 0.10704:  99%|███████████████████▊| 89/90 [00:13<00:00,  6.94it/s]01/02/2026 17:19:11 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 17:19:11 - INFO - __main__ -     Num examples = 1413
01/02/2026 17:19:11 - INFO - __main__ -     Batch size = 128
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
01/02/2026 17:19:12 - INFO - __main__ -   ***** Eval results *****
01/02/2026 17:19:12 - INFO - __main__ -     eval_acc = 0.305
01/02/2026 17:19:12 - INFO - __main__ -     eval_f1 = 0.0931
01/02/2026 17:19:12 - INFO - __main__ -     eval_mcc = 0.2559
01/02/2026 17:19:12 - INFO - __main__ -     eval_precision = 0.1055
01/02/2026 17:19:12 - INFO - __main__ -     eval_recall = 0.0934
01/02/2026 17:19:12 - INFO - __main__ -     ********************
01/02/2026 17:19:12 - INFO - __main__ -     Best Acc:0.305
01/02/2026 17:19:12 - INFO - __main__ -     ********************
01/02/2026 17:19:13 - INFO - __main__ -   Saving model checkpoint to ./saved_models/checkpoint-best-acc/cnnteacher.bin
epoch 23 loss 0.10704: 100%|████████████████████| 90/90 [00:14<00:00,  6.09it/s]
epoch 24 loss 0.09701:  99%|███████████████████▊| 89/90 [00:13<00:00,  6.86it/s]01/02/2026 17:19:26 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 17:19:26 - INFO - __main__ -     Num examples = 1413
01/02/2026 17:19:26 - INFO - __main__ -     Batch size = 128
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
01/02/2026 17:19:27 - INFO - __main__ -   ***** Eval results *****
01/02/2026 17:19:27 - INFO - __main__ -     eval_acc = 0.276
01/02/2026 17:19:27 - INFO - __main__ -     eval_f1 = 0.0891
01/02/2026 17:19:27 - INFO - __main__ -     eval_mcc = 0.232
01/02/2026 17:19:27 - INFO - __main__ -     eval_precision = 0.1041
01/02/2026 17:19:27 - INFO - __main__ -     eval_recall = 0.088
epoch 24 loss 0.09701: 100%|████████████████████| 90/90 [00:13<00:00,  6.55it/s]
epoch 25 loss 0.09236:  99%|███████████████████▊| 89/90 [00:13<00:00,  6.92it/s]01/02/2026 17:19:40 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 17:19:40 - INFO - __main__ -     Num examples = 1413
01/02/2026 17:19:40 - INFO - __main__ -     Batch size = 128
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
01/02/2026 17:19:40 - INFO - __main__ -   ***** Eval results *****
01/02/2026 17:19:40 - INFO - __main__ -     eval_acc = 0.3029
01/02/2026 17:19:40 - INFO - __main__ -     eval_f1 = 0.0952
01/02/2026 17:19:40 - INFO - __main__ -     eval_mcc = 0.2557
01/02/2026 17:19:40 - INFO - __main__ -     eval_precision = 0.1037
01/02/2026 17:19:40 - INFO - __main__ -     eval_recall = 0.0993
epoch 25 loss 0.09236: 100%|████████████████████| 90/90 [00:13<00:00,  6.56it/s]
epoch 26 loss 0.08921:  99%|███████████████████▊| 89/90 [00:13<00:00,  6.81it/s]01/02/2026 17:19:54 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 17:19:54 - INFO - __main__ -     Num examples = 1413
01/02/2026 17:19:54 - INFO - __main__ -     Batch size = 128
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
01/02/2026 17:19:54 - INFO - __main__ -   ***** Eval results *****
01/02/2026 17:19:54 - INFO - __main__ -     eval_acc = 0.281
01/02/2026 17:19:54 - INFO - __main__ -     eval_f1 = 0.0938
01/02/2026 17:19:54 - INFO - __main__ -     eval_mcc = 0.2381
01/02/2026 17:19:54 - INFO - __main__ -     eval_precision = 0.1225
01/02/2026 17:19:54 - INFO - __main__ -     eval_recall = 0.0866
epoch 26 loss 0.08921: 100%|████████████████████| 90/90 [00:13<00:00,  6.56it/s]
epoch 27 loss 0.0864:  99%|████████████████████▊| 89/90 [00:13<00:00,  6.69it/s]01/02/2026 17:20:07 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 17:20:07 - INFO - __main__ -     Num examples = 1413
01/02/2026 17:20:07 - INFO - __main__ -     Batch size = 128
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
01/02/2026 17:20:08 - INFO - __main__ -   ***** Eval results *****
01/02/2026 17:20:08 - INFO - __main__ -     eval_acc = 0.3001
01/02/2026 17:20:08 - INFO - __main__ -     eval_f1 = 0.0922
01/02/2026 17:20:08 - INFO - __main__ -     eval_mcc = 0.2538
01/02/2026 17:20:08 - INFO - __main__ -     eval_precision = 0.1008
01/02/2026 17:20:08 - INFO - __main__ -     eval_recall = 0.0939
epoch 27 loss 0.0864: 100%|█████████████████████| 90/90 [00:13<00:00,  6.55it/s]
epoch 28 loss 0.08515:  99%|███████████████████▊| 89/90 [00:13<00:00,  6.90it/s]01/02/2026 17:20:21 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 17:20:21 - INFO - __main__ -     Num examples = 1413
01/02/2026 17:20:21 - INFO - __main__ -     Batch size = 128
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
01/02/2026 17:20:22 - INFO - __main__ -   ***** Eval results *****
01/02/2026 17:20:22 - INFO - __main__ -     eval_acc = 0.2895
01/02/2026 17:20:22 - INFO - __main__ -     eval_f1 = 0.089
01/02/2026 17:20:22 - INFO - __main__ -     eval_mcc = 0.2417
01/02/2026 17:20:22 - INFO - __main__ -     eval_precision = 0.1052
01/02/2026 17:20:22 - INFO - __main__ -     eval_recall = 0.0876
epoch 28 loss 0.08515: 100%|████████████████████| 90/90 [00:13<00:00,  6.55it/s]
epoch 29 loss 0.0812:  99%|████████████████████▊| 89/90 [00:13<00:00,  6.84it/s]01/02/2026 17:20:35 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 17:20:35 - INFO - __main__ -     Num examples = 1413
01/02/2026 17:20:35 - INFO - __main__ -     Batch size = 128
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
01/02/2026 17:20:35 - INFO - __main__ -   ***** Eval results *****
01/02/2026 17:20:35 - INFO - __main__ -     eval_acc = 0.2703
01/02/2026 17:20:35 - INFO - __main__ -     eval_f1 = 0.0822
01/02/2026 17:20:35 - INFO - __main__ -     eval_mcc = 0.2249
01/02/2026 17:20:35 - INFO - __main__ -     eval_precision = 0.0982
01/02/2026 17:20:35 - INFO - __main__ -     eval_recall = 0.0834
epoch 29 loss 0.0812: 100%|█████████████████████| 90/90 [00:13<00:00,  6.55it/s]
epoch 30 loss 0.08119:  99%|███████████████████▊| 89/90 [00:13<00:00,  6.87it/s]01/02/2026 17:20:49 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 17:20:49 - INFO - __main__ -     Num examples = 1413
01/02/2026 17:20:49 - INFO - __main__ -     Batch size = 128
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
01/02/2026 17:20:49 - INFO - __main__ -   ***** Eval results *****
01/02/2026 17:20:49 - INFO - __main__ -     eval_acc = 0.2859
01/02/2026 17:20:49 - INFO - __main__ -     eval_f1 = 0.0869
01/02/2026 17:20:49 - INFO - __main__ -     eval_mcc = 0.2396
01/02/2026 17:20:49 - INFO - __main__ -     eval_precision = 0.1015
01/02/2026 17:20:49 - INFO - __main__ -     eval_recall = 0.0865
epoch 30 loss 0.08119: 100%|████████████████████| 90/90 [00:13<00:00,  6.56it/s]
epoch 31 loss 0.07348:  99%|███████████████████▊| 89/90 [00:13<00:00,  6.82it/s]01/02/2026 17:21:02 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 17:21:02 - INFO - __main__ -     Num examples = 1413
01/02/2026 17:21:02 - INFO - __main__ -     Batch size = 128
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
01/02/2026 17:21:03 - INFO - __main__ -   ***** Eval results *****
01/02/2026 17:21:03 - INFO - __main__ -     eval_acc = 0.2845
01/02/2026 17:21:03 - INFO - __main__ -     eval_f1 = 0.0892
01/02/2026 17:21:03 - INFO - __main__ -     eval_mcc = 0.24
01/02/2026 17:21:03 - INFO - __main__ -     eval_precision = 0.0967
01/02/2026 17:21:03 - INFO - __main__ -     eval_recall = 0.0914
epoch 31 loss 0.07348: 100%|████████████████████| 90/90 [00:13<00:00,  6.57it/s]
epoch 32 loss 0.0769:  99%|████████████████████▊| 89/90 [00:13<00:00,  6.89it/s]01/02/2026 17:21:16 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 17:21:16 - INFO - __main__ -     Num examples = 1413
01/02/2026 17:21:16 - INFO - __main__ -     Batch size = 128
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
01/02/2026 17:21:17 - INFO - __main__ -   ***** Eval results *****
01/02/2026 17:21:17 - INFO - __main__ -     eval_acc = 0.2824
01/02/2026 17:21:17 - INFO - __main__ -     eval_f1 = 0.0809
01/02/2026 17:21:17 - INFO - __main__ -     eval_mcc = 0.2339
01/02/2026 17:21:17 - INFO - __main__ -     eval_precision = 0.1003
01/02/2026 17:21:17 - INFO - __main__ -     eval_recall = 0.0805
epoch 32 loss 0.0769: 100%|█████████████████████| 90/90 [00:13<00:00,  6.58it/s]
epoch 33 loss 0.07349:  99%|███████████████████▊| 89/90 [00:13<00:00,  6.74it/s]01/02/2026 17:21:30 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 17:21:30 - INFO - __main__ -     Num examples = 1413
01/02/2026 17:21:30 - INFO - __main__ -     Batch size = 128
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
01/02/2026 17:21:31 - INFO - __main__ -   ***** Eval results *****
01/02/2026 17:21:31 - INFO - __main__ -     eval_acc = 0.2958
01/02/2026 17:21:31 - INFO - __main__ -     eval_f1 = 0.085
01/02/2026 17:21:31 - INFO - __main__ -     eval_mcc = 0.2501
01/02/2026 17:21:31 - INFO - __main__ -     eval_precision = 0.0939
01/02/2026 17:21:31 - INFO - __main__ -     eval_recall = 0.0871
epoch 33 loss 0.07349: 100%|████████████████████| 90/90 [00:13<00:00,  6.44it/s]
epoch 34 loss 0.07199:  99%|███████████████████▊| 89/90 [00:13<00:00,  6.78it/s]01/02/2026 17:21:44 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 17:21:44 - INFO - __main__ -     Num examples = 1413
01/02/2026 17:21:44 - INFO - __main__ -     Batch size = 128
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
01/02/2026 17:21:45 - INFO - __main__ -   ***** Eval results *****
01/02/2026 17:21:45 - INFO - __main__ -     eval_acc = 0.2873
01/02/2026 17:21:45 - INFO - __main__ -     eval_f1 = 0.0791
01/02/2026 17:21:45 - INFO - __main__ -     eval_mcc = 0.2408
01/02/2026 17:21:45 - INFO - __main__ -     eval_precision = 0.0849
01/02/2026 17:21:45 - INFO - __main__ -     eval_recall = 0.0802
epoch 34 loss 0.07199: 100%|████████████████████| 90/90 [00:13<00:00,  6.43it/s]
epoch 35 loss 0.06568:  99%|███████████████████▊| 89/90 [00:13<00:00,  6.70it/s]01/02/2026 17:21:58 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 17:21:58 - INFO - __main__ -     Num examples = 1413
01/02/2026 17:21:58 - INFO - __main__ -     Batch size = 128
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
01/02/2026 17:21:59 - INFO - __main__ -   ***** Eval results *****
01/02/2026 17:21:59 - INFO - __main__ -     eval_acc = 0.2916
01/02/2026 17:21:59 - INFO - __main__ -     eval_f1 = 0.0916
01/02/2026 17:21:59 - INFO - __main__ -     eval_mcc = 0.2434
01/02/2026 17:21:59 - INFO - __main__ -     eval_precision = 0.1099
01/02/2026 17:21:59 - INFO - __main__ -     eval_recall = 0.0903
epoch 35 loss 0.06568: 100%|████████████████████| 90/90 [00:13<00:00,  6.43it/s]
epoch 36 loss 0.0621:  99%|████████████████████▊| 89/90 [00:13<00:00,  6.80it/s]01/02/2026 17:22:12 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 17:22:12 - INFO - __main__ -     Num examples = 1413
01/02/2026 17:22:12 - INFO - __main__ -     Batch size = 128
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
01/02/2026 17:22:12 - INFO - __main__ -   ***** Eval results *****
01/02/2026 17:22:12 - INFO - __main__ -     eval_acc = 0.2873
01/02/2026 17:22:12 - INFO - __main__ -     eval_f1 = 0.0794
01/02/2026 17:22:12 - INFO - __main__ -     eval_mcc = 0.2416
01/02/2026 17:22:12 - INFO - __main__ -     eval_precision = 0.0929
01/02/2026 17:22:12 - INFO - __main__ -     eval_recall = 0.0825
epoch 36 loss 0.0621: 100%|█████████████████████| 90/90 [00:13<00:00,  6.44it/s]
epoch 37 loss 0.06483:  99%|███████████████████▊| 89/90 [00:13<00:00,  6.80it/s]01/02/2026 17:22:26 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 17:22:26 - INFO - __main__ -     Num examples = 1413
01/02/2026 17:22:26 - INFO - __main__ -     Batch size = 128
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
01/02/2026 17:22:26 - INFO - __main__ -   ***** Eval results *****
01/02/2026 17:22:26 - INFO - __main__ -     eval_acc = 0.2866
01/02/2026 17:22:26 - INFO - __main__ -     eval_f1 = 0.0888
01/02/2026 17:22:26 - INFO - __main__ -     eval_mcc = 0.2426
01/02/2026 17:22:26 - INFO - __main__ -     eval_precision = 0.1018
01/02/2026 17:22:26 - INFO - __main__ -     eval_recall = 0.0895
epoch 37 loss 0.06483: 100%|████████████████████| 90/90 [00:13<00:00,  6.44it/s]
epoch 38 loss 0.05759:  99%|███████████████████▊| 89/90 [00:13<00:00,  6.74it/s]01/02/2026 17:22:40 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 17:22:40 - INFO - __main__ -     Num examples = 1413
01/02/2026 17:22:40 - INFO - __main__ -     Batch size = 128
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
01/02/2026 17:22:40 - INFO - __main__ -   ***** Eval results *****
01/02/2026 17:22:40 - INFO - __main__ -     eval_acc = 0.2979
01/02/2026 17:22:40 - INFO - __main__ -     eval_f1 = 0.0912
01/02/2026 17:22:40 - INFO - __main__ -     eval_mcc = 0.249
01/02/2026 17:22:40 - INFO - __main__ -     eval_precision = 0.1007
01/02/2026 17:22:40 - INFO - __main__ -     eval_recall = 0.0917
epoch 38 loss 0.05759: 100%|████████████████████| 90/90 [00:13<00:00,  6.45it/s]
epoch 39 loss 0.05678:  99%|███████████████████▊| 89/90 [00:13<00:00,  6.80it/s]01/02/2026 17:22:54 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 17:22:54 - INFO - __main__ -     Num examples = 1413
01/02/2026 17:22:54 - INFO - __main__ -     Batch size = 128
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
01/02/2026 17:22:54 - INFO - __main__ -   ***** Eval results *****
01/02/2026 17:22:54 - INFO - __main__ -     eval_acc = 0.2958
01/02/2026 17:22:54 - INFO - __main__ -     eval_f1 = 0.0898
01/02/2026 17:22:54 - INFO - __main__ -     eval_mcc = 0.2497
01/02/2026 17:22:54 - INFO - __main__ -     eval_precision = 0.1001
01/02/2026 17:22:54 - INFO - __main__ -     eval_recall = 0.0887
epoch 39 loss 0.05678: 100%|████████████████████| 90/90 [00:13<00:00,  6.45it/s]
epoch 40 loss 0.05064:  99%|███████████████████▊| 89/90 [00:13<00:00,  6.61it/s]01/02/2026 17:23:08 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 17:23:08 - INFO - __main__ -     Num examples = 1413
01/02/2026 17:23:08 - INFO - __main__ -     Batch size = 128
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
01/02/2026 17:23:08 - INFO - __main__ -   ***** Eval results *****
01/02/2026 17:23:08 - INFO - __main__ -     eval_acc = 0.2845
01/02/2026 17:23:08 - INFO - __main__ -     eval_f1 = 0.088
01/02/2026 17:23:08 - INFO - __main__ -     eval_mcc = 0.2373
01/02/2026 17:23:08 - INFO - __main__ -     eval_precision = 0.0988
01/02/2026 17:23:08 - INFO - __main__ -     eval_recall = 0.0864
epoch 40 loss 0.05064: 100%|████████████████████| 90/90 [00:14<00:00,  6.41it/s]
epoch 41 loss 0.05023:  99%|███████████████████▊| 89/90 [00:13<00:00,  6.80it/s]01/02/2026 17:23:22 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 17:23:22 - INFO - __main__ -     Num examples = 1413
01/02/2026 17:23:22 - INFO - __main__ -     Batch size = 128
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
01/02/2026 17:23:22 - INFO - __main__ -   ***** Eval results *****
01/02/2026 17:23:22 - INFO - __main__ -     eval_acc = 0.2845
01/02/2026 17:23:22 - INFO - __main__ -     eval_f1 = 0.0907
01/02/2026 17:23:22 - INFO - __main__ -     eval_mcc = 0.2385
01/02/2026 17:23:22 - INFO - __main__ -     eval_precision = 0.1027
01/02/2026 17:23:22 - INFO - __main__ -     eval_recall = 0.0888
epoch 41 loss 0.05023: 100%|████████████████████| 90/90 [00:13<00:00,  6.44it/s]
epoch 42 loss 0.04611:  99%|███████████████████▊| 89/90 [00:13<00:00,  6.74it/s]01/02/2026 17:23:36 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 17:23:36 - INFO - __main__ -     Num examples = 1413
01/02/2026 17:23:36 - INFO - __main__ -     Batch size = 128
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
01/02/2026 17:23:36 - INFO - __main__ -   ***** Eval results *****
01/02/2026 17:23:36 - INFO - __main__ -     eval_acc = 0.2831
01/02/2026 17:23:36 - INFO - __main__ -     eval_f1 = 0.0868
01/02/2026 17:23:36 - INFO - __main__ -     eval_mcc = 0.2375
01/02/2026 17:23:36 - INFO - __main__ -     eval_precision = 0.0902
01/02/2026 17:23:36 - INFO - __main__ -     eval_recall = 0.0896
epoch 42 loss 0.04611: 100%|████████████████████| 90/90 [00:13<00:00,  6.43it/s]
epoch 43 loss 0.04503:  99%|███████████████████▊| 89/90 [00:13<00:00,  6.77it/s]01/02/2026 17:23:50 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 17:23:50 - INFO - __main__ -     Num examples = 1413
01/02/2026 17:23:50 - INFO - __main__ -     Batch size = 128
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
01/02/2026 17:23:50 - INFO - __main__ -   ***** Eval results *****
01/02/2026 17:23:50 - INFO - __main__ -     eval_acc = 0.2958
01/02/2026 17:23:50 - INFO - __main__ -     eval_f1 = 0.0895
01/02/2026 17:23:50 - INFO - __main__ -     eval_mcc = 0.2476
01/02/2026 17:23:50 - INFO - __main__ -     eval_precision = 0.0957
01/02/2026 17:23:50 - INFO - __main__ -     eval_recall = 0.0904
epoch 43 loss 0.04503: 100%|████████████████████| 90/90 [00:14<00:00,  6.43it/s]
epoch 44 loss 0.04108:  99%|███████████████████▊| 89/90 [00:13<00:00,  6.70it/s]01/02/2026 17:24:04 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 17:24:04 - INFO - __main__ -     Num examples = 1413
01/02/2026 17:24:04 - INFO - __main__ -     Batch size = 128
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
01/02/2026 17:24:04 - INFO - __main__ -   ***** Eval results *****
01/02/2026 17:24:04 - INFO - __main__ -     eval_acc = 0.2944
01/02/2026 17:24:04 - INFO - __main__ -     eval_f1 = 0.0869
01/02/2026 17:24:04 - INFO - __main__ -     eval_mcc = 0.2443
01/02/2026 17:24:04 - INFO - __main__ -     eval_precision = 0.1019
01/02/2026 17:24:04 - INFO - __main__ -     eval_recall = 0.0827
epoch 44 loss 0.04108: 100%|████████████████████| 90/90 [00:14<00:00,  6.43it/s]
epoch 45 loss 0.03649:  99%|███████████████████▊| 89/90 [00:13<00:00,  6.67it/s]01/02/2026 17:24:18 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 17:24:18 - INFO - __main__ -     Num examples = 1413
01/02/2026 17:24:18 - INFO - __main__ -     Batch size = 128
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
01/02/2026 17:24:18 - INFO - __main__ -   ***** Eval results *****
01/02/2026 17:24:18 - INFO - __main__ -     eval_acc = 0.2972
01/02/2026 17:24:18 - INFO - __main__ -     eval_f1 = 0.0838
01/02/2026 17:24:18 - INFO - __main__ -     eval_mcc = 0.2482
01/02/2026 17:24:18 - INFO - __main__ -     eval_precision = 0.1024
01/02/2026 17:24:18 - INFO - __main__ -     eval_recall = 0.0811
epoch 45 loss 0.03649: 100%|████████████████████| 90/90 [00:13<00:00,  6.45it/s]
epoch 46 loss 0.03481:  99%|███████████████████▊| 89/90 [00:13<00:00,  6.79it/s]01/02/2026 17:24:32 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 17:24:32 - INFO - __main__ -     Num examples = 1413
01/02/2026 17:24:32 - INFO - __main__ -     Batch size = 128
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
01/02/2026 17:24:32 - INFO - __main__ -   ***** Eval results *****
01/02/2026 17:24:32 - INFO - __main__ -     eval_acc = 0.3015
01/02/2026 17:24:32 - INFO - __main__ -     eval_f1 = 0.0918
01/02/2026 17:24:32 - INFO - __main__ -     eval_mcc = 0.253
01/02/2026 17:24:32 - INFO - __main__ -     eval_precision = 0.0992
01/02/2026 17:24:32 - INFO - __main__ -     eval_recall = 0.0903
epoch 46 loss 0.03481: 100%|████████████████████| 90/90 [00:13<00:00,  6.44it/s]
epoch 47 loss 0.0317:  99%|████████████████████▊| 89/90 [00:13<00:00,  6.79it/s]01/02/2026 17:24:46 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 17:24:46 - INFO - __main__ -     Num examples = 1413
01/02/2026 17:24:46 - INFO - __main__ -     Batch size = 128
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
01/02/2026 17:24:46 - INFO - __main__ -   ***** Eval results *****
01/02/2026 17:24:46 - INFO - __main__ -     eval_acc = 0.2937
01/02/2026 17:24:46 - INFO - __main__ -     eval_f1 = 0.0901
01/02/2026 17:24:46 - INFO - __main__ -     eval_mcc = 0.2448
01/02/2026 17:24:46 - INFO - __main__ -     eval_precision = 0.0993
01/02/2026 17:24:46 - INFO - __main__ -     eval_recall = 0.0884
epoch 47 loss 0.0317: 100%|█████████████████████| 90/90 [00:13<00:00,  6.46it/s]
epoch 48 loss 0.02773:  99%|███████████████████▊| 89/90 [00:13<00:00,  6.73it/s]01/02/2026 17:25:00 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 17:25:00 - INFO - __main__ -     Num examples = 1413
01/02/2026 17:25:00 - INFO - __main__ -     Batch size = 128
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
01/02/2026 17:25:00 - INFO - __main__ -   ***** Eval results *****
01/02/2026 17:25:00 - INFO - __main__ -     eval_acc = 0.2979
01/02/2026 17:25:00 - INFO - __main__ -     eval_f1 = 0.0895
01/02/2026 17:25:00 - INFO - __main__ -     eval_mcc = 0.2489
01/02/2026 17:25:00 - INFO - __main__ -     eval_precision = 0.0997
01/02/2026 17:25:00 - INFO - __main__ -     eval_recall = 0.088
epoch 48 loss 0.02773: 100%|████████████████████| 90/90 [00:13<00:00,  6.44it/s]
epoch 49 loss 0.0262:  99%|████████████████████▊| 89/90 [00:13<00:00,  6.79it/s]01/02/2026 17:25:13 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 17:25:13 - INFO - __main__ -     Num examples = 1413
01/02/2026 17:25:13 - INFO - __main__ -     Batch size = 128
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
01/02/2026 17:25:14 - INFO - __main__ -   ***** Eval results *****
01/02/2026 17:25:14 - INFO - __main__ -     eval_acc = 0.3008
01/02/2026 17:25:14 - INFO - __main__ -     eval_f1 = 0.0937
01/02/2026 17:25:14 - INFO - __main__ -     eval_mcc = 0.2518
01/02/2026 17:25:14 - INFO - __main__ -     eval_precision = 0.1082
01/02/2026 17:25:14 - INFO - __main__ -     eval_recall = 0.0902
epoch 49 loss 0.0262: 100%|█████████████████████| 90/90 [00:13<00:00,  6.45it/s]
100%|██████████████████████████████████████| 1314/1314 [00:03<00:00, 333.93it/s]
01/02/2026 17:25:19 - INFO - __main__ -   ***** Running Test *****
01/02/2026 17:25:19 - INFO - __main__ -     Num examples = 1314
01/02/2026 17:25:19 - INFO - __main__ -     Batch size = 128
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
01/02/2026 17:25:19 - INFO - __main__ -   ***** Test results *****
01/02/2026 17:25:19 - INFO - __main__ -     eval_acc = 0.3227
01/02/2026 17:25:19 - INFO - __main__ -     eval_f1 = 0.097
01/02/2026 17:25:19 - INFO - __main__ -     eval_mcc = 0.28
01/02/2026 17:25:19 - INFO - __main__ -     eval_precision = 0.1097
01/02/2026 17:25:19 - INFO - __main__ -     eval_recall = 0.1021
done writing predictions






!python student_codebert_main.py \
    --alpha 0.7 \
    --output_dir=./saved_models \
    --model_name=soft_distil_model_07.bin \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --data_labels_path=/kaggle/working/processed_data \
    --train_data_file=/kaggle/working/processed_data/merged_train.csv \
    --eval_data_file=/kaggle/working/processed_data/merged_valid.csv \
    --test_data_file=/kaggle/working/processed_data/merged_test.csv \
    --do_train \
    --do_test \
    --block_size 512 \
    --epochs 50 \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456

01/02/2026 17:26:13 - INFO - __main__ -   ***** Running training *****
01/02/2026 17:26:13 - INFO - __main__ -     Num examples = 11508
01/02/2026 17:26:13 - INFO - __main__ -     Num Epochs = 50
01/02/2026 17:26:13 - INFO - __main__ -     Instantaneous batch size per GPU = 8
01/02/2026 17:26:13 - INFO - __main__ -     Total train batch size = 8
01/02/2026 17:26:13 - INFO - __main__ -     Gradient Accumulation steps = 1
01/02/2026 17:26:13 - INFO - __main__ -     Total optimization steps = 71950
epoch 0 loss 4.55212: 100%|████████████████▉| 1438/1439 [11:05<00:00,  2.17it/s]01/02/2026 17:37:18 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 17:37:18 - INFO - __main__ -     Num examples = 1413
01/02/2026 17:37:18 - INFO - __main__ -     Batch size = 8
01/02/2026 17:37:42 - INFO - __main__ -   ***** Eval results *****
01/02/2026 17:37:42 - INFO - __main__ -     best_beta = 0.5
01/02/2026 17:37:42 - INFO - __main__ -     eval_acc = 0.1019
01/02/2026 17:37:42 - INFO - __main__ -     ********************
01/02/2026 17:37:42 - INFO - __main__ -     Best Acc:0.1019
01/02/2026 17:37:42 - INFO - __main__ -     Best Beta:0.5
01/02/2026 17:37:42 - INFO - __main__ -     ********************
01/02/2026 17:37:43 - INFO - __main__ -   Saving model checkpoint to ./saved_models/checkpoint-best-acc/soft_distil_model_07.bin
epoch 0 loss 4.55212: 100%|█████████████████| 1439/1439 [11:29<00:00,  2.09it/s]
epoch 1 loss 3.70081: 100%|████████████████▉| 1438/1439 [11:05<00:00,  2.16it/s]01/02/2026 17:48:48 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 17:48:48 - INFO - __main__ -     Num examples = 1413
01/02/2026 17:48:48 - INFO - __main__ -     Batch size = 8
01/02/2026 17:49:12 - INFO - __main__ -   ***** Eval results *****
01/02/2026 17:49:12 - INFO - __main__ -     best_beta = 0.9
01/02/2026 17:49:12 - INFO - __main__ -     eval_acc = 0.1394
01/02/2026 17:49:12 - INFO - __main__ -     ********************
01/02/2026 17:49:12 - INFO - __main__ -     Best Acc:0.1394
01/02/2026 17:49:12 - INFO - __main__ -     Best Beta:0.9
01/02/2026 17:49:12 - INFO - __main__ -     ********************
01/02/2026 17:49:13 - INFO - __main__ -   Saving model checkpoint to ./saved_models/checkpoint-best-acc/soft_distil_model_07.bin
epoch 1 loss 3.70081: 100%|█████████████████| 1439/1439 [11:30<00:00,  2.08it/s]
epoch 2 loss 3.41922: 100%|████████████████▉| 1438/1439 [11:05<00:00,  2.17it/s]01/02/2026 18:00:18 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 18:00:18 - INFO - __main__ -     Num examples = 1413
01/02/2026 18:00:18 - INFO - __main__ -     Batch size = 8
01/02/2026 18:00:42 - INFO - __main__ -   ***** Eval results *****
01/02/2026 18:00:42 - INFO - __main__ -     best_beta = 0.9
01/02/2026 18:00:42 - INFO - __main__ -     eval_acc = 0.1748
01/02/2026 18:00:42 - INFO - __main__ -     ********************
01/02/2026 18:00:42 - INFO - __main__ -     Best Acc:0.1748
01/02/2026 18:00:42 - INFO - __main__ -     Best Beta:0.9
01/02/2026 18:00:42 - INFO - __main__ -     ********************
01/02/2026 18:00:43 - INFO - __main__ -   Saving model checkpoint to ./saved_models/checkpoint-best-acc/soft_distil_model_07.bin
epoch 2 loss 3.41922: 100%|█████████████████| 1439/1439 [11:30<00:00,  2.08it/s]
epoch 3 loss 3.19157: 100%|████████████████▉| 1438/1439 [11:04<00:00,  2.16it/s]01/02/2026 18:11:48 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 18:11:48 - INFO - __main__ -     Num examples = 1413
01/02/2026 18:11:48 - INFO - __main__ -     Batch size = 8
01/02/2026 18:12:12 - INFO - __main__ -   ***** Eval results *****
01/02/2026 18:12:12 - INFO - __main__ -     best_beta = 0.6
01/02/2026 18:12:12 - INFO - __main__ -     eval_acc = 0.1677
epoch 3 loss 3.19157: 100%|█████████████████| 1439/1439 [11:28<00:00,  2.09it/s]
epoch 4 loss 2.93258: 100%|████████████████▉| 1438/1439 [11:04<00:00,  2.17it/s]01/02/2026 18:23:17 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 18:23:17 - INFO - __main__ -     Num examples = 1413
01/02/2026 18:23:17 - INFO - __main__ -     Batch size = 8
01/02/2026 18:23:41 - INFO - __main__ -   ***** Eval results *****
01/02/2026 18:23:41 - INFO - __main__ -     best_beta = 0.9
01/02/2026 18:23:41 - INFO - __main__ -     eval_acc = 0.1875
01/02/2026 18:23:41 - INFO - __main__ -     ********************
01/02/2026 18:23:41 - INFO - __main__ -     Best Acc:0.1875
01/02/2026 18:23:41 - INFO - __main__ -     Best Beta:0.9
01/02/2026 18:23:41 - INFO - __main__ -     ********************
01/02/2026 18:23:42 - INFO - __main__ -   Saving model checkpoint to ./saved_models/checkpoint-best-acc/soft_distil_model_07.bin
epoch 4 loss 2.93258: 100%|█████████████████| 1439/1439 [11:29<00:00,  2.09it/s]
epoch 5 loss 2.61614: 100%|████████████████▉| 1438/1439 [11:04<00:00,  2.17it/s]01/02/2026 18:34:46 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 18:34:46 - INFO - __main__ -     Num examples = 1413
01/02/2026 18:34:46 - INFO - __main__ -     Batch size = 8
01/02/2026 18:35:10 - INFO - __main__ -   ***** Eval results *****
01/02/2026 18:35:10 - INFO - __main__ -     best_beta = 0.8
01/02/2026 18:35:10 - INFO - __main__ -     eval_acc = 0.1684
epoch 5 loss 2.61614: 100%|█████████████████| 1439/1439 [11:28<00:00,  2.09it/s]
epoch 6 loss 2.2666: 100%|█████████████████▉| 1438/1439 [11:04<00:00,  2.16it/s]01/02/2026 18:46:15 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 18:46:15 - INFO - __main__ -     Num examples = 1413
01/02/2026 18:46:15 - INFO - __main__ -     Batch size = 8
01/02/2026 18:46:39 - INFO - __main__ -   ***** Eval results *****
01/02/2026 18:46:39 - INFO - __main__ -     best_beta = 0.5
01/02/2026 18:46:39 - INFO - __main__ -     eval_acc = 0.1635
epoch 6 loss 2.2666: 100%|██████████████████| 1439/1439 [11:28<00:00,  2.09it/s]
epoch 7 loss 1.92644: 100%|████████████████▉| 1438/1439 [11:04<00:00,  2.17it/s]01/02/2026 18:57:43 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 18:57:43 - INFO - __main__ -     Num examples = 1413
01/02/2026 18:57:43 - INFO - __main__ -     Batch size = 8
01/02/2026 18:58:08 - INFO - __main__ -   ***** Eval results *****
01/02/2026 18:58:08 - INFO - __main__ -     best_beta = 0.7
01/02/2026 18:58:08 - INFO - __main__ -     eval_acc = 0.1635
epoch 7 loss 1.92644: 100%|█████████████████| 1439/1439 [11:28<00:00,  2.09it/s]
epoch 8 loss 1.61902: 100%|████████████████▉| 1438/1439 [11:04<00:00,  2.17it/s]01/02/2026 19:09:12 - INFO - __main__ -   ***** Running evaluation *****
01/02/2026 19:09:12 - INFO - __main__ -     Num examples = 1413
01/02/2026 19:09:12 - INFO - __main__ -     Batch size = 8