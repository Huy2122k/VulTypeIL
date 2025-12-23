12/23/2025 16:53:07 - INFO - __main__ -   Processing Task 1
12/23/2025 16:53:07 - INFO - __main__ -   Loading training data for Task 1
100%|██████████████████████████████████████| 2344/2344 [00:07<00:00, 331.55it/s]
12/23/2025 16:53:15 - INFO - __main__ -   *** Example ***
12/23/2025 16:53:15 - INFO - __main__ -   label: 98
12/23/2025 16:53:15 - INFO - __main__ -   group: 2
12/23/2025 16:53:15 - INFO - __main__ -   *** Example ***
12/23/2025 16:53:15 - INFO - __main__ -   label: 98
12/23/2025 16:53:15 - INFO - __main__ -   group: 2
12/23/2025 16:53:15 - INFO - __main__ -   *** Example ***
12/23/2025 16:53:15 - INFO - __main__ -   label: 98
12/23/2025 16:53:15 - INFO - __main__ -   group: 2
100%|████████████████████████████████████████| 275/275 [00:01<00:00, 263.47it/s]
12/23/2025 16:53:16 - INFO - __main__ -   Training model for Task 1
12/23/2025 16:53:19 - INFO - __main__ -   ***** Running training *****
12/23/2025 16:53:19 - INFO - __main__ -     Num examples = 2344
12/23/2025 16:53:19 - INFO - __main__ -     Num Epochs = 10
12/23/2025 16:53:19 - INFO - __main__ -     Instantaneous batch size per GPU = 8
12/23/2025 16:53:19 - INFO - __main__ -     Total train batch size = 8
12/23/2025 16:53:19 - INFO - __main__ -     Gradient Accumulation steps = 1
12/23/2025 16:53:19 - INFO - __main__ -     Total optimization steps = 2930
epoch 0 loss 5.60162: 100%|███████████████████| 293/293 [00:08<00:00, 33.32it/s]
epoch 1 loss 3.70723: 100%|███████████████████| 293/293 [00:07<00:00, 36.84it/s]
epoch 2 loss 3.02062: 100%|███████████████████| 293/293 [00:07<00:00, 36.71it/s]
epoch 3 loss 2.84175: 100%|███████████████████| 293/293 [00:07<00:00, 36.77it/s]
epoch 4 loss 2.72471: 100%|███████████████████| 293/293 [00:08<00:00, 36.51it/s]
epoch 5 loss 2.63505: 100%|███████████████████| 293/293 [00:08<00:00, 36.61it/s]
epoch 6 loss 2.56499: 100%|███████████████████| 293/293 [00:07<00:00, 36.66it/s]
epoch 7 loss 2.50756: 100%|███████████████████| 293/293 [00:07<00:00, 36.73it/s]
epoch 8 loss 2.46321: 100%|███████████████████| 293/293 [00:07<00:00, 36.87it/s]
epoch 9 loss 2.44621: 100%|██████████████████▉| 292/293 [00:07<00:00, 36.78it/s]12/23/2025 16:54:39 - INFO - __main__ -   ***** Running evaluation *****
12/23/2025 16:54:39 - INFO - __main__ -     Num examples = 275
12/23/2025 16:54:39 - INFO - __main__ -     Batch size = 8
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
12/23/2025 16:54:40 - INFO - __main__ -   ***** Eval results *****
12/23/2025 16:54:40 - INFO - __main__ -     eval_acc = 0.24
12/23/2025 16:54:40 - INFO - __main__ -     eval_f1 = 0.0486
12/23/2025 16:54:40 - INFO - __main__ -     eval_mcc = 0.156
12/23/2025 16:54:40 - INFO - __main__ -     eval_precision = 0.0557
12/23/2025 16:54:40 - INFO - __main__ -     eval_recall = 0.0653
12/23/2025 16:54:40 - INFO - __main__ -     ********************
12/23/2025 16:54:40 - INFO - __main__ -     Best Acc:0.24
12/23/2025 16:54:40 - INFO - __main__ -     ********************
12/23/2025 16:54:40 - INFO - __main__ -   Saving model checkpoint to output_bags/checkpoint-best-acc/model.bin
epoch 9 loss 2.44621: 100%|███████████████████| 293/293 [00:08<00:00, 33.33it/s]
12/23/2025 16:54:41 - INFO - __main__ -   Evaluating model trained on Task 1 on all other tasks
100%|████████████████████████████████████████| 228/228 [00:00<00:00, 304.34it/s]
12/23/2025 16:54:42 - INFO - __main__ -   Testing on Task 1
12/23/2025 16:54:42 - INFO - __main__ -   ***** Running Test *****
12/23/2025 16:54:42 - INFO - __main__ -     Num examples = 228
12/23/2025 16:54:42 - INFO - __main__ -     Batch size = 8
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
12/23/2025 16:54:42 - INFO - __main__ -   ***** Test results *****
12/23/2025 16:54:42 - INFO - __main__ -     eval_acc = 0.1886
12/23/2025 16:54:42 - INFO - __main__ -     eval_f1 = 0.0315
12/23/2025 16:54:42 - INFO - __main__ -     eval_mcc = 0.1262
12/23/2025 16:54:42 - INFO - __main__ -     eval_precision = 0.06
12/23/2025 16:54:42 - INFO - __main__ -     eval_recall = 0.0521
{}
100%|████████████████████████████████████████| 269/269 [00:00<00:00, 294.73it/s]
12/23/2025 16:54:43 - INFO - __main__ -   Testing on Task 2
12/23/2025 16:54:43 - INFO - __main__ -   ***** Running Test *****
12/23/2025 16:54:43 - INFO - __main__ -     Num examples = 269
12/23/2025 16:54:43 - INFO - __main__ -     Batch size = 8
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
12/23/2025 16:54:43 - INFO - __main__ -   ***** Test results *****
12/23/2025 16:54:43 - INFO - __main__ -     eval_acc = 0.1636
12/23/2025 16:54:43 - INFO - __main__ -     eval_f1 = 0.0214
12/23/2025 16:54:43 - INFO - __main__ -     eval_mcc = 0.0886
12/23/2025 16:54:43 - INFO - __main__ -     eval_precision = 0.016
12/23/2025 16:54:43 - INFO - __main__ -     eval_recall = 0.0409
{}
100%|████████████████████████████████████████| 218/218 [00:00<00:00, 316.18it/s]
12/23/2025 16:54:44 - INFO - __main__ -   Testing on Task 3
12/23/2025 16:54:44 - INFO - __main__ -   ***** Running Test *****
12/23/2025 16:54:44 - INFO - __main__ -     Num examples = 218
12/23/2025 16:54:44 - INFO - __main__ -     Batch size = 8
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
12/23/2025 16:54:44 - INFO - __main__ -   ***** Test results *****
12/23/2025 16:54:44 - INFO - __main__ -     eval_acc = 0.0321
12/23/2025 16:54:44 - INFO - __main__ -     eval_f1 = 0.0031
12/23/2025 16:54:44 - INFO - __main__ -     eval_mcc = 0.0002
12/23/2025 16:54:44 - INFO - __main__ -     eval_precision = 0.0019
12/23/2025 16:54:44 - INFO - __main__ -     eval_recall = 0.0081
{}
100%|████████████████████████████████████████| 311/311 [00:00<00:00, 328.35it/s]
12/23/2025 16:54:45 - INFO - __main__ -   Testing on Task 4
12/23/2025 16:54:45 - INFO - __main__ -   ***** Running Test *****
12/23/2025 16:54:45 - INFO - __main__ -     Num examples = 311
12/23/2025 16:54:45 - INFO - __main__ -     Batch size = 8
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
12/23/2025 16:54:45 - INFO - __main__ -   ***** Test results *****
12/23/2025 16:54:45 - INFO - __main__ -     eval_acc = 0.0675
12/23/2025 16:54:45 - INFO - __main__ -     eval_f1 = 0.0053
12/23/2025 16:54:45 - INFO - __main__ -     eval_mcc = 0.0045
12/23/2025 16:54:45 - INFO - __main__ -     eval_precision = 0.0032
12/23/2025 16:54:45 - INFO - __main__ -     eval_recall = 0.0192
{}
100%|████████████████████████████████████████| 288/288 [00:00<00:00, 394.20it/s]
12/23/2025 16:54:46 - INFO - __main__ -   Testing on Task 5
12/23/2025 16:54:46 - INFO - __main__ -   ***** Running Test *****
12/23/2025 16:54:46 - INFO - __main__ -     Num examples = 288
12/23/2025 16:54:46 - INFO - __main__ -     Batch size = 8
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
12/23/2025 16:54:46 - INFO - __main__ -   ***** Test results *****
12/23/2025 16:54:46 - INFO - __main__ -     eval_acc = 0.1146
12/23/2025 16:54:46 - INFO - __main__ -     eval_f1 = 0.0093
12/23/2025 16:54:46 - INFO - __main__ -     eval_mcc = 0.0214
12/23/2025 16:54:46 - INFO - __main__ -     eval_precision = 0.0061
12/23/2025 16:54:46 - INFO - __main__ -     eval_recall = 0.0244
{}
12/23/2025 16:54:46 - INFO - __main__ -   Processing Task 2
12/23/2025 16:54:46 - INFO - __main__ -   Loading training data for Task 2
100%|██████████████████████████████████████| 2293/2293 [00:06<00:00, 332.36it/s]
12/23/2025 16:54:53 - INFO - __main__ -   *** Example ***
12/23/2025 16:54:53 - INFO - __main__ -   label: 201
12/23/2025 16:54:53 - INFO - __main__ -   group: 2
12/23/2025 16:54:53 - INFO - __main__ -   *** Example ***
12/23/2025 16:54:53 - INFO - __main__ -   label: 201
12/23/2025 16:54:53 - INFO - __main__ -   group: 2
12/23/2025 16:54:53 - INFO - __main__ -   *** Example ***
12/23/2025 16:54:53 - INFO - __main__ -   label: 201
12/23/2025 16:54:53 - INFO - __main__ -   group: 2
100%|████████████████████████████████████████| 285/285 [00:00<00:00, 352.00it/s]
12/23/2025 16:54:54 - INFO - __main__ -   Training model for Task 2
12/23/2025 16:54:54 - INFO - __main__ -   ***** Running training *****
12/23/2025 16:54:54 - INFO - __main__ -     Num examples = 2293
12/23/2025 16:54:54 - INFO - __main__ -     Num Epochs = 10
12/23/2025 16:54:54 - INFO - __main__ -     Instantaneous batch size per GPU = 8
12/23/2025 16:54:54 - INFO - __main__ -     Total train batch size = 8
12/23/2025 16:54:54 - INFO - __main__ -     Gradient Accumulation steps = 1
12/23/2025 16:54:54 - INFO - __main__ -     Total optimization steps = 2870
epoch 0 loss 3.88005: 100%|███████████████████| 287/287 [00:07<00:00, 36.56it/s]
epoch 1 loss 3.49718: 100%|███████████████████| 287/287 [00:07<00:00, 36.68it/s]
epoch 2 loss 3.18794: 100%|███████████████████| 287/287 [00:07<00:00, 36.71it/s]
epoch 3 loss 2.9577: 100%|████████████████████| 287/287 [00:07<00:00, 36.79it/s]
epoch 4 loss 2.80753: 100%|███████████████████| 287/287 [00:07<00:00, 36.74it/s]
epoch 5 loss 2.67809: 100%|███████████████████| 287/287 [00:07<00:00, 36.82it/s]
epoch 6 loss 2.59325: 100%|███████████████████| 287/287 [00:07<00:00, 36.78it/s]
epoch 7 loss 2.51106: 100%|███████████████████| 287/287 [00:07<00:00, 36.82it/s]
epoch 8 loss 2.45939: 100%|███████████████████| 287/287 [00:07<00:00, 36.75it/s]
epoch 9 loss 2.42919:  99%|██████████████████▊| 284/287 [00:07<00:00, 36.53it/s]12/23/2025 16:56:12 - INFO - __main__ -   ***** Running evaluation *****
12/23/2025 16:56:12 - INFO - __main__ -     Num examples = 285
12/23/2025 16:56:12 - INFO - __main__ -     Batch size = 8
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
12/23/2025 16:56:12 - INFO - __main__ -   ***** Eval results *****
12/23/2025 16:56:12 - INFO - __main__ -     eval_acc = 0.1263
12/23/2025 16:56:12 - INFO - __main__ -     eval_f1 = 0.028
12/23/2025 16:56:12 - INFO - __main__ -     eval_mcc = 0.0575
12/23/2025 16:56:12 - INFO - __main__ -     eval_precision = 0.0362
12/23/2025 16:56:12 - INFO - __main__ -     eval_recall = 0.0478
12/23/2025 16:56:12 - INFO - __main__ -     ********************
12/23/2025 16:56:12 - INFO - __main__ -     Best Acc:0.1263
12/23/2025 16:56:12 - INFO - __main__ -     ********************
12/23/2025 16:56:13 - INFO - __main__ -   Saving model checkpoint to output_bags/checkpoint-best-acc/model.bin
epoch 9 loss 2.42919: 100%|███████████████████| 287/287 [00:09<00:00, 31.75it/s]
12/23/2025 16:56:14 - INFO - __main__ -   Evaluating model trained on Task 2 on all other tasks
100%|████████████████████████████████████████| 228/228 [00:00<00:00, 355.41it/s]
12/23/2025 16:56:15 - INFO - __main__ -   Testing on Task 1
12/23/2025 16:56:15 - INFO - __main__ -   ***** Running Test *****
12/23/2025 16:56:15 - INFO - __main__ -     Num examples = 228
12/23/2025 16:56:15 - INFO - __main__ -     Batch size = 8
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
12/23/2025 16:56:15 - INFO - __main__ -   ***** Test results *****
12/23/2025 16:56:15 - INFO - __main__ -     eval_acc = 0.1009
12/23/2025 16:56:15 - INFO - __main__ -     eval_f1 = 0.0221
12/23/2025 16:56:15 - INFO - __main__ -     eval_mcc = 0.0687
12/23/2025 16:56:15 - INFO - __main__ -     eval_precision = 0.059
12/23/2025 16:56:15 - INFO - __main__ -     eval_recall = 0.049
{}
100%|████████████████████████████████████████| 269/269 [00:00<00:00, 351.64it/s]
12/23/2025 16:56:16 - INFO - __main__ -   Testing on Task 2
12/23/2025 16:56:16 - INFO - __main__ -   ***** Running Test *****
12/23/2025 16:56:16 - INFO - __main__ -     Num examples = 269
12/23/2025 16:56:16 - INFO - __main__ -     Batch size = 8
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
12/23/2025 16:56:16 - INFO - __main__ -   ***** Test results *****
12/23/2025 16:56:16 - INFO - __main__ -     eval_acc = 0.2379
12/23/2025 16:56:16 - INFO - __main__ -     eval_f1 = 0.032
12/23/2025 16:56:16 - INFO - __main__ -     eval_mcc = 0.1497
12/23/2025 16:56:16 - INFO - __main__ -     eval_precision = 0.0326
12/23/2025 16:56:16 - INFO - __main__ -     eval_recall = 0.0487
{}
100%|████████████████████████████████████████| 218/218 [00:00<00:00, 379.07it/s]
12/23/2025 16:56:16 - INFO - __main__ -   Testing on Task 3
12/23/2025 16:56:16 - INFO - __main__ -   ***** Running Test *****
12/23/2025 16:56:16 - INFO - __main__ -     Num examples = 218
12/23/2025 16:56:16 - INFO - __main__ -     Batch size = 8
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
12/23/2025 16:56:17 - INFO - __main__ -   ***** Test results *****
12/23/2025 16:56:17 - INFO - __main__ -     eval_acc = 0.1055
12/23/2025 16:56:17 - INFO - __main__ -     eval_f1 = 0.0078
12/23/2025 16:56:17 - INFO - __main__ -     eval_mcc = 0.0212
12/23/2025 16:56:17 - INFO - __main__ -     eval_precision = 0.0048
12/23/2025 16:56:17 - INFO - __main__ -     eval_recall = 0.0213
{}
100%|████████████████████████████████████████| 311/311 [00:00<00:00, 384.84it/s]
12/23/2025 16:56:17 - INFO - __main__ -   Testing on Task 4
12/23/2025 16:56:17 - INFO - __main__ -   ***** Running Test *****
12/23/2025 16:56:17 - INFO - __main__ -     Num examples = 311
12/23/2025 16:56:17 - INFO - __main__ -     Batch size = 8
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
12/23/2025 16:56:18 - INFO - __main__ -   ***** Test results *****
12/23/2025 16:56:18 - INFO - __main__ -     eval_acc = 0.119
12/23/2025 16:56:18 - INFO - __main__ -     eval_f1 = 0.0141
12/23/2025 16:56:18 - INFO - __main__ -     eval_mcc = 0.0267
12/23/2025 16:56:18 - INFO - __main__ -     eval_precision = 0.0122
12/23/2025 16:56:18 - INFO - __main__ -     eval_recall = 0.0259
{}
100%|████████████████████████████████████████| 288/288 [00:00<00:00, 465.89it/s]
12/23/2025 16:56:18 - INFO - __main__ -   Testing on Task 5
12/23/2025 16:56:18 - INFO - __main__ -   ***** Running Test *****
12/23/2025 16:56:18 - INFO - __main__ -     Num examples = 288
12/23/2025 16:56:18 - INFO - __main__ -     Batch size = 8
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
12/23/2025 16:56:19 - INFO - __main__ -   ***** Test results *****
12/23/2025 16:56:19 - INFO - __main__ -     eval_acc = 0.1007
12/23/2025 16:56:19 - INFO - __main__ -     eval_f1 = 0.0117
12/23/2025 16:56:19 - INFO - __main__ -     eval_mcc = 0.0274
12/23/2025 16:56:19 - INFO - __main__ -     eval_precision = 0.0115
12/23/2025 16:56:19 - INFO - __main__ -     eval_recall = 0.0284
{}
12/23/2025 16:56:19 - INFO - __main__ -   Processing Task 3
12/23/2025 16:56:19 - INFO - __main__ -   Loading training data for Task 3
100%|██████████████████████████████████████| 2350/2350 [00:06<00:00, 348.02it/s]
12/23/2025 16:56:25 - INFO - __main__ -   *** Example ***
12/23/2025 16:56:25 - INFO - __main__ -   label: 98
12/23/2025 16:56:25 - INFO - __main__ -   group: 2
12/23/2025 16:56:25 - INFO - __main__ -   *** Example ***
12/23/2025 16:56:25 - INFO - __main__ -   label: 56
12/23/2025 16:56:25 - INFO - __main__ -   group: 2
12/23/2025 16:56:25 - INFO - __main__ -   *** Example ***
12/23/2025 16:56:25 - INFO - __main__ -   label: 56
12/23/2025 16:56:25 - INFO - __main__ -   group: 2
100%|████████████████████████████████████████| 279/279 [00:00<00:00, 398.31it/s]
12/23/2025 16:56:26 - INFO - __main__ -   Training model for Task 3
12/23/2025 16:56:26 - INFO - __main__ -   ***** Running training *****
12/23/2025 16:56:26 - INFO - __main__ -     Num examples = 2350
12/23/2025 16:56:26 - INFO - __main__ -     Num Epochs = 10
12/23/2025 16:56:26 - INFO - __main__ -     Instantaneous batch size per GPU = 8
12/23/2025 16:56:26 - INFO - __main__ -     Total train batch size = 8
12/23/2025 16:56:26 - INFO - __main__ -     Gradient Accumulation steps = 1
12/23/2025 16:56:26 - INFO - __main__ -     Total optimization steps = 2940
epoch 0 loss 4.63578: 100%|███████████████████| 294/294 [00:08<00:00, 36.68it/s]
epoch 1 loss 4.06173: 100%|███████████████████| 294/294 [00:08<00:00, 36.75it/s]
epoch 2 loss 3.57907: 100%|███████████████████| 294/294 [00:08<00:00, 36.66it/s]
epoch 3 loss 3.26263: 100%|███████████████████| 294/294 [00:08<00:00, 36.74it/s]
epoch 4 loss 3.03852: 100%|███████████████████| 294/294 [00:07<00:00, 36.77it/s]
epoch 5 loss 2.87062: 100%|███████████████████| 294/294 [00:08<00:00, 36.74it/s]
epoch 6 loss 2.7455: 100%|████████████████████| 294/294 [00:07<00:00, 36.75it/s]
epoch 7 loss 2.65808: 100%|███████████████████| 294/294 [00:07<00:00, 36.75it/s]
epoch 8 loss 2.58504: 100%|███████████████████| 294/294 [00:07<00:00, 36.76it/s]
epoch 9 loss 2.54971:  99%|██████████████████▊| 292/294 [00:07<00:00, 36.73it/s]12/23/2025 16:57:46 - INFO - __main__ -   ***** Running evaluation *****
12/23/2025 16:57:46 - INFO - __main__ -     Num examples = 279
12/23/2025 16:57:46 - INFO - __main__ -     Batch size = 8
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
12/23/2025 16:57:46 - INFO - __main__ -   ***** Eval results *****
12/23/2025 16:57:46 - INFO - __main__ -     eval_acc = 0.1685
12/23/2025 16:57:46 - INFO - __main__ -     eval_f1 = 0.0247
12/23/2025 16:57:46 - INFO - __main__ -     eval_mcc = 0.0408
12/23/2025 16:57:46 - INFO - __main__ -     eval_precision = 0.0209
12/23/2025 16:57:46 - INFO - __main__ -     eval_recall = 0.0365
12/23/2025 16:57:46 - INFO - __main__ -     ********************
12/23/2025 16:57:46 - INFO - __main__ -     Best Acc:0.1685
12/23/2025 16:57:46 - INFO - __main__ -     ********************
12/23/2025 16:57:47 - INFO - __main__ -   Saving model checkpoint to output_bags/checkpoint-best-acc/model.bin
epoch 9 loss 2.54971: 100%|███████████████████| 294/294 [00:09<00:00, 32.02it/s]
12/23/2025 16:57:48 - INFO - __main__ -   Evaluating model trained on Task 3 on all other tasks
100%|████████████████████████████████████████| 228/228 [00:00<00:00, 355.65it/s]
12/23/2025 16:57:49 - INFO - __main__ -   Testing on Task 1
12/23/2025 16:57:49 - INFO - __main__ -   ***** Running Test *****
12/23/2025 16:57:49 - INFO - __main__ -     Num examples = 228
12/23/2025 16:57:49 - INFO - __main__ -     Batch size = 8
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
12/23/2025 16:57:49 - INFO - __main__ -   ***** Test results *****
12/23/2025 16:57:49 - INFO - __main__ -     eval_acc = 0.1009
12/23/2025 16:57:49 - INFO - __main__ -     eval_f1 = 0.0178
12/23/2025 16:57:49 - INFO - __main__ -     eval_mcc = 0.0596
12/23/2025 16:57:49 - INFO - __main__ -     eval_precision = 0.0191
12/23/2025 16:57:49 - INFO - __main__ -     eval_recall = 0.0437
{}
100%|████████████████████████████████████████| 269/269 [00:00<00:00, 344.08it/s]
12/23/2025 16:57:50 - INFO - __main__ -   Testing on Task 2
12/23/2025 16:57:50 - INFO - __main__ -   ***** Running Test *****
12/23/2025 16:57:50 - INFO - __main__ -     Num examples = 269
12/23/2025 16:57:50 - INFO - __main__ -     Batch size = 8
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
12/23/2025 16:57:50 - INFO - __main__ -   ***** Test results *****
12/23/2025 16:57:50 - INFO - __main__ -     eval_acc = 0.223
12/23/2025 16:57:50 - INFO - __main__ -     eval_f1 = 0.0197
12/23/2025 16:57:50 - INFO - __main__ -     eval_mcc = 0.1267
12/23/2025 16:57:50 - INFO - __main__ -     eval_precision = 0.0149
12/23/2025 16:57:50 - INFO - __main__ -     eval_recall = 0.035
{}
100%|████████████████████████████████████████| 218/218 [00:00<00:00, 376.65it/s]
12/23/2025 16:57:51 - INFO - __main__ -   Testing on Task 3
12/23/2025 16:57:51 - INFO - __main__ -   ***** Running Test *****
12/23/2025 16:57:51 - INFO - __main__ -     Num examples = 218
12/23/2025 16:57:51 - INFO - __main__ -     Batch size = 8
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
12/23/2025 16:57:51 - INFO - __main__ -   ***** Test results *****
12/23/2025 16:57:51 - INFO - __main__ -     eval_acc = 0.1514
12/23/2025 16:57:51 - INFO - __main__ -     eval_f1 = 0.0229
12/23/2025 16:57:51 - INFO - __main__ -     eval_mcc = 0.0792
12/23/2025 16:57:51 - INFO - __main__ -     eval_precision = 0.0304
12/23/2025 16:57:51 - INFO - __main__ -     eval_recall = 0.0363
{}
100%|████████████████████████████████████████| 311/311 [00:00<00:00, 388.05it/s]
12/23/2025 16:57:52 - INFO - __main__ -   Testing on Task 4
12/23/2025 16:57:52 - INFO - __main__ -   ***** Running Test *****
12/23/2025 16:57:52 - INFO - __main__ -     Num examples = 311
12/23/2025 16:57:52 - INFO - __main__ -     Batch size = 8
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
12/23/2025 16:57:52 - INFO - __main__ -   ***** Test results *****
12/23/2025 16:57:52 - INFO - __main__ -     eval_acc = 0.1158
12/23/2025 16:57:52 - INFO - __main__ -     eval_f1 = 0.0114
12/23/2025 16:57:52 - INFO - __main__ -     eval_mcc = 0.0058
12/23/2025 16:57:52 - INFO - __main__ -     eval_precision = 0.0099
12/23/2025 16:57:52 - INFO - __main__ -     eval_recall = 0.0248
{}
100%|████████████████████████████████████████| 288/288 [00:00<00:00, 457.26it/s]
12/23/2025 16:57:52 - INFO - __main__ -   Testing on Task 5
12/23/2025 16:57:52 - INFO - __main__ -   ***** Running Test *****
12/23/2025 16:57:52 - INFO - __main__ -     Num examples = 288
12/23/2025 16:57:52 - INFO - __main__ -     Batch size = 8
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
12/23/2025 16:57:53 - INFO - __main__ -   ***** Test results *****
12/23/2025 16:57:53 - INFO - __main__ -     eval_acc = 0.1042
12/23/2025 16:57:53 - INFO - __main__ -     eval_f1 = 0.0114
12/23/2025 16:57:53 - INFO - __main__ -     eval_mcc = 0.0402
12/23/2025 16:57:53 - INFO - __main__ -     eval_precision = 0.0172
12/23/2025 16:57:53 - INFO - __main__ -     eval_recall = 0.031
{}
12/23/2025 16:57:53 - INFO - __main__ -   Processing Task 4
12/23/2025 16:57:53 - INFO - __main__ -   Loading training data for Task 4
100%|██████████████████████████████████████| 2270/2270 [00:06<00:00, 340.08it/s]
12/23/2025 16:58:00 - INFO - __main__ -   *** Example ***
12/23/2025 16:58:00 - INFO - __main__ -   label: 31
12/23/2025 16:58:00 - INFO - __main__ -   group: 2
12/23/2025 16:58:00 - INFO - __main__ -   *** Example ***
12/23/2025 16:58:00 - INFO - __main__ -   label: 272
12/23/2025 16:58:00 - INFO - __main__ -   group: 2
12/23/2025 16:58:00 - INFO - __main__ -   *** Example ***
12/23/2025 16:58:00 - INFO - __main__ -   label: 272
12/23/2025 16:58:00 - INFO - __main__ -   group: 2
100%|████████████████████████████████████████| 266/266 [00:00<00:00, 311.24it/s]
12/23/2025 16:58:01 - INFO - __main__ -   Training model for Task 4
12/23/2025 16:58:01 - INFO - __main__ -   ***** Running training *****
12/23/2025 16:58:01 - INFO - __main__ -     Num examples = 2270
12/23/2025 16:58:01 - INFO - __main__ -     Num Epochs = 10
12/23/2025 16:58:01 - INFO - __main__ -     Instantaneous batch size per GPU = 8
12/23/2025 16:58:01 - INFO - __main__ -     Total train batch size = 8
12/23/2025 16:58:01 - INFO - __main__ -     Gradient Accumulation steps = 1
12/23/2025 16:58:01 - INFO - __main__ -     Total optimization steps = 2840
epoch 0 loss 4.04776: 100%|███████████████████| 284/284 [00:07<00:00, 36.69it/s]
epoch 1 loss 3.71405: 100%|███████████████████| 284/284 [00:07<00:00, 36.77it/s]
epoch 2 loss 3.37126: 100%|███████████████████| 284/284 [00:07<00:00, 36.74it/s]
epoch 3 loss 3.12565: 100%|███████████████████| 284/284 [00:07<00:00, 36.74it/s]
epoch 4 loss 2.93087: 100%|███████████████████| 284/284 [00:07<00:00, 36.76it/s]
epoch 5 loss 2.78675: 100%|███████████████████| 284/284 [00:07<00:00, 36.80it/s]
epoch 6 loss 2.65999: 100%|███████████████████| 284/284 [00:07<00:00, 36.74it/s]
epoch 7 loss 2.57772: 100%|███████████████████| 284/284 [00:07<00:00, 36.76it/s]
epoch 8 loss 2.5108: 100%|████████████████████| 284/284 [00:07<00:00, 36.75it/s]
epoch 9 loss 2.47219:  99%|██████████████████▋| 280/284 [00:07<00:00, 36.76it/s]12/23/2025 16:59:18 - INFO - __main__ -   ***** Running evaluation *****
12/23/2025 16:59:18 - INFO - __main__ -     Num examples = 266
12/23/2025 16:59:18 - INFO - __main__ -     Batch size = 8
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
12/23/2025 16:59:18 - INFO - __main__ -   ***** Eval results *****
12/23/2025 16:59:18 - INFO - __main__ -     eval_acc = 0.1429
12/23/2025 16:59:18 - INFO - __main__ -     eval_f1 = 0.0342
12/23/2025 16:59:18 - INFO - __main__ -     eval_mcc = 0.09
12/23/2025 16:59:18 - INFO - __main__ -     eval_precision = 0.0318
12/23/2025 16:59:18 - INFO - __main__ -     eval_recall = 0.0576
12/23/2025 16:59:18 - INFO - __main__ -     ********************
12/23/2025 16:59:18 - INFO - __main__ -     Best Acc:0.1429
12/23/2025 16:59:18 - INFO - __main__ -     ********************
12/23/2025 16:59:19 - INFO - __main__ -   Saving model checkpoint to output_bags/checkpoint-best-acc/model.bin
epoch 9 loss 2.47219: 100%|███████████████████| 284/284 [00:08<00:00, 31.78it/s]
12/23/2025 16:59:20 - INFO - __main__ -   Evaluating model trained on Task 4 on all other tasks
100%|████████████████████████████████████████| 228/228 [00:00<00:00, 341.48it/s]
12/23/2025 16:59:20 - INFO - __main__ -   Testing on Task 1
12/23/2025 16:59:20 - INFO - __main__ -   ***** Running Test *****
12/23/2025 16:59:20 - INFO - __main__ -     Num examples = 228
12/23/2025 16:59:20 - INFO - __main__ -     Batch size = 8
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
12/23/2025 16:59:20 - INFO - __main__ -   ***** Test results *****
12/23/2025 16:59:20 - INFO - __main__ -     eval_acc = 0.0702
12/23/2025 16:59:20 - INFO - __main__ -     eval_f1 = 0.0173
12/23/2025 16:59:20 - INFO - __main__ -     eval_mcc = 0.0486
12/23/2025 16:59:20 - INFO - __main__ -     eval_precision = 0.0246
12/23/2025 16:59:20 - INFO - __main__ -     eval_recall = 0.041
{}
100%|████████████████████████████████████████| 269/269 [00:00<00:00, 345.00it/s]
12/23/2025 16:59:21 - INFO - __main__ -   Testing on Task 2
12/23/2025 16:59:21 - INFO - __main__ -   ***** Running Test *****
12/23/2025 16:59:21 - INFO - __main__ -     Num examples = 269
12/23/2025 16:59:21 - INFO - __main__ -     Batch size = 8
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
12/23/2025 16:59:22 - INFO - __main__ -   ***** Test results *****
12/23/2025 16:59:22 - INFO - __main__ -     eval_acc = 0.1822
12/23/2025 16:59:22 - INFO - __main__ -     eval_f1 = 0.018
12/23/2025 16:59:22 - INFO - __main__ -     eval_mcc = 0.0876
12/23/2025 16:59:22 - INFO - __main__ -     eval_precision = 0.019
12/23/2025 16:59:22 - INFO - __main__ -     eval_recall = 0.0289
{}
100%|████████████████████████████████████████| 218/218 [00:00<00:00, 354.54it/s]
12/23/2025 16:59:22 - INFO - __main__ -   Testing on Task 3
12/23/2025 16:59:22 - INFO - __main__ -   ***** Running Test *****
12/23/2025 16:59:22 - INFO - __main__ -     Num examples = 218
12/23/2025 16:59:22 - INFO - __main__ -     Batch size = 8
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
12/23/2025 16:59:22 - INFO - __main__ -   ***** Test results *****
12/23/2025 16:59:22 - INFO - __main__ -     eval_acc = 0.1101
12/23/2025 16:59:22 - INFO - __main__ -     eval_f1 = 0.0093
12/23/2025 16:59:22 - INFO - __main__ -     eval_mcc = 0.0266
12/23/2025 16:59:22 - INFO - __main__ -     eval_precision = 0.0058
12/23/2025 16:59:22 - INFO - __main__ -     eval_recall = 0.0302
{}
100%|████████████████████████████████████████| 311/311 [00:01<00:00, 295.82it/s]
12/23/2025 16:59:23 - INFO - __main__ -   Testing on Task 4
12/23/2025 16:59:23 - INFO - __main__ -   ***** Running Test *****
12/23/2025 16:59:23 - INFO - __main__ -     Num examples = 311
12/23/2025 16:59:23 - INFO - __main__ -     Batch size = 8
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
12/23/2025 16:59:24 - INFO - __main__ -   ***** Test results *****
12/23/2025 16:59:24 - INFO - __main__ -     eval_acc = 0.1511
12/23/2025 16:59:24 - INFO - __main__ -     eval_f1 = 0.027
12/23/2025 16:59:24 - INFO - __main__ -     eval_mcc = 0.0626
12/23/2025 16:59:24 - INFO - __main__ -     eval_precision = 0.0282
12/23/2025 16:59:24 - INFO - __main__ -     eval_recall = 0.0417
{}
100%|████████████████████████████████████████| 288/288 [00:00<00:00, 477.81it/s]
12/23/2025 16:59:24 - INFO - __main__ -   Testing on Task 5
12/23/2025 16:59:24 - INFO - __main__ -   ***** Running Test *****
12/23/2025 16:59:24 - INFO - __main__ -     Num examples = 288
12/23/2025 16:59:24 - INFO - __main__ -     Batch size = 8
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
12/23/2025 16:59:25 - INFO - __main__ -   ***** Test results *****
12/23/2025 16:59:25 - INFO - __main__ -     eval_acc = 0.1146
12/23/2025 16:59:25 - INFO - __main__ -     eval_f1 = 0.0139
12/23/2025 16:59:25 - INFO - __main__ -     eval_mcc = 0.0436
12/23/2025 16:59:25 - INFO - __main__ -     eval_precision = 0.0112
12/23/2025 16:59:25 - INFO - __main__ -     eval_recall = 0.0294
{}
12/23/2025 16:59:25 - INFO - __main__ -   Processing Task 5
12/23/2025 16:59:25 - INFO - __main__ -   Loading training data for Task 5
100%|██████████████████████████████████████| 2251/2251 [00:05<00:00, 380.64it/s]
12/23/2025 16:59:31 - INFO - __main__ -   *** Example ***
12/23/2025 16:59:31 - INFO - __main__ -   label: 85
12/23/2025 16:59:31 - INFO - __main__ -   group: 2
12/23/2025 16:59:31 - INFO - __main__ -   *** Example ***
12/23/2025 16:59:31 - INFO - __main__ -   label: 85
12/23/2025 16:59:31 - INFO - __main__ -   group: 2
12/23/2025 16:59:31 - INFO - __main__ -   *** Example ***
12/23/2025 16:59:31 - INFO - __main__ -   label: 58
12/23/2025 16:59:31 - INFO - __main__ -   group: 2
100%|████████████████████████████████████████| 308/308 [00:00<00:00, 378.99it/s]
12/23/2025 16:59:32 - INFO - __main__ -   Training model for Task 5
12/23/2025 16:59:32 - INFO - __main__ -   ***** Running training *****
12/23/2025 16:59:32 - INFO - __main__ -     Num examples = 2251
12/23/2025 16:59:32 - INFO - __main__ -     Num Epochs = 10
12/23/2025 16:59:32 - INFO - __main__ -     Instantaneous batch size per GPU = 8
12/23/2025 16:59:32 - INFO - __main__ -     Total train batch size = 8
12/23/2025 16:59:32 - INFO - __main__ -     Gradient Accumulation steps = 1
12/23/2025 16:59:32 - INFO - __main__ -     Total optimization steps = 2820
epoch 0 loss 4.48982: 100%|███████████████████| 282/282 [00:07<00:00, 36.67it/s]
epoch 1 loss 3.93684: 100%|███████████████████| 282/282 [00:07<00:00, 36.76it/s]
epoch 2 loss 3.48274: 100%|███████████████████| 282/282 [00:07<00:00, 36.73it/s]
epoch 3 loss 3.15959: 100%|███████████████████| 282/282 [00:07<00:00, 36.69it/s]
epoch 4 loss 2.93981: 100%|███████████████████| 282/282 [00:07<00:00, 36.79it/s]
epoch 5 loss 2.77645: 100%|███████████████████| 282/282 [00:07<00:00, 36.84it/s]
epoch 6 loss 2.65231: 100%|███████████████████| 282/282 [00:07<00:00, 36.78it/s]
epoch 7 loss 2.56025: 100%|███████████████████| 282/282 [00:07<00:00, 36.77it/s]
epoch 8 loss 2.50086: 100%|███████████████████| 282/282 [00:07<00:00, 36.76it/s]
epoch 9 loss 2.45674:  99%|██████████████████▊| 280/282 [00:07<00:00, 37.00it/s]12/23/2025 17:00:48 - INFO - __main__ -   ***** Running evaluation *****
12/23/2025 17:00:48 - INFO - __main__ -     Num examples = 308
12/23/2025 17:00:48 - INFO - __main__ -     Batch size = 8
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
12/23/2025 17:00:48 - INFO - __main__ -   ***** Eval results *****
12/23/2025 17:00:48 - INFO - __main__ -     eval_acc = 0.2045
12/23/2025 17:00:48 - INFO - __main__ -     eval_f1 = 0.0282
12/23/2025 17:00:48 - INFO - __main__ -     eval_mcc = 0.1295
12/23/2025 17:00:48 - INFO - __main__ -     eval_precision = 0.0254
12/23/2025 17:00:48 - INFO - __main__ -     eval_recall = 0.0414
12/23/2025 17:00:48 - INFO - __main__ -     ********************
12/23/2025 17:00:48 - INFO - __main__ -     Best Acc:0.2045
12/23/2025 17:00:48 - INFO - __main__ -     ********************
12/23/2025 17:00:49 - INFO - __main__ -   Saving model checkpoint to output_bags/checkpoint-best-acc/model.bin
epoch 9 loss 2.45674: 100%|███████████████████| 282/282 [00:08<00:00, 31.65it/s]
12/23/2025 17:00:50 - INFO - __main__ -   Evaluating model trained on Task 5 on all other tasks
100%|████████████████████████████████████████| 228/228 [00:00<00:00, 357.33it/s]
12/23/2025 17:00:51 - INFO - __main__ -   Testing on Task 1
12/23/2025 17:00:51 - INFO - __main__ -   ***** Running Test *****
12/23/2025 17:00:51 - INFO - __main__ -     Num examples = 228
12/23/2025 17:00:51 - INFO - __main__ -     Batch size = 8
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
12/23/2025 17:00:51 - INFO - __main__ -   ***** Test results *****
12/23/2025 17:00:51 - INFO - __main__ -     eval_acc = 0.0921
12/23/2025 17:00:51 - INFO - __main__ -     eval_f1 = 0.0197
12/23/2025 17:00:51 - INFO - __main__ -     eval_mcc = 0.0179
12/23/2025 17:00:51 - INFO - __main__ -     eval_precision = 0.0375
12/23/2025 17:00:51 - INFO - __main__ -     eval_recall = 0.0238
{}
100%|████████████████████████████████████████| 269/269 [00:00<00:00, 353.47it/s]
12/23/2025 17:00:52 - INFO - __main__ -   Testing on Task 2
12/23/2025 17:00:52 - INFO - __main__ -   ***** Running Test *****
12/23/2025 17:00:52 - INFO - __main__ -     Num examples = 269
12/23/2025 17:00:52 - INFO - __main__ -     Batch size = 8
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
12/23/2025 17:00:52 - INFO - __main__ -   ***** Test results *****
12/23/2025 17:00:52 - INFO - __main__ -     eval_acc = 0.1004
12/23/2025 17:00:52 - INFO - __main__ -     eval_f1 = 0.0207
12/23/2025 17:00:52 - INFO - __main__ -     eval_mcc = 0.0307
12/23/2025 17:00:52 - INFO - __main__ -     eval_precision = 0.0355
12/23/2025 17:00:52 - INFO - __main__ -     eval_recall = 0.0285
{}
100%|████████████████████████████████████████| 218/218 [00:00<00:00, 358.21it/s]
12/23/2025 17:00:53 - INFO - __main__ -   Testing on Task 3
12/23/2025 17:00:53 - INFO - __main__ -   ***** Running Test *****
12/23/2025 17:00:53 - INFO - __main__ -     Num examples = 218
12/23/2025 17:00:53 - INFO - __main__ -     Batch size = 8
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
12/23/2025 17:00:53 - INFO - __main__ -   ***** Test results *****
12/23/2025 17:00:53 - INFO - __main__ -     eval_acc = 0.0505
12/23/2025 17:00:53 - INFO - __main__ -     eval_f1 = 0.0116
12/23/2025 17:00:53 - INFO - __main__ -     eval_mcc = 0.0238
12/23/2025 17:00:53 - INFO - __main__ -     eval_precision = 0.027
12/23/2025 17:00:53 - INFO - __main__ -     eval_recall = 0.0283
{}
100%|████████████████████████████████████████| 311/311 [00:00<00:00, 386.41it/s]
12/23/2025 17:00:54 - INFO - __main__ -   Testing on Task 4
12/23/2025 17:00:54 - INFO - __main__ -   ***** Running Test *****
12/23/2025 17:00:54 - INFO - __main__ -     Num examples = 311
12/23/2025 17:00:54 - INFO - __main__ -     Batch size = 8
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
12/23/2025 17:00:54 - INFO - __main__ -   ***** Test results *****
12/23/2025 17:00:54 - INFO - __main__ -     eval_acc = 0.1093
12/23/2025 17:00:54 - INFO - __main__ -     eval_f1 = 0.022
12/23/2025 17:00:54 - INFO - __main__ -     eval_mcc = 0.0557
12/23/2025 17:00:54 - INFO - __main__ -     eval_precision = 0.0206
12/23/2025 17:00:54 - INFO - __main__ -     eval_recall = 0.0306
{}
100%|████████████████████████████████████████| 288/288 [00:00<00:00, 489.57it/s]
12/23/2025 17:00:54 - INFO - __main__ -   Testing on Task 5
12/23/2025 17:00:54 - INFO - __main__ -   ***** Running Test *****
12/23/2025 17:00:54 - INFO - __main__ -     Num examples = 288
12/23/2025 17:00:54 - INFO - __main__ -     Batch size = 8
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
12/23/2025 17:00:55 - INFO - __main__ -   ***** Test results *****
12/23/2025 17:00:55 - INFO - __main__ -     eval_acc = 0.1424
12/23/2025 17:00:55 - INFO - __main__ -     eval_f1 = 0.0254
12/23/2025 17:00:55 - INFO - __main__ -     eval_mcc = 0.0749
12/23/2025 17:00:55 - INFO - __main__ -     eval_precision = 0.0252
12/23/2025 17:00:55 - INFO - __main__ -     eval_recall = 0.0312
{}