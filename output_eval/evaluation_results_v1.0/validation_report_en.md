# Meituan Review Model Validation Report (v6.0)

## 1. Basic Information
- Total Validation Samples: 65
- Successfully Parsed Samples: 36
- JSON Parsing Success Rate: 55.38%
- Inference Device: cuda:0 | Batch Size: 16
- Validation Time: 2025-08-23 18:08:36

## 2. Sentiment Classification (Main Task)
| Metric               | Value       |
|----------------------|-------------|
| Accuracy             | 0.9722 |
| Weighted F1 Score    | 0.9725 |
| Macro F1 Score       | 0.9694 |

### Classification Details
- Negative Recall: 1.0000
- Positive Recall: 0.9583
- Negative Precision: 0.9231
- Positive Precision: 1.0000

## 3. Response Quality (v6.0 Key Optimization Target)
| Metric                     | Value                 |
|----------------------------|-----------------------|
| Avg. Core Word Coverage    | 0.9583 |
| High Coverage Rate (≥50%)  | 97.22% |
| Avg. Response Length       | 28.0 chars |
| Valid Response Rate (≥10 chars) | 100.00% |

## 4. Key Issue Extraction
- Avg. ROUGE-L: 0.9691

## 5. Format Error Analysis
| Error Type               | Sample Count | Percentage |
| model_error: invalid_output | 12          | 41.4      %
| json_error: Extra data: line 1 c | 6           | 20.7      %
| no_json_structure    | 4           | 13.8      %
| json_error: Extra data: line 6 c | 3           | 10.3      %
| json_error: Extra data: line 2 c | 2           | 6.9       %
| missing_keys: {'response_to_customer'} | 2           | 6.9       %

## 6. Confusion Matrix
| True \ Predicted | Negative(0) | Positive(1) | Total |
|-------------------|-------------|-------------|-------|
| Negative(0)       | 12 | 0 | 12 |
| Positive(1)       | 1 | 23 | 24 |
| Total             | 13 | 23 | 36 |

## 7. Validation Conclusion
- JSON Parsing: ❌ Fail (Target: ≥90%)
- Core Word Coverage: ✅ Pass (Target: ≥50%)
- Sentiment Accuracy: ✅ Pass (Target: ≥95%)
