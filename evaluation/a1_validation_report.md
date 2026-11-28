# A1 – Image Quality Agent Validation Report
## 1. Overview
The A1 Image Quality Agent evaluates uploaded skin lesion images before they proceed to further analysis.

It checks:
- Image blur
- Brightness
- Dark fraction
- Glare
- Hair or ruler obstruction

## 2. Validation Dataset

- Dataset: HAM10000 validation set
- Total validation samples: 1,942
- Evaluation method: Precomputed image quality features
- Image preprocessing: Resize to 224 × 224 pixels

## 3. Validation Results

| Metric | Result |
|---|---:|
| Total images | 1,942 |
| PASS | 1,825 |
| FAIL | 117 |
| PASS rate | 93.98% |
| FAIL rate | 6.02% |

## 4. Issue Distribution

| Quality Issue | Count |
|---|---:|
| Blurry | 68 |
| Glare | 59 |
| Hair/ruler obstruction | 50 |
| Too dark | 29 |
| Heavy black border/vignette | 23 |
| Over-exposed | 8 |

Note: Issue counts may overlap because one image can contain multiple quality issues.

## 5. Live Feature Verification

A separate test was performed on 20 real validation images.

- Images tested: 20
- Total feature comparisons: 120
- Matches: 120
- Mismatches: 0

Result: All calculated feature values matched the expected feature values.

## 6. Input Validation Tests

The following invalid input cases were tested:

- Empty image bytes
- Invalid metadata
- Undecodable image data

The agent safely returned FAIL responses for these cases instead of crashing.

## 7. Conclusion

The A1 Image Quality Agent successfully completed the validation checks.

The validation evaluation reported a 6.02% FAIL rate. Live feature verification achieved 120/120 matches across 20 real images.

The agent is ready for the next integration checks with the Safety Agent.

Note: This is a prototype image-quality screening component, not a medical diagnostic system.