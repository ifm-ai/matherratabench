# MathErrataBench results

Each model was evaluated on the same 200 papers.

| Model | Effort | Issue recovery | Serious recovery | Severity score | Strict issue recovery | Strict formal recovery |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GPT-Astra | max | 77.41% | 95.74% | 81.71% | 41.98% | 50.29% |
| GPT-Sol | medium | 54.11% | 82.98% | 70.62% | 22.45% | 26.95% |
| Claude Opus 5.5 | max | 53.84% | 80.85% | 81.22% | 30.13% | 38.76% |
| Claude Fable 5.1 | medium | 51.88% | 78.72% | 80.38% | 26.36% | 34.58% |
| GLM-5.3 | max | 51.46% | 89.36% | 77.91% | 25.24% | 33.86% |
| Kimi-K3 | max | 48.81% | 68.09% | 82.00% | 26.50% | 33.72% |
| MiMo-V2.6-Pro-RL | max | 36.82% | 63.83% | 78.79% | 19.53% | 26.22% |
| K2-Horizon-375B-A23B | high | 35.70% | 55.32% | 75.59% | 15.48% | 18.16% |
| DeepSeek-V4.1-Flash | max | 33.47% | 57.45% | 71.88% | 13.81% | 16.86% |
| GLM-5.3-Flash | max | 30.68% | 57.45% | 76.82% | 15.76% | 20.75% |
| Gemini-3.8-Flash | medium | 21.62% | 53.19% | 71.29% | 9.62% | 14.84% |

- **Issue recovery** is the share of all 717 substantive gold issues matched.
- **Serious recovery** is the share of the 47 serious gold issues matched.
- **Severity score** averages credit over matched issues. An exact severity earns 100%, an adjacent severity earns 50%, and other or ungraded labels earn 0%.
- **Strict issue recovery** is the share of all 717 gold issues matched with the exact severity and target type, plus the exact primary tag for formal issues.
- **Strict formal recovery** is the weighted share of the 476 formal gold issues matched with the exact severity and primary tag. Minor issues count half, and moderate and serious issues count one.

## Answer and judgment coverage

| Model | Parsed answers | Why fewer than 200 were parsed | Accepted judgments | Parsed answers reporting no issues |
| --- | ---: | --- | ---: | ---: |
| GPT-Astra | 200 | All 200 parsed | 200 | 0 |
| GPT-Sol | 200 | All 200 parsed | 200 | 0 |
| Claude Opus 5.5 | 199 | 1 with unparseable JSON | 196 | 3 |
| Claude Fable 5.1 | 199 | 1 with no scoreable answer on file | 199 | 0 |
| GLM-5.3 | 196 | 4 with unparseable JSON | 196 | 0 |
| Kimi-K3 | 199 | 1 with unparseable JSON | 197 | 2 |
| MiMo-V2.6-Pro-RL | 198 | 2 with unparseable JSON | 194 | 4 |
| K2-Horizon-375B-A23B | 195 | 5 with unparseable JSON | 195 | 0 |
| DeepSeek-V4.1-Flash | 198 | 2 with no scoreable answer on file | 196 | 2 |
| GLM-5.3-Flash | 176 | 24 with unparseable JSON | 174 | 2 |
| Gemini-3.8-Flash | 195 | 5 with no scoreable answer on file | 168 | 27 |

All 2,115 answers that reported findings received an accepted judgment. The 45 cases without a scoreable answer and the 40 parsed answers that reported no issues stayed in the recovery denominator.
