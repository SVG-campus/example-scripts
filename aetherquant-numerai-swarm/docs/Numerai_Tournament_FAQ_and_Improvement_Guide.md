# Numerai Tournament FAQ, Rankings & Model Optimization Guide
## Sovereign Entanglement Asset Systems (SEAS) & Numerai v5.3 Architecture

---

## ❓ 1. Working on the Tutorial by Hand (Is That Okay?)

**YES, absolutely!**  
Doing the onboarding tutorial by hand on the website ([numer.ai/onboarding](https://numer.ai/onboarding)) is 100% recommended because:
* It checks off the **5-step account setup progress bar** on your Numerai dashboard.
* It verifies your model slots (`aetherquant`, `aetherquant_fn`, `aetherquant_te`).
* Once you upload `feature_neutralization.pkl` and `target_ensemble.pkl` on the site, our python script (`src/automated_daily_pipeline.py`) takes over and automates all future daily submissions via API!

---

## 📅 2. How Many Submissions Can We Do Per Day?

* **Per Model Slot**: You can submit **1 live submission per model per daily round**.
* **Per Account Cap**: Each Numerai account is allowed up to **30 model slots** (e.g. `aetherquant`, `aetherquant_fn`, `aetherquant_te`, `aetherquant_lgb_small`, `aetherquant_swarm`, etc.).
* **Total Daily Volume**: We can automate up to **30 distinct model predictions** every single day!

---

## 🏆 3. How Does Numerai Rank Models?

Numerai ranks models using four primary quantitative metrics:

| Metric Name | Full Name | Meaning & Calculation | Impact on Payouts |
|---|---|---|---|
| **CORR** | Pearson Correlation | Correlation between your predictions and true stock return targets. | Base payout metric. |
| **MMC** | Meta Model Contribution | Unique value your predictions add to the overall Numerai Meta Model ensemble. | **Highest Payout Multiplier** |
| **FNC** | Feature Neutral Correlation | Correlation after removing linear component explained by standard features. | Risk & stability check. |
| **TC** | True Contribution | Long-term portfolio quarterly contribution score. | Grandmaster Leaderboards |

---

## 🚀 4. How We Improve Model Performance (4-Step Strategy)

1. **LightGBM / XGBoost Model Training (`src/train_lgb_small.py`)**:
   - Train decision tree ensembles directly on `v5.3/train.parquet` using `small` (42 features) and `medium` (780 features) feature sets.
2. **Feature Neutralization (`aetherquant_fn`)**:
   - Apply linear projection in `src/topological_disentangler.py` (`feature_neutralize(proportion=0.5)`) to eliminate feature exposure and prevent drawdowns during regime shifts.
3. **Target Ensembling (`aetherquant_te`)**:
   - Ensemble predictions trained across multiple target horizons (`target_agnes_20`, `target_alpha_20`, `target_caroline_20`, `target_charlie_20`).
4. **MERA-KMPA Swarm Persistent Topology ($\beta_0, \beta_1$)**:
   - Filter predictions using Complex Projective Space ($\mathbb{CP}^n$) Fubini-Study phase distance to isolate non-linear Betti loops from noise.

---

*(PDF version saved at [`Numerai_Tournament_Master_Guide.pdf`](file:///C:/Users/svillalobosgonzalez1/Documents/GitHub/aetherquant-numerai-tournament/docs/Numerai_Tournament_Master_Guide.pdf)). All systems operational! 🚀*
