# Numerai Tournament & MERA-KMPA Integration Master Guide
## Automated v5.3 Quantum Dataset Ingestion, Submissions & Season 2026 Staking Roadmap
**Account**: `aetherquant` (`santiago@aetherquant.cloud`)  
**Audit Verified**: July 25, 2026  
**Repository**: `aetherquant-numerai-tournament`

---

## 📊 1. Account Authentication & NMR Staking Wallet Verification

| Field / Property | Verified Status / Value | Security & Scope | Operational Role |
|---|---|---|---|
| **Username** | `aetherquant` | VERIFIED User Account | Primary Tournament Account |
| **Email** | `santiago@aetherquant.cloud` | Verified Primary Email | Official Notifications & Scores |
| **API Public ID** | `2PPYXJYSNU4O5P7BU2A25...` | Full Scope Access | Automated Predictions Upload |
| **MCP Auth Token** | `OGSPTFKUEKAS54WSB76...` | MCP Server Tool Auth | Direct Agent API Execution |
| **NMR Wallet Balance** | **1.62028792 NMR** | Confirmed On-Chain | Season 2026 Staking Ready |
| **Model Name / ID** | `aetherquant` (ID: `d5b859e1...`) | Primary Tournament Model | Live Daily Submissions (Active) |

---

## 🚀 2. Verified Live Submission Event Log

* **Submission ID**: `549ab1b6-c1e0-43ee-97bb-2ef045c03828`
* **Model Name**: `aetherquant`
* **Asset Rows Submitted**: 7,096 live predictions
* **Dataset Version**: v5.3 'Quantum' (807 new features)
* **Status**: **100% Verified & Accepted by Numerai API**

---

## 🛠️ 3. Season 2026 Staking & Automated Daily Pipeline Roadmap

1. **Season 2026 Staking Requirement**:
   - 20 qualified staked submissions with minimum NMR stake per round.
   - With **1.62028792 NMR** confirmed in the wallet, initial rounds can be staked immediately, scaling to 20 NMR over the coming weeks.

2. **Automated Daily Pipeline (`src/automated_daily_pipeline.py`)**:
   - Programmatically downloads daily `v5.3/live.parquet`.
   - Applies MERA-KMPA topological feature neutralization.
   - Rank-normalizes predictions to uniform $[0, 1]$.
   - Uploads live predictions automatically to Numerai API.

---

*(PDF version generated at [`Numerai_Tournament_Master_Guide.pdf`](file:///C:/Users/svillalobosgonzalez1/Documents/GitHub/aetherquant-numerai-tournament/docs/Numerai_Tournament_Master_Guide.pdf)). All systems operational! 🚀*
