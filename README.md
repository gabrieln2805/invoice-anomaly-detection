# invoice-anomaly-detection

1. Problem Definition

The goal of this system is to automatically identify unusual invoice-like transactions that should be reviewed by accounting or internal audit teams.

Key constraints:

- No reliable fraud labels available

- Large transaction volume

- Focus on prioritization, not automated decisions

Therefore, the system uses unsupervised anomaly detection, flagging transactions that deviate from learned “normal” behavior.

2. Dataset Overview

The application uses the PaySim synthetic financial transactions dataset, which simulates real-world payment behavior.

Each record represents a transaction with attributes such as:

- transaction type

- amount

- origin account (treated as vendor)

- timestamp proxy (step)

Only PAYMENT transactions are used, as they most closely resemble invoice payments in accounting systems.

3. Feature Engineering

The model does not use raw transaction data directly.
Instead, it relies on contextual, vendor-normalized features, which are more meaningful for financial controls.

3.1 Vendor-Level Aggregations

For each vendor (nameOrig), the following statistics are computed:

Feature	             Description
vendor_avg	         Average historical invoice amount for the vendor
vendor_std	         Standard deviation of invoice amounts for the vendor
vendor_tx_count	     Number of invoices observed for the vendor

These provide behavioral context, allowing the model to distinguish between:

- a large invoice from a frequent vendor

- a large invoice from a one-off vendor

3.2 Amount Z-score (amount_zscore)

Measures how unusual an invoice amount is relative to the vendor’s own history and normalizes for vendor-specific pricing patterns

Z-score value	Meaning
≈ 0	Typical invoice for this vendor
> +1.5	Higher-than-usual invoice
< -1.5	Lower-than-usual invoice

This is a standard statistical control frequently used in audit analytics.

3.3 Temporal Feature (hour_bucket)

Derived as:

hour_bucket = step % 24


Although PaySim does not include real timestamps, this feature acts as a proxy for transaction timing, enabling the model to learn typical processing periods and unusual temporal patterns

4. Feature Scaling

All features are standardized using z-score normalization. This ensures:

equal contribution of all features

stable neural network training

no dominance by high-magnitude values (e.g., amount)

5. Model Architecture
5.1 Why an Autoencoder?

An autoencoder is well suited for this use case because:

- it learns normal behavior without labels

- anomalies are defined as patterns that cannot be reconstructed well

- it scales to large datasets

This aligns with real-world accounting scenarios, where confirmed fraud labels are rare or delayed.

5.2 Network Structure
Input (5 features)
   ↓
Dense (16) + ReLU
   ↓
Dense (8)  + ReLU   ← Latent representation
   ↓
Dense (16) + ReLU
   ↓
Output (5 features)


- Encoder compresses normal transaction patterns

- Decoder reconstructs expected values

- Reconstruction quality is the basis for anomaly scoring

6. Training Objective

The model minimizes Mean Squared Error (MSE).

This forces the autoencoder to accurately reconstruct common patterns and struggle with rare or inconsistent ones

7. Anomaly Scoring
7.1 Reconstruction Error (recon_error)

Low error = transaction fits learned normal behavior

High error = transaction deviates across one or more dimensions

This score is continuous, not binary.

7.2 Anomaly Thresholding

Instead of predicting fraud, the system flags the top N% of transactions by reconstruction error

Default setting:

- threshold = 98th percentile

This mirrors audit practice,controls workload adapts to data volume and avoids arbitrary absolute cutoffs

8. Result Interpretation
8.1 Why some invoices with low Z-score are flagged

An invoice can have a normal amount (amount_zscore ≈ 0) but still be anomalous due to unusual timing, rare vendor behavior and multivariate interactions. This is expected and desirable since anomalies are detected based on overall pattern deviation, not a single metric.

8.2 Vendor Frequency Effect

Due to dataset characteristics most vendors appear only once or twice and vendor_tx_count is often low.

As a result the model emphasizes contextual and amount-based anomalies, this behavior is documented and expected.

In real ERP systems, vendor histories would be longer, improving vendor risk profiling.

9. Dashboard Outputs
10. 
KPIs

Total invoices processed

Number of flagged anomalies

Anomaly rate (%)

Visualizations

Invoice amount vs vendor frequency

Anomalies over time

Interactive table for manual review

Export

CSV export of flagged invoices for downstream audit workflows

10. Limitations & Future Improvements
Known limitations

Synthetic dataset

Limited vendor recurrence

No ground truth labels

Possible extensions

Vendor risk scoring over time

Hybrid rule + ML controls

Sequence-based models (LSTM)

Explanation models (SHAP / rule-based)

11. Intended Use

This system is designed to 

- assist accounting and audit teams and prioritize review effort

- increase coverage without increasing workload

