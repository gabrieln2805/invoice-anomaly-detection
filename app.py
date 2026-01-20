import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Invoice Anomaly Detection", layout="wide")
st.title("🧾 Intelligent Invoice Anomaly Detection")
st.caption("Unsupervised PyTorch Autoencoder for Accounting & Audit")

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df = df[df["type"] == "PAYMENT"].copy()
    return df

DATA_PATH = "data_sample.csv"
df = load_data(DATA_PATH)

df = df.sort_values("step")

vendor_stats = df.groupby("nameOrig")["amount"].agg(
    vendor_avg="mean",
    vendor_std="std",
    vendor_tx_count="count"
).reset_index()

df = df.merge(vendor_stats, on="nameOrig", how="left")

df["amount_zscore"] = (
    (df["amount"] - df["vendor_avg"]) / df["vendor_std"]
)

df["hour_bucket"] = df["step"] % 24

df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

features = [
    "amount",
    "vendor_avg",
    "vendor_tx_count",
    "amount_zscore",
    "hour_bucket"
]

X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

class InvoiceAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

@st.cache_resource
def train_model(input_dim, X_scaled):
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    model = InvoiceAutoencoder(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for _ in range(25):
        optimizer.zero_grad()
        recon = model(X_tensor)
        loss = criterion(recon, X_tensor)
        loss.backward()
        optimizer.step()

    return model

model = train_model(X_scaled.shape[1], X_scaled)

with torch.no_grad():
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    recon = model(X_tensor)
    df["recon_error"] = torch.mean((X_tensor - recon) ** 2, dim=1).numpy()

threshold = np.percentile(df["recon_error"], 98)
df["is_anomaly"] = df["recon_error"] > threshold

c1, c2, c3 = st.columns(3)
c1.metric("Total Invoices", len(df))
c2.metric("Flagged Anomalies", int(df["is_anomaly"].sum()))
c3.metric("Anomaly Rate", f"{df['is_anomaly'].mean()*100:.2f}%")

st.subheader("Invoice Amount vs Vendor Frequency")

fig, ax = plt.subplots(figsize=(8, 4))
sns.scatterplot(
    data=df.sample(5000),
    x="vendor_tx_count",
    y="amount",
    hue="is_anomaly",
    alpha=0.6,
    ax=ax
)
st.pyplot(fig)

st.subheader("Anomalous Payments Over Time")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df["step"], df["amount"], alpha=0.3)
ax.scatter(
    df[df["is_anomaly"]]["step"],
    df[df["is_anomaly"]]["amount"],
    color="red",
    s=10
)
st.pyplot(fig)

st.subheader("Flagged Invoices")

min_amount = st.slider(
    "Minimum Invoice Amount",
    int(df["amount"].min()),
    int(df["amount"].max()),
    10000
)

flagged = df[
    (df["is_anomaly"]) &
    (df["amount"] >= min_amount)
]

st.dataframe(
    flagged[[
        "step",
        "nameOrig",
        "amount",
        "vendor_avg",
        "vendor_tx_count",
        "amount_zscore",
        "recon_error"
    ]].sort_values("recon_error", ascending=False),
    use_container_width=True
)

st.download_button(
    "📥 Download Flagged Invoices (CSV)",
    flagged.to_csv(index=False),
    "flagged_invoices.csv",
    "text/csv"
)

