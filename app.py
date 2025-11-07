# ======================================================
# LearnKAN — Split-View Functional Evolution Dashboard
# ======================================================

import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import zipfile
import os
from PIL import Image
import pandas as pd
import time

# ======================================================
# Branding & Theme Setup
# ======================================================
icon_path = "learnkan_icon.png"
try:
    icon_image = Image.open(icon_path)
except:
    icon_image = None

st.set_page_config(
    page_title="LearnKAN — Kolmogorov–Arnold Network Lab",
    layout="wide",
    page_icon=icon_image if icon_image else "🧩",
)

theme_choice = st.radio("🌗 Choose Theme for Plots:", ["Light", "Dark"], horizontal=True)

if theme_choice == "Dark":
    st.markdown("<style>body {background-color: #0e1117; color: #e0e0e0;}</style>", unsafe_allow_html=True)
    primary_color = "#90caf9"
    plt_style = "dark_background"
else:
    st.markdown("<style>body {background-color: #ffffff; color: #000000;}</style>", unsafe_allow_html=True)
    primary_color = "#2a9d8f"
    plt_style = "seaborn-v0_8-whitegrid"

if icon_image:
    st.image(icon_image, width=80)
st.title("LearnKAN — Functional Evolution Dashboard")
st.caption("A synchronized visualization platform for functional learning and pooling analysis in Kolmogorov–Arnold Networks (KANs).")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================================
# Function Generator
# ======================================================
functions = {
    "sin(x)": np.sin,
    "x²": lambda x: x ** 2,
    "exp(x)": np.exp,
    "|x|": np.abs,
    "sigmoid(x)": lambda x: 1 / (1 + np.exp(-x)),
}

def generate_data(func, n=400):
    x = np.linspace(-2, 2, n)
    y = func(x)
    x_norm = (x - x.min()) / (x.max() - x.min()) * 2 - 1
    return torch.tensor(x_norm, dtype=torch.float32).unsqueeze(1), torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# ======================================================
# Pooling Operator
# ======================================================
class PoolingOperator(nn.Module):
    def __init__(self, pooling_type, input_dim=64):
        super().__init__()
        self.pooling_type = pooling_type
        if pooling_type == "parametric":
            self.alpha = nn.Parameter(torch.ones(1, input_dim))
        elif pooling_type == "attention":
            self.attn = nn.Linear(input_dim, 1)
        elif pooling_type == "spline_weighted":
            self.weight = nn.Parameter(torch.linspace(0.1, 1.0, input_dim))

    def forward(self, x):
        if self.pooling_type == "mean":
            return x.mean(dim=1, keepdim=True)
        elif self.pooling_type == "max":
            return x.max(dim=1, keepdim=True)[0]
        elif self.pooling_type == "min":
            return x.min(dim=1, keepdim=True)[0]
        elif self.pooling_type == "integral":
            return torch.trapz(x, dim=1, keepdim=True)
        elif self.pooling_type == "spline_weighted":
            return (x * self.weight).sum(dim=1, keepdim=True) / self.weight.sum()
        elif self.pooling_type == "parametric":
            return (x * torch.sigmoid(self.alpha)).mean(dim=1, keepdim=True)
        elif self.pooling_type == "attention":
            attn_weights = torch.softmax(self.attn(x), dim=1)
            return (attn_weights * x).sum(dim=1, keepdim=True)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")

# ======================================================
# LearnKAN Model
# ======================================================
class KANLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = torch.tanh

    def forward(self, x):
        return self.activation(self.linear(x))

class LearnKAN(nn.Module):
    def __init__(self, pooling_type="mean", depth=3):
        super().__init__()
        dims = [1, 64, 32, 16, 1]
        self.layers = nn.ModuleList([KANLayer(dims[i], dims[i+1]) for i in range(depth)])
        self.pool = PoolingOperator(pooling_type, input_dim=dims[depth])
        self._init_weights()

    def _init_weights(self):
        for layer in self.layers:
            if isinstance(layer.linear, nn.Linear):
                nn.init.xavier_uniform_(layer.linear.weight)
                nn.init.zeros_(layer.linear.bias)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.pool(x)

# ======================================================
# Metrics
# ======================================================
def compute_metrics(X, Y, pred):
    X_np = X.cpu().numpy().flatten()
    Y_np = Y.cpu().numpy().flatten()
    pred_np = pred.flatten()
    residuals = pred_np - Y_np
    mse = np.mean(residuals**2)
    dx = X_np[1] - X_np[0]
    d2_pred = np.gradient(np.gradient(pred_np, dx), dx)
    curvature_energy = np.mean(d2_pred**2)
    smoothness = 1 / (1 + curvature_energy)
    stability = 1 / (1 + np.var(residuals))
    return {"MSE": mse, "Curvature Energy": curvature_energy, "Smoothness": smoothness, "Stability": stability}

# ======================================================
# UI Controls
# ======================================================
col1, col2 = st.columns(2)
with col1:
    func_name = st.selectbox("Select Function:", list(functions.keys()))
    pooling_modes = st.multiselect(
        "Select Pooling Strategies (up to 3 for side-by-side view):",
        ["mean", "spline_weighted", "attention", "parametric", "max", "min", "integral"],
        default=["mean", "spline_weighted", "attention"],
    )
    epochs = st.slider("Training Epochs:", 100, 1000, 300, 50)
    lr = st.number_input("Learning Rate:", 0.001, 0.05, 0.01, step=0.001)
with col2:
    st.markdown("""
    ### Pooling Categories
    - 🟦 **Pointwise:** mean, max, min  
    - 🟩 **Global functional:** integral, spline-weighted  
    - 🟧 **Adaptive:** parametric, attention  
    """)

# ======================================================
# Split-View Functional Evolution (with Time Factor)
# ======================================================
if st.button("🚀 Run Split-View Evolution"):
    func = functions[func_name]
    X, Y = generate_data(func)
    X, Y = X.to(device), Y.to(device)

    plt.style.use(plt_style)
    cols = st.columns(len(pooling_modes))
    plot_placeholders = [col.empty() for col in cols]
    progress = st.progress(0)
    metrics_results = []

    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize models for all pooling modes
        models = {p: LearnKAN(pooling_type=p).to(device) for p in pooling_modes}
        opts = {p: torch.optim.Adam(models[p].parameters(), lr=lr) for p in pooling_modes}
        criterion = nn.MSELoss()
        losses = {p: [] for p in pooling_modes}
        times = {p: 0.0 for p in pooling_modes}

        for epoch in range(epochs):
            for pooling in pooling_modes:
                start_time = time.time()

                model = models[pooling]
                optimizer = opts[pooling]
                optimizer.zero_grad()
                pred = model(X)
                loss = criterion(pred, Y)
                loss.backward()
                optimizer.step()

                end_time = time.time()
                times[pooling] += (end_time - start_time)
                losses[pooling].append(loss.item())

            # Live animation update
            if epoch % max(epochs // 20, 1) == 0:
                for idx, pooling in enumerate(pooling_modes):
                    model = models[pooling]
                    y_pred = model(X).detach().cpu().numpy()

                    plt.rcParams.update({
                        "font.size": 8,
                        "axes.titlesize": 9,
                        "axes.labelsize": 8,
                        "legend.fontsize": 7,
                        "xtick.labelsize": 7,
                        "ytick.labelsize": 7,
                    })
                    fig_size = (3.2, 2.0)
                    fig, ax = plt.subplots(figsize=fig_size)
                    ax.plot(X.cpu().numpy(), Y.cpu().numpy(), label="True", linewidth=1.0,
                            color="white" if theme_choice == "Dark" else "black")
                    ax.plot(X.cpu().numpy(), y_pred, linestyle="--", color=primary_color, linewidth=1.0)
                    ax.set_title(f"{pooling} (Epoch {epoch})", fontsize=8)
                    ax.grid(True, alpha=0.2)
                    ax.legend(fontsize=7, loc="best", frameon=False)
                    fig.tight_layout(pad=0.8)
                    plot_placeholders[idx].pyplot(fig, use_container_width=False)
                    plt.close(fig)

            progress.progress((epoch + 1) / epochs)

        # After training: compute metrics + time
        for pooling in pooling_modes:
            model = models[pooling]
            y_pred = model(X).detach().cpu().numpy()
            metrics = compute_metrics(X, Y, y_pred)
            metrics["Time (s)"] = round(times[pooling], 3)
            metrics["Efficiency"] = metrics["Smoothness"] / (metrics["MSE"] * (1 + metrics["Time (s)"]))
            metrics_results.append({"Pooling": pooling, **metrics})
            np.save(os.path.join(tmpdir, f"{pooling}_pred.npy"), y_pred)
            np.save(os.path.join(tmpdir, f"{pooling}_loss.npy"), np.array(losses[pooling]))

        df = pd.DataFrame(metrics_results).sort_values(by="MSE")
        st.subheader("📊 Functional Metrics Summary (with Time)")

        # Adaptive table styling
        styled = df.style.format({
            "MSE": "{:.3e}",
            "Curvature Energy": "{:.3e}",
            "Smoothness": "{:.4f}",
            "Stability": "{:.4f}",
            "Time (s)": "{:.2f}",
            "Efficiency": "{:.2e}",
        })

        if theme_choice == "Dark":
            styled = styled.set_properties(**{"color": "#ffffff", "background-color": "#0e1117"})
        else:
            styled = styled.set_properties(**{"color": "#000000", "background-color": "#ffffff"})

        st.dataframe(styled)

        # Download zip
        zip_path = os.path.join(tmpdir, f"{func_name}_LearnKAN_SplitView_Timed.zip")
        with zipfile.ZipFile(zip_path, "w") as zipf:
            for file in os.listdir(tmpdir):
                zipf.write(os.path.join(tmpdir, file), file)
        with open(zip_path, "rb") as f:
            st.download_button("📥 Download Timed Experiment Bundle", f.read(), f"{func_name}_LearnKAN_SplitView_Timed.zip", mime="application/zip")

st.markdown("---")
st.markdown("💡 *Tip:* Compare both accuracy (MSE) and efficiency (Smoothness ÷ MSE ÷ Time) to see which pooling performs better under time constraints.*")
