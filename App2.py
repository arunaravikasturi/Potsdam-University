import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings("ignore", category=DataConversionWarning)

# File paths / config
INPUT_FILE = "dataset02.csv"
TRAIN_FILE = "dataset02_training.csv"
TEST_FILE = "dataset02_testing.csv"
OUTPUT_PDF = "UE_04_App2_ScatterVisualizationAndOlsModel.pdf"

TEST_SIZE = 0.2
RANDOM_STATE = 42
IQR_MULTIPLIER = 1.5

def prepare_data(df):
    """Clean, remove outliers, and normalize the dataset."""
    print("Original shape:", df.shape)

    # Convert all columns to numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    # Drop rows with missing values
    df = df.dropna()
    print("After removing NaN:", df.shape)

    # IQR Outlier removal
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    mask = ~((df < (Q1 - IQR_MULTIPLIER * IQR)) | 
             (df > (Q3 + IQR_MULTIPLIER * IQR))).any(axis=1)
    df = df[mask]

    print("After outlier removal:", df.shape)

    if df.empty:
        raise ValueError("All rows removed as outliers. Adjust IQR_MULTIPLIER.")

    # Normalize
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    return df

try:
    df = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    print("Input file not found. Creating dummy dataset...")

    df = pd.DataFrame({
        "x1": np.random.rand(100) * 10,
        "x2": np.random.rand(100) * 5 + 3,
        "y":  np.random.rand(100) * 50
    })

    # Add some noise
    df.loc[5, "x1"] = "bad"
    df.loc[10, "x2"] = np.nan
    df.loc[15, "y"] = 1000  # Outlier

    df.to_csv(INPUT_FILE, index=False)

df.head()

df_clean = prepare_data(df)

train_df, test_df = train_test_split(
    df_clean,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

train_df.to_csv(TRAIN_FILE, index=False)
test_df.to_csv(TEST_FILE, index=False)

print("Training saved:", train_df.shape)
print("Testing saved:", test_df.shape)

target = df_clean.columns[-1]
features = df_clean.columns[:-1]

X_train = train_df[features]
y_train = train_df[target]

X_train_const = sm.add_constant(X_train)

model = sm.OLS(y_train, X_train_const).fit()
print(model.summary())

# Predictions for plotting
y_pred_train = model.predict(X_train_const)

print("Saving PDF:", OUTPUT_PDF)

with PdfPages(OUTPUT_PDF) as pdf:

    num_features = len(features)
    rows = int(np.ceil(num_features / 2))
    cols = 2 if num_features > 1 else 1
    
    fig, axes = plt.subplots(rows, cols, figsize=(10, 5 * rows), squeeze=False)

    X_test = test_df[features]
    y_test = test_df[target]

    for i, col in enumerate(features):
        ax = axes.flat[i]

        # Scatter plots
        ax.scatter(X_train[col], y_train, alpha=0.7, label="Train")
        ax.scatter(X_test[col], y_test, alpha=0.7, label="Test")

        # Regression line
        order = X_train[col].argsort()
        ax.plot(
            X_train[col].iloc[order],
            y_pred_train.iloc[order],
            label="OLS Fit"
        )

        ax.set_title(f"{target} vs {col}")
        ax.set_xlabel(col)
        ax.set_ylabel(target)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)

    # Hide unused axes
    for j in range(i + 1, len(axes.flat)):
        axes.flat[j].set_visible(False)

    # Save the figure to the PDF
    pdf.savefig(fig)

    # Show inside Jupyter notebook
    plt.show()

    # Close the figure to avoid display duplication
    plt.close(fig)

print("Done.")


BOXPLOT_PDF = "UE_04_App2_BoxPlot.pdf"
print("Saving Box Plot PDF:", BOXPLOT_PDF)

with PdfPages(BOXPLOT_PDF) as pdf:

    plt.figure(figsize=(8, 6))

    # Standard boxplot (y-axis NOT reversed)
    df_clean.boxplot()

    plt.title("Box Plot of All Data Dimensions")
    plt.ylabel("Scaled Values")

    # Match your image: no reversed axis
    # plt.ylim(1.5, -1.5)  <-- REMOVE this line

    # Optional: remove grid (because your image has none)
    # plt.grid(False)

    pdf.savefig()
    plt.show()
    plt.close()

print("Box plot generation complete.")

from matplotlib.backends.backend_pdf import PdfPages

DIAG_PDF = "App02_Diagnosis.pdf"
print("Saving Diagnosis PDF:", DIAG_PDF)

# Collect diagnostics
fitted_vals = model.fittedvalues
residuals = model.resid
influence = model.get_influence()
student_resid = influence.resid_studentized_internal
leverage = influence.hat_matrix_diag

# Save into PDF
with PdfPages(DIAG_PDF) as pdf:

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle("OLS Diagnostic Plots", fontsize=16)

    # 1. Residuals vs Fitted
    ax = axes[0, 0]
    ax.scatter(fitted_vals, residuals, alpha=0.5)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_title("Residuals vs Fitted")
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")

    # 2. Normal Q-Q
    ax = axes[0, 1]
    sm.qqplot(residuals, line="45", fit=True, ax=ax)
    ax.set_title("Normal Q-Q")

    # 3. Scale-Location
    ax = axes[1, 0]
    ax.scatter(fitted_vals, (abs(student_resid) ** 0.5), alpha=0.5)
    axes[1, 0].set_title("Scale-Location")
    axes[1, 0].set_xlabel("Fitted values")
    axes[1, 0].set_ylabel("√(|Standardized Residuals|)")

    # 4. Residuals vs Leverage
    ax = axes[1, 1]
    sm.graphics.plot_leverage_resid2(model, ax=ax)
    ax.set_title("Residuals vs Leverage")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save figure to PDF
    pdf.savefig(fig)

    # Also display normally (like your boxplot example shows + closes)
    plt.show()
    plt.close(fig)

print("✅ Diagnosis plots saved into App02_Diagnosis.pdf")
