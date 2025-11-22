import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score, mean_squared_error

# Set plot style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12})

# ==============================================================================
# 1. DATA GENERATION (Simulating the Molecular Dynamics Output)
# ==============================================================================
def get_metallic_glass_data(n_samples=100):
    """
    Generates a synthetic dataset representing MD simulations of Cu-Zr metallic glasses.
    Features mirror the Voronoi analysis described in the report.
    """
    np.random.seed(42)
    
    # --- INPUT PARAMETERS ---
    # Log Cooling Rate (9.0 to 13.0). Slower cooling (9) allows more order.
    log_cooling_rate = np.random.uniform(9.0, 13.0, n_samples)
    
    # Temperature during tensile test (100K to 500K)
    temperature = np.random.uniform(100, 500, n_samples)

    # --- ENGINEERED STRUCTURAL FEATURES (Voronoi Analysis) ---
    # Physics: Slower cooling rate -> Higher fraction of Full Icosahedra (<0,0,12,0>)
    # We add noise to simulate the variance inherent in MD simulations.
    ico_fraction = 0.65 - (0.035 * log_cooling_rate) + np.random.normal(0, 0.015, n_samples)
    
    # Physics: Distorted clusters don't correlate strongly with cooling rate
    distorted_fraction = np.random.uniform(0.15, 0.20, n_samples)
    
    # Physics: Free Volume is inversely proportional to packing efficiency (Icosahedra)
    # More icosahedra = Less free volume
    free_volume = 0.12 - (0.2 * ico_fraction) + np.random.normal(0, 0.005, n_samples)
    
    # A "Noise" feature to test if LASSO correctly eliminates it
    simulation_id_noise = np.random.rand(n_samples)

    # --- TARGET LABEL: YIELD STRENGTH (GPa) ---
    # Physics Ground Truth: Strength comes from Icosahedra. 
    # Free Volume and Temperature weaken the material.
    yield_strength = (
        -1.0 
        + (8.5 * ico_fraction)       # Strong positive driver
        - (5.0 * free_volume)        # Negative driver
        - (0.0015 * temperature)     # Thermal softening
        + (0.0 * distorted_fraction) # No effect (LASSO should find this is 0)
        + np.random.normal(0, 0.05, n_samples) # Experimental/MD noise
    )

    # Create DataFrame
    df = pd.DataFrame({
        'Log_Cooling_Rate': log_cooling_rate,
        'Temperature_K': temperature,
        'Ico_Fraction': ico_fraction,
        'Distorted_Frac': distorted_fraction,
        'Free_Volume': free_volume,
        'Noise_Feat': simulation_id_noise,
        'Yield_Strength_GPa': yield_strength
    })
    
    return df

# Load data
print("Generating synthetic MD dataset...")
df = get_metallic_glass_data(150)
print(f"Dataset Shape: {df.shape}")

# ==============================================================================
# 2. PREPROCESSING
# ==============================================================================
# Separate Target (y) and Features (X)
X = df.drop('Yield_Strength_GPa', axis=1)
y = df['Yield_Strength_GPa']

# Split: 80% Training, 20% Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SCALING: Critical for LASSO. 
# We must scale features so the penalty (Lambda) applies fairly to all coefficients.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================================================================
# 3. MODEL TRAINING (LASSO with Cross-Validation)
# ==============================================================================
print("\nTraining LASSO Model with 5-Fold Cross-Validation...")

# LassoCV automatically finds the best alpha (lambda)
lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
lasso.fit(X_train_scaled, y_train)

print(f"Optimal Regularization Parameter (Alpha/Lambda): {lasso.alpha_:.4f}")

# ==============================================================================
# 4. EVALUATION
# ==============================================================================
y_pred = lasso.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("\n--- PERFORMANCE ON TEST SET ---")
print(f"R^2 Score (Accuracy): {r2:.4f}")
print(f"Mean Squared Error:   {mse:.4f} GPa^2")

# ==============================================================================
# 5. INTERPRETATION & PLOTTING
# ==============================================================================
# Extract coefficients
coefs = pd.Series(lasso.coef_, index=X.columns)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Actual vs Predicted
axes[0].scatter(y_test, y_pred, color='#005088', alpha=0.7, edgecolors='k')
axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel("Actual Yield Strength (GPa) [MD Simulation]")
axes[0].set_ylabel("Predicted Yield Strength (GPa) [LASSO Model]")
axes[0].set_title(f"Model Accuracy (R^2 = {r2:.2f})")
axes[0].legend()
axes[0].grid(True, linestyle='--', alpha=0.6)

# Plot 2: Feature Importance (The Scientific Discovery)
# Color bars based on positive/negative impact
colors = ['#dc2626' if c < 0 else '#16a34a' for c in coefs]
coefs.sort_values().plot(kind='barh', ax=axes[1], color=colors, edgecolor='black')
axes[1].axvline(0, color='black', linestyle='-', linewidth=0.8)
axes[1].set_title("Feature Importance (LASSO Coefficients)")
axes[1].set_xlabel("Impact on Yield Strength (Normalized Beta)")
axes[1].grid(axis='x', linestyle='--', alpha=0.6)

# Annotate the interpretation
print("\n--- SCIENTIFIC INTERPRETATION ---")
for feature, val in coefs.sort_values(ascending=False).items():
    if abs(val) < 0.01:
        print(f"[ELIMINATED] {feature}: \t{val:.4f} (Not a physical driver)")
    else:
        direction = "INCREASES" if val > 0 else "DECREASES"
        print(f"[DRIVER]     {feature}: \t{val:.4f} -> {direction} Strength")

plt.tight_layout()
plt.savefig('metallic_glass_results.png', dpi=300)
print("\nPlot saved as 'metallic_glass_results.png'")
plt.show()