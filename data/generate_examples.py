"""Generate deterministic synthetic datasets with injected anomalies.

Run directly (`python data/generate_examples.py`) to regenerate the bundled
CSV examples in the `data/` directory. Each dataset uses a fixed random seed
and includes a hidden `_ground_truth` column (1 = anomaly, 0 = normal) so
the UI can display live precision and recall alongside Isolation Forest output.
"""

from pathlib import Path
import numpy as np
import pandas as pd

RANDOM_SEED = 42
DATA_DIR = Path(__file__).resolve().parent


def generate_credit_card_transactions(rng: np.random.Generator) -> pd.DataFrame:
    """Generate 1,000 credit card transactions with ~2% fraud-like outliers."""
    n_total = 1000
    n_anomalies = 20  # 2.0%
    n_normal = n_total - n_anomalies

    # Normal spending: moderate log-normal amounts, daytime/evening hours, low risk
    normal_amount = np.round(rng.lognormal(mean=3.8, sigma=0.65, size=n_normal), 2)
    normal_amount = np.clip(normal_amount, 4.50, 460.0)
    normal_hour = np.round(rng.normal(loc=14.8, scale=3.6, size=n_normal), 2)
    normal_hour = np.clip(normal_hour, 5.5, 23.8)
    normal_risk = np.round(rng.beta(a=2.0, b=8.5, size=n_normal), 4)

    # Fraud-like outliers: high ticket late-night bursts + elevated merchant risk
    fraud_amount = np.round(rng.uniform(680.0, 2950.0, size=n_anomalies), 2)
    fraud_hour = np.round(rng.uniform(0.2, 4.6, size=n_anomalies), 2)
    fraud_risk = np.round(rng.uniform(0.68, 0.98, size=n_anomalies), 4)

    df_normal = pd.DataFrame({
        "amount": normal_amount,
        "hour_of_day": normal_hour,
        "merchant_risk_score": normal_risk,
        "_ground_truth": 0,
    })
    df_fraud = pd.DataFrame({
        "amount": fraud_amount,
        "hour_of_day": fraud_hour,
        "merchant_risk_score": fraud_risk,
        "_ground_truth": 1,
    })

    combined = pd.concat([df_normal, df_fraud], ignore_index=True)
    return combined.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)


def generate_network_traffic(rng: np.random.Generator) -> pd.DataFrame:
    """Generate 800 network flow records with ~3% injected attack bursts."""
    n_total = 800
    n_flood = 14
    n_slow = 10
    n_normal = n_total - n_flood - n_slow

    normal_pps = np.round(rng.normal(loc=145.0, scale=38.0, size=n_normal), 1)
    normal_pps = np.clip(normal_pps, 25.0, 295.0)
    normal_size = np.round(rng.normal(loc=640.0, scale=130.0, size=n_normal), 1)
    normal_size = np.clip(normal_size, 210.0, 1250.0)
    normal_duration = np.round(rng.gamma(shape=4.0, scale=4.5, size=n_normal), 2)
    normal_duration = np.clip(normal_duration, 1.5, 58.0)

    # Volumetric flood bursts: extreme packet rate, tiny payloads, brief connections
    flood_pps = np.round(rng.uniform(620.0, 1450.0, size=n_flood), 1)
    flood_size = np.round(rng.uniform(54.0, 145.0, size=n_flood), 1)
    flood_duration = np.round(rng.uniform(0.15, 2.2, size=n_flood), 2)

    # Slow-rate exhaustion attacks: very long duration, minimal packet rate
    slow_pps = np.round(rng.uniform(4.0, 18.0, size=n_slow), 1)
    slow_size = np.round(rng.uniform(60.0, 120.0, size=n_slow), 1)
    slow_duration = np.round(rng.uniform(115.0, 240.0, size=n_slow), 2)

    df_normal = pd.DataFrame({
        "packets_per_sec": normal_pps,
        "avg_packet_size": normal_size,
        "connection_duration": normal_duration,
        "_ground_truth": 0,
    })
    df_attack = pd.DataFrame({
        "packets_per_sec": np.concatenate([flood_pps, slow_pps]),
        "avg_packet_size": np.concatenate([flood_size, slow_size]),
        "connection_duration": np.concatenate([flood_duration, slow_duration]),
        "_ground_truth": 1,
    })

    combined = pd.concat([df_normal, df_attack], ignore_index=True)
    return combined.sample(frac=1.0, random_state=RANDOM_SEED + 1).reset_index(drop=True)


def generate_manufacturing_quality(rng: np.random.Generator) -> pd.DataFrame:
    """Generate 650 sensor batch readings with defective-batch outliers and minor sensor NaNs."""
    n_total = 650
    n_anomalies = 16  # ~2.5%
    n_normal = n_total - n_anomalies

    normal_temp = np.round(rng.normal(loc=185.0, scale=3.1, size=n_normal), 2)
    normal_vib = np.round(rng.normal(loc=1.38, scale=0.22, size=n_normal), 3)
    normal_vib = np.clip(normal_vib, 0.55, 2.25)
    normal_press = np.round(rng.normal(loc=101.4, scale=2.3, size=n_normal), 2)

    # Defective batches: thermal drift + excessive bearing vibration + pressure drops/spikes
    defect_temp = np.round(rng.uniform(201.0, 217.5, size=n_anomalies), 2)
    defect_vib = np.round(rng.uniform(2.95, 4.95, size=n_anomalies), 3)
    half = n_anomalies // 2
    defect_press = np.round(
        np.concatenate([
            rng.uniform(82.0, 90.5, size=half),
            rng.uniform(112.5, 122.0, size=n_anomalies - half),
        ]),
        2,
    )

    df_normal = pd.DataFrame({
        "temperature": normal_temp,
        "vibration": normal_vib,
        "pressure": normal_press,
        "_ground_truth": 0,
    })
    df_defect = pd.DataFrame({
        "temperature": defect_temp,
        "vibration": defect_vib,
        "pressure": defect_press,
        "_ground_truth": 1,
    })

    combined = (
        pd.concat([df_normal, df_defect], ignore_index=True)
        .sample(frac=1.0, random_state=RANDOM_SEED + 2)
        .reset_index(drop=True)
    )

    # Inject a small number of missing sensor readings in normal rows to demonstrate imputation
    normal_indices = combined.index[combined["_ground_truth"] == 0].to_numpy()
    drop_idx = rng.choice(normal_indices, size=6, replace=False)
    combined.loc[drop_idx[:2], "temperature"] = np.nan
    combined.loc[drop_idx[2:4], "vibration"] = np.nan
    combined.loc[drop_idx[4:6], "pressure"] = np.nan

    return combined


def main() -> None:
    """Generate and save all three example CSV datasets."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    datasets = {
        "credit_card_transactions.csv": generate_credit_card_transactions(rng),
        "network_traffic.csv": generate_network_traffic(rng),
        "manufacturing_quality.csv": generate_manufacturing_quality(rng),
    }

    for filename, df in datasets.items():
        output_path = DATA_DIR / filename
        df.to_csv(output_path, index=False)
        anomalies = int(df["_ground_truth"].sum())
        print(f"Wrote {output_path.name}: {len(df)} rows ({anomalies} anomalies)")


if __name__ == "__main__":
    main()
