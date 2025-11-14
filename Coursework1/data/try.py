import pandas as pd
import matplotlib.pyplot as plt
import os

print(os.getcwd())

csv_path = "../results/spmm_benchmark_results.csv"  # change if your path is different
df = pd.read_csv(csv_path)

densities = [0.1, 0.2, 0.3]

for dens in densities:
    subset = df[df["densityA"] == dens].sort_values("m")

    plt.figure(figsize=(7, 5))
    plt.plot(subset["m"], subset["RDD_COO_ms"], marker="o", label="RDD COO (ms)")
    plt.plot(subset["m"], subset["RDD_CSR_ms"], marker="s", linestyle="--", label="CSRxCSC (ms)")
    plt.title(f"Runtime vs Matrix Size (density = {dens})")
    plt.xlabel("Matrix Size (m)")
    plt.ylabel("Runtime (ms)")
    plt.grid(True)
    plt.legend()

    file_path = f"../../../Submission/CSR_runtime_vs_size_density_{str(dens).replace('.', '_')}.png"
    plt.savefig(file_path, bbox_inches="tight", dpi=160)
    plt.close()
