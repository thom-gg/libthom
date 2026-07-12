import csv
import sys
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.ticker import FuncFormatter

if len(sys.argv) < 2:
    print("Usage: python plot.py file1.csv file2.csv ...")
    sys.exit(1)

csv_files = sys.argv[1:]

# Dictionnaire pour stocker les données par kernel
data = defaultdict(list)

for filename in csv_files:
    with open(filename, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                size = int(row['size'])
                gflops = float(row['gflops'])
                kernel = filename
                data[kernel].append((size, gflops))
            except KeyError:
                print(f"Colonnes manquantes dans {filename}")
            except ValueError:
                print(f"Valeur invalide dans {filename}: {row}")

# Tracer les courbes
plt.figure(figsize=(10,6))

for kernel, values in data.items():
    values.sort(key=lambda x: x[0])
    sizes = [v[0] for v in values]
    gflops = [v[1] for v in values]
    plt.plot(sizes, gflops, marker='o', label=kernel)

plt.xlabel("M=N=K")
plt.ylabel("GFLOP/s")
plt.title("Performance")
plt.legend()
plt.grid(True)

# Formatter l'axe X pour afficher "10M" au lieu de 10000000
def millions(x, pos):
    if x >= 1_000_000:
        return f"{int(x/1_000_000)}M"
    elif x >= 1_000:
        return f"{int(x/1_000)}k"
    else:
        return str(int(x))

plt.gca().xaxis.set_major_formatter(FuncFormatter(millions))

plt.tight_layout()
plt.savefig("plot.png")
plt.show()