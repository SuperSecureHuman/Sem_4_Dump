import csv
import matplotlib.pyplot as plt
import psutil
import random
import time
import editdistance
import Levenshtein
import polyleven
import rapidfuzz
import edlib
import numpy as np
from tqdm import tqdm

# Define the input sizes you want to test
input_sizes = np.arange(1000, 100000, 1000)

runtimes_editdistance = []
runtimes_Levenshtein = []
runtimes_polyleven = []
runtimes_rapidfuzz = []
runtimes_edlib = []
cpu_usage = []

# Initialize CPU and RAM usage per library
cpu_usage_editdistance = []
cpu_usage_Levenshtein = []
cpu_usage_polyleven = []
cpu_usage_rapidfuzz = []
cpu_usage_edlib = []
ram_usage_editdistance = []
ram_usage_Levenshtein = []
ram_usage_polyleven = []
ram_usage_rapidfuzz = []
ram_usage_edlib = []

# Progress bar
pbar = tqdm(total=len(input_sizes))

# Run the tests
for size in input_sizes:
    # Generate random strings
    str1 = ''.join(random.choice('atcg') for _ in range(size))
    str2 = ''.join(random.choice('atcg') for _ in range(size))

    start_time = time.time()
    editdistance.eval(str1, str2)
    end_time = time.time()
    runtime = end_time - start_time
    runtimes_editdistance.append(runtime)
    cpu_percent = psutil.cpu_percent()
    cpu_usage.append(cpu_percent)
    cpu_usage_editdistance.append(cpu_percent)
    ram_usage_editdistance.append(psutil.virtual_memory().percent)

    start_time = time.time()
    Levenshtein.distance(str1, str2)
    end_time = time.time()
    runtime = end_time - start_time
    runtimes_Levenshtein.append(runtime)
    cpu_percent = psutil.cpu_percent()
    cpu_usage.append(cpu_percent)
    cpu_usage_Levenshtein.append(cpu_percent)
    ram_usage_Levenshtein.append(psutil.virtual_memory().percent)

    start_time = time.time()
    polyleven.levenshtein(str1, str2)
    end_time = time.time()
    runtime = end_time - start_time
    runtimes_polyleven.append(runtime)
    cpu_percent = psutil.cpu_percent()
    cpu_usage.append(cpu_percent)
    cpu_usage_polyleven.append(cpu_percent)
    ram_usage_polyleven.append(psutil.virtual_memory().percent)

    start_time = time.time()
    rapidfuzz.distance.Levenshtein.distance(str1, str2)
    end_time = time.time()
    runtime = end_time - start_time
    runtimes_rapidfuzz.append(runtime)
    cpu_percent = psutil.cpu_percent()
    cpu_usage.append(cpu_percent)
    cpu_usage_rapidfuzz.append(cpu_percent)
    ram_usage_rapidfuzz.append(psutil.virtual_memory().percent)

    start_time = time.time()
    edlib.align(str1, str2)
    end_time = time.time()
    runtime = end_time - start_time
    runtimes_edlib.append(runtime)
    cpu_percent = psutil.cpu_percent()
    cpu_usage.append(cpu_percent)
    cpu_usage_edlib.append(cpu_percent)
    ram_usage_edlib.append(psutil.virtual_memory().percent)

    pbar.update(1)

pbar.close()

# Plotting
plt.figure(figsize=(12, 8))

# Plot runtimes
plt.subplot(2, 2, 1)
plt.plot(input_sizes, runtimes_editdistance, label='editdistance', marker='o')
plt.plot(input_sizes, runtimes_Levenshtein, label='Levenshtein', marker='o')
plt.plot(input_sizes, runtimes_polyleven, label='polyleven', marker='o')
plt.plot(input_sizes, runtimes_rapidfuzz, label='rapidfuzz', marker='o')
plt.plot(input_sizes, runtimes_edlib, label='edlib', marker='o')
plt.xlabel('Input Size')
plt.ylabel('Runtime (seconds)')
plt.title('Runtime')
plt.legend()

# Plot CPU usage
plt.subplot(2, 2, 2)
#plt.plot(input_sizes, cpu_usage_editdistance, label='editdistance', marker='o')
plt.plot(input_sizes, cpu_usage_Levenshtein, label='Levenshtein', marker='o')
plt.plot(input_sizes, cpu_usage_polyleven, label='polyleven', marker='o')
plt.plot(input_sizes, cpu_usage_rapidfuzz, label='rapidfuzz', marker='o')
plt.plot(input_sizes, cpu_usage_edlib, label='edlib', marker='o')
plt.xlabel('Input Size')
plt.ylabel('CPU Usage (%)')
plt.title('CPU Usage')
plt.legend()

# Plot RAM usage
plt.subplot(2, 2, 3)
#plt.plot(input_sizes, ram_usage_editdistance, label='editdistance', marker='o')
plt.plot(input_sizes, ram_usage_Levenshtein, label='Levenshtein', marker='o')
plt.plot(input_sizes, ram_usage_polyleven, label='polyleven', marker='o')
plt.plot(input_sizes, ram_usage_rapidfuzz, label='rapidfuzz', marker='o')
plt.plot(input_sizes, ram_usage_edlib, label='edlib', marker='o')
plt.xlabel('Input Size')
plt.ylabel('RAM Usage (%)')
plt.title('RAM Usage')
plt.legend()

# Adjust the spacing between subplots
plt.tight_layout()

# Display the plots
plt.show()
