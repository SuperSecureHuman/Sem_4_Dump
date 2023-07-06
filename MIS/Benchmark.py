import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from admm_parallel import ADMM
from torch_cpu import ADMM as ADMM_CPU
from torch_gpu import ADMM as ADMM_GPU
from tqdm import tqdm
import pandas as pd


def generate_dummy_data(n_samples, n_features):
    X = np.random.randn(n_samples, n_features)
    true_coefficients = np.random.uniform(low=-5, high=5, size=n_features)  # True coefficients for each feature
    noise = np.random.normal(loc=0, scale=1, size=n_samples)  # Noise term
    y = X.dot(true_coefficients) + noise
    return X, y


def benchmark(n_samples_list):
    admm_vect_times = []
    admm_vect_errors = []
    admm_par_times = []
    admm_par_errors = []
    admm_cpu_times = []
    admm_cpu_errors = []
    admm_gpu_times = []
    admm_gpu_errors = []
    lasso_times = []
    lasso_errors = []

    for n_samples in tqdm(n_samples_list):
        n_features = 1000  # Number of features (assumed to be fixed)
        x, y = generate_dummy_data(n_samples, n_features)
        # reshape y
        y = y.reshape(-1, 1)
        

        # Benchmark ADMM Parallel
        admm_parallel = ADMM(x, y, parallel=True)
        start_time = time.time()
        for _ in range(2):
            admm_parallel.step()
        admm_parallel_time = time.time() - start_time
        admm_par_times.append(admm_parallel_time)
        admm_par_errors.append(mean_squared_error(admm_parallel.A.dot(admm_parallel.X), admm_parallel.b))


        # Benchmark ADMM Iterative
        admm_iter = ADMM(x, y, parallel=False)
        start_time = time.time()
        for _ in range(2):
            admm_iter.step()
        admm_iter_time = time.time() - start_time
        admm_vect_times.append(admm_iter_time)
        admm_vect_errors.append(mean_squared_error(admm_iter.A.dot(admm_iter.X), admm_iter.b))

        # Benchmark ADMM CPU
        admm_cpu = ADMM_CPU(x, y)
        start_time = time.time()
        for _ in range(2):
            admm_cpu.step()
        admm_cpu_time = time.time() - start_time
        admm_cpu_times.append(admm_cpu_time)
        admm_cpu_errors.append(mean_squared_error(admm_cpu.A.numpy().dot(admm_cpu.X.numpy()), admm_cpu.b.numpy()))


        # Benchmark ADMM GPU
        admm_gpu = ADMM_GPU(x, y)
        start_time = time.time()
        for _ in range(2):
            admm_gpu.step()
        admm_gpu_time = time.time() - start_time
        admm_gpu_times.append(admm_gpu_time)
        admm_gpu_errors.append(mean_squared_error(admm_gpu.A.cpu().numpy().dot(
            admm_gpu.X.cpu().numpy()), admm_gpu.b.cpu().numpy()))



        # Benchmark Lasso regression
        lasso = Lasso(max_iter=100000, warm_start=True)
        start_time = time.time()
        lasso.fit(x, y)
        lasso_time = time.time() - start_time
        lasso_times.append(lasso_time)
        lasso_errors.append(mean_squared_error(lasso.predict(x), y))

    # Save results to CSV using pandas
    results_df = pd.DataFrame({
        'n_samples': n_samples_list,
        'admm_iter_time': admm_vect_times,
        'admm_iter_errors': admm_vect_errors,
        'admm_parallel_time': admm_par_times,
        'admm_parallel_errors': admm_par_errors,
        'admm_cpu_time': admm_cpu_times,
        'admm_cpu_errors': admm_cpu_errors,
        'admm_gpu_time': admm_gpu_times,
        'admm_gpu_errors': admm_gpu_errors,
        'lasso_time': lasso_times,
        'lasso_errors': lasso_errors
    })
    results_df.to_csv('benchmark_results.csv', index=False)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(n_samples_list, admm_vect_times, label="ADMM Iterative")
    plt.plot(n_samples_list, admm_par_times, label="ADMM Parallel")
    plt.plot(n_samples_list, admm_cpu_times, label="ADMM CPU")
    plt.plot(n_samples_list, admm_gpu_times, label="ADMM GPU")
    plt.plot(n_samples_list, lasso_times, label="Lasso")
    plt.xlabel("Number of Samples")
    plt.ylabel("Time (seconds)")
    plt.title("Time Increase in Solving")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(n_samples_list, admm_vect_errors, label="ADMM Iterative")
    plt.plot(n_samples_list, admm_par_errors, label="ADMM Parallel")
    plt.plot(n_samples_list, admm_cpu_errors, label="ADMM CPU")
    plt.plot(n_samples_list, admm_gpu_errors, label="ADMM GPU")
    plt.plot(n_samples_list, lasso_errors, label="Lasso")
    plt.xlabel("Number of Samples")
    plt.ylabel("Mean Squared Error")
    plt.title("Error Comparison")
    plt.legend()

    plt.tight_layout()
    plt.show()


n_samples_list = np.arange(10000, 100000, 1000)

# Run the benchmark
benchmark(n_samples_list)
