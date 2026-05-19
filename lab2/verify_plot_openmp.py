import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def verify_for_size(n, folder="matrices"):
    try:
        A = np.loadtxt(f"{folder}/matrix_A_{n}.txt", skiprows=1)
        B = np.loadtxt(f"{folder}/matrix_B_{n}.txt", skiprows=1)
        C_computed = np.loadtxt(f"{folder}/matrix_C_{n}.txt", skiprows=1)
        C_expected = np.dot(A, B)
        
        if np.allclose(C_computed, C_expected, rtol=1e-5, atol=1e-5):
            return True
        else:
            return False
    except FileNotFoundError:
        print(f"Files for size {n} not found")
        return False
    except Exception as e:
        print(f"Error checking size {n}: {e}")
        return False

def plot_results():
    df = pd.read_csv("results_openmp.csv")
    
    sizes = sorted(df['size'].unique())
    threads = sorted(df['threads'].unique())
    
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for t in threads:
        sub_df = df[df['threads'] == t].sort_values(by='size')
        plt.plot(sub_df['size'], sub_df['time_seconds'], 'o-', linewidth=2, label=f'{t} threads')
    
    plt.xlabel('Matrix size nxn', fontsize=11)
    plt.ylabel('Time (seconds)', fontsize=11)
    plt.title('Execution Time (OpenMP)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    t1_times = df[df['threads'] == 1].sort_values(by='size')['time_seconds'].values
    
    for t in threads:
        sub_df = df[df['threads'] == t].sort_values(by='size')
        tn_times = sub_df['time_seconds'].values
        speedup = t1_times / tn_times
        plt.plot(sub_df['size'], speedup, 's--', linewidth=2, label=f'{t} threads')
        
    plt.xlabel('Matrix size nxn', fontsize=11)
    plt.ylabel('Speedup S = T1 / Tn', fontsize=11)
    plt.title('Speedup Coefficient', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plot_openmp.png', dpi=150)
    print("Graph saved as plot_openmp.png")

def main():
    sizes = [200, 400, 800, 1200, 1600, 2000]
    print("--- Verification ---")
    for n in sizes:
        if verify_for_size(n):
            print(f"Size {n}: PASSED")
        else:
            print(f"Size {n}: FAIL")
            
    print("\n--- Plotting ---")
    plot_results()

if __name__ == "__main__":
    main()
