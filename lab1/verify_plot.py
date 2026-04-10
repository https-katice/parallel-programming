import numpy as np
import matplotlib.pyplot as plt

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
        print(f"Файлы для размера {n} не найдены")
        return False
    except Exception as e:
        print(f"Ошибка при проверке размера {n}: {e}")
        return False

def plot_results():
    data = np.loadtxt("results.csv", delimiter=",", skiprows=1)
    sizes = data[:, 0]
    times = data[:, 1]
    
    idx = np.argsort(sizes)
    sizes = sizes[idx]
    times = times[idx]
    
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times, 'bo-', linewidth=2, markersize=8, label='Время выполнения')
    plt.xlabel('Размер матрицы nxn', fontsize=12)
    plt.ylabel('Время (секунды)', fontsize=12)
    plt.title('Зависимость времени перемножения от размера матрицы', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    for size, t in zip(sizes, times):
        plt.annotate(f'{t:.2f}с', (size, t), textcoords="offset points", xytext=(5, 5), ha='center')
    
    plt.tight_layout()
    plt.savefig('plot.png', dpi=150)

def main():
    sizes = [200, 400, 800, 1200, 1600]
    for n in sizes:
        if verify_for_size(n):
            print(f"size passed")
        else:
            print(f"size fail")
    plot_results()

if __name__ == "__main__":
    main()