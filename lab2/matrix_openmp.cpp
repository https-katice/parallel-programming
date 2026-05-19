#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <chrono>
#include <stdexcept>
#include <direct.h>
#include <omp.h> 

using namespace std;

void create_directory(const string& path) {
    _mkdir(path.c_str());
}

double** allocate_matrix(int n) {
    double** mat = new double* [n];
    for (int i = 0; i < n; i++) {
        mat[i] = new double[n];
    }
    return mat;
}

void free_matrix(double** mat, int n) {
    for (int i = 0; i < n; i++) {
        delete[] mat[i];
    }
    delete[] mat;
}

void generate_random_matrix(double** mat, int n) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(0.0, 10.0);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            mat[i][j] = dist(gen);
        }
    }
}

void save_matrix_to_file(const string& filename, double** mat, int n) {
    ofstream f(filename.c_str());
    if (!f.is_open()) {
        throw runtime_error("Could not open file " + filename + " for writing");
    }
    f << n << "\n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            f << mat[i][j];
            if (j < n - 1) f << " ";
        }
        f << "\n";
    }
    f.close();
}

void multiply_matrices_omp(double** A, double** B, double** C, int n, int num_threads) {
    omp_set_num_threads(num_threads);

#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

int main() {
    try {
        int sizes[] = { 200, 400, 800, 1200, 1600, 2000 };
        int num_sizes = 6;

        int threads_config[] = { 1, 2, 4, 8 };
        int num_configs = 4;

        create_directory("matrices");

        ofstream results("results_openmp.csv");
        if (!results.is_open()) {
            throw runtime_error("Could not create results_openmp.csv");
        }
        results << "size,threads,time_seconds,arithmetic_operations\n";
        results.close();

        cout << "size\tthreads\ttime(s)\t\toperations\n";

        for (int i = 0; i < num_sizes; i++) {
            int n = sizes[i];

            double** A = allocate_matrix(n);
            double** B = allocate_matrix(n);
            double** C = allocate_matrix(n);

            generate_random_matrix(A, n);
            generate_random_matrix(B, n);

            save_matrix_to_file("matrices/matrix_A_" + to_string(n) + ".txt", A, n);
            save_matrix_to_file("matrices/matrix_B_" + to_string(n) + ".txt", B, n);

            for (int t = 0; t < num_configs; t++) {
                int num_threads = threads_config[t];

                auto start = chrono::high_resolution_clock::now();
                multiply_matrices_omp(A, B, C, n, num_threads);
                auto end = chrono::high_resolution_clock::now();
                double elapsed = chrono::duration<double>(end - start).count();

                if (num_threads == 8 || num_threads == threads_config[num_configs - 1]) {
                    save_matrix_to_file("matrices/matrix_C_" + to_string(n) + ".txt", C, n);
                }

                long long arithmetic_ops = 2LL * n * n * n;

                cout << n << "\t" << num_threads << "\t" << elapsed << "\t\t" << arithmetic_ops << "\n";

                results.open("results_openmp.csv", ios::app);
                if (!results.is_open()) {
                    throw runtime_error("Could not open results_openmp.csv for appending");
                }
                results << n << "," << num_threads << "," << elapsed << "," << arithmetic_ops << "\n";
                results.close();
            }

            free_matrix(A, n);
            free_matrix(B, n);
            free_matrix(C, n);
        }
    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    return 0;
}