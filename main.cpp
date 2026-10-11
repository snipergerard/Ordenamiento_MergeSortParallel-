#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <mpi.h>
#include <string.h>
#include <time.h>
#include <stdarg.h>

/*
mpiexec -hosts 2 192.168.100.12 1 192.168.100.9 1 \\192.168.100.12\MPI_Compartido\FuerzaBruta\bin\Debug\FuerzaBruta.exe

Actividad 1.5 Ordenamiento en merge short parallel distribuido utilizando MPI con OpenMP
Equipo 9 Programacion Paralela
Gerardo Josue Rubio Calderon
Argenis Adan Gutierrez Hurtado


*/


void registrarLog(FILE* logFile, int consola, const char* formato, ...) {
    va_list args1, args2;
    va_start(args1, formato);
    va_copy(args2, args1);
    #pragma omp critical
    {
        if (consola) {
            vprintf(formato, args1);
            fflush(stdout);
        }
        if (logFile != NULL) {
            vfprintf(logFile, formato, args2);
            fflush(logFile);
        }
    }
    va_end(args1);
    va_end(args2);
}

// Función segura para imprimir el arreglo
void imprimirArreglo(int* A, int N, const char* titulo, FILE* logFile) {
    if (N > 100) {
        registrarLog(logFile, 1, "[MAESTRO] %s (Impresion omitida por seguridad, N > 100)\n", titulo);
        return;
    }
    registrarLog(logFile, 1, "\n--- %s ---\n", titulo);
    for(int i = 0; i < N; i++) {
        registrarLog(logFile, 1, "%d ", A[i]);
    }
    registrarLog(logFile, 1, "\n-----------------------\n");
}

void merge(int* A, int* temp, int left, int mid, int right) {
    int i = left, j = mid + 1, k = left;
    while (i <= mid && j <= right) {
        if (A[i] <= A[j]) temp[k++] = A[i++];
        else temp[k++] = A[j++];
    }
    while (i <= mid) temp[k++] = A[i++];
    while (j <= right) temp[k++] = A[j++];
    for (i = left; i <= right; i++) A[i] = temp[i];
}

void parallel_merge_sort(int* A, int* temp, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        if (right - left > 10000) {
            #pragma omp task shared(A, temp)
            parallel_merge_sort(A, temp, left, mid);
            #pragma omp task shared(A, temp)
            parallel_merge_sort(A, temp, mid + 1, right);
            #pragma omp taskwait
            merge(A, temp, left, mid, right);
        } else {
            parallel_merge_sort(A, temp, left, mid);
            parallel_merge_sort(A, temp, mid + 1, right);
            merge(A, temp, left, mid, right);
        }
    }
}

int busquedaSecuencialBloque(int* arreglo, int size, int target, int offset, const char* equipo, int rank, FILE* logFile, int imprimir) {
    int encontrado = -1;
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        if (arreglo[i] == target) {
            // Candado con "nombre" para que no choque con el de registrarLog
            #pragma omp critical (act_encontrado)
            {
                encontrado = offset + i;
            }
            if (imprimir) {
                // registrarLog ya tiene su propio candado interno, lo dejamos correr libre
                registrarLog(logFile, 1, "[Equipo: %s] [Nodo MPI: %d] [Hilo OpenMP: %d] [Bloque: %d - %d] [Algoritmo: Secuencial] [Hallado en indice: %d]\n",
                             equipo, rank, omp_get_thread_num(), offset, offset + size - 1, offset + i);
            }
        }
    }
    return encontrado;
}

int busquedaBinariaBloque(int* arreglo, int size, int target, int offset, const char* equipo, int rank, FILE* logFile, int imprimir) {
    int encontrado = -1;
    if (size > 0 && target >= arreglo[0] && target <= arreglo[size - 1]) {
        #pragma omp parallel
        {
            int num_hilos = omp_get_num_threads();
            int id_hilo = omp_get_thread_num();
            int chunk_size = size / num_hilos;
            int inicio = id_hilo * chunk_size;
            int fin = (id_hilo == num_hilos - 1) ? size - 1 : inicio + chunk_size - 1;

            if (target >= arreglo[inicio] && target <= arreglo[fin]) {
                int l = inicio, r = fin;
                while (l <= r) {
                    int m = l + (r - l) / 2;
                    if (arreglo[m] == target) {
                        #pragma omp critical (act_encontrado)
                        {
                            encontrado = offset + m;
                        }
                        if (imprimir) {
                            registrarLog(logFile, 1, "[Equipo: %s] [Nodo MPI: %d] [Hilo OpenMP: %d] [Bloque: %d - %d] [Algoritmo: Binaria] [Hallado en indice: %d]\n",
                                         equipo, rank, id_hilo, offset + inicio, offset + fin, offset + m);
                        }
                        break;
                    }
                    if (arreglo[m] < target) l = m + 1;
                    else r = m - 1;
                }
            }
        }
    }
    return encontrado;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int num_procesos, rank_mpi, long_name;
    char nombre_equipo[MPI_MAX_PROCESSOR_NAME];

    MPI_Comm_size(MPI_COMM_WORLD, &num_procesos);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_mpi);
    MPI_Get_processor_name(nombre_equipo, &long_name);

    char nombre_archivo_log[256];
    sprintf(nombre_archivo_log, "log_equipo_%s_nodo_%d.txt", nombre_equipo, rank_mpi);
    FILE* logFile = fopen(nombre_archivo_log, "w");

    registrarLog(logFile, 0, "=== Sesion Iniciada: [Equipo: %s] [Nodo MPI: %d] ===\n", nombre_equipo, rank_mpi);

    int N = 0;
    int *A = NULL;
    int *sub_A = NULL;
    int opcion_menu = 0;
    int target = 0, modo = 0;
    double t_inicio, t_fin;
    double tiempo_ordenamiento = 0, tiempo_secuencial_loc = 0, tiempo_secuencial_dist = 0;
    double tiempo_binaria_loc = 0, tiempo_binaria_dist = 0;

    do {
        if (rank_mpi == 0) {
            printf("\n=== MENU PRINCIPAL (BUSQUEDA HIBRIDA) ===\n");
            printf("1. Generar Arreglo Dinamico\n");
            printf("2. Ordenar Arreglo con OpenMP\n");
            printf("3. Ejecutar Busqueda Secuencial\n");
            printf("4. Ejecutar Busqueda Binaria\n");
            printf("5. Mostrar Tiempos y Comparativa\n");
            printf("0. Salir\n");
            printf("Seleccione una opcion: ");
            fflush(stdout);
            scanf("%d", &opcion_menu);
        }

        MPI_Bcast(&opcion_menu, 1, MPI_INT, 0, MPI_COMM_WORLD);

        switch (opcion_menu) {
            case 1:
                if (rank_mpi == 0) {
                    printf("Ingrese el tamano del arreglo (N): ");
                    fflush(stdout);
                    scanf("%d", &N);
                    if (N % num_procesos != 0) {
                        N = N + (num_procesos - (N % num_procesos));
                            }
                }
                MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

                if (A) free(A);
                if (sub_A) free(sub_A);

                sub_A = (int*)malloc((N / num_procesos) * sizeof(int));

                if (rank_mpi == 0) {
                    A = (int*)malloc(N * sizeof(int));
                    srand(time(NULL));
                    for (int i = 0; i < N; i++) A[i] = (rand() % 1000) + 1; // Rango de 1 a 1000 para probar facil
                    registrarLog(logFile, 1, "[MAESTRO] Arreglo de %d elementos generado.\n", N);

                    // Llama a la funcion de impresion de seguridad
                    imprimirArreglo(A, N, "ARREGLO DESORDENADO", logFile);
                }
                break;

            case 2:
                if (rank_mpi == 0 && A != NULL) {
                    int* temp = (int*)malloc(N * sizeof(int));
                    t_inicio = MPI_Wtime();
                    #pragma omp parallel
                    {
                        #pragma omp single
                        parallel_merge_sort(A, temp, 0, N - 1);
                    }
                    t_fin = MPI_Wtime();
                    tiempo_ordenamiento = t_fin - t_inicio;
                    free(temp);
                    registrarLog(logFile, 1, "[MAESTRO] Ordenamiento completado en %f seg.\n", tiempo_ordenamiento);

                    // Imprime el arreglo ordenado si N es pequeno
                    imprimirArreglo(A, N, "ARREGLO ORDENADO", logFile);
                }
                break;

            case 3:
            case 4:
                if (N == 0) break;
                if (rank_mpi == 0) {
                    printf("Ingrese el valor a buscar (Target): ");
                    fflush(stdout);
                    scanf("%d", &target);
                    printf("Modo (1 = Local Maestro, 2 = Distribuido): ");
                    fflush(stdout);
                    scanf("%d", &modo);
                }
                MPI_Bcast(&target, 1, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Bcast(&modo, 1, MPI_INT, 0, MPI_COMM_WORLD);

                if (modo == 1) {
                    if (rank_mpi == 0) {
                        t_inicio = MPI_Wtime();
                        int idx = -1;
                        if (opcion_menu == 3) idx = busquedaSecuencialBloque(A, N, target, 0, nombre_equipo, rank_mpi, logFile, (N <= 100));
                        else idx = busquedaBinariaBloque(A, N, target, 0, nombre_equipo, rank_mpi, logFile, (N <= 100));
                        t_fin = MPI_Wtime();

                        if (opcion_menu == 3) tiempo_secuencial_loc = t_fin - t_inicio;
                        else tiempo_binaria_loc = t_fin - t_inicio;

                        if (idx != -1) registrarLog(logFile, 1, ">>> [MAESTRO] Encontrado en indice %d <<<\n", idx);
                        else registrarLog(logFile, 1, ">>> [MAESTRO] No encontrado. <<<\n");
                    }
                } else if (modo == 2) {
                    int size_local = N / num_procesos;
                    MPI_Scatter(A, size_local, MPI_INT, sub_A, size_local, MPI_INT, 0, MPI_COMM_WORLD);

                    t_inicio = MPI_Wtime();
                    int idx_local = -1;
                    int offset = rank_mpi * size_local;

                    if (opcion_menu == 3) idx_local = busquedaSecuencialBloque(sub_A, size_local, target, offset, nombre_equipo, rank_mpi, logFile, (N <= 100));
                    else idx_local = busquedaBinariaBloque(sub_A, size_local, target, offset, nombre_equipo, rank_mpi, logFile, (N <= 100));

                    int idx_global = -1;
                    MPI_Reduce(&idx_local, &idx_global, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
                    t_fin = MPI_Wtime();

                    if (rank_mpi == 0) {
                        if (opcion_menu == 3) tiempo_secuencial_dist = t_fin - t_inicio;
                        else tiempo_binaria_dist = t_fin - t_inicio;

                        if (idx_global != -1) registrarLog(logFile, 1, ">>> [RED] Encontrado en indice global %d <<<\n", idx_global);
                        else registrarLog(logFile, 1, ">>> [RED] No encontrado en ningun bloque. <<<\n");
                    }
                }
                break;

            case 5:
                if (rank_mpi == 0) {
                    printf("\n--- COMPARATIVA DE RENDIMIENTO ---\n");
                    printf("Ordenamiento Paralelo (OpenMP): %f seg\n", tiempo_ordenamiento);
                    printf("Secuencial (Local):             %f seg\n", tiempo_secuencial_loc);
                    printf("Secuencial (Dist):              %f seg\n", tiempo_secuencial_dist);
                    printf("Binaria (Local):                %f seg\n", tiempo_binaria_loc);
                    printf("Binaria (Dist):                 %f seg\n", tiempo_binaria_dist);
                    printf("----------------------------------\n");
                    fflush(stdout);
                }
                break;
        }
    } while (opcion_menu != 0);

    if (A) free(A);
    if (sub_A) free(sub_A);
    if (logFile) fclose(logFile);

    MPI_Finalize();
    return 0;
}

