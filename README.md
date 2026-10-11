# 🚀 Sistema Híbrido de Ordenamiento y Búsqueda Paralela (MPI + OpenMP)

Este proyecto implementa una solución de **Computación de Alto Rendimiento (HPC)** en C que combina el modelo de **memoria compartida (OpenMP)** y **memoria distribuida (MPI)**. 

El sistema permite generar arreglos masivos de datos, ordenarlos de manera paralela utilizando **Merge Sort con Tareas** y realizar búsquedas paralelas (secuenciales y binarias) tanto en un solo nodo como distribuidas a través de un clúster de procesos.

---

## 🛠️ Características Principales

* **Ordenamiento Paralelo (`parallel_merge_sort`)**: 
  * Implementa Merge Sort recursivo usando `#pragma omp task` de OpenMP.
  * Incluye un **umbral (threshold)** dinámico para evitar la sobrecarga de hilos en tramos pequeños de datos.
* **Búsqueda Secuencial Paralela (`busquedaSecuencialBloque`)**:
  * Paralelización de bucles con `#pragma omp parallel for` para inspeccionar bloques de memoria no ordenados.
* **Búsqueda Binaria Paralela (`busquedaBinariaBloque`)**:
  * Divide el arreglo ordenado entre los hilos del proceso.
  * Cada hilo descarta su bloque en tiempo $O(1)$ revisando solo sus extremos, garantizando que únicamente el hilo correspondiente ejecute la búsqueda binaria.
* **Arquitectura Híbrida MPI + OpenMP**:
  * Distribución de bloques de datos entre múltiples nodos o procesos con `MPI_Scatter`.
  * Reducción y consolidación de resultados globales mediante `MPI_Reduce`.
* **Sistema de Logging Thread-Safe (`registrarLog`)**:
  * Manejo de argumentos variables (`va_list`) con protección de secciones críticas (`#pragma omp critical`) para evitar corrupción de registros en consola y archivos de texto.

---

## 📋 Requisitos Previos

Para compilar y ejecutar este programa necesitas:

* Un compilador de C (como `gcc` o `clang`) con soporte para **OpenMP**.
* Una implementación de **MPI** instalada en tu sistema (ej. **OpenMPI** o **MPICH**).

En sistemas basados en Debian/Ubuntu puedes instalarlos con:
```bash
sudo apt update
sudo apt install build-essential openmpi-bin libopenmpi-dev
