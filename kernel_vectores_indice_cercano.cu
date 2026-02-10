#include <cmath>
#include <iostream>
#include <float.h>

__device__ 
float calcular_distancia(const float* vectores, const float* vector_consulta, 
                         int idx, int dimension) {
    float distancia = 0.0f;
    for (int d = 0; d < dimension; ++d) {
        float diff = vectores[idx * dimension + d] - vector_consulta[d];
        distancia += diff * diff;
    }
    return sqrtf(distancia);
}

__global__ 
void kernel_vectores_indice_cercano(const float* __restrict__ vectores, 
                                     const float* __restrict__ vector_consulta, 
                                     int num_vectores, int dimension, 
                                     float* distancias_min, int* indice_cercano) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_vectores) {
        float distancia = calcular_distancia(vectores, vector_consulta, idx, dimension);
        
        // Comparar con distancia mínima usando atomicMin_block (comparar floats)
        if (distancia < *distancias_min) {
            atomicMin((unsigned int*)distancias_min, __float_as_uint(distancia));
            *indice_cercano = idx;
        }
    }
}

__host__ 
void encontrar_indice_cercano_gpu(const float* vectores, const float* vector_consulta, 
                                   int num_vectores, int dimension, 
                                   float* distancias_min, int* indice_cercano) {
    int threads_per_block = 256;
    int blocks = (num_vectores + threads_per_block - 1) / threads_per_block;
    kernel_vectores_indice_cercano<<<blocks, threads_per_block>>>(
        vectores, vector_consulta, num_vectores, dimension, distancias_min, indice_cercano);
}

extern "C" 
void lanzar_encontrar_indice_cercano(const float* vectores, const float* vector_consulta, 
                                      int num_vectores, int dimension, 
                                      float* distancias_min, int* indice_cercano) {
    encontrar_indice_cercano_gpu(vectores, vector_consulta, num_vectores, dimension, 
                                  distancias_min, indice_cercano);
}

