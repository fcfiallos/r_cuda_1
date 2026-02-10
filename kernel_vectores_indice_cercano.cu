#include <cmath>
#include <iostream>
#include <float.h>
#include <cuda_runtime.h>

__device__ float calcular_distancia(const float *vectores, const float *vector_consulta,
                                    int idx, int dimension)
{
    float distancia = 0.0f;
    for (int d = 0; d < dimension; ++d)
    {
        float diff = vectores[idx * dimension + d] - vector_consulta[d];
        distancia += diff * diff;
    }
    return distancia;
}

__global__ void kernel_minimo_por_bloque(const float *__restrict__ vectores,
                                         const float *__restrict__ vector_consulta,
                                         int num_vectores, int dimension,
                                         float *distancias_min_bloque,
                                         int *indice_cercano_bloque)
{
    extern __shared__ unsigned char shared_raw[];
    float *shared_dist = reinterpret_cast<float *>(shared_raw);
    int *shared_idx = reinterpret_cast<int *>(shared_dist + blockDim.x);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float distancia = FLT_MAX;
    int indice = -1;

    if (idx < num_vectores)
    {
        distancia = calcular_distancia(vectores, vector_consulta, idx, dimension);
        indice = idx;
    }

    shared_dist[threadIdx.x] = distancia;
    shared_idx[threadIdx.x] = indice;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1)
    {
        if (threadIdx.x < offset)
        {
            float other_dist = shared_dist[threadIdx.x + offset];
            int other_idx = shared_idx[threadIdx.x + offset];
            if (other_dist < shared_dist[threadIdx.x])
            {
                shared_dist[threadIdx.x] = other_dist;
                shared_idx[threadIdx.x] = other_idx;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        distancias_min_bloque[blockIdx.x] = shared_dist[0];
        indice_cercano_bloque[blockIdx.x] = shared_idx[0];
    }
}

__global__ void kernel_reducir_minimos(const float *__restrict__ distancias_min_bloque,
                                       const int *__restrict__ indice_cercano_bloque,
                                       int num_bloques,
                                       float *distancias_min,
                                       int *indice_cercano)
{
    extern __shared__ unsigned char shared_raw[];
    float *shared_dist = reinterpret_cast<float *>(shared_raw);
    int *shared_idx = reinterpret_cast<int *>(shared_dist + blockDim.x);

    float distancia = FLT_MAX;
    int indice = -1;

    for (int i = threadIdx.x; i < num_bloques; i += blockDim.x)
    {
        float d = distancias_min_bloque[i];
        if (d < distancia)
        {
            distancia = d;
            indice = indice_cercano_bloque[i];
        }
    }

    shared_dist[threadIdx.x] = distancia;
    shared_idx[threadIdx.x] = indice;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1)
    {
        if (threadIdx.x < offset)
        {
            float other_dist = shared_dist[threadIdx.x + offset];
            int other_idx = shared_idx[threadIdx.x + offset];
            if (other_dist < shared_dist[threadIdx.x])
            {
                shared_dist[threadIdx.x] = other_dist;
                shared_idx[threadIdx.x] = other_idx;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        *distancias_min = shared_dist[0];
        *indice_cercano = shared_idx[0];
    }
}

__host__ void encontrar_indice_cercano_gpu(const float *vectores, const float *vector_consulta,
                                           int num_vectores, int dimension,
                                           float *distancias_min, int *indice_cercano)
{
    int threads_per_block = 256;
    int blocks = (num_vectores + threads_per_block - 1) / threads_per_block;

    float *d_distancias_min_bloque = nullptr;
    int *d_indice_cercano_bloque = nullptr;
    cudaMalloc(&d_distancias_min_bloque, sizeof(float) * blocks);
    cudaMalloc(&d_indice_cercano_bloque, sizeof(int) * blocks);

    size_t shared_bytes = (sizeof(float) + sizeof(int)) * threads_per_block;
    kernel_minimo_por_bloque<<<blocks, threads_per_block, shared_bytes>>>(
        vectores, vector_consulta, num_vectores, dimension,
        d_distancias_min_bloque, d_indice_cercano_bloque);

    kernel_reducir_minimos<<<1, threads_per_block, shared_bytes>>>(
        d_distancias_min_bloque, d_indice_cercano_bloque, blocks,
        distancias_min, indice_cercano);

    cudaFree(d_distancias_min_bloque);
    cudaFree(d_indice_cercano_bloque);
}

extern "C" void lanzar_encontrar_indice_cercano(const float *vectores, const float *vector_consulta,
                                                int num_vectores, int dimension,
                                                float *distancias_min, int *indice_cercano)
{
    encontrar_indice_cercano_gpu(vectores, vector_consulta, num_vectores, dimension,
                                 distancias_min, indice_cercano);
}
