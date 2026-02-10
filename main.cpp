#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>
#include <climits>
#include <cfloat>
#include <fmt/core.h>
#include <cuda_runtime.h>

#define num_vectores 1000000
#define dimension 3

#define CHECK(expr)                                                                                             \
    {                                                                                                           \
        auto internal_error = (expr);                                                                           \
        if (internal_error != cudaSuccess)                                                                      \
        {                                                                                                       \
            fmt::println("ERROR: {} in {} at line {}", cudaGetErrorString(internal_error), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                                                 \
        }                                                                                                       \
    }

extern "C" void lanzar_encontrar_indice_cercano(const float *vectores, const float *vector_consulta, int num_vectores, int dimension, float *distancias_min, int *indice_cercano);

std::vector<float> generar_vectores(int num_vectores, int dimension)
{
    std::vector<float> vectores(num_vectores * dimension);
    for (int i = 0; i < num_vectores * dimension; ++i)
    {
        vectores[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    return vectores;
}

void calcular_indice_cercano_serial(const std::vector<float> &vectores,
                                    const std::vector<float> &vector_consulta,
                                    int num_vectores, int dimension,
                                    float &distancia_min, int &indice_cercano)
{
    distancia_min = FLT_MAX;
    indice_cercano = -1;

    for (int i = 0; i < num_vectores; ++i)
    {
        float distancia = 0.0f;
        for (int j = 0; j < dimension; ++j)
        {
            float diff = vectores[i * dimension + j] - vector_consulta[j];
            distancia += diff * diff;
        }
        if (distancia < distancia_min)
        {
            distancia_min = distancia;
            indice_cercano = i;
        }
    }
}

int main()
{
    int device_id = 0;
    CHECK(cudaSetDevice(device_id));

    cudaDeviceProp props;
    CHECK(cudaGetDeviceProperties(&props, device_id));
    fmt::println("Device: {}", props.name);
    fmt::println("Max Threads per Block: {}", props.maxThreadsPerBlock);

    std::vector<float> vectores = generar_vectores(num_vectores, dimension);
    std::vector<float> vector_consulta = {0.5f, 0.5f, 0.5f};

    float h_distancia_min_serial = FLT_MAX;
    int h_indice_cercano_serial = -1;
    calcular_indice_cercano_serial(vectores, vector_consulta, num_vectores, dimension,
                                   h_distancia_min_serial, h_indice_cercano_serial);

    fmt::println("Índice del vector más cercano (CPU): {}", h_indice_cercano_serial);
    fmt::println("Distancia mínima (CPU): {:.6f}", h_distancia_min_serial);

    float *d_distancias_min;
    int *d_indice_cercano;
    float *d_vectores;
    float *d_vector_consulta;
    float h_distancia_min_gpu = FLT_MAX;
    int h_indice_cercano_gpu = -1;

    CHECK(cudaMalloc(&d_distancias_min, sizeof(float)));
    CHECK(cudaMalloc(&d_indice_cercano, sizeof(int)));
    CHECK(cudaMalloc(&d_vectores, sizeof(float) * num_vectores * dimension));
    CHECK(cudaMalloc(&d_vector_consulta, sizeof(float) * dimension));

    CHECK(cudaMemcpy(d_vectores, vectores.data(), sizeof(float) * num_vectores * dimension, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_vector_consulta, vector_consulta.data(), sizeof(float) * dimension, cudaMemcpyHostToDevice));

    CHECK(cudaMemcpy(d_distancias_min, &h_distancia_min_gpu, sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_indice_cercano, &h_indice_cercano_gpu, sizeof(int), cudaMemcpyHostToDevice));

    lanzar_encontrar_indice_cercano(d_vectores, d_vector_consulta, num_vectores, dimension,
                                    d_distancias_min, d_indice_cercano);

    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(&h_distancia_min_gpu, d_distancias_min, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&h_indice_cercano_gpu, d_indice_cercano, sizeof(int), cudaMemcpyDeviceToHost));

    fmt::println("Índice del vector más cercano (GPU): {}", h_indice_cercano_gpu);
    fmt::println("Distancia mínima (GPU): {:.6f}", h_distancia_min_gpu);

    CHECK(cudaFree(d_distancias_min));
    CHECK(cudaFree(d_indice_cercano));
    CHECK(cudaFree(d_vectores));
    CHECK(cudaFree(d_vector_consulta));
    return 0;
}