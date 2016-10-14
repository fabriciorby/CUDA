#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define BLOCK_SIZE 50

//Um teste comparando a eficiência de uma
//multiplicação de matrizes por CPU ou GPU
//utilizando memória compartilhada ou global

typedef struct {
  int width;
  int height;
  int stride;
  float *elements;
} Matrix;

void startSeed()
{
    srand(time(NULL));
    int seed = rand();
    srand(seed);
}

void draw_random(Matrix mat) {
     for (int i = 0; i < mat.height*mat.width; i++)
     {
      mat.elements[i] = (float) (rand() % 10);
     }
}

void disp_img(Matrix mat) {
    for (int i = 0; i < mat.height; i++)
     {
        for (int j = 0; j < mat.width; j++)
        {
          printf("%5.0f", mat.elements[i*mat.width + j]);
        }
        printf("\n");
     }
     printf("\n");
}

Matrix createMatrix(int height, int width)
{
    Matrix mat;

    mat.width = width;
    mat.height = height;
    mat.elements = (float*) malloc(mat.width*mat.height*sizeof(float));

    for(int i = 0; i < mat.height; i++)
      for(int j = 0; j < mat.width; j++)
        mat.elements[i*mat.width + j] = 0;
    
    return mat;
}

void multiMatrixCPU(Matrix A, Matrix B, Matrix C)
{
    for (int i = 0; i < A.height; i++) {
        for (int j = 0; j < B.width; j++) {
            C.elements[j + i * B.width] = 0;
            for (int k = 0; k < A.width; k++) {
                C.elements[j + i * C.width] += A.elements[k + i * A.width] * B.elements[j + k * B.width];
            }
        }
    }
}

__device__ float GetElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}

__device__ void SetElement(Matrix A, int row, int col,
                           float value)
{
    A.elements[row * A.stride + col] = value;
}

 __device__ Matrix GetSubMatrix(Matrix A, int row, int col) 
{
    Matrix Asub;
    Asub.width    = BLOCK_SIZE;
    Asub.height   = BLOCK_SIZE;
    Asub.stride   = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return Asub;
}

 __global__ void MatMulKernelShared(Matrix A, Matrix B, Matrix C)
{

    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

    float Cvalue = 0;

    int row = threadIdx.y;
    int col = threadIdx.x;

    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {

        Matrix Asub = GetSubMatrix(A, blockRow, m);

        Matrix Bsub = GetSubMatrix(B, m, blockCol);

        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);

        __syncthreads();

        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];

        __syncthreads();
    }

    SetElement(Csub, row, col, Cvalue);
}

void MatMulShared(const Matrix A, const Matrix B, Matrix C)
{
    Matrix d_A;
    d_A.width = d_A.stride = A.width;
    d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);

    Matrix d_B;
    d_B.width = d_B.stride = B.width;
    d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);

    Matrix d_C;
    d_C.width = d_C.stride = C.width;
    d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    
    MatMulKernelShared<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) 
{
  // Each thread computes one element of C
  // by accumulating results into Cvalue
  float Cvalue = 0;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < A.height && col < B.width)
  {
    for (int e = 0; e < A.width; e++)
      Cvalue += (A.elements[row * A.width + e]) * (B.elements[e * B.width + col]);
    C.elements[row * C.width + col] = Cvalue;
  }
  
}

void MatMul(const Matrix A, const Matrix B, Matrix C) 
{
  // Load A and B to device memory 
  Matrix d_A;
  d_A.width = A.width;
  d_A.height = A.height;
  size_t size = A.width * A.height * sizeof(float);
  cudaMalloc(&d_A.elements, size);
  cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
  
  Matrix d_B;
  d_B.width = B.width;
  d_B.height = B.height;
  size = B.width * B.height * sizeof(float);
  cudaMalloc(&d_B.elements, size);
  cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
  
  // Allocate C in device memory
  Matrix d_C;
  d_C.width = C.width;
  d_C.height = C.height;
  size = C.width * C.height * sizeof(float);
  cudaMalloc(&d_C.elements, size);
  
  // Invoke kernel
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid((B.width + dimBlock.x - 1) / dimBlock.x, (A.height + dimBlock.y - 1) / dimBlock.y);
  MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
  cudaThreadSynchronize();
  
  // Read C from device memory
  cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
  
  // Free device memory
  cudaFree(d_A.elements);
  cudaFree(d_B.elements);
  cudaFree(d_C.elements);
}

int main(int argc, char* argv[])
{
  clock_t tic;
  clock_t toc;

  Matrix A;
  Matrix B;
  Matrix C;

  int a;
  int b;
  int c;

  startSeed();

  int num_devices, device;
  printf("Multiplicação de Matrizes\n");
  printf("Neste programa foi utilizado um BLOCK_SIZE = 50\n\n");
  cudaGetDeviceCount(&num_devices);
  for (device = 0; device < num_devices; device++) {
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, device);
    cudaSetDevice(device);
    printf("Utilizando uma %s:\n\n", properties.name);
    for (int i = 100; i <= 1000; i += 100)
    {
      a = i;
      b = i;
      c = i;

      A = createMatrix(a, b);
      B = createMatrix(b, c);
      C = createMatrix(A.height, B.width);

      printf("A[%d][%d] * B[%d][%d]\n", a, b, b, c);
      
      draw_random(A);
      draw_random(B);

      tic = clock();
      MatMul(A, B, C);
      toc = clock();
      printf("GPU (global): %.3fms\n", (double)(toc - tic) / CLOCKS_PER_SEC*1000);
      C = createMatrix(A.height, B.width);

      draw_random(A);
      draw_random(B);

      tic = clock();
      MatMulShared(A, B, C);
      toc = clock();
      printf("GPU (shared): %.3fms\n", (double)(toc - tic) / CLOCKS_PER_SEC*1000);
      C = createMatrix(A.height, B.width);

      draw_random(A);
      draw_random(B);

      tic = clock();
      multiMatrixCPU(A, B, C);
      toc = clock();
      printf("CPU: %.3fms\n", (double)(toc - tic) / CLOCKS_PER_SEC*1000);
      printf("\n");
    }
  }
}