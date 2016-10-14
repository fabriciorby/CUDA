#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define BLOCK_SIZE 16

typedef struct {
int width;
int height;
float* elements;
} Matrix;

// Matrix multiplication kernel called by MatMul()
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

int main(int argc, char* argv[])
{
  Matrix A;
  Matrix B;
  Matrix C;

  int a = 4;
  int b = 4;
  int c = 8;

  A = createMatrix(a, b);
  B = createMatrix(b, c);
  C = createMatrix(A.height, B.width);

  startSeed();
  
  draw_random(A);
  draw_random(B);

  printf(" Matriz A \n");
  disp_img(A);
  printf(" Matriz B \n");
  disp_img(B);
  MatMul(A, B, C);
  printf(" Matriz C \n");
  disp_img(C);
}