#define N	16
#define k 5
#define BLOCK_SIZE 16

#include <float.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

//algoritmo de transformada de distância em GPU

// incluir valores aleatórios < N no vetor
void draw_random(float *im, int n) {
     // initialize random generator with a random seed only once
     srand((unsigned int)time(NULL));
     static int seed = rand();
     // use same seed to draw the same image again on every test
     srand(seed);
     for (int i = 0; i < n; i++) {
      if((rand() % 2) == 0)
	       im[i] = 0.00;
      else
        im[i] = (float) k;
     }
}
// imprimir vertor
void disp_img(float const *img, int n) {
    for (int col = 0; col < n; col++)
  printf("%2.0f ", img[col]);
    printf("\n");
}

__global__ void kernel(float *in, float *out) {
  int gindex = threadIdx.x + blockIdx.x * blockDim.x;
  // Apply 
  float elemento = in[gindex];
  float result;
  bool flagDir = false;
  bool flagEsq = false;
  int i = gindex + 1;
  int contEsq = 0;
  int contDir = 0;
  if (elemento == k)
  {
    while(i < N)
    {
      elemento = in[i];
      if (elemento == 0)
      {
        flagDir = true;
        break;
      }
      i++;
      contDir++;
    }
    i = gindex - 1;
    while(i >= 0)
    {
      elemento = in[i];
      if (elemento == 0)
      {
        flagEsq = true;
        break;
      }
      i--;
      contEsq++;
    }
    if (flagDir && flagEsq)
    {
      result = fminf(contDir, contEsq);
    } else if (flagDir)
    {
      result = contDir;
    } else
    {
      result = contEsq;
    }
      
  }
  out[gindex] = result;
}

int main(void) {
  float *in, *out;        // host copies of a, b, c
  float *d_in, *d_out;    // device copies of a, b, c
  size_t size = (N) * sizeof(float);

  // Alloc space for host copies and setup values
  in  = (float*) malloc(size); 
  out = (float*) malloc(size);

  draw_random(in,  N);
  disp_img(in, N);
 
  // Alloc space for device copies
  cudaMalloc((void **)&d_in,  size);
  cudaMalloc((void **)&d_out, size);

  // Copy to device
  cudaMemcpy(d_in,  in,  size, cudaMemcpyHostToDevice);

  // Launch kernel on GPU
  kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_in, d_out);

  // Copy result back to host
  cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

  disp_img(out, N);

  // Cleanup
  free(in); free(out);
  cudaFree(d_in); cudaFree(d_out);
  return 0;
}
