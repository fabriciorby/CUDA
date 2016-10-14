#define N			16
#define RADIUS			1
#define BLOCK_SIZE 		16

#include <float.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

//algoritmo de dilatação em GPU

// incluir valores aleatórios < N no vetor
void draw_random(float *im, int n) {
     // initialize random generator with a random seed only once
     srand((unsigned int)time(NULL));
     static int seed = rand();
     // use same seed to draw the same image again on every test
     srand(seed);
     for (int i = 0; i<RADIUS; i++)
        im[i] = im[n-i-1] = 0;
     for (int i = RADIUS; i < n - RADIUS; i++) {
	im[i] = (float) (rand() % N) ;
     }
}
// imprimir vertor
void disp_img(float const *img, int n) {
    for (int col = 0; col < n; col++)
	printf("%2.0f ", img[col]);
    printf("\n");
}

__global__ void kernel(float *in, float *out) {
  __shared__ float temp[BLOCK_SIZE + 2 * RADIUS];
  int gindex = threadIdx.x + blockIdx.x * blockDim.x;
  int lindex = threadIdx.x + RADIUS;
  // Read input elements into shared memory
  temp[lindex] = in[gindex];
  if (threadIdx.x < RADIUS) {
     temp[lindex - RADIUS]     = in[gindex - RADIUS];
     temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
  }

  // Synchronize (ensure all the data is available)
  __syncthreads();
  // Apply 
  float result = in[gindex];
  for (int offset = -RADIUS ; offset <= RADIUS ; offset++)
      result = fmaxf(result, temp[lindex + offset]);

  // Store the result
  out[gindex] = result;
}

int main(void) {
  float *in, *out;        // host copies of a, b, c
  float *d_in, *d_out;    // device copies of a, b, c
  size_t size = (N + 2*RADIUS) * sizeof(float);

  // Alloc space for host copies and setup values
  in  = (float*) malloc(size); 
  out = (float*) malloc(size);

  draw_random(in,  N + 2*RADIUS);
  disp_img(in, N + 2*RADIUS);
 
  // Alloc space for device copies
  cudaMalloc((void **)&d_in,  size);
  cudaMalloc((void **)&d_out, size);

  // Copy to device
  cudaMemcpy(d_in,  in,  size, cudaMemcpyHostToDevice);

  // Launch kernel on GPU
  kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_in+RADIUS, d_out+RADIUS);

  // Copy result back to host
  cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

  disp_img(out, N + 2*RADIUS);

  // Cleanup
  free(in); free(out);
  cudaFree(d_in); cudaFree(d_out);
  return 0;
}
