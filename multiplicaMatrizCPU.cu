#include <stdlib.h>
//#include <float.h>
#include <stdio.h>
//#include <string.h>
//#include <math.h>
#include <time.h>

typedef struct {
  int width;
  int height;
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

void multiMatrix(Matrix A, Matrix B, Matrix C)
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

long timediff(clock_t t1, clock_t t2) {
    long elapsed;
    elapsed = ((double)t2 - t1) / CLOCKS_PER_SEC * 1000;
    return elapsed;
}

int main(int argc, char* argv[])
{
  int a = 4;
  int b = 4;
  int c = 8;
  
  Matrix A;
  Matrix B;
  Matrix C;

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
  multiMatrix(A, B, C);
  printf(" Matriz C \n");
  disp_img(C);
}


