#include <cuda.h>
#include <cuda_runtime_api.h>



////////////////////////////////////////////////////////////////////////////////
// GPU-specific defines
////////////////////////////////////////////////////////////////////////////////
//Maps to a single instruction on G8x / G9x / G10x
#define IMAD(a, b, c) ( __mul24((a), (b)) + (c) )

//Use unrolled innermost convolution loop
#define UNROLL_INNER 1

//Round a / b to nearest higher integer value
inline int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

//Align a to nearest higher multiple of b
inline int iAlignUp(int a, int b)
{
    return (a % b != 0) ? (a - a % b + b) : a;
}





texture<float, 2, cudaReadModeElementType> texSrc;
texture<float, 2, cudaReadModeElementType> texSrc2;





////////////////////////////////////////////////////////////////////////////////
// Slinding window
////////////////////////////////////////////////////////////////////////////////
__global__ void slidingWindowKernel(
    float *d_Dst,
    int matrixW,
    int matrixH,
    int KERNEL_LENGTH1,
    int stride
)
{
    const   int ix = stride -1 + IMAD(blockDim.x, blockIdx.x, threadIdx.x) * stride;
    const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
    const float  x = (float)ix + 0.5f;
    const float  y = (float)iy + 0.5f;


    if (ix >= matrixW - KERNEL_LENGTH1 || ix < KERNEL_LENGTH1 || iy >= matrixH)
    {
        return;
    }

    float sum = 0;



    for (int k = 0; k < KERNEL_LENGTH1; k++)
    {
        sum += (tex2D(texSrc, x - (float)k, y) - tex2D(texSrc2, (float)(KERNEL_LENGTH1 - k - 1), y))*(tex2D(texSrc, x - (float)k, y) - tex2D(texSrc2, (float)(KERNEL_LENGTH1 - k - 1), y));
    }



    d_Dst[IMAD(iy, matrixW, ix)] = 1.0/(1e-6+sum/float(KERNEL_LENGTH1));

}



extern "C" void slidingWindowGPU(
    float *d_Dst,
    cudaArray *a_Src,
    cudaArray *b_Src,
    int matrixW,
    int matrixH,
    int KERNEL_LENGTH1,
    int stride

)
{
    dim3 threads(16, 12);
    dim3 blocks(iDivUp(matrixW, threads.x), iDivUp(matrixH, threads.y));

    cudaBindTextureToArray(texSrc, a_Src);
    cudaBindTextureToArray(texSrc2, b_Src);



    slidingWindowKernel<<<blocks, threads>>>(
        d_Dst,
        matrixW,
        matrixH,
        KERNEL_LENGTH1,
        stride	
    );
    cudaUnbindTexture(texSrc);
    cudaUnbindTexture(texSrc2);
}



extern "C" {
void cuda_distance(float *a, float *h_OutputGPU, size_t W, size_t H, size_t KL1, size_t strd)
{
    float
    *h_Input,
    *h_Input2;

    cudaArray
    *a_Src;

    cudaArray
    *b_Src;

    cudaChannelFormatDesc floatTex = cudaCreateChannelDesc<float>();

    float
    *d_Output;

    const int matrixW = W;
    const int matrixH = H;
    const int KERNEL_LENGTH1 = KL1;
    const int stride = strd;	

    h_Input     = (float *)malloc(matrixW* matrixH * sizeof(float));
    h_Input2     = (float *)malloc(KERNEL_LENGTH1 * matrixH * sizeof(float));
    cudaMallocArray(&a_Src, &floatTex, matrixW, matrixH);
    cudaMallocArray(&b_Src, &floatTex, KERNEL_LENGTH1, matrixH);
    cudaMalloc((void **)&d_Output, matrixW * matrixH * sizeof(float));


    unsigned int w=0;
    for (unsigned int j = 0; j < matrixH; j++)
    {
	for (unsigned int i = 0; i < matrixW; i++) 
	{
	        h_Input[w] = a[i+j*matrixW]; 
		w=w+1; 
	}
    }
    w=0;
    for (unsigned int j = 0; j < matrixH; j++)
    {
	for (unsigned int i = matrixW-KERNEL_LENGTH1; i < matrixW; i++) 
	{
	        h_Input2[w] = a[i+j*matrixW]; 
		w=w+1; 
	}
    }

    cudaMemcpyToArray(a_Src, 0, 0, h_Input, matrixW * matrixH * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToArray(b_Src, 0, 0, h_Input2, KERNEL_LENGTH1 * matrixH * sizeof(float), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();


    slidingWindowGPU(
            d_Output,
            a_Src,
            b_Src,
            matrixW,
            matrixH,
            KERNEL_LENGTH1,
            stride
    );


    cudaDeviceSynchronize();

    cudaMemcpy(h_OutputGPU, d_Output, matrixW * matrixH * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_Output);
    cudaFreeArray(a_Src);
    cudaFreeArray(b_Src);
    		    

    free(h_Input);
    free(h_Input2);

}
}


