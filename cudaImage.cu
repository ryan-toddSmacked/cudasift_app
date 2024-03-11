//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//

#include <cstdio>

#include "cudautils.h"
#include "cudaImage.h"

int iDivUp(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }
int iDivDown(int a, int b) { return a / b; }
int iAlignUp(int a, int b) { return (a % b != 0) ? (a - a % b + b) : a; }
int iAlignDown(int a, int b) { return a - a % b; }

void CudaImage::Allocate(int w, int h, int p, bool host, float *devmem, float *hostmem)
{
    width = w;
    height = h;
    pitch = p;
    d_data = devmem;
    h_data = hostmem;
    t_data = NULL;
    if (devmem == NULL)
    {
        safeCall(cudaMallocPitch((void **)&d_data, (size_t *)&pitch, (size_t)(sizeof(float) * width), (size_t)height));
        pitch /= sizeof(float);
        d_internalAlloc = true;
    }
    if (host && hostmem == NULL)
    {
        h_data = (float *)malloc(sizeof(float) * pitch * height);
        h_internalAlloc = true;
    }
}

CudaImage::CudaImage() : width(0), height(0), d_data(NULL), h_data(NULL), t_data(NULL), d_internalAlloc(false), h_internalAlloc(false)
{
}

CudaImage::~CudaImage()
{
    if (d_internalAlloc && d_data != NULL)
        safeCall(cudaFree(d_data));
    d_data = NULL;
    if (h_internalAlloc && h_data != NULL)
        free(h_data);
    h_data = NULL;
    if (t_data != NULL)
        safeCall(cudaFreeArray((cudaArray *)t_data));
    t_data = NULL;
}

double CudaImage::Download()
{
    TimerGPU timer(0);
    int p = sizeof(float) * pitch;
    if (d_data != NULL && h_data != NULL)
        safeCall(cudaMemcpy2D(d_data, p, h_data, sizeof(float) * width, sizeof(float) * width, height, cudaMemcpyHostToDevice));
    return timer.read();
}


