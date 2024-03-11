//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//

#pragma once
#ifndef CUDAIMAGE_H
#define CUDAIMAGE_H

class CudaImage
{
public:
    int width, height;
    int pitch;
    float *h_data;
    float *d_data;
    float *t_data;
    bool d_internalAlloc;
    bool h_internalAlloc;

public:
    CudaImage();
    ~CudaImage();
    void Allocate(int width, int height, int pitch, bool withHost, float *devMem = NULL, float *hostMem = NULL);
    double Download();
};

int iDivUp(int a, int b);
int iDivDown(int a, int b);
int iAlignUp(int a, int b);
int iAlignDown(int a, int b);

#endif // CUDAIMAGE_H
