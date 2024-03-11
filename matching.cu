#include "cudaSift.h"
#include "cudautils.h"

//================= Device matching functions =====================//

// Keep
__global__ void CleanMatches(SiftPoint *sift1, int numPts1)
{
    const int p1 = min(blockIdx.x * 64 + threadIdx.x, numPts1 - 1);
    sift1[p1].score = 0.0f;
}

#define M7W 32
#define M7H 32
#define M7R 4
#define NRX 2
#define NDIM 128

// Keep
__global__ void FindMaxCorr10(SiftPoint *sift1, SiftPoint *sift2, int numPts1, int numPts2)
{
    __shared__ float4 buffer1[M7W * NDIM / 4];
    __shared__ float4 buffer2[M7H * NDIM / 4];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bp1 = M7W * blockIdx.x;
    for (int j = ty; j < M7W; j += M7H / M7R)
    {
        int p1 = min(bp1 + j, numPts1 - 1);
        for (int d = tx; d < NDIM / 4; d += M7W)
            buffer1[j * NDIM / 4 + (d + j) % (NDIM / 4)] = ((float4 *)&sift1[p1].data)[d];
    }

    float max_score[NRX];
    float sec_score[NRX];
    int index[NRX];
    for (int i = 0; i < NRX; i++)
    {
        max_score[i] = 0.0f;
        sec_score[i] = 0.0f;
        index[i] = -1;
    }
    int idx = ty * M7W + tx;
    int ix = idx % (M7W / NRX);
    int iy = idx / (M7W / NRX);
    for (int bp2 = 0; bp2 < numPts2 - M7H + 1; bp2 += M7H)
    {
        for (int j = ty; j < M7H; j += M7H / M7R)
        {
            int p2 = min(bp2 + j, numPts2 - 1);
            for (int d = tx; d < NDIM / 4; d += M7W)
                buffer2[j * NDIM / 4 + d] = ((float4 *)&sift2[p2].data)[d];
        }
        __syncthreads();

        if (idx < M7W * M7H / M7R / NRX)
        {
            float score[M7R][NRX];
            for (int dy = 0; dy < M7R; dy++)
                for (int i = 0; i < NRX; i++)
                    score[dy][i] = 0.0f;
            for (int d = 0; d < NDIM / 4; d++)
            {
                float4 v1[NRX];
                for (int i = 0; i < NRX; i++)
                    v1[i] = buffer1[((M7W / NRX) * i + ix) * NDIM / 4 + (d + (M7W / NRX) * i + ix) % (NDIM / 4)];
                for (int dy = 0; dy < M7R; dy++)
                {
                    float4 v2 = buffer2[(M7R * iy + dy) * (NDIM / 4) + d];
                    for (int i = 0; i < NRX; i++)
                    {
                        score[dy][i] += v1[i].x * v2.x;
                        score[dy][i] += v1[i].y * v2.y;
                        score[dy][i] += v1[i].z * v2.z;
                        score[dy][i] += v1[i].w * v2.w;
                    }
                }
            }
            for (int dy = 0; dy < M7R; dy++)
            {
                for (int i = 0; i < NRX; i++)
                {
                    if (score[dy][i] > max_score[i])
                    {
                        sec_score[i] = max_score[i];
                        max_score[i] = score[dy][i];
                        index[i] = min(bp2 + M7R * iy + dy, numPts2 - 1);
                    }
                    else if (score[dy][i] > sec_score[i])
                        sec_score[i] = score[dy][i];
                }
            }
        }
        __syncthreads();
    }

    float *scores1 = (float *)buffer1;
    float *scores2 = &scores1[M7W * M7H / M7R];
    int *indices = (int *)&scores2[M7W * M7H / M7R];
    if (idx < M7W * M7H / M7R / NRX)
    {
        for (int i = 0; i < NRX; i++)
        {
            scores1[iy * M7W + (M7W / NRX) * i + ix] = max_score[i];
            scores2[iy * M7W + (M7W / NRX) * i + ix] = sec_score[i];
            indices[iy * M7W + (M7W / NRX) * i + ix] = index[i];
        }
    }
    __syncthreads();

    if (ty == 0)
    {
        float max_score = scores1[tx];
        float sec_score = scores2[tx];
        int index = indices[tx];
        for (int y = 0; y < M7H / M7R; y++)
            if (index != indices[y * M7W + tx])
            {
                if (scores1[y * M7W + tx] > max_score)
                {
                    sec_score = max(max_score, sec_score);
                    max_score = scores1[y * M7W + tx];
                    index = indices[y * M7W + tx];
                }
                else if (scores1[y * M7W + tx] > sec_score)
                    sec_score = scores1[y * M7W + tx];
            }
        sift1[bp1 + tx].score = max_score;
        sift1[bp1 + tx].match = index;
        sift1[bp1 + tx].match_xpos = sift2[index].xpos;
        sift1[bp1 + tx].match_ypos = sift2[index].ypos;
        sift1[bp1 + tx].ambiguity = sec_score / (max_score + 1e-6f);
    }
}

// Keep
template <int size>
__device__ void InvertMatrix(float elem[size][size], float res[size][size])
{
    int indx[size];
    float b[size];
    float vv[size];
    for (int i = 0; i < size; i++)
        indx[i] = 0;
    int imax = 0;
    float d = 1.0;
    for (int i = 0; i < size; i++)
    { // find biggest element for each row
        float big = 0.0;
        for (int j = 0; j < size; j++)
        {
            float temp = fabs(elem[i][j]);
            if (temp > big)
                big = temp;
        }
        if (big > 0.0)
            vv[i] = 1.0 / big;
        else
            vv[i] = 1e16;
    }
    for (int j = 0; j < size; j++)
    {
        for (int i = 0; i < j; i++)
        {                                       // i<j
            float sum = elem[i][j];             // i<j (lower left)
            for (int k = 0; k < i; k++)         // k<i<j
                sum -= elem[i][k] * elem[k][j]; // i>k (upper right), k<j (lower left)
            elem[i][j] = sum;                   // i<j (lower left)
        }
        float big = 0.0;
        for (int i = j; i < size; i++)
        {                                       // i>=j
            float sum = elem[i][j];             // i>=j (upper right)
            for (int k = 0; k < j; k++)         // k<j<=i
                sum -= elem[i][k] * elem[k][j]; // i>k (upper right), k<j (lower left)
            elem[i][j] = sum;                   // i>=j (upper right)
            float dum = vv[i] * fabs(sum);
            if (dum >= big)
            {
                big = dum;
                imax = i;
            }
        }
        if (j != imax)
        { // imax>j
            for (int k = 0; k < size; k++)
            {
                float dum = elem[imax][k]; // upper right and lower left
                elem[imax][k] = elem[j][k];
                elem[j][k] = dum;
            }
            d = -d;
            vv[imax] = vv[j];
        }
        indx[j] = imax;
        if (elem[j][j] == 0.0) // j==j (upper right)
            elem[j][j] = 1e-16;
        if (j != (size - 1))
        {
            float dum = 1.0 / elem[j][j];
            for (int i = j + 1; i < size; i++) // i>j
                elem[i][j] *= dum;             // i>j (upper right)
        }
    }
    for (int j = 0; j < size; j++)
    {
        for (int k = 0; k < size; k++)
            b[k] = 0.0;
        b[j] = 1.0;
        int ii = -1;
        for (int i = 0; i < size; i++)
        {
            int ip = indx[i];
            float sum = b[ip];
            b[ip] = b[i];
            if (ii != -1)
                for (int j = ii; j < i; j++)
                    sum -= elem[i][j] * b[j]; // i>j (upper right)
            else if (sum != 0.0)
                ii = i;
            b[i] = sum;
        }
        for (int i = size - 1; i >= 0; i--)
        {
            float sum = b[i];
            for (int j = i + 1; j < size; j++)
                sum -= elem[i][j] * b[j]; // i<j (lower left)
            b[i] = sum / elem[i][i];      // i==i (upper right)
        }
        for (int i = 0; i < size; i++)
            res[i][j] = b[i];
    }
}

// Keep
__global__ void ComputeHomographies(float *coord, int *randPts, float *homo,
                                    int numPts)
{
    float a[8][8], ia[8][8];
    float b[8];
    const int bx = blockIdx.x;
    const int tx = threadIdx.x;
    const int idx = blockDim.x * bx + tx;
    const int numLoops = blockDim.x * gridDim.x;
    for (int i = 0; i < 4; i++)
    {
        int pt = randPts[i * numLoops + idx];
        float x1 = coord[pt + 0 * numPts];
        float y1 = coord[pt + 1 * numPts];
        float x2 = coord[pt + 2 * numPts];
        float y2 = coord[pt + 3 * numPts];
        float *row1 = a[2 * i + 0];
        row1[0] = x1;
        row1[1] = y1;
        row1[2] = 1.0;
        row1[3] = row1[4] = row1[5] = 0.0;
        row1[6] = -x2 * x1;
        row1[7] = -x2 * y1;
        float *row2 = a[2 * i + 1];
        row2[0] = row2[1] = row2[2] = 0.0;
        row2[3] = x1;
        row2[4] = y1;
        row2[5] = 1.0;
        row2[6] = -y2 * x1;
        row2[7] = -y2 * y1;
        b[2 * i + 0] = x2;
        b[2 * i + 1] = y2;
    }
    InvertMatrix<8>(a, ia);
    __syncthreads();
    for (int j = 0; j < 8; j++)
    {
        float sum = 0.0f;
        for (int i = 0; i < 8; i++)
            sum += ia[j][i] * b[i];
        homo[j * numLoops + idx] = sum;
    }
    __syncthreads();
}

#define TESTHOMO_TESTS 16 // number of tests per block,  alt. 32, 32
#define TESTHOMO_LOOPS 16 // number of loops per block,  alt.  8, 16

// Keep
__global__ void TestHomographies(float *d_coord, float *d_homo,
                                 int *d_counts, int numPts, float thresh2)
{
    __shared__ float homo[8 * TESTHOMO_LOOPS];
    __shared__ int cnts[TESTHOMO_TESTS * TESTHOMO_LOOPS];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int idx = blockIdx.y * blockDim.y + tx;
    const int numLoops = blockDim.y * gridDim.y;
    if (ty < 8 && tx < TESTHOMO_LOOPS)
        homo[tx * 8 + ty] = d_homo[idx + ty * numLoops];
    __syncthreads();
    float a[8];
    for (int i = 0; i < 8; i++)
        a[i] = homo[ty * 8 + i];
    int cnt = 0;
    for (int i = tx; i < numPts; i += TESTHOMO_TESTS)
    {
        float x1 = d_coord[i + 0 * numPts];
        float y1 = d_coord[i + 1 * numPts];
        float x2 = d_coord[i + 2 * numPts];
        float y2 = d_coord[i + 3 * numPts];
        float nomx = __fmul_rz(a[0], x1) + __fmul_rz(a[1], y1) + a[2];
        float nomy = __fmul_rz(a[3], x1) + __fmul_rz(a[4], y1) + a[5];
        float deno = __fmul_rz(a[6], x1) + __fmul_rz(a[7], y1) + 1.0f;
        float errx = __fmul_rz(x2, deno) - nomx;
        float erry = __fmul_rz(y2, deno) - nomy;
        float err2 = __fmul_rz(errx, errx) + __fmul_rz(erry, erry);
        if (err2 < __fmul_rz(thresh2, __fmul_rz(deno, deno)))
            cnt++;
    }
    int kty = TESTHOMO_TESTS * ty;
    cnts[kty + tx] = cnt;
    __syncthreads();
    int len = TESTHOMO_TESTS / 2;
    while (len > 0)
    {
        if (tx < len)
            cnts[kty + tx] += cnts[kty + tx + len];
        len /= 2;
        __syncthreads();
    }
    if (tx < TESTHOMO_LOOPS && ty == 0)
        d_counts[idx] = cnts[TESTHOMO_TESTS * tx];
    __syncthreads();
}

//================= Host matching functions =====================//

// Keep
double FindHomography(SiftData &data, float *homography, int *numMatches, int numLoops, float minScore, float maxAmbiguity, float thresh)
{
    *numMatches = 0;
    homography[0] = homography[4] = homography[8] = 1.0f;
    homography[1] = homography[2] = homography[3] = 0.0f;
    homography[5] = homography[6] = homography[7] = 0.0f;
#ifdef MANAGEDMEM
    SiftPoint *d_sift = data.m_data;
#else
    if (data.d_data == NULL)
        return 0.0f;
    SiftPoint *d_sift = data.d_data;
#endif
    TimerGPU timer(0);
    numLoops = iDivUp(numLoops, 16) * 16;
    int numPts = data.numPts;
    if (numPts < 8)
        return 0.0f;
    int numPtsUp = iDivUp(numPts, 16) * 16;
    float *d_coord, *d_homo;
    int *d_randPts, *h_randPts;
    int randSize = 4 * sizeof(int) * numLoops;
    int szFl = sizeof(float);
    int szPt = sizeof(SiftPoint);
    safeCall(cudaMalloc((void **)&d_coord, 4 * sizeof(float) * numPtsUp));
    safeCall(cudaMalloc((void **)&d_randPts, randSize));
    safeCall(cudaMalloc((void **)&d_homo, 8 * sizeof(float) * numLoops));
    h_randPts = (int *)malloc(randSize);
    float *h_scores = (float *)malloc(sizeof(float) * numPtsUp);
    float *h_ambiguities = (float *)malloc(sizeof(float) * numPtsUp);
    safeCall(cudaMemcpy2D(h_scores, szFl, &d_sift[0].score, szPt, szFl, numPts, cudaMemcpyDeviceToHost));
    safeCall(cudaMemcpy2D(h_ambiguities, szFl, &d_sift[0].ambiguity, szPt, szFl, numPts, cudaMemcpyDeviceToHost));
    int *validPts = (int *)malloc(sizeof(int) * numPts);
    int numValid = 0;
    for (int i = 0; i < numPts; i++)
    {
        if (h_scores[i] > minScore && h_ambiguities[i] < maxAmbiguity)
            validPts[numValid++] = i;
    }
    free(h_scores);
    free(h_ambiguities);
    if (numValid >= 8)
    {
        for (int i = 0; i < numLoops; i++)
        {
            int p1 = rand() % numValid;
            int p2 = rand() % numValid;
            int p3 = rand() % numValid;
            int p4 = rand() % numValid;
            while (p2 == p1)
                p2 = rand() % numValid;
            while (p3 == p1 || p3 == p2)
                p3 = rand() % numValid;
            while (p4 == p1 || p4 == p2 || p4 == p3)
                p4 = rand() % numValid;
            h_randPts[i + 0 * numLoops] = validPts[p1];
            h_randPts[i + 1 * numLoops] = validPts[p2];
            h_randPts[i + 2 * numLoops] = validPts[p3];
            h_randPts[i + 3 * numLoops] = validPts[p4];
        }
        safeCall(cudaMemcpy(d_randPts, h_randPts, randSize, cudaMemcpyHostToDevice));
        safeCall(cudaMemcpy2D(&d_coord[0 * numPtsUp], szFl, &d_sift[0].xpos, szPt, szFl, numPts, cudaMemcpyDeviceToDevice));
        safeCall(cudaMemcpy2D(&d_coord[1 * numPtsUp], szFl, &d_sift[0].ypos, szPt, szFl, numPts, cudaMemcpyDeviceToDevice));
        safeCall(cudaMemcpy2D(&d_coord[2 * numPtsUp], szFl, &d_sift[0].match_xpos, szPt, szFl, numPts, cudaMemcpyDeviceToDevice));
        safeCall(cudaMemcpy2D(&d_coord[3 * numPtsUp], szFl, &d_sift[0].match_ypos, szPt, szFl, numPts, cudaMemcpyDeviceToDevice));
        ComputeHomographies<<<numLoops / 16, 16>>>(d_coord, d_randPts, d_homo, numPtsUp);
        safeCall(cudaDeviceSynchronize());
        checkMsg("ComputeHomographies() execution failed\n");
        dim3 blocks(1, numLoops / TESTHOMO_LOOPS);
        dim3 threads(TESTHOMO_TESTS, TESTHOMO_LOOPS);
        TestHomographies<<<blocks, threads>>>(d_coord, d_homo, d_randPts, numPtsUp, thresh * thresh);
        safeCall(cudaDeviceSynchronize());
        checkMsg("TestHomographies() execution failed\n");
        safeCall(cudaMemcpy(h_randPts, d_randPts, sizeof(int) * numLoops, cudaMemcpyDeviceToHost));
        int maxIndex = -1, maxCount = -1;
        for (int i = 0; i < numLoops; i++)
            if (h_randPts[i] > maxCount)
            {
                maxCount = h_randPts[i];
                maxIndex = i;
            }
        *numMatches = maxCount;
        safeCall(cudaMemcpy2D(homography, szFl, &d_homo[maxIndex], sizeof(float) * numLoops, szFl, 8, cudaMemcpyDeviceToHost));
    }
    free(validPts);
    free(h_randPts);
    safeCall(cudaFree(d_homo));
    safeCall(cudaFree(d_randPts));
    safeCall(cudaFree(d_coord));
    double gpuTime = timer.read();
#ifdef VERBOSE
    printf("FindHomography time =         %.2f ms\n", gpuTime);
#endif
    return gpuTime;
}

// Keep
double MatchSiftData(SiftData &data1, SiftData &data2)
{
    TimerGPU timer(0);
    int numPts1 = data1.numPts;
    int numPts2 = data2.numPts;
    if (!numPts1 || !numPts2)
        return 0.0;
#ifdef MANAGEDMEM
    SiftPoint *sift1 = data1.m_data;
    SiftPoint *sift2 = data2.m_data;
#else
    if (data1.d_data == NULL || data2.d_data == NULL)
        return 0.0f;
    SiftPoint *sift1 = data1.d_data;
    SiftPoint *sift2 = data2.d_data;
#endif

// Combined version with no global memory requirement using global locks
    dim3 blocksMax3(iDivUp(numPts1, 16), iDivUp(numPts2, 512));
    dim3 threadsMax3(16, 16);
    CleanMatches<<<iDivUp(numPts1, 64), 64>>>(sift1, numPts1);
    blocksMax3 = dim3(iDivUp(numPts1, M7W));
    threadsMax3 = dim3(M7W, M7H / M7R);
    FindMaxCorr10<<<blocksMax3, threadsMax3>>>(sift1, sift2, numPts1, numPts2);
    safeCall(cudaDeviceSynchronize());
    checkMsg("FindMaxCorr5() execution failed\n");


    if (data1.h_data != NULL)
    {
        float *h_ptr = &data1.h_data[0].score;
        float *d_ptr = &data1.d_data[0].score;
        safeCall(cudaMemcpy2D(h_ptr, sizeof(SiftPoint), d_ptr, sizeof(SiftPoint), 5 * sizeof(float), data1.numPts, cudaMemcpyDeviceToHost));
    }

    return timer.read();
}
