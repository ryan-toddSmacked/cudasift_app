//********************************************************//
// CUDA SIFT extractor by Marten Björkman aka Celebrandil //
//              celle @ csc.kth.se                       //
//********************************************************//  

#include <iostream>
#include <cstdio>
#include <cmath>
#include <iomanip>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cuda_runtime.h>

#include "cudaImage.h"
#include "cudaSift.h"
#include "Logger.hpp"
#include "geomFuncs.hpp"

enum CUDASIFT_ERROR
{
    NO_ERROR=0,
    CUDA_ALLOC_ERROR,
    UNKNOWN_ERROR
};

constexpr double inverse_sqrt2 = 0.7071067811865475;

const char* usage();
const char* authors();


inline void parseargs(int argc, char **argv, std::string &image1, std::string &image2, std::string &output_file, int &num_features, float &feature_threshold);
inline void initcuda();
CUDASIFT_ERROR cudasift(const cv::Mat &cpu_img1, const cv::Mat &cpu_img2, float *homography, int num_features, float feature_threshold);
inline void readimages(const char *filename1, const char *filename2, cv::Mat &cpu_img1, cv::Mat &cpu_img2);
inline void formatimages(cv::Mat &cpu_img1, cv::Mat &cpu_img2);
inline void scaleimages(cv::Mat &cpu_img1, cv::Mat &cpu_img2, double scale_factor);
inline void writehomography(const char *filename, const char *image1, const char *image2, const float *homography, const cv::Size &size1, const cv::Size &size2, const cv::Size &final_size1, const cv::Size &final_size2, double elapsed_time);
inline void displayhomography(const float *homography);

int main(int argc, char **argv)
{
    std::chrono::high_resolution_clock::time_point t1, t2;
    std::chrono::duration<double> time_span;
    std::string image1, image2, output_file;
    int num_features = 4096;
    float feature_threshold = 3.0f;
    cv::Mat cpu_img1, cpu_img2;
    float homography[9];
    CUDASIFT_ERROR error;
    int max_trials = 5;
    int trial = 0;
    bool success = false;
    cv::Size original_size1, original_size2;
    cv::Size final_size1, final_size2;
    double upscale_factor = 1.0;        // This is the factor by which the homography needs to be scaled up, when downscaling the images. the shift is also scaled by this factor.

    // Need at least 3 arguments
    if (argc < 4)
    {
        std::cout << usage() << std::endl;
        return 1;
    }

    // Print authors
    printf("%s\n", authors());

    t1 = std::chrono::high_resolution_clock::now();
    // Parse the arguments
    parseargs(argc, argv, image1, image2, output_file, num_features, feature_threshold);

    // Read the images
    readimages(image1.c_str(), image2.c_str(), cpu_img1, cpu_img2);

    original_size1 = cpu_img1.size();
    original_size2 = cpu_img2.size();

    // Format the images
    formatimages(cpu_img1, cpu_img2);

    // Initialize CUDA
    initcuda();

    do
    {
        trial++;

        // Call the cudasift function
        error = cudasift(cpu_img1, cpu_img2, homography, num_features, feature_threshold);

        // Check for memory allocation errors
        switch (error)
        {

            case NO_ERROR:
                // Just break
                success = true;
                break;

            case CUDA_ALLOC_ERROR:
                if (trial > max_trials)
                {
                    Logger::error("Memory allocation error for CUDA images. Tried scaling the images down, but still not enough memory available.\n");
                    cudaDeviceReset();
                    return error;
                }
                Logger::warning("Memory allocation error. Trying to scale the images down to use less memory...\n");
                scaleimages(cpu_img1, cpu_img2, inverse_sqrt2);
                break;

            case UNKNOWN_ERROR:
                Logger::error("Unknown error. nothing to be done...\n");
                cudaDeviceReset();
                return error;

            default:
                Logger::warning("Unknown error. nothing to be done...\n");
                cudaDeviceReset();
                return error;
        }

        final_size1 = cpu_img1.size();
        final_size2 = cpu_img2.size();
        if (!success)
        {
            // scale the value by which the homography needs to be scaled up.
            upscale_factor /= inverse_sqrt2;
        }
    } while (!success);

    t2 = std::chrono::high_resolution_clock::now();
    time_span = (t2 - t1);
    // Reset the device
    cudaDeviceReset();

    if (upscale_factor != 1.0)
    {
        // Only scale the homography if the images were downscaled.
        // The values to scale should only be the shift values.
        // Rotations and scaling should not be scaled. As they would not be effected by the scaling of the images.
        homography[2] *= upscale_factor;
        homography[5] *= upscale_factor;
    }

    // Write the homography to the output file    
    writehomography(output_file.c_str(), image1.c_str(), image2.c_str(), homography, original_size1, original_size2, final_size1, final_size2, time_span.count());

    // Display homography to stdout
    displayhomography(homography);

    t2 = std::chrono::high_resolution_clock::now();
    time_span = (t2 - t1);

    Logger::log("Total time taken: %.6lf s\n\n", time_span.count());
}


const char* usage()
{
    return "Usage: cudasift <image1> <image2> <output file> [OPTIONAL, in this order] <number of features> <feature threshold>\n"
    " [REQUIRED]\n"
    "  <image1>             - First image to match\n"
    "  <image2>             - Second image to match\n"
    "  <output file>        - File to write the output homography to\n"
    " [OPTIONAL]\n"
    "  <number of features> - Number of features to extract from each image\n"
    "  <feature threshold>  - Feature threshold.\n\n"
    " Default values for optional parameters are 4096 and 3.0 respectively.\n"
    " The resulting homography is used to warp the first image to the second.\n"
    " The homography is written to the output file in JSON format.\n"
    " It is a ROW-MAJOR 3x3 floating point matrix.\n"
    " The feature threshold argument is a value used to filter out weak features.\n"
    " As the value increases, the number of features decreases.\n";
}

const char* authors()
{
    return "\t<--- CudaSift - SIFT features with CUDA --->\n"
    "Authors: M. Björkman, N. Bergström and D. Kragic,\n"
    "\"Detecting, segmenting and tracking unknown objects using multi-label MRF inference\", CVIU, 118, pp. 111-127, January 2014.\n"
    "[ScienceDirect](http://www.sciencedirect.com/science/article/pii/S107731421300194X)\n";
}

inline void parseargs(int argc, char **argv, std::string &image1, std::string &image2, std::string &output_file, int &num_features, float &feature_threshold)
{
    image1 = argv[1];
    image2 = argv[2];
    output_file = argv[3];

    if (argc > 4)
    {
        try
        {
            num_features = std::stoul(argv[4]);
        }
        catch(...)
        {
            // This is okay, the default value will be used.
            Logger::warning("Invalid value for number of features. Using default value of 4096\n");
        }
    }

    if (argc > 5)
    {
        try
        {
            feature_threshold = std::stof(argv[5]);
        }
        catch(...)
        {
            // This is okay, the default value will be used.
            Logger::warning("Invalid value for feature threshold. Using default value of 3.0\n");
        }
    }
}

inline void initcuda()
{
    std::chrono::high_resolution_clock::time_point t1, t2;
    std::chrono::duration<double> time_span;

    // Initialize CUDA
    Logger::log("Initializing CUDA\n");
    try
    {
        t1 = std::chrono::high_resolution_clock::now();
        InitCuda();
        t2 = std::chrono::high_resolution_clock::now();
    }
    catch(const std::exception& e)
    {
        // Log the error
        Logger::error("Error initializing CUDA: %s\n", e.what());
        // Exit the program
        exit(1);
    }
    time_span = (t2 - t1);
    Logger::log("CUDA initialized (%.6lf s)\n", time_span.count());
}

CUDASIFT_ERROR cudasift(const cv::Mat &cpu_img1, const cv::Mat &cpu_img2, float *homography, int num_features, float feature_threshold)
{

    std::chrono::high_resolution_clock::time_point t1, t2;
    std::chrono::duration<double> time_span;
    double elapsed_time;

    const unsigned int w1 = cpu_img1.cols;
    const unsigned int h1 = cpu_img1.rows;
    const unsigned int w2 = cpu_img2.cols;
    const unsigned int h2 = cpu_img2.rows;

    const unsigned int max_width = (w1 > w2) ? w1 : w2;
    const unsigned int max_height = (h1 > h2) ? h1 : h2;

    int find_homography_num_loops;
    int num_matches;
    int num_inliers;

    SiftData siftData1, siftData2;  // Sift data for the two images, these do not have destructor, so they need to be freed manually.
    bool free_sift_data1 = false, free_sift_data2 = false;  // Flags to check if the sift data needs to be freed, before returning from the function.
    CudaImage img1, img2;
    SiftTempMem tempMem;


    // Allocate memory for the images on the GPU
    Logger::log("Allocating memory for CUDA images\n");
    try
    {
        t1 = std::chrono::high_resolution_clock::now();
        img1.Allocate(w1, h1, iAlignUp(w1, 128), false, NULL, (float*)cpu_img1.data);
        img2.Allocate(w2, h2, iAlignUp(w2, 128), false, NULL, (float*)cpu_img2.data);
        t2 = std::chrono::high_resolution_clock::now();
    }
    catch(const std::exception& e)
    {
        Logger::error("Could not allocate memory for CUDA images: %s\n", e.what());
        if (free_sift_data1) FreeSiftData(siftData1);   // I think the compiler will optimize this out...
        if (free_sift_data2) FreeSiftData(siftData2);
        if (e.what() == std::string("out of memory"))
        {
            return CUDASIFT_ERROR::CUDA_ALLOC_ERROR;
        }
        return CUDASIFT_ERROR::UNKNOWN_ERROR;
    }
    time_span = (t2 - t1);
    Logger::log("Memory allocated for CUDA images (%.6lf s)\n", time_span.count());


    // Download the images to the device
    Logger::log("Downloading CUDA images to device\n");
    try
    {
        elapsed_time = img1.Download();
        elapsed_time += img2.Download();
    }
    catch(const std::exception& e)
    {
        Logger::error("Could not download CUDA images to device: %s\n", e.what());
        if (free_sift_data1) FreeSiftData(siftData1);   // I think the compiler will optimize this out...
        if (free_sift_data2) FreeSiftData(siftData2);
        if (e.what() == std::string("out of memory"))
        {
            return CUDASIFT_ERROR::CUDA_ALLOC_ERROR;
        }
        return CUDASIFT_ERROR::UNKNOWN_ERROR;
    }
    Logger::log("CUDA images downloaded to device (%.6lf s)\n", elapsed_time / 1000.0);


    // Initialize SiftData for the two images
    Logger::log("Initializing SiftData for the two images\n");
    try
    {
        t1 = std::chrono::high_resolution_clock::now();
        InitSiftData(siftData1, num_features, true, true);
        free_sift_data1 = true; // Set the flag to true, so that the data is freed before returning from the function.
        InitSiftData(siftData2, num_features, true, true);
        free_sift_data2 = true; // Set the flag to true, so that the data is freed before returning from the function.
        t2 = std::chrono::high_resolution_clock::now();
    }
    catch(const std::exception& e)
    {
        Logger::error("Could not initialize SiftData: %s\n", e.what());
        if (free_sift_data1) FreeSiftData(siftData1);   // I think the compiler will optimize this out...
        if (free_sift_data2) FreeSiftData(siftData2);
        if (e.what() == std::string("out of memory"))
        {
            return CUDASIFT_ERROR::CUDA_ALLOC_ERROR;
        }
        return CUDASIFT_ERROR::UNKNOWN_ERROR;
    }
    time_span = (t2 - t1);
    Logger::log("SiftData initialized (%.6lf s)\n", time_span.count());


    // Allocate temporary memory for Sift extraction
    Logger::log("Allocating temporary memory for Sift extraction\n");
    try
    {
        t1 = std::chrono::high_resolution_clock::now();
        tempMem.Allocate(max_width, max_height, 6);
        t2 = std::chrono::high_resolution_clock::now();
    }
    catch(const std::exception& e)
    {
        Logger::error("Could not allocate temporary memory for sift extraction: %s\n", e.what());
        if (free_sift_data1) FreeSiftData(siftData1);
        if (free_sift_data2) FreeSiftData(siftData2);
        if (e.what() == std::string("out of memory"))
        {
            return CUDASIFT_ERROR::CUDA_ALLOC_ERROR;
        }
        return CUDASIFT_ERROR::UNKNOWN_ERROR;
    }
    time_span = (t2 - t1);
    Logger::log("Temporary memory allocated for Sift extraction (%.6lf s)\n", time_span.count());


    // Extract Sift features from the two images
    Logger::log("Extracting Sift features from the two images\n");
    try
    {
        elapsed_time = ExtractSift(siftData1, img1, 6, 1.0, feature_threshold, 0.0f, tempMem.get_device_pointer());
    }
    catch(const std::exception& e)
    {
        Logger::error("Could not extract SiftData for image 1: %s\n", e.what());
        if (free_sift_data1) FreeSiftData(siftData1);
        if (free_sift_data2) FreeSiftData(siftData2);
        if (e.what() == std::string("out of memory"))
        {
            return CUDASIFT_ERROR::CUDA_ALLOC_ERROR;
        }
        return CUDASIFT_ERROR::UNKNOWN_ERROR;
    }
    Logger::log("Sift features extracted from image 1 (%.6lf s)\n", elapsed_time / 1000.0);


    // Extract Sift features from the second image
    try
    {
        elapsed_time = ExtractSift(siftData2, img2, 6, 1.0, feature_threshold, 0.0f, tempMem.get_device_pointer());
    }
    catch(const std::exception& e)
    {
        Logger::error("Could not extract SiftData for image 2: %s\n", e.what());
        if (free_sift_data1) FreeSiftData(siftData1);
        if (free_sift_data2) FreeSiftData(siftData2);
        if (e.what() == std::string("out of memory"))
        {
            return CUDASIFT_ERROR::CUDA_ALLOC_ERROR;
        }
        return CUDASIFT_ERROR::UNKNOWN_ERROR;
    }
    Logger::log("Sift features extracted from image 2 (%.6lf s)\n", elapsed_time / 1000.0);
    Logger::log("Image1 SiftFeatures: %d\n", siftData1.numPts);
    Logger::log("Image2 SiftFeatures: %d\n", siftData2.numPts);


    // Match the Sift features from the two images
    Logger::log("Matching Sift features from the two images\n");
    try
    {
        elapsed_time = MatchSiftData(siftData1, siftData2);
    }
    catch(const std::exception& e)
    {
        Logger::error("Could not match SiftData beteen images: %s\n", e.what());
        if (free_sift_data1) FreeSiftData(siftData1);
        if (free_sift_data2) FreeSiftData(siftData2);
        if (e.what() == std::string("out of memory"))
        {
            return CUDASIFT_ERROR::CUDA_ALLOC_ERROR;
        }
        return CUDASIFT_ERROR::UNKNOWN_ERROR;
    }
    Logger::log("Sift features matched (%.6lf s)\n", elapsed_time / 1000.0);


    // Set the number of loops for finding the homography, 10 * the maximum number of points.
    find_homography_num_loops = 10 * ((siftData1.numPts > siftData2.numPts) ? siftData1.numPts : siftData2.numPts);

    // Find the homography between the two images
    Logger::log("Finding the homography between the two images\n");
    try
    {
        elapsed_time = FindHomography(siftData1, homography, &num_matches, find_homography_num_loops, 0.85f, 0.95f, 5.0f);
    }
    catch(const std::exception& e)
    {
        Logger::error("could not find homography: %s\n", e.what());
        if (free_sift_data1) FreeSiftData(siftData1);
        if (free_sift_data2) FreeSiftData(siftData2);
        if (e.what() == std::string("out of memory"))
        {
            return CUDASIFT_ERROR::CUDA_ALLOC_ERROR;
        }
        return CUDASIFT_ERROR::UNKNOWN_ERROR;
    }
    Logger::log("Homography found (%.6lf s)\n", elapsed_time / 1000.0);
    Logger::log("Number of matches: %d\n", num_matches);

    // Improve the homography
    Logger::log("Improving the homography\n");
    try
    {
        t1 = std::chrono::high_resolution_clock::now();
        num_inliers = ImproveHomography(siftData1, homography, 5, 0.85f, 0.95f, 3.5f);
        t2 = std::chrono::high_resolution_clock::now();
    }
    catch(const std::exception& e)
    {
        Logger::error("Could not improve homography: %s\n", e.what());
        if (free_sift_data1) FreeSiftData(siftData1);
        if (free_sift_data2) FreeSiftData(siftData2);
        if (e.what() == std::string("out of memory"))
        {
            return CUDASIFT_ERROR::CUDA_ALLOC_ERROR;
        }
        return CUDASIFT_ERROR::UNKNOWN_ERROR;
    }
    time_span = (t2 - t1);
    Logger::log("Homography improved (%.6lf s)\n", time_span.count());
    Logger::log("Number of inliers: %d\n", num_inliers);
    Logger::log("Inlier ratio: %.4lf%%\n", 100.0 * (num_inliers / (double)siftData1.numPts));

    // Free the SiftData
    if (free_sift_data1) FreeSiftData(siftData1);
    if (free_sift_data2) FreeSiftData(siftData2);

    return CUDASIFT_ERROR::NO_ERROR;
}


inline void readimages(const char *filename1, const char *filename2, cv::Mat &cpu_img1, cv::Mat &cpu_img2)
{
    std::chrono::high_resolution_clock::time_point t1, t2;
    std::chrono::duration<double> time_span;

    Logger::log("Reading image %s\n", filename1);
    // This function can exit the program if the images are not found.
    t1 = std::chrono::high_resolution_clock::now();
    cpu_img1 = cv::imread(filename1, cv::IMREAD_GRAYSCALE);
    t2 = std::chrono::high_resolution_clock::now();
    time_span = (t2 - t1);

    if (cpu_img1.empty())
    {
        Logger::error("Error reading image %s\n", filename1);
        exit(1);
    }

    Logger::log("Image %s read (%.6lf s)\n", filename1, time_span.count());

    Logger::log("Reading image %s\n", filename2);
    t1 = std::chrono::high_resolution_clock::now();
    cpu_img2 = cv::imread(filename2, cv::IMREAD_GRAYSCALE);
    t2 = std::chrono::high_resolution_clock::now();
    time_span = (t2 - t1);

    if (cpu_img2.empty())
    {
        Logger::error("Error reading image %s\n", filename2);
        exit(1);
    }

    Logger::log("Image %s read (%.6lf s)\n", filename2, time_span.count());
}

inline void formatimages(cv::Mat &cpu_img1, cv::Mat &cpu_img2)
{
    std::chrono::high_resolution_clock::time_point t1, t2;
    std::chrono::duration<double> time_span;

    // Convert images to single channel floating point images
    Logger::log("Converting images to single channel floating point images\n");
    t1 = std::chrono::high_resolution_clock::now();
    cpu_img1.convertTo(cpu_img1, CV_32FC1);
    cpu_img2.convertTo(cpu_img2, CV_32FC1);
    t2 = std::chrono::high_resolution_clock::now();
    time_span = (t2 - t1);
    Logger::log("Images converted (%.6lf s)\n", time_span.count());
}

inline void scaleimages(cv::Mat &cpu_img1, cv::Mat &cpu_img2, double scale_factor)
{
    std::chrono::high_resolution_clock::time_point t1, t2;
    std::chrono::duration<double> time_span;

    // Scale images in each dimension
    // This should only be called if the imagestake up too much memory on GPU.
    // Use the cubic interpolation method.
    Logger::log("Scaling images by %.4lf\n", scale_factor);
    t1 = std::chrono::high_resolution_clock::now();
    cv::resize(cpu_img1, cpu_img1, cv::Size(), scale_factor, scale_factor, cv::INTER_CUBIC);
    cv::resize(cpu_img2, cpu_img2, cv::Size(), scale_factor, scale_factor, cv::INTER_CUBIC);
    t2 = std::chrono::high_resolution_clock::now();
    time_span = (t2 - t1);
    Logger::log("Images scaled (%.6lf s)\n", time_span.count());
}


inline void writehomography(const char *filename, const char *image1, const char *image2, const float *homography, const cv::Size &size1, const cv::Size &size2, const cv::Size &final_size1, const cv::Size &final_size2, double elapsed_time)
{
    std::chrono::high_resolution_clock::time_point t1, t2;
    std::chrono::duration<double> time_span;
    FILE *file;

    Logger::log("Writing homography to file %s\n", filename);
    t1 = std::chrono::high_resolution_clock::now();
    file = fopen(filename, "w");
    if (file == NULL)
    {
        Logger::error("Error opening file %s for writing\n", filename);
        exit(1);
    }
    fprintf(file, "{\n");
    fprintf(file, "\t\"image1\": \"%s\",\n", image1);
    fprintf(file, "\t\"image2\": \"%s\",\n", image2);
    fprintf(file, "\t\"homography\": [\n");
    // Write as a row-major 3x3 matrix
    fprintf(file, "\t\t[%.9e, %.9e, %.9e],\n", homography[0], homography[1], homography[2]);
    fprintf(file, "\t\t[%.9e, %.9e, %.9e],\n", homography[3], homography[4], homography[5]);
    fprintf(file, "\t\t[%.9e, %.9e, %.9e]\n", homography[6], homography[7], homography[8]);
    fprintf(file, "\t],\n");
    fprintf(file, "\t\"original_size1\": [%d, %d],\n", size1.width, size1.height);
    fprintf(file, "\t\"original_size2\": [%d, %d],\n", size2.width, size2.height);
    fprintf(file, "\t\"final_size1\": [%d, %d],\n", final_size1.width, final_size1.height);
    fprintf(file, "\t\"final_size2\": [%d, %d],\n", final_size2.width, final_size2.height);
    fprintf(file, "\t\"elapsed_time\": %.6lf\n", elapsed_time);
    fprintf(file, "}\n");
    fclose(file);
    t2 = std::chrono::high_resolution_clock::now();
    time_span = (t2 - t1);
    Logger::log("Homography written to file (%.6lf s)\n", time_span.count());
}

inline void displayhomography(const float *homography)
{
    float val;
    printf("Homography:\n");
    for (int i = 0; i < 3; i++)
    {
        printf("\t|");
        for (int j = 0; j < 3; j++)
        {
            val = homography[i * 3 + j];
            if (val == 0.0f) val = 0.0f;
            if (val < 0.0f) printf(" %.9e", val);
            else printf("  %.9e", val);    
        }
        printf(" |\n");
    }
}

