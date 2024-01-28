 
#include <boost/program_options.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

// Check if we have cudawarping from opencv-contrib
#ifdef HAVE_OPENCV_CUDAWARPING
#include <opencv2/cudawarping.hpp>
#endif

#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <chrono>
#include <stack>

#include "cudaImage.h"
#include "cudaSift.h"
#include "geomFuncs.h"
#include "tmpMem.h"

namespace po = boost::program_options;

/// @brief Give credit to the authors
std::string printAuthors();

/// @brief Verifies the command line arguments 
void verifyArgs(const po::variables_map &vm, const po::options_description &desc);

/// @brief Read images from disk
void readImages(const std::vector<std::string> &files, cv::Mat &img1, cv::Mat &img2, bool _clahe=false);

/// @brief Load images to the device
void loadImages(const cv::Mat &img1, const cv::Mat &img2, CudaImage &img1Cuda, CudaImage &img2Cuda);

/// @brief Warp image 2 onto image 1 dimensions
void warpImage(const cv::Mat &img1, const cv::Mat &img2, cv::Mat &img2Warped, const float *homography);

/// @brief imfuse function from Matlab
void imfuse(const cv::Mat &img1, const cv::Mat &img2, cv::Mat &imgFused, const cv::Mat &img2Mask=cv::Mat());

/// @brief generate grayscale difference image
void graydiff(const cv::Mat &img1, const cv::Mat &img2, cv::Mat &imgDiff, const cv::Mat &img2Mask=cv::Mat());

/// @brief Generate Results
void generateResults(
    cv::Mat &img1, cv::Mat &img2,
    const float* homography, double inlier_ratio,
    const std::string &outputDir, const std::string &outputPrefix,
    const std::string &img1Name, const std::string &img2Name,
    uint32_t numMatches, uint32_t numFit, uint32_t numFeatures,
    uint32_t numOctaves, double initBlur, double thresh, double lowestScale,
    uint32_t numLoops_find, double minScore_find, double maxAmbiguity_find, double threshFit_find,
    uint32_t numLoops_improve, double minScore_improve, double maxAmbiguity_improve, double threshFit_improve);


/// @brief Get now time
inline std::string Now();

/// @brief Tic
inline void tic();

/// @brief Toc
inline double toc();

/// @brief Log
inline void Log(const std::string &msg)
{
    std::cout << "[" << Now() << "]: " << msg << std::endl;
}

/// @brief pretty print homograpy
inline void printHomography(const float *homography)
{
    for (int i = 0; i < 9; i++)
    {
        const float val = homography[i];
        (val < 0.0f) ? (printf(" %.7e", val)) : (printf("  %.7e", val));
        if ((i + 1) % 3 == 0) printf("\n");
    }
}

static std::stack<std::chrono::high_resolution_clock::time_point> tictoc_stack;


int main(int argc, char **argv) 
{
    // IO parameters
    std::vector<std::string> files;     // Input image files
    std::string outputDir;              // Output directory
    std::string outputPrefix;           // Output file prefix

    // Sift parameters
    uint32_t numFeatures = 2000;        // Number of features to extract (default 2000)
    uint32_t numOctaves = 5;            // Number of octaves to extract (default 5)
    double initBlur = 1.0;              // Initial blur level (default 1.0)
    double thresh = 3.0;                // Threshold on difference of Gaussians (default 3.0)
    double lowestScale = 0.0;           // Lowest scale to detect (default 0.0)

    // Find homography parameters
    uint32_t numLoops_find = 2000;       // Number of RANSAC iterations (default 2000)
    double minScore_find = 0.8;          // Minimum required score between feature matches (default 0.8)
    std::string mScoreStr_find;
    double maxAmbiguity_find = 0.95;     // Maximum ambiguity between feature matches (default 0.95)
    std::string mAmbStr_find;
    double threshFit_find = 5.0;         // Threshold on fit error (default 5.0)

    // Improve homography parameters
    uint32_t numLoops_improve = 5;       // Number of RANSAC iterations (default 5)
    double minScore_improve = 0.8;       // Minimum required score between feature matches (default 0.8)
    std::string mScoreStr_improve;
    double maxAmbiguity_improve = 0.95;  // Maximum ambiguity between feature matches (default 0.95)
    std::string mAmbStr_improve;
    double threshFit_improve = 3.0;      // Threshold on fit error (default 3.0)

    cv::Mat img1, img2;
    cv::Mat img2Warped; // using the homography matrix to warp image 2 onto image 1 dimensions
    cv::Mat imgFused;   // fused image
    bool clahe = false;
    SiftData siftData1, siftData2;
    po::variables_map vm;
    float homography[9];
    int numMatches;
    int numFit;
    uint32_t seed = 0;
    int device = 0;
    double confidence;

    // Declare the General options
    po::options_description desc("Options");
    desc.add_options()
        ("help,h", "produce help message")
        ("device,d", po::value<int>(&device)->default_value(0), "cuda device")
        ("input,i", po::value<std::vector<std::string>>(&files)->multitoken(), "input files")
        ("output,o", po::value<std::string>(&outputDir)->default_value("./"), "output directory")
        ("prefix", po::value<std::string>(&outputPrefix)->default_value("sift"), "output file prefix")
        ("seed", po::value<uint32_t>(&seed)->default_value(0), "random seed")
        ("histeq", po::bool_switch(&clahe), "apply histogram equalization");

    // Declare the Sift options
    po::options_description descSift("Sift Options");
    descSift.add_options()
        ("nfeatures", po::value<uint32_t>(&numFeatures)->default_value(2000), "number of features to extract")
        ("noctaves", po::value<uint32_t>(&numOctaves)->default_value(5), "number of octaves to extract")
        ("init-blur", po::value<double>(&initBlur)->default_value(1.0), "initial blur level")
        ("thresh", po::value<double>(&thresh)->default_value(3.0), "threshold on difference of Gaussians")
        ("lowest-scale", po::value<double>(&lowestScale)->default_value(0.0), "lowest scale to detect");

    // Declare the Find Homography options
    po::options_description descFind("Find Homography Options");
    descFind.add_options()
        ("nloops-find", po::value<uint32_t>(&numLoops_find)->default_value(2000), "number of RANSAC iterations")
        ("min-score-find", po::value<std::string>(&mScoreStr_find)->default_value("0.8"), "minimum required score between feature matches")
        ("max-ambiguity-find", po::value<std::string>(&mAmbStr_find)->default_value("0.95"), "maximum ambiguity between feature matches")
        ("thresh-find", po::value<double>(&threshFit_find)->default_value(5.0), "threshold on fit error");

    // Declare the Improve Homography options
    po::options_description descImprove("Improve Homography Options");
    descImprove.add_options()
        ("nloops-improve", po::value<uint32_t>(&numLoops_improve)->default_value(5), "number of RANSAC iterations")
        ("min-score-improve", po::value<std::string>(&mScoreStr_improve)->default_value("0.8"), "minimum required score between feature matches")
        ("max-ambiguity-improve", po::value<std::string>(&mAmbStr_improve)->default_value("0.95"), "maximum ambiguity between feature matches")
        ("thresh-improve", po::value<double>(&threshFit_improve)->default_value(3.0), "threshold on fit error");

    // Add all options to the description
    desc.add(descSift).add(descFind).add(descImprove);

    // Parse the command line options
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    // Print the authors
    printf("%s", printAuthors().c_str());

    // Verify the command line options
    verifyArgs(vm, desc);

    // Convert the string arguments to double
    minScore_find = std::stod(mScoreStr_find);
    maxAmbiguity_find = std::stod(mAmbStr_find);
    minScore_improve = std::stod(mScoreStr_improve);
    maxAmbiguity_improve = std::stod(mAmbStr_improve);

    numOctaves = numOctaves > 5 ? 5 : numOctaves;

    // Initialize SiftGPU
    InitCuda(device);

    Log("Starting SIFT\n");

    // Randomize the seed
    srand(seed);
    {
        CudaImage img1Cuda, img2Cuda;

        tic();
        // Read the images
        readImages(files, img1, img2, clahe);
        printf(" Read images (%lf s)\n", toc());
        printf("  Image1 size: %d x %d\n", img1.cols, img1.rows);
        printf("  Image2 size: %d x %d\n", img2.cols, img2.rows);

        tic();
        // Load the images to the device
        loadImages(img1, img2, img1Cuda, img2Cuda);
        printf(" Uploaded images (%lf s)\n", toc());

        tic();
        // Initialize SiftData
        InitSiftData(siftData1, numFeatures, true, true);
        InitSiftData(siftData2, numFeatures, true, true);
        printf(" Initialized SiftData (%lf s)\n", toc());

        // Temporary memory for laplace buffer
        {
            TmpMem memoryTmp;
            int w1 = img1.cols; int h1 = img1.rows; int w2 = img2.cols; int h2 = img2.rows;

            memoryTmp.alloc((w1>w2)?w1:w2, (h1>h2)?h1:h2, 5);

            // Extract SiftData
            tic();
            ExtractSift(siftData1, img1Cuda, numOctaves, initBlur, thresh, lowestScale, false, memoryTmp.data());
            ExtractSift(siftData2, img2Cuda, numOctaves, initBlur, thresh, lowestScale, false, memoryTmp.data());
            printf(" Extracted SiftData (%lf s)\n", toc());
            printf("  Image1 features: %d\n", siftData1.numPts);
            printf("  Image2 features: %d\n", siftData2.numPts);
        }
    }

    tic();
    // Match SiftData
    MatchSiftData(siftData1, siftData2);
    printf(" Matched SiftData (%lf s)\n", toc());

    // Free SiftData2, it is no longer needed
    FreeSiftData(siftData2);

    tic();
    // Find homography
    FindHomography(siftData1, homography, &numMatches, numLoops_find, minScore_find, maxAmbiguity_find, threshFit_find);
    printf(" Found homography (%lf s)\n", toc());
    printHomography(homography);
    printf(" Number of matches: %d\n", numMatches);


    tic();
    // Improve homography
    numFit = ImproveHomography(siftData1, homography, numLoops_improve, minScore_improve, maxAmbiguity_improve, threshFit_improve);
    printf(" Improved homography (%lf s)\n", toc());
    printHomography(homography);
    printf(" Number of inliers: %d\n", numFit);

    confidence = (double)numFit / (double)siftData1.numPts;

    // Free SiftData1, it is no longer needed
    numFeatures = siftData1.numPts;
    FreeSiftData(siftData1);

    tic();
    // Generate results
    generateResults(
        img1, img2,
        homography, confidence,
        outputDir, outputPrefix,
        files[0], files[1],
        numMatches, numFit, numFeatures,
        numOctaves, initBlur, thresh, lowestScale,
        numLoops_find, minScore_find, maxAmbiguity_find, threshFit_find,
        numLoops_improve, minScore_improve, maxAmbiguity_improve, threshFit_improve);
    printf(" Generated outputs (%lf s)\n\n", toc());

    // Confidence
    printf(" Confidence (inlier ratio): %.3lf%%\n\n", confidence * 100.0);

    Log("Finished SIFT\n");
}

std::string printAuthors()
{
    std::string str = "\n";
    //printf("\n");
    //printf("         <-- CudaSift - SIFT features with CUDA -->\n\n");
    //printf(" M. Björkman, N. Bergström and D. Kragic\n");
    //printf(" \"Detecting, segmenting and tracking unknown objects using multi-label MRF inference\",");
    //printf(" CVIU, 118, pp. 111-127, January 2014.\n");
    //printf(" http://www.sciencedirect.com/science/article/pii/S107731421300194X\n\n");

    str += "         <-- CudaSift - SIFT features with CUDA -->\n\n";
    str += " M. Björkman, N. Bergström and D. Kragic\n";
    str += " \"Detecting, segmenting and tracking unknown objects using multi-label MRF inference\",";
    str += " CVIU, 118, pp. 111-127, January 2014.\n";
    str += " http://www.sciencedirect.com/science/article/pii/S107731421300194X\n\n";

    return str;
}

void verifyArgs(const po::variables_map &vm, const po::options_description &desc)
{
    if (vm.count("help")) {
        std::cout << desc;
        exit(0);
    }

    if (vm.count("input") == 0) {
        std::cout << "No input files specified" << std::endl;
        std::cout << desc;
        exit(0);
    }

    if (vm["input"].as<std::vector<std::string>>().size() != 2) {
        std::cout << "Two input files must be specified" << std::endl;
        std::cout << desc;
        exit(0);
    }
}

void readImages(const std::vector<std::string> &files, cv::Mat &img1, cv::Mat &img2, bool _clahe)
{
    img1 = cv::imread(files[0], cv::IMREAD_GRAYSCALE);
    if (img1.empty()) {
        std::cout << "Could not read image " << files[0] << std::endl;
        exit(0);
    }    
    
    img2 = cv::imread(files[1], cv::IMREAD_GRAYSCALE);
    if (img2.empty()) {
        std::cout << "Could not read image " << files[1] << std::endl;
        exit(0);
    }

    // Apply CLAHE, to improve contrast
    if (_clahe)
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
        clahe->setClipLimit(4);
        clahe->apply(img1, img1);
        clahe->apply(img2, img2);
    }

    // For both images, find all pixels that are 0
    // Then set those pixels to 1
    cv::Mat img1Mask = img1 == 0;
    cv::Mat img2Mask = img2 == 0;
    img1.setTo(cv::Scalar(1.0), img1Mask);
    img2.setTo(cv::Scalar(1.0), img2Mask);

    // Convert to 32-bit float
    img1.convertTo(img1, CV_32FC1);
    img2.convertTo(img2, CV_32FC1);

}

void loadImages(const cv::Mat &img1, const cv::Mat &img2, CudaImage &img1Cuda, CudaImage &img2Cuda)
{
    img1Cuda.Allocate(img1.cols, img1.rows, img1.cols, false, NULL, (float*)img1.data);
    img2Cuda.Allocate(img2.cols, img2.rows, img2.cols, false, NULL, (float*)img2.data);

    // Download images to the device
    img1Cuda.Download();
    img2Cuda.Download();
}

void warpImage(const cv::Mat &img1, const cv::Mat &img2, cv::Mat &img2Warped, const float *homography)
{
    // Warp image 2 onto image 1 dimensions
    cv::Mat H(3, 3, CV_32FC1, (void*)homography);
    H = H.inv();

#ifdef HAVE_OPENCV_CUDAWARPING
    // Use CUDA warping
    cv::cuda::GpuMat img2Gpu(img2);
    cv::cuda::GpuMat img2WarpedGpu;
    cv::cuda::warpPerspective(img2Gpu, img2WarpedGpu, H, img1.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT);

    // Download the image
    img2WarpedGpu.download(img2Warped);
#else
    cv::warpPerspective(img2, img2Warped, H, img1.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT);
#endif
}

void imfuse(const cv::Mat &img1, const cv::Mat &img2, cv::Mat &imgFused, const cv::Mat &img2Mask)
{
    // Make sure the images are the same size
    if (img1.size() != img2.size())
    {
        std::cout << "Images must be the same size" << std::endl;
        exit(0);
    }

    // imgFused shall be an RGB image, opencv RGB is BGR
    // The red channel shall be filled with img2
    // The green channel shall be filled with img1
    // The blue channel shall be filled with img2
    imgFused.create(img1.rows, img1.cols, CV_8UC3);
    std::vector<cv::Mat> channels(3);
    channels[0] = img2;
    channels[1] = img1;
    channels[2] = img2;

    cv::merge(channels, imgFused);

    // If img2Mask is not empty, then use it to mask out the black pixels in img2
    if (!img2Mask.empty())
        imgFused.setTo(cv::Scalar(0, 0, 0), ~img2Mask);
}

void graydiff(const cv::Mat &img1, const cv::Mat &img2, cv::Mat &imgDiff, const cv::Mat &img2Mask)
{
    // Make sure the images are the same size
    if (img1.size() != img2.size())
    {
        std::cout << "Images must be the same size" << std::endl;
        exit(0);
    }

    // imgDiff shall be a 8-bit unsigned integer
    // It will be a pixel-wise squared difference between img1 and img2
    imgDiff.create(img1.rows, img1.cols, CV_8UC1);

    // Calculate the difference
    cv::absdiff(img1, img2, imgDiff);

    // check if img2Mask is empty
    if (!img2Mask.empty())
    {
        // If img2Mask is not empty, then use it to mask out the black pixels in img2
        // The img2Mask need to be inverted, since the black pixels shall be masked out
        imgDiff.setTo(cv::Scalar(0.0), ~img2Mask);
    }

    // Square the difference
    //cv::pow(imgDiff, 2.0, imgDiff);

}

void generateResults(
    cv::Mat &img1, cv::Mat &img2,
    const float* homography, double inlier_ratio,
    const std::string &outputDir, const std::string &outputPrefix,
    const std::string &img1Name_orig, const std::string &img2Name_orig,
    uint32_t numMatches, uint32_t numFit, uint32_t numFeatures,
    uint32_t numOctaves, double initBlur, double thresh, double lowestScale,
    uint32_t numLoops_find, double minScore_find, double maxAmbiguity_find, double threshFit_find,
    uint32_t numLoops_improve, double minScore_improve, double maxAmbiguity_improve, double threshFit_improve)
{
    float original_homography[9];
    memcpy((void*)original_homography, (void*)homography, 9 * sizeof(float));
    float val;

    // Convert images to 8-bit unsigned integer
    img1.convertTo(img1, CV_8UC1);
    img2.convertTo(img2, CV_8UC1);

    // Write the homography and inlier ratio to disk, json format
    // Write all the parameters to disk, json format
    std::string homographyName = outputDir + "/" + outputPrefix + "_results.json";
    FILE *fp = fopen(homographyName.c_str(), "w");
    fprintf(fp, "{\n");
    fprintf(fp, "  \"homography\": [\n");
    fprintf(fp, "    [%.9e, %.9e, %.9e],\n", homography[0], homography[1], homography[2]);
    fprintf(fp, "    [%.9e, %.9e, %.9e],\n", homography[3], homography[4], homography[5]);
    fprintf(fp, "    [%.9e, %.9e, %.9e]\n", homography[6], homography[7], homography[8]);
    fprintf(fp, "  ],\n");
    fprintf(fp, "  \"inlier_ratio\": %.3lf,\n", 100.0 * inlier_ratio);
    fprintf(fp, "  \"num_matches\": %u,\n", numMatches);
    fprintf(fp, "  \"num_inliers\": %u,\n", numFit);
    fprintf(fp, "  \"num_features\": %u,\n", numFeatures);
    fprintf(fp, "  \"num_octaves\": %u,\n", numOctaves);
    fprintf(fp, "  \"init_blur\": %.9e,\n", initBlur);
    fprintf(fp, "  \"thresh\": %.9e,\n", thresh);
    fprintf(fp, "  \"lowest_scale\": %.9e,\n", lowestScale);
    fprintf(fp, "  \"num_loops_find\": %u,\n", numLoops_find);
    fprintf(fp, "  \"min_score_find\": %.9e,\n", minScore_find);
    fprintf(fp, "  \"max_ambiguity_find\": %.9e,\n", maxAmbiguity_find);
    fprintf(fp, "  \"thresh_fit_find\": %.9e,\n", threshFit_find);
    fprintf(fp, "  \"num_loops_improve\": %u,\n", numLoops_improve);
    fprintf(fp, "  \"min_score_improve\": %.9e,\n", minScore_improve);
    fprintf(fp, "  \"max_ambiguity_improve\": %.9e,\n", maxAmbiguity_improve);
    fprintf(fp, "  \"thresh_fit_improve\": %.9e\n", threshFit_improve);
    fprintf(fp, "}\n");
    fclose(fp);

    // Warp image 2 onto image 1 dimensions
    cv::Mat img2Warped;
    warpImage(img1, img2, img2Warped, homography);

    // Find the pixels that are not black in image 2
    cv::Mat img2WarpedMask;
    cv::threshold(img2Warped, img2WarpedMask, 0.0, 255.0, cv::THRESH_BINARY);

    // Fuse the images
    cv::Mat imgFused;
    imfuse(img1, img2Warped, imgFused, img2WarpedMask);

    // Generate grayscale difference image
    cv::Mat imgDiff;
    graydiff(img1, img2Warped, imgDiff, img2WarpedMask);

    // Write the images to disk
    std::string img1Name = outputDir + "/" + outputPrefix + "_1.png";
    std::string img2WarpedName = outputDir + "/" + outputPrefix + "_2.png";
    std::string imgFusedName = outputDir + "/" + outputPrefix + "_fused.jpg";
    std::string imgDiffName = outputDir + "/" + outputPrefix + "_graydiff.jpg";

    cv::imwrite(img1Name, img1);
    cv::imwrite(imgFusedName, imgFused);
    cv::imwrite(img2WarpedName, img2Warped);
    cv::imwrite(imgDiffName, imgDiff);

    // Generate html file showing the resulting images
    std::string htmlName = outputDir + "/" + outputPrefix + ".html";
    fp = fopen(htmlName.c_str(), "w");
    fprintf(fp, "<html>\n");
    fprintf(fp, "<head>\n");
    fprintf(fp, "<title>%s</title>\n", outputPrefix.c_str());
    fprintf(fp, "</head>\n");
    
    fprintf(fp, "<body>\n");
    fprintf(fp, "<h1>%s</h1>\n", " CudaSift - SIFT features with CUDA");

    // Print the homography in a table, Then print the inlier ratio to the right of the table
    fprintf(fp, "<h2>%s</h2>\n", "Homography");
    // Add borders around the cells in the table
    // Padding between the cell border and the content by 5 pixels
    fprintf(fp, "<table border=\"1\" cellpadding=\"5\">\n");
    fprintf(fp, "<tr>\n");
    for (int i = 0; i < 9; i++)
    {
        val = original_homography[i];
        fprintf(fp, "<td>%.9f</td>\n", val);
        if ((i + 1) % 3 == 0) fprintf(fp, "</tr>\n<tr>\n");
    }
    fprintf(fp, "</tr>\n");
    fprintf(fp, "</table>\n");

    // Print the original images side by side.
    // Add a caption below the images, with the image names from disk
    // Add line break
    fprintf(fp, "<br>\n");
    fprintf(fp, "<h2>%s</h2>\n", "Input images");
    fprintf(fp, "<table border=\"1\">\n");
    fprintf(fp, "<tr>\n");
    fprintf(fp, "<td><img src=\"%s\" width=\"100%%\"></td>\n", img1Name_orig.c_str());
    fprintf(fp, "<td><img src=\"%s\" width=\"100%%\"></td>\n", img2Name_orig.c_str());
    fprintf(fp, "</tr>\n");
    fprintf(fp, "<tr>\n");
    fprintf(fp, "<td><p>%s</p></td>\n", img1Name_orig.c_str());
    fprintf(fp, "<td><p>%s</p></td>\n", img2Name_orig.c_str());
    fprintf(fp, "</tr>\n");
    fprintf(fp, "</table>\n");

    // Print the resulting images side by side.
    // Add a caption below the images, with the image names from disk
    fprintf(fp, "<br>\n");
    fprintf(fp, "<h2>%s</h2>\n", "Resulting images after SIFT - RANSAC - Warp");
    fprintf(fp, "<table border=\"1\">\n");
    fprintf(fp, "<tr>\n");
    fprintf(fp, "<td><img src=\"%s\" width=\"100%%\"></td>\n", img1Name.c_str());
    fprintf(fp, "<td><img src=\"%s\" width=\"100%%\"></td>\n", img2WarpedName.c_str());
    fprintf(fp, "</tr>\n");
    fprintf(fp, "<tr>\n");
    fprintf(fp, "<td><p>%s</p></td>\n", img1Name.c_str());
    fprintf(fp, "<td><p>%s</p></td>\n", img2WarpedName.c_str());
    fprintf(fp, "</tr>\n");
    fprintf(fp, "</table>\n");

    // Print the fused image
    // Add a caption below the image, with the image name from disk
    fprintf(fp, "<br>\n");
    fprintf(fp, "<h2>%s</h2>\n", "Fused image");
    fprintf(fp, "<table border=\"1\">\n");
    fprintf(fp, "<tr>\n");
    fprintf(fp, "<td><img src=\"%s\" width=\"100%%\"></td>\n", imgFusedName.c_str());
    fprintf(fp, "</tr>\n");
    fprintf(fp, "<tr>\n");
    fprintf(fp, "<td><p>%s</p></td>\n", imgFusedName.c_str());
    fprintf(fp, "</tr>\n");
    fprintf(fp, "</table>\n");

    // Print the grayscale difference image
    // Add a caption below the image, with the image name from disk
    fprintf(fp, "<br>\n");
    fprintf(fp, "<h2>%s</h2>\n", "Grayscale difference image");
    fprintf(fp, "<table border=\"1\">\n");
    fprintf(fp, "<tr>\n");
    fprintf(fp, "<td><img src=\"%s\" width=\"100%%\"></td>\n", imgDiffName.c_str());
    fprintf(fp, "</tr>\n");
    fprintf(fp, "<tr>\n");
    fprintf(fp, "<td><p>%s</p></td>\n", imgDiffName.c_str());
    fprintf(fp, "</tr>\n");
    fprintf(fp, "</table>\n");

    fprintf(fp, "<br>\n");
    fprintf(fp, "<p>%s\n", "M. Björkman, N. Bergström and D. Kragic");
    fprintf(fp, "%s\n", "\"Detecting, segmenting and tracking unknown objects using multi-label MRF inference\", CVIU, 118, pp. 111-127, January 2014.");
    fprintf(fp, "<a href=\"%s\">%s</a></p>\n", "http://www.sciencedirect.com/science/article/pii/S107731421300194X", "http://www.sciencedirect.com/science/article/pii/S107731421300194X");
    fprintf(fp, "</body>\n");
}

inline std::string Now()
{
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%d-%m-%Y %H:%M:%S");
    return oss.str();
}

inline void tic()
{
    auto now = std::chrono::high_resolution_clock::now();
    tictoc_stack.push(now);
}

inline double toc()
{
    auto now = std::chrono::high_resolution_clock::now();
    if (tictoc_stack.empty())
        return 0.0;
    auto then = tictoc_stack.top();
    tictoc_stack.pop();

    // Return the time in seconds as a double
    return std::chrono::duration<double>(now - then).count();
}

