
#include <iostream>
#include <string>
#include <vector>
#include <limits>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <boost/program_options.hpp>

#include <hdf5.h>

#include "cudaImage.h"
#include "cudaSift.h"


namespace po = boost::program_options;

// Global variables

// Generic parameters
static bool s_verbose = false;
static std::vector<std::string> s_input_files;
static std::string s_output_file;
static std::string s_identifier;
static int s_device_num = 0;

// smart search parameters
static bool s_smart_search = false;
static int s_smart_search_iterations;
static int s_smart_search_children;
static float s_target_x_scaling;
static float s_target_y_scaling;
static float s_target_x_shearing;
static float s_target_y_shearing;

// extract feature parameters
static int s_num_features;
static double s_init_blur;
static float s_feature_thresh;
static float s_lowest_scale;

// find homography parameters
static int s_find_homography_num_loops;
static float s_find_homography_min_score;
static float s_find_homography_max_ambiguity;
static float s_find_homography_thresh;

// improve homography parameters
static int s_improve_homography_num_loops;
static float s_improve_homography_min_score;
static float s_improve_homography_max_ambiguity;
static float s_improve_homography_thresh;

void parseArgs(int argc, char** argv);
void ImproveHomography(SiftData &data, float *homography, int numLoops, float minScore, float maxAmbiguity, float thresh);
void writeResults(const float* homography);
void prettyPrint(const float* homography);


void singleProcess_quiet();
void singleProcess_verbose();

void smartProcess_quiet();
void smartProcess_verbose();

void scoreHomography(const float* homography, float* score);

class TempSiftMem
{
    private:
        float* m_memoryTmp;

    public:
        TempSiftMem() : m_memoryTmp(NULL) {}
        TempSiftMem(int w, int h, int numOctaves)
        {
            m_memoryTmp = AllocSiftTempMemory(w, h, numOctaves, false);
        }

        ~TempSiftMem()
        {
            FreeSiftTempMemory(m_memoryTmp);
        }

        void allocate(int w, int h, int numOctaves)
        {
            m_memoryTmp = AllocSiftTempMemory(w, h, numOctaves, false);
        }

        float* get()
        {
            return m_memoryTmp;
        }
};


int main(int argc, char** argv)
{
    std::chrono::high_resolution_clock::time_point total_1, total_2;
    parseArgs(argc, argv);

    if (s_verbose && !s_smart_search)
    {
        total_1 = std::chrono::high_resolution_clock::now();
        std::cout << "===================================================" << std::endl;
        std::cout << "CudaSift Single Process\n" << std::endl;

        InitCuda(s_device_num, s_verbose);
        singleProcess_verbose();
        total_2 = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(total_2 - total_1);
        std::cout << " Total time: " << time_span.count() << " [s]" << std::endl;
        std::cout << "===================================================" << std::endl;
    }
    else if (!s_verbose && !s_smart_search)
    {
        InitCuda(s_device_num, s_verbose);
        singleProcess_quiet();
    }
    else if (s_verbose && s_smart_search)
    {
        total_1 = std::chrono::high_resolution_clock::now();
        std::cout << "===================================================" << std::endl;
        std::cout << "CudaSift Smart Search\n" << std::endl;

        InitCuda(s_device_num, s_verbose);
        smartProcess_verbose();
        total_2 = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(total_2 - total_1);
        std::cout << " Total time: " << time_span.count() << " [s]" << std::endl;
        std::cout << "===================================================" << std::endl;
    }
    else
    {
        InitCuda(s_device_num, s_verbose);
        smartProcess_quiet();
    }

    return 0;
}

void parseArgs(int argc, char** argv)
{
    po::options_description desc("Generic options");
    po::options_description smart_search_options("Smart search options");
    po::options_description feature_options("Extract SIFT feature options");
    po::options_description homography_options("Find homography options");
    po::options_description improve_homography_options("Improve homography options");

    desc.add_options()
        ("help,h", "Produce this help message")
        ("verbose,v", "Verbose output")
        ("device,d", po::value<int>(&s_device_num)->default_value(0), "CUDA device number")
        ("input,i", po::value<std::vector<std::string>>(&s_input_files)->multitoken(), "Input files")
        ("output,o", po::value<std::string>(&s_output_file), "Output file")
        ("identifier", po::value<std::string>(&s_identifier), "Identifier for type of images being processed");
    
    smart_search_options.add_options()
        ("smart-search", "Use smart search to find homography")
        ("smart-search-iterations", po::value<int>(&s_smart_search_iterations)->default_value(6), "Number of iterations to run smart search, this will max out at 10")
        ("smart-search-children", po::value<int>(&s_smart_search_children)->default_value(8), "Number of children to spawn per iteration")
        ("target-x-scaling", po::value<float>(&s_target_x_scaling)->default_value(1.0f), "Target x scaling")
        ("target-y-scaling", po::value<float>(&s_target_y_scaling)->default_value(1.0f), "Target y scaling")
        ("target-x-shearing", po::value<float>(&s_target_x_shearing)->default_value(0.0f), "Target x shearing")
        ("target-y-shearing", po::value<float>(&s_target_y_shearing)->default_value(0.0f), "Target y shearing");

    feature_options.add_options()
        ("num-features", po::value<int>(&s_num_features)->default_value(2000), "Number of features to extract")
        ("init-blur", po::value<double>(&s_init_blur)->default_value(1.0), "Initial blur level")
        ("feature-thresh", po::value<float>(&s_feature_thresh)->default_value(3.0), "Feature threshold")
        ("lowest-scale", po::value<float>(&s_lowest_scale)->default_value(0.0), "Lowest scale to detect features");

    homography_options.add_options()
        ("find-homography-num-loops", po::value<int>(&s_find_homography_num_loops)->default_value(1000), "Number of loops to find homography")
        ("find-homography-min-score", po::value<float>(&s_find_homography_min_score)->default_value(0.85), "Minimum score to find homography")
        ("find-homography-max-ambiguity", po::value<float>(&s_find_homography_max_ambiguity)->default_value(0.95), "Maximum ambiguity to find homography")
        ("find-homography-thresh", po::value<float>(&s_find_homography_thresh)->default_value(5.0), "Threshold to find homography");

    improve_homography_options.add_options()
        ("improve-homography-num-loops", po::value<int>(&s_improve_homography_num_loops)->default_value(5), "Number of loops to improve homography")
        ("improve-homography-min-score", po::value<float>(&s_improve_homography_min_score)->default_value(0.0), "Minimum score to improve homography")
        ("improve-homography-max-ambiguity", po::value<float>(&s_improve_homography_max_ambiguity)->default_value(0.8), "Maximum ambiguity to improve homography")
        ("improve-homography-thresh", po::value<float>(&s_improve_homography_thresh)->default_value(3.0), "Threshold to improve homography");

    desc.add(smart_search_options).add(feature_options).add(homography_options).add(improve_homography_options);

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help"))
    {
        // Print help message
        std::cout << desc << std::endl;
        exit(0);
    }

    if (vm.count("verbose"))
        s_verbose = true;

    if (vm.count("smart-search"))
        s_smart_search = true;

    // input files must contain 2 files
    if (s_input_files.size() != 2)
    {
        std::cerr << desc << std::endl;
        std::cerr << "Error: --input or -i must be followed by two files to process" << std::endl;
        exit(1);
    }

    // output file must be specified
    if (s_output_file.empty())
    {
        std::cerr << desc << std::endl;
        std::cerr << "Error: --output or -o must be followed by an output file" << std::endl;
        exit(1);
    }

    // Clip smart search iterations to 10
    s_smart_search_iterations = std::min(s_smart_search_iterations, 10);
}

void ImproveHomography(SiftData &data, float *homography, int numLoops, float minScore, float maxAmbiguity, float thresh)
{
#ifdef MANAGEDMEM
  SiftPoint *mpts = data.m_data;
#else
  if (data.h_data==NULL)
    return;
  SiftPoint *mpts = data.h_data;
#endif
  float limit = thresh*thresh;
  int numPts = data.numPts;
  cv::Mat M(8, 8, CV_64FC1);
  cv::Mat A(8, 1, CV_64FC1), X(8, 1, CV_64FC1);
  double Y[8];
  for (int i=0;i<8;i++) 
    A.at<double>(i, 0) = homography[i] / homography[8];
  for (int loop=0;loop<numLoops;loop++) {
    M = cv::Scalar(0.0);
    X = cv::Scalar(0.0);
    for (int i=0;i<numPts;i++) {
      SiftPoint &pt = mpts[i];
      if (pt.score<minScore || pt.ambiguity>maxAmbiguity)
	continue;
      float den = A.at<double>(6)*pt.xpos + A.at<double>(7)*pt.ypos + 1.0f;
      float dx = (A.at<double>(0)*pt.xpos + A.at<double>(1)*pt.ypos + A.at<double>(2)) / den - pt.match_xpos;
      float dy = (A.at<double>(3)*pt.xpos + A.at<double>(4)*pt.ypos + A.at<double>(5)) / den - pt.match_ypos;
      float err = dx*dx + dy*dy;
      float wei = (err<limit ? 1.0f : 0.0f); //limit / (err + limit);
      Y[0] = pt.xpos;
      Y[1] = pt.ypos;
      Y[2] = 1.0;
      Y[3] = Y[4] = Y[5] = 0.0;
      Y[6] = - pt.xpos * pt.match_xpos;
      Y[7] = - pt.ypos * pt.match_xpos;
      for (int c=0;c<8;c++) 
        for (int r=0;r<8;r++) 
          M.at<double>(r,c) += (Y[c] * Y[r] * wei);
      X += (cv::Mat(8,1,CV_64FC1,Y) * pt.match_xpos * wei);
      Y[0] = Y[1] = Y[2] = 0.0;
      Y[3] = pt.xpos;
      Y[4] = pt.ypos; 
      Y[5] = 1.0;
      Y[6] = - pt.xpos * pt.match_ypos;
      Y[7] = - pt.ypos * pt.match_ypos;
      for (int c=0;c<8;c++) 
        for (int r=0;r<8;r++) 
          M.at<double>(r,c) += (Y[c] * Y[r] * wei);
      X += (cv::Mat(8,1,CV_64FC1,Y) * pt.match_ypos * wei);
    }
    cv::solve(M, X, A, cv::DECOMP_CHOLESKY);
  }
  int numfit = 0;
  for (int i=0;i<numPts;i++) {
    SiftPoint &pt = mpts[i];
    float den = A.at<double>(6)*pt.xpos + A.at<double>(7)*pt.ypos + 1.0;
    float dx = (A.at<double>(0)*pt.xpos + A.at<double>(1)*pt.ypos + A.at<double>(2)) / den - pt.match_xpos;
    float dy = (A.at<double>(3)*pt.xpos + A.at<double>(4)*pt.ypos + A.at<double>(5)) / den - pt.match_ypos;
    float err = dx*dx + dy*dy;
    if (err<limit) 
      numfit++;
    pt.match_error = sqrt(err);
  }
  for (int i=0;i<8;i++) 
    homography[i] = A.at<double>(i);
  homography[8] = 1.0f;
  return;
}

void writeResults(const float* homography)
{
    const char* filename = s_output_file.c_str();
    const char* image1 = s_input_files[0].c_str();
    const char* image2 = s_input_files[1].c_str();
    const char* identifier = s_identifier.c_str();

    hid_t file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    // write the homography
    {
        hsize_t dims[2] = {3, 3};
        hid_t dataspace_id = H5Screate_simple(2, dims, NULL);
        hid_t dataset_id = H5Dcreate(file_id, "homography", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, homography);
        H5Dclose(dataset_id);
        H5Sclose(dataspace_id);
    }

    // write the image1 string as a variable length
    {
        hid_t dataspace_id = H5Screate(H5S_SCALAR);
        hid_t string_type = H5Tcopy(H5T_C_S1);
        H5Tset_size(string_type, H5T_VARIABLE);

        hid_t dataset_id = H5Dcreate(file_id, "image1", string_type, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dataset_id, string_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, &image1);
        H5Dclose(dataset_id);
        H5Sclose(dataspace_id);
    }

    // write the image2 string as a variable length
    {
        hid_t dataspace_id = H5Screate(H5S_SCALAR);
        hid_t string_type = H5Tcopy(H5T_C_S1);
        H5Tset_size(string_type, H5T_VARIABLE);

        hid_t dataset_id = H5Dcreate(file_id, "image2", string_type, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dataset_id, string_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, &image2);
        H5Dclose(dataset_id);
        H5Sclose(dataspace_id);
    }

    // Write the identifier as a variable length
    {
        hid_t dataspace_id = H5Screate(H5S_SCALAR);
        hid_t string_type = H5Tcopy(H5T_C_S1);
        H5Tset_size(string_type, H5T_VARIABLE);

        hid_t dataset_id = H5Dcreate(file_id, "identifier", string_type, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dataset_id, string_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, &identifier);
        H5Dclose(dataset_id);
        H5Sclose(dataspace_id);
    }
}

void prettyPrint(const float* homography)
{
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3;j++)
        {
            fprintf(stdout, " %15.10f", homography[i * 3 + j]);
        }
        fprintf(stdout, "\n");
    }
    fprintf(stdout, "\n");
}


void singleProcess_quiet()
{
    SiftData siftData1;
    SiftData siftData2;
    float homography[9];
    int numMatches;

    {
        CudaImage img1, img2;
        {
            cv::Mat limg, rimg;

            cv::imread(s_input_files[0], 0).convertTo(limg, CV_32FC1);
            cv::imread(s_input_files[1], 0).convertTo(rimg, CV_32FC1);

            if (limg.empty())
            {
                std::cerr << "Error: could not read image " << s_input_files[0] << std::endl;
                exit(1);
            }
            if (rimg.empty())
            {
                std::cerr << "Error: could not read image " << s_input_files[1] << std::endl;
                exit(1);
            }

            unsigned int w1 = limg.cols;
            unsigned int h1 = limg.rows;
            unsigned int w2 = rimg.cols;
            unsigned int h2 = rimg.rows;

            img1.Allocate(w1, h1, iAlignUp(w1, 128), false, NULL, (float*)limg.data);
            img2.Allocate(w2, h2, iAlignUp(w2, 128), false, NULL, (float*)rimg.data);

            img1.Download();
            img2.Download();
        }

        InitSiftData(siftData1, s_num_features, true, true);
        InitSiftData(siftData2, s_num_features, true, true);

        {
            TempSiftMem tmpMem;

            // Allocate space for the largest possible image
            int w = std::max(img1.width, img2.width);
            int h = std::max(img1.height, img2.height);
            tmpMem.allocate(w, h, 5);

            ExtractSift(siftData1, img1, 5, s_init_blur, s_feature_thresh, s_lowest_scale, false, tmpMem.get());
            ExtractSift(siftData2, img2, 5, s_init_blur, s_feature_thresh, s_lowest_scale, false, tmpMem.get());
        }
    }

    MatchSiftData(siftData1, siftData2);

    FindHomography(siftData1, homography, &numMatches, s_find_homography_num_loops, s_find_homography_min_score, s_find_homography_max_ambiguity, s_find_homography_thresh);

    ImproveHomography(siftData1, homography, s_improve_homography_num_loops, s_improve_homography_min_score, s_improve_homography_max_ambiguity, s_improve_homography_thresh);

    FreeSiftData(siftData1);
    FreeSiftData(siftData2);

    writeResults(homography);
}

void singleProcess_verbose()
{
    SiftData siftData1;
    SiftData siftData2;
    float homography[9];
    int numMatches;
    std::chrono::high_resolution_clock::time_point t1, t2;
    std::chrono::duration<double> time_span;

    {
        CudaImage img1, img2;
        {
            cv::Mat limg, rimg;

            std::cout << " Reading images from disk\n  " << s_input_files[0] << "\n  " << s_input_files[1] << std::endl;

            t1 = std::chrono::high_resolution_clock::now();
            cv::imread(s_input_files[0], 0).convertTo(limg, CV_32FC1);
            cv::imread(s_input_files[1], 0).convertTo(rimg, CV_32FC1);
            t2 = std::chrono::high_resolution_clock::now();
            time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

            if (limg.empty())
            {
                std::cerr << "Error: could not read image " << s_input_files[0] << std::endl;
                exit(1);
            }
            if (rimg.empty())
            {
                std::cerr << "Error: could not read image " << s_input_files[1] << std::endl;
                exit(1);
            }

            std::cout << " ...Done " << time_span.count() << " [s]\n" << std::endl;

            unsigned int w1 = limg.cols;
            unsigned int h1 = limg.rows;
            unsigned int w2 = rimg.cols;
            unsigned int h2 = rimg.rows;

            std::cout << " Allocating device memory for images" << std::endl;
            t1 = std::chrono::high_resolution_clock::now();
            img1.Allocate(w1, h1, iAlignUp(w1, 128), false, NULL, (float*)limg.data);
            img2.Allocate(w2, h2, iAlignUp(w2, 128), false, NULL, (float*)rimg.data);
            t2 = std::chrono::high_resolution_clock::now();

            time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

            std::cout << " ...Done " << time_span.count() << " [s]\n" << std::endl;

            std::cout << " Downloading images to device" << std::endl;

            t1 = std::chrono::high_resolution_clock::now();
            img1.Download();
            img2.Download();
            t2 = std::chrono::high_resolution_clock::now();

            time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

            std::cout << " ...Done " << time_span.count() << " [s]\n" << std::endl;
        }

        std::cout << " Allocating device memory for SIFT features" << std::endl;

        t1 = std::chrono::high_resolution_clock::now();
        InitSiftData(siftData1, s_num_features, true, true);
        InitSiftData(siftData2, s_num_features, true, true);
        t2 = std::chrono::high_resolution_clock::now();

        time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

        std::cout << " ...Done " << time_span.count() << " [s]\n" << std::endl;

        {
            TempSiftMem tmpMem;

            // Allocate space for the largest possible image
            int w = std::max(img1.width, img2.width);
            int h = std::max(img1.height, img2.height);
            tmpMem.allocate(w, h, 5);

            std::cout << " Extracting SIFT features" << std::endl;

            t1 = std::chrono::high_resolution_clock::now();
            ExtractSift(siftData1, img1, 5, s_init_blur, s_feature_thresh, s_lowest_scale, false, tmpMem.get());
            ExtractSift(siftData2, img2, 5, s_init_blur, s_feature_thresh, s_lowest_scale, false, tmpMem.get());
            t2 = std::chrono::high_resolution_clock::now();

            time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

            std::cout << " ...Done " << time_span.count() << " [s]\n" << std::endl;
        }
    }

    std::cout << " Matching SIFT features" << std::endl;

    t1 = std::chrono::high_resolution_clock::now();
    MatchSiftData(siftData1, siftData2);
    t2 = std::chrono::high_resolution_clock::now();

    time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

    std::cout << " ...Done " << time_span.count() << " [s]\n" << std::endl;

    std::cout << " Finding homography" << std::endl;

    t1 = std::chrono::high_resolution_clock::now();
    FindHomography(siftData1, homography, &numMatches, s_find_homography_num_loops, s_find_homography_min_score, s_find_homography_max_ambiguity, s_find_homography_thresh);
    t2 = std::chrono::high_resolution_clock::now();

    time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

    std::cout << " ...Done " << time_span.count() << " [s]\n" << std::endl;

    prettyPrint(homography);

    std::cout << " Improving homography" << std::endl;

    t1 = std::chrono::high_resolution_clock::now();
    ImproveHomography(siftData1, homography, s_improve_homography_num_loops, s_improve_homography_min_score, s_improve_homography_max_ambiguity, s_improve_homography_thresh);
    t2 = std::chrono::high_resolution_clock::now();

    time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

    std::cout << " ...Done " << time_span.count() << " [s]\n" << std::endl;

    prettyPrint(homography);

    FreeSiftData(siftData1);
    FreeSiftData(siftData2);

    std::cout << " Writing results to disk" << std::endl;

    t1 = std::chrono::high_resolution_clock::now();
    writeResults(homography);
    t2 = std::chrono::high_resolution_clock::now();

    time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

    std::cout << " ...Done " << time_span.count() << " [s]\n" << std::endl;
}



inline void linspace(std::vector<float>& vec, float a, float b, int n)
{
    const float step = (b - a) / (float)(n - 1);
    vec.resize(n);
    for (int i = 0; i < n; i++)
        vec[i] = a + (float)i * step;
}

void smartProcess_verbose()
{
    SiftData siftData1;
    SiftData siftData2;
    float homography[9];
    float best_homography[9];
    float largest_best_score = std::numeric_limits<float>::max();
    int numMatches;
    std::chrono::high_resolution_clock::time_point t1, t2;
    std::chrono::duration<double> time_span;

    // We will only attempt to evolve the init_blur and feature_thresh parameters
    std::vector<float> initBlur_children(s_smart_search_children);
    std::vector<float> featureThresh_children(s_smart_search_children);

    // We will also need to keep track of the scores for each child
    std::vector<float> scores(s_smart_search_children);

    // vector of homographies
    std::vector<float> homographies(s_smart_search_children * 9);

    // Initial range for next generation, centered around the current best
    float thresh_range = 1.0f;
    float blur_range = 0.2f;

    // Initialize the children
    // init_blur should be between 1.0 and 2.0, linspace equally between the two
    // feature_thresh should be between 1.6 and 10.0, linspace equally between the two
    linspace(initBlur_children, 1.0, 2.0, s_smart_search_children);
    linspace(featureThresh_children, 1.6, 10.0, s_smart_search_children);

    int iteration = 0;

    {
        CudaImage img1, img2;
        {
            cv::Mat limg, rimg;

            std::cout << " Reading images from disk\n  " << s_input_files[0] << "\n  " << s_input_files[1] << std::endl;

            t1 = std::chrono::high_resolution_clock::now();
            cv::imread(s_input_files[0], 0).convertTo(limg, CV_32FC1);
            cv::imread(s_input_files[1], 0).convertTo(rimg, CV_32FC1);
            t2 = std::chrono::high_resolution_clock::now();
            time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

            if (limg.empty())
            {
                std::cerr << "Error: could not read image " << s_input_files[0] << std::endl;
                exit(1);
            }
            if (rimg.empty())
            {
                std::cerr << "Error: could not read image " << s_input_files[1] << std::endl;
                exit(1);
            }

            std::cout << " ...Done " << time_span.count() << " [s]\n" << std::endl;

            unsigned int w1 = limg.cols;
            unsigned int h1 = limg.rows;
            unsigned int w2 = rimg.cols;
            unsigned int h2 = rimg.rows;

            std::cout << " Allocating device memory for images" << std::endl;
            t1 = std::chrono::high_resolution_clock::now();
            img1.Allocate(w1, h1, iAlignUp(w1, 128), false, NULL, (float*)limg.data);
            img2.Allocate(w2, h2, iAlignUp(w2, 128), false, NULL, (float*)rimg.data);
            t2 = std::chrono::high_resolution_clock::now();

            time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

            std::cout << " ...Done " << time_span.count() << " [s]\n" << std::endl;

            std::cout << " Downloading images to device" << std::endl;

            t1 = std::chrono::high_resolution_clock::now();
            img1.Download();
            img2.Download();
            t2 = std::chrono::high_resolution_clock::now();

            time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

            std::cout << " ...Done " << time_span.count() << " [s]\n" << std::endl;
        }

        std::cout << " Allocating device memory for SIFT features" << std::endl;

        t1 = std::chrono::high_resolution_clock::now();
        InitSiftData(siftData1, s_num_features, true, true);
        InitSiftData(siftData2, s_num_features, true, true);
        t2 = std::chrono::high_resolution_clock::now();

        time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

        std::cout << " ...Done " << time_span.count() << " [s]\n" << std::endl;

        TempSiftMem tmpMem;

        // Allocate space for the largest possible image
        int w = std::max(img1.width, img2.width);
        int h = std::max(img1.height, img2.height);
        tmpMem.allocate(w, h, 5);


        do
        {
            std::cout << "---------------------------------------------------" << std::endl;
            std::cout << "Smart Search Iteration: " << iteration+1  << " | " << s_smart_search_iterations << std::endl;

            t1 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < s_smart_search_children; i++)
            {
                ExtractSift(siftData1, img1, 5, initBlur_children[i], featureThresh_children[i], s_lowest_scale, false, tmpMem.get());
                ExtractSift(siftData2, img2, 5, initBlur_children[i], featureThresh_children[i], s_lowest_scale, false, tmpMem.get());

                MatchSiftData(siftData1, siftData2);

                FindHomography(siftData1, homography, &numMatches, s_find_homography_num_loops, s_find_homography_min_score, s_find_homography_max_ambiguity, s_find_homography_thresh);

                ImproveHomography(siftData1, homography, s_improve_homography_num_loops, s_improve_homography_min_score, s_improve_homography_max_ambiguity, s_improve_homography_thresh);

                scoreHomography(homography, &scores[i]);

                // Copy the homography into the homographies vector
                memcpy(&homographies[i * 9], homography, sizeof(float) * 9);
            }

            iteration++;

            // Divide the ranges by 2, and center them around the best score
            thresh_range /= 2.0f;
            blur_range /= 2.0f;

            // Find the best score
            int best_index = 0;
            float best_score = scores[0];
            for (int i = 1; i < s_smart_search_children; i++)
            {
                if (scores[i] < best_score)
                {
                    best_score = scores[i];
                    best_index = i;
                }
            }
            // Update the init_blur and feature_thresh parameters
            linspace(initBlur_children, initBlur_children[best_index] - blur_range, initBlur_children[best_index] + blur_range, s_smart_search_children);
            linspace(featureThresh_children, featureThresh_children[best_index] - thresh_range, featureThresh_children[best_index] + thresh_range, s_smart_search_children);

            // If the best score is better than the current best, update the current best
            if (best_score < largest_best_score)
            {
                largest_best_score = best_score;
                memcpy(best_homography, &homographies[best_index * 9], sizeof(float) * 9);
            }

            t2 = std::chrono::high_resolution_clock::now();

            time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

            std::cout << " ...Done " << time_span.count() << " [s]\n" << std::endl;

            std::cout << " Best score: " << largest_best_score << std::endl;
            std::cout << " Best homography: " << std::endl;
            prettyPrint(best_homography);

            std::cout << "---------------------------------------------------\n" << std::endl;

        } while (iteration < s_smart_search_iterations);
    }

    FreeSiftData(siftData1);
    FreeSiftData(siftData2);

    std::cout << " Best homography:" << std::endl;
    prettyPrint(best_homography);

    std::cout << " Writing results to disk" << std::endl;

    t1 = std::chrono::high_resolution_clock::now();
    writeResults(best_homography);
    t2 = std::chrono::high_resolution_clock::now();

    time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

    std::cout << " ...Done " << time_span.count() << " [s]\n" << std::endl;

}


void smartProcess_quiet()
{
    SiftData siftData1;
    SiftData siftData2;
    float homography[9];
    float best_homography[9];
    float largest_best_score = std::numeric_limits<float>::max();
    int numMatches;

    // We will only attempt to evolve the init_blur and feature_thresh parameters
    std::vector<float> initBlur_children(s_smart_search_children);
    std::vector<float> featureThresh_children(s_smart_search_children);

    // We will also need to keep track of the scores for each child
    std::vector<float> scores(s_smart_search_children);

    // vector of homographies
    std::vector<float> homographies(s_smart_search_children * 9);

    // Initial range for next generation, centered around the current best
    float thresh_range = 1.0f;
    float blur_range = 0.2f;

    // Initialize the children
    // init_blur should be between 1.0 and 2.0, linspace equally between the two
    // feature_thresh should be between 1.6 and 10.0, linspace equally between the two
    linspace(initBlur_children, 1.0, 2.0, s_smart_search_children);
    linspace(featureThresh_children, 1.6, 10.0, s_smart_search_children);

    int iteration = 0;

    {
        CudaImage img1, img2;
        {
            cv::Mat limg, rimg;

            cv::imread(s_input_files[0], 0).convertTo(limg, CV_32FC1);
            cv::imread(s_input_files[1], 0).convertTo(rimg, CV_32FC1);

            if (limg.empty())
            {
                std::cerr << "Error: could not read image " << s_input_files[0] << std::endl;
                exit(1);
            }
            if (rimg.empty())
            {
                std::cerr << "Error: could not read image " << s_input_files[1] << std::endl;
                exit(1);
            }


            unsigned int w1 = limg.cols;
            unsigned int h1 = limg.rows;
            unsigned int w2 = rimg.cols;
            unsigned int h2 = rimg.rows;

            img1.Allocate(w1, h1, iAlignUp(w1, 128), false, NULL, (float*)limg.data);
            img2.Allocate(w2, h2, iAlignUp(w2, 128), false, NULL, (float*)rimg.data);

            img1.Download();
            img2.Download();
        }
        
        InitSiftData(siftData1, s_num_features, true, true);
        InitSiftData(siftData2, s_num_features, true, true);

        TempSiftMem tmpMem;

        // Allocate space for the largest possible image
        int w = std::max(img1.width, img2.width);
        int h = std::max(img1.height, img2.height);
        tmpMem.allocate(w, h, 5);


        do
        {
            for (int i = 0; i < s_smart_search_children; i++)
            {
                ExtractSift(siftData1, img1, 5, initBlur_children[i], featureThresh_children[i], s_lowest_scale, false, tmpMem.get());
                ExtractSift(siftData2, img2, 5, initBlur_children[i], featureThresh_children[i], s_lowest_scale, false, tmpMem.get());

                MatchSiftData(siftData1, siftData2);

                FindHomography(siftData1, homography, &numMatches, s_find_homography_num_loops, s_find_homography_min_score, s_find_homography_max_ambiguity, s_find_homography_thresh);

                ImproveHomography(siftData1, homography, s_improve_homography_num_loops, s_improve_homography_min_score, s_improve_homography_max_ambiguity, s_improve_homography_thresh);

                scoreHomography(homography, &scores[i]);

                // Copy the homography into the homographies vector
                memcpy(&homographies[i * 9], homography, sizeof(float) * 9);
            }

            iteration++;

            // Divide the ranges by 2, and center them around the best score
            thresh_range /= 2.0f;
            blur_range /= 2.0f;

            // Find the best score
            int best_index = 0;
            float best_score = scores[0];
            for (int i = 1; i < s_smart_search_children; i++)
            {
                if (scores[i] < best_score)
                {
                    best_score = scores[i];
                    best_index = i;
                }
            }
            // Update the init_blur and feature_thresh parameters
            linspace(initBlur_children, initBlur_children[best_index] - blur_range, initBlur_children[best_index] + blur_range, s_smart_search_children);
            linspace(featureThresh_children, featureThresh_children[best_index] - thresh_range, featureThresh_children[best_index] + thresh_range, s_smart_search_children);

            // If the best score is better than the current best, update the current best
            if (best_score < largest_best_score)
            {
                largest_best_score = best_score;
                memcpy(best_homography, &homographies[best_index * 9], sizeof(float) * 9);
            }

        } while (iteration < s_smart_search_iterations);
    }

    FreeSiftData(siftData1);
    FreeSiftData(siftData2);

    writeResults(best_homography);
}


void scoreHomography(const float* H, float* score)
{
    // H = | a b c |
    //     | d e f |
    //     | g h 1 |

    // x_scaling -> a = H[0]
    // y_scaling -> e = H[4]
    // x_shearing -> b = H[1]
    // y_shearing -> d = H[3]

    *score = fabs(H[0] - s_target_x_scaling) +
             fabs(H[4] - s_target_y_scaling) +
             fabs(H[1] - s_target_x_shearing) +
             fabs(H[3] - s_target_y_shearing);
}

