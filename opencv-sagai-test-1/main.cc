/*
 * main.cc
 *
 *  Created on: Oct 19, 2016
 *      Author: amyznikov
 *
 *  new_image = a*image + beta.
 *  image.convertTo(new_image, -1, alpha, beta);
 *
 *  http://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/
 *
 */

#include <opencv2/opencv.hpp>
#include <sys/types.h>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace cv;
using namespace std;


enum algo {
  algo_sub,
  algo_div
};




#define DEFAULT_MKDIR_MODE (S_IRWXU|S_IRGRP|S_IXGRP|S_IROTH|S_IXOTH)

static bool create_path(const char * path, mode_t mode = DEFAULT_MKDIR_MODE)
{
  size_t size;
  char tmp[(size = strlen(path ? path : "")) + 1];

  if ( strcpy(tmp, path ? path : "")[size - 1] == '/' ) {
    tmp[size - 1] = 0;
  }

  for ( char * p = tmp + 1; *p; p++ ) {
    if ( *p == '/' ) {
      *p = 0;
      if ( mkdir(tmp, mode) != 0 && errno != EEXIST ) {
        return false;
      }
      *p = '/';
    }
  }

  return mkdir(tmp, mode) == 0 || errno == EEXIST ? true : false;
}



static bool get_files(vector<string> & v, const char * path, const char * suffix)
{
  DIR * dir = 0;
  struct dirent * e = NULL;

  size_t name_len;
  size_t suffix_len = 0;

  char outfilename[PATH_MAX];

  int n = -1;

  if ( !(dir = opendir(path) )) {
    fprintf(stderr,"opendir(%s) fails: %s\n", path, strerror(errno) );
    goto end;
  }


  if ( suffix ) {
    suffix_len = strlen(suffix);
  }

  n = 0;
  while ( (e = readdir(dir)) ) {

    if ( (e->d_type == DT_LNK) || (e->d_type == DT_REG) ) {

      if ( suffix_len  ) {
        if ((name_len = strlen(e->d_name)) < suffix_len || strcmp(e->d_name+name_len-suffix_len, suffix) != 0 ) {
          continue;
        }
      }

      sprintf(outfilename, "%s/%s", path, e->d_name);
      v.emplace_back( outfilename );
      ++n;
    }
  }


  closedir(dir);

end:

  return n >= 0;
}



static void calc_hist(const Mat & image, double bins[], size_t numbins, double minval, double maxval)
{
  int b;

  for( int y = 0; y < image.rows; y++ ) {
    const float * row = reinterpret_cast<const float*>(image.ptr(y));
    for( int x = 0; x < image.cols; x++ ) {

      if ( (b = (row[x] - minval) * numbins / (maxval - minval)) < 0 ) {
        b = 0;
      }
      else if ( b >= (int)numbins ) {
        b = numbins - 1;
      }

      ++bins[b];
    }
  }
}

static int central_region_radius = 150;

static void get_energy_metrics(const Mat & image, double * mean, double * pwr)
{
  const double rr = min(min(central_region_radius, image.rows), image.cols);
  const double x0 = image.cols / 2, y0 = image.rows / 2;

  *pwr = 0, *mean = 0;

  size_t npix = 0;

  for ( int y = 0; y < image.rows; y++ ) {
    const float * row = reinterpret_cast<const float*>(image.ptr(y));
    for ( int x = 0; x < image.cols; x++ ) {
      if ( hypot(x - x0, y - y0) < rr ) {
        *mean += row[x];
        ++npix;
      }
    }
  }
  *mean /= npix;


  for ( int y = 0; y < image.rows; y++ ) {
    const float * row = reinterpret_cast<const float*>(image.ptr(y));
    for ( int x = 0; x < image.cols; x++ ) {
      if ( hypot(x - x0, y - y0) < rr ) {
        *pwr += (row[x] - *mean) * (row[x] - *mean);
      }
    }
  }

  *pwr = sqrt(*pwr / npix) ;
}



static void threshold_pixels(Mat & image, double minval, double maxval)
{
  for( int y = 0; y < image.rows; y++ ) {
    float * row = reinterpret_cast<float*>(image.ptr(y));
    for( int x = 0; x < image.cols; x++ ) {
      if ( row[x] < minval ) {
        row[x] = minval;
      }
      else if ( row[x] >= maxval ) {
        row[x] = maxval;
      }
    }
  }
}

static void normalize_pixels(Mat & image, double smin, double smax, double dmin, double dmax)
{
  for ( int y = 0; y < image.rows; y++ ) {
    float * row = reinterpret_cast<float*>(image.ptr(y));
    for ( int x = 0; x < image.cols; x++ ) {
      if ( row[x] < smin ) {
        row[x] = dmin;
      }
      else if ( row[x] >= smax ) {
        row[x] = dmax;
      }
      else {
        row[x] = (row[x] - smin) * (dmax - dmin) / (smax - smin);
      }
    }
  }
}



static void detect_blobs(Mat & image, std::vector<KeyPoint> & blobs)
{
  // Setup SimpleBlobDetector parameters.
  SimpleBlobDetector::Params params;

  // Change thresholds
  params.minThreshold = 10;
  params.maxThreshold = 50;
  params.thresholdStep = 10;
  params.minRepeatability = 3;
  params.minDistBetweenBlobs = 10;

  params.filterByArea = true;
  params.minArea = 4;
  params.maxArea = 250;

  params.filterByColor = true;
  params.blobColor = 255;

  params.filterByCircularity = true;
  params.minCircularity = 0.5;
  params.maxCircularity = 1;

  params.filterByConvexity = false;
  //params.minConvexity = 0.0;
  //params.maxConvexity = 1;

  params.filterByInertia = false;

  // Detect blobs.
#if CV_MAJOR_VERSION < 3   // If you are using OpenCV 2
  SimpleBlobDetector detector(params);
  detector.detect( im, keypoints);
#else
  cv::Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
  detector->detect(image, blobs );
#endif

//  // Draw detected blobs as red circles.
//  // DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures the size of the circle corresponds to the size of blob
//  Mat im_with_keypoints;
//  drawKeypoints( image, blobs, im_with_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
}








static int flat_field_blur_size = 100;
static int local_field_blur_size = 3;

static bool load_image(Mat & image, const string & fname)
{
  Mat fld;

  if ( !(image = imread(fname, IMREAD_UNCHANGED)).data ) {
    fprintf(stderr, "imread(%s) fails\n", fname.c_str());
    return false;
  }

  image.convertTo(image, CV_32FC1);

  if ( local_field_blur_size > 0 ) {
    blur(image, image, Size(local_field_blur_size, local_field_blur_size));
  }

  if ( flat_field_blur_size > 0 ) {
    blur(image, fld, Size(flat_field_blur_size, flat_field_blur_size));
    image /= fld;
  }

  return true;
}


void dump_hist(const char * fname, const Mat & image, double gmin, double gmax)
{
  const size_t numbins = 100;
  double bins[numbins] = { 0 };
  double p = 0, vlo, vhi;
  size_t lo = 0, hi = numbins - 1;
  double minval, maxval, range;

  FILE * fp = NULL;


  printf("=====================\n");

  minMaxLoc(image, &minval, &maxval);
  range = maxval - minval;
  printf("%s global: minVal=%g\tmaxVal=%g\trange=%g\n", fname, minval, maxval, range);

  minval = gmin;
  maxval = gmax;
  range = maxval - minval;
  calc_hist(image, bins, numbins, minval, maxval);

  for ( size_t i = 0; i < numbins; ++i ) {
    p += bins[i];
  }

  if ( !(fp = fopen(fname, "w")) ) {
    fprintf(stderr, "Fatal: can not open %s: %s\n", fname, strerror(errno));
    fp = stdout;
  }


  fprintf(fp, "BIN\tCENTER\tPOW\n");
  for ( size_t i = 0; i < numbins; ++i ) {
    bins[i] *= 100 / p;
    fprintf(fp, "%3zu\t%+12.3g\t%+12.3g\n", i, minval + i * range / numbins, bins[i]);
  }

  for( double s = 0; lo < numbins; ++lo ) {
    if ( (s += bins[lo]) > 1 ) {
      --lo;
      break;
    }
  }

  for( double s = 0; hi > 0; --hi ) {
    if ( (s += bins[hi]) > 1 ) {
      ++hi;
      break;
    }
  }

  vlo = minval + lo * range / numbins;
  vhi = minval + hi * range / numbins;
  printf("lo=%zu\thi=%zu\tvlo=%g\tvhi=%g\n", lo, hi, vlo, vhi);

  if ( fp && fp != stdout ) {
    fclose(fp);
  }


}

int main(int argc, char *argv[])
{
  const char * input_directory_name = NULL;
  const char * output_directory_name = "./output-images";
  char outfilename[PATH_MAX];

  vector<string> input_files;

  Mat prev, current, diff;
  double current_mean = 0, current_pwr = 0;

  size_t i, j;

  enum algo algo = algo_div;

  double gmin_sub = -0.08, gmax_sub = 0.08;
  double gmin_div = -0.1, gmax_div = 0.1;

  double gmin, gmax, gamma = 1;
  bool gmin_set = false, gmax_set = false;

  double alpha = 0.1;

  bool dump_histogram = false;



  for( int i = 1; i < argc; ++i ) {

    if ( strcmp(argv[i], "-help") == 0 || strcmp(argv[i], "--help") == 0 ) {
      printf("Usage:\n");
      printf("  opencv-sagai-test-1 <input-directory> [OPTIONS]\n\n");
      printf("OPTIONS:\n");
      printf("  -o <output-directory-for-processed-images> [%s] \n", output_directory_name);
      printf("  -g  dump internal histogram\n");
      printf("  gmin=<low-output-threshod> [sub=%g div=%g] \n", gmin_sub, gmin_div);
      printf("  gmax=<high-output-threshod> [sub=%g div=%g] \n", gmax_sub, gmax_div);
      printf("  gamma=<gamma-correction-of-output-frames> [%g] \n", gamma);
      printf("  alpha=<contrast-parameter> [%g]\n", alpha);
      printf("  b=<flat-field-blur-size> [%d] \n", flat_field_blur_size);
      printf("  f=<local-field-blur-size> [%d] \n", local_field_blur_size);
      printf("  cr=<central_region_radius> [%d] \n", central_region_radius);
      printf("  a=<sub|div> select 'frame difference' algorithm\n");

      return 0;
    }



    if ( strcmp(argv[i], "-o") == 0 ) {
      if ( ++i >= argc ) {
        fprintf(stderr, "Missing argument after %s\n", argv[i]);
        return 1;
      }

      output_directory_name = argv[i];
    }
    else if ( strcmp(argv[i], "-g") == 0 ) {
      dump_histogram = true;
    }
    else if ( strncmp(argv[i], "a=", 2) == 0 ) {
      if ( strcmp(argv[i] + 2, "sub") == 0 ) {
        algo = algo_sub;
      }
      else if ( strcmp(argv[i] + 2, "div") == 0 ) {
        algo = algo_div;
      }
      else {
        fprintf(stderr, "Invalid algorithm %s selected. Use one of a=sub or a=div\n", argv[i]);
        return 1;
      }
    }
    else if ( strncmp(argv[i], "gmin=", 5) == 0 ) {
      if ( sscanf(argv[i] + 5, "%lf", &gmin) != 1 ) {
        fprintf(stderr, "Syntax error on %s\n", argv[i]);
        return 1;
      }
      gmin_set = true;
    }
    else if ( strncmp(argv[i], "gmax=", 5) == 0 ) {
      if ( sscanf(argv[i] + 5, "%lf", &gmax) != 1 ) {
        fprintf(stderr, "Syntax error on %s\n", argv[i]);
        return 1;
      }
      gmax_set = true;
    }
    else if ( strncmp(argv[i], "gamma=", 6) == 0 ) {
      if ( sscanf(argv[i] + 6, "%lf", &gamma) != 1 ) {
        fprintf(stderr, "Syntax error on %s\n", argv[i]);
        return 1;
      }
    }
    else if ( strncmp(argv[i], "alpha=", 6) == 0 ) {
      if ( sscanf(argv[i] + 6, "%lf", &alpha) != 1 ) {
        fprintf(stderr, "Syntax error on %s\n", argv[i]);
        return 1;
      }
    }
    else if ( strncmp(argv[i], "b=", 2) == 0 ) {
      if ( sscanf(argv[i] + 2, "%d", &flat_field_blur_size) != 1 ) {
        fprintf(stderr, "Syntax error on %s\n", argv[i]);
        return 1;
      }
    }
    else if ( strncmp(argv[i], "f=", 2) == 0 ) {
      if ( sscanf(argv[i] + 2, "%d", &local_field_blur_size) != 1 ) {
        fprintf(stderr, "Syntax error on %s\n", argv[i]);
        return 1;
      }
    }
    else if ( strncmp(argv[i], "cr=", 3) == 0 ) {
      if ( sscanf(argv[i] + 3, "%d", &central_region_radius) != 1 ) {
        fprintf(stderr, "Syntax error on %s\n", argv[i]);
        return 1;
      }
    }
    else if ( !input_directory_name ) {
      input_directory_name = argv[i];
    }
    else {
      fprintf(stderr, "Invalid argument %s\n", argv[i]);
      return 1;
    }
  }


  if ( !gmin_set ) {
    switch ( algo ) {
    case algo_sub :
      gmin = gmin_sub;
      break;
    case algo_div :
      gmin = gmin_div;
      break;
    }
  }

  if ( !gmax_set ) {
    switch ( algo ) {
    case algo_sub :
      gmax = gmax_sub;
      break;
    case algo_div :
      gmax = gmax_div;
      break;
    }
  }


  if ( !input_directory_name ) {
    input_directory_name = ".";
  }

  if ( !create_path(output_directory_name)) {
    fprintf(stderr, "create_path(%s) fails: %s\n", output_directory_name, strerror(errno));
    return 1;
  }

  if ( !get_files(input_files, input_directory_name, ".tif") ) {
    fprintf(stderr, "get_files(%s) fails: %s\n", input_directory_name, strerror(errno));
    return 1;
  }


  sort(input_files.begin(), input_files.end(), less<string>());


  for ( j = 0, i = 0; i < input_files.size(); ++i ) {

    double mv = 0, pw = 0;
    bool process_blobs = true;
    std::vector<KeyPoint> blobs;

    if ( !load_image(current, input_files[i]) ) {
      continue;
    }

    get_energy_metrics(current, &current_mean, &current_pwr);
    //printf("%s: mean=%g pwr=%g\n", input_files[i].c_str(), current_mean, current_pwr);

    current -= current_mean;
    current /= current_pwr;

    if ( !prev.data ) {
      // this is fisrt image in sequence
      prev = current;
      continue;
    }

    // next image in sequence

    get_energy_metrics(current, &mv, &pw);
    printf("%s: current_mean=%g -> mv=%g\n", input_files[i].c_str(), current_mean, mv);

    if ( algo == algo_sub ) {
      diff = current * (1 + alpha) - prev;
    }
    else {
      // fixme: div not implemented yet
      diff = current * (1 + alpha) - prev;
    }

    prev = current;


    threshold_pixels(diff, gmin, gmax);

    if ( dump_histogram ) {
      sprintf(outfilename, "%s/frame%03zu.hist", output_directory_name, j);
      dump_hist(outfilename, diff, gmin, gmax);
    }


    if ( gamma == 1 ) {
      normalize_pixels(diff, gmin, gmax, 0, 255);
    }
    else {
      normalize_pixels(diff, gmin, gmax, 0, 1);
      pow(diff, gamma, diff);
      normalize_pixels(diff, 0, 1, 0, 255);
    }

    if ( process_blobs ) {

      Mat tmp = diff.clone();
      normalize_pixels(tmp, 0, 255, 0, 1);

      tmp -= mean(tmp);
      pow(tmp, 2, tmp);
      blur(tmp, tmp, Size(5, 5));
      normalize_pixels(tmp, 0, 1, 0, 255);
      tmp.convertTo(tmp, CV_8UC1);

      detect_blobs(tmp, blobs);

      if ( blobs.size() < 1 ) {
        fprintf(stderr, "No blobs detected\n");
      }
    }


    diff.convertTo(diff, CV_8UC1);

    if ( process_blobs && blobs.size() ) {
      //drawKeypoints(diff, blobs, diff, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
      for ( size_t i = 0; i < blobs.size(); ++i ) {
        circle(diff, blobs[i].pt, 15, Scalar(0, 0, 255), 2, LINE_4, 0);
      }
    }

    sprintf(outfilename, "%s/frame%03zu.tif", output_directory_name, j);
    if ( !imwrite(outfilename, diff) ) {
      fprintf(stderr, "fatal: imwrite(%s) fails\n", outfilename);
      return 1;
    }

    ++j;

  }



  return 0;
}


#if 0

// some unused but usefull routines
static void translate_image(const Mat & ref, Mat & image)
{
  // Define the motion model
  const int warp_mode = MOTION_TRANSLATION;

  // Set a 2x3 or 3x3 warp matrix depending on the motion model.
  Mat warp_matrix;

  // Initialize the matrix to identity
  if ( warp_mode == MOTION_HOMOGRAPHY ) {
    warp_matrix = Mat::eye(3, 3, CV_32F);
  }
  else {
    warp_matrix = Mat::eye(2, 3, CV_32F);
  }

  // Specify the number of iterations.
  int number_of_iterations = 5000;

  // Specify the threshold of the increment
  // in the correlation coefficient between two iterations
  double termination_eps = 1e-7;

  // Define termination criteria
  TermCriteria criteria(TermCriteria::COUNT + TermCriteria::EPS, number_of_iterations, termination_eps);

  // Run the ECC algorithm. The results are stored in warp_matrix.
  printf("C findTransformECC()\n");
  findTransformECC(ref, image, warp_matrix, warp_mode, criteria);
  printf("R findTransformECC()\n");

  if ( warp_mode != MOTION_HOMOGRAPHY ) {
    // Use warpAffine for Translation, Euclidean and Affine
    warpAffine(image, image, warp_matrix, image.size(), INTER_NEAREST + WARP_INVERSE_MAP); // INTER_LINEAR
  }
  else {
    // Use warpPerspective for Homography
    warpPerspective(image, image, warp_matrix, image.size(), INTER_LINEAR + WARP_INVERSE_MAP);
  }

}


#endif
