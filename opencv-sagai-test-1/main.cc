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

static void get_energy_metrics(const Mat & image, double * pwr, double * mean)
{
  const double rr = min(min(central_region_radius, image.rows), image.cols);
  const double x0 = image.cols / 2, y0 = image.rows / 2;

  *pwr = 0, * mean = 0;

  size_t npix = 0;

  for ( int y = 0; y < image.rows; y++ ) {
    const float * row = reinterpret_cast<const float*>(image.ptr(y));
    for ( int x = 0; x < image.cols; x++ ) {
      if ( hypot(x - x0, y - y0) < rr ) {
        * pwr += row[x] * row[x];
        * mean += row[x];
        ++npix;
      }
    }
  }

  *mean /= npix;
  *pwr = sqrt(*pwr);
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


static int global_field_blur_size = 100;

static bool load_image(Mat & image, const string & fname, double * pwr)
{
  Mat fld, tmp;
  double mv;

  if ( !(image = imread(fname, IMREAD_UNCHANGED)).data ) {
    fprintf(stderr, "imread(%s) fails\n", fname.c_str());
    return false;
  }

  image.convertTo(image, CV_32FC1);

  //blur(image, image, Size(1, 1));
  blur(image, fld, Size(global_field_blur_size, global_field_blur_size));

  get_energy_metrics(image, pwr, &mv);
  (image -= mv) /= fld;
  get_energy_metrics(image, pwr, &mv);

  printf("%s\n", fname.c_str());

  return true;
}


void dump_hist(const Mat & image)
{
  const size_t numbins = 256;
  double bins[numbins] = { 0 };
  double p = 0, vlo, vhi;
  size_t lo = 0, hi = numbins - 1;
  double minval, maxval, range;

  printf("=====================\n");

  minMaxLoc(image, &minval, &maxval);

  range = maxval - minval;
  //printf("minVal=%g\tmaxVal=%g\trange=%g\n", minval, maxval, range);

  calc_hist(image, bins, numbins, minval, maxval);

  for ( size_t i = 0; i < numbins; ++i ) {
    p += bins[i];
  }

  for ( size_t i = 0; i < numbins; ++i ) {
    bins[i] *= 100 / p;
    printf("%3zu\t%+12.3g\t%+12.3g\n", i, minval + i * range / numbins, bins[i]);
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

}

int main(int argc, char *argv[])
{
  const char * input_directory_name = NULL;
  const char * output_directory_name = "./output-images";
  char outfilename[PATH_MAX];

  vector<string> input_files;

  Mat prev, current, diff;
  double first_pwr, current_pwr;

  size_t i, j;

  double gmin = -0.08, gmax = 0.08;
  bool dump_histogram = false;

  for( int i = 1; i < argc; ++i ) {

    if ( strcmp(argv[i], "-help") == 0 || strcmp(argv[i], "--help") == 0 ) {
      printf("Usage:\n");
      printf("  opencv-sagai-test-1 <input-directory> [OPTIONS]\n\n");
      printf("OPTIONS:\n");
      printf("  -o <output-directory-for-processed-images> [%s] \n", output_directory_name);
      printf("  -g  dump internal histogram\n");
      printf("  gmin=<low-output-threshod> [%g] \n", gmin);
      printf("  gmax=<high-output-threshod> [%g] \n", gmax);
      printf("  b=<global-field-blur-size> [%d] \n", global_field_blur_size);
      printf("  cr=<central_region_radius> [%d] \n", central_region_radius);


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
    else if ( strncmp(argv[i], "gmin=", 5) == 0 ) {
      if ( sscanf(argv[i] + 5, "%lf", &gmin) != 1 ) {
        fprintf(stderr, "Syntax error on %s\n", argv[i]);
        return 1;
      }
    }
    else if ( strncmp(argv[i], "gmax=", 5) == 0 ) {
      if ( sscanf(argv[i] + 5, "%lf", &gmax) != 1 ) {
        fprintf(stderr, "Syntax error on %s\n", argv[i]);
        return 1;
      }
    }
    else if ( strncmp(argv[i], "b=", 2) == 0 ) {
      if ( sscanf(argv[i] + 2, "%d", &global_field_blur_size) != 1 ) {
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


  if ( input_files.size() < 2 ) {
    fprintf(stderr, "at least 2 input images need to create a differential movie, only %zu found\n", input_files.size());
    return 1;
  }

  sort(input_files.begin(), input_files.end(), less<string>());


  i = 0;
  while ( i < input_files.size() && !load_image(prev, input_files[i], &first_pwr) ) {
    ++i;
  }

  if ( i == input_files.size() ) {
    fprintf(stderr, "fatal: at least 2 input images need to create a differential movie\n");
    return 1;
  }

  // fisrt = prev.clone();

  for ( j = 0, ++i; i < input_files.size(); ++i ) {

    while ( i < input_files.size() && !load_image(current, input_files[i],  &current_pwr) ) {
      ++i;
    }

    if ( current.data ) {

      diff = current * (first_pwr / current_pwr) - prev;
      //diff = current - prev;
      prev = current;

      diff -= mean(diff);

      if ( dump_histogram ) {
        dump_hist(diff);
      }

      threshold_pixels(diff, gmin, gmax);

      sprintf(outfilename, "%s/frame%03zu.tif", output_directory_name, j);

      normalize(diff, diff, 0, 255, NORM_MINMAX);
      diff.convertTo(diff, CV_8UC1);

      if ( !imwrite(outfilename, diff) ) {
        fprintf(stderr, "fatal: imwrite(%s) fails\n",outfilename);
        return 1;
      }

      ++j;
    }

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
