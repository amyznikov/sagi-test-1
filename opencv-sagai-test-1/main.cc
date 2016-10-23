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

static void calc_circle_stats(const Mat & image, int x0, int y0, int r, double * m, double * v)
{
  size_t n = 0;
  int xmin = x0 <= r ? 0 : x0 - r;
  int ymin = y0 <= r ? 0 : y0 - r;
  int xmax = x0 + r >= image.cols ? image.cols - 1 : x0 + r;
  int ymax = y0 + r >= image.rows ? image.rows - 1 : y0 + r;

  *m = 0, *v = 0;

  for ( int y = ymin; y <= ymax; ++y ) {
    const float * row = reinterpret_cast<const float*>(image.ptr(y));
    for ( int x = xmin; x <= xmax; ++x ) {
      if ( hypot(x - x0, y - y0) <= r ) {
        *m += row[x];
        *v += row[x] * row[x];
        ++n;
      }
    }
  }

  if ( n > 0 ) {
    *m /= n;
    *v = sqrt((*v / n) - (*m * *m));
  }
}


static void get_energy_metrics(const Mat & image, double * m, double * p)
{
  const double rr = min(min(central_region_radius, image.rows), image.cols);
  const double x0 = image.cols / 2, y0 = image.rows / 2;
  size_t n = 0;

  *p = 0, *m = 0;

  for ( int y = 0; y < image.rows; y++ ) {
    const float * row = reinterpret_cast<const float*>(image.ptr(y));
    for ( int x = 0; x < image.cols; x++ ) {
      if ( hypot(x - x0, y - y0) < rr ) {
        *m += row[x];
        *p += row[x] * row[x];
        ++n;
      }
    }
  }

  if ( n ) {
    *m /= n;
    *p = sqrt((*p / n) - (*m * *m));
  }
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
      if ( (row[x] = (row[x] - smin) * (dmax - dmin) / (smax - smin) + dmin) < dmin ) {
        row[x] = dmin;
      }
      else if ( row[x] >= dmax ) {
        row[x] = dmax;
      }
    }
  }
}



static void detect_blobs(Mat & image, std::vector<KeyPoint> & blobs)
{
  // Setup SimpleBlobDetector parameters.
  SimpleBlobDetector::Params params;

  // Set thresholds
  params.minThreshold = 0;
  params.maxThreshold = 255;
  params.thresholdStep = (params.maxThreshold - params.minThreshold);
  params.minRepeatability = 1;
  params.minDistBetweenBlobs = 3;

  params.filterByArea = true;
  params.minArea = 1;
  params.maxArea = 400;

  params.filterByColor = false;
  params.blobColor = 255;

  params.filterByCircularity = true;
  params.minCircularity = 0.7;
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

static void save_blobs(const char * fname, const vector<KeyPoint> & blobs )
{
  FILE * fp;
  int i = 0;

  if ( !(fp = fopen(fname, "w")) ) {
    fprintf(stderr, "Can not write %s: %s\n", fname, strerror(errno));
    goto end;
  }

  fprintf(fp, "IDX\tX\tY\tS\tA\tOCT\tCLS\n");
  for ( vector<KeyPoint>::const_iterator ii = blobs.begin(); ii != blobs.end(); ++ii, ++i ) {
    fprintf(fp, "%3d\t%8g\t%8g\t%8g\t%8g\t%3d\t%5d\n", i, ii->pt.x, ii->pt.y, ii->size, ii->angle, ii->octave, ii->class_id);
  }


end:

  if ( fp ) {
    fclose(fp);
  }
}







static bool load_image(Mat & image, const string & fname, int gbs, int lbs)
{
  Mat fld;

  if ( !(image = imread(fname, IMREAD_UNCHANGED)).data ) {
    fprintf(stderr, "imread(%s) fails\n", fname.c_str());
    return false;
  }

  image = image(Rect(32, 52, 960, 720));
  image.convertTo(image, CV_32FC1);

  if ( gbs > 0 ) {
    blur(image, fld, Size(gbs, gbs));
    image /= fld;
  }

  if ( lbs > 0 ) {
    blur(image, image, Size(lbs, lbs));
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


static void draw_blobs(Mat & img, const std::vector<KeyPoint> & blobs, const char * output_directory_name, int frame_index, bool draw=true)
{
  if ( draw ) {
    Scalar color1 = Scalar(0, 0, 0);
    Scalar color2 = Scalar(255, 255, 255);
    for ( size_t i = 0; i < blobs.size(); ++i ) {
      circle(img, blobs[i].pt, blobs[i].size, color1, 1, LINE_4, 0);
      circle(img, blobs[i].pt, blobs[i].size, color2, 1, LINE_4, 0);
    }
  }

  if ( output_directory_name ) {
    char outfilename[PATH_MAX];
    sprintf(outfilename, "%s/blobs.%03d.tif", output_directory_name, frame_index);
    if ( !imwrite(outfilename, img) ) {
      fprintf(stderr, "fatal: imwrite(%s) fails\n", outfilename);
    }
  }
}


static void variance_filter(Mat & src, Mat & dst)
{
  double r1 = 6;
  double r2 = 25;
  //double m1, v1, m2, v2;
  double minVal, maxVal;
  Mat sigma1, sigma2, I1, M1, I2, M2;

  static Mat ring1, ring2;

  if ( !ring1.data ) {
    int npix = 0;
    ring1 = Mat::zeros(r1 * 2 + 1, r1 * 2 + 1, CV_32FC1);
    for ( int y = 0; y < r1 * 2 + 1; ++y ) {
      float * row = reinterpret_cast<float*>(ring1.ptr(y));
      for ( int x = 0; x < r1 * 2 + 1; ++x ) {
        row[x] = hypot(x - r1, y - r1) <= r1;
        ++npix;
      }
    }
    ring1 /= npix;
  }

  if ( !ring2.data ) {
    int npix = 0;
    ring2 = Mat::zeros(r2 * 2 + 1, r2 * 2 + 1, CV_32FC1);
    for ( int y = 0; y < r2 * 2 + 1; ++y ) {
      float * row = reinterpret_cast<float*>(ring2.ptr(y));
      for ( int x = 0; x < r2 * 2 + 1; ++x ) {
        row[x] = hypot(x - r2, y - r2) <= r2;
        ++npix;
      }
    }
    ring2 /= npix;
  }

  //blur(src, src, Size(3, 3));
  //GaussianBlur(src, src, Size(0, 0), 2, 2);


  //GaussianBlur(src, M1, Size(0, 0), r1, r1);
  filter2D(src, M1, -1, ring1);
  //GaussianBlur(src.mul(src), I1, Size(0, 0), r1, r1);
  filter2D(src.mul(src), I1, -1, ring1);
  cv::sqrt(I1 - M1.mul(M1), sigma1);
  minMaxLoc(sigma1, &minVal, &maxVal);
  printf("SIGMA1: minVal=%g maxVal=%g\n", minVal, maxVal);

  //GaussianBlur(src, M2, Size(0, 0), r2, r2);
  filter2D(src, M2, -1, ring2);
  //GaussianBlur(src.mul(src), I2, Size(0, 0), r2, r2);
  filter2D(src.mul(src), I2, -1, ring2);
  cv::sqrt(I2 - M2.mul(M2), sigma2);
  //sigma2 = I2 - M2.mul(M2);
  minMaxLoc(sigma2, &minVal, &maxVal);
  printf("SIGMA2: minVal=%g maxVal=%g\n", minVal, maxVal);

  divide(sigma1, sigma2, dst);

  minMaxLoc(dst, &minVal, &maxVal);
  printf("VAR: minVal=%g maxVal=%g\n", minVal, maxVal);

  normalize_pixels(dst, 1.5, 3, 0, 255);
}



struct blob {
  double x, y, sx, sy, xmin, xmax, ymin, ymax, size;
  int n;
  blob(double x_, double y_, double sx_, double sy_, double size_, int n_)
      : x(x_), y(y_), sx(sx_), sy(sy_), xmin(x_), xmax(x_), ymin(y_), ymax(y_), size(size_), n(n_)
  {
  }
  blob(double x_, double y_, double size_)
      : x(x_), y(y_), sx(0), sy(0), xmin(x_), xmax(x_), ymin(y_), ymax(y_), size(size_), n(1)
  {
  }
};

typedef vector<blob>
  bloblist;


static bloblist::iterator find_blob(bloblist::iterator beg, bloblist::iterator end, double x, double y, double r)
{
  for ( ; beg != end; ++beg ) {
    if ( hypot(beg->x - x, beg->y - y) < r ) {
      break;
    }
  }
  return beg;
}


static void match_blobs(bloblist & blobs)
{
  int njoins = 0;

  do {
    njoins = 0;
    for ( bloblist::iterator b = blobs.begin(); b != blobs.end(); ++b ) {

      double x = b->x, y = b->y;
      double x2 = x * x, y2 = y * y;
      double xmin = b->xmin, xmax = b->xmax;
      double ymin = b->ymin, ymax = b->ymax;
      double size = b->size;
      double r = 10;
      int n = 1;

      bloblist::iterator match = b + 1;
      while ( (match = find_blob(match, blobs.end(), x, y, r)) != blobs.end() ) {
        x += match->x;
        y += match->y;
        x2 += match->x * match->x;
        y2 += match->y * match->y;

        if ( match->x < xmin ) {
          xmin = match->x;
        }
        else if ( match->x > xmax ) {
          xmax = match->x;
        }

        if ( match->y < ymin ) {
          ymin = match->y;
        }
        else if ( match->y > ymax ) {
          ymax = match->y;
        }

        if ( match->size > size ) {
          size = match->size;
        }

        ++n;
        match = blobs.erase(match);
      }

      if ( n > 1 ) {

        b->x = x / n;
        b->y = y / n;
        b->sx = sqrt(x2 / n - b->x * b->x);
        b->sy = sqrt(y2 / n - b->y * b->y);
        b->n += n - 1;
        b->xmin = xmin;
        b->xmax = xmax;
        b->ymin = ymin;
        b->ymax = ymax;
        b->size = size;

        ++njoins;
        //fprintf(stderr, "n=%d\n", n);
      }
    }

    fprintf(stderr, "njoins=%d\n", njoins);
    //usleep(100*1000);

  } while (njoins > 0 );


}


static bool detect_dust(const vector<string> & input_files, vector<KeyPoint> & gblobs, const char * output_directory_name)
{
  Mat prev, current, diff, var;
  double mv, pw;
  size_t i, n;

  bloblist common;

  for ( n = 0, i = 0; i < input_files.size(); ++i ) {

    vector<KeyPoint> blobs;
    bloblist tmp;


    if ( !load_image(current, input_files[i], 0, 0) ) {
      continue;
    }

    get_energy_metrics(current, &mv, &pw);
    current -= mv;
    current /= pw;

    if ( !prev.data ) {
      prev = current;
      continue;
    }

    diff = current - prev;
    prev = current;

    variance_filter(diff, var);
    var.convertTo(var, CV_8UC1);
    detect_blobs(var, blobs);
    draw_blobs(var, blobs, output_directory_name, n, false);

    for ( vector<KeyPoint>::const_iterator ii = blobs.begin(); ii != blobs.end(); ++ii ) {
      tmp.emplace_back(blob(ii->pt.x, ii->pt.y, ii->size));
    }

    match_blobs(tmp);
    for ( bloblist::iterator b = tmp.begin(); b != tmp.end(); ++b ) {
      b->n = 1;
    }
    common.insert(common.end(), tmp.begin(), tmp.end());



    ++n;
  }

  if ( n ) {
    match_blobs(common);

    for ( bloblist::iterator b = common.begin(); b != common.end(); ++b ) {
      b->sx = b->xmax - b->xmin;
      b->sy = b->ymax - b->ymin;
      if ( b->n > 2 && hypot(b->sx, b->sy) <= 6 ) {
        gblobs.emplace_back(b->x, b->y, b->size, -1, 0, b->n); // hypot(b->sx, b->sy)
      }
    }

    if ( 1 ) {

      char fname[PATH_MAX];

      sprintf(fname, "%s/blobs.txt", output_directory_name);

      FILE * fp = fopen(fname, "w");

      fprintf(fp, "X\tY\tSX\tSY\tS\tN\n");
      for ( bloblist::const_iterator b = common.begin(); b != common.end(); ++b ) {
        if ( b->n != 1 ) {
          fprintf(fp, "%8g\t%8g\t%8g\t%8g\t%8g\t%4d\n", b->x, b->y, b->sx, b->sy, hypot(b->sx, b->sy), b->n);
        }
      }
      fclose(fp);
    }



//
//    if ( 1 ) {
//
//      char fname[PATH_MAX];
//
//      sprintf(fname, "%s/blobs.txt", output_directory_name);
//
//      FILE * fp = fopen(fname, "w");
//
//      fprintf(fp, "X\tY\tS\tOCT\n");
//      for ( vector<KeyPoint>::iterator b = gblobs.begin(); b != gblobs.end(); ++b ) {
//        fprintf(fp, "%8g\t%8g\t%8g\t%4d\n", b->pt.x, b->pt.y, b->size, b->octave);
//      }
//      fclose(fp);
//    }

  }


  return n > 0;
}


static void save_normalized_image(const Mat & image, const char * output_directory_name, const char * prefix, int frame_index )
{
  Mat gray;
  char outfilename[PATH_MAX];

  //normalize_pixels(diff, gmin, gmax, 0, 255);
  image.convertTo(gray, CV_8UC1);

  sprintf(outfilename, "%s/%s.%03d.tif", output_directory_name, prefix, frame_index);
  if ( !imwrite(outfilename, gray) ) {
    fprintf(stderr, "fatal: imwrite(%s) fails\n", outfilename);
  }
}


int main(int argc, char *argv[])
{
  const char * input_directory_name = NULL;
  const char * output_directory_name = "./output-images";
  char outfilename[PATH_MAX];

  vector<string> input_files;

  Mat prev, current, diff, avg, bsimage;
  double mv = 0, pw = 0;

  size_t i, j;

  enum algo algo = algo_div;

  int flat_field_blur_size = 100;
  int local_field_blur_size = 3;

  double gmin_sub = -0.08, gmax_sub = 0.08;
  double gmin_div = -0.1, gmax_div = 0.1;

  double gmin, gmax, gamma = 1;
  bool gmin_set = false, gmax_set = false;

  double alpha = 0.1;

  bool dump_histogram = false;

  std::vector<KeyPoint> gblobs;



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

  if ( !detect_dust(input_files, gblobs, output_directory_name) ) {
    fprintf(stderr, "detect_dust(%s) fails\n", input_directory_name);
    return 1;
  }

  fprintf(stderr, "GLOBAL BLOBS: %zu detections\n", gblobs.size());
  //return 0;


  for ( j = 0, i = 0; i < input_files.size(); ++i ) {

    Mat var;

    std::vector<KeyPoint> blobs;

    if ( !load_image(current, input_files[i], flat_field_blur_size, local_field_blur_size ) ) {
      continue;
    }

    get_energy_metrics(current, &mv, &pw);
    current -= mv;
    current /= pw;

    if ( !prev.data ) {
      prev = current;
      continue;
    }

//    diff = current - prev;
//    variance_filter(diff, var);
//    var.convertTo(bsimage, CV_8UC1);
//    detect_blobs(bsimage, blobs);
//    draw_blobs(bsimage, blobs, output_directory_name, j);
//    save_normalized_image(bsimage, output_directory_name, "var", j );



    if ( algo == algo_sub ) {
      diff = current * (1 + alpha) - prev;
    }
    else {
      // fixme: div not implemented yet
      diff = current * (1 + alpha)/prev;
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

    if ( true ) {
      for ( std::vector<KeyPoint>::const_iterator ii =  gblobs.begin(); ii != gblobs.end(); ++ii ) {

        Rect rc(
            std::max((int) (ii->pt.x - ii->size ), 0),
            std::max((int) (ii->pt.y - ii->size ), 0),
            std::min((int)(2*ii->size), (int)(diff.cols-ii->pt.x-1)),
            std::min((int)(2*ii->size), (int)(diff.rows-ii->pt.y-1))
        );

        Scalar m = mean(diff(rc));
        //randn(diff(rc), m, Scalar::all(100));
        randu(diff(rc), m - Scalar::all(70), m + Scalar::all(70));
        GaussianBlur(diff(rc), diff(rc), Size(0,0), 0.66, 0.66);
      }
    }


    diff.convertTo(diff, CV_8UC1);

    if ( false ) {
      draw_blobs(diff, gblobs, NULL, -10 );
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


static void match_blob(blob & b, vector<bloblist>::iterator frame, vector<bloblist>::iterator endframe)
{
  if ( frame != endframe ) {
    double x = b.x;
    double y = b.y;

    bloblist::iterator mb = find_blob(frame->begin(), frame->end(), x, y);
    while ( mb != frame->end() ) {

      match_blob(*mb, frame + 1, endframe);
      b.x += mb->x;
      b.y += mb->y;
      b.sx += mb->x * mb->x;
      b.sy += mb->y * mb->y;
      b.n += mb->n;

      frame->erase(mb);
      mb = find_blob(frame->begin(), frame->end(), x, y);
    }
  }
}


static void match_blobs(vector<bloblist > & frames, bloblist & common)
{
  double x0, y0, sx, sy, dr;
  for ( vector<bloblist >::iterator frame = frames.begin(); frame != frames.end(); ++frame ) {
    for ( bloblist::iterator b = frame->begin(); b != frame->end(); ++b ) {
      match_blob(*b, frame + 1, frames.end());

      x0 = b->x / b->n;
      y0 = b->y / b->n;

      if ( b->n > 1 ) {
        sx = sqrt(b->sx / b->n - x0 * x0);
        sy = sqrt(b->sy / b->n - y0 * y0);
      }
      else {
        sx = sy = 0;
      }

      common.emplace_back(blob(x0, y0, sx, sy, b->n));
    }
  }
}


#endif
