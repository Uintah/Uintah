
// macros.h
// david hart
// jan 1998

#ifndef __MACROS_H__
#define __MACROS_H__

#include <math.h>
#include <float.h>
#include <string.h>
#include <stdlib.h>

#include <iostream>
using std::cerr;
using std::endl;
using std::ostream;

#include <values.h>
#include <ctype.h>

#ifdef _WIN32
#else
#include <unistd.h>
#endif

/*
#ifndef bool
#define bool int
#endif

#ifndef true
#define true 1
#endif

#ifndef false
#define false 0
#endif
*/

//extern int strcasecmp(char* s1, char* s2);

//----------------------------------------------------------------------
// MACROS & CONSTANTS
//----------------------------------------------------------------------

//#ifdef _DEBUG
#define DEBUG(variable) cerr << #variable << " = "\
<< variable << ", " << __FILE__ << ":" << __LINE__ << endl
//#else
//#define DEBUG(variable) {}
//#endif

const double EPSILON3	= 5e-2;
const double EPSILON6	= 5e-5;
const double EPSILON9	= 5e-8;
const double EPSILON12	= 5e-11;

const double DINFINITY	= DBL_MAX;
const float  FINFINITY	= FLT_MAX;
const double DNEGINFINITY	= DBL_MIN;
const float  FNEGINFINITY	= FLT_MIN;

const double RMAX = (double)(((int)RAND_MAX) + 1);

#ifndef		M_PI
const double	M_PI	= 3.1415926535897932384626433832795; // ;)
#endif

//----------------------------------------------------------------------
// THE FUNCTIONS FORMERLY KNOWN AS MACROS
//----------------------------------------------------------------------

// RAD2DEG -------------------------------------------------------------
template <class T>
inline T RAD2DEG(T x) {
  return (x*180.0/M_PI);
}

// DEG2RAD -------------------------------------------------------------
template <class T>
inline T DEG2RAD(T x) {
  return (M_PI*x/180.0);
}

// SWAP ----------------------------------------------------------------
template <class T>
inline void SWAP(T& a, T& b) {
  T temp;
  temp = a;
  a = b;
  b = temp;
}

// MAX -----------------------------------------------------------------
template <class T>
inline T MAX(T a, T b) { 
  return ((a < b) ? b : a); 
}

// WHICHMAX ------------------------------------------------------------
template <class T>
inline int WHICHMAX(T a, T b) {
  return ((a < b) ? 1 : 0);
}

// MIN -----------------------------------------------------------------
template <class T>
inline T MIN(T a, T b) { 
  return ((a > b) ? b : a);
}

// WHICHMIN ------------------------------------------------------------
template <class T>
inline int WHICHMIN(T a, T b) {
  return ((a > b) ? 1 : 0);
}

// MAX -----------------------------------------------------------------
template <class T>
inline T MAX(T a, T b, T c) {
  if (a > b && a > c) return a;
  else if (b > c) return b;
  else return c;
}

// MAX3 ----------------------------------------------------------------
template <class T>
inline T MAX3(T a, T b, T c) {
  if (a > b && a > c) return a;
  else if (b > c) return b;
  else return c;
}

// WHICHMAX3 -----------------------------------------------------------
template <class T>
inline int WHICHMAX3(T a, T b, T c) {
  if (a > b && a > c) return 0;
  else if (b > c) return 1;
  else return 2;
}

// WHICHMAX ------------------------------------------------------------
template <class T>
inline int WHICHMAX(T a, T b, T c) {
  if (a > b && a > c) return 0;
  else if (b > c) return 1;
  else return 2;
}

// MIN3 ----------------------------------------------------------------
template <class T>
inline T MIN3(T a, T b, T c) {
  if (a < b && a < c) return a;
  else if (b < c) return b;
  else return c;
}

// MIN -----------------------------------------------------------------
template <class T>
inline T MIN(T a, T b, T c) {
  if (a < b && a < c) return a;
  else if (b < c) return b;
  else return c;
}

// WHICHMIN3 -----------------------------------------------------------
template <class T>
inline int WHICHMIN3(T a, T b, T c) {
  if (a < b && a < c) return 0;
  else if (b < c) return 1;
  else return 2;
}

// WHICHMIN ------------------------------------------------------------
template <class T>
inline int WHICHMIN(T a, T b, T c) {
  if (a < b && a < c) return 0;
  else if (b < c) return 1;
  else return 2;
}

// CLAMP ---------------------------------------------------------------
template <class T>
inline T CLAMP(T a, T min, T max) {
  return MIN(max, MAX(min, a));
}

// ABS -----------------------------------------------------------------
template <class T>
inline T ABS(T x) {
  return ((x >= 0.0) ? x : -x);
}

// SIGN ----------------------------------------------------------------
template <class T>
inline T SIGN(T x) { 
  return ((x >= 0.0) ? 1.0 : -1.0);
}

// SIGN ----------------------------------------------------------------
template <class T>
inline T SIGN(T a, T b) {
  return (b >= 0.0 ? ABS(a) : -ABS(a));
}

// ROUND ---------------------------------------------------------------
template <class T>
inline T ROUND(T x) {
  return (((x - floor(x)) >= 0.5) ? ceil(x) : floor(x));
}

#ifdef _WIN32
extern "C" {
// sgenrand ------------------------------------------------------------
void sgenrand(unsigned long seed);

// genrand -------------------------------------------------------------
double genrand();
}
#endif

// sRANDOM ------------------------------------------------------------
inline void sRANDOM(unsigned int seed) {
#ifdef _WIN32
  sgenrand(seed);
#else
  srand48((long)seed);
#endif
}

// dRANDOM -------------------------------------------------------------
inline double dRANDOM() {
#ifdef _WIN32
  return genrand();
#else
  return drand48();
#endif
}

// fRANDOM -------------------------------------------------------------
inline float fRANDOM() {
  return (float)dRANDOM();
}

// iRANDOM -------------------------------------------------------------
inline int iRANDOM() {
  return (int)(dRANDOM() * MAXINT);
}

// SQR -----------------------------------------------------------------
template <class T>
inline T SQR(T x) {
  return (x*x);
}

// NEAR ----------------------------------------------------------------
template <class T>
inline int NEAR(T a, T b) {
  return (ABS(a-b) < EPSILON3);
}

// CLOSE ---------------------------------------------------------------
template <class T>
inline int CLOSE(T a, T b) { 
  return (ABS(a-b) < EPSILON6);
}

// VERYCLOSE -----------------------------------------------------------
template <class T>
inline int VERYCLOSE(T a, T b) { 
  return (ABS(a-b) < EPSILON9);
}

// EXTREMELYCLOSE ------------------------------------------------------
template <class T>
inline int EXTREMELYCLOSE(T a, T b) {
  return (ABS(a-b) < EPSILON12);
}

// STRCASECMP ----------------------------------------------------------
/*
int strcasecmp(char* s1, char* s2) {
	return _strcasecmp(s1, s2);
}
*/

// STRCMP --------------------------------------------------------------
inline int STRCMP(const char* str1, const char* str2) {
#ifndef stricmp
  return(strcasecmp(str1, str2));
#else
  return(stricmp(str1, str2));
#endif
}

// EQUAL ---------------------------------------------------------------
// case INsensitive string comparison
inline int EQUAL(const char* str1, const char* str2) {
  return(STRCMP(str1, str2) == 0);
}

// ISNUMERIC -------------------------------------------------------------
inline int ISNUMERIC(char c) {
				// the '/' character is a hack for
				// allowing parsing .obj file
				// formats.  i will remove this
				// later.
  return (c >= '0' && c <= '9' || c == '-' ||
	  c == 'e' || c == '+' || c == '.' || c == '/');
}

// ISLOWER -------------------------------------------------------------
inline int ISLOWER(char c) {
  return (c >= 'a' && c <= 'z');
}

// ISUPPER -------------------------------------------------------------
inline int ISUPPER(char c) {
  return (c >= 'A' && c <= 'Z');
}

// ISALPHA ---------------------------------------------------------------
inline int ISALPHA(char c) {
  return (ISLOWER(c) || ISUPPER(c));
}

// ISALPHANUMERIC --------------------------------------------------------
inline int ISALPHANUMERIC(char c) {
  return (ISALPHA(c) || ISNUMERIC(c));
}

// LENGTH --------------------------------------------------------------
inline int LENGTH(char* str) { 
  return strlen(str);
}

// ISNUMERIC -------------------------------------------------------------
inline int ISNUMERIC(char* str) {
  for (int i = 0; i < LENGTH(str); i++) {
    if (!ISNUMERIC(str[i])) return 0;
  }
  return 1;
}

// ISALPHA ---------------------------------------------------------------
inline int ISALPHA(char* str) {
  for (int i = 0; i < LENGTH(str); i++) {
    if (!ISALPHA(str[i])) return 0;
  }
  return 1;
}

// ISALPHANUMERIC --------------------------------------------------------
inline int ISALPHANUMERIC(char* str) {
  for (int i = 0; i < LENGTH(str); i++) {
    if (!ISALPHANUMERIC(str[i])) return 0;
  }
  return 1;
}

// LOWER ---------------------------------------------------------------
inline char LOWER(char c) {
  if (ISUPPER(c)) return (c - 'A' + 'a');
  else return c;
}

// UPPER ---------------------------------------------------------------
inline char UPPER(char c) {
  if (ISLOWER(c)) return (c - 'a' + 'A');
  else return c;
}

// LOWER ---------------------------------------------------------------
inline void LOWER(char* str) {
  int n = LENGTH(str);
  for (int i = 0; i < n; i++) {
    str[i] = LOWER(str[i]);
  }
}

// UPPER ---------------------------------------------------------------
inline void UPPER(char* str) {
  int n = LENGTH(str);
  for (int i = 0; i < n; i++) {
    str[i] = UPPER(str[i]);
  }
}

// EVEN ----------------------------------------------------------------
inline int EVEN(int n) {
  return ((n % 2) == 0);
}

// ODD -----------------------------------------------------------------
inline int ODD(int n) {
  return ((n % 2) == 1);
}

// BINSEARCH -----------------------------------------------------------
				// binary search -- returns the bin
				// number of the greatest value less
				// than val

				// works on class defined with <, >,
				// and == operators.
template <class T>
inline int BINSEARCH(T& val, T* array, int n) {
  int l=0, r=n-1, i;
  if (array == NULL || n == 0 || val < array[0]) return -1;
  if (val > array[n-1]) return n;
  if (val == array[n-1]) return n-1;
  while (r > l+1) {
    i = (l+r)/2;
    if (val < array[i]) r = i;
    else l = i;
  }
  return l;
}

// BINSEARCH -----------------------------------------------------------
				// binary search -- returns the bin
				// number of the greatest value less
				// than val

				// works on class defined with <, >,
				// and == operators.
template <class T>
inline int BINSEARCHPTR(T* val, T** array, int n) {
  int l=0, r=n-1, i;
  if (array == NULL || n == 0 || (*val) < (*(array[0]))) return -1;
  if ((*val) > (*(array[n-1]))) return n;
  if ((*val) == (*(array[n-1]))) return n-1;
  while (r > l+1) {
    i = (l+r)/2;
    if ((*val) < (*(array[i]))) r = i;
    else l = i;
  }
  return l;
}

//----------------------------------------------------------------------
template <class T>
inline T LERP(const double x, const T* v) {
  return (v[0]*(1.0-x) + v[1]*x);
}

//----------------------------------------------------------------------
template <class T>
inline T LERP(const double x, const T& v1, const T& v2) {
  return (v1*(1.0-x) + v2*x);
}

//----------------------------------------------------------------------
template <class T>
inline T LINEARINTERPOLATE(const double x, const T* v) {
  return (v[0]*(1.0-x) + v[1]*x);
}

//----------------------------------------------------------------------
template <class T>
inline T BILINEARINTERPOLATE(double x, double y, T* v[2]) {
  T w[2];
  w[0] = LINEARINTERPOLATE(x, v[0]);
  w[1] = LINEARINTERPOLATE(x, v[1]);
  return LINEARINTERPOLATE(y, w);
}

//----------------------------------------------------------------------
template <class T>
inline T TRILINEARINTERPOLATE(double x, double y, double z, T** v[2]) {
  T w[2];
  w[0] = BILINEARINTERPOLATE(x, y, v[0]);
  w[1] = BILINEARINTERPOLATE(x, y, v[1]);
  return LINEARINTERPOLATE(z, w);
}

//----------------------------------------------------------------------
template <class T>
inline T CUBICINTERPOLATE(double x, T* v) {
  double xx[4];
  xx[0] = x+1.0;
  xx[1] = x+0.0;
  xx[2] = x-1.0;
  xx[3] = x-2.0;
  return (
    v[0]*((-1.0/6.0)*xx[1]*xx[2]*xx[3]) +
    v[1]*(( 1.0/2.0)*xx[0]*xx[2]*xx[3]) +
    v[2]*((-1.0/2.0)*xx[0]*xx[1]*xx[3]) +
    v[3]*(( 1.0/6.0)*xx[0]*xx[1]*xx[2])
    );
}

//----------------------------------------------------------------------
template <class T>
inline T BICUBICINTERPOLATE(double x, double y, T* v[4]) {
  T w[4];
  w[0] = CUBICINTERPOLATE(x, v[0]);
  w[1] = CUBICINTERPOLATE(x, v[1]);
  w[2] = CUBICINTERPOLATE(x, v[2]);
  w[3] = CUBICINTERPOLATE(x, v[3]);
  return CUBICINTERPOLATE(y, w);
}

//----------------------------------------------------------------------
template <class T>
inline T TRICUBICINTERPOLATE(double x, double y, double z, T** v[4]) {
  T w[4];
  w[0] = BICUBICINTERPOLATE(x, y, v[0]);
  w[1] = BICUBICINTERPOLATE(x, y, v[1]);
  w[2] = BICUBICINTERPOLATE(x, y, v[2]);
  w[3] = BICUBICINTERPOLATE(x, y, v[3]);
  return CUBICINTERPOLATE(z, w);
}

//----------------------------------------------------------------------
template <class T>
inline int RLE_ENCODE(T* input, int input_len, T* rle_output,
  T run_code, T code_replace, int max_run_length, int input_delta=1) {
  
  T* left;
  T* right;
  T* rle_loc;
  int run_length;
  
  for (left = input, rle_loc = rle_output;
       (left-input)/input_delta < input_len; left = right) {
    for (right = left+input_delta, run_length=1;
	 (right-input)/input_delta < input_len &&
	   run_length < max_run_length &&
	   *left == *right;
	 right += input_delta, run_length++) {
      // do nothing - skip over run
    }
				// encode run -- but only of length 3+
    if (run_length == 1) {
      *(rle_loc++) = (*left == run_code) ? code_replace : *left;
    }
    else if (run_length == 2) {      
      *(rle_loc++) = (*left == run_code) ? code_replace : *left;
      *(rle_loc++) = (*(left+input_delta) == run_code) ? code_replace : *left;
    }
    else /* if (run_length > 2) */ {
      *(rle_loc++) = run_code;
      *(rle_loc++) = T(run_length);
      *(rle_loc++) = *left;
    }
  }
				// return left of encoded array
  return (rle_loc - rle_output);
}

//----------------------------------------------------------------------
template <class T>
inline int RLE_DECODE(T* rle_input, int input_len, T* output,
  T run_code, int output_delta=1) {

  T* rle_loc;
  T* output_loc;
  int i;

  for (rle_loc = rle_input, output_loc = output;
       (rle_loc-rle_input) < input_len; ) {
    if (rle_loc[0] == run_code) {
      for (i = 0; i < int(rle_loc[1]); i++) {
	*output_loc = rle_loc[2];
	output_loc += output_delta;
      }
      rle_loc += 3;
    }
    else {
      *output_loc = *(rle_loc++);
      output_loc += output_delta;
    }
  }
				// return length of decoded array
  return (output_loc - output);
}

//----------------------------------------------------------------------
template <class T>
inline int PERMUTE(T* array, int len) {
  int i, j, k;
  
                                // find the largest k such that p[k] <
                                // p[k+1].  if no such k exists, stop.
  for (k = -1, i = 0; i < len-1; i++)
    if (array[i] < array[i+1]) k = i;
  if (k == -1) return 0;
  
                                // find the smallest entry p[k+j] to
                                // the right of p[k] that is larger
                                // than p[k].
  for (j = k+1, i = j+1; i < len; i++)
    if (array[i] < array[j] && array[i] > array[k]) j = i;

                                // interchange array[k] and array[k+j]
  SWAP(array[k], array[j]);

                                // reverse the order of array[k+1] array[k+2]
                                // ... array[n]
  for (i = k+1, j = len-1; i <= j; i++, j--) {
    SWAP(array[i], array[j]);
  }

  return 1;
}

//----------------------------------------------------------------------
template <class T>
int lt(const void* c1, const void* c2) {
  return (*((T*)c1) < *((T*)c2)) ? (-1) :
    ((*((T*)c1) > *((T*)c2)) ? (1) : (0));
}

//----------------------------------------------------------------------
template <class T>
int gt(const void* c1, const void* c2) {
  return (*((T*)c1) > *((T*)c2)) ? (-1) :
    ((*((T*)c1) < *((T*)c2)) ? (1) : (0));
}

// SORT ----------------------------------------------------------------
template <class T>
inline void SORT(T* array, int len) {
  qsort(array, len, sizeof(T), lt<T>);
}

// SORTUP --------------------------------------------------------------
template <class T>
inline void SORTUP(T* array, int len) {
  qsort(array, len, sizeof(T), lt<T>);
}

// SORTDOWN ------------------------------------------------------------
template <class T>
inline void SORTDOWN(T* array, int len) {
  qsort(array, len, sizeof(T), gt<T>);
}

//----------------------------------------------------------------------
template <class T>
inline int PERMUTE_SKIP(T* array, int len, int skip_loc) {
  skip_loc = MAX(0, MIN(len-1, skip_loc));
  SORTDOWN(&array[skip_loc+1], len-skip_loc-1);
  return PERMUTE(array, len);
}

//----------------------------------------------------------------------
inline void
SLEEP(float seconds) {
#ifdef _WIN32
  crapcrapcrap
#else
  sleep(int(floor(seconds)));
  usleep(int(1000000.0 * (seconds-floor(seconds))));
#endif
}

//----------------------------------------------------------------------

#endif
