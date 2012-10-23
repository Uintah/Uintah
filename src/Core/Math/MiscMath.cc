/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


/*
 *  MiscMath.cc
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   June 2004
 *
 */

#include <iostream>
using namespace std;
#include <cstdio>
#include <Core/Math/MiscMath.h>
#include <Core/Math/Expon.h>
#include <cmath>

#ifdef __digital__
#  include <fp_class.h>
#endif
#ifdef _WIN32
#  include <cfloat>
#  define finite _finite
#endif

namespace SCIRun {

double MakeReal(double value)
{
  if (!finite(value))
  {
    int is_inf = 0;
#ifdef __digital__
    // on dec, we have fp_class which can tell us if a number is +/- infinity
    int fpclass = fp_class(value);
    if (fpclass == FP_POS_INF) is_inf = 1;
    if (fpclass == FP_NEG_INF) is_inf = -1;
#elif defined(__sgi)
    fpclass_t c = fpclass(value);
    if (c == FP_PINF) is_inf = 1;
    if (c == FP_NINF) is_inf = -1;
#elif defined(_WIN32)
    int c = _fpclass(value);
    if (c == _FPCLASS_PINF) is_inf = 1;
    if (c == _FPCLASS_NINF) is_inf = -1;
#else
#  if defined( REDSTORM ) || defined( _AIX ) || defined( __PGI )
    is_inf  = isinf(value);
#  else
    is_inf  = std::isinf(value);
#  endif
#endif
    if (is_inf == 1) value = (double)(0x7fefffffffffffffULL);
    if (is_inf == -1) value = (double)(0x0010000000000000ULL);
    else value = 0.0; // Assumed NaN
  }
  return value;
}

void findFactorsNearRoot(const int value, int &factor1, int &factor2) {
  int f1,f2;
  f1 = f2 = (int) Sqrt((double)value);
  // now we are basically looking for a pair of multiples that are closest to
  // the square root.
  while ((f1 * f2) != value) {
    // look for another multiple
    for(int i = f1+1; i <= value; i++) {
      if (value%i == 0) {
        // we've found a root
        f1 = i;
        f2 = value/f1;
        break;
      }
    }
  }
  factor1 = f1;
  factor2 = f2;
}

//Computes the cubed root of a using Halley's algorithm.  The initial guess
//is set to the parameter provided, if no parameter is provided then the 
//result of the last call is used as the initial guess.  INT_MIN is used
//as a sentinal to signal that no guess was provided.
double cubeRoot(double a, double guess)
{
  double xold;
  static double xnew=1;   //start initial guess at last answer
  double small_num=5*DBL_EPSILON;

  if(guess!=DBL_MIN)
    xnew=guess;

  double atimes2=a*2.0;
  double x3;
  double diff=DBL_MAX, last_diff;
   
  //do a couple of iterations without checking the convergence criteria
  xold=xnew;
  x3=xold*xold*xold;
  xnew=xold*(x3+atimes2)/(2*x3+a);

  xold=xnew;
  x3=xold*xold*xold;
  xnew=xold*(x3+atimes2)/(2*x3+a);

  xold=xnew;
  x3=xold*xold*xold;
  xnew=xold*(x3+atimes2)/(2*x3+a);

  diff=fabs(xold-xnew);

  do
  {
    xold=xnew;
    x3=xold*xold*xold;
    xnew=xold*(x3+atimes2)/(2*x3+a);

    last_diff=diff;
    diff=fabs(xold-xnew);
    //terminate if the /*absolute or*/ relative difference is less than machine precision or if the convergence has stopped
  } while(/*diff>small_num &&*/ diff>small_num*xold && last_diff>diff);
  
  if(last_diff<diff)
    return xold;
  else
    return xnew;
}

} // End namespace SCIRun

