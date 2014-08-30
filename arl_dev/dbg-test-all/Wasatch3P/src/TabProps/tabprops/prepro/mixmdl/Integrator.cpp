/*
 * Copyright (c) 2014 The University of Utah
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

#include <cmath>
#include <cassert>

#include <tabprops/prepro/mixmdl/Integrator.h>

using namespace std;

//#define INTDEBUG
#ifdef INTDEBUG
#include <iostream>
#endif

//==============================================================================

Integrator::Integrator( const double lo, const double hi )
{
  loBound_ = lo;
  hiBound_ = hi;

  func_ = NULL;

  absErrTol_ = 1.0e-8;
  relErrTol_ = 1.0e-5;

  singul_.clear();
}

//====================================================================

double BasicIntegrator::integrate()
{
  assert( NULL != func_ );

  double integral = 0.0;
  nMaxEvals = 0;
  npts = nCoarse;

  double inc = (hiBound_-loBound_)/double(nCoarse-1);
  double a = loBound_;
  double b = a+inc;

  for( int ii=0; ii<nCoarse-1; ii++ ){
    double fa   = (*func_)( a );
    double fb   = (*func_)( b );
    double mid  = 0.5*(a+b);
    double fmid = (*func_)( mid );

    // compute the integral on this interval
    double h = 0.5*(b-a);
    double I = simpson( fa, fmid, fb, h );

    integral += refine( a, b, fa, fb, fmid, I, 0 );

    // move to next interval
    a = b;
    b = a+inc;
  }

#ifdef INTDEBUG
  cout << npts << " points required to achieve tolerance of " << absErrTol_ << endl;
  if ( nMaxEvals > 0 )
    cout << "  MAX recursion level reached " << nMaxEvals << " times" << endl;
#endif

  return integral;
}
//--------------------------------------------------------------------
double
BasicIntegrator::refine( const double a,  const double b,
			 const double fa, const double fb,
			 const double fmid,
			 const double S_whole,
			 int lev )
{
  npts += 4;

  const double h   = 0.5*(b-a);   // Size of interval
  const double h2  = 0.5*h;       // Size of subinterval
  const double mid = 0.5*(a+b);   // Midpoint of interval

  double x = a+h2;
  const double f_mid_left  = (*func_)(x);  // func val at midpt of left subinterval
  x = b-h2;
  const double f_mid_right = (*func_)(x);  // func val at midpt of right subinterval

  const double S_left  = simpson( fa,   f_mid_left,  fmid, h2 );  // Simpson's result for interval [a, (a+b)/2]
  const double S_right = simpson( fmid, f_mid_right, fb,   h2 );  // Simpson's result for interval [(a+b)/2, b]

  if (lev >= maxLevels ){
    nMaxEvals++;
    // sum up and return
    return ( S_left + S_right );
  }
  else{
    // check for convergence in this interval
    const double absErr = std::abs( S_whole - (S_left + S_right) );
    const double relErr = std::abs( absErr ) / ( std::abs(S_whole) + absErrTol_ );
    if ( absErr < absErrTol_  &&
	 relErr < relErrTol_ ){
      // sum up and return
      return ( S_left + S_right );
    }
    else{    // Refine the interval to improve accuracy
      double sum = 0;
      sum += refine( a,   mid, fa,   fmid, f_mid_left,  S_left,  lev+1 );
      sum += refine( mid, b,   fmid, fb,   f_mid_right, S_right, lev+1 );

      return( sum );
    }
  }
}
//--------------------------------------------------------------------
double
Integrator::f( const double x)
{
  return std::sin(x)+1.0;
}
double
Integrator::f3( const double x )
{
  return std::abs(sin(x));
}
double
Integrator::result( const double lo, const double hi )
{
  return ( cos(hi) - cos(lo) +(hi-lo));
}
double
Integrator::result3( const double lo, const double hi )
{
  return ( sin(hi)/std::abs(sin(hi))*cos(hi)
	   -sin(lo)/std::abs(sin(lo))*cos(lo) );
}


//==============================================================================
//==============================================================================

/*
#include <iostream>
int main()
{
  using namespace std;
  cout << "setting up problem..." << endl;

  BasicIntegrator myIntegral( -1.0, 1.0, 11 );

  // create the integrand functor
  DoubleFunctor<BasicIntegrator> fun( &myIntegral, &Integrator::f );

  // set the integrand functor in the integrator.
  myIntegral.set_integrand_func( &fun );

  cout << "done." << endl
       << "now integrating the function." << endl;
  double ans = myIntegral.integrate();
  double trueAns = myIntegral.result(-1.0,1.0);
  cout << endl << "finished!" << endl
       << " result = " << ans << endl
       << "          " << trueAns << endl
       << " rel error % " << (trueAns-ans)/trueAns << endl;
}
*/




