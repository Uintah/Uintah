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

#include <cassert>
#include <vector>
#include <set>
#include <algorithm>

#include <tabprops/prepro/mixmdl/GaussKronrod.h>

using std::vector;
using std::cout; using std::endl;

//====================================================================
//====================================================================


GaussKronrod::GaussKronrod( const double lo,
			    const double hi,
			    const int nIntervals )
  : Integrator( lo, hi ),
    initIntervals_( std::max(nIntervals,1) ),

    maxLevels_   ( 50   ),
    maxIntervals_( 1000 )
{
  nIntervals_ = std::max(nIntervals,1);  // interval counter

  interrogate_ = false;
  outfileName_ = "integrator.dat";

  // do a bunch of consistency checking on the Gauss-Kronrod data...
  assert( nGaussKronrod == (nGauss*2 + 1) );
  for( int i=0; i<nGauss; i++ ){
    // be sure that every other Gauss-Kronrod point is a Gauss point
    if( gaussPoint[i] != gaussKronrodPoint[2*i+1] )
      cout << gaussPoint[i] << " " << gaussKronrodPoint[2*i+1] << " " << i << endl;
    assert( gaussPoint[i] == gaussKronrodPoint[2*i+1] );

    // assure proper symmetry
    assert( gaussPoint [i] == -gaussPoint [nGauss-i-1] );
    assert( gaussWeight[i] ==  gaussWeight[nGauss-i-1] );
  }
  // assure proper symmetry
  for( int i=0; i<nGaussKronrod; i++ ){
    if( gaussKronrodPoint[i] != -gaussKronrodPoint[nGaussKronrod-i-1] )
      cout << gaussKronrodPoint[i] << " " << gaussKronrodPoint[nGaussKronrod-i-1] << endl;
    assert( gaussKronrodPoint[i] == -gaussKronrodPoint[nGaussKronrod-i-1] );
    if( gaussKronrodWeight[i] != gaussKronrodWeight[nGaussKronrod-i-1] )
      cout << gaussKronrodWeight[i] << " " << gaussKronrodWeight[nGaussKronrod-i-1] << " " << i << endl;
    assert( gaussKronrodWeight[i] == gaussKronrodWeight[nGaussKronrod-i-1] );
  }
}
//--------------------------------------------------------------------
GaussKronrod::~GaussKronrod()
{}
//--------------------------------------------------------------------
double
GaussKronrod::integrate()
{
  if( interrogate_ ){
    fout.open( outfileName_.c_str(), std::ofstream::trunc );
    assert( fout.is_open() );
    fout << "# eval pt     fun val    interval width" << std::endl;
  }

  // set the bounds
  std::set<double> bounds;
  set_bounds( bounds );

  double sum = 0.0;
  nIntervals_ = initIntervals_;

  // evaluate the integral over each interval, refining as necessary
  std::set<double>::const_iterator iset2= bounds.begin();
  std::set<double>::const_iterator iset1 = iset2;
  ++iset2;
  assert( iset2 != bounds.end() );
  do{
    sum += evaluate( *iset1, *iset2, 0 );
    if( interrogate_ ) fout.flush();
    iset1 = iset2;
    ++iset2;
  } while( iset2 != bounds.end() );

  if( interrogate_ ) fout.close();

  return double(sum);
}
//--------------------------------------------------------------------
void
GaussKronrod::set_bounds( std::set<double> & bounds )
{
  // add singularities
  bounds = singul_;

  // add specified interval endpoints
  const double dx = (hiBound_-loBound_)/double(initIntervals_);
  for( int i=0; i<=initIntervals_; i++ ){
    bounds.insert( loBound_ + double(i)*dx );
  }

  // remove any points outside the integral bounds
  std::set<double>::iterator iter;
  for( iter  = bounds.begin();
       iter != bounds.end();
       iter++ )
    {
      if( *iter > hiBound_ || *iter < loBound_ )
	bounds.erase( iter );
    }
}
//--------------------------------------------------------------------
double
GaussKronrod::evaluate( const double a,
			const double b,
			const int level )
{
  assert( b > a );
  //
  // we must compute the integral over the range [-1,1]
  // so we transform the limits [a,b] to [-1,1] as follows:
  //
  //  int_{a}^{b} f(x) dx  =
  //	 int_{-1}^{1}  (b-a)/2 * f( 0.5*((b+a)+t*(b-a)) ) dt
  //
  const double halfWidth = 0.5*(b-a);
  const double midpoint  = a + halfWidth;

  double sum      = 0.0;
  double gaussSum = 0.0;

  int j=0;
  for( int i=0; i<nGaussKronrod; i++ ){
    // set the point to evaluate the function,
    // converting from [-1,1] to [a,b]
    const double xpt = halfWidth * gaussKronrodPoint[i] + midpoint;
    const double funVal = (*func_)( xpt );

    if( interrogate_ ){
      fout << xpt << " " << funVal << " " << b-a << endl;
    }

    sum += funVal * gaussKronrodWeight[i];
    if( gaussPoint[j] == gaussKronrodPoint[i] ){
      gaussSum += funVal * gaussWeight[j];
      j++;
    }
  }

  // transform the integrated result back to [a,b]
  sum      *= halfWidth;
  gaussSum *= halfWidth;

  const double absErr = std::abs( std::abs(sum)-std::abs(gaussSum) );
  const double relErr = absErr / ( std::abs(sum)+absErrTol_ );

  if( level >= maxLevels_ || nIntervals_ >= maxIntervals_ ){
    //    cout << "Maximum number of refinements exceeded!  Returning current estimate." << endl;
    return sum;
  }

  if( absErr > absErrTol_ || relErr > relErrTol_ ){
    sum  = evaluate( a, midpoint, level+1 );
    sum += evaluate( midpoint, b, level+1 );
    nIntervals_++;
  }

  return sum;
}
//--------------------------------------------------------------------
