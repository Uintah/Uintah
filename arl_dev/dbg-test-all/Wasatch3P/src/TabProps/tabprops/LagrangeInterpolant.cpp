/**
 *  \file   LagrangeInterpolant.cpp
 *  \date   Jun 24, 2013
 *  \author "James C. Sutherland"
 *
 *
 * The MIT License
 *
 * Copyright (c) 2013 The University of Utah
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
 *
 */

#include "LagrangeInterpolant.h"

# include <tabprops/Archive.h>

#include <stdexcept>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cassert>
#include <limits>

using std::cout;
using std::endl;
using std::vector;
using std::pair; using std::make_pair;

//===================================================================

//#define VERBOSE_COMPARE

bool vec_compare( const std::vector<double>& v1, const std::vector<double>& v2 ){
  if( v1.size() != v2.size() ) return false;
  const size_t nval = v1.size();

  double maxVal = 0.0;
  for( size_t i=0; i<nval; ++i )  maxVal += std::max( maxVal, std::abs(v1[i]) );

  const double RTOL = 5e-10;

  size_t ndiff=0;
  for( size_t i=0; i<nval; ++i ){
    const double abserr = std::abs( v1[i] - v2[i] );
    const double relerr = abserr / maxVal;
#   ifdef VERBOSE_COMPARE
    if( relerr > RTOL ){
      ++ndiff;
      cout << " differs at location " << i << " : " << v1[i] << " : " << v2[i] << " (" << abserr << ", " << relerr << ")" << std::endl;
    }
#   else
    if( relerr > RTOL ) return false;
#   endif
  }
# ifdef VERBOSE_COMPARE
  if( ndif>0 ) cout << ndiff << " / " << v2.size() << " locations differ\n";
  return ndiff==0;
# else
  return true;
# endif
}

//===================================================================

bool
is_monotonic( const vector<double>& x )
{
  assert( x.size() > 1 );
  if( x[1]>x[0] ){ // presume ascending order
    double lo = x[0];
    for( size_t i=1; i<x.size(); ++i ){
      if( x[i] < lo ) return false;
      lo = x[i];
    }
  }
  else{ // presume descending order
    double hi = x[0];
    for( size_t i=1; i<x.size(); ++i ){
      if( x[i] > hi ) return false;
      hi = x[i];
    }
  }
  return true;
}

//===================================================================

bool
is_increasing( const vector<double>& x )
{
  assert( x.size() > 1 );
  double lo = x[0];
  for( size_t i=1; i<x.size(); ++i ){
    if( x[i] < lo ) return false;
    lo = x[i];
  }
  return true;
}

//===================================================================

void enforce_incresing( const vector<double>& x, const std::string str )
{
  if( !is_increasing(x) ){
    std::ostringstream msg;
    msg << endl << endl << "ERROR from " << __FILE__ << " : " << __LINE__
        << endl << endl << str << endl
        << "Variable must be monontonically increasing." << endl;
    throw std::invalid_argument( msg.str() );
  }
}

//===================================================================

pair<double,double>
bounds( const vector<double>& x )
{
  assert( is_monotonic(x) );
  const double x1 = x.front();
  const double x2 = x.back();
  return ( x2 > x1 )      ?
      make_pair( x1, x2 ) :  // ascending
      make_pair( x2, x1 ) ;  // descending;
}

//===================================================================

bool is_uniform( const std::vector<double>& xvals )
{
  assert( xvals.size() > 1 );
  double dx = xvals[1]-xvals[0];
  for( size_t i=1; i<xvals.size(); ++i ){
    if( std::abs(dx - (xvals[i]-xvals[i-1])) / dx > 1e-10 ) return false;
  }
  return true;
}

//===================================================================

template< typename IndexT, typename ValT >
IndexT
index_finder( const ValT& x,
              const vector<ValT>& xgrid,
              const bool allowClipping=false )
{
//  std::cout << "Index Finder: x=" << x << "  (" << xgrid.front() << "," << xgrid.back() << ")\n";
  assert( !std::isnan(x) );

  const size_t nx = xgrid.size();
  IndexT ilo = 0;
  IndexT ihi = nx-1;

  if( allowClipping && x<xgrid.front() ) return ilo;
  if( allowClipping && x>xgrid.back()  ) return ihi;

  // sanity check
  if( x<xgrid.front() || x>xgrid.back() ){
    std::ostringstream msg;
    msg << __FILE__ << " : " << __LINE__ << endl
        << "root is not bracketed!" << endl;
    throw std::runtime_error(msg.str());
  }

  // regula falsi method to find lower index

  while( ihi-ilo > 1 ){
    const ValT m = ( xgrid[ihi]-xgrid[ilo] ) / ValT( ihi-ilo );
    const IndexT c = std::max( ilo+1, IndexT( ihi - (xgrid[ihi]-x)/m ) );
    assert( c>0 && c<xgrid.size() );
    if( x >= xgrid[c] )
      ilo = std::min(ihi-1,c);
    else
      ihi = std::max(ilo+1,c);
  }
  // error checking:
  if( !allowClipping && (x > xgrid[ihi] || x < xgrid[ilo]) ){
    std::ostringstream msg;
    msg << __FILE__ << " : " << __LINE__ << std::endl
        << " Regula falsi failed to converge properly!" << std::endl
        << " Target x=" << x << std::endl
        << " xlo=" << xgrid[ilo] << ", xhi=" << xgrid[ihi]
        << std::endl << std::endl;
    throw std::runtime_error( msg.str() );
  }

  return ilo;
}

//===================================================================

inline size_t low_index( const vector<double>& xvals,
                         const double x,
                         const unsigned order )
{
  int i = index_finder<size_t,double>( x, xvals, true );
  // adjust for order
  const int half = int(order)/2;
  i = std::max( 0, i-half );
  // adjust at boundaries
  const int imax = xvals.size()-1;
  i = std::min( imax-int(order), std::max( 0, i ) );
  return i;
}

//===================================================================

inline size_t low_index( const double xlo,
                         const double xhi,
                         const double dx,
                         const double x,
                         const unsigned order )
{
  int i = (x-xlo) / dx;
  // adjust for order
  const int half = int(order)/2;
  i = std::max( 0, i-half );
  // adjust at boundaries
  const int imax = (xhi-xlo)/dx;
  i = std::min( imax-int(order), std::max( 0, i ) );
  return i;
}

//===================================================================

// macro to expand a calculation of a single fraction term in the lagrange polynomial expansions
#define BASIS_FUN( k, i, xvals, x ) (x-xvals[i])/(xvals[k]-xvals[i])

/**
 * @brief compute the 1D basis functions for Lagrange polynomial interpolants
 *
 * @param xvals the independent variable function values
 * @param x the independent variable value where we want to interpolate
 * @param order the order of interpolant
 * @param coefs the coefficients that we will set here
 * @param isUniform if true then the search for the bracketing interval will use
 * a faster algorithm.  This should only be used for uniform spacing of the
 * independent variable.
 *
 * @return the lo index for accessing the relevant data (can prevent further searches later)
 */
size_t basis_funs_1d( const vector<double>& xvals,
                      const double x,
                      const unsigned order,
                      double* const coefs,
                      const bool isUniform = false )
{
  assert( order>=1 );

  size_t ilo = 0;
  if( isUniform ) ilo = low_index( xvals[0], xvals[xvals.size()-1], xvals[1]-xvals[0], x, order );
  else            ilo = low_index( xvals, x, order );

  switch (order) {
  case 1:
    coefs[0] = BASIS_FUN(ilo+0,ilo+1,xvals,x);
    coefs[1] = BASIS_FUN(ilo+1,ilo  ,xvals,x);
    break;
  case 2:
    coefs[0] = BASIS_FUN(ilo+0,ilo+1,xvals,x) * BASIS_FUN(ilo+0,ilo+2,xvals,x);
    coefs[1] = BASIS_FUN(ilo+1,ilo  ,xvals,x) * BASIS_FUN(ilo+1,ilo+2,xvals,x);
    coefs[2] = BASIS_FUN(ilo+2,ilo  ,xvals,x) * BASIS_FUN(ilo+2,ilo+1,xvals,x);
    break;
  default:
    for( size_t k=0; k<=order; ++k ){
      coefs[k] = 1;
      for( size_t i=0; i<=order; ++i ){
        if( i==k ) continue;
        coefs[k] *= BASIS_FUN( ilo+k, ilo+i, xvals, x );
      }
    }
    break;
  }
  return ilo;
}

inline size_t basis_funs_1d( const vector<double>& xvals,
                             const double x,
                             const unsigned order,
                             vector<double>& coefs,
                             const bool isUniform = false )
{
# ifndef NDEBUG
  assert( coefs.size() == order+1 );
  assert( order>=1 );
# endif
  return basis_funs_1d( xvals, x, order, &coefs[0], isUniform );
}

//===================================================================

size_t der_basis_funs_1d( const vector<double>& xvals,
                          const double x,
                          const unsigned order,
                          double* coefs,
                          const bool isUniform = false )
{
# ifndef NDEBUG
  assert( order>=1 );
# endif

  size_t ilo = 0;
  if( isUniform ) ilo = low_index( xvals[0], xvals[xvals.size()-1], xvals[1]-xvals[0], x, order );
  else            ilo = low_index( xvals, x, order );

  switch (order) { // unroll the loops for first and second order
  case 1:
    coefs[0] = 1.0 / (xvals[ilo  ]-xvals[ilo+1]);
    coefs[1] = 1.0 / (xvals[ilo+1]-xvals[ilo  ]);
    break;
  case 2:
    coefs[0] = BASIS_FUN(ilo+0,ilo+1,xvals,x) / (xvals[ilo+0]-xvals[ilo+2]) + BASIS_FUN(ilo+0,ilo+2,xvals,x) / (xvals[ilo+0]-xvals[ilo+1]);
    coefs[1] = BASIS_FUN(ilo+1,ilo+0,xvals,x) / (xvals[ilo+1]-xvals[ilo+2]) + BASIS_FUN(ilo+1,ilo+2,xvals,x) / (xvals[ilo+1]-xvals[ilo+0]);
    coefs[2] = BASIS_FUN(ilo+2,ilo+0,xvals,x) / (xvals[ilo+2]-xvals[ilo+1]) + BASIS_FUN(ilo+2,ilo+1,xvals,x) / (xvals[ilo+2]-xvals[ilo+0]);
    break;
  case 3:{
    const double tmp01 = BASIS_FUN( ilo+0, ilo+1, xvals, x );
    const double tmp02 = BASIS_FUN( ilo+0, ilo+2, xvals, x );
    const double tmp03 = BASIS_FUN( ilo+0, ilo+3, xvals, x );
    const double tmp10 = BASIS_FUN( ilo+1, ilo+0, xvals, x );
    const double tmp12 = BASIS_FUN( ilo+1, ilo+2, xvals, x );
    const double tmp13 = BASIS_FUN( ilo+1, ilo+3, xvals, x );
    const double tmp20 = BASIS_FUN( ilo+2, ilo+0, xvals, x );
    const double tmp21 = BASIS_FUN( ilo+2, ilo+1, xvals, x );
    const double tmp23 = BASIS_FUN( ilo+2, ilo+3, xvals, x );
    const double tmp30 = BASIS_FUN( ilo+3, ilo+0, xvals, x );
    const double tmp31 = BASIS_FUN( ilo+3, ilo+1, xvals, x );
    const double tmp32 = BASIS_FUN( ilo+3, ilo+2, xvals, x );
    coefs[0] = tmp02 * tmp03 / ( xvals[ilo+0] - xvals[ilo+1] )
             + tmp01 * tmp03 / ( xvals[ilo+0] - xvals[ilo+2] )
             + tmp01 * tmp02 / ( xvals[ilo+0] - xvals[ilo+3] );
    coefs[1] = tmp12 * tmp13 / ( xvals[ilo+1] - xvals[ilo+0] )
             + tmp10 * tmp13 / ( xvals[ilo+1] - xvals[ilo+2] )
             + tmp10 * tmp12 / ( xvals[ilo+1] - xvals[ilo+3] );
    coefs[2] = tmp21 * tmp23 / ( xvals[ilo+2] - xvals[ilo+0] )
             + tmp20 * tmp23 / ( xvals[ilo+2] - xvals[ilo+1] )
             + tmp20 * tmp21 / ( xvals[ilo+2] - xvals[ilo+3] );
    coefs[3] = tmp31 * tmp32 / ( xvals[ilo+3] - xvals[ilo+0] )
             + tmp30 * tmp32 / ( xvals[ilo+3] - xvals[ilo+1] )
             + tmp30 * tmp31 / ( xvals[ilo+3] - xvals[ilo+2] );
    break;
  }
  default:
    // this is fairly slow because of the duplicate calls to BASIS_FUN
    for( size_t k=0; k<=order; ++k ){
      coefs[k] = 0.0;
      for( size_t i=0; i<=order; ++i ){
        if( i==k ) continue;
        double tmpprod = 1.0;
        for( size_t j=0; j<=order; ++j ){
          if( j==i || j==k ) continue;
          tmpprod *= BASIS_FUN(ilo+k,ilo+j,xvals,x);
        }
        coefs[k] += tmpprod / ( xvals[ilo+k]-xvals[ilo+i] );
      }
    }
    break;
  }
  return ilo;
}

inline size_t der_basis_funs_1d( const vector<double>& xvals,
                                 const double x,
                                 const unsigned order,
                                 vector<double>& coefs,
                                 const bool isUniform = false )
{
# ifndef NDEBUG
  assert( coefs.size() == order+1 );
  assert( order>=1 );
# endif
  return der_basis_funs_1d( xvals, x, order, &coefs[0], isUniform );
}

//===================================================================

# define TMPVAL( i,j,xvals )( 1.0/(xvals[i]-xvals[j]) )

inline size_t second_der_basis_funs_1d( const vector<double>& xvals,
                                        const double x,
                                        const unsigned order,
                                        double* coefs,
                                        const bool isUniform = false )
{
# ifndef NDEBUG
  assert( order>=1 );
# endif

  if( order == 1 ){
    coefs[0] = 0; coefs[1] = 0;
    return 0;  // index doesn't matter in this case.
  }

  size_t ilo = 0;
  if( isUniform ) ilo = low_index( xvals[0], xvals[xvals.size()-1], xvals[1]-xvals[0], x, order );
  else            ilo = low_index( xvals, x, order );

  switch (order) { // unroll the loops for first and second order
  case 2:{
    coefs[0] = 2.0*TMPVAL( ilo  , ilo+1, xvals ) * TMPVAL( ilo  , ilo+2, xvals );
    coefs[1] = 2.0*TMPVAL( ilo+1, ilo  , xvals ) * TMPVAL( ilo+1, ilo+2, xvals );
    coefs[2] = 2.0*TMPVAL( ilo+2, ilo  , xvals ) * TMPVAL( ilo+2, ilo+1, xvals );
    break;
  }
  case 3:{
    const double bf01 = BASIS_FUN( ilo  , ilo+1, xvals, x );
    const double bf02 = BASIS_FUN( ilo  , ilo+2, xvals, x );
    const double bf03 = BASIS_FUN( ilo  , ilo+3, xvals, x );
    const double bf10 = BASIS_FUN( ilo+1, ilo+0, xvals, x );
    const double bf12 = BASIS_FUN( ilo+1, ilo+2, xvals, x );
    const double bf13 = BASIS_FUN( ilo+1, ilo+3, xvals, x );
    const double bf20 = BASIS_FUN( ilo+2, ilo+0, xvals, x );
    const double bf21 = BASIS_FUN( ilo+2, ilo+1, xvals, x );
    const double bf23 = BASIS_FUN( ilo+2, ilo+3, xvals, x );
    const double bf30 = BASIS_FUN( ilo+3, ilo+0, xvals, x );
    const double bf31 = BASIS_FUN( ilo+3, ilo+1, xvals, x );
    const double bf32 = BASIS_FUN( ilo+3, ilo+2, xvals, x );

    const double tmp01 = TMPVAL( ilo  , ilo+1, xvals );
    const double tmp02 = TMPVAL( ilo  , ilo+2, xvals );
    const double tmp03 = TMPVAL( ilo  , ilo+3, xvals );
    const double tmp10 = TMPVAL( ilo+1, ilo  , xvals );
    const double tmp12 = TMPVAL( ilo+1, ilo+2, xvals );
    const double tmp13 = TMPVAL( ilo+1, ilo+3, xvals );
    const double tmp20 = TMPVAL( ilo+2, ilo  , xvals );
    const double tmp21 = TMPVAL( ilo+2, ilo+1, xvals );
    const double tmp23 = TMPVAL( ilo+2, ilo+3, xvals );
    const double tmp30 = TMPVAL( ilo+3, ilo  , xvals );
    const double tmp31 = TMPVAL( ilo+3, ilo+1, xvals );
    const double tmp32 = TMPVAL( ilo+3, ilo+2, xvals );

    coefs[0] = tmp01*( tmp02*bf03 + tmp03*bf02 ) + tmp02*( tmp01*bf03 + tmp03*bf01 ) + tmp03*( tmp01*bf02 + tmp02*bf01 );
    coefs[1] = tmp10*( tmp12*bf13 + tmp13*bf12 ) + tmp12*( tmp10*bf13 + tmp13*bf10 ) + tmp13*( tmp10*bf12 + tmp12*bf10 );
    coefs[2] = tmp20*( tmp21*bf23 + tmp23*bf21 ) + tmp21*( tmp20*bf23 + tmp23*bf20 ) + tmp23*( tmp20*bf21 + tmp21*bf20 );
    coefs[3] = tmp30*( tmp31*bf32 + tmp32*bf31 ) + tmp31*( tmp30*bf32 + tmp32*bf30 ) + tmp32*( tmp30*bf31 + tmp31*bf30 );
    break;
  }
  default:{ // not unrolled. Slow.
    for( size_t k=0; k<=order; ++k ){
      coefs[k] = 0.0;
      for( size_t i=0; i<=order; ++i ){
        if( i == k ) continue;
        double tmp = 0.0;
        for( size_t j=0; j<=order; ++j ){
          if( j == k || j == i ) continue;
          double bf = 1.0;
          for( size_t m=0; m<=order; ++m ){
          if( m == k || m == i || m == j ) continue;
            bf *= BASIS_FUN( ilo+k, ilo+m, xvals, x );
          }
          tmp += TMPVAL( ilo+k, ilo+j, xvals ) * bf;
        }
        coefs[k] +=  tmp * TMPVAL( ilo+k, ilo+i, xvals );
      }
    }
    break;
  } // default
  } // switch( order )

  return ilo;
}

inline size_t second_der_basis_funs_1d( const vector<double>& xvals,
                                        const double x,
                                        const unsigned order,
                                        vector<double>& coefs,
                                        const bool isUniform = false )
{
# ifndef NDEBUG
  assert( coefs.size() == order+1 );
  assert( order>=1 );
# endif
  return second_der_basis_funs_1d(xvals,x,order,&coefs[0],isUniform);
}

//===================================================================

template<typename Archive> void
LagrangeInterpolant::serialize( Archive& ar, const unsigned version )
{
  ar & order_ & allowClipping_;
}

//===================================================================

LagrangeInterpolant1D::LagrangeInterpolant1D( const unsigned order,
                                              const vector<double>& xvals,
                                              const vector<double>& fvals,
                                              const bool clipValues )
: LagrangeInterpolant( order, clipValues ),
  bounds_( bounds(xvals) ),
  xvals_( xvals ),
  fvals_( fvals ),
  isUniform_( is_uniform(xvals) )
{
  assert( fvals_.size() == xvals_.size() );
  assert( xvals_.size() > order+1 );
  enforce_incresing( xvals_, "1D interpolant independent variable" );
}

//-------------------------------------------------------------------

LagrangeInterpolant1D::LagrangeInterpolant1D( const LagrangeInterpolant1D& other )
: LagrangeInterpolant( other.order_, other.allowClipping_ ),
  bounds_( other.bounds_ ),
  xvals_( other.xvals_ ),
  fvals_( other.fvals_ ),
  isUniform_( other.isUniform_ )
{
  assert( fvals_.size() == xvals_.size() );
}

//-------------------------------------------------------------------

LagrangeInterpolant1D::LagrangeInterpolant1D()
: LagrangeInterpolant(1,true)
{}

//-------------------------------------------------------------------

double
LagrangeInterpolant1D::value( const double* const indep ) const
{
  const double x = allowClipping_ ? std::max( std::min( *indep, bounds_.second ), bounds_.first ) : *indep;

  size_t ilo = 0;
  if( isUniform_ ) ilo = low_index( xvals_[0], xvals_[xvals_.size()-1], xvals_[1]-xvals_[0], x, order_ );
  else             ilo = low_index( xvals_, x, order_ );

  double val=0;

  switch (order_){ // for up to third order, unroll the loops:
  case 1:
    val = fvals_[ilo  ] * BASIS_FUN( ilo+0, ilo+1, xvals_, x )
        + fvals_[ilo+1] * BASIS_FUN( ilo+1, ilo  , xvals_, x );
    break;
  case 2:
    val = fvals_[ilo  ] * BASIS_FUN( ilo+0, ilo+1, xvals_, x ) * BASIS_FUN( ilo+0, ilo+2, xvals_, x )
        + fvals_[ilo+1] * BASIS_FUN( ilo+1, ilo  , xvals_, x ) * BASIS_FUN( ilo+1, ilo+2, xvals_, x )
        + fvals_[ilo+2] * BASIS_FUN( ilo+2, ilo  , xvals_, x ) * BASIS_FUN( ilo+2, ilo+1, xvals_, x );
    break;
  case 3:
    val = fvals_[ilo  ] * BASIS_FUN( ilo+0, ilo+1, xvals_, x ) * BASIS_FUN( ilo+0, ilo+2, xvals_, x ) * BASIS_FUN( ilo+0, ilo+3, xvals_, x )
        + fvals_[ilo+1] * BASIS_FUN( ilo+1, ilo  , xvals_, x ) * BASIS_FUN( ilo+1, ilo+2, xvals_, x ) * BASIS_FUN( ilo+1, ilo+3, xvals_, x )
        + fvals_[ilo+2] * BASIS_FUN( ilo+2, ilo  , xvals_, x ) * BASIS_FUN( ilo+2, ilo+1, xvals_, x ) * BASIS_FUN( ilo+2, ilo+3, xvals_, x )
        + fvals_[ilo+3] * BASIS_FUN( ilo+3, ilo  , xvals_, x ) * BASIS_FUN( ilo+3, ilo+1, xvals_, x ) * BASIS_FUN( ilo+3, ilo+2, xvals_, x );
    break;
  default: // slower - use loops
    for( size_t k=0; k<=order_; ++k ){
      double tmp=1;
      for( size_t i=0; i<=order_; ++i ){
        if( i==k ) continue;
        tmp *= BASIS_FUN( ilo+k, ilo+i, xvals_, x );
      }
      val += fvals_[k+ilo] * tmp;
    }
    break;
  }
  return val;
}

//-------------------------------------------------------------------

double
LagrangeInterpolant1D::derivative( const double* const indep, const int dim ) const
{
  const double x = allowClipping_ ? std::max( std::min( *indep, bounds_.second ), bounds_.first ) : *indep;

  size_t ilo = 0;
  if( isUniform_ ) ilo = low_index( xvals_[0], xvals_[xvals_.size()-1], xvals_[1]-xvals_[0], x, order_ );
  else             ilo = low_index( xvals_, x, order_ );

  double val = 0.0;

  switch (order_) { // unroll the loops for first and second order
  case 1:
    val = fvals_[ilo  ] / (xvals_[ilo  ]-xvals_[ilo+1])
        + fvals_[ilo+1] / (xvals_[ilo+1]-xvals_[ilo  ]);
    break;
  case 2:
    val = fvals_[ilo  ] * ( BASIS_FUN( ilo+0, ilo+1, xvals_, x ) / ( xvals_[ilo+0] - xvals_[ilo+2] )
                          + BASIS_FUN( ilo+0, ilo+2, xvals_, x ) / ( xvals_[ilo+0] - xvals_[ilo+1] ) )
        + fvals_[ilo+1] * ( BASIS_FUN( ilo+1, ilo+0, xvals_, x ) / ( xvals_[ilo+1] - xvals_[ilo+2] )
                          + BASIS_FUN( ilo+1, ilo+2, xvals_, x ) / ( xvals_[ilo+1] - xvals_[ilo+0] ) )
        + fvals_[ilo+2] * ( BASIS_FUN( ilo+2, ilo+0, xvals_, x ) / ( xvals_[ilo+2] - xvals_[ilo+1] )
                          + BASIS_FUN( ilo+2, ilo+1, xvals_, x ) / ( xvals_[ilo+2] - xvals_[ilo+0] ) );
    break;
  case 3:{
    // store a few off to avoid repeated calculation since there are duplicate usage
    const double tmp01 = BASIS_FUN( ilo+0, ilo+1, xvals_, x );
    const double tmp02 = BASIS_FUN( ilo+0, ilo+2, xvals_, x );
    const double tmp03 = BASIS_FUN( ilo+0, ilo+3, xvals_, x );
    const double tmp10 = BASIS_FUN( ilo+1, ilo+0, xvals_, x );
    const double tmp12 = BASIS_FUN( ilo+1, ilo+2, xvals_, x );
    const double tmp13 = BASIS_FUN( ilo+1, ilo+3, xvals_, x );
    const double tmp20 = BASIS_FUN( ilo+2, ilo+0, xvals_, x );
    const double tmp21 = BASIS_FUN( ilo+2, ilo+1, xvals_, x );
    const double tmp23 = BASIS_FUN( ilo+2, ilo+3, xvals_, x );
    const double tmp30 = BASIS_FUN( ilo+3, ilo+0, xvals_, x );
    const double tmp31 = BASIS_FUN( ilo+3, ilo+1, xvals_, x );
    const double tmp32 = BASIS_FUN( ilo+3, ilo+2, xvals_, x );
    val = fvals_[ilo  ] * ( tmp02*tmp03/(xvals_[ilo+0]-xvals_[ilo+1])
                          + tmp01*tmp03/(xvals_[ilo+0]-xvals_[ilo+2])
                          + tmp01*tmp02/(xvals_[ilo+0]-xvals_[ilo+3]) )
        + fvals_[ilo+1] * ( tmp12*tmp13/(xvals_[ilo+1]-xvals_[ilo+0])
                          + tmp10*tmp13/(xvals_[ilo+1]-xvals_[ilo+2])
                          + tmp10*tmp12/(xvals_[ilo+1]-xvals_[ilo+3]) )
        + fvals_[ilo+2] * ( tmp21*tmp23/(xvals_[ilo+2]-xvals_[ilo+0])
                          + tmp20*tmp23/(xvals_[ilo+2]-xvals_[ilo+1])
                          + tmp20*tmp21/(xvals_[ilo+2]-xvals_[ilo+3]) )
        + fvals_[ilo+3] * ( tmp31*tmp32/(xvals_[ilo+3]-xvals_[ilo+0])
                          + tmp30*tmp32/(xvals_[ilo+3]-xvals_[ilo+1])
                          + tmp30*tmp31/(xvals_[ilo+3]-xvals_[ilo+2]) );
    break;
  }
  default: // not unrolled.  This will be significantly slower due to repeated calculation of BASIS_FUN
    for( size_t k=0; k<=order_; ++k ){
      double tmp = 0.0;
      for( size_t i=0; i<=order_; ++i ){
        if( i==k ) continue;
        double tmpprod = 1.0;
        for( size_t j=0; j<=order_; ++j ){
          if( j==i || j==k ) continue;
          tmpprod *= BASIS_FUN( ilo+k, ilo+j, xvals_, x );
        }
        tmp += tmpprod / ( xvals_[ilo+k] - xvals_[ilo+i] );
      }
      val += fvals_[k+ilo] * tmp;
    }
    break;
  }
  return val;
}

//-------------------------------------------------------------------

double
LagrangeInterpolant1D::second_derivative( const double* const indep, const int dim1, const int dim2 ) const
{
  if( order_ == 1 ) return 0;  // For first order, the second derivative is zero.

  const double x = allowClipping_ ? std::max( std::min( *indep, bounds_.second ), bounds_.first ) : *indep;

  double coefs[order_+1];
  const size_t ilo = second_der_basis_funs_1d( xvals_, x, order_, coefs, isUniform_ );
  double val = 0.0;
  for( short k=0; k<=order_; ++k ) val += fvals_[ilo+k]*coefs[k];
  return val;
}

//-------------------------------------------------------------------

LagrangeInterpolant1D::~LagrangeInterpolant1D()
{}

//-------------------------------------------------------------------

vector<pair<double,double> >
LagrangeInterpolant1D::get_bounds() const
{
  vector<pair<double,double> > b;
  b.push_back(bounds_);
  return b;
}

//-------------------------------------------------------------------

bool
LagrangeInterpolant1D::operator==( const LagrangeInterpolant& other ) const
{
  const LagrangeInterpolant1D& a = dynamic_cast<const LagrangeInterpolant1D&>(other);
  return order_ == a.get_order()
      && allowClipping_ == a.clipping()
      && vec_compare( xvals_, a.xvals_ )
      && vec_compare( fvals_, a.fvals_ )
      && bounds_ == a.bounds_;
}

//-------------------------------------------------------------------

template<typename Archive> void
LagrangeInterpolant1D::serialize( Archive& ar, const unsigned version )
{
  ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP( LagrangeInterpolant )
     & BOOST_SERIALIZATION_NVP( xvals_ )
     & BOOST_SERIALIZATION_NVP( fvals_ )
     & BOOST_SERIALIZATION_NVP( isUniform_ );
  bounds_ = bounds(xvals_);
}

//===================================================================

LagrangeInterpolant2D::LagrangeInterpolant2D( const unsigned order,
                                              const vector<double>& xvals,
                                              const vector<double>& yvals,
                                              const vector<double>& fvals,
                                              const bool clipValues )
: LagrangeInterpolant(order,clipValues),
  xbounds_( bounds(xvals) ),
  ybounds_( bounds(yvals) ),
  xvals_( xvals ),
  yvals_( yvals ),
  fvals_( fvals ),
  isUniformX_( is_uniform(xvals) ),
  isUniformY_( is_uniform(yvals) )
{
  assert( fvals_.size() == xvals_.size() * yvals_.size() );

  assert( xvals_.size() > order_+1 );
  assert( yvals_.size() > order_+1 );

  enforce_incresing( xvals_, "2D interpolant independent variable 1" );
  enforce_incresing( yvals_, "2D interpolant independent variable 2" );

  xcoef1d_.resize( order_+1 );
  ycoef1d_.resize( order_+1 );
}

//-------------------------------------------------------------------

LagrangeInterpolant2D::LagrangeInterpolant2D( const LagrangeInterpolant2D& other )
: LagrangeInterpolant(other.order_,other.allowClipping_),
  xbounds_( other.xbounds_ ),
  ybounds_( other.ybounds_ ),
  xvals_( other.xvals_ ),
  yvals_( other.yvals_ ),
  fvals_( other.fvals_ ),
  isUniformX_( other.isUniformX_ ),
  isUniformY_( other.isUniformY_ )
{
  assert( fvals_.size() == xvals_.size() * yvals_.size() );

  xcoef1d_.resize( order_+1 );
  ycoef1d_.resize( order_+1 );
}

//-------------------------------------------------------------------

LagrangeInterpolant2D::LagrangeInterpolant2D()
: LagrangeInterpolant(1,true)
{}

//-------------------------------------------------------------------

double
LagrangeInterpolant2D::value( const double* const indep ) const
{
  const size_t nx = xvals_.size();

  const double x = allowClipping_ ? std::min( std::max( indep[0], xbounds_.first ), xbounds_.second ) : indep[0];
  const double y = allowClipping_ ? std::min( std::max( indep[1], ybounds_.first ), ybounds_.second ) : indep[1];

  const size_t ilo = basis_funs_1d( xvals_, x, order_, xcoef1d_, isUniformX_ );
  const size_t jlo = basis_funs_1d( yvals_, y, order_, ycoef1d_, isUniformY_ );

# ifndef NDEBUG
  assert( ilo+order_ < xvals_.size() );
  assert( jlo+order_ < yvals_.size() );
# endif

  double val = 0.0;

  switch ( order_ ){
  case 1:
    val = fvals_[ilo  +(jlo  )*nx] * xcoef1d_[0] * ycoef1d_[0]
        + fvals_[ilo+1+(jlo  )*nx] * xcoef1d_[1] * ycoef1d_[0]
        + fvals_[ilo  +(jlo+1)*nx] * xcoef1d_[0] * ycoef1d_[1]
        + fvals_[ilo+1+(jlo+1)*nx] * xcoef1d_[1] * ycoef1d_[1];
    break;
  default:
    for( size_t j=0; j<=order_; ++j ){
      const size_t jix = (jlo+j)*nx;
      for( size_t i=0; i<=order_; ++i ){
        val += fvals_[ ilo+i + jix ] * xcoef1d_[i] * ycoef1d_[j];
      }
    }
    break;
  }
  return val;
}

//-------------------------------------------------------------------

double
LagrangeInterpolant2D::derivative( const double* const indep,
                                   const int dim ) const
{
  assert( dim <  2 );
  assert( dim >= 0);
  const size_t nx = xvals_.size();

  const double x = allowClipping_ ? std::min( std::max( indep[0], xbounds_.first ), xbounds_.second ) : indep[0];
  const double y = allowClipping_ ? std::min( std::max( indep[1], ybounds_.first ), ybounds_.second ) : indep[1];

  size_t ilo, jlo;
  if( dim==0 ){
    ilo = der_basis_funs_1d( xvals_, x, order_, xcoef1d_, isUniformX_ );
    jlo =     basis_funs_1d( yvals_, y, order_, ycoef1d_, isUniformY_ );
  }
  else{
    ilo =     basis_funs_1d( xvals_, x, order_, xcoef1d_, isUniformX_ );
    jlo = der_basis_funs_1d( yvals_, y, order_, ycoef1d_, isUniformY_ );
  }

# ifndef NDEBUG
  assert( ilo+order_ < xvals_.size() );
  assert( jlo+order_ < yvals_.size() );
# endif

  double val = 0.0;

  switch ( order_ ){
  case 1:
    val = fvals_[ilo  +(jlo  )*nx] * xcoef1d_[0] * ycoef1d_[0]
        + fvals_[ilo+1+(jlo  )*nx] * xcoef1d_[1] * ycoef1d_[0]
        + fvals_[ilo  +(jlo+1)*nx] * xcoef1d_[0] * ycoef1d_[1]
        + fvals_[ilo+1+(jlo+1)*nx] * xcoef1d_[1] * ycoef1d_[1];
    break;
  default:
    for( size_t j=0; j<=order_; ++j ){
      const size_t jix = (jlo+j)*nx;
      for( size_t i=0; i<=order_; ++i ){
        val += fvals_[ ilo+i + jix ] * xcoef1d_[i] * ycoef1d_[j];
      }
    }
    break;
  }
  return val;
}

//-------------------------------------------------------------------

double
LagrangeInterpolant2D::second_derivative( const double* const indep, const int dim1, const int dim2 ) const
{
  assert( dim1 < 2 );
  assert( dim2 < 2 );

  const double x = allowClipping_ ? std::min( std::max( indep[0], xbounds_.first ), xbounds_.second ) : indep[0];
  const double y = allowClipping_ ? std::min( std::max( indep[1], ybounds_.first ), ybounds_.second ) : indep[1];

  double xcoefs[order_+1], ycoefs[order_+1];

  size_t ilo=0, jlo=0;

  switch( dim1 ){
    case 0:{
      switch( dim2 ){
        case 0:{
          ilo = second_der_basis_funs_1d( xvals_, x, order_, xcoefs, isUniformX_ );
          jlo =            basis_funs_1d( yvals_, y, order_, ycoefs, isUniformY_ );
          break;
        }
        case 1:{
          ilo = der_basis_funs_1d( xvals_, x, order_, xcoefs, isUniformX_ );
          jlo = der_basis_funs_1d( yvals_, y, order_, ycoefs, isUniformY_ );
          break;
        }
        break;
      }
      break;
    }
    case 1:{
      switch( dim2 ){
        case 0:{
          ilo = der_basis_funs_1d( xvals_, x, order_, xcoefs, isUniformX_ );
          jlo = der_basis_funs_1d( yvals_, y, order_, ycoefs, isUniformY_ );
          break;
        }
        case 1:{
          ilo =            basis_funs_1d( xvals_, x, order_, xcoefs, isUniformX_ );
          jlo = second_der_basis_funs_1d( yvals_, y, order_, ycoefs, isUniformY_ );
          break;
        }
        break;
      }
      break;
    }
    default:
      throw std::invalid_argument("invalid dimension specification for second derivative");
  }

  const size_t nx = xvals_.size();
  double val = 0.0;
  for( size_t ky=0; ky<=order_; ++ky ){
    const size_t i = ilo + (ky+jlo)*nx;
    for( size_t kx=0; kx<=order_; ++kx ){
      val += fvals_[kx+i] * xcoefs[kx] * ycoefs[ky];
    }
  }
  return val;
}

//-------------------------------------------------------------------

LagrangeInterpolant2D::~LagrangeInterpolant2D()
{}

//-------------------------------------------------------------------

vector<pair<double,double> >
LagrangeInterpolant2D::get_bounds() const
{
  vector<pair<double,double> > b;
  b.push_back( bounds(xvals_) );
  b.push_back( bounds(yvals_) );
  return b;
}

//-------------------------------------------------------------------

bool
LagrangeInterpolant2D::operator==( const LagrangeInterpolant& other ) const
{
  const LagrangeInterpolant2D& a = dynamic_cast<const LagrangeInterpolant2D&>(other);
  return order_ == a.get_order()
      && allowClipping_ == a.clipping()
      && vec_compare( xvals_, a.xvals_ )
      && vec_compare( yvals_, a.yvals_ )
      && vec_compare( fvals_, a.fvals_ )
      && xbounds_ == a.xbounds_
      && ybounds_ == a.ybounds_;
}

//-------------------------------------------------------------------

template<typename Archive> void
LagrangeInterpolant2D::serialize( Archive& ar, const unsigned version )
{
  ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP( LagrangeInterpolant )
     & BOOST_SERIALIZATION_NVP( xvals_ )
     & BOOST_SERIALIZATION_NVP( yvals_ )
     & BOOST_SERIALIZATION_NVP( fvals_ )
     & BOOST_SERIALIZATION_NVP( isUniformX_ )
     & BOOST_SERIALIZATION_NVP( isUniformY_ );
  xbounds_ = bounds(xvals_);
  ybounds_ = bounds(yvals_);
  xcoef1d_.resize( order_+1 );
  ycoef1d_.resize( order_+1 );
}

//===================================================================

LagrangeInterpolant3D::LagrangeInterpolant3D( const unsigned order,
                                              const vector<double>& xvals,
                                              const vector<double>& yvals,
                                              const vector<double>& zvals,
                                              const vector<double>& fvals,
                                              const bool clipValues )
: LagrangeInterpolant(order,clipValues),
  xbounds_( bounds(xvals) ),
  ybounds_( bounds(yvals) ),
  zbounds_( bounds(zvals) ),
  xvals_( xvals ),
  yvals_( yvals ),
  zvals_( zvals ),
  fvals_( fvals ),
  isUniformX_( is_uniform(xvals) ),
  isUniformY_( is_uniform(yvals) ),
  isUniformZ_( is_uniform(zvals) )
{
  assert( fvals_.size() == xvals_.size() * yvals_.size() * zvals_.size() );

  enforce_incresing( xvals_, "3D interpolant independent variable 1" );
  enforce_incresing( yvals_, "3D interpolant independent variable 2" );
  enforce_incresing( zvals_, "3D interpolant independent variable 3" );

  xcoef1d_.resize( order_+1 );
  ycoef1d_.resize( order_+1 );
  zcoef1d_.resize( order_+1 );
}

//-------------------------------------------------------------------

LagrangeInterpolant3D::LagrangeInterpolant3D( const LagrangeInterpolant3D& other )
: LagrangeInterpolant(other.order_,other.allowClipping_),
  xbounds_( other.xbounds_ ),
  ybounds_( other.ybounds_ ),
  zbounds_( other.zbounds_ ),
  xvals_( other.xvals_ ),
  yvals_( other.yvals_ ),
  zvals_( other.zvals_ ),
  fvals_( other.fvals_ ),
  isUniformX_( other.isUniformX_ ),
  isUniformY_( other.isUniformY_ ),
  isUniformZ_( other.isUniformZ_ )
{
  assert( fvals_.size() == xvals_.size() * yvals_.size() * zvals_.size() );

  xcoef1d_.resize( order_+1 );
  ycoef1d_.resize( order_+1 );
  zcoef1d_.resize( order_+1 );
}

//-------------------------------------------------------------------

LagrangeInterpolant3D::LagrangeInterpolant3D()
: LagrangeInterpolant(1,true)
{}

//-------------------------------------------------------------------

double
LagrangeInterpolant3D::value( const double* const indep ) const
{
  const size_t nx = xvals_.size();
  const size_t ny = yvals_.size();

  const double x = allowClipping_ ? std::min( std::max( indep[0], xbounds_.first ), xbounds_.second ) : indep[0];
  const double y = allowClipping_ ? std::min( std::max( indep[1], ybounds_.first ), ybounds_.second ) : indep[1];
  const double z = allowClipping_ ? std::min( std::max( indep[2], zbounds_.first ), zbounds_.second ) : indep[2];

  const size_t ilo = basis_funs_1d( xvals_, x, order_, xcoef1d_, isUniformX_ );
  const size_t jlo = basis_funs_1d( yvals_, y, order_, ycoef1d_, isUniformY_ );
  const size_t klo = basis_funs_1d( zvals_, z, order_, zcoef1d_, isUniformZ_ );

  double val = 0.0;

  for( size_t k=0; k<=order_; ++k ){
    const size_t kix = (klo+k)*nx*ny;
    for( size_t j=0; j<=order_; ++j ){
      const size_t jix = (jlo+j)*nx;
      const double tmp = ycoef1d_[j] * zcoef1d_[k];
      for( size_t i=0; i<=order_; ++i ){
        const size_t ix = ilo+i + jix + kix;
        val += fvals_[ix] * xcoef1d_[i] * tmp;
      }
    }
  }

  return val;
}

//-------------------------------------------------------------------

double
LagrangeInterpolant3D::derivative( const double* const indep,
                                   const int dim ) const
{
  const size_t nx = xvals_.size();
  const size_t ny = yvals_.size();

  const double x = allowClipping_ ? std::min( std::max( indep[0], xbounds_.first ), xbounds_.second ) : indep[0];
  const double y = allowClipping_ ? std::min( std::max( indep[1], ybounds_.first ), ybounds_.second ) : indep[1];
  const double z = allowClipping_ ? std::min( std::max( indep[2], zbounds_.first ), zbounds_.second ) : indep[2];

  size_t ilo, jlo, klo;
  if( dim==0 ){
    ilo = der_basis_funs_1d( xvals_, x, order_, xcoef1d_, isUniformX_ );
    jlo =     basis_funs_1d( yvals_, y, order_, ycoef1d_, isUniformY_ );
    klo =     basis_funs_1d( zvals_, z, order_, zcoef1d_, isUniformZ_ );
  }
  else if( dim==1 ){
    ilo =     basis_funs_1d( xvals_, x, order_, xcoef1d_, isUniformX_ );
    jlo = der_basis_funs_1d( yvals_, y, order_, ycoef1d_, isUniformY_ );
    klo =     basis_funs_1d( zvals_, z, order_, zcoef1d_, isUniformZ_ );
  }
  else{
    assert( dim==2 );
    ilo =     basis_funs_1d( xvals_, x, order_, xcoef1d_, isUniformX_ );
    jlo =     basis_funs_1d( yvals_, y, order_, ycoef1d_, isUniformY_ );
    klo = der_basis_funs_1d( zvals_, z, order_, zcoef1d_, isUniformZ_ );
  }
  double val = 0.0;

  for( size_t k=0; k<=order_; ++k ){
    const size_t kix = (klo+k)*nx*ny;
    for( size_t j=0; j<=order_; ++j ){
      const size_t jix = (jlo+j)*nx;
      const double tmp = ycoef1d_[j] * zcoef1d_[k];
      for( size_t i=0; i<=order_; ++i ){
        const size_t ix = ilo+i + jix + kix;
        val += fvals_[ix] * xcoef1d_[i] * tmp;
      }
    }
  }

  return val;
}

//-------------------------------------------------------------------

double
LagrangeInterpolant3D::second_derivative( const double* const indep, const int dim1, const int dim2 ) const
{
  if( order_ == 1 ) return 0;

  assert( dim1 < 3 );
  assert( dim2 < 3 );

  const double x = allowClipping_ ? std::min( std::max( indep[0], xbounds_.first ), xbounds_.second ) : indep[0];
  const double y = allowClipping_ ? std::min( std::max( indep[1], ybounds_.first ), ybounds_.second ) : indep[1];
  const double z = allowClipping_ ? std::min( std::max( indep[2], zbounds_.first ), zbounds_.second ) : indep[2];

  double xcoefs[order_+1], ycoefs[order_+1], zcoefs[order_+1];

  size_t ilo=0, jlo=0, klo=0;

  switch( dim1 ){
    case 0:{
      switch( dim2 ){
        case 0:{
          ilo = second_der_basis_funs_1d( xvals_, x, order_, xcoefs, isUniformX_ );
          jlo =            basis_funs_1d( yvals_, y, order_, ycoefs, isUniformY_ );
          klo =            basis_funs_1d( zvals_, z, order_, zcoefs, isUniformZ_ );
          break;
        }
        case 1:{
          ilo = der_basis_funs_1d( xvals_, x, order_, xcoefs, isUniformX_ );
          jlo = der_basis_funs_1d( yvals_, y, order_, ycoefs, isUniformY_ );
          klo =     basis_funs_1d( zvals_, z, order_, zcoefs, isUniformZ_ );
          break;
        }
        case 2:{
          ilo = der_basis_funs_1d( xvals_, x, order_, xcoefs, isUniformX_ );
          jlo =     basis_funs_1d( yvals_, y, order_, ycoefs, isUniformY_ );
          klo = der_basis_funs_1d( zvals_, z, order_, zcoefs, isUniformZ_ );
          break;
        }
        break;
      }
      break;
    }
    case 1:{
      switch( dim2 ){
        case 0:{
          ilo = der_basis_funs_1d( xvals_, x, order_, xcoefs, isUniformX_ );
          jlo = der_basis_funs_1d( yvals_, y, order_, ycoefs, isUniformY_ );
          klo =     basis_funs_1d( zvals_, z, order_, zcoefs, isUniformZ_ );
          break;
        }
        case 1:{
          ilo =            basis_funs_1d( xvals_, x, order_, xcoefs, isUniformX_ );
          jlo = second_der_basis_funs_1d( yvals_, y, order_, ycoefs, isUniformY_ );
          klo =            basis_funs_1d( zvals_, z, order_, zcoefs, isUniformZ_ );
          break;
        }
        case 2:{
          ilo =     basis_funs_1d( xvals_, x, order_, xcoefs, isUniformX_ );
          jlo = der_basis_funs_1d( yvals_, y, order_, ycoefs, isUniformY_ );
          klo = der_basis_funs_1d( zvals_, z, order_, zcoefs, isUniformZ_ );

        }
        break;
      }
      break;
    }
    case 2:{
      switch( dim2 ){
        case 0:{
          ilo = der_basis_funs_1d( xvals_, x, order_, xcoefs, isUniformX_ );
          jlo =     basis_funs_1d( yvals_, y, order_, ycoefs, isUniformY_ );
          klo = der_basis_funs_1d( zvals_, z, order_, zcoefs, isUniformZ_ );
          break;
        }
        case 1:{
          ilo =     basis_funs_1d( xvals_, x, order_, xcoefs, isUniformX_ );
          jlo = der_basis_funs_1d( yvals_, y, order_, ycoefs, isUniformY_ );
          klo = der_basis_funs_1d( zvals_, z, order_, zcoefs, isUniformZ_ );
          break;
        }
        case 2:{
          ilo =            basis_funs_1d( xvals_, x, order_, xcoefs, isUniformX_ );
          jlo =            basis_funs_1d( yvals_, y, order_, ycoefs, isUniformY_ );
          klo = second_der_basis_funs_1d( zvals_, z, order_, zcoefs, isUniformZ_ );
        }
        break;
      }
      break;
    }
    default:
      throw std::invalid_argument("invalid dimension specification for second derivative");
  }

  const size_t nx = xvals_.size();
  const size_t ny = yvals_.size();
  double val = 0.0;
  for( size_t kz=0; kz<=order_; ++kz ){
    const size_t k = klo + kz;
    for( size_t ky=0; ky<=order_; ++ky ){
      const size_t j = jlo + ky;
      for( size_t kx=0; kx<=order_; ++kx ){
        val += fvals_[ilo+kx + j*nx + k*nx*ny] * xcoefs[kx]*ycoefs[ky]*zcoefs[kz];
      }
    }
  }
  return val;
}

//-------------------------------------------------------------------

LagrangeInterpolant3D::~LagrangeInterpolant3D()
{}

//-------------------------------------------------------------------

vector<pair<double,double> >
LagrangeInterpolant3D::get_bounds() const
{
  vector<pair<double,double> > bounds;
  bounds.push_back( xbounds_ );
  bounds.push_back( ybounds_ );
  bounds.push_back( zbounds_ );
  return bounds;
}

//-------------------------------------------------------------------

bool
LagrangeInterpolant3D::operator==( const LagrangeInterpolant& other ) const
{
  const LagrangeInterpolant3D& a = dynamic_cast<const LagrangeInterpolant3D&>(other);

  return order_ == a.get_order()
      && allowClipping_ == a.clipping()
      && vec_compare(xvals_, a.xvals_)
      && vec_compare(yvals_, a.yvals_)
      && vec_compare(zvals_, a.zvals_)
      && vec_compare(fvals_, a.fvals_)
      && xbounds_ == a.xbounds_
      && ybounds_ == a.ybounds_
      && zbounds_ == a.zbounds_;
}

//-------------------------------------------------------------------

template<typename Archive> void
LagrangeInterpolant3D::serialize( Archive& ar, const unsigned version )
{
  ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP( LagrangeInterpolant )
     & BOOST_SERIALIZATION_NVP( xvals_ )
     & BOOST_SERIALIZATION_NVP( yvals_ )
     & BOOST_SERIALIZATION_NVP( zvals_ )
     & BOOST_SERIALIZATION_NVP( fvals_ )
     & BOOST_SERIALIZATION_NVP( isUniformX_)
     & BOOST_SERIALIZATION_NVP( isUniformY_)
     & BOOST_SERIALIZATION_NVP( isUniformZ_);

  xbounds_ = bounds(xvals_);
  ybounds_ = bounds(yvals_);
  zbounds_ = bounds(zvals_);

  xcoef1d_.resize( order_+1 );
  ycoef1d_.resize( order_+1 );
  zcoef1d_.resize( order_+1 );
}

//===================================================================

LagrangeInterpolant4D::LagrangeInterpolant4D( const unsigned order,
                                              const vector<double>& x1vals,
                                              const vector<double>& x2vals,
                                              const vector<double>& x3vals,
                                              const vector<double>& x4vals,
                                              const vector<double>& fvals,
                                              const bool clipValues )
: LagrangeInterpolant(order,clipValues),
  x1vals_( x1vals ),
  x2vals_( x2vals ),
  x3vals_( x3vals ),
  x4vals_( x4vals ),
  fvals_ ( fvals  )
{
  assert( fvals_.size() == x1vals_.size() * x2vals_.size() * x3vals_.size() * x4vals_.size() );

  enforce_incresing( x1vals_, "4D interpolant independent variable 1" );
  enforce_incresing( x2vals_, "4D interpolant independent variable 2" );
  enforce_incresing( x3vals_, "4D interpolant independent variable 3" );
  enforce_incresing( x4vals_, "4D interpolant independent variable 4" );

  x1coef1d_.resize( order_+1 );
  x2coef1d_.resize( order_+1 );
  x3coef1d_.resize( order_+1 );
  x4coef1d_.resize( order_+1 );

  isUniform_[0] = is_uniform(x1vals);
  isUniform_[1] = is_uniform(x2vals);
  isUniform_[2] = is_uniform(x3vals);
  isUniform_[3] = is_uniform(x4vals);

  bounds_.push_back( bounds(x1vals) );
  bounds_.push_back( bounds(x2vals) );
  bounds_.push_back( bounds(x3vals) );
  bounds_.push_back( bounds(x4vals) );
}

//-------------------------------------------------------------------

LagrangeInterpolant4D::LagrangeInterpolant4D( const LagrangeInterpolant4D& other )
: LagrangeInterpolant(other.order_,other.allowClipping_),
  bounds_( other.bounds_ ),
  x1vals_( other.x1vals_ ),
  x2vals_( other.x2vals_ ),
  x3vals_( other.x3vals_ ),
  x4vals_( other.x4vals_ ),
  fvals_ ( other.fvals_  )
{
  assert( fvals_.size() == x1vals_.size() * x2vals_.size() * x3vals_.size() * x4vals_.size() );

  x1coef1d_.resize( order_+1 );
  x2coef1d_.resize( order_+1 );
  x3coef1d_.resize( order_+1 );
  x4coef1d_.resize( order_+1 );

  for( short i=0; i<4; ++i ) isUniform_[i] = other.isUniform_[i];
}

//-------------------------------------------------------------------

LagrangeInterpolant4D::LagrangeInterpolant4D()
: LagrangeInterpolant(1,true)
{}

//-------------------------------------------------------------------

double
LagrangeInterpolant4D::value( const double* const indep ) const
{
  const size_t nx1 = x1vals_.size();
  const size_t nx2 = x2vals_.size();
  const size_t nx3 = x3vals_.size();
  const size_t nx4 = x4vals_.size();

  const double x1 = allowClipping_ ? std::min( std::max( indep[0], bounds_[0].first ), bounds_[0].second ) : indep[0];
  const double x2 = allowClipping_ ? std::min( std::max( indep[1], bounds_[1].first ), bounds_[1].second ) : indep[1];
  const double x3 = allowClipping_ ? std::min( std::max( indep[2], bounds_[2].first ), bounds_[2].second ) : indep[2];
  const double x4 = allowClipping_ ? std::min( std::max( indep[3], bounds_[3].first ), bounds_[3].second ) : indep[3];

  const size_t i1lo = basis_funs_1d( x1vals_, x1, order_, x1coef1d_, isUniform_[0] );
  const size_t i2lo = basis_funs_1d( x2vals_, x2, order_, x2coef1d_, isUniform_[1] );
  const size_t i3lo = basis_funs_1d( x3vals_, x3, order_, x3coef1d_, isUniform_[2] );
  const size_t i4lo = basis_funs_1d( x4vals_, x4, order_, x4coef1d_, isUniform_[3] );

  double val = 0.0;

  for( size_t k4=0; k4<=order_; ++k4 ){
    const size_t i4 = (i4lo+k4)*nx1*nx2*nx3;
    for( size_t k3=0; k3<=order_; ++k3 ){
      const double tmp = x3coef1d_[k3] * x4coef1d_[k4];
      const size_t i3 = (i3lo+k3)*nx1*nx2;
      for( size_t k2=0; k2<=order_; ++k2 ){
        const double tmp2 = x2coef1d_[k2] * tmp;
        const size_t i2 = (i2lo+k2)*nx1;
        for( size_t k1=0; k1<=order_; ++k1 ){
          const size_t i1 = i1lo+k1 + i2 + i3 + i4;
          assert( i1 < fvals_.size() );
          val += fvals_[i1] * x1coef1d_[k1] * tmp2;
        }
      }
    }
  }
  return val;
}

//-------------------------------------------------------------------

double
LagrangeInterpolant4D::derivative( const double* const indep,
                                   const int dim ) const
{
  const size_t nx1 = x1vals_.size();
  const size_t nx2 = x2vals_.size();
  const size_t nx3 = x3vals_.size();
  const size_t nx4 = x4vals_.size();

  const double x1 = allowClipping_ ? std::min( std::max( indep[0], bounds_[0].first ), bounds_[0].second ) : indep[0];
  const double x2 = allowClipping_ ? std::min( std::max( indep[1], bounds_[1].first ), bounds_[1].second ) : indep[1];
  const double x3 = allowClipping_ ? std::min( std::max( indep[2], bounds_[2].first ), bounds_[2].second ) : indep[2];
  const double x4 = allowClipping_ ? std::min( std::max( indep[3], bounds_[3].first ), bounds_[3].second ) : indep[3];

  size_t i1lo, i2lo, i3lo, i4lo;
  switch( dim ){
  case 0:
    i1lo = der_basis_funs_1d( x1vals_, x1, order_, x1coef1d_, isUniform_[0] );
    i2lo =     basis_funs_1d( x2vals_, x2, order_, x2coef1d_, isUniform_[1] );
    i3lo =     basis_funs_1d( x3vals_, x3, order_, x3coef1d_, isUniform_[2] );
    i4lo =     basis_funs_1d( x4vals_, x4, order_, x4coef1d_, isUniform_[3] );
    break;
  case 1:
    i1lo =     basis_funs_1d( x1vals_, x1, order_, x1coef1d_, isUniform_[0]);
    i2lo = der_basis_funs_1d( x2vals_, x2, order_, x2coef1d_, isUniform_[1]);
    i3lo =     basis_funs_1d( x3vals_, x3, order_, x3coef1d_, isUniform_[2]);
    i4lo =     basis_funs_1d( x4vals_, x4, order_, x4coef1d_, isUniform_[3]);
    break;
  case 2:
    i1lo =     basis_funs_1d( x1vals_, x1, order_, x1coef1d_, isUniform_[0]);
    i2lo =     basis_funs_1d( x2vals_, x2, order_, x2coef1d_, isUniform_[1]);
    i3lo = der_basis_funs_1d( x3vals_, x3, order_, x3coef1d_, isUniform_[2]);
    i4lo =     basis_funs_1d( x4vals_, x4, order_, x4coef1d_, isUniform_[3]);
    break;
  default:
    assert( dim==3 );
    i1lo =     basis_funs_1d( x1vals_, x1, order_, x1coef1d_, isUniform_[0] );
    i2lo =     basis_funs_1d( x2vals_, x2, order_, x2coef1d_, isUniform_[1] );
    i3lo =     basis_funs_1d( x3vals_, x3, order_, x3coef1d_, isUniform_[2] );
    i4lo = der_basis_funs_1d( x4vals_, x4, order_, x4coef1d_, isUniform_[3] );
  } // switch(dim)

  double val = 0.0;

  for( size_t k4=0; k4<=order_; ++k4 ){
    const size_t i4 = (i4lo+k4)*nx1*nx2*nx3;
    for( size_t k3=0; k3<=order_; ++k3 ){
      const double tmp = x3coef1d_[k3] * x4coef1d_[k4];
      const size_t i3 = (i3lo+k3)*nx1*nx2;
      for( size_t k2=0; k2<=order_; ++k2 ){
        const double tmp2 = x2coef1d_[k2] * tmp;
        const size_t i2 = (i2lo+k2)*nx1;
        for( size_t k1=0; k1<=order_; ++k1 ){
          const size_t i1 = i1lo+k1 + i2 + i3 + i4;
#         ifndef NDEBUG
          assert( i1 < fvals_.size() );
#         endif
          val += fvals_[i1] * x1coef1d_[k1] * tmp2;
        }
      }
    }
  }
  return val;
}

//-------------------------------------------------------------------

double
LagrangeInterpolant4D::second_derivative( const double* const indep, const int dim1, const int dim2 ) const
{
  if( order_ == 1 ) return 0;

  assert( dim1 < 4 );
  assert( dim2 < 4 );

  const double x1 = allowClipping_ ? std::min( std::max( indep[0], bounds_[0].first ), bounds_[0].second ) : indep[0];
  const double x2 = allowClipping_ ? std::min( std::max( indep[1], bounds_[1].first ), bounds_[1].second ) : indep[1];
  const double x3 = allowClipping_ ? std::min( std::max( indep[2], bounds_[2].first ), bounds_[2].second ) : indep[2];
  const double x4 = allowClipping_ ? std::min( std::max( indep[3], bounds_[3].first ), bounds_[3].second ) : indep[3];

  size_t i1lo, i2lo, i3lo, i4lo;

  switch( dim1 ){
    case 0:{
      switch( dim2 ){
        case 0:
          i1lo = second_der_basis_funs_1d( x1vals_, x1, order_, x1coef1d_, isUniform_[0] );
          i2lo =            basis_funs_1d( x2vals_, x2, order_, x2coef1d_, isUniform_[1] );
          i3lo =            basis_funs_1d( x3vals_, x3, order_, x3coef1d_, isUniform_[2] );
          i4lo =            basis_funs_1d( x4vals_, x4, order_, x4coef1d_, isUniform_[3] );
          break;
        case 1:
          i1lo = der_basis_funs_1d( x1vals_, x1, order_, x1coef1d_, isUniform_[0] );
          i2lo = der_basis_funs_1d( x2vals_, x2, order_, x2coef1d_, isUniform_[1] );
          i3lo =     basis_funs_1d( x3vals_, x3, order_, x3coef1d_, isUniform_[2] );
          i4lo =     basis_funs_1d( x4vals_, x4, order_, x4coef1d_, isUniform_[3] );
          break;
        case 2:
          i1lo = der_basis_funs_1d( x1vals_, x1, order_, x1coef1d_, isUniform_[0] );
          i2lo =     basis_funs_1d( x2vals_, x2, order_, x2coef1d_, isUniform_[1] );
          i3lo = der_basis_funs_1d( x3vals_, x3, order_, x3coef1d_, isUniform_[2] );
          i4lo =     basis_funs_1d( x4vals_, x4, order_, x4coef1d_, isUniform_[3] );
          break;
        case 3:
          i1lo = der_basis_funs_1d( x1vals_, x1, order_, x1coef1d_, isUniform_[0] );
          i2lo =     basis_funs_1d( x2vals_, x2, order_, x2coef1d_, isUniform_[1] );
          i3lo =     basis_funs_1d( x3vals_, x3, order_, x3coef1d_, isUniform_[2] );
          i4lo = der_basis_funs_1d( x4vals_, x4, order_, x4coef1d_, isUniform_[3] );
          break;
      }
      break;
    } // case 0 on switch(dim1)
    case 1:{
      switch( dim2 ){
        case 0:
          i1lo = der_basis_funs_1d( x1vals_, x1, order_, x1coef1d_, isUniform_[0] );
          i2lo = der_basis_funs_1d( x2vals_, x2, order_, x2coef1d_, isUniform_[1] );
          i3lo =     basis_funs_1d( x3vals_, x3, order_, x3coef1d_, isUniform_[2] );
          i4lo =     basis_funs_1d( x4vals_, x4, order_, x4coef1d_, isUniform_[3] );
          break;
        case 1:
          i1lo =            basis_funs_1d( x1vals_, x1, order_, x1coef1d_, isUniform_[0] );
          i2lo = second_der_basis_funs_1d( x2vals_, x2, order_, x2coef1d_, isUniform_[1] );
          i3lo =            basis_funs_1d( x3vals_, x3, order_, x3coef1d_, isUniform_[2] );
          i4lo =            basis_funs_1d( x4vals_, x4, order_, x4coef1d_, isUniform_[3] );
          break;
        case 2:
          i1lo =     basis_funs_1d( x1vals_, x1, order_, x1coef1d_, isUniform_[0] );
          i2lo = der_basis_funs_1d( x2vals_, x2, order_, x2coef1d_, isUniform_[1] );
          i3lo = der_basis_funs_1d( x3vals_, x3, order_, x3coef1d_, isUniform_[2] );
          i4lo =     basis_funs_1d( x4vals_, x4, order_, x4coef1d_, isUniform_[3] );
          break;
        case 3:
          i1lo =     basis_funs_1d( x1vals_, x1, order_, x1coef1d_, isUniform_[0] );
          i2lo = der_basis_funs_1d( x2vals_, x2, order_, x2coef1d_, isUniform_[1] );
          i3lo =     basis_funs_1d( x3vals_, x3, order_, x3coef1d_, isUniform_[2] );
          i4lo = der_basis_funs_1d( x4vals_, x4, order_, x4coef1d_, isUniform_[3] );
          break;
      }
      break;
    } // case 1 on switch(dim1)
    case 2:{
      switch( dim2 ){
        case 0:
          i1lo = der_basis_funs_1d( x1vals_, x1, order_, x1coef1d_, isUniform_[0] );
          i2lo =     basis_funs_1d( x2vals_, x2, order_, x2coef1d_, isUniform_[1] );
          i3lo = der_basis_funs_1d( x3vals_, x3, order_, x3coef1d_, isUniform_[2] );
          i4lo =     basis_funs_1d( x4vals_, x4, order_, x4coef1d_, isUniform_[3] );
          break;
        case 1:
          i1lo =     basis_funs_1d( x1vals_, x1, order_, x1coef1d_, isUniform_[0] );
          i2lo = der_basis_funs_1d( x2vals_, x2, order_, x2coef1d_, isUniform_[1] );
          i3lo = der_basis_funs_1d( x3vals_, x3, order_, x3coef1d_, isUniform_[2] );
          i4lo =     basis_funs_1d( x4vals_, x4, order_, x4coef1d_, isUniform_[3] );
          break;
        case 2:
          i1lo =            basis_funs_1d( x1vals_, x1, order_, x1coef1d_, isUniform_[0] );
          i2lo =            basis_funs_1d( x2vals_, x2, order_, x2coef1d_, isUniform_[1] );
          i3lo = second_der_basis_funs_1d( x3vals_, x3, order_, x3coef1d_, isUniform_[2] );
          i4lo =            basis_funs_1d( x4vals_, x4, order_, x4coef1d_, isUniform_[3] );
          break;
        case 3:
          i1lo =     basis_funs_1d( x1vals_, x1, order_, x1coef1d_, isUniform_[0] );
          i2lo =     basis_funs_1d( x2vals_, x2, order_, x2coef1d_, isUniform_[1] );
          i3lo = der_basis_funs_1d( x3vals_, x3, order_, x3coef1d_, isUniform_[2] );
          i4lo = der_basis_funs_1d( x4vals_, x4, order_, x4coef1d_, isUniform_[3] );
          break;
      }
      break;
    } // case 2 on switch(dim1)
    case 3:{
      switch( dim2 ){
        case 0:
          i1lo = der_basis_funs_1d( x1vals_, x1, order_, x1coef1d_, isUniform_[0] );
          i2lo =     basis_funs_1d( x2vals_, x2, order_, x2coef1d_, isUniform_[1] );
          i3lo =     basis_funs_1d( x3vals_, x3, order_, x3coef1d_, isUniform_[2] );
          i4lo = der_basis_funs_1d( x4vals_, x4, order_, x4coef1d_, isUniform_[3] );
          break;
        case 1:
          i1lo =     basis_funs_1d( x1vals_, x1, order_, x1coef1d_, isUniform_[0] );
          i2lo = der_basis_funs_1d( x2vals_, x2, order_, x2coef1d_, isUniform_[1] );
          i3lo =     basis_funs_1d( x3vals_, x3, order_, x3coef1d_, isUniform_[2] );
          i4lo = der_basis_funs_1d( x4vals_, x4, order_, x4coef1d_, isUniform_[3] );
          break;
        case 2:
          i1lo =     basis_funs_1d( x1vals_, x1, order_, x1coef1d_, isUniform_[0] );
          i2lo =     basis_funs_1d( x2vals_, x2, order_, x2coef1d_, isUniform_[1] );
          i3lo = der_basis_funs_1d( x3vals_, x3, order_, x3coef1d_, isUniform_[2] );
          i4lo = der_basis_funs_1d( x4vals_, x4, order_, x4coef1d_, isUniform_[3] );
          break;
        case 3:
          i1lo =            basis_funs_1d( x1vals_, x1, order_, x1coef1d_, isUniform_[0] );
          i2lo =            basis_funs_1d( x2vals_, x2, order_, x2coef1d_, isUniform_[1] );
          i3lo =            basis_funs_1d( x3vals_, x3, order_, x3coef1d_, isUniform_[2] );
          i4lo = second_der_basis_funs_1d( x4vals_, x4, order_, x4coef1d_, isUniform_[3] );
          break;
      }
      break;
    } // case 3 on switch(dim1)
    default:
      throw std::invalid_argument("invalid dimension specification for second derivative");
  }

  const size_t nx1 = x1vals_.size();
  const size_t nx2 = x2vals_.size();
  const size_t nx3 = x3vals_.size();

  double val = 0.0;

  for( size_t k4=0; k4<=order_; ++k4 ){
    const size_t i4 = (i4lo+k4)*nx1*nx2*nx3;
    for( size_t k3=0; k3<=order_; ++k3 ){
      const double tmp = x3coef1d_[k3] * x4coef1d_[k4];
      const size_t i3 = (i3lo+k3)*nx1*nx2;
      for( size_t k2=0; k2<=order_; ++k2 ){
        const double tmp2 = x2coef1d_[k2] * tmp;
        const size_t i2 = (i2lo+k2)*nx1;
        for( size_t k1=0; k1<=order_; ++k1 ){
          const size_t i1 = i1lo+k1 + i2 + i3 + i4;
#         ifndef NDEBUG
          assert( i1 < fvals_.size() );
#         endif
          val += fvals_[i1] * x1coef1d_[k1] * tmp2;
        }
      }
    }
  }
  return val;
}

//-------------------------------------------------------------------

LagrangeInterpolant4D::~LagrangeInterpolant4D()
{}

//-------------------------------------------------------------------

vector<pair<double,double> >
LagrangeInterpolant4D::get_bounds() const
{
  return bounds_;
}

//-------------------------------------------------------------------

bool
LagrangeInterpolant4D::operator==( const LagrangeInterpolant& other ) const
{
  const LagrangeInterpolant4D& a = dynamic_cast<const LagrangeInterpolant4D&>(other);
  return order_ == a.get_order()
      && allowClipping_ == a.clipping()
      && vec_compare( x1vals_, a.x1vals_ )
      && vec_compare( x2vals_, a.x2vals_ )
      && vec_compare( x3vals_, a.x3vals_ )
      && vec_compare( x4vals_, a.x4vals_ )
      && vec_compare( fvals_ , a.fvals_  )
      && bounds_ == a.bounds_;
}

//-------------------------------------------------------------------

template<typename Archive> void
LagrangeInterpolant4D::serialize( Archive& ar, const unsigned version )
{
  ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP( LagrangeInterpolant )
     & BOOST_SERIALIZATION_NVP( x1vals_ )
     & BOOST_SERIALIZATION_NVP( x2vals_ )
     & BOOST_SERIALIZATION_NVP( x3vals_ )
     & BOOST_SERIALIZATION_NVP( x4vals_ )
     & BOOST_SERIALIZATION_NVP( fvals_  )
     & BOOST_SERIALIZATION_NVP( isUniform_ );

  bounds_.clear();
  bounds_.push_back( bounds(x1vals_) );
  bounds_.push_back( bounds(x2vals_) );
  bounds_.push_back( bounds(x3vals_) );
  bounds_.push_back( bounds(x4vals_) );

  x1coef1d_.resize( order_+1 );
  x2coef1d_.resize( order_+1 );
  x3coef1d_.resize( order_+1 );
  x4coef1d_.resize( order_+1 );
}

//===================================================================

LagrangeInterpolant5D::LagrangeInterpolant5D( const unsigned order,
                                              const vector<double>& x1vals,
                                              const vector<double>& x2vals,
                                              const vector<double>& x3vals,
                                              const vector<double>& x4vals,
                                              const vector<double>& x5vals,
                                              const vector<double>& fvals,
                                              const bool clipValues )
: LagrangeInterpolant(order,clipValues),
  x1vals_( x1vals ),
  x2vals_( x2vals ),
  x3vals_( x3vals ),
  x4vals_( x4vals ),
  x5vals_( x5vals ),
  fvals_ ( fvals  )
{
  assert( fvals_.size() == x1vals_.size() * x2vals_.size() * x3vals_.size() * x4vals_.size() * x5vals_.size() );

  enforce_incresing( x1vals_, "4D interpolant independent variable 1" );
  enforce_incresing( x2vals_, "4D interpolant independent variable 2" );
  enforce_incresing( x3vals_, "4D interpolant independent variable 3" );
  enforce_incresing( x4vals_, "4D interpolant independent variable 4" );
  enforce_incresing( x5vals_, "4D interpolant independent variable 5" );

  x1coef1d_.resize( order_+1 );
  x2coef1d_.resize( order_+1 );
  x3coef1d_.resize( order_+1 );
  x4coef1d_.resize( order_+1 );
  x5coef1d_.resize( order_+1 );

  bounds_.push_back( bounds(x1vals_) );
  bounds_.push_back( bounds(x2vals_) );
  bounds_.push_back( bounds(x3vals_) );
  bounds_.push_back( bounds(x4vals_) );
  bounds_.push_back( bounds(x5vals_) );

  isUniform_[0] = is_uniform(x1vals);
  isUniform_[1] = is_uniform(x2vals);
  isUniform_[2] = is_uniform(x3vals);
  isUniform_[3] = is_uniform(x4vals);
  isUniform_[4] = is_uniform(x5vals);
}

//-------------------------------------------------------------------

LagrangeInterpolant5D::LagrangeInterpolant5D( const LagrangeInterpolant5D& other )
: LagrangeInterpolant(other.order_,other.allowClipping_),
  bounds_( other.bounds_ ),
  x1vals_( other.x1vals_ ),
  x2vals_( other.x2vals_ ),
  x3vals_( other.x3vals_ ),
  x4vals_( other.x4vals_ ),
  x5vals_( other.x5vals_ ),
  fvals_ ( other.fvals_  )
{
  assert( fvals_.size() == x1vals_.size() * x2vals_.size() * x3vals_.size() * x4vals_.size() * x5vals_.size() );

  x1coef1d_.resize( order_+1 );
  x2coef1d_.resize( order_+1 );
  x3coef1d_.resize( order_+1 );
  x4coef1d_.resize( order_+1 );
  x5coef1d_.resize( order_+1 );

  for( short i=0; i<5; ++i ) isUniform_[i] = other.isUniform_[i];
}

//-------------------------------------------------------------------

LagrangeInterpolant5D::LagrangeInterpolant5D()
: LagrangeInterpolant(1,true)
{}

//-------------------------------------------------------------------

double
LagrangeInterpolant5D::value( const double* const indep ) const
{
  const size_t nx1 = x1vals_.size();
  const size_t nx2 = x2vals_.size();
  const size_t nx3 = x3vals_.size();
  const size_t nx4 = x4vals_.size();
  const size_t nx5 = x5vals_.size();

  const double x1 = allowClipping_ ? std::min( std::max( indep[0], bounds_[0].first ), bounds_[0].second ) : indep[0];
  const double x2 = allowClipping_ ? std::min( std::max( indep[1], bounds_[1].first ), bounds_[1].second ) : indep[1];
  const double x3 = allowClipping_ ? std::min( std::max( indep[2], bounds_[2].first ), bounds_[2].second ) : indep[2];
  const double x4 = allowClipping_ ? std::min( std::max( indep[3], bounds_[3].first ), bounds_[3].second ) : indep[3];
  const double x5 = allowClipping_ ? std::min( std::max( indep[4], bounds_[4].first ), bounds_[4].second ) : indep[4];

  const size_t i1lo = basis_funs_1d( x1vals_, x1, order_, x1coef1d_, isUniform_[0] );
  const size_t i2lo = basis_funs_1d( x2vals_, x2, order_, x2coef1d_, isUniform_[1] );
  const size_t i3lo = basis_funs_1d( x3vals_, x3, order_, x3coef1d_, isUniform_[2] );
  const size_t i4lo = basis_funs_1d( x4vals_, x4, order_, x4coef1d_, isUniform_[3] );
  const size_t i5lo = basis_funs_1d( x5vals_, x5, order_, x5coef1d_, isUniform_[4] );

  double val = 0.0;

  for( size_t k5=0; k5<=order_; ++k5 ){
    const size_t i5 = (i5lo+k5)*nx1*nx2*nx3*nx4;
    for( size_t k4=0; k4<=order_; ++k4 ){
      const double tmp = x4coef1d_[k4] * x5coef1d_[k5];
      const size_t i4 = (i4lo+k4)*nx1*nx2*nx3;
      for( size_t k3=0; k3<=order_; ++k3 ){
        const double tmp2 = x3coef1d_[k3] * tmp;
        const size_t i3 = (i3lo+k3)*nx1*nx2;
        for( size_t k2=0; k2<=order_; ++k2 ){
          const double tmp3 = x2coef1d_[k2] * tmp2;
          const size_t i2 = (i2lo+k2)*nx1;
          for( size_t k1=0; k1<=order_; ++k1 ){
            const size_t i1 = i1lo+k1 + i2 + i3 + i4 + i5;
#           ifndef NDEBUG
            assert( i1 < fvals_.size() );
#           endif
            val += fvals_[i1] * x1coef1d_[k1] * tmp3;
          }
        }
      }
    }
  }
  return val;
}

//-------------------------------------------------------------------

double
LagrangeInterpolant5D::derivative( const double* const indep,
                                   const int dim ) const
{
  const size_t nx1 = x1vals_.size();
  const size_t nx2 = x2vals_.size();
  const size_t nx3 = x3vals_.size();
  const size_t nx4 = x4vals_.size();
  const size_t nx5 = x5vals_.size();

  const double x1 = allowClipping_ ? std::min( std::max( indep[0], bounds_[0].first ), bounds_[0].second ) : indep[0];
  const double x2 = allowClipping_ ? std::min( std::max( indep[1], bounds_[1].first ), bounds_[1].second ) : indep[1];
  const double x3 = allowClipping_ ? std::min( std::max( indep[2], bounds_[2].first ), bounds_[2].second ) : indep[2];
  const double x4 = allowClipping_ ? std::min( std::max( indep[3], bounds_[3].first ), bounds_[3].second ) : indep[3];
  const double x5 = allowClipping_ ? std::min( std::max( indep[4], bounds_[4].first ), bounds_[4].second ) : indep[4];

  size_t i1lo, i2lo, i3lo, i4lo, i5lo;

  switch( dim ){
    case 0:
      i1lo = der_basis_funs_1d( x1vals_, x1, order_, x1coef1d_, isUniform_[0] );
      i2lo =     basis_funs_1d( x2vals_, x2, order_, x2coef1d_, isUniform_[1] );
      i3lo =     basis_funs_1d( x3vals_, x3, order_, x3coef1d_, isUniform_[2] );
      i4lo =     basis_funs_1d( x4vals_, x4, order_, x4coef1d_, isUniform_[3] );
      i5lo =     basis_funs_1d( x5vals_, x5, order_, x5coef1d_, isUniform_[4] );
      break;
    case 1:
      i1lo =     basis_funs_1d( x1vals_, x1, order_, x1coef1d_, isUniform_[0] );
      i2lo = der_basis_funs_1d( x2vals_, x2, order_, x2coef1d_, isUniform_[1] );
      i3lo =     basis_funs_1d( x3vals_, x3, order_, x3coef1d_, isUniform_[2] );
      i4lo =     basis_funs_1d( x4vals_, x4, order_, x4coef1d_, isUniform_[3] );
      i5lo =     basis_funs_1d( x5vals_, x5, order_, x5coef1d_, isUniform_[4] );
      break;
    case 2:
      i1lo =     basis_funs_1d( x1vals_, x1, order_, x1coef1d_, isUniform_[0] );
      i2lo =     basis_funs_1d( x2vals_, x2, order_, x2coef1d_, isUniform_[1] );
      i3lo = der_basis_funs_1d( x3vals_, x3, order_, x3coef1d_, isUniform_[2] );
      i4lo =     basis_funs_1d( x4vals_, x4, order_, x4coef1d_, isUniform_[3] );
      i5lo =     basis_funs_1d( x5vals_, x5, order_, x5coef1d_, isUniform_[4] );
      break;
    case 3:
      i1lo =     basis_funs_1d( x1vals_, x1, order_, x1coef1d_, isUniform_[0] );
      i2lo =     basis_funs_1d( x2vals_, x2, order_, x2coef1d_, isUniform_[1] );
      i3lo =     basis_funs_1d( x3vals_, x3, order_, x3coef1d_, isUniform_[2] );
      i4lo = der_basis_funs_1d( x4vals_, x4, order_, x4coef1d_, isUniform_[3] );
      i5lo =     basis_funs_1d( x5vals_, x5, order_, x5coef1d_, isUniform_[4] );
      break;
    case 4:
      i1lo =     basis_funs_1d( x1vals_, x1, order_, x1coef1d_, isUniform_[0] );
      i2lo =     basis_funs_1d( x2vals_, x2, order_, x2coef1d_, isUniform_[1] );
      i3lo =     basis_funs_1d( x3vals_, x3, order_, x3coef1d_, isUniform_[2] );
      i4lo =     basis_funs_1d( x4vals_, x4, order_, x4coef1d_, isUniform_[3] );
      i5lo = der_basis_funs_1d( x5vals_, x5, order_, x5coef1d_, isUniform_[4] );
      break;
    default:
      assert(false);
  }

  double val = 0.0;

  for( size_t k5=0; k5<=order_; ++k5 ){
    const size_t i5 = (i5lo+k5)*nx1*nx2*nx3*nx4;
    for( size_t k4=0; k4<=order_; ++k4 ){
      const double tmp = x4coef1d_[k4] * x5coef1d_[k5];
      const size_t i4 = (i4lo+k4)*nx1*nx2*nx3;
      for( size_t k3=0; k3<=order_; ++k3 ){
        const double tmp2 = x3coef1d_[k3] * tmp;
        const size_t i3 = (i3lo+k3)*nx1*nx2;
        for( size_t k2=0; k2<=order_; ++k2 ){
          const double tmp3 = x2coef1d_[k2] * tmp2;
          const size_t i2 = (i2lo+k2)*nx1;
          for( size_t k1=0; k1<=order_; ++k1 ){
            const size_t i1 = i1lo+k1 + i2 + i3 + i4 + i5;
            assert( i1 < fvals_.size() );
            val += fvals_[i1] * x1coef1d_[k1] * tmp3;
          }
        }
      }
    }
  }
  return val;
}

//-------------------------------------------------------------------

double
LagrangeInterpolant5D::second_derivative( const double* const indep, const int dim1, const int dim2 ) const
{
  if( order_ == 1 ) return 0.0;

  assert( dim1 < 5 );
  assert( dim1 < 5 );

  const double x1 = allowClipping_ ? std::min( std::max( indep[0], bounds_[0].first ), bounds_[0].second ) : indep[0];
  const double x2 = allowClipping_ ? std::min( std::max( indep[1], bounds_[1].first ), bounds_[1].second ) : indep[1];
  const double x3 = allowClipping_ ? std::min( std::max( indep[2], bounds_[2].first ), bounds_[2].second ) : indep[2];
  const double x4 = allowClipping_ ? std::min( std::max( indep[3], bounds_[3].first ), bounds_[3].second ) : indep[3];
  const double x5 = allowClipping_ ? std::min( std::max( indep[4], bounds_[4].first ), bounds_[4].second ) : indep[4];

  size_t i1lo, i2lo, i3lo, i4lo, i5lo;

  switch( dim1 ){
    case 0:{
      switch( dim2 ){
        case 0:
          i1lo = second_der_basis_funs_1d( x1vals_, x1, order_, x1coef1d_, isUniform_[0] );
          i2lo =            basis_funs_1d( x2vals_, x2, order_, x2coef1d_, isUniform_[1] );
          i3lo =            basis_funs_1d( x3vals_, x3, order_, x3coef1d_, isUniform_[2] );
          i4lo =            basis_funs_1d( x4vals_, x4, order_, x4coef1d_, isUniform_[3] );
          i5lo =            basis_funs_1d( x5vals_, x5, order_, x5coef1d_, isUniform_[4] );
          break;
        case 1:
          i1lo = der_basis_funs_1d( x1vals_, x1, order_, x1coef1d_, isUniform_[0] );
          i2lo = der_basis_funs_1d( x2vals_, x2, order_, x2coef1d_, isUniform_[1] );
          i3lo =     basis_funs_1d( x3vals_, x3, order_, x3coef1d_, isUniform_[2] );
          i4lo =     basis_funs_1d( x4vals_, x4, order_, x4coef1d_, isUniform_[3] );
          i5lo =     basis_funs_1d( x5vals_, x5, order_, x5coef1d_, isUniform_[4] );
          break;
        case 2:
          i1lo = der_basis_funs_1d( x1vals_, x1, order_, x1coef1d_, isUniform_[0] );
          i2lo =     basis_funs_1d( x2vals_, x2, order_, x2coef1d_, isUniform_[1] );
          i3lo = der_basis_funs_1d( x3vals_, x3, order_, x3coef1d_, isUniform_[2] );
          i4lo =     basis_funs_1d( x4vals_, x4, order_, x4coef1d_, isUniform_[3] );
          i5lo =     basis_funs_1d( x5vals_, x5, order_, x5coef1d_, isUniform_[4] );
          break;
        case 3:
          i1lo = der_basis_funs_1d( x1vals_, x1, order_, x1coef1d_, isUniform_[0] );
          i2lo =     basis_funs_1d( x2vals_, x2, order_, x2coef1d_, isUniform_[1] );
          i3lo =     basis_funs_1d( x3vals_, x3, order_, x3coef1d_, isUniform_[2] );
          i4lo = der_basis_funs_1d( x4vals_, x4, order_, x4coef1d_, isUniform_[3] );
          i5lo =     basis_funs_1d( x5vals_, x5, order_, x5coef1d_, isUniform_[4] );
          break;
        case 4:
          i1lo = der_basis_funs_1d( x1vals_, x1, order_, x1coef1d_, isUniform_[0] );
          i2lo =     basis_funs_1d( x2vals_, x2, order_, x2coef1d_, isUniform_[1] );
          i3lo =     basis_funs_1d( x3vals_, x3, order_, x3coef1d_, isUniform_[2] );
          i4lo =     basis_funs_1d( x4vals_, x4, order_, x4coef1d_, isUniform_[3] );
          i5lo = der_basis_funs_1d( x5vals_, x5, order_, x5coef1d_, isUniform_[4] );
          break;
      }
      break;
    } // case 0 on switch(dim1)
    case 1:{
      switch( dim2 ){
        case 0:
          i1lo = der_basis_funs_1d( x1vals_, x1, order_, x1coef1d_, isUniform_[0] );
          i2lo = der_basis_funs_1d( x2vals_, x2, order_, x2coef1d_, isUniform_[1] );
          i3lo =     basis_funs_1d( x3vals_, x3, order_, x3coef1d_, isUniform_[2] );
          i4lo =     basis_funs_1d( x4vals_, x4, order_, x4coef1d_, isUniform_[3] );
          i5lo =     basis_funs_1d( x5vals_, x5, order_, x5coef1d_, isUniform_[4] );
          break;
        case 1:
          i1lo =            basis_funs_1d( x1vals_, x1, order_, x1coef1d_, isUniform_[0] );
          i2lo = second_der_basis_funs_1d( x2vals_, x2, order_, x2coef1d_, isUniform_[1] );
          i3lo =            basis_funs_1d( x3vals_, x3, order_, x3coef1d_, isUniform_[2] );
          i4lo =            basis_funs_1d( x4vals_, x4, order_, x4coef1d_, isUniform_[3] );
          i5lo =            basis_funs_1d( x5vals_, x5, order_, x5coef1d_, isUniform_[4] );
          break;
        case 2:
          i1lo =     basis_funs_1d( x1vals_, x1, order_, x1coef1d_, isUniform_[0] );
          i2lo = der_basis_funs_1d( x2vals_, x2, order_, x2coef1d_, isUniform_[1] );
          i3lo = der_basis_funs_1d( x3vals_, x3, order_, x3coef1d_, isUniform_[2] );
          i4lo =     basis_funs_1d( x4vals_, x4, order_, x4coef1d_, isUniform_[3] );
          i5lo =     basis_funs_1d( x5vals_, x5, order_, x5coef1d_, isUniform_[4] );
          break;
        case 3:
          i1lo =     basis_funs_1d( x1vals_, x1, order_, x1coef1d_, isUniform_[0] );
          i2lo = der_basis_funs_1d( x2vals_, x2, order_, x2coef1d_, isUniform_[1] );
          i3lo =     basis_funs_1d( x3vals_, x3, order_, x3coef1d_, isUniform_[2] );
          i4lo = der_basis_funs_1d( x4vals_, x4, order_, x4coef1d_, isUniform_[3] );
          i5lo =     basis_funs_1d( x5vals_, x5, order_, x5coef1d_, isUniform_[4] );
          break;
        case 4:
          i1lo =     basis_funs_1d( x1vals_, x1, order_, x1coef1d_, isUniform_[0] );
          i2lo = der_basis_funs_1d( x2vals_, x2, order_, x2coef1d_, isUniform_[1] );
          i3lo =     basis_funs_1d( x3vals_, x3, order_, x3coef1d_, isUniform_[2] );
          i4lo =     basis_funs_1d( x4vals_, x4, order_, x4coef1d_, isUniform_[3] );
          i5lo = der_basis_funs_1d( x5vals_, x5, order_, x5coef1d_, isUniform_[4] );
          break;
      }
      break;
    } // case 1 on switch(dim1)
    case 2:{
      switch( dim2 ){
        case 0:
          i1lo = der_basis_funs_1d( x1vals_, x1, order_, x1coef1d_, isUniform_[0] );
          i2lo =     basis_funs_1d( x2vals_, x2, order_, x2coef1d_, isUniform_[1] );
          i3lo = der_basis_funs_1d( x3vals_, x3, order_, x3coef1d_, isUniform_[2] );
          i4lo =     basis_funs_1d( x4vals_, x4, order_, x4coef1d_, isUniform_[3] );
          i5lo =     basis_funs_1d( x5vals_, x5, order_, x5coef1d_, isUniform_[4] );
          break;
        case 1:
          i1lo =     basis_funs_1d( x1vals_, x1, order_, x1coef1d_, isUniform_[0] );
          i2lo = der_basis_funs_1d( x2vals_, x2, order_, x2coef1d_, isUniform_[1] );
          i3lo = der_basis_funs_1d( x3vals_, x3, order_, x3coef1d_, isUniform_[2] );
          i4lo =     basis_funs_1d( x4vals_, x4, order_, x4coef1d_, isUniform_[3] );
          i5lo =     basis_funs_1d( x5vals_, x5, order_, x5coef1d_, isUniform_[4] );
          break;
        case 2:
          i1lo =            basis_funs_1d( x1vals_, x1, order_, x1coef1d_, isUniform_[0] );
          i2lo =            basis_funs_1d( x2vals_, x2, order_, x2coef1d_, isUniform_[1] );
          i3lo = second_der_basis_funs_1d( x3vals_, x3, order_, x3coef1d_, isUniform_[2] );
          i4lo =            basis_funs_1d( x4vals_, x4, order_, x4coef1d_, isUniform_[3] );
          i5lo =            basis_funs_1d( x5vals_, x5, order_, x5coef1d_, isUniform_[4] );
          break;
        case 3:
          i1lo =     basis_funs_1d( x1vals_, x1, order_, x1coef1d_, isUniform_[0] );
          i2lo =     basis_funs_1d( x2vals_, x2, order_, x2coef1d_, isUniform_[1] );
          i3lo = der_basis_funs_1d( x3vals_, x3, order_, x3coef1d_, isUniform_[2] );
          i4lo = der_basis_funs_1d( x4vals_, x4, order_, x4coef1d_, isUniform_[3] );
          i5lo =     basis_funs_1d( x5vals_, x5, order_, x5coef1d_, isUniform_[4] );
          break;
        case 4:
          i1lo =     basis_funs_1d( x1vals_, x1, order_, x1coef1d_, isUniform_[0] );
          i2lo =     basis_funs_1d( x2vals_, x2, order_, x2coef1d_, isUniform_[1] );
          i3lo = der_basis_funs_1d( x3vals_, x3, order_, x3coef1d_, isUniform_[2] );
          i4lo =     basis_funs_1d( x4vals_, x4, order_, x4coef1d_, isUniform_[3] );
          i5lo = der_basis_funs_1d( x5vals_, x5, order_, x5coef1d_, isUniform_[4] );
          break;
      }
      break;
    } // case 2 on switch(dim1)
    case 3:{
      switch( dim2 ){
        case 0:
          i1lo = der_basis_funs_1d( x1vals_, x1, order_, x1coef1d_, isUniform_[0] );
          i2lo =     basis_funs_1d( x2vals_, x2, order_, x2coef1d_, isUniform_[1] );
          i3lo =     basis_funs_1d( x3vals_, x3, order_, x3coef1d_, isUniform_[2] );
          i4lo = der_basis_funs_1d( x4vals_, x4, order_, x4coef1d_, isUniform_[3] );
          i5lo =     basis_funs_1d( x5vals_, x5, order_, x5coef1d_, isUniform_[4] );
          break;
        case 1:
          i1lo =     basis_funs_1d( x1vals_, x1, order_, x1coef1d_, isUniform_[0] );
          i2lo = der_basis_funs_1d( x2vals_, x2, order_, x2coef1d_, isUniform_[1] );
          i3lo =     basis_funs_1d( x3vals_, x3, order_, x3coef1d_, isUniform_[2] );
          i4lo = der_basis_funs_1d( x4vals_, x4, order_, x4coef1d_, isUniform_[3] );
          i5lo =     basis_funs_1d( x5vals_, x5, order_, x5coef1d_, isUniform_[4] );
          break;
        case 2:
          i1lo =     basis_funs_1d( x1vals_, x1, order_, x1coef1d_, isUniform_[0] );
          i2lo =     basis_funs_1d( x2vals_, x2, order_, x2coef1d_, isUniform_[1] );
          i3lo = der_basis_funs_1d( x3vals_, x3, order_, x3coef1d_, isUniform_[2] );
          i4lo = der_basis_funs_1d( x4vals_, x4, order_, x4coef1d_, isUniform_[3] );
          i5lo =     basis_funs_1d( x5vals_, x5, order_, x5coef1d_, isUniform_[4] );
          break;
        case 3:
          i1lo =            basis_funs_1d( x1vals_, x1, order_, x1coef1d_, isUniform_[0] );
          i2lo =            basis_funs_1d( x2vals_, x2, order_, x2coef1d_, isUniform_[1] );
          i3lo =            basis_funs_1d( x3vals_, x3, order_, x3coef1d_, isUniform_[2] );
          i4lo = second_der_basis_funs_1d( x4vals_, x4, order_, x4coef1d_, isUniform_[3] );
          i5lo =            basis_funs_1d( x5vals_, x5, order_, x5coef1d_, isUniform_[4] );
          break;
        case 4:
          i1lo =     basis_funs_1d( x1vals_, x1, order_, x1coef1d_, isUniform_[0] );
          i2lo =     basis_funs_1d( x2vals_, x2, order_, x2coef1d_, isUniform_[1] );
          i3lo =     basis_funs_1d( x3vals_, x3, order_, x3coef1d_, isUniform_[2] );
          i4lo = der_basis_funs_1d( x4vals_, x4, order_, x4coef1d_, isUniform_[3] );
          i5lo = der_basis_funs_1d( x5vals_, x5, order_, x5coef1d_, isUniform_[4] );
          break;
      }
      break;
    } // case 3 on switch(dim1)
    case 4:{
      switch( dim2 ){
        case 0:
          i1lo = der_basis_funs_1d( x1vals_, x1, order_, x1coef1d_, isUniform_[0] );
          i2lo =     basis_funs_1d( x2vals_, x2, order_, x2coef1d_, isUniform_[1] );
          i3lo =     basis_funs_1d( x3vals_, x3, order_, x3coef1d_, isUniform_[2] );
          i4lo =     basis_funs_1d( x4vals_, x4, order_, x4coef1d_, isUniform_[3] );
          i5lo = der_basis_funs_1d( x5vals_, x5, order_, x5coef1d_, isUniform_[4] );
          break;
        case 1:
          i1lo =     basis_funs_1d( x1vals_, x1, order_, x1coef1d_, isUniform_[0] );
          i2lo = der_basis_funs_1d( x2vals_, x2, order_, x2coef1d_, isUniform_[1] );
          i3lo =     basis_funs_1d( x3vals_, x3, order_, x3coef1d_, isUniform_[2] );
          i4lo =     basis_funs_1d( x4vals_, x4, order_, x4coef1d_, isUniform_[3] );
          i5lo = der_basis_funs_1d( x5vals_, x5, order_, x5coef1d_, isUniform_[4] );
          break;
        case 2:
          i1lo =     basis_funs_1d( x1vals_, x1, order_, x1coef1d_, isUniform_[0] );
          i2lo =     basis_funs_1d( x2vals_, x2, order_, x2coef1d_, isUniform_[1] );
          i3lo = der_basis_funs_1d( x3vals_, x3, order_, x3coef1d_, isUniform_[2] );
          i4lo =     basis_funs_1d( x4vals_, x4, order_, x4coef1d_, isUniform_[3] );
          i5lo = der_basis_funs_1d( x5vals_, x5, order_, x5coef1d_, isUniform_[4] );
          break;
        case 3:
          i1lo =     basis_funs_1d( x1vals_, x1, order_, x1coef1d_, isUniform_[0] );
          i2lo =     basis_funs_1d( x2vals_, x2, order_, x2coef1d_, isUniform_[1] );
          i3lo =     basis_funs_1d( x3vals_, x3, order_, x3coef1d_, isUniform_[2] );
          i4lo = der_basis_funs_1d( x4vals_, x4, order_, x4coef1d_, isUniform_[3] );
          i5lo = der_basis_funs_1d( x5vals_, x5, order_, x5coef1d_, isUniform_[4] );
          break;
        case 4:
          i1lo =            basis_funs_1d( x1vals_, x1, order_, x1coef1d_, isUniform_[0] );
          i2lo =            basis_funs_1d( x2vals_, x2, order_, x2coef1d_, isUniform_[1] );
          i3lo =            basis_funs_1d( x3vals_, x3, order_, x3coef1d_, isUniform_[2] );
          i4lo =            basis_funs_1d( x4vals_, x4, order_, x4coef1d_, isUniform_[3] );
          i5lo = second_der_basis_funs_1d( x5vals_, x5, order_, x5coef1d_, isUniform_[4] );
          break;
      }
      break;
    } // case 4 on switch(dim1)
    default:
      throw std::invalid_argument("invalid dimension specification for second derivative");
  }

  const size_t nx1 = x1vals_.size();
  const size_t nx2 = x2vals_.size();
  const size_t nx3 = x3vals_.size();
  const size_t nx4 = x4vals_.size();

  double val = 0.0;

  for( size_t k5=0; k5<=order_; ++k5 ){
    const size_t i5 = (i5lo+k5)*nx1*nx2*nx3*nx4;
    for( size_t k4=0; k4<=order_; ++k4 ){
      const double tmp = x4coef1d_[k4] * x5coef1d_[k5];
      const size_t i4 = (i4lo+k4)*nx1*nx2*nx3;
      for( size_t k3=0; k3<=order_; ++k3 ){
        const double tmp2 = x3coef1d_[k3] * tmp;
        const size_t i3 = (i3lo+k3)*nx1*nx2;
        for( size_t k2=0; k2<=order_; ++k2 ){
          const double tmp3 = x2coef1d_[k2] * tmp2;
          const size_t i2 = (i2lo+k2)*nx1;
          for( size_t k1=0; k1<=order_; ++k1 ){
            const size_t i1 = i1lo+k1 + i2 + i3 + i4 + i5;
            assert( i1 < fvals_.size() );
            val += fvals_[i1] * x1coef1d_[k1] * tmp3;
          }
        }
      }
    }
  }
  return val;
}

//-------------------------------------------------------------------

LagrangeInterpolant5D::~LagrangeInterpolant5D()
{}

//-------------------------------------------------------------------

vector<pair<double,double> >
LagrangeInterpolant5D::get_bounds() const
{
  return bounds_;
}

//-------------------------------------------------------------------

bool
LagrangeInterpolant5D::operator==( const LagrangeInterpolant& other ) const
{
  const LagrangeInterpolant5D& a = dynamic_cast<const LagrangeInterpolant5D&>(other);
  return order_ == a.get_order()
      && allowClipping_ == a.clipping()
      && vec_compare( x1vals_, a.x1vals_ )
      && vec_compare( x2vals_, a.x2vals_ )
      && vec_compare( x3vals_, a.x3vals_ )
      && vec_compare( x4vals_, a.x4vals_ )
      && vec_compare( x5vals_, a.x5vals_ )
      && vec_compare( fvals_ , a.fvals_  )
      && bounds_ == a.bounds_;
}

//-------------------------------------------------------------------

template<typename Archive> void
LagrangeInterpolant5D::serialize( Archive& ar, const unsigned version )
{
  ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP( LagrangeInterpolant )
     & BOOST_SERIALIZATION_NVP( x1vals_ )
     & BOOST_SERIALIZATION_NVP( x2vals_ )
     & BOOST_SERIALIZATION_NVP( x3vals_ )
     & BOOST_SERIALIZATION_NVP( x4vals_ )
     & BOOST_SERIALIZATION_NVP( x5vals_ )
     & BOOST_SERIALIZATION_NVP( fvals_  )
     & BOOST_SERIALIZATION_NVP( isUniform_ );

  bounds_.clear();
  bounds_.push_back( bounds(x1vals_) );
  bounds_.push_back( bounds(x2vals_) );
  bounds_.push_back( bounds(x3vals_) );
  bounds_.push_back( bounds(x4vals_) );
  bounds_.push_back( bounds(x5vals_) );

  x1coef1d_.resize( order_+1 );
  x2coef1d_.resize( order_+1 );
  x3coef1d_.resize( order_+1 );
  x4coef1d_.resize( order_+1 );
  x5coef1d_.resize( order_+1 );
}

//===================================================================

// explicit template instantiation

template void LagrangeInterpolant  ::serialize<OutputArchive>( OutputArchive&, const unsigned );
template void LagrangeInterpolant  ::serialize<InputArchive >( InputArchive& , const unsigned );

template void LagrangeInterpolant1D::serialize<OutputArchive>( OutputArchive&, const unsigned );
template void LagrangeInterpolant1D::serialize<InputArchive >( InputArchive& , const unsigned );

template void LagrangeInterpolant2D::serialize<OutputArchive>( OutputArchive&, const unsigned );
template void LagrangeInterpolant2D::serialize<InputArchive >( InputArchive& , const unsigned );

template void LagrangeInterpolant3D::serialize<OutputArchive>( OutputArchive&, const unsigned );
template void LagrangeInterpolant3D::serialize<InputArchive >( InputArchive& , const unsigned );

template void LagrangeInterpolant4D::serialize<OutputArchive>( OutputArchive&, const unsigned );
template void LagrangeInterpolant4D::serialize<InputArchive >( InputArchive& , const unsigned );

template void LagrangeInterpolant5D::serialize<OutputArchive>( OutputArchive&, const unsigned );
template void LagrangeInterpolant5D::serialize<InputArchive >( InputArchive& , const unsigned );

BOOST_CLASS_EXPORT_IMPLEMENT( LagrangeInterpolant   )   BOOST_CLASS_VERSION( LagrangeInterpolant  , 1 )
BOOST_CLASS_EXPORT_IMPLEMENT( LagrangeInterpolant1D )   BOOST_CLASS_VERSION( LagrangeInterpolant1D, 1 )
BOOST_CLASS_EXPORT_IMPLEMENT( LagrangeInterpolant2D )   BOOST_CLASS_VERSION( LagrangeInterpolant2D, 1 )
BOOST_CLASS_EXPORT_IMPLEMENT( LagrangeInterpolant3D )   BOOST_CLASS_VERSION( LagrangeInterpolant3D, 1 )
BOOST_CLASS_EXPORT_IMPLEMENT( LagrangeInterpolant4D )   BOOST_CLASS_VERSION( LagrangeInterpolant4D, 1 )
BOOST_CLASS_EXPORT_IMPLEMENT( LagrangeInterpolant5D )   BOOST_CLASS_VERSION( LagrangeInterpolant5D, 1 )
