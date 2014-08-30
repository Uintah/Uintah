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

#include <algorithm>
#include <cmath>
#include <cassert>

#include "LagrangePoly.h"

#include <iostream>
using std::cout; using std::endl;

//--------------------------------------------------------------------

LagrangeCoefficients::LagrangeCoefficients( const std::vector<double> xpts )
  : xpts_( xpts )
{
}

//--------------------------------------------------------------------

LagrangeCoefficients::LagrangeCoefficients( const double* const xbegin,
                                            const double* const xend )
  : xpts_( xbegin, xend )
{
}

//--------------------------------------------------------------------

LagrangeCoefficients::~LagrangeCoefficients()
{
}

//--------------------------------------------------------------------

void
LagrangeCoefficients::
get_bounds( const double x,
            const int polyOrder,
            int& ilo, int& nlo,
            int& nhi ) const
{
  // obtain an iterator to the point on the (-) side that brackets x:
  ConstVecIter iterlo = std::lower_bound( xpts_.begin(), xpts_.end(), x ) - 1;
  assert( iterlo != xpts_.end() );
  const int nx = xpts_.size();
  ilo = iterlo - xpts_.begin();

  assert( ilo < nx );
  if( ilo<0 ){
    ilo-=ilo;
    iterlo=xpts_.begin();
  }

  const int npts = polyOrder+1;

  nhi = npts/2;
  nlo = nhi;
  if( npts%2 != 0 ){
    // odd number of points.  Are we closer to lower or upper?
    if( ilo+1 >= nx )
      ++nlo;
    else{
      if( std::fabs(*iterlo-x) > std::fabs(*(iterlo+1)-x) )
        ++nhi;
      else
        ++nlo;
    }
  }

#ifdef DEBUG_COEFFS
  cout << "order: " << polyOrder
       << ", npts=" << npts
       << ",  nlo=" << nlo
       << ",  nhi=" << nhi
       << endl;
#endif

  // adjust for boundaries
  const int loExcess = (ilo-nlo+1);
  if( loExcess < 0 ){
#ifdef DEBUG_COEFFS
    cout << "adjusting lower bound to stay in domain." << endl;
#endif
    nlo += loExcess;
    nhi-=loExcess;
  }
  const int hiExcess = (nx-(ilo+nhi+1));
  if( hiExcess < 0 ){
#ifdef DEBUG_COEFFS
    cout << "adjusting upper bound to stay in domain." << endl;
#endif
    nlo -= hiExcess;
    nhi += hiExcess;
  }
}

//--------------------------------------------------------------------

void
LagrangeCoefficients::
get_interp_coefs_indices( const double x,
                          const int polyOrder,
                          std::vector<double>& coefs,
                          std::vector<int>& indices ) const
{
  int ilo=0, nlo=0, nhi=0;
  get_bounds( x, polyOrder, ilo, nlo, nhi );

  coefs.clear();
  indices.clear();

  // get the coefficients.
  int kindex = ilo-nlo+1;
  const int koffset = kindex;
  for( int k=0; k<=polyOrder; ++k, ++kindex ){

    const double xk = xpts_[kindex];
    double coef = 1.0;

    // assemble this coefficient.
    bool haveEntry = false;
    for( int i=ilo-nlo+1; i!=ilo+nhi+1; ++i ){
      const double xi = xpts_[i];
      if( xk != xi ){
        haveEntry = true;
        coef *= (x-xi)/(xk-xi);
      }
    }
    if( haveEntry ){
      coefs.push_back( coef );
      indices.push_back( k+koffset );
    }
  }
}

//--------------------------------------------------------------------

void
LagrangeCoefficients::
get_derivative_coefs_indices( const double x,
                              const int polyOrder,
                              std::vector<double>& coefs,
                              std::vector<int>& indices ) const
{
  int ilo=0, nlo=0, nhi=0;
  get_bounds( x, polyOrder, ilo, nlo, nhi );

  coefs.clear();
  indices.clear();

  // get the coefficients.
  int kindex = ilo-nlo+1;
  const int koffset = kindex;
  for( int k=0; k<=polyOrder; ++k, ++kindex ){

    const double xk = xpts_[kindex];

    // assemble the denominator:
#ifdef DEBUG_COEFFS
    cout << endl << "k=" << k << " of " << polyOrder << endl;
    cout << "  den = ";
#endif
    bool haveEntry = false;
    double den = 1.0;
    for( int i=ilo-nlo+1; i!=ilo+nhi+1; ++i ){
      const double xi = xpts_[i];
      if( xi != xk ){
        haveEntry = true;
        den *= (xk-xi);
#ifdef DEBUG_COEFFS
        cout << "(x"  << kindex-koffset << "-x" << i-koffset << ")";
#endif
      }
    }

    // assemble the numerator
    // jcs this seems to be a performance bottleneck.  Look into it...
#ifdef DEBUG_COEFFS
    cout << endl
         << "  num = ";
#endif
    double num = 0.0;
    if( polyOrder == 1 ) num=1.0;
    for( int j=ilo-nlo+1; j!=ilo+nhi+1; ++j ){
      const double xj = xpts_[j];
      if( xj == xk ) continue;
      bool tmpEntry = false;
      double tmp = 1.0;
      for( int i=ilo-nlo+1; i!=ilo+nhi+1; ++i ){
        const double xi = xpts_[i];
        if( xi == xk || xi == xj ) continue;
        tmp *= (x-xi);
#ifdef DEBUG_COEFFS
        cout << "(" << "x" << "-" << "x" << i-koffset << ")";
#endif
        tmpEntry = true;
      }
      if( tmpEntry ) num += tmp;
    }
#ifdef DEBUG_COEFFS
    cout << endl;
    cout << "  num: " << num << ",  den: " << den << endl;
#endif
    if( haveEntry ){
      coefs.push_back( num/den );
      indices.push_back( k+koffset );
    }
  }
}

//--------------------------------------------------------------------

LagrangeInterpolant::
LagrangeInterpolant( const std::vector<double>& xpts,
                     const std::vector<double>& ypts,
                     const int order )
  : coefs_( xpts ),
    ypts_ ( ypts ),
    order_( order )
{
  assert( xpts.size() == ypts.size() );
}

//--------------------------------------------------------------------

LagrangeInterpolant::~LagrangeInterpolant()
{
}

//--------------------------------------------------------------------

double
LagrangeInterpolant::value( const double x,
                            const int order ) const
{
  coefs_.get_interp_coefs_indices( x, order, coefVals_, indices_ );
  double val = 0.0;
  std::vector<double>::const_iterator icoef = coefVals_.begin();
  std::vector<double>::const_iterator iy    = ypts_    .begin() + indices_[0];
  for( ; icoef!=coefVals_.end(); ++icoef, ++iy ){
    val += *iy * *icoef;
  }
  return val;
}

//--------------------------------------------------------------------

void
LagrangeInterpolant::
get_coefs_indices( const double x,
                   const int order,
                   std::vector<double>& coefs,
                   std::vector<int>& indices ) const
{
  coefs_.get_interp_coefs_indices( x, order, coefs, indices );
}

//--------------------------------------------------------------------




//--------------------------------------------------------------------

LagrangeDerivative::
LagrangeDerivative( const std::vector<double>& x,
                    const std::vector<double>& y,
                    const int order )
  : coefs_( x ),
    ypts_ ( y ),
    order_( order )
{
  assert( x.size() == y.size() );
}

//--------------------------------------------------------------------

LagrangeDerivative::~LagrangeDerivative()
{
}

//--------------------------------------------------------------------

double
LagrangeDerivative::value( const double x, const int order ) const
{
  coefs_.get_derivative_coefs_indices( x, order, coefVals_, indices_ );
  double val = 0.0;
  std::vector<double>::const_iterator icoef = coefVals_.begin();
  std::vector<double>::const_iterator iy    = ypts_    .begin() + indices_[0];
  for( ; icoef!=coefVals_.end(); ++icoef, ++iy ){
    val += *iy * *icoef;
  }
  return val;
}

//--------------------------------------------------------------------

void
LagrangeDerivative::
get_coefs_indices( const double x,
                   const int order,
                   std::vector<double>& coefs,
                   std::vector<int>& indices ) const
{
  coefs_.get_derivative_coefs_indices( x, order, coefs, indices );
}

//--------------------------------------------------------------------
