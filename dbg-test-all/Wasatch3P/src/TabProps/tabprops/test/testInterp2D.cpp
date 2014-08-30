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

#include <tabprops/TabPropsConfig.h>
#include <tabprops/Archive.h>

#include "TestHelper.h"

#include <boost/chrono.hpp>

#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>

#include <tabprops/TabProps.h>

using std::cout;
using std::endl;
using std::vector;

//==============================================================================

void time_it( const InterpT& interp, const double range )
{
  typedef boost::chrono::duration<long long, boost::micro> microseconds;
  typedef boost::chrono::high_resolution_clock Clock;
  Clock::time_point t1 = Clock::now();
  const unsigned n=100000;
  double x[2];
  for( int unsigned i=0; i<n; ++i ){
    x[0] = double(rand())/double(RAND_MAX) * range;
    x[1] = double(rand())/double(RAND_MAX) * range;
    interp.value(x);
  }
  boost::chrono::duration<double> t2 = Clock::now() - t1;
  std::cout << "\t" << std::scientific << std::setprecision(3) << n/t2.count() << " interpolations per second" << std::endl;

  t1 = Clock::now();
  for( int unsigned i=0; i<n; ++i ){
    x[0] = double(rand())/double(RAND_MAX) * range;
    x[1] = double(rand())/double(RAND_MAX) * range;
    interp.derivative(x,0);
  }
  t2 = Clock::now() - t1;
  std::cout << "\t" << n/t2.count() << " derivatives per second" << std::endl;

  t1 = Clock::now();
  for( int unsigned i=0; i<n; ++i ){
    x[0] = double(rand())/double(RAND_MAX) * range;
    x[1] = double(rand())/double(RAND_MAX) * range;
    interp.second_derivative(x,0,0);
  }
  t2 = Clock::now() - t1;
  std::cout << "\t" << n/t2.count() << " second derivatives per second" << std::endl;
}

//==============================================================================

vector<double> set_grid( const size_t n,
                         const double L,
                         const bool isUniform )
{
  vector<double> x(n,0);
  const double dx = L/double(n-1);
  for( size_t i=0; i<n; ++i ){
    if( isUniform )  x[i] = i*dx;
    else             x[i] = i*dx + (std::rand()/RAND_MAX-0.5) * dx/4;
  }
  return x;
}

vector<double> set_function( const vector<double>& x,
                             const vector<double>& y )
{
  vector<double> f;
  for( size_t j=0; j<y.size(); ++j ){
    for( size_t i=0; i<x.size(); ++i ){
      f.push_back( 5.0*x[i]*x[i]*y[j] + 4.0*y[j]*y[j]*x[i] + sin( x[i] ) + cos( y[j] ) );
    }
  }
  return f;
}

// test the interpolant to ensure that it precisely interpolates the grid values
bool test_interp( const vector<double>& x,
                  const vector<double>& y,
                  const Interp2D& interp,
                  const double atol,
                  const double derTol,
                  const double der2Tol )
{
//  std::ofstream fout("df.dat");

  TestHelper status(false);
  const vector<double> f = set_function( x, y );
  size_t k=0;
  for( size_t j=0; j<y.size(); ++j ){
    for( size_t i=0; i<x.size(); ++i, ++k ){
      double query[2] = { x[i], y[j] };

      const double fi = interp.value( query );
      status( std::abs( fi-f[k] ) < atol );
      if( std::abs( fi-f[k] ) > atol ){
        cout << "  ** f=" << f[k] << ", finterp=" << fi << ", err: " << std::abs(fi-f[k]) << endl;
      }

      const double dfdx = interp.derivative(query,0);
      const double dfdxx = 10.0*x[i]*y[j] + 4.0*y[j]*y[j] + std::cos(x[i]);
      double err = std::abs(dfdx-dfdxx);
      status( err < derTol, "dfdx" );
      if( err >= derTol ){
        cout << "  ** dfdx=" << dfdxx << ", df_approx=" << dfdx << " err=" << err << endl;
      }

      const double dfdy = interp.derivative(query,1);
      const double dfdyx = 5*x[i]*x[i] + 8.0*y[j]*x[i] - std::sin(y[j]);
      err = std::abs(dfdy-dfdyx);
      status( err < derTol, "dfdy" );
      if( err >= derTol ){
        cout << "  ** dfdy=" << dfdyx << ", df_approx=" << dfdy << " err=" << err << endl;
      }

      //-- second derivatives
      if( interp.get_order() > 1 ){
        const double d2fdx2 = interp.second_derivative(query,0,0);
        const double d2fdx2x= 10.0*y[j] - std::sin(x[i]);
        err = std::abs(d2fdx2-d2fdx2x);
        status( err < der2Tol, "d2fdx2" );
        if( err >= der2Tol ){
          cout << " ** d2fdx2=" << d2fdx2x << " approx=" << d2fdx2 << " err=" << err << endl;
        }

        const double d2fdy2 = interp.second_derivative(query,1,1);
        const double d2fdy2x= 8.0*x[i] - std::cos(y[j]);
        err = std::abs(d2fdy2-d2fdy2x);
        status( err < der2Tol, "d2fdy2" );
        if( err >= der2Tol ){
          cout << " ** d2fdy2=" << d2fdy2x << " approx=" << d2fdy2 << " err=" << err << endl;
        }

        const double d2fdxy = interp.second_derivative(query,0,1);
        const double d2fdxyx= 10*x[i] + 8*y[j];
        err = std::abs(d2fdxy-d2fdxyx);
        status( err < der2Tol, "d2fdxy" );
        if( err >= der2Tol ){
          cout << " ** d2fdxy=" << d2fdxyx << " approx=" << d2fdxy << " err=" << err << endl;
        }

        const double d2fdyx = interp.second_derivative(query,1,0);
        const double d2fdyxx= 10*x[i]+8*y[j];
        err = std::abs(d2fdyx-d2fdyxx);
        status( err < der2Tol, "d2fdyx" );
        if( err >= der2Tol ){
          cout << " ** d2fdyx=" << d2fdyxx << " approx=" << d2fdyx << " err=" << err << endl;
        }
      }

//      fout << x[i] << "\t" << y[j] << "\t" << dfdxx << "\t" << dfdx << "\t" << dfdyx << "\t" << dfdy << std::endl;
    }
  }
  return status.ok();
}

bool test( const size_t nx,
           const size_t ny,
           const size_t order,
           const bool isUniform )
{
  TestHelper status(false);

  const double lx = 1.0;
  const double ly = 1.0;

  const vector<double> x = set_grid( nx, lx, isUniform );
  const vector<double> y = set_grid( ny, ly, isUniform );

  // build the interpolant
  Interp2D funcInterp( order, x, y, set_function(x,y) );

  status( funcInterp.get_bounds()[0].first  == x[0   ], "x lo bound" );
  status( funcInterp.get_bounds()[0].second == x[nx-1], "x hi bound" );
  status( funcInterp.get_bounds()[1].first  == y[0   ], "y lo bound" );
  status( funcInterp.get_bounds()[1].second == y[ny-1], "y hi bound" );

  double atol = 3e-14;
  double derTol = atol, der2Tol=derTol;
  switch (order){
  case 1 : atol=9.4e-4;  derTol=2.7e-1;                   break;
  case 2 : atol=6.0e-7;  derTol=9.4e-4; der2Tol=5.4e-2;  break;
  case 3 : atol=8.9e-9;  derTol=3.7e-5; der2Tol=2.6e-3;  break;
  default: atol=6.0e-11;  derTol=1.6e-6; der2Tol=1.3e-4;  break;
  }
  status( test_interp( x, y, funcInterp, atol, derTol, der2Tol ), "interpolation" );

  {
    std::ofstream outFile("lin2d.out", std::ios_base::out|std::ios_base::trunc );
    OutputArchive oa(outFile);
    oa << BOOST_SERIALIZATION_NVP( funcInterp );
  }
  {
    std::ifstream inFile("lin2d.out");
    InputArchive ia(inFile);
    Interp2D interp2;
    ia >> BOOST_SERIALIZATION_NVP( interp2 );
    status( funcInterp == interp2 , "Serialization" );
  }

  time_it( funcInterp, x[nx-1] );

  const InterpT* const myClone = funcInterp.clone();
  status( *myClone == funcInterp, "clone" );
  delete myClone;
  return status.ok();
}


int main()
{
  TestHelper status( true );
  try{
    for( unsigned order=1; order<=4; ++order ){
      std::cout << std::endl << "Testing for order = " << order << std::endl;
      status( test( 20, 25, order, true  ), "20x25 uniform" );
      status( test( 20, 25, order, false ), "20x25 nonuniform" );
      status( test( 25, 20, order, true  ), "25x20 uniform" );
      status( test( 25, 20, order, false ), "25x20 nonuniform" );
    }
    if( status.ok() ){
      std::cout << std::endl << "PASS!" << std::endl;
      return 0;
    }
  }
  catch( std::exception& e ){
    std::cout << e.what() << std::endl;
  }
  std::cout << std::endl << "FAIL!" << std::endl;
  return -1;
}
