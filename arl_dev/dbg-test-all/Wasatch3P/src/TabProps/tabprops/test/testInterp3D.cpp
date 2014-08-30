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

#include <cmath>
#include <iostream>
#include <fstream>

#include <boost/chrono.hpp>

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
  double x[3];
  for( int unsigned i=0; i<n; ++i ){
    x[0] = double(rand())/double(RAND_MAX) * range;
    x[1] = double(rand())/double(RAND_MAX) * range;
    x[2] = double(rand())/double(RAND_MAX) * range;
    interp.value(x);
  }
  boost::chrono::duration<double> t2 = Clock::now() - t1;
  std::cout << n/t2.count() << " interpolations per second" << std::endl;

  t1 = Clock::now();
  for( int unsigned i=0; i<n; ++i ){
    x[0] = double(rand())/double(RAND_MAX) * range;
    x[1] = double(rand())/double(RAND_MAX) * range;
    x[2] = double(rand())/double(RAND_MAX) * range;
    interp.derivative(x,0);
  }
  t2 = Clock::now() - t1;
  std::cout << n/t2.count() << " derivatives per second" << std::endl;

  t1 = Clock::now();
  for( int unsigned i=0; i<n; ++i ){
    x[0] = double(rand())/double(RAND_MAX) * range;
    x[1] = double(rand())/double(RAND_MAX) * range;
    x[2] = double(rand())/double(RAND_MAX) * range;
    interp.second_derivative(x,0,0);
  }
  t2 = Clock::now() - t1;
  std::cout << n/t2.count() << " second derivatives per second" << std::endl;
}

//==============================================================================

vector<double> set_grid( const bool isUniform,
                         const size_t n,
                         const double L = 1.0 )
{
  vector<double> x(n,0);
  for( size_t i=0; i<n; ++i ){
    if( isUniform ){
      x[i] = L/double(n-1) * double(i);
    }
    else{
      x[i] = L/double((n-1)*(n-1)) * i*i;
    }
  }
  return x;
}

vector<double> set_function( const vector<double>& x,
                             const vector<double>& y,
                             const vector<double>& z )
{
  // jcs note that this is a weak function for testing second
  // derivatives because the mixed partials are all zero.
  vector<double> f;
  size_t m=0;
  for( size_t k=0; k<z.size(); ++k ){
    for( size_t j=0; j<y.size(); ++j ){
      for( size_t i=0; i<x.size(); ++i ){
        f.push_back( sin( x[i] ) + cos( y[j] ) + z[k]*z[k] );
      }
    }
  }
  return f;
}

// test the interpolant to ensure that it precisely interpolates the grid values
bool test_interp( const vector<double>& x,
                  const vector<double>& y,
                  const vector<double>& z,
                  const Interp3D& interp,
                  const vector<double>& f,
                  const double atol,
                  const double derTol,
                  const double der2Tol )
{
  TestHelper status(false);

  long int nquery=0;
  const size_t nx=x.size(), ny=y.size(), nz=z.size();
  for( size_t k=0; k<nz; ++k ){
    for( size_t j=0; j<ny; ++j ){
      for( size_t i=0; i<nx; ++i ){
        double query[3] = { x[i], y[j], z[k] };
        const double fi = interp.value( query );
        ++nquery;
        const size_t m = i + j*nx + k*nx*ny;
        status( std::abs( fi-f[m] ) < atol );
        if( std::abs( fi-f[m] ) >= atol ){
          cout << "  ** f=" << f[m] << ", finterp=" << fi << ", err: " << std::abs(fi-f[m])
               << "  (i,j,k)=(" << i << "," << j << "," << k << ")"<< endl;
        }

        const double dfdx = interp.derivative(query,0);
        const double dfdy = interp.derivative(query,1);
        const double dfdz = interp.derivative(query,2);

        const double dx =  cos(x[i]);
        const double dy = -sin(y[j]);
        const double dz =  2*z[k];

        status( std::abs(dfdx-dx) < derTol );
        status( std::abs(dfdy-dy) < derTol );
        status( std::abs(dfdz-dz) < derTol );

        if( interp.get_order() > 1 ){
          const double d2fdx2  = interp.second_derivative(query,0,0);
          const double d2fdy2  = interp.second_derivative(query,1,1);
          const double d2fdz2  = interp.second_derivative(query,2,2);
          const double d2fdxdy = interp.second_derivative(query,0,1);
          const double d2fdydx = interp.second_derivative(query,1,0);
          const double d2fdxdz = interp.second_derivative(query,0,2);
          const double d2fdzdx = interp.second_derivative(query,2,0);
          const double d2fdydz = interp.second_derivative(query,1,2);
          const double d2fdzdy = interp.second_derivative(query,2,1);

          const double d2x = -sin(x[i]);
          const double d2y = -cos(y[j]);
          const double d2z = 2;

          status( std::abs(d2fdxdy-d2fdydx) < 1e-14 );
          status( std::abs(d2fdxdz-d2fdxdz) < 1e-14 );
          status( std::abs(d2fdydz-d2fdzdy) < 1e-14 );

          double err = std::abs(d2fdx2-d2x);
          status( err < der2Tol );
          if( err > der2Tol ) cout << d2x << " " << d2fdx2 << " " << err << endl;

          err = std::abs(d2fdy2-d2y);
          status( err < der2Tol );
          if( err > der2Tol ) cout << d2y << " " << d2fdy2 << " " << err << endl;

          err = std::abs(d2fdz2-d2z);
          status( err < der2Tol );
          if( err > der2Tol ) cout << d2z << " " << d2fdz2 << " " << err << endl;
        }
      }
    }
  }

  time_it( interp, x[nx-1] );

  return status.ok();
}

bool test( const size_t nx,
           const size_t ny,
           const size_t nz,
           const bool isUniform,
           const unsigned order )
{
  TestHelper status(false);

  const double lx = 1.0;
  const double ly = 1.0;
  const double lz = 1.0;

  const vector<double> x = set_grid( isUniform, nx, lx );
  const vector<double> y = set_grid( isUniform, ny, ly );
  const vector<double> z = set_grid( isUniform, nz, lz );
  const vector<double> f = set_function(x,y,z);

  // build the interpolant
  Interp3D funcInterp( order, x, y, z, f );

  status( funcInterp.get_bounds()[0].first  == x[0   ], "x lo bound" );
  status( funcInterp.get_bounds()[0].second == x[nx-1], "x hi bound" );
  status( funcInterp.get_bounds()[1].first  == y[0   ], "y lo bound" );
  status( funcInterp.get_bounds()[1].second == y[ny-1], "y hi bound" );
  status( funcInterp.get_bounds()[2].first  == z[0   ], "z lo bound" );
  status( funcInterp.get_bounds()[2].second == z[nz-1], "z hi bound" );

  double derTol = 0.0, der2Tol=derTol;
  switch (order){
  case 1 : derTol=9e-2;                 break;
  case 2 : derTol=2e-2; der2Tol=6.5e-2; break;
  case 3 : derTol=9e-4; der2Tol=5.5e-3; break;
  default: derTol=1e-4; der2Tol=3e-4;   break;
  }

  status( test_interp( x, y, z, funcInterp, f, 5e-15, derTol, der2Tol ), "interpolant test");

  {
    std::ofstream outFile("lin3d.out", std::ios_base::out|std::ios_base::trunc );
    OutputArchive oa(outFile);
    oa << BOOST_SERIALIZATION_NVP( funcInterp );
  }
  {
    std::ifstream inFile("lin3d.out");
    InputArchive ia(inFile);
    Interp3D interp3;
    ia >> BOOST_SERIALIZATION_NVP( interp3 );
    status( interp3 == funcInterp, "serialization" );
    status( test_interp( x, y, z, interp3, f, 5e-15, derTol, der2Tol ), "interp on reloaded object" );
  }

  InterpT* myClone = funcInterp.clone();
  status( *myClone == funcInterp, "clone" );
  delete myClone;

  return status.ok();
}


int main()
{
  TestHelper status(true);
  try{
    for( unsigned order=1; order<=4; ++order ){
      cout << endl << "Order: " << order << endl;
      status( test( 20,  14,  30,  true, order ), "interpolate uniform (20,14,30)" );
      status( test( 30,  31,  35, false, order ), "interpolate nonuniform (30,31,35)" );
    }
    if( status.ok() ){
      cout << "PASS!" << endl;
      return 0;
    }
  }
  catch( std::exception& err ){
    cout << err.what() << endl;
  }
  cout << "FAIL!" << endl;
  return -1;
}
