/**
 * Copyright (c) 2012 The University of Utah
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
#include <iomanip>
#include <fstream>
#include <cstdlib>


#include <boost/chrono.hpp>

#include <tabprops/TabProps.h>

using std::cout;
using std::endl;
using std::vector;

//==============================================================================

void time_it( const InterpT& interp, const double range )
{
  typedef boost::chrono::high_resolution_clock Clock;
  Clock::time_point t1 = Clock::now();
  const unsigned n=200;
  double x[4];
  for( int unsigned i=0; i<n; ++i ){
    x[0] = double(rand())/double(RAND_MAX) * range;
    x[1] = double(rand())/double(RAND_MAX) * range;
    x[2] = double(rand())/double(RAND_MAX) * range;
    x[3] = double(rand())/double(RAND_MAX) * range;
    interp.value(x);
  }
  boost::chrono::duration<double> t2 = Clock::now() - t1;
  std::cout << "\t" << std::scientific << std::setprecision(3) << n/t2.count() << " interpolations per second" << std::endl;

  t1 = Clock::now();
  for( int unsigned i=0; i<n; ++i ){
    x[0] = double(rand())/double(RAND_MAX) * range;
    x[1] = double(rand())/double(RAND_MAX) * range;
    x[2] = double(rand())/double(RAND_MAX) * range;
    x[3] = double(rand())/double(RAND_MAX) * range;
    interp.derivative(x,0);
  }
  t2 = Clock::now() - t1;
  std::cout << "\t" << n/t2.count() << " derivatives per second" << std::endl;

  t1 = Clock::now();
  for( int unsigned i=0; i<n; ++i ){
    x[0] = double(rand())/double(RAND_MAX) * range;
    x[1] = double(rand())/double(RAND_MAX) * range;
    x[2] = double(rand())/double(RAND_MAX) * range;
    x[3] = double(rand())/double(RAND_MAX) * range;
    interp.second_derivative(x,0,0);
  }
  t2 = Clock::now() - t1;
  std::cout << "\t" << n/t2.count() << " second derivatives per second" << std::endl;
}

//==============================================================================

vector<double>
set_grid( const bool isUniform,
          const size_t n,
          const double L = 1.0 )
{
  vector<double> x(n,0);
  const double dx = L/double(n-1);
  for( size_t i=0; i<n; ++i ){
    if( isUniform )  x[i] = i*dx;
    else             x[i] = i*dx + (std::rand()/RAND_MAX-0.5) * dx/4;
  }
  return x;
}

vector<double>
set_function( const vector<double>& x0,
              const vector<double>& x1,
              const vector<double>& x2,
              const vector<double>& x3 )
{
  vector<double> f;
  for( size_t i3=0; i3<x3.size(); ++i3 ){
    for( size_t i2=0; i2<x2.size(); ++i2 ){
      for( size_t i1=0; i1<x1.size(); ++i1 ){
        for( size_t i0=0; i0<x0.size(); ++i0 ){
          f.push_back( sin( x0[i0] ) + cos( x1[i1] ) + x2[i2]*x2[i2] + cos(x3[i3]*x3[i3]) );
        }
      }
    }
  }
  return f;
}

// test the interpolant to ensure that it precisely interpolates the grid values
bool test_interp( const vector<double>& x0,
                  const vector<double>& x1,
                  const vector<double>& x2,
                  const vector<double>& x3,
                  const Interp4D& interp,
                  const vector<double>& f,
                  const double atol,
                  const double derTol,
                  const double der2Tol )
{
  TestHelper status( false );
  TestHelper d2status( true );

  long int nquery=0;
  const size_t n[4] = { x0.size(), x1.size(), x2.size(), x3.size() };
  for( size_t kk=0; kk<1; ++kk ){
    for( size_t i3=0; i3<n[3]; ++i3 ){
      for( size_t i2=0; i2<n[2]; ++i2 ){
        for( size_t i1=0; i1<n[1]; ++i1 ){
          for( size_t i0=0; i0<n[0]; ++i0 ){
            double query[4] = { x0[i0], x1[i1], x2[i2], x3[i3] };
            const double fi = interp.value( query );
            ++nquery;
            const size_t m = i0 + i1*n[0] + i2*n[0]*n[1] + i3*n[0]*n[1]*n[2];
            double err = std::abs( fi-f[m] );
            status( err <= atol );
            if( err > atol ){
              cout << fi << ", " << f[m] << "  ** err: " << err << endl;
            }

            const double dfdx0 = interp.derivative(query,0);
            const double dfdx1 = interp.derivative(query,1);
            const double dfdx2 = interp.derivative(query,2);
            const double dfdx3 = interp.derivative(query,3);

            const double dx0 = cos(x0[i0]);
            const double dx1 = -sin(x1[i1]);
            const double dx2 = 2*x2[i2];
            const double dx3 = -2*x3[i3]*sin(x3[i3]*x3[i3]);

            status( std::abs(dx0-dfdx0) < derTol && std::abs(dx1-dfdx1) < derTol &&
                    std::abs(dx2-dfdx2) < derTol && std::abs(dx3-dfdx3) < derTol );

            if( std::abs(dx0-dfdx0) > derTol || std::abs(dx1-dfdx1) > derTol ||
                std::abs(dx2-dfdx2) > derTol || std::abs(dx3-dfdx3) > derTol ){
              cout << "failed derivatives: \n\t"
                  << dx0 << " : " << dfdx0 << "\t " << dx0-dfdx0 << "\n\t"
                  << dx1 << " : " << dfdx1 << "\t " << dx1-dfdx1 << "\n\t"
                  << dx2 << " : " << dfdx2 << "\t " << dx2-dfdx2 << "\n\t"
                  << dx3 << " : " << dfdx3 << "\t " << dx3-dfdx3 << std::endl;
            }

            if( interp.get_order() == 1 ) continue;

            // jcs since the current function doesn't have nonzero mixed partials, don't test them for now - this will speed up the testing.
            const double d2fdx0x0 = interp.second_derivative(query,0,0);
//            const double d2fdx0x1 = interp.second_derivative(query,0,1);
//            const double d2fdx0x2 = interp.second_derivative(query,0,2);
//            const double d2fdx0x3 = interp.second_derivative(query,0,3);

            const double d2x0x0 = -sin(x0[i0]);
            const double d2x0x1 = 0.0;
            const double d2x0x2 = 0.0;
            const double d2x0x3 = 0.0;

//            const double d2fdx1x0 = interp.second_derivative(query,1,0);
            const double d2fdx1x1 = interp.second_derivative(query,1,1);
//            const double d2fdx1x2 = interp.second_derivative(query,1,2);
//            const double d2fdx1x3 = interp.second_derivative(query,1,3);

            const double d2x1x0 = 0.0;
            const double d2x1x1 = -cos(x1[i1]);
            const double d2x1x2 = 0.0;
            const double d2x1x3 = 0.0;

//            const double d2fdx2x0 = interp.second_derivative(query,2,0);
//            const double d2fdx2x1 = interp.second_derivative(query,2,1);
            const double d2fdx2x2 = interp.second_derivative(query,2,2);
//            const double d2fdx2x3 = interp.second_derivative(query,2,3);

            const double d2x2x0 = 0.0;
            const double d2x2x1 = 0.0;
            const double d2x2x2 = 2.0;
            const double d2x2x3 = 0.0;

//            const double d2fdx3x0 = interp.second_derivative(query,3,0);
//            const double d2fdx3x1 = interp.second_derivative(query,3,1);
//            const double d2fdx3x2 = interp.second_derivative(query,3,2);
            const double d2fdx3x3 = interp.second_derivative(query,3,3);

            const double d2x3x0 = 0.0;
            const double d2x3x1 = 0.0;
            const double d2x3x2 = 0.0;
            const double d2x3x3 = -2*sin(x3[i3]*x3[i3]) - 4*x3[i3]*x3[i3]*cos(x3[i3]*x3[i3]);

            err = std::abs(d2fdx0x0 - d2x0x0);
            status( err < der2Tol, "d2fdx0x0" );
            if( err > der2Tol ) cout << "d^2f / dx0^2 -> " << d2fdx0x0 << " : " << d2x0x0 << "\t " << err << endl;

            err = std::abs(d2fdx1x1 - d2x1x1);
            status( err < der2Tol, "d2fdx1x1" );
            if( err > der2Tol ) cout << "d^2f / dx1^2 -> " << d2fdx1x1 << " : " << d2x1x1 << "\t " << err << endl;

            err = std::abs(d2fdx2x2 - d2x2x2);
            status( err < der2Tol, "d2fdx2x2" );
            if( err > der2Tol ) cout << "d^2f / dx2^2 -> " << d2fdx2x2 << " : " << d2x2x2 << "\t " << err << endl;

            err = std::abs(d2fdx3x3 - d2x3x3);
            status( err < der2Tol, "d2fdx3x3" );
            if( err > der2Tol ) cout << "d^2f / dx3^2 -> " << d2fdx3x3 << " : " << d2x3x3 << "\t " << err << endl;

            // mixed partials should be equal
//            status( std::abs(d2fdx0x1 - d2fdx1x0) < 1e-14 );
//            status( std::abs(d2fdx0x2 - d2fdx2x0) < 1e-14 );
//            status( std::abs(d2fdx0x3 - d2fdx3x0) < 1e-14 );
//            status( std::abs(d2fdx1x2 - d2fdx2x1) < 1e-14 );
//            status( std::abs(d2fdx1x3 - d2fdx3x1) < 1e-14 );
//            status( std::abs(d2fdx2x3 - d2fdx3x2) < 1e-14 );

          }
        }
      }
    }
  }

  time_it( interp, x0[n[0]-1] );

  return status.ok();
}

bool test( const std::vector<size_t> n,
           const bool isUniform,
           const unsigned order )
{
  TestHelper status(false);

  double len[4];
  vector< vector<double> > x;
  for( size_t i=0; i<4; ++i ){
    len[i] = 1.0;
    x.push_back( set_grid( isUniform, n[i], len[i] ) );
  }
  vector<double> f = set_function( x[0], x[1], x[2], x[3] );

  Interp4D interp( order, x[0], x[1], x[2], x[3], f );

  status( interp.get_bounds()[0].first  == x[0][0     ], "x1 lo bound" );
  status( interp.get_bounds()[0].second == x[0][n[0]-1], "x1 hi bound" );
  status( interp.get_bounds()[1].first  == x[1][0     ], "x2 lo bound" );
  status( interp.get_bounds()[1].second == x[1][n[1]-1], "x2 hi bound" );
  status( interp.get_bounds()[2].first  == x[2][0     ], "x3 lo bound" );
  status( interp.get_bounds()[2].second == x[2][n[2]-1], "x3 hi bound" );
  status( interp.get_bounds()[3].first  == x[3][0     ], "x4 lo bound" );
  status( interp.get_bounds()[3].second == x[3][n[3]-1], "x4 hi bound" );

  {
    status( Interp4D(interp) == interp, "copy constructor" );
    InterpT* myClone = interp.clone();
    status( *myClone == interp, "clone" );
    delete myClone;
  }

  const double atol = 3e-14;
  double derTol = atol, der2Tol=derTol;
  switch (order){
  case 1 : derTol=8e-2;  der2Tol=0.0   ;  break;
  case 2 : derTol=2e-3;  der2Tol=4.4e-2;  break;
  case 3 : derTol=6e-4;  der2Tol=5.3e-2;  break;
  default: derTol=8e-5;  der2Tol=8.2e-3;  break;
  }

  status( test_interp( x[0],x[1],x[2],x[3], interp, f, atol ,derTol, der2Tol ), "interpolant test");

  {
    std::ofstream outFile("lin4d.out", std::ios_base::out|std::ios_base::trunc );
    OutputArchive oa(outFile);
    oa << BOOST_SERIALIZATION_NVP( interp );
  }
  {
    std::ifstream inFile("lin4d.out");
    InputArchive ia(inFile);
    Interp4D interp2;
    ia >> BOOST_SERIALIZATION_NVP( interp2 );
    status( interp2 == interp, "serialization" );
//    status( test_interp( x[0],x[1],x[2],x[3], interp2, f, 1e-14, derTol, der2Tol ), "interp on reloaded object" );
  }

  return status.ok();
}

std::vector<size_t>
make_vec( const int n0, const int n1, const int n2, const int n3 )
{
  std::vector<size_t> n(4,0);
  n[0]=n0; n[1]=n1; n[2]=n2; n[3]=n3;
  return n;
}

int main()
{
  TestHelper status(true);
  try{
    for( unsigned order=1; order<=4; ++order ){
      cout << "\nOrder: " << order << endl;
      status( test( make_vec( 24, 23, 28, 26 ), true,  order ), "interpolate uniform (21,17,28,26)" );
      status( test( make_vec( 24, 23, 28, 26), false, order ), "interpolate nonuniform (21,17,28,26)" );
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

