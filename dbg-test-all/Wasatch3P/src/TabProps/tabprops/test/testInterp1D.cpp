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

//==============================================================================

void time_it( const InterpT& interp, const double range )
{
  typedef boost::chrono::duration<long long, boost::micro> microseconds;
  typedef boost::chrono::high_resolution_clock Clock;
  Clock::time_point t1 = Clock::now();
  const unsigned n=1000;
  for( int unsigned i=0; i<n; ++i ){
    const double x = double(rand())/double(RAND_MAX) * range;
    interp.value( &x );
  }
  boost::chrono::duration<double> t2 = Clock::now() - t1;
  std::cout << "\t" << std::scientific << std::setprecision(2) << n/t2.count() << " interpolations per second" << std::endl;

  t1 = Clock::now();
  for( int unsigned i=0; i<n; ++i ){
    const double x = double(rand())/double(RAND_MAX) * range;
    interp.derivative( &x, 0 );
  }
  t2 = Clock::now() - t1;
  std::cout << "\t" << n/t2.count() << " derivatives per second" << std::endl;

  t1 = Clock::now();
  for( int unsigned i=0; i<n; ++i ){
    const double x = double(rand())/double(RAND_MAX) * range;
    interp.second_derivative( &x, 0, 0 );
  }
  t2 = Clock::now() - t1;
  std::cout << "\t" << n/t2.count() << " second derivatives per second" << std::endl;
}

//==============================================================================

bool test_uniform( const unsigned order, const int n )
{
  TestHelper status(false);

  std::vector<double> x, fx;
  const double PI = 3.141;
  const double deltax = 2*PI/double(n-1);

  for( int i=0; i<n; ++i ){
    const double xx = i * deltax;
    x.push_back(xx);
    fx.push_back( 2+std::sin(xx) );
  }

  const Interp1D interpf( order, x, fx );
  const Interp1D interpx( order, x,  x );

  // ensure that we exactly interpolate the original points
  for( int i=0; i<n; ++i ){
    const double xx  = i*deltax;
    const double xxi = interpx.value( &xx );
    const double dx  = interpx.derivative( &xx, 0 );

    status( std::abs(xx-xxi) < 3e-14, "x val" );

    const double dxe = 1.0;
    status( std::abs(dx-dxe) < 3e-14, "x der" );
    if( std::abs(dx-dxe) >= 3e-14 ){
      std::cout << "dx failed at x=" << xx << ",  dx: " << dx << ", exact: " << dxe << std::endl;
    }

    const double f  = 2+std::sin(xx);
    const double fi = interpf.value( &xx );

    status( std::abs(f-fi) < 1e-13, "f" );
    if( std::abs(f-fi) > 1e-13 ){
      std::cout <<"*** f: " << f << "," << fi << "," << f-fi << std::endl;
    }
  }

  if( status.isfailed() ){
    std::cout << "The original points are not correctly interpolated!" << std::endl;
  }

//  std::ostringstream fnam; fnam << "1dresults_" << order << ".dat";
//  std::ofstream fout( fnam.str() );

  // note that this tolerance was set to be consistent with the
  // maximum observed error for the value of n=20.
  double relerr = 1e-4, abserr = 1e-14;
  switch( order ){
  case 1 : relerr = 0.015; abserr=3e-1; break;
  case 2 : relerr = 2e-3;  abserr=4e-2; break;
  case 3 : relerr = 3e-4;  abserr=4e-3; break;
  case 4 : relerr = 8e-5;  abserr=3e-3; break;
  default: relerr = 1e-5;  abserr=1e-4; break;
  }

  for( int i=0; i<5*n; ++i ){
    const double xx  = i*2*PI / double(5*n-1);
    const double xxi = interpx.value( &xx );

    status( std::abs(xx-xxi) < 1e-14 );
    if( std::abs(xx-xxi) > 1e-14 ){
      std::cout << "*** " << i << "  " << xx << "," << xxi << "," << xx-xxi << std::endl;
    }

    const double f   = 2+std::sin(xx);
    const double fi  = interpf.value( &xx );
    const double df  = interpf.derivative(&xx,0);
    const double d2f = interpf.second_derivative(&xx,0,0);
    const double dfdx   =  std::cos(xx);
    const double d2fdx2 = -std::sin(xx);

    status( std::abs((f-fi)/(f+1e-3)) < relerr );
    if( std::abs((f-fi)/(fi+1e-3)) > relerr ){
      std::cout << "*** " << f << "," << fi << "," << f-fi << ", " << std::abs((f-fi)/(f+1e-9)) << std::endl;
    }

    status( std::abs(df-dfdx) < abserr );
    if( std::abs(df-dfdx) > abserr ){
      std::cout << "*** " << df << "\t" << dfdx << "\t" << df-dfdx << std::endl;
    }

    status( std::abs(d2f-d2fdx2) < 30*abserr );
    if( std::abs(d2f-d2fdx2) >= 30*abserr ){
      std::cout << "*** d2f/dx2: " << d2f << "\t" << d2fdx2 << "\t" << d2f-d2fdx2 << std::endl;
    }

//    fout << xxi << "\t" << f << "\t" << fi << "\t" << dfdx << "\t" << df << "\t" << d2fdx2 << "\t" << d2f << std::endl;
  }
  if( status.isfailed() ){
    std::cout << "general interpolation failed" << std::endl;
  }

  time_it( interpf, 2*PI );

  {
    std::ofstream outFile("lin1d.out", std::ios_base::out|std::ios_base::trunc );
    OutputArchive oa(outFile);
    oa << BOOST_SERIALIZATION_NVP( interpx );
    oa << BOOST_SERIALIZATION_NVP( interpf );
    outFile.close();

    std::ifstream inFile("lin1d.out");
    InputArchive ia(inFile);
    Interp1D interpx2, interpf2;
    ia >> BOOST_SERIALIZATION_NVP( interpx2 );
    ia >> BOOST_SERIALIZATION_NVP( interpf2 );

    status( interpf2.value( &x[n/2] ) == interpf.value( &x[n/2] ) && interpf == interpf2, "Serialization" );

    InterpT* interp = new Interp1D( order, x, fx );
    std::ofstream outFile2("lin1dptr.out", std::ios_base::out|std::ios_base::trunc );
    OutputArchive oa2(outFile2);
    oa2 << BOOST_SERIALIZATION_NVP( *interp );
    delete interp;
  }

  return status.ok();
}

//==============================================================================

bool test_nonuniform( const unsigned order, const int n )
{
  std::vector<double> x, fx;
  const double xL = 2.0;
  const double dx = xL/double(n-1);
  for( int i=0; i<n; ++i ){
    const double xx = i*dx + (std::rand()/RAND_MAX-0.5)*dx/2;
    x.push_back( xx );
    fx.push_back( x[i]*x[i] );
  }

  const Interp1D interpf( order, x, fx );
  const Interp1D interpx( order, x,  x );

  TestHelper status(false);

  status( interpf.get_bounds()[0].first  == x[0  ], "low bound"  );
  status( interpf.get_bounds()[0].second == x[n-1], "high bound" );

  // ensure that we exactly interpolate the original points
  for( int i=0; i<n; ++i ){
    const double xx  = x[i];
    const double xxi = interpx.value( &xx );

    status( std::abs(xx-xxi) < 1e-15 );

    const double f  = xx*xx;
    const double fi = interpf.value( &xx );

    status( std::abs(f-fi) < 1e-15 );
    if( std::abs(f-fi) > 1e-15 ){
      std::cout << f << " : " << fi << "\t " << std::abs(f-fi) << std::endl;
    }
  }

  if( status.isfailed() ){
    std::cout << "The original points are not correctly interpolated!" << std::endl;
  }

  time_it( interpf, xL/double(n*n) );

  {
    std::ofstream outFile("lin1d.out", std::ios_base::out|std::ios_base::trunc );
    OutputArchive oa(outFile);
    oa << BOOST_SERIALIZATION_NVP( interpx );
    oa << BOOST_SERIALIZATION_NVP( interpf );
    outFile.close();

    std::ifstream inFile("lin1d.out");
    InputArchive ia(inFile);
    Interp1D interpx2, interpf2;
    ia >> BOOST_SERIALIZATION_NVP( interpx2 );
    ia >> BOOST_SERIALIZATION_NVP( interpf2 );

    status( interpf.value( &x[n/2] ) == interpf2.value( &x[n/2] ) && interpf == interpf2, "Serialization" );
  }

  InterpT* myClone = interpf.clone();
  status( *myClone == interpf, "clone" );
  delete myClone;

  return status.ok();
}

//==============================================================================

int main()
{
  TestHelper status(true);
  try{
    const int npts = 20;
    for( unsigned order=1; order<=4; ++order ){
      std::cout << std::endl << "Testing for order = " << order << std::endl;
      status( test_uniform(order,npts), "uniform mesh interpolant" );
      status( test_nonuniform(order,npts), "non-uniform mesh interpolant" );
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

//==============================================================================

