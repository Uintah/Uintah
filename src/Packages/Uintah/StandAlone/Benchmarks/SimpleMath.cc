/*
 *  SimpleMath.cc: Simple math benchmark.
 *
 *  Written by:
 *   Randy Jones
 *   Department of Computer Science
 *   University of Utah
 *   July 31, 2003
 *
 *  Copyright (C) 2000 U of U
 */

#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Thread/Time.h>
#include <iostream>

using namespace SCIRun;
using namespace std;

const long SIZE_DEFAULT = 400;

void usage ( void )
{
  cerr << "Usage: SimpleMath <size>" << endl;
  cerr << endl;
  cerr << "  <size>    This must be a number greater than zero." << endl;
  cerr << endl;
  cerr << "  This program will perform the operation 'result = a*x+b'" << endl;
  cerr << "  <size> cubed times using Uintah data structures and" << endl;
  cerr << "  calculate the MFLOPS achieved." << endl;
}

int main ( int argc, char** argv )
{
  long size = SIZE_DEFAULT;

  /*
   * Parse arguments
   */
  if ( argc > 1 ) {
     size = strtol( argv[1], (char**)NULL, 10 );

     if (size <= 0) {
       usage();
       return EXIT_FAILURE;
     }
  }
  else {
    usage();
    return EXIT_FAILURE;
  }

  cout << "Simple Math Benchmark: Using size of " << size << endl;

  IntVector low ( 0,0,0 );
  IntVector high( size,size,size );

  CCVariable<double> result, a, x, b;

  result.allocate( low, high );
  a.allocate( low, high );
  x.allocate( low, high );
  b.allocate( low, high );
  
  a.initialize( 5 );
  x.initialize( 6 );
  b.initialize( 2 );

  double startTime = Time::currentSeconds();

  for ( CellIterator iter(low, high); !iter.done(); iter++ )
    result[*iter] = a[*iter] * x[*iter] + b[*iter];

  double deltaTime = Time::currentSeconds() - startTime;
  double megaFlops = (size * size * size * 2.0) / 1000000.0 / deltaTime;

  cout << "Completed in " << deltaTime << " seconds.";
  cout << " (" << megaFlops << " MFLOPS)" << endl;

  return EXIT_SUCCESS;
}
