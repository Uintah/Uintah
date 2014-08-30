/**
The MIT License

Copyright (c) 2014 The University of Utah

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.


\file   field_operations.cpp
\date   Jul 6, 2014
\author "James C. Sutherland"

\page example-field-operations  Operations using fields

# Goal of this example
This example will show how to use SpatialFields in SpatialOps to perform a
variety of field operations.

# Key Concepts

 -# Most mathematical operations/functions are supported in Nebo's
    \link NeboOperations field operations \endlink.  They are lifted over fields.
    This includes standard mathematical
    operations such as \c sin \c cos \c exp, etc.

 -# Nebo supports conditional operations that unroll to the equivalent of point-wise
    \c if statements.  These are achieved via the \link NeboCond nebo cond \endlink
    construct.

 -# For fields with CPU-allocated memory, \c print_field will write the field with
    some formatting to the given output stream. Note that \c print_field
    prints the lowest index first and ends with the highest index.


\sa \ref example-field-creation
\sa \ref NeboOperations

# Example Code
\c examples/field_operations.cpp
\include field_operations.cpp

# Try This
Modify \c field_operations.cpp to do the following:
 -# Create a function `sin(x)` and then clip its values to be within [-0.1,0.2].

*/

#include <spatialops/structured/FVStaggered.h>
#include <spatialops/structured/Grid.h>        // a convenient way to define coordinates
#include <spatialops/structured/FieldHelper.h> // provides methods (print_field()) to view small fields

using namespace SpatialOps;

// If we are compiling with GPU CUDA support, create fields on the device.
// Otherwise, create them on the host.
#ifdef ENABLE_CUDA
# define LOCATION GPU_INDEX
#else
# define LOCATION CPU_INDEX
#endif

int main()
{
  // Define the size of the domain
  const IntVec fieldDim( 5, 5, 1 );         //  (nx,ny,nz)
  const DoubleVec length( 1.0, 1.0, 1.0 );  // a cube of unit length

  //----------------------------------------------------------------------------
  // Create fields:
  typedef SpatialOps::SVolField FieldT;

  // Use default values to create objects that are required to construct a field.
  const GhostData nghost(0);
  const BoundaryCellInfo bcInfo = BoundaryCellInfo::build<FieldT>( true, true, true );
  const MemoryWindow window( get_window_with_ghost( fieldDim, nghost, bcInfo ) );

  FieldT x( window, bcInfo, nghost, NULL, InternalStorage, LOCATION );
  FieldT y( window, bcInfo, nghost, NULL, InternalStorage, LOCATION );
  FieldT z( window, bcInfo, nghost, NULL, InternalStorage, LOCATION );

  FieldT f( window, bcInfo, nghost, NULL, InternalStorage, LOCATION );
  FieldT g( window, bcInfo, nghost, NULL, InternalStorage, LOCATION );

  //----------------------------------------------------------------------------
  // Build a grid. This is a convenient way to set coordinate values that will
  // be used below.
  const Grid grid( fieldDim, length );
  grid.set_coord<XDIR>(x);
  grid.set_coord<YDIR>(y);
  grid.set_coord<ZDIR>(z);

  //----------------------------------------------------------------------------
  // Print fields to standard output:
  std::cout << "x:" << std::endl;   print_field( x, std::cout );
  std::cout << "y:" << std::endl;   print_field( y, std::cout );
  std::cout << "z:" << std::endl;   print_field( z, std::cout );

  //----------------------------------------------------------------------------
  // Perform operations on fields

  // Assign field values.  Note that this is a vectorized statement that will
  // work on GPU, serial CPU and multicore CPU.
  f <<= sin(x) + cos(y) + tanh(z);

# ifdef ENABLE_CUDA
  //If f uses GPU memory, to print f, f needs to be copied to CPU memory.
  f.add_device( CPU_INDEX );
# endif
  std::cout << "f:" << std::endl;
  print_field( f, std::cout );

  //----------------------------------------------------------------------------
  // Conditional (if/then/else...) evaluation.
  // cond in Nebo creates conditional expressions. Each cond clause, except the last,
  // must have two arguments.  The first argument is the condition (if this), and the
  // second is the result (then that).  The final clause takes only one argument
  // (else other), which is returned only if all previous conditions fail.
  //
  // The conditions are evaluated first to last, and evaluation stops when a
  // condition returns true. Thus, the order of conditions can and will effect
  // the results.
  g <<= cond( f > 2.0, x+z )          // if     ( f[i] > 2.0 ) g[i] = x[i]+z[i];
            ( f > 1.5, y   )          // else if( f[i] > 1.5 ) g[i] = y[i];
            ( f > 1.0, -f  )          // else if( f[i] > 1.0 ) g[i] = -f[i];
            ( f );                    // else                  g[i] = f[i];

# ifdef ENABLE_CUDA
  //If g uses GPU memory, to print g, g needs to be copied to CPU memory.
  g.add_device( CPU_INDEX );
# endif
  std::cout << "g:" << std::endl;
  print_field( g, std::cout );

  return 0;
}
