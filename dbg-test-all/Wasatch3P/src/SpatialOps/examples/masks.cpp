/**
 *
 * The MIT License
 *
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
 *


\file masks.cpp
\page example-masks Example of using Masks with fields

# Goal
Illustrate how to use Nebo masks.

# Key Concepts
 -# A mask is a set of points where conditions are different and require different
    calculations.
 -# A mask is used in a `cond` expression in Nebo, and can be used on CPU or GPU.

# Example Code
\c examples/masks.cpp
  \include masks.cpp

# Try this:
Modify \c masks.cpp to do the following:
 - Modify the number of ghost cells (try 1 or 2) and notice what happens to the output.
   Note that you can also specify different number of ghost cells per face via
   \code{.cpp} GhostData nghost( nxMinus, nxPlux, nyMinus, nyPlus, nzMinus, nzPlus ) \endcode
   or the same number of ghosts on each face via
   \code{.cpp} GhostData nghost( n ); \endcode
   as in the example.
 - Modify the mask points
 - Modify the `cond` statement to obtain different values on the masked points.
   For example, try this:
   \code{.cpp}
    f <<= cond( mask, 10.0 - f )
              ( f );
   \endcode
 */

#include <spatialops/structured/FVStaggered.h>
#include <spatialops/structured/Grid.h>
#include <spatialops/structured/FieldHelper.h>

using namespace SpatialOps;
using namespace std;

// If we are compiling with GPU CUDA support, create fields on the device.
// Otherwise, create them on the host.
#ifdef ENABLE_CUDA
# define LOCATION GPU_INDEX
#else
# define LOCATION CPU_INDEX
#endif

int main()
{
  //----------------------------------------------------------------------------
  // Define the domain size and number of points
  const DoubleVec length( 5, 5, 5 ); // a cube of length 5.0
  const IntVec fieldDim( 5, 5, 1 );  // a 5 x 5 x 1 problem

  //----------------------------------------------------------------------------
  // Create fields
  const GhostData nghost(0);  // try changing this to 1 and see what happens.
  const BoundaryCellInfo sVolBCInfo = BoundaryCellInfo::build<SVolField>( true, true, true );
  const MemoryWindow sVolWindow( get_window_with_ghost( fieldDim, nghost, sVolBCInfo) );

  SVolField x( sVolWindow, sVolBCInfo, nghost, NULL, InternalStorage, LOCATION );
  SVolField y( sVolWindow, sVolBCInfo, nghost, NULL, InternalStorage, LOCATION );
  SVolField f( sVolWindow, sVolBCInfo, nghost, NULL, InternalStorage, LOCATION );

  //----------------------------------------------------------------------------
  // Build a coordinates.
  const Grid grid( fieldDim, length );
  grid.set_coord<XDIR>(x);
  grid.set_coord<YDIR>(y);

  //----------------------------------------------------------------------------
  // Assign f:
  f <<= x + y;

  //----------------------------------------------------------------------------
  // Build a mask.
  // A mask is a set of indices where something different should happen, such
  // as a boundary condition: Edge of a flame, fuel injection or exhaust pipe.
  // Indexing for masks has the origin at the first interior point.
  // (Ghost points are negative indices.)
  vector<IntVec> maskSet;
  for( int i=0; i<fieldDim[1]; ++i ){
    maskSet.push_back( IntVec(0, i, 0) );
  }

  // Creating a mask requires a prototype field and a vector of indices.
  SpatialMask<SVolField> mask( f, maskSet );
# ifdef ENABLE_CUDA
  // Note that these are created only on a CPU, and must be explicitly
  // transferred to the GPU for CUDA enabled cases as shown below.
  mask.add_consumer( GPU_INDEX );
# endif

  //----------------------------------------------------------------------------
  // Use the mask:
# ifdef ENABLE_CUDA
  // If f uses GPU memory, f needs to be copied to CPU memory to print it.
  f.add_device( CPU_INDEX );
# endif
  cout << "f before applying mask:" << endl;
  print_field( f, cout );

  f <<= cond( mask, 90.0 )
            ( f );

# ifdef ENABLE_CUDA
  // If f uses GPU memory, f needs to be copied to CPU memory to print it.
  f.add_device( CPU_INDEX );
# endif
  cout << "f after applying mask:" << endl;
  print_field( f, cout );

  return 0;
}
