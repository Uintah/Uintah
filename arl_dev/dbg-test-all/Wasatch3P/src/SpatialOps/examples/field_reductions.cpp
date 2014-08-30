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


\file   field_reductions.cpp
\date   Jul 10, 2014
\author "James C. Sutherland"


\page example-field-reductions Performing reduction operators (min/max, etc.) on fields

# Goal of this example
This example will show how to use SpatialFields in SpatialOps to perform
reduction operations (norm, max, min) on field

# Key Concepts

 -# Nebo supports reduction operations on fields.
 -# Currently, field reductions only have support for CPU fields.

\sa \ref example-field-creation
\sa \ref example-field-operations

# Example Code
\c examples/field_reductions.cpp
\include field_reductions.cpp
*/


#include <spatialops/structured/FVStaggered.h>
#include <spatialops/structured/Grid.h>

#include <iostream>
#include <iomanip>

using namespace SpatialOps;
using namespace std;

int main()
{
  typedef SVolField FieldT;

  // Define the size and number of points in the domain.
  const IntVec fieldDim( 10, 10, 10 );            // 10 x 10 x 10 points
  const DoubleVec length( 1.0, 1.0, 1.0 );  // a cube of unit length

  //----------------------------------------------------------------------------
  // Create fields of type FieldT.

  const bool bcx=true, bcy=true, bcz=true;
  const GhostData nghost(0);
  const BoundaryCellInfo bcInfo = BoundaryCellInfo::build<FieldT>( bcx, bcy, bcz );
  const MemoryWindow window( get_window_with_ghost( fieldDim, nghost, bcInfo ) );

  FieldT x( window, bcInfo, nghost, NULL, InternalStorage );
  FieldT y( window, bcInfo, nghost, NULL, InternalStorage );
  FieldT z( window, bcInfo, nghost, NULL, InternalStorage );

  FieldT f( window, bcInfo, nghost, NULL, InternalStorage );

  //----------------------------------------------------------------------------
  // Build coordinates.
  const Grid grid( fieldDim, length );
  grid.set_coord<XDIR>(x);
  grid.set_coord<YDIR>(y);
  grid.set_coord<ZDIR>(z);

  //----------------------------------------------------------------------------
  // Perform operations on fields.

  f <<= x + y + z;  // set values in a field

  const double fnorm = nebo_norm( f );  // L-2 norm of f
  const double fmax  = nebo_max ( f );  // maximum value in f
  const double fmin  = nebo_min ( f );  // minimum value in f

  //----------------------------------------------------------------------------
  // Print out the domain extents for this field type as well as the max and min
  // of the "f" field that we created above.
  cout << setprecision(4) << left
       << " X-min:  " << setprecision(4) << left << setw(10) << nebo_min(x)
       << " X-max:  " << setprecision(4) << left << setw(10) << nebo_max(x) << endl
       << " Y-min:  " << setprecision(4) << left << setw(10) << nebo_min(y)
       << " Y-max:  " << setprecision(4) << left << setw(10) << nebo_max(y) << endl
       << " Z-min:  " << setprecision(4) << left << setw(10) << nebo_min(z)
       << " Z-max:  " << setprecision(4) << left << setw(10) << nebo_max(z) << endl
       << " f-Min:  " << setprecision(4) << left << setw(10) << fmin
       << " f-Max:  " << setprecision(4) << left << setw(10) << fmax        << endl
       << " f-Norm: " << setprecision(4) << left << setw(10) << fnorm
       << endl << endl;

  return 0;
}
