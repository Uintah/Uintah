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


\file   field_types.cpp
\date   Jul 10, 2014
\author James C. Sutherland


\page example-field-types  Some natively supported field types

# Goal of this example
This example will show all 16 Field types and illustrate how they are different.

# Key Concepts

 -# This example also illustrates the different layouts of fields.

\sa \ref example-field-creation
\sa \ref example-field-type-inference
\sa The \ref fieldtypes module, which documents the field types and type
 inference tools in SpatialOps.

# Example Code
\c examples/field_types.cpp
\include field_types.cpp
*/


#include <spatialops/structured/FVStaggered.h>
#include <spatialops/structured/Grid.h>
#include <spatialops/structured/FieldHelper.h>

#include <iostream>
#include <iomanip>
#include <string>

using namespace SpatialOps;
using namespace std;

//==============================================================================

template<typename FieldT>
void driver( const string tag )
{
  cout << tag << std::endl;

  // Define the size and number of points in the domain
  const IntVec fieldDim( 5, 5, 1 );               // 5 x 5 x 1 points
  const DoubleVec domainLength( 5.0, 5.0, 5.0 );  // a cube of length 5.0

  //----------------------------------------------------------------------------
  // Create coordinate fields of type FieldT.
  const bool bcx=true, bcy=true, bcz=true;
  const GhostData nghost(0);
  const BoundaryCellInfo bcInfo = BoundaryCellInfo::build<FieldT>( bcx, bcy, bcz );
  const MemoryWindow window( get_window_with_ghost( fieldDim, nghost, bcInfo ) );

  FieldT x( window, bcInfo, nghost, NULL, InternalStorage );
  FieldT y( window, bcInfo, nghost, NULL, InternalStorage );
  FieldT z( window, bcInfo, nghost, NULL, InternalStorage );

  //----------------------------------------------------------------------------
  // Set coordinate fields.
  const Grid grid( fieldDim, domainLength );
  grid.set_coord<XDIR>(x);
  grid.set_coord<YDIR>(y);
  grid.set_coord<ZDIR>(z);

  //----------------------------------------------------------------------------
  // Print coordinates
  cout << "x:" << std::endl;   print_field( x, cout );
  cout << "y:" << std::endl;   print_field( y, cout );
  cout << "z:" << std::endl;   print_field( z, cout );
}

//==============================================================================

int main()
{
  driver<  SVolField>( "SVolField   - volume field on the scalar volume" );
  driver<SSurfXField>( "SSurfXField - x-surface field on the scalar volume" );
  driver<SSurfYField>( "SSurfYField - y-surface field on the scalar volume" );
  driver<SSurfZField>( "SSurfZField - z-surface field on the scalar volume" );

  driver<  XVolField>( "XVolField   - volume field on the x-staggered volume" );
  driver<XSurfXField>( "XSurfXField - x-surface field on the x-staggered volume" );
  driver<XSurfYField>( "XSurfYField - y-surface field on the x-staggered volume" );
  driver<XSurfZField>( "XSurfZField - z-surface field on the x-staggered volume" );

  driver<  YVolField>( "YVolField   - volume field on the y-staggered volume" );
  driver<YSurfXField>( "YSurfXField - x-surface field on the y-staggered volume" );
  driver<YSurfYField>( "YSurfYField - y-surface field on the y-staggered volume" );
  driver<YSurfZField>( "YSurfZField - z-surface field on the y-staggered volume" );

  driver<  ZVolField>( "ZVolField   - volume field on the z-staggered volume" );
  driver<ZSurfXField>( "ZSurfXField - x-surface field on the z-staggered volume" );
  driver<ZSurfYField>( "ZSurfYField - y-surface field on the z-staggered volume" );
  driver<ZSurfZField>( "ZSurfZField - z-surface field on the z-staggered volume" );

  return 0;
}
