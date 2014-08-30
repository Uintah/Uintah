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



\file   3D_Laplacian.h
\date   Jul 10, 2014
\author James C. Sutherland

\page example-3d-laplacian The 3D Laplacian in action

Using the laplacian operator,
\f$
\frac{\partial^2}{\partial x} +
\frac{\partial^2}{\partial y} +
\frac{\partial^2}{\partial z}
\f$,
as a case-study, this example shows how to create a function template to compute
the 3D laplacian on compatible field types.

# Goals
 - Illustrate how to use stencil operators
 - Illustrate type inference from fields to operators

# Key Concepts
 -# There are a number of predefined operators/stencils in SpatialOps.  The
    simplest way to obtain these is via the \link SpatialOps::BasicOpTypes
    BasicOpTypes\endlink struct.
 -# Stencils can be stored and retrieved in the \link SpatialOps::OperatorDatabase
    OperatorDatabase\endlink which provides simple storage and retrieval of
    operators by type.
 -# Using type inference, highly generic code can be written that is also very
    robust.

\sa \ref example-field-types
\sa \ref example-stencil-type-inference
\sa \ref example-stencils

# Example Code
\c examples/3D_Laplacian.h
\include 3D_Laplacian.h

*/

#include <spatialops/structured/FVStaggered.h>


// Here FieldT is a generic field that is assumed to live on a volume.
//
// Valid input field types:
// ------------------------
//   SVolField - non-staggered (scalar) volume field
//   XVolField - x-staggered volume field (x-velocity/momentum)
//   YVolField - y-staggered volume field (y-velocity/momentum)
//   ZVolField - z-staggered volume field (z-velocity/momentum)
//
// Inputs:
// -------
//  opDB - the OperatorDatabase that holds all operators
//  src  - the source field (to apply the Laplacian to)
//
// Output:
// -------
//  dest - the destination field (result of Laplacian applied to src)
//
template< typename FieldT >
void
calculate_laplacian_3D( const SpatialOps::OperatorDatabase& opDB,
                        const FieldT& src,
                        FieldT& dest )
{
  using namespace SpatialOps;

  //---------------------------------------------------------------------------
  // Infer operator types for the 3D laplacian
  // If this function is used with a field type that is not a volume field
  // then compiler errors result due to invalid type inference for operators.
  typedef typename BasicOpTypes< FieldT >::GradX GradX;
  typedef typename BasicOpTypes< FieldT >::GradY GradY;
  typedef typename BasicOpTypes< FieldT >::GradZ GradZ;

  typedef typename BasicOpTypes< FieldT >::DivX  DivX;
  typedef typename BasicOpTypes< FieldT >::DivY  DivY;
  typedef typename BasicOpTypes< FieldT >::DivZ  DivZ;


  //---------------------------------------------------------------------------
  // obtain the appropriate operators from the operator database
  // The OperatorDatabase is simply a container for operators that
  // provides access by type.
  GradX& gradX = *opDB.retrieve_operator<GradX>();
  GradY& gradY = *opDB.retrieve_operator<GradY>();
  GradZ& gradZ = *opDB.retrieve_operator<GradZ>();

  DivX&  divX  = *opDB.retrieve_operator<DivX >();
  DivY&  divY  = *opDB.retrieve_operator<DivY >();
  DivZ&  divZ  = *opDB.retrieve_operator<DivZ >();


  //---------------------------------------------------------------------------
  // Finally, perform the calculation.  This is the easy part!
  // Note that this will run on multicore and GPU!
  dest <<= divX( gradX( src ) )
         + divY( gradY( src ) )
         + divZ( gradZ( src ) );
}
