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



\file   stencil_type_inference.cpp
\date   Jul 10, 2014
\author James C. Sutherland

\page example-stencil-type-inference Inferring operator types

# Goal
Illustrate how to obtain stencil operator types.


# Key Concepts

 -# There are a number of predefined operators/stencils in SpatialOps. They
    can be obtained via two simple ways:
    - The simplest way to obtain these is via the \link SpatialOps::BasicOpTypes
      BasicOpTypes\endlink struct.  This requires the volume field type and
      provides the types of all operators associated with that field.
    - Alternatively, one can use the \link SpatialOps::OperatorTypeBuilder
      OperatorTypeBuilder \endlink struct, which provides a bit more flexibility
      but requires more type knowledge.

    These approaches are both illustrated in the example below.

 -# Each stencil defines the type of field that it consumes (the source field)
    as well as the type of field it produces (the destination field) as follows:
    \code{.cpp}
     typedef OpType::SrcFieldType  SrcType;
     typedef OpType::DestFieldType DestType;
    \endcode

 -# Because of this strong typing and type inference, SpatialOps can guarantee,
    at compile time, compatibility between fields and operators.

\sa \ref example-3d-laplacian, which uses the concepts shown here to
    create a very generic 3D Laplacian function.

\sa The \ref optypes module, which provides additional documentation on operator type inference.

# Example Code
\c examples/stencil_type_inference.cpp
\include stencil_type_inference.cpp


*/

#include <spatialops/structured/FVStaggered.h> // defines field and operator types for structured meshes
#include <spatialops/SpatialOpsTools.h>        // for is_same_type

#include <cassert>

using namespace SpatialOps;

int main()
{
  //----------------------------------------------------------------------------
  // Illustrate type inference to obtain various operator types from a given
  // volume field type.  If an invalid "volume" field is provided, a compiler
  // error will result.
  typedef BasicOpTypes<SVolField>::GradX        GradX;  // x-derivative operator type
  typedef BasicOpTypes<SVolField>::GradY        GradY;  // y-derivative operator type
  typedef BasicOpTypes<SVolField>::GradZ        GradZ;  // z-derivative operator type

  typedef BasicOpTypes<SVolField>::InterpC2FX InterpX;  // x-interpolant operator type
  typedef BasicOpTypes<SVolField>::InterpC2FY InterpY;  // y-interpolant operator type
  typedef BasicOpTypes<SVolField>::InterpC2FZ InterpZ;  // z-interpolant operator type

  typedef BasicOpTypes<SVolField>::DivX          DivX;  // x-divergence operator
  typedef BasicOpTypes<SVolField>::DivY          DivY;  // y-divergence operator
  typedef BasicOpTypes<SVolField>::DivZ          DivZ;  // z-divergence operator


  //----------------------------------------------------------------------------
  // Illustrate more general way to obtain operator types using OperatorTypeBuilder
  // Here, we use the operation we want to accomplish (e.g., Interpolant) as
  // well as the source and destination field types to obtain the operator type
  // of interest.  If no such operator has been defined, a compiler error will result.
  typedef OperatorTypeBuilder< Interpolant, SVolField,   SSurfXField >::type InterpX2;
  typedef OperatorTypeBuilder< Gradient,    SVolField,   SSurfXField >::type GradX2;
  typedef OperatorTypeBuilder< Divergence,  SSurfXField, SVolField   >::type DivX2;

  assert(( is_same_type< InterpX,InterpX2 >() ));
  assert(( is_same_type< GradX,  GradX2   >() ));
  assert(( is_same_type< DivX,   DivX2    >() ));


  //----------------------------------------------------------------------------
  // Operators define the types of fields that they operate on (source field types)
  // as well as the type of field that they produce (destination field type)
  assert(( is_same_type< InterpX::SrcFieldType,  SVolField  >() ));
  assert(( is_same_type< InterpX::DestFieldType, SSurfXField>() ));

  assert(( is_same_type< DivY::SrcFieldType,  SSurfYField>() ));
  assert(( is_same_type< DivY::DestFieldType, SVolField  >() ));


  //----------------------------------------------------------------------------
  // The examples above were dealing with the non-staggered mesh (with field types
  // SVolField, SSurfXField, etc.) but we can also apply type inference for the
  // staggered meshes.
  typedef BasicOpTypes<XVolField>::GradX    XVolGradX;
  typedef BasicOpTypes<XVolField>::DivX     XVolDivX;

  assert(( is_same_type< XVolGradX::DestFieldType, XVolDivX::SrcFieldType>() ));

  return 0;
}
