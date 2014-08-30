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


\file heat_equation.cpp

\page example-heat-equation Example: Solving a Heat Equation

# Goal
Illustrate how to use Nebo to implement a simple two-dimensional heat transfer simulation:
\f[
\frac{\partial \phi}{\partial t}
 = \frac{\partial}{\partial x}\left(\alpha\frac{\partial \phi}{\partial x}\right)
 + \frac{\partial}{\partial y}\left(\alpha\frac{\partial \phi}{\partial y}\right)
\f]

# Key Concepts
 -# Use Nebo assignment, reductions, stencils, masks, and `cond` to create a simple heat transfer equation.
 -# Show a very simple way to implement Dirichlet boundary conditions.
 -# Use ghost fields in conjunction with operators.

\sa \ref example-stencil-type-inference
\sa \ref example-stencils
\sa \ref example-masks

# Example Code
\c examples/heat_equation.cpp
\include heat_equation.cpp

*/


#include <spatialops/structured/FVStaggered.h>
#include <spatialops/structured/Grid.h>
#include <spatialops/structured/FieldHelper.h>

using namespace SpatialOps;

// If we are compiling with GPU CUDA support, create fields on the device.
// Otherwise, create them on the host.
#ifdef ENABLE_CUDA
# define LOCATION GPU_INDEX
#else
# define LOCATION CPU_INDEX
#endif

//==============================================================================

template<typename FieldT>
void
initialize_thermal_diffusivity( const Grid& grid, FieldT & alpha )
{
  SpatFldPtr<FieldT> x = SpatialFieldStore::get<FieldT>( alpha );
  SpatFldPtr<FieldT> y = SpatialFieldStore::get<FieldT>( alpha );

  grid.set_coord<XDIR>( *x );
  grid.set_coord<YDIR>( *y );
  //alpha <<= 0.1 + 0.4 * (x + y + 1.0);
  //alpha <<= 0.1 + 0.4 * (x + y);
  //alpha <<= (x + y + .2) /2;
  alpha <<= 1.0;

# ifdef ENABLE_CUDA
  alpha.add_device( CPU_INDEX );  // transfer to facilitate printing its values
# endif
  std::cout << "Initial alpha:" << std::endl;
  print_field( alpha, std::cout, true );
}

//==============================================================================

void
initialize_mask_points( const Grid& grid,
                        std::vector<IntVec>& leftSet,
                        std::vector<IntVec>& rightSet,
                        std::vector<IntVec>& topSet,
                        std::vector<IntVec>& bottomSet )
{
  for( int i=-1; i<=grid.extent()[1]; ++i ){
    leftSet .push_back( IntVec(-1, i, 0) );
    rightSet.push_back( IntVec(grid.extent()[0], i, 0) );
  }

  for(int i = -1; i <= grid.extent()[0]; i++){
    topSet   .push_back( IntVec(i, -1, 0) );
    bottomSet.push_back( IntVec(i, grid.extent()[1], 0) );
  }
}

//==============================================================================

template<typename FieldT>
double
find_deltaT( const FieldT& alpha, const Grid& grid )
{
  const double deltaX = grid.spacing<XDIR>();
  const double deltaY = grid.spacing<YDIR>();
  const double sqrdDeltaX = deltaX * deltaX;
  const double sqrdDeltaY = deltaY * deltaY;
  const double sqrdDeltaXYmult = sqrdDeltaX * sqrdDeltaY;
  const double sqrdDeltaXYplus = sqrdDeltaX + sqrdDeltaY;

  return 0.25 * sqrdDeltaXYmult / ( sqrdDeltaXYplus * nebo_min(alpha) );
}

//==============================================================================

int main()
{
  typedef SVolField FieldT;

  //----------------------------------------------------------------------------
  // Define the domain size and number of points
  const DoubleVec domainLength(1,1,1);  // a cube of unit length
  const IntVec fieldDim( 6, 6, 1 );     // 6 x 6 x 1 points

  //----------------------------------------------------------------------------
  // Create fields
  const GhostData nghost(1);
  const BoundaryCellInfo bcInfo = BoundaryCellInfo::build<FieldT>( true, true, true );
  const MemoryWindow window( get_window_with_ghost( fieldDim, nghost, bcInfo) );

  FieldT   phi( window, bcInfo, nghost, NULL, InternalStorage, LOCATION );
  FieldT   rhs( window, bcInfo, nghost, NULL, InternalStorage, LOCATION );
  FieldT alpha( window, bcInfo, nghost, NULL, InternalStorage, LOCATION );

  const Grid grid( fieldDim, domainLength );

  //----------------------------------------------------------------------------
  // Initialize alpha, thermal diffusivity
  initialize_thermal_diffusivity( grid, alpha );

  //----------------------------------------------------------------------------
  // Build and initialize masks:
  std::vector<IntVec> leftSet, rightSet, topSet, bottomSet;

  initialize_mask_points( grid, leftSet, rightSet, topSet, bottomSet );

  SpatialMask<FieldT> left  ( phi, leftSet   );
  SpatialMask<FieldT> right ( phi, rightSet  );
  SpatialMask<FieldT> top   ( phi, topSet    );
  SpatialMask<FieldT> bottom( phi, bottomSet );

# ifdef ENABLE_CUDA
  // Masks are created on CPU so we need to explicitly transfer them to GPU
  left  .add_consumer( GPU_INDEX );
  right .add_consumer( GPU_INDEX );
  top   .add_consumer( GPU_INDEX );
  bottom.add_consumer( GPU_INDEX );
# endif

  //----------------------------------------------------------------------------
  // Build stencils:
  OperatorDatabase opDB;         // holds stencils that can be retrieved easily
  build_stencils( grid, opDB );  // builds stencils and places them in opDB

  typedef BasicOpTypes<SVolField>::DivX          DivX;  // x-divergence operator type
  typedef BasicOpTypes<SVolField>::GradX        GradX;  // x-gradient operator type
  typedef BasicOpTypes<SVolField>::InterpC2FX InterpX;  // x-interpolant operator type

  const DivX&       divX = *opDB.retrieve_operator<DivX   >(); // retrieve the DivX operator
  const GradX&     gradX = *opDB.retrieve_operator<GradX  >(); // retrieve the GradX operator
  const InterpX& interpX = *opDB.retrieve_operator<InterpX>(); // retrieve the InterpX operator

  typedef BasicOpTypes<SVolField>::DivY          DivY;  // y-divergence operator type
  typedef BasicOpTypes<SVolField>::GradY        GradY;  // y-gradient operator type
  typedef BasicOpTypes<SVolField>::InterpC2FY InterpY;  // y-interpolant operator type

  const DivY&       divY = *opDB.retrieve_operator<DivY   >(); // retrieve the DivY operator
  const GradY&     gradY = *opDB.retrieve_operator<GradY  >(); // retrieve the GradY operator
  const InterpY& interpY = *opDB.retrieve_operator<InterpY>(); // retrieve the InterpY operator

  //----------------------------------------------------------------------------
  // Determine a safe deltaT:
  const double deltaT = find_deltaT( alpha, grid );

  //----------------------------------------------------------------------------
  // Initialize phi:
  phi <<= cond( left, 10.0 )
              ( right, 0.0 )
              ( 5.0 );

  std::cout << "Initial phi:" << std::endl;
# ifdef ENABLE_CUDA
  phi.add_device( CPU_INDEX );
# endif
  print_field( phi, std::cout, true );

  //----------------------------------------------------------------------------
  // Take time steps:
  const int nSteps = 40;
  const int printInterval = 1;
  for( int i=0; i<=nSteps; ++i ){

    // Calculate the RHS (change in phi)
    rhs <<= divX( interpX(alpha) * gradX(phi) )
          + divY( interpY(alpha) * gradY(phi) );

    // update the solution
    phi <<= phi + deltaT * rhs;

    // Reset boundaries:
    phi <<= cond( left,         10.0 )
                ( right,         0.0 )
                ( top || bottom, 5.0 )
                ( phi );

    // print current state:
    if( i%printInterval == 0 ){
      std::cout << "phi after " << i + 1 << " time steps:" << std::endl;

#     ifdef ENABLE_CUDA
      // If f uses GPU memory, f needs to be copied to CPU memory to print it.
      phi.add_device( CPU_INDEX );
#     endif

      print_field( phi, std::cout, true );
    }
  }

  return 0;
}
