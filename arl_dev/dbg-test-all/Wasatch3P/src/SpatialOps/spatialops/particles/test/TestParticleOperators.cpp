
#include <set>
#include <iostream>
using std::cout;
using std::endl;

#include <spatialops/structured/FVStaggeredFieldTypes.h>

#include <spatialops/particles/ParticleFieldTypes.h>
#include <spatialops/particles/ParticleOperators.h>

#include <spatialops/structured/FieldHelper.h>
#include <spatialops/structured/Grid.h>

#include <test/TestHelper.h>

typedef SpatialOps::SVolField CellField;
using namespace SpatialOps;

int main()
{
  const IntVec dim(10,1,1);
  const DoubleVec length( 10,1,1 );
  const Grid grid( dim, length );

  const GhostData cg(3);
  const BoundaryCellInfo cbc = BoundaryCellInfo::build<CellField>();
  const MemoryWindow mw = get_window_with_ghost( dim, cg, BoundaryCellInfo::build<CellField>(false,false,false) );

  //
  // build the fields
  //
  CellField xcoord( mw, cbc, cg, NULL );
  CellField f     ( mw, cbc, cg, NULL );
  CellField ctmp  ( mw, cbc, cg, NULL );

  const GhostData pg(0);
  const BoundaryCellInfo pbc = BoundaryCellInfo::build<SpatialOps::Particle::ParticleField>();
  const size_t np=1;
  const MemoryWindow pmw( IntVec(np,1,1) );

  SpatialOps::Particle::ParticleField pCoord( pmw, pbc, pg, NULL );
  SpatialOps::Particle::ParticleField pSize ( pmw, pbc, pg, NULL );
  SpatialOps::Particle::ParticleField pfield( pmw, pbc, pg, NULL );
  SpatialOps::Particle::ParticleField ptmp  ( pmw, pbc, pg, NULL );


  grid.set_coord<SpatialOps::XDIR>( xcoord ); // set the coordinates.
  f <<= 10*xcoord;                            // set the field values
//  print_field( xcoord, std::cout );
//  print_field( f, std::cout );

  pCoord[0] = 3.25;  // location of the particle
  pSize [0] = 5;     // size of the particle
  pfield[0] = 20;    // value on the particle

  //
  // build the operators
  //
  typedef SpatialOps::Particle::CellToParticle<CellField> C2P;
  C2P c2p( xcoord[1]-xcoord[0], xcoord[cg.get_minus(0)] );
  c2p.set_coordinate_information( &pCoord, NULL, NULL, &pSize );

  typedef SpatialOps::Particle::ParticleToCell<CellField> P2C;
  P2C p2c( xcoord[1] - xcoord[0], xcoord[cg.get_minus(0)] );
  p2c.set_coordinate_information( &pCoord, NULL, NULL, &pSize );

  //
  // interpolate to particles
  //
  c2p.apply_to_field( f, ptmp );
  p2c.apply_to_field( pfield, ctmp );

//  std::cout << "  Interpolated gas value to particle field : " << ptmp[0] << std::endl;

//  CellField::const_iterator ix=xcoord.begin();
//  for( CellField::const_iterator i=ctmp.begin(); i!=ctmp.end(); ++i, ++ix )
//    std::cout << "  Interpolated particle value to gas field at x=" << *ix << " = "<< *i << std::endl;

//  print_field( f,      std::cout );
//  print_field( pfield, std::cout );
//  print_field( ptmp,   std::cout );
//  print_field( ctmp,   std::cout );

  TestHelper status(true);
  status( ptmp[0] == 32.5, "c2p" );

  const size_t ishift = cg.get_minus(0);
  status( ctmp[ishift+0] == 1, "p2c [0]" );
  status( ctmp[ishift+1] == 4, "p2c [1]" );
  status( ctmp[ishift+2] == 4, "p2c [2]" );
  status( ctmp[ishift+3] == 4, "p2c [3]" );
  status( ctmp[ishift+4] == 4, "p2c [4]" );
  status( ctmp[ishift+5] == 3, "p2c [5]" );
  status( ctmp[ishift+6] == 0, "p2c [6]" );
  status( ctmp[ishift+7] == 0, "p2c [7]" );
  status( ctmp[ishift+8] == 0, "p2c [8]" );
  status( ctmp[ishift+9] == 0, "p2c [9]" );

  if( status.ok() ) return 0;
  return -1;
}
