#include <spatialops/structured/Grid.h>
#include <spatialops/structured/FVStaggeredFieldTypes.h>
#include <test/TestHelper.h>

#include <cmath>

using namespace SpatialOps;
using std::cout;
using std::endl;


template< typename FieldT >
bool test_field( const Grid& grid,
                 const bool bcx,
                 const bool bcy,
                 const bool bcz,
                 const double xlo,
                 const double ylo,
                 const double zlo,
                 const double xhi,
                 const double yhi,
                 const double zhi )
{
  const GhostData ghost(1);
  const BoundaryCellInfo bc = BoundaryCellInfo::build<FieldT>( bcx,bcy,bcz );
  const MemoryWindow window = get_window_with_ghost( grid.extent(), ghost, bc );
  FieldT f( window, bc, ghost, NULL );

  const MemoryWindow& interior = f.window_without_ghost();
  const int ilo = interior.flat_index( IntVec(0,0,0) );
  const int ihi = interior.flat_index( interior.extent()-IntVec(1,1,1) );

  TestHelper status(false);

  grid.set_coord<XDIR>( f );
  // cout << f[ilo] << "," << xlo << ", " << f[ilo]-xlo << endl
  //      << f[ihi] << "," << xhi << ", " << f[ihi]-xhi << endl;
  status( std::abs( f[ilo] - xlo ) < 1e-15, "xlo" );
  status( std::abs( f[ihi] - xhi ) < 1e-15, "xhi" );

  grid.set_coord<YDIR>( f );
  // cout << f[ilo] << "," << ylo << ", " << f[ilo]-ylo << endl
  //      << f[ihi] << "," << yhi << ", " << f[ihi]-yhi << endl;
  status( std::abs( f[ilo] - ylo ) < 1e-15, "ylo" );
  status( std::abs( f[ihi] - yhi ) < 1e-15, "yhi" );

  grid.set_coord<ZDIR>( f );
  // cout << f[ilo] << "," << zlo << ", " << f[ilo]-zlo << endl
  //      << f[ihi] << "," << zhi << ", " << f[ihi]-zhi << endl;
  status( std::abs( f[ilo] - zlo ) < 1e-15, "zlo" );
  status( std::abs( f[ihi] - zhi ) < 1e-15, "zhi" );

  return status.ok();
}


int main()
{
  const IntVec npts( 3, 3, 3 );
  const DoubleVec length(1.0,11.0,111.0);

  const Grid grid( npts, length );

  const DoubleVec spc = grid.spacing();

  cout << "Grid info: [nx,ny,nz] = " << npts << endl
       << "           [Lx,Ly,Lz] = " << "[" << length[0] << "," << length[1] << "," << length[2] << "]" << endl
       << "           [dx,dy,dz] = " << "[" << spc[0] << "," << spc[1] << "," << spc[2] << "]" << endl
       << endl;

  TestHelper status(true);

  status( spc[0] == length[0]/npts[0], "dx" );
  status( spc[1] == length[1]/npts[1], "dy" );
  status( spc[2] == length[2]/npts[2], "dz" );

  status( test_field<SVolField  >( grid, false, false, false, spc[0]/2.0, spc[1]/2.0, spc[2]/2.0, length[0]-spc[0]/2.0, length[1]-spc[1]/2.0, length[2]-spc[2]/2.0 ), "SVolField (n,n,n)" );
  status( test_field<SVolField  >( grid,  true,  true,  true, spc[0]/2.0, spc[1]/2.0, spc[2]/2.0, length[0]-spc[0]/2.0, length[1]-spc[1]/2.0, length[2]-spc[2]/2.0 ), "SVolField (n,n,n)" );
  status( test_field<SSurfXField>( grid, false, false, false, 0         , spc[1]/2.0, spc[2]/2.0, length[0]-spc[0]    , length[1]-spc[1]/2.0, length[2]-spc[2]/2.0 ), "SSurfXField (n,n,n)" );
  status( test_field<SSurfXField>( grid,  true,  true,  true, 0         , spc[1]/2.0, spc[2]/2.0, length[0]           , length[1]-spc[1]/2.0, length[2]-spc[2]/2.0 ), "SSurfXField (y,y,y)" );
  status( test_field<SSurfYField>( grid, false, false, false, spc[0]/2.0, 0         , spc[2]/2.0, length[0]-spc[0]/2.0, length[1]-spc[1]    , length[2]-spc[2]/2.0 ), "SSurfYField (n,n,n)" );
  status( test_field<SSurfYField>( grid,  true,  true,  true, spc[0]/2.0, 0         , spc[2]/2.0, length[0]-spc[0]/2.0, length[1]           , length[2]-spc[2]/2.0 ), "SSurfYField (y,y,y)" );
  status( test_field<SSurfZField>( grid, false, false, false, spc[0]/2.0, spc[1]/2.0, 0         , length[0]-spc[0]/2.0, length[1]-spc[1]/2.0, length[2]-spc[2]     ), "SSurfZField (n,n,n)" );
  status( test_field<SSurfZField>( grid,  true,  true,  true, spc[0]/2.0, spc[1]/2.0, 0         , length[0]-spc[0]/2.0, length[1]-spc[1]/2.0, length[2]            ), "SSurfZField (y,y,y)" );

  status( test_field<XVolField  >( grid, false, false, false, 0         , spc[1]/2.0, spc[2]/2.0, length[0]-spc[0]    , length[1]-spc[1]/2.0, length[2]-spc[2]/2.0 ), "XVolField (n,n,n)" );
  status( test_field<XVolField  >( grid,  true,  true,  true, 0         , spc[1]/2.0, spc[2]/2.0, length[0]           , length[1]-spc[1]/2.0, length[2]-spc[2]/2.0 ), "XVolField (n,n,n)" );
  status( test_field<XSurfXField>( grid, false, false, false, spc[0]/2.0, spc[1]/2.0, spc[2]/2.0, length[0]-spc[0]/2.0, length[1]-spc[1]/2.0, length[2]-spc[2]/2.0 ), "XSurfXField (n,n,n)" );
  status( test_field<XSurfXField>( grid,  true,  true,  true, spc[0]/2.0, spc[1]/2.0, spc[2]/2.0, length[0]-spc[0]/2.0, length[1]-spc[1]/2.0, length[2]-spc[2]/2.0 ), "XSurfXField (y,y,y)" );
  status( test_field<XSurfYField>( grid, false, false, false, 0         , 0         , spc[2]/2.0, length[0]-spc[0]    , length[1]-spc[1]    , length[2]-spc[2]/2.0 ), "XSurfYField (n,n,n)" );
  status( test_field<XSurfYField>( grid,  true,  true,  true, 0         , 0         , spc[2]/2.0, length[0]-spc[0]    , length[1]           , length[2]-spc[2]/2.0 ), "XSurfYField (y,y,y)" );
  status( test_field<XSurfZField>( grid, false, false, false, 0         , spc[1]/2.0, 0         , length[0]-spc[0]    , length[1]-spc[1]/2.0, length[2]-spc[2]     ), "XSurfZField (n,n,n)" );
  status( test_field<XSurfZField>( grid,  true,  true,  true, 0         , spc[1]/2.0, 0         , length[0]-spc[0]    , length[1]-spc[1]/2.0, length[2]            ), "XSurfZField (y,y,y)" );

  status( test_field<YVolField  >( grid, false, false, false, spc[0]/2.0, 0         , spc[2]/2.0, length[0]-spc[0]/2.0, length[1]-spc[1]    , length[2]-spc[2]/2.0 ), "YVolField (n,n,n)" );
  status( test_field<YVolField  >( grid,  true,  true,  true, spc[0]/2.0, 0         , spc[2]/2.0, length[0]-spc[0]/2.0, length[1]           , length[2]-spc[2]/2.0 ), "YVolField (n,n,n)" );
  status( test_field<YSurfXField>( grid, false, false, false, 0         , 0         , spc[2]/2.0, length[0]-spc[0]    , length[1]-spc[1]    , length[2]-spc[2]/2.0 ), "YSurfXField (n,n,n)" );
  status( test_field<YSurfXField>( grid,  true,  true,  true, 0         , 0         , spc[2]/2.0, length[0]           , length[1]-spc[1]    , length[2]-spc[2]/2.0 ), "YSurfXField (y,y,y)" );
  status( test_field<YSurfYField>( grid, false, false, false, spc[0]/2.0, spc[1]/2.0, spc[2]/2.0, length[0]-spc[0]/2.0, length[1]-spc[1]/2.0, length[2]-spc[2]/2.0 ), "YSurfYField (n,n,n)" );
  status( test_field<YSurfYField>( grid,  true,  true,  true, spc[0]/2.0, spc[1]/2.0, spc[2]/2.0, length[0]-spc[0]/2.0, length[1]-spc[1]/2.0, length[2]-spc[2]/2.0 ), "YSurfYField (y,y,y)" );
  status( test_field<YSurfZField>( grid, false, false, false, spc[0]/2.0, 0         , 0         , length[0]-spc[0]/2.0, length[1]-spc[1]    , length[2]-spc[2]     ), "YSurfZField (n,n,n)" );
  status( test_field<YSurfZField>( grid,  true,  true,  true, spc[0]/2.0, 0         , 0         , length[0]-spc[0]/2.0, length[1]-spc[1]    , length[2]            ), "YSurfZField (y,y,y)" );

  status( test_field<ZVolField  >( grid, false, false, false, spc[0]/2.0, spc[1]/2.0, 0         , length[0]-spc[0]/2.0, length[1]-spc[1]/2.0, length[2]-spc[2]     ), "ZVolField (n,n,n)" );
  status( test_field<ZVolField  >( grid,  true,  true,  true, spc[0]/2.0, spc[1]/2.0, 0         , length[0]-spc[0]/2.0, length[1]-spc[1]/2.0, length[2]            ), "ZVolField (n,n,n)" );
  status( test_field<ZSurfXField>( grid, false, false, false, 0         , spc[1]/2.0, 0         , length[0]-spc[0]    , length[1]-spc[1]/2.0, length[2]-spc[2]     ), "ZSurfXField (n,n,n)" );
  status( test_field<ZSurfXField>( grid,  true,  true,  true, 0         , spc[1]/2.0, 0         , length[0]           , length[1]-spc[1]/2.0, length[2]-spc[2]     ), "ZSurfXField (y,y,y)" );
  status( test_field<ZSurfYField>( grid, false, false, false, spc[0]/2.0, 0         , 0         , length[0]-spc[0]/2.0, length[1]-spc[1]    , length[2]-spc[2]     ), "ZSurfYField (n,n,n)" );
  status( test_field<ZSurfYField>( grid,  true,  true,  true, spc[0]/2.0, 0         , 0         , length[0]-spc[0]/2.0, length[1]           , length[2]-spc[2]     ), "ZSurfYField (y,y,y)" );
  status( test_field<ZSurfZField>( grid, false, false, false, spc[0]/2.0, spc[1]/2.0, spc[2]/2.0, length[0]-spc[0]/2.0, length[1]-spc[1]/2.0, length[2]-spc[2]/2.0 ), "ZSurfZField (n,n,n)" );
  status( test_field<ZSurfZField>( grid,  true,  true,  true, spc[0]/2.0, spc[1]/2.0, spc[2]/2.0, length[0]-spc[0]/2.0, length[1]-spc[1]/2.0, length[2]-spc[2]/2.0 ), "ZSurfZField (y,y,y)" );


  if( status.ok() ){
    cout << "Grid tests PASSED" << endl;
    return 0;
  }

  cout << "AT LEAST ONE GRID TEST FAILED!!!" << endl;
  return -1;
}
