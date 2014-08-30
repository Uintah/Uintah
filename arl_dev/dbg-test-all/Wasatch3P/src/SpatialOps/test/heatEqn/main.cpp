#include <iostream>
using std::cout;
using std::endl;

//--- SpatialOps includes ---//
#include <spatialops/SpatialOpsConfigure.h>
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/FVStaggeredFieldTypes.h>
#include <spatialops/Nebo.h>
#include <spatialops/structured/Grid.h>

#include <spatialops/structured/stencil/FVStaggeredOperatorTypes.h>
#include <spatialops/structured/stencil/StencilBuilder.h>

typedef SpatialOps::SVolField   CellField;
typedef SpatialOps::SSurfXField XSideField;
typedef SpatialOps::SSurfYField YSideField;
typedef SpatialOps::SSurfZField ZSideField;

typedef SpatialOps::BasicOpTypes<CellField>::GradX      GradX;
typedef SpatialOps::BasicOpTypes<CellField>::InterpC2FX InterpX;
typedef SpatialOps::BasicOpTypes<CellField>::DivX       DivX;

typedef SpatialOps::BasicOpTypes<CellField>::GradY      GradY;
typedef SpatialOps::BasicOpTypes<CellField>::InterpC2FY InterpY;
typedef SpatialOps::BasicOpTypes<CellField>::DivY       DivY;

typedef SpatialOps::BasicOpTypes<CellField>::GradZ      GradZ;
typedef SpatialOps::BasicOpTypes<CellField>::InterpC2FZ InterpZ;
typedef SpatialOps::BasicOpTypes<CellField>::DivZ       DivZ;


//--- local includes ---//
#include "tools.h"

//-- boost includes ---//
#include <boost/program_options.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

namespace po = boost::program_options;
using namespace SpatialOps;

int main( int iarg, char* carg[] )
{
  size_t ntime;
  IntVec npts;
  DoubleVec length;

  // parse the command line options input describing the problem
  {
    po::options_description desc("Supported Options");
    desc.add_options()
      ( "help", "print help message" )
      ( "ntime", po::value<size_t>(&ntime)->default_value(1000), "number of 'iterations'" )
      //( "ntime", po::value<size_t>(&ntime), "number of 'iterations'" )
      ( "nx", po::value<int>(&npts[0])->default_value(10), "Grid in x" )
      ( "ny", po::value<int>(&npts[1])->default_value(10), "Grid in y" )
      ( "nz", po::value<int>(&npts[2])->default_value(10), "Grid in z" )
      ( "Lx", po::value<double>(&length[0])->default_value(1.0),"Length in x")
      ( "Ly", po::value<double>(&length[1])->default_value(1.0),"Length in y")
      ( "Lz", po::value<double>(&length[2])->default_value(1.0),"Length in z");

    po::variables_map args;
    po::store( po::parse_command_line(iarg,carg,desc), args );
    po::notify(args);

    if (args.count("help")) {
      cout << desc << "\n";
      return 1;
    }
  }

  cout << " [nx,ny,nz] = [" << npts[0] << "," << npts[1] << "," << npts[2] << "]" << endl
       << " ntime = " << ntime << endl
#      ifdef ENABLE_THREADS
       << " NTHREADS = " << NTHREADS << endl
#      endif
       << endl;

  // set mesh spacing (uniform, structured mesh)
  const DoubleVec spacing = length/npts;

  const Grid grid( npts, length );

  // build the spatial operators
  SpatialOps::OperatorDatabase sodb;
  SpatialOps::build_stencils( grid, sodb );

  // grab pointers to the operators
  const GradX* const gradx = sodb.retrieve_operator<GradX>();
  const GradY* const grady = sodb.retrieve_operator<GradY>();
  const GradZ* const gradz = sodb.retrieve_operator<GradZ>();

  const DivX* const divx = sodb.retrieve_operator<DivX>();
  const DivY* const divy = sodb.retrieve_operator<DivY>();
  const DivZ* const divz = sodb.retrieve_operator<DivZ>();

  const InterpX* const interpx = sodb.retrieve_operator<InterpX>();
  const InterpY* const interpy = sodb.retrieve_operator<InterpY>();
  const InterpZ* const interpz = sodb.retrieve_operator<InterpZ>();

  // build fields
  const GhostData ghost(1);

  const BoundaryCellInfo cellBC = BoundaryCellInfo::build< CellField>(true,true,true);

  const MemoryWindow vwindow( get_window_with_ghost(npts,ghost,cellBC) );

  CellField temperature( vwindow, cellBC, ghost, NULL );
  CellField thermCond  ( vwindow, cellBC, ghost, NULL );
  CellField rhoCp      ( vwindow, cellBC, ghost, NULL );
  CellField xcoord     ( vwindow, cellBC, ghost, NULL );
  CellField ycoord     ( vwindow, cellBC, ghost, NULL );
  CellField zcoord     ( vwindow, cellBC, ghost, NULL );
  CellField rhs        ( vwindow, cellBC, ghost, NULL );

  grid.set_coord<SpatialOps::XDIR>( xcoord );
  grid.set_coord<SpatialOps::YDIR>( ycoord );
  grid.set_coord<SpatialOps::ZDIR>( zcoord );

  const BoundaryCellInfo xBC = BoundaryCellInfo::build<XSideField>(true,true,true);
  const BoundaryCellInfo yBC = BoundaryCellInfo::build<YSideField>(true,true,true);
  const BoundaryCellInfo zBC = BoundaryCellInfo::build<ZSideField>(true,true,true);
  const MemoryWindow xwindow( get_window_with_ghost(npts,ghost,xBC) );
  const MemoryWindow ywindow( get_window_with_ghost(npts,ghost,yBC) );
  const MemoryWindow zwindow( get_window_with_ghost(npts,ghost,zBC) );

  XSideField xflux( xwindow, xBC, ghost, NULL );
  YSideField yflux( ywindow, yBC, ghost, NULL );
  ZSideField zflux( zwindow, zBC, ghost, NULL );

  {
    using namespace SpatialOps;
    temperature <<= sin( xcoord ) + cos( ycoord ) + sin( zcoord );
    thermCond <<= xcoord + ycoord + zcoord;
    rhoCp <<= 1.0;
  }

  try{
    cout << "beginning 'timestepping'" << endl;

    const boost::posix_time::ptime time_start( boost::posix_time::microsec_clock::universal_time() );

    // mimic the effects of solving this PDE in time.
    for( size_t itime=0; itime<ntime; ++itime ){

      using namespace SpatialOps;

      if( npts[0]>1 ) calculate_flux( *gradx, *interpx, temperature, thermCond, xflux );
      if( npts[1]>1 ) calculate_flux( *grady, *interpy, temperature, thermCond, yflux );
      if( npts[2]>1 ) calculate_flux( *gradz, *interpz, temperature, thermCond, zflux );

      rhs <<= 0.0;
      if( npts[0]>1 ) calculate_rhs( *divx, xflux, rhoCp, rhs );
      if( npts[1]>1 ) calculate_rhs( *divy, yflux, rhoCp, rhs );
      if( npts[2]>1 ) calculate_rhs( *divz, zflux, rhoCp, rhs );

      // ordinarily at this point we would use rhs to update
      // temperature.  however, there are some other complicating
      // factors like setting boundary conditions that we have neglected
      // here to simplify things.

      //    cout << itime+1 << " of " << ntime << endl;
    }

    const boost::posix_time::ptime time_end( boost::posix_time::microsec_clock::universal_time() );
    const boost::posix_time::time_duration time_dur = time_end - time_start;

    cout << "done" << endl << endl
         << "time taken: "
         << time_dur.total_microseconds()*1e-6
         << endl << endl;


    return 0;
  }
  catch( std::exception& err ){
    cout << "ERROR!" << endl;
  }
  return -1;
}
