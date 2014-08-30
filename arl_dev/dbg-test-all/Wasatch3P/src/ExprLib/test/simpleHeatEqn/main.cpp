#include <iostream>
using std::cout;
using std::endl;

#include <spatialops/structured/FVStaggered.h>
#include <spatialops/structured/Grid.h>

namespace so = SpatialOps;
typedef so::SVolField   CellField;
typedef so::SSurfXField XSideField;
typedef so::SSurfYField YSideField;
typedef so::SSurfZField ZSideField;

typedef so::BasicOpTypes<CellField>::GradX      GradX;
typedef so::BasicOpTypes<CellField>::InterpC2FX InterpX;
typedef so::BasicOpTypes<CellField>::DivX       DivX;

typedef so::BasicOpTypes<CellField>::GradY      GradY;
typedef so::BasicOpTypes<CellField>::InterpC2FY InterpY;
typedef so::BasicOpTypes<CellField>::DivY       DivY;

typedef so::BasicOpTypes<CellField>::GradZ      GradZ;
typedef so::BasicOpTypes<CellField>::InterpC2FZ InterpZ;
typedef so::BasicOpTypes<CellField>::DivZ       DivZ;


//-- boost includes ---//
#include <boost/program_options.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include <expression/ExprLib.h>

//--- local includes ---//
#include "RHSExpr.h"
#include "Flux.h"
#include "MonolithicRHS.h"

namespace po = boost::program_options;

void jcs_pause()
{
  std::cout << "Press <ENTER> to continue...";
  std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}

int main( int iarg, char* carg[] )
{
  size_t ntime;
  so::IntVec npts;
  std::vector<double> length(3,1.0);
  std::vector<double> spacing(3,1.0);
# ifdef FIELD_EXPRESSION_THREADS
  int soThreadCount, exprThreadCount;
# endif

  // parse the command line options input describing the problem
  {
    po::options_description desc("Supported Options");
    desc.add_options()
      ( "help", "print help message" )
      ( "ntime", po::value<size_t>(&ntime)->default_value(1000), "number of 'iterations'" )
#     ifdef FIELD_EXPRESSION_THREADS
      ( "tc", po::value<int>(&soThreadCount)->default_value(NTHREADS), "Number of threads for Nebo")
      ( "etc", po::value<int>(&exprThreadCount)->default_value(1), "Number of threads for ExprLib")
#     endif
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

# ifdef FIELD_EXPRESSION_THREADS
  cout << "FIELD_EXPRESSION_THREADS is ON" << endl;
  SpatialOps::set_hard_thread_count( NTHREADS );
  SpatialOps::set_soft_thread_count( soThreadCount );
  Expr::set_hard_thread_count( NTHREADS );
  Expr::set_soft_thread_count( exprThreadCount );
  sleep(1);
# endif

  cout << " [nx,ny,nz] = " << npts << endl
       << " ntime = " << ntime << endl
#     ifdef FIELD_EXPRESSION_THREADS
       << " SpatialOps NTHREADS (can set at runtime) = " << SpatialOps::get_soft_thread_count()
       << " out of " << SpatialOps::get_hard_thread_count() << endl
       << " ExprLib    NTHREADS (can set at runtime) = "
       << SpatialOps::ThreadPool::get_pool_size() << " out of "
       << SpatialOps::ThreadPool::get_pool_capacity() << endl
#     endif
       << endl;

  // set mesh spacing (uniform, structured mesh)
  for( size_t i=0; i<3; ++i )
    spacing[i] = length[i]/double(npts[i]);

  // set face areas
  std::vector<double> area(3,1.0);
  area[0] = spacing[1]*spacing[2];
  area[1] = spacing[0]*spacing[2];
  area[2] = spacing[0]*spacing[1];

  // build the spatial operators
  SpatialOps::OperatorDatabase sodb;
  so::build_stencils( npts[0],   npts[1],   npts[2],
                      length[0], length[1], length[2],
                      sodb );

  Expr::ExpressionFactory segFactory, monoFactory;

  const Expr::Tag tempt( "Temperature", Expr::STATE_NONE );
  const Expr::Tag dct( "ThermalCond", Expr::STATE_NONE );

  Expr::ExpressionID segRHSID, monoRHSID;

  { //-- segregated approach:
    const Expr::Tag
    xfluxt( "HeatFluxX", Expr::STATE_NONE ),
    yfluxt( "HeatFluxY", Expr::STATE_NONE ),
    zfluxt( "HeatFluxZ", Expr::STATE_NONE );

    segFactory.register_expression( new Flux<GradX,InterpX>::Builder( xfluxt, tempt, dct ) );
    segFactory.register_expression( new Flux<GradY,InterpY>::Builder( yfluxt, tempt, dct ) );
    segFactory.register_expression( new Flux<GradZ,InterpZ>::Builder( zfluxt, tempt, dct ) );

    segRHSID = segFactory.register_expression(
        new RHSExpr<CellField>::Builder( Expr::Tag("segrhs",Expr::STATE_NONE),
                                         npts[0]>1 ? xfluxt : Expr::Tag(),
                                         npts[1]>1 ? yfluxt : Expr::Tag(),
                                         npts[2]>1 ? zfluxt : Expr::Tag() ) );
  }
  { //-- monolithic approach
    monoRHSID = monoFactory.register_expression(
        new MonolithicRHS<CellField>::Builder( Expr::Tag("monorhs",Expr::STATE_NONE),
                                               dct, tempt ) );
  }

  //-- build a few coordinate fields
  const so::BoundaryCellInfo cellBCInfo = so::BoundaryCellInfo::build<CellField>(true,true,true);
  const so::GhostData cellGhosts(1);
  const so::MemoryWindow vwindow( so::get_window_with_ghost(npts,cellGhosts,cellBCInfo) );
  CellField xcoord( vwindow, cellBCInfo, cellGhosts, NULL );
  CellField ycoord( vwindow, cellBCInfo, cellGhosts, NULL );
  CellField zcoord( vwindow, cellBCInfo, cellGhosts, NULL );
  so::Grid grid( npts, length );
  grid.set_coord<SpatialOps::XDIR>( xcoord );
  grid.set_coord<SpatialOps::YDIR>( ycoord );
  grid.set_coord<SpatialOps::ZDIR>( zcoord );

# ifdef ENABLE_CUDA
  xcoord.add_device( GPU_INDEX );
  ycoord.add_device( GPU_INDEX );
  zcoord.add_device( GPU_INDEX );
# endif
  CellField diffCoef( vwindow, cellBCInfo, cellGhosts, NULL );
  segFactory .register_expression( new Expr::PlaceHolder<CellField>::Builder(dct  ) );
  segFactory .register_expression( new Expr::PlaceHolder<CellField>::Builder(tempt) );
  monoFactory.register_expression( new Expr::PlaceHolder<CellField>::Builder(dct  ) );
  monoFactory.register_expression( new Expr::PlaceHolder<CellField>::Builder(tempt) );

  //-- create the expression tree
  Expr::ExpressionTree  segTree(  segRHSID,  segFactory, 0 );
  Expr::ExpressionTree monoTree( monoRHSID, monoFactory, 0 );

  Expr::FieldManagerList fml;
  segTree .register_fields( fml );
  monoTree.register_fields( fml );
  {
    std::ofstream  seg("seggraph.dot" );   segTree.write_tree(seg );
    std::ofstream mono("monograph.dot");  monoTree.write_tree(mono);
  }

  try{
    //-- allocate all fields on the patch for this segTree
    fml.allocate_fields( Expr::FieldAllocInfo( npts, 0, 0, false, false, false ) );

    //-- bind fields & operators
    segTree .bind_fields   ( fml  );
    segTree .bind_operators( sodb );
    monoTree.bind_fields   ( fml  );
    monoTree.bind_operators( sodb );
  }
  catch( std::exception& err ){
   std::cout << "ERROR allocating/binding fields" << std::endl
       << err.what() << std::endl;
   return -1;
  }
  //-- initialize the fields - temperature and thermal cond.
  {
    using namespace SpatialOps;
    CellField& temp = fml.field_manager<CellField>().field_ref(tempt);
    CellField& dc   = fml.field_manager<CellField>().field_ref(dct);
    temp <<= sin( xcoord ) + cos( ycoord ) + sin( zcoord );
    dc <<= xcoord + ycoord + zcoord;
  }

  try{
    boost::posix_time::ptime start, stop;
    boost::posix_time::time_duration elapsed;

    start = boost::posix_time::microsec_clock::universal_time();
    for( int iter=0; iter<ntime; ++iter ){
      segTree.execute_tree();
    }
    stop = boost::posix_time::microsec_clock::universal_time();
    elapsed = stop-start;
    cout << "sequential - time taken: "
         << elapsed.total_microseconds()*1e-6
         << endl;

    start = boost::posix_time::microsec_clock::universal_time();
    for( int iter=0; iter<ntime; ++iter ){
      monoTree.execute_tree();
    }
    stop = boost::posix_time::microsec_clock::universal_time();
    elapsed = stop-start;
    cout << "monolithic - time taken: "
         << elapsed.total_microseconds()*1e-6
         << endl << endl;

    return 0;
  }
  catch( std::exception& err ){
    cout << "ERROR!" << endl
         << err.what() << endl;
  }
  return -1;

}
