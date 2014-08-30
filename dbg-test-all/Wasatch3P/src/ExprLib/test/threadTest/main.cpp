#include <sstream>
#include <fstream>
#include <stdexcept>
#include <iostream>
using std::cout;
using std::endl;

//-- boost includes ---//
#include <boost/program_options.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
namespace po = boost::program_options;

#include <expression/ExprLib.h>

#include "defs.h"
#include "test.h"

typedef Expr::ConstantExpr<VolT>  ConstExpr;

typedef std::vector< Expr::ExpressionID > IdInfo;

IdInfo exprs;

void
register_expressions( Expr::ExpressionFactory& exprFactory,
                      const int nvar,
                      const bool doDiffusion,
                      const bool coupledSrc,
                      const bool indepSrc )
{
  using Expr::Tag;
  using Expr::STATE_NONE;

  for( int i=0; i<nvar; ++i ){

    std::ostringstream nam;
    nam << "var_" << i;

    const std::string varname = nam.str();

    const Tag varTag( varname, STATE_NONE );
    const Tag rhsTag( varname+"_rhs", STATE_NONE );

    Tag fluxTag = Tag();
    if( doDiffusion ){
      fluxTag = Tag( varname+"_flux", STATE_NONE );
      exprFactory.register_expression( new FluxExpr::Builder( fluxTag, varTag, 1.0 ) );
    }
    else{
    }
    Tag busyTag;
    if( coupledSrc ){
      busyTag = Tag( varname+"_src", STATE_NONE );
      exprFactory.register_expression( new CoupledBusyWork::Builder(busyTag,varTag,nvar) );
    }
    else if( indepSrc ){
      busyTag = Tag( varname+"_src", STATE_NONE );
      exprFactory.register_expression( new BusyWork::Builder(busyTag,varTag,nvar) );
    }

    exprFactory.register_expression( new ConstExpr::Builder(varTag,i) );
    exprs.push_back( exprFactory.register_expression( new RHSExpr::Builder( rhsTag, fluxTag, busyTag ) ) );
  }

}

//--------------------------------------------------------------------

void build_operators( PatchT& patch,
                      const std::vector<double>& spacing )
{
  const bool xbcPlus = patch.has_physical_bc_xplus();
  const bool ybcPlus = patch.has_physical_bc_yplus();
  const bool zbcPlus = patch.has_physical_bc_zplus();

  const std::vector<int>& dim = patch.dim();

  SpatialOps::OperatorDatabase& opDB = patch.operator_database();
  opDB.register_new_operator( new XDivT ( SpatialOps::build_two_point_coef_collection( -1.0/spacing[0], 1.0/spacing[1] ) ) );
  opDB.register_new_operator( new XGradT( SpatialOps::build_two_point_coef_collection( -1.0/spacing[0], 1.0/spacing[1] ) ) );
}

//--------------------------------------------------------------------

int main( int argc, char* argv[] )
{
  int nvar = 10;
  int nstep = 200000/nvar;
  std::vector<int> dim(3,1);
  dim[0] = 25;
  dim[1] = 10;
  dim[2] = 10;
  bool doDiffusion, coupledSrc, indepSrc;

#ifdef FIELD_EXPRESSION_THREADS
  int soThreadCount, exprThreadCount;
#endif

  // parse the command line options input describing the problem
  {
    po::options_description desc("Supported Options");
    desc.add_options()
      ( "help", "print help message" )
      ( "nvar", po::value<int>(&nvar)->default_value(10), "number of variables" )
      ( "nstep", po::value<int>(&nstep)->default_value(1000), "number of 'iterations'" )
      ( "diffusion",      po::bool_switch(&doDiffusion)->default_value(false), "activates diffusive flux terms" )
      ( "coupled-source", po::bool_switch(&coupledSrc )->default_value(false), "activates coupled source term" )
      ( "indep-source",   po::bool_switch(&indepSrc   )->default_value(false), "activates independent source term" )
#ifdef FIELD_EXPRESSION_THREADS
      ( "tc", po::value<int>(&soThreadCount)->default_value(1), "Number of threads for Nebo")
      ( "etc", po::value<int>(&exprThreadCount)->default_value(NTHREADS), "Number of threads for ExprLib")
#endif
      ( "nx", po::value<int>(&dim[0])->default_value(25), "Grid in x" )
      ( "ny", po::value<int>(&dim[1])->default_value(10), "Grid in y" )
      ( "nz", po::value<int>(&dim[2])->default_value(10), "Grid in z" );

    po::variables_map args;
    po::store( po::parse_command_line(argc,argv,desc), args );
    po::notify(args);

    if (args.count("help")) {
      cout << desc << endl;
      return 1;
    }
  }

#ifdef FIELD_EXPRESSION_THREADS
  cout << "FIELD_EXPRESSION_THREADS is ON" << endl;
  SpatialOps::set_hard_thread_count( NTHREADS );
  SpatialOps::set_soft_thread_count( soThreadCount );
  Expr::set_hard_thread_count( NTHREADS );
  Expr::set_soft_thread_count( exprThreadCount );
  sleep(1);
#endif

  if( coupledSrc && indepSrc ){
    cout << "cannot use coupled & independent source together" << endl;
    return -1;
  }
  cout << endl;
  if( coupledSrc  ) cout << " Using coupled source terms" << endl;
  if( indepSrc    ) cout << " Using independent source terms" << endl;
  if( doDiffusion ) cout << " Using diffusion" << endl;

  cout << " [nx,ny,nz] = [" << dim[0] << "," << dim[1] << "," << dim[2] << "]" << endl
       << " nsteps = " << nstep << endl
       << " nvar   = " << nvar << endl
#     ifdef FIELD_EXPRESSION_THREADS
       << " SpatialOps NTHREADS (can set at runtime) = " << SpatialOps::get_soft_thread_count()
       << " out of " << SpatialOps::get_hard_thread_count() << endl
       << " ExprLib    NTHREADS (can set at runtime) = "
       << SpatialOps::ThreadPool::get_pool_size() << " out of "
       << SpatialOps::ThreadPool::get_pool_capacity() << endl
#     endif
       << endl;

  std::vector<double> length(3,1);
  std::vector<double> spacing(3,1.0);
  for( int i=0; i<3; ++i ){
    if( dim[i]>1 ) spacing[i] = length[i]/dim[i];
  }

  PatchT patch( dim[0], dim[1], dim[2] );

  if( doDiffusion ) build_operators( patch, spacing );

  Expr::ExpressionFactory exprFactory;
  register_expressions( exprFactory, nvar, doDiffusion, coupledSrc, indepSrc );

  IdInfo::const_iterator irhs=exprs.begin();
  Expr::ExpressionTree tree( *irhs, exprFactory, patch.id() );
  ++irhs;
  for( ; irhs!=exprs.end(); ++irhs ){
    tree.insert_tree( *irhs );
  }

   {
     std::ofstream fout("tree.dot");
     tree.write_tree(fout);
     fout.close();
   }

  try{
    Expr::FieldManagerList& fml = patch.field_manager_list();
    tree.register_fields( fml );
    fml.allocate_fields( patch.field_info() );
    tree.bind_fields( fml );
    tree.bind_operators( patch.operator_database() );

    const boost::posix_time::ptime start = boost::posix_time::microsec_clock::universal_time();

    for( int i=0; i<nstep; ++i ){
      tree.execute_tree();
      cout << i+1 << " of " << nstep << endl;
    }

    const boost::posix_time::time_duration elapsed = boost::posix_time::microsec_clock::universal_time() - start;

    cout << endl << "thread "<<NTHREADS<< " time_per_step_per_var "<< elapsed.total_microseconds()*1e-6 <<endl;
  }
  catch( std::runtime_error& err ){
    cout << err.what() << endl;
    return -1;
  }

  return 0;
}
