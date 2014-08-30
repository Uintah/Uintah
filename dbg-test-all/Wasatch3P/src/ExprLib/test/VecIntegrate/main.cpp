#include <sstream>
#include "RHS.h"

#include <expression/ExprLib.h>
#include <expression/TimeStepper.h>
#include <spatialops/Nebo.h>

using namespace SpatialOps;

/*
  This test solves the ODE:

     d yi
     -----  = sin( fi * t )
      d t

   Assuming yi(0)=0, the analytic solution is

     yi(t) = ( 1-cos( fi * t ) ) / fi

 */

//--------------------------------------------------------------------

double check_results( const std::vector<SingleValueField*>& phi,
                      const std::vector<double>& freq,
                      SingleValueField& t )
{
# ifdef ENABLE_CUDA
  t.set_device_as_active( CPU_INDEX );
# endif
  assert( phi.size() == freq.size() );
  std::vector<SingleValueField*>::const_iterator iphi  = phi.begin();
  std::vector<double           >::const_iterator ifreq = freq.begin();
  double l2norm=0.0;
  for( ; iphi!=phi.end(); ++iphi, ++ifreq ){
    const double exact =  (1.0-std::cos( *ifreq * t[0] )) / *ifreq;
#   ifdef ENABLE_CUDA
    (*iphi)->set_device_as_active(CPU_INDEX);
#   endif

    const double absErr = std::abs( (**iphi)[0]-exact );
    l2norm += absErr*absErr;
    std::cout << (**iphi)[0] << ", " << absErr << ", " << l2norm << ", " << t[0] << std::endl;
  }

  return sqrt(l2norm);
}

//--------------------------------------------------------------------

int main()
{
  const unsigned int nvar = 20;

  Expr::ExprPatch patch(1);
  Expr::ExpressionFactory exprFactory;
  Expr::FieldManagerList& fml = patch.field_manager_list();
  Expr::TimeStepper ts( exprFactory, Expr::SSPRK3 );
  Expr::FieldMgrSelector<SingleValueField>::type& fmdouble = fml.field_manager<SingleValueField>();

  const Expr::Tag timeTag( "time", Expr::STATE_NONE );
  exprFactory.register_expression( new Expr::PlaceHolder<SingleValueField>::Builder(timeTag) );

  // register the RHS expressions
  Expr::TagList rhsTags;
  std::vector<double> freqs;
  for( unsigned int i=0; i<nvar; ++i ){
    std::ostringstream nam;
    nam << "rhs_" << i;
    rhsTags.push_back( Expr::Tag( nam.str(), Expr::STATE_NONE ) );
    freqs.push_back(i+1);
  }
  const Expr::ExpressionID rhsID = exprFactory.register_expression( new RHS::Builder(rhsTags,freqs,timeTag) );

  // add equations to the time integrator
  std::vector<std::string> varNames;
  for( unsigned int i=0; i<nvar; ++i ){
    std::ostringstream nam;
    nam << "var_" << i;
    varNames.push_back( nam.str() );
  }
  ts.add_equations<SingleValueField>( varNames, rhsID );

  // finalize the integrator - registers variables, allocates memory, etc.
  ts.finalize( fml, patch.operator_database(), patch.field_info() );

  {
    std::ofstream fout( "rhs.dot" );
    ts.get_tree()->write_tree( fout );
    fout.close();
  }

  std::vector<double> l2err, dtlist;
  double dt = 1;
  const size_t nt = 8;
  const double tend = 5.0;
  SingleValueField& t = fmdouble.field_ref( timeTag );

# ifdef ENABLE_CUDA
  t.add_device(CPU_INDEX);
# endif

  for( size_t idt=0; idt<nt; ++idt ){

    t <<= 0;

    std::cout << "Integration loop " << idt+1 << " of " << nt << std::endl;

    // set initial conditions
    std::vector<SingleValueField*> vars;
    for( unsigned int i=0; i<nvar; ++i ){
      vars.push_back( &fmdouble.field_ref( Expr::Tag(varNames[i], Expr::STATE_N) ) );
      *vars[i] <<= 0.0;
    }

#   ifdef ENABLE_CUDA
    t.validate_device(CPU_INDEX);
    t.set_device_as_active(CPU_INDEX);
    ts.set_time( t[0] );
    t.set_device_as_active(GPU_INDEX);
#   else
    ts.set_time( t[0] );
#   endif

    // integrate in time

    while( ts.get_time() < tend ) ts.step(dt);

    l2err.push_back( check_results( vars, freqs, t ) );

    dtlist.push_back( dt );

    dt *= 0.5;
  }


  std::ofstream fout( "conv.csv" );
  fout << "dt, L2 error, first, second, third, fourth" << std::endl;
  for( size_t i=0; i<nt; ++i ){
    fout << dtlist[i] << ", " << l2err[i];
    for( size_t iord=1; iord<5; ++iord ){
      const double err = l2err[0] * std::pow( dtlist[i]/dtlist[0], double(iord) );
      fout << ", " << err;
    }
    fout << std::endl;
  }


  // Verify that the solution is at least fourth order, which it should
  // be for this situation where the RHS is a function of time only
  const double err4ord = l2err[0] * std::pow( dtlist[nt-1]/dtlist[0], 4.0 );
  if( l2err[nt-1] < 2*err4ord ){
    std::cout << "PASS" << std::endl;
    return 0;
  }

  std::cout << "FAIL" << std::endl;
  return -1;

}
