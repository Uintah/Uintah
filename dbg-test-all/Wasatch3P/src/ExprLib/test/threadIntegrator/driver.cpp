#include <expression/ExprLib.h>
#include <iostream>
#include "expressions.h"

#include <test/TestHelper.h>

//====================================================================

#include <spatialops/structured/FVStaggered.h>
typedef Expr::ExprPatch  PatchT;
typedef SpatialOps::SVolField  FieldT;

//====================================================================

bool run()
{
  using std::cout;
  using std::endl;
  using namespace SpatialOps;

  //           012345
  PatchT patch(100000); // solve 100K copies of the odes

  const double k = 1.0;

  // register expressions
  const Expr::Tag c1t("C1_RHS",Expr::STATE_N);
  const Expr::Tag c2t("C2_RHS",Expr::STATE_N);
  Expr::ExpressionFactory exprFactory;
  const Expr::ExpressionID rhsC1ID = exprFactory.register_expression( new C1<FieldT>::Builder(c1t,k) );
  const Expr::ExpressionID rhsC2ID = exprFactory.register_expression( new C2<FieldT>::Builder(c2t,k) );

  // build the integrator and attach equations.
  Expr::FieldManagerList& fml = patch.field_manager_list();
  Expr::TimeStepper integrator( exprFactory );
  integrator.add_equation<FieldT>( "C1", rhsC1ID );
  integrator.add_equation<FieldT>( "C2", rhsC2ID );
  integrator.finalize( fml, patch.operator_database(), patch.field_info() );
  {
    std::ofstream fout("tree.dot");
    integrator.get_tree()->write_tree(fout);
  }

# ifdef ENABLE_THREADS
  cout << "Executing with " << NTHREADS << " threads." << endl;
# else
  cout << "Executing without threads." << endl;
# endif

  // set initial conditions:
  FieldT& c1 = fml.field_manager<FieldT>().field_ref(Expr::Tag("C1",Expr::STATE_N));
  FieldT& c2 = fml.field_manager<FieldT>().field_ref(Expr::Tag("C2",Expr::STATE_N));
  c1 <<= 1.0;
  c2 <<= 0.0;

  // integrate in time:
  const double dt = 1.0e-3;
  const double endt = 1.0;
  const double tmon = endt/10.0;
  for( double t=0.0; t<=endt; t+=dt ){
    integrator.step(dt);
    if( std::fmod(t,tmon) < dt ) cout << "t=" << t << endl;
  }

# ifdef ENABLE_CUDA
  c1.add_device( CPU_INDEX );
  c2.add_device( CPU_INDEX );
# endif
  const FieldT& c3 = const_cast<FieldT&>(c1);
  const FieldT& c4 = const_cast<FieldT&>(c2);

  cout << "c1=" << *c3.begin()
       << ",  c2=" << *c4.begin()
       << endl;

  TestHelper status(false);

  double err = std::abs( *c3.begin() - 0.367695 );
  status( err<1e-5, "c3" );
  err = std::abs( *c4.begin() - 0.632305 );
  status( err<1e-5, "c4" );
  if( status.isfailed() )
    cout << endl << "FAIL" << endl
         << " Error: " << err << endl;
  else
    cout << endl << "PASS" << endl;

  return status.ok();
}

//====================================================================

int main()
{
  if( run() ) return 0;
  return -1;
}

//====================================================================
