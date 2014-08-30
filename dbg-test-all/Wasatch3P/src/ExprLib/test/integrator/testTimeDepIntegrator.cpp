#include <iostream>
#include <cmath>

#include <expression/ExprLib.h>

using namespace SpatialOps;

typedef Expr::ExprPatch  PatchT;

/*
 * RHS for the ode:  \f$ \frac{ d\phi}{ dt} = \exp(-t) \f$
 * The analytic solution is
 *   \f$ \phi = \phi_0 + \exp(-t) - 1 \f$
 */
class RHS : public Expr::Expression<SingleValueField>
{
public:
  void advertise_dependents( Expr::ExprDeps& exprDeps )
  {
    exprDeps.requires_expression( Expr::Tag("time", Expr::STATE_NONE) );
  }

  void bind_fields( const Expr::FieldManagerList& fml )
  {
    t_ = &fml.field_manager<SingleValueField>().field_ref( Expr::Tag("time", Expr::STATE_NONE) );
  }

  void evaluate()
  {
    using namespace SpatialOps;
    SingleValueField& rhs = this->value();
    rhs <<= exp( -*t_ );
  }

  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Expr::ExpressionBase* build() const{ return new RHS(); }
    Builder(const Expr::Tag& tag) : ExpressionBuilder(tag) {}
    ~Builder(){}
  private:
  };

private:

  RHS() : Expr::Expression<SingleValueField>()
  {
#   ifdef ENABLE_CUDA
    this->set_gpu_runnable( true );
#   endif
  }
  const SingleValueField* t_;
};

double test_integrator( double dt,
                        const double endTime )
{
  const Expr::Tag rhsTag("RHS",Expr::STATE_NONE), phiTag("phi",Expr::STATE_N);
  PatchT patch(1);
  Expr::ExpressionFactory exprFactory;
  Expr::FieldManagerList& fml = patch.field_manager_list();
  const Expr::ExpressionID rhsID =
    exprFactory.register_expression( new RHS::Builder(rhsTag) );
  const Expr::Tag timeTag("time",Expr::STATE_NONE);
  exprFactory.register_expression( new Expr::PlaceHolder<SingleValueField>::Builder(timeTag) );
  Expr::TimeStepper ts( exprFactory, Expr::SSPRK3, patch.id(), timeTag );
  ts.add_equation<SingleValueField>( "phi", rhsID );
//  ts.get_tree()->lock_fields(fml);
  ts.finalize( fml, patch.operator_database(), patch.field_info() );

  // be sure that some access methods on the integrator work as expected
  {
    std::vector<std::string> solnVars;
    ts.get_variables( solnVars );
    if( solnVars[0] != "phi" ){
      std::cout << "solution variable name on time integrator is inconsistent" << std::endl;
      return false;
    }

    std::vector<Expr::Tag> rhsTags;
    ts.get_rhs_tags( rhsTags );
    if( rhsTags[0] != rhsTag ){
      std::cout << "RHS tag on time integrator is inconsistent" << std::endl;
      return false;
    }
  }

  // set the initial condition
  Expr::FieldMgrSelector<SingleValueField>::type& doubleFM = fml.field_manager<SingleValueField>();
  const double phi0 = 1.0;
  {
    SingleValueField& phi = doubleFM.field_ref(phiTag);
    phi <<= phi0; // initial condition.
  }

  double time=0.0;
  ts.set_time(time);
  while( time<endTime ){
    if( endTime-time < dt ) dt=endTime-time;
    ts.step( dt );
    time += dt;
  }
  assert(time==ts.get_time());
# ifndef ENABLE_CUDA
  std::cout << time << " : " << doubleFM.field_ref(timeTag)[0] << std::endl;
# endif

  SingleValueField& phi = doubleFM.field_ref(phiTag);
  const double phiexact = phi0 + 1.0 - std::exp( -endTime );
# ifdef ENABLE_CUDA
  phi.add_device( CPU_INDEX );
# endif

  const SingleValueField& phi1 = const_cast<SingleValueField&>(phi);
  const double err = std::abs( phi1[0] - phiexact );
  std::cout << "found   : " << phi1[0] << std::endl
            << "expected: " << phiexact << std::endl
            << "error   : " << err << std::endl
            << std::endl;

  return err;
}


int main()
{
  double err[3];
  double dt[] = { 2.0e-1, 1.0e-1, 5.0e-2 };
  try{
    for( int i=0; i<3; ++i ){
      err[i] = test_integrator( dt[i], 1.1 );
    }
  }
  catch( std::exception& err ){
    std::cout << err.what() << std::endl;
    return -1;
  }

  const double rat1 = err[0]/err[1];
  const double rat2 = err[1]/err[2];

  std::cout << err[0] << ", " << err[1] << ", " << err[2] << std::endl
            << rat1 << std::endl << rat2 << std::endl;

  const bool isOkay = ( rat1>15. && rat2>15. );

  if( isOkay ){
    std::cout << "PASS" << std::endl;
    return 0;
  }

  std::cout << "FAIL" << std::endl;
  return -1;
}
