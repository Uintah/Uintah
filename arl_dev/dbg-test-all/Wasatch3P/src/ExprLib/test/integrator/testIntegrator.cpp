#include <iostream>
#include <vector>

#include <expression/ExprLib.h>

#include <spatialops/structured/FVStaggeredFieldTypes.h>
#include <spatialops/Nebo.h>

#include <test/TestHelper.h>

using namespace SpatialOps;

struct Loc{
  typedef SpatialOps::NODIR   FaceDir;
  typedef IndexTriplet<0,0,0> Offset;
  typedef IndexTriplet<0,0,0> BCExtra;
};

typedef Expr::ExprPatch  PatchT;
typedef SpatialField<Loc>  FieldT;

//====================================================================

class ExpDecay : public Expr::Expression<FieldT>
{
public:
  void advertise_dependents( Expr::ExprDeps& exprDeps )
  {
    exprDeps.requires_expression( ctag_ );
  }

  void bind_fields( const Expr::FieldManagerList& fml )
  {
    c_ = &fml.field_manager<FieldT>().field_ref( ctag_ );
  }

  void evaluate()
  {
    FieldT& f = this->value();
    f <<= -k_ * *c_;
  }

  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Expr::ExpressionBase* build() const
    {
      return new ExpDecay( c_, k_ );
    }
    Builder( const Expr::Tag& exprValue,
             const Expr::Tag& c,
             const double k )
    : Expr::ExpressionBuilder(exprValue),
      c_(c), k_(k) {}
  private:
    const Expr::Tag c_;
    const double k_;
  };

private:

  ExpDecay( const Expr::Tag& c,
	    const double k )
    : Expr::Expression<FieldT>(),
      ctag_( c ),
      k_( k )
  {
    this->set_gpu_runnable( true );
  }

  const Expr::Tag ctag_;
  const double k_;
  const FieldT *c_;
};

//====================================================================

double test_integrator( const double dt,
			const double endTime,
			FieldT& c,
			const double k,
			Expr::TimeStepper& timeStepper )
{
  double time = 0;
  const double c0 = 1.0;
  c <<= c0;

  while( time<endTime ){
    timeStepper.step( dt );
    time += dt;
  }

  const double cexact = c0 * exp(-k*time);
  FieldT& tmp = c;
  tmp <<= cexact - tmp;
# ifdef ENABLE_CUDA
  tmp.add_device( CPU_INDEX );
# endif
  const double absErr = field_norm( tmp );

  return absErr;
}

//====================================================================

bool
setup_and_test_integrator( const Expr::TSMethod method )
{
  TestHelper status(true);
  using std::cout;
  using std::endl;

  PatchT patch(1);

  const double k = 1.0;

  Expr::ExpressionFactory exprFactory;
  const Expr::Tag ctag("c",Expr::STATE_N);
  const Expr::ExpressionID rhsID = exprFactory.register_expression( new ExpDecay::Builder(Expr::Tag("RHS",Expr::STATE_N),ctag,k),
								    true );
  Expr::FieldManagerList& fml = patch.field_manager_list();

  Expr::TimeStepper timeIntegrator( exprFactory, method, patch.id() );
  timeIntegrator.add_equation<FieldT>( "c", rhsID );
  timeIntegrator.finalize( fml, patch.operator_database(), patch.field_info() );

  FieldT& c = fml.field_ref<FieldT>(ctag);

  const size_t ndt = 8;
  double timesteps[ndt] = { 0.4, 0.2, 0.1, 0.05, 0.025, 0.01, 0.005, 0.0025 };
  std::vector<double> errs;
  for( size_t i=0; i<ndt; ++i ){
    errs.push_back( test_integrator( timesteps[i], 1.0, c, k, timeIntegrator ) );
  }

//   cout << "delta t    Error" << endl
//        << "----------------" << endl;
//   for( size_t i=0; i<ndt; ++i ){
//     cout << timesteps[i] << "  " << errs[i] << endl;
//   }

  switch (method) {
    case Expr::ForwardEuler:{
      double xerrs[ndt] = {0.0851942,0.0401994,0.0190605,0.00939352,0.004647,0.0018471,0.000921619,0.000460327};
      for( size_t i=0; i<ndt; ++i ){
        const double diff = std::abs( errs[i] - xerrs[i] );
        const double threshhold = 1.0e-5*xerrs[i];
        status(  diff < threshhold, boost::lexical_cast<char>(i)   );
      }
      break;
    }
    case Expr::SSPRK3:{
      double xerrs[ndt] = {0.00132812,0.000143957,1.65291e-05,1.99429e-06,2.44345e-07,1.54515e-08,1.92372e-09,2.39996e-10};
      std::cout << std::endl;
      for( size_t i=0; i<ndt; ++i ){
        status( std::abs( errs[i] - xerrs[i] ) < 1.0e-5*xerrs[i], boost::lexical_cast<char>(i) );
        std::cout << errs[i] << " ";
      }
      std::cout << std::endl;
      break;
    }
  } // switch

  return status.ok();
}

//====================================================================

int main()
{
  try{
    TestHelper status( true );
    status( setup_and_test_integrator( Expr::ForwardEuler ), "ForwardEuler" );
    status( setup_and_test_integrator( Expr::SSPRK3 ), "SSPRK3" );
    if( status.ok() ) return 0;
  }
  catch( std::exception& err ){
    std::cout << err.what() << std::endl;
  }
  return -1;
}
