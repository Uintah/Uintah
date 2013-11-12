#include "DensityCalculator.h"

#include <boost/foreach.hpp>


extern "C"{
  void dgesv_( const int* n,    // # of equations
               const int* nrhs, // # of rhs to solve
               double A[],      // lhs matrix
               const int* lda,  // leading dim of A (#eqn for us)
               int* ipiv,       // integer work array
               double rhs[],    // rhs array
               const int* ldb,  // dimension of rhs (#eqn for us)
               int* info );
}

//-------------------------------------------------------------------

DensityCalculatorBase::
DensityCalculatorBase( const int neq,
                       const double rtol,
                       const size_t maxIter )
: neq_    ( neq     ),
  rtol_   ( rtol    ),
  maxIter_( maxIter )
{
  jac_ .resize( neq_*neq_ );
  res_ .resize( neq_      );
  ipiv_.resize( neq_      );
}

DensityCalculatorBase::~DensityCalculatorBase(){}

bool DensityCalculatorBase::solve( const DoubleVec& passThrough,
                                   DoubleVec& soln )
{
  int niter = 0;
  double relErr = 0.0;

  do{
    calc_jacobian_and_res( passThrough, soln, jac_, res_ );
    switch( neq_ ){
      case 1: // avoid the overhead of the general equation solver for this trivial case.
        res_[0] /= jac_[0];  // put the displacement from Newton's method in the residual.
        break;
      default:
        // Solving J * delta = rhs  (note the missing minus sign, which is handled in the update below)
        // note that on entry, res_ is the rhs and on exit, it is the solution (delta).
        const int one=1; int info;
        dgesv_( &neq_, &one, &jac_[0], &neq_, &ipiv_[0], &res_[0], &neq_, &info );
        assert( info==0 );
        break;
    } // switch
    relErr = 0.0;
    for( size_t i=0; i<neq_; ++i ){
      soln[i] -= res_[i];
      relErr += std::abs( res_[i]/get_normalization_factor(i) );
      // clip the solution to the valid range
      const std::pair<double,double> bounds = get_bounds(i);
      soln[i] = std::max( std::min( bounds.second, soln[i] ), bounds.first );
    }
    ++niter;
//    if( niter>2 )
//      std::cout << "\t" << res_[0];
  } while( relErr > rtol_ && niter < maxIter_ );
//if(niter>2)
//  std::cout << "\n -> converged in " << niter << " iterations  (" << soln[0] << ")\n";
  return niter < maxIter_;
}


//===================================================================


template< typename FieldT >
DensFromMixfrac<FieldT>::
DensFromMixfrac( const InterpT& rhoEval,
                 const Expr::Tag& rhoFTag )
  : Expr::Expression<FieldT>(),
    DensityCalculatorBase( 1, 1e-6, 5 ),
    rhoEval_( rhoEval ),
    rhoFTag_( rhoFTag ),
    bounds_( rhoEval.get_bounds()[0] )
{
  this->set_gpu_runnable(false);
}

//--------------------------------------------------------------------

template< typename FieldT >
DensFromMixfrac<FieldT>::
~DensFromMixfrac()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
DensFromMixfrac<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( rhoFTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
DensFromMixfrac<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const typename Expr::FieldMgrSelector<FieldT>::type& fm = fml.template field_manager<FieldT>();
  rhoF_ = &fm.field_ref( rhoFTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
DensFromMixfrac<FieldT>::
evaluate()
{
  FieldT& rho = this->value();

  typename FieldT::const_iterator irhoF = rhoF_->begin();
  typename FieldT::iterator irho = rho.begin();
  const typename FieldT::iterator irhoe = rho.end();
  size_t nbad = 0;
  DoubleVec soln(1), vals(1);
  for( ; irho!=irhoe; ++irho, ++irhoF ){
    vals[0] = *irhoF;
    soln[0] = *irhoF / *irho;   // initial guess for the mixture fraction
    const bool converged = this->solve( vals, soln );  // soln contains the mixture fraction
    if( !converged ) ++nbad;
    *irho = *irhoF / soln[0];
  }
  if( nbad>0 ){
    std::cout << "\tConvergence failed at " << nbad << " points.\n";
  }
}

//--------------------------------------------------------------------

template<typename FieldT>
void
DensFromMixfrac<FieldT>::
calc_jacobian_and_res( const DensityCalculatorBase::DoubleVec& passThrough,
                       const DensityCalculatorBase::DoubleVec& soln,
                       DensityCalculatorBase::DoubleVec& jac,
                       DensityCalculatorBase::DoubleVec& res )
{
  const double rhoF = passThrough[0];
  const double& f = soln[0];
  const double rhoCalc = rhoEval_.value( &f );
  jac[0] = rhoCalc + f * rhoEval_.derivative( &f, 0 );
  res[0] = f * rhoCalc - rhoF;
}

//--------------------------------------------------------------------

template<typename FieldT>
double
DensFromMixfrac<FieldT>::get_normalization_factor( const unsigned i ) const
{
  return 0.5; // nominal value for mixture fraction
}

//--------------------------------------------------------------------

template<typename FieldT>
std::pair<double,double>
DensFromMixfrac<FieldT>::get_bounds( const unsigned i ) const
{
  return bounds_;
}

//--------------------------------------------------------------------

template< typename FieldT >
DensFromMixfrac<FieldT>::
Builder::Builder( const InterpT& rhoEval,
                  const Expr::Tag& resultTag,
                  const Expr::Tag& rhoFTag  )
  : ExpressionBuilder( resultTag ),
    rhoEval_( rhoEval.clone() ),
    rhoFTag_( rhoFTag )
{
  if( resultTag.context() != Expr::CARRY_FORWARD ){
    std::ostringstream msg;
    msg << "ERROR: Density must have CARRY_FORWARD context so that an initial guess is available\n\t"
        << __FILE__ << " : " << __LINE__ << std::endl;
    throw std::runtime_error( msg.str() );
  }
}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
DensFromMixfrac<FieldT>::
Builder::build() const
{
  return new DensFromMixfrac<FieldT>( *rhoEval_, rhoFTag_ );
}



//===================================================================

template< typename FieldT >
DensHeatLossMixfrac<FieldT>::
DensHeatLossMixfrac( const Expr::Tag& hTag,
                     const Expr::Tag& rhofTag,
                     const Expr::Tag& rhohTag,
                     const InterpT& densEvaluator,
                     const InterpT& enthEvaluator )
  : Expr::Expression<FieldT>(),
    DensityCalculatorBase( 2, 1e-6, 5 ),
    hTag_   ( hTag    ),
    rhofTag_( rhofTag ),
    rhohTag_( rhohTag ),
    densEval_( densEvaluator ),
    enthEval_( enthEvaluator ),
    bounds_( densEvaluator.get_bounds() )
{}

//--------------------------------------------------------------------

template< typename FieldT >
DensHeatLossMixfrac<FieldT>::
~DensHeatLossMixfrac()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
DensHeatLossMixfrac<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( hTag_    );
  exprDeps.requires_expression( rhofTag_ );
  exprDeps.requires_expression( rhohTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
DensHeatLossMixfrac<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const typename Expr::FieldMgrSelector<FieldT>::type& fm = fml.template field_manager<FieldT>();
  rhof_ = &fm.field_ref( rhofTag_ );
  rhoh_ = &fm.field_ref( rhohTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
DensHeatLossMixfrac<FieldT>::
evaluate()
{
  typename Expr::Expression<FieldT>::ValVec& result  = this->get_value_vec();
  FieldT& density = *result[0];
  FieldT& gamma   = *result[1];

  typename FieldT::const_iterator irhof = rhof_->begin();
  typename FieldT::const_iterator irhoh = rhoh_->begin();
  typename FieldT::iterator irho = density.begin();
  typename FieldT::iterator igam = gamma.begin();
  const typename FieldT::iterator irhoe = density.end();

  DoubleVec soln(2), vals(2);
  for( ; irho!=irhoe; ++irho, ++igam, ++irhof, ++irhoh ){
    vals[0] = *irhof;
    vals[1] = *irhoh;
    soln[0] = *irhof / *irho;
    soln[1] = *igam;          // jcs this would require that gamma is CARRY_FORWARD.
    this->solve( vals, soln );
    *irho = *irhof / soln[0]; // set solution for density
    *igam = soln[1];          // heat loss
  }
}

//--------------------------------------------------------------------

template< typename FieldT >
void
DensHeatLossMixfrac<FieldT>::
calc_jacobian_and_res( const DensityCalculatorBase::DoubleVec& passThrough,
                       const DensityCalculatorBase::DoubleVec& soln,
                       DensityCalculatorBase::DoubleVec& jac,
                       DensityCalculatorBase::DoubleVec& res )
{
  const double& rhof = passThrough[0];
  const double& rhoh = passThrough[1];
  const double& f    = soln[0];
  const double& gam  = soln[1];

  // evaluate density and enthalpy given the current guess for f and gamma.
  const double rho = densEval_.value( soln );
  const double h   = enthEval_.value( soln );

  // evaluate the residual function
  res[0] = f * rho - rhof;
  res[1] = rho * h - rhoh;

  // evaluate derivative for use in the jacobian matrix
  const double drhodf   = densEval_.derivative( soln, 0 );
  const double drhodgam = densEval_.derivative( soln, 1 );
  const double dhdf     = enthEval_.derivative( soln, 0 );
  const double dhdgam   = enthEval_.derivative( soln, 1 );

  jac[0] = rho + f * drhodf;
  jac[1] = f * drhodgam;
  jac[2] = rho*dhdf   + h*drhodf;
  jac[3] = rho*dhdgam + h*drhodgam;
}

//--------------------------------------------------------------------

template<typename FieldT>
double
DensHeatLossMixfrac<FieldT>::get_normalization_factor( const unsigned i ) const
{
  return 0.5; // nominal value for mixture fraction and heat loss (which range [0,1] and [-1,1] respectively).
}

//--------------------------------------------------------------------

template<typename FieldT>
std::pair<double,double>
DensHeatLossMixfrac<FieldT>::get_bounds( const unsigned i ) const
{
  return bounds_[i];
}

//--------------------------------------------------------------------

template< typename FieldT >
DensHeatLossMixfrac<FieldT>::
Builder::Builder( const Expr::Tag& rhoTag,
                  const Expr::Tag& gammaTag,
                  const Expr::Tag& hTag,
                  const Expr::Tag& rhofTag,
                  const Expr::Tag& rhohTag,
                  const InterpT& densEvaluator,
                  const InterpT& enthEvaluator )
  : ExpressionBuilder( tag_list(rhoTag,gammaTag) ),
    hTag_   ( hTag    ),
    rhofTag_( rhofTag ),
    rhohTag_( rhohTag ),
    densEval_( densEvaluator.clone() ),
    enthEval_( enthEvaluator.clone() )
{
  if( rhoTag.context() != Expr::CARRY_FORWARD ){
    std::ostringstream msg;
    msg << "ERROR: Density must have CARRY_FORWARD context so that an initial guess is available\n\t"
        << __FILE__ << " : " << __LINE__ << std::endl;
    throw std::runtime_error( msg.str() );
  }
  if( hTag.context() != Expr::CARRY_FORWARD ){
    std::ostringstream msg;
    msg << "ERROR: Heat loss must have CARRY_FORWARD context so that an initial guess is available\n\t"
        << __FILE__ << " : " << __LINE__ << std::endl;
    throw std::runtime_error( msg.str() );
  }
}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
DensHeatLossMixfrac<FieldT>::
Builder::build() const
{
  return new DensHeatLossMixfrac<FieldT>( hTag_, rhofTag_,rhohTag_,*densEval_,*enthEval_ );
}

//====================================================================

// explicit template instantiation
#include <spatialops/structured/FVStaggeredFieldTypes.h>
template class DensFromMixfrac    <SpatialOps::structured::SVolField>;
template class DensHeatLossMixfrac<SpatialOps::structured::SVolField>;

