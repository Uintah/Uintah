#include <CCA/Components/Wasatch/Expressions/DensityCalculator.h>

#include <sci_defs/uintah_defs.h>

#include <boost/foreach.hpp>

#define DGESV FIX_NAME(dgesv)

extern "C"{
  void DGESV( const int    * n,    // # of equations
              const int    * nrhs, // # of rhs to solve
                    double   A[],  // lhs matrix
              const int    * lda,  // leading dim of A (#eqn for us)
                    int    * ipiv, // integer work array
                    double   rhs[],// rhs array
              const int    * ldb,  // dimension of rhs (#eqn for us)
                    int    * info );
}

//-------------------------------------------------------------------
namespace OldDensityCalculator{

DensityCalculatorBase::
DensityCalculatorBase( const int neq,
                       const double rtol,
                       const size_t maxIter )
: rtol_   ( rtol    ),
  maxIter_( maxIter ),
  neq_    ( neq     )
{
  jac_ .resize( neq_*neq_ );
  res_ .resize( neq_      );
  ipiv_.resize( neq_      );
}

DensityCalculatorBase::~DensityCalculatorBase(){}

bool DensityCalculatorBase::solve( const DoubleVec& passThrough,
                                   DoubleVec& soln,
                                   double& relError)
{
  unsigned niter = 0;
  double relErr = 0.0;
  if (maxIter_ == 0) return true;
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
        DGESV( &neq_, &one, &jac_[0], &neq_, &ipiv_[0], &res_[0], &neq_, &info );
        if( info != 0 ){
          std::cout << "\nSOLVER FAILED: "<< info << "  " << soln[0] << ", " << soln[1] << std::endl;
        }
        assert( info==0 );
        break;
    } // switch
    relErr = 0.0;
    for( int i=0; i<neq_; ++i ){
      soln[i] -= res_[i];
      relErr += std::abs( res_[i]/get_normalization_factor(i) );
      // clip the solution to the valid range
      const std::pair<double,double>& bounds = get_bounds(i);
      soln[i] = std::max( std::min( bounds.second, soln[i] ), bounds.first );
    }
    ++niter;
  } while( relErr > rtol_ && niter < maxIter_ );
  relError = relErr;
  return niter < maxIter_;
}


//===================================================================

template< typename FieldT >
DensHeatLossMixfrac<FieldT>::
DensHeatLossMixfrac( const Expr::Tag& rhoOldTag,
                     const Expr::Tag& gammaOldTag,
                     const Expr::Tag& rhofTag,
                     const Expr::Tag& rhohTag,
                     const InterpT& densEvaluator,
                     const InterpT& enthEvaluator )
  : Expr::Expression<FieldT>(),
    DensityCalculatorBase( 2, 1e-6, 5 ),
    densEval_( densEvaluator ),
    enthEval_( enthEvaluator ),
    bounds_( densEvaluator.get_bounds() )
{
  rhoOld_ = this->template create_field_request<FieldT>(rhoOldTag);
  gammaOld_ = this->template create_field_request<FieldT>(gammaOldTag);
  rhof_ = this->template create_field_request<FieldT>(rhofTag);
  rhoh_ = this->template create_field_request<FieldT>(rhohTag);
}

//--------------------------------------------------------------------

template< typename FieldT >
DensHeatLossMixfrac<FieldT>::
~DensHeatLossMixfrac()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
DensHeatLossMixfrac<FieldT>::
evaluate()
{
  typename Expr::Expression<FieldT>::ValVec& result  = this->get_value_vec();
  FieldT& density = *result[0];
  FieldT& gamma   = *result[1];
  FieldT& dRhodF  = *result[2];
  FieldT& dRhodH  = *result[3];

  density <<= rhoOld_->field_ref();
  gamma   <<= gammaOld_->field_ref();
  
  const FieldT& rhof = rhof_->field_ref();
  const FieldT& rhoh = rhoh_->field_ref();
  
  typename FieldT::const_iterator irhof = rhof   .begin();
  typename FieldT::const_iterator irhoh = rhoh   .begin();
  typename FieldT::iterator irho        = density.begin();
  typename FieldT::iterator igam        = gamma  .begin();
  typename FieldT::iterator idRhodF     = dRhodF .begin();
  typename FieldT::iterator idRhodH     = dRhodH .begin();
  const typename FieldT::iterator irhoe = density.end();

  size_t nbad=0;
  DoubleVec soln(2), vals(2);
  double relError = 0.0;
  for( ; irho!=irhoe; ++irho, ++igam, ++irhof, ++irhoh, ++idRhodF, ++idRhodH ){
    vals[0] = *irhof;
    vals[1] = *irhoh;
    soln[0] = *irhof / *irho; // mixture fraction
    soln[1] = *igam;          // heat loss
    const bool converged = this->solve( vals, soln, relError );
    if( !converged ) ++nbad;
    *irho = *irhof / soln[0]; // set solution for density
    *igam = soln[1];          // heat loss

    *idRhodF = densEval_.derivative( soln, 0 );
    const double dRhodGamma = densEval_.derivative( soln, 1 );
    const double dHdGamma   = enthEval_.derivative( soln, 1 );
    assert( dHdGamma != 0 );

    // calculate d(rho)/d(h) using the chain rule
    *idRhodH = dRhodGamma/dHdGamma;
  }
  if( nbad>0 ){
    std::cout << "\tConvergence failed at " << nbad << " of " << density.window_with_ghost().local_npts() << " points.\n";
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
  //const double& gam  = soln[1];

  // if we hit the bounds on mixture fraction, we have a degenerate case,
  // so don't solve for heat loss in that situation.
  const double tol = 1e-4;
  const bool atBounds = ( std::abs(f-bounds_[0].first ) < tol ||
                          std::abs(f-bounds_[0].second) < tol );

  // evaluate density and enthalpy given the current guess for f and gamma.
  const double rho =                       densEval_.value( soln );
  const double h   = atBounds ? rhoh/rho : enthEval_.value( soln );
  // evaluate the residual function
  res[0] = f * rho - rhof;
  res[1] = atBounds ? 0 : rho * h - rhoh;

  // evaluate derivative for use in the jacobian matrix
  const double drhodf   =                densEval_.derivative( soln, 0 );
  const double drhodgam = atBounds ? 0 : densEval_.derivative( soln, 1 );
  const double dhdf     = atBounds ? 0 : enthEval_.derivative( soln, 0 );
  const double dhdgam   = atBounds ? 0 : enthEval_.derivative( soln, 1 );

  // strange ordering because of fortran/c conventions.
  jac[0] = rho + f * drhodf;
  jac[2] = atBounds ? 0 : f * drhodgam;
  jac[1] = atBounds ? 0 : rho*dhdf   + h*drhodf;
  jac[3] = atBounds ? 1 : rho*dhdgam + h*drhodgam;
}

//--------------------------------------------------------------------

template< typename FieldT >
DensHeatLossMixfrac<FieldT>::
Builder::Builder( const Expr::Tag& rhoTag,
                  const Expr::Tag& gammaTag,
                  const Expr::Tag& dRhodFTag,
                  const Expr::Tag& dRhodHTag,
                  const Expr::Tag& rhoOldTag,
                  const Expr::Tag& gammaOldTag,
                  const Expr::Tag& rhofTag,
                  const Expr::Tag& rhohTag,
                  const InterpT& densEvaluator,
                  const InterpT& enthEvaluator )
  : ExpressionBuilder( tag_list(rhoTag,gammaTag, dRhodFTag, dRhodHTag) ),
    rhoOldTag_   ( rhoOldTag             ),
    gammaOldTag_ ( gammaOldTag           ),
    rhofTag_     ( rhofTag               ),
    rhohTag_     ( rhohTag               ),
    densEval_    ( densEvaluator.clone() ),
    enthEval_    ( enthEvaluator.clone() )
{}

//====================================================================
// explicit template instantiation
#include <spatialops/structured/FVStaggeredFieldTypes.h>
template class DensHeatLossMixfrac<SpatialOps::SVolField>;
}// namespace OldDensityCalculator

