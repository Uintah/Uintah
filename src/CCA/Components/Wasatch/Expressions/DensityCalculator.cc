#include "DensityCalculator.h"

#include <boost/foreach.hpp>

//-------------------------------------------------------------------

template< typename FieldT >
DensityCalculator<FieldT>::
DensityCalculator( const InterpT& evaluator,
                   const Expr::TagList& rhoEtaTags,
                   const Expr::TagList& etaTags,
                   const Expr::TagList& orderedIvarTags )
: Expr::Expression<FieldT>(),
  rhoEtaTags_     ( rhoEtaTags      ),
  etaTags_        ( etaTags         ),
  orderedIvarTags_( orderedIvarTags ),
  evaluator_      ( evaluator       )
{
  etaIndex_.clear();
  size_t counter=0;
  BOOST_FOREACH( const Expr::Tag& ivarTag, orderedIvarTags_ ){
    const typename Expr::TagList::const_iterator ii = std::find( etaTags_.begin(), etaTags_.end(), ivarTag );
    if( ii != etaTags_.end() ) etaIndex_.push_back( counter );
    else                       thetaTags_.push_back( ivarTag );
    ++counter;
  }
}

//--------------------------------------------------------------------

template< typename FieldT >
DensityCalculator<FieldT>::
~DensityCalculator()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
DensityCalculator<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  BOOST_FOREACH( const Expr::Tag& tag, rhoEtaTags_ ) exprDeps.requires_expression( tag );
  BOOST_FOREACH( const Expr::Tag& tag, thetaTags_  ) exprDeps.requires_expression( tag );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
DensityCalculator<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const typename Expr::FieldMgrSelector<FieldT>::type& fm = fml.template field_manager<FieldT>();

  rhoEta_.clear();
  BOOST_FOREACH( const Expr::Tag& tag, rhoEtaTags_ ){
    rhoEta_.push_back( &fm.field_ref( tag ) );
  }

  theta_.clear();
  BOOST_FOREACH( const Expr::Tag& tag, thetaTags_ ){
    theta_.push_back( &fm.field_ref( tag ) );
  }
}

//--------------------------------------------------------------------

template< typename FieldT >
void
DensityCalculator<FieldT>::
evaluate()
{
  FieldT& rho = this->value();

  thetaIters_.clear();
  rhoEtaIters_.clear();

  for( typename IndepVarVec::iterator i=theta_.begin(); i!=theta_.end(); ++i ){
    thetaIters_.push_back( (*i)->begin() );
  }
  for( typename IndepVarVec::const_iterator i=rhoEta_.begin(); i!=rhoEta_.end(); ++i ){
    rhoEtaIters_.push_back( (*i)->begin() );
  }

  unsigned int convergeFailed=0;

  // loop over grid points.
  for( typename FieldT::iterator irho= rho.begin(); irho!=rho.end(); ++irho ){

    // extract indep vars at this grid point
    etaPoint_.clear();
    rhoEtaPoint_.clear();
    for( typename ConstIter::const_iterator i=rhoEtaIters_.begin(); i!=rhoEtaIters_.end(); ++i ){
      rhoEtaPoint_.push_back( **i );
      etaPoint_.push_back( **i / (*irho + 1e-11) );
    }
    thetaPoint_.clear();
    for( typename ConstIter::const_iterator i=thetaIters_.begin(); i!=thetaIters_.end(); ++i ){
      thetaPoint_.push_back( **i );
    }

    // calculate the result
    const bool converged = nonlinear_solver( *irho , etaPoint_, thetaPoint_, rhoEtaPoint_, etaIndex_, evaluator_, 1e-9 );
    if( !converged )  ++convergeFailed;

    // increment all iterators to the next grid point
    for( typename ConstIter::iterator i=rhoEtaIters_.begin(); i!=rhoEtaIters_.end(); ++i )  ++(*i);
    for( typename ConstIter::iterator i=thetaIters_.begin();  i!=thetaIters_.end();  ++i )  ++(*i);

  } // grid loop

  if( convergeFailed > 0 )
    std::cout << convergeFailed << " of " << rho.window_with_ghost().local_npts() << " points failed to converge on density solver" << std::endl;
}

//--------------------------------------------------------------------

template< typename FieldT >
DensityCalculator<FieldT>::
Builder::Builder( const Expr::Tag& result,
                  const InterpT& densEvaluator,
                  const Expr::TagList& rhoEtaTags,
                  const Expr::TagList& etaTags,
                  const Expr::TagList& orderedEtaTags )
: ExpressionBuilder(result),
  rhoEtaTs_    ( rhoEtaTags     ),
  etaTs_       ( etaTags        ),
  orderedIvarTs_( orderedEtaTags ),
  interp_       ( densEvaluator.clone() )
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
DensityCalculator<FieldT>::
Builder::build() const
{
  return new DensityCalculator<FieldT>( *interp_, rhoEtaTs_, etaTs_, orderedIvarTs_);
}

//====================================================================

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

template< typename FieldT >
bool
DensityCalculator<FieldT>::nonlinear_solver( double& rho,
                                             std::vector<double>& eta,
                                             const std::vector<double>& theta,
                                             const std::vector<double>& rhoEta,
                                             const std::vector<size_t>& etaIndex,
                                             const InterpT& eval,
                                             const double rtol )
{
  using namespace std;
  const size_t neq = rhoEta.size();
  unsigned int itCounter = 0;
  const unsigned int maxIter = 5;
  
  orderedIvars_.clear();
  orderedIvars_ = theta;

  if( neq==0 ){
    // no solve required - just a straight function evaluation.
    rho = eval.value( orderedIvars_ );
    return true;
  }
  else{
    // Create the ordered eta vector
    for( size_t i=0; i<etaIndex.size(); ++i ) {
      orderedIvars_.insert( orderedIvars_.begin()+etaIndex[i], eta[i] );
    }
  }

  jac_   .resize(neq*neq);  // vector for the jacobian matrix
  g_     .resize(neq    );  // vector for the rhs functions in non-linear solver
  delta_ .resize(neq    );  // finite difference vector
  etaTmp_.resize(neq    );  // work array for assembling the jacobian
  ipiv_  .resize(neq    );  // work array for linear solver

  double relErr=0.0;

  for( size_t i=0; i<neq; ++i ){
    delta_[i] = 1e-6 * eta[i] + 1e-11;
  }

  const std::vector<std::pair<double,double> > bounds = eval.get_bounds();

  do{
    ++itCounter;

    // update the ordered eta vector with the current values of eta.
    // jcs note that we could move this to the bottom of the loop, but
    // it results in slightly different answers and diffs regression tests.
    // Really, it should be at the bottom since otherwise we don't get
    // the benefit of the final iteration...
    for( size_t i=0; i<etaIndex.size(); ++i ) {
      orderedIvars_[ etaIndex[i] ] = eta[i];
    }

    rho = eval.value( orderedIvars_ );

    // Loop over different etas to construct the linear system
    for( size_t k=0; k<neq; ++k ){
//#define USE_OLD_SOLVER
#ifdef USE_OLD_SOLVER
      for( size_t i=0; i<neq; ++i ) {
        if( k==i ) etaTmp_[i] = eta[k] + delta_[k];
        else       etaTmp_[i] = eta[i];
      }

      // update the ordered eta vector with the modified eta vector
      for( size_t i=0; i<etaIndex.size(); ++i ) {
        orderedIvars_[ etaIndex[i] ] = etaTmp_[i];
      }

      const double rhoplus = eval.value( &orderedIvars_[0] );

      // Calculating the rhs vector components
      g_[k] = -( eta[k] - (rhoEta[k] / rho));
      for( size_t i=0; i<neq; ++i ) {
        jac_[i + k*neq] = (( etaTmp_[i] - rhoEta[i]/rhoplus ) - ( eta[i] - rhoEta[i]/rho )) / delta_[k];
      }
#else
      g_[k] = -( rho*eta[k] - rhoEta[k] );
      for( size_t i=0; i<neq; ++i ){
        // jcs could move the density derivative evaluation outside the loop and save it off.  Otherwise, this is a lot of duplicate calculation...
        const double der = eval.derivative( &orderedIvars_[0], i );
        jac_[i+k*neq] = eta[i] * der + rho;
      }
#endif
    } // linear system construction

    // Solve the linear system
    const int one=1; int info;

    const int numEqns = neq;
    // Solving J * delta = g
    // note that on entry, g is the rhs and on exit, g is the solution (delta).
    dgesv_( &numEqns, &one, &jac_[0], &numEqns, &ipiv_[0], &g_[0], &numEqns, &info );
    assert( info==0 );

    // relative error calculations
    relErr = 0.0;
    for( size_t i=0; i<neq; ++i ){
      eta[i] += g_[i];
      eta[i] = std::max( std::min( bounds[i].second,  eta[i] ), bounds[i].first );
      relErr += std::abs( 2*g_[i]/(bounds[i].second+bounds[i].first) );
    }
    relErr /= neq;

  } while( relErr>rtol && itCounter < maxIter );

  if( itCounter >= maxIter ){
//    std::cout << " problems!  " << rho << " , " << relErr << ", " << g_[0] << ", " << eta[0] <<", "<< rhoEta[0] << std::endl;
    return false;
  }
  //    std::cout << "converged in " << itCounter << " iterations.  eta=" << eta[0] << ", rhoeta=" << rho*eta[0] << ", rho=" << rho << std::endl;

  return true;
}

//====================================================================

// explicit template instantiation
#include <spatialops/structured/FVStaggeredFieldTypes.h>
template class DensityCalculator<SpatialOps::structured::SVolField>;

