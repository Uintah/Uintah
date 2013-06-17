
#include "DensityCalculator.h"

//-------------------------------------------------------------------

template< typename FieldT >
DensityCalculator<FieldT>::
DensityCalculator( const InterpT* const evaluator,
                   const Expr::TagList& rhoEtaTags,
                   const Expr::TagList& rhoetaIncEtaNames,
                   const Expr::TagList& orderedEtaTags )
: Expr::Expression<FieldT>(),
  rhoEtaTags_       ( rhoEtaTags        ),
  rhoEtaIncEtaNames_( rhoetaIncEtaNames ),
  orderedEtaTags_   ( orderedEtaTags    ),
  evaluator_        ( evaluator         )
{
  reIindex_.clear();
  int counter=0;
  bool match;
  for( typename Expr::TagList::const_iterator i=orderedEtaTags_.begin();
      i!=orderedEtaTags_.end();
      ++i, ++counter )
  {
    match=0;
    for( typename Expr::TagList::const_iterator j=rhoEtaIncEtaNames_.begin();
         j!=rhoEtaIncEtaNames_.end();
         ++j )
    {
      if (*i==*j) {
        reIindex_.push_back( counter );
        match=1;
        break;
      }
    }

    if (!match)  {
      rhoEtaExEtaNames_.push_back( *i );
    }
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
  for( Expr::TagList::const_iterator inam=rhoEtaTags_.begin(); inam!=rhoEtaTags_.end(); ++inam ){
    exprDeps.requires_expression( *inam );
  }
  for( Expr::TagList::const_iterator inam=rhoEtaExEtaNames_.begin(); inam!=rhoEtaExEtaNames_.end(); ++inam ){
    exprDeps.requires_expression( *inam );
  }
}

//--------------------------------------------------------------------

template< typename FieldT >
void
DensityCalculator<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const typename Expr::FieldMgrSelector<FieldT>::type& fm = fml.template field_manager<FieldT>();

  rhoEta_.clear();
  for( Expr::TagList::const_iterator inam=rhoEtaTags_.begin(); inam!=rhoEtaTags_.end(); ++inam ){
    rhoEta_.push_back( &fm.field_ref( *inam ) );
  }
  rhoEtaExEta_.clear();
  for( Expr::TagList::const_iterator inam=rhoEtaExEtaNames_.begin(); inam!=rhoEtaExEtaNames_.end(); ++inam ){
    rhoEtaExEta_.push_back( &fm.field_ref( *inam ) );
  }
}

//--------------------------------------------------------------------

template< typename FieldT >
void
DensityCalculator<FieldT>::
evaluate()
{
  DepVarVec& results = this->get_value_vec();

  DepVarVec rhoEtaIncEta;
  typename DepVarVec::iterator iReIEta = results.begin();
  FieldT& rho = **iReIEta;
  ++iReIEta;
  for( ; iReIEta!=results.end(); ++iReIEta ){
    rhoEtaIncEta.push_back( *iReIEta );
  }

  rhoEtaIncEtaIters_.clear();
  rhoEtaExEtaIters_.clear();
  rhoEtaIters_.clear();

  for( typename DepVarVec::iterator i=rhoEtaIncEta.begin(); i!=rhoEtaIncEta.end(); ++i ){
    rhoEtaIncEtaIters_.push_back( (*i)->begin() );
  }
  for( typename IndepVarVec::iterator i=rhoEtaExEta_.begin(); i!=rhoEtaExEta_.end(); ++i ){
    rhoEtaExEtaIters_.push_back( (*i)->begin() );
  }
  for( typename IndepVarVec::const_iterator i=rhoEta_.begin(); i!=rhoEta_.end(); ++i ){
    rhoEtaIters_.push_back( (*i)->begin() );
  }

  unsigned int convergeFailed=0;

  // loop over grid points.
  for( typename FieldT::iterator irho= rho.begin(); irho!=rho.end(); ++irho ){

    // extract indep vars at this grid point
    rhoEtaIncEtaPoint_.clear();
    rhoEtaPoint_.clear();
    for( typename ConstIter::const_iterator i=rhoEtaIters_.begin(); i!=rhoEtaIters_.end(); ++i ){
      rhoEtaPoint_.push_back( **i );
      rhoEtaIncEtaPoint_.push_back( **i / (*irho + 1e-11) );
    }
    rhoEtaExEtaPoint_.clear();
    for( typename ConstIter::const_iterator i=rhoEtaExEtaIters_.begin(); i!=rhoEtaExEtaIters_.end(); ++i ){
      rhoEtaExEtaPoint_.push_back( **i );
    }

    // calculate the result
    const bool converged = nonlinear_solver( rhoEtaIncEtaPoint_, rhoEtaExEtaPoint_, rhoEtaPoint_, reIindex_, *irho , *evaluator_, 1e-9 );
    if( !converged )  ++convergeFailed;

    // increment all iterators to the next grid point
    typename PointValues::const_iterator iReIEta=rhoEtaIncEtaPoint_.begin();
    for( typename VarIter::iterator i=rhoEtaIncEtaIters_.begin(); i!=rhoEtaIncEtaIters_.end(); ++i, ++iReIEta ){
      **i = *iReIEta;
      ++(*i);
    }
    for( typename ConstIter::iterator i=rhoEtaIters_.begin(); i!=rhoEtaIters_.end(); ++i )  ++(*i);
    for( typename ConstIter::iterator i=rhoEtaExEtaIters_.begin(); i!=rhoEtaExEtaIters_.end(); ++i )  ++(*i);

  } // grid loop

  if( convergeFailed>0 )
    std::cout << convergeFailed << " of " << rho.window_with_ghost().local_npts() << " points failed to converge on density solver" << std::endl;
}

//--------------------------------------------------------------------

template< typename FieldT >
DensityCalculator<FieldT>::
Builder::Builder( const Expr::Tag& result,
                  const InterpT* const densEvaluator,
                  const Expr::TagList& rhoEtaTags,
                  const Expr::TagList& rhoEtaIncEtaNames,
                  const Expr::TagList& orderedEtaTags )
: ExpressionBuilder(result),
  rhoEtaTs_      ( rhoEtaTags        ),
  rhoEtaIncEtaNs_( rhoEtaIncEtaNames ),
  orderedEtaTs_  ( orderedEtaTags    ),
  spline_        ( densEvaluator     )
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
DensityCalculator<FieldT>::
Builder::build() const
{
  return new DensityCalculator<FieldT>( spline_, rhoEtaTs_, rhoEtaIncEtaNs_, orderedEtaTs_);
}

//====================================================================

extern "C"{
  void dgetrf_( const int* m,
                const int* n,
                double A[],
                const int* lda,
                int* ipiv,
                int* info );

  void dgetrs_( const char* trans,
                const int* n,
                const int* nrhs,
                const double A[],
                const int* lda,
                const int* ipiv,
                double b[],
                const int* ldb,
                int* info );
}

template< typename FieldT >
bool
DensityCalculator<FieldT>::nonlinear_solver( std::vector<double>& reIeta,
                                             const std::vector<double>& reEeta,
                                             const std::vector<double>& rhoEta,
                                             const std::vector<int>& reIindex,
                                             double& rho,
                                             const InterpT& eval,
                                             const double rtol )
{
  using namespace std;
  const size_t neq = rhoEta.size();
  unsigned int itCounter = 0;
  const unsigned int maxIter = 20;  // the lowest iteration number seen in the tests is 5 which makes maxIter=6.
  
  if( neq==0 ){
    orderedEta_.clear();
    orderedEta_=reEeta;
    rho = eval.value(orderedEta_);
  }
  else{
    jac_.resize(neq*neq);       // A vector for the jacobian matrix
    g_.resize(neq);             // A vector for the rhs functions in non-linear solver
    std::vector<int> ipiv(neq); // work array for linear solver

    PointValues reIetaTmp(reIeta);
    double relErr;

    std::vector<double> deleta(neq); 
    for (size_t i=0; i<neq; ++i) 
      deleta[i] = 1e-6 * reIeta[i] + 1e-11;

    do{
      ++itCounter;
      // Create the ordered eta vector
      orderedEta_.clear();
      orderedEta_=reEeta;
      for( size_t i=0; i<reIindex.size(); i++ ) {
        orderedEta_.insert(orderedEta_.begin()+reIindex[i],reIeta[i]);
      }

      double rhotmp = eval.value( orderedEta_ );

      // Loop over different etas
      for( size_t k=0; k<neq; ++k ){
        for( size_t i=0; i<neq; ++i ) {
          if( k==i )
            reIetaTmp[i]=reIeta[k] + deleta[k];
          else
            reIetaTmp[i] = reIeta[i];
        }

        // Recreate the ordered eta vector with the modified eta vector
        orderedEta_.clear();
        orderedEta_=reEeta;
        for( size_t i=0; i<reIindex.size(); i++ ) {
          orderedEta_.insert( orderedEta_.begin()+reIindex[i], reIetaTmp[i] );
        }

        const double rhoplus = eval.value(&orderedEta_[0]);

        // Calculating the rhs vextor components
        g_[k] = -( reIeta[k] - (rhoEta[k] / rhotmp));
        for( size_t i=0; i<neq; ++i ) {
          jac_[i + k*neq] = (( reIetaTmp[i] - rhoEta[i]/rhoplus ) - ( reIeta[i] - rhoEta[i]/rhotmp )) / deleta[k];
        }
      }
      
      // Solve the linear system
      const char mode = 'n';
      int one=1, info;

      // jcs why not use dgesv instead?
      // Factor general matrix J using Gaussian elimination with partial pivoting
      const int numEqns = neq;
      dgetrf_(&numEqns, &numEqns, &jac_[0], &numEqns, &ipiv[0], &info);
      assert( info==0 );
      // Solving J * delta = g
      // note that on entry, g is the rhs and on exit, g is the solution (delta).
      dgetrs_(&mode, &numEqns, &one, &jac_[0], &numEqns, &ipiv[0], &g_[0], &numEqns, &info);
      assert(info==0);

      // relative error calculations
      relErr = 0.0;
      for( size_t i=0; i<neq; ++i ){
        reIeta[i] += g_[i];
        relErr += std::abs( g_[i]/(std::abs(reIeta[i])+rtol) );
      }

      rho = rhotmp;  // Updating rho
    } while (relErr>rtol && itCounter<maxIter);
    if( itCounter >= maxIter ){
//      std::cout << itCounter << setprecision(15) << " problems!  " << rho << " , " << relErr << ", " << g[0] << ", " << ReIeta[0] <<", "<<rhoEta[0]<< ", " << testrho << ", " << ReEeta[0] << std::endl;
      return false;
    }
//    std::cout << "converged in " << itCounter << " iterations.  eta=" << ReIeta[0] << ", rhoeta=" << rho*ReIeta[0] << ", rho=" << rho << std::endl;  

  } 

  return true;
}

//====================================================================

// explicit template instantiation
#include <spatialops/structured/FVStaggeredFieldTypes.h>
template class DensityCalculator<SpatialOps::structured::SVolField>;

