#ifndef DensityCalculator_Expr_h
#define DensityCalculator_Expr_h

#include <tabprops/BSpline.h>
#include <tabprops/StateTable.h>

#include <expression/Expr_Expression.h>

/**
 *  \ingroup WasatchExpressions
 *  \class  DensityCalculator
 *  \author Amir Biglari
 *  \date   Feb, 2011
 *
 *  \brief Evaluates density and eta if we are given rho*eta, using TabProps,
 *         an external library for tabular property evaluation.
 */
template< typename FieldT >
class DensityCalculator
: public Expr::Expression<FieldT>
{
  typedef std::vector<      FieldT*>  DepVarVec;
  typedef std::vector<const FieldT*>  IndepVarVec;
  
  typedef std::vector< typename FieldT::      iterator > VarIter;
  typedef std::vector< typename FieldT::const_iterator > ConstIter;
  
  typedef std::vector<double> PointValues;
  
  /* RhoetaIncEta and ReIEta mean "Rhoeta Included Eta" and are using to show a 
   subset of the eta's which can be weighted by density */
  /* RhoetaExEta and ReEEta mean "Rhoeta Excluded Eta" and are using to show  a 
   subset of the eta's which can NOT be weighted by density */
  const Expr::TagList rhoEtaTags_, rhoEtaIncEtaNames_, orderedEtaTags_;
  Expr::TagList rhoEtaExEtaNames_;
  
  const BSpline* const evaluator_;
  
  IndepVarVec rhoEta_;
  IndepVarVec rhoEtaExEta_;
  VarIter     rhoEtaIncEtaIters_;
  ConstIter   rhoEtaExEtaIters_;
  ConstIter   rhoEtaIters_;
  
  PointValues rhoEtaIncEtaPoint_;
  PointValues rhoEtaExEtaPoint_;
  PointValues rhoEtaPoint_;
  
  std::vector<int> ReIindex;
  
  // Linear solver function variables
  std::vector<double> J;           ///< A vector for the jacobian matrix
  std::vector<double> g;           ///< A vector for the functions
  std::vector<double> orderedEta;  ///< A vector to store all eta values in the same order as the table
  
  
  DensityCalculator( const BSpline* const spline,
                     const Expr::TagList& RhoEtaTags,           ///< rho*eta tag
                     const Expr::TagList& RhoetaIncEtaNames,    ///< Tag for ReIEta
                     const Expr::TagList& OrderedEtaTags,       ///< Tag for all of the eta's in the corect order
                     const Expr::ExpressionID& id,
                     const Expr::ExpressionRegistry& reg );
  
  void lin_solver( std::vector<double>& ReIeta, 
                   const std::vector<double>& ReEeta,
                   const std::vector<double>& RhoEta,
                   std::vector<int>& ReIindex,
                   double& rho,
                   const BSpline&,
                   const double rtol );
  
public:
  class Builder : public Expr::ExpressionBuilder
  {
    const Expr::TagList rhoEtaTs_, rhoEtaIncEtaNs_, rhoEtaExEtaNs_, orderedEtaTs_;
    const BSpline* const spline_;
  public:
    Builder( const BSpline* const spline,
             const Expr::TagList& rhoEtaTags,
             const Expr::TagList& rhoEtaIncEtaNames,
             const Expr::TagList& orderedEtaTags );
    
    Expr::ExpressionBase*
    build( const Expr::ExpressionID& id,
          const Expr::ExpressionRegistry& reg ) const;
  };
  
  ~DensityCalculator();
  
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void evaluate();
};



// ###################################################################
//
//                          Implementation
//
// ###################################################################



template< typename FieldT >
DensityCalculator<FieldT>::
DensityCalculator( const BSpline* const spline,
                   const Expr::TagList& rhoEtaTags,
                   const Expr::TagList& rhoetaIncEtaNames,
                   const Expr::TagList& orderedEtaTags,
                   const Expr::ExpressionID& id,
                   const Expr::ExpressionRegistry& reg )
: Expr::Expression<FieldT>(id,reg),
  rhoEtaTags_       ( rhoEtaTags        ),
  rhoEtaIncEtaNames_( rhoetaIncEtaNames ),
  orderedEtaTags_   ( orderedEtaTags    ),
  evaluator_        ( spline            )
{
  ReIindex.clear();
  int counter=0;
  bool match;
  for (typename Expr::TagList::const_iterator i=orderedEtaTags_.begin(); i!=orderedEtaTags_.end(); i++, counter++) {
    match=0;
    for (typename Expr::TagList::const_iterator j=rhoEtaIncEtaNames_.begin(); j!=rhoEtaIncEtaNames_.end(); j++) {
      if (*i==*j) {
        ReIindex.push_back( counter );
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
  const Expr::FieldManager<FieldT>& fm = fml.template field_manager<FieldT>();
  
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
  
  
  // loop over grid points.
  for( typename FieldT::iterator irho= rho.begin(); irho!=rho.end(); ++irho ){
    
    // initialize the density with a rough guess. (here we should put this equal to the density from the last time step)
    *irho = 0.2;
    
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
    lin_solver( rhoEtaIncEtaPoint_, rhoEtaExEtaPoint_, rhoEtaPoint_, ReIindex, *irho , *evaluator_, 1e-11 );
    
    // increment all iterators to the next grid point
    typename PointValues::const_iterator iReIEta=rhoEtaIncEtaPoint_.begin();
    for( typename VarIter::iterator i=rhoEtaIncEtaIters_.begin(); i!=rhoEtaIncEtaIters_.end(); ++i, ++iReIEta ){
      **i = *iReIEta;
      ++(*i);
    }
    for( typename ConstIter::iterator i=rhoEtaIters_.begin(); i!=rhoEtaIters_.end(); ++i )  ++(*i);
    for( typename ConstIter::iterator i=rhoEtaExEtaIters_.begin(); i!=rhoEtaExEtaIters_.end(); ++i )  ++(*i);
    
  } // grid loop
}

//--------------------------------------------------------------------

template< typename FieldT >
DensityCalculator<FieldT>::
Builder::Builder( const BSpline* const spline,
                  const Expr::TagList& rhoEtaTags,
                  const Expr::TagList& rhoEtaIncEtaNames,
                  const Expr::TagList& orderedEtaTags )
: spline_        ( spline            ),
  rhoEtaIncEtaNs_( rhoEtaIncEtaNames ),
  rhoEtaTs_      ( rhoEtaTags        ),
  orderedEtaTs_  ( orderedEtaTags    )
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
DensityCalculator<FieldT>::
Builder::build( const Expr::ExpressionID& id,
                const Expr::ExpressionRegistry& reg ) const
{
  return new DensityCalculator<FieldT>( spline_, rhoEtaTs_, rhoEtaIncEtaNs_, orderedEtaTs_, id, reg );
}

//====================================================================

extern "C"{
  void dgetrf_( int* m,
                int* n,
                double A[],
                int* lda,
                int* ipiv,
                int* info );
  
  void dgetrs_( char* trans,
                int* n,
                int* nrhs,
                double A[],
                int* lda,
                int* ipiv,
                double b[],
                int* ldb,
                int* info );
}

template< typename FieldT >
void
DensityCalculator<FieldT>::lin_solver( std::vector<double>& ReIeta, 
                                      const std::vector<double>& ReEeta,                                      
                                      const std::vector<double>& rhoEta,
                                      std::vector<int>& ReIindex,
                                      double& rho,
                                      const BSpline& eval,
                                      const double rtol )
{
  const size_t size = rhoEta.size();
  int itCounter = 0;
  double rlxFac = 1.0;
  
  if (size==0) {
    orderedEta.clear();
    orderedEta=ReEeta;
    rho = eval.value(orderedEta);
  }
  else {
    std::cout << ", Lambda=" << ReEeta[0] << ", rhoeta(real)=" << rhoEta[0] << std::endl;
    // A vector for the jacobian matrix
    J.resize(size*size);
    // A vector for the functions
    g.resize(size);
    
    PointValues ReIetaTmp(ReIeta);
    double relErr=rtol+1;
    
    do {
      itCounter++;
      // Creating the ordered eta vector
      orderedEta.clear();
      orderedEta=ReEeta;
      for (int i=0; i<ReIindex.size(); i++) {
        orderedEta.insert(orderedEta.begin()+ReIindex[i],ReIeta[i]);
      }
      
      const double rhotmp = eval.value( orderedEta );
      
      // Loop over different etas
      for( int k=0; k<size; ++k ){
        for (int i=0; i<size; ++i) {
          if (k==i) 
            ReIetaTmp[i]=ReIeta[k] + (std::abs(ReIeta[k])+rtol) * 0.05;
          else
            ReIetaTmp[i] = ReIeta[i];
        }
        
        // Recreating the ordered eta vector
        orderedEta.clear();
        orderedEta=ReEeta;
        for (int i=0; i<ReIindex.size(); i++) {
          orderedEta.insert( orderedEta.begin()+ReIindex[i], ReIetaTmp[i] );
        }
        
        const double rhoplus = eval.value(&orderedEta[0]);
        
        // Calculating the function members
        g[k] = -( ReIeta[k] - (rhoEta[k] / rhotmp));
        
        for (int i=0; i<size; ++i) {        
          J[i + k*size] = (( ReIetaTmp[i] - rhoEta[i]/rhoplus ) - ( ReIeta[i] - rhoEta[i]/rhotmp )) / ((std::abs(ReIeta[k])+ rtol) * 0.05 );
        }
      }
      
      //Solving the equation
      std::vector<int> ipiv(size);
      std::vector<double> delta = g;
      int info;
      char mode = 'n';
      int neq = size;
      int one = 1;
      
      // Factor general matrix J using Gaussian elimination with partial pivoting
      dgetrf_(&neq, &neq, &J[0], &neq, &ipiv[0], &info);
      // Solving J * delta = g
      dgetrs_(&mode, &neq, &one, &J[0], &neq, &ipiv[0], &delta[0], &neq, &info);
//      std::cout << "delta: " << delta[0] << "   " << delta[1] << std::endl;
      
      // relaxations !!! This part may be removed after adding density update part !!!
      if (itCounter==100) rlxFac = 0.8;
      else if (itCounter==200) rlxFac = 0.6;      
      else if (itCounter==300) rlxFac = 1.2;
      else if (itCounter==400) rlxFac = 1.4;
      else if (itCounter==500){
        itCounter=0;
        rlxFac=1.0;
      }
      
      relErr = 0.0;
      for( int i=0; i<size; ++i ){
        ReIeta[i] = ReIeta[i] + delta[i]*rlxFac;
        relErr = relErr + std::abs( delta[i]/(std::abs(ReIeta[i])+rtol) );
      }
      
/*      if (size==2) {
        std::cout << "rho: " << rhotmp << std::endl
        << "eta: " << ReIeta[0] << "   " << ReIeta[1] << std::endl;
        std::cout << "relerr: " << relErr << std::endl;
      }
      else {
        std::cout << "rho: " << rhotmp << std::endl
        << "eta: " << ReIeta[0] << std::endl;
        std::cout << "relerr: " << relErr << std::endl;
      }
*/
      
      // Update rho
      rho=rhotmp;
      
    } while (relErr>rtol);
    
    std::cout << "converged.  eta=" << ReIeta[0] << ", rhoeta=" << rho*ReIeta[0] << std::endl;
    std::cout << "it No. : " << itCounter << std::endl;
  }
}

//====================================================================

#endif // DensityCalculator_Expr_h
