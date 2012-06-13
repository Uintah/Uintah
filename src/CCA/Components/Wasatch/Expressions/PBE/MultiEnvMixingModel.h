#ifndef MultiEnvMixingModel_Expr_h
#define MultiEnvMixingModel_Expr_h
#include <spatialops/structured/FVStaggeredFieldTypes.h>
#include <spatialops/structured/FVStaggeredOperatorTypes.h>

#include <expression/Expression.h>

/**
 *  \ingroup WasatchExpressions
 *  \class MultiEnvMixingModel
 *  \author Alex Abboud	 
 *  \date June 2012
 *  \tparam FieldT the type of field.
 *  \brief Implements a basic three absciassae multi environment mixing model
 *  fixes w_1 at mixfrac = 0 & w_3 at mixfrac = 1
 *  closure of this is that w_2 = <Z>
 *  for precipitation, reaction only occurs at w_2
 *  this returns a vector of weights
 *  with a vector of dw/dt base on scalr diss
 *  [w1 dw1/dt w2 dw2/dt w3 dw3/dt] 
 */
template< typename FieldT >
class MultiEnvMixingModel
: public Expr::Expression<FieldT>
{
  
  const Expr::Tag mixFracTag_, scalarVarTag_, scalarDissTag_;    //this will correspond to proper tags for mix frac & sclar var
  const FieldT* mixFrac_; 											 // mixture fraction from grid
  const FieldT* scalarVar_; 										 // sclar variance form grid
  const FieldT* scalarDiss_;
  
  MultiEnvMixingModel( const Expr::Tag& mixFracTag_,
                       const Expr::Tag& scalarVarTag_,
                      const Expr::Tag& scalarDissTag_);
  
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::TagList& results,
             const Expr::Tag& mixFracTag,
             const Expr::Tag& scalarVarTag,
             const Expr::Tag& scalarDissTag)
    : ExpressionBuilder(results),
    mixfract_(mixFracTag),
    scalarvart_(scalarVarTag),
    scalardisst_(scalarDissTag)
    {}
    
    ~Builder(){}
    
    Expr::ExpressionBase* build() const
    {
      return new MultiEnvMixingModel<FieldT>( mixfract_, scalarvart_, scalardisst_ );
    }
    
  private:
    const Expr::Tag mixfract_, scalarvart_, scalardisst_;
  };
  
  ~MultiEnvMixingModel();
  
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
};


// ###################################################################
//
//                          Implementation
//
// ###################################################################

template< typename FieldT >
MultiEnvMixingModel<FieldT>::
MultiEnvMixingModel( const Expr::Tag& mixFracTag,
                     const Expr::Tag& scalarVarTag,
                    const Expr::Tag& scalarDissTag)
: Expr::Expression<FieldT>(),
mixFracTag_(mixFracTag),
scalarVarTag_(scalarVarTag),
scalarDissTag_(scalarDissTag)
{}

//--------------------------------------------------------------------

template< typename FieldT >
MultiEnvMixingModel<FieldT>::
~MultiEnvMixingModel()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
MultiEnvMixingModel<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( mixFracTag_ );
  exprDeps.requires_expression( scalarVarTag_ );
  exprDeps.requires_expression( scalarDissTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
MultiEnvMixingModel<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldManager<FieldT>& fm = fml.template field_manager<FieldT>();
  mixFrac_ = &fm.field_ref( mixFracTag_ );
  scalarVar_ = &fm.field_ref( scalarVarTag_ );
  scalarDiss_ = &fm.field_ref( scalarDissTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
MultiEnvMixingModel<FieldT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
MultiEnvMixingModel<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  typedef std::vector<FieldT*> ResultsVec;
  ResultsVec& results = this->get_value_vec();
  const int nEnv = 3;
  const int wSize = 2*nEnv;
  
  const FieldT* sampleField = scalarVar_;  //dummy iterator field
  typename FieldT::const_interior_iterator sampleIterator = sampleField->interior_begin(); 

  typename FieldT::const_interior_iterator mixfracIter = mixFrac_->interior_begin();
  typename FieldT::const_interior_iterator scalarvarIter = scalarVar_->interior_begin();
  typename FieldT::const_interior_iterator scalardissIter = scalarDiss_->interior_begin();
  //loop to set results iterators
  std::vector<typename FieldT::interior_iterator> resultsIterators;
  for ( int i = 0; i < wSize; i++ ) {
    typename FieldT::interior_iterator thisResultsIterator = results[i]->interior_begin();
    resultsIterators.push_back(thisResultsIterator);
  }
  
  while (sampleIterator != sampleField->interior_end() ) {

    if ( *mixfracIter != 1.0 && *mixfracIter != 0.0 ) {
      *resultsIterators[0] = *scalarvarIter / *mixfracIter;
      *resultsIterators[1] = - *scalardissIter / *mixfracIter;
      *resultsIterators[2] = ( *scalarvarIter - *mixfracIter + *mixfracIter * *mixfracIter ) / ( *mixfracIter * *mixfracIter - *mixfracIter );
      *resultsIterators[3] = *scalardissIter / ( *mixfracIter - *mixfracIter * *mixfracIter );
      *resultsIterators[4] = - *scalarvarIter / ( *mixfracIter - 1.0 );
      *resultsIterators[5] = - *scalardissIter / ( 1.0 - *mixfracIter );
      
    } else if ( *mixfracIter == 1.0 ) {
      *resultsIterators[0] = 0.0;
      *resultsIterators[1] = 0.0;
      *resultsIterators[2] = 0.0;
      *resultsIterators[3] = 0.0;
      *resultsIterators[4] = 1.0;
      *resultsIterators[5] = 0.0;
      
    } else if ( *mixfracIter == 0.0 ) {
      *resultsIterators[0] = 1.0;
      *resultsIterators[1] = 0.0;
      *resultsIterators[2] = 0.0;
      *resultsIterators[3] = 0.0;
      *resultsIterators[4] = 0.0;
      *resultsIterators[5] = 0.0;
    }
  
    //increment iterators
    ++sampleIterator;
    ++mixfracIter;
    ++scalarvarIter;
    ++scalardissIter;
    for ( int i = 0; i < wSize; i++ ) {
      resultsIterators[i] += 1;
    }
  }
}

#endif