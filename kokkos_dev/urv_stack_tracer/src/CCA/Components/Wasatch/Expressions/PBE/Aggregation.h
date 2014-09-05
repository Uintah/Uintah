/*
 * The MIT License
 *
 * Copyright (c) 2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef Aggregation_Expr_h
#define Aggregation_Expr_h

#include <expression/Expression.h>

#define ri *abscissae_[i]
#define rj *abscissae_[j]
#define wi *weights_[i]
#define wj *weights_[j]

/**
 *  \class Aggregation
 *  \authors Alex Abboud, Tony Saad
 *  \date June 2012
 *
 *  \brief Implementation of the aggregation term in particulate system
 *  This term has both a death and birth component; in terms of QMOM this is
 *  \f$ (1/2 \sum_i w_i \sum_j w_j ( r_i^3 + r_j^3)^{k/3} \beta_{ij} - \sum_i r_i^k w_i \sum_j \beta_{ij} w_j) * \alpha \f$
 *  where k is the moment order and r & w are the abscissae and weights, and \f$ \beta_{ij} \f$ is the frequency based on model
 *  \f$ \alpha \f$ is an efficiency from an expression, or times a cosntant efficiency
 *  the efficiency can also be set to be dependent on the particle sizes of the collision
 */
template< typename FieldT >
class Aggregation
: public Expr::Expression<FieldT>
{
  
  const Expr::TagList weightsTagList_; // these are the tags of all weights
  const Expr::TagList abscissaeTagList_; // these are the tags of all abscissae
  const Expr::TagList efficiencyTagList_; //tags for collison efficiencies
  const Expr::Tag aggCoefTag_;    //optional coefficent which contaisn fluid properties
  const double momentOrder_;      // order of this moment
  const double effCoef_;          //efficiency coefficient of frequency
  const std::string aggModel_;    //strign with type of aggregation model to use for fequency
  const bool useEffTags_;         //boolean to use efficiency tags
  
  typedef std::vector<const FieldT*> FieldVec;
  FieldVec weights_;
  FieldVec abscissae_;
  FieldVec efficiency_;
  const FieldT* aggCoef_;
  
  Aggregation( const Expr::TagList& weightsTagList,
               const Expr::TagList& abscissaeTagList,
               const Expr::TagList& efficiencyTagList,
               const Expr::Tag& aggCoefTag,
               const double momentOrder,
               const double effCoef,
               const std::string aggModel,
               const bool useEffTags);
  
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& result,
             const Expr::TagList& weightsTagList,
             const Expr::TagList& abscissaeTagList,
             const Expr::TagList& efficiencyTagList,
             const Expr::Tag& aggCoefTag,
             const double momentOrder,
             const double effCoef,
             const std::string aggModel,
             const bool useEffTags)
    : ExpressionBuilder(result),
    weightstaglist_(weightsTagList),
    abscissaetaglist_(abscissaeTagList),
    efficiencytaglist_(efficiencyTagList),
    aggcoeft_(aggCoefTag),
    momentorder_(momentOrder),
    effcoef_(effCoef),
    aggmodel_(aggModel),
    useefftags_(useEffTags)
    {}
    ~Builder(){}
    Expr::ExpressionBase* build() const
    {
      return new Aggregation<FieldT>( weightstaglist_,abscissaetaglist_, efficiencytaglist_, aggcoeft_, momentorder_, effcoef_, aggmodel_, useefftags_ );
    }
    
  private:
    const Expr::TagList weightstaglist_; // these are the tags of all the known moments
    const Expr::TagList abscissaetaglist_; // these are the tags of all the known moments
    const Expr::TagList efficiencytaglist_; 
    const Expr::Tag aggcoeft_;
    const double momentorder_;
    const double effcoef_;
    const std::string aggmodel_;
    const bool useefftags_;
  };
  
  ~Aggregation();
  
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
Aggregation<FieldT>::
Aggregation( const Expr::TagList& weightsTagList,
             const Expr::TagList& abscissaeTagList,
             const Expr::TagList& efficiencyTagList,
             const Expr::Tag& aggCoefTag,
             const double momentOrder,
             const double effCoef,
             const std::string aggModel,
             const bool useEffTags)
: Expr::Expression<FieldT>(),
  weightsTagList_(weightsTagList),
  abscissaeTagList_(abscissaeTagList),
  efficiencyTagList_(efficiencyTagList),
  aggCoefTag_(aggCoefTag),
  momentOrder_(momentOrder),
  effCoef_(effCoef),
  aggModel_(aggModel),
  useEffTags_(useEffTags)
{
  this->set_gpu_runnable( true );
}

//--------------------------------------------------------------------

template< typename FieldT >
Aggregation<FieldT>::
~Aggregation()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
Aggregation<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( weightsTagList_ );
  exprDeps.requires_expression( abscissaeTagList_ );
  if ( aggCoefTag_ != Expr::Tag () ) 
    exprDeps.requires_expression( aggCoefTag_ );
  if ( useEffTags_ )
    exprDeps.requires_expression( efficiencyTagList_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
Aggregation<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const typename Expr::FieldMgrSelector<FieldT>::type& volfm = fml.template field_manager<FieldT>();
  weights_.clear();
  abscissae_.clear();
  efficiency_.clear();
  for (Expr::TagList::const_iterator iweight=weightsTagList_.begin(); iweight!=weightsTagList_.end(); iweight++) {
    weights_.push_back(&volfm.field_ref(*iweight));
  }
  for (Expr::TagList::const_iterator iabscissa=abscissaeTagList_.begin(); iabscissa!=abscissaeTagList_.end(); iabscissa++) {
    abscissae_.push_back(&volfm.field_ref(*iabscissa));
  }
  if ( aggCoefTag_ != Expr::Tag() )
    aggCoef_ = &volfm.field_ref(aggCoefTag_) ;
  if (useEffTags_) {
    for (Expr::TagList::const_iterator iefficiency=efficiencyTagList_.begin(); iefficiency!=efficiencyTagList_.end(); iefficiency++) {
      efficiency_.push_back(&volfm.field_ref(*iefficiency));
    }
  }
}

//--------------------------------------------------------------------

template< typename FieldT >
void
Aggregation<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  result <<= 0.0;

  SpatFldPtr<FieldT> tmp = SpatialFieldStore::get<FieldT>( result );

  int nEnv = weights_.size();
  
  if (aggModel_ == "CONSTANT") {  // \beta_{ij} = 1
    for (int i=0; i<nEnv; i++) {
      for (int j =0 ; j<nEnv; j++) {
        *tmp <<= 0.5 * wi*wj * pow( ri*ri*ri + rj*rj*rj, momentOrder_/3.0 ) - pow( ri , momentOrder_ ) * wi*wj;
        if (useEffTags_) *tmp <<= *efficiency_[i*nEnv  + j] * *tmp;
        result <<= result + *tmp;
      }
    }
  } else if (aggModel_ == "BROWNIAN") {  // \beta_{ij} = (r_i + r_j)^2 / r_i / r_j
    for (int i=0; i<nEnv; i++) {
      for (int j =0 ; j<nEnv; j++) {
        *tmp <<= 0.5 * wi*wj * pow( ri*ri*ri + rj*rj*rj, momentOrder_/3.0 )* (ri+rj) * (ri+rj) / ri / rj - pow( ri , momentOrder_ ) * wi*wj * (ri+rj) * (ri+rj) /ri / rj;
        if (useEffTags_) *tmp <<= *efficiency_[i*nEnv  + j] * *tmp;
        result <<= result + *tmp;
      }
    }
  } else if (aggModel_ == "HYDRODYNAMIC" ) { // \beta_{ij} = (r_i + r_j)^3
    for (int i=0; i<nEnv; i++) {
      for (int j =0 ; j<nEnv; j++) {
        *tmp <<= 0.5 * wi*wj * pow( ri*ri*ri + rj*rj*rj, momentOrder_/3.0 )* (ri+rj) * (ri+rj)* (ri+rj) - pow( ri , momentOrder_ ) * wi*wj* (ri+rj) * (ri+rj)* (ri+rj);
        if (useEffTags_) *tmp <<= *efficiency_[i*nEnv  + j] * *tmp;
        result <<= result + *tmp;
      }
    }
  }

  result <<= effCoef_ * result;

  if ( aggCoefTag_ != Expr::Tag () )
    result <<= result * *aggCoef_;  
}

#endif // Aggregation_Expr_h

