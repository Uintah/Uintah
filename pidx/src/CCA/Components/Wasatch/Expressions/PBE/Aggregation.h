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
#include <spatialops/structured/FVStaggeredFieldTypes.h>
#include <spatialops/structured/FVStaggeredOperatorTypes.h>
#include <spatialops/FieldExpressionsExtended.h>

/**
 *  \class Aggregation
 *  \author Alex Abboud
 *  \date June 2012
 *
 *  \brief Implementation of the aggregation term in particulate system
 *  This term has both a death and birth component; in terms of QMOM this is
 *  \f$ (1/2 \sum_i w_i \sum_j w_j ( r_i^3 + r_j^3)^{k/3} \beta_{ij} - \sum_i r_i^k w_i \sum_j \beta_{ij} w_j) * \alpha \f$
 *  where k is the moment order and r & w are the abscissae and weights, and \f$ \beta_{ij} \f$ is the frequency based on model
 *  \f$ \alpha \f$ is an efficiency from an expression, or times a cosntant efficiency
 */
template< typename FieldT >
class Aggregation
: public Expr::Expression<FieldT>
{
  
  const Expr::TagList weightsTagList_; // these are the tags of all weights
  const Expr::TagList abscissaeTagList_; // these are the tags of all abscissae
  const Expr::Tag aggCoefTag_;    //optional coefficent which contaisn fluid properties
  const double momentOrder_;      // order of this moment
  const double effCoef_;          //efficiency coefficient of frequency
  const std::string aggModel_;    //strign with type of aggregation model to use for fequency
  
  typedef std::vector<const FieldT*> FieldVec;
  FieldVec weights_;
  FieldVec abscissae_;
  const FieldT* aggCoef_;
  
  Aggregation( const Expr::TagList& weightsTagList,
               const Expr::TagList& abscissaeTagList,
               const Expr::Tag& aggCoefTag,
               const double momentOrder,
               const double effCoef,
               const std::string aggModel);
  
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& result,
             const Expr::TagList& weightsTagList,
             const Expr::TagList& abscissaeTagList,
             const Expr::Tag& aggCoefTag,
             const double momentOrder,
             const double effCoef,
             const std::string aggModel)
    : ExpressionBuilder(result),
    weightstaglist_(weightsTagList),
    abscissaetaglist_(abscissaeTagList),
    aggcoeft_(aggCoefTag),
    momentorder_(momentOrder),
    effcoef_(effCoef),
    aggmodel_(aggModel)
    {}
    ~Builder(){}
    Expr::ExpressionBase* build() const
    {
      return new Aggregation<FieldT>( weightstaglist_,abscissaetaglist_, aggcoeft_, momentorder_, effcoef_, aggmodel_ );
    }
    
  private:
    const Expr::TagList weightstaglist_; // these are the tags of all the known moments
    const Expr::TagList abscissaetaglist_; // these are the tags of all the known moments
    const Expr::Tag aggcoeft_;
    const double momentorder_;
    const double effcoef_;
    const std::string aggmodel_;
  };
  
  ~Aggregation();
  
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
Aggregation<FieldT>::
Aggregation( const Expr::TagList& weightsTagList,
             const Expr::TagList& abscissaeTagList,
             const Expr::Tag& aggCoefTag,
             const double momentOrder,
             const double effCoef,
             const std::string aggModel)
: Expr::Expression<FieldT>(),
weightsTagList_(weightsTagList),
abscissaeTagList_(abscissaeTagList),
aggCoefTag_(aggCoefTag),
momentOrder_(momentOrder),
effCoef_(effCoef),
aggModel_(aggModel)
{}

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
  for (Expr::TagList::const_iterator iweight=weightsTagList_.begin(); iweight!=weightsTagList_.end(); iweight++) {
    weights_.push_back(&volfm.field_ref(*iweight));
  }
  for (Expr::TagList::const_iterator iabscissa=abscissaeTagList_.begin(); iabscissa!=abscissaeTagList_.end(); iabscissa++) {
    abscissae_.push_back(&volfm.field_ref(*iabscissa));
  }
  if ( aggCoefTag_ != Expr::Tag() )
    aggCoef_ = &volfm.field_ref(aggCoefTag_) ;
}

//--------------------------------------------------------------------

template< typename FieldT >
void
Aggregation<FieldT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
Aggregation<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  result <<= 0.0;
  
  typename FieldT::interior_iterator resultsIter = result.interior_begin();
  int nEnv = weights_.size();

  const FieldT* sampleField = weights_[0];
  typename FieldT::const_interior_iterator sampleIterator = sampleField->interior_begin();
  
  std::vector<typename FieldT::const_interior_iterator> weightIterators;
  std::vector<typename FieldT::const_interior_iterator> abscissaeIterators;
  for (int i=0; i<nEnv; ++i) {
    typename FieldT::const_interior_iterator thisIterator = weights_[i]->interior_begin();
    weightIterators.push_back(thisIterator);
    
    typename FieldT::const_interior_iterator otherIterator = abscissae_[i]->interior_begin();
    abscissaeIterators.push_back(otherIterator);
  }
  double Sum;
  
  if (aggModel_ == "CONSTANT") {  // \beta_{ij} = 1
    while ( sampleIterator!=sampleField->interior_end() ) {
      Sum = 0.0;
      
      for (int i=0; i<nEnv; i++) {
        for (int j =0 ; j<nEnv; j++) {
          Sum = Sum + 0.5 * *weightIterators[i] * *weightIterators[j] * 
                            pow( *abscissaeIterators[i] * *abscissaeIterators[i] * *abscissaeIterators[i] +
                            *abscissaeIterators[j] * *abscissaeIterators[j] * *abscissaeIterators[j], momentOrder_/3.0); //birth term
          Sum = Sum - pow( *abscissaeIterators[i] , momentOrder_ ) * *weightIterators[i] * *weightIterators[j]; //death term
        }
      }
      *resultsIter = effCoef_ * Sum;
      
      ++sampleIterator;
      ++resultsIter;
      for (int i=0; i<nEnv; i++ ) {
        weightIterators[i] += 1;
        abscissaeIterators[i] += 1;
      }
    }  
  } else if (aggModel_ == "BROWNIAN") {  // \beta_{ij} = (r_i + r_j)^2 / r_i / r_j
    while ( sampleIterator!=sampleField->interior_end() ) {
      Sum = 0.0;
      
      for (int i=0; i<nEnv; i++) {
        for (int j =0 ; j<nEnv; j++) {
          Sum = Sum + 0.5 * *weightIterators[i] * *weightIterators[j] * 
                pow( *abscissaeIterators[i] * *abscissaeIterators[i] * *abscissaeIterators[i] +
                *abscissaeIterators[j] * *abscissaeIterators[j] * *abscissaeIterators[j], momentOrder_/3.0) *
                (*abscissaeIterators[i] + *abscissaeIterators[j]) * (*abscissaeIterators[i] + *abscissaeIterators[j]) 
                 / *abscissaeIterators[i] / *abscissaeIterators[j]; //birth term
          Sum = Sum - pow( *abscissaeIterators[i] , momentOrder_ ) * *weightIterators[i] * *weightIterators[j] *
                (*abscissaeIterators[i] + *abscissaeIterators[j]) * (*abscissaeIterators[i] + *abscissaeIterators[j]) 
                / *abscissaeIterators[i] / *abscissaeIterators[j]; //death term
        }
      }
      *resultsIter = effCoef_ * Sum;
      
      ++sampleIterator;
      ++resultsIter;
      for (int i=0; i<nEnv; i++ ) {
        weightIterators[i] += 1;
        abscissaeIterators[i] += 1;
      }
    }
  } else if (aggModel_ == "HYDRODYNAMIC" ) { // \beta_{ij} = (r_i + r_j)^3
    while ( sampleIterator!=sampleField->interior_end() ) {
      Sum = 0.0;
      
      for (int i=0; i<nEnv; i++) {
        for (int j =0 ; j<nEnv; j++) {
          Sum = Sum + 0.5 * *weightIterators[i] * *weightIterators[j] * 
                pow( *abscissaeIterators[i] * *abscissaeIterators[i] * *abscissaeIterators[i] +
                *abscissaeIterators[j] * *abscissaeIterators[j] * *abscissaeIterators[j], momentOrder_/3.0) *
                (*abscissaeIterators[i] + *abscissaeIterators[j]) * (*abscissaeIterators[i] + *abscissaeIterators[j]) 
                * ( *abscissaeIterators[i] + *abscissaeIterators[j]); //birth term
          Sum = Sum - pow( *abscissaeIterators[i] , momentOrder_ ) * *weightIterators[i] * *weightIterators[j] *
                (*abscissaeIterators[i] + *abscissaeIterators[j]) * (*abscissaeIterators[i] + *abscissaeIterators[j]) 
                * ( *abscissaeIterators[i] + *abscissaeIterators[j]); //death term
        }
      }
      *resultsIter = effCoef_ * Sum;
      
      ++sampleIterator;
      ++resultsIter;
      for (int i=0; i<nEnv; ++i ) {
        weightIterators[i] += 1;
        abscissaeIterators[i] += 1;
      }
    }
  }
  
  if ( aggCoefTag_ != Expr::Tag () ) 
    result <<= result * *aggCoef_;
}

#endif // Aggregation_Expr_h

