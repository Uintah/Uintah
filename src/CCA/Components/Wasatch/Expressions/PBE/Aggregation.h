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
  if (useEffTags_) {
    int nEff = efficiency_.size();
    int index;
    std::vector<typename FieldT::const_interior_iterator> efficiencyIterators;
    for (int i = 0; i<nEff; ++i) {
      typename FieldT::const_interior_iterator thisIterator = efficiency_[i]->interior_begin();  
      efficiencyIterators.push_back(thisIterator);
    }
    
    if (aggModel_ == "CONSTANT") {  // \beta_{ij} = 1
      while ( sampleIterator!=sampleField->interior_end() ) {
        Sum = 0.0;
        index = 0;
        for (int i=0; i<nEnv; i++) {
          for (int j =0 ; j<nEnv; j++) {
            Sum = Sum + *efficiencyIterators[index]* 0.5 * *weightIterators[i] * *weightIterators[j] * 
                        pow( *abscissaeIterators[i] * *abscissaeIterators[i] * *abscissaeIterators[i] +
                        *abscissaeIterators[j] * *abscissaeIterators[j] * *abscissaeIterators[j], momentOrder_/3.0); //birth term
            Sum = Sum - *efficiencyIterators[index]* pow( *abscissaeIterators[i] , momentOrder_ ) * *weightIterators[i] * *weightIterators[j]; //death term
            index++;
          }
        }
        *resultsIter = effCoef_ * Sum;
        
        ++sampleIterator;
        ++resultsIter;
        for (int i=0; i<nEnv; i++ ) {
          weightIterators[i] += 1;
          abscissaeIterators[i] += 1;
        }
        for (int i = 0; i<nEff; i++) {
          efficiencyIterators[i] +=1; 
        }
      }  
    } else if (aggModel_ == "BROWNIAN") {  // \beta_{ij} = (r_i + r_j)^2 / r_i / r_j
      while ( sampleIterator!=sampleField->interior_end() ) {
        Sum = 0.0;
        index = 0;
        for (int i=0; i<nEnv; i++) {
          for (int j =0 ; j<nEnv; j++) {
            Sum = Sum + *efficiencyIterators[index]* 0.5 * *weightIterators[i] * *weightIterators[j] * 
                        pow( *abscissaeIterators[i] * *abscissaeIterators[i] * *abscissaeIterators[i] +
                        *abscissaeIterators[j] * *abscissaeIterators[j] * *abscissaeIterators[j], momentOrder_/3.0) *
                        (*abscissaeIterators[i] + *abscissaeIterators[j]) * (*abscissaeIterators[i] + *abscissaeIterators[j]) 
                        / *abscissaeIterators[i] / *abscissaeIterators[j]; //birth term
            Sum = Sum - *efficiencyIterators[index]* pow( *abscissaeIterators[i] , momentOrder_ ) * *weightIterators[i] * *weightIterators[j] *
                        (*abscissaeIterators[i] + *abscissaeIterators[j]) * (*abscissaeIterators[i] + *abscissaeIterators[j]) 
                        / *abscissaeIterators[i] / *abscissaeIterators[j]; //death term
            index++;
          }
        }
        *resultsIter = effCoef_ * Sum;
        
        ++sampleIterator;
        ++resultsIter;
        for (int i=0; i<nEnv; i++ ) {
          weightIterators[i] += 1;
          abscissaeIterators[i] += 1;
        }
        for (int i = 0; i<nEff; i++) {
          efficiencyIterators[i] +=1; 
        }
      }
    } else if (aggModel_ == "HYDRODYNAMIC" ) { // \beta_{ij} = (r_i + r_j)^3
      while ( sampleIterator!=sampleField->interior_end() ) {
        Sum = 0.0;
        index = 0;
        for (int i=0; i<nEnv; i++) {
          for (int j =0 ; j<nEnv; j++) {
            Sum = Sum + *efficiencyIterators[index]* 0.5 * *weightIterators[i] * *weightIterators[j] * 
                        pow( *abscissaeIterators[i] * *abscissaeIterators[i] * *abscissaeIterators[i] +
                        *abscissaeIterators[j] * *abscissaeIterators[j] * *abscissaeIterators[j], momentOrder_/3.0) *
                        (*abscissaeIterators[i] + *abscissaeIterators[j]) * (*abscissaeIterators[i] + *abscissaeIterators[j]) 
                        * ( *abscissaeIterators[i] + *abscissaeIterators[j]); //birth term
            Sum = Sum - *efficiencyIterators[index]* pow( *abscissaeIterators[i] , momentOrder_ ) * *weightIterators[i] * *weightIterators[j] *
                        (*abscissaeIterators[i] + *abscissaeIterators[j]) * (*abscissaeIterators[i] + *abscissaeIterators[j]) 
                        * ( *abscissaeIterators[i] + *abscissaeIterators[j]); //death term
            index++;
          }
        }
        *resultsIter = effCoef_ * Sum;
        
        ++sampleIterator;
        ++resultsIter;
        for (int i=0; i<nEnv; ++i ) {
          weightIterators[i] += 1;
          abscissaeIterators[i] += 1;
        }
        for (int i = 0; i<nEff; i++) {
          efficiencyIterators[i] +=1; 
        }
      }
    }
    
  } else {
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
  }  
  
  if ( aggCoefTag_ != Expr::Tag () ) 
    result <<= result * *aggCoef_;  
}

#endif // Aggregation_Expr_h

