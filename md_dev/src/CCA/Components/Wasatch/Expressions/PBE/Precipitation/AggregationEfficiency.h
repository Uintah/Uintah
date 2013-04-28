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

#ifndef AggregationEfficiency_Expr_h
#define AggregationEfficiency_Expr_h

#include <expression/Expression.h>


/**
 *  \class AggregationEfficiency
 *  \author Alex Abboud
 *  \date March 2013
 *
 *  \brief Implementation of the aggregation effciency term in liquid-particulate system
 *  this is a size dependent coefficient with one value for each absicassae combination as \f$ \psi = m1/(1+m1) \f$
 *  with \f$ m1 = L * G(r_i) / \rho \bar{d}^2 \epsilon \f$, \f$ m1 \f$ is specific to each particle combination
 *  \f$ L \f$ is a physical property, \f$ G(r_i) \f$ is the growth rate specific to that particl size, 
 *  \f$ \bar{d}^2 \f$ is the average particle size of that collision and $\f epislon \f$ is the energy dissipation
 */
template< typename FieldT >
class AggregationEfficiency
: public Expr::Expression<FieldT>
{
  const Expr::TagList efficiencyTagList_; //tags of efficiencies, nEnv^2
  const Expr::TagList abscissaeTagList_;  // these are the tags of all abscissae
  const Expr::Tag growthCoefTag_;         //coefficient tag for growth
  const Expr::Tag dissipationTag_;        //energy dissipation tag
  const Expr::Tag densityTag_;            //fluid density tag

  const double lengthParam_;              //parameter for scaling the efficiency model and matching units 
  const std::string growthModel_;         //string with type of growth rate model to use 
  
  typedef std::vector<const FieldT*> FieldVec;
  FieldVec abscissae_;
  const FieldT* growthCoef_;
  const FieldT* dissipation_;
  const FieldT* density_;
  
  AggregationEfficiency(const Expr::TagList& abscissaeTagList,
                        const Expr::Tag& growthCoefTag_,
                        const Expr::Tag& dissipationTag_,
                        const Expr::Tag& densityTag_,
                        const double lengthParam_,
                        const std::string growthModel_);
  
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::TagList& result,
             const Expr::TagList& abscissaeTagList,
             const Expr::Tag& growthCoefTag,
             const Expr::Tag& dissipationTag,
             const Expr::Tag& densityTag,
             const double lengthParam,
             const std::string growthModel)
    : ExpressionBuilder(result),
    abscissaetaglist_(abscissaeTagList),
    growthcoeft_(growthCoefTag),
    dissipationt_(dissipationTag),
    densityt_(densityTag),
    lengthparam_(lengthParam),
    growthmodel_(growthModel)
    {}
    ~Builder(){}
    Expr::ExpressionBase* build() const
    {
      return new AggregationEfficiency<FieldT>(abscissaetaglist_, growthcoeft_, dissipationt_, densityt_,  lengthparam_, growthmodel_ );
    }
    
  private:
    const Expr::TagList abscissaetaglist_; // these are the tags of all the known moments
    const Expr::Tag growthcoeft_;
    const Expr::Tag dissipationt_;
    const Expr::Tag densityt_;
    const double lengthparam_;
    const std::string growthmodel_;
  };
  
  ~AggregationEfficiency();
  
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
AggregationEfficiency<FieldT>::
AggregationEfficiency( const Expr::TagList& abscissaeTagList,
                       const Expr::Tag& growthCoefTag,
                       const Expr::Tag& dissipationTag,
                       const Expr::Tag& densityTag,
                       const double lengthParam,
                       const std::string growthModel)
: Expr::Expression<FieldT>(),
abscissaeTagList_(abscissaeTagList),
growthCoefTag_(growthCoefTag),
dissipationTag_(dissipationTag),
densityTag_(densityTag),
lengthParam_(lengthParam),
growthModel_(growthModel)
{}

//--------------------------------------------------------------------

template< typename FieldT >
AggregationEfficiency<FieldT>::
~AggregationEfficiency()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
AggregationEfficiency<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( abscissaeTagList_ );
  exprDeps.requires_expression( growthCoefTag_ );
  exprDeps.requires_expression( dissipationTag_ );
  exprDeps.requires_expression( densityTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
AggregationEfficiency<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const typename Expr::FieldMgrSelector<FieldT>::type& volfm = fml.template field_manager<FieldT>();
  abscissae_.clear();
  for (Expr::TagList::const_iterator iabscissa=abscissaeTagList_.begin(); iabscissa!=abscissaeTagList_.end(); iabscissa++) {
    abscissae_.push_back(&volfm.field_ref(*iabscissa));
  }
  growthCoef_ = &volfm.field_ref( growthCoefTag_) ;
  dissipation_ = &volfm.field_ref( dissipationTag_ );
  density_ = &volfm.field_ref( densityTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
AggregationEfficiency<FieldT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
AggregationEfficiency<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  typedef std::vector<FieldT*> ResultsVec;
  ResultsVec& result = this->get_value_vec();
  
  int nEnv = abscissae_.size();
  int nEff = nEnv*nEnv;
  
  std::vector<typename FieldT::interior_iterator> resultsIter;
  for ( int i =0; i<nEff; ++i) {
    typename FieldT::interior_iterator thisResultsIterator = result[i]->interior_begin();
    resultsIter.push_back(thisResultsIterator);
  }
  
  const FieldT* sampleField = abscissae_[0];
  typename FieldT::const_interior_iterator sampleIterator = sampleField->interior_begin();
  
  std::vector<typename FieldT::const_interior_iterator> abscissaeIterators;
  for (int i=0; i<nEnv; ++i) {
    typename FieldT::const_interior_iterator thisIterator = abscissae_[i]->interior_begin();
    abscissaeIterators.push_back(thisIterator);
  }

  typename FieldT::const_interior_iterator growthCoefIter = growthCoef_->interior_begin();
  typename FieldT::const_interior_iterator densityIter = density_->interior_begin();
  typename FieldT::const_interior_iterator dissipationIter = dissipation_->interior_begin();
  
  double m1;
  int index;
  if (growthModel_ == "BULK_DIFFUSION") {
    while ( sampleIterator!=sampleField->interior_end() ) {
      index = 0;
      for (int i=0; i<nEnv; i++) {
        for (int j =0 ; j<nEnv; j++) {
          if (*densityIter > 0.0 && *dissipationIter > 0.0 ) {
            if (*abscissaeIterators[i] > *abscissaeIterators[j] ) {
              m1 = lengthParam_ * *growthCoefIter / (*abscissaeIterators[i] * *densityIter * (*abscissaeIterators[i] + *abscissaeIterators[j]) 
                                                     * (*abscissaeIterators[i] + *abscissaeIterators[j]) * *dissipationIter);
            } else {
              m1 = lengthParam_ * *growthCoefIter / (*abscissaeIterators[j] * *densityIter * (*abscissaeIterators[i] + *abscissaeIterators[j]) 
                                                     * (*abscissaeIterators[i] + *abscissaeIterators[j]) * *dissipationIter);
            }
          } else {
            m1 = 0.0;
          }
          *resultsIter[index] = m1/(1.0+m1);
          index++;
        }
      }
      ++sampleIterator;
      ++growthCoefIter;
      ++dissipationIter;
      ++densityIter;
      for (int i=0; i<nEnv; i++ ) {
        abscissaeIterators[i] += 1;
      }
      for (int i=0; i<nEff; i++) {
        resultsIter[i] += 1;
      }
    }  
    
  } else if (growthModel_ == "MONOSURFACE") {
    while ( sampleIterator!=sampleField->interior_end() ) {
      index = 0;
      for (int i=0; i<nEnv; i++) {
        for (int j =0 ; j<nEnv; j++) {
          if (*densityIter > 0.0 && *dissipationIter > 0.0 ) {
            if (*abscissaeIterators[i] > *abscissaeIterators[j] ) {
              m1 = lengthParam_ * *growthCoefIter / (*abscissaeIterators[i] * *densityIter * (*abscissaeIterators[i] + *abscissaeIterators[j]) 
                                                     * (*abscissaeIterators[i] + *abscissaeIterators[j]) * *dissipationIter);
            } else {
              m1 = lengthParam_ * *growthCoefIter / (*abscissaeIterators[j] * *densityIter * (*abscissaeIterators[i] + *abscissaeIterators[j]) 
                                                     * (*abscissaeIterators[i] + *abscissaeIterators[j]) * *dissipationIter);
            }
          } else {
            m1 = 0.0;
          }
          *resultsIter[index] = m1/(1.0+m1);
          index++;
        }
      }
      ++sampleIterator;
      ++growthCoefIter;
      ++dissipationIter;
      ++densityIter;
      for (int i=0; i<nEnv; i++ ) {
        abscissaeIterators[i] += 1;
      }
      for (int i=0; i<nEff; i++) {
        resultsIter[i] += 1;
      }
    } 
    
  } else if (growthModel_ == "CONSTANT" || growthModel_ == "KINETIC") {
    index = 0;
    while ( sampleIterator!=sampleField->interior_end() ) {
      for (int i=0; i<nEnv; i++) {
        for (int j =0 ; j<nEnv; j++) {
          if (*densityIter > 0.0 && *dissipationIter > 0.0 ) {
            m1 = lengthParam_ * *growthCoefIter  / ( *densityIter * (*abscissaeIterators[i] + *abscissaeIterators[j])  * (*abscissaeIterators[i] + *abscissaeIterators[j]) * *dissipationIter);
          } else {
            m1 = 0.0;
          }
          *resultsIter[index] = m1/(1.0+m1);
          index++;
        }
      }
      ++sampleIterator;
      ++growthCoefIter;
      ++dissipationIter;
      ++densityIter;
      for (int i=0; i<nEnv; i++ ) {
        abscissaeIterators[i] += 1;
      }
      for (int i=0; i<nEff; i++) {
        resultsIter[i] += 1;
      }
    }
  }
}

#endif // AggregationEfficiency_Expr_h

