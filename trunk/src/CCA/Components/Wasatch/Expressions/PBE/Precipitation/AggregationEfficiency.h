/*
 * The MIT License
 *
 * Copyright (c) 2012-2018 The University of Utah
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
 *  \authors Alex Abboud, Tony Saad
 *  \date March 2013
 *
 *  \brief Implementation of the aggregation effciency term in liquid-particulate system
 *  this is a size dependent coefficient with one value for each absicassae combination as \f$ \psi = m1/(1+m1) \f$
 *  with \f$ m1 = L * G(r_i) / \rho \bar{d}^2 \epsilon \f$, \f$ m1 \f$ is specific to each particle combination
 *  \f$ L \f$ is a physical property, \f$ G(r_i) \f$ is the growth rate specific to that particle size,
 *  \f$ \bar{d}^2 \f$ is the average particle size of that collision and \f$ \epsilon \f$ is the energy dissipation
 */
template< typename FieldT >
class AggregationEfficiency
: public Expr::Expression<FieldT>
{
//  const Expr::TagList abscissaeTagList_;  // these are the tags of all abscissae
//  const Expr::Tag growthCoefTag_;         //coefficient tag for growth
//  const Expr::Tag dissipationTag_;        //energy dissipation tag
//  const Expr::Tag densityTag_;            //fluid density tag

  const double l_;                        //parameter for scaling the efficiency model and matching units
  const std::string growthModel_;         //string with type of growth rate model to use 
  
  DECLARE_VECTOR_OF_FIELDS(FieldT, abscissae_)
  DECLARE_FIELDS(FieldT, g0_, eps_, rho_)
  
  AggregationEfficiency(const Expr::TagList& abscissaeTagList,
                        const Expr::Tag& growthCoefTag,
                        const Expr::Tag& dissipationTag,
                        const Expr::Tag& densityTag,
                        const double lengthParam,
                        const std::string growthModel);
  
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
    l_(lengthParam),
    growthmodel_(growthModel)
    {}
    ~Builder(){}
    Expr::ExpressionBase* build() const
    {
      return new AggregationEfficiency<FieldT>(abscissaetaglist_, growthcoeft_, dissipationt_, densityt_,  l_, growthmodel_ );
    }
    
  private:
    const Expr::TagList abscissaetaglist_; // these are the tags of all the known moments
    const Expr::Tag growthcoeft_;
    const Expr::Tag dissipationt_;
    const Expr::Tag densityt_;
    const double l_;
    const std::string growthmodel_;
  };
  
  ~AggregationEfficiency();
  
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
  l_(lengthParam),
  growthModel_(growthModel)
{
  this->set_gpu_runnable( true );
  this->template create_field_vector_request<FieldT>(abscissaeTagList, abscissae_);
   g0_ = this->template create_field_request<FieldT>(growthCoefTag);
   eps_ = this->template create_field_request<FieldT>(dissipationTag);
   rho_ = this->template create_field_request<FieldT>(densityTag);
}

//--------------------------------------------------------------------

template< typename FieldT >
AggregationEfficiency<FieldT>::
~AggregationEfficiency()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
AggregationEfficiency<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  typedef typename Expr::Expression<FieldT>::ValVec ResultsVec;
  ResultsVec& results = this->get_value_vec();
  const FieldT& rho = rho_->field_ref();
  const FieldT& g0  = g0_->field_ref();
  const FieldT& eps = eps_->field_ref();
  SpatialOps::SpatFldPtr<FieldT> tmp = SpatialOps::SpatialFieldStore::get<FieldT>( *results[0] );

  int nEnv = abscissae_.size();

  int idx = 0;

//#define ri abscissae_[i]->field_ref()
//#define rj abscissae_[j]->field_ref()

  if (growthModel_ == "BULK_DIFFUSION") {
    
    for (int i=0; i<nEnv; i++) {
      for (int j =0 ; j<nEnv; j++) {
        *tmp <<= cond( rho > 0.0 && eps > 0.0,
                       cond ( ri > rj, l_ * g0 / (ri * rho * (ri + rj) * (ri + rj) * eps) )
                            ( l_ * g0 / (rj * rho * (ri + rj) * (ri + rj) * eps) ) )
                     ( 0.0 );
        *tmp <<= cond( *tmp > 0.0, *tmp)
                     (0.0); 
        *results[idx++] <<= *tmp/(1.0 + *tmp);
      }
    }
    
  } else if (growthModel_ == "MONOSURFACE") {
    
    for (int i=0; i<nEnv; i++) {
      for (int j =0 ; j<nEnv; j++) {
        *tmp <<= cond( rho > 0.0 && eps > 0.0,
                       cond ( ri > rj, l_ * g0 / (ri * rho * (ri + rj) * (ri + rj) * eps) )
                            ( l_ * g0 / (rj * rho * (ri + rj) * (ri + rj) * eps) ) )
                     ( 0.0 );
        *tmp <<= cond( *tmp > 0.0, *tmp)
                     (0.0);
        *results[idx++] <<= *tmp/(1.0 + *tmp);
      }
    }         
    
  } else if (growthModel_ == "CONSTANT" || growthModel_ == "KINETIC") {
    
    for (int i=0; i<nEnv; i++) {
      for (int j =0 ; j<nEnv; j++) {
        *tmp <<= cond( rho > 0.0 && eps > 0.0, l_ * g0  / ( rho * (ri + rj)  * (ri + rj) * eps) )
                     ( 0.0 );
        *tmp <<= cond( *tmp > 0.0, *tmp)
                     (0.0);
        *results[idx++] <<= *tmp/(1.0 + *tmp);
      }
    }
    
  }
}

#endif // AggregationEfficiency_Expr_h

