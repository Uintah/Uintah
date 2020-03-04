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

#ifndef TwoFluidDensityModel_Expr_h
#define TwoFluidDensityModel_Expr_h

#include <expression/Expression.h>

/**
 *  @class TwoFluidModelDensity
 *  @author Josh McConnell
 *  @date February, 2020
 *  @brief computes density as a function of mixture fraction using the following formula
 */

template< typename FieldT>
class TwoFluidDensityfromMixFrac
 : public Expr::Expression<FieldT>
{
  DECLARE_FIELDS( FieldT, f_ )
  const bool isStepFcn_, computedRhodF_;
  const double &rho0_,  &rho1_;
  const double f0_, f1_, slope_, intercept_;

  TwoFluidDensityfromMixFrac( const Expr::Tag& fTag,
                              const double&    rho0,
                              const double&    rho1,
                              const double&    fTransition,
                              const double&    fRange,
                              const bool       computedRhodF )
  : Expr::Expression<FieldT>(),
  isStepFcn_    ( fRange == 0. ),
  computedRhodF_(computedRhodF ),
  rho0_         ( rho0         ),
  rho1_         ( rho1         ),
  f0_           ( fTransition - 0.5*fRange ),
  f1_           ( fTransition + 0.5*fRange ),
  slope_        ( isStepFcn_ ? 0 : (rho1 - rho0)/fRange ),
  intercept_    ( isStepFcn_ ? 0 : rho0 - slope_*f0_    )
{
  this->set_gpu_runnable(true);
  f_ = this->template create_field_request<FieldT>( fTag );
}

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  @brief Build a TwoFluidDensityfromMixFrac expression
     *  @param rhoTag tag to density computed by this expression
     *  @param dRhodFTag tag to \f$\frac{\partial \rho}{\partial f}\f$, computed by this expression
     */
    Builder( const Expr::Tag rhoTag,
             const Expr::Tag dRhodFTag,
             const Expr::Tag fTag,
             const double    rho0,
             const double    rho1,
             const double    fTransition,
             const double    fRange )
    : ExpressionBuilder( Expr::tag_list(rhoTag, dRhodFTag) ),
    fTag_         ( fTag        ),
    rho0_         ( rho0        ),
    rho1_         ( rho1        ),
    fTransition_  ( fTransition ),
    fRange_       ( fRange      ),
    computedRhodF_( true        )
    {}
    
    Builder( const Expr::Tag rhoTag,
             const Expr::Tag fTag,
             const double    rho0,
             const double    rho1,
             const double    fTransition,
             const double    fRange )
    : ExpressionBuilder( Expr::tag_list(rhoTag) ),
    fTag_         ( fTag        ),
    rho0_         ( rho0        ),
    rho1_         ( rho1        ),
    fTransition_  ( fTransition ),
    fRange_       ( fRange      ),
    computedRhodF_( false       )
    {}

    Expr::ExpressionBase* build() const
    {
      return new TwoFluidDensityfromMixFrac<FieldT>( fTag_, rho0_, rho1_, fTransition_, fRange_, computedRhodF_ );
    }

  private:
    const Expr::Tag fTag_;
    const double rho0_,  rho1_, fTransition_, fRange_;
    const bool computedRhodF_;
  };

  void evaluate()
  {
    using namespace SpatialOps;

    typename Expr::Expression<FieldT>::ValVec& results = this->get_value_vec();
    FieldT& rho    = *results[0];

    const FieldT& f = this->f_->field_ref();
    if( isStepFcn_ ){
      rho <<= cond( (f < f0_), rho0_ )
                     ( rho1_         );         
    }
    else{
      rho <<= cond( (f < f0_), rho0_      )
                  ( (f > f1_), rho1_      )
                  ( slope_*f + intercept_ );
    }

    if(computedRhodF_){
      FieldT& dRhodF = *results[1];

      if( isStepFcn_ ){
        // in reality, this has a value of sign(density0-density1)*inf, when f = fTransition, 
        // but that would yield garbage if used in a diffusive varable density problem 
        dRhodF <<= 0.;
      }
      else{
        dRhodF <<= cond( (f < f0_), 0 )
                      ( (f > f1_), 0 )
                      ( slope_ );
      }
    }
  };
};

//===================================================================

template< typename FieldT>
class TwoFluidDensityfromRhoF
 : public Expr::Expression<FieldT>
{
  DECLARE_FIELDS( FieldT, rhoF_ )
  const bool isStepFcn_, computedRhodF_;
  const double &rho0_,  &rho1_;
  const double f0_, f1_, rhoF0_, rhoF1_, slope_, intercept_;

  TwoFluidDensityfromRhoF( const Expr::Tag& rhoFTag,
                           const double&    rho0,
                           const double&    rho1,
                           const double&    fTransition,
                           const double&    fRange,
                           const bool       computedRhodF )
  : Expr::Expression<FieldT>(),
  isStepFcn_    ( fRange == 0. ),
  computedRhodF_(computedRhodF ),
  rho0_         ( rho0         ),
  rho1_         ( rho1         ),
  f0_           ( fTransition - 0.5*fRange ),
  f1_           ( fTransition + 0.5*fRange ),
  rhoF0_        ( rho0*f0_ ),
  rhoF1_        ( rho1*f1_ ),
  slope_        ( isStepFcn_ ? 0 : (rho1 - rho0)/fRange ),
  intercept_    ( isStepFcn_ ? 0 : rho0 - slope_*f0_    )
{
  this->set_gpu_runnable(true);
  rhoF_ = this->template create_field_request<FieldT>( rhoFTag );
}

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  @brief Build a TwoFluidDensityfromRhoF expression
     *  @param rhoTag the tag for the value that this expression computes
     */
    Builder( const Expr::Tag rhoTag,
             const Expr::Tag dRhodFTag,
             const Expr::Tag rhoFTag,
             const double    rho0,
             const double    rho1,
             const double    fTransition,
             const double    fRange )
    : ExpressionBuilder( Expr::tag_list(rhoTag, dRhodFTag) ),
    rhoFTag_      ( rhoFTag     ),
    rho0_         ( rho0        ),
    rho1_         ( rho1        ),
    fTransition_  ( fTransition ),
    fRange_       ( fRange      ),
    computedRhodF_( true        )
    {}

    Builder( const Expr::Tag rhoTag,
             const Expr::Tag rhoFTag,
             const double    rho0,
             const double    rho1,
             const double    fTransition,
             const double    fRange )
    : ExpressionBuilder( Expr::tag_list(rhoTag) ),
    rhoFTag_      ( rhoFTag     ),
    rho0_         ( rho0        ),
    rho1_         ( rho1        ),
    fTransition_  ( fTransition ),
    fRange_       ( fRange      ),
    computedRhodF_( false       )
    {}

    Expr::ExpressionBase* build() const
    {
      return new TwoFluidDensityfromRhoF<FieldT>( rhoFTag_, rho0_, rho1_, fTransition_, fRange_, computedRhodF_ );
    }

  private:
    const Expr::Tag rhoFTag_;
    const double rho0_, rho1_, fTransition_, fRange_;
    const bool computedRhodF_;
  };

  void evaluate()
  {
    using namespace SpatialOps;

    typename Expr::Expression<FieldT>::ValVec& results = this->get_value_vec();
    FieldT& rho    = *results[0];

    const FieldT& rhoF = this->rhoF_->field_ref();

    if( isStepFcn_ ){
      rho <<= cond( (rhoF < rhoF0_), rho0_ )
                    ( rho1_                );
    }
    else{
      rho <<= cond( (rhoF < rhoF0_), rho0_   )
                    ( (rhoF > rhoF1_), rho1_ )
                    ( 0.5*(-intercept_ + sqrt(intercept_*intercept_ + 4*slope_*rhoF) ) + intercept_ );
    }


    if(computedRhodF_){
      FieldT& dRhodF = *results[1];

      if( isStepFcn_ ){
        // in reality, this has a value of sign(density0-density1)*inf, when f = fTransition, 
        // but that would yield garbage if used in a diffusive varable density problem 
        dRhodF <<= 0.;
      }
      else{
        dRhodF <<= cond( (rhoF < rhoF0_), 0 )
                      ( (rhoF > rhoF1_), 0 )
                      ( slope_ );
      }
    }
  };
};

#endif // TwoFluidDensityModel_Expr_h
