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
 *  @class TwoFluidPropertyFromMixFrac
 *  @author Josh McConnell
 *  @date February, 2020
 *  @brief computes a property, \f$\G\f$, as a function of mixture fraction, \f$\f\f$ using the following formula
 *  \f[
 *    G =
 *    \begin{cases}
 *      G_0                                                      & f \leq z_0,
 *      \frac{G_1 - G_0}{f_{range}} \left( f - f_0 \right) + G_0 & f_0 < f < f_1,
 *      G_1                                                      & f \geq f_1, 
 *    \end{cases}
 *  \f]
 * where \f$\f_0 = f_{transition} - 0.5 f_{range}\f$, \f$\f_1 = f_{transition} + 0.5 f_{range}\f$, and
 * \f$f_{transition}, f_{range}\f$ are input parameters.
 */

template< typename FieldT>
class TwoFluidPropertyFromMixFrac
 : public Expr::Expression<FieldT>
{
  DECLARE_FIELDS( FieldT, f_ )
  const bool isStepFcn_, computedGdF_;
  const double &g0_,  &g1_;
  const double f0_, f1_, slope_, intercept_;

  TwoFluidPropertyFromMixFrac( const Expr::Tag& fTag,
                               const double&    g0,
                               const double&    g1,
                               const double&    fTransition,
                               const double&    fRange,
                               const bool       computedGdF )
  : Expr::Expression<FieldT>(),
  isStepFcn_   ( fRange == 0. ),
  computedGdF_ ( computedGdF  ),
  g0_          ( g0           ),
  g1_          ( g1           ),
  f0_          ( fTransition - 0.5*fRange ),
  f1_          ( fTransition + 0.5*fRange ),
  slope_       ( isStepFcn_ ? 0 : (g1 - g0)/fRange ),
  intercept_   ( isStepFcn_ ? 0 : g0 - slope_*f0_  )
{
  this->set_gpu_runnable(true);
  f_ = this->template create_field_request<FieldT>( fTag );
}

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  @brief Build a TwoFluidPropertyFromMixFrac expression
     *  @param gTag tag to property, \f$G\f$, computed by this expression
     *  @param dGdFTag tag to \f$\frac{\partial G}{\partial f}\f$, computed by this expression
     *  @param fTag tag to mixture fraction
     *  @param g0 value of \f$G\f$  when \f$f = f_{transition} - 0.5 f_{range}\f$ 
     *  @param g1 value of \f$G\f$  when \f$f = f_{transition} + 0.5 f_{range}\f$ 
     *  @param fTransition transition value of  value of mixture fraction, \f$f\f$
     *  @param fRange width of mixture fraction range over which \f$G\f$ changes
     */
    Builder( const Expr::Tag gTag,
             const Expr::Tag dGdFTag,
             const Expr::Tag fTag,
             const double    g0,
             const double    g1,
             const double    fTransition,
             const double    fRange )
    : ExpressionBuilder( Expr::tag_list(gTag, dGdFTag) ),
    fTag_       ( fTag        ),
    g0_         ( g0          ),
    g1_         ( g1          ),
    fTransition_( fTransition ),
    fRange_     ( fRange      ),
    computedGdF_( true        )
    {}
    
    Builder( const Expr::Tag gTag,
             const Expr::Tag fTag,
             const double    g0,
             const double    g1,
             const double    fTransition,
             const double    fRange )
    : ExpressionBuilder( Expr::tag_list(gTag) ),
    fTag_       ( fTag        ),
    g0_         ( g0          ),
    g1_         ( g1          ),
    fTransition_( fTransition ),
    fRange_     ( fRange      ),
    computedGdF_( false       )
    {}

    Expr::ExpressionBase* build() const
    {
      return new TwoFluidPropertyFromMixFrac<FieldT>( fTag_, g0_, g1_, fTransition_, fRange_, computedGdF_ );
    }

  private:
    const Expr::Tag fTag_;
    const double g0_,  g1_, fTransition_, fRange_;
    const bool computedGdF_;
  };

  void evaluate()
  {
    using namespace SpatialOps;

    typename Expr::Expression<FieldT>::ValVec& results = this->get_value_vec();
    FieldT& g  = *results[0];

    const FieldT& f = this->f_->field_ref();
    if( isStepFcn_ ){
      g <<= cond( (f < f0_), g0_ )
                ( g1_            );         
    }
    else{
      g <<= cond( (f < f0_), g0_        )
                ( (f > f1_), g1_        )
                ( slope_*f + intercept_ );
    }

    if(computedGdF_){
      FieldT& dGdF = *results[1];

      if( isStepFcn_ ){
        // in reality, this has a value of sign(density0-density1)*inf, when f = fTransition, 
        // but that would yield garbage if used in a diffusive varable density problem 
        dGdF <<= 0.;
      }
      else{
        dGdF <<= cond( (f < f0_), 0 )
                     ( (f > f1_), 0 )
                     ( slope_ );
      }
    }
  };
};

//===================================================================

/**
 *  @class TwoFluidDensityfromRhoF
 *  @author Josh McConnell
 *  @date February, 2020
 *  @brief computes density, \f$\rho\f$ from a density-weighted mixture fraction, \f$\rho f\f$, 
 *  where density has the following form
 *  \f[
 *    G =
 *    \begin{cases}
 *      G_0                                                      & f \leq z_0,
 *      \frac{G_1 - G_0}{f_{range}} \left( f - f_0 \right) + G_0 & f_0 < f < f_1,
 *      G_1                                                      & f \geq f_1, 
 *    \end{cases}
 *  \f]
 * where \f$\f_0 = f_{transition} - 0.5 f_{range}\f$, \f$\f_1 = f_{transition} + 0.5 f_{range}\f$, and
 * \f$f_{transition}, f_{range}\f$ are input parameters.
 */
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
     *  @param rhoTag tag to density, \f$\rho\f$, computed by this expression
     *  @param dRhodFTag tag to \f$\frac{\partial \rho}{\partial f}\f$, computed by this expression
     *  @param rhoFTag tag to density-weighted mixture fraction
     *  @param rho0 value of \f$\rho\f$  when \f$f = f_{transition} - 0.5 f_{range}\f$ 
     *  @param rho1 value of \f$\rho\f$  when \f$f = f_{transition} + 0.5 f_{range}\f$ 
     *  @param fTransition transition value of  value of mixture fraction, \f$f\f$
     *  @param fRange width of mixture fraction range over which \f$\rho\f$ changes
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
