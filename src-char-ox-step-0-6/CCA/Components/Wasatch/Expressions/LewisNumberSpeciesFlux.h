/*
 * The MIT License
 *
 * Copyright (c) 2015-2018 The University of Utah
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

#ifndef LewisNumberDiffFlux_Expr_h
#define LewisNumberDiffFlux_Expr_h

#include <expression/Expression.h>

#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/stencil/FVStaggeredOperatorTypes.h>

/**
 *  \class LewisNumberDiffFlux
 *  \author James C. Sutherland
 *  \date May, 2016
 *
 *  Given the Lewis number of a species, this calculates the mass diffusive flux of a species.
 *
 *  The Lewis number is defined as
 *  \f[
 *    Le_i = \frac{Sc_i}{Pr} = \frac{\alpha}{D_i} = \frac{\lambda}{\rho c_p D_i}.
 *  \f]
 *
 *  Consistent with our form for the mass diffusive flux for mixture-averaged
 *  transport, we have
 *  \f[
 *    j_i = -\frac{\rho}{\bar{M}} D_i \nabla (y_i \bar{M})
 *  \f]
 *  where \f$\bar{M}\f$ is the mixture molecular weight.  Substituting, we find
 *  \f[
 *    j_i = -\rho D_i \nabla y_i = -\frac{\lambda \bar{M}}{c_p Le_i} \nabla( y_i \bar{M})
 *  \f]
 *
 *  Here, we enforce that \f$\sum_{i=1}^N j_i = 0\f$ by adjusting the diffusive flux
 *  of each species by \f$\sigma \equiv \sum_{i=1}^N j_i\f$ as
 *  \f[
 *    j_i = -\sigma y_i - \frac{\lambda \bar{M}}{c_p Le_i} \nabla( y_i \bar{M}).
 *  \f]
 *
 *  \tparam FluxT   the type of field for the resulting flux
 */
template< typename FluxT >
class LewisNumberDiffFluxes
 : public Expr::Expression<FluxT>
{
  typedef typename SpatialOps::VolType<FluxT>::VolField ScalarT;
  typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Interpolant,ScalarT,FluxT>::type  InterpOp;
  typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Gradient,   ScalarT,FluxT>::type  GradOp;

  const std::vector<double>& lewisNum_;
  const size_t nspec_;

  DECLARE_VECTOR_OF_FIELDS( ScalarT, yi_ )
  DECLARE_FIELDS( ScalarT, thermCond_, cp_ )
  
  InterpOp* interpOp_;
  GradOp* gradOp_;

  LewisNumberDiffFluxes( const std::vector<double>& lewisNum,
                         const Expr::TagList& yiTags,
                         const Expr::Tag& thermCondTag,
                         const Expr::Tag& cpTag );

public:

  class Builder : public Expr::ExpressionBuilder
  {
    const std::vector<double> lewisNum_;
    const Expr::TagList yiTags_;
    const Expr::Tag thermCondTag_, cpTag_;
  public:
    /**
     *  @brief Build a LewisNumberDiffFlux expression
     *  @param fluxTags the tag for the diffusive flux
     *  @param lewisNums the Lewis number for each species
     *  @param specMassFracTags the species mass fractions, \f$y_i\f$
     *  @param thermCondTag the thermal conductivity
     *  @param cpTag the isobaric heat capacity
     *  @param nghost the number of ghost cells to compute on
     */
    Builder( const Expr::TagList& fluxTags,
             const std::vector<double>& lewisNum,
             const Expr::TagList& specMassFracTags,
             const Expr::Tag& thermCondTag,
             const Expr::Tag& cpTag,
             const int nghost = DEFAULT_NUMBER_OF_GHOSTS )
      : ExpressionBuilder( fluxTags, nghost ),
        lewisNum_( lewisNum ),
        yiTags_( specMassFracTags ),
        thermCondTag_( thermCondTag ),
        cpTag_( cpTag )
    {}

    Expr::ExpressionBase* build() const{
      return new LewisNumberDiffFluxes<FluxT>( lewisNum_, yiTags_, thermCondTag_, cpTag_ );
    }

  };  /* end of Builder class */

  ~LewisNumberDiffFluxes();
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
};



// ###################################################################
//
//                          Implementation
//
// ###################################################################



template< typename FluxT >
LewisNumberDiffFluxes<FluxT>::
LewisNumberDiffFluxes( const std::vector<double>& lewisNum,
                       const Expr::TagList& yiTags,
                       const Expr::Tag& thermCondTag,
                       const Expr::Tag& cpTag )
  : Expr::Expression<FluxT>(),
    lewisNum_( lewisNum ),
    nspec_( yiTags.size() )
{
  this->set_gpu_runnable(true);

  this->template create_field_vector_request<ScalarT>( yiTags, yi_ );

  thermCond_ = this->template create_field_request<ScalarT>( thermCondTag );
  cp_        = this->template create_field_request<ScalarT>( cpTag        );
}

//--------------------------------------------------------------------

template< typename FluxT >
LewisNumberDiffFluxes<FluxT>::
~LewisNumberDiffFluxes()
{}

//--------------------------------------------------------------------

template< typename FluxT >
void
LewisNumberDiffFluxes<FluxT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  interpOp_ = opDB.retrieve_operator<InterpOp>();
  gradOp_   = opDB.retrieve_operator<GradOp  >();
}

//--------------------------------------------------------------------

template< typename FluxT >
void
LewisNumberDiffFluxes<FluxT>::
evaluate()
{
  typename Expr::Expression<FluxT>::ValVec& fluxes = this->get_value_vec();

  SpatialOps::SpatFldPtr<FluxT> fluxSum = SpatialOps::SpatialFieldStore::get<FluxT>( *(fluxes[0]) );

  for( size_t i=0; i<nspec_; ++i ){
    FluxT& flux = *fluxes[i];
    const ScalarT& yi        = yi_[i]    ->field_ref();
    const ScalarT& thermCond = thermCond_->field_ref();
    const ScalarT& cp        = cp_       ->field_ref();

    flux <<= - (*interpOp_)( thermCond / ( cp * lewisNum_[i] ) ) * (*gradOp_)( yi );

    if( i==0 ) *fluxSum <<= flux;
    else       *fluxSum <<= *fluxSum + flux;
  }

  // enforce that the sum of fluxes equal 0
  for( size_t i=0; i<nspec_; ++i ){
    const ScalarT& yi = yi_[i]->field_ref();
    FluxT& flux = *fluxes[i];
    flux <<= flux - *fluxSum * (*interpOp_) ( yi );
  }
}

//--------------------------------------------------------------------

#endif // LewisNumberDiffFlux_Expr_h
