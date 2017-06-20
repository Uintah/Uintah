/*
 * The MIT License
 *
 * Copyright (c) 2015-2017 The University of Utah
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

#ifndef TurbDiffFlux_Expr_h
#define TurbDiffFlux_Expr_h

#include <expression/Expression.h>
#include <spatialops/OperatorDatabase.h>

/**
 *  \ingroup WasatchExpressions
 *  \class  TurbDiffFlux
 *  \author James C. Sutherland
 *  \date   May, 2016
 *
 *  \brief Calculates a generic turbulent diffusive flux, \f$J = -\rho \Gamma_T
 *         \frac{\partial \phi}{\partial x}\f$, where
 *         \f$\Gamma_T = \frac{\mu_{turb}}{\bar{\rho} Sc_{turb}}\f$\f$.
 *
 *  Given turbulent Schmidt number, \f$Sc_{turb}\f$, and turbulent viscosity,
 *  \f$\mu_{turb}\f$, and filtered density, \f$\bar{\rho}\f$, this calculates
 *  the turbulent diffusive flux, \f$D_{turb} = \frac{\mu_{turb}}{\bar{\rho} Sc_{turb}}\f$.
 *
 *  \tparam FluxT the type for the diffusive flux.
 */
template< typename FluxT >
class TurbDiffFlux
 : public Expr::Expression<FluxT>
{
  typedef typename SpatialOps::VolType<FluxT>::VolField ScalarT;

  typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Gradient,   ScalarT,  FluxT>::type  GradT;
  typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Interpolant,ScalarT,  FluxT>::type  InterpT;

  const double scTurb_;
  DECLARE_FIELDS( ScalarT, turbVisc_, phi_ )

  GradT*   gradOp_;
  InterpT* interpOp_;
  
  TurbDiffFlux( const Expr::Tag& turbViscTag,
                const double turbSchmidt,
                const Expr::Tag& phiTag );

public:

  class Builder : public Expr::ExpressionBuilder
  {
    const double turbSc_;
    const Expr::Tag turbViscTag_, phiTag_;
  public:
    /**
     *  @brief Build a TurbDiffFlux expression
     *  @param fluxTag the tag for the turbulent diffusive flux
     *  @param turbViscTag the turbulent viscosity
     *  @param turbSchmidt the turbulent Schmidt number: \f$Sc = \frac{\mu}{\rho D}\f$
     *  @param phiTag the scalar whose turbulent flux we are computing
     *  @param nghost the number of ghost cells to compute
     */
    Builder( const Expr::Tag& fluxTag,
             const Expr::Tag& turbViscTag,
             const double turbSchmidt,
             const Expr::Tag& phiTag,
             const int nghost = DEFAULT_NUMBER_OF_GHOSTS )
      : ExpressionBuilder( fluxTag, nghost ),
        turbSc_( turbSchmidt ),
        turbViscTag_( turbViscTag ),
        phiTag_( phiTag )
    {}

    Expr::ExpressionBase* build() const{
      return new TurbDiffFlux<FluxT>( turbViscTag_, turbSc_, phiTag_ );
    }

  };  /* end of Builder class */

  ~TurbDiffFlux();
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
};



// ###################################################################
//
//                          Implementation
//
// ###################################################################



template< typename FluxT >
TurbDiffFlux<FluxT>::

TurbDiffFlux( const Expr::Tag& turbViscTag,
              const double turbSchmidt,
              const Expr::Tag& phiTag )
  : Expr::Expression<FluxT>(),
    scTurb_( turbSchmidt )
{
  this->set_gpu_runnable( true );

  turbVisc_ = this->template create_field_request<ScalarT>( turbViscTag );
  phi_      = this->template create_field_request<ScalarT>( phiTag      );
}

//--------------------------------------------------------------------

template< typename FluxT >
TurbDiffFlux<FluxT>::
~TurbDiffFlux()
{}

//--------------------------------------------------------------------

template< typename FluxT >
void
TurbDiffFlux<FluxT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  gradOp_   = opDB.retrieve_operator<GradT  >();
  interpOp_ = opDB.retrieve_operator<InterpT>();
}

//--------------------------------------------------------------------

template< typename FluxT >
void
TurbDiffFlux<FluxT>::
evaluate()
{
  FluxT& flux = this->value();

  const ScalarT& turbVisc = turbVisc_->field_ref();
  const ScalarT& phi      = phi_     ->field_ref();

  const double scInv = 1.0 / scTurb_;

  flux <<= -(*interpOp_)( turbVisc * scInv ) * (*gradOp_)( phi );
}

//--------------------------------------------------------------------

#endif // TurbDiffFlux_Expr_h
