/*
 * The MIT License
 *
 * Copyright (c) 2010-2023 The University of Utah
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

#include <expression/Expression.h>
#include <pokitt/CanteraObjects.h>

#ifndef PressureMaterialDerivative_h
#define PressureMaterialDerivative_h

/**
 *  \class PressureMaterialDerivative
 *  \author Josh McConnell
 *  \date   November 2018
 *
 *  \brief Computes
 *  \f[
 *   \frac{DP}{Dt} = \widehat{\frac{DP}{Dt}} \frac{D P}{D t}  - \gamma P \nabla \cdot \mathbf{u}
 *  \f]
 *   where \f$P\f$ is pressure, \f$\gamma = \frac{c_p M}{c_p M - R}\f$, \f$\rho\f$ is density, \f$\mathbf{u}\f$ is velocity,
 *   \f$\c_p\f$ is heat capacity, and \f$\R\f$ is the universal gas constant.
 */
template<typename FieldT>
class PressureMaterialDerivative : public Expr::Expression<FieldT>
{
  DECLARE_FIELDS(FieldT, partialDPDt_, gamma_, p_,  divu_ )

  const double gasConst_;

  PressureMaterialDerivative( const Expr::Tag& partialDPDtTag,
                              const Expr::Tag& gammaTag,
                              const Expr::Tag& pressureTag,
                              const Expr::Tag& divuTag );

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  @brief Build a PressureMaterialDerivative expression
     *  @param resultTag the tag for the value that this expression computes
     */

    Builder( const Expr::Tag& resultTag,
             const Expr::Tag& partialDPDtTag,
             const Expr::Tag& gammaTag,
             const Expr::Tag& pressureTag,
             const Expr::Tag& divuTag );

    Expr::ExpressionBase* build() const{
      return new PressureMaterialDerivative( partialDPDtTag_,
                                             gammaTag_,
                                             pressureTag_,
                                             divuTag_ );
    }

  private:
    const Expr::Tag partialDPDtTag_, gammaTag_, pressureTag_, divuTag_;
  };

  void evaluate();

};


// ###################################################################
//
//                          Implementation
//
// ###################################################################

template<typename FieldT>
PressureMaterialDerivative<FieldT>::
PressureMaterialDerivative( const Expr::Tag& partialDPDtTag,
                            const Expr::Tag& gammaTag,
                            const Expr::Tag& pressureTag,
			    const Expr::Tag& divuTag )
: Expr::Expression<FieldT>(),
  gasConst_( CanteraObjects::gas_constant() )
{
  this->set_gpu_runnable(true);

  partialDPDt_ = this->template create_field_request<FieldT>( partialDPDtTag );
  gamma_       = this->template create_field_request<FieldT>( gammaTag       );
  p_           = this->template create_field_request<FieldT>( pressureTag    );
  divu_        = this->template create_field_request<FieldT>( divuTag        );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
PressureMaterialDerivative<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();

  const FieldT& partialDPDt = partialDPDt_->field_ref();
  const FieldT& gamma       = gamma_      ->field_ref();
  const FieldT& p           = p_          ->field_ref();
  const FieldT& divu        = divu_       ->field_ref();

  result <<= partialDPDt - (gamma * p * divu);
}

//--------------------------------------------------------------------

template<typename FieldT>
PressureMaterialDerivative<FieldT>::
Builder::Builder( const Expr::Tag& resultTag,
                  const Expr::Tag& partialDPDtTag,
                  const Expr::Tag& gammaTag,
                  const Expr::Tag& pressureTag,
	          const Expr::Tag& divuTag )
  : ExpressionBuilder( resultTag ),
    partialDPDtTag_( partialDPDtTag ),
    gammaTag_      ( gammaTag       ),
    pressureTag_   ( pressureTag    ),
    divuTag_       ( divuTag        )
{}

//====================================================================
#endif // PressureMaterialDerivative_h

