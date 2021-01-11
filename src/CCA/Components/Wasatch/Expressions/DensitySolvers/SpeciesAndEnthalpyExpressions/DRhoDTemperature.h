/*
 * The MIT License
 *
 * Copyright (c) 2010-2018 The University of Utah
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

#ifndef DRhoDTemperature_h
#define DRhoDTemperature_h

namespace WasatchCore{

/**
 *  \class DRhoDTemperature
 *  \author Josh McConnell
 *  \date   May 2018
 *
 *  \brief Computes \f$\frac{\partial \rho}{\partial T}\f$
 *  given as
 *  \f[
 *   \frac{\partial \rho}{\partial T} = -\frac{\rho}{ T},
 *  \f]
 *  where \f$\T\f$ is the temperature and \f$\rho\f$ is the
 *  mixture heat density.
 */
template<typename FieldT>
class DRhoDTemperature : public Expr::Expression<FieldT>
{
  DECLARE_FIELDS(FieldT, rho_, t_ )

    DRhoDTemperature( const Expr::Tag& rhoTag,
                      const Expr::Tag& temperatureTag );

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  @brief Build a DRhoDTemperature expression
     *  @param resultTag the tag for the value that this expression computes
     */

    Builder( const Expr::Tag& resultTag,
             const Expr::Tag& rhoTag,
             const Expr::Tag& temperatureTag );

    Expr::ExpressionBase* build() const{
      return new DRhoDTemperature( rhoTag_, temperatureTag_ );
    }

  private:
    const Expr::Tag rhoTag_, temperatureTag_;
  };

  void evaluate();

};


// ###################################################################
//
//                          Implementation
//
// ###################################################################

template<typename FieldT>
DRhoDTemperature<FieldT>::
DRhoDTemperature( const Expr::Tag& rhoTag,
                  const Expr::Tag& temperatureTag )
: Expr::Expression<FieldT>()
{
  this->set_gpu_runnable(true);

  rho_ = this->template create_field_request<FieldT>( rhoTag         );
  t_   = this->template create_field_request<FieldT>( temperatureTag );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
DRhoDTemperature<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();

  const FieldT& rho = rho_->field_ref();
  const FieldT& t   = t_  ->field_ref();

  result <<= -rho / (t);
}

//--------------------------------------------------------------------

template<typename FieldT>
DRhoDTemperature<FieldT>::
Builder::Builder( const Expr::Tag& resultTag,
                  const Expr::Tag& rhoTag,
                  const Expr::Tag& temperatureTag )
  : ExpressionBuilder( resultTag ),
    rhoTag_        ( rhoTag         ),
    temperatureTag_( temperatureTag )
{}

//====================================================================
}
#endif // DRhoDTemperature_h

