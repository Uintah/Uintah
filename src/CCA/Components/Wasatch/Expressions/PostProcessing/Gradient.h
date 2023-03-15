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
#include <spatialops/structured/FVStaggered.h>
#include <CCA/Components/Wasatch/Expressions/ScalarEOSHelper.h>

#ifndef Wasatch_Gradient_h
#define Wasatch_Gradient_h

/**
 *  \class Gradient
 *  \author Josh McConnell
 *  \date   November 2018
 *
 *  \brief Computes
 *  \f[
 *  \nabla_{x_i} \phi
 *  \f]
 */

namespace WasatchCore{

template<typename FaceT, typename DestT>
class Gradient : public Expr::Expression<DestT>
{
  typedef typename SpatialOps::VolType<FaceT>::VolField ScalarT;
  typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Gradient,ScalarT,FaceT>::type GradT;

  typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Interpolant,FaceT,DestT>::type Face2DestInterpT;

  DECLARE_FIELD( ScalarT, phi_ )


  const GradT*            gradOp_;
  const Face2DestInterpT* face2DestOp_;

  Gradient( const Expr::Tag& phiTag );

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  @brief Build a Gradient expression
     *  @param resultTag the tag for the value that this expression computes
     */

    Builder( const Expr::Tag& resultTag,
             const Expr::Tag& phiTag );

    Expr::ExpressionBase* build() const{
      return new Gradient( phiTag_ );
    }

  private:
    const Expr::Tag phiTag_;
  };

  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();

};


// ###################################################################
//
//                          Implementation
//
// ###################################################################

template<typename FaceT, typename DestT>
Gradient<FaceT, DestT>::
Gradient( const Expr::Tag& phiTag )
: Expr::Expression<DestT>()

{
  this->set_gpu_runnable(true);

  phi_ = this->template create_field_request<ScalarT>( phiTag );
}

//--------------------------------------------------------------------

template<typename FaceT, typename DestT>
void Gradient<FaceT, DestT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  gradOp_      = opDB.retrieve_operator<GradT           >();
  face2DestOp_ = opDB.retrieve_operator<Face2DestInterpT>();
}
//--------------------------------------------------------------------

template<typename FaceT, typename DestT>
void
Gradient<FaceT, DestT>::
evaluate()
{
  using namespace SpatialOps;
  DestT& result = this->value();

  result <<= (*face2DestOp_)( (*gradOp_)( phi_->field_ref() ) );
}

//--------------------------------------------------------------------

template<typename FaceT, typename DestT>
Gradient<FaceT, DestT>::
Builder::Builder( const Expr::Tag& resultTag,
                  const Expr::Tag& phiTag )
  : ExpressionBuilder( resultTag ),
    phiTag_( phiTag )
{}

//====================================================================
}

#endif // Wasatch_Gradient_h

