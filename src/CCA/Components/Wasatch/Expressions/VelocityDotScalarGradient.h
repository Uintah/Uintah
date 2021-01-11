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
#include <spatialops/structured/FVStaggered.h>

#ifndef VelocityDotScalarGradient_h
#define VelocityDotScalarGradient_h

/**
 *  \class VelocityDotScalarGradient
 *  \author Josh McConnell
 *  \date   October 2018
 *
 *  \brief Computes
 *  \f[
 *   u_{x_i} \cdot \nabla_{x_i} \phi
 *  \f]
 */


template<typename FaceT, typename VelT>
class VelocityDotScalarGradient : public Expr::Expression<typename SpatialOps::VolType<FaceT>::VolField>
{
  typedef typename SpatialOps::VolType<FaceT>::VolField ScalarT;
  typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Gradient,ScalarT,FaceT>::type GradT;

  typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Interpolant,FaceT,ScalarT>::type Face2ScalarInterpT;
  typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Interpolant,VelT, FaceT  >::type Vel2FaceInterpT;

  DECLARE_FIELD( ScalarT, phi_      )
  DECLARE_FIELD( VelT   , velocity_ )


  const GradT*              gradOp_;
  const Face2ScalarInterpT* face2ScalarOp_;
  const Vel2FaceInterpT*    vel2FaceOp_;

  VelocityDotScalarGradient( const Expr::Tag& phiTag,
                             const Expr::Tag& velTag );

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  @brief Build a VelocityDotScalarGradient expression
     *  @param dpdtPartTag the tag for the value that this expression computes
     */

    Builder( const Expr::Tag& resultTag,
             const Expr::Tag& phiTag,
             const Expr::Tag& velTag );

    Expr::ExpressionBase* build() const{
      return new VelocityDotScalarGradient( phiTag_, velTag_ );
    }

  private:
    const Expr::Tag phiTag_, velTag_;
  };

  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();

};


// ###################################################################
//
//                          Implementation
//
// ###################################################################

template<typename FaceT, typename VelT>
VelocityDotScalarGradient<FaceT, VelT>::
VelocityDotScalarGradient( const Expr::Tag& phiTag,
                           const Expr::Tag& velTag )
: Expr::Expression<ScalarT>()

{
  this->set_gpu_runnable(true);

  phi_       = this->template create_field_request<ScalarT>( phiTag );
  velocity_  = this->template create_field_request<VelT   >( velTag );
}

//--------------------------------------------------------------------

template<typename FaceT, typename VelT>
void VelocityDotScalarGradient<FaceT, VelT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  gradOp_        = opDB.retrieve_operator<GradT             >();
  face2ScalarOp_ = opDB.retrieve_operator<Face2ScalarInterpT>();
  vel2FaceOp_    = opDB.retrieve_operator<Vel2FaceInterpT   >();
}
//--------------------------------------------------------------------

template<typename FaceT, typename VelT>
void
VelocityDotScalarGradient<FaceT, VelT>::
evaluate()
{
  using namespace SpatialOps;
  ScalarT& result = this->value();

  const ScalarT& phi = phi_     ->field_ref();
  const VelT&    vel = velocity_->field_ref();

  result <<= (*face2ScalarOp_)( (*vel2FaceOp_)( vel ) * (*gradOp_)( phi ) );
}

//--------------------------------------------------------------------

template<typename FaceT, typename VelT>
VelocityDotScalarGradient<FaceT, VelT>::
Builder::Builder( const Expr::Tag& resultTag,
                  const Expr::Tag& phiTag,
                  const Expr::Tag& velTag)
  : ExpressionBuilder( resultTag ),
    phiTag_( phiTag ),
    velTag_( velTag )
{}

//====================================================================

#endif // VelocityDotScalarGradient_h

