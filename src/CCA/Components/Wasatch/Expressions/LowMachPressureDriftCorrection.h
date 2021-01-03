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

#ifndef LowMachPressureDriftCorrectionExpr_h
#define LowMachPressureDriftCorrectionExpr_h

#include <expression/Expression.h>

//-- SpatialOps Includes --//
#include <spatialops/structured/FVStaggered.h>

namespace WasatchCore{

template< typename ScalarT >
class LowMachPressureDriftCorrection : public Expr::Expression<ScalarT>
{
  /**
   *  \class   LowMachPressureDriftCorrection
   *  \author  Josh McConnell
   *  \date    December, 2018
   *  \ingroup Expressions
   *
   *  \brief Calculates the time derivative of any field.
   *
   */

  LowMachPressureDriftCorrection( const Expr::Tag& thermoPressureTag,
                                  const Expr::Tag& backgroundPressureTag,
                                  const Expr::Tag& adiabaticIndexTag,
                                  const Expr::Tag& xVelocityTag,
                                  const Expr::Tag& yVelocityTag,
                                  const Expr::Tag& zVelocityTag,
                                  const Expr::Tag& timestepTag );

  typedef typename SpatialOps::SingleValueField TimeFieldT;

  typedef SpatialOps::XVolField XVelT;
  typedef SpatialOps::YVolField YVelT;
  typedef SpatialOps::ZVolField ZVelT;

  typedef typename SpatialOps::FaceTypes<ScalarT> FaceTypes;
  typedef typename FaceTypes::XFace XFaceT;
  typedef typename FaceTypes::YFace YFaceT;
  typedef typename FaceTypes::ZFace ZFaceT;

  typedef typename SpatialOps::BasicOpTypes<ScalarT> OpTypes;
  typedef typename OpTypes::GradX GradX;
  typedef typename OpTypes::GradY GradY;
  typedef typename OpTypes::GradZ GradZ;

  typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Interpolant, XFaceT, ScalarT>::type XFace2ScalarInterpT;
  typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Interpolant, YFaceT, ScalarT>::type YFace2ScalarInterpT;
  typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Interpolant, ZFaceT, ScalarT>::type ZFace2ScalarInterpT;

  typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Interpolant, XVelT, XFaceT>::type XVel2XFaceInterpT;
  typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Interpolant, YVelT, YFaceT>::type YVel2YFaceInterpT;
  typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Interpolant, ZVelT, ZFaceT>::type ZVel2ZFaceInterpT;

  DECLARE_FIELDS(ScalarT, pThermo_, pBackground_, gamma_)

  DECLARE_FIELD(XVelT, xVel_)
  DECLARE_FIELD(YVelT, yVel_)
  DECLARE_FIELD(ZVelT, zVel_)

  DECLARE_FIELD(TimeFieldT, dt_)

  const bool doX_, doY_, doZ_, is3D_;

  const GradX* gradXOp_;
  const GradY* gradYOp_;
  const GradZ* gradZOp_;

  const XFace2ScalarInterpT* xFace2ScalarOp_;
  const YFace2ScalarInterpT* yFace2ScalarOp_;
  const ZFace2ScalarInterpT* zFace2ScalarOp_;

  const XVel2XFaceInterpT* xVel2FaceOp_;
  const YVel2YFaceInterpT* yVel2FaceOp_;
  const ZVel2ZFaceInterpT* zVel2FaceOp_;
public:
  
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& result,
	     const Expr::Tag& thermoPressureTag,
	     const Expr::Tag& backgroundPressureTag,
	     const Expr::Tag& adiabaticIndexTag,
	     const Expr::Tag& xVelocityTag,
	     const Expr::Tag& yVelocityTag,
	     const Expr::Tag& zVelocityTag,
	     const Expr::Tag& timestepTag );

    Expr::ExpressionBase* build() const{
      return new LowMachPressureDriftCorrection( thermoPressureTag_,
                                                 backgroundPressureTag_,
                                                 adiabaticIndexTag_,
                                                 xVelocityTag_,
                                                 yVelocityTag_,
                                                 zVelocityTag_,
                                                 timestepTag_ );
    }

    ~Builder(){}

  private:
    const Expr::Tag thermoPressureTag_, backgroundPressureTag_, adiabaticIndexTag_,
                    xVelocityTag_, yVelocityTag_, zVelocityTag_, timestepTag_;
  };
  
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
  
};

//====================================================================

//--------------------------------------------------------------------

template<typename ScalarT>
LowMachPressureDriftCorrection<ScalarT>::
LowMachPressureDriftCorrection( const Expr::Tag& thermoPressureTag,
                                const Expr::Tag& backgroundPressureTag,
                                const Expr::Tag& adiabaticIndexTag,
                                const Expr::Tag& xVelocityTag,
                                const Expr::Tag& yVelocityTag,
                                const Expr::Tag& zVelocityTag,
                                const Expr::Tag& timestepTag )
: Expr::Expression<ScalarT>(),
  doX_ ( xVelocityTag != Expr::Tag() ),
  doY_ ( yVelocityTag != Expr::Tag() ),
  doZ_ ( zVelocityTag != Expr::Tag() ),
  is3D_( doX_ && doY_ && doZ_        )
{
  this->set_gpu_runnable( true );

  pThermo_     = this->template create_field_request<ScalarT>(thermoPressureTag    );
  pBackground_ = this->template create_field_request<ScalarT>(backgroundPressureTag);
  gamma_       = this->template create_field_request<ScalarT>(adiabaticIndexTag    );

  if( doX_ ) xVel_ = this->template create_field_request<XVelT>(xVelocityTag);
  if( doY_ ) yVel_ = this->template create_field_request<YVelT>(yVelocityTag);
  if( doZ_ ) zVel_ = this->template create_field_request<ZVelT>(zVelocityTag);

  dt_ = this->template create_field_request<TimeFieldT>(timestepTag);
}

//------------------------------------------------------------------

template<typename ScalarT>
void
LowMachPressureDriftCorrection<ScalarT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  if( doX_ ){
    gradXOp_        = opDB.retrieve_operator<GradX              >();
    xFace2ScalarOp_ = opDB.retrieve_operator<XFace2ScalarInterpT>();
    xVel2FaceOp_    = opDB.retrieve_operator<XVel2XFaceInterpT  >();
  }
  if( doY_ ){
    gradYOp_        = opDB.retrieve_operator<GradY              >();
    yFace2ScalarOp_ = opDB.retrieve_operator<YFace2ScalarInterpT>();
    yVel2FaceOp_    = opDB.retrieve_operator<YVel2YFaceInterpT  >();
  }
  if( doZ_ ){
    gradZOp_        = opDB.retrieve_operator<GradZ              >();
    zFace2ScalarOp_ = opDB.retrieve_operator<ZFace2ScalarInterpT>();
    zVel2FaceOp_    = opDB.retrieve_operator<ZVel2ZFaceInterpT  >();
  }

}

//--------------------------------------------------------------------

template< typename ScalarT >
void
LowMachPressureDriftCorrection<ScalarT>::
evaluate()
{
  using namespace SpatialOps;
  ScalarT& result = this->value();

  const ScalarT & gamma   = gamma_      ->field_ref();
  const ScalarT & pThermo = pThermo_    ->field_ref();
  const ScalarT & p0      = pBackground_->field_ref();

  const TimeFieldT& dt = dt_->field_ref();

  result  <<= (pThermo - p0)/dt;

  if(is3D_){
    const XVelT& xVel = xVel_->field_ref();
    const YVelT& yVel = yVel_->field_ref();
    const ZVelT& zVel = zVel_->field_ref();

    result <<= result - (*xFace2ScalarOp_)( (*xVel2FaceOp_)(xVel) * (*gradXOp_)(pThermo) )
                      - (*yFace2ScalarOp_)( (*yVel2FaceOp_)(yVel) * (*gradYOp_)(pThermo) )
                      - (*zFace2ScalarOp_)( (*zVel2FaceOp_)(zVel) * (*gradZOp_)(pThermo) );
  }
  else{
    if(doX_){
      const XVelT& xVel = xVel_->field_ref();

      result <<= result - (*xFace2ScalarOp_)( (*xVel2FaceOp_)(xVel) * (*gradXOp_)(pThermo) );
    }
    else if(doY_){
      const YVelT& yVel = yVel_->field_ref();

      result <<= result - (*yFace2ScalarOp_)( (*yVel2FaceOp_)(yVel) * (*gradYOp_)(pThermo) );
    }
    else if(doZ_){
      const ZVelT& zVel = zVel_->field_ref();

      result <<= result - (*zFace2ScalarOp_)( (*zVel2FaceOp_)(zVel) * (*gradZOp_)(pThermo) );
    }
  }

  result <<= result/(gamma * pThermo);

}

//--------------------------------------------------------------------

template< typename ScalarT >
LowMachPressureDriftCorrection<ScalarT>::
Builder::
Builder( const Expr::Tag& result,
         const Expr::Tag& thermoPressureTag,
         const Expr::Tag& backgroundPressureTag,
         const Expr::Tag& adiabaticIndexTag,
         const Expr::Tag& xVelocityTag,
         const Expr::Tag& yVelocityTag,
         const Expr::Tag& zVelocityTag,
         const Expr::Tag& timestepTag  )
: ExpressionBuilder(result),
  thermoPressureTag_    ( thermoPressureTag     ),
  backgroundPressureTag_( backgroundPressureTag ),
  adiabaticIndexTag_    ( adiabaticIndexTag     ),
  xVelocityTag_         ( xVelocityTag          ),
  yVelocityTag_         ( yVelocityTag          ),
  zVelocityTag_         ( zVelocityTag          ),
  timestepTag_          ( timestepTag           )
{}

//--------------------------------------------------------------------

}//namespece WasatchCore
#endif // LowMachPressureDriftCorrectionExpr_h
