/*
 * The MIT License
 *
 * Copyright (c) 2012 The University of Utah
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

#ifndef StrainTensorBase_Expr_h
#define StrainTensorBase_Expr_h

//-- Wasatch Includes --//
#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>

//-- ExprLib Includes --//
#include <expression/Expression.h>

/**
 *  \class StrainTensorBase
 *  \authors Tony Saad
 *  \date    April, 2013
 *  \ingroup Expressions
 *
 *  \brief This class provides a unified framework for calculating LES eddy-viscosity turbulence tensors.
A common feature of LES eddy-viscosity turbulence models is that they represent dissipation at the small
scales using one form of a velocity gradient tensor. Given this common feature, this base class defines
the necessary derivative and interpolant operators that are expected to be used in the eddy-viscosity 
models. Typical eddy-viscosity models are the Constant Smagorinsky, the Vreman, and the WALE models.
Furthermore, all operators defined here will be used in the Dynamic Smagorinsky Coefficient expression.
 *
 */
class StrainTensorBase : public Expr::Expression<SVolField>
{
protected:
  
  const Expr::TagList velTags_;
  
  //A_SURF_B_Field = A vol, B surface
  typedef SpatialOps::SpatFldPtr<SVolField> SVolPtr;
  typedef std::vector< SVolPtr  > SVolVecT;
  typedef std::vector< SVolVecT > SVolTensorT;
  
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Gradient, XVolField, SpatialOps::XSurfYField >::type dudyT;
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Gradient, YVolField, SpatialOps::YSurfXField >::type dvdxT;
  
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Gradient, XVolField, SpatialOps::XSurfZField >::type dudzT;
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Gradient, ZVolField, SpatialOps::ZSurfXField >::type dwdxT;
  
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Gradient, YVolField, SpatialOps::YSurfZField >::type dvdzT;
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Gradient, ZVolField, SpatialOps::ZSurfYField >::type dwdyT;
  
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Gradient, XVolField, SVolField >::type dudxT;
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Gradient, YVolField, SVolField >::type dvdyT;
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Gradient, ZVolField, SVolField >::type dwdzT;
  
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, SpatialOps::XSurfYField, SVolField >::type XYInterpT;
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, SpatialOps::YSurfXField, SVolField >::type YXInterpT;
  
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, SpatialOps::XSurfZField, SVolField >::type XZInterpT;
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, SpatialOps::ZSurfXField, SVolField >::type ZXInterpT;
  
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, SpatialOps::YSurfZField, SVolField >::type YZInterpT;
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, SpatialOps::ZSurfYField, SVolField >::type ZYInterpT;
  
  const XVolField* vel1_;
  const YVolField* vel2_;
  const ZVolField* vel3_;
  
  const bool doX_, doY_, doZ_;
  
  const dudyT* dudyOp_;
  const dvdxT* dvdxOp_;
  const dudzT* dudzOp_;
  const dwdxT* dwdxOp_;
  const dvdzT* dvdzOp_;
  const dwdyT* dwdyOp_;
  const dudxT* dudxOp_;
  const dvdyT* dvdyOp_;
  const dwdzT* dwdzOp_;
  
  const XYInterpT* xyInterpOp_;
  const YXInterpT* yxInterpOp_;
  
  const XZInterpT* xzInterpOp_;
  const ZXInterpT* zxInterpOp_;
  
  const YZInterpT* yzInterpOp_;
  const ZYInterpT* zyInterpOp_;
  
  StrainTensorBase( const Expr::TagList& velTags );
  
  void calculate_strain_tensor_components(SVolField& strTsrMag,
                                          const XVolField& u,
                                          const YVolField& v,
                                          const ZVolField& w,
                                          SVolField& S11,
                                          SVolField& S12,
                                          SVolField& S13,
                                          SVolField& S22,
                                          SVolField& S23,
                                          SVolField& S33);
  
public:  
  ~StrainTensorBase();
  
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
}; // class StrainTensorBase

#endif
