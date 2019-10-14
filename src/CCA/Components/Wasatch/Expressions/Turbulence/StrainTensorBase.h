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
template<typename ResultT, typename Vel1T, typename Vel2T, typename Vel3T>
class StrainTensorBase : public Expr::Expression<ResultT>
{
protected:
    
  typedef typename SpatialOps::FaceTypes<Vel1T> UFaceTypes;
  typedef typename SpatialOps::FaceTypes<Vel2T> VFaceTypes;
  typedef typename SpatialOps::FaceTypes<Vel3T> WFaceTypes;
  
  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Gradient, Vel1T, typename UFaceTypes::YFace >::type dudyT;
  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Gradient, Vel2T, typename VFaceTypes::XFace >::type dvdxT;
  
  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Gradient, Vel1T, typename UFaceTypes::ZFace >::type dudzT;
  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Gradient, Vel3T, typename WFaceTypes::XFace >::type dwdxT;
  
  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Gradient, Vel2T, typename VFaceTypes::ZFace >::type dvdzT;
  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Gradient, Vel3T, typename WFaceTypes::YFace >::type dwdyT;
  
  typedef typename SpatialOps::OperatorTypeBuilder< typename WasatchCore::GradOpSelector<Vel1T, SpatialOps::XDIR>::Gradient, Vel1T, ResultT >::type dudxT;
  typedef typename SpatialOps::OperatorTypeBuilder< typename WasatchCore::GradOpSelector<Vel2T, SpatialOps::YDIR>::Gradient, Vel2T, ResultT >::type dvdyT;
  typedef typename SpatialOps::OperatorTypeBuilder< typename WasatchCore::GradOpSelector<Vel3T, SpatialOps::ZDIR>::Gradient, Vel3T, ResultT >::type dwdzT;
  
  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, typename UFaceTypes::YFace, ResultT >::type XYInterpT;
  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, typename VFaceTypes::XFace, ResultT >::type YXInterpT;
  
  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, typename UFaceTypes::ZFace, ResultT >::type XZInterpT;
  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, typename WFaceTypes::XFace, ResultT >::type ZXInterpT;
  
  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, typename VFaceTypes::ZFace, ResultT >::type YZInterpT;
  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, typename WFaceTypes::YFace, ResultT >::type ZYInterpT;
  
  DECLARE_FIELD(Vel1T, u_)
  DECLARE_FIELD(Vel2T, v_)
  DECLARE_FIELD(Vel3T, w_)
  
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
  
  // for symmetric strain tensor on structured staggered grids
  void calculate_strain_tensor_components(ResultT& strTsrMag,
                                          const Vel1T& u,
                                          const Vel2T& v,
                                          const Vel3T& w,
                                          ResultT& S11,
                                          ResultT& S12,
                                          ResultT& S13,
                                          ResultT& S22,
                                          ResultT& S23,
                                          ResultT& S33);
  
public:  
  ~StrainTensorBase();
  
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
}; // class StrainTensorBase

#endif
