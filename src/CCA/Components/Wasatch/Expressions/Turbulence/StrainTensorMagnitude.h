/*
 * The MIT License
 *
 * Copyright (c) 2012-2017 The University of Utah
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

#ifndef StrainTensorMagnitude_Expr_h
#define StrainTensorMagnitude_Expr_h

#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>
#include "StrainTensorBase.h"
#include <expression/Expression.h>

/**
 *  \class   StrainTensorSquare
 *  \authors Tony Saad
 *  \date    Jan, 2012. (Originally created: June, 2012).
 *  \ingroup Expressions
 *
 *  \brief Given the components of a velocity field, \f$\bar{u}_i\f$, this
 expression calculates the square of the filtered strain tensor,
 \f$\tilde{S}_{kl}\tilde{S}_{kl}\f$ where
 \f$S_{kl}=\frac{1}{2}(\frac{\partial\tilde{u}_k}{\partial x_l}+\frac{\partial\tilde{u}_l}{\partial x_k})\f$.
 Note that \f$ \tilde{S}_{kl}\tilde{S}_{kl} \equiv \frac{1}{2}|\tilde{S}|^2 \f$.
 The reason for calculating \f$ \frac{1}{2}|\tilde{S}|^2 \f$ instead of \f$ |\tilde{S}| \f$
 is that it will be used in the WALE and VREMAN models which makes it easy to implement
 a unified interface across these turbulence models.
 *
 */
class StrainTensorSquare : Expr::Expression<SVolField> {
  
  StrainTensorSquare( const Expr::Tag& s11Tag,
                      const Expr::Tag& s21Tag,
                      const Expr::Tag& s31Tag,
                      const Expr::Tag& s22Tag,
                      const Expr::Tag& s32Tag,
                      const Expr::Tag& s33Tag );

  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, SpatialOps::XSurfXField, SVolField >::type XXInterpT;
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, SpatialOps::YSurfYField, SVolField >::type YYInterpT;
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, SpatialOps::ZSurfZField, SVolField >::type ZZInterpT;
    
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, SpatialOps::XSurfYField, SVolField >::type XYInterpT;
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, SpatialOps::XSurfZField, SVolField >::type XZInterpT;
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, SpatialOps::YSurfZField, SVolField >::type YZInterpT;

  // NOTATION: We follow the classical continuum mechanics notation for tensors:
  // S_ij is the strain rate on the i-th face in the j-th direction.
  // Although the tensor is symmetric at a point and there is no distinction between S_ij and S_ji,
  // this distinction is important when dealing with staggered grids where S_ij cannot really be compared to S_ji
  // because these quantities live on different grid locations.

  // XVOL related strain rates
  typedef SpatialOps::XSurfXField S11T;
  typedef SpatialOps::XSurfYField S21T;
  typedef SpatialOps::XSurfZField S31T;
  DECLARE_FIELD(S11T, S11_) // strain on x face in x direction
  DECLARE_FIELD(S21T, S21_) // strain on y face in x direction
  DECLARE_FIELD(S31T, S31_) // strain on z face in x direction
  
  // YVOL related strain rates
  // const SpatialOps::YSurfXField* S12_; // strain on x face in y direction
  typedef SpatialOps::YSurfYField S22T;
  typedef SpatialOps::YSurfZField S32T;
  DECLARE_FIELD(S22T, S22_) // strain on y face in Y direction
  DECLARE_FIELD(S32T, S32_) // strain on z face in Y direction
  
  // ZVOL related strain rates
//  const SpatialOps::ZSurfXField* S13_; // strain on x face in z direction
//  const SpatialOps::ZSurfYField* S23_; // strain on y face in z direction
  typedef SpatialOps::ZSurfZField S33T;
  DECLARE_FIELD(S33T, S33_) // strain on z face in Z direction
//  const SVolField* dil_;
  
  const XXInterpT* xxInterpOp_;
  const YYInterpT* yyInterpOp_;
  const ZZInterpT* zzInterpOp_;
  const XYInterpT* xyInterpOp_;
  const XZInterpT* xzInterpOp_;
  const YZInterpT* yzInterpOp_;

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder(const Expr::Tag& result,
            const Expr::Tag& s11Tag,
            const Expr::Tag& s21Tag,
            const Expr::Tag& s31Tag,
            const Expr::Tag& s22Tag,
            const Expr::Tag& s32Tag,
            const Expr::Tag& s33Tag );

    ~Builder(){}
    Expr::ExpressionBase* build() const;
    
  private:
    const Expr::Tag S11Tag_, S21Tag_, S31Tag_, S22Tag_, S32Tag_, S33Tag_;
  };
  
  ~StrainTensorSquare();

  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
};

/**
 *  \class  WaleTensorMagnitude
 *  \author Tony Saad
 *  \date   June, 2012
 *  \ingroup Expressions
 *
 *  \brief This calculates the square velocity gradient tensor. 
           This is used in the W.A.L.E. turbulence model. 
           See:
           Nicoud and Ducros, 1999, Subgrid-Scale Stress Modelling Based on the
           Square of the Velocity Gradient Tensor
 *
 */
template<typename ResultT, typename Vel1T, typename Vel2T, typename Vel3T>
class WaleTensorMagnitude : public StrainTensorBase<ResultT, Vel1T, Vel2T, Vel3T> {

  WaleTensorMagnitude( const Expr::TagList& velTags );  
  public:
    class Builder : public Expr::ExpressionBuilder
    {
    public:
      
      /**
       *  \param vel1tag the first component of the velocity 
       *  \param vel2tag the second component of the velocity 
       *  \param vel3tag the third component of the velocity 
       */
      Builder( const Expr::Tag& result,
               const Expr::TagList& velTags );
      ~Builder(){}
      Expr::ExpressionBase* build() const;
      
    private:
      const Expr::TagList velTags_;
    };
  
  ~WaleTensorMagnitude();
  void evaluate();
};

/**
 *  \class  VremanTensorMagnitude
 *  \author Tony Saad
 *  \date   June, 2012
 *  \ingroup Expressions
 *
 *  \brief This calculates the Vreman tensor magnitude.
 This is used in the Vreman turbulent model. 
 See:
 Vreman 2004, An eddy-viscosity subgrid-scale model for turbulent shear flow:
 Algebraic theory and applications
 */
template<typename ResultT, typename Vel1T, typename Vel2T, typename Vel3T>
class VremanTensorMagnitude : public StrainTensorBase<ResultT, Vel1T, Vel2T, Vel3T> {
  
  VremanTensorMagnitude( const Expr::TagList& velTags );  
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    
    /**
     *  \param vel1tag the first component of the velocity 
     *  \param vel2tag the second component of the velocity 
     *  \param vel3tag the third component of the velocity 
     */
    Builder( const Expr::Tag& result,
             const Expr::TagList& velTags );
    ~Builder(){}
    Expr::ExpressionBase* build() const;
    
  private:
    const Expr::TagList velTags_;
  };
  
  ~VremanTensorMagnitude();
  void evaluate();
};
#endif // StrainTensorMagnitude_Expr_h
