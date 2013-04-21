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

#ifndef StrainTensorMagnitude_Expr_h
#define StrainTensorMagnitude_Expr_h

#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>
#include "StrainTensorBase.h"
#include <expression/Expression.h>

/**
 *  \brief obtain the tag for the strain tensor magnitude
 */
Expr::Tag straintensormagnitude_tag();
Expr::Tag wale_tensormagnitude_tag();
Expr::Tag vreman_tensormagnitude_tag();

/**
 *  \class   StrainTensorSquare
 *  \authors Amir Biglari, Tony Saad
 *  \date    Jan, 2012. (Originally created: June, 2012).
 *  \ingroup Expressions
 *
 *  \brief Given the components of a velocity field, \f$\bar{u}_i\f$, this
 expression calculates the square of the filtered strain tensor,
 \f$\tilde{S}_{kl}\tilde{S}_{kl}\f$ where
 \f$S_{kl}=\frac{1}{2}(\frac{\partial\tilde{u}_k}{\partial x_l}+\frac{\partial\tilde{u}_l}{\partial x_k})\f$.
 Note that \f$ \tilde{S}_{kl}\tilde{S}_{kl} \equiv \tfrac{1}{2}|\tilde{S}|^2 \f$.
 The reason for calculating \f$ \tfrac{1}{2}|\tilde{S}|^2 \f$ instead of \f$ |\tilde{S}| \f$
 is that it will be used in the WALE and VREMAN models which makes it easy to implement
 a unified interface across these turbulence models.
 *
 */
class StrainTensorSquare : public StrainTensorBase {
  
  StrainTensorSquare( const Expr::Tag& vel1tag,
                      const Expr::Tag& vel2tag,
                      const Expr::Tag& vel3tag);
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& result,
             const Expr::Tag& vel1tag,
             const Expr::Tag& vel2tag,
             const Expr::Tag& vel3tag );
    ~Builder(){}
    Expr::ExpressionBase* build() const;
    
  private:
    const Expr::Tag v1t_, v2t_, v3t_;
  };
  
  ~StrainTensorSquare();
  void evaluate();
};

/**
 *  \class  WaleTensorMagnitude
 *  \author Tony Saad
 *  \date   June, 2012
 *  \ingroup Expressions
 *
 *  \brief This calculates the square velocity gradient tensor. 
           This is used in the W.A.L.E. turbulent model. 
           See:
           Nicoud and Ducros, 1999, Subgrid-Scale Stress Modelling Based on the
           Square of the Velocity Gradient Tensor
 *
 */
class WaleTensorMagnitude : public StrainTensorBase {

  WaleTensorMagnitude( const Expr::Tag& vel1tag,
                       const Expr::Tag& vel2tag,
                       const Expr::Tag& vel3tag);  
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
               const Expr::Tag& vel1tag,
               const Expr::Tag& vel2tag,
               const Expr::Tag& vel3tag );
      ~Builder(){}
      Expr::ExpressionBase* build() const;
      
    private:
      const Expr::Tag v1t_, v2t_, v3t_;
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
 Vreman 2004, An eddy-viscosity subgrid-scale model for turbulent shear ﬂow:
 Algebraic theory and applications
 *
 */
class VremanTensorMagnitude : public StrainTensorBase {
  
  VremanTensorMagnitude( const Expr::Tag& vel1tag,
                         const Expr::Tag& vel2tag,
                         const Expr::Tag& vel3tag);  
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
             const Expr::Tag& vel1tag,
             const Expr::Tag& vel2tag,
             const Expr::Tag& vel3tag );
    ~Builder(){}
    Expr::ExpressionBase* build() const;
    
  private:
    const Expr::Tag v1t_, v2t_, v3t_;
  };
  
  ~VremanTensorMagnitude();
  void evaluate();
};
#endif // StrainTensorMagnitude_Expr_h
