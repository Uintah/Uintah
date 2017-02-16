/*
 * The MIT License
 *
 * Copyright (c) 2012-2016 The University of Utah
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

#ifndef BoundaryConditions_h
#define BoundaryConditions_h

#include "BoundaryConditionBase.h"
#include <expression/Expression.h>

//-------------------------------------------------------------------------------------------------
/**
 *  \class 	ConstantBC
 *  \ingroup 	Expressions
 *  \author 	Tony Saad
 *  \date    September, 2012
 *
 *  \brief Provides an expression to set basic Dirichlet and Neumann boundary
 *  conditions. Given a BCValue, we set the ghost value such that
 *  \f$ f[ghost] = \alpha f[interior] + \beta BCValue \f$
 *
 *  \tparam FieldT - The type of field for the expression on which this bc applies.
 */
template< typename FieldT >
class ConstantBC : public BoundaryConditionBase<FieldT>
{
public:
  ConstantBC( const double bcValue) :
  bcValue_(bcValue)
  {
    this->set_gpu_runnable(true);
  }
  
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     * @param result Tag of the resulting expression.
     * @param bcValue   constant boundary condition value.
     */
    Builder( const Expr::Tag& resultTag, const double bcValue )
    : ExpressionBuilder(resultTag),
      bcValue_(bcValue)
    {}
    inline Expr::ExpressionBase* build() const{ return new ConstantBC(bcValue_); }
  private:
    const double bcValue_;
  };
  
  ~ConstantBC(){}
  void evaluate();
  
private:
  const double bcValue_;
};

//-------------------------------------------------------------------------------------------------
/**
 *  \class 	OneSidedDirichletBC
 *  \ingroup 	Expressions
 *  \author 	Amir Biglari
 *  \date    May, 2014
 *
 *  \brief Implements a dirichlet boundary condition but on the interior cell.
 *
 *  \tparam FieldT - The type of field for the expression on which this bc applies.
 */

template< typename FieldT >
class OneSidedDirichletBC : public BoundaryConditionBase<FieldT>
{
public:
  OneSidedDirichletBC( const double bcValue) :
  bcValue_(bcValue)
  {}
  
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     * @param result Tag of the resulting expression.
     * @param bcValue   constant boundary condition value.
     */
    Builder( const Expr::Tag& resultTag,
            const double bcValue )
    : ExpressionBuilder(resultTag),
      bcValue_(bcValue)
    {}
    inline Expr::ExpressionBase* build() const{ return new OneSidedDirichletBC(bcValue_); }
  private:
    const double bcValue_;
  };
  
  ~OneSidedDirichletBC(){}
  void evaluate();
  
private:
  const double bcValue_;
};

//-------------------------------------------------------------------------------------------------
/**
 *  \class 	LinearBC
 *  \ingroup 	Expressions
 *  \author 	Tony Saad
 *  \date    September, 2012
 *
 *  \brief Implements a linear profile at the boundary.
 *
 *  \tparam FieldT - The type of field for the expression on which this bc applies.
 */
template< typename FieldT >
class LinearBC : public BoundaryConditionBase<FieldT>
{
  LinearBC( const Expr::Tag& indepVarTag,
            const double a,
            const double b )
  : a_(a), b_(b)
  {
    this->set_gpu_runnable(true);
    x_ = this->template create_field_request<FieldT>(indepVarTag);
  }
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& resultTag,
             const Expr::Tag& indepVarTag,
             const double a,
             const double b )
    : ExpressionBuilder(resultTag),
      indepVarTag_ (indepVarTag),
      a_(a), b_(b)
    {}
    inline Expr::ExpressionBase* build() const{ return new LinearBC(indepVarTag_, a_, b_); }
  private:
    const Expr::Tag indepVarTag_;
    const double a_, b_;
  };
  
  ~LinearBC(){}
  void evaluate();
  
private:
  DECLARE_FIELD(FieldT, x_)
  const double a_, b_;
};

//-------------------------------------------------------------------------------------------------
/**
 *  \class 	ParabolicBC
 *  \ingroup 	Expressions
 *  \author 	Tony Saad
 *  \date    September, 2012
 *
 *  \brief Implements a parabolic profile at the boundary of the form: a*x^2 + b*x + c.
 *
 *  \tparam FieldT - The type of field for the expression on which this bc applies.
 */
template< typename FieldT >
class ParabolicBC : public BoundaryConditionBase<FieldT>
{
  ParabolicBC( const Expr::Tag& indepVarTag,
               const double a, const double b,
               const double c, const double x0 )
  : a_(a), b_(b), c_(c), x0_(x0)
  {
    this->set_gpu_runnable(true);
    x_ = this->template create_field_request<FieldT>(indepVarTag);
  }
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  \param resultTag The tag of the resulting expression computed here.
     *  \param indepVarTag The tag of the independent variable
     *  \param a  The coefficient of x^2 in the parabolic formula
     *  \param b  The coefficient of x in the parabolic formula
     *  \param c  The constant in the parabolic formula
     *  \param x0 The value of the point (independent variable) where the parabola assumes its maximum/minimum
     */
    Builder( const Expr::Tag& resultTag,
             const Expr::Tag& indepVarTag,
            const double a, const double b,
            const double c, const double x0 )
    : ExpressionBuilder(resultTag),
      indepVarTag_ (indepVarTag),
      a_(a), b_(b), c_(c), x0_(x0)
    {}
    inline Expr::ExpressionBase* build() const{ return new ParabolicBC(indepVarTag_, a_, b_, c_, x0_); }
  private:
    const Expr::Tag indepVarTag_;
    const double a_, b_, c_, x0_;
  };
  
  ~ParabolicBC(){}
  void evaluate();
  
private:
  DECLARE_FIELD(FieldT, x_)
  const double a_, b_, c_, x0_;
};

//-------------------------------------------------------------------------------------------------
/**
 *  \class 	PowerLawBC
 *  \ingroup 	Expressions
 *  \author 	Tony Saad
 *  \date    September, 2012
 *
 *  \brief Implements a powerlaw profile at the boundary.
 *
 *  \tparam FieldT - The type of field for the expression on which this bc applies.
 */
template< typename FieldT >
class PowerLawBC : public BoundaryConditionBase<FieldT>
{
  PowerLawBC( const Expr::Tag& indepVarTag,
              const double x0, const double phiCenter,
             const double halfHeight, const double n )
  : x0_(x0), phic_(phiCenter), R_(halfHeight), n_(n)
  {
    this->set_gpu_runnable(true);
    x_ = this->template create_field_request<FieldT>(indepVarTag);
  }
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& resultTag,
             const Expr::Tag& indepVarTag,
             const double x0, const double phiCenter,
             const double halfHeight, const double n)
    : ExpressionBuilder(resultTag),
      indepVarTag_ (indepVarTag),
      x0_(x0), phic_(phiCenter), R_(halfHeight), n_(n)
    {}
    inline Expr::ExpressionBase* build() const{ return new PowerLawBC(indepVarTag_, x0_, phic_, R_, n_); }
  private:
    const Expr::Tag indepVarTag_;
    const double x0_, phic_, R_, n_;
  };
  
  ~PowerLawBC(){}
  void evaluate();
  
private:
  DECLARE_FIELD(FieldT, x_)
  const double    x0_, phic_, R_, n_;
};

//-------------------------------------------------------------------------------------------------
/**
 *  \class 	BCCopier
 *  \ingroup 	Expressions
 *  \author 	Amir Biglari
 *  \date    September, 2012
 *
 *  \brief Provides a mechanism to copy boundary values from one field to another.
 *
 *  \tparam FieldT - The type of field for the expression on which this bc applies.
 */
template< typename FieldT >
class BCCopier : public BoundaryConditionBase<FieldT>
{
  BCCopier( const Expr::Tag& srcTag )
  {
    this->set_gpu_runnable(true);
     src_ = this->template create_field_request<FieldT>(srcTag);
  }
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& resultTag,
             const Expr::Tag& srcTag )
    : ExpressionBuilder(resultTag),
      srcTag_ (srcTag)
    {}
    inline Expr::ExpressionBase* build() const{ return new BCCopier(srcTag_); }
  private:
    const Expr::Tag srcTag_;
  };
  
  ~BCCopier(){}
  void evaluate();
private:
  DECLARE_FIELD(FieldT, src_)
};

//-------------------------------------------------------------------------------------------------
/**
 *  \class 	BCPrimVar
 *  \ingroup 	Expressions
 *  \author 	Tony Saad
 *  \date    March, 2014
 *
 *  \brief Provides a mechanism to copy boundary values from one field to another.
 *
 *  \tparam FieldT - the type of field for the RHS.
 */
template< typename FieldT >
class BCPrimVar
: public BoundaryConditionBase<FieldT>
{
  BCPrimVar( const Expr::Tag& srcTag,
           const Expr::Tag& densityTag) :
  hasDensity_(densityTag != Expr::Tag() )
  {
    this->set_gpu_runnable(true);
    src_ = this->template create_field_request<FieldT>(srcTag);
    if (hasDensity_)  rho_ = this->template create_field_request<SVolField>(densityTag);
  }
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& resultTag,
            const Expr::Tag& srcTag,
            const Expr::Tag densityTag = Expr::Tag() )
    : ExpressionBuilder(resultTag),
    srcTag_ (srcTag),
    densityTag_(densityTag)
    {}
    inline Expr::ExpressionBase* build() const{ return new BCPrimVar(srcTag_, densityTag_); }
  private:
    const Expr::Tag srcTag_, densityTag_;
  };
  
  ~BCPrimVar(){}
  void evaluate();
  void bind_operators( const SpatialOps::OperatorDatabase& opdb );
private:
  const bool hasDensity_;
  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, SVolField, FieldT >::type DenInterpT;
  const DenInterpT* rhoInterpOp_;

  typedef typename SpatialOps::BasicOpTypes<FieldT>::GradX      GradX;
  typedef typename SpatialOps::BasicOpTypes<FieldT>::GradY      GradY;
  typedef typename SpatialOps::BasicOpTypes<FieldT>::GradZ      GradZ;
  
  typedef typename SpatialOps::NeboBoundaryConditionBuilder<GradX> Neum2XOpT;
  typedef typename SpatialOps::NeboBoundaryConditionBuilder<GradY> Neum2YOpT;
  typedef typename SpatialOps::NeboBoundaryConditionBuilder<GradZ> Neum2ZOpT;
  const Neum2XOpT* neux_;
  const Neum2YOpT* neuy_;
  const Neum2ZOpT* neuz_;
  
  DECLARE_FIELD(FieldT, src_)
  DECLARE_FIELD(SVolField, rho_)
};

//-------------------------------------------------------------------------------------------------
/**
 *  \class 	BCOneSidedConvFluxDiv
 *  \ingroup 	Expressions
 *  \author 	Tony Saad
 *  \date    March, 2014
 *
 *  \brief One sided convective flux BC. Used in NSCBCs.
 *
 *  \tparam FieldT - the type of field for the RHS.
 *  \tparam DivOpT - the type of divergence operator to invert.
 */
template< typename FieldT, typename DivOpT >
class BCOneSidedConvFluxDiv
: public BoundaryConditionBase<FieldT>
{
  BCOneSidedConvFluxDiv( const Expr::Tag& uTag,
                         const Expr::Tag& phiTag )
  {
    this->set_gpu_runnable(true);
    u_ = this->template create_field_request<FieldT>(uTag);
    phi_ = this->template create_field_request<FieldT>(phiTag);
  }
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& resultTag,
             const Expr::Tag& uTag,
             const Expr::Tag& phiTag )
    : ExpressionBuilder(resultTag),
    uTag_  (uTag),
    phiTag_(phiTag)
    {}
    inline Expr::ExpressionBase* build() const{ return new BCOneSidedConvFluxDiv(uTag_, phiTag_); }
  private:
    const Expr::Tag uTag_, phiTag_;
  };
  
  ~BCOneSidedConvFluxDiv(){}
  void evaluate();
  void bind_operators( const SpatialOps::OperatorDatabase& opdb );
private:
  const DivOpT*  divOp_;
  
  DECLARE_FIELDS(FieldT, u_, phi_)
};

//-------------------------------------------------------------------------------------------------
/**
 *  \class 	BCOneSidedGradP
 *  \ingroup 	Expressions
 *  \author 	Tony Saad
 *  \date    March, 2014
 *
 *  \brief One sided gradp BC. Used in NSCBCs.
 *
 *  \tparam FieldT - the type of field for the RHS.
 *  \tparam GradOpT - the type of gradient operator to invert.
 */
template< typename FieldT, typename GradOpT >
class BCOneSidedGradP
: public BoundaryConditionBase<FieldT>
{
  BCOneSidedGradP( const Expr::Tag& pTag )
  {
    this->set_gpu_runnable(true);
    p_ = this->template create_field_request<FieldT>(pTag);
  }
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& resultTag,
            const Expr::Tag& pTag)
    : ExpressionBuilder(resultTag),
    pTag_  (pTag)
    {}
    inline Expr::ExpressionBase* build() const{ return new BCOneSidedGradP(pTag_); }
  private:
    const Expr::Tag pTag_;
  };
  
  ~BCOneSidedGradP(){}
  void evaluate();
  void bind_operators( const SpatialOps::OperatorDatabase& opdb );
private:
  const GradOpT*  gradOp_;
  
  DECLARE_FIELDS(FieldT, p_);
};

//-------------------------------------------------------------------------------------------------
/**
 *  \class 	ConstantBCNew
 *  \ingroup 	Expressions
 *  \author 	Tony Saad
 *  \date    Dec, 2016
 *
 *  \brief Prototype for new style of Boundary Conditions. 
 *
 *  \tparam FieldT - the type of field for the RHS.
 */
template< typename FieldT, typename OpT >
class ConstantBCNew
: public BoundaryConditionBase<FieldT>
{
  ConstantBCNew(double bcVal):
  bcVal_(bcVal)
  {
    this->set_gpu_runnable(true);
  }
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& resultTag, double bcVal)
    : ExpressionBuilder(resultTag), bcVal_(bcVal)
    {}
    inline Expr::ExpressionBase* build() const{ return new ConstantBCNew(bcVal_); }
  private:
    double bcVal_;
  };
  
  ~ConstantBCNew(){}
  void evaluate()
  {
    FieldT& lhs = this->value();
    (*op_)(*this->interiorSvolSpatialMask_, lhs, bcVal_, this->isMinusFace_);
  }
  void bind_operators( const SpatialOps::OperatorDatabase& opdb )
  {
    op_ = opdb.retrieve_operator<OpT>();
  }
private:
  const OpT* op_;
  double bcVal_;
};

//-------------------------------------------------------------------------------------------------
#endif // BoundaryConditions_h
