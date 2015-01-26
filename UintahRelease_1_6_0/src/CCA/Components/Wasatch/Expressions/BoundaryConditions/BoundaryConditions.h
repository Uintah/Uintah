/*
 * The MIT License
 *
 * Copyright (c) 2012-2015 The University of Utah
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
  {}
  
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
    Expr::ExpressionBase* build() const{ return new ConstantBC(bcValue_); }
  private:
    const double bcValue_;
  };
  
  ~ConstantBC(){}
  void advertise_dependents( Expr::ExprDeps& exprDeps ){}
  void bind_fields( const Expr::FieldManagerList& fml ){}
  void evaluate();
  
private:
  const double bcValue_;
};

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
    Expr::ExpressionBase* build() const{ return new OneSidedDirichletBC(bcValue_); }
  private:
    const double bcValue_;
  };
  
  ~OneSidedDirichletBC(){}
  void advertise_dependents( Expr::ExprDeps& exprDeps ){}
  void bind_fields( const Expr::FieldManagerList& fml ){}
  void evaluate();
  
private:
  const double bcValue_;
};

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
  : indepVarTag_ (indepVarTag),
    a_(a), b_(b)
  {}
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
    Expr::ExpressionBase* build() const{ return new LinearBC(indepVarTag_, a_, b_); }
  private:
    const Expr::Tag indepVarTag_;
    const double a_, b_;
  };
  
  ~LinearBC(){}
  void advertise_dependents( Expr::ExprDeps& exprDeps ){ exprDeps.requires_expression( indepVarTag_ );}
  void bind_fields( const Expr::FieldManagerList& fml ){ x_ = &fml.template field_ref<FieldT>( indepVarTag_ );}
  void evaluate();
  
private:
  const FieldT* x_;
  const Expr::Tag indepVarTag_;
  const double a_, b_;
};


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
  : indepVarTag_ (indepVarTag),
    a_(a), b_(b), c_(c), x0_(x0)
  {}
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
    Expr::ExpressionBase* build() const{ return new ParabolicBC(indepVarTag_, a_, b_, c_, x0_); }
  private:
    const Expr::Tag indepVarTag_;
    const double a_, b_, c_, x0_;
  };
  
  ~ParabolicBC(){}
  void advertise_dependents( Expr::ExprDeps& exprDeps ){ exprDeps.requires_expression( indepVarTag_ );}
  void bind_fields( const Expr::FieldManagerList& fml ){ x_ = &fml.template field_ref<FieldT>( indepVarTag_ );}  
  void evaluate();
  
private:
  const FieldT* x_;
  const Expr::Tag indepVarTag_;
  const double a_, b_, c_, x0_;
};

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
  : indepVarTag_ (indepVarTag),
    x0_(x0), phic_(phiCenter), R_(halfHeight), n_(n)
  {}
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
    Expr::ExpressionBase* build() const{ return new PowerLawBC(indepVarTag_, x0_, phic_, R_, n_); }
  private:
    const Expr::Tag indepVarTag_;
    const double x0_, phic_, R_, n_;
  };
  
  ~PowerLawBC(){}
  void advertise_dependents( Expr::ExprDeps& exprDeps ){ exprDeps.requires_expression( indepVarTag_ );}
  void bind_fields( const Expr::FieldManagerList& fml ){ x_ = &fml.template field_ref<FieldT>( indepVarTag_ );}
  void evaluate();
  
private:
  const FieldT*   x_;
  const Expr::Tag indepVarTag_;
  const double    x0_, phic_, R_, n_;
};

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
  BCCopier( const Expr::Tag& srcTag ) : srcTag_(srcTag){}
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& resultTag,
             const Expr::Tag& srcTag )
    : ExpressionBuilder(resultTag),
      srcTag_ (srcTag)
    {}
    Expr::ExpressionBase* build() const{ return new BCCopier(srcTag_); }
  private:
    const Expr::Tag srcTag_;
  };
  
  ~BCCopier(){}
  void advertise_dependents( Expr::ExprDeps& exprDeps ){ exprDeps.requires_expression( srcTag_ ); }
  void bind_fields( const Expr::FieldManagerList& fml ){ src_ = &fml.template field_ref<FieldT>(srcTag_); }
  void evaluate();
private:
  const FieldT* src_;
  const Expr::Tag srcTag_;
};

#endif // BoundaryConditions_h
