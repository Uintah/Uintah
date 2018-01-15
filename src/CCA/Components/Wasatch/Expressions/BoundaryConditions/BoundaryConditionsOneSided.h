/*
 * The MIT License
 *
 * Copyright (c) 2018 The University of Utah
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

#ifndef SRC_CCA_COMPONENTS_WASATCH_EXPRESSIONS_BOUNDARYCONDITIONS_BOUNDARYCONDITIONONESIDED_H_
#define SRC_CCA_COMPONENTS_WASATCH_EXPRESSIONS_BOUNDARYCONDITIONS_BOUNDARYCONDITIONONESIDED_H_

/*
 *  NOTE: we have split the implementations into several files to reduce the size
 *        of object files when compiling with ncvv, since we were crashing the
 *        linker in some cases.
 */

#include "BoundaryConditionBase.h"
#include <expression/Expression.h>

namespace WasatchCore{

  //-------------------------------------------------------------------------------------------------
  /**
   *  \class    BCOneSidedConvFluxDiv
   *  \ingroup  Expressions
   *  \author   Tony Saad
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
   *  \class    BCOneSidedGradP
   *  \ingroup  Expressions
   *  \author   Tony Saad
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

    DECLARE_FIELDS( FieldT, p_ )
  };


  //-------------------------------------------------------------------------------------------------
  /**
   *  \class    OneSidedDirichletBC
   *  \ingroup  Expressions
   *  \author   Amir Biglari
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

} // namespace WasatchCore

#endif /* SRC_CCA_COMPONENTS_WASATCH_EXPRESSIONS_BOUNDARYCONDITIONS_BOUNDARYCONDITIONONESIDED_H_ */
