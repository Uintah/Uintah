/*
 * The MIT License
 *
 * Copyright (c) 2013-2018 The University of Utah
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

#ifndef TabPropsHeatLossEvaluator_Expr_h
#define TabPropsHeatLossEvaluator_Expr_h

#include <tabprops/TabPropsConfig.h>
#include <tabprops/StateTable.h>

#include <expression/Expression.h>

/**
 *  \ingroup	Expressions
 *  \class  	TabPropsHeatLossEvaluator
 *  \author 	James C. Sutherland
 *  \date   	June, 2013
 *
 *  \brief Evaluates the heat loss for tables that have non-adiabatic
 *         information in them.
 *
 *  \tparam FieldT the type of field for both the independent and
 *          dependent variables.
 */
template< typename FieldT >
class TabPropsHeatLossEvaluator
 : public Expr::Expression<FieldT>
{
  DECLARE_VECTOR_OF_FIELDS(FieldT, indepVars_)

  const size_t hlIx_;
  std::vector<double> ivarsPoint_;
  const InterpT* const enthEval_;
  const InterpT* const adEnthEval_;
  const InterpT* const sensEnthEval_;

  TabPropsHeatLossEvaluator( const InterpT* const adEnthInterp,
                             const InterpT* const sensEnthInterp,
                             const InterpT* const enthInterp,
                             const size_t hlIx,
                             const Expr::TagList& ivarNames );

public:
  class Builder : public Expr::ExpressionBuilder
  {
    const InterpT& adEnthInterp_;
    const InterpT& sensEnthInterp_;
    const InterpT& enthInterp_;
    const size_t hlIx_;
    const Expr::TagList ivarNames_;
  public:
    Builder( const Expr::Tag& result,
             const InterpT& adEnthInterp,
             const InterpT& sensEnthInterp,
             const InterpT& enthInterp,
             const size_t hlIx,
             const Expr::TagList& ivarNames );
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  };

  ~TabPropsHeatLossEvaluator();
  void evaluate();
};

#endif // TabPropsHeatLossEvaluator_Expr_h
