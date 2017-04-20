/*
 * The MIT License
 *
 * Copyright (c) 2014-2017 The University of Utah
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

#ifndef TimeAdvance_Expr_h
#define TimeAdvance_Expr_h

#include <expression/Expression.h>

#include <spatialops/structured/FVStaggered.h>

#include <CCA/Components/Wasatch/TimeIntegratorTools.h>

/**
 *  \class TimeAdvance
 *  \author Tony Saad
 *  \ingroup Expressions
 *
 *  \brief calculates an updated solution variable
 *
 *  \tparam FieldT the field type for the TimeAdvance (nominally the scalar volume field)
 */
template< typename FieldT >
class TimeAdvance
 : public Expr::Expression<FieldT>
{
  typedef typename SpatialOps::SingleValueField SingleValue;
  const Expr::Tag phiOldt_, rhst_, dtt_, rkstaget_;

  const WasatchCore::TimeIntegrator timeIntInfo_;
  
//  const SingleValue* dt_;
//  const SingleValue* rkStage_;
  
  DECLARE_FIELDS(FieldT, phiOld_, phiNew_, rhs_)
  DECLARE_FIELDS(SingleValue, dt_, rkStage_)
  
  TimeAdvance( const std::string& solnVarName,
               const Expr::Tag& rhsTag,
               const WasatchCore::TimeIntegrator timeIntInfo );

  TimeAdvance( const std::string& solnVarName,
               const Expr::Tag& phiOldTag,
               const Expr::Tag& rhsTag,
               const WasatchCore::TimeIntegrator timeIntInfo );

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:

    /**
     *  \param result tag for the variable at the new time state
     *  \param rhsTag the tag for the RHS evaluation of this variable's evolution PDE.
     *  \param timeIntInfo
     */
    Builder( const Expr::Tag& result,
             const Expr::Tag& rhsTag,
             const WasatchCore::TimeIntegrator timeIntInfo );

    Builder( const Expr::Tag& result,
             const Expr::Tag& phiOldTag,
             const Expr::Tag& rhsTag,
             const WasatchCore::TimeIntegrator timeIntInfo );
    
    ~Builder(){}
    Expr::ExpressionBase* build() const;

  private:
    const std::string solnVarName_;
    const Expr::Tag phiOldt_, rhst_;
    const WasatchCore::TimeIntegrator timeIntInfo_;
  };

  ~TimeAdvance();
  void evaluate();
};

#endif // TimeAdvance_Expr_h
