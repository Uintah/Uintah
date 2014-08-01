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

#ifndef TimeAdvance_Expr_h
#define TimeAdvance_Expr_h

#include <expression/Expression.h>

#include <spatialops/structured/FVStaggered.h>

#include <CCA/Components/Wasatch/TimeIntegratorTools.h>

/**
 *  \class TimeAdvance
 *  \ingroup Expressions
 *
 *  \brief calculates \f$ \nabla\cdot\mathbf{u} \f$
 *
 *  \tparam FieldT the field type for the TimeAdvance (nominally the scalar volume field)
 *  \tparam XmomT  the field type for the first velocity component
 *  \tparam YmomT  the field type for the second velocity component
 *  \tparam ZmomT  the field type for the third velocity component
 */
template< typename FieldT >
class TimeAdvance
 : public Expr::Expression<FieldT>
{
  typedef typename SpatialOps::SingleValueField SingleValue;
  const Expr::Tag phioldt_, rhst_, dtt_, rkstaget_;

  const Wasatch::TimeIntegrator timeIntInfo_;
  int rkStage_;
  double a_, b_;
  
  const SingleValue* dt_;
  const SingleValue* rks_;
  
  const FieldT* phiOld_;
  const FieldT* phiNew_;
  const FieldT* rhs_;
  
  TimeAdvance( const std::string& solnVarName,
               const Expr::Tag& rhsTag,
               const Wasatch::TimeIntegrator timeIntInfo );

  TimeAdvance( const std::string& solnVarName,
              const Expr::Tag& phiOldTag,
              const Expr::Tag& rhsTag,
              const Wasatch::TimeIntegrator timeIntInfo );

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:

    /**
     *  \param XmomTag the velocity corresponding to the XmomT template parameter
     *  \param YmomTag the velocity corresponding to the YmomT template parameter
     *  \param ZmomTag the velocity corresponding to the ZmomT template parameter
     */
    Builder( const Expr::Tag& result,
             const Expr::Tag& rhsTag,
             const Wasatch::TimeIntegrator timeIntInfo );

    Builder( const Expr::Tag& result,
             const Expr::Tag& phiOldTag,
             const Expr::Tag& rhsTag,
             const Wasatch::TimeIntegrator timeIntInfo );
    
    ~Builder(){}
    Expr::ExpressionBase* build() const;

  private:
    const std::string solnVarName_;
    const Expr::Tag phioldt_, rhst_;
    const Wasatch::TimeIntegrator timeIntInfo_;
  };

  ~TimeAdvance();

  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void set_integrator_stage(const int rkStage){
    rkStage_ = rkStage;
  }
  void evaluate();
};

#endif // TimeAdvance_Expr_h
