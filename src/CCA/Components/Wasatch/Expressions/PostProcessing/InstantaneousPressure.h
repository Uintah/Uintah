/*
 * The MIT License
 *
 * Copyright (c) 1997-2026 The University of Utah
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

#ifndef InstantaneousPressure_Expr_h
#define InstantaneousPressure_Expr_h

#include <expression/Expression.h>

#include <spatialops/structured/stencil/FVStaggeredOperatorTypes.h>

/**
 *  \class InstantaneousPressure
 *  \ingroup PostProcessing
 *  \author Mokbel Karam
 *
 *  \brief construct the instantaneous pressure value using pseudo-pressure values from previous time levels.
 *
 */
class InstantaneousPressure
 : public Expr::Expression<SpatialOps::SVolField>
{
  typedef SpatialOps::SVolField PFieldT;
  typedef SpatialOps::SingleValueField TimeField;
  typedef boost::shared_ptr<const Expr::FieldRequest<PFieldT> > fieldRequestPtr;
  typedef boost::shared_ptr<const Expr::FieldRequest<SpatialOps::SpatialField<SpatialOps::SingleValue> > > timeFieldRequestPtr;
  
  const int order_;

  Expr::TagList oldPressureTags_;
  
  DECLARE_FIELD(TimeField, timestep_)
  DECLARE_VECTOR_OF_FIELDS( PFieldT, old_pressure_Fields_ ) 

  // constructor
  InstantaneousPressure(const int order);

public:
  class Builder : public Expr::ExpressionBuilder
  {
    const int order_;
  public:
    /**
     *  \param result the result of this expression
     *  \param order the order of the constructed instantaneous pressure value
     */
    Builder( const Expr::Tag& result,
            const int order);
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  };

  ~InstantaneousPressure();
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
  
};

#endif // InstantaneousPressure_Expr_h
