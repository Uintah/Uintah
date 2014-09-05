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

#ifndef Interpolate_Expr_h
#define Interpolate_Expr_h

#include <expression/Expression.h>
#include <spatialops/structured/FVStaggeredOperatorTypes.h>
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>

/**
 *  \class   InterpolateExpression
 *  \author  Tony Saad
 *  \date    February, 2012
 *  \ingroup Expressions
 *
 *  \brief An expression that interpolates between different field types.
           For example, this can be usedto calculate cell centered velocities.
           This expression is currently specialized for staggered-to-cell centered
           interpolation.
 *  \tparam SrcT: Source field type.
    \tparam DestT: Destination field type.
 *
 */
template< typename SrcT, typename DestT >
class InterpolateExpression
: public Expr::Expression<DestT>
{
  const Expr::Tag srct_;
  
  typedef typename SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Interpolant, SrcT, DestT >::type InpterpSrcT2DestT;
  
  const SrcT* src_;
  
  const InpterpSrcT2DestT* InpterpSrcT2DestTOp_;
  
  InterpolateExpression( const Expr::Tag& srctag );
  
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    
    /**
     *  \param srctag  Tag of the source field
     *  \param desttag Tag of the destination field
     */
    Builder( const Expr::Tag& result,
             const Expr::Tag& srctag );
    ~Builder(){}
    Expr::ExpressionBase* build() const;
    
  private:
    const Expr::Tag srct_;
  };
  
  ~InterpolateExpression();
  
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
};

#endif // Interpolate_Expr_h
