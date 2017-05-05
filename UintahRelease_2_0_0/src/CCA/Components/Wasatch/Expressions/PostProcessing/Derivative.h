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

#ifndef Interpolate_Expr_h
#define Interpolate_Expr_h

#include <expression/Expression.h>
#include <spatialops/OperatorDatabase.h>
#include <spatialops/particles/ParticleOperators.h>
#include <spatialops/structured/SpatialFieldStore.h>
#include <spatialops/particles/ParticleFieldTypes.h>
#include <spatialops/structured/stencil/FVStaggeredOperatorTypes.h>
//-- Wasatch Includes --//
#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>

/**
 *  \class   Derivative
 *  \author  Tony Saad
 *  \date    October, 2016
 *  \ingroup Expressions
 *
 *  \brief An expression that computes a derivative. Used for NSCBCs.
 *  \tparam SrcT: Source field type.
    \tparam DestT: Destination field type.
 *
 */
template< typename SrcT, typename DestT, typename DirT >
class Derivative
: public Expr::Expression<DestT>
{
  DECLARE_FIELD(SrcT, src_)
  
  typedef typename SpatialOps::OperatorTypeBuilder< typename WasatchCore::GradOpSelector<SrcT, DirT>::Gradient, SrcT, DestT >::type DerivativeT;

  const DerivativeT* derivativeOp_;
  
  Derivative( const Expr::Tag& srctag );
  
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
  
  ~Derivative();
  
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
};

#endif // Interpolate_Expr_h
