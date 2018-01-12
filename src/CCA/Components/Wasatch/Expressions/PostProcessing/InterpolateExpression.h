/*
 * The MIT License
 *
 * Copyright (c) 2012-2018 The University of Utah
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

/**
 *  \class   InterpolateExpression
 *  \author  Tony Saad
 *  \date    February, 2012
 *  \ingroup Expressions
 *
 *  \brief An expression that interpolates between different field types.
           For example, this can be used to calculate cell centered velocities.
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
  DECLARE_FIELD(SrcT, src_)
  
  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, SrcT, DestT >::type InterpSrcT2DestT;
  const InterpSrcT2DestT* interpSrcT2DestTOp_;
  
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
  
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
};

/**
 *  \class   InterpolateParticleExpression
 *  \author  Tony Saad
 *  \date    October, 2014
 *  \ingroup Expressions
 *
 *  \brief An expression computes Eulerian scalars from Lagrangian particle properties.
 *  \tparam DestT: Destination field type - usually SVolField.
 *
 */
template< typename DestT >
class InterpolateParticleExpression
: public Expr::Expression<DestT>
{
  DECLARE_FIELDS(ParticleField, src_, psize_, px_, py_, pz_)
  
  typedef typename SpatialOps::Particle::ParticleToCell<DestT> P2CellOpT;
  P2CellOpT* p2CellOp_; // particle to cell operator

  typedef typename SpatialOps::Particle::ParticlesPerCell<DestT> PPerCellOpT;
  PPerCellOpT* pPerCellOp_; // operator that counts the number of particles per cell
  
  InterpolateParticleExpression( const Expr::Tag& srctag,
                                 const Expr::Tag& particleSizeTag,
                                 const Expr::TagList& particlePositionTags);
  
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    
    /**
     *  \param desttag Tag of the destination field
     */
    Builder( const Expr::Tag& result,
            const Expr::Tag& srctag,
            const Expr::Tag& particleSizeTag,
            const Expr::TagList& particlePositionTags);
    ~Builder(){}
    Expr::ExpressionBase* build() const;
    
  private:
    const Expr::Tag srct_, pSizeTag_;
    const Expr::TagList pPosTags_;
  };
  
  ~InterpolateParticleExpression();
  
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
};

#endif // Interpolate_Expr_h
