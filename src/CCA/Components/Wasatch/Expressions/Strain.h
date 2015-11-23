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

#ifndef Strain_Expr_h
#define Strain_Expr_h

#include <expression/Expression.h>
#include <spatialops/structured/stencil/FVStaggeredOperatorTypes.h>

/**
 *  \class 	Strain
 *  \author 	James C. Sutherland, Tony Saad
 *  \date 	 June 2012, (Originally created: December, 2010).
 *  \ingroup	Expressions
 *
 *  \brief Calculates a component of the Strain tensor.
 *
 *  The Strain tensor is given as
 *  \f[ S_{ij} = - \left( \frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i} \right) + \frac{2}{3} \delta_{ij} \frac{\partial u_k}{\partial x_k} \f]
 *
 *  \tparam StrainT The type of field for this Strain component.
 *  \tparam Vel1T   The type of field for the first velocity component.
 *  \tparam Vel2T   The type of field for the second velocity component.
 */
template< typename StrainT,
typename Vel1T,
typename Vel2T >
class Strain
: public Expr::Expression<StrainT>
{
  
  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, SVolField, StrainT >::type  SVol2StrainInterpT;
  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Gradient,    Vel1T,     StrainT >::type  Vel1GradT;
  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Gradient,    Vel2T,     StrainT >::type  Vel2GradT;
  
  const SVol2StrainInterpT* svolInterpOp_; ///< Interpolate viscosity to the face where we are building the Strain
  const Vel1GradT*          vel1GradOp_;   ///< Calculate the velocity gradient dui/dxj at the Strain face
  const Vel2GradT*          vel2GradOp_;   ///< Calculate the velocity gradient duj/dxi at the Strain face
  
  DECLARE_FIELD( Vel1T, u1_ )
  DECLARE_FIELD( Vel2T, u2_ )
  
  Strain( const Expr::Tag& vel1Tag,
          const Expr::Tag& vel2Tag );
  
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  \param result the Strain component being calculated
     *  \param viscTag the viscosity
     *  \param vel1Tag the first velocity component
     *  \param vel2Tag the second velocity component
     *
     *  \param dilTag the dilatation tag. If supplied, the
     *         dilatational Strain term will be added.
     */
    Builder( const Expr::Tag& result,
             const Expr::Tag& vel1Tag,
             const Expr::Tag& vel2Tag,
             const Expr::Tag& dilTag );
    ~Builder(){}
    Expr::ExpressionBase* build() const;
    
  private:
    const Expr::Tag vel1t_, vel2t_;
  };
  
  ~Strain();
  
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
  
};



/**
 *  \brief specialized version of the Strain tensor for normal Strain.
 */
template< typename StrainT,
typename VelT >
class Strain< StrainT, VelT, VelT >
: public Expr::Expression<StrainT>
{
 
  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, SVolField, StrainT >::type  SVol2StrainInterpT;
  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Gradient,    VelT,      StrainT >::type  VelGradT;
  
  const SVol2StrainInterpT* svolInterpOp_; ///< Interpolate viscosity to the face where we are building the Strain
  const VelGradT*           velGradOp_;    ///< Calculate the velocity gradient dui/dxj at the Strain face
  
  DECLARE_FIELD( VelT,      u_   )
  DECLARE_FIELD( SVolField, dil_ )
  
  Strain( const Expr::Tag& velTag,
          const Expr::Tag& dilTag );
  
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  \param result the Strain component calculated here
     *  \param vel1Tag the first velocity component
     *  \param vel2Tag the second velocity component
     *
     *  \param dilTag the dilatation tag. If supplied, the
     *         dilatational Strain term will be added.
     *
     *  note that in this case, the second velocity component will be
     *  ignored.  It is kept for consistency with the off-diagonal
     *  Strain builder.
     */
    Builder( const Expr::Tag& result,
             const Expr::Tag& vel1Tag,
             const Expr::Tag& vel2Tag,
             const Expr::Tag& dilTag );
    ~Builder(){}
    Expr::ExpressionBase* build() const;
    
  private:
    const Expr::Tag velt_, dilt_;
  };
  
  ~Strain();
  
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
};

#endif // Strain_Expr_h
