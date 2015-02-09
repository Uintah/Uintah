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

#ifndef Velocity_Magnitude_Expr_h
#define Velocity_Magnitude_Expr_h

#include <expression/Expression.h>
#include <spatialops/structured/stencil/FVStaggeredOperatorTypes.h>

/**
 *  \class 	VelocityMagnitude
 *  \author Tony Saad
 *  \date 	 February, 2012
 *  \ingroup	Expressions
 *
 *  \brief Calculates a cell centered velocity magnitude.
 *
 *  The Velocity magnitude is given by:
 *  \f[ \left |  \mathbf{u} \right | = \sqrt(\mathbf u \cdot \mathbf u) \f]     
 *
 *  \tparam FieldT  The type of field for the velocity magnitue - usually SVOL.
 *  \tparam Vel1T   The type of field for the first velocity component.
 *  \tparam Vel2T   The type of field for the second velocity component.
 *  \tparam Vel3T   The type of field for the second velocity component. 
 */
template< typename FieldT,
typename Vel1T,
typename Vel2T,
typename Vel3T >
class VelocityMagnitude
: public Expr::Expression<FieldT>
{
  
protected:
  const Expr::Tag vel1t_, vel2t_, vel3t_;
  const bool is3d_;

  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, Vel1T, FieldT >::type InterpVel1T2FieldT;
  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, Vel2T, FieldT >::type InterpVel2T2FieldT;
  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, Vel3T, FieldT >::type InterpVel3T2FieldT;
    
  const Vel1T* vel1_;
  const Vel2T* vel2_;
  const Vel3T* vel3_;
  
  const InterpVel1T2FieldT* interpVel1T2FieldTOp_;
  const InterpVel2T2FieldT* interpVel2T2FieldTOp_;
  const InterpVel3T2FieldT* interpVel3T2FieldTOp_;
  
  VelocityMagnitude( const Expr::Tag& vel1tag,
                     const Expr::Tag& vel2tag,
                     const Expr::Tag& vel3tag );
  
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    
    /**
     *  \param vel1tag the velocity corresponding to the Vel1T template parameter
     *  \param vel2tag the velocity corresponding to the Vel2T template parameter
     *  \param vel3tag the velocity corresponding to the Vel3T template parameter
     */
    Builder( const Expr::Tag& result,
             const Expr::Tag& vel1tag,
             const Expr::Tag& vel2tag,
             const Expr::Tag& vel3tag );
    ~Builder(){}
    Expr::ExpressionBase* build() const;
    
  private:
    const Expr::Tag v1t_, v2t_, v3t_;
  };
  
  ~VelocityMagnitude();
  
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
};

#endif // VelocityMagnitude_Expr_h
