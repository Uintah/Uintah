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

#ifndef Kinetic_Energy_Expr_h
#define Kinetic_Energy_Expr_h

#include <expression/Expression.h>

#include "VelocityMagnitude.h"
#include <CCA/Components/Wasatch/Expressions/ReductionBase.h>

//----------------------------------------------------------------------------//

/**
 *  \class  KineticEnergy
 *  \author Tony Saad
 *  \date   June, 2013
 *  \ingroup Expressions
 *
 *  \brief Calculates the cell centered, pointwise, kinetic energy.
 *
 *  \warning This is likely inaccurate for the kinetic energy given that the 
 velocities are staggered. ideally, we'd want to use the individual velocity
 components. The total KE (i.e. volumetric), on the other hand, can be easily calculated
 using nebo syntax without the need for "shift" operators. For this, see 
 the TotalKineticEnergy expression.
 *
 *  The Kinetic Energy is given by:
 *  \f[\frac{1}{2} \mathbf{u}\cdot\mathbf{u} \f]     
 *
 *  \tparam FieldT  The type of field for the kinetic energy - usually SVOL.
 *  \tparam Vel1T   The type of field for the first velocity component.
 *  \tparam Vel2T   The type of field for the second velocity component.
 *  \tparam Vel3T   The type of field for the second velocity component. 
 */
template< typename FieldT,
typename Vel1T,
typename Vel2T,
typename Vel3T >
class KineticEnergy
: public VelocityMagnitude<FieldT,Vel1T,Vel2T,Vel3T>
{
  KineticEnergy( const Expr::Tag& vel1tag,
                 const Expr::Tag& vel2tag,
                 const Expr::Tag& vel3tag );
  
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    
    /**
     *  \param resultTag The tag of the resulting expression
     *  \param vel1Tag   The velocity component corresponding to the Vel1T template parameter
     *  \param vel2Tag   The velocity component corresponding to the Vel2T template parameter
     *  \param vel3Tag   The velocity component corresponding to the Vel3T template parameter
     */
    Builder( const Expr::Tag& resultTag,
             const Expr::Tag& vel1Tag,
             const Expr::Tag& vel2Tag,
             const Expr::Tag& vel3Tag );
    ~Builder(){}
    Expr::ExpressionBase* build() const;
    
  private:
    const Expr::Tag v1t_, v2t_, v3t_;
  };
  
  ~KineticEnergy();
  void evaluate();
}; // Class KineticEnergy

//----------------------------------------------------------------------------//

/**
 *  \class 	 TotalKineticEnergy
 *  \author  Tony Saad
 *  \date 	 June, 2013
 *  \ingroup Expressions
 *
 *  \brief Calculates the total kinetic energy in the domain.
 *
 *  The total Kinetic Energy is given by:
 *  \f[ E_\text{kinetic} = \frac{1}{2} \sum_\text{cells} \mathbf{u}\cdot\mathbf{u} \f]
 *
 *  \tparam Vel1T   The type of field for the first velocity component.
 *  \tparam Vel2T   The type of field for the second velocity component.
 *  \tparam Vel3T   The type of field for the second velocity component.
 */
template< typename Vel1T, typename Vel2T, typename Vel3T >
class TotalKineticEnergy
: public Expr::Expression<SpatialOps::SingleValueField>
{
  
private:
  DECLARE_FIELD(Vel1T, u_)
  DECLARE_FIELD(Vel2T, v_)
  DECLARE_FIELD(Vel3T, w_)
  
  const bool doX_, doY_, doZ_, is3d_;
  
  TotalKineticEnergy( const Expr::Tag& resultTag,
                      const Expr::Tag& vel1Tag,
                      const Expr::Tag& vel2Tag,
                      const Expr::Tag& vel3Tag );
  
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    
    /**
     *  \param resultTag  The tag of the resulting expression
     *  \param vel1tag The velocity corresponding to the Vel1T template parameter
     *  \param vel2tag The velocity corresponding to the Vel2T template parameter
     *  \param vel3tag The velocity corresponding to the Vel3T template parameter
     */
    Builder( const Expr::Tag& resultTag,
             const Expr::Tag& vel1Tag,
             const Expr::Tag& vel2Tag,
             const Expr::Tag& vel3Tag );
    ~Builder(){}
    Expr::ExpressionBase* build() const;
    
  private:
    const Expr::Tag resultTag_, v1t_, v2t_, v3t_;
  };
  
  ~TotalKineticEnergy();

  void evaluate();
}; // Class TotalKineticEnergy

#endif // KineticEnergy_Expr_h
