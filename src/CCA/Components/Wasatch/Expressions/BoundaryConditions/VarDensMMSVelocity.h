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

#ifndef Var_Dens_MMS_Vel_Expr_h
#define Var_Dens_MMS_Vel_Expr_h

#include <expression/Expression.h>

#ifndef PI
#define PI 3.1415926535897932384626433832795
#endif

/**
 *  \class 	VarDensMMSVelocity
 *  \ingroup  Expressions
 *  \author   Amir Biglari
 *  \date   December, 2012
 *
 *  \brief Provides an expression for velocity at the boundaries in the MMS 
 *         that is written for the pressure projection method verification
 *
 *  \tparam FieldT - the type of field for the velocity.
 */

template< typename FieldT >
class VarDensMMSVelocity
: public BoundaryConditionBase<FieldT>
{
public:
  enum BCSide{
    RIGHT,
    LEFT
  };
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  \brief Construct an expression for the velocity at the boundaries, given the tag for time
     *         at x = 15 and x = -15
     *
     *  \param indepVarTag the Expr::Tag for holding the time variable.
     *
     *  \param side an enum for specifying the side of the boundary, wether it is on the left or the right side
     */
    Builder( const Expr::Tag& resultTag,
             const Expr::Tag& indepVarTag,
             const BCSide side ) :
    ExpressionBuilder(resultTag),
    indepVarTag_ (indepVarTag),
    side_ (side)
    {}
    Expr::ExpressionBase* build() const{ return new VarDensMMSVelocity(indepVarTag_, side_); }
  private:
    const Expr::Tag indepVarTag_;
    const BCSide side_;
  };
  
  ~VarDensMMSVelocity(){}
  void advertise_dependents( Expr::ExprDeps& exprDeps ){  exprDeps.requires_expression( indepVarTag_ );}
  void bind_fields( const Expr::FieldManagerList& fml ){
    t_    = &fml.template field_manager<double>().field_ref( indepVarTag_ );
  }
  void evaluate();
private:
  VarDensMMSVelocity( const Expr::Tag& indepVarTag,
              const BCSide side ) :
  indepVarTag_ (indepVarTag),
  side_ (side)
  {}  
  const double* t_;
  const Expr::Tag indepVarTag_;
  const BCSide side_;
};

// ###################################################################
//
//                          Implementation
//
// ###################################################################


//--------------------------------------------------------------------

template< typename FieldT >
void
VarDensMMSVelocity<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& f = this->value();
  const double ci = this->ci_;
  const double cg = this->cg_;
  std::vector<int>::const_iterator ia = this->flatGhostPoints_.begin(); // ia is the ghost flat index
  std::vector<int>::const_iterator ib = this->flatInteriorPoints_.begin(); // ib is the interior flat index
  if (side_==RIGHT) {
    for( ; ia != this->flatGhostPoints_.end(); ++ia, ++ib )
      f[*ia] = ( ( ((-5 * *t_)/( *t_ * *t_ + 1)) * sin(10 * PI / (*t_ + 10) ) ) - ci*f[*ib] ) / cg;
  }
  else if (side_==LEFT) {
    for( ; ia != this->flatGhostPoints_.end(); ++ia, ++ib )
      f[*ia] = ( ( ((-5 * *t_)/( *t_ * *t_ + 1)) * sin(-10 * PI / (*t_ + 10) ) ) - ci*f[*ib] ) / cg;
  }
}

#endif // Var_Dens_MMS_Vel_Expr_h
