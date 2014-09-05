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

#ifndef Var_Dens_MMS_Soln_Var_Expr_h
#define Var_Dens_MMS_Soln_Var_Expr_h

#include <expression/Expression.h>

/**
 *  \class 	VarDensMMSSolnVar
 *  \ingroup  Expressions
 *  \author   Amir Biglari
 *  \date   December, 2012
 *
 *  \brief Provides an expression for solution variable at the boundaries in the MMS 
 *         that is written for the pressure projection method verification
 *
 *  \tparam FieldT - the type of field for the solution variable.
 */

template< typename FieldT >
class VarDensMMSSolnVar
: public BoundaryConditionBase<FieldT>
{
  VarDensMMSSolnVar( const Expr::Tag& indepVarTag ) :
  indepVarTag_ (indepVarTag)
  {}
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  \brief Construct an expression for the solution variable at the boundaries, given the tag for time
     *         at x = 15 and x = -15
     *
     *  \param indepVarTag the Expr::Tag for holding the time variable.
     */
    Builder( const Expr::Tag& resultTag,
             const Expr::Tag& indepVarTag) :
    ExpressionBuilder(resultTag),
    indepVarTag_ (indepVarTag)
    {}
    Expr::ExpressionBase* build() const{ return new VarDensMMSSolnVar(indepVarTag_); }
  private:
    const Expr::Tag indepVarTag_;
  };
  
  ~VarDensMMSSolnVar(){}
  void advertise_dependents( Expr::ExprDeps& exprDeps ){ exprDeps.requires_expression( indepVarTag_ );}
  void bind_fields( const Expr::FieldManagerList& fml ){ t_ = &fml.template field_ref<double>( indepVarTag_ );
  }
  void evaluate();
private:
  const double* t_;
  const Expr::Tag indepVarTag_;
};

// ###################################################################
//
//                          Implementation
//
// ###################################################################


//--------------------------------------------------------------------

template< typename FieldT >
void
VarDensMMSSolnVar<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& f = this->value();
  const double ci = this->ci_;
  const double cg = this->cg_;
  if ( (this->vecGhostPts_) && (this->vecInteriorPts_) ) {
    std::vector<SpatialOps::structured::IntVec>::const_iterator ig = (this->vecGhostPts_)->begin();    // ig is the ghost flat index
    std::vector<SpatialOps::structured::IntVec>::const_iterator ii = (this->vecInteriorPts_)->begin(); // ii is the interior flat index
    const double bcValue = -5/(exp(1125/(*t_ + 10))*(2 * *t_ + 5) * ((5/(exp(1125/(*t_ + 10))*(2 * *t_ + 5)) - 1)/1.29985 - 5/(0.081889 * exp(1125/(*t_ + 10))*(2 * *t_ + 5))));
    for( ; ig != (this->vecGhostPts_)->end(); ++ig, ++ii ){
      f(*ig) = ( bcValue - ci*f(*ii) ) / cg;
    }
  }
}

#endif // Var_Dens_MMS_Soln_Var_Expr_h
