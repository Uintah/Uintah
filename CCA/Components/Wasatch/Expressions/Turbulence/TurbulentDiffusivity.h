/*
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

#ifndef TurbulentDiffusivity_Expr_h
#define TurbulentDiffusivity_Expr_h

#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>
#include "TurbulenceParameters.h"

#include <expression/Expression.h>

/**
 *  \brief obtain the tag for the turbulent diffusivity
 */
Expr::Tag turbulent_diffusivity_tag();

/**
 *  \class TurbulentViscosity
 *  \author Amir Biglari
 *  \date   July, 2012. (Originally created: Jan, 2012).
 *  \ingroup Expressions
 *  \brief given turbulent schmidt number, \f$Sc_{turb}\f$, and turbulent viscosity,
 *   \f$\mu_{turb}\f$, and filtered density, \f$\bar{\rho}\f$, this calculates
 *   turbulent diffusivity, \f$D_{turb} = \frac{\mu_{turb}}{\bar{\rho} Sc_{turb}}\f$.
 *
 *   Note: It is currently assumed that turbulent diffusivity viscosity is a "SVolField" type.  
 *         Therefore, variables should be interpolated into "SVolField" at the end. 
 *
 *   Note: It is currently assumed that filtered densit, turbulent viscosity  and Sc number are 
 *         "SVolField" type. Therefore, there is no need to interpolate it.
 *
 */

class TurbulentDiffusivity
 : public Expr::Expression<SVolField>
{
  
  const Expr::Tag tViscTag_, rhoTag_;
  const double tSchmidt_;

  const SVolField *tVisc_, *rho_;

  TurbulentDiffusivity( const Expr::Tag rhoTag,
                        const double    tSchmidt,
                        const Expr::Tag tViscTag );
  
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  \brief Construct a turbulent diffusivity given expressions for
     *         \f$\bar{\rho}\f$, \f$Sc_{turb}\f$ and \f$\mu_{turb}\f$.
     *
     *  \param rhoTag the Expr::Tag for the density.
     *
     *  \param tScTag the Expr::Tag holding the turbulent Schmidt number 
     *
     *  \param tViscTag the Expr::Tagnholding the turbulent viscosity
     *
     */    
    Builder( const Expr::Tag& result,
             const Expr::Tag  rhoTag,
             const double     tSchmidt,
             const Expr::Tag  tViscTag )
      : ExpressionBuilder(result),
        rhot_   ( rhoTag   ),
        tVisct_ ( tViscTag ),
        tSc_    ( tSchmidt )
    {}
    
    Expr::ExpressionBase* build() const
    {
      return new TurbulentDiffusivity( rhot_, tSc_, tVisct_ );      
    }    
  private:
    const Expr::Tag rhot_, tVisct_;
    const double tSc_;
  };

  ~TurbulentDiffusivity();

  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();

};

#endif // TurbulentDiffusivity_Expr_h
