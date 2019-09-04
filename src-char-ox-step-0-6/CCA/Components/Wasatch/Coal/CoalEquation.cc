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
#include <CCA/Components/Wasatch/Coal/CoalEquation.h>

#include <CCA/Components/Wasatch/Expressions/ExprAlgebra.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <expression/ExprLib.h>
#include <expression/Expression.h>
#include <cmath>

#include <stdexcept>
#include <sstream>
#include <string>

namespace Coal{

  CoalEquation::
  CoalEquation( const std::string& solnVarName,
                const Expr::Tag&   particleMassTag,
                const double       initialMassFraction,
                GraphCategories&   gc )
  : EquationBase::EquationBase( gc, solnVarName, WasatchCore::NODIR ),
    pMassTag_    ( particleMassTag     ),
    initialValue_( initialMassFraction )
  {
    if(pMassTag_ == Expr::Tag()){
      std::ostringstream msg;
      msg << std::endl
          << __FILE__ << " : " << __LINE__ << std::endl
          << "Tag for particle mass passed to CoalEquation is invalid" << std::endl;
      throw std::invalid_argument( msg.str() );
    }
    setup();
  }

  CoalEquation::
  CoalEquation( const std::string& solnVarName,
                const double       initialValue,
                GraphCategories&   gc )
  : EquationBase::EquationBase( gc, solnVarName, WasatchCore::NODIR ),
    pMassTag_   ( Expr::Tag() ),
    initialValue_( initialValue )
  {
    setup();
  }


  //------------------------------------------------------------------

  void CoalEquation::setup()
  {
    //Expr::ExpressionFactory& factory = *gc_[WasatchCore::ADVANCE_SOLUTION]->exprFactory;
  }


  //------------------------------------------------------------------

  Expr::ExpressionID
  CoalEquation::
  initial_condition( Expr::ExpressionFactory& exprFactory )
  {
    if( pMassTag_ == Expr::Tag() ){
      exprFactory.register_expression( new Expr::ConstantExpr<ParticleField>::
                                           Builder( initial_condition_tag(),
                                                    initialValue_ ) );
    }
    else{
      const Expr::Tag pMassInit(pMassTag_.name(), Expr::STATE_NONE);
      exprFactory.register_expression( new Expr::LinearFunction<ParticleField>::
                                           Builder( initial_condition_tag(),
                                                    pMassInit,
                                                    initialValue_,
                                                    0.0) );
    }
    return exprFactory.get_id( initial_condition_tag() );
  }

} // namespace Coal
