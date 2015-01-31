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

#include <CCA/Components/Wasatch/Transport/ParticleMassEquation.h>
#include <CCA/Components/Wasatch/Expressions/ExprAlgebra.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <expression/ExprLib.h>
#include <expression/Expression.h>
#include <cmath>

#ifndef PI
#  define PI 3.1415926535897932384626433832795
#endif

// ###################################################################
//
//                          Implementation
//
// ###################################################################

ParticleMassIC::
ParticleMassIC( const Expr::Tag& pRhoTag,
               const Expr::Tag& pDiameterTag )
: Expr::Expression<ParticleField>(),
pRhoTag_( pRhoTag ),
pDiameterTag_( pDiameterTag )
{}

//--------------------------------------------------------------------

ParticleMassIC::
~ParticleMassIC()
{}

//--------------------------------------------------------------------

void
ParticleMassIC::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( pRhoTag_ );
  exprDeps.requires_expression( pDiameterTag_ );
}

//--------------------------------------------------------------------

void
ParticleMassIC::
bind_fields( const Expr::FieldManagerList& fml )
{
  pRho_ = &fml.field_ref< ParticleField >( pRhoTag_ );
  pDiameter_ = &fml.field_ref< ParticleField >( pDiameterTag_ );
}

//--------------------------------------------------------------------

void
ParticleMassIC::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{}

//--------------------------------------------------------------------

void
ParticleMassIC::
evaluate()
{
  using namespace SpatialOps;
  ParticleField& result = this->value();
  result <<= (PI/6.0) * *pRho_ * pow(*pDiameter_,3);
}

//--------------------------------------------------------------------

ParticleMassIC::
Builder::Builder( const Expr::Tag& resultTag,
                 const Expr::Tag& pRhoTag,
                 const Expr::Tag& pDiameterTag )
: ExpressionBuilder( resultTag ),
pRhoTag_( pRhoTag ),
pDiameterTag_( pDiameterTag )
{}

//--------------------------------------------------------------------

Expr::ExpressionBase*
ParticleMassIC::
Builder::build() const
{
  return new ParticleMassIC( pRhoTag_,pDiameterTag_ );
}

namespace Wasatch{


  // #################################################################
  //
  //                          Implementation
  //
  // #################################################################


  ParticleMassEquation::
  ParticleMassEquation( const std::string& solnVarName,
                            const Direction pdir,
                            const Expr::TagList& particlePositionTags,
                           const Expr::Tag& particleSizeTag,
                           Uintah::ProblemSpecP particleEqsSpec,
                           GraphCategories& gc )
  : ParticleEquationBase( solnVarName, pdir, particlePositionTags, particleSizeTag, particleEqsSpec, gc ),
    pSrcTag_( parse_nametag(particleEqsSpec->findBlock("ParticleMass")->findBlock("SourceTerm")) ),
    pRhoTag_( parse_nametag(particleEqsSpec->findBlock("ParticleDensity")) )
  {
    setup();
  }

  //------------------------------------------------------------------
  
  void ParticleMassEquation::setup()
  {
    rhsExprID_ = setup_rhs();
    gc_[ADVANCE_SOLUTION]->rootIDs.insert( rhsExprID_ );
  }
  
  //------------------------------------------------------------------
  
  Expr::ExpressionID ParticleMassEquation::setup_rhs()
  {
    Expr::ExpressionFactory& factory = *gc_[ADVANCE_SOLUTION]->exprFactory;
    typedef Expr::ConstantExpr<ParticleField>::Builder RHSBuilder;
    return factory.register_expression( scinew RHSBuilder(rhsTag_, 0.0 ) );
  }
  
  //------------------------------------------------------------------
  
  ParticleMassEquation::~ParticleMassEquation()
  {}

  //------------------------------------------------------------------

  Expr::ExpressionID
  ParticleMassEquation::
  initial_condition( Expr::ExpressionFactory& exprFactory )
  {
    // register expression to calculate the momentum initial condition from the initial conditions on
    // velocity and density in the cases that we are initializing velocity in the input file
    const Expr::Tag pRhoInit(pRhoTag_.name(), Expr::STATE_NONE);
    const Expr::Tag pSizeInit(pSizeTag_.name(), Expr::STATE_NONE);
    exprFactory.register_expression( new ParticleMassIC::Builder( initial_condition_tag(),
                                                                  pRhoInit, pSizeInit ) );
    return exprFactory.get_id( initial_condition_tag() );
  }

  //==================================================================

} // namespace Particle
