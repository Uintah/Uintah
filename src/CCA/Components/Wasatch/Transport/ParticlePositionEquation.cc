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

#include <CCA/Components/Wasatch/Transport/ParticlePositionEquation.h>
#include <CCA/Components/Wasatch/Expressions/Particles/ParticlePositionRHS.h>
#include <Core/Exceptions/ProblemSetupException.h>

namespace WasatchCore{


  // #################################################################
  //
  //                          Implementation
  //
  // #################################################################




  ParticlePositionEquation::
  ParticlePositionEquation( const std::string& solnVarName,
                            const Direction pdir,
                            const Expr::TagList& particlePositionTags,
                           const Expr::Tag& particleSizeTag,
                           Uintah::ProblemSpecP particleEqsSpec,
                           GraphCategories& gc )
  : ParticleEquationBase( solnVarName, pdir, particlePositionTags, particleSizeTag, gc )
  {
    Uintah::ProblemSpecP pPosSpec = particleEqsSpec->findBlock("ParticlePosition");
    // get the particle velocities
    pUTag_ = parse_nametag(pPosSpec->findBlock("Velocity")->findBlock("XVel"));
    pVTag_ = parse_nametag(pPosSpec->findBlock("Velocity")->findBlock("YVel"));
    pWTag_ = parse_nametag(pPosSpec->findBlock("Velocity")->findBlock("ZVel"));
    setup();
  }

  //------------------------------------------------------------------
  
  void ParticlePositionEquation::setup()
  {
    rhsExprID_ = setup_rhs();
    gc_[ADVANCE_SOLUTION]->rootIDs.insert( rhsExprID_ );
  }
  
  //------------------------------------------------------------------
  
  Expr::ExpressionID ParticlePositionEquation::setup_rhs()
  {
    Expr::ExpressionFactory& factory = *gc_[ADVANCE_SOLUTION]->exprFactory;
    
    Expr::Tag pVelTag;
    switch (direction_) {
      case XDIR:
        pVelTag = pUTag_;
        break;
      case YDIR:
        pVelTag = pVTag_;
        break;
      case ZDIR:
        pVelTag = pWTag_;
      default:
        break;
    }

    typedef ParticlePositionRHS::Builder RHSBuilder;
    return factory.register_expression( scinew RHSBuilder(rhsTag_, pVelTag ) );    
  }
  
  //------------------------------------------------------------------
  
  ParticlePositionEquation::~ParticlePositionEquation()
  {}

  //------------------------------------------------------------------

  Expr::ExpressionID
  ParticlePositionEquation::
  initial_condition( Expr::ExpressionFactory& exprFactory )
  {
    return exprFactory.get_id( initial_condition_tag() );
  }

  //==================================================================

} // namespace Particle
