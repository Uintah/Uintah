/*
 * The MIT License
 *
 * Copyright (c) 2012-2014 The University of Utah
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

// Uintah Includes
#include <Core/Exceptions/ProblemSetupException.h>

// Wasatch Includes
#include <CCA/Components/Wasatch/Transport/ParticleMomentumEquation.h>

#include <CCA/Components/Wasatch/Expressions/Particles/ParticleRe.h>
#include <CCA/Components/Wasatch/Expressions/Particles/ParticleDragForce.h>
#include <CCA/Components/Wasatch/Expressions/Particles/ParticleDensity.h>
#include <CCA/Components/Wasatch/Expressions/Particles/ParticleBodyForce.h>
#include <CCA/Components/Wasatch/Expressions/Particles/ParticleMomentumRHS.h>
#include <CCA/Components/Wasatch/Expressions/Particles/ParticleResponseTime.h>
#include <CCA/Components/Wasatch/Expressions/Particles/ParticleDragCoefficient.h>


namespace Wasatch{


  // #################################################################
  //
  //                          Implementation
  //
  // #################################################################

  ParticleMomentumEquation::
  ParticleMomentumEquation( const std::string& solnVarName,
                            const Direction pdir,
                            const Expr::TagList& particlePositionTags,
                            const Expr::Tag& particleSizeTag,
                            Uintah::ProblemSpecP particleEqsSpec,
                            GraphCategories& gc )
  : ParticleEquationBase( solnVarName, pdir, particlePositionTags, particleSizeTag, particleEqsSpec, gc ),
  pMassTag_( parse_nametag(particleEqsSpec->findBlock("ParticleMass"    )) ),
  pRhoTag_ ( parse_nametag(particleEqsSpec->findBlock("ParticleDensity" )) ),
  gViscTag_( parse_nametag(particleEqsSpec->findBlock("ParticleMomentum")->findBlock("GasProperties")->findBlock("GasViscosity")) ),
  gRhoTag_ ( parse_nametag(particleEqsSpec->findBlock("ParticleMomentum")->findBlock("GasProperties")->findBlock("GasDensity"  )) )
  {
    Uintah::ProblemSpecP pMomSpec = particleEqsSpec->findBlock("ParticleMomentum");
    
    // get the particle velocities
    std::string puname, pvname, pwname;
    pMomSpec->getAttribute("x",puname);
    pMomSpec->getAttribute("y",pvname);
    pMomSpec->getAttribute("z",pwname);
    pUTag_ = Expr::Tag(puname, Expr::STATE_DYNAMIC);
    pVTag_ = Expr::Tag(pvname, Expr::STATE_DYNAMIC);
    pWTag_ = Expr::Tag(pwname, Expr::STATE_DYNAMIC);
    
    // get the gas velocities
    gUTag_ = parse_nametag(pMomSpec->findBlock("GasProperties")->findBlock("GasVelocity")->findBlock("XVel"));
    gVTag_ = parse_nametag(pMomSpec->findBlock("GasProperties")->findBlock("GasVelocity")->findBlock("YVel"));
    gWTag_ = parse_nametag(pMomSpec->findBlock("GasProperties")->findBlock("GasVelocity")->findBlock("ZVel"));
    
    // pick up the gravity terms
    doGravity_ = false;
    for( Uintah::ProblemSpecP gravitySpec=pMomSpec->findBlock("Gravity");
        gravitySpec != 0;
        gravitySpec=gravitySpec->findNextBlock("Gravity") )
    {
      std::string gDir;
      gravitySpec->getAttribute("direction",gDir);
      if      (gDir == "X" && dir_name() == "X") doGravity_ = true;
      else if (gDir == "Y" && dir_name() == "Y") doGravity_ = true;
      else if (gDir == "Z" && dir_name() == "Z") doGravity_ = true;
    }
    doDrag_ = true;
    doDrag_ = !(pMomSpec->findBlock("DisableDragForce"));
    setup();
  }

  //------------------------------------------------------------------
  
  void ParticleMomentumEquation::setup()
  {
    rhsExprID_ = setup_rhs();
    gc_[ADVANCE_SOLUTION]->rootIDs.insert( rhsExprID_ );
  }
  
  //------------------------------------------------------------------
  
  Expr::ExpressionID ParticleMomentumEquation::setup_rhs()
  {
    Expr::ExpressionFactory& factory = *gc_[ADVANCE_SOLUTION]->exprFactory;
    
    //_____________________________
    // build a particle density expression
    if( !factory.have_entry( pRhoTag_ ) )
    {
      typedef ParticleDensity::Builder ParticleDensity;
      factory.register_expression( scinew ParticleDensity(pRhoTag_, pMassTag_, pSizeTag_ ) );
    }

    //_____________________________
    // build a bodyforce expression
    Expr::Tag pBodyForceTag;
    if (doGravity_)
    {
      pBodyForceTag = Expr::Tag("p.gravity_" + dir_name(), Expr::STATE_NONE);
      typedef ParticleBodyForce<SVolField>::Builder BodyForce;
      factory.register_expression( scinew BodyForce(pBodyForceTag, gRhoTag_, pRhoTag_, pSizeTag_, pPosTags_ ) );
    }

    Expr::Tag pDragForceTag;
    if (doDrag_) {
      //_____________________________
      // build a particle relaxation time expression
      const Expr::Tag pTauTag("p.tau",Expr::STATE_NONE);
      if( !factory.have_entry( pTauTag ) )
      {
        typedef ParticleResponseTime<SVolField>::Builder ParticleTau;
        factory.register_expression( scinew ParticleTau(pTauTag, pRhoTag_, pSizeTag_, gViscTag_, pPosTags_ ) );
      }
      
      //_____________________________
      // build a particle Re expression
      const Expr::Tag pReTag("p.re",Expr::STATE_NONE);
      if (!factory.have_entry( pReTag) )
      {
        typedef ParticleRe<XVolField, YVolField, ZVolField, SVolField>::Builder ParticleReB;
        const Expr::TagList gVelTags(tag_list(gUTag_, gVTag_, gWTag_));
        const Expr::TagList pVelTags(tag_list(pUTag_, pVTag_, pWTag_));
        factory.register_expression( scinew ParticleReB(pReTag, pSizeTag_, gRhoTag_, gViscTag_, pPosTags_, pVelTags, gVelTags ) );
      }
      
      //_____________________________
      // build a drag coefficient expression
      const Expr::Tag pDragCoefTag("p.cd", Expr::STATE_NONE);
      if (!factory.have_entry(pDragCoefTag)) {
        typedef  ParticleDragCoefficient::Builder DragCoef;
        factory.register_expression( scinew DragCoef(pDragCoefTag, pReTag ) );
      }
      
      //_____________________________
      // Drag Force
      pDragForceTag =  Expr::Tag("p.drag_" + dir_name(), Expr::STATE_NONE);
      switch (direction_) {
        case XDIR:
        {
          typedef ParticleDragForce<XVolField>::Builder DragForce;
          factory.register_expression( scinew DragForce(pDragForceTag, gUTag_, pDragCoefTag, pTauTag, solution_variable_tag(), pSizeTag_, pPosTags_ ) );
        }
          break;
        case YDIR:
        {
          typedef ParticleDragForce<YVolField>::Builder DragForce;
          factory.register_expression( scinew DragForce(pDragForceTag, gVTag_, pDragCoefTag, pTauTag, solution_variable_tag(), pSizeTag_, pPosTags_ ) );
        }
          break;
        case ZDIR:
        {
          typedef ParticleDragForce<ZVolField>::Builder DragForce;
          factory.register_expression( scinew DragForce(pDragForceTag, gWTag_, pDragCoefTag, pTauTag, solution_variable_tag(), pSizeTag_, pPosTags_ ) );
        }
          break;
          
        default:
          break;
      }
    }
    
    typedef ParticleMomentumRHS::Builder RHSBuilder;
    return factory.register_expression( scinew RHSBuilder(rhsTag_, pBodyForceTag, pDragForceTag ) );
  }
  
  //------------------------------------------------------------------
  
  ParticleMomentumEquation::~ParticleMomentumEquation()
  {}

  //------------------------------------------------------------------

  Expr::ExpressionID
  ParticleMomentumEquation::
  initial_condition( Expr::ExpressionFactory& exprFactory )
  {
    return exprFactory.get_id( solution_variable_tag() );
  }

  //==================================================================

} // namespace Particle
