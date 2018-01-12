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

#include <CCA/Components/Wasatch/Transport/ParticleMomentumEquation.h>

// Uintah Includes
#include <Core/Exceptions/ProblemSetupException.h>

// Wasatch Includes
#include <CCA/Components/Wasatch/TagNames.h>
#include <CCA/Components/Wasatch/Expressions/Particles/ParticleRe.h>
#include <CCA/Components/Wasatch/Expressions/Particles/ParticleDensity.h>
#include <CCA/Components/Wasatch/Expressions/Particles/ParticleDragForce.h>
#include <CCA/Components/Wasatch/Expressions/Particles/ParticleBodyForce.h>
#include <CCA/Components/Wasatch/Expressions/Particles/ParticleMomentumRHS.h>
#include <CCA/Components/Wasatch/Expressions/Particles/ParticleResponseTime.h>
#include <CCA/Components/Wasatch/Expressions/Particles/ParticleDragCoefficient.h>
#include <CCA/Components/Wasatch/Expressions/BoundaryConditions/ParticleWallBC.h>

namespace WasatchCore{

  extern bool is_normal_to_boundary( const Direction stagLoc,
                                     const Uintah::Patch::FaceType face );

  // #################################################################
  //
  //                          Implementation
  //
  // #################################################################
  template<typename GasVel1T, typename GasVel2T, typename GasVel3T>
  ParticleMomentumEquation<GasVel1T,GasVel2T,GasVel3T>::
  ParticleMomentumEquation( const std::string& solnVarName,
                            const Direction pdir,
                            const Expr::TagList& particlePositionTags,
                            const Expr::Tag& particleSizeTag,
                            Uintah::ProblemSpecP particleEqsSpec,
                            GraphCategories& gc )
  : ParticleEquationBase( solnVarName, pdir, particlePositionTags, particleSizeTag, gc ),
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
    pUTag_ = Expr::Tag( puname, Expr::STATE_DYNAMIC );
    pVTag_ = Expr::Tag( pvname, Expr::STATE_DYNAMIC );
    pWTag_ = Expr::Tag( pwname, Expr::STATE_DYNAMIC );
    
    // get the gas velocities
    gUTag_ = parse_nametag( pMomSpec->findBlock("GasProperties")->findBlock("GasVelocity")->findBlock("XVel") );
    gVTag_ = parse_nametag( pMomSpec->findBlock("GasProperties")->findBlock("GasVelocity")->findBlock("YVel") );
    gWTag_ = parse_nametag( pMomSpec->findBlock("GasProperties")->findBlock("GasVelocity")->findBlock("ZVel") );
    
    // pick up the gravity terms
    doGravity_ = false;
    for( Uintah::ProblemSpecP gravitySpec=pMomSpec->findBlock("Gravity"); gravitySpec != nullptr; gravitySpec = gravitySpec->findNextBlock("Gravity") ) {
      std::string gDir;
      gravitySpec->getAttribute("direction",gDir);
      if     ( gDir == "X" && dir_name() == "x" ) { doGravity_ = true; }
      else if( gDir == "Y" && dir_name() == "y" ) { doGravity_ = true; }
      else if( gDir == "Z" && dir_name() == "z" ) { doGravity_ = true; }
    }
    
    // check if drag was disabled
    doDrag_ = !(pMomSpec->findBlock("DisableDragForce"));
    setup();
  }

  //------------------------------------------------------------------
  template<typename GasVel1T, typename GasVel2T, typename GasVel3T>
  void   ParticleMomentumEquation<GasVel1T,GasVel2T,GasVel3T>::setup()
  {
    rhsExprID_ = setup_rhs();
    gc_[ADVANCE_SOLUTION]->rootIDs.insert( rhsExprID_ );
  }
  
  //------------------------------------------------------------------
  template<typename GasVel1T, typename GasVel2T, typename GasVel3T>
  Expr::ExpressionID ParticleMomentumEquation<GasVel1T,GasVel2T,GasVel3T>::setup_rhs()
  {
    Expr::ExpressionFactory& factory = *gc_[ADVANCE_SOLUTION]->exprFactory;
    
    //_____________________________
    // build a particle density expression
    if( !factory.have_entry( pRhoTag_ ) ){
      typedef ParticleDensity::Builder ParticleDensity;
      factory.register_expression( scinew ParticleDensity(pRhoTag_, pMassTag_, pSizeTag_ ) );
    }

    //_____________________________
    // build a bodyforce expression
    Expr::Tag pBodyForceTag;
    if( doGravity_ ){
      switch( direction_ ){
        case XDIR: pBodyForceTag = TagNames::self().pbodyx; break;
        case YDIR: pBodyForceTag = TagNames::self().pbodyy; break;
        case ZDIR: pBodyForceTag = TagNames::self().pbodyz; break;
        default:                                            break;
      }
      typedef typename ParticleBodyForce<SVolField>::Builder BodyForce;
      factory.register_expression( scinew BodyForce(pBodyForceTag, gRhoTag_, pRhoTag_, pSizeTag_, pPosTags_ ) );
    }

    Expr::Tag pDragForceTag;
    if( doDrag_ ){
      //_____________________________
      // build a particle relaxation time expression
      const Expr::Tag pTauTag = TagNames::self().presponse;
      if( !factory.have_entry( pTauTag ) ){
        typedef typename ParticleResponseTime<SVolField>::Builder ParticleTau;
        factory.register_expression( scinew ParticleTau(pTauTag, pRhoTag_, pSizeTag_, gViscTag_, pPosTags_ ) );
      }
      
      //_____________________________
      // build a particle Re expression
      const Expr::Tag pReTag = TagNames::self().preynolds;
      if( !factory.have_entry( pReTag) ){
        typedef typename ParticleRe<GasVel1T, GasVel2T, GasVel3T, SVolField>::Builder ParticleReB;
        const Expr::TagList gVelTags(tag_list(gUTag_, gVTag_, gWTag_));
        const Expr::TagList pVelTags(tag_list(pUTag_, pVTag_, pWTag_));
        factory.register_expression( scinew ParticleReB(pReTag, pSizeTag_, gRhoTag_, gViscTag_, pPosTags_, pVelTags, gVelTags ) );
      }
      
      //_____________________________
      // build a drag coefficient expression
      const Expr::Tag pDragCoefTag = TagNames::self().pdragcoef;
      if( !factory.have_entry(pDragCoefTag) ){
        typedef  ParticleDragCoefficient::Builder DragCoef;
        factory.register_expression( scinew DragCoef(pDragCoefTag, pReTag ) );
      }
      
      //_____________________________
      // Drag Force
      switch( direction_ ){
        case XDIR:{
          pDragForceTag = TagNames::self().pdragx;
          typedef typename ParticleDragForce<GasVel1T>::Builder DragForce;
          factory.register_expression( scinew DragForce(pDragForceTag, gUTag_, pDragCoefTag, pTauTag, solution_variable_tag(), pSizeTag_, pPosTags_ ) );
          break;
        }
        case YDIR:{
          pDragForceTag = TagNames::self().pdragy;
          typedef typename ParticleDragForce<GasVel2T>::Builder DragForce;
          factory.register_expression( scinew DragForce(pDragForceTag, gVTag_, pDragCoefTag, pTauTag, solution_variable_tag(), pSizeTag_, pPosTags_ ) );
          break;
        }
        case ZDIR:{
          pDragForceTag = TagNames::self().pdragz;
          typedef typename ParticleDragForce<GasVel3T>::Builder DragForce;
          factory.register_expression( scinew DragForce(pDragForceTag, gWTag_, pDragCoefTag, pTauTag, solution_variable_tag(), pSizeTag_, pPosTags_ ) );
          break;
        }
        default:
          break;
      }
    }
    
    typedef ParticleMomentumRHS::Builder RHSBuilder;
    return factory.register_expression( scinew RHSBuilder(rhsTag_, pBodyForceTag, pDragForceTag ) );
  }
  
  //------------------------------------------------------------------
  template<typename GasVel1T, typename GasVel2T, typename GasVel3T>
  ParticleMomentumEquation<GasVel1T,GasVel2T,GasVel3T>::~ParticleMomentumEquation()
  {}

  //------------------------------------------------------------------
  template<typename GasVel1T, typename GasVel2T, typename GasVel3T>
  Expr::ExpressionID
  ParticleMomentumEquation<GasVel1T,GasVel2T,GasVel3T>::
  initial_condition( Expr::ExpressionFactory& exprFactory )
  {
    return exprFactory.get_id( initial_condition_tag() );
  }

  //------------------------------------------------------------------
  template<typename GasVel1T, typename GasVel2T, typename GasVel3T>
  void
  ParticleMomentumEquation<GasVel1T,GasVel2T,GasVel3T>::setup_boundary_conditions( WasatchBCHelper& bcHelper,
                                                       GraphCategories& graphCat )
  {
    Expr::ExpressionFactory& advSlnFactory = *(graphCat[ADVANCE_SOLUTION]->exprFactory);
    
    // make logical decisions based on the specified boundary types
    BOOST_FOREACH( const BndMapT::value_type& bndPair, bcHelper.get_boundary_information() )
    {
      const std::string& bndName = bndPair.first;
      const BndSpec& myBndSpec = bndPair.second;
      const bool isNormal = is_normal_to_boundary(direction_, myBndSpec.face);
      
      const Uintah::BCGeomBase::ParticleBndSpec pBndSpec = myBndSpec.particleBndSpec;
      if( pBndSpec.hasParticleBC() ){

        switch( pBndSpec.bndType ){
          case Uintah::BCGeomBase::ParticleBndSpec::WALL:{
            const double restCoef = pBndSpec.restitutionCoef;
            if( isNormal ){
              // create particle wall bcs
              const Expr::Tag pVelBCTag( solution_variable_name() + "_" + bndName +"_wallbc", Expr::STATE_NONE);
              BndCondSpec particleWallBCSpec = {solution_variable_name(), pVelBCTag.name(), 0.0, DIRICHLET, FUNCTOR_TYPE};
              advSlnFactory.register_expression ( new ParticleWallBC::Builder(pVelBCTag, restCoef, false) );
              bcHelper.add_boundary_condition(bndName, particleWallBCSpec);
            }
            else{
              if( pBndSpec.wallType != Uintah::BCGeomBase::ParticleBndSpec::ELASTIC ){
                // create particle wall bcs
                const Expr::Tag pVelBCTag( solution_variable_name() + "_" + bndName +"_wallbc", Expr::STATE_NONE);
                BndCondSpec particleWallBCSpec = {solution_variable_name(), pVelBCTag.name(), 0.0, DIRICHLET, FUNCTOR_TYPE};
                advSlnFactory.register_expression ( new ParticleWallBC::Builder(pVelBCTag, restCoef, true) );
                bcHelper.add_boundary_condition(bndName, particleWallBCSpec);
              }
            }
          }
          break;

          default:
            break;
        }
      }
    }
  }
  
  //==================================================================
  template<typename GasVel1T, typename GasVel2T, typename GasVel3T>
  void ParticleMomentumEquation<GasVel1T,GasVel2T,GasVel3T>::
  apply_boundary_conditions( const GraphHelper& graphHelper,
                             WasatchBCHelper& bcHelper )
  {
    const Category taskCat = ADVANCE_SOLUTION;
    // set bcs for particle momentum
    bcHelper.apply_boundary_condition<ParticleField>( this->solnvar_np1_tag(), taskCat );
  }
  //==================================================================
  
  // Explicit Template Instantiation:
  template class ParticleMomentumEquation< XVolField, YVolField, ZVolField >;
  template class ParticleMomentumEquation< SVolField, SVolField, SVolField >;


} // namespace Particle
