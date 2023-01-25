/*
 * The MIT License
 *
 * Copyright (c) 2016-2018 The University of Utah
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

/**
 *  \file   ParseParticleEquations.cc
 *  \date   Aug 30, 2016
 *  \author james
 */

#include <spatialops/structured/FVStaggered.h>

#include <Core/Exceptions/ProblemSetupException.h>

//-- Wasatch includes --//
#include <CCA/Components/Wasatch/TagNames.h>
#include <CCA/Components/Wasatch/Wasatch.h>
#include <CCA/Components/Wasatch/Transport/ParseParticleEquations.h>
#include <CCA/Components/Wasatch/Transport/EquationAdaptors.h>
#include <CCA/Components/Wasatch/Transport/ParticleSizeEquation.h>
#include <CCA/Components/Wasatch/Transport/ParticleMassEquation.h>
#include <CCA/Components/Wasatch/Transport/ParticlePositionEquation.h>
#include <CCA/Components/Wasatch/Transport/ParticleTemperatureEquation.h>
#include <CCA/Components/Wasatch/Transport/ParticleMomentumEquation.h>
#include <CCA/Components/Wasatch/Expressions/Particles/ParticleGasMomentumSrc.h>
#include <CCA/Components/Wasatch/Expressions/Turbulence/TurbulenceParameters.h>
#include <CCA/Components/Wasatch/Expressions/StableTimestep.h>

namespace WasatchCore{

  template<typename GasVel1T, typename GasVel2T, typename GasVel3T>
  std::vector<EqnTimestepAdaptorBase*>
  parse_particle_transport_equations( Uintah::ProblemSpecP particleSpec,
                                      Uintah::ProblemSpecP wasatchSpec,
                                      const bool useAdaptiveDt,
                                      GraphCategories& gc )
  {
    typedef std::vector<EqnTimestepAdaptorBase*> EquationAdaptors;
    EquationAdaptors adaptors;

    std::string pxname,pyname,pzname;
    Uintah::ProblemSpecP posSpec = particleSpec->findBlock("ParticlePosition");
    posSpec->getAttribute( "x", pxname );
    posSpec->getAttribute( "y", pyname );
    posSpec->getAttribute( "z", pzname );

    const Expr::Tag pXTag(pxname,Expr::STATE_DYNAMIC);
    const Expr::Tag pYTag(pyname,Expr::STATE_DYNAMIC);
    const Expr::Tag pZTag(pzname,Expr::STATE_DYNAMIC);
    const Expr::TagList pPosTags( tag_list(pXTag,pYTag,pZTag) );

    const Expr::Tag pSizeTag = parse_nametag(particleSpec->findBlock("ParticleSize"));
    const std::string pSizeName=pSizeTag.name();

    //___________________________________________________________________________
    // resolve the particle equations
    //
    proc0cout << "------------------------------------------------" << std::endl
    << "Creating particle equations..." << std::endl;
    proc0cout << "------------------------------------------------" << std::endl;

    proc0cout << "Setting up particle x-coordinate equation" << std::endl;
    EquationBase* pxeq = scinew ParticlePositionEquation( pxname,
                                                         XDIR,
                                                         pPosTags,
                                                         pSizeTag,
                                                         particleSpec,
                                                         gc );
    adaptors.push_back( scinew EqnTimestepAdaptor<ParticleField>(pxeq) );

    proc0cout << "Setting up particle y-coordinate equation" << std::endl;
    EquationBase* pyeq = scinew ParticlePositionEquation( pyname,
                                                         YDIR,
                                                         pPosTags,
                                                         pSizeTag,
                                                         particleSpec,
                                                         gc );
    adaptors.push_back( scinew EqnTimestepAdaptor<ParticleField>(pyeq) );

    proc0cout << "Setting up particle z-coordinate equation" << std::endl;
    EquationBase* pzeq = scinew ParticlePositionEquation( pzname,
                                                         ZDIR,
                                                         pPosTags,
                                                         pSizeTag,
                                                         particleSpec,
                                                         gc );
    adaptors.push_back( scinew EqnTimestepAdaptor<ParticleField>(pzeq) );


    std::string puname,pvname,pwname;
    Uintah::ProblemSpecP pMomSpec = particleSpec->findBlock("ParticleMomentum");
    pMomSpec->getAttribute( "x", puname );
    pMomSpec->getAttribute( "y", pvname );
    pMomSpec->getAttribute( "z", pwname );

    //___________________________________________________________________________
    // resolve the particle mass equation to be solved and create the adaptor for it.
    //
    const Expr::Tag pMassTag    = parse_nametag(particleSpec->findBlock("ParticleMass"));
    const std::string pMassName = pMassTag.name();
    proc0cout << "Setting up particle mass equation" << std::endl;
    EquationBase* pmeq = scinew ParticleMassEquation( pMassName,
                                                     NODIR,
                                                     pPosTags,
                                                     pSizeTag,
                                                     particleSpec,
                                                     gc );
    adaptors.push_back( scinew EqnTimestepAdaptor<ParticleField>(pmeq) );

    //___________________________________________________________________________
    // if specified, resolve the particle temperature equation to be solved and create the adaptor for it.
    //
    Uintah::ProblemSpecP pTempSpec = particleSpec->findBlock("ParticleTemperature");
    if( pTempSpec ){
      std::string gasViscosityName;
      Uintah::ProblemSpecP gasSpec = particleSpec->findBlock("ParticleMomentum")->findBlock("GasProperties");
      gasSpec->findBlock("GasViscosity")->getAttribute( "name", gasViscosityName );
      const Expr::Tag gViscTag( gasViscosityName, Expr::STATE_NONE );
      const Expr::Tag pTempTag    = parse_nametag(pTempSpec);
      const std::string pTempName = pTempTag.name();
      proc0cout << "Setting up particle temperature equation" << std::endl;
      EquationBase* pTeq = scinew ParticleTemperatureEquation( pTempName,
                                                               pPosTags,
                                                               pSizeTag,
                                                               pMassTag,
                                                               gViscTag,
                                                               wasatchSpec,
                                                               pTempSpec,
                                                               gc );
      adaptors.push_back( scinew EqnTimestepAdaptor<ParticleField>(pTeq) );
    }

    //___________________________________________________________________________
    // resolve the momentum equation to be solved and create the adaptor for it.
    //
    Expr::ExpressionFactory& factory = *(gc[ADVANCE_SOLUTION]->exprFactory);
    proc0cout << "Setting up particle x-momentum equation" << std::endl;
    EquationBase* pueq = scinew ParticleMomentumEquation<GasVel1T,GasVel2T,GasVel3T>( puname,
                                                                                     XDIR,
                                                                                     pPosTags,
                                                                                     pSizeTag,
                                                                                     particleSpec,
                                                                                     gc );
    adaptors.push_back( scinew EqnTimestepAdaptor<ParticleField>(pueq) );

    proc0cout << "Setting up particle y-momentum equation" << std::endl;
    EquationBase* pveq = scinew ParticleMomentumEquation<GasVel1T,GasVel2T,GasVel3T>( pvname,
                                                                                     YDIR,
                                                                                     pPosTags,
                                                                                     pSizeTag,
                                                                                     particleSpec,
                                                                                     gc );
    adaptors.push_back( scinew EqnTimestepAdaptor<ParticleField>(pveq) );

    proc0cout << "Setting up particle z-momentum equation" << std::endl;
    EquationBase* pweq = scinew ParticleMomentumEquation<GasVel1T,GasVel2T,GasVel3T>( pwname,
                                                                                     ZDIR,
                                                                                     pPosTags,
                                                                                     pSizeTag,
                                                                                     particleSpec,
                                                                                     gc );
    adaptors.push_back( scinew EqnTimestepAdaptor<ParticleField>(pweq) );

    //___________________________________________________________________________
    // resolve the particle size equation to be solved and create the adaptor for it.
    //
    proc0cout << "Setting up particle size equation" << std::endl;
    EquationBase* psizeeq = scinew ParticleSizeEquation( pSizeName,
                                                        NODIR,
                                                        pPosTags,
                                                        pSizeTag,
                                                        particleSpec,
                                                        gc );
    adaptors.push_back( scinew EqnTimestepAdaptor<ParticleField>(psizeeq) );

    //___________________________________________________________________________
    // Two way coupling between particles and the gas phase
    //
    if( !particleSpec->findBlock("ParticleMomentum")->findBlock("DisableTwoWayCoupling") ){
      Uintah::ProblemSpecP momentumSpec  = wasatchSpec->findBlock("MomentumEquations");
      if( momentumSpec ){

        std::string xmomname, ymomname, zmomname;
        const Uintah::ProblemSpecP doxmom = momentumSpec->get( "X-Momentum", xmomname );
        const Uintah::ProblemSpecP doymom = momentumSpec->get( "Y-Momentum", ymomname );
        const Uintah::ProblemSpecP dozmom = momentumSpec->get( "Z-Momentum", zmomname );

        const TagNames tNames = TagNames::self();
        if( doxmom ){
          typedef typename ParticleGasMomentumSrc<GasVel1T>::Builder XMomSrcT;
          const Expr::Tag xMomRHSTag (xmomname + "_rhs_partial", Expr::STATE_NONE);
          factory.register_expression( scinew XMomSrcT( tNames.pmomsrcx, tNames.pdragx, pMassTag, pSizeTag, pPosTags ));
          factory.attach_dependency_to_expression(tNames.pmomsrcx, xMomRHSTag);
        }

        if( doymom ){
          typedef typename ParticleGasMomentumSrc<GasVel2T>::Builder YMomSrcT;
          const Expr::Tag yMomRHSTag (ymomname + "_rhs_partial", Expr::STATE_NONE);
          factory.register_expression( scinew YMomSrcT( tNames.pmomsrcy, tNames.pdragy, pMassTag, pSizeTag, pPosTags ));
          factory.attach_dependency_to_expression(tNames.pmomsrcy, yMomRHSTag);
        }

        if( dozmom ){
          typedef typename ParticleGasMomentumSrc<GasVel3T>::Builder ZMomSrcT;
          const Expr::Tag zMomRHSTag (zmomname + "_rhs_partial", Expr::STATE_NONE);
          factory.register_expression( scinew ZMomSrcT( tNames.pmomsrcz, tNames.pdragz, pMassTag, pSizeTag, pPosTags ));
          factory.attach_dependency_to_expression(tNames.pmomsrcz, zMomRHSTag);
        }
      }
    }

    //
    // loop over the local adaptors and set the initial and boundary conditions on each equation attached to that adaptor
    for( EquationAdaptors::const_iterator ia=adaptors.begin(); ia!=adaptors.end(); ++ia ){
      EqnTimestepAdaptorBase* const adaptor = *ia;
      EquationBase* particleEq = adaptor->equation();
      //_____________________________________________________
      // set up initial conditions on this momentum equation
      try{
        proc0cout << "Setting initial conditions for particle equation: "
        << particleEq->solution_variable_name()
        << std::endl;
        GraphHelper* const icGraphHelper = gc[INITIALIZATION];
        icGraphHelper->rootIDs.insert( particleEq->initial_condition( *icGraphHelper->exprFactory ) );
      }
      catch( std::runtime_error& e ){
        std::ostringstream msg;
        msg << e.what()
        << std::endl
        << "ERORR while setting initial conditions on particle equation "
        << particleEq->solution_variable_name()
        << std::endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
    }

    proc0cout << "------------------------------------------------" << std::endl;

    //
    // ADD ADAPTIVE TIMESTEPPING in case it was not parsed in the momentum equations
    if( useAdaptiveDt ){
      // if no stabletimestep expression has been registered, then register one. otherwise return.
      if( !factory.have_entry(TagNames::self().stableTimestep) ){

        std::string gasViscosityName, gasDensityName;
        Uintah::ProblemSpecP gasSpec = particleSpec->findBlock("ParticleMomentum")->findBlock("GasProperties");
        gasSpec->findBlock("GasViscosity")->getAttribute( "name", gasViscosityName );
        gasSpec->findBlock("GasDensity")->getAttribute( "name", gasDensityName );
        Uintah::ProblemSpecP gasVelSpec = gasSpec->findBlock("GasVelocity");
        std::string uVelName, vVelName, wVelName;
        gasVelSpec->findBlock("XVel")->getAttribute("name",uVelName);
        gasVelSpec->findBlock("YVel")->getAttribute("name",vVelName);
        gasVelSpec->findBlock("ZVel")->getAttribute("name",wVelName);
        const Expr::Tag xVelTag = Expr::Tag(uVelName, Expr::STATE_NONE);
        const Expr::Tag yVelTag = Expr::Tag(vVelName, Expr::STATE_NONE);
        const Expr::Tag zVelTag = Expr::Tag(wVelName, Expr::STATE_NONE);
        const Expr::Tag viscTag = Expr::Tag(gasViscosityName, Expr::STATE_NONE);
        const Expr::Tag densityTag = Expr::Tag(gasDensityName, Expr::STATE_NONE);

        const Expr::Tag puTag = Expr::Tag(puname, Expr::STATE_DYNAMIC);
        const Expr::Tag pvTag = Expr::Tag(pvname, Expr::STATE_DYNAMIC);
        const Expr::Tag pwTag = Expr::Tag(pwname, Expr::STATE_DYNAMIC);

        const bool isCompressible = (Wasatch::flow_treatment() == COMPRESSIBLE);
        Expr::ExpressionID stabDtID;
        if (isCompressible){
          stabDtID = factory.register_expression(scinew StableTimestep<SVolField,SVolField,SVolField>::Builder( TagNames::self().stableTimestep,
                                                                                                                                        densityTag, viscTag,
                                                                                                                                        xVelTag,yVelTag,zVelTag, puTag, pvTag, pwTag, TagNames::self().soundspeed ),true );
        } else {
          const Expr::Tag soundspeedTag = isCompressible ? TagNames::self().soundspeed : Expr::Tag();
          stabDtID = factory.register_expression(scinew StableTimestep<XVolField,YVolField,ZVolField>::Builder( TagNames::self().stableTimestep,
                                                                                                                                        densityTag, viscTag,
                                                                                                                                        xVelTag,yVelTag,zVelTag, puTag, pvTag, pwTag, Expr::Tag() ),true );
          
        }
                                            
        // force this onto the graph.
        gc[ADVANCE_SOLUTION]->rootIDs.insert( stabDtID );
      }
    }

    //
    return adaptors;
  }

  //----------------------------------------------------------------------------
  // explicit template instantiation
  template std::vector<EqnTimestepAdaptorBase*>
  parse_particle_transport_equations<SVolField,SVolField,SVolField>( Uintah::ProblemSpecP, Uintah::ProblemSpecP, const bool, GraphCategories& );

  template std::vector<EqnTimestepAdaptorBase*>
  parse_particle_transport_equations<XVolField,YVolField,ZVolField>( Uintah::ProblemSpecP, Uintah::ProblemSpecP, const bool, GraphCategories& );
  //----------------------------------------------------------------------------

}// namespace WasatchCore
