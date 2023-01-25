
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

//-- Wasatch Includes --//
#include <sci_defs/wasatch_defs.h>
#include <CCA/Components/Wasatch/Wasatch.h>
#include <CCA/Components/Wasatch/TagNames.h>
#include <CCA/Components/Wasatch/ParseTools.h>
#include <CCA/Components/Wasatch/Transport/ParseEquation.h>
#include <CCA/Components/Wasatch/Transport/EquationAdaptors.h>
#include <CCA/Components/Wasatch/DualTimeMatrixManager.h>

//-- Add headers for individual transport equations here --//
#include <CCA/Components/Wasatch/Transport/TransportEquation.h>
#include <CCA/Components/Wasatch/Transport/ScalarTransportEquation.h>
#include <CCA/Components/Wasatch/Transport/ScalabilityTestTransportEquation.h>
#include <CCA/Components/Wasatch/Transport/MomentumTransportEquationBase.h>
#include <CCA/Components/Wasatch/Transport/LowMachMomentumTransportEquation.h>
#include <CCA/Components/Wasatch/Transport/CompressibleMomentumTransportEquation.h>
#include <CCA/Components/Wasatch/Transport/TotalInternalEnergyTransportEquation.h>
#include <CCA/Components/Wasatch/Transport/EquationBase.h>
#include <CCA/Components/Wasatch/Transport/MomentTransportEquation.h>
#include <CCA/Components/Wasatch/Transport/EnthalpyTransportEquation.h>
#ifdef HAVE_POKITT
#include "SpeciesTransportEquation.h"
#include <CCA/Components/Wasatch/Transport/TarTransportEquation.h>
#include <CCA/Components/Wasatch/Transport/SootTransportEquation.h>
#include <CCA/Components/Wasatch/Transport/SootParticleTransportEquation.h>
#include <pokitt/CanteraObjects.h>
#include <pokitt/transport/ViscosityMix.h>
#endif

//-- includes for the expressions built here --//
#include <CCA/Components/Wasatch/Expressions/PoissonExpression.h>
#include <CCA/Components/Wasatch/Expressions/StableTimestep.h>
#include <CCA/Components/Wasatch/Expressions/MMS/Functions.h>
#include <CCA/Components/Wasatch/Expressions/MMS/VardenMMS.h>
#include <CCA/Components/Wasatch/Expressions/MMS/Varden2DMMS.h>
#include <CCA/Components/Wasatch/Expressions/SimpleEmission.h>
#include <CCA/Components/Wasatch/Expressions/DORadSolver.h>

//-- Uintah includes --//
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Parallel/Parallel.h>

//-- Expression Library includes --//
#include <expression/ExpressionFactory.h>

#include <iostream>

namespace WasatchCore{



  //==================================================================

  EqnTimestepAdaptorBase* parse_scalar_equation( Uintah::ProblemSpecP scalarEqnParams,
                                                 Uintah::ProblemSpecP wasatchParams,
                                                 TurbulenceParameters turbParams,
                                                 const Expr::Tag densityTag,
                                                 GraphCategories& gc,
                                                 WasatchCore::DualTimeMatrixInfo& dualTimeMatrixInfo,
                                                 std::set<std::string>& persistentFields )
  {
    EqnTimestepAdaptorBase* adaptor = nullptr;
    EquationBase* transeqn = nullptr;

    std::string eqnLabel, solnVariable;

    scalarEqnParams->getAttribute( "equation", eqnLabel );
    scalarEqnParams->get( "SolutionVariable", solnVariable );



    //___________________________________________________________________________
    // resolve the transport equation to be solved and create the adaptor for it.
    //
    proc0cout << "------------------------------------------------" << std::endl
    << "Creating transport equation for '" << solnVariable << "'" << std::endl;

    if( eqnLabel == "generic" || eqnLabel=="mixturefraction"){
      typedef ScalarTransportEquation< SVolField > ScalarTransEqn;
      transeqn = scinew ScalarTransEqn( ScalarTransEqn::get_solnvar_name( scalarEqnParams ),
                                       scalarEqnParams,
                                       gc,
                                       densityTag,
                                       turbParams,
                                       persistentFields );
      adaptor = scinew EqnTimestepAdaptor< SVolField >( transeqn );

      dualTimeMatrixInfo.add_scalar_equation( transeqn->solution_variable_tag(), transeqn->rhs_tag() );

    }
    else if( eqnLabel == "enthalpy" ){
      typedef EnthalpyTransportEquation TransEqn;
      transeqn = scinew TransEqn( ScalarTransportEquation<SVolField>::get_solnvar_name(scalarEqnParams),
                                 scalarEqnParams,
                                 wasatchParams,
                                 gc,
                                 densityTag,
                                 turbParams,
                                 persistentFields );
      adaptor = scinew EqnTimestepAdaptor<SVolField>(transeqn);
    }
    else {
      std::ostringstream msg;
      msg << "ERROR: No transport equation was specified '" << eqnLabel << "'. Please revise your input file" << std::endl;
      throw Uintah::InvalidValue( msg.str(), __FILE__, __LINE__ );
    }

    assert( transeqn != nullptr );
    assert( adaptor  != nullptr );

    //_____________________________________________________
    // set up initial conditions on this transport equation
    try{
      proc0cout << "Setting initial conditions for transport equation '" << solnVariable << "'" << std::endl;
      GraphHelper* const icGraphHelper = gc[INITIALIZATION  ];
      icGraphHelper->rootIDs.insert( transeqn->initial_condition( *icGraphHelper->exprFactory ) );
    }
    catch( std::runtime_error& e ){
      std::ostringstream msg;
      msg << e.what()
      << std::endl
      << "ERORR while setting initial conditions on equation '" << eqnLabel << "'"
      << std::endl;
      throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
    }

    proc0cout << "------------------------------------------------" << std::endl;
    return adaptor;
  }

  //==================================================================

  std::vector<EqnTimestepAdaptorBase*>
  parse_species_equations( Uintah::ProblemSpecP params,
                           Uintah::ProblemSpecP wasatchSpec,
                           Uintah::ProblemSpecP momentumParams,
                           const TurbulenceParameters& turbParams,
                           const Expr::Tag& densityTag,
                           GraphCategories& gc,
                           WasatchCore::DualTimeMatrixInfo& dualTimeMatrixInfo,
                           const bool computeKineticsJacobian )
  {
#   ifdef HAVE_POKITT
    if( turbParams.turbModelName != TurbulenceParameters::NOTURBULENCE ){
      // jcs We should at least include convective closure (turbulent diffusive flux)
      //     Source term closure is a much more challenging issue.
      throw Uintah::ProblemSetupException( "Turbulent closure is not yet supported for species transport", __FILE__, __LINE__ );
    }

    // because we need the velocity tags for the one-sided convective flux treatment used
    // for NSCBCs, but this means SpeciesTransport depends on MomentumTransport...
    std::string xvelname, yvelname, zvelname;
    const Uintah::ProblemSpecP doxvel = momentumParams->get( "X-Velocity", xvelname );
    const Uintah::ProblemSpecP doyvel = momentumParams->get( "Y-Velocity", yvelname );
    const Uintah::ProblemSpecP dozvel = momentumParams->get( "Z-Velocity", zvelname );

    const Expr::Tag xVelTag = doxvel ? Expr::Tag(xvelname, Expr::STATE_NONE) : Expr::Tag();
    const Expr::Tag yVelTag = doyvel ? Expr::Tag(yvelname, Expr::STATE_NONE) : Expr::Tag();
    const Expr::Tag zVelTag = dozvel ? Expr::Tag(zvelname, Expr::STATE_NONE) : Expr::Tag();
    const Expr::TagList velTags = tag_list(xVelTag, yVelTag, zVelTag);

    return setup_species_equations( params,
                                    wasatchSpec,
                                    turbParams,
                                    densityTag,
                                    velTags,
                                    TagNames::self().temperature,
                                    gc,
                                    dualTimeMatrixInfo,
                                    computeKineticsJacobian );
#   else
    // nothing to do - return empty equation set.
    std::vector<EqnTimestepAdaptorBase*> eqns;
    return eqns;
#   endif
  }

  //==================================================================

  std::vector<EqnTimestepAdaptorBase*>
  parse_tar_and_soot_equations( Uintah::ProblemSpecP params,
                               const TurbulenceParameters& turbParams,
                               const Expr::Tag& densityTag,
                               GraphCategories& gc )
  {
    std::vector<EqnTimestepAdaptorBase*> adaptors;

#   ifdef HAVE_POKITT
    // setup a transport equation for tar
    adaptors.push_back(setup_tar_equation( params,
                                           turbParams,
                                           densityTag,
                                           gc ) );

    // setup a transport equation for soot
    adaptors.push_back(setup_soot_equation( params,
                                            turbParams,
                                            densityTag,
                                            gc ) );

    // setup a transport equation for soot particle number density
    adaptors.push_back(setup_soot_particle_equation( params,
                                                     turbParams,
                                                     densityTag,
                                                     gc ) );
    return adaptors;
#   else
    // nothing to do - return empty equation set.

    return adaptors;
#   endif
  }

  //==================================================================

  std::vector<EqnTimestepAdaptorBase*>
  parse_scalability_test( Uintah::ProblemSpecP params,
                         GraphCategories& gc )
  {
    typedef std::vector<EqnTimestepAdaptorBase*> EquationAdaptors;
    EquationAdaptors adaptors;

    std::string basePhiName;
    params->get( "SolutionVariable", basePhiName );

    int nEqs = 1;
    params->get( "NumberOfEquations", nEqs );

    GraphHelper* const icGraphHelper = gc[INITIALIZATION];

    for( int iEq=0; iEq<nEqs; iEq++ ){

      std::stringstream ss;
      ss << iEq;
      std::string thisPhiName = basePhiName + ss.str();
      // set initial condition and register it
      const Expr::Tag icTag( thisPhiName, Expr::STATE_NONE );
      const Expr::Tag indepVarTag( "XSVOL", Expr::STATE_NONE );
      typedef Expr::SinFunction<SVolField>::Builder Builder;
      icGraphHelper->exprFactory->register_expression( scinew Builder( icTag, indepVarTag, 1.0, 1, 0.0) );

      // create the transport equation with all-to-all source term
      typedef ScalabilityTestTransportEquation< SVolField > ScalTestEqn;
      EquationBase* scaltesteqn = scinew ScalTestEqn( gc, thisPhiName, params );
      adaptors.push_back( scinew EqnTimestepAdaptor< SVolField >( scaltesteqn ) );

      //_____________________________________________________
      // set up initial conditions on this equation
      try{
        proc0cout << "Setting initial conditions for scalability test equation: "
        << scaltesteqn->solution_variable_name()
        << std::endl;
        icGraphHelper->rootIDs.insert( scaltesteqn->initial_condition( *icGraphHelper->exprFactory ) );
      }
      catch( std::runtime_error& e ){
        std::ostringstream msg;
        msg << e.what()
        << std::endl
        << "ERROR while setting initial conditions on scalability test equation "
        << scaltesteqn->solution_variable_name()
        << std::endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
    }

    proc0cout << "------------------------------------------------" << std::endl;
    return adaptors;
  }

  //==================================================================

  void parse_poisson_equation( Uintah::ProblemSpecP poissonEqParams,
                              GraphCategories& gc,
                              Uintah::SolverInterface& linSolver,
                              Uintah::MaterialManagerP& materialManager )
  {
    std::string slnVariableName;
    poissonEqParams->get("SolutionVariable", slnVariableName);
    const Expr::Tag poissonVariableTag(slnVariableName, Expr::STATE_N);
    const Expr::TagList poissontags( tag_list( poissonVariableTag, Expr::Tag( poissonVariableTag.name() + "_rhs_poisson_expr", TagNames::self().pressure.context() ) ) );
    const Expr::Tag rhsTag = parse_nametag( poissonEqParams->findBlock("PoissonRHS")->findBlock("NameTag"));
    bool useRefPoint = false;
    double refValue = 0.0;
    Uintah::IntVector refLocation(0,0,0);

    if( poissonEqParams->findBlock("ReferenceValue") ){
      useRefPoint = true;
      Uintah::ProblemSpecP refPhiParams = poissonEqParams->findBlock("ReferenceValue");
      refPhiParams->getAttribute("value", refValue);
      refPhiParams->get("ReferenceCell", refLocation);
    }

    bool use3DLaplacian = true;
    poissonEqParams->getWithDefault( "Use3DLaplacian",use3DLaplacian, true );

    linSolver.readParameters( poissonEqParams, "" );
    linSolver.getParameters()->setSolveOnExtraCells( false );
    linSolver.getParameters()->setUseStencil4( true );
    linSolver.getParameters()->setOutputFileName( "WASATCH" );

    PoissonExpression::poissonTagList.push_back(poissonVariableTag);

    Expr::ExpressionBuilder* pbuilder  = new PoissonExpression::Builder( poissontags, rhsTag, useRefPoint, refValue, refLocation, use3DLaplacian, linSolver);
    Expr::ExpressionBuilder* pbuilder1 = new PoissonExpression::Builder( poissontags, rhsTag, useRefPoint, refValue, refLocation, use3DLaplacian, linSolver);

    GraphHelper* const icgraphHelper  = gc[INITIALIZATION  ];
    GraphHelper* const slngraphHelper = gc[ADVANCE_SOLUTION];

    const Expr::ExpressionID slnPoissonID = slngraphHelper->exprFactory->register_expression( pbuilder1 );
    slngraphHelper->exprFactory->cleave_from_parents(slnPoissonID);
    const Expr::ExpressionID icPoissonID = icgraphHelper->exprFactory->register_expression( pbuilder );
    //icgraphHelper->exprFactory->cleave_from_parents(icPoissonID);

    slngraphHelper->rootIDs.insert( slnPoissonID );
    icgraphHelper ->rootIDs.insert( icPoissonID  );
  }

  //==================================================================

  void parse_radiation_solver( Uintah::ProblemSpecP params,
                               GraphHelper& gh,
                               Uintah::SolverInterface& linSolver,
                               Uintah::MaterialManagerP& materialManager,
                               std::set<std::string>& persistentFields )
  {
    const Expr::Tag tempTag = parse_nametag( params->findBlock("Temperature")->findBlock("NameTag") );
    const Expr::Tag divQTag = parse_nametag( params->findBlock("DivQ")->findBlock("NameTag") );

    Expr::Tag absCoefTag;
    if( params->findBlock("AbsorptionCoefficient") )
      absCoefTag = parse_nametag( params->findBlock("AbsorptionCoefficient")->findBlock("NameTag") );

    if( params->findBlock("SimpleEmission") ){
      Uintah::ProblemSpecP envTempParams = params->findBlock("SimpleEmission")->findBlock("EnvironmentTemperature");
      if( envTempParams->findBlock("Constant") ){
        double envTempVal = 0.0;
        envTempParams->findBlock("Constant")->getAttribute( "value", envTempVal );
        gh.exprFactory->register_expression( new SimpleEmission<SVolField>::Builder( divQTag, tempTag, envTempVal, absCoefTag ) );
      }
      else{
        const Expr::Tag envTempTag = parse_nametag( envTempParams );
        gh.exprFactory->register_expression( new SimpleEmission<SVolField>::Builder( divQTag, tempTag, envTempTag, absCoefTag ) );
      }
    }
    else if( params->findBlock("DiscreteOrdinates") ){
      linSolver.readParameters( params, "" );

      int order = 2;
      params->findBlock("DiscreteOrdinates")->getAttribute("order",order);

      Expr::Tag scatCoef;  // currently we are not ready for scattering.
      //      if( params->findBlock("ScatteringCoefficient") ) parse_nametag( params->findBlock("ScatteringCoefficient") );

      const OrdinateDirections discOrd( order );
      Expr::TagList intensityTags;
      for( size_t i=0; i< discOrd.number_of_directions(); ++i ){
        const OrdinateDirections::SVec& svec = discOrd.get_ordinate_information(i);
        const std::string intensity( "intensity_" + boost::lexical_cast<std::string>(i) );
        std::cout << "registering expression for " << intensity << std::endl;
        DORadSolver::Builder* radSolver = new DORadSolver::Builder( intensity, svec, absCoefTag, scatCoef, tempTag, linSolver );
        const Expr::ExpressionID id = gh.exprFactory->register_expression( radSolver );
        gh.exprFactory->cleave_from_children( id );
        gh.exprFactory->cleave_from_parents ( id );
        BOOST_FOREACH( const Expr::Tag& tag, radSolver->get_tags() ){
          persistentFields.insert( tag.name() );
        }
      }
      gh.exprFactory->register_expression( new DORadSrc::Builder( divQTag, tempTag, absCoefTag, discOrd ) );
    }
  }

  //==================================================================

  void parse_var_den_mms( Uintah::ProblemSpecP wasatchParams,
                         Uintah::ProblemSpecP varDensMMSParams,
                         const bool computeContinuityResidual,
                         GraphCategories& gc )
  {
    std::string solnVarName, primVarName;
    double rho0=1.29985, rho1=0.081889, D=0.0658;
    varDensMMSParams->get("ConservedScalar",solnVarName);
    varDensMMSParams->get("Scalar",primVarName);
    if (wasatchParams->findBlock("TwoStreamMixing")){
    Uintah::ProblemSpecP varDensityParams = wasatchParams->findBlock("TwoStreamMixing");
    varDensityParams->getAttribute("rho0",rho0);
    varDensityParams->getAttribute("rho1",rho1);
    }

    const TagNames& tagNames = TagNames::self();

    const Expr::Tag fNP1Tag(primVarName, Expr::STATE_NP1);
    const Expr::Tag scalarEOSCouplingTag(primVarName + "_EOS_Coupling", Expr::STATE_NONE);

    for( Uintah::ProblemSpecP bcExprParams = wasatchParams->findBlock("BCExpression");
        bcExprParams != nullptr;
        bcExprParams = bcExprParams->findNextBlock("BCExpression") )
    {
      if( bcExprParams->findBlock("VarDenMMSMomentum") ){
        double bcRho0=1.29985, bcRho1=0.081889;
        Uintah::ProblemSpecP valParams = bcExprParams->findBlock("VarDenMMSMomentum");
        if (wasatchParams->findBlock("TwoStreamMixing")){
        Uintah::ProblemSpecP bDensityParams = wasatchParams->findBlock("TwoStreamMixing");
        bDensityParams->getAttribute("rho0",bcRho0);
        bDensityParams->getAttribute("rho1",bcRho1);
        }
        if( rho0!=bcRho0 || rho1!=bcRho1 ){
          std::ostringstream msg;
          msg << "ERROR: the values of rho0 and rho1 should be exactly the same in the \"VariableDensityMMS\" block and the \"VarDen1DMMSMomentum\" BCExpression. In \"VariableDensityMMS\" rho0=" << rho0 << " and rho1=" << rho1 << " while in \"VarDen1DMMSMomentum\" BCExpression rho0=" << bcRho0 << " and rho1=" << bcRho1 << std::endl;
          throw Uintah::InvalidValue( msg.str(), __FILE__, __LINE__ );
        }
      }

      else if( bcExprParams->findBlock("VarDenMMSDensity") ){
        double bcRho0=1.29985, bcRho1=0.081889;
        Uintah::ProblemSpecP valParams = bcExprParams->findBlock("VarDenMMSDensity");
        if (wasatchParams->findBlock("TwoStreamMixing")){
        Uintah::ProblemSpecP bDensityParams = wasatchParams->findBlock("TwoStreamMixing");
        bDensityParams->getAttribute("rho0",bcRho0);
        bDensityParams->getAttribute("rho1",bcRho1);
        }
        if( rho0!=bcRho0 || rho1!=bcRho1 ){
          std::ostringstream msg;
          msg << "ERROR: the values of rho0 and rho1 should be exactly the same in the \"VariableDensityMMS\" block and the \"VarDen1DMMSDensity\" BCExpression. In \"VariableDensityMMS\" rho0=" << rho0 << " and rho1=" << rho1 << " while in \"VarDen1DMMSDensity\" BCExpression rho0=" << bcRho0 << " and rho1=" << bcRho1 << std::endl;
          throw Uintah::InvalidValue( msg.str(), __FILE__, __LINE__ );
        }
      }

      else if( bcExprParams->findBlock("VarDenMMSSolnVar") ){
        double bcRho0=1.29985, bcRho1=0.081889;
        Uintah::ProblemSpecP valParams = bcExprParams->findBlock("VarDenMMSSolnVar");
        if (wasatchParams->findBlock("TwoStreamMixing")){
        Uintah::ProblemSpecP bDensityParams = wasatchParams->findBlock("TwoStreamMixing");
        bDensityParams->getAttribute("rho0",bcRho0);
        bDensityParams->getAttribute("rho1",bcRho1);
        }
        if( rho0!=bcRho0 || rho1!=bcRho1 ){
          std::ostringstream msg;
          msg << "ERROR: the values of rho0 and rho1 should be exactly the same in the \"VariableDensityMMS\" block and the \"VarDen1DMMSSolnVar\" BCExpression. In \"VariableDensityMMS\" rho0=" << rho0 << " and rho1=" << rho1 << " while in \"VarDen1DMMSSolnVar\" BCExpression rho0=" << bcRho0 << " and rho1=" << bcRho1 << std::endl;
          throw Uintah::InvalidValue( msg.str(), __FILE__, __LINE__ );
        }
      }
    }
    varDensMMSParams->get("D",D);

    const Expr::Tag solnVarRHSTag = Expr::Tag(solnVarName+"_rhs",Expr::STATE_NONE);

    Expr::ExpressionFactory& factory = *gc[ADVANCE_SOLUTION]->exprFactory;
    typedef VarDen1DMMSMixFracSrc<SVolField>::Builder MMSMixFracSrc;

    /* We can't build MMS mixture fraction source at initialization because the time field time field is not initialized.
     * So STATE_N and STATE_NP1 values of the mixture fraction source are calculated at each time step.
     */
    factory.register_expression( new MMSMixFracSrc(tagNames.mms_mixfracsrc,tagNames.xsvolcoord, tagNames.time, tagNames.dt, D, rho0, rho1, false));
    factory.attach_dependency_to_expression(tagNames.mms_mixfracsrc, solnVarRHSTag);

    const  Expr::Tag mmsMixfracSrcNP1Tag = Expr::Tag(tagNames.mms_mixfracsrc.name() + "_NP1", Expr::STATE_NONE);
    factory.register_expression( new MMSMixFracSrc(mmsMixfracSrcNP1Tag,tagNames.xsvolcoord, tagNames.time, tagNames.dt, D, rho0, rho1, true));

    Uintah::ProblemSpecP densityParams = wasatchParams->findBlock("Density");
    std::string densityName;
    densityParams->findBlock("NameTag")->getAttribute( "name", densityName );
    const Expr::Tag densityTag  = Expr::Tag(densityName, Expr::STATE_N  );
    const Expr::Tag densNP1Tag  = Expr::Tag(densityName, Expr::STATE_NP1);
    const Expr::Tag drhodfTag   = tagNames.derivative_tag( densityTag, fNP1Tag );

    // attach Sf_{n+1} to the scalar EOS coupling term
    const Expr::Tag mms_EOSMixFracSrcTag(tagNames.mms_mixfracsrc.name() + "_EOS", Expr::STATE_NONE);
    factory.register_expression( new VarDenEOSCouplingMixFracSrc<SVolField>::Builder(mms_EOSMixFracSrcTag, mmsMixfracSrcNP1Tag, densNP1Tag, drhodfTag));

    factory.attach_dependency_to_expression(mms_EOSMixFracSrcTag, scalarEOSCouplingTag);


    std::string xvelname, yvelname, zvelname;
    Uintah::ProblemSpecP doxvel,doyvel, dozvel;
    Expr::TagList velTags;
    Uintah::ProblemSpecP momEqnParams  = wasatchParams->findBlock("MomentumEquations");
    doxvel = momEqnParams->get( "X-Velocity", xvelname );
    doyvel = momEqnParams->get( "Y-Velocity", yvelname );
    dozvel = momEqnParams->get( "Z-Velocity", zvelname );
    if( doxvel ) velTags.push_back(Expr::Tag(xvelname, Expr::STATE_NONE));
    else         velTags.push_back( Expr::Tag() );
    if( doyvel ) velTags.push_back(Expr::Tag(yvelname, Expr::STATE_NONE));
    else         velTags.push_back( Expr::Tag() );
    if( dozvel ) velTags.push_back(Expr::Tag(zvelname, Expr::STATE_NONE));
    else         velTags.push_back( Expr::Tag() );

    factory.register_expression( new VarDen1DMMSContinuitySrc<SVolField>::Builder( tagNames.mms_continuitysrc, rho0, rho1, tagNames.xsvolcoord, tagNames.time, tagNames.dt));
    factory.register_expression( new VarDen1DMMSPressureContSrc<SVolField>::Builder( tagNames.mms_pressurecontsrc, tagNames.mms_continuitysrc, densNP1Tag, fNP1Tag, drhodfTag, tagNames.dt));
    factory.attach_dependency_to_expression(tagNames.mms_pressurecontsrc, scalarEOSCouplingTag);

    if (computeContinuityResidual)
    {
      factory.attach_dependency_to_expression(tagNames.mms_continuitysrc, tagNames.drhodt);
    }
  }

  //==================================================================

  void parse_var_den_oscillating_mms( Uintah::ProblemSpecP wasatchParams,
                                     Uintah::ProblemSpecP varDens2DMMSParams,
                                     const bool computeContinuityResidual,
                                     GraphCategories& gc)
  {
    std::string solnVarName, primVarName;
    double rho0, rho1, d, w, k, uf, vf;
    const Expr::Tag diffTag = parse_nametag( varDens2DMMSParams->findBlock("DiffusionCoefficient")->findBlock("NameTag") );
    varDens2DMMSParams->get("ConservedScalar",solnVarName);
    varDens2DMMSParams->get("Scalar",primVarName);
    varDens2DMMSParams->getAttribute("rho0",rho0);
    varDens2DMMSParams->getAttribute("rho1",rho1);
    varDens2DMMSParams->getAttribute("uf",uf);
    varDens2DMMSParams->getAttribute("vf",vf);
    varDens2DMMSParams->getAttribute("k",k);
    varDens2DMMSParams->getAttribute("w",w);
    varDens2DMMSParams->getAttribute("d",d);

    const TagNames& tagNames = TagNames::self();

    Uintah::ProblemSpecP densityParams = wasatchParams->findBlock("Density");
    Uintah::ProblemSpecP momEqnParams  = wasatchParams->findBlock("MomentumEquations");

    const Expr::Tag densityTag = parse_nametag( densityParams->findBlock("NameTag") );
    const Expr::Tag densNP1Tag = Expr::Tag( densityTag.name(), Expr::STATE_NP1 );
    const Expr::Tag solnVarRHSTag( solnVarName+"_rhs", Expr::STATE_NONE );

    const Expr::Tag drhodfTag = tagNames.derivative_tag( densityTag, primVarName );
    const Expr::Tag scalarEOSCouplingTag(primVarName + "_EOS_Coupling", Expr::STATE_NONE);


    std::string x1="X", x2="Y";
    if( varDens2DMMSParams->findAttribute("x1") ) varDens2DMMSParams->getAttribute("x1",x1);
    if( varDens2DMMSParams->findAttribute("x2") ) varDens2DMMSParams->getAttribute("x2",x2);

    Expr::Tag x1Tag, x2Tag, x1XTag, x1YTag, x2XTag, x2YTag;

    if      (x1 == "X")  x1Tag = tagNames.xsvolcoord;
    else if (x1 == "Y")  x1Tag = tagNames.ysvolcoord;
    else if (x1 == "Z")  x1Tag = tagNames.zsvolcoord;

    if      (x2 == "X")  x2Tag = tagNames.xsvolcoord;
    else if (x2 == "Y")  x2Tag = tagNames.ysvolcoord;
    else if (x2 == "Z")  x2Tag = tagNames.zsvolcoord;

    Expr::ExpressionFactory& factory = *gc[ADVANCE_SOLUTION]->exprFactory;
    typedef VarDenMMSOscillatingMixFracSrc<SVolField>::Builder MMSMixFracSrc;

    // attach the mixture fraction source term, Sf_n to the RHS of the mixture fraction equation
    factory.register_expression( new MMSMixFracSrc(tagNames.mms_mixfracsrc, x1Tag, x2Tag, tagNames.time, rho0, rho1, d, w, k, uf, vf, false));
    factory.attach_dependency_to_expression(tagNames.mms_mixfracsrc, solnVarRHSTag);

    // We need to compute the EOS coupling term at n+1. This term requires the mixture fraction src at n+1,
    // here we register that term: Sf_{n+1}
    const Expr::Tag mmsMixfracSrcNP1Tag = Expr::Tag(tagNames.mms_mixfracsrc.name() + "_NP1", Expr::STATE_NONE);
    factory.register_expression( new MMSMixFracSrc(mmsMixfracSrcNP1Tag, x1Tag, x2Tag, tagNames.time, rho0, rho1, d, w, k, uf, vf, true));

    // attach Sf_{n+1} to the scalar EOS coupling term
    const Expr::Tag mms_EOSMixFracSrcTag(tagNames.mms_mixfracsrc.name() + "_EOS", Expr::STATE_NONE);
    factory.register_expression( new VarDenEOSCouplingMixFracSrc<SVolField>::Builder(mms_EOSMixFracSrcTag, mmsMixfracSrcNP1Tag, densNP1Tag, drhodfTag));

    factory.attach_dependency_to_expression(mms_EOSMixFracSrcTag, scalarEOSCouplingTag);
  }

  //==================================================================

  std::vector<EqnTimestepAdaptorBase*>
  parse_momentum_equations( Uintah::ProblemSpecP wasatchSpec,
                            const TurbulenceParameters turbParams,
                            const bool useAdaptiveDt,
                            const bool doParticles,
                            const Expr::Tag densityTag,
                            GraphCategories& gc,
                            Uintah::SolverInterface& linSolver,
                            Uintah::MaterialManagerP& materialManager,
                            WasatchCore::DualTimeMatrixInfo& dualTimeMatrixInfo,
                            std::set<std::string>& persistentFields )
  {
    typedef std::vector<EqnTimestepAdaptorBase*> EquationAdaptors;
    EquationAdaptors adaptors;

    const bool isCompressible = (Wasatch::flow_treatment() == COMPRESSIBLE);

    Uintah::ProblemSpecP momentumSpec = wasatchSpec->findBlock("MomentumEquations");
    std::string xvelname, yvelname, zvelname;
    const Uintah::ProblemSpecP doxvel = momentumSpec->get( "X-Velocity", xvelname );
    const Uintah::ProblemSpecP doyvel = momentumSpec->get( "Y-Velocity", yvelname );
    const Uintah::ProblemSpecP dozvel = momentumSpec->get( "Z-Velocity", zvelname );

    std::string xmomname, ymomname, zmomname;
    const Uintah::ProblemSpecP doxmom = momentumSpec->get( "X-Momentum", xmomname );
    const Uintah::ProblemSpecP doymom = momentumSpec->get( "Y-Momentum", ymomname );
    const Uintah::ProblemSpecP dozmom = momentumSpec->get( "Z-Momentum", zmomname );

    const Expr::Tag viscTag = (momentumSpec->findBlock("Viscosity"))
        ? parse_nametag( momentumSpec->findBlock("Viscosity")->findBlock("NameTag") )
        : Expr::Tag();
#   ifdef HAVE_POKITT
    if( viscTag != Expr::Tag() ){
      if( momentumSpec->findBlock("Viscosity")->findBlock("FromPoKiTT") ){
        typedef pokitt::Viscosity<SVolField>::Builder Visc;
        Expr::TagList yiTags;
        for( int i=0; i<CanteraObjects::number_species(); ++i ){
          yiTags.push_back( Expr::Tag( CanteraObjects::species_name(i), Expr::STATE_NONE ) );
        }
        gc[ADVANCE_SOLUTION]->exprFactory->register_expression( scinew Visc( viscTag, TagNames::self().temperature, yiTags ) );
      }
    }
#   endif

    if( isCompressible && momentumSpec->findBlock("Pressure") ){
      std::ostringstream msg;
      msg << "ERROR: There is no need to specify a pressure tag in the momentum equations block."
          << "Please revise your input file" << std::endl;
      throw Uintah::InvalidValue( msg.str(), __FILE__, __LINE__ );
    }

    // check if none of the momentum directions were specified
    if( !(doxvel || doyvel || dozvel) ){
      std::ostringstream msg;
      msg << "ERROR: No Direction was specified for momentum equations."
      << "Please revise your input file" << std::endl;
      throw Uintah::InvalidValue( msg.str(), __FILE__, __LINE__ );
    }

    // parse body force expression
    std::string bodyForceDir;
    Expr::Tag xBodyForceTag, yBodyForceTag, zBodyForceTag;
    for( Uintah::ProblemSpecP bodyForceParams=momentumSpec->findBlock("BodyForce");
        bodyForceParams != nullptr;
        bodyForceParams=bodyForceParams->findNextBlock("BodyForce") ){
      bodyForceParams->getAttribute("direction", bodyForceDir );
      if (bodyForceDir == "X") xBodyForceTag = parse_nametag( bodyForceParams->findBlock("NameTag") );
      if (bodyForceDir == "Y") yBodyForceTag = parse_nametag( bodyForceParams->findBlock("NameTag") );
      if (bodyForceDir == "Z") zBodyForceTag = parse_nametag( bodyForceParams->findBlock("NameTag") );
    }

    // parse source expression
    std::string srcTermDir;
    Expr::TagList xSrcTermTags, ySrcTermTags, zSrcTermTags;

    for( Uintah::ProblemSpecP srcTermParams=momentumSpec->findBlock("SourceTerm");
        srcTermParams != nullptr;
        srcTermParams=srcTermParams->findNextBlock("SourceTerm") ){
      srcTermParams->getAttribute("direction", srcTermDir );
      if (srcTermDir == "X") xSrcTermTags.push_back( parse_nametag( srcTermParams->findBlock("NameTag") ) );
      if (srcTermDir == "Y") ySrcTermTags.push_back( parse_nametag( srcTermParams->findBlock("NameTag") ) );
      if (srcTermDir == "Z") zSrcTermTags.push_back( parse_nametag( srcTermParams->findBlock("NameTag") ) );
    }

    GraphHelper* const solnGraphHelper = gc[ADVANCE_SOLUTION  ];
    GraphHelper* const icGraphHelper   = gc[INITIALIZATION    ];

    //___________________________________________________________________________‰‰
    // resolve the momentum equation to be solved and create the adaptor for it.
    //
    proc0cout << "------------------------------------------------" << std::endl
              << "Creating momentum equations..." << std::endl;

    if( isCompressible ){
      std::string rhoETotalName, eTotalName;
      Expr::Tag e0Tag;
      Uintah::ProblemSpecP energySpec = wasatchSpec->findBlock("EnergyEquation");
      if( !energySpec ){
        std::ostringstream msg;
        msg << "ERROR: When solving a compressible flow problem you must specify an energy equation." << std::endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      } else {
        energySpec->get("SolutionVariable", rhoETotalName);
        energySpec->get("PrimitiveVariable", eTotalName);
        e0Tag = Expr::Tag(eTotalName, Expr::STATE_NONE);
      }

      dualTimeMatrixInfo.density = densityTag;
      dualTimeMatrixInfo.viscosity = viscTag;
      dualTimeMatrixInfo.temperature = TagNames::self().temperature;
      dualTimeMatrixInfo.pressure = TagNames::self().pressure;
      dualTimeMatrixInfo.doCompressible = true;

      Expr::Tag xVelTag, yVelTag, zVelTag;
      if( doxvel && doxmom ) {
        proc0cout << "Setting up X momentum transport equation" << std::endl;

        typedef CompressibleMomentumTransportEquation<SpatialOps::XDIR> XMomEq;
        EquationBase* momtranseq = scinew XMomEq( wasatchSpec,
                                                  WasatchCore::XDIR,
                                                  xvelname,
                                                  xmomname,
                                                  densityTag,
                                                  TagNames::self().temperature,
                                                  TagNames::self().mixMW,
                                                  e0Tag,
                                                  xBodyForceTag,
                                                  xSrcTermTags,
                                                  gc,
                                                  momentumSpec,
                                                  turbParams );

        adaptors.push_back( scinew EqnTimestepAdaptor<SVolField>(momtranseq) );

        xVelTag = Expr::Tag(xvelname, Expr::STATE_NONE);

        dualTimeMatrixInfo.doX = true;
        dualTimeMatrixInfo.xVelocity = xVelTag;
        dualTimeMatrixInfo.xMomentum = Expr::Tag( xmomname, Expr::STATE_DYNAMIC );
      }

      if( doyvel && doymom ){
        proc0cout << "Setting up Y momentum transport equation" << std::endl;

        typedef CompressibleMomentumTransportEquation<SpatialOps::YDIR> YMomEq;
        EquationBase* momtranseq = scinew YMomEq( wasatchSpec,
                                                  WasatchCore::YDIR, yvelname,
                                                  ymomname,
                                                  densityTag,
                                                  TagNames::self().temperature,
                                                  TagNames::self().mixMW,
                                                  e0Tag,
                                                  yBodyForceTag,
                                                  ySrcTermTags,
                                                  gc,
                                                  momentumSpec,
                                                  turbParams );

        adaptors.push_back( scinew EqnTimestepAdaptor<SVolField>(momtranseq) );

        yVelTag = Expr::Tag(yvelname, Expr::STATE_NONE);

        dualTimeMatrixInfo.doY = true;
        dualTimeMatrixInfo.yVelocity = yVelTag;
        dualTimeMatrixInfo.yMomentum = Expr::Tag( ymomname, Expr::STATE_DYNAMIC );
      }

      if( dozvel && dozmom ){
        proc0cout << "Setting up Z momentum transport equation" << std::endl;
        typedef CompressibleMomentumTransportEquation<SpatialOps::ZDIR> ZMomEq;
        EquationBase* momtranseq = scinew ZMomEq( wasatchSpec,
                                                  WasatchCore::ZDIR, zvelname,
                                                  zmomname,
                                                  densityTag,
                                                  TagNames::self().temperature,
                                                  TagNames::self().mixMW,
                                                  e0Tag,
                                                  zBodyForceTag,
                                                  zSrcTermTags,
                                                  gc,
                                                  momentumSpec,
                                                  turbParams );

        adaptors.push_back( scinew EqnTimestepAdaptor<SVolField>(momtranseq) );

        zVelTag = Expr::Tag(zvelname, Expr::STATE_NONE);

        dualTimeMatrixInfo.doZ = true;
        dualTimeMatrixInfo.zVelocity = zVelTag;
        dualTimeMatrixInfo.zMomentum = Expr::Tag( zmomname, Expr::STATE_DYNAMIC );
      }


      // register continuity equation
      EquationBase* contEq = scinew ContinuityTransportEquation( densityTag,
                                                                 TagNames::self().temperature,
                                                                 TagNames::self().mixMW,
                                                                 gc,
                                                                 xVelTag,
                                                                 yVelTag,
                                                                 zVelTag,
                                                                 wasatchSpec );
      adaptors.push_back( scinew EqnTimestepAdaptor<SVolField>(contEq) );

      // register total internal energy equation
      const Expr::TagList velTags = tag_list(xVelTag, yVelTag, zVelTag);
      const Expr::TagList bodyForceTags = tag_list(xBodyForceTag,yBodyForceTag,zBodyForceTag);
      proc0cout << "Creating TotalInternalEnergyTransportEquation" << std::endl;
      EquationBase* totalEEq = scinew TotalInternalEnergyTransportEquation( rhoETotalName,
                                                                            wasatchSpec,
                                                                            energySpec,
                                                                            gc,
                                                                            densityTag,
                                                                            TagNames::self().temperature,
                                                                            TagNames::self().pressure,
                                                                            velTags,
                                                                            bodyForceTags,
                                                                            viscTag,
                                                                            TagNames::self().dilatation,
                                                                            turbParams,
                                                                            dualTimeMatrixInfo,
                                                                            persistentFields );
      adaptors.push_back( scinew EqnTimestepAdaptor<SVolField>(totalEEq) );

    } // isCompressible
    else{ // low mach

      /* Fix density context passed to Low-Mach constructor as STATE_N. This is done because the algorithm used
       * requires solving for density at the next time level in order to obtain an estimate of the velocity
       * divergence field, which then becomes density at STATE_N for the time step that follows, etc, etc.
       */
      Expr::Tag lowMachDensityTag = Wasatch::flow_treatment() == INCOMPRESSIBLE ?
                                    densityTag :
                                    Expr::Tag(densityTag.name(), Expr::STATE_N);

      if( doxvel && doxmom ){
        proc0cout << "Setting up X momentum transport equation" << std::endl;
        typedef LowMachMomentumTransportEquation< XVolField > MomTransEq;
        EquationBase* momtranseq = scinew MomTransEq( WasatchCore::XDIR, xvelname,
                                                      xmomname,
                                                      lowMachDensityTag,
                                                      xBodyForceTag,
                                                      xSrcTermTags,
                                                      gc,
                                                      momentumSpec,
                                                      turbParams,
                                                      linSolver, materialManager );
        adaptors.push_back( scinew EqnTimestepAdaptor<XVolField>(momtranseq) );
      }

      if( doyvel && doymom ){
        proc0cout << "Setting up Y momentum transport equation" << std::endl;
        typedef LowMachMomentumTransportEquation< YVolField > MomTransEq;
        EquationBase* momtranseq = scinew MomTransEq( WasatchCore::YDIR, yvelname,
                                                      ymomname,
                                                      lowMachDensityTag,
                                                      yBodyForceTag,
                                                      ySrcTermTags,
                                                      gc,
                                                      momentumSpec,
                                                      turbParams,
                                                      linSolver,materialManager );
        adaptors.push_back( scinew EqnTimestepAdaptor<YVolField>(momtranseq) );
      }

      if( dozvel && dozmom ){
        proc0cout << "Setting up Z momentum transport equation" << std::endl;
        typedef LowMachMomentumTransportEquation< ZVolField > MomTransEq;
        EquationBase* momtranseq = scinew MomTransEq( WasatchCore::ZDIR, zvelname,
                                                      zmomname,
                                                      lowMachDensityTag,
                                                      zBodyForceTag,
                                                      zSrcTermTags,
                                                      gc,
                                                      momentumSpec,
                                                      turbParams,
                                                      linSolver,materialManager );
        adaptors.push_back( scinew EqnTimestepAdaptor<ZVolField>(momtranseq) );
      }
    }

    //
    // ADD ADAPTIVE TIMESTEPPING
    if( useAdaptiveDt ){
      const Expr::Tag xVelTag = doxvel ? Expr::Tag(xvelname, Expr::STATE_NONE) : Expr::Tag();
      const Expr::Tag yVelTag = doyvel ? Expr::Tag(yvelname, Expr::STATE_NONE) : Expr::Tag();
      const Expr::Tag zVelTag = dozvel ? Expr::Tag(zvelname, Expr::STATE_NONE) : Expr::Tag();

      Expr::Tag puTag, pvTag, pwTag;
      if (doParticles) {
        Uintah::ProblemSpecP particleSpec = wasatchSpec->findBlock("ParticleTransportEquations");
        Uintah::ProblemSpecP particleMomSpec = particleSpec->findBlock("ParticleMomentum");
        std::string puname, pvname,pwname;
        particleMomSpec->getAttribute("x",puname);
        particleMomSpec->getAttribute("y",pvname);
        particleMomSpec->getAttribute("z",pwname);
        puTag = Expr::Tag(puname,Expr::STATE_DYNAMIC);
        pvTag = Expr::Tag(pvname,Expr::STATE_DYNAMIC);
        pwTag = Expr::Tag(pwname,Expr::STATE_DYNAMIC);
      }
      Expr::ExpressionID stabDtID;
      if (isCompressible) {
        stabDtID = solnGraphHelper->exprFactory->register_expression(scinew StableTimestep<SVolField,SVolField,SVolField>::Builder( TagNames::self().stableTimestep,
                                                                                                                                                            densityTag,
                                                                                                                                                            viscTag,
                                                                                                                                                            xVelTag,yVelTag,zVelTag, puTag, pvTag, pwTag, TagNames::self().soundspeed ), true);

      } else {
        stabDtID = solnGraphHelper->exprFactory->register_expression(scinew StableTimestep<XVolField,YVolField,ZVolField>::Builder( TagNames::self().stableTimestep,
                                                                                                                                                            densityTag,
                                                                                                                                                            viscTag,
                                                                                                                                                            xVelTag,yVelTag,zVelTag, puTag, pvTag, pwTag, Expr::Tag() ), true);

      }
      // force this onto the graph.
      solnGraphHelper->rootIDs.insert( stabDtID );
    }

    //
    // loop over the local adaptors and set the initial and boundary conditions on each equation attached to that adaptor
    for( EquationAdaptors::const_iterator ia=adaptors.begin(); ia!=adaptors.end(); ++ia ){
      EqnTimestepAdaptorBase* const adaptor = *ia;
      EquationBase* momtranseq = adaptor->equation();
      //_____________________________________________________
      // set up initial conditions on this momentum equation
      try{
        proc0cout << "Setting initial conditions for momentum equation: "
        << momtranseq->solution_variable_name()
        << std::endl;
        icGraphHelper->rootIDs.insert( momtranseq->initial_condition( *icGraphHelper->exprFactory ) );
      }
      catch( std::runtime_error& e ){
        std::ostringstream msg;
        msg << e.what()
        << std::endl
        << "ERORR while setting initial conditions on momentum equation "
        << momtranseq->solution_variable_name()
        << std::endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
    }

    //_____________________________________________________
    // set up initial conditions on the pressure
    try{
      proc0cout << "Setting initial conditions for pressure: "
      << TagNames::self().pressure.name()
      << std::endl;
      icGraphHelper->rootIDs.insert( (*icGraphHelper->exprFactory).get_id( TagNames::self().pressure ) );
    }
    catch( std::runtime_error& e ){
      std::ostringstream msg;
      msg << e.what()
      << std::endl
      << "ERORR while setting initial conditions on pressure. "
      << TagNames::self().pressure.name()
      << std::endl;
      throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
    }

    proc0cout << "------------------------------------------------" << std::endl;
    //
    return adaptors;
  }

  //==================================================================

  std::vector<EqnTimestepAdaptorBase*>
  parse_moment_transport_equations( Uintah::ProblemSpecP params,
                                   Uintah::ProblemSpecP wasatchParams,
                                   GraphCategories& gc )
  {
    typedef std::vector<EqnTimestepAdaptorBase*> EquationAdaptors;
    EquationAdaptors adaptors;

    proc0cout << "Parsing moment transport equations\n";
    GraphHelper* const solnGraphHelper = gc[ADVANCE_SOLUTION];
    GraphHelper* const icGraphHelper   = gc[INITIALIZATION  ];
    Expr::TagList transportedMomentTags;
    Expr::TagList abscissaeTags;
    Expr::TagList weightsTags;

    std::string basePhiName;
    params->get( "PopulationName", basePhiName );
    basePhiName = "m_" + basePhiName;

    int nEnv = 1;
    params->get( "NumberOfEnvironments", nEnv );
    const int nEqs = 2*nEnv;

    //________________________
    //get the initial moments
    std::vector< double> initialMoments;
    for( Uintah::ProblemSpecP exprParams = wasatchParams->findBlock("MomentInitialization");
        exprParams != nullptr;
        exprParams = exprParams->findNextBlock("MomentInitialization") )
    {
      std::string populationName;
      exprParams->get("PopulationName", populationName);
      const std::string inputMomentName = "m_" + populationName;

      if( basePhiName.compare(inputMomentName) == 0 ){
        exprParams->get("Values", initialMoments,nEqs);
      }
    }

    for( int iMom=0; iMom<nEqs; iMom++ ){
      const double momentID = (double) iMom; //here we will add any fractional moments
      std::stringstream ss;
      ss << iMom;
      std::string thisPhiName = basePhiName + "_" + ss.str();

      // create moment transport equation
      typedef MomentTransportEquation< SVolField > MomTransEq;
      EquationBase* momtranseq = scinew MomTransEq( thisPhiName,
                                                   gc,
                                                   momentID,
                                                   params,
                                                   initialMoments[iMom] );

      adaptors.push_back( scinew EqnTimestepAdaptor< SVolField >( momtranseq ) );

      // tsaad: MUST INSERT ROOT IDS INTO THE SOLUTION GRAPH HELPER. WE NEVER DO
      // THAT ELSEWHERE, BUT THIS IS NEEDED TO MAKE THINGS EASIER WHEN USING
      // WASATCH IN ARCHES.
      solnGraphHelper->rootIDs.insert( momtranseq->get_rhs_id() );
    }

    //
    // loop over the local adaptors and set the initial and boundary conditions on each equation attached to that adaptor
    for( EquationAdaptors::const_iterator ia=adaptors.begin(); ia!=adaptors.end(); ++ia ){
      EqnTimestepAdaptorBase* const adaptor = *ia;
      EquationBase* momtranseq = adaptor->equation();

      //_____________________________________________________
      // set up initial conditions on this moment equation
      try{
        proc0cout << "Setting initial conditions for moment transport equation: "
        << momtranseq->solution_variable_name()
        << std::endl;
        icGraphHelper->rootIDs.insert( momtranseq->initial_condition( *icGraphHelper->exprFactory ) );
      }
      catch( std::runtime_error& e ){
        std::ostringstream msg;
        msg << e.what()
        << std::endl
        << "ERORR while setting initial conditions on moment transport equation "
        << momtranseq->solution_variable_name()
        << std::endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
    }

    proc0cout << "------------------------------------------------" << std::endl;
    return adaptors;
  }

  //------------------------------------------------------------------
} // namespace WasatchCore
