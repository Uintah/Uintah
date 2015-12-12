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

//-- Wasatch Includes --//
#include "ParseEquation.h"
#include <CCA/Components/Wasatch/Wasatch.h>
#include <CCA/Components/Wasatch/TimeStepper.h>
#include <CCA/Components/Wasatch/TagNames.h>
#include <CCA/Components/Wasatch/ParseTools.h>
#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>

//-- Add headers for individual transport equations here --//
#include "TransportEquation.h"
#include "ScalarTransportEquation.h"
#include "ScalabilityTestTransportEquation.h"
#include "MomentumTransportEquationBase.h"
#include "LowMachMomentumTransportEquation.h"
#include "CompressibleMomentumTransportEquation.h"
#include "TotalInternalEnergyTransportEquation.h"
#include "EquationBase.h"

#include "MomentTransportEquation.h"
#include "ParticlePositionEquation.h"
#include "ParticleMomentumEquation.h"
#include "ParticleSizeEquation.h"
#include "ParticleMassEquation.h"
#include "EnthalpyTransportEquation.h"

//-- includes for the expressions built here --//
#include <CCA/Components/Wasatch/Expressions/PBE/QMOM.h>
#include <CCA/Components/Wasatch/Expressions/ConvectiveFlux.h>
#include <CCA/Components/Wasatch/Expressions/PoissonExpression.h>
#include <CCA/Components/Wasatch/ConvectiveInterpolationMethods.h>
#include <CCA/Components/Wasatch/Expressions/DiffusiveFlux.h>
#include <CCA/Components/Wasatch/Expressions/DiffusiveVelocity.h>
#include <CCA/Components/Wasatch/Expressions/StableTimestep.h>
#include <CCA/Components/Wasatch/Expressions/Pressure.h>
#include <CCA/Components/Wasatch/Expressions/MMS/Functions.h>
#include <CCA/Components/Wasatch/Expressions/MMS/VardenMMS.h>
#include <CCA/Components/Wasatch/Expressions/MMS/Varden2DMMS.h>
#include <CCA/Components/Wasatch/Expressions/Particles/ParticleGasMomentumSrc.h>
#include <CCA/Components/Wasatch/Expressions/SimpleEmission.h>
#include <CCA/Components/Wasatch/Expressions/DORadSolver.h>

//-- Uintah includes --//
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Parallel/Parallel.h>

//-- Expression Library includes --//
#include <expression/ExpressionFactory.h>

#include <iostream>

namespace WasatchCore{


  /**
   *  \class EqnTimestepAdaptor
   *  \authors James C. Sutherland, Tony Saad, Amir Biglari
   *  \date July, 2011. (Originally created: June, 2010).
   *
   *  \brief Strongly typed adaptor provides the key functionality to
   *         plug a transport equation into a TimeStepper.
   */
  template< typename FieldT >
  class EqnTimestepAdaptor : public EqnTimestepAdaptorBase
  {
  public:
    EqnTimestepAdaptor( EquationBase* eqn ) : EqnTimestepAdaptorBase(eqn) {}
    void hook( TimeStepper& ts ) const
    {
      ts.add_equation<FieldT>( eqn_->solution_variable_name(),
                               eqn_->get_rhs_id() );
    }
  };

  //==================================================================

  EqnTimestepAdaptorBase::EqnTimestepAdaptorBase( EquationBase* eqn )
    : eqn_(eqn)
  {}

  //------------------------------------------------------------------

  EqnTimestepAdaptorBase::~EqnTimestepAdaptorBase()
  {
    delete eqn_;
  }

  //==================================================================

  EqnTimestepAdaptorBase* parse_scalar_equation( Uintah::ProblemSpecP params,
                                                 TurbulenceParameters turbParams,
                                                 const Expr::Tag densityTag,
                                                 const bool isConstDensity,
                                                 GraphCategories& gc )
  {
    EqnTimestepAdaptorBase* adaptor = NULL;
    EquationBase* transeqn = NULL;

    std::string eqnLabel, solnVariable;

    params->getAttribute( "equation", eqnLabel );
    params->get( "SolutionVariable", solnVariable );


    //___________________________________________________________________________
    // resolve the transport equation to be solved and create the adaptor for it.
    //
    proc0cout << "------------------------------------------------" << std::endl
              << "Creating transport equation for '" << solnVariable << "'" << std::endl;

    if( eqnLabel == "generic" ){
       typedef ScalarTransportEquation< SVolField > ScalarTransEqn;
        transeqn = scinew ScalarTransEqn( ScalarTransEqn::get_solnvar_name( params ),
                                          params,
                                          gc,
                                          densityTag,
                                          isConstDensity,
                                          turbParams );
        adaptor = scinew EqnTimestepAdaptor< SVolField >( transeqn );
    }
    else if( eqnLabel == "enthalpy" ){
      typedef EnthalpyTransportEquation TransEqn;
      transeqn = scinew TransEqn( ScalarTransportEquation<SVolField>::get_solnvar_name(params),
                                  params,
                                  gc,
                                  densityTag,
                                  isConstDensity,
                                  turbParams );
      adaptor = scinew EqnTimestepAdaptor<SVolField>(transeqn);
    }
    else {
      std::ostringstream msg;
      msg << "ERROR: No transport equation was specified '" << eqnLabel << "'. Please revise your input file" << std::endl;
      throw Uintah::InvalidValue( msg.str(), __FILE__, __LINE__ );
    }

    assert( transeqn != NULL );
    assert( adaptor  != NULL );

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
                << "ERORR while setting initial conditions on scalability test equation "
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
                               Uintah::SimulationStateP& sharedState )
  {
    std::string slnVariableName;
    poissonEqParams->get("SolutionVariable", slnVariableName);
    const Expr::Tag poissonVariableTag(slnVariableName, Expr::STATE_N);
    const Expr::TagList poissontags( tag_list( poissonVariableTag, Expr::Tag( poissonVariableTag.name() + "_rhs_poisson_expr", TagNames::self().pressure.context() ) ) );
    const Expr::Tag rhsTag = parse_nametag( poissonEqParams->findBlock("PoissonRHS")->findBlock("NameTag"));    
    bool useRefPoint = false;
    double refValue = 0.0;
    SCIRun::IntVector refLocation(0,0,0);
    
    if( poissonEqParams->findBlock("ReferenceValue") ){
      useRefPoint = true;
      Uintah::ProblemSpecP refPhiParams = poissonEqParams->findBlock("ReferenceValue");      
      refPhiParams->getAttribute("value", refValue);
      refPhiParams->get("ReferenceCell", refLocation);
    }
    
    bool use3DLaplacian = true;
    poissonEqParams->getWithDefault( "Use3DLaplacian",use3DLaplacian, true );
        
    Uintah::SolverParameters* sparams = linSolver.readParameters( poissonEqParams, "", sharedState );
    sparams->setSolveOnExtraCells( false );
    sparams->setUseStencil4( true );
    sparams->setOutputFileName( "WASATCH" );
    
    PoissonExpression::poissonTagList.push_back(poissonVariableTag);
    
    Expr::ExpressionBuilder* pbuilder  = new PoissonExpression::Builder( poissontags, rhsTag, useRefPoint, refValue, refLocation, use3DLaplacian, *sparams, linSolver);
    Expr::ExpressionBuilder* pbuilder1 = new PoissonExpression::Builder( poissontags, rhsTag, useRefPoint, refValue, refLocation, use3DLaplacian, *sparams, linSolver);
    
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
                               Uintah::SimulationStateP& sharedState,
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
      Uintah::SolverParameters* sparams = linSolver.readParameters( params, "", sharedState );

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
        DORadSolver::Builder* radSolver = new DORadSolver::Builder( intensity, svec, absCoefTag, scatCoef, tempTag, *sparams, linSolver );
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
    varDensMMSParams->get("rho1",rho1);
    varDensMMSParams->get("rho0",rho0);

    const Expr::Tag fStarTag(primVarName + "*", Expr::STATE_NONE);
    const Expr::Tag drhodfStarTag("drhod" + primVarName + "*", Expr::STATE_NONE);
    const Expr::Tag scalarEOSCouplingTag(primVarName + "*_EOS_Coupling", Expr::STATE_NONE);
    
    for( Uintah::ProblemSpecP bcExprParams = wasatchParams->findBlock("BCExpression");
         bcExprParams != 0;
         bcExprParams = bcExprParams->findNextBlock("BCExpression") )
    {
      if( bcExprParams->findBlock("VarDenMMSMomentum") ){
        double bcRho0=1.29985, bcRho1=0.081889;
        Uintah::ProblemSpecP valParams = bcExprParams->findBlock("VarDenMMSMomentum");
        valParams->get("rho0",bcRho0);
        valParams->get("rho1",bcRho1);
        if( rho0!=bcRho0 || rho1!=bcRho1 ){
          std::ostringstream msg;
          msg << "ERROR: the values of rho0 and rho1 should be exactly the same in the \"VariableDensityMMS\" block and the \"VarDen1DMMSMomentum\" BCExpression. In \"VariableDensityMMS\" rho0=" << rho0 << " and rho1=" << rho1 << " while in \"VarDen1DMMSMomentum\" BCExpression rho0=" << bcRho0 << " and rho1=" << bcRho1 << std::endl;
          throw Uintah::InvalidValue( msg.str(), __FILE__, __LINE__ );
        }
      }
      
      else if( bcExprParams->findBlock("VarDenMMSDensity") ){
        double bcRho0=1.29985, bcRho1=0.081889;
        Uintah::ProblemSpecP valParams = bcExprParams->findBlock("VarDenMMSDensity");
        valParams->get("rho0",bcRho0);
        valParams->get("rho1",bcRho1);
        if( rho0!=bcRho0 || rho1!=bcRho1 ){
          std::ostringstream msg;
          msg << "ERROR: the values of rho0 and rho1 should be exactly the same in the \"VariableDensityMMS\" block and the \"VarDen1DMMSDensity\" BCExpression. In \"VariableDensityMMS\" rho0=" << rho0 << " and rho1=" << rho1 << " while in \"VarDen1DMMSDensity\" BCExpression rho0=" << bcRho0 << " and rho1=" << bcRho1 << std::endl;
          throw Uintah::InvalidValue( msg.str(), __FILE__, __LINE__ );
        }
      }
      
      else if( bcExprParams->findBlock("VarDenMMSSolnVar") ){
        double bcRho0=1.29985, bcRho1=0.081889;
        Uintah::ProblemSpecP valParams = bcExprParams->findBlock("VarDenMMSSolnVar");
        valParams->get("rho0",bcRho0);
        valParams->get("rho1",bcRho1);
        if( rho0!=bcRho0 || rho1!=bcRho1 ){
          std::ostringstream msg;
          msg << "ERROR: the values of rho0 and rho1 should be exactly the same in the \"VariableDensityMMS\" block and the \"VarDen1DMMSSolnVar\" BCExpression. In \"VariableDensityMMS\" rho0=" << rho0 << " and rho1=" << rho1 << " while in \"VarDen1DMMSSolnVar\" BCExpression rho0=" << bcRho0 << " and rho1=" << bcRho1 << std::endl;
          throw Uintah::InvalidValue( msg.str(), __FILE__, __LINE__ );
        }
      }
    }
    varDensMMSParams->get("D",D);
    const TagNames& tagNames = TagNames::self();

    const Expr::Tag solnVarRHSTag     = Expr::Tag(solnVarName+"_rhs",Expr::STATE_NONE);
    
    GraphHelper* const slngraphHelper = gc[ADVANCE_SOLUTION];
    slngraphHelper->exprFactory->register_expression( new VarDen1DMMSMixFracSrc<SVolField>::Builder(tagNames.mms_mixfracsrc,tagNames.xsvolcoord, tagNames.time, tagNames.dt, D, rho0, rho1, false));
    slngraphHelper->exprFactory->attach_dependency_to_expression(tagNames.mms_mixfracsrc, solnVarRHSTag);
    

    const Expr::Tag mms_mixfracsrcPlusTag(tagNames.mms_mixfracsrc.name() + "*", Expr::STATE_NONE);
    slngraphHelper->exprFactory->register_expression( new VarDen1DMMSMixFracSrc<SVolField>::Builder(mms_mixfracsrcPlusTag,tagNames.xsvolcoord, tagNames.time, tagNames.dt, D, rho0, rho1, true));
    
    Uintah::ProblemSpecP densityParams = wasatchParams->findBlock("Density");
    const Expr::Tag densityTag   = parse_nametag( densityParams->findBlock("NameTag") );
    const Expr::Tag densStarTag  = tagNames.make_star(densityTag, Expr::STATE_NONE);
    
    // attach Sf_{n+1} to the scalar EOS coupling term
    const Expr::Tag mms_EOSMixFracSrcTag(tagNames.mms_mixfracsrc.name() + "*_EOS", Expr::STATE_NONE);
    slngraphHelper->exprFactory->register_expression( new VarDenEOSCouplingMixFracSrc<SVolField>::Builder(mms_EOSMixFracSrcTag, mms_mixfracsrcPlusTag, densStarTag, drhodfStarTag));
    
    slngraphHelper->exprFactory->attach_dependency_to_expression(mms_EOSMixFracSrcTag, scalarEOSCouplingTag);

    
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
    
    slngraphHelper->exprFactory->register_expression( new VarDen1DMMSContinuitySrc<SVolField>::Builder( tagNames.mms_continuitysrc, rho0, rho1, tagNames.xsvolcoord, tagNames.time, tagNames.dt));
    slngraphHelper->exprFactory->register_expression( new VarDen1DMMSPressureContSrc<SVolField>::Builder( tagNames.mms_pressurecontsrc, tagNames.mms_continuitysrc, densStarTag, fStarTag, drhodfStarTag, tagNames.dt));
    slngraphHelper->exprFactory->attach_dependency_to_expression(tagNames.mms_pressurecontsrc, scalarEOSCouplingTag);
    
    if (computeContinuityResidual)
    {
      slngraphHelper->exprFactory->attach_dependency_to_expression(tagNames.mms_continuitysrc, tagNames.drhodtstar);
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
    
    
    const Expr::Tag fStarTag(primVarName + "*", Expr::STATE_NONE);
    const Expr::Tag drhodfStarTag("drhod" + primVarName + "*", Expr::STATE_NONE);
    const Expr::Tag scalarEOSCouplingTag(primVarName + "*_EOS_Coupling", Expr::STATE_NONE);
    
    const TagNames& tagNames = TagNames::self();
    
    Uintah::ProblemSpecP densityParams = wasatchParams->findBlock("Density");
    Uintah::ProblemSpecP momEqnParams  = wasatchParams->findBlock("MomentumEquations");
    
    const Expr::Tag densityTag = parse_nametag( densityParams->findBlock("NameTag") );
    const Expr::Tag densStarTag = tagNames.make_star( densityTag, Expr::STATE_NONE );
    const Expr::Tag solnVarRHSTag( solnVarName+"_rhs", Expr::STATE_NONE );

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

    // attach the mixture fraction source term, Sf_n to the RHS of the mixture fraction equation
    GraphHelper* const slngraphHelper = gc[ADVANCE_SOLUTION];
    slngraphHelper->exprFactory->register_expression( new VarDenMMSOscillatingMixFracSrc<SVolField>::Builder(tagNames.mms_mixfracsrc, x1Tag, x2Tag, tagNames.time, rho0, rho1, d, w, k, uf, vf, false));
    slngraphHelper->exprFactory->attach_dependency_to_expression(tagNames.mms_mixfracsrc, solnVarRHSTag);

    // We need to compute the EOS coupling term at n+1. This term requires the mixture fraction src at n+1,
    // here we register that term: Sf_{n+1}
    const Expr::Tag mms_mixfracsrcPlusTag(tagNames.mms_mixfracsrc.name() + "*", Expr::STATE_NONE);
    slngraphHelper->exprFactory->register_expression( new VarDenMMSOscillatingMixFracSrc<SVolField>::Builder(mms_mixfracsrcPlusTag, x1Tag, x2Tag, tagNames.time, rho0, rho1, d, w, k, uf, vf, true));
    
    // attach Sf_{n+1} to the scalar EOS coupling term
    const Expr::Tag mms_EOSMixFracSrcTag(tagNames.mms_mixfracsrc.name() + "*_EOS", Expr::STATE_NONE);
    slngraphHelper->exprFactory->register_expression( new VarDenEOSCouplingMixFracSrc<SVolField>::Builder(mms_EOSMixFracSrcTag, mms_mixfracsrcPlusTag, densStarTag, drhodfStarTag));
    
    slngraphHelper->exprFactory->attach_dependency_to_expression(mms_EOSMixFracSrcTag, scalarEOSCouplingTag);
  }
  
  //==================================================================
  
  std::vector<EqnTimestepAdaptorBase*>
  parse_momentum_equations( Uintah::ProblemSpecP wasatchSpec,
                            const TurbulenceParameters turbParams,
                            const bool useAdaptiveDt,
                            const bool doParticles,
                            const bool isConstDensity,
                            const Expr::Tag densityTag,
                            GraphCategories& gc,
                            Uintah::SolverInterface& linSolver, Uintah::SimulationStateP& sharedState )
  {
    typedef std::vector<EqnTimestepAdaptorBase*> EquationAdaptors;
    EquationAdaptors adaptors;

    bool isCompressible = (Wasatch::flow_treatment() == COMPRESSIBLE);
    
    Uintah::ProblemSpecP momentumSpec = wasatchSpec->findBlock("MomentumEquations");
    std::string xvelname, yvelname, zvelname;
    const Uintah::ProblemSpecP doxvel = momentumSpec->get( "X-Velocity", xvelname );
    const Uintah::ProblemSpecP doyvel = momentumSpec->get( "Y-Velocity", yvelname );
    const Uintah::ProblemSpecP dozvel = momentumSpec->get( "Z-Velocity", zvelname );

    std::string xmomname, ymomname, zmomname;
    const Uintah::ProblemSpecP doxmom = momentumSpec->get( "X-Momentum", xmomname );
    const Uintah::ProblemSpecP doymom = momentumSpec->get( "Y-Momentum", ymomname );
    const Uintah::ProblemSpecP dozmom = momentumSpec->get( "Z-Momentum", zmomname );

    if (isCompressible && momentumSpec->findBlock("Pressure")) {
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
        bodyForceParams != 0;
        bodyForceParams=bodyForceParams->findNextBlock("BodyForce") ){
      bodyForceParams->getAttribute("direction", bodyForceDir );
      if (bodyForceDir == "X") xBodyForceTag = parse_nametag( bodyForceParams->findBlock("NameTag") );
      if (bodyForceDir == "Y") yBodyForceTag = parse_nametag( bodyForceParams->findBlock("NameTag") );
      if (bodyForceDir == "Z") zBodyForceTag = parse_nametag( bodyForceParams->findBlock("NameTag") );
    }

    // parse source expression
    std::string srcTermDir;
    Expr::Tag xSrcTermTag, ySrcTermTag, zSrcTermTag;
    for( Uintah::ProblemSpecP srcTermParams=momentumSpec->findBlock("SourceTerm");
        srcTermParams != 0;
        srcTermParams=srcTermParams->findNextBlock("SourceTerm") ){
      srcTermParams->getAttribute("direction", srcTermDir );
      if (srcTermDir == "X") xSrcTermTag = parse_nametag( srcTermParams->findBlock("NameTag") );
      if (srcTermDir == "Y") ySrcTermTag = parse_nametag( srcTermParams->findBlock("NameTag") );
      if (srcTermDir == "Z") zSrcTermTag = parse_nametag( srcTermParams->findBlock("NameTag") );
    }
    
    GraphHelper* const solnGraphHelper = gc[ADVANCE_SOLUTION  ];
    GraphHelper* const icGraphHelper   = gc[INITIALIZATION    ];

    //___________________________________________________________________________‰‰
    // resolve the momentum equation to be solved and create the adaptor for it.
    //
    proc0cout << "------------------------------------------------" << std::endl
              << "Creating momentum equations..." << std::endl;

    if (isCompressible) {
      
      
      const Expr::Tag temperatureTag = parse_nametag(momentumSpec->findBlock("EnergyEquation")->findBlock("Temperature")->findBlock("NameTag"));
      const Expr::Tag mixMWTag = parse_nametag(momentumSpec->findBlock("EnergyEquation")->findBlock("MolecularWeight")->findBlock("NameTag"));
      const double R = 8.314459848;  // gas constant J/(mol K)
      
      Expr::Tag xVelTag, yVelTag, zVelTag;
      Expr::Tag rhoTag(densityTag.name(), Expr::STATE_DYNAMIC);
      if( doxvel && doxmom ) {
        proc0cout << "Setting up X momentum transport equation" << std::endl;
      
        typedef CompressibleMomentumTransportEquation<SpatialOps::XDIR> XMomEq;
        EquationBase* momtranseq = scinew XMomEq( WasatchCore::XDIR,
                                                     xvelname,
                                                     xmomname,
                                                     rhoTag,
                                                     temperatureTag,
                                                     mixMWTag,
                                                     R,
                                                     xBodyForceTag,
                                                     xSrcTermTag,
                                                     gc,
                                                     momentumSpec,
                                                     turbParams);
          
        adaptors.push_back( scinew EqnTimestepAdaptor<SVolField>(momtranseq) );
        
        xVelTag = Expr::Tag(xvelname, Expr::STATE_NONE);
      }
      
      if( doyvel && doymom ){
        proc0cout << "Setting up Y momentum transport equation" << std::endl;

        typedef CompressibleMomentumTransportEquation<SpatialOps::YDIR> YMomEq;
        EquationBase* momtranseq = scinew YMomEq( WasatchCore::YDIR, yvelname,
                                                     ymomname,
                                                     rhoTag,
                                                     temperatureTag,
                                                     mixMWTag,
                                                     R,
                                                     xBodyForceTag,
                                                     xSrcTermTag,
                                                     gc,
                                                     momentumSpec,
                                                     turbParams );
          
        adaptors.push_back( scinew EqnTimestepAdaptor<SVolField>(momtranseq) );

        yVelTag = Expr::Tag(yvelname, Expr::STATE_NONE);
      }
      
      if( dozvel && dozmom ){
        proc0cout << "Setting up Z momentum transport equation" << std::endl;
        typedef CompressibleMomentumTransportEquation<SpatialOps::ZDIR> ZMomEq;
        EquationBase* momtranseq = scinew ZMomEq( WasatchCore::ZDIR, zvelname,
                                                     zmomname,
                                                     rhoTag,
                                                     temperatureTag,
                                                     mixMWTag,
                                                     R,
                                                     xBodyForceTag,
                                                     xSrcTermTag,
                                                     gc,
                                                     momentumSpec,
                                                     turbParams );
          
        adaptors.push_back( scinew EqnTimestepAdaptor<SVolField>(momtranseq) );

        zVelTag = Expr::Tag(zvelname, Expr::STATE_NONE);
      }


      // register continuity equation
      EquationBase* contEq = scinew ContinuityTransportEquation(rhoTag, temperatureTag, mixMWTag, R, gc, xVelTag, yVelTag, zVelTag);
      adaptors.push_back( scinew EqnTimestepAdaptor<SVolField>(contEq) );
      
      // register total internal energy equation
      const Expr::TagList velTags = tag_list(xVelTag, yVelTag, zVelTag);
      const Expr::Tag viscTag = (momentumSpec->findBlock("Viscosity")) ? parse_nametag( momentumSpec->findBlock("Viscosity")->findBlock("NameTag") ) : Expr::Tag();
      std::string rhoETotal;
      Uintah::ProblemSpecP energySpec = momentumSpec->findBlock("EnergyEquation");
      if (!energySpec) {
        std::ostringstream msg;
        msg << "ERROR: When solving a compressible flow problem you must specify an energy equation." << std::endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
      momentumSpec->findBlock("EnergyEquation")->get("SolutionVariable", rhoETotal);
      EquationBase* totalEEq = scinew TotalInternalEnergyTransportEquation(rhoETotal, momentumSpec->findBlock("EnergyEquation"), gc, rhoTag, temperatureTag, TagNames::self().pressure, velTags, viscTag, turbParams);
      adaptors.push_back( scinew EqnTimestepAdaptor<SVolField>(totalEEq) );

    } else {
      
      if( doxvel && doxmom ){
        proc0cout << "Setting up X momentum transport equation" << std::endl;
        typedef LowMachMomentumTransportEquation< XVolField > MomTransEq;
      EquationBase* momtranseq = scinew MomTransEq( WasatchCore::XDIR, xvelname,
                                         xmomname,
                                         densityTag,
                                         isConstDensity,
                                         xBodyForceTag,
                                         xSrcTermTag,
                                         gc,
                                         momentumSpec,
                                         turbParams,
                                         linSolver, sharedState );
        adaptors.push_back( scinew EqnTimestepAdaptor<XVolField>(momtranseq) );
      }
      
      if( doyvel && doymom ){
        proc0cout << "Setting up Y momentum transport equation" << std::endl;
        typedef LowMachMomentumTransportEquation< YVolField > MomTransEq;
      EquationBase* momtranseq = scinew MomTransEq( WasatchCore::YDIR, yvelname,
                                         ymomname,
                                         densityTag,
                                         isConstDensity,
                                         yBodyForceTag,
                                         ySrcTermTag,
                                         gc,
                                         momentumSpec,
                                         turbParams,
                                         linSolver,sharedState );
        adaptors.push_back( scinew EqnTimestepAdaptor<YVolField>(momtranseq) );
      }
      
      if( dozvel && dozmom ){
        proc0cout << "Setting up Z momentum transport equation" << std::endl;
        typedef LowMachMomentumTransportEquation< ZVolField > MomTransEq;
      EquationBase* momtranseq = scinew MomTransEq( WasatchCore::ZDIR, zvelname,
                                         zmomname,
                                         densityTag,
                                         isConstDensity,
                                         zBodyForceTag,
                                         zSrcTermTag,
                                         gc,
                                         momentumSpec,
                                         turbParams,
                                         linSolver,sharedState );
        adaptors.push_back( scinew EqnTimestepAdaptor<ZVolField>(momtranseq) );
      }
    }
    
    //
    // ADD ADAPTIVE TIMESTEPPING
    if( useAdaptiveDt ){
      const Expr::Tag xVelTag = doxvel ? Expr::Tag(xvelname, Expr::STATE_NONE) : Expr::Tag();
      const Expr::Tag yVelTag = doyvel ? Expr::Tag(yvelname, Expr::STATE_NONE) : Expr::Tag();
      const Expr::Tag zVelTag = dozvel ? Expr::Tag(zvelname, Expr::STATE_NONE) : Expr::Tag();
      const Expr::Tag viscTag = (momentumSpec->findBlock("Viscosity")) ? parse_nametag( momentumSpec->findBlock("Viscosity")->findBlock("NameTag") ) : Expr::Tag();

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
      const Expr::ExpressionID stabDtID = solnGraphHelper->exprFactory->register_expression(scinew StableTimestep::Builder( TagNames::self().stableTimestep,
                                                                                                                           densityTag,
                                                                                                                           viscTag,
                                                                                                                           xVelTag,yVelTag,zVelTag, puTag, pvTag, pwTag ), true);
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
  parse_particle_transport_equations( Uintah::ProblemSpecP particleSpec,
                                      Uintah::ProblemSpecP wasatchSpec,
                                      const bool useAdaptiveDt,
                                      GraphCategories& gc)
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
    // resolve the momentum equation to be solved and create the adaptor for it.
    //
    Expr::ExpressionFactory& factory = *(gc[ADVANCE_SOLUTION]->exprFactory);
    proc0cout << "Setting up particle x-momentum equation" << std::endl;
    EquationBase* pueq = scinew ParticleMomentumEquation( puname,
                                                          XDIR,
                                                          pPosTags,
                                                          pSizeTag,
                                                          particleSpec,
                                                          gc );
    adaptors.push_back( scinew EqnTimestepAdaptor<ParticleField>(pueq) );
    
    proc0cout << "Setting up particle y-momentum equation" << std::endl;
    EquationBase* pveq = scinew ParticleMomentumEquation( pvname,
                                                          YDIR,
                                                          pPosTags,
                                                          pSizeTag,
                                                          particleSpec,
                                                          gc );
    adaptors.push_back( scinew EqnTimestepAdaptor<ParticleField>(pveq) );
    
    proc0cout << "Setting up particle z-momentum equation" << std::endl;
    EquationBase* pweq = scinew ParticleMomentumEquation( pwname,
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
    if (!particleSpec->findBlock("ParticleMomentum")->findBlock("DisableTwoWayCoupling"))
    {
      Uintah::ProblemSpecP momentumSpec  = wasatchSpec->findBlock("MomentumEquations");
      if (momentumSpec) {
                
        std::string xmomname, ymomname, zmomname;
        const Uintah::ProblemSpecP doxmom = momentumSpec->get( "X-Momentum", xmomname );
        const Uintah::ProblemSpecP doymom = momentumSpec->get( "Y-Momentum", ymomname );
        const Uintah::ProblemSpecP dozmom = momentumSpec->get( "Z-Momentum", zmomname );
        
        const TagNames tNames = TagNames::self();
        if (doxmom) {
          typedef ParticleGasMomentumSrc<XVolField>::Builder XMomSrcT;
          const Expr::Tag xMomRHSTag (xmomname + "_rhs_partial", Expr::STATE_NONE);
          factory.register_expression( scinew XMomSrcT( tNames.pmomsrcx, tNames.pdragx, pMassTag, pSizeTag, pPosTags ));
          factory.attach_dependency_to_expression(tNames.pmomsrcx, xMomRHSTag);
        }
        
        if (doymom) {
          typedef ParticleGasMomentumSrc<YVolField>::Builder YMomSrcT;
          const Expr::Tag yMomRHSTag (ymomname + "_rhs_partial", Expr::STATE_NONE);
          factory.register_expression( scinew YMomSrcT( tNames.pmomsrcy, tNames.pdragy, pMassTag, pSizeTag, pPosTags ));
          factory.attach_dependency_to_expression(tNames.pmomsrcy, yMomRHSTag);
        }
        
        if (dozmom) {
          typedef ParticleGasMomentumSrc<ZVolField>::Builder ZMomSrcT;
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
      if (!factory.have_entry(TagNames::self().stableTimestep)) {
        
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
        
        const Expr::ExpressionID stabDtID = factory.register_expression(scinew StableTimestep::Builder( TagNames::self().stableTimestep,
                                                                                                                             densityTag,
                                                                                                                             viscTag,
                                                                                                                             xVelTag,yVelTag,zVelTag, puTag, pvTag, pwTag ), true);
        // force this onto the graph.
        gc[ADVANCE_SOLUTION]->rootIDs.insert( stabDtID );
      }
    }

    //
    return adaptors;
  }

  //==================================================================

  std::vector<EqnTimestepAdaptorBase*>
  parse_moment_transport_equations( Uintah::ProblemSpecP params,
                                    Uintah::ProblemSpecP wasatchParams,
                                    const bool isConstDensity,
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
        exprParams != 0;
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
                                                         isConstDensity,
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

  //-----------------------------------------------------------------

  template< typename FieldT >
  void setup_convective_flux_expression( const std::string& dir,
                                         const Expr::Tag& solnVarTag,
                                         Expr::Tag convFluxTag,
                                         const ConvInterpMethods convMethod,
                                         const Expr::Tag& advVelocityTag,
                                         Expr::ExpressionFactory& factory,
                                         FieldTagInfo& info )
  {
    typedef OpTypes<FieldT> Ops;
    typedef typename FaceTypes<FieldT>::XFace XFace;
    typedef typename FaceTypes<FieldT>::YFace YFace;
    typedef typename FaceTypes<FieldT>::ZFace ZFace;

    if( advVelocityTag == Expr::Tag() ){
      std::ostringstream msg;
      msg << "ERROR: no advective velocity set for transport equation '" << solnVarTag.name() << "'" << std::endl;
      throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
    }

    if( convFluxTag == Expr::Tag() ){
      const TagNames& tagNames = TagNames::self();
      convFluxTag = Expr::Tag( solnVarTag.name() + tagNames.convectiveflux + dir, Expr::STATE_NONE );
      // make new Tag for solnVar by adding the appropriate suffix ( "_*" or nothing ). This
      // is because we need the ScalarRHS at time step n+1 for our pressure projection method

      Expr::ExpressionBuilder* builder = NULL;

      const std::string interpMethod = get_conv_interp_method( convMethod );
      if( dir=="X" ){
        proc0cout << "SETTING UP CONVECTIVE FLUX EXPRESSION IN X DIRECTION USING " << interpMethod << std::endl;
        if (Wasatch::flow_treatment() == WasatchCore::COMPRESSIBLE) {
          typedef typename ConvectiveFluxLimiter<
              typename Ops::InterpC2FXLimiter,
              typename Ops::InterpC2FXUpwind,
              typename OperatorTypeBuilder<Interpolant,FieldT,   XFace>::type, // scalar interp type
              typename OperatorTypeBuilder<Interpolant,SVolField,XFace>::type  // velocity interp type
              >::Builder ConvFluxLim;
          builder = scinew ConvFluxLim( convFluxTag, solnVarTag, advVelocityTag, convMethod, info[VOLUME_FRAC] );
        } else {
          typedef typename ConvectiveFluxLimiter<
              typename Ops::InterpC2FXLimiter,
              typename Ops::InterpC2FXUpwind,
              typename OperatorTypeBuilder<Interpolant,FieldT,   XFace>::type, // scalar interp type
              typename OperatorTypeBuilder<Interpolant,XVolField,XFace>::type  // velocity interp type
              >::Builder ConvFluxLim;
        builder = scinew ConvFluxLim( convFluxTag, solnVarTag, advVelocityTag, convMethod, info[VOLUME_FRAC] );
        }
        
      }
      else if( dir=="Y" ){
        proc0cout << "SETTING UP CONVECTIVE FLUX EXPRESSION IN Y DIRECTION USING " << interpMethod << std::endl;
        if (Wasatch::flow_treatment() == WasatchCore::COMPRESSIBLE) {
          typedef typename ConvectiveFluxLimiter<
          typename Ops::InterpC2FYLimiter,
          typename Ops::InterpC2FYUpwind,
          typename OperatorTypeBuilder<Interpolant,FieldT,   YFace>::type, // scalar interp type
          typename OperatorTypeBuilder<Interpolant,SVolField,YFace>::type  // velocity interp type
          >::Builder ConvFluxLim;
          builder = scinew ConvFluxLim( convFluxTag, solnVarTag, advVelocityTag, convMethod, info[VOLUME_FRAC] );
        } else {
          typedef typename ConvectiveFluxLimiter<
          typename Ops::InterpC2FYLimiter,
          typename Ops::InterpC2FYUpwind,
          typename OperatorTypeBuilder<Interpolant,FieldT,   YFace>::type, // scalar interp type
          typename OperatorTypeBuilder<Interpolant,YVolField,YFace>::type  // velocity interp type
          >::Builder ConvFluxLim;
        builder = scinew ConvFluxLim( convFluxTag, solnVarTag, advVelocityTag, convMethod, info[VOLUME_FRAC] );
        }

      }
      else if( dir=="Z") {
        proc0cout << "SETTING UP CONVECTIVE FLUX EXPRESSION IN Z DIRECTION USING " << interpMethod << std::endl;
        if (Wasatch::flow_treatment() == WasatchCore::COMPRESSIBLE) {
          typedef typename ConvectiveFluxLimiter<
          typename Ops::InterpC2FZLimiter,
          typename Ops::InterpC2FZUpwind,
          typename OperatorTypeBuilder<Interpolant,FieldT,   ZFace>::type, // scalar interp type
          typename OperatorTypeBuilder<Interpolant,SVolField,ZFace>::type  // velocity interp type
          >::Builder ConvFluxLim;
          builder = scinew ConvFluxLim( convFluxTag, solnVarTag, advVelocityTag, convMethod, info[VOLUME_FRAC] );
        } else {
          typedef typename ConvectiveFluxLimiter<
          typename Ops::InterpC2FZLimiter,
          typename Ops::InterpC2FZUpwind,
          typename OperatorTypeBuilder<Interpolant,FieldT,   ZFace>::type, // scalar interp type
          typename OperatorTypeBuilder<Interpolant,ZVolField,ZFace>::type  // velocity interp type
          >::Builder ConvFluxLim;
        builder = scinew ConvFluxLim( convFluxTag, solnVarTag, advVelocityTag, convMethod, info[VOLUME_FRAC] );
        }
      }

      if( builder == NULL ){
        std::ostringstream msg;
        msg << "ERROR: Could not build a convective flux expression for '"
            << solnVarTag.name() << "'" << std::endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
      factory.register_expression( builder );
    }

    FieldSelector fs;
    if     ( dir=="X" ) fs = CONVECTIVE_FLUX_X;
    else if( dir=="Y" ) fs = CONVECTIVE_FLUX_Y;
    else if( dir=="Z" ) fs = CONVECTIVE_FLUX_Z;
    else{
      std::ostringstream msg;
      msg << "Invalid direction selection for convective flux expression on " << solnVarTag.name() << std::endl;
      throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
    }
    info[ fs ] = convFluxTag;
  }

  template< typename FieldT >
  void setup_convective_flux_expression( Uintah::ProblemSpecP convFluxParams,
                                         const Expr::Tag& solnVarTag,
                                         Expr::ExpressionFactory& factory,
                                         FieldTagInfo& info )
  {
    Expr::Tag convFluxTag, advVelocityTag, advVelocityCorrectedTag;

    std::string dir, interpMethod;
    convFluxParams->getAttribute("direction",dir);
    convFluxParams->getAttribute("method",interpMethod);

    // get the tag for the advective velocity
    Uintah::ProblemSpecP advVelocityTagParam = convFluxParams->findBlock( "AdvectiveVelocity" );
    if( advVelocityTagParam ){
      advVelocityTag = parse_nametag( advVelocityTagParam->findBlock( "NameTag" ) );
    }

    // see if we have an expression set for the advective flux.
    Uintah::ProblemSpecP nameTagParam = convFluxParams->findBlock("NameTag");
    if( nameTagParam ) convFluxTag = parse_nametag( nameTagParam );

    setup_convective_flux_expression<FieldT>( dir,
                                              solnVarTag, convFluxTag,
                                              get_conv_interp_method(interpMethod),
                                              advVelocityTag,
                                              factory,
                                              info );
  }

  //-----------------------------------------------------------------

  template< typename FluxT >
  Expr::ExpressionBuilder*
  build_diff_flux_expr( Uintah::ProblemSpecP diffFluxParams,
                        const Expr::Tag& diffFluxTag,
                        const Expr::Tag& primVarTag,
                        const Expr::Tag& densityTag,
                        const Expr::Tag& turbDiffTag )
  {
    typedef typename DiffusiveFlux<FluxT>::Builder Flux;
    bool isDiffusiveVelocity = false;
    if( diffFluxParams->findAttribute("isvelocity") ){
      diffFluxParams->getAttribute("isvelocity",isDiffusiveVelocity);
    }
    
    if( diffFluxParams->findAttribute("coefficient") ){
      double coef;
      diffFluxParams->getAttribute("coefficient",coef);
      if (isDiffusiveVelocity) return scinew typename DiffusiveVelocity<FluxT>::Builder( diffFluxTag, primVarTag, coef, turbDiffTag );
      return scinew Flux( diffFluxTag, primVarTag, coef, turbDiffTag, densityTag );
    }
    else if( diffFluxParams->findBlock("DiffusionCoefficient") ){
      /**
       *  \todo need to ensure that the type that the user gives
       *        for the diffusion coefficient field matches the
       *        type implied here.  Alternatively, we don't let
       *        the user specify the type for the diffusion
       *        coefficient.  But there is the matter of what
       *        independent variable is used when calculating the
       *        coefficient...  Arrrgghh.
       */
      const Expr::Tag coef = parse_nametag( diffFluxParams->findBlock("DiffusionCoefficient")->findBlock("NameTag") );
      if (isDiffusiveVelocity) return scinew typename DiffusiveVelocity<FluxT>::Builder( diffFluxTag, primVarTag, coef, turbDiffTag );
      return scinew Flux( diffFluxTag, primVarTag, coef, turbDiffTag, densityTag );
    }
    else {
      std::ostringstream msg;
      msg << "You mus provide a coefficient for your diffusive flux expressions. Please revise your input file." << std::endl;
      throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
    }
    return NULL;
  }

  template< typename FieldT>
  void setup_diffusive_flux_expression( Uintah::ProblemSpecP diffFluxParams,
                                        const Expr::Tag densityTag,
                                        const Expr::Tag primVarTag,
                                        const Expr::Tag turbDiffTag,  
                                        const std::string suffix,
                                        Expr::ExpressionFactory& factory,
                                        FieldTagInfo& info )
  {
    typedef typename FaceTypes<FieldT>::XFace XFaceT;
    typedef typename FaceTypes<FieldT>::YFace YFaceT;
    typedef typename FaceTypes<FieldT>::ZFace ZFaceT;
    const std::string& primVarName = primVarTag.name();
    Expr::Tag diffFluxTag;  // we will populate this.

    std::string direction;
    diffFluxParams->getAttribute("direction",direction);

    const bool singleDirection = (direction == "X" || direction == "Y" || direction == "Z");
    // see if we have an expression set for the diffusive flux.
    Uintah::ProblemSpecP nameTagParam = diffFluxParams->findBlock("NameTag");
    if( nameTagParam ){
      if( singleDirection ) diffFluxTag = parse_nametag( nameTagParam );
      else{
        std::ostringstream msg;
        msg << "You cannot build a diffusive flux expression with a specified nametag for '" << primVarName << "' in multiple directions" << std::endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
      FieldSelector fs;
      if     ( direction == "X" ) fs=DIFFUSIVE_FLUX_X;
      else if( direction == "Y" ) fs=DIFFUSIVE_FLUX_Y;
      else if( direction == "Z" ) fs=DIFFUSIVE_FLUX_Z;
      else{
        std::ostringstream msg;
        msg << "Invalid direction selection for diffusive flux expression" << std::endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
      
      info[ fs ] = diffFluxTag;
    }
    else{ // build an expression for the diffusive flux.

      for( std::string::iterator it = direction.begin(); it != direction.end(); ++it ){
        std::string dir(1,*it);
        const TagNames& tagNames = TagNames::self();
        diffFluxTag = Expr::Tag( primVarName + suffix + tagNames.diffusiveflux + dir, Expr::STATE_NONE );
        // make new Tags for density and primVar by adding the appropriate suffix ( "_*" or nothing ). This
        // is because we need the ScalarRHS at time step n+1 for our pressure projection method
        const Expr::Tag densityCorrectedTag = Expr::Tag(densityTag.name() + suffix, Expr::STATE_NONE);
        const Expr::Tag primVarCorrectedTag = Expr::Tag(primVarTag.name() + suffix, Expr::STATE_NONE);
        
        Expr::ExpressionBuilder* builder = NULL;
        if     ( dir=="X" ) builder = build_diff_flux_expr<XFaceT>(diffFluxParams,diffFluxTag,primVarCorrectedTag,densityCorrectedTag,turbDiffTag);
        else if( dir=="Y" ) builder = build_diff_flux_expr<YFaceT>(diffFluxParams,diffFluxTag,primVarCorrectedTag,densityCorrectedTag,turbDiffTag);
        else if( dir=="Z" ) builder = build_diff_flux_expr<ZFaceT>(diffFluxParams,diffFluxTag,primVarCorrectedTag,densityCorrectedTag,turbDiffTag);
        
        if( builder == NULL ){
          std::ostringstream msg;
          msg << "Could not build a diffusive flux expression for '" << primVarName << "'" << std::endl;
          throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
        }
        factory.register_expression( builder );
        
        FieldSelector fs;
        if     ( dir=="X" ) fs=DIFFUSIVE_FLUX_X;
        else if( dir=="Y" ) fs=DIFFUSIVE_FLUX_Y;
        else if( dir=="Z" ) fs=DIFFUSIVE_FLUX_Z;
        else{
          std::ostringstream msg;
          msg << "Invalid direction selection for diffusive flux expression" << std::endl;
          throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
        }
        
        info[ fs ] = diffFluxTag;
      }
    }
  }

  //------------------------------------------------------------------

  template< typename VelT >
  Expr::ExpressionBuilder*
  build_diff_vel_expr( Uintah::ProblemSpecP diffVelParams,
                       const Expr::Tag& diffVelTag,
                       const Expr::Tag& primVarTag,
                       const Expr::Tag& turbDiffTag )
  {
    typedef typename DiffusiveVelocity<VelT>::Builder Velocity;
    
    if( diffVelParams->findAttribute("coefficient") ){
      double coef;
      diffVelParams->getAttribute("coefficient",coef);
      return scinew Velocity( diffVelTag, primVarTag, coef, turbDiffTag );
    }
    else if( diffVelParams->findBlock("DiffusionCoefficient") ){
      /**
       *  \todo need to ensure that the type that the user gives
       *        for the diffusion coefficient field matches the
       *        type implied here.  Alternatively, we don't let
       *        the user specify the type for the diffusion
       *        coefficient.  But there is the matter of what
       *        independent variable is used when calculating the
       *        coefficient...  Arrrgghh.
       */
      const Expr::Tag coef = parse_nametag( diffVelParams->findBlock("DiffusionCoefficient")->findBlock("NameTag") );
      return scinew Velocity( diffVelTag, primVarTag, coef, turbDiffTag );
    }
    return NULL;
  }

  template< typename FieldT>
  void setup_diffusive_velocity_expression( Uintah::ProblemSpecP diffVelParams,
                                            const Expr::Tag primVarTag,
                                            const Expr::Tag turbDiffTag,  
                                            Expr::ExpressionFactory& factory,
                                            FieldTagInfo& info )
  {
    typedef typename FaceTypes<FieldT>::XFace XFaceT;
    typedef typename FaceTypes<FieldT>::YFace YFaceT;
    typedef typename FaceTypes<FieldT>::ZFace ZFaceT;

    const std::string& primVarName = primVarTag.name();
    Expr::Tag diffVelTag;  // we will populate this.

    std::string direction;
    diffVelParams->getAttribute("direction",direction);

    const bool singleDirection = (direction == "X" || direction == "Y" || direction == "Z");
    // see if we have an expression set for the diffusive velocity.
    Uintah::ProblemSpecP nameTagParam = diffVelParams->findBlock("NameTag");
    if( nameTagParam ){
      if( singleDirection ) diffVelTag = parse_nametag( nameTagParam );
      else {
        std::ostringstream msg;
        msg << "You cannot build a diffusive velocity expression with a specified nametag for '" << primVarName << "' in multiple directions" << std::endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
      
      FieldSelector fs;
      if     ( direction == "X" ) fs=DIFFUSIVE_FLUX_X;
      else if( direction == "Y" ) fs=DIFFUSIVE_FLUX_Y;
      else if( direction == "Z" ) fs=DIFFUSIVE_FLUX_Z;
      else{
        std::ostringstream msg;
        msg << "Invalid direction selection for diffusive velocity expression" << std::endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
      info[ fs ] = diffVelTag;

    }
    else { // build an expression for the diffusive velocity.

      for( std::string::iterator it = direction.begin(); it != direction.end(); ++it ){
        std::string dir(1,*it);
        diffVelTag = Expr::Tag( primVarName+"_diffVelocity_"+dir, Expr::STATE_NONE );
        
        Expr::ExpressionBuilder* builder = NULL;
        if     ( dir=="X" )  builder = build_diff_vel_expr<XFaceT>(diffVelParams,diffVelTag,primVarTag,turbDiffTag);
        else if( dir=="Y" )  builder = build_diff_vel_expr<YFaceT>(diffVelParams,diffVelTag,primVarTag,turbDiffTag);
        else if( dir=="Z" )  builder = build_diff_vel_expr<ZFaceT>(diffVelParams,diffVelTag,primVarTag,turbDiffTag);
        
        if( builder == NULL ){
          std::ostringstream msg;
          msg << "Could not build a diffusive velocity expression for '"
          << primVarName << "'" << std::endl;
          throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
        }
        factory.register_expression( builder );
        
        FieldSelector fs;
        if     ( dir=="X" ) fs=DIFFUSIVE_FLUX_X;
        else if( dir=="Y" ) fs=DIFFUSIVE_FLUX_Y;
        else if( dir=="Z" ) fs=DIFFUSIVE_FLUX_Z;
        else{
          std::ostringstream msg;
          msg << "Invalid direction selection for diffusive velocity expression" << std::endl;
          throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
        }
        info[ fs ] = diffVelTag;
      }
    }
  }

  //------------------------------------------------------------------

  //==================================================================
  // explicit template instantiation
#define INSTANTIATE_DIFFUSION( FIELDT )                         \
                                                                \
    template void setup_diffusive_flux_expression<FIELDT>(      \
       Uintah::ProblemSpecP diffFluxParams,                     \
       const Expr::Tag densityTag,                              \
       const Expr::Tag primVarTag,                              \
       const Expr::Tag turbDiffTag,                             \
       const std::string suffix,                                \
       Expr::ExpressionFactory& factory,                        \
       FieldTagInfo& info );                                    \
                                                                \
    template void setup_diffusive_velocity_expression<FIELDT>(  \
       Uintah::ProblemSpecP diffVelParams,                      \
       const Expr::Tag primVarTag,                              \
       const Expr::Tag turbDiffTag,                             \
       Expr::ExpressionFactory& factory,                        \
       FieldTagInfo& info );

#define INSTANTIATE_CONVECTION( FIELDT )                        \
    template void setup_convective_flux_expression<FIELDT>(     \
        const std::string& dir,                                 \
        const Expr::Tag& solnVarTag,                            \
        Expr::Tag convFluxTag,                                  \
        const ConvInterpMethods convMethod,                     \
        const Expr::Tag& advVelocityTag,                        \
        Expr::ExpressionFactory& factory,                       \
        FieldTagInfo& info );                                   \
                                                                \
    template void setup_convective_flux_expression<FIELDT>(     \
        Uintah::ProblemSpecP convFluxParams,                    \
        const Expr::Tag& solnVarName,                           \
        Expr::ExpressionFactory& factory,                       \
        FieldTagInfo& info );

  // diffusive fluxes only for scalars.
  INSTANTIATE_DIFFUSION ( SVolField )

  // convective fluxes are supported for momentum as well.
  INSTANTIATE_CONVECTION( SVolField )

  //-----------------------------------------------------------------

} // namespace WasatchCore
