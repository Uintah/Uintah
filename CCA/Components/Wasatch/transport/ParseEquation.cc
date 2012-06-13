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

//-- Wasatch Includes --//
#include "ParseEquation.h"
#include <CCA/Components/Wasatch/TimeStepper.h>
#include <CCA/Components/Wasatch/StringNames.h>
#include <CCA/Components/Wasatch/ParseTools.h>
#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>

//-- Add headers for individual transport equations here --//
#include "TransportEquation.h"
#include "ScalarTransportEquation.h"
#include "ScalabilityTestTransportEquation.h"
#include "MomentumTransportEquation.h"
#include "MomentTransportEquation.h"
#include <CCA/Components/Wasatch/Expressions/PBE/QMOM.h>
#include <CCA/Components/Wasatch/Expressions/PBE/MultiEnvMixingModel.h>

//-- includes for the expressions built here --//
#include <CCA/Components/Wasatch/Expressions/ConvectiveFlux.h>
#include <CCA/Components/Wasatch/Expressions/PoissonExpression.h>
#include <CCA/Components/Wasatch/ConvectiveInterpolationMethods.h>
#include <CCA/Components/Wasatch/Expressions/DiffusiveFlux.h>
#include <CCA/Components/Wasatch/Expressions/DiffusiveVelocity.h>
#include <CCA/Components/Wasatch/Expressions/Pressure.h>

//-- Uintah includes --//
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Parallel/Parallel.h>

//-- Expression Library includes --//
#include <expression/ExpressionFactory.h>

#include <iostream>


namespace Wasatch{


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
    EqnTimestepAdaptor( Wasatch::TransportEquation* eqn ) : EqnTimestepAdaptorBase(eqn) {}
    void hook( TimeStepper& ts ) const
    {
      ts.add_equation<FieldT>( eqn_->solution_variable_name(),
                               eqn_->get_rhs_id() );
    }
  };

  //==================================================================

  EqnTimestepAdaptorBase::EqnTimestepAdaptorBase( Wasatch::TransportEquation* eqn )
    : eqn_(eqn)
  {}

  //------------------------------------------------------------------

  EqnTimestepAdaptorBase::~EqnTimestepAdaptorBase()
  {
    delete eqn_;
  }

  //==================================================================

  EqnTimestepAdaptorBase* parse_equation( Uintah::ProblemSpecP params,
                                          const Expr::Tag densityTag,
                                          const bool isConstDensity,
                                          GraphCategories& gc )
  {
    EqnTimestepAdaptorBase* adaptor = NULL;
    Wasatch::TransportEquation* transeqn = NULL;

    std::string eqnLabel, solnVariable;

    params->getAttribute( "equation", eqnLabel );
    params->get( "SolutionVariable", solnVariable );

    GraphHelper* const solnGraphHelper = gc[ADVANCE_SOLUTION];
    GraphHelper* const icGraphHelper   = gc[INITIALIZATION  ];

    //___________________________________________________________________________
    // resolve the transport equation to be solved and create the adaptor for it.
    //
    proc0cout << "------------------------------------------------" << std::endl;
    proc0cout << "Creating transport equation for '" << eqnLabel << "'" << std::endl;
    Expr::ExpressionID rhsID;

    if( eqnLabel == "generic" ){
       // find out if this corresponds to a staggered or non-staggered field
      std::string staggeredDirection;
      Uintah::ProblemSpecP scalarStaggeredParams = params->get( "StaggeredDirection", staggeredDirection );

      if ( scalarStaggeredParams ) {
        // if staggered, then determine the staggering direction
        proc0cout << "Detected staggered scalar '" << eqnLabel << "'" << std::endl;

        // make proper calls based on the direction
        if ( staggeredDirection=="X" ) {
          proc0cout << "Setting up staggered scalar transport equation in direction: '" << staggeredDirection << "'" << std::endl;
          typedef ScalarTransportEquation< XVolField > ScalarTransEqn;
          rhsID = ScalarTransEqn::get_rhs_expr_id( densityTag, isConstDensity, *solnGraphHelper->exprFactory, params );
          transeqn = scinew ScalarTransEqn( ScalarTransEqn::get_solnvar_name( params ),
                                            params,
                                            densityTag,
                                            isConstDensity,
                                            rhsID );
          adaptor = scinew EqnTimestepAdaptor< XVolField >( transeqn );

        } else if ( staggeredDirection=="Y" ) {
          proc0cout << "Setting up staggered scalar transport equation in direction: '" << staggeredDirection << "'" << std::endl;
          typedef ScalarTransportEquation< YVolField > ScalarTransEqn;
          rhsID = ScalarTransEqn::get_rhs_expr_id( densityTag, isConstDensity, *solnGraphHelper->exprFactory, params );
          transeqn = scinew ScalarTransEqn( ScalarTransEqn::get_solnvar_name( params ),
                                           params,
                                           densityTag,
                                           isConstDensity,
                                           rhsID );
          adaptor = scinew EqnTimestepAdaptor< YVolField >( transeqn );

        } else if (staggeredDirection=="Z") {
          proc0cout << "Setting up staggered scalar transport equation in direction: '" << staggeredDirection << "'" << std::endl;
          typedef ScalarTransportEquation< ZVolField > ScalarTransEqn;
          rhsID = ScalarTransEqn::get_rhs_expr_id( densityTag, isConstDensity, *solnGraphHelper->exprFactory, params );
          transeqn = scinew ScalarTransEqn( ScalarTransEqn::get_solnvar_name( params ),
                                           params,
                                           densityTag,
                                           isConstDensity,
                                           rhsID );
          adaptor = scinew EqnTimestepAdaptor< ZVolField >( transeqn );

        } else {
          std::ostringstream msg;
          msg << "ERROR: No direction is specified for staggered field '" << eqnLabel << "'. Please revise your input file." << std::endl;
          throw Uintah::InvalidValue( msg.str(), __FILE__, __LINE__ );
        }

      } else if ( !scalarStaggeredParams ) {
        // in this case, the scalar field is not staggered
        proc0cout << "Detected non-staggered scalar '" << eqnLabel << "'" << std::endl;
        typedef ScalarTransportEquation< SVolField > ScalarTransEqn;
        rhsID = ScalarTransEqn::get_rhs_expr_id( densityTag, isConstDensity, *solnGraphHelper->exprFactory, params );
        transeqn = scinew ScalarTransEqn( ScalarTransEqn::get_solnvar_name( params ),
                                         params,
                                         densityTag,
                                         isConstDensity,
                                         rhsID );
        adaptor = scinew EqnTimestepAdaptor< SVolField >( transeqn );
      }
    } else {
      std::ostringstream msg;
      msg << "ERROR: No transport equation was specified '" << eqnLabel << "'. Please revise your input file" << std::endl;
      throw Uintah::InvalidValue( msg.str(), __FILE__, __LINE__ );
    }
    // insert the rhsID into the rootIDs of the solution graph helper
    solnGraphHelper->rootIDs.insert(rhsID);
    //_____________________________________________________
    // set up initial conditions on this transport equation
    try{
      proc0cout << "Setting initial conditions for transport equation '" << eqnLabel << "'" << std::endl;
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

    std::string stagLoc;
    Uintah::ProblemSpecP stagLocParams = params->get( "StaggeredDirection", stagLoc );

    int nEqs = 1;
    params->get( "NumberOfEquations", nEqs );

    GraphHelper* const solnGraphHelper = gc[ADVANCE_SOLUTION];
    GraphHelper* const icGraphHelper   = gc[INITIALIZATION  ];


    for( int iEq=0; iEq<nEqs; iEq++ ){

      Wasatch::TransportEquation* scaltesteqn = NULL;

      std::stringstream ss;
      ss << iEq;
      std::string thisPhiName = basePhiName + ss.str();
      // set initial condition and register it
      Expr::Tag icTag( thisPhiName, Expr::STATE_N );
      Expr::ExpressionID rhsID;

      if( stagLocParams ){

        // X-Staggered scalar
        if( stagLoc=="X" ){
          Expr::Tag indepVarTag( "XXVOL", Expr::STATE_NONE );
          typedef Expr::SinFunction<XVolField>::Builder Builder;
          icGraphHelper->exprFactory->register_expression( scinew Builder( icTag, indepVarTag, 1.0, 1, 0.0) );

          // create the transport equation with all-to-all source term
          typedef ScalabilityTestTransportEquation< XVolField > ScalTestEqn;
          rhsID = ScalTestEqn::get_rhs_expr_id( thisPhiName,
                                                *solnGraphHelper->exprFactory,
                                                params );
          scaltesteqn = scinew ScalTestEqn( thisPhiName, rhsID );
          adaptors.push_back( scinew EqnTimestepAdaptor< XVolField >( scaltesteqn ) );
        }
        else if( stagLoc=="Y" ){
          Expr::Tag indepVarTag( "XYVOL", Expr::STATE_NONE );
          typedef Expr::SinFunction<YVolField>::Builder Builder;
          icGraphHelper->exprFactory->register_expression( scinew Builder( icTag, indepVarTag, 1.0, 1, 0.0) );

          // create the transport equation with all-to-all source term
          typedef ScalabilityTestTransportEquation< YVolField > ScalTestEqn;
          rhsID = ScalTestEqn::get_rhs_expr_id( thisPhiName,
                                                *solnGraphHelper->exprFactory,
                                                params );
          scaltesteqn = scinew ScalTestEqn( thisPhiName, rhsID );
          adaptors.push_back( scinew EqnTimestepAdaptor< YVolField >( scaltesteqn ) );
        }
        else if(stagLoc=="Z" ){
          Expr::Tag indepVarTag( "ZZVOL", Expr::STATE_NONE );
          typedef Expr::SinFunction<ZVolField>::Builder Builder;
          icGraphHelper->exprFactory->register_expression( scinew Builder( icTag, indepVarTag, 1.0, 1, 0.0) );

          // create the transport equation with all-to-all source term
          typedef ScalabilityTestTransportEquation< ZVolField > ScalTestEqn;
          rhsID = ScalTestEqn::get_rhs_expr_id( thisPhiName,
                                                *solnGraphHelper->exprFactory,
                                                params );
          scaltesteqn = scinew ScalTestEqn( thisPhiName, rhsID );
          adaptors.push_back( scinew EqnTimestepAdaptor< ZVolField >( scaltesteqn ) );
        }

      }
      else{
        Expr::Tag indepVarTag( "XSVOL", Expr::STATE_NONE );
        typedef Expr::SinFunction<SVolField>::Builder Builder;
        icGraphHelper->exprFactory->register_expression( scinew Builder( icTag, indepVarTag, 1.0, 1, 0.0) );

        // create the transport equation with all-to-all source term
        typedef ScalabilityTestTransportEquation< SVolField > ScalTestEqn;
        rhsID = ScalTestEqn::get_rhs_expr_id( thisPhiName,
                                              *solnGraphHelper->exprFactory,
                                              params );
        scaltesteqn = scinew ScalTestEqn( thisPhiName, rhsID );
        adaptors.push_back( scinew EqnTimestepAdaptor< SVolField >( scaltesteqn ) );
      }

      solnGraphHelper->rootIDs.insert(rhsID);

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
      proc0cout << "------------------------------------------------" << std::endl;
    }
    return adaptors;
  }

  //==================================================================
  
  void parse_poisson_equation( Uintah::ProblemSpecP poissonEqParams,
                              GraphCategories& gc,
                              Uintah::SolverInterface& linSolver) {
    std::string slnVariableName;
    poissonEqParams->get("SolutionVariable", slnVariableName);
    Expr::TagList poissontags;
    const Expr::Tag poissonVariableTag(slnVariableName, Expr::STATE_N);    
    poissontags.push_back( poissonVariableTag );
    poissontags.push_back( Expr::Tag( poissonVariableTag.name() + "_rhs_poisson_expr", pressure_tag().context() ) );
    const Expr::Tag rhsTag = parse_nametag( poissonEqParams->findBlock("PoissonRHS")->findBlock("NameTag"));    
    bool useRefPoint = false;
    double refValue = 0.0;
    SCIRun::IntVector refLocation(0,0,0);
    
    if (poissonEqParams->findBlock("ReferenceValue")) {
      useRefPoint = true;
      Uintah::ProblemSpecP refPhiParams = poissonEqParams->findBlock("ReferenceValue");      
      refPhiParams->getAttribute("value", refValue);
      refPhiParams->get("ReferenceCell", refLocation);
    }
    
    bool use3DLaplacian = true;
    poissonEqParams->getWithDefault("Use3DLaplacian",use3DLaplacian, true);
        
    Uintah::SolverParameters* sparams = linSolver.readParameters( poissonEqParams, "" );
    sparams->setSolveOnExtraCells( false );
    sparams->setUseStencil4( true );
    sparams->setOutputFileName( "WASATCH" );
    
    PoissonExpression::poissonTagList.push_back(poissonVariableTag);
    
    const Expr::ExpressionBuilder* const pbuilder  = new PoissonExpression::Builder( poissontags, rhsTag,useRefPoint, refValue, refLocation, use3DLaplacian,*sparams, linSolver);    
    const Expr::ExpressionBuilder* const pbuilder1 = new PoissonExpression::Builder( poissontags, rhsTag,useRefPoint, refValue, refLocation, use3DLaplacian,*sparams, linSolver);            
    
    GraphHelper* const icgraphHelper = gc[INITIALIZATION];        
    GraphHelper* const slngraphHelper = gc[ADVANCE_SOLUTION];    

    slngraphHelper->exprFactory->register_expression( pbuilder1 );            
    icgraphHelper->exprFactory->register_expression( pbuilder );    

    slngraphHelper->rootIDs.insert( slngraphHelper->exprFactory->get_id(poissonVariableTag) );
    icgraphHelper->rootIDs.insert( icgraphHelper->exprFactory->get_id(poissonVariableTag) );    
  }

  //==================================================================
  
  std::vector<EqnTimestepAdaptorBase*> parse_momentum_equations( Uintah::ProblemSpecP params,
                                                                 TurbulenceParameters turbParams,
                                                                 const Expr::Tag densityTag,
                                                                 GraphCategories& gc,
                                                                 Uintah::SolverInterface& linSolver )
  {
    typedef std::vector<EqnTimestepAdaptorBase*> EquationAdaptors;
    EquationAdaptors adaptors;
    EqnTimestepAdaptorBase* adaptor = NULL;
    Wasatch::TransportEquation* momtranseq = NULL;

    std::string xvelname, yvelname, zvelname;
    const Uintah::ProblemSpecP doxvel = params->get( "X-Velocity", xvelname );
    const Uintah::ProblemSpecP doyvel = params->get( "Y-Velocity", yvelname );
    const Uintah::ProblemSpecP dozvel = params->get( "Z-Velocity", zvelname );

    std::string xmomname, ymomname, zmomname;
    const Uintah::ProblemSpecP doxmom = params->get( "X-Momentum", xmomname );
    const Uintah::ProblemSpecP doymom = params->get( "Y-Momentum", ymomname );
    const Uintah::ProblemSpecP dozmom = params->get( "Z-Momentum", zmomname );

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
    for( Uintah::ProblemSpecP bodyForceParams=params->findBlock("BodyForce");
        bodyForceParams != 0;
        bodyForceParams=bodyForceParams->findNextBlock("BodyForce") ){
      bodyForceParams->get("Direction", bodyForceDir );
      if (bodyForceDir == "X") xBodyForceTag = parse_nametag( bodyForceParams->findBlock("NameTag") );
      if (bodyForceDir == "Y") yBodyForceTag = parse_nametag( bodyForceParams->findBlock("NameTag") );
      if (bodyForceDir == "Z") zBodyForceTag = parse_nametag( bodyForceParams->findBlock("NameTag") );
    }

    GraphHelper* const solnGraphHelper = gc[ADVANCE_SOLUTION];
    GraphHelper* const icGraphHelper   = gc[INITIALIZATION  ];

    //___________________________________________________________________________
    // resolve the momentum equation to be solved and create the adaptor for it.
    //
    proc0cout << "------------------------------------------------" << std::endl;
    proc0cout << "Creating momentum equations..." << std::endl;

    if( doxvel && doxmom ){
      proc0cout << "Setting up X momentum transport equation" << std::endl;
      typedef MomentumTransportEquation< XVolField > MomTransEq;
      const Expr::ExpressionID rhsID = MomTransEq::get_mom_rhs_id( *solnGraphHelper->exprFactory, xvelname, xmomname, params, linSolver );
      momtranseq = scinew MomTransEq( xvelname,
                                      xmomname,
                                      densityTag,
                                      xBodyForceTag,
                                      *solnGraphHelper->exprFactory,
                                      params,
                                      turbParams,
                                      rhsID,
                                      linSolver );
      solnGraphHelper->rootIDs.insert(rhsID);

      adaptor = scinew EqnTimestepAdaptor< XVolField >( momtranseq );
      adaptors.push_back(adaptor);
    }

    if( doyvel && doymom ){
      proc0cout << "Setting up Y momentum transport equation" << std::endl;
      typedef MomentumTransportEquation< YVolField > MomTransEq;
      const Expr::ExpressionID rhsID = MomTransEq::get_mom_rhs_id( *solnGraphHelper->exprFactory, yvelname, ymomname, params, linSolver );
      momtranseq = scinew MomTransEq( yvelname,
                                     ymomname,
                                     densityTag,
                                     yBodyForceTag,
                                     *solnGraphHelper->exprFactory,
                                     params,
                                     turbParams,
                                     rhsID,
                                     linSolver );
      solnGraphHelper->rootIDs.insert(rhsID);

      adaptor = scinew EqnTimestepAdaptor< YVolField >( momtranseq );
      adaptors.push_back(adaptor);
    }

    if( dozvel && dozmom ){
      proc0cout << "Setting up Z momentum transport equation" << std::endl;
      typedef MomentumTransportEquation< ZVolField > MomTransEq;
      const Expr::ExpressionID rhsID = MomTransEq::get_mom_rhs_id( *solnGraphHelper->exprFactory, zvelname, zmomname, params, linSolver );
      momtranseq = scinew MomTransEq( zvelname,
                                     zmomname,
                                     densityTag,
                                     zBodyForceTag,
                                     *solnGraphHelper->exprFactory,
                                     params,
                                     turbParams,
                                     rhsID,
                                     linSolver );
      solnGraphHelper->rootIDs.insert(rhsID);

      adaptor = scinew EqnTimestepAdaptor< ZVolField >( momtranseq );
      adaptors.push_back(adaptor);
    }

    //
    // loop over the local adaptors and set the initial and boundary conditions on each equation attached to that adaptor
    for( EquationAdaptors::const_iterator ia=adaptors.begin(); ia!=adaptors.end(); ++ia ){
      EqnTimestepAdaptorBase* const adaptor = *ia;
      Wasatch::TransportEquation* momtranseq = adaptor->equation();
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
        << "ERORR while setting initial conditions on momentum equation"
        << momtranseq->solution_variable_name()
        << std::endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
      proc0cout << "------------------------------------------------" << std::endl;
    }

    //_____________________________________________________
    // set up initial conditions on the pressure
    try{
      proc0cout << "Setting initial conditions for pressure: "
      << pressure_tag().name()
      << std::endl;
      icGraphHelper->rootIDs.insert( (*icGraphHelper->exprFactory).get_id( Expr::Tag(pressure_tag().name(), Expr::STATE_N) ) );
    }
    catch( std::runtime_error& e ){
      std::ostringstream msg;
      msg << e.what()
      << std::endl
      << "ERORR while setting initial conditions on pressure. When solving for the momentum equations, you must provide an initial condition for the pressure from the input file."
      << pressure_tag().name()
      << std::endl;
      throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
    }
    proc0cout << "------------------------------------------------" << std::endl;
    //
    return adaptors;
  }

  //==================================================================

  template<typename FieldT>
  void preprocess_moment_transport_qmom( Uintah::ProblemSpecP momentEqsParams,
                                         Expr::ExpressionFactory& factory,
                                         Expr::TagList& transportedMomentTags,
                                         Expr::TagList& abscissaeTags,
                                         Expr::TagList& weightsTags,
                                         Expr::TagList& multiEnvWeightsTags)
  {
    // check for supersaturation
    Expr::Tag superSaturationTag = Expr::Tag();
    Uintah::ProblemSpecP superSaturationParams = momentEqsParams->findBlock("SupersaturationExpression");
    if (superSaturationParams) {
      superSaturationTag = parse_nametag( superSaturationParams->findBlock("NameTag") );
    }
    //
    std::string populationName;
    momentEqsParams->get( "PopulationName", populationName );
    int nEnv = 1;
    momentEqsParams->get( "NumberOfEnvironments", nEnv );
    Expr::TagList weightsAndAbscissaeTags;
    //
    // fill in the weights and abscissae tags
    //
    std::stringstream envID;
    for (int i=0; i<nEnv; i++) {
      envID.str(std::string());
      envID << i;
      weightsAndAbscissaeTags.push_back(Expr::Tag("w_" + populationName + "_" + envID.str(), Expr::STATE_NONE));
      weightsAndAbscissaeTags.push_back(Expr::Tag("a_" + populationName + "_" + envID.str(), Expr::STATE_NONE));
      weightsTags.push_back(Expr::Tag("w_" + populationName + "_" + envID.str(), Expr::STATE_NONE));
      abscissaeTags.push_back(Expr::Tag("a_" + populationName + "_"+ envID.str(), Expr::STATE_NONE));
    }
    //
    // construct the transported moments taglist. this will be used to register the
    // qmom expression
    //
    const int nEqs = 2*nEnv;
    std::stringstream strMomID;
    for (int iEq=0; iEq<nEqs; iEq++) {
      strMomID.str(std::string());
      const double momentOrder = (double) iEq;
      strMomID << momentOrder;
      const std::string thisPhiName = "m_" + populationName + "_" + strMomID.str();
      transportedMomentTags.push_back(Expr::Tag(thisPhiName, Expr::STATE_N));
    }
    //
    // register the qmom expression
    //
    factory.register_expression( scinew typename QMOM<FieldT>::Builder(weightsAndAbscissaeTags,transportedMomentTags,superSaturationTag) );
    
    //register the multi environment mixign model calculation
    //register this here since only need one expr for the weights
    if (momentEqsParams->findBlock("MultiEnvMixingModel") ) {
      std::stringstream wID;
      const int numEnv = 3;
      //create the expression for weights and derivatives if block found  
      for (int i=0; i<numEnv; i++) {
        wID.str(std::string());
        wID << i;
        multiEnvWeightsTags.push_back(Expr::Tag("w_MultiEnv_" + wID.str(), Expr::STATE_NONE) );
        multiEnvWeightsTags.push_back(Expr::Tag("dwdt_MultiEnv_" + wID.str(), Expr::STATE_NONE) );
      }
      
      Uintah::ProblemSpecP multiEnvParams = momentEqsParams->findBlock("MultiEnvMixingModel");
      const Expr::Tag mixFracTag = parse_nametag( multiEnvParams->findBlock("MixtureFraction")->findBlock("NameTag") );
      const Expr::Tag scalarVarTag = parse_nametag( multiEnvParams->findBlock("ScalarVariance")->findBlock("NameTag") );
      const Expr::Tag scalarDissTag = parse_nametag( multiEnvParams->findBlock("ScalarDissipation")->findBlock("NameTag") );
      factory.register_expression( scinew typename MultiEnvMixingModel<FieldT>::Builder(multiEnvWeightsTags, mixFracTag, scalarVarTag, scalarDissTag) );
    }
  }

  //==================================================================

  std::vector<EqnTimestepAdaptorBase*>
  parse_moment_transport_equations( Uintah::ProblemSpecP params,
                                    Uintah::ProblemSpecP wasatchParams,
                                    GraphCategories& gc )
  {
    typedef std::vector<EqnTimestepAdaptorBase*> EquationAdaptors;
    EquationAdaptors adaptors;
    EqnTimestepAdaptorBase* adaptor = NULL;
    Wasatch::TransportEquation* momtranseq = NULL;

    std::cout << "Parsing moment transport equations\n";
    GraphHelper* const solnGraphHelper = gc[ADVANCE_SOLUTION];
    GraphHelper* const icGraphHelper   = gc[INITIALIZATION  ];
    Expr::TagList transportedMomentTags;
    Expr::TagList abscissaeTags;
    Expr::TagList weightsTags;

    std::string basePhiName;
    params->get( "PopulationName", basePhiName );
    basePhiName = "m_" + basePhiName;

    std::string stagLoc;
    Uintah::ProblemSpecP stagLocParams = params->get( "StaggeredDirection", stagLoc );
    int nEnv = 1;
    params->get( "NumberOfEnvironments", nEnv );
    const int nEqs = 2*nEnv;
    
    std::vector< double> initialMoments;
    double val;
    initialMoments = std::vector< double >(nEqs);
    //loop over all basic exprs to find initialized moments 
    for( int i=0; i<nEqs; i++) { 
      std::stringstream ss;
      ss << i;
      std::string thisPhiName = basePhiName + "_" + ss.str();
      
      for( Uintah::ProblemSpecP expressionParams=wasatchParams->findBlock("BasicExpression");
          expressionParams != 0;
          expressionParams=expressionParams->findNextBlock("BasicExpression") ){

        std::string exprName;
        expressionParams->findBlock("NameTag")->getAttribute("name",exprName);
        
        if (exprName == thisPhiName ){
          expressionParams->get("Constant",val);
          initialMoments[i] = val;
          std::cout << "getting initial moment [" << i << "] = " << val <<std::endl; //quick debug statement
        }
      }
    }

    Expr::TagList multiEnvWeightsTags;
    
    if (stagLocParams) {

      // X-Staggered scalar
      if (stagLoc=="X") {
        for (int iMom=0; iMom<nEqs; iMom++) {
          preprocess_moment_transport_qmom<XVolField>(params, *solnGraphHelper->exprFactory,
                                        transportedMomentTags, abscissaeTags,
                                        weightsTags, multiEnvWeightsTags);

          const double momentID = (double) iMom; //here we will add any fractional moments
          std::stringstream ss;
          ss << iMom;
          std::string thisPhiName = basePhiName + "_" + ss.str();

          // create moment transport equation
          typedef MomentTransportEquation< XVolField > MomTransEq;
          momtranseq = scinew MomTransEq( thisPhiName,
                                          MomTransEq::get_moment_rhs_id( *solnGraphHelper->exprFactory,
                                                                      params,
                                                                      weightsTags,
                                                                      abscissaeTags,
                                                                      momentID,
                                                                      multiEnvWeightsTags,
                                                                      initialMoments[iMom])
                                         );
          adaptor = scinew EqnTimestepAdaptor< XVolField >( momtranseq );
          adaptors.push_back(adaptor);
        }
      } else if (stagLoc=="Y") {
        preprocess_moment_transport_qmom<YVolField>(params, *solnGraphHelper->exprFactory,
                                                 transportedMomentTags, abscissaeTags,
                                                 weightsTags, multiEnvWeightsTags);
        for (int iMom=0; iMom<nEqs; iMom++) {
          const double momentID = (double) iMom; //here we will add any fractional moments
          std::stringstream ss;
          ss << iMom;
          std::string thisPhiName = basePhiName + "_" + ss.str();

          // create moment transport equation
          typedef MomentTransportEquation< YVolField > MomTransEq;
          momtranseq = scinew MomTransEq( thisPhiName,
                                         MomTransEq::get_moment_rhs_id( *solnGraphHelper->exprFactory,
                                                                       params,
                                                                       weightsTags,
                                                                       abscissaeTags,
                                                                       momentID,
                                                                       multiEnvWeightsTags,
                                                                       initialMoments[iMom])
                                         );
          adaptor = scinew EqnTimestepAdaptor< YVolField >( momtranseq );
          adaptors.push_back(adaptor);
        }

      } else if (stagLoc=="Z") {
        preprocess_moment_transport_qmom<ZVolField>(params, *solnGraphHelper->exprFactory,
                                                 transportedMomentTags, abscissaeTags,
                                                 weightsTags, multiEnvWeightsTags);

        for (int iMom=0; iMom<nEqs; iMom++) {
          const double momentID = (double) iMom; //here we will add any fractional moments
          std::stringstream ss;
          ss << iMom;
          std::string thisPhiName = basePhiName + "_" + ss.str();

          // create moment transport equation
          typedef MomentTransportEquation< ZVolField > MomTransEq;
          momtranseq = scinew MomTransEq( thisPhiName,
                                         MomTransEq::get_moment_rhs_id( *solnGraphHelper->exprFactory,
                                                                       params,
                                                                       weightsTags,
                                                                       abscissaeTags,
                                                                       momentID,
                                                                       multiEnvWeightsTags,
                                                                       initialMoments[iMom])
                                         );
          adaptor = scinew EqnTimestepAdaptor< ZVolField >( momtranseq );
          adaptors.push_back(adaptor);
        }
      }
    } else if (!stagLocParams) {
      preprocess_moment_transport_qmom<SVolField>(params, *solnGraphHelper->exprFactory,
                                               transportedMomentTags, abscissaeTags,
                                               weightsTags, multiEnvWeightsTags);

      for (int iMom=0; iMom<nEqs; iMom++) {
        const double momentID = (double) iMom; //here we will add any fractional moments
        std::stringstream ss;
        ss << iMom;
        std::string thisPhiName = basePhiName + "_" + ss.str();

        // create moment transport equation
        typedef MomentTransportEquation< SVolField > MomTransEq;
        const Expr::ExpressionID rhsID = MomTransEq::get_moment_rhs_id( *solnGraphHelper->exprFactory,
                                                                       params,
                                                                       weightsTags,
                                                                       abscissaeTags,
                                                                       momentID,
                                                                       multiEnvWeightsTags,
                                                                       initialMoments[iMom]);
        momtranseq = scinew MomTransEq( thisPhiName, rhsID);
        adaptor = scinew EqnTimestepAdaptor< SVolField >( momtranseq );
        adaptors.push_back(adaptor);
        // tsaad: MUST INSERT ROOT IDS INTO THE SOLUTION GRAPH HELPER. WE NEVER DO
        // THAT ELSEWHERE, BUT THIS IS NEEDED TO MAKE THINGS EASIER WHEN USING
        // WASATCH IN ARCHES.
        solnGraphHelper->rootIDs.insert(rhsID);
        }
      }
    //
    // loop over the local adaptors and set the initial and boundary conditions on each equation attached to that adaptor
    for( EquationAdaptors::const_iterator ia=adaptors.begin(); ia!=adaptors.end(); ++ia ){
      EqnTimestepAdaptorBase* const adaptor = *ia;
      Wasatch::TransportEquation* momtranseq = adaptor->equation();

      //_____________________________________________________
      // set up initial conditions on this momentum equation
      try{
        proc0cout << "Setting initial conditions for scalability test equation: "
        << momtranseq->solution_variable_name()
        << std::endl;
        icGraphHelper->rootIDs.insert( momtranseq->initial_condition( *icGraphHelper->exprFactory ) );
      }
      catch( std::runtime_error& e ){
        std::ostringstream msg;
        msg << e.what()
        << std::endl
        << "ERORR while setting initial conditions on scalability test equation "
        << momtranseq->solution_variable_name()
        << std::endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
      proc0cout << "------------------------------------------------" << std::endl;
    }
    //
    return adaptors;
  }

  //-----------------------------------------------------------------

  template< typename FieldT >
  void setup_convective_flux_expression( const std::string dir,
                                         const Expr::Tag solnVarTag,
                                         Expr::Tag convFluxTag,
                                         const Expr::Tag volFracTag,
                                         const ConvInterpMethods convMethod,
                                         const Expr::Tag advVelocityTag,
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
      convFluxTag = Expr::Tag( solnVarTag.name() + "_convective_flux_" + dir, Expr::STATE_NONE );

      Expr::ExpressionBuilder* builder = NULL;

      const std::string interpMethod = get_conv_interp_method( convMethod );
      if( dir=="X" ){
        proc0cout << "SETTING UP CONVECTIVE FLUX EXPRESSION IN X DIRECTION USING " << interpMethod << std::endl;
        typedef typename ConvectiveFluxLimiter<
            typename Ops::InterpC2FXLimiter,
            typename Ops::InterpC2FXUpwind,
            typename OperatorTypeBuilder<Interpolant,FieldT,   XFace>::type, // scalar interp type
            typename OperatorTypeBuilder<Interpolant,XVolField,XFace>::type  // velocity interp type
            >::Builder ConvFluxLim;
        builder = scinew ConvFluxLim( convFluxTag, solnVarTag, advVelocityTag, convMethod, volFracTag );
      }
      else if( dir=="Y" ){
        proc0cout << "SETTING UP CONVECTIVE FLUX EXPRESSION IN Y DIRECTION USING " << interpMethod << std::endl;
        typedef typename ConvectiveFluxLimiter<
            typename Ops::InterpC2FYLimiter,
            typename Ops::InterpC2FYUpwind,
            typename OperatorTypeBuilder<Interpolant,FieldT,   YFace>::type, // scalar interp type
            typename OperatorTypeBuilder<Interpolant,YVolField,YFace>::type  // velocity interp type
            >::Builder ConvFluxLim;
        builder = scinew ConvFluxLim( convFluxTag, solnVarTag, advVelocityTag, convMethod, volFracTag );
      }
      else if( dir=="Z") {
        proc0cout << "SETTING UP CONVECTIVE FLUX EXPRESSION IN Z DIRECTION USING " << interpMethod << std::endl;
        typedef typename ConvectiveFluxLimiter<
            typename Ops::InterpC2FZLimiter,
            typename Ops::InterpC2FZUpwind,
            typename OperatorTypeBuilder<Interpolant,FieldT,   ZFace>::type, // scalar interp type
            typename OperatorTypeBuilder<Interpolant,ZVolField,ZFace>::type  // velocity interp type
            >::Builder ConvFluxLim;
        builder = scinew ConvFluxLim( convFluxTag, solnVarTag, advVelocityTag, convMethod, volFracTag );
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
    if      ( dir=="X" ) fs = CONVECTIVE_FLUX_X;
    else if ( dir=="Y" ) fs = CONVECTIVE_FLUX_Y;
    else if ( dir=="Z" ) fs = CONVECTIVE_FLUX_Z;
    else{
      std::ostringstream msg;
      msg << "Invalid direction selection for convective flux expression on " << solnVarTag.name() << std::endl;
      throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
    }
    info[ fs ] = convFluxTag;
  }

  template< typename FieldT >
  void setup_convective_flux_expression( Uintah::ProblemSpecP convFluxParams,
                                         const Expr::Tag solnVarTag,
                                         const Expr::Tag volFracTag,
                                         Expr::ExpressionFactory& factory,
                                         FieldTagInfo& info )
  {
    typedef OpTypes<FieldT> Ops;
    Expr::Tag convFluxTag, advVelocityTag;

    std::string dir, interpMethod;
    convFluxParams->get("Direction",dir);
    convFluxParams->get("Method",interpMethod);

    // get the tag for the advective velocity
    Uintah::ProblemSpecP advVelocityTagParam = convFluxParams->findBlock( "AdvectiveVelocity" );
    if( advVelocityTagParam ){
      advVelocityTag = parse_nametag( advVelocityTagParam->findBlock( "NameTag" ) );
    }

    // see if we have an expression set for the advective flux.
    Uintah::ProblemSpecP nameTagParam = convFluxParams->findBlock("NameTag");
    if( nameTagParam ) convFluxTag = parse_nametag( nameTagParam );

    setup_convective_flux_expression<FieldT>( dir,
                                              solnVarTag, convFluxTag, volFracTag,
                                              Wasatch::get_conv_interp_method(interpMethod),
                                              advVelocityTag,
                                              factory,
                                              info );
  }

  //-----------------------------------------------------------------

  template< typename OpT >
  Expr::ExpressionBuilder*
  build_diff_flux_expr( Uintah::ProblemSpecP diffFluxParams,
                        const Expr::Tag& diffFluxTag,
                        const Expr::Tag& primVarTag,
                        const Expr::Tag& densityTag )
  {
    if( diffFluxParams->findBlock("ConstantDiffusivity") ){
      double coef;
      diffFluxParams->get("ConstantDiffusivity",coef);
      typedef typename DiffusiveFlux< typename OpT::SrcFieldType,
                                      typename OpT::DestFieldType
                                      >::Builder Flux;
      return scinew Flux( diffFluxTag, primVarTag, coef, densityTag );
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
      typedef typename DiffusiveFlux2< typename OpT::SrcFieldType,
                                       typename OpT::DestFieldType
                                       >::Builder Flux;
      return scinew Flux( diffFluxTag, primVarTag, coef, densityTag );
    }
    return NULL;
  }

  template< typename FieldT>
  void setup_diffusive_flux_expression( Uintah::ProblemSpecP diffFluxParams,
                                        const Expr::Tag densityTag,
                                        const Expr::Tag primVarTag,
                                        const bool isStrong,
                                        Expr::ExpressionFactory& factory,
                                        FieldTagInfo& info )
  {
    typedef OpTypes<FieldT> MyOpTypes;
    const std::string& primVarName = primVarTag.name();
    Expr::Tag diffFluxTag;  // we will populate this.

    std::string dir;
    diffFluxParams->get("Direction",dir);

    // see if we have an expression set for the diffusive flux.
    Uintah::ProblemSpecP nameTagParam = diffFluxParams->findBlock("NameTag");
    if( nameTagParam ){
      diffFluxTag = parse_nametag( nameTagParam );
    }
    else{ // build an expression for the diffusive flux.

      diffFluxTag = Expr::Tag( primVarName+"_diffFlux_"+dir, Expr::STATE_NONE );

      Expr::ExpressionBuilder* builder = NULL;
      if( dir=="X" )      builder = build_diff_flux_expr<typename MyOpTypes::GradX>(diffFluxParams,diffFluxTag,primVarTag,densityTag);
      else if( dir=="Y" ) builder = build_diff_flux_expr<typename MyOpTypes::GradY>(diffFluxParams,diffFluxTag,primVarTag,densityTag);
      else if( dir=="Z")  builder = build_diff_flux_expr<typename MyOpTypes::GradZ>(diffFluxParams,diffFluxTag,primVarTag,densityTag);

      if( builder == NULL ){
        std::ostringstream msg;
        msg << "Could not build a diffusive flux expression for '" << primVarName << "'" << std::endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
      factory.register_expression( builder );
    }

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

  //------------------------------------------------------------------

  template< typename GradT, typename InterpT >
  Expr::ExpressionBuilder*
  build_diff_vel_expr( Uintah::ProblemSpecP diffVelParams,
                       const Expr::Tag& diffVelTag,
                       const Expr::Tag& primVarTag )
  {
    if( diffVelParams->findBlock("ConstantDiffusivity") ){
      typedef typename DiffusiveVelocity<GradT>::Builder Velocity;
      double coef;
      diffVelParams->get("ConstantDiffusivity",coef);
      return scinew Velocity( diffVelTag, primVarTag, coef );
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
      typedef typename DiffusiveVelocity2< GradT, InterpT >::Builder Velocity;
      const Expr::Tag coef = parse_nametag( diffVelParams->findBlock("DiffusionCoefficient")->findBlock("NameTag") );
      return scinew Velocity( diffVelTag, primVarTag, coef );
    }
    return NULL;
  }

  template< typename FieldT>
  void setup_diffusive_velocity_expression( Uintah::ProblemSpecP diffVelParams,
                                            const Expr::Tag primVarTag,
                                            Expr::ExpressionFactory& factory,
                                            FieldTagInfo& info )
  {
    typedef OpTypes<FieldT> MyOpTypes;
    const std::string& primVarName = primVarTag.name();
    Expr::Tag diffVelTag;  // we will populate this.

    std::string dir;
    diffVelParams->get("Direction",dir);

    // see if we have an expression set for the diffusive velocity.
    Uintah::ProblemSpecP nameTagParam = diffVelParams->findBlock("NameTag");
    if( nameTagParam ){
      diffVelTag = parse_nametag( nameTagParam );
    }
    else{ // build an expression for the diffusive velocity.

      diffVelTag = Expr::Tag( primVarName+"_diffVelocity_"+dir, Expr::STATE_NONE );

      Expr::ExpressionBuilder* builder = NULL;
      if( dir=="X" )       builder = build_diff_vel_expr<typename MyOpTypes::GradX,typename MyOpTypes::InterpC2FX>(diffVelParams,diffVelTag,primVarTag);
      else if( dir=="Y" )  builder = build_diff_vel_expr<typename MyOpTypes::GradY,typename MyOpTypes::InterpC2FY>(diffVelParams,diffVelTag,primVarTag);
      else if( dir=="Z")   builder = build_diff_vel_expr<typename MyOpTypes::GradZ,typename MyOpTypes::InterpC2FZ>(diffVelParams,diffVelTag,primVarTag);

      if( builder == NULL ){
        std::ostringstream msg;
        msg << "Could not build a diffusive velocity expression for '"
            << primVarName << "'" << std::endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
      factory.register_expression( builder );
    }

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

  //------------------------------------------------------------------

  //==================================================================
  // explicit template instantiation
#define INSTANTIATE( FIELDT )                                   \
                                                                \
    template void setup_diffusive_flux_expression<FIELDT>(      \
       Uintah::ProblemSpecP diffFluxParams,                     \
       const Expr::Tag densityTag,                              \
       const Expr::Tag primVarTag,                              \
       const bool isStrong,                                     \
       Expr::ExpressionFactory& factory,                        \
       FieldTagInfo& info );                                    \
                                                                \
    template void setup_diffusive_velocity_expression<FIELDT>(  \
       Uintah::ProblemSpecP diffVelParams,                      \
       const Expr::Tag primVarTag,                              \
       Expr::ExpressionFactory& factory,                        \
       FieldTagInfo& info );                                    \
                                                                \
    template void setup_convective_flux_expression<FIELDT>(     \
        const std::string dir,                                  \
        const Expr::Tag solnVarTag,                             \
        Expr::Tag convFluxTag,                                  \
        const Expr::Tag volFracTag,                             \
        const ConvInterpMethods convMethod,                     \
        const Expr::Tag advVelocityTag,                         \
        Expr::ExpressionFactory& factory,                       \
        FieldTagInfo& info );                                   \
                                                                \
    template void setup_convective_flux_expression<FIELDT>(     \
        Uintah::ProblemSpecP convFluxParams,                    \
        const Expr::Tag solnVarName,                            \
        const Expr::Tag volFracTag,                             \
        Expr::ExpressionFactory& factory,                       \
        FieldTagInfo& info );

  INSTANTIATE( SVolField )
  INSTANTIATE( XVolField )
  INSTANTIATE( YVolField )
  INSTANTIATE( ZVolField )

  //-----------------------------------------------------------------

} // namespace Wasatch
