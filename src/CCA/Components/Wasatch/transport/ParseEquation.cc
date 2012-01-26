//-- Wasatch Includes --//
#include "ParseEquation.h"
#include "../TimeStepper.h"
#include <CCA/Components/Wasatch/StringNames.h>
#include "../ParseTools.h"

//-- Add headers for individual transport equations here --//
#include "ScalarTransportEquation.h"
#include "ScalabilityTestTransportEquation.h"
#include "TemperatureTransportEquation.h"
#include "MomentumTransportEquation.h"
#include "MomentTransportEquation.h"
#include <CCA/Components/Wasatch/Expressions/PBE/QMOM.h>

//-- Uintah includes --//
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Parallel/Parallel.h>

//-- Expression Library includes --//
#include <CCA/Components/Wasatch/transport/TransportEquation.h>

#include <iostream>


namespace Wasatch{


  /**
   *  \class EqnTimestepAdaptor
   *  \author James C. Sutherland
   *  \date June, 2010
   *  \modifier Amir Biglari
   *  \date July, 2011
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
    const StringNames& sName = StringNames::self();

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

    } else if( eqnLabel == sName.temperature ){
      transeqn = scinew TemperatureTransportEquation( *solnGraphHelper->exprFactory );
      adaptor = scinew EqnTimestepAdaptor< TemperatureTransportEquation::FieldT >( transeqn );

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

  std::vector<EqnTimestepAdaptorBase*> parse_scalability_test( Uintah::ProblemSpecP params,
                                                              GraphCategories& gc )
  {
    typedef std::vector<EqnTimestepAdaptorBase*> EquationAdaptors;
    EquationAdaptors adaptors;
    EqnTimestepAdaptorBase* adaptor = NULL;
    Wasatch::TransportEquation* scaltesteqn = NULL;

    std::string basePhiName;
    params->get( "SolutionVariable", basePhiName );

    std::string stagLoc;
    Uintah::ProblemSpecP stagLocParams = params->get( "StaggeredDirection", stagLoc );

    int nEqs = 1;
    params->get( "NumberOfEquations", nEqs );

    GraphHelper* const solnGraphHelper = gc[ADVANCE_SOLUTION];
    GraphHelper* const icGraphHelper   = gc[INITIALIZATION  ];

      if (stagLocParams) {

        // X-Staggered scalar
        if (stagLoc=="X") {
          for (int iEq=0; iEq<nEqs; iEq++) {
            std::stringstream ss;
            ss << iEq;
            std::string thisPhiName = basePhiName + ss.str();
            // set initial condition and register it
            Expr::Tag icTag( thisPhiName, Expr::STATE_N );
            Expr::Tag indepVarTag( "XXVOL", Expr::STATE_NONE );
            typedef Expr::SinFunction<XVolField>::Builder Builder;
            icGraphHelper->exprFactory->register_expression( scinew Builder( icTag, indepVarTag, 1.0, 1, 0.0) );

            // create the transport equation with all-to-all source term
            typedef ScalabilityTestTransportEquation< XVolField > ScalTestEqn;
            const Expr::ExpressionID rhsID = ScalTestEqn::get_rhs_expr_id( thisPhiName,
                                                                          *solnGraphHelper->exprFactory,
                                                                          params );
            scaltesteqn = scinew ScalTestEqn( basePhiName,
                                             thisPhiName,
                                             rhsID );
            solnGraphHelper->rootIDs.insert(rhsID);
            
            adaptor = scinew EqnTimestepAdaptor< XVolField >( scaltesteqn );
            adaptors.push_back(adaptor);
          }
        } else if (stagLoc=="Y") {
          for (int iEq=0; iEq<nEqs; iEq++) {
            std::stringstream ss;
            ss << iEq;
            std::string thisPhiName = basePhiName + ss.str();

            // set initial condition and register it
            Expr::Tag icTag( thisPhiName, Expr::STATE_N );
            Expr::Tag indepVarTag( "XYVOL", Expr::STATE_NONE );
            typedef Expr::SinFunction<YVolField>::Builder Builder;
            icGraphHelper->exprFactory->register_expression( scinew Builder( icTag, indepVarTag, 1.0, 1, 0.0) );

            // create the transport equation with all-to-all source term
            typedef ScalabilityTestTransportEquation< YVolField > ScalTestEqn;
            const Expr::ExpressionID rhsID = ScalTestEqn::get_rhs_expr_id( thisPhiName,
                                                                          *solnGraphHelper->exprFactory,
                                                                          params );
            scaltesteqn = scinew ScalTestEqn( basePhiName,
                                             thisPhiName,
                                             rhsID );
            solnGraphHelper->rootIDs.insert(rhsID);

            adaptor = scinew EqnTimestepAdaptor< YVolField >( scaltesteqn );
            adaptors.push_back(adaptor);
          }

        } else if (stagLoc=="Z") {
          for (int iEq=0; iEq<nEqs; iEq++) {
            std::stringstream ss;
            ss << iEq;
            std::string thisPhiName = basePhiName + ss.str();

            // set initial condition and register it
            Expr::Tag icTag( thisPhiName, Expr::STATE_N );
            Expr::Tag indepVarTag( "ZZVOL", Expr::STATE_NONE );
            typedef Expr::SinFunction<ZVolField>::Builder Builder;
            icGraphHelper->exprFactory->register_expression( scinew Builder( icTag, indepVarTag, 1.0, 1, 0.0) );

            // create the transport equation with all-to-all source term
            typedef ScalabilityTestTransportEquation< ZVolField > ScalTestEqn;
            const Expr::ExpressionID rhsID = ScalTestEqn::get_rhs_expr_id( thisPhiName,
                                                                          *solnGraphHelper->exprFactory,
                                                                          params );
            scaltesteqn = scinew ScalTestEqn( basePhiName,
                                             thisPhiName,
                                             rhsID );
            solnGraphHelper->rootIDs.insert(rhsID);

            adaptor = scinew EqnTimestepAdaptor< ZVolField >( scaltesteqn );
            adaptors.push_back(adaptor);
          }

        }
      } else if (!stagLocParams) {

        for (int iEq=0; iEq<nEqs; iEq++) {

          std::stringstream ss;
          ss << iEq;
          std::string thisPhiName = basePhiName + ss.str();

          // set initial condition and register it
          Expr::Tag icTag( thisPhiName, Expr::STATE_N );
          Expr::Tag indepVarTag( "XSVOL", Expr::STATE_NONE );
          typedef Expr::SinFunction<SVolField>::Builder Builder;
          icGraphHelper->exprFactory->register_expression( scinew Builder( icTag, indepVarTag, 1.0, 1, 0.0) );

          // create the transport equation with all-to-all source term
          typedef ScalabilityTestTransportEquation< SVolField > ScalTestEqn;
          const Expr::ExpressionID rhsID = ScalTestEqn::get_rhs_expr_id( thisPhiName,
                                                                        *solnGraphHelper->exprFactory,
                                                                        params );
          scaltesteqn = scinew ScalTestEqn( basePhiName,
                                           thisPhiName,
                                           rhsID );
          solnGraphHelper->rootIDs.insert(rhsID);

          adaptor = scinew EqnTimestepAdaptor< SVolField >( scaltesteqn );
          adaptors.push_back(adaptor);
        }
      }
    //
    // loop over the local adaptors and set the initial and boundary conditions on each equation attached to that adaptor
    for( EquationAdaptors::const_iterator ia=adaptors.begin(); ia!=adaptors.end(); ++ia ){
      EqnTimestepAdaptorBase* const adaptor = *ia;
      Wasatch::TransportEquation* scaltesteqn = adaptor->equation();

      //_____________________________________________________
      // set up initial conditions on this momentum equation
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
    //
    return adaptors;
  }

  //==================================================================

  std::vector<EqnTimestepAdaptorBase*> parse_momentum_equations( Uintah::ProblemSpecP params,
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
                                      *solnGraphHelper->exprFactory,
                                      params,
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
                                     *solnGraphHelper->exprFactory,
                                     params,
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
                                     *solnGraphHelper->exprFactory,
                                     params,
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
    //
    return adaptors;
  }

  //==================================================================

  template<typename FieldT>
  void preprocess_moment_transport_qmom( Uintah::ProblemSpecP momentEqsParams,
                                         Expr::ExpressionFactory& factory,
                                         Expr::TagList& transportedMomentTags,
                                         Expr::TagList& abscissaeTags,
                                         Expr::TagList& weightsTags )
  {
    // check for supersaturation
    Expr::Tag superSaturationTag = Expr::Tag();
    Uintah::ProblemSpecP superSaturationParams = momentEqsParams->findBlock("SuperSaturationExpression");
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
  }

  //==================================================================

  std::vector<EqnTimestepAdaptorBase*>
  parse_moment_transport_equations( Uintah::ProblemSpecP params,
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

    if (stagLocParams) {

      // X-Staggered scalar
      if (stagLoc=="X") {
        for (int iMom=0; iMom<nEqs; iMom++) {
          preprocess_moment_transport_qmom<XVolField>(params, *solnGraphHelper->exprFactory,
                                        transportedMomentTags, abscissaeTags,
                                        weightsTags);

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
                                                                      momentID)
                                         );
          adaptor = scinew EqnTimestepAdaptor< XVolField >( momtranseq );
          adaptors.push_back(adaptor);
        }
      } else if (stagLoc=="Y") {
        preprocess_moment_transport_qmom<YVolField>(params, *solnGraphHelper->exprFactory,
                                                 transportedMomentTags, abscissaeTags,
                                                 weightsTags);
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
                                                                       momentID)
                                         );
          adaptor = scinew EqnTimestepAdaptor< YVolField >( momtranseq );
          adaptors.push_back(adaptor);
        }

      } else if (stagLoc=="Z") {
        preprocess_moment_transport_qmom<ZVolField>(params, *solnGraphHelper->exprFactory,
                                                 transportedMomentTags, abscissaeTags,
                                                 weightsTags);

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
                                                                       momentID)
                                         );
          adaptor = scinew EqnTimestepAdaptor< ZVolField >( momtranseq );
          adaptors.push_back(adaptor);
        }
      }
    } else if (!stagLocParams) {
      preprocess_moment_transport_qmom<SVolField>(params, *solnGraphHelper->exprFactory,
                                               transportedMomentTags, abscissaeTags,
                                               weightsTags);

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
                                                                       momentID);
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

} // namespace Wasatch
