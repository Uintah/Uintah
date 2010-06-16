#include "ParseEquation.h"

#include <iostream>

#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/ProblemSpec/ProblemSpec.h>

#include <CCA/Components/Wasatch/StringNames.h>

//-- Add headers for individual transport equations here --//
#include "TemperatureTransportEquation.h"


namespace Wasatch{

  //------------------------------------------------------------------

  EqnTimestepAdaptorBase::EqnTimestepAdaptorBase( Expr::TransportEquation* eqn )
    : eqn_(eqn)
  {}

  //------------------------------------------------------------------

  EqnTimestepAdaptorBase::~EqnTimestepAdaptorBase()
  {
    delete eqn_;
  }

  //==================================================================

  EqnTimestepAdaptorBase* parse_equation( Uintah::ProblemSpecP params,
                                          GraphCategories& gc )
  {
    const StringNames& sName = StringNames::self();

    EqnTimestepAdaptorBase* adaptor = NULL;
    Expr::TransportEquation* transeqn = NULL;

    std::string eqnLabel;

    params->getAttribute( "label", eqnLabel );
    std::cout << "Building equation with label '" << eqnLabel << "'" << std::endl;

    GraphHelper* const solnGraphHelper = gc[ADVANCE_SOLUTION];
    GraphHelper* const icGraphHelper   = gc[INITIALIZATION  ];

    //___________________________________________________________________________
    // resolve the transport equation to be solved and create the adaptor for it.
    //
    std::cout << "Creating transport equation for '" << eqnLabel << "'" << std::endl;
    // jcs get this timestepper stuff working...
    if( eqnLabel == sName.temperature ){
      transeqn = new TemperatureTransportEquation( *solnGraphHelper->exprFactory, params );
      adaptor = new EqnTimestepAdaptor<TemperatureTransportEquation::FieldT>( transeqn );
    }

    /* add additional transport equations here */

    else{
      std::ostringstream msg;
      msg << "No transport equation was resolved for '" << eqnLabel << "'" << std::endl;
      throw Uintah::InvalidValue( msg.str(), __FILE__, __LINE__ );
    }

    //_____________________________________________________
    // set up initial conditions on this transport equation
    try{
      std::cout << "setting ICs for transport equation '" << eqnLabel << "'" << std::endl;
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

    //______________________________________________________
    // set up boundary conditions on this transport equation
    try{
      std::cout << "setting BCs for transport equation '" << eqnLabel << "'" << std::endl;
      transeqn->setup_boundary_conditions( *solnGraphHelper->exprFactory );
    }
    catch( std::runtime_error& e ){
      std::ostringstream msg;
      msg << e.what()
          << std::endl
          << "ERORR while setting boundary conditions on equation '" << eqnLabel << "'"
          << std::endl;
      throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
    }

    return adaptor;
  }
  
} // namespace Wasatch
