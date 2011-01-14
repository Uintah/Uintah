//-- Wasatch Includes --//
#include "ParseEquation.h"
#include "../TimeStepper.h"
#include <CCA/Components/Wasatch/StringNames.h>

//-- Add headers for individual transport equations here --//
#include "ScalarTransportEquation.h"
#include "TemperatureTransportEquation.h"

//-- Uintah includes --//
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

//-- Expression Library includes --//
#include <expression/TransportEquation.h>

#include <iostream>


namespace Wasatch{


  /**
   *  \class EqnTimestepAdaptor
   *  \author James C. Sutherland
   *  \date June, 2010
   *
   *  \brief Strongly typed adaptor provides the key functionality to
   *         plug a transport equation into a TimeStepper.
   */
  template< typename FieldT >
  class EqnTimestepAdaptor : public EqnTimestepAdaptorBase
  {
  public:
    EqnTimestepAdaptor( Expr::TransportEquation* eqn, Uintah::ProblemSpecP transEqnParams ) : EqnTimestepAdaptorBase(eqn, transEqnParams) {}
    void hook( TimeStepper& ts ) const
    {
      ts.add_equation<FieldT>( eqn_->solution_variable_name(),
                               eqn_->get_rhs_id() );
    }
  };

  //==================================================================

  EqnTimestepAdaptorBase::EqnTimestepAdaptorBase( Expr::TransportEquation* eqn,
                                                 Uintah::ProblemSpecP transEqnParams)
    : eqn_(eqn), transEqnParams_(transEqnParams)
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

    std::string eqnLabel, solnVariable;

    params->getAttribute( "equation", eqnLabel );
    params->get( "SolutionVariable", solnVariable );
    
    GraphHelper* const solnGraphHelper = gc[ADVANCE_SOLUTION];
    GraphHelper* const icGraphHelper   = gc[INITIALIZATION  ];

    //___________________________________________________________________________
    // resolve the transport equation to be solved and create the adaptor for it.
    //
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Creating transport equation for '" << eqnLabel << "'" << std::endl;

    if( eqnLabel == "generic" ){
       // find out if this corresponds to a staggered or non-staggered field
      std::string staggeredDirection;
      Uintah::ProblemSpecP scalarStaggeredParams = params->get( "StaggeredDirection", staggeredDirection );
      
      if ( scalarStaggeredParams ) {
        // if staggered, then determine the staggering direction
        std::cout << "Detected staggered scalar '" << eqnLabel << "'" << std::endl;
        
        // make proper calls based on the direction
        if ( staggeredDirection=="X" ) {
          std::cout << "Setting up staggered scalar transport equation in direction: '" << staggeredDirection << "'" << std::endl;
          typedef ScalarTransportEquation< XVolField > ScalarTransEqn;
          transeqn = scinew ScalarTransEqn( ScalarTransEqn::get_phi_name( params ),
                                            ScalarTransEqn::get_rhs_expr_id( *solnGraphHelper->exprFactory, params ) );
          adaptor = scinew EqnTimestepAdaptor< XVolField >( transeqn, params );
          
        } else if ( staggeredDirection=="Y" ) {
          std::cout << "Setting up staggered scalar transport equation in direction: '" << staggeredDirection << "'" << std::endl;
          typedef ScalarTransportEquation< YVolField > ScalarTransEqn;
          transeqn = scinew ScalarTransEqn( ScalarTransEqn::get_phi_name( params ),
                                            ScalarTransEqn::get_rhs_expr_id( *solnGraphHelper->exprFactory, params ) );
          adaptor = scinew EqnTimestepAdaptor< YVolField >( transeqn, params );
          
        } else if (staggeredDirection=="Z") {
          std::cout << "Setting up staggered scalar transport equation in direction: '" << staggeredDirection << "'" << std::endl;
          typedef ScalarTransportEquation< ZVolField > ScalarTransEqn;
          transeqn = scinew ScalarTransEqn( ScalarTransEqn::get_phi_name( params ),
                                            ScalarTransEqn::get_rhs_expr_id( *solnGraphHelper->exprFactory, params ) );
          adaptor = scinew EqnTimestepAdaptor< ZVolField >( transeqn, params );
          
        } else {
          std::ostringstream msg;
          msg << "ERROR: No direction is specified for staggered field '" << eqnLabel << "'. Please revise your input file." << std::endl;
          throw Uintah::InvalidValue( msg.str(), __FILE__, __LINE__ );                             
        }
        
      } else if ( !scalarStaggeredParams ) {
        // in this case, the scalar field is not staggered
        std::cout << "Detected non-staggered scalar '" << eqnLabel << "'" << std::endl;
        typedef ScalarTransportEquation< SVolField > ScalarTransEqn;
        transeqn = scinew ScalarTransEqn( ScalarTransEqn::get_phi_name( params ),
                                       ScalarTransEqn::get_rhs_expr_id( *solnGraphHelper->exprFactory, params ) );
        adaptor = scinew EqnTimestepAdaptor< SVolField >( transeqn, params );
      }
      
    } else if( eqnLabel == sName.temperature ){
      transeqn = scinew TemperatureTransportEquation( *solnGraphHelper->exprFactory );
      adaptor = scinew EqnTimestepAdaptor< TemperatureTransportEquation::FieldT >( transeqn, params );
      
    } else {
      std::ostringstream msg;
      msg << "ERROR: No transport equation was specified '" << eqnLabel << "'. Please revise your input file" << std::endl;
      throw Uintah::InvalidValue( msg.str(), __FILE__, __LINE__ );
    }

    //_____________________________________________________
    // set up initial conditions on this transport equation
    try{
      std::cout << "Setting initial conditions for transport equation '" << eqnLabel << "'" << std::endl;
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
      std::cout << "Setting BCs for transport equation '" << eqnLabel << "'" << std::endl;
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
    std::cout << "------------------------------------------------" << std::endl;
    return adaptor;
  }
  
} // namespace Wasatch
