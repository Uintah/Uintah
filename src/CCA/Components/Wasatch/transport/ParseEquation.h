#ifndef Wasatch_ParseEquations_h
#define Wasatch_ParseEquations_h

#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

#include "../GraphHelperTools.h"


/**
 *  \file ParseEquation.h
 *  \brief Parser tools for transport equations.
 */

namespace Expr{ class TransportEquation; }

namespace Wasatch{

  class TimeStepper;

  /** \addtogroup WasatchParser
   *  @{
   */

  /**
   *  \class EqnTimestepAdaptorBase
   *  \author James C. Sutherland
   *  \date June, 2010
   *
   *  This serves as a means to have a container of adaptors.  These
   *  adaptors will plug a strongly typed transport equation into a
   *  time integrator, preserving the type information which is
   *  required for use by the integrator.
   */
  class EqnTimestepAdaptorBase
  {
  protected:
    EqnTimestepAdaptorBase( Expr::TransportEquation* eqn, 
                           Uintah::ProblemSpecP transEqnParams );
    Expr::TransportEquation* const eqn_;
    Uintah::ProblemSpecP transEqnParams_;
  public:
    virtual ~EqnTimestepAdaptorBase();
    virtual void hook( TimeStepper& ts ) const = 0;
    Expr::TransportEquation* equation(){ return eqn_; }
    const Expr::TransportEquation* equation() const{ return eqn_; }
    Uintah::ProblemSpecP transEqnParams() { return transEqnParams_;}
  };


  /**
   *  \brief Build the transport equation specified by "params"
   *
   *  \param params the tag from the input file specifying the
   *         transport equation.
   *
   *  \param gc the GraphCategories.
   *
   *  \return an EqnTimestepAdaptorBase object that can be used to
   *          plug this transport equation into a TimeStepper.
   */
  EqnTimestepAdaptorBase*
  parse_equation( Uintah::ProblemSpecP params,
                  GraphCategories& gc );
  
  /**
   *  \brief Build the momentum equation specified by "params"
   *
   *  \param params The XML block from the input file specifying the
   *         momentum equation. This will be <MomentumEquations>.
   *
   *  \param gc The GraphCategories.
   *
   *  \return an EqnTimestepAdaptorBase object that can be used to
   *          plug this transport equation into a TimeStepper.
   */  
  EqnTimestepAdaptorBase* parse_momentum_equations( Uintah::ProblemSpecP params,
                                                   GraphCategories& gc );

  /** @} */

}// namespace Wasatch

#endif // Wasatch_ParseEquations_h
