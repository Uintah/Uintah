#ifndef Wasatch_ParseEquations_h
#define Wasatch_ParseEquations_h

#include <Core/ProblemSpec/ProblemSpecP.h>

#include "../GraphHelperTools.h"
#include "../TimeStepper.h"

#include <expression/TransportEquation.h>

namespace Wasatch{

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
    EqnTimestepAdaptorBase( Expr::TransportEquation* eqn );
    Expr::TransportEquation* const eqn_;
  public:
    virtual ~EqnTimestepAdaptorBase();
    virtual void hook( TimeStepper& ts ) const = 0;
    Expr::TransportEquation* equation(){ return eqn_; }
    const Expr::TransportEquation* equation() const{ return eqn_; }
  };


  /**
   *  \class EqnTimestepAdaptor
   *  \Author James C. Sutherland
   *  \date June, 2010
   *
   *  \brief Strongly typed adaptor provides the key functionality to
   *  plug a transport equation into a TimeStepper.
   */
  template< typename FieldT >
  class EqnTimestepAdaptor : public EqnTimestepAdaptorBase
  {
  public:
    EqnTimestepAdaptor( Expr::TransportEquation* eqn ) : EqnTimestepAdaptorBase(eqn) {}
    void hook( TimeStepper& ts ) const
    {
      ts.add_equation<FieldT>( eqn_->solution_variable_name(),
                               eqn_->get_rhs_id() );
    }
  };


  /**
   *  \brief Build the transport equation specified by "params"
   *
   *  \param params the tag from the input file specifying the
   *         transport equation.
   *
   *  \param gc the GraphCategories
   *
   *  \param ts the TimeStepper object that we will load this
   *         transport equation on.
   *
   *  \return an EqnTimestepAdaptorBase object that can be used to
   *          plug this transport equation into a TimeStepper.
   */
  EqnTimestepAdaptorBase*
  parse_equation( Uintah::ProblemSpecP params,
                  GraphCategories& gc );

}// namespace Wasatch

#endif // Wasatch_ParseEquations_h
