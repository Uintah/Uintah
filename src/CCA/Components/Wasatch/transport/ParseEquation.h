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

#ifndef Wasatch_ParseEquations_h
#define Wasatch_ParseEquations_h

#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Ports/SolverInterface.h>

#include "../GraphHelperTools.h"
#include <CCA/Components/Wasatch/transport/TransportEquation.h>

/**
 *  \file ParseEquation.h
 *  \brief Parser tools for transport equations.
 */

namespace Wasatch{

  class TimeStepper;
  class TransportEquation;

  /** \addtogroup WasatchParser
   *  @{
   */

  /**
   *  \class EqnTimestepAdaptorBase
   *  \author James C. Sutherland
   *  \date June, 2010
   *  \modifier Amir Biglari
   *  \date July, 2011
   *
   *  This serves as a means to have a container of adaptors.  These
   *  adaptors will plug a strongly typed transport equation into a
   *  time integrator, preserving the type information which is
   *  required for use by the integrator.
   */
  class EqnTimestepAdaptorBase
  {
  protected:
    EqnTimestepAdaptorBase( TransportEquation* eqn);
    TransportEquation* const eqn_;

  public:
    virtual ~EqnTimestepAdaptorBase();
    virtual void hook( TimeStepper& ts ) const = 0;
    TransportEquation* equation(){ return eqn_; }
    const TransportEquation* equation() const{ return eqn_; }
  };


  /**
   *  \brief Build the transport equation specified by "params"
   *
   *  \param params the tag from the input file specifying the
   *         transport equation.
   *
   *  \param densityTag a tag for the density to be passed to
   *         the scalar transport equations if it is needed.
   *         othwise it will be an empty tag.
   *
   *  \param gc the GraphCategories.
   *
   *  \return an EqnTimestepAdaptorBase object that can be used to
   *          plug this transport equation into a TimeStepper.
   */
  EqnTimestepAdaptorBase*
  parse_equation( Uintah::ProblemSpecP params,
                  const Expr::Tag densityTag,
                  const bool isConstDensity,
                  GraphCategories& gc );

  /**
   *  \brief Build the momentum equation specified by "params"
   *
   *  \param params The XML block from the input file specifying the
   *         momentum equation. This will be \verbatim <MomentumEquations>\endverbatim.
   *
   *  \param gc The GraphCategories.
   *
   *  \return an EqnTimestepAdaptorBase object that can be used to
   *          plug this transport equation into a TimeStepper.
   */
  std::vector<EqnTimestepAdaptorBase*> parse_scalability_test( Uintah::ProblemSpecP params,
                                                               GraphCategories& gc );

  /**
   *  \brief Build the momentum equation specified by "params"
   *
   *  \param params The XML block from the input file specifying the
   *         momentum equation. This will be \verbatim <MomentumEquations>\endverbatim.
   *
   *  \param gc The GraphCategories.
   *
   *  \return an EqnTimestepAdaptorBase object that can be used to
   *          plug this transport equation into a TimeStepper.
   */
  std::vector<EqnTimestepAdaptorBase*> parse_momentum_equations( Uintah::ProblemSpecP params,
                                                                 const Expr::Tag densityTag,
                                                                 GraphCategories& gc,
                                                                 Uintah::SolverInterface& linSolver);

  /**
   *  \brief Build moment transport equations specified by "params"
   *
   *  \param params The XML block from the input file specifying the
   *         momentum equation. This will be <MomentumEquations>.
   *
   *  \param densityTag a tag for the density to be passed to
   *         the momentum equations if it is needed. othwise
   *         it will be an empty tag.
   *
   *  \param gc The GraphCategories.
   *
   *  \return a vector of EqnTimestepAdaptorBase objects that can be used to
   *          plug this transport equation into a TimeStepper.
   */
  std::vector<EqnTimestepAdaptorBase*> parse_moment_transport_equations( Uintah::ProblemSpecP params,
                                                                        GraphCategories& gc);

//  template<typename FieldT>
//  void process_moment_transport_qmom(Uintah::ProblemSpecP params,
//                                     Expr::ExpressionFactory& factory,
//                                     Expr::TagList& transportedMomentTags,
//                                     Expr::TagList& abscissaeTags,
//                                     Expr::TagList& weightsTags);


  /** @} */

}// namespace Wasatch

#endif // Wasatch_ParseEquations_h
