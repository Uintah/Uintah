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

#ifndef ParticleMassEquation_h
#define ParticleMassEquation_h

#include "ParticleEquationBase.h"
#include <expression/ExprLib.h>


namespace WasatchCore{


  /**
   *  @class ParticleMassEquation
   *  @ingroup WasatchParticles
   *  @author Tony Saad
   *  @date   June 2014
   *  @brief Solves for particle mass.
   */
  class ParticleMassEquation : public ParticleEquationBase
  {
  public:

    /**
     * \brief Construct a transport equation for the mass of a particle
     *
     * \param solnVarName The name of the solution variable for this equation
     *
     * \param pdir Specifies which position or momentum component this equation solves.
     *
     * \param particlePositionTags A taglist containing the tags of x, y, and z
     *        particle coordiantes. Those may be needed by some particle
     *        expressions that require particle operators
     *
     * \param particleSizeTag Particle size tag. May be needed by some expressions.
     *
     * \param particleEqsSpec the Uintah parser information for this Particle equation
     *
     * \param gc The GraphCategories object from Wasatch
     *
     */
    ParticleMassEquation( const std::string& solnVarName,
                          const Direction pdir,
                          const Expr::TagList& particlePositionTags,
                          const Expr::Tag& particleSizeTag,
                          Uintah::ProblemSpecP particleEqsSpec,
                          GraphCategories& gc );

    ~ParticleMassEquation();

    /**
     *  \brief setup the initial conditions for this transport equation.
     */
    Expr::ExpressionID initial_condition( Expr::ExpressionFactory& );

    /**
     *  \brief Used to check the validity of the boundary conditions specified
     *   by the user at a given boundary and also to infer/add new BCs on the
     *   type of boundary.  Example: at a stationary impermeable wall, we can
     *   immediately infer zero-velocity boundary conditions and check whether
     *   the user has specified any velocity BCs at that boundary. See examples
     *   in the momentum transport equation.
     */
    void setup_boundary_conditions( WasatchBCHelper& bcHelper,
                                    GraphCategories& graphCat)
    {}
    
    /**
     *  \brief Set up the boundary condition on initial conditions equation.
     */
    void apply_initial_boundary_conditions( const GraphHelper& graphHelper,
                                            WasatchBCHelper& bcHelper )
    {}
    
    /**
     *  \brief Set up the boundary condition evaluators for this
     *  Equation.
     */
    void apply_boundary_conditions( const GraphHelper& graphHelper,
                                    WasatchBCHelper& bcHelper )
    {}

  private:
    const Expr::Tag pSrcTag_, pRhoTag_;
    void setup();
    Expr::ExpressionID setup_rhs();
  };

} // namespace Particle

#endif // ParticleMassEquation_h
