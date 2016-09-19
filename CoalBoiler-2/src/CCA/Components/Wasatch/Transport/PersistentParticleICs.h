#ifndef PersistentParticleICs_h
#define PersistentParticleICs_h
#include "ParticleEquationBase.h"

namespace WasatchCore{


/**
 *  @class PersistentInitialParticleSize
 *  @ingroup WasatchParticles
 *  @author Josh McConnell
 *  @date August 2016
 *  @brief Solves d(pd_0) / dt = 0 where pd_0 is the initial particle size
 *  todo: replace this with code that doesn't unnecessarily solve an ODE
 */

  class PersistentInitialParticleSize
    : public ParticleEquationBase
  {
  public:

    /**
     * \brief Construct a (non-changing) transport equation for the initial particle size
     *
     * \param solnVarName The name of the solution variable for this equation
     *
     * \param particlePositionTags A taglist containing the tags of x, y, and z
     *        particle coordinates. Those may be needed by some particle
     *        expressions that require particle operators
     *
     * \param particleSizeTag Particle size tag. May be needed by some expressions.
     *
     * \param particleEqsSpec the Uintah parser information for this Particle equation
     *
     * \param gc The GraphCategories object from Wasatch
     *
     */
    PersistentInitialParticleSize( const std::string&   solnVarName,
                                   const Expr::TagList& particlePositionTags,
                                   const Expr::Tag&     particleSizeTag,
                                   GraphCategories&     gc );

    ~PersistentInitialParticleSize();

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
     *
     *  \param bcHelper
     *  \param graphCat
     */
    void setup_boundary_conditions( WasatchBCHelper& bcHelper,
                                    GraphCategories& graphCat)
    {}

    /**
     *  \brief Set up the boundary condition on initial conditions evaluators for this
     *  Equation.
     *  \param graphHelper
     *  \param bcHelper
     */
    void apply_initial_boundary_conditions( const GraphHelper& graphHelper,
                                            WasatchBCHelper&   bcHelper )
    {}

    /**
     *  \brief Set up the boundary condition evaluators for this
     *  Equation.
     *  \param graphHelper
     *  \param bcHelper
     */
    void apply_boundary_conditions( const GraphHelper& graphHelper,
                                    WasatchBCHelper&   bcHelper )
    {}

  private:
    void setup();
    Expr::ExpressionID setup_rhs();
  };

  //------------------------------------------------------------------
  //------------------------------------------------------------------
  class PersistentInitialParticleMass : public ParticleEquationBase
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
    PersistentInitialParticleMass( const std::string& solnVarName,
                          const Expr::TagList&        particlePositionTags,
                          const Expr::Tag&            particleSizeTag,
                          Uintah::ProblemSpecP        particleEqsSpec,
                          GraphCategories&            gc );

    ~PersistentInitialParticleMass();

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
    const Expr::Tag pRhoTag_;
    void setup();
    const Expr::Tag get_initial_density_tag();
    Expr::ExpressionID setup_rhs();
  };

} // namespace WasatchCore

#endif /* PersistentParticleICs_h */
