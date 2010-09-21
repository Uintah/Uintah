/*

  The MIT License

  Copyright (c) 1997-2010 Center for the Simulation of Accidental Fires and 
  Explosions (CSAFE), and  Institute for Clean and Secure Energy (ICSE), 
  University of Utah.

  License for the specific language governing rights and limitations under
  Permission is hereby granted, free of charge, to any person obtaining a 
  copy of this software and associated documentation files (the "Software"),
  to deal in the Software without restriction, including without limitation 
  the rights to use, copy, modify, merge, publish, distribute, sublicense, 
  and/or sell copies of the Software, and to permit persons to whom the 
  Software is furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included 
  in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
  DEALINGS IN THE SOFTWARE.

*/


/**
   \file
 
   \mainpage
 
   \author James C. Sutherland
   \date June, 2010
 
   Wasatch is a Uintah component that interfaces to the Expression
   library and SpatialOps library to facilitate more rapid
   development of time-dependent PDE solvers.

   Wasatch is designed to be a highly flexible simulation component
   for solution of transient PDEs. Like Uintah itself, Wasatch is
   based on graph theory.  However, it allows dynamic construction and
   "discovery" of graphs.  Each node in a graph defines its direct
   dependencies, and the graph is constructed recursively.  This is in
   contrast to Uintah where the user must define the graph completely
   and statically.  This dynamic graph construction allows very easy
   definition of complex algorithms and results in highly localized
   changes when modifying models.

   The Wasatch component is designed to insulate users from Uintah to
   a large degree.  If you find yourself using anything in the Uintah
   namespace, you are probably doing something wrong.

   Most Wasatch related tools are in the Wasatch namespace.  We do not
   open up the Uintah namespace intentionally to make it clear when we
   are using Uintah functionality.

   There is extensive usage of two external libraries: ExprLib and
   SpatialOps.  These have been developed by James C. Sutherland and
   support graph construction as well as field/operator operations,
   respectively.
 
 
   \par Development in Wasatch
 
   Most developers will need to focus on the Expression library and
   not on Wasatch directly.  Typically, development should focus on
   creating Expressions that encapsulate a concise bit of
   functionality.  These should be written to be as generic as
   possible.  Notably:

    - Expressions should NOT be cognizant of ANYTHING within Uintah.
      This includes parser objects, exception handling, etc.  They are
      meant to be portable.

    - Exressions should use Expr::Tag objects to define dependencies,
      and should NOT use hard-coded strings for naming of any
      variables.  This also destroys portability and can lead to
      fragile code dependencies.

   For developers that need to deal with details in Wasatch, it
   interfaces to Uintah through several key areas:
 
    - Parsing.  Uintah has its own XML parser (Uintah::ProblemSpec).
      Developers should work to minimize the intrusion of parser
      objects into application code as much as possible, as it
      destroys portability of the code.
 
    - Task creation.  Uintah tasks are typically created by wrapping
      Expression tree objects using the Wasatch::TaskInterface class.

  \defgroup WasatchExpressions	Expressions in Wasatch
  \defgroup WasatchFields	Fields and field tools
  \defgroup WasatchOperators	Operators
  \defgroup WasatchCore		Wasatch Core
  \defgroup WasatchGraph	Wasatch Graph
  \defgroup WasatchParser	Wasatch Parsers
*/

#ifndef Packages_Uintah_CCA_Components_Examples_Wasatch_h
#define Packages_Uintah_CCA_Components_Examples_Wasatch_h

#include <list>
#include <map>
#include <string>

//-- Uintah Framework Includes --//
#include <Core/Parallel/UintahParallelComponent.h>
#include <CCA/Ports/SimulationInterface.h>
#include <Core/Grid/SimulationStateP.h>

//-- Wasatch includes --//
#include "PatchInfo.h"
#include "GraphHelperTools.h"
#include "FieldTypes.h"

namespace Expr{ class ExpressionID; }

namespace Uintah{ class Task; }

namespace Wasatch{

  class EqnTimestepAdaptorBase;
  class TimeStepper;

  /**
   *  \ingroup WasatchCore
   *  \class  Wasatch
   *  \author James C. Sutherland
   *  \date   June, 2010
   *
   *  \brief Defines the Wasatch simulation component within Uintah.
   *
   * \todo Create a tree on the initialization that duplicates the
   *       "advance solution" tree so that we have everything required
   *       for output at the initial time step.
   */
  class UINTAHSHARE Wasatch :
    public Uintah::UintahParallelComponent,
    public Uintah::SimulationInterface
  {
  public:

    Wasatch( const Uintah::ProcessorGroup* myworld );

    ~Wasatch();

    /**
     *  \brief Required method from base class. Sets up the problem.
     *
     *  Performs the following tasks:
     *  <ul>
     *  <li> Registers the material(s) for use in solving this problem.
     *  <li> Creates any expressions triggered directly from the input
     *       file.  These are typically associated with calculating
     *       initial conditions.
     *  <li> Sets up property evaluation.
     *  <li> Creates a time integrator for solving the set of PDEs.
     *  <li> Parses and constructs transport equations.  This, in
     *       turn, registers relevant expressions required to solve
     *       each transport equation and also plugs them into the time
     *       integrator.
     *  </ul>
     */
    void problemSetup( const Uintah::ProblemSpecP& params, 
                       const Uintah::ProblemSpecP& restart_prob_spec, 
                       Uintah::GridP& grid,
                       Uintah::SimulationStateP& );

    /**
     *  \brief Set up initial condition task(s)
     *
     *  Performs the following:
     *  <ul>
     *  <li> Set up spatial operators associated with each patch and
     *       store them for use by expressions later on.
     *  <li> Set up the Uintah::Task(s) that will set the initial conditions.
     *  </ul>
     */
    void scheduleInitialize( const Uintah::LevelP& level,
                             Uintah::SchedulerP& sched );

    /**
     *  \brief Set up the Uintah::Task that will calculate the timestep.
     */
    void scheduleComputeStableTimestep( const Uintah::LevelP& level,
                                        Uintah::SchedulerP& );

    /**
     *  <ul>
     *  <li> Plug each transport equation that has been set up into
     *       the time stepper and generate the full ExpressionTree
     *       that will calculate the RHS for all of the equations.
     *  <li> Create the Uintah::Task(s) associated with the
     *       ExpressionTree to advance the solution for the PDE system
     *       forward one time step.
     *  </ul>
     */
    void scheduleTimeAdvance( const Uintah::LevelP& level, 
                              Uintah::SchedulerP& );

    /**
     *  \brief specify that we want coordinate information
     *  \param dir the coordinate (XDIR -> x coord)
     *
     *  This method can be called to request the coordinate for the
     *  given field type. For example,
     *
     *  \code requires_coordinate<SVolField>(XDIR) \endcode
     *
     *  results in the X-coordinate of the scalar volumes being
     *  populated as a field.  These can subsequently be used in a
     *  graph.
     *
     *  Note that coordinates have very specific names that should be
     *  used in the input file.  See Wasatch.cc and specifically the
     *  implementation of register_coord_fields for more information.
     */
    template<typename FieldT> void requires_coordinate( const Direction dir );

  private:

    bool needCoords_,
      xSVolCoord_, ySVolCoord_,zSVolCoord_,
      xXVolCoord_, yXVolCoord_,zXVolCoord_,
      xYVolCoord_, yYVolCoord_,zYVolCoord_,
      xZVolCoord_, yZVolCoord_,zZVolCoord_;

    Uintah::VarLabel *xSVol_, *ySVol_, *zSVol_;
    Uintah::VarLabel *xXVol_, *yXVol_, *zXVol_;
    Uintah::VarLabel *xYVol_, *yYVol_, *zYVol_;
    Uintah::VarLabel *xZVol_, *yZVol_, *zZVol_;

    Uintah::SimulationStateP sharedState_; ///< access to some common things like the current timestep.

    /**
     *  a container of information for constructing ExprLib graphs.
     *  These are then wrapped as Wasath::TaskInterface objects and
     *  plugged into Uintah.
     */
    GraphCategories graphCategories_;

    PatchInfoMap patchInfoMap_; ///< Information about each patch

    TimeStepper* timeStepper_;  ///< The TimeStepper used to advance equations registered here.

    typedef std::vector<EqnTimestepAdaptorBase*> EquationAdaptors;
    EquationAdaptors adaptors_;

    Wasatch( const Wasatch& ); // disallow copying
    Wasatch& operator=( const Wasatch& ); // disallow assignment

    /** \brief a convenience function */
    void create_timestepper_on_patches( const Uintah::PatchSet* const localPatches,
                                        const Uintah::MaterialSet* const materials,
                                        Uintah::SchedulerP& sched );

    // jcs this should disappear soon?
    void computeDelT( const Uintah::ProcessorGroup*,
                      const Uintah::PatchSubset* patches,
                      const Uintah::MaterialSubset* matls,
                      Uintah::DataWarehouse* old_dw,
                      Uintah::DataWarehouse* new_dw );

    /** \brief sets the requested grid variables - callback for an initialization task */
    void set_grid_variables( const Uintah::ProcessorGroup* const pg,
                             const Uintah::PatchSubset* const patches,
                             const Uintah::MaterialSubset* const materials,
                             Uintah::DataWarehouse* const oldDW,
                             Uintah::DataWarehouse* const newDW );

    /** \brief registers requested coordinate fields */
    void register_coord_fields( Uintah::Task* const task,
                                const Uintah::PatchSet* const ps,
                                const Uintah::MaterialSet* const mss );

  };


  template<> void Wasatch::requires_coordinate<SVolField>( const Direction dir )
  {
    needCoords_ = true;
    switch (dir) {
      case XDIR : xSVolCoord_=true; break;
      case YDIR : ySVolCoord_=true; break;
      case ZDIR : zSVolCoord_=true; break;
      }
  }
  template<> void Wasatch::requires_coordinate<XVolField>( const Direction dir )
  {
    needCoords_ = true;
    switch (dir) {
      case XDIR : xXVolCoord_=true; break;
      case YDIR : yXVolCoord_=true; break;
      case ZDIR : zXVolCoord_=true; break;
      }
  }
  template<> void Wasatch::requires_coordinate<YVolField>( const Direction dir )
  {
    needCoords_ = true;
    switch (dir) {
      case XDIR : xYVolCoord_=true; break;
      case YDIR : yYVolCoord_=true; break;
      case ZDIR : zYVolCoord_=true; break;
      }
  }
  template<> void Wasatch::requires_coordinate<ZVolField>( const Direction dir )
  {
    needCoords_ = true;
    switch (dir) {
      case XDIR : xZVolCoord_=true; break;
      case YDIR : yZVolCoord_=true; break;
      case ZDIR : zZVolCoord_=true; break;
      }
  }

} // namespace Wasatch

#endif
