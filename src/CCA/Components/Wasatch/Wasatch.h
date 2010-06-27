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


namespace Expr{ class ExpressionID; }

namespace Wasatch{

  class EqnTimestepAdaptorBase;
  class TimeStepper;

  /**
   *  \class  Wasatch
   *  \author James C. Sutherland
   *  \date   June, 2010
   *
   *  \brief Defines the Wasatch simulation component within Uintah.
   *
   *  Wasatch is designed to be a highly flexible simulation component
   *  for solution of transient PDEs. Like Uintah itself, Wasatch is
   *  based on graph theory.  However, it allows dynamic construction
   *  and "discovery" of graphs.  Each node in a graph defines its
   *  direct dependencies, and the graph is constructed recursively.
   *  This is in contrast to Uintah where the user must define the
   *  graph completely and statically.  This dynamic graph
   *  construction allows very easy definition of complex algorithms
   *  and results in highly localized changes when modifying models.
   *
   *  The Wasatch component is designed to insulate users from Uintah
   *  to a large degree.  If you find yourself using anything in the
   *  Uintah namespace, you are probably doing something wrong.
   *
   *  Most Wasatch related tools are in the Wasatch namespace.  We do
   *  not open up the Uintah namespace intentionally to make it clear
   *  when we are using Uintah functionality.
   *
   *  There is extensive usage of two external libraries: ExprLib and
   *  SpatialOps.  These have been developed by James S. Sutherland
   *  and support graph construction as well as field/operator
   *  operations, respectively.
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
     *  Required method from base class. Sets up the problem.
     */
    void problemSetup( const Uintah::ProblemSpecP& params, 
                       const Uintah::ProblemSpecP& restart_prob_spec, 
                       Uintah::GridP& grid,
                       Uintah::SimulationStateP& );

    void scheduleInitialize( const Uintah::LevelP& level,
                             Uintah::SchedulerP& sched );

    void scheduleComputeStableTimestep( const Uintah::LevelP& level,
                                        Uintah::SchedulerP& );

    void scheduleTimeAdvance( const Uintah::LevelP& level, 
                              Uintah::SchedulerP& );

  private:

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

    /** a convenience function */
    void create_timestepper_on_patches( const Uintah::PatchSet* const localPatches,
                                        const Uintah::MaterialSet* const materials,
                                        Uintah::SchedulerP& sched );

    // jcs this should disappear soon?
    void computeDelT( const Uintah::ProcessorGroup*,
                      const Uintah::PatchSubset* patches,
                      const Uintah::MaterialSubset* matls,
                      Uintah::DataWarehouse* old_dw,
                      Uintah::DataWarehouse* new_dw );

  };

} // namespace Wasatch

#endif
