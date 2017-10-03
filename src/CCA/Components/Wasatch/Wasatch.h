/*
 * The MIT License
 *
 * Copyright (c) 2010-2017 The University of Utah
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

/**
   \file

   \mainpage

   \author James C. Sutherland, Tony Saad
   \date June, 2010 - present

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
   SpatialOps.  These have been developed by James C. Sutherland and his
   collaborators and support graph construction as well as field/operator
   operations, respectively.


   \par Development in Wasatch

   Most developers will need to focus on writing Expressions and not on
   Wasatch internals directly. Typically, development should focus on
   creating Expressions that encapsulate a concise bit of functionality.
   These should be written to be as generic as possible. Notably:

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
      Expression tree objects using the WasatchCore::TaskInterface class.
      If you find yourself writing a Uintah::Task directly in Wasatch,
      you are probably doing something wrong.

  \defgroup Expressions		Expressions in Wasatch
  \defgroup WasatchFields	Fields and field tools
  \defgroup WasatchOperators	Operators
  \defgroup WasatchCore		Wasatch Core
  \defgroup WasatchGraph	Wasatch Graph
  \defgroup WasatchParser	Wasatch Parsers
  \defgroup WasatchParticles    Particle Transport

*/

#ifndef Packages_Uintah_CCA_Components_Examples_Wasatch_h
#define Packages_Uintah_CCA_Components_Examples_Wasatch_h

#include <list>
#include <map>
#include <string>
#include <set>

//-- Uintah Framework Includes --//
#include <Core/Parallel/UintahParallelComponent.h>
#include <CCA/Ports/SimulationInterface.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Ports/SolverInterface.h>

//-- Wasatch includes --//
#include "PatchInfo.h"
#include "GraphHelperTools.h"
#include "FieldTypes.h"
#include "BCHelper.h"
#include "WasatchBCHelper.h"
#include "TimeIntegratorTools.h"
//-- ExprLib Includes --//
#include <expression/ExpressionFactory.h>
#include <expression/dualtime/FixedPointBDFDualTimeIntegrator.h>
#include <expression/dualtime/VariableImplicitBDFDualTimeIntegrator.h>

namespace Expr{
  class ExpressionID;
}

namespace Uintah {
  class Task;
  class Ray;
  class ParticlesHelper;
}

  class CellType;

namespace WasatchCore{
  void force_expressions_on_graph( Expr::TagList& exprTagList,
                                   GraphHelper* const graphHelper );
  
  void check_periodicity_extra_cells( const Uintah::ProblemSpecP& params,
                                      Uintah::IntVector& extraCells,
                                      bool& isPeriodic);
  
  class EqnTimestepAdaptorBase;
  class TimeStepper;
  class TaskInterface;
  class WasatchParticlesHelper;
  struct TimeIntegrator;
  
  enum FlowTreatment
  {
    INCOMPRESSIBLE,
    LOWMACH,
    COMPRESSIBLE
  };

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
   *
   * \todo Allow other "root" tasks to be registerd that are not
   *       necessarily associated with the RHS of a governing
   *       equation.
   *
   */
  class Wasatch :
    public Uintah::UintahParallelComponent,
    public Uintah::SimulationInterface
  {

  public:

    typedef std::vector<EqnTimestepAdaptorBase*> EquationAdaptors;
    typedef std::map< int, WasatchBCHelper* > BCHelperMapT; //<<< LevelID, BCHelper >>>
    // we need a dual time integrator per patch since each of the RHS trees will need a dual time integrator for the patch it is working on
    typedef std::map< int, Expr::DualTime::BDFDualTimeIntegrator* > DTIntegratorMapT; //<<< PatchID, DualTimeIntegrator >>>
    
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
     *  \brief Set up initial condition task(s)
     *
     *  Performs the following:
     *  <ul>
     *  <li> Set up spatial operators associated with each patch and
     *       store them for use by expressions later on.
     *  <li> Set up the Uintah::Task(s) that will set the initial conditions.
     *  </ul>
     */
    void scheduleComputeDualTimeResidual( const Uintah::LevelP& level,
                                          Uintah::SchedulerP& sched );

    /**
     *  \brief Set up initial condition task(s) on a restart
     */
    void scheduleRestartInitialize( const Uintah::LevelP& level,
                                    Uintah::SchedulerP& sched );

    /**
     *  \brief Set up things that need to be done on a restart
     */
    void restartInitialize();

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

    void scheduleUpdateCurrentTime( const Uintah::LevelP& level,
                             Uintah::SchedulerP&, const int rkStage );
    
    void scheduleSetInitialTime( const Uintah::LevelP& level,
                                   Uintah::SchedulerP& );

    void preGridProblemSetup(const Uintah::ProblemSpecP& params,
                             Uintah::GridP& grid,
                             Uintah::SimulationStateP& state);

    //__________________________________
    //  AMR
    virtual void scheduleCoarsen(const Uintah::LevelP& coarseLevel,
                                 Uintah::SchedulerP& sched);

    virtual void scheduleRefineInterface(const Uintah::LevelP& /*fineLevel*/,
                                         Uintah::SchedulerP& /*scheduler*/,
                                         bool, bool);

    virtual int computeTaskGraphIndex();

    const EquationAdaptors& equation_adaptors() const{ return adaptors_; }
    GraphCategories& graph_categories(){ return graphCategories_; }
    Expr::ExpressionFactory* solution_factory(){ return (graphCategories_[ADVANCE_SOLUTION])->exprFactory; }
    void disable_timestepper_creation(){ buildTimeIntegrator_ = false; }
    void disable_wasatch_material(){ buildWasatchMaterial_ = false; }
    const PatchInfoMap& patch_info_map() const{ return patchInfoMap_; }
    std::list< const TaskInterface* >& task_interface_list(){ return taskInterfaceList_; }
    const std::set<std::string>& persistent_fields() const{ return persistentFields_; }
    std::set<std::string>& persistent_fields(){ return persistentFields_; }
    
    void make_field_persistent( const std::string& fieldName ){
      if( persistentFields_.find(fieldName) == persistentFields_.end() ) persistentFields_.insert(fieldName);
    }
    
    void set_wasatch_materials( const Uintah::MaterialSet* const materials ) { materials_ = materials; }
    const Uintah::MaterialSet* get_wasatch_materials() const{ return materials_; }
    const Uintah::ProblemSpecP get_wasatch_spec(){return wasatchSpec_;}

    static void set_flow_treatment( const FlowTreatment treat ){ flowTreatment_ = treat; }
    static void set_flow_treatment( const std::string treat )
    {
      flowTreatment_ = INCOMPRESSIBLE;
      if      ( treat == "LOWMACH"      ) flowTreatment_ = LOWMACH;
      else if ( treat == "COMPRESSIBLE" ) flowTreatment_ = COMPRESSIBLE;
    }
    static FlowTreatment flow_treatment(){return flowTreatment_;}

    static void need_pressure_solve( const bool needPSolve ){ needPressureSolve_ = needPSolve; }
    static bool need_pressure_solve(){ return needPressureSolve_; }

    static void has_dual_time( const bool hasDualTime ){ hasDualTime_ = hasDualTime; }
    static bool has_dual_time(){ return hasDualTime_; }

  private:
    bool buildTimeIntegrator_;   ///< used for Wasatch-Arches coupling
    bool buildWasatchMaterial_;  ///< used for Wasatch-Arches coupling
    bool isRestarting_;
    int  nRKStages_;
    bool isPeriodic_;
    bool doRadiation_;
    int  radCalcFrequency_;
    bool doParticles_;
    unsigned long totalDualTimeIterations_;
    
    TimeIntegrator timeIntegrator_;
    
    DTIntegratorMapT dualTimeIntegrators_;
    
    std::set<std::string> persistentFields_;   ///< prevent the ExpressionTree from reclaiming memory on these fields.
    Uintah::SimulationStateP sharedState_; ///< access to some common things like the current timestep.
    const Uintah::MaterialSet* materials_;
    Uintah::ProblemSpecP wasatchSpec_;

    WasatchParticlesHelper* particlesHelper_;
    
    BCFunctorMap bcFunctorMap_;
    BCHelperMapT bcHelperMap_;



    /**
     *  a container of information for constructing ExprLib graphs.
     *  These are then wrapped as WasatchCore::TaskInterface objects and
     *  plugged into Uintah.
     */
    GraphCategories graphCategories_;

    PatchInfoMap patchInfoMap_; ///< Information about each patch

    TimeStepper* timeStepper_;  ///< The TimeStepper used to advance equations registered here.

    Uintah::SolverInterface* linSolver_;

    Uintah::Ray* rmcrt_; // RMCRT solver. needed to pass along to other tasks and expressions...
    CellType* cellType_;
    
    EquationAdaptors adaptors_;  ///< set of transport equations to be solved

    std::list< const TaskInterface*  > taskInterfaceList_;
    std::map< int, const Uintah::PatchSet* > patchesForOperators_;

    static FlowTreatment flowTreatment_;
    static bool needPressureSolve_;
    static bool hasDualTime_;
    
    Uintah::SchedulerP subsched_; // needed for dualtime
    bool dualTime_;
    bool compileDualTimeSched_;
    
    Uintah::VarLabel *dtLabel_, *tLabel_, *tStepLabel_, *rkStageLabel_;
    
    Wasatch( const Wasatch& );            // disallow copying
    Wasatch& operator=( const Wasatch& ); // disallow assignment
    
    /** \brief a convenience function */
    void create_timestepper_on_patches( const Uintah::PatchSet* const localPatches,
                                        const Uintah::MaterialSet* const materials,
                                        const Uintah::LevelP& level,
                                        Uintah::SchedulerP& sched,
                                        const int RKStage );

    void create_dual_timestepper_on_patches( const Uintah::PatchSet* const localPatches,
                                             const Uintah::MaterialSet* const materials,
                                             const Uintah::LevelP& level,
                                             Uintah::SchedulerP& sched );

    // jcs this should disappear soon?
    void computeDelT( const Uintah::ProcessorGroup*,
                      const Uintah::PatchSubset* patches,
                      const Uintah::MaterialSubset* matls,
                      Uintah::DataWarehouse* old_dw,
                      Uintah::DataWarehouse* new_dw );


    void setup_patchinfo_map( const Uintah::LevelP& level,
                              Uintah::SchedulerP& sched );

    void dualTimeAdvance(const Uintah::ProcessorGroup* pg,
                         const Uintah::PatchSubset* patches,
                         const Uintah::MaterialSubset* matls,
                         Uintah::DataWarehouse* old_dw,
                         Uintah::DataWarehouse* new_dw,
                         Uintah::LevelP level, Uintah::Scheduler* sched,
                         Expr::ExpressionFactory* const factory);
    

    void computeDualTimeResidual(const Uintah::ProcessorGroup* pg,
                         const Uintah::PatchSubset* patches,
                         const Uintah::MaterialSubset* matls,
                         Uintah::DataWarehouse* old_dw,
                         Uintah::DataWarehouse* new_dw,
                         Uintah::LevelP level, Uintah::Scheduler* sched);

    void
    update_current_time( const Uintah::ProcessorGroup* const pg,
                        const Uintah::PatchSubset* const patches,
                        const Uintah::MaterialSubset* const materials,
                        Uintah::DataWarehouse* const oldDW,
                        Uintah::DataWarehouse* const newDW,
                        const int rkStage );

    void
    set_initial_time( const Uintah::ProcessorGroup* const pg,
                        const Uintah::PatchSubset* const patches,
                        const Uintah::MaterialSubset* const materials,
                        Uintah::DataWarehouse* const oldDW,
                        Uintah::DataWarehouse* const newDW );

    enum PatchsetSelector{
      USE_FOR_TASKS,
      USE_FOR_OPERATORS
    };

    /** \brief obtain the set of patches to operate on */
    const Uintah::PatchSet* get_patchset( const PatchsetSelector,
                                          const Uintah::LevelP& level,
                                          Uintah::SchedulerP& sched );
  };

} // namespace WasatchCore

#endif
