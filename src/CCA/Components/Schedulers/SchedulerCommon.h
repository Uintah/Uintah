/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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


#ifndef UINTAH_HOMEBREW_SCHEDULERCOMMON_H
#define UINTAH_HOMEBREW_SCHEDULERCOMMON_H

#include <CCA/Components/Schedulers/OnDemandDataWarehouseP.h>
#include <CCA/Components/Schedulers/Relocate.h>
#include <CCA/Ports/Scheduler.h>

#include <CCA/Components/Schedulers/RuntimeStatsEnum.h>

#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/MaterialManagerP.h>
#include <Core/Parallel/UintahParallelComponent.h>
#include <Core/Util/Timers/Timers.hpp>

#include <iosfwd>
#include <map>
#include <set>
#include <vector>

namespace Uintah {

class ApplicationInterface;
class LoadBalancer;
class Output;
class DetailedTask;
class DetailedTasks;
class TaskGraph;
class LocallyComputedPatchVarMap;
  
using LabelMatlMap            = std::map<const VarLabel*, MaterialSubset*, VarLabel::Compare>;
using VarLabelMaterialListMap = std::map< std::string, std::list<int> >;
using ReductionTasksMap       = std::map<VarLabelMatl<Level, DataWarehouse>, Task*>;
using VarLabelList            = std::vector<std::vector<const VarLabel*> >;


/**************************************

CLASS
   SchedulerCommon


GENERAL INFORMATION

   SchedulerCommon.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)


KEYWORDS
   SchedulerCommon


DESCRIPTION

****************************************/

class SchedulerCommon : public Scheduler, public UintahParallelComponent {

  public:

    SchedulerCommon( const ProcessorGroup * myworld );

    virtual ~SchedulerCommon();

    virtual void setComponents(  UintahParallelComponent *comp );
    virtual void getComponents();
    virtual void releaseComponents();

    // TEMPORARY
    virtual ApplicationInterface *getApplication() { return m_application; };
  
    virtual void problemSetup( const ProblemSpecP     & prob_spec,
                               const MaterialManagerP & materialManager );

    virtual void doEmitTaskGraphDocs();

    virtual void checkMemoryUse( unsigned long & memUsed,
                                 unsigned long & highwater,
                                 unsigned long & maxMemUsed );

    // sbrk memory start location (for memory tracking)
    virtual void   setStartAddr( char * start ) { start_addr = start; }

    virtual char* getStartAddr() { return start_addr; }

    virtual void   resetMaxMemValue();

    // For calculating memory usage when sci-malloc is disabled...
    static char* start_addr;

    virtual void initialize( int numOldDW = 1, int numNewDW = 1 );

    virtual void setParentDWs( DataWarehouse * parent_old_dw, DataWarehouse * parent_new_dw );

    virtual void clearMappings();

    virtual void mapDataWarehouse( Task::WhichDW, int dwTag );

    void compile();

    /// For more complicated models
    virtual void addTaskGraph( tgType type, int index /* == -1 */ );

    virtual int getNumTaskGraphs() { return m_task_graphs.size(); }

    virtual void setNumTaskGraphs( const int num_task_graphs ) {
      ASSERT( num_task_graphs  >= 1 );
      m_num_task_graphs = num_task_graphs;
    }

    virtual void addTask(       Task        * task
                        , const PatchSet    * patches
                        , const MaterialSet * matls
                        , const int           tg_num = -1
                        );

    virtual bool useSmallMessages() { return m_use_small_messages; }

    /// Get all of the requires needed from the old data warehouse (carried forward).
    virtual const std::vector<const Task::Dependency*>&         getInitialRequires()     const { return m_init_requires; }
    virtual const std::set<const VarLabel*, VarLabel::Compare>& getInitialRequiredVars() const { return m_init_required_vars; }
    virtual const std::set<const VarLabel*, VarLabel::Compare>& getComputedVars()        const { return m_computed_vars; }

    virtual LoadBalancer * getLoadBalancer() { return m_loadBalancer; };

    virtual DataWarehouse* get_dw( int idx );

    virtual DataWarehouse* getLastDW();

    virtual void logMemoryUse();

    virtual void advanceDataWarehouse( const GridP & grid, bool initialization = false );

    virtual void fillDataWarehouses( const GridP & grid );

    virtual void replaceDataWarehouse( int           index
                                     , const GridP & grid
                                     , bool initialization = false
                                     );

    // Get the SuperPatch (set of connected patches making a larger rectangle)
    // for the given label and patch and find the largest extents encompassing
    // the expected ghost cells (requiredLow, requiredHigh) and the requested
    // ghost cells as well (requestedLow, requestedHigh) for each of the
    // patches.  Required and requested will besame if requestedNumGCells = 0.
    virtual const std::vector<const Patch*>*
    getSuperPatchExtents( const VarLabel         * label
                        ,       int                matlIndex
                        , const Patch            * patch
                        ,       Ghost::GhostType   requestedGType
                        ,       int                requestedNumGCells
                        ,       IntVector        & requiredLow
                        ,       IntVector        & requiredHigh
                        ,       IntVector        & requestedLow
                        ,       IntVector        & requestedHigh
                        ) const;

    // Makes and returns a map that maps strings to VarLabels of
    // that name and a list of material indices for which that
    // variable is valid (at least according to d_allcomps).
    virtual VarLabelMaterialMap* makeVarLabelMaterialMap();

    virtual bool isOldDW( int idx ) const;

    virtual bool isNewDW( int idx ) const;

    // Only called by the SimulationController, and only once, and only
    // if the simulation has been "restarted."
    virtual void setGeneration( int id ) { m_generation = id; }

    // This function will copy the data from the old grid to the new grid.
    // The PatchSubset structure will contain a patch on the new grid.
    void copyDataToNewGrid( const ProcessorGroup *
                          , const PatchSubset    * patches
                          , const MaterialSubset *
                          ,       DataWarehouse  * old_dw
                          ,       DataWarehouse  * new_dw
                          );

    //////////
    // Schedule particle relocation without the need to supply pre_relocation variables.
    //   Use with caution until, as this requires further testing (tsaad).
    virtual void scheduleParticleRelocation( const LevelP       & coarsestLevelwithParticles
                                           , const VarLabel     * posLabel
                                           , const VarLabelList & otherLabels
                                           , const MaterialSet  * matls
                                           );


    virtual void scheduleParticleRelocation( const LevelP       & level
                                           , const VarLabel     * old_posLabel
                                           , const VarLabelList & old_labels
                                           , const VarLabel     * new_posLabel
                                           , const VarLabelList & new_labels
                                           , const VarLabel     * particleIDLabel
                                           , const MaterialSet  * matls
                                           ,       int which
                                           );

    virtual void scheduleParticleRelocation( const LevelP       & coarsestLevelwithParticles
                                           , const VarLabel     * old_posLabel
                                           , const VarLabelList & old_labels
                                           , const VarLabel     * new_posLabel
                                           , const VarLabelList & new_labels
                                           , const VarLabel     * particleIDLabel
                                           , const MaterialSet  * matls
                                           );

    virtual void setPositionVar( const VarLabel* posLabel ) { m_reloc_new_pos_label = posLabel; }

    virtual void scheduleAndDoDataCopy( const GridP & grid );

    // Clear the recorded task monitoring attribute values.
    virtual void clearTaskMonitoring();

    // Schedule the recording of the task monitoring attribute values.
    virtual void scheduleTaskMonitoring( const LevelP& level );
    virtual void scheduleTaskMonitoring( const PatchSet* patches );

    // Record the task monitoring attribute values.
    virtual void recordTaskMonitoring(const ProcessorGroup*,  
                                      const PatchSubset* patches,
                                      const MaterialSubset* /*matls*/,
                                      DataWarehouse* old_dw,
                                      DataWarehouse* new_dw);
  
    //! override default behavior of copying, scrubbing, and such
    virtual void overrideVariableBehavior( const std::string & var
                                         ,       bool          treatAsOld
                                         ,       bool          copyData
                                         ,       bool          noScrub
                                         ,       bool          notCopyData   = false
                                         ,       bool          notCheckpoint = false
                                         );

    const std::set<std::string>& getNoScrubVars() { return m_no_scrub_vars;}

    const std::set<std::string>& getCopyDataVars() { return m_copy_data_vars;}

    const std::set<std::string>& getNotCopyDataVars() { return m_no_copy_data_vars;}

    virtual const std::set<std::string>& getNotCheckPointVars() const { return m_no_checkpoint_vars;}

    virtual bool useInternalDeps();

    int getMaxGhost() { return m_max_ghost_cells; }

    int getMaxDistalGhost() { return m_max_distal_ghost_cells; }

    int getMaxLevelOffset() { return m_max_level_offset; }

    bool isCopyDataTimestep() { return m_is_copy_data_timestep; }
      
    bool copyTimestep() { return (m_is_copy_data_timestep ||
                                  m_is_init_timestep); }

    void setInitTimestep( bool isInitTimestep ) { m_is_init_timestep = isInitTimestep; }

    void setRestartInitTimestep( bool isRestartInitTimestep ) { m_is_restart_init_timestep = isRestartInitTimestep; }

    virtual bool isRestartInitTimestep() { return m_is_restart_init_timestep; }

    const VarLabel* m_reloc_new_pos_label{nullptr};

    void setRuntimeStats( ReductionInfoMapper< RuntimeStatsEnum, double > *runtimeStats) { d_runtimeStats = runtimeStats; };

  protected:

    void finalizeTimestep();

    void makeTaskGraphDoc( const DetailedTasks * dtask, int rank = 0 );

    void emitNode( const DetailedTask * dtask
                 ,       double         start
                 ,       double         duration
                 ,       double         execution_duration
                 );

    void finalizeNodes( int process=0 );

    template< class T >
    void    printTrackedValues(       GridVariable<T> * var
                              , const IntVector       & start
                              , const IntVector       & end
                              );

    void printTrackedVars( DetailedTask* dtask, int when );

    virtual void verifyChecksum() = 0;

    enum {
        PRINT_BEFORE_COMM = 1
      , PRINT_BEFORE_EXEC = 2
      , PRINT_AFTER_EXEC  = 4
    };

    // Some places need to know if this is a copy data timestep or
    // a normal timestep.  (A copy data timestep is AMR's current 
    // method of getting data from an old to a new grid).
    bool                                m_is_copy_data_timestep{false};
    bool                                m_is_init_timestep{false};
    bool                                m_is_restart_init_timestep{false};
    int                                 m_current_task_graph{-1};
    int                                 m_generation{0};
    int                                 m_dwmap[Task::TotalDWs];

    ApplicationInterface * m_application  {nullptr};
    LoadBalancer         * m_loadBalancer {nullptr};
    Output               * m_output       {nullptr};
  
    MaterialManagerP                    m_materialManager{nullptr};
    std::vector<OnDemandDataWarehouseP> m_dws;
    std::vector<TaskGraph*>             m_task_graphs;

    //! These are so we can track certain variables over the taskgraph's execution.
    int                        m_tracking_vars_print_location{0};
    int                        m_tracking_patch_id{-1};
    int                        m_tracking_level{-1};
    double                     m_tracking_start_time{1.0};
    double                     m_tracking_end_time{0.0};
    IntVector                  m_tracking_start_index{IntVector(-9, -9, -9)};
    IntVector                  m_tracking_end_index{IntVector(-9, -9, -9)};
    std::vector<std::string>   m_tracking_vars;
    std::vector<std::string>   m_tracking_tasks;
    std::vector<Task::WhichDW> m_tracking_dws;

    // Optional task monitoring.
    MaterialSubset* m_dummy_matl{0};

    bool m_monitoring{false};          // Monitoring on/off.
    bool m_monitoring_per_cell{false}; // Record the task runtime attributes
                                       // on a per cell basis rather than a
                                       // per patch basis.
    // Maps for the global or local tasks to be monitored.
    std::map<std::string, const VarLabel *>       m_monitoring_tasks[2];
    std::map<std::string, std::map<int, double> > m_monitoring_values[2];

    // Method for summing up the task contributions.
    void sumTaskMonitoringValues(DetailedTask * dtask);

    // so we can manually copy vars between AMR levels
    std::set<std::string> m_copy_data_vars;

    // ignore copying these vars between AMR levels
    std::set<std::string> m_no_copy_data_vars;

    // vars manually set not to scrub (normally when needed between a normal taskgraph and the regridding phase)
    std::set<std::string> m_no_scrub_vars;

    // treat variable as an "old" var - will be checkpointed, copied, and only scrubbed from an OldDW
    std::set<std::string> m_treat_as_old_vars;

    // do not checkpoint these variables
    std::set<std::string> m_no_checkpoint_vars;

    ReductionInfoMapper< RuntimeStatsEnum, double > *d_runtimeStats{nullptr};

  private:

    // helper method for primary addTask()
    void addTask(       std::shared_ptr<Task>
                , const PatchSet    * patches
                , const MaterialSet * matls
                , const int           tg_num
                );

    // eliminate copy, assignment and move
    SchedulerCommon( const SchedulerCommon & )            = delete;
    SchedulerCommon& operator=( const SchedulerCommon & ) = delete;
    SchedulerCommon( SchedulerCommon && )                 = delete;
    SchedulerCommon& operator=( SchedulerCommon && )      = delete;

    ProblemSpecP                m_graph_doc{nullptr};
    ProblemSpecP                m_graph_nodes{nullptr};

    std::ofstream*              m_mem_logfile{nullptr};

    Relocate                    m_relocate_1;
    Relocate                    m_relocate_2;

    // whether or not to send a small message (takes more work to organize)
    // or a larger one (more communication time)
    bool m_use_small_messages{true};
    bool m_emit_task_graph{false};
    int  m_num_task_graphs{1};
    int  m_num_tasks{0};
    int  m_num_old_dws{0};

    //! These are to store which vars we have to copy to the new grid
    //! in a copy data task.  Set in scheduleDataCopy and used in
    //! copyDataToNewGrid.
    std::vector<LabelMatlMap>   m_label_matls;

    ReductionTasksMap           m_reduction_tasks;

    LocallyComputedPatchVarMap* m_locallyComputedPatchVarMap{nullptr};

    //! set in addTask - can be used until initialize is called...
    std::vector<const Task::Dependency*>         m_init_requires;
    std::set<const VarLabel*, VarLabel::Compare> m_init_required_vars;
    std::set<const VarLabel*, VarLabel::Compare> m_computed_vars;

    // Maximum memory used as sampled across a given timestep.
    unsigned long               m_max_mem_used{0};

    // max ghost cells of standard tasks - will be used for loadbalancer to create neighborhood
    int m_max_ghost_cells{0};

    // max ghost cells for tasks with distal requirements (e.g. RMCRT) - will be used for loadbalancer to create neighborhood
    int m_max_distal_ghost_cells{0};

    // max level offset of all tasks - will be used for loadbalancer to create neighborhood
    int m_max_level_offset{0};

    // task-graph needs access to reduction task map, etc
    friend class TaskGraph;

  };
} // End namespace Uintah

#endif
