/*
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
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


#ifndef CCA_COMPONENTS_SCHEDULERS_SCHEDULERCOMMON_H
#define CCA_COMPONENTS_SCHEDULERS_SCHEDULERCOMMON_H

#include <CCA/Components/Schedulers/OnDemandDataWarehouseP.h>
#include <CCA/Components/Schedulers/Relocate.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Parallel/UintahParallelComponent.h>
#include <Core/Util/DebugStream.h>

#include   <iosfwd>
#include   <set>

namespace Uintah {


class Output;
class DetailedTask;
class DetailedTasks;
class TaskGraph;
class LocallyComputedPatchVarMap;

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

  
WARNING
  
****************************************/

class SchedulerCommon : public Scheduler, public UintahParallelComponent {

  public:

    SchedulerCommon( const ProcessorGroup * myworld, const Output * oport );

    virtual ~SchedulerCommon();

    virtual void problemSetup( const ProblemSpecP& prob_spec, SimulationStateP& state );

    virtual void doEmitTaskGraphDocs();

    virtual void checkMemoryUse( unsigned long& memuse, unsigned long & highwater, unsigned long& maxMemUse );

    // sbrk memory start location (for memory tracking)
    virtual void   setStartAddr( char * start ) { start_addr = start; }

    virtual char* getStartAddr() { return start_addr; }

    virtual void   resetMaxMemValue();

    // For calculating memory usage when sci-malloc is disabled...
    static char* start_addr;

    virtual void initialize( int numOldDW = 1, int numNewDW = 1 );

    virtual void setParentDWs( DataWarehouse* parent_old_dw, DataWarehouse* parent_new_dw );

    virtual void clearMappings();

    virtual void mapDataWarehouse( Task::WhichDW, int dwTag );

    void compile();

    /// For more complicated models 
    virtual void addTaskGraph(tgType type);

    virtual int getNumTaskGraphs() { return m_graphs.size(); }

    virtual void addTask( Task* t, const PatchSet*, const MaterialSet* );

    virtual bool useSmallMessages() { return m_use_small_messages; }

    /// Get all of the requires needed from the old data warehouse (carried forward).
    virtual const std::vector<const Task::Dependency*>&         getInitialRequires() const     { return m_init_requires; }

    virtual const std::set<const VarLabel*, VarLabel::Compare>& getInitialRequiredVars() const { return m_init_requires_vars; }

    virtual const std::set<const VarLabel*, VarLabel::Compare>& getComputedVars() const        { return m_computed_vars; }

    virtual LoadBalancer* getLoadBalancer();

    virtual void releaseLoadBalancer();
       
    virtual DataWarehouse* get_dw( int idx );

    virtual DataWarehouse* getLastDW( void );

    virtual void logMemoryUse();
      
    virtual void advanceDataWarehouse( const GridP& grid, bool initialization=false );

    virtual void fillDataWarehouses( const GridP& grid );

    virtual void replaceDataWarehouse( int index, const GridP& grid, bool initialization=false );

    virtual void setRestartable( bool restartable );

    // Get the expected extents that may be needed for a particular variable
    // on a particular patch (which should include expected ghost cells.
    //virtual void
    //getExpectedExtents(const VarLabel* label, const Patch* patch,
    //	       IntVector& lowIndex, IntVector& highIndex) const;

    // Get the SuperPatch (set of connected patches making a larger rectangle)
    // for the given label and patch and find the largest extents encompassing
    // the expected ghost cells (requiredLow, requiredHigh) and the requested
    // ghost cells as well (requestedLow, requestedHigh) for each of the
    // patches.  Required and requested will be the same if requestedNumGCells = 0.
    virtual const std::vector<const Patch*>*
    getSuperPatchExtents(const VarLabel* label,
                               int matlIndex,
                         const Patch* patch,
                               Ghost::GhostType requestedGType,
                         int requestedNumGCells,
                               IntVector& requiredLow,
                               IntVector& requiredHigh,
                               IntVector& requestedLow,
                               IntVector& requestedHigh) const;

    // Makes and returns a map that maps strings to VarLabels of
    // that name and a list of material indices for which that
    // variable is valid (at least according to d_allcomps).
    typedef std::map< std::string, std::list<int> > VarLabelMaterialMap;
    virtual VarLabelMaterialMap* makeVarLabelMaterialMap();

    virtual bool isOldDW(int idx) const;

    virtual bool isNewDW(int idx) const;

    // Only called by the SimulationController, and only once, and only
    // if the simulation has been "restarted."
    virtual void setGeneration( int id ) { m_generation = id; }

    // This function will copy the data from the old grid to the new grid.
    // The PatchSubset structure will contain a patch on the new grid.
    void copyDataToNewGrid(const ProcessorGroup*,
                           const PatchSubset*     patches,
                           const MaterialSubset*,
                                 DataWarehouse*  old_dw,
                                 DataWarehouse*  new_dw);

    //////////
    // Schedule particle relocation without the need to supply pre_relocation variables. Use with
    // caution until as this requires further testing (tsaad).
    virtual void scheduleParticleRelocation(const LevelP& coarsestLevelwithParticles,
                                            const VarLabel* posLabel,
                                            const std::vector<std::vector<const VarLabel*> >& otherLabels,
                                            const MaterialSet* matls);

    
    virtual void scheduleParticleRelocation(const LevelP& level,
                                            const VarLabel* old_posLabel,
                                            const std::vector<std::vector<const VarLabel*> >& old_labels,
                                            const VarLabel* new_posLabel,
                                            const std::vector<std::vector<const VarLabel*> >& new_labels,
                                            const VarLabel* particleIDLabel,
                                            const MaterialSet* matls,
                                                  int which);

    virtual void scheduleParticleRelocation(const LevelP& coarsestLevelwithParticles,
                                            const VarLabel* old_posLabel,
                                            const std::vector<std::vector<const VarLabel*> >& old_labels,
                                            const VarLabel* new_posLabel,
                                            const std::vector<std::vector<const VarLabel*> >& new_labels,
                                            const VarLabel* particleIDLabel,
                                            const MaterialSet* matls);
    
    virtual void setPositionVar(const VarLabel* posLabel) { m_reloc_new_posLabel = posLabel; }

    virtual void scheduleAndDoDataCopy( const GridP& grid, SimulationInterface* sim );

    //! override default behavior of copying, scrubbing, and such
    virtual void overrideVariableBehavior(const std::string & var,
                                          bool treatAsOld,
                                          bool copyData,
                                          bool noScrub,
                                          bool notCopyData = false,
                                          bool notCheckpoint = false);

    const std::set<std::string>& getNoScrubVars()     { return m_no_scrub_vars;}

    const std::set<std::string>& getCopyDataVars()    { return m_copy_data_vars;}

    const std::set<std::string>& getNotCopyDataVars() { return m_not_copy_data_vars;}

    virtual const std::set<std::string>& getNotCheckPointVars() const { return m_not_checkpoint_vars;}

    virtual bool useInternalDeps();
    
    const VarLabel* m_reloc_new_posLabel;

    // TODO replace after Mira DDT problem is debugged (APH - 03/24/15)
    int getMaxGhost()       {return maxGhost;}
    int getMaxLevelOffset() {return maxLevelOffset;}
//    const std::map<int, int>& getMaxGhostCells() { return maxGhostCells; }
//    const std::map<int, int>& getMaxLevelOffsets() { return maxLevelOffsets; }

    bool isCopyDataTimestep() { return m_shared_state->isCopyDataTimestep() || m_is_init_timestep; }

    void setInitTimestep( bool isInitTimestep ) { m_is_init_timestep = isInitTimestep; }

    void setRestartInitTimestep( bool isRestartInitTimestep ) { m_is_restart_init_timestep = isRestartInitTimestep; }

    virtual bool isRestartInitTimestep() { return m_is_restart_init_timestep; }


    typedef std::map<VarLabelMatl<Level>, Task*> ReductionTasksMap;
    ReductionTasksMap reductionTasks;

  protected:

    void finalizeTimestep();
    
    void makeTaskGraphDoc( const DetailedTasks* dt, int rank = 0 );

    void emitNode( const DetailedTask* dt, double start, double duration, double execution_duration );

    void finalizeNodes( int process=0 );

    enum { PRINT_BEFORE_COMM = 1, PRINT_BEFORE_EXEC = 2, PRINT_AFTER_EXEC = 4 };

    void printTrackedVars(DetailedTask* dt, int when);
    
    bool m_is_init_timestep;
    bool m_is_restart_init_timestep;
   
    /**
    * output the task name and the level it's executing on.
    * and each of the patches
    */
    void printTask( std::ostream& out, DetailedTask* task );
    
    /**
    *  Output the task name and the level it's executing on
    *  only first patch of that level
    */
    void printTaskLevels( const ProcessorGroup* d_myworld,
                                DebugStream& out,
                                DetailedTask* task );
    
    
    virtual void verifyChecksum() = 0;

    std::vector<TaskGraph*>             m_graphs;
    int                                 m_currentTG;
    int                                 m_num_tasks;
    int                                 m_generation;
    SimulationStateP                    m_shared_state;
    std::vector<OnDemandDataWarehouseP> m_dws;
    int                                 m_num_old_dws;
    int                                 m_dw_map[Task::TotalDWs];
    const Output*                       m_output_port;
    bool                                m_restartable;

    //! These are so we can track certain variables over the taskgraph's execution.
    std::vector<std::string>   m_tracking_vars;
    std::vector<std::string>   m_tracking_tasks;
    std::vector<Task::WhichDW> m_tracking_dws;
    int                        m_tracking_vars_print_location;
    int                        m_tracking_patch_id;
    double                     m_tracking_start_time;
    double                     m_tracking_end_time;
    int                        m_tracking_level;
    IntVector                  m_tracking_start_index;
    IntVector                  m_tracking_end_index;
    int                        m_num_particle_ghost_cells;
    Ghost::GhostType           m_particle_ghost_type;

    // so we can manually copy vars between AMR levels
    std::set<std::string> m_copy_data_vars;

    // ignore copying these vars between AMR levels
    std::set<std::string> m_not_copy_data_vars;

    // vars manually set not to scrub (normally when needed between a normal taskgraph
    // and the regridding phase)
    std::set<std::string> m_no_scrub_vars;

    // treat variable as an "old" var - will be checkpointed, copied, and only scrubbed from an OldDW
    std::set<std::string> m_treat_as_old_vars;
    
    // do not checkpoint these variables
    std::set<std::string> m_not_checkpoint_vars;

  private:

    // eliminate copy, assignment and move
    SchedulerCommon( const SchedulerCommon & )            = delete;
    SchedulerCommon& operator=( const SchedulerCommon & ) = delete;
    SchedulerCommon( SchedulerCommon && )                 = delete;
    SchedulerCommon& operator=( SchedulerCommon && )      = delete;

    // Maximum memory use as sampled across a given timestep.
    unsigned long                m_max_mem_use;

    ProblemSpecP                 m_graph_doc;
    ProblemSpecP                 m_nodes;
    std::ofstream              * m_memlog_file;
    bool                         m_emit_taskgraph;

    LocallyComputedPatchVarMap * m_locally_computed_patchvar_map;
    Relocate                     m_reloc_1;
    Relocate                     m_reloc_2;

    // whether or not to send a small message (takes more work to organize)
    // or a larger one (more communication time)
    bool m_use_small_messages;

    //! These are to store which vars we have to copy to the new grid
    //! in a copy data task.  Set in scheduleDataCopy and used in
    //! copyDataToNewGrid.
    typedef std::map<const VarLabel*, MaterialSubset*, VarLabel::Compare> label_matl_map;
    std::vector<label_matl_map> m_label_matls;

    //! set in addTask - can be used until initialize is called...
    std::vector<const Task::Dependency*>         m_init_requires;
    std::set<const VarLabel*, VarLabel::Compare> m_init_requires_vars;
    std::set<const VarLabel*, VarLabel::Compare> m_computed_vars;

    // TODO replace after Mira DDT problem is debugged (APH - 03/24/15)
    //max ghost cells of all tasks - will be used for loadbalancer to create neighborhood
    int maxGhost;
    //max level offset of all tasks - will be used for loadbalancer to create neighborhood
    int maxLevelOffset;
//    // max ghost cells of all tasks (per level) - will be used by loadbalancer to create neighborhood
//    // map levelIndex to maxGhostCells
//    //   this is effectively maximum horizontal range considered by the loadbalanceer for the neighborhood creation
//    std::map<int, int> maxGhostCells;
//
//    // max level offset of all tasks (per level) - will be used for loadbalancer to create neighborhood
//    // map levelIndex to maxLevelOffset
//    //   this is effectively maximum vertical range considered by the loadbalanceer for the neighborhood creation
//    std::map<int, int> maxLevelOffsets;
    
  };
} // namespace Uintah


#endif // CCA_COMPONENTS_SCHEDULERS_SCHEDULERCOMMON_H
