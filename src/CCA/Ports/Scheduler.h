/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

#ifndef CCA_PORTS_SCHEDULER_H
#define CCA_PORTS_SCHEDULER_H

#include <Core/Parallel/UintahParallelPort.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/Task.h>

#include <list>
#include <map>
#include <set>
#include <string>
#include <vector>


namespace Uintah {

  class LoadBalancerPort;
  class Task;
  class SimulationInterface;

/**************************************

CLASS
   Scheduler
   

GENERAL INFORMATION

   Scheduler.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   
KEYWORDS
   Scheduler


DESCRIPTION

  
****************************************/

class Scheduler : public UintahParallelPort {

  public:

    Scheduler();

    virtual ~Scheduler();
   
    // Only called by the SimulationController, and only once, and only
    // if the simulation has been "restarted".
    virtual void setGeneration( int id ) = 0;

    virtual void problemSetup( const ProblemSpecP& prob_spec, SimulationStateP& state ) = 0;

    virtual void checkMemoryUse( unsigned long & memUsed,
				 unsigned long & highwater,
				 unsigned long & maxMemUsed ) = 0;
    
    virtual void setStartAddr( char * start ) = 0;  // sbrk memory start location (for memory tracking)

    virtual char* getStartAddr() = 0;

    virtual void resetMaxMemValue() = 0;

    virtual void initialize( int numOldDW = 1, int numNewDW = 1 ) = 0;

    virtual void setParentDWs( DataWarehouse * parent_old_dw, DataWarehouse * parent_new_dw ) = 0;

    virtual void clearMappings() = 0;

    virtual void mapDataWarehouse( Task::WhichDW, int dwTag ) = 0;

    virtual void doEmitTaskGraphDocs() = 0;
    
    virtual void compile() = 0;

    virtual void execute( int tgnum = 0, int iteration = 0 ) = 0;

    virtual SchedulerP createSubScheduler() = 0;
       
    enum tgType { NormalTaskGraph, IntermediateTaskGraph };

    virtual void addTaskGraph( tgType type, int index = -1 ) = 0;

    virtual int getNumTaskGraphs() = 0;

    virtual void setNumTaskGraphs( const int num_task_graphs = 1) = 0;
    
    virtual bool useSmallMessages() = 0;
    
    virtual void addTask( Task* t, const PatchSet*, const MaterialSet*, const int tgnum = -1 ) = 0;
    
    virtual const std::vector<const Task::Dependency*>&         getInitialRequires() const = 0;

    virtual const std::set<const VarLabel*, VarLabel::Compare>& getInitialRequiredVars() const = 0;

    virtual const std::set<const VarLabel*, VarLabel::Compare>& getComputedVars() const = 0;

    virtual const std::set<std::string>&                        getNotCheckPointVars() const = 0;    

    virtual LoadBalancerPort * getLoadBalancer() = 0;

    virtual void releaseLoadBalancer() = 0;

    virtual DataWarehouse* get_dw( int idx ) = 0;

    virtual DataWarehouse* getLastDW() = 0;

    virtual bool isOldDW( int idx ) const = 0;

    virtual bool isNewDW( int idx ) const = 0;

    virtual void logMemoryUse() = 0;
      
    virtual void advanceDataWarehouse( const GridP & grid, bool initialization = false ) = 0;

    virtual void fillDataWarehouses( const GridP& grid) = 0;

    virtual void replaceDataWarehouse( int index, const GridP & grid, bool initialization = false ) = 0;

    virtual void setRestartable( bool restartable ) = 0;
    
    virtual bool isRestartInitTimestep() = 0;

//        protected:

    virtual void setPositionVar( const VarLabel * posLabel ) = 0;
    
    using VarLabelList = std::vector<std::vector<const VarLabel*> >;
    virtual void scheduleParticleRelocation( const LevelP       & coarsestLevelwithParticles
                                           , const VarLabel     * posLabel
                                           , const VarLabelList & labels
                                           , const VarLabel     * new_posLabel
                                           , const VarLabelList & new_labels
                                           , const VarLabel     * particleIDLabel
                                           , const MaterialSet  * matls
                                           ) = 0;

    virtual void scheduleParticleRelocation( const LevelP       & level
                                           , const VarLabel     * posLabel
                                           , const VarLabelList & labels
                                           , const VarLabel     * new_posLabel
                                           , const VarLabelList & new_labels
                                           , const VarLabel     * particleIDLabel
                                           , const MaterialSet  * matls
                                           ,       int            w
                                           ) = 0;

    //////////
    // Schedule particle relocation without the need to provide pre-relocation labels. Warning: This
    // is experimental and has not been fully tested yet. Use with caution (tsaad).
    virtual void scheduleParticleRelocation( const LevelP       & coarsestLevelwithParticles
                                           , const VarLabel     * posLabel
                                           , const VarLabelList & otherLabels
                                           , const MaterialSet  * matls
                                           ) = 0;

    //! Schedule copying data to new grid after regridding
    virtual void scheduleAndDoDataCopy( const GridP & grid, SimulationInterface * sim ) = 0;

    virtual void clearTaskMonitoring() = 0;
    virtual void scheduleTaskMonitoring( const LevelP& level ) = 0;
    virtual void scheduleTaskMonitoring( const PatchSet* patches ) = 0;

    virtual void overrideVariableBehavior( const std::string & var
                                         ,       bool          treatAsOld
                                         ,       bool          copyData
                                         ,       bool          noScrub
                                         ,       bool          notCopyData
                                         ,       bool          noCheckpoint
                                         ) = 0;

    // Get the SuperPatch (set of connected patches making a larger rectangle)
    // for the given label and patch and find the largest extents encompassing
    // the expected ghost cells (requiredLow, requiredHigh) and the requested
    // ghost cells as well (requestedLow, requestedHigh) for each of the
    // patches.  Required and requested will be the same if requestedNumGCells = 0.
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
                        ) const = 0;

    // Makes and returns a map that maps strings to VarLabels of
    // that name and a list of material indices for which that
    // variable is valid (at least according to d_allcomps).
    using VarLabelMaterialMap = std::map< std::string, std::list<int> >;
    virtual VarLabelMaterialMap* makeVarLabelMaterialMap() = 0;

    virtual int getMaxGhost() = 0;

    virtual int getMaxDistalGhost() = 0;

    virtual int getMaxLevelOffset() = 0;

    virtual bool isCopyDataTimestep() = 0;

    virtual void setInitTimestep( bool ) = 0;

    virtual void setRestartInitTimestep( bool ) = 0;

  private:

    // eliminate copy, assignment and move
    Scheduler( const Scheduler & )            = delete;
    Scheduler& operator=( const Scheduler & ) = delete;
    Scheduler( Scheduler && )                 = delete;
    Scheduler& operator=( Scheduler && )      = delete;
};

} // End namespace Uintah

#endif // CCA_PORTS_SCHEDULER_H
