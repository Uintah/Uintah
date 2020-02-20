/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

#ifndef UINTAH_HOMEBREW_APPLICATIONCOMMON_H
#define UINTAH_HOMEBREW_APPLICATIONCOMMON_H

#include <Core/Parallel/UintahParallelComponent.h>
#include <CCA/Ports/ApplicationInterface.h>
#include <CCA/Ports/LoadBalancer.h>
#include <CCA/Ports/SchedulerP.h>
#include <CCA/Ports/SolverInterface.h>

#include <CCA/Components/Application/ApplicationReductionVariable.h>

#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/MaterialManagerP.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/OS/Dir.h>
#include <Core/Util/Handle.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

#include <sci_defs/visit_defs.h>

namespace Uintah {

  class DataWarehouse;
  class Output;
  class Regridder;

/**************************************

CLASS
   ApplicationCommon
   
   Short description...

GENERAL INFORMATION

   ApplicationCommon.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   
KEYWORDS
   Application Interface

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/
  class ApplicationCommon : public UintahParallelComponent,
                            public ApplicationInterface {

    friend class Switcher;
    friend class PostProcessUda;

  public:
    ApplicationCommon(const ProcessorGroup* myworld,
                      const MaterialManagerP materialManager);

    virtual ~ApplicationCommon();
  
    // Methods for managing the components attached via the ports.
    virtual void setComponents( UintahParallelComponent *comp );
    virtual void getComponents();
    virtual void releaseComponents();

    virtual Scheduler *getScheduler() { return m_scheduler; }
    virtual Regridder *getRegridder() { return m_regridder; }
    virtual Output    *getOutput()    { return m_output; }

    // Top level problem set up called by sus.
    virtual void problemSetup( const ProblemSpecP &prob_spec );
    virtual void problemSetupDeltaT( const ProblemSpecP &prob_spec );
    
    // Top level problem set up called by simulation controller.
    virtual void problemSetup( const ProblemSpecP     & params,
                               const ProblemSpecP     & restart_prob_spec,
                                     GridP            & grid ) = 0;

    // Called to add missing grid based UPS specs.
    virtual void preGridProblemSetup( const ProblemSpecP & params, 
                                            GridP        & grid ) {};

    // Used to write parts of the problem spec.
    virtual void outputProblemSpec( ProblemSpecP & ps ) {};
      
    // Schedule the inital setup of the problem.
    virtual void scheduleInitialize( const LevelP     & level,
                                           SchedulerP & scheduler ) = 0;
                                 
    // On a restart schedule an initialization task.
    virtual void scheduleRestartInitialize( const LevelP     & level,
                                                  SchedulerP & scheduler ) = 0;

    // Used by the switcher
    virtual void setupForSwitching() {}

    // restartInitialize() is called once and only once if and when a
    // simulation is restarted.  This allows the simulation component
    // to handle initializations that are necessary when a simulation
    // is restarted.
    virtual void restartInitialize() {}

    // Get the task graph the application wants to execute. Returns an
    // index into the scheduler's list of task graphs.
    virtual void setTaskGraphIndex( int index ) { m_taskGraphIndex = index; }
    virtual int  getTaskGraphIndex() { return m_taskGraphIndex; }

    // Schedule the inital switching.
    virtual void scheduleSwitchInitialization( const LevelP     & level,
                                                     SchedulerP & sched ) {}
      
    virtual void scheduleSwitchTest( const LevelP &     level,
                                           SchedulerP & scheduler ) {};

    // Schedule the actual time step advencement tasks.
    virtual void scheduleTimeAdvance( const LevelP     & level,
                                            SchedulerP & scheduler );

    // Optionally schedule tasks that can not be done in scheduleTimeAdvance.
    virtual void scheduleFinalizeTimestep( const LevelP     & level,
                                                 SchedulerP & scheduler ) {}

    // Optionally schedule analysis tasks.
    virtual void scheduleAnalysis( const LevelP     & level,
                                         SchedulerP & scheduler) {}

    // Optionally schedule a task that determines the next delt T value.
    virtual void scheduleComputeStableTimeStep( const LevelP & level,
                                                SchedulerP   & scheduler ) = 0;
      
    // Reduce the system wide values such as the next delta T.
    virtual void scheduleReduceSystemVars(const GridP      & grid,
                                          const PatchSet   * perProcPatchSet,
                                                SchedulerP & scheduler);

    void reduceSystemVars( const ProcessorGroup *,
                           const PatchSubset    * patches,
                           const MaterialSubset * matls,
                                 DataWarehouse  * old_dw,
                                 DataWarehouse  * new_dw );

    // An optional call for the application to check their reduction vars.
    virtual void checkReductionVars( const ProcessorGroup * pg,
                                     const PatchSubset    * patches,
                                     const MaterialSubset * matls,
                                           DataWarehouse  * old_dw,
                                           DataWarehouse  * new_dw ) {};
    
    // Schedule the initialization of system values such at the time step.
    virtual void scheduleInitializeSystemVars(const GridP      & grid,
                                              const PatchSet   * perProcPatchSet,
                                                    SchedulerP & scheduler);
    
    void initializeSystemVars( const ProcessorGroup *,
                               const PatchSubset    * patches,
                               const MaterialSubset * /*matls*/,
                                     DataWarehouse  * /*old_dw*/,
                                     DataWarehouse  * new_dw );
    
    // Schedule the updating of system values such at the time step.
    virtual void scheduleUpdateSystemVars(const GridP      & grid,
                                          const PatchSet   * perProcPatchSet,
                                                SchedulerP & scheduler);
    
    virtual void updateSystemVars( const ProcessorGroup *,
                                   const PatchSubset    * patches,
                                   const MaterialSubset * /*matls*/,
                                         DataWarehouse  * /*old_dw*/,
                                         DataWarehouse  * new_dw );
    
    // Methods used for scheduling AMR regridding.
    virtual void scheduleRefine( const PatchSet   * patches,
                                       SchedulerP & scheduler );
    
    virtual void scheduleRefineInterface( const LevelP     & fineLevel, 
                                                SchedulerP & scheduler,
                                                bool         needCoarseOld,
                                                bool         needCoarseNew );

    virtual void scheduleCoarsen( const LevelP     & coarseLevel, 
                                        SchedulerP & scheduler );

    // Schedule to mark flags for AMR regridding
    virtual void scheduleErrorEstimate( const LevelP     & coarseLevel,
                                              SchedulerP & scheduler);

    // Schedule to mark initial flags for AMR regridding
    virtual void scheduleInitialErrorEstimate(const LevelP     & coarseLevel,
                                                    SchedulerP & sched);

    // Used to get the progress ratio of an AMR regridding subcycle.
    virtual double getSubCycleProgress(DataWarehouse* fineNewDW);


    // Recompute a time step if current time advance is not
    // converging.  The returned time is the new delta T.
    virtual void   recomputeDelT();
    virtual double recomputeDelT( const double delT );

    // Updates the time step and the delta T.
    virtual void prepareForNextTimeStep();

    // Asks the application if it needs to be recompiled.
    virtual bool needRecompile( const GridP & grid ) { return false; };

    // Labels for access value in the data warehouse.
    virtual const VarLabel* getTimeStepLabel() const { return m_timeStepLabel; }
    virtual const VarLabel* getSimTimeLabel() const { return m_simulationTimeLabel; }
    virtual const VarLabel* getDelTLabel() const { return m_delTLabel; }

    //////////
    virtual void setAMR(bool val) { m_AMR = val; }
    virtual bool isAMR() const { return m_AMR; }
  
    virtual void setLockstepAMR(bool val) { m_lockstepAMR = val; }
    virtual bool isLockstepAMR() const { return m_lockstepAMR; }

    virtual void setDynamicRegridding(bool val) {m_dynamicRegridding = val; }
    virtual bool isDynamicRegridding() const { return m_dynamicRegridding; }
  
    // Boolean for vars chanegd by the in-situ.
    virtual void haveModifiedVars( bool val ) { m_haveModifiedVars = val; }
    virtual bool haveModifiedVars() const { return m_haveModifiedVars; }
     
    // For restarting.
    virtual bool isRestartTimeStep() const { return m_isRestartTimestep; } 
    virtual void setRestartTimeStep( bool val ){ m_isRestartTimestep = val; }

    // For regridding.
    virtual bool isRegridTimeStep() const { return m_isRegridTimeStep; }
    virtual void setRegridTimeStep( bool val ) {
      m_isRegridTimeStep = val;
      
      if( m_isRegridTimeStep )
        m_lastRegridTimestep = m_timeStep;
    }
    virtual int  getLastRegridTimeStep() { return m_lastRegridTimestep; }

    // Some applications can set reduction variables
    virtual unsigned int numReductionVariable() const
    {
      return m_appReductionVars.size();
    }

    virtual void addReductionVariable( std::string name,
                                       const TypeDescription *varType,
                                       bool varActive = false )
    {
      m_appReductionVars[name] = new ApplicationReductionVariable( name, varType, varActive );
    }

    virtual void activateReductionVariable(std::string name, bool val) { m_appReductionVars[name]->setActive( val ); }
    virtual bool activeReductionVariable(std::string name) const {
      if( m_appReductionVars.find(name) != m_appReductionVars.end() )
	return m_appReductionVars.find(name)->second->getActive();
      else
        return false;
    }

    virtual bool isBenignReductionVariable( std::string name ) { return m_appReductionVars[name]->isBenignValue(); }
    virtual void overrideReductionVariable( DataWarehouse * new_dw, std::string name,   bool val) { m_appReductionVars[name]->setValue( new_dw, val ); }
    virtual void overrideReductionVariable( DataWarehouse * new_dw, std::string name, double val) { m_appReductionVars[name]->setValue( new_dw, val ); }
    virtual void setReductionVariable( DataWarehouse * new_dw, std::string name,   bool val) { m_appReductionVars[name]->setValue( new_dw, val ); }
    virtual void setReductionVariable( DataWarehouse * new_dw, std::string name, double val) { m_appReductionVars[name]->setValue( new_dw, val ); }
    // Get application specific reduction values all cast to doubles.
    virtual double getReductionVariable( std::string name ) const
    {
      if( m_appReductionVars.find(name) != m_appReductionVars.end() )
        return m_appReductionVars.at(name)->getValue();
      else
        return 0;
    }
    
    virtual double getReductionVariable( unsigned int index ) const
    {
      for ( const auto & var : m_appReductionVars )
      {
        if( index == 0 )
          return var.second->getValue();
        else
          --index;
      }

      return 0;
    }
    
    virtual std::string getReductionVariableName( unsigned int index ) const
    {
      for ( const auto & var : m_appReductionVars )
      {
        if( index == 0 )
          return var.second->getName();
        else
          --index;
      }

      return "Unknown";
    }
    
    virtual unsigned int getReductionVariableCount( unsigned int index ) const
    {
      for ( const auto & var : m_appReductionVars )
      {
        if( index == 0 )
          return var.second->getCount();
        else
          --index;
      }

      return 0;
    }
    
    virtual bool overriddenReductionVariable( unsigned int index ) const
    {
      for ( const auto & var : m_appReductionVars )
      {
        if( index == 0 )
          return var.second->overridden();
        else
          --index;
      }

      return false;
    }
    
    // Access methods for member classes.
    virtual MaterialManagerP getMaterialManagerP() const { return m_materialManager; }
  
    virtual ReductionInfoMapper< ApplicationStatsEnum,
                                 double > & getApplicationStats()
    { return m_application_stats; };

    virtual void resetApplicationStats( double val )
    { m_application_stats.reset( val ); };
      
    virtual void reduceApplicationStats( bool allReduce,
                                         const ProcessorGroup* myWorld )
    { m_application_stats.reduce( allReduce, myWorld ); };      
 
  public:
    virtual void   setDelTOverrideRestart( double val ) { m_delTOverrideRestart = val; }
    virtual double getDelTOverrideRestart() const { return m_delTOverrideRestart; }

    virtual void   setDelTInitialMax( double val ) { m_delTInitialMax = val; }
    virtual double getDelTInitialMax() const { return m_delTInitialMax; }

    virtual void   setDelTInitialRange( double val ) { m_delTInitialRange = val; }
    virtual double getDelTInitialRange() const { return m_delTInitialRange; }

    virtual void   setDelTMultiplier( double val ) { m_delTMultiplier = val; }
    virtual double getDelTMultiplier() const { return m_delTMultiplier; }

    virtual void   setDelTMaxIncrease( double val ) { m_delTMaxIncrease = val; }
    virtual double getDelTMaxIncrease() const { return m_delTMaxIncrease; }
    
    virtual void   setDelTMin( double val ) { m_delTMin = val; }
    virtual double getDelTMin() const { return m_delTMin; }

    virtual void   setDelTMax( double val ) { m_delTMax = val; }
    virtual double getDelTMax() const { return m_delTMax; }

    virtual void   setSimTimeEndAtMax( bool val ) { m_simTimeEndAtMax = val; }
    virtual bool   getSimTimeEndAtMax() const  { return m_simTimeEndAtMax; }

    virtual void   setSimTimeMax( double val ) { m_simTimeMax = val; }
    virtual double getSimTimeMax() const { return m_simTimeMax; }

    virtual void   setSimTimeClampToOutput( bool val ) { m_simTimeClampToOutput = val; }
    virtual bool   getSimTimeClampToOutput() const { return m_simTimeClampToOutput; }

    virtual void   setTimeStepsMax( int val ) { m_timeStepsMax = val; }
    virtual int    getTimeStepsMax() const { return m_timeStepsMax; }

    virtual void   setWallTimeMax( double val ) { m_wallTimeMax = val; }
    virtual double getWallTimeMax() const { return m_wallTimeMax; }

  private:
    // The classes are private because only the top level application
    // should be changing them. This only really matters when there are
    // applications built upon multiple applications. The children
    // applications will not have valid values. They should ALWAYS get
    // the values via the data warehouse.
    
    // Flag for outputting or checkpointing if the next delta is invalid
    virtual void         setOutputIfInvalidNextDelT( ValidateFlag flag ) { m_outputIfInvalidNextDelTFlag = flag; }
    virtual ValidateFlag getOutputIfInvalidNextDelT() const { return m_outputIfInvalidNextDelTFlag; }

    virtual void         setCheckpointIfInvalidNextDelT( ValidateFlag flag ) { m_checkpointIfInvalidNextDelTFlag = flag; }
    virtual ValidateFlag getCheckpointIfInvalidNextDelT() const { return m_checkpointIfInvalidNextDelTFlag; }

    //////////
    virtual void   setDelT( double delT ) { m_delT = delT; }
    virtual double getDelT() const { return m_delT; }
    virtual void   setDelTForAllLevels(       SchedulerP& scheduler,
                                        const GridP & grid,
                                        const int totalFine );

    virtual void         setNextDelT( double delT, bool restart = false );
    virtual double       getNextDelT() const { return m_delTNext; }
    virtual ValidateFlag validateNextDelT( double &delTNext, unsigned int level );
    
    //////////
    virtual   void setSimTime( double simTime );
    virtual double getSimTime() const { return m_simTime; };
    
    // Returns the integer time step index of the simulation.  All
    // simulations start with a time step number of 0.  This value is
    // incremented by one before a time step is processed.  The 'set'
    // function should only be called by the SimulationController at the
    // beginning of a simulation.  The 'increment' function is called by
    // the SimulationController at the beginning of each time step.
    virtual void setTimeStep( int timeStep );
    virtual void incrementTimeStep();
    virtual int  getTimeStep() const { return m_timeStep; }

    virtual bool isLastTimeStep( double walltime );
    virtual bool maybeLastTimeStep( double walltime ) const;


  protected:
    Scheduler       * m_scheduler    {nullptr};
    LoadBalancer    * m_loadBalancer {nullptr};
    SolverInterface * m_solver       {nullptr};
    Regridder       * m_regridder    {nullptr};
    Output          * m_output       {nullptr};

    // Use a map to store the reduction variables. 
    std::map< std::string, ApplicationReductionVariable* > m_appReductionVars;
    
    enum VALIDATE_ENUM  // unsigned char
    {
      DELTA_T_MAX_INCREASE     = 0x01,
      DELTA_T_MIN              = 0x02,
      DELTA_T_MAX              = 0x04,
      DELTA_T_INITIAL_MAX      = 0x08,

      CLAMP_TIME_TO_OUTPUT     = 0x10,
      CLAMP_TIME_TO_CHECKPOINT = 0x20,
      CLAMP_TIME_TO_MAX        = 0x40
    };

  private:
    bool m_AMR {false};
    bool m_lockstepAMR {false};

    bool m_dynamicRegridding {false};
  
    bool m_isRestartTimestep {false};
    
    bool m_isRegridTimeStep {false};
    // While it may not have been a "re"-grid, the original grid is
    // created on time step 0.
    int m_lastRegridTimestep { 0 };

    bool m_haveModifiedVars {false};

    const VarLabel* m_timeStepLabel;
    const VarLabel* m_simulationTimeLabel;
    const VarLabel* m_delTLabel;
  
    // Some applications may use multiple task graphs.
    int m_taskGraphIndex{0};
    
    // The simulation runs to either the maximum number of time steps
    // (timeStepsMax) or the maximum simulation time (simTimeMax), which
    // ever comes first. If the "max_Timestep" is not specified in the .ups
    // file, then it is set to zero.
    double m_delT{0.0};
    double m_delTNext{0.0};

    double m_delTOverrideRestart{0}; // Override the restart delta T value
    double m_delTInitialMax{0};      // Maximum initial delta T
    double m_delTInitialRange{0};    // Simulation time range for the initial delta T

    double m_delTMin{0};             // Minimum delta T
    double m_delTMax{0};             // Maximum delta T
    double m_delTMultiplier{1.0};    // Multiple for increasing delta T
    double m_delTMaxIncrease{0};     // Maximum delta T increase.

    double m_simTime{0.0};           // Current sim time
    double m_simTimeMax{0};          // Maximum simulation time
    bool   m_simTimeEndAtMax{false}; // End the simulation at exactly this sim time.
    bool   m_simTimeClampToOutput{false}; // Clamp the simulation time to the next output or checkpoint


    int    m_timeStep{0};            // Current time step
    int    m_timeStepsMax{0};        // Maximum number of time steps to run.  

    double m_wallTimeMax{0};         // Maximum wall time.
  
    ValidateFlag     m_outputIfInvalidNextDelTFlag{0};
    ValidateFlag m_checkpointIfInvalidNextDelTFlag{0};

  protected:
    
    MaterialManagerP m_materialManager{nullptr};

    ReductionInfoMapper< ApplicationStatsEnum,
                         double > m_application_stats;
    
  private:
    ApplicationCommon(const ApplicationCommon&);
    ApplicationCommon& operator=(const ApplicationCommon&);

#ifdef HAVE_VISIT
  public:
    // Reduction analysis variables for on the fly analysis
    virtual std::vector< analysisVar > & getAnalysisVars() { return m_analysisVars; }
  
    // Interactive variables from the UPS problem spec.
    virtual std::vector< interactiveVar > & getUPSVars() { return m_UPSVars; }
  
    // Interactive state variables from the application.
    virtual std::vector< interactiveVar > & getStateVars() { return m_stateVars; }
  
    virtual void setVisIt( unsigned int val ) { m_doVisIt = val; }
    virtual unsigned int  getVisIt() { return m_doVisIt; }

  protected:
    // Reduction analysis variables for on the fly analysis
    std::vector< analysisVar > m_analysisVars;
  
    // Interactive variables from the UPS problem spec.
    std::vector< interactiveVar > m_UPSVars;

    // Interactive state variables from the application.
    std::vector< interactiveVar > m_stateVars;

    unsigned int m_doVisIt{false};
#endif
  };

} // End namespace Uintah
   
#endif
