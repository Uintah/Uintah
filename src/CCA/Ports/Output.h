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

#ifndef UINTAH_HOMEBREW_OUTPUT_H
#define UINTAH_HOMEBREW_OUTPUT_H

#include <CCA/Components/Schedulers/RuntimeStatsEnum.h>
#include <CCA/Ports/SchedulerP.h>

#include <Core/Containers/ConsecutiveRangeSet.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/MaterialManagerP.h>
#include <Core/Grid/Variables/MaterialSetP.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/OS/Dir.h>
#include <Core/Parallel/UintahParallelPort.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Util/InfoMapper.h>

#include <string>
#include <map>

namespace Uintah {

  class UintahParallelComponent;
  class ProcessorGroup;

  class Patch;
  class VarLabel;

/**************************************

CLASS
   Output
   
   Short description...

GENERAL INFORMATION

   Output.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   
KEYWORDS
   Output

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class Output : public UintahParallelPort {
  public:
    Output();
    virtual ~Output();
      
    // Methods for managing the components attached via the ports.
    virtual void setComponents( UintahParallelComponent *comp ) = 0;
    virtual void getComponents() = 0;
    virtual void releaseComponents() = 0;

    //////////
    // Insert Documentation Here:
    virtual void problemSetup( const ProblemSpecP     & params,
                               const ProblemSpecP     & restart_prob_spec,
                               const MaterialManagerP & materialManager ) = 0;

    virtual void initializeOutput( const ProblemSpecP & params,
                                   const GridP        & grid ) = 0;
    
    //////////
    // Call this when restarting from a checkpoint after calling
    // problemSetup.
    virtual void restartSetup( const Dir  & restartFromDir,
                               const int    startTimeStep,
                               const int    timeStep,
                               const double simTime,
                               const bool   fromScratch,
                               const bool   removeOldDir) = 0;
    //////////
    // set timeinfoFlags and 
    virtual void postProcessUdaSetup( Dir & fromDir ) = 0;

    virtual bool needRecompile( const GridP & grid ) = 0;

    virtual void recompile( const GridP & grid ) = 0;

    //////////
    // Call this after all other tasks have been added to the scheduler
    virtual void finalizeTimeStep(const GridP      & grid,
                                        SchedulerP & scheduler,
                                        bool         recompile = false ) = 0;

    // schedule all output tasks
    virtual void sched_allOutputTasks( const GridP      & grid,
                                             SchedulerP & scheduler,
                                             bool         recompile = false ) = 0;

    //////////
    // Call this after a time step restart where delt is adjusted to
    // make sure there still will be output and/or checkpoint time step
    virtual void reevaluate_OutputCheckPointTimeStep( const double simTime,
                                                      const double delT ) = 0;

    //////////
    // Call this after the time step has been executed to find the
    // next time step to output
    virtual void findNext_OutputCheckPointTimeStep( const bool    restart,
                                                    const GridP & grid ) = 0;
    
    //////////
    // update or write to the xml files
    virtual void writeto_xml_files( const GridP& grid ) = 0;
    
    virtual void writeto_xml_files( std::map< std::string,
                                              std::pair< std::string,
                                                         std::string > > & modifiedVars ) = 0;
     
    //////////
    // Insert Documentation Here:
    virtual const std::string getOutputLocation() const = 0;

    virtual void setScrubSavedVariables( bool val ) = 0;

    // Get the time/time step the next output will occur
    virtual double getNextOutputTime() const = 0;
    virtual int    getNextOutputTimeStep() const = 0;

    // Pushes output back by one time step.
    virtual void postponeNextOutputTimeStep() = 0; 

    // Get the time / time step/ wall time that the next checkpoint will occur
    virtual double getNextCheckpointTime()     const = 0; // Simulation time (seconds and fractions there of)
    virtual int    getNextCheckpointTimeStep() const = 0; // integer - time step
    virtual int    getNextCheckpointWallTime() const = 0; // integer - seconds
      
    // Returns true if data will be output this time step
    virtual void setOutputTimeStep( bool val, const GridP& grid ) = 0;
    virtual bool isOutputTimeStep() const = 0;

    // Returns true if data will be checkpointed this time step
    virtual void setCheckpointTimeStep( bool val, const GridP& grid ) = 0;
    virtual bool isCheckpointTimeStep() const = 0;

    // Returns true if the label is being saved
    virtual bool isLabelSaved( const std::string & label ) const = 0;

    // output interval
    virtual void   setOutputInterval( double inv ) = 0;
    virtual double getOutputInterval() const = 0;
    virtual void   setOutputTimeStepInterval( int inv ) = 0;
    virtual int    getOutputTimeStepInterval() const = 0;
    
    // get checkpoint interval
    virtual void   setCheckpointInterval( double inv ) = 0;
    virtual double getCheckpointInterval() const = 0;
    virtual void   setCheckpointTimeStepInterval( int inv ) = 0;
    virtual int    getCheckpointTimeStepInterval() const = 0;
    virtual void   setCheckpointWallTimeInterval( int inv ) = 0;
    virtual int    getCheckpointWallTimeInterval() const = 0;

    // Returns true if the UPS file has specified to save the UDA using PIDX format.
    virtual bool   savingAsPIDX() const = 0;

    // Instructs the output source (DataArchivers) on which format to use when saving data.
    virtual void   setSaveAsUDA() = 0;
    virtual void   setSaveAsPIDX() = 0;

    //! Called by the in situ VisIt to output and chaeckpoint on
    //! demand.
    virtual void outputTimeStep( const GridP& grid,
                                 SchedulerP& sched,
                                 bool previous ) = 0;

    virtual void checkpointTimeStep( const GridP& grid,
                                     SchedulerP& sched,
                                     bool previous ) = 0;

    virtual void maybeLastTimeStep( bool val ) = 0;
    virtual bool maybeLastTimeStep() = 0;

    virtual void   setElapsedWallTime( double val ) = 0;
    virtual double getElapsedWallTime() const = 0;

    virtual void   setCheckpointCycle( int val ) = 0;
    virtual double getCheckpointCycle() const = 0;

    virtual void setUseLocalFileSystems( bool val ) = 0;
    virtual bool getUseLocalFileSystems() const = 0;

    virtual void setSwitchState(bool val) = 0;
    virtual bool getSwitchState() const = 0;
    
    //////////
    // Get the directory of the current time step for outputting info.
    virtual const std::string& getLastTimeStepOutputLocation() const = 0;
    
    virtual void setRuntimeStats( ReductionInfoMapper< RuntimeStatsEnum, double > *runtimeStats) = 0;

    // Returns trus if an output or checkpoint exists for the time step
    virtual bool outputTimeStepExists( unsigned int ts ) = 0;
    virtual bool checkpointTimeStepExists( unsigned int ts ) = 0;
    
  private:
    
    Output( const Output& );
    Output& operator=( const Output& );
  };

} // End namespace Uintah

#endif
