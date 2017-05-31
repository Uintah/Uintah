#ifndef UINTAH_HOMEBREW_OUTPUT_H
#define UINTAH_HOMEBREW_OUTPUT_H

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

#include <CCA/Ports/SchedulerP.h>
#include <Core/Containers/ConsecutiveRangeSet.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/Variables/MaterialSetP.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/OS/Dir.h>
#include <Core/Parallel/UintahParallelPort.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

#include <string>
#include <map>

namespace Uintah {

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

class SimulationState;

  class Output : public UintahParallelPort {
  public:
    Output();
    virtual ~Output();
      
    //////////
    // Insert Documentation Here:
    virtual void problemSetup(const ProblemSpecP& params,
			      const ProblemSpecP& restart_prob_spec,
                              SimulationState* state) = 0;

    virtual void initializeOutput(const ProblemSpecP& params) = 0;
    //////////
    // Call this when restarting from a checkpoint after calling
    // problemSetup.
    virtual void restartSetup(Dir& restartFromDir, int startTimestep,
                              int timestep, double time, bool fromScratch,
                              bool removeOldDir) = 0;
    //////////
    // set timeinfoFlags and 
    virtual void reduceUdaSetup(Dir& fromDir) = 0;

    virtual bool needRecompile(double time, double delt,
                               const GridP& grid) = 0;

    //////////
    // Call this after all other tasks have been added to the scheduler
    virtual void finalizeTimestep(double t, double delt, const GridP&,
                                  SchedulerP&, bool recompile = false ) = 0;

    // schedule all output tasks
    virtual void sched_allOutputTasks(double delt, const GridP&,
                                      SchedulerP&, bool recompile = false ) = 0;

    //////////
    // Call this after a timestep restart where delt is adjusted to
    // make sure there still will be output and/or checkpoint timestep
    virtual void reevaluate_OutputCheckPointTimestep(double time) = 0;

    //////////
    // Call this after the timestep has been executed to find the
    // next time step to output
    virtual void findNext_OutputCheckPointTimestep(double time,
						   bool restart = false) = 0;
    
    //////////
    // update or write to the xml files
    virtual void writeto_xml_files(double delt, const GridP& grid) = 0;
    virtual void writeto_xml_files( std::map< std::string, std::pair<std::string, std::string> > &modifiedVars ) = 0;
     
      //////////
      // Insert Documentation Here:
    virtual const std::string getOutputLocation() const = 0;

    // Get the time/timestep the next output will occur
    virtual double getNextOutputTime() const = 0;
    virtual int    getNextOutputTimestep() const = 0;

    // Pushes output back by one timestep.
    virtual void postponeNextOutputTimestep() = 0; 

    // Get the time / timestep/ walltime that the next checkpoint will occur
    virtual double getNextCheckpointTime()     const = 0; // Simulation time (seconds and fractions there of)
    virtual int    getNextCheckpointTimestep() const = 0; // integer - time step
    virtual int    getNextCheckpointWalltime() const = 0; // integer - seconds
      
    // Returns true if data will be output this timestep
    virtual bool isOutputTimestep() const = 0;

    // Returns true if data will be checkpointed this timestep
    virtual bool isCheckpointTimestep() const = 0;

    // Returns true if the label is being saved
    virtual bool isLabelSaved( const std::string & label ) const = 0;

    // update output interval
    virtual void updateOutputInterval( double inv ) = 0;
    virtual void updateOutputTimestepInterval( int inv ) = 0;

    // get output interval
    virtual double getOutputInterval() const = 0;
    virtual int    getOutputTimestepInterval() const = 0;
    
    // update checkpoint interval
    virtual void updateCheckpointInterval( double inv ) = 0;
    virtual void updateCheckpointTimestepInterval( int inv ) = 0;

    // get checkpoint interval
    virtual double getCheckpointInterval() const = 0;
    virtual int    getCheckpointTimestepInterval() const = 0;
    virtual int    getCheckpointWalltimeInterval() const = 0;

    // Returns true if the UPS file has specified to save the UDA using PIDX format.
    virtual bool   savingAsPIDX() const = 0;

    // Instructs the output source (DataArchivers) on which format to use when saving data.
    virtual void   setSaveAsUDA() = 0;
    virtual void   setSaveAsPIDX() = 0;

    //! Called by In-situ VisIt to force the dump of a time step's data.
    virtual void outputTimestep( double time, double delt,
				 const GridP& grid, SchedulerP& sched ) = 0;

    virtual void checkpointTimestep( double time, double delt,
				     const GridP& grid, SchedulerP& sched ) = 0;
    //////////
    // Get the directory of the current time step for outputting info.
    virtual const std::string& getLastTimestepOutputLocation() const = 0;
    
  private:
    
    Output( const Output& );
    Output& operator=( const Output& );
  };

} // End namespace Uintah

#endif
