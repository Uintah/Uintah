/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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

#include <Core/Parallel/UintahParallelPort.h>
#include <Core/Grid/GridP.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Ports/SchedulerP.h>
#include <Core/OS/Dir.h>
#include <string>


namespace Uintah {

  using SCIRun::Dir;

  class ProcessorGroup;
  class Patch;

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
                              SimulationState* state) = 0;

    virtual void initializeOutput(const ProblemSpecP& params) = 0;
    //////////
    // Call this when restarting from a checkpoint after calling
    // problemSetup.
    virtual void restartSetup(Dir& restartFromDir, int startTimestep,
			      int timestep, double time, bool fromScratch,
			      bool removeOldDir) = 0;
    //////////
    // Call this when doing a combine_patches run after calling
    // problemSetup.  
    virtual void combinePatchSetup(Dir& fromDir) = 0;

    virtual bool needRecompile(double time, double delt,
			       const GridP& grid) = 0;

    //////////
    // Call this after all other tasks have been added to the scheduler
    virtual void finalizeTimestep(double t, double delt, const GridP&,
				  SchedulerP&, bool recompile = false,
                                  int addMaterial = 0) = 0;

    //////////
    // Call this after a timestep restart to make sure we still
    // have an output timestep
    virtual void reEvaluateOutputTimestep(double old_delt, double new_delt)=0;

    //////////
    // Call this after the timestep has been executed.
    virtual void executedTimestep(double delt, const GridP&) = 0;
     
      //////////
      // Insert Documentation Here:
    virtual const std::string getOutputLocation() const = 0;

    //////////
    // Get the current time step
    virtual int getCurrentTimestep() = 0;

    //////////
    // Get the current time step
    virtual double getCurrentTime() = 0;

    // Get the time the next output will occur
    virtual double getNextOutputTime() = 0;

    // Get the timestep the next output will occur
    virtual int getNextOutputTimestep() = 0;

    // Get the time the next checkpoint will occur
    virtual double getNextCheckpointTime() = 0;

    // Get the timestep the next checkpoint will occur
    virtual int getNextCheckpointTimestep() = 0;
      
    // Returns true if data will be output this timestep
    virtual bool isOutputTimestep() = 0;

    // Returns true if data will be checkpointed this timestep
    virtual bool isCheckpointTimestep() = 0;

    // Returns true if the label is being saved
    virtual bool isLabelSaved(std::string label) = 0;

    //////////
    // Get the directory of the current time step for outputting info.
    virtual const std::string& getLastTimestepOutputLocation() const = 0;
  private:
    Output(const Output&);
    Output& operator=(const Output&);
  };

} // End namespace Uintah

#endif
