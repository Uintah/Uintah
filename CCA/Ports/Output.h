#ifndef UINTAH_HOMEBREW_OUTPUT_H
#define UINTAH_HOMEBREW_OUTPUT_H

#include <Packages/Uintah/Core/Parallel/UintahParallelPort.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/CCA/Ports/SchedulerP.h>
#include <Core/OS/Dir.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

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
  
   Copyright (C) 2000 SCI Group

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
      
    //////////
    // Insert Documentation Here:
    virtual void problemSetup(const ProblemSpecP& params) = 0;

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
				  SchedulerP&, bool = false) = 0;

    //////////
    // Call this after the timestep has been executed.
    virtual void executedTimestep() = 0;
     
      //////////
      // Insert Documentation Here:
    virtual const std::string getOutputLocation() const = 0;

    //////////
    // Get the current time step
    virtual int getCurrentTimestep() = 0;

      //////////
      // Get the current time step
    virtual double getCurrentTime() = 0;
      
    //////////
    // Returns true if the last timestep was one
    // in which data was output.
    virtual bool wasOutputTimestep() = 0;

    //////////
    // Get the directory of the current time step for outputting info.
    virtual const std::string& getLastTimestepOutputLocation() const = 0;
  private:
    Output(const Output&);
    Output& operator=(const Output&);
  };

} // End namespace Uintah

#endif
