#ifndef UINTAH_HOMEBREW_OUTPUT_H
#define UINTAH_HOMEBREW_OUTPUT_H

#include <Uintah/Parallel/UintahParallelPort.h>
#include <Uintah/Grid/GridP.h>
#include <Uintah/Grid/LevelP.h>
#include <Uintah/Grid/SimulationStateP.h>
#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Interface/ProblemSpecP.h>
#include <Uintah/Interface/SchedulerP.h>
#include <SCICore/OS/Dir.h>
#include <string>

namespace Uintah {
   using namespace SCICore::OS;

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

      //////////
      // Insert Documentation Here:
      virtual void finalizeTimestep(double t, double delt, const LevelP&,
				    SchedulerP&,
				    DataWarehouseP&) = 0;

      //////////
      // Insert Documentation Here:
      virtual const std::string getOutputLocation() const = 0;

      //////////
      // Get the current time step
      virtual int getCurrentTimestep() = 0;

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

} // end namespace Uintah

//
// $Log$
// Revision 1.12  2000/09/08 17:02:51  witzel
// Added virtual methods for getting the current timestep, the last
// timestep directory (where data was output), and whether the last
// timestep was one in which data was output.
//
// Revision 1.11  2000/07/26 20:14:12  jehall
// Moved taskgraph/dependency output files to UDA directory
// - Added output port parameter to schedulers
// - Added getOutputLocation() to Uintah::Output interface
// - Renamed output files to taskgraph[.xml]
//
// Revision 1.10  2000/06/17 07:06:46  sparker
// Changed ProcessorContext to ProcessorGroup
//
// Revision 1.9  2000/06/16 19:46:46  sparker
// Output interface doesn't need old_dw
// DataWarehouse no longer needs carryForward
//
// Revision 1.8  2000/05/30 20:19:41  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.7  2000/05/15 19:39:52  sparker
// Implemented initial version of DataArchive (output only so far)
// Other misc. cleanups
//
// Revision 1.6  2000/04/26 06:49:11  sparker
// Streamlined namespaces
//
// Revision 1.5  2000/04/11 07:10:53  sparker
// Completing initialization and problem setup
// Finishing Exception modifications
//
// Revision 1.4  2000/03/17 09:30:03  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.3  2000/03/16 22:08:23  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//

#endif
