#ifndef UINTAH_HOMEBREW_DataArchiver_H
#define UINTAH_HOMEBREW_DataArchiver_H

#include <Uintah/Interface/Output.h>
#include <Uintah/Parallel/UintahParallelComponent.h>
#include <SCICore/OS/Dir.h>

namespace Uintah {
   class VarLabel;
   using SCICore::OS::Dir;
   
   /**************************************
     
     CLASS
       DataArchiver
      
       Short Description...
      
     GENERAL INFORMATION
      
       DataArchiver.h
      
       Steven G. Parker
       Department of Computer Science
       University of Utah
      
       Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
       Copyright (C) 2000 SCI Group
      
     KEYWORDS
       DataArchiver
      
     DESCRIPTION
       Long description...
      
     WARNING
      
     ****************************************/
    
   class DataArchiver : public Output, public UintahParallelComponent {
   public:
      DataArchiver(int MpiRank, int MpiProcess);
      virtual ~DataArchiver();

      //////////
      // Insert Documentation Here:
      virtual void problemSetup(const ProblemSpecP& params);

      //////////
      // Insert Documentation Here:
      virtual void finalizeTimestep(double t, double delt, const LevelP&,
				    SchedulerP&,
				    DataWarehouseP& old_dw,
				    DataWarehouseP&);

      //////////
      // Insert Documentation Here:
      void output(const ProcessorContext*,
		  const Patch* patch,
		  DataWarehouseP& old_dw,
		  DataWarehouseP& new_dw,
		  int timestep,
		  const VarLabel*,
		  int matlindex);
   private:
      std::string d_filebase;
      int d_outputInterval;
      int d_currentTimestep;
      Dir d_dir;

      DataArchiver(const DataArchiver&);
      DataArchiver& operator=(const DataArchiver&);
      
   };
} // end namespace Uintah

//
// $Log$
// Revision 1.3  2000/05/30 20:18:55  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.2  2000/05/20 08:09:04  sparker
// Improved TypeDescription
// Finished I/O
// Use new XML utility libraries
//
// Revision 1.1  2000/05/15 19:39:35  sparker
// Implemented initial version of DataArchive (output only so far)
// Other misc. cleanups
//
//

#endif
