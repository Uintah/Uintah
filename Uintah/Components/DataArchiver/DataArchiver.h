#ifndef UINTAH_HOMEBREW_DataArchiver_H
#define UINTAH_HOMEBREW_DataArchiver_H

#include <Uintah/Interface/Output.h>
#include <Uintah/Parallel/UintahParallelComponent.h>
#include <SCICore/OS/Dir.h>
#include <SCICore/Containers/ConsecutiveRangeSet.h>

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
      DataArchiver(const ProcessorGroup* myworld);
      virtual ~DataArchiver();

      //////////
      // Insert Documentation Here:
      virtual void problemSetup(const ProblemSpecP& params);

      //////////
      // Insert Documentation Here:
      virtual void finalizeTimestep(double t, double delt, const LevelP&,
				    SchedulerP&,
				    DataWarehouseP&);

      //////////
      // Insert Documentation Here:
      virtual const std::string getOutputLocation() const;

      //////////
      // Insert Documentation Here:
      void output(const ProcessorGroup*,
		  const Patch* patch,
		  DataWarehouseP& old_dw,
		  DataWarehouseP& new_dw,
		  int timestep,
		  const VarLabel*,
		  int matlindex);

      // Method to output reduction variables to a single file
      void outputReduction(const ProcessorGroup*,
			   DataWarehouseP& old_dw,
			   DataWarehouseP& new_dw,
			   double time);

      //////////
      // Get the current time step
      virtual int getCurrentTimestep()
      { return d_currentTimestep; }

      //////////
      // Returns true if the last timestep was one
      // in which data was output.
      virtual bool wasOutputTimestep()
      { return d_wasOutputTimestep; }

      //////////
      // Get the directory of the current time step for outputting info.
      virtual const std::string& getLastTimestepOutputLocation() const
      { return d_lastTimestepLocation; }
   private:
      void initSaveLabels(SchedulerP& sched);
     
      std::string d_filebase;
      double d_outputInterval;
      double d_nextOutputTime;
      int d_currentTimestep;
      Dir d_dir;
      bool d_writeMeta;
      std::string d_lastTimestepLocation;
      bool d_wasOutputTimestep;

      // d_saveLabelNames is a temporary list containing VarLabel
      // names to be saved and the materials to save them for.  The
      // information will be basically transferred to d_saveLabels or
      // d_saveReductionLabels after mapping VarLabel names to their
      // actual VarLabel*'s.
      struct SaveNameItem {
	 std::string labelName;
         ConsecutiveRangeSet matls;
      };
      std::list< SaveNameItem > d_saveLabelNames;

      struct SaveItem {
	 const VarLabel* label;
         ConsecutiveRangeSet matls;
      };
      std::vector< SaveItem > d_saveLabels;
      std::vector< SaveItem > d_saveReductionLabels;

      DataArchiver(const DataArchiver&);
      DataArchiver& operator=(const DataArchiver&);
      
   };
} // end namespace Uintah

//
// $Log$
// Revision 1.12  2000/12/06 23:59:40  witzel
// Added variable save functionality via the DataArchiver problem spec
//
// Revision 1.11  2000/09/08 17:00:11  witzel
// Added functions for getting the last timestep directory, the current
// timestep, and whether the last timestep was one in which data was
// output.  These functions are needed by the Scheduler to archive taskgraph
// data.
//
// Revision 1.10  2000/08/25 17:41:15  sparker
// All output from an MPI run now goes into a single UDA dir
//
// Revision 1.9  2000/07/26 20:14:09  jehall
// Moved taskgraph/dependency output files to UDA directory
// - Added output port parameter to schedulers
// - Added getOutputLocation() to Uintah::Output interface
// - Renamed output files to taskgraph[.xml]
//
// Revision 1.8  2000/06/17 07:06:30  sparker
// Changed ProcessorContext to ProcessorGroup
//
// Revision 1.7  2000/06/16 19:48:19  sparker
// Use updated output interface
//
// Revision 1.6  2000/06/14 21:50:54  guilkey
// Changed DataArchiver to create uda files based on a per time basis
// rather than a per number of timesteps basis.
//
// Revision 1.5  2000/06/03 05:24:26  sparker
// Finished/changed reduced variable emits
// Fixed bug in directory version numbers where the index was getting
//   written to the base file directory instead of the versioned one.
//
// Revision 1.4  2000/06/01 23:09:39  guilkey
// Added beginnings of code to store integrated quantities.
//
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
