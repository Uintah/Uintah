#ifndef UINTAH_HOMEBREW_DataArchiver_H
#define UINTAH_HOMEBREW_DataArchiver_H

#include <Packages/Uintah/CCA/Ports/Output.h>
#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Core/OS/Dir.h>
#include <Core/Containers/ConsecutiveRangeSet.h>

namespace Uintah {
   
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

} // End namespace Uintah

#endif
