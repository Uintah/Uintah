#ifndef UINTAH_HOMEBREW_DataArchiver_H
#define UINTAH_HOMEBREW_DataArchiver_H

#include <Packages/Uintah/CCA/Ports/Output.h>
#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/Core/Grid/MaterialSetP.h>
#include <Core/OS/Dir.h>
#include <Core/Containers/ConsecutiveRangeSet.h>
#include <Core/Thread/Mutex.h>

class DOM_Document;

namespace Uintah {

using std::string;
using std::vector;
using std::list;
using std::pair;

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
      // Call this when restarting from a checkpoint after calling
      // problemSetup.  This will copy timestep directories and dat
      // files up to the specified timestep from restartFromDir if
      // fromScratch is false and will set time and timestep variables
      // appropriately to continue smoothly from that timestep.
      // If timestep is negative, then all timesteps will get copied
      // if they are to be copied at all (fromScratch is false).
      virtual void restartSetup(Dir& restartFromDir, int startTimestep,
				int timestep, double time, bool fromScratch,
				bool removeOldDir);

      // Copy a section from another uda's index.xml.
      void copySection(Dir& fromDir, Dir& toDir, string section);
      void copySection(Dir& fromDir, string section)
      { copySection(fromDir, d_dir, section); }

      //////////
      // Insert Documentation Here:
      virtual void finalizeTimestep(double t, double delt, const LevelP&,
				    SchedulerP&);
      //////////
      // Insert Documentation Here:
      virtual const string getOutputLocation() const;

      virtual bool need_recompile(double time, double dt,
				  const LevelP& level);

      //////////
      // Insert Documentation Here:
      void output(const ProcessorGroup*,
		  const PatchSubset* patch,
		  const MaterialSubset* matls,
		  DataWarehouse* old_dw,
		  DataWarehouse* new_dw,
		  Dir* p_dir,
		  const VarLabel*);

      // Method to output reduction variables to a single file
      void outputReduction(const ProcessorGroup*,
			   const PatchSubset* patch,
			   const MaterialSubset* matls,
			   DataWarehouse* old_dw,
			   DataWarehouse* new_dw);

      // This calls output for all of the checkpoint reduction variables
      // which will end up in globals.xml / globals.data -- in this way,
      // all this data will be output by one process avoiding conflicts.
      void outputCheckpointReduction(const ProcessorGroup* world,
				     const PatchSubset* patch,
				     const MaterialSubset* matls,
				     DataWarehouse* old_dw,
				     DataWarehouse* new_dw);

      //////////
      // Get the current time step
      virtual int getCurrentTimestep()
      { return d_currentTimestep; }
       
      //////////
      // Get the current time
      virtual double getCurrentTime()
      { return  d_currentTime; }

      //////////
      // Returns true if the last timestep was one
      // in which data was output.
      virtual bool wasOutputTimestep()
      { return d_wasOutputTimestep; }

      //////////
      // Get the directory of the current time step for outputting info.
      virtual const string& getLastTimestepOutputLocation() const
      { return d_lastTimestepLocation; }
   private:
     
      struct SaveNameItem {
	string labelName;
	string compressionMode;
	ConsecutiveRangeSet matls;
      };

      class SaveItem {
      public:
	void setMaterials(const ConsecutiveRangeSet& matls,
			  ConsecutiveRangeSet& prevMatls,
			  MaterialSetP& prevMatlSet);

	const MaterialSet* getMaterialSet() const
	{ return matlSet_.get_rep(); }
	  
	const VarLabel* label_;
      private:
	MaterialSetP matlSet_;
      };

      void initSaveLabels(SchedulerP& sched);
      void initCheckpoints(SchedulerP& sched);
     
      // helper for finalizeTimestep
      void outputTimestep(Dir& dir, vector<SaveItem>& saveLabels,
			  double time, double delt,
			  const LevelP& level,
			  string* pTimestepDir /* passed back */,
			  bool hasGlobals = false);

      void scheduleOutputTimestep(Dir& dir, vector<SaveItem>& saveLabels,
				  const LevelP& level, SchedulerP& sched);
      void beginOutputTimestep(double time, double delt,
			       const LevelP& level);

      // helper for problemSetup
      void createIndexXML(Dir& dir);

      // helpers for restartSetup
      void addRestartStamp(DOM_Document indexDoc, Dir& fromDir,
			   int timestep);

      void copyTimesteps(Dir& fromDir, Dir& toDir, int startTimestep,
			 int maxTimestep, bool removeOld,
			 bool areCheckpoints = false);
      void copyDatFiles(Dir& fromDir, Dir& toDir, int startTimestep,
			int maxTimestep, bool removeOld);
   
      // add saved global (reduction) variables to index.xml
      void indexAddGlobals();

      string d_filebase;

      // only one of these should be nonzero
      double d_outputInterval;
      int d_outputTimestepInterval;
     
      double d_nextOutputTime; // used when d_outputInterval != 0
      int d_nextOutputTimestep; // used when d_outputTimestepInterval != 0
      int d_currentTimestep;
      Dir d_dir;
      bool d_writeMeta;
      string d_lastTimestepLocation;
      bool d_wasOutputTimestep;
      bool d_wasCheckpointTimestep;
      bool d_saveParticleVariables;
      bool d_saveP_x;

      double d_currentTime;

      // d_saveLabelNames is a temporary list containing VarLabel
      // names to be saved and the materials to save them for.  The
      // information will be basically transferred to d_saveLabels or
      // d_saveReductionLabels after mapping VarLabel names to their
      // actual VarLabel*'s.
      list< SaveNameItem > d_saveLabelNames;
      vector< SaveItem > d_saveLabels;
      vector< SaveItem > d_saveReductionLabels;

      // for efficiency of SaveItem's
      ConsecutiveRangeSet prevMatls_;
      MaterialSetP prevMatlSet_;     
     
      // d_checkpointLabelNames is a temporary list containing
      // the names of labels to save when checkpointing
      vector< SaveItem > d_checkpointLabels;
      vector< SaveItem > d_checkpointReductionLabels;

      // only one of these should be nonzero
      double d_checkpointInterval;
      int d_checkpointTimestepInterval;

      int d_checkpointCycle;

      Dir d_checkpointsDir;
      list<string> d_checkpointTimestepDirs;
      double d_nextCheckpointTime; // used when d_checkpointInterval != 0
      int d_nextCheckpointTimestep; // used when d_checkpointTimestepInterval != 0
      Mutex d_outputLock;
      Mutex d_outputReductionLock;

      DataArchiver(const DataArchiver&);
      DataArchiver& operator=(const DataArchiver&);
      
   };

} // End namespace Uintah

#endif
