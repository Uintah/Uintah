#ifndef UINTAH_HOMEBREW_DataArchiver_H
#define UINTAH_HOMEBREW_DataArchiver_H

#include <Packages/Uintah/CCA/Ports/Output.h>
#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/Core/Grid/MaterialSetP.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Core/OS/Dir.h>
#include <Core/Containers/ConsecutiveRangeSet.h>
#include <Core/Thread/Mutex.h>

namespace Uintah {

using SCIRun::ConsecutiveRangeSet;
using SCIRun::Mutex;

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
      DataArchiver(const ProcessorGroup* myworld, int udaSuffix = -1);
      virtual ~DataArchiver();

      //////////
      // Sets up when the DataArchiver will output and what data, according
      // to params.  Also stores state to keep track of time and timesteps
      // in the simulation.  (If you only need to use DataArchiver to copy 
      // data, then you can pass a NULL SimulationState
      virtual void problemSetup(const ProblemSpecP& params,
                                SimulationState* state);

      //////////
      // This function will set up the output for the simulation.  As part
      // of this it will output the input.xml and index.xml in the uda
      // directory.  Call after calling all problemSetups.
      virtual void initializeOutput(const ProblemSpecP& params);

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

      //////////
      // Call this when doing a combine_patches run after calling
      // problemSetup.  It will copy the data files over and make it ignore
      // dumping reduction variables.
      virtual void combinePatchSetup(Dir& fromDir);

      // Copy a section from another uda's index.xml.
      void copySection(Dir& fromDir, Dir& toDir, string section);
      void copySection(Dir& fromDir, string section)
      { copySection(fromDir, d_dir, section); }

      //////////
      // Call this after all other tasks have been added to the scheduler
      virtual void finalizeTimestep(double t, double delt, const GridP&,
				    SchedulerP&, bool recompile=false);

      //////////
      // Call this after the timestep has been executed.
      virtual void executedTimestep(double delt);
     
      //////////
      // Insert Documentation Here:
      virtual const string getOutputLocation() const;

      virtual bool needRecompile(double time, double dt,
				  const GridP& grid);

      //////////
      // Insert Documentation Here:
      void output(const ProcessorGroup*,
		  const PatchSubset* patch,
		  const MaterialSubset* matls,
		  DataWarehouse* old_dw,
		  DataWarehouse* new_dw,
		  Dir* p_dir,
		  const VarLabel*,
		  bool isThisCheckpoint);

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
      // Use the sharedState directly if you can, but if you must use these...
      virtual int getCurrentTimestep()
      { return d_sharedState->getCurrentTopLevelTimeStep(); }
       
      virtual double getCurrentTime()
      { return  d_sharedState->getElapsedTime(); }

      // Get the time the next output will occur
      virtual double getNextOutputTime() { return d_nextOutputTime; }
      
      // Get the timestep the next output will occur
      virtual int getNextOutputTimestep() { return d_nextOutputTimestep; }
      
      // Get the time the next checkpoint will occur
      virtual double getNextCheckpointTime() { return d_nextCheckpointTime; }
      
      // Get the timestep the next checkpoint will occur
      virtual int getNextCheckpointTimestep(){return d_nextCheckpointTimestep;}

      //////////
      // Returns true if the last timestep was one
      // in which data was output.
      virtual bool wasOutputTimestep()
      { return d_wasOutputTimestep; }

      //////////
      // Get the directory of the current time step for outputting info.
      virtual const string& getLastTimestepOutputLocation() const
      { return d_lastTimestepLocation; }

     public:
     
      struct SaveNameItem {
	string labelName;
	string compressionMode;
	ConsecutiveRangeSet matls;
        ConsecutiveRangeSet levels;
      };

      class SaveItem {
      public:
	void setMaterials(const ConsecutiveRangeSet& matls,
			  ConsecutiveRangeSet& prevMatls,
			  MaterialSetP& prevMatlSet);

	const MaterialSet* getMaterialSet() const
	{ return matlSet_.get_rep(); }
	  
	const VarLabel* label_;

        // store as CRS as the grid can change
        ConsecutiveRangeSet levels;
      private:
	MaterialSetP matlSet_;
      };

   private:
      // returns a ProblemSpecP that you need to call releaseDocument on
      ProblemSpecP loadDocument(std::string xmlName);     

      //! creates the uda directory with a trailing version suffix
      void makeVersionedDir();
      
      void initSaveLabels(SchedulerP& sched);
      void initCheckpoints(SchedulerP& sched);
     
      // helper for finalizeTimestep
      void outputTimestep(Dir& dir, vector<SaveItem>& saveLabels,
			  double time, double delt,
			  const GridP& grid,
			  string* pTimestepDir /* passed back */,
			  bool hasGlobals = false);

      void scheduleOutputTimestep(Dir& dir, vector<SaveItem>& saveLabels,
				  const GridP& grid, SchedulerP& sched,
				  bool isThisCheckpoint);
      void beginOutputTimestep(double time, double delt,
			       const GridP& grid);

      //////////
      // Call this after a timestep restart to make sure we still
      // have an output timestep
      virtual void reEvaluateOutputTimestep(double old_delt, double new_delt);
      // helper for problemSetup
      void createIndexXML(Dir& dir);

      // helpers for restartSetup
      void addRestartStamp(ProblemSpecP indexDoc, Dir& fromDir,
			   int timestep);

      void copyTimesteps(Dir& fromDir, Dir& toDir, int startTimestep,
			 int maxTimestep, bool removeOld,
			 bool areCheckpoints = false);
      void copyDatFiles(Dir& fromDir, Dir& toDir, int startTimestep,
			int maxTimestep, bool removeOld);
   
      // add saved global (reduction) variables to index.xml
      void indexAddGlobals();

      string d_filebase;
      SimulationStateP d_sharedState;
      double d_tempElapsedTime; // set in finalizeTimestep for output tasks

      // only one of these should be nonzero
      double d_outputInterval;
      int d_outputTimestepInterval;
     
      double d_nextOutputTime; // used when d_outputInterval != 0
      int d_nextOutputTimestep; // used when d_outputTimestepInterval != 0
      //int d_currentTimestep;
      Dir d_dir;
      bool d_writeMeta;
      string d_lastTimestepLocation;
      bool d_wasOutputTimestep;
      bool d_wasCheckpointTimestep;
      bool d_saveParticleVariables;
      bool d_saveP_x;

      //double d_currentTime;

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

      // how much real time to pass (in seconds) to wait for checkpoint
      // can be used with or without one of the above two
      // walltimeStart cannot be used without walltimeInterval
      int d_checkpointWalltimeStart;     //how long to wait before first 
      int d_checkpointWalltimeInterval;
      
      int d_checkpointCycle;

      Dir d_checkpointsDir;
      list<string> d_checkpointTimestepDirs;
      double d_nextCheckpointTime; // used when d_checkpointInterval != 0
      int d_nextCheckpointTimestep; // used when d_checkpointTimestepInterval != 0
      int d_nextCheckpointWalltime; // used when d_checkpointWalltimeInterval != 0
      //-----------------------------------------------------------
      // RNJ - 
      //
      // In order to avoid having to open and close index.xml,
      // p<xxxxx>.xml, and p<xxxxx>.data when we want to update
      // each variable, we will keep track of some XML docs and
      // file handles and only open and close them once per
      // timestep if they are needed.
      //-----------------------------------------------------------

      // We need to have two separate XML Index Docs
      // because it is possible to do an output
      // and a checkpoint at the same time.

      ProblemSpecP d_XMLIndexDoc;
      ProblemSpecP d_CheckpointXMLIndexDoc;

      // Each level needs it's own data file handle 
      // and if we are outputting and checkpointing
      // at the same time we need two different sets.

      map< int, int > d_DataFileHandles;
      map< int, int > d_CheckpointDataFileHandles;

      // Each level needs it's own XML Data Doc
      // and if we are outputting and checkpointing
      // at the same time we need two different sets.

      map< int, ProblemSpecP > d_XMLDataDocs;
      map< int, ProblemSpecP > d_CheckpointXMLDataDocs;


      //-----------------------------------------------------------
      // RNJ - 
      //
      // If the <DataArchiver> section of the .ups file contains:
      //
      //   <outputDoubleAsFloat />
      //
      // Then we will set the d_OutputDoubleAsFloat boolean to true
      // and we will try to output floats instead of doubles.
      //
      // NOTE: This does not affect checkpoints as they will
      //       always be outputting doubles for accuracy.
      //-----------------------------------------------------------

      bool d_outputDoubleAsFloat;

      string TranslateVariableType( string type, bool isThisCheckpoint );


      //-----------------------------------------------------------
      // RNJ - 
      //
      // This is the number of times the DataArchiver will retry
      // a file system operation before it gives up and throws
      // an exception.
      //-----------------------------------------------------------

      int d_fileSystemRetrys;


      //! This is if you want to pass in the uda extension on the command line
      int d_udaSuffix;

      Mutex d_outputLock;

      DataArchiver(const DataArchiver&);
      DataArchiver& operator=(const DataArchiver&);
      
   };

} // End namespace Uintah

#endif
