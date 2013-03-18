/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

#ifndef UINTAH_HOMEBREW_DataArchiver_H
#define UINTAH_HOMEBREW_DataArchiver_H

#include <CCA/Ports/Output.h>
#include <Core/Parallel/UintahParallelComponent.h>
#include <Core/Grid/Variables/MaterialSetP.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Util/Assert.h>
#include <Core/OS/Dir.h>
#include <Core/Containers/ConsecutiveRangeSet.h>
#include <Core/Thread/Mutex.h>

namespace Uintah {
class DataWarehouse;
using SCIRun::ConsecutiveRangeSet;
using SCIRun::Mutex;


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
      
             
     KEYWORDS
       DataArchiver
      
     DESCRIPTION
       Long description...
      
     WARNING
      
     ****************************************/
    
   //! Handles outputting the data.
   class DataArchiver : public Output, public UintahParallelComponent {
     public:
       DataArchiver(const ProcessorGroup* myworld, int udaSuffix = -1);
       virtual ~DataArchiver();

       static bool wereSavesAndCheckpointsInitialized;

       //! Sets up when the DataArchiver will output and what data, according
       //! to params.  Also stores state to keep track of time and timesteps
       //! in the simulation.  (If you only need to use DataArchiver to copy 
       //! data, then you can pass a NULL SimulationState
       virtual void problemSetup(const ProblemSpecP& params,
           SimulationState* state);

       //! This function will set up the output for the simulation.  As part
       //! of this it will output the input.xml and index.xml in the uda
       //! directory.  Call after calling all problemSetups.
       virtual void initializeOutput(const ProblemSpecP& params);

       //! Call this when restarting from a checkpoint after calling
       //! problemSetup.  This will copy timestep directories and dat
       //! files up to the specified timestep from restartFromDir if
       //! fromScratch is false and will set time and timestep variables
       //! appropriately to continue smoothly from that timestep.
       //! If timestep is negative, then all timesteps will getg copied
       //! if they are to be copied at all (fromScratch is false).
       virtual void restartSetup(Dir& restartFromDir, int startTimestep,
           int timestep, double time, bool fromScratch,
           bool removeOldDir);

       //! Call this when doing a combine_patches run after calling
       //! problemSetup.  It will copy the data files over and make it ignore
       //! dumping reduction variables.
       virtual void combinePatchSetup(Dir& fromDir);

       //! Copy a section between udas' index.xml.
       void copySection(Dir& fromDir, Dir& toDir, std::string section);

       //! Copy a section from another uda's to our index.xml.
       void copySection(Dir& fromDir, std::string section)
       { copySection(fromDir, d_dir, section); }

       //! Checks to see if this is an output timestep. 
       //! If it is, setup directories and xml files that we need to output.
       //! Will also setup the tasks if we are recompiling the taskgraph.
       //! Call once per timestep, and if recompiling,
       //! after all the other tasks are scheduled.
       virtual void finalizeTimestep(double t, double delt, const GridP&,
           SchedulerP&, bool recompile=false,
           int addMaterial=0);

       //! Find the next times to output and dumps open files to disk.
       //! Call after timestep has completed.
       virtual void executedTimestep(double delt, const GridP&);

       //! Returns as a string the name of the top of the output directory.
       virtual const std::string getOutputLocation() const;

       //! Asks if we need to recompile the task graph.
       virtual bool needRecompile(double time, double dt,
           const GridP& grid);

       //! The task that handles the outputting.  Scheduled in finalizeTimestep.
       //! Handles outputs and checkpoints and differentiates between them in the
       //! last argument.  Outputs as binary the data acquired from VarLabel in 
       //! p_dir.
       void output(const ProcessorGroup*, const PatchSubset* patch,
           const MaterialSubset* matls, DataWarehouse* old_dw,
           DataWarehouse* new_dw, int type);

       //! Task that handles outputting non-checkpoint reduction variables.
       //! Scheduled in finalizeTimestep.
       void outputReduction(const ProcessorGroup*,
           const PatchSubset* patch,
           const MaterialSubset* matls,
           DataWarehouse* old_dw,
           DataWarehouse* new_dw);

       //! Recommended to use sharedState directly if you can.
       virtual int getCurrentTimestep()
       { return d_sharedState->getCurrentTopLevelTimeStep(); }

       //! Recommended to use sharedState directly if you can.
       virtual double getCurrentTime()
       { return  d_sharedState->getElapsedTime(); }

       //! Get the time the next output will occur
       virtual double getNextOutputTime() { return d_nextOutputTime; }

       //! Get the timestep the next output will occur
       virtual int getNextOutputTimestep() { return d_nextOutputTimestep; }

       //! Get the time the next checkpoint will occur
       virtual double getNextCheckpointTime() { return d_nextCheckpointTime; }

       //! Get the timestep the next checkpoint will occur
       virtual int getNextCheckpointTimestep(){return d_nextCheckpointTimestep;}

       //! Returns true if data will be output this timestep
       virtual bool isOutputTimestep()
       { return d_isOutputTimestep; }

       //! Returns true if data will be checkpointed this timestep
       virtual bool isCheckpointTimestep()
       { return d_isCheckpointTimestep; }

       //! Get the directory of the current time step for outputting info.
       virtual const std::string& getLastTimestepOutputLocation() const
       { return d_lastTimestepLocation; }

       bool isLabelSaved( std::string label );
       void updateOutputInv(double inv);
       void updateCheckpointInv(double inv);

     public:

       //! problemSetup parses the ups file into a list of these
       //! (d_saveLabelNames)
       struct SaveNameItem {
         std::string labelName;
         std::string compressionMode;
         ConsecutiveRangeSet matls;
         ConsecutiveRangeSet levels;
       };

       class SaveItem {
         public:
           void setMaterials(int level, const ConsecutiveRangeSet& matls,
               ConsecutiveRangeSet& prevMatls,
               MaterialSetP& prevMatlSet);

           MaterialSet* getMaterialSet(int level)
           { return matlSet_[level].get_rep(); }

           const VarLabel* label_;

           std::map<int, MaterialSetP> matlSet_;
       };

     private:
       //! returns a ProblemSpecP reading the xml file xmlName.
       //! You will need to that you need to call ProblemSpec::releaseDocument
       ProblemSpecP loadDocument(std::string xmlName);     

       //! creates the uda directory with a trailing version suffix
       void makeVersionedDir();

       void initSaveLabels(SchedulerP& sched, bool initTimestep);
       void initCheckpoints(SchedulerP& sched);

       //! helper for beginOutputTimestep - creates and writes
       //! the necessary directories and xml files to begin the 
       //! output timestep.
       void outputTimestep(Dir& dir, std::vector<SaveItem>& saveLabels,
           double time, double delt, const GridP& grid,
           std::string* pTimestepDir /* passed back */, bool hasGlobals = false);

       //! helper for finalizeTimestep - schedules a task for each var's output
       void scheduleOutputTimestep(std::vector<SaveItem>& saveLabels,
           const GridP& grid, SchedulerP& sched,
           bool isThisCheckpoint);

       //! Helper for finalizeTimestep - determines if, based on the current
       //! time and timestep, this will be an output or checkpoint timestep.
       void beginOutputTimestep(double time, double delt,
           const GridP& grid);

       //! After a timestep restart (delt adjusted), we need to see if we are 
       //! still an output timestep.
       virtual void reEvaluateOutputTimestep(double old_delt, double new_delt);

       //! helper for initializeOutput - writes the initial index.xml file,
       //! both setting the d_indexDoc var and writing it to disk.
       void createIndexXML(Dir& dir);

       //! helper for restartSetup - adds the restart field to index.xml
       void addRestartStamp(ProblemSpecP indexDoc, Dir& fromDir,
           int timestep);

       //! helper for restartSetup - copies the timestep directories AND
       //! timestep entries in index.xml
       void copyTimesteps(Dir& fromDir, Dir& toDir, int startTimestep,
           int maxTimestep, bool removeOld,
           bool areCheckpoints = false);

       //! helper for restartSetup - copies the reduction dat files to 
       //! new uda dir (from startTimestep to maxTimestep)
       void copyDatFiles(Dir& fromDir, Dir& toDir, int startTimestep,
           int maxTimestep, bool removeOld);

       //! add saved global (reduction) variables to index.xml
       void indexAddGlobals();

       //! string for uda dir (actual dir will have postpended numbers
       //! i.e., filebase.000
       std::string d_filebase;

       //! pointer to simulation state, to get timestep and time info
       SimulationStateP d_sharedState;

       //! set in finalizeTimestep for output tasks to see how far the 
       //! next timestep will go.  Stored as temp in case of a
       //! timestep restart.
       double d_tempElapsedTime; 

       // Only one of these should be non-zero.  The value is read from the .ups file.
       double d_outputInterval;         // In seconds.
       int d_outputTimestepInterval;    // Number of time steps.

       double d_nextOutputTime; // used when d_outputInterval != 0
       int d_nextOutputTimestep; // used when d_outputTimestepInterval != 0
       //int d_currentTimestep;
       Dir d_dir; //!< top of uda dir

       //! Represents whether this proc will output non-processor-specific
       //! files
       bool d_writeMeta;

       //! Whether or not to save the initialization timestep
       bool d_outputInitTimestep;

       //! last timestep dir (filebase.000/t#)
       std::string d_lastTimestepLocation;
       bool d_isOutputTimestep; //!< set if this is an output timestep
       bool d_isCheckpointTimestep; //!< set if a checkpoint timestep

       //! Whether or not particle vars are saved
       //! Requires p.x to be set
       bool d_saveParticleVariables; 

       //! Wheter or not p.x is saved 
       bool d_saveP_x;

       //double d_currentTime;

       //! d_saveLabelNames is a temporary list containing VarLabel
       //! names to be saved and the materials to save them for.  The
       //! information will be basically transferred to d_saveLabels or
       //! d_saveReductionLabels after mapping VarLabel names to their
       //! actual VarLabel*'s.
       std::list< SaveNameItem > d_saveLabelNames;
       std::vector< SaveItem > d_saveLabels;
       std::vector< SaveItem > d_saveReductionLabels;

       // for efficiency of SaveItem's
       ConsecutiveRangeSet prevMatls_;
       MaterialSetP prevMatlSet_;     

       //! d_checkpointLabelNames is a temporary list containing
       //! the names of labels to save when checkpointing
       std::vector< SaveItem > d_checkpointLabels;
       std::vector< SaveItem > d_checkpointReductionLabels;

       // Only one of these should be non-zero.
       double d_checkpointInterval;        // In seconds.
       int d_checkpointTimestepInterval;   // In seconds.

       // How much real time (in seconds) to wait for checkpoint can be
       // used with or without one of the above two.  WalltimeStart
       // cannot be used without walltimeInterval.
       int d_checkpointWalltimeStart;     // Amount of (real) time to wait before first checkpoint.
       int d_checkpointWalltimeInterval;  // Amount of (real) time to between checkpoints.

       //! How many checkpoint dirs to keep around
       int d_checkpointCycle;

       //! Top of checkpoints dir
       Dir d_checkpointsDir;

       //! List of current checkpoint dirs
       std::list<std::string> d_checkpointTimestepDirs;
       double d_nextCheckpointTime; //!< used when d_checkpointInterval != 0
       int d_nextCheckpointTimestep; //!< used when d_checkpointTimestepInterval != 0
       int d_nextCheckpointWalltime; //!< used when d_checkpointWalltimeInterval != 0

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

       //! index.xml
       ProblemSpecP d_XMLIndexDoc; 

       //! checkpoints/index.xml
       ProblemSpecP d_CheckpointXMLIndexDoc;

       ProblemSpecP d_upsFile;

       // Each level needs it's own data file handle 
       // and if we are outputting and checkpointing
       // at the same time we need two different sets.
       // Also store the filename for error-tracking purposes.

       std::map< int, std::pair<int, char*> > d_DataFileHandles;
       std::map< int, std::pair<int, char*> > d_CheckpointDataFileHandles;

       // Each level needs it's own XML Data Doc
       // and if we are outputting and checkpointing
       // at the same time we need two different sets.

       std::map< int, ProblemSpecP > d_XMLDataDocs;
       std::map< int, ProblemSpecP > d_CheckpointXMLDataDocs;


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

       std::string TranslateVariableType( std::string type, bool isThisCheckpoint );


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

       //! The number of levels the DA knows about.  If this changes, we need to 
       //! redo output and Checkpoint tasks.
       int d_numLevelsInOutput;

#if SCI_ASSERTION_LEVEL >= 2
       //! double-check to make sure that DA::output is only called once per level per processor per type
       std::vector<bool> d_outputCalled;
       std::vector<bool> d_checkpointCalled;
       bool d_checkpointReductionCalled;
#endif
       Mutex d_outputLock;

       DataArchiver(const DataArchiver&);
       DataArchiver& operator=(const DataArchiver&);
      
   };

} // End namespace Uintah

#endif
