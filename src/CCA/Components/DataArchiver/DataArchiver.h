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

#ifndef UINTAH_HOMEBREW_DataArchiver_H
#define UINTAH_HOMEBREW_DataArchiver_H

#include <CCA/Ports/Output.h>
#include <CCA/Ports/PIDXOutputContext.h>
#include <Core/Parallel/MasterLock.h>
#include <Core/Parallel/UintahParallelComponent.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/MaterialSetP.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Util/Assert.h>
#include <Core/OS/Dir.h>
#include <Core/Containers/ConsecutiveRangeSet.h>

//#include <mutex>

namespace Uintah {

class DataWarehouse;


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
       DataArchiver( const ProcessorGroup * myworld, int udaSuffix = -1 );
       virtual ~DataArchiver();

       static bool d_wereSavesAndCheckpointsInitialized;

       //! Sets up when the DataArchiver will output and what data, according
       //! to params.  Also stores state to keep track of time and timesteps
       //! in the simulation.  (If you only need to use DataArchiver to copy 
       //! data, then you can pass a nullptr SimulationState
       virtual void problemSetup( const ProblemSpecP    & params,
				  const ProblemSpecP    & restart_prob_spec,
                                        SimulationState * state );

       virtual void outputProblemSpec( ProblemSpecP & root_ps );

       //! This function will set up the output for the simulation.  As part
       //! of this it will output the input.xml and index.xml in the uda
       //! directory.  Call after calling all problemSetups.
       virtual void initializeOutput( const ProblemSpecP & params );

       //! Call this when restarting from a checkpoint after calling
       //! problemSetup.  This will copy timestep directories and dat
       //! files up to the specified timestep from restartFromDir if
       //! fromScratch is false and will set time and timestep variables
       //! appropriately to continue smoothly from that timestep.
       //! If timestep is negative, then all timesteps will get copied
       //! if they are to be copied at all (fromScratch is false).
       virtual void restartSetup( Dir    & restartFromDir,
                                  int      startTimestep,
                                  int      timestep,
                                  double   time,
                                  bool     fromScratch,
                                  bool     removeOldDir );

       //! Call this after problemSetup it will copy the data and
       //! checkpoint files ignore dumping reduction variables.
       virtual void reduceUdaSetup( Dir& fromDir );

       //! Copy a section between udas .
       void copySection( Dir & fromDir, Dir & toDir, const std::string & file, const std::string & section );

       //! Copy a section from another uda's to our index.xml.
       void copySection( Dir & fromDir, const std::string & section ) { copySection(fromDir, d_dir, "index.xml", section); }

       //! Checks to see if this is an output timestep. 
       //! If it is, setup directories and xml files that we need to output.
       //! Call once per timestep, and if recompiling,
       //! after all the other tasks are scheduled.
       virtual void finalizeTimestep(       double       time,
                                            double       delt,
                                      const GridP      &,
                                            SchedulerP &,
                                            bool         recompile = false );
           
       //! schedule the output tasks if we are recompiling the taskgraph.  
       virtual void sched_allOutputTasks(       double       delt,
                                          const GridP      & /* grid */,
                                                SchedulerP & /* scheduler */,
                                                bool         recompile = false );
                                      
       //! Call this after a timestep restart where delt is adjusted to
       //! make sure there still will be output and/or checkpoint timestep
       virtual void reevaluate_OutputCheckPointTimestep(double time);

     //! Call this after the timestep has been executed to find the
       //! next time step to output
       virtual void findNext_OutputCheckPointTimestep( double time,
						       bool restart = false );

       //! write meta data to xml files 
       //! Call after timestep has completed.
       virtual void writeto_xml_files( double delt, const GridP & grid );
       virtual void writeto_xml_files( std::map< std::string, std::pair<std::string, std::string> > &modifiedVars );

       //! Returns as a string the name of the top of the output directory.
       virtual const std::string getOutputLocation() const;

       //! Asks if we need to recompile the task graph.
       virtual bool needRecompile( double time, double dt, const GridP & grid );

       //! The task that handles the outputting.  Scheduled in
       //! finalizeTimestep.  Handles outputs and checkpoints and
       //! differentiates between them in the last argument.  Outputs
       //! as binary the data acquired from VarLabel in p_dir.
       void outputVariables( const ProcessorGroup *,
                             const PatchSubset    * patch,
                             const MaterialSubset * matls, 
                                   DataWarehouse  * old_dw,
                                   DataWarehouse  * new_dw, 
                                   int              type );

       //! Task that handles outputting non-checkpoint reduction variables.
       //! Scheduled in finalizeTimestep.
       void outputReductionVars( const ProcessorGroup *,
                                 const PatchSubset    * patch,
                                 const MaterialSubset * matls,
                                       DataWarehouse  * old_dw,
                                       DataWarehouse  * new_dw );

       //! Get the time the next output will occur
       virtual double getNextOutputTime() const { return d_nextOutputTime; }

       //! Get the timestep the next output will occur
       virtual int  getNextOutputTimestep() const { return d_nextOutputTimestep; }
       virtual void postponeNextOutputTimestep() { d_nextOutputTimestep++; } // Pushes output back by one timestep.

       //! Get the time/timestep/walltime of the next checkpoint will occur
       virtual double getNextCheckpointTime()     const { return d_nextCheckpointTime; }
       virtual int    getNextCheckpointTimestep() const { return d_nextCheckpointTimestep; }
       virtual int    getNextCheckpointWalltime() const { return d_nextCheckpointWalltime; }

       //! Returns true if data will be output this timestep
       virtual bool isOutputTimestep() const { return d_isOutputTimestep; }

       //! Returns true if data will be checkpointed this timestep
       virtual bool isCheckpointTimestep() const { return d_isCheckpointTimestep; }

       //! Get the directory of the current time step for outputting info.
       virtual const std::string& getLastTimestepOutputLocation() const { return d_lastTimestepLocation; }

       bool isLabelSaved ( const std::string & label ) const;
       
       //! Allow a component to define the output and checkpoint interval on the fly.
       void updateOutputInterval( double inv );
       void updateOutputTimestepInterval( int inv );
       void updateCheckpointInterval( double inv );
       void updateCheckpointTimestepInterval( int inv );

       double getOutputInterval()         const { return d_outputInterval; }
       int    getOutputTimestepInterval() const { return d_outputTimestepInterval; }

       double getCheckpointInterval()         const { return d_checkpointInterval; }
       int    getCheckpointTimestepInterval() const { return d_checkpointTimestepInterval; }
       int    getCheckpointWalltimeInterval() const { return d_checkpointWalltimeInterval; }

       bool   savingAsPIDX() const { return ( d_outputFileFormat == PIDX ); } 

       // Instructs the DataArchive to save data using the original UDA format or using PIDX.
       void   setSaveAsUDA()  { d_outputFileFormat = UDA; }
       void   setSaveAsPIDX() { d_outputFileFormat = PIDX; }

       //! Called by In-situ VisIt to force the dump of a time step's data.
       void outputTimestep( double time, double delt,
                            const GridP& grid, SchedulerP& sched );

       void checkpointTimestep( double time, double delt,
                                const GridP& grid, SchedulerP& sched );

     public:

       //! problemSetup parses the ups file into a list of these
       //! (d_saveLabelNames)
       struct SaveNameItem {
         std::string         labelName;
         std::string         compressionMode;
         ConsecutiveRangeSet matls;
         ConsecutiveRangeSet levels;
       };

       class SaveItem {
         public:
         
           void setMaterials(      int                   level, 
                             const ConsecutiveRangeSet & matls,
                                   ConsecutiveRangeSet & prevMatls,
                                   MaterialSetP        & prevMatlSet );

           MaterialSet* getMaterialSet( int level ) { return matlSet[level].get_rep(); }
           
           const MaterialSubset* getMaterialSubset( const Level * level );
           
           const VarLabel* label;
           std::map<int, MaterialSetP> matlSet;
       };

     private:
     
       enum outputFileFormat { UDA, PIDX };
       outputFileFormat d_outputFileFormat;
      
       //__________________________________
       //         PIDX related
       //! output the all of the saveLabels in PIDX format
       size_t
          saveLabels_PIDX( std::vector< SaveItem >   & saveLabels,
                           const ProcessorGroup      * pg,
                           const PatchSubset         * patches,      
                           DataWarehouse             * new_dw,          
                           int                         type,
                           const TypeDescription::Type TD,
                           Dir                         ldir,        // uda/timestep/levelIndex
                           const std::string         & dirName,     // CCVars, SFC*Vars
                           ProblemSpecP              & doc );
                           
       //! returns a vector of SaveItems with a common type description
       std::vector<DataArchiver::SaveItem> 
          findAllVariableTypes( std::vector< SaveItem >& saveLabels,
                                 const TypeDescription::Type TD );
      
       //! bulletproofing so user can't save unsupported var type
       void isVarTypeSupported( std::vector< SaveItem >& saveLabels,
                                std::vector<TypeDescription::Type> pidxVarTypes );
           
       void createPIDX_dirs( std::vector< SaveItem >& saveLabels,
                             Dir& levelDir );

       // Writes out the <Grid> and <Data> sections into the timestep.xml file by creating a DOM and then writing it out.
       void writeGridOriginal(   const bool hasGlobals, const GridP & grid, ProblemSpecP rootElem );

       // Writes out the <Grid> and <Data> sections (respectively) to separate files (that are associated with timestep.xml) using a XML streamer.
       void writeGridTextWriter( const bool hasGlobals, const std::string & grid_path, const GridP & grid );
       void writeDataTextWriter( const bool hasGlobals, const std::string & data_path, const GridP & grid,
                                 const std::vector< std::vector<bool> > & procOnLevel );

       // Writes out the <Grid> section (associated with timestep.xml) to a separate binary file.
       void writeGridBinary(     const bool hasGlobals, const std::string & grid_path, const GridP & grid );

       //__________________________________
       //! returns a ProblemSpecP reading the xml file xmlName.
       //! You will need to that you need to call ProblemSpec::releaseDocument
       ProblemSpecP loadDocument( const std::string & xmlName );

       //! creates the uda directory with a trailing version suffix
       void makeVersionedDir();

       void initSaveLabels(  SchedulerP& sched, bool initTimestep );
       void initCheckpoints( SchedulerP& sched );

       //! helper for beginOutputTimestep - creates and writes
       //! the necessary directories and xml files to begin the 
       //! output timestep.
       void makeTimestepDirs( Dir& dir,
                              std::vector<SaveItem>& saveLabels,
                              const GridP& grid,
                              std::string* pTimestepDir );

       PIDXOutputContext::PIDX_flags d_PIDX_flags; // Contains the knobs & switches
#if HAVE_PIDX       

       std::vector<MPI_Comm> d_pidxComms; // Array of MPI Communicators for PIDX usage...
       
       //! creates communicator every AMR level required for PIDX
       void createPIDXCommunicator(       std::vector<SaveItem> & saveLabels,
                                    const GridP                 & grid, 
                                          SchedulerP            & sched,
                                          bool                    isThisACheckpoint );
#endif

       //! helper for finalizeTimestep - schedules a task for each var's output
       void scheduleOutputTimestep(std::vector<SaveItem>& saveLabels,
                                   const GridP& grid, 
                                   SchedulerP& sched,
                                   bool isThisCheckpoint);

       //! Helper for finalizeTimestep - determines if, based on the current
       //! time and timestep, this will be an output or checkpoint timestep.
       void beginOutputTimestep( const double   time,
                                 const double   delt,
                                 const GridP  & grid );

       //! helper for initializeOutput - writes the initial index.xml file,
       //! both setting the d_indexDoc var and writing it to disk.
       void createIndexXML(Dir& dir);

       //! helper for restartSetup - adds the restart field to index.xml
       void addRestartStamp( ProblemSpecP   indexDoc,
                             Dir          & fromDir,
                             int            timestep );

       //! helper for restartSetup - copies the timestep directories AND
       //! timestep entries in index.xml
       void copyTimesteps( Dir  & fromDir,
                           Dir  & toDir,
                           int    startTimestep,
                           int    maxTimestep,
                           bool   removeOld,
                           bool   areCheckpoints = false );

       //! helper for restartSetup - copies the reduction dat files to 
       //! new uda dir (from startTimestep to maxTimestep)
       void copyDatFiles( Dir & fromDir,
                          Dir & toDir,
                          int   startTimestep,
                          int   maxTimestep,
                          bool  removeOld );

       //! add saved global (reduction) variables to index.xml
       void indexAddGlobals();

       // setupLocalFileSystems() and setupSharedFileSystem() are used to 
       // create the UDA (versioned) directory.  setupLocalFileSystems() is
       // old method of determining which ranks should output UDA
       // metadata and handles the case when each node has its own local file system
       // (as opposed to a shared file system across all nodes). setupLocalFileSystems()
       // will only be used if specifically turned on via a
       // command line arg to sus when running using MPI.
       void setupLocalFileSystems();
       void setupSharedFileSystem(); // Verifies that all ranks see a shared FS.
       void saveSVNinfo();

       //! string for uda dir (actual dir will have postpended numbers
       //! i.e., filebase.000
       std::string d_filebase;

       //! pointer to simulation state, to get timestep and time info
       SimulationStateP d_sharedState;

       // Only one of these should be non-zero.  The value is read
       // from the .ups file.
       double d_outputInterval;         // In seconds.
       int    d_outputTimestepInterval; // Number of time steps.

       double d_nextOutputTime;         // used when d_outputInterval != 0
       int    d_nextOutputTimestep;     // used when d_outputTimestepInterval != 0

       bool   d_outputLastTimestep;     // Output the last time step.
     
       //int d_currentTimestep;
       Dir    d_dir;                    //!< top of uda dir

       //! Represents whether this proc will output non-processor-specific
       //! files
       bool   d_writeMeta;

       //! Whether or not to save the initialization timestep
       bool   d_outputInitTimestep;

       //! last timestep dir (filebase.000/t#)
       std::string d_lastTimestepLocation;
       bool        d_isOutputTimestep;         //!< set if this is an output timestep
       bool        d_isCheckpointTimestep;     //!< set if a checkpoint timestep

       //! Whether or not particle vars are saved
       //! Requires p.x to be set
       bool d_saveParticleVariables; 

       //! Wheter or not p.x is saved 
       bool d_saveP_x;
     
       std::string d_particlePositionName;

       //double d_currentTime;

       //! d_saveLabelNames is a temporary list containing VarLabel
       //! names to be saved and the materials to save them for.  The
       //! information will be basically transferred to d_saveLabels or
       //! d_saveReductionLabels after mapping VarLabel names to their
       //! actual VarLabel*'s.
       std::list< SaveNameItem > d_saveLabelNames;
       std::vector< SaveItem >   d_saveLabels;
       std::vector< SaveItem >   d_saveReductionLabels;

       // for efficiency of SaveItem's
       ConsecutiveRangeSet d_prevMatls;
       MaterialSetP d_prevMatlSet;     

       //! d_checkpointLabelNames is a temporary list containing
       //! the names of labels to save when checkpointing
       std::vector< SaveItem > d_checkpointLabels;
       std::vector< SaveItem > d_checkpointReductionLabels;

       // Only one of these should be non-zero.
       double d_checkpointInterval;        // In seconds.
       int d_checkpointTimestepInterval;   // Number of time steps.

       // How much real time (in seconds) to wait for checkpoint can be
       // used with or without one of the above two.  WalltimeStart
       // cannot be used without walltimeInterval.
       int d_checkpointWalltimeStart;     // Amount of (real) time (in seconds) to wait before first checkpoint.
       int d_checkpointWalltimeInterval;  // Amount of (real) time (in seconds) to between checkpoints.

       bool d_checkpointLastTimestep;     // Checkpoint the last time step.

       //! How many checkpoint dirs to keep around
       int d_checkpointCycle;

       //! Top of checkpoints dir
       Dir d_checkpointsDir;

       //! List of current checkpoint dirs
       std::list<std::string> d_checkpointTimestepDirs;
       double d_nextCheckpointTime;      //!< used when d_checkpointInterval != 0.          Simulation time (seconds (and fractions there of))
       int    d_nextCheckpointTimestep;  //!< used when d_checkpointTimestepInterval != 0.  Integer - time step
       int    d_nextCheckpointWalltime;  //!< used when d_checkpointWalltimeInterval != 0.  Integer Seconds.

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

       //__________________________________
       //  reduceUda related
       //  used for migrating timestep directories
       std::map< int, int> d_restartTimestepIndicies;
       bool d_usingReduceUda;
       
       Dir d_fromDir;                   // keep track of the original uda
       void copy_outputProblemSpec(Dir& fromDir, Dir& toDir);
       
       // returns either the top level timestep or if reduceUda is used
       // a value from the index.xml file
       int getTimestepTopLevel();

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
       using Mutex = Uintah::MasterLock;
       Mutex d_outputLock;

       DataArchiver(const DataArchiver&);
       DataArchiver& operator=(const DataArchiver&);
      
   };

} // End namespace Uintah

#endif
