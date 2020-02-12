/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

#include <CCA/Components/Schedulers/RuntimeStatsEnum.h>

#include <Core/Containers/ConsecutiveRangeSet.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/MaterialSetP.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/MaterialManagerP.h>
#include <Core/OS/Dir.h>
#include <Core/Parallel/MasterLock.h>
#include <Core/Parallel/UintahParallelComponent.h>
#include <Core/Util/Assert.h>

namespace Uintah {

class DataWarehouse;
class ApplicationInterface;
class LoadBalancer;

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
  class DataArchiver : public UintahParallelComponent, public Output {

  public:
    DataArchiver( const ProcessorGroup * myworld, int udaSuffix = -1 );
    virtual ~DataArchiver();

    // Methods for managing the components attached via the ports.
    virtual void setComponents( UintahParallelComponent *parent ) {};
    virtual void getComponents();
    virtual void releaseComponents();
    
    static bool m_wereSavesAndCheckpointsInitialized;

    //! Sets up when the DataArchiver will output and what data, according
    //! to params.  If you only need to use DataArchiver to copy 
    //! data, then MaterialManager can be a nullptr.
    virtual void problemSetup( const ProblemSpecP    & params,
                               const ProblemSpecP    & restart_prob_spec,
                               const MaterialManagerP& materialManager );

    virtual void outputProblemSpec( ProblemSpecP & root_ps );

    //! This function will set up the output for the simulation.  As part
    //! of this it will output the input.xml and index.xml in the uda
    //! directory.  Call after calling all problemSetups.
    virtual void initializeOutput( const ProblemSpecP & params,
                                   const GridP& grid );

    //! Call this when restarting from a checkpoint after calling
    //! problemSetup.  This will copy time step directories and dat
    //! files up to the specified time step from restartFromDir if
    //! fromScratch is false and will set time and time step variables
    //! appropriately to continue smoothly from that time step.
    //! If time step is negative, then all time steps will get copied
    //! if they are to be copied at all (fromScratch is false).
    virtual void restartSetup( const Dir    & restartFromDir,
                               const int      startTimeStep,
                               const int      timeStep,
                               const double   simTime,
                               const bool     fromScratch,
                               const bool     removeOldDir );

    //! Call this after problemSetup it will copy the data and
    //! checkpoint files ignore dumping reduction/sole variables.
    virtual void postProcessUdaSetup( Dir& fromDir );

    //! Copy a section between udas .
    void copySection( const Dir & fromDir, const Dir & toDir, const std::string & file, const std::string & section );

    //! Copy a section from another uda's to our index.xml.
    void copySection( Dir & fromDir, const std::string & section ) { copySection( fromDir, m_outputDir, "index.xml", section ); }

    //! Checks to see if this is an output time step. 
    //! If it is, setup directories and xml files that we need to output.
    //! Call once per time step, and if recompiling,
    //! after all the other tasks are scheduled.
    virtual void finalizeTimeStep( const GridP      & /* grid */,
                                   SchedulerP & /* scheduler */,
                                   bool         recompile = false );
           
    //! schedule the output tasks if we are recompiling the taskgraph.  
    virtual void sched_allOutputTasks( const GridP      & /* grid */,
                                       SchedulerP & /* scheduler */,
                                       bool         recompile = false );
                                      
    //! Call this after the time step has been executed to find the
    //! next time step to output
    virtual void findNext_OutputCheckPointTimeStep( const bool restart,
                                                    const GridP& grid );

    //! Called after a time step recompute where delta t is adjusted
    //! to make sure an output and/or checkpoint time step is needed.
    virtual void recompute_OutputCheckPointTimeStep();

    //! write meta data to xml files 
    //! Call after time step has completed.
    virtual void writeto_xml_files( const GridP & grid );

    virtual void writeto_xml_files( std::map< std::string,
                                    std::pair<std::string,
                                    std::string> > &modifiedVars );

    //! Returns as a string the name of the top of the output directory.
    virtual const std::string getOutputLocation() const;

    //! Normally saved vars are scrubbed if not needed for the next
    //! time step. By pass scubbing when running in situ or if wanting
    //! to save the previous time step.
    virtual void setScrubSavedVariables( bool val ) { scrubSavedVariables = val; };

    //! Asks if the task graph needs to be recompiled.
    virtual bool needRecompile( const GridP & grid );

    virtual void recompile(const GridP& grid);

    //! The task that handles the outputting.  Scheduled in
    //! finalizeTimeStep.  Handles outputs and checkpoints and
    //! differentiates between them in the last argument.  Outputs
    //! as binary the data acquired from VarLabel in p_dir.
    void outputVariables( const ProcessorGroup *,
                          const PatchSubset    * patch,
                          const MaterialSubset * matls, 
                          DataWarehouse  * old_dw,
                          DataWarehouse  * new_dw, 
                          int              type );

    //! Task that handles outputting non-checkpoint global
    //! reduction/sole variables.  Scheduled in finalizeTimeStep.
    void outputGlobalVars( const ProcessorGroup *,
                           const PatchSubset    * patch,
                           const MaterialSubset * matls,
                                 DataWarehouse  * old_dw,
                                 DataWarehouse  * new_dw );

    //! Get the time the next output will occur
    virtual double getNextOutputTime() const { return m_nextOutputTime; }
    //! Get the time step the next output will occur
    virtual int  getNextOutputTimeStep() const { return m_nextOutputTimeStep; }
    // Pushes output back by one time step.
    virtual void postponeNextOutputTimeStep() { ++m_nextOutputTimeStep; }

    //! Get the time/time step/wall time of the next checkpoint will occur
    virtual double getNextCheckpointTime()     const { return m_nextCheckpointTime; }
    virtual int    getNextCheckpointTimeStep() const { return m_nextCheckpointTimeStep; }
    virtual int    getNextCheckpointWallTime() const { return m_nextCheckpointWallTime; }

    //! Returns true if data will be output this time step
    virtual void setOutputTimeStep( bool val, const GridP & grid );
    virtual bool isOutputTimeStep() const { return m_isOutputTimeStep; }

    //! Returns true if data will be checkpointed this time step
    virtual void setCheckpointTimeStep( bool val, const GridP & grid );
    virtual bool isCheckpointTimeStep() const { return m_isCheckpointTimeStep; }

    //! Get the directory of the current time step for outputting info.
    virtual const std::string& getLastTimeStepOutputLocation() const { return m_lastTimeStepLocation; }

    bool isLabelSaved ( const std::string & label ) const;
       
    //! Allow a component to adjust the output and checkpoint
    //! interval on the fly.
    void   setOutputInterval( double inv );
    double getOutputInterval()         const { return m_outputInterval; }
    void   setOutputTimeStepInterval( int inv );
    int    getOutputTimeStepInterval() const { return m_outputTimeStepInterval; }

    void   setCheckpointInterval( double inv );
    double getCheckpointInterval()         const { return m_checkpointInterval; }
    void   setCheckpointTimeStepInterval( int inv );
    int    getCheckpointTimeStepInterval() const { return m_checkpointTimeStepInterval; }

    void   setCheckpointWallTimeInterval( int inv );
    int    getCheckpointWallTimeInterval() const { return m_checkpointWallTimeInterval; }

    bool   savingAsPIDX() const { return ( m_outputFileFormat == PIDX ); } 

    // Instructs the DataArchive to save data using the original UDA format or using PIDX.
    void   setSaveAsUDA()  { m_outputFileFormat = UDA; }
    void   setSaveAsPIDX() { m_outputFileFormat = PIDX; }

    //! Called by in situ VisIt to force the dump of a time step's data.
    void outputTimeStep( const GridP& grid,
                         SchedulerP& sched,
                         bool previous );

    void checkpointTimeStep( const GridP& grid,
                             SchedulerP& sched,
                             bool previous );

    void maybeLastTimeStep( bool val ) { m_maybeLastTimeStep = val; };
    bool maybeLastTimeStep() { return m_maybeLastTimeStep; };
     
    void setSwitchState(bool val) { m_switchState = val; }
    bool getSwitchState() const { return m_switchState; }   
    
    void   setElapsedWallTime( double val );
    double getElapsedWallTime() const { return m_elapsedWallTime; };
     
    void   setCheckpointCycle( int val );
    double getCheckpointCycle() const { return m_checkpointCycle; };
     
    void setUseLocalFileSystems(bool val) { m_useLocalFileSystems = val; };
    bool getUseLocalFileSystems() const { return m_useLocalFileSystems; };
     
    void setRuntimeStats( ReductionInfoMapper< RuntimeStatsEnum, double > *runtimeStats) { m_runtimeStats = runtimeStats; };
     
    // Returns trus if an output or checkpoint exists for the time step
    bool outputTimeStepExists( unsigned int ts );
    bool checkpointTimeStepExists( unsigned int ts );
    
    //! problemSetup parses the ups file into a list of these
    //! (m_saveLabelNames)
    struct SaveNameItem {
      std::string         labelName;
      std::string         compressionMode;
      ConsecutiveRangeSet matls;
      ConsecutiveRangeSet levels;
    };

    class SaveItem {
    public:

      void setMaterials( const int                   level, 
                         const ConsecutiveRangeSet & matls,
                               ConsecutiveRangeSet & prevMatls,
                               MaterialSetP        & prevMatlSet );

      const MaterialSet*    getMaterialSet( int level ) const { return matlSet.at( level ).get_rep(); }
      const MaterialSubset* getMaterialSubset( const Level * level ) const;

      const VarLabel* label;
      std::map<int, MaterialSetP> matlSet;
    };

  private:
     
    enum outputFileFormat { UDA, PIDX };
    outputFileFormat m_outputFileFormat { UDA };
      
    //__________________________________
    //         PIDX related
    //! output the all of the saveLabels in PIDX format
    size_t
    saveLabels_PIDX( const ProcessorGroup      * pg,
                     const PatchSubset         * patches,      
                     DataWarehouse       * new_dw,          
                     int                   type,
                     std::vector< SaveItem > & saveLabels,
                     const TypeDescription::Type TD,
                     Dir                   ldir,        // uda/timeStep/levelIndex
                     const std::string         & dirName,     // CCVars, SFC*Vars
                     ProblemSpecP        & doc );
                           
    //! Searches through "saveLabels" and returns all the SaveItems that are of the same "type".
    std::vector<DataArchiver::SaveItem> 
    findAllVariablesWithType( const std::vector< SaveItem > & saveLabels,
                              const TypeDescription::Type     type );
      
    //! bulletproofing so user can't save unsupported var type
    void isVarTypeSupported( const std::vector< SaveItem >              & saveLabels,
                             const std::vector< TypeDescription::Type > & pidxVarTypes );
           
    // Writes out the <Grid> and <Data> sections into the
    // timestep.xml file by creating a DOM and then writing it out.
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

    void initSaveLabels(  SchedulerP & sched, bool initTimeStep );
    void initCheckpoints( const SchedulerP & sched );

    //! helper for beginOutputTimeStep - creates and writes
    //! the necessary directories and xml files to begin the 
    //! output time step.
    void makeTimeStepDirs(      Dir& dir,
                                std::vector<SaveItem>& saveLabels,
                          const GridP& grid,
                                std::string* pTimeStepDir );

    PIDXOutputContext::PIDX_flags m_PIDX_flags; // Contains the knobs & switches
#if HAVE_PIDX       

    std::vector<MPI_Comm> m_pidxComms; // Array of MPI Communicators for PIDX usage...
       
    //! creates communicator every AMR level required for PIDX
    void createPIDXCommunicator(       std::vector<SaveItem> & saveLabels,
                                 const GridP                 & grid, 
                                       SchedulerP            & sched,
                                       bool                    isThisACheckpoint );

    // Timestep # of the last time we saved "timestep.xml". -1 == not
    // yet saved. Only save timestep.xml as needed (ie, when a
    // regrid occurs), otherwise a given timestep will refer (symlink)
    // to the last time it was saved.  Note, this is in reference to
    // IO timesteps.  We always generate and save timestep.xml for
    // Checkpoint output.
#endif
    int m_lastOutputOfTimeStepXML = -1; 

    //! helper for finalizeTimeStep - schedules a task for each var's output
    void scheduleOutputTimeStep(       std::vector<SaveItem> & saveLabels,
                                 const GridP                 & grid, 
                                       SchedulerP            & sched,
                                       bool                    isThisCheckpoint );

    //! Helper for finalizeTimeStep - determines if, based on the current
    //! time and time step, this will be an output or checkpoint time step.
    void beginOutputTimeStep( const GridP & grid );

    //! helper for initializeOutput - writes the initial index.xml file,
    //! both setting the m_indexDoc var and writing it to disk.
    void createIndexXML(Dir& dir);

    //! helper for restartSetup - adds the restart field to index.xml
    void addRestartStamp(       ProblemSpecP   indexDoc,
                          const Dir          & fromDir,
                          const int            timestep );

    //! helper for restartSetup - copies the time step directories AND
    //! time step entries in index.xml
    void copyTimeSteps( const Dir  & fromDir,
                        const Dir  & toDir,
                        const int    startTimeStep,
                        const int    maxTimeStep,
                        const bool   removeOld,
                        const bool   areCheckpoints = false );

    //! helper for restartSetup - copies the global dat files to 
    //! new uda dir (from startTimeStep to maxTimeStep)
    void copyDatFiles( const Dir & fromDir,
                       const Dir & toDir,
                       const int   startTimeStep,
                       const int   maxTimeStep,
                       const bool  removeOld );

    //! add saved global (reduction/sole) variables to index.xml
    void indexAddGlobals();

    // setupLocalFileSystems() and setupSharedFileSystem() are used to
    // create the UDA (versioned) directory.  setupLocalFileSystems()
    // is old method of determining which ranks should output UDA
    // metadata and handles the case when each node has its own local
    // file system (as opposed to a shared file system across all
    // nodes). setupLocalFileSystems() will only be used if
    // specifically turned on via a command line arg to sus when
    // running using MPI.
    void setupLocalFileSystems();
    void setupSharedFileSystem(); // Verifies that all ranks see a shared FS.
    void saveSVNinfo();

    //! string for uda dir (actual dir will have postpended numbers
    //! i.e., filebase.000
    std::string m_filebase { "" };

    ApplicationInterface * m_application{nullptr};
    LoadBalancer         * m_loadBalancer{nullptr};
    
    //! pointer to simulation state, to get time step and time info
    MaterialManagerP m_materialManager;

    // Only one of these should be non-zero.  The value is read
    // from the .ups file.
    double m_outputInterval {0};         // In seconds.
    int    m_outputTimeStepInterval {0}; // Number of time steps.

    double m_nextOutputTime {0};     // used when m_outputInterval != 0
    int    m_nextOutputTimeStep {0}; // used when m_outputTimeStepInterval != 0

    bool   m_outputLastTimeStep {false}; // Output the last time step.
     
    Dir    m_outputDir;                    //!< top of uda dir

    //! Represents whether this proc will output non-processor-specific
    //! files
    bool   m_writeMeta {false};

    //! Whether or not to save the initialization time step
    bool   m_outputInitTimeStep {false};

    //! last timestep dir (filebase.000/t#)
    std::string m_lastTimeStepLocation {"invalid"};

    //! List of current output dirs
    std::list<std::string> m_outputTimeStepDirs;
    
    bool m_isOutputTimeStep {false};      //!< set if an output time step
    bool m_isCheckpointTimeStep {false};  //!< set if a checkpoint time step

    //! Whether or not particle vars are saved
    //! Requires p.x to be set
    bool m_saveParticleVariables {false}; 

    //! Wheter or not p.x is saved 
    bool m_saveP_x {false};
     
    std::string m_particlePositionName{ "p.x" };

    double m_elapsedWallTime {0};
    bool m_maybeLastTimeStep{false};

    bool m_switchState{false};

    // Tells the data archiver that we are running with each MPI node
    // having a separate file system.  (Simulation defaults to running
    // on a shared file system.)
    bool m_useLocalFileSystems{false};

    //! m_saveLabelNames is a temporary list containing VarLabel
    //! names to be saved and the materials to save them for.  The
    //! information will be basically transferred to m_saveLabels or
    //! m_saveGlobalLabels after mapping VarLabel names to their
    //! actual VarLabel*'s.
    std::list< SaveNameItem > m_saveLabelNames;
    std::vector< SaveItem >   m_saveLabels;
    std::vector< SaveItem >   m_saveGlobalLabels;

    // for efficiency of SaveItem's
    ConsecutiveRangeSet m_prevMatls;
    MaterialSetP m_prevMatlSet {nullptr};

    //! m_checkpointLabelNames is a temporary list containing
    //! the names of labels to save when checkpointing
    std::vector< SaveItem > m_checkpointLabels;
    std::vector< SaveItem > m_checkpointGlobalLabels;

    // Only one of these should be non-zero.
    double m_checkpointInterval {0};        // In seconds.
    int m_checkpointTimeStepInterval {0};   // Number of time steps.

    // How much real time (in seconds) to wait before writting the
    // first checkpoint. Can be used with or without one of the above
    // two.  WallTimeStart cannot be used without WallTimeInterval.
    int m_checkpointWallTimeStart {0};
    int m_checkpointWallTimeInterval {0};

    bool m_checkpointLastTimeStep {false}; // Checkpoint the last time step.

    //! How many checkpoint dirs to keep around
    int m_checkpointCycle {2};

    //! Top of checkpoints dir
    Dir m_checkpointsDir {""};

    //! List of current checkpoint dirs
    std::list<std::string> m_checkpointTimeStepDirs;
    
    //!< used when m_checkpointInterval != 0. Simulation time in seconds.
    double m_nextCheckpointTime {0};
    //!< used when m_checkpointTimeStepInterval != 0.  Integer - time step
    int    m_nextCheckpointTimeStep {0};
    //!< used when m_checkpointWallTimeInterval != 0.  Integer seconds.
    int    m_nextCheckpointWallTime {0};

    // 
    bool m_outputPreviousTimeStep     {false};
    bool m_checkpointPreviousTimeStep {false};
    
    MaterialSubset *m_tmpMatSubset {nullptr};

    //-----------------------------------------------------------
    // RNJ - 
    //
    // In order to avoid having to open and close index.xml,
    // p<xxxxx>.xml, and p<xxxxx>.data when we want to update
    // each variable, we will keep track of some XML docs and
    // file handles and only open and close them once per
    // time step if they are needed.
    //-----------------------------------------------------------

    // We need to have two separate XML Index Docs
    // because it is possible to do an output
    // and a checkpoint at the same time.

    //! index.xml
    ProblemSpecP m_XMLIndexDoc {nullptr};

    //! checkpoints/index.xml
    ProblemSpecP m_CheckpointXMLIndexDoc {nullptr};

    ProblemSpecP m_upsFile {nullptr};

    // Each level needs it's own data file handle 
    // and if we are outputting and checkpointing
    // at the same time we need two different sets.
    // Also store the filename for error-tracking purposes.

    std::map< int, std::pair<int, char*> > m_DataFileHandles;
    std::map< int, std::pair<int, char*> > m_CheckpointDataFileHandles;

    // Each level needs it's own XML Data Doc
    // and if we are outputting and checkpointing
    // at the same time we need two different sets.

    std::map< int, ProblemSpecP > m_XMLDataDocs;
    std::map< int, ProblemSpecP > m_CheckpointXMLDataDocs;

    // Hacky variable to ensure that PIDX checkpoint and IO tasks that
    // happen to fall on the same time step run in a serialized manner
    // (as it appears that PIDX is not thread safe).  If there was a
    // better way to synchronize tasks, we should do that...
    VarLabel * m_sync_io_label;

    //__________________________________
    //  PostProcessUda related
    bool m_doPostProcessUda {false};
    //  Used for migrating restart time step directories.
    std::map< int, int> m_restartTimeStepIndicies;
       
    Dir m_fromDir {""};              // keep track of the original uda
    void copy_outputProblemSpec(Dir& fromDir, Dir& toDir);
       
    // Returns either the top level time step or if postProcessUda is used
    // a value from the index.xml file
    int getTimeStepTopLevel();

    //! Normally saved vars are scrubbed if not needed for the next
    //! time step. By pass scubbing when running in situ or if wanting
    //! to save the previous time step.
    bool scrubSavedVariables { true };
    
    //-----------------------------------------------------------
    // RNJ - 
    //
    // If the <DataArchiver> section of the .ups file contains:
    //
    //   <outputDoubleAsFloat />
    //
    // Then we will set the m_OutputDoubleAsFloat boolean to true
    // and we will try to output floats instead of doubles.
    //
    // NOTE: This does not affect checkpoints as they will
    //       always be outputting doubles for accuracy.
    //-----------------------------------------------------------

    bool m_outputDoubleAsFloat {false};

    //-----------------------------------------------------------

    // These four variables affect the global var output only.
    
    // For outputing the sim time and/or time step with the global vars
    bool m_outputGlobalVarsTimeStep {false};
    bool m_outputGlobalVarsSimTime  {true};
    
    // For modulating the output frequency global vars. By default
    // they are output every time step. Note: Frequency > OnTimeStep
    unsigned int m_outputGlobalVarsFrequency {1};
    unsigned int m_outputGlobalVarsOnTimeStep {0};

    
    //-----------------------------------------------------------
    std::string TranslateVariableType( std::string type, bool isThisCheckpoint );

    //-----------------------------------------------------------
    // RNJ - 
    //
    // This is the number of times the DataArchiver will retry
    // a file system operation before it gives up and throws
    // an exception.
    //-----------------------------------------------------------

    int m_fileSystemRetrys {10};

    //! This is if you want to pass in the uda extension on the command line
    int m_udaSuffix {-1};

    //! The number of levels the DA knows about.  If this changes,
    //! we need to redo output and Checkpoint tasks.
    int m_numLevelsInOutput {0};

    ReductionInfoMapper< RuntimeStatsEnum, double > *m_runtimeStats;

#ifdef HAVE_PIDX
    bool m_pidx_need_to_recompile {false};
    bool m_pidx_restore_nth_rank {false};
    int  m_pidx_requested_nth_rank {-1};
    bool m_pidx_checkpointing {false};
#endif
    
#if SCI_ASSERTION_LEVEL >= 2
    //! double-check to make sure that DA::output is only called once per level per processor per type
    std::vector<bool> m_outputCalled;
    std::vector<bool> m_checkpointCalled;
    bool m_checkpointGlobalCalled {false};
#endif
    Uintah::MasterLock m_outputLock;

    DataArchiver(const DataArchiver&);
    DataArchiver& operator=(const DataArchiver&);      
  };

} // End namespace Uintah

#endif
