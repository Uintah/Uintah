#ifndef UINTAH_HOMEBREW_DataArchive_H
#define UINTAH_HOMEBREW_DataArchive_H

/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

#include <Core/Containers/ConsecutiveRangeSet.h>
//#include <Core/Containers/HashTable.h>
#include <Core/Exceptions/VariableNotFoundInGrid.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/VarnameMatlPatch.h>
#include <Core/Parallel/MasterLock.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/Handle.h>
#include <Core/Util/Timers/Timers.hpp>

#if HAVE_PIDX
#  include <PIDX.h>
#endif

#include <list>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

namespace Uintah {

class VarLabel;
class DataWarehouse;
class LoadBalancer;

/**************************************

  CLASS
  DataArchive

  Short Description...

  GENERAL INFORMATION

  DataArchive.h

  Kurt Zimmerman
  Department of Computer Science
  University of Utah

  Center for the Simulation of Accidental Fires and Explosions (C-SAFE)

  KEYWORDS
  DataArchive

  DESCRIPTION
  Long description...

  WARNING

****************************************/

#if HAVE_PIDX
// For PIDX usage:
struct BufferAndSizeTuple {
  BufferAndSizeTuple() { buffer = nullptr; size = -1; }
  unsigned char * buffer;
  long            size; // not unsigned so I can set to -1 for sanity checking.
};
#endif
  
//! Container to hold UCF data when read in from disk.
class DataArchive {

private:
  struct DataFileInfo; // Forward declaration

public:
  DataArchive( const std::string & filebase,               // <- Name to save UDA under.
               const int           processor     = 0,      // use if you want to different processors to read different parts of the archive.
               const int           numProcessors = 1,
               const bool          verbose       = true ); // If you want error messages printed to the screen.

  // GROUP: Destructors
  //////////
  // Destructor
  virtual ~DataArchive();

  // Returns true if this DataArchive is using the PIDX format.
  bool isPIDXFormat() const { return d_fileFormat == PIDX; }

  // Processor Information for Visualization
  int queryPatchwiseProcessor( const Patch* patch, const int index );

  // Returns the name of the UDA (on disk).
  std::string name() { return d_filebase; }

  // Return the name of the particle position variable if specified by the user. if not, this will return p.x.
  std::string getParticlePositionName() const { return d_particlePositionName; }

  //! Set up data arachive for restarting a Uintah simulation
  void restartInitialize( const int       timestep,
                          const GridP   & grid,
                          DataWarehouse * dw,
                          LoadBalancer  * lb,
                          double        * pTime /* passed back */ );


  //__________________________________
  //  This is used by postProcessUda component.  It reads in the data and puts it into the DW.
  //  This is a specialization of restartInitialize().
  void postProcess_ReadUda( const ProcessorGroup * pg,
                            const int              timestep,
                            const GridP          & grid,
                            const PatchSubset    * patches,
                            DataWarehouse        * dw,
                            LoadBalancer         * lb ); 

  // GROUP:  Information Access
  //////////
  // However, we need a means of determining the names of existing
  // variables. We also need to determine the type of each variable.
  // Get a list of scalar or vector variable names and  
  // a list of corresponding data types
  void queryVariables( std::vector<std::string>              & names,
                       std::vector<int>                      & num_matls,
                       std::vector<const TypeDescription *>  & types );

  void queryGlobals(   std::vector<std::string>              & names,
                       std::vector<const TypeDescription *>  &  );

  void queryTimesteps( std::vector<int>                      & index,
                       std::vector<double>                   & times );

  void queryProcessors( unsigned int & nProcs );
  
  //! The ups is for the assignBCS that needs to happen if we are reading the simulation grid from the uda,
  //! and thus is only necessary on a true restart.
  GridP queryGrid( int index, const ProblemSpecP & ups = nullptr, bool assignBCs = true );

  //////////
  // Does a variable exist on a patch at this timestep?
  bool exists( const std::string & varname,
               const Patch       * patch,
               const int           timeStep );
  //////////
  // how long does a particle live?  Not variable specific.
  void queryLifetime( double & min, double & max, particleId id );

  //////////
  // How long does a patch live?  Not variable specific.
  void queryLifetime( double & min, double & max, const Patch * patch );

  ConsecutiveRangeSet queryMaterials( const std::string & varname,
                                      const Patch       * patch,
                                      int           index );

  int queryNumMaterials( const Patch* patch, int index );

  bool query(       Variable     & var,
              const std::string  & name,
              const int            matlIndex, 
              const Patch        * patch,
              const int            timeIndex,
                    DataFileInfo * dfi = nullptr );

  bool query(       Variable         & var,
              const std::string      & name,
              const int                matlIndex, 
              const Patch            * patch,
              const int                timeIndex,
              const Ghost::GhostType   ghostType,
              const int                numGhostCells );

  void queryRegion(       Variable    & var,
                    const std::string & name,
                    const int           matlIndex, 
                    const Level       * level,
                    const int           timeIndex,
                    const IntVector   & low,
                    const IntVector   & high );

  //////////
  // query the variable value for a particular particle  overtime;
  // T = double/float/vector/Tensor I'm not sure of the proper
  // syntax.
  template<class T>
  void query( ParticleVariable< T >&, const std::string& name, int matlIndex,
              particleId id,
              double min, double max);

  //////////
  // query the variable value for a particular particle  overtime;
  // T = double/float/vector/Tensor I'm not sure of the proper
  // syntax.
  template<class T>
  void query( NCVariable< T >&, const std::string& name, int matlIndex,
              const IntVector& index,
              double min, double max);

  //////////
  // query the variable value for a particular particle  overtime;
  // T = double/float/vector/Tensor I'm not sure of the proper
  // syntax.
  template<class T>
  void query( CCVariable< T >&, const std::string& name, int matlIndex,
              const IntVector& index,
              double min, double max);

  //////////
  // query the variable value for a particular particle  overtime;
  template<class T>
  void query(std::vector<T>& values, const std::string& name,
             int matlIndex, long64 particleID, int levelIndex,
             double startTime, double endTime) ;
  //////////
  // similarly, we want to be able to track variable values in a particular
  // patch cell over time.
  template<class T>
  void query(std::vector<T>& values, const std::string& name, int matlIndex,
             IntVector loc, double startTime, double endTime, int level=-1);

  //////////
  // Pass back the timestep number specified in the "restart" tag of the
  // index file, or return false if such a tag does not exist.
  bool queryRestartTimestep(int& timestep);
#if 0
  //////////
  // In other cases we will have noticed something interesting and we
  // will want to access some small portion of a patch.  We will need
  // to request some range of data in index space.
  template<class T> void get(T& data, const std::string& name,
                             const Patch* patch, cellIndex min, cellIndex max);
#endif

  // Reads the appropriate timestep.xml file and returns the <oldDelt>
  double getOldDelt( int restart_index );

  // Parses the timestep.xml file that corrensponds to the restart_index and creates
  // a problem spec with the Component's portion (ie: the portion after </Data>).
  ProblemSpecP getTimestepDocForComponent( int restart_index );

  // Only cache a single timestep
  void turnOnXMLCaching();
      
  // Cache the default number of timesteps
  void turnOffXMLCaching();
      
  // Cache new_size number of timesteps.  Calls the
  // TimeHashMaps::updateCacheSize function with new_size.  See
  // corresponding documentation.
  void setTimestepCacheSize(int new_size);

  // This is added to allow simple geometric scaling of the entire domain
  void setCellScale( Vector& s ){ d_cell_scale = s; }
  // This is added so that particles can see if the domain has been scaled
  // and change the particle locations appropriately.
  Vector getCellScale(){ return d_cell_scale; }

  // This is a list of the last n timesteps accessed.  Data from
  // only the last timestep_cache_size timesteps is stored, unless
  // timestep_cache_size is less than or equal to zero then the size
  // is unbounded.
  std::list<int> d_lastNtimesteps;

  // Tells you the number of timesteps to cache. Less than or equal to
  // zero means to cache all of them.
  int timestep_cache_size;

  // This will be the default number of timesteps cached, determined
  // by the number of processors.
  int default_cache_size;

  // Used as the first byte of the grid.xml file to designate/verify that it is stored in binary format.
  static const unsigned int GRID_MAGIC_NUMBER{ 0xdeadbeef };

protected:
  DataArchive();

private:

#if HAVE_PIDX
  void queryPIDX(       BufferAndSizeTuple * data,
                  const PIDX_variable      & varDesc,
                  const TypeDescription    * td ,
                  const std::string        & name,
                  const int                  matlIndex, 
                  const Patch              * patch,
                  const int                  timeIndex );

  bool setupQueryPIDX(       PIDX_access     & access,
                             PIDX_file       & idxFile,
                             PIDX_variable   & varDesc,
                       const LevelP          & level,
                       const TypeDescription * td,
                       const std::string     & name,
                       const int               matlIndex,
                       const int               timeIndex );

  bool queryPIDXSerial(       Variable     & var,
                        const std::string  & name,
                        const int            matlIndex,
                        const Patch        * patch,
                        const int            timeIndex );
#endif

  // Sets d_particlePositionName if found. Note, rewinds 'xml_fp', thus starting at the top of the file.
  void queryAndSetParticlePositionName( FILE * xml_fp ); 

  // Sets d_fileFormat
  void queryAndSetFileFormat( FILE * xml_fp );

  static void queryEndiannessAndBits( ProblemSpecP doc, std::string & endianness, int & numBits );

  // Note, this function rewinds 'fp', and thus starts at the top of the file.
  static void queryEndiannessAndBits( FILE* fp,  std::string & endianness, int & numBits );

  ////////////////

  std::map<std::string, VarLabel*> d_createdVarLabels;

  // What we need to store on a per-variable basis, everything else can be retrieved from a higher level.
  struct DataFileInfo {
    DataFileInfo(long s, long e, long np) : start(s), end(e), numParticles(np) {}
    DataFileInfo() {}
    long start;
    long end;
    int numParticles;
  };

  // store these in separate arrays so we don't have to store nearly as many of them
  struct VarData {
    std::string type;
    std::string compression;
    std::string filename;            // needed for PIDX
    IntVector boundaryLayer;
  };

  struct PatchData {
    PatchData() : parsed(false), proc(-1) {}
    bool parsed;
    int proc;
    std::string datafilename;
  };

  //  typedef Uintah::HashTable<VarnameMatlPatch, DataFileInfo> VarHashMap;
  //  typedef Uintah::HashTableIter<VarnameMatlPatch, DataFileInfo> VarHashMapIterator;

  //! Top of DataArchive structure for storing hash maps of variable data
  //! - containing data for each time step.
  class TimeData {
  public:    
    TimeData( DataArchive * da, const std::string & timestepPathAndFilename );
    ~TimeData();
    VarData& findVariableInfo(const std::string& name, const Patch* patch, int matl);

    // Reads timestep.xml and prepares the data xml files to be read.
    void init();
    void purgeCache(); // purge the cached data

    // Makes sure that (if the patch data exists) then it is parsed.  Try logic to pick
    // the right file first, and if you can't, parse everything.
    void parsePatch( const Patch* patch );

    // Parse an individual data file and load appropriate storage.
    void parseFile( const std::string & filename, int levelNum, int basePatch );

    // This would be private data, except we want DataArchive to have access,
    // so we would mark DataArchive as 'friend', but we're already a private
    // nested class of DataArchive...

    // Info in the data file about the patch-matl-var.
    // Replaced the "VarHashMap d_datafileInfo;" with the two vectors (below) in order to
    // keep the variables in a defined order.  This occurred after svn revision r56540.
    std::vector<VarnameMatlPatch> d_datafileInfoIndex;
    std::vector<DataFileInfo>     d_datafileInfoValue;

    // Patch info (separate by levels) - proc, whether parsed, datafile, etc.
    // Gets expanded and proc is set during queryGrid.  Other fields are set
    // when parsed
    // Organized in a contiguous array, by patch-level-index
    std::vector<std::vector<PatchData> >   d_patchInfo; 

    // Wheter a material is active per level.
    std::vector<std::vector<bool> >        d_matlInfo;

    // Var info - type, compression, and boundary layer.
    std::map<std::string, VarData>         d_varInfo;

    // Xml filenames referred to in timestep.xml.
    std::vector< std::vector<std::string> > d_xmlFilenames;
    std::vector< std::vector<bool> >        d_xmlParsed;

    std::string   d_globaldata;

    ConsecutiveRangeSet d_matls;  // materials available this timestep

    GridP         d_grid;
    bool          d_initialized;

    ProblemSpecP  d_timestep_ps_for_component;    // timestep.xml's xml for components.
    std::string   d_ts_path_and_filename;         // Path to timestep.xml.
    std::string   d_ts_directory;                 // Directory that contains timestep.xml.
    bool          d_swapBytes;
    int           d_nBytes;
    DataArchive * d_parent_da;                    // Pointer to parent DA.  Need for backward-compatibility with endianness, etc.
  };

  DataArchive( const DataArchive& );
  DataArchive& operator=( const DataArchive& );

  // There can be multiple DataArchives so must static.
  static DebugStream dbg;

  //__________________________________
  //  PIDX related
  enum FileFormatType { UDA, PIDX, NOT_SPECIFIED };
  FileFormatType d_fileFormat; 
      
  enum VarType { BLANK, REDUCTION_VAR, PATCH_VAR };
 
  bool isPIDXEnabled(){
#if HAVE_PIDX
    return true;  
#else
    return false;
#endif
  };

  void createPIDXCommunicator( const GridP & grid, LoadBalancer * lb );
  std::vector<MPI_Comm> d_pidxComms; // Array of MPI Communicators for PIDX usage...


  //______________________________________________________________________
  //
  void queryVariables( FILE                                * fp,
                       std::vector<std::string>            & names,
                       std::vector<int>                    & num_matls,
                       std::vector<const TypeDescription*> & types,
                       bool                                  globals = false );

  TimeData & getTimeData( int index );

  std::string   d_filebase;
  FILE        * d_indexFile; // File pointer to XML index document.

  bool   d_simRestart;
  Vector d_cell_scale; // Used for scaling the physical data size.

  std::vector<TimeData> d_timeData;
  std::vector<int>      d_ts_indices;
  std::vector<double>   d_ts_times;

  // global bits and endianness - read from index.xml ONLY if not in timestep.xml
  std::string d_globalEndianness;
  int         d_globalNumBits;

  typedef std::map<std::pair<int, const Patch*>, Handle<ParticleSubset> > psetDBType;
  psetDBType d_psetDB;

  // if used, different processors read different parts of the archive
  int d_processor;
  int d_numProcessors;

  Uintah::MasterLock d_lock;
    
  std::string d_particlePositionName;

  void findPatchAndIndex( const GridP            grid,
                          Patch         *& patch,
                          particleIndex  & idx,
                          const long64           particleID,
                          const int              matIndex,
                          const int              levelIndex,
                          const int              index );

  static bool        d_types_initialized;     
};

  //______________________________________________________________________
  //
  template<class T>
  void
  DataArchive::query( NCVariable< T >&, const std::string& name, int matlIndex,
                      const IntVector& index, double min, double max ) {
    std::cerr << "DataArchive::query not finished\n";
  }
  //______________________________________________________________________
  //
  template<class T>
  void
  DataArchive::query( CCVariable< T >&, const std::string& name, int matlIndex,
                      const IntVector& index, double min, double max ) {
    std::cerr << "DataArchive::query not finished\n";
  }
  //______________________________________________________________________
  //
  template<class T>
  void
  DataArchive::query( ParticleVariable< T >& var, const std::string& name,
                      int matlIndex, particleId id, double min, double max ) {
    std::cerr << "DataArchive::query not finished\n";
  }
  //______________________________________________________________________
  //
  template<class T>
  void
  DataArchive::query(       std::vector<T> & values,
                      const std::string    & name,
                            int              matlIndex,
                            long64           particleID,
                            int              levelIndex,
                            double           startTime,
                            double           endTime ) {
    Timers::Simple timer;
    timer.start();

    std::vector<int> index;
    std::vector<double> times;
    queryTimesteps( index, times ); // build timesteps if not already done

    // figure out what kind of variable we're looking for
    std::vector<std::string>            type_names;
    std::vector<int>                    num_matls;
    std::vector<const TypeDescription*> type_descriptions;

    // README FIXME qwerty ... who calls this?  We are now opening and reading a file
    // every time queryVariables() is called, so if this happens a lot, we might need to rethink it...

    queryVariables( type_names, num_matls, type_descriptions );

    const TypeDescription* type = nullptr;
    std::vector<std::string>::iterator name_iter = type_names.begin();
    std::vector<const TypeDescription*>::iterator type_iter = type_descriptions.begin();
    for ( ; name_iter != type_names.end() && type == nullptr; name_iter++, type_iter++ ) {
      if (*name_iter == name) {
        type = *type_iter;
      }
    }
    if( type == nullptr ) {
      throw InternalError("Unable to determine variable type", __FILE__, __LINE__);
    }
    if( type->getType() != TypeDescription::ParticleVariable ) {
      throw InternalError("Variable type is not ParticleVariable", __FILE__, __LINE__);
    }
    // find the first timestep
    int ts = 0;
    while( (ts < (int)d_ts_times.size()) && (startTime > d_ts_times[ts]) ) {
      ts++;
    }

    // Idx needs to be initialized before it is used in findPatchAndIndex.
    particleIndex idx = 0;
    for ( ; (ts < (int)d_ts_times.size()) && (d_ts_times[ts] <= endTime); ts++) {
      // figure out what patch contains the cell. As far as I can tell,
      // nothing prevents this from changing between timesteps, so we have to
      // do this every time -- if that can't actually happen we might be able
      // to speed this up.
      Patch * patch = nullptr;
      GridP   grid  = queryGrid( ts );
      findPatchAndIndex( grid, patch, idx, particleID, matlIndex, levelIndex, ts );

      if( patch == nullptr ) {
        throw VariableNotFoundInGrid( name, particleID, matlIndex, "DataArchive::query", __FILE__, __LINE__ );
      }

      ParticleVariable<T> var;
      query( var, name, matlIndex, patch, ts );
      // Now find the index that corresponds to the particleID
      // std::cerr <<" time = "<<t<<",  value = "<<var[idx]<<std::endl;
      values.push_back(var[idx]);
    }

    dbg << "DataArchive::query(values) completed in " << timer().seconds()
        << " seconds\n";
  }  
  //______________________________________________________________________
  //
  template<class T>
  void
  DataArchive::query(       std::vector<T> & values,
                      const std::string    & name,
                            int              matlIndex,
                            IntVector        loc,
                            double           startTime,
                            double           endTime,
                            int              levelIndex /* = -1 */ ) {
    Timers::Simple timer;
    timer.start();

    std::vector<int> index;
    std::vector<double> times;
    queryTimesteps(index, times); // build timesteps if not already done

    // figure out what kind of variable we're looking for
    std::vector<std::string>            type_names;
    std::vector<int>                    num_matls;
    std::vector<const TypeDescription*> type_descriptions;
    queryVariables( type_names, num_matls, type_descriptions );
    const TypeDescription* type = nullptr;
    std::vector<std::string>::iterator name_iter = type_names.begin();
    std::vector<const TypeDescription*>::iterator type_iter = type_descriptions.begin();
    for ( ; name_iter != type_names.end() && type == nullptr; name_iter++, type_iter++) {
      if (*name_iter == name) {
        type = *type_iter;
      }
    }
    if ( type == nullptr ) {
      throw InternalError( "Unable to determine variable type", __FILE__, __LINE__ );
    }

    // Find the first timestep.
    int ts = 0;
    while( (ts < (int)d_ts_times.size()) && (startTime > d_ts_times[ts]) ) {
      ts++;
    }

    for ( ; (ts < (int)d_ts_times.size()) && (d_ts_times[ts] <= endTime); ts++) {
      // figure out what patch contains the cell. As far as I can tell,
      // nothing prevents this from changing between timesteps, so we have to
      // do this every time -- if that can't actually happen we might be able
      // to speed this up.
      Patch * patch = nullptr;
      GridP   grid  = queryGrid( ts );

      // which levels to query between.
      int startLevel, endLevel;
      if (levelIndex == -1) {
        startLevel = 0;
        endLevel = grid->numLevels();
      }
      else {
        startLevel = levelIndex;
        endLevel = levelIndex+1;
      }

      for (int level_nr = startLevel; (level_nr < endLevel) && (patch == nullptr); level_nr++) {
        const LevelP level = grid->getLevel(level_nr);

        switch (type->getType()) {
        case TypeDescription::CCVariable:
          for (Level::const_patch_iterator iter = level->patchesBegin();
               (iter != level->patchesEnd()) && (patch == nullptr); iter++) {
            if ((*iter)->containsCell(loc)) {
              patch = *iter;
              // We found our patch, quit looking.
              break;
            }
          }
          break;

        case TypeDescription::NCVariable:
          for (Level::const_patch_iterator iter = level->patchesBegin();
               (iter != level->patchesEnd()) && (patch == nullptr); iter++) {
            if ((*iter)->containsNode(loc)) {
              patch = *iter;
              break;
            }
          }
          break;
        case TypeDescription::SFCXVariable:
          for (Level::const_patch_iterator iter = level->patchesBegin();
               (iter != level->patchesEnd()) && (patch == nullptr); iter++) {
            if ((*iter)->containsSFCX(loc)) {
              patch = *iter;
              break;
            }
          }
          break;
        case TypeDescription::SFCYVariable:
          for (Level::const_patch_iterator iter = level->patchesBegin();
               (iter != level->patchesEnd()) && (patch == nullptr); iter++) {
            if ((*iter)->containsSFCY(loc)) {
              patch = *iter;
              break;
            }
          }
          break;
        case TypeDescription::SFCZVariable:
          for (Level::const_patch_iterator iter = level->patchesBegin();
               (iter != level->patchesEnd()) && (patch == nullptr); iter++) {
            if ((*iter)->containsSFCZ(loc)) {
              patch = *iter;
              break;
            }
          }
          break;

        default:
          std::cerr << "Variable of unsupported type for this cell-based query: " << type->getType() << '\n';
          break;
        }
      }
      if (patch == nullptr) {
        throw VariableNotFoundInGrid(name,loc,matlIndex,"DataArchive::query", __FILE__, __LINE__);
      }

      switch (type->getType()) {
      case TypeDescription::CCVariable: {
        CCVariable<T> var;
        query(var, name, matlIndex, patch, ts);
        values.push_back(var[loc]);
      } break;

      case TypeDescription::NCVariable: {
        NCVariable<T> var;
        query(var, name, matlIndex, patch, ts);
        values.push_back(var[loc]);
      } break;

      case TypeDescription::SFCXVariable: {
        SFCXVariable<T> var;
        query(var, name, matlIndex, patch, ts);
        values.push_back(var[loc]);
      } break;

      case TypeDescription::SFCYVariable: {
        SFCYVariable<T> var;
        query(var, name, matlIndex, patch, ts);
        values.push_back(var[loc]);
      } break;

      case TypeDescription::SFCZVariable: {
        SFCZVariable<T> var;
        query(var, name, matlIndex, patch, ts);
        values.push_back(var[loc]);
      } break;

      default:
        // Dd: Is this correct?  Error here?
        break;
      }
      //std::cerr << "DataArchive::query:data extracted" << std::endl;
    }

    dbg << "DataArchive::query(values) completed in " << timer().seconds()
        << " seconds\n";
  }
  
} // end namespace Uintah

#endif

