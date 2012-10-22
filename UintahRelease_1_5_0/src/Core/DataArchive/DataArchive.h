/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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

#ifndef UINTAH_HOMEBREW_DataArchive_H
#define UINTAH_HOMEBREW_DataArchive_H

#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/Variables/VarnameMatlPatch.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Util/Handle.h>
#include <Core/Exceptions/VariableNotFoundInGrid.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Time.h>
#include <Core/Util/DebugStream.h>
#include <Core/Containers/ConsecutiveRangeSet.h>
#include <Core/Containers/HashTable.h>

#include   <string>
#include   <vector>
#include   <list>

#include <fcntl.h>

#ifndef _WIN32
#  include <unistd.h>
#endif


namespace Uintah {

  using namespace SCIRun;

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

  //! Container to hold UCF data when read in from disk.
  class DataArchive {
    private:

      // what we need to store on a per-variable basis
      // everything else can be retrieved from a higher level
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
        IntVector boundaryLayer;
      };

      struct PatchData {
        PatchData() : parsed(false), proc(-1) {}
        bool parsed;
        int proc;
        std::string datafilename;
      };

      typedef HashTable<VarnameMatlPatch, DataFileInfo> VarHashMap;
      typedef HashTableIter<VarnameMatlPatch, DataFileInfo> VarHashMapIterator;

      //! Top of DataArchive structure for storing hash maps of variable data
      //! - containing data for each time step.
      class TimeData {
        public:    
          TimeData(DataArchive* da, ProblemSpecP timestepDoc, std::string timestepURL);
          ~TimeData();
          VarData& findVariableInfo(const std::string& name, const Patch* patch, int matl);

          // reads timestep.xml and prepares the data xml files to be read
          void init();
          void purgeCache(); // purge the cached data

          // makes sure that (if the patch data exists) then it is parsed.  Try logic to pick
          // the right file first, and if you can't, parse everything
          void parsePatch(const Patch* patch);

          // parse an individual data file and load appropriate storage
          void parseFile(std::string file, int levelNum, int basePatch);

          // This would be private data, except we want DataArchive to have access,
          // so we would mark DataArchive as 'friend', but we're already a private
          // nested class of DataArchive...

          // info in the data file about the patch-matl-var
          VarHashMap d_datafileInfo;

          // Patch info (separate by levels) - proc, whether parsed, datafile, etc.
          // Gets expanded and proc is set during queryGrid.  Other fields are set
          // when parsed
          // Organized in a contiguous array, by patch-level-index
          std::vector<std::vector<PatchData> > d_patchInfo; 

          // Wheter a material is active per level
          std::vector<std::vector<bool> > d_matlInfo;

          // var info - type, compression, and boundary layer
          std::map<std::string, VarData> d_varInfo; 

          // xml urls referred to in timestep.xml
          std::vector<std::vector<std::string> > d_xmlUrls;
          std::vector<std::vector<bool> > d_xmlParsed;

          std::string d_globaldata;

          ConsecutiveRangeSet matls;  // materials available this timestep

          GridP d_grid;               // store the grid...
          bool d_initialized;         // Flagged once this patch's init is called
          ProblemSpecP d_tstop;       // ProblemSpecP of timestep.xml
          std::string d_tsurl;        // path to timestep.xml
          std::string d_tsurldir;     // dir that contains timestep.xml
          bool d_swapBytes;
          int d_nBytes;
          DataArchive* da;            // pointer for parent DA.  Need for backward-compatibility with endianness, etc.
      };

    public:
      DataArchive(const std::string& filebase,
          int processor = 0 /* use if you want to different processors
                               to read different parts of the archive */,
          int numProcessors = 1,
          bool verbose = true ); // If you want error messages printed to the screen.

      // GROUP: Destructors
      //////////
      // Destructor
      virtual ~DataArchive();

      TimeData& getTimeData(int index);


      // Processor Information for Visualization
      int queryPatchwiseProcessor( const Patch* patch, int index );
      int queryNumProcs(int index);

      std::string name(){ return d_filebase;}

      //! Set up data arachive for restarting a Uintah simulation   
      void restartInitialize(int timestep, const GridP& grid, DataWarehouse* dw,
          LoadBalancer* lb, double* pTime /* passed back */);

      inline ProblemSpecP getTimestepDoc(int index) { return getTimeData(index).d_tstop; }

      static void queryEndiannessAndBits(ProblemSpecP, std::string& endianness, int& numBits);  

      // GROUP:  Information Access
      //////////
      // However, we need a means of determining the names of existing
      // variables. We also need to determine the type of each variable.
      // Get a list of scalar or vector variable names and  
      // a list of corresponding data types
      void queryVariables( std::vector< std::string>& names,
          std::vector< const TypeDescription *>&  );
      void queryGlobals( std::vector< std::string>& names,
          std::vector< const TypeDescription *>&  );
      void queryTimesteps( std::vector<int>& index,
          std::vector<double>& times );

      //! the ups is for the assignBCS that needs to happen
      //! if we are reading the simulation grid from the uda,
      //! and thus is only necessary on a true restart.
      GridP queryGrid(int index, const ProblemSpec* ups = 0);




#if 0
      //////////
      // Does a variable exist in a particular patch?
      bool exists(const std::string&, const Patch*, int) {
        return true;
      }
#endif

      //////////
      // how long does a particle live?  Not variable specific.
      void queryLifetime( double& min, double& max, particleId id);

      //////////
      // how long does a patch live?  Not variable specific
      void queryLifetime( double& min, double& max, const Patch* patch);

      ConsecutiveRangeSet queryMaterials(const std::string& varname,
          const Patch* patch, int index);

      int queryNumMaterials(const Patch* patch, int index);

      // Queries a variable for a material, patch, and index in time.
      // Optionally pass in DataFileInfo if you're iterating over
      // entries in the hash table (like restartInitialize does)
      void query( Variable& var, const std::string& name, int matlIndex, 
          const Patch* patch, int timeIndex, DataFileInfo* dfi = 0);

      void query( Variable& var, const std::string& name, int matlIndex, 
          const Patch* patch, int timeIndex,
          Ghost::GhostType, int ngc);

      void queryRegion( Variable& var, const std::string& name, int matlIndex, 
          const Level* level, int timeIndex, IntVector low, IntVector high );

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

      // Only cache a single timestep
      void turnOnXMLCaching();
      // Cache the default number of timesteps
      void turnOffXMLCaching();
      // Cache new_size number of timesteps.  Calls the
      // TimeHashMaps::updateCacheSize function with new_size.  See
      // corresponding documentation.
      void setTimestepCacheSize(int new_size);

      // These are here for the LockingHandle interface.  The names should
      // match those found in Core/Datatypes/Datatype.h.
      int ref_cnt;
      Mutex lock;


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

    protected:
      DataArchive();

    private:
      friend class DataArchive::TimeData;
      DataArchive(const DataArchive&);
      DataArchive& operator=(const DataArchive&);

      void queryVariables( const ProblemSpecP vars, std::vector<std::string>& names,
          std::vector<const TypeDescription*>& types);

      std::string d_filebase;  
      ProblemSpecP d_indexDoc;
      ProblemSpecP d_restartTimestepDoc;
      std::string d_restartTimestepURL;

      bool d_simRestart;
      Vector d_cell_scale; //used for scaling the physical data size

      std::vector<TimeData> d_timeData;
      std::vector<int> d_tsindex;
      std::vector<double> d_tstimes;

      // global bits and endianness - read from index.xml ONLY if not in timestep.xml
      std::string d_globalEndianness;
      int d_globalNumBits;

      typedef std::map<std::pair<int, const Patch*>, Handle<ParticleSubset> > psetDBType;
      psetDBType d_psetDB;

      // if used, different processors read different parts of the archive
      int d_processor;
      int d_numProcessors;

      Mutex d_lock;

      void findPatchAndIndex(GridP grid, Patch*& patch, particleIndex& idx,
          long64 particleID, int matIndex, int levelIndex,
          int index);

      static DebugStream dbg;
  };


  template<class T>
    void DataArchive::query( NCVariable< T >&, const std::string& name, int matlIndex,
        const IntVector& index,
        double min, double max)
    {
      std::cerr << "DataArchive::query not finished\n";
    }

  template<class T>
    void DataArchive::query( CCVariable< T >&, const std::string& name, int matlIndex,
        const IntVector& index,
        double min, double max)
    {
      std::cerr << "DataArchive::query not finished\n";
    }

  template<class T>
    void DataArchive::query(ParticleVariable< T >& var, const std::string& name,
        int matlIndex, particleId id,
        double min, double max)
    {
      std::cerr << "DataArchive::query not finished\n";
    }


  template<class T>
    void DataArchive::query(std::vector<T>& values, const std::string& name,
        int matlIndex, long64 particleID,
        int levelIndex,
        double startTime, double endTime)
    {
      double call_start = SCIRun::Time::currentSeconds();

      std::vector<int> index;
      std::vector<double> times;
      queryTimesteps(index, times); // build timesteps if not already done

      // figure out what kind of variable we're looking for
      std::vector<std::string> type_names;
      std::vector<const TypeDescription*> type_descriptions;
      queryVariables(type_names, type_descriptions);
      const TypeDescription* type = NULL;
      std::vector<std::string>::iterator name_iter = type_names.begin();
      std::vector<const TypeDescription*>::iterator type_iter = type_descriptions.begin();
      for ( ; name_iter != type_names.end() && type == NULL;
          name_iter++, type_iter++) {
        if (*name_iter == name)
          type = *type_iter;
      }
      if (type == NULL)
        throw InternalError("Unable to determine variable type", __FILE__, __LINE__);
      if (type->getType() != TypeDescription::ParticleVariable)    
        throw InternalError("Variable type is not ParticleVariable", __FILE__, __LINE__);
      // find the first timestep
      int ts = 0;
      while ((ts < (int)d_tstimes.size()) && (startTime > d_tstimes[ts]))
        ts++;

      // idx needs to be initialized before it is used in findPatchAndIndex.
      particleIndex idx = 0;
      for ( ; (ts < (int)d_tstimes.size()) && (d_tstimes[ts] <= endTime); ts++) {
        // figure out what patch contains the cell. As far as I can tell,
        // nothing prevents this from changing between timesteps, so we have to
        // do this every time -- if that can't actually happen we might be able
        // to speed this up.
        Patch* patch = NULL;
        GridP grid = queryGrid( ts);
        findPatchAndIndex(grid, patch, idx, particleID, matlIndex, levelIndex, ts);
        //    std::cerr <<" Patch = 0x"<<hex<<patch<<dec<<", index = "<<idx;
        if (patch == NULL)
          throw VariableNotFoundInGrid(name,particleID,matlIndex,
              "DataArchive::query", __FILE__, __LINE__);

        ParticleVariable<T> var;
        query(var, name, matlIndex, patch, ts);
        //now find the index that corresponds to the particleID
        //std::cerr <<" time = "<<t<<",  value = "<<var[idx]<<std::endl;
        values.push_back(var[idx]);

      }
      dbg << "DataArchive::query(values) completed in "
        << (SCIRun::Time::currentSeconds() - call_start) << " seconds\n";
    }  

  template<class T>
    void DataArchive::query(std::vector<T>& values, const std::string& name,
        int matlIndex, IntVector loc,
        double startTime, double endTime,
        int levelIndex /*=-1*/)
    {
      double call_start = SCIRun::Time::currentSeconds();

      std::vector<int> index;
      std::vector<double> times;
      queryTimesteps(index, times); // build timesteps if not already done

      // figure out what kind of variable we're looking for
      std::vector<std::string> type_names;
      std::vector<const TypeDescription*> type_descriptions;
      queryVariables(type_names, type_descriptions);
      const TypeDescription* type = NULL;
      std::vector<std::string>::iterator name_iter = type_names.begin();
      std::vector<const TypeDescription*>::iterator type_iter = type_descriptions.begin();
      for ( ; name_iter != type_names.end() && type == NULL;
          name_iter++, type_iter++) {
        if (*name_iter == name)
          type = *type_iter;
      }
      if (type == NULL)
        throw InternalError("Unable to determine variable type", __FILE__, __LINE__);

      // find the first timestep
      int ts = 0;
      while ((ts < (int)d_tstimes.size()) && (startTime > d_tstimes[ts]))
        ts++;

      for ( ; (ts < (int)d_tstimes.size()) && (d_tstimes[ts] <= endTime); ts++) {
        // figure out what patch contains the cell. As far as I can tell,
        // nothing prevents this from changing between timesteps, so we have to
        // do this every time -- if that can't actually happen we might be able
        // to speed this up.
        Patch* patch = NULL;
        GridP grid = queryGrid(ts);

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

        for (int level_nr = startLevel;
            (level_nr < endLevel) && (patch == NULL); level_nr++) {
          const LevelP level = grid->getLevel(level_nr);

          switch (type->getType()) {
            case TypeDescription::CCVariable:
              for (Level::const_patchIterator iter = level->patchesBegin();
                  (iter != level->patchesEnd()) && (patch == NULL); iter++) {
                if ((*iter)->containsCell(loc)) {
                  patch = *iter;
                  // We found our patch, quit looking.
                  break;
                }
              }
              break;

            case TypeDescription::NCVariable:
              for (Level::const_patchIterator iter = level->patchesBegin();
                  (iter != level->patchesEnd()) && (patch == NULL); iter++) {
                if ((*iter)->containsNode(loc)) {
                  patch = *iter;
                  break;
                }
              }
              break;
            case TypeDescription::SFCXVariable:
              for (Level::const_patchIterator iter = level->patchesBegin();
                  (iter != level->patchesEnd()) && (patch == NULL); iter++) {
                if ((*iter)->containsSFCX(loc)) {
                  patch = *iter;
                  break;
                }
              }
              break;
            case TypeDescription::SFCYVariable:
              for (Level::const_patchIterator iter = level->patchesBegin();
                  (iter != level->patchesEnd()) && (patch == NULL); iter++) {
                if ((*iter)->containsSFCY(loc)) {
                  patch = *iter;
                  break;
                }
              }
              break;
            case TypeDescription::SFCZVariable:
              for (Level::const_patchIterator iter = level->patchesBegin();
                  (iter != level->patchesEnd()) && (patch == NULL); iter++) {
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
        if (patch == NULL) {
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

      dbg << "DataArchive::query(values) completed in "
        << (SCIRun::Time::currentSeconds() - call_start) << " seconds\n";
    }
  
} // end namespace Uintah

#endif

