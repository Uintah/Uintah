#ifndef UINTAH_HOMEBREW_DataArchive_H
#define UINTAH_HOMEBREW_DataArchive_H

#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/ProblemSpec/Handle.h>
#include <Packages/Uintah/Core/Exceptions/VariableNotFoundInGrid.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Time.h>
#include <Core/Util/DebugStream.h>
#include <Core/Containers/ConsecutiveRangeSet.h>
#include <Core/Containers/HashTable.h>

#include <sgi_stl_warnings_off.h>
#include <string>
#include <vector>
#include <list>
#include <sgi_stl_warnings_on.h>

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#define IRIX
#pragma set woff 1375
#endif
//#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/util/XMLURL.hpp>
#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1375
#endif
#include <fcntl.h>
#include <unistd.h>

namespace Uintah {

using namespace SCIRun;
using std::string;
using std::vector;

class VarLabel;
  class DataWarehouse;

   struct eqstr { // comparison class used in hash_map to compare keys
     bool operator()(const char* s1, const char* s2) const {
       return strcmp(s1, s2) == 0;
     }
   };

   typedef HashTable<string, pair<ProblemSpecP, XMLURL> > VarHashMap;

   typedef HashTableIter<string, pair<ProblemSpecP, XMLURL> >
   VarHashMapIterator;

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
      
       Copyright (C) 2000 SCI Group
      
     KEYWORDS
       DataArchive
      
     DESCRIPTION
       Long description...
      
     WARNING
      
     ****************************************/
   
class DataArchive {
private:

  /* Helper classes for storing hash maps of variable data. */
  class PatchHashMaps;
  class MaterialHashMaps;
  class TimeHashMaps;
  friend class PatchHashMaps;
  friend class MaterialHashMaps;
  friend class TimeHashMaps;
  
  // Top of data structure for storing hash maps of variable data
  // - containing data for each time step.
  class TimeHashMaps {
  public:
    TimeHashMaps(const vector<double>& tsTimes,
		 const vector<XMLURL>& tsUrls,
		 const vector<ProblemSpecP>& tsTopNodes,
		 int processor, int numProcessors);
    
    inline ProblemSpecP findVariable(const string& name, const Patch* patch,
				     int matl, double time, XMLURL& foundUrl);
    
    inline PatchHashMaps* findTimeData(double time);
    void purgeTimeData(double time); // purge some caching
    
    MaterialHashMaps* findPatchData(double time, const Patch* patch);
  private:
    map<double, PatchHashMaps> d_patchHashMaps;
    map<double, PatchHashMaps>::iterator d_lastFoundIt;
  };
  
  // Second layer of data structure for storing hash maps of variable data
  // - containing data for each patch at a certain time step.
  class PatchHashMaps {
    friend class TimeHashMaps;
  public:
    PatchHashMaps();
    ~PatchHashMaps();  // to free the saved XML Data
    void init(XMLURL tsUrl, ProblemSpecP tsTopNode,
	      int processor, int numProcessors);
    void purgeCache(); // purge the cached data
    inline ProblemSpecP findVariable(const string& name, const Patch* patch,
				 int matl, XMLURL& foundUrl);
    MaterialHashMaps* findPatchData(const Patch* patch);
  private:
    void parse();    
    void add(const string& name, int patchid, int matl,
	     ProblemSpecP varNode, XMLURL url)
    { d_matHashMaps[patchid].add(name, matl, varNode, url); }
    
    map<int, MaterialHashMaps> d_matHashMaps;
    map<int, MaterialHashMaps>::iterator d_lastFoundIt;
    list<XMLURL> d_xmlUrls;
    bool d_isParsed;
    vector<ProblemSpecP> docs; // kept around for memory cleanup purposes
  };
  
  // Third layer of data structure for storing hash maps of variable data
  // - containing data for each material at a certain patch and time step.
  class MaterialHashMaps {
    friend class PatchHashMaps;
  public:
    MaterialHashMaps() {}
    
    ProblemSpecP findVariable(const string& name, int matl,
			  XMLURL& foundUrl);
    
    // note that vector is offset by one to allow for matl=-1 at element 0
    const vector<VarHashMap>& getVarHashMaps() const
    { return d_varHashMaps; }
  private:
    void add(const string& name, int matl, ProblemSpecP varNode, XMLURL url);
    
    vector<VarHashMap> d_varHashMaps;
  };
  
public:
  DataArchive(const string& filebase,
	      int processor=0 /* use if you want to different processors
				 to read different parts of the archive */,
	      int numProcessors=1);
  
  // GROUP: Destructors
  //////////
  // Destructor
  virtual ~DataArchive();
  
  void restartInitialize(int& timestep, const GridP& grid, DataWarehouse* dw,
			 double* pTime /* passed back */,
			 double* pDelt /* passed back */);
  
  // GROUP:  Information Access
  //////////
  // However, we need a means of determining the names of existing
  // variables. We also need to determine the type of each variable.
  // Get a list of scalar or vector variable names and  
  // a list of corresponding data types
  void queryVariables( vector< string>& names,
		       vector< const TypeDescription *>&  );
  void queryGlobals( vector< string>& names,
		     vector< const TypeDescription *>&  );
  void queryTimesteps( vector<int>& index,
		       vector<double>& times );
  GridP queryGrid( double time );

  void purgeTimestepCache(double time)
  { if (d_varHashMaps) d_varHashMaps->purgeTimeData(time); }
  
#if 0
  //////////
  // Does a variable exist in a particular patch?
  bool exists(const string&, const Patch*, int) {
    return true;
  }
#endif
   
  //////////
  // how long does a particle live?  Not variable specific.
  void queryLifetime( double& min, double& max, particleId id);
  
  //////////
  // how long does a patch live?  Not variable specific
  void queryLifetime( double& min, double& max, const Patch* patch);
  
  ConsecutiveRangeSet queryMaterials(const string& name,
				     const Patch* patch, double time);
  
  int queryNumMaterials(const Patch* patch, double time);
  
  void query( Variable& var, const string& name,
	      int matlIndex, const Patch* patch, double tine );
  
  //////////
  // query the variable value for a particular particle  overtime;
  // T = double/float/vector/Tensor I'm not sure of the proper
  // syntax.
  template<class T>
  void query( ParticleVariable< T >&, const string& name, int matlIndex,
	      particleId id,
	      double min, double max);
  
  //////////
  // query the variable value for a particular particle  overtime;
  // T = double/float/vector/Tensor I'm not sure of the proper
  // syntax.
  template<class T>
  void query( NCVariable< T >&, const string& name, int matlIndex,
	      const IntVector& index,
	      double min, double max);
  
  //////////
  // query the variable value for a particular particle  overtime;
  // T = double/float/vector/Tensor I'm not sure of the proper
  // syntax.
  template<class T>
  void query( CCVariable< T >&, const string& name, int matlIndex,
	      const IntVector& index,
	      double min, double max);
  
  //////////
  // query the variable value for a particular particle  overtime;
  template<class T>
  void query(vector<T>& values, const string& name,
	     int matlIndex, long64 particleID,
	     double startTime, double endTime) ;
  //////////
  // similarly, we want to be able to track variable values in a particular
  // patch cell over time.
  template<class T>
  void query(vector<T>& values, const string& name, int matlIndex,
	     IntVector loc, double startTime, double endTime);

  //////////
  // Pass back the timestep number specified in the "restart" tag of the
  // index file, or return false if such a tag does not exist.
  bool queryRestartTimestep(int& timestep);
#if 0
  //////////
  // In other cases we will have noticed something interesting and we
  // will want to access some small portion of a patch.  We will need
  // to request some range of data in index space.
  template<class T> void get(T& data, const string& name,
			     const Patch* patch, cellIndex min, cellIndex max);
#endif

  static bool cacheOnlyCurrentTimestep;
protected:
  DataArchive();
  
private:
  DataArchive(const DataArchive&);
  DataArchive& operator=(const DataArchive&);
  
  ProblemSpecP getTimestep(double time, XMLURL& url);
  string queryEndianness();  
  int queryNBits();  
  void query( Variable& var, ProblemSpecP vnode, XMLURL url,
	      int matlIndex,	const Patch* patch );

  void queryVariables( const ProblemSpecP vars, vector<string>& names,
		       vector<const TypeDescription*>& types);

  
  TimeHashMaps* getTopLevelVarHashMaps()
  {
    if (d_varHashMaps == NULL) {
      vector<int> indices;
      vector<double> times;
      queryTimesteps(indices, times);
      d_varHashMaps = scinew TimeHashMaps(times, d_tsurl, d_tstop,
					  d_processor, d_numProcessors);
    }
    return d_varHashMaps;
  }
  
  // for restartInitialize
  void initVariable(const Patch* patch,
		    DataWarehouse* new_dw,
		    VarLabel* label, int matl,
		    pair<ProblemSpecP, XMLURL> dataRef);   
  
  std::string d_filebase;  
  ProblemSpecP d_indexDoc;
  XMLURL d_base;

  bool d_swapBytes;
  int d_nBytes;
  
  bool have_timesteps;
  std::vector<int> d_tsindex;
  std::vector<double> d_tstimes;
  std::vector<ProblemSpecP> d_tstop;
  std::vector<XMLURL> d_tsurl;
  TimeHashMaps* d_varHashMaps;
  
  typedef map<pair<int, const Patch*>, Handle<ParticleSubset> > psetDBType;
  psetDBType d_psetDB;
 
  // if used, different processors read different parts of the archive
  int d_processor;
  int d_numProcessors;
  
  Mutex d_lock;
  
  ProblemSpecP findVariable(const string& name, const Patch* patch,
			int matl, double time, XMLURL& url);
  void findPatchAndIndex(GridP grid, Patch*& patch, particleIndex& idx,
			 long64 particleID, int matIndex,
			 double time);
  static DebugStream dbg;
};

   
  template<class T>
  void DataArchive::query( NCVariable< T >&, const string& name, int matlIndex,
			   const IntVector& index,
			   double min, double max)
  {
    cerr << "DataArchive::query not finished\n";
  }
  
  template<class T>
  void DataArchive::query( CCVariable< T >&, const string& name, int matlIndex,
			   const IntVector& index,
			   double min, double max)
  {
    cerr << "DataArchive::query not finished\n";
  }
  
  template<class T>
  void DataArchive::query(ParticleVariable< T >& var, const string& name,
			  int matlIndex, particleId id,
			  double min, double max)
  {
    cerr << "DataArchive::query not finished\n";
  }
  
  
  template<class T>
  void DataArchive::query(vector<T>& values, const string& name,
			  int matlIndex, long64 particleID,
			  double startTime, double endTime)
  {
    double call_start = SCIRun::Time::currentSeconds();
    
    if (!have_timesteps) {
      vector<int> index;
      vector<double> times;
      queryTimesteps(index, times);
      // will build d_ts* as a side effect
    }
    // figure out what kind of variable we're looking for
    vector<string> type_names;
    vector<const TypeDescription*> type_descriptions;
    queryVariables(type_names, type_descriptions);
    const TypeDescription* type = NULL;
    vector<string>::iterator name_iter = type_names.begin();
    vector<const TypeDescription*>::iterator type_iter = type_descriptions.begin();
    for ( ; name_iter != type_names.end() && type == NULL;
	  name_iter++, type_iter++) {
      if (*name_iter == name)
	type = *type_iter;
    }
    if (type == NULL)
      throw InternalError("Unable to determine variable type");
    if (type->getType() != TypeDescription::ParticleVariable)    
      throw InternalError("Variable type is not ParticleVariable");
    // find the first timestep
    int ts = 0;
    while ((ts < (int)d_tstimes.size()) && (startTime > d_tstimes[ts]))
      ts++;
    GridP grid = queryGrid( d_tstimes[ts] );
    Patch* patch = NULL;
    // idx needs to be initialized before it is used in findPatchAndIndex.
    particleIndex idx = 0;
    for ( ; (ts < (int)d_tstimes.size()) && (d_tstimes[ts] <= endTime); ts++) {
      double t = d_tstimes[ts];
      // figure out what patch contains the cell. As far as I can tell,
      // nothing prevents this from changing between timesteps, so we have to
      // do this every time -- if that can't actually happen we might be able
      // to speed this up.
      findPatchAndIndex(grid, patch, idx, particleID, matlIndex, t);
      //    cerr <<" Patch = 0x"<<hex<<patch<<dec<<", index = "<<idx;
      if (patch == NULL)
	throw VariableNotFoundInGrid(name,particleID,matlIndex,
				     "DataArchive::query");
      
      ParticleVariable<T> var;
      query(var, name, matlIndex, patch, t);
      //now find the index that corresponds to the particleID
      //cerr <<" time = "<<t<<",  value = "<<var[idx]<<endl;
      values.push_back(var[idx]);
      
    }
    dbg << "DataArchive::query(values) completed in "
	<< (SCIRun::Time::currentSeconds() - call_start) << " seconds\n";
  }  
  
  template<class T>
  void DataArchive::query(vector<T>& values, const string& name,
			  int matlIndex, IntVector loc,
			  double startTime, double endTime)
  {
    double call_start = SCIRun::Time::currentSeconds();
    
    if (!have_timesteps) {
      vector<int> index;
      vector<double> times;
      queryTimesteps(index, times);
      // will build d_ts* as a side effect
    }
    
    // figure out what kind of variable we're looking for
    vector<string> type_names;
    vector<const TypeDescription*> type_descriptions;
    queryVariables(type_names, type_descriptions);
    const TypeDescription* type = NULL;
    vector<string>::iterator name_iter = type_names.begin();
    vector<const TypeDescription*>::iterator type_iter = type_descriptions.begin();
    for ( ; name_iter != type_names.end() && type == NULL;
	  name_iter++, type_iter++) {
      if (*name_iter == name)
	type = *type_iter;
    }
    if (type == NULL)
      throw InternalError("Unable to determine variable type");
    
    // find the first timestep
    int ts = 0;
    while ((ts < (int)d_tstimes.size()) && (startTime > d_tstimes[ts]))
      ts++;
    
    for ( ; (ts < (int)d_tstimes.size()) && (d_tstimes[ts] <= endTime); ts++) {
      double t = d_tstimes[ts];
      
      // figure out what patch contains the cell. As far as I can tell,
      // nothing prevents this from changing between timesteps, so we have to
      // do this every time -- if that can't actually happen we might be able
      // to speed this up.
      Patch* patch = NULL;
      GridP grid = queryGrid(t);
      for (int level_nr = 0;
	   (level_nr < grid->numLevels()) && (patch == NULL); level_nr++) {
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
	  cerr << "Variable of unsupported type for this cell-based query: " << type->getType() << '\n';
	  break;
	}
      }
      if (patch == NULL) {
	throw VariableNotFoundInGrid(name,loc,matlIndex,"DataArchive::query");
      }
     
      switch (type->getType()) {
      case TypeDescription::CCVariable: {
	CCVariable<T> var;
	query(var, name, matlIndex, patch, t);
	values.push_back(var[loc]);
      } break;
      
      case TypeDescription::NCVariable: {
	NCVariable<T> var;
	query(var, name, matlIndex, patch, t);
	values.push_back(var[loc]);
      } break;

      case TypeDescription::SFCXVariable: {
	SFCXVariable<T> var;
	query(var, name, matlIndex, patch, t);
	values.push_back(var[loc]);
      } break;
      
      case TypeDescription::SFCYVariable: {
	SFCYVariable<T> var;
	query(var, name, matlIndex, patch, t);
	values.push_back(var[loc]);
      } break;
      
      case TypeDescription::SFCZVariable: {
	SFCZVariable<T> var;
	query(var, name, matlIndex, patch, t);
	values.push_back(var[loc]);
      } break;
      
      default:
	// Dd: Is this correct?  Error here?
	break;
      }
      //cerr << "DataArchive::query:data extracted" << endl;
    }
    
    dbg << "DataArchive::query(values) completed in "
        << (SCIRun::Time::currentSeconds() - call_start) << " seconds\n";
  }
  
} // end namespace Uintah

#endif

