#ifndef UINTAH_HOMEBREW_DataArchive_H
#define UINTAH_HOMEBREW_DataArchive_H

#include <Uintah/Grid/ParticleVariable.h>
#include <Uintah/Grid/NCVariable.h>
#include <Uintah/Grid/CCVariable.h>
#include <Uintah/Grid/GridP.h>
#include <SCICore/Thread/Mutex.h>
#include <SCICore/Thread/Time.h>
#include <SCICore/Util/DebugStream.h>
#include <string>
#include <vector>
#include <list>
#include <hash_map>

#ifdef __sgi
#define IRIX
#pragma set woff 1375
#endif
#include <util/PlatformUtils.hpp>
#include <parsers/DOMParser.hpp>
#include <dom/DOM_Node.hpp>
#include <dom/DOM_NamedNodeMap.hpp>
#include <util/XMLURL.hpp>
#ifdef __sgi
#pragma reset woff 1375
#endif
#include <fcntl.h>
#include <unistd.h>

namespace Uintah {
   class Patch;
   using SCICore::Thread::Mutex;
   using SCICore::Thread::Time;
   using namespace PSECore::XMLUtil;
   using SCICore::Util::DebugStream;
   using namespace std;

   struct eqstr { // comparison class used in hash_map to compare keys
     bool operator()(const char* s1, const char* s2) const {
       return strcmp(s1, s2) == 0;
     }
   };

   typedef hash_map<const char*, pair<DOM_Node, XMLURL*>, hash<const char*>, 
                    eqstr> VarHashMap;

   typedef hash_map<const char*, pair<DOM_Node, XMLURL*>, hash<const char*>,
                    eqstr>::iterator VarHashMapIterator;

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

  // Top of data structure for storing hash maps of variable data
  // - containing data for each time step.
  class TimeHashMaps {
  public:
    TimeHashMaps(const vector<double>& tsTimes,
		 const vector<XMLURL>& tsUrls,
		 const vector<DOM_Node>& tsTopNodes);
    
    DOM_Node findVariable(const string& name, const Patch* patch, int matl,
			  double time, XMLURL& foundUrl);
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
    void init(XMLURL tsUrl, DOM_Node tsTopNode);
    
    DOM_Node findVariable(const string& name, const Patch* patch,
			  int matl, XMLURL& foundUrl)      ;
  private:
    void parse();    
    void add(const string& name, int patchid, int matl,
	     DOM_Node varNode, XMLURL* pUrl)
    { d_matHashMaps[patchid].add(name, matl, varNode, pUrl); }

    map<int, MaterialHashMaps> d_matHashMaps;
    map<int, MaterialHashMaps>::iterator d_lastFoundIt;
    list<XMLURL> d_xmlUrls;
    bool d_isParsed;
  };

  // Third layer of data structure for storing hash maps of variable data
  // - containing data for each material at a certain patch and time step.
  class MaterialHashMaps {
    friend class PatchHashMaps;
  public:
    MaterialHashMaps() {}

    DOM_Node findVariable(const string& name, int matl,
			  XMLURL& foundUrl);
  private:
    void add(const string& name, int matl, DOM_Node varNode, XMLURL* pUrl);

    vector<VarHashMap> d_varHashMaps;

    // store a copy of the variable names so that char*'s don't become invalid
    // in the hash table.
    list<string> d_varNames;
  };
  
public:
   DataArchive(const std::string& filebase);
  // GROUP: Destructors
  //////////
  // Destructor
  virtual ~DataArchive();

    
   // GROUP:  Information Access
   //////////
   // However, we need a means of determining the names of existing
   // variables. We also need to determine the type of each variable.
   // Get a list of scalar or vector variable names and  
   // a list of corresponding data types
   void queryVariables( std::vector< std::string>& names,
		       std::vector< const TypeDescription *>&  );
   void queryTimesteps( std::vector<int>& index,
		       std::vector<double>& times );
   GridP queryGrid( double time );
   
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

   int queryNumMaterials(const std::string& name, const Patch* patch,
			double time);

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
   void query( ParticleVariable< T >&, const std::string& name, int matlIndex,
	      const Patch*, double time );
   
   //////////
   // query the variable value for a particular particle  overtime;
   // T = double/float/vector/Tensor I'm not sure of the proper
   // syntax.
   template<class T>
   void query( NCVariable< T >&, const std::string& name, int matlIndex,
	      const Patch*, double time );


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
	      const Patch*, double time );


   //////////
   // query the variable value for a particular particle  overtime;
   // T = double/float/vector/Tensor I'm not sure of the proper
   // syntax.
   template<class T>
   void query( CCVariable< T >&, const std::string& name, int matlIndex,
	      const IntVector& index,
	      double min, double max);
   
   //////////
   // similarly, we want to be able to track variable values in a particular
   // patch cell over time.
   template<class T>
   void query(std::vector<T>& values, const std::string& name, int matlIndex,
    	      IntVector loc, double startTime, double endTime);
   
#if 0
   //////////
   // In other cases we will have noticed something interesting and we
   // will want to access some small portion of a patch.  We will need
   // to request some range of data in index space.
   template<class T> void get(T& data, const std::string& name,
			      const Patch* patch, cellIndex min, cellIndex max);
#endif
   
protected:
   DataArchive();
   
private:
   DataArchive(const DataArchive&);
   DataArchive& operator=(const DataArchive&);

   DOM_Node getTimestep(double time, XMLURL& url);
   
   std::string d_filebase;
   DOM_Document d_indexDoc;
   XMLURL d_base;

   bool have_timesteps;
   std::vector<int> d_tsindex;
   std::vector<double> d_tstimes;
   std::vector<DOM_Node> d_tstop;
   std::vector<XMLURL> d_tsurl;
   TimeHashMaps* d_varHashMaps;
  
   Mutex d_lock;

   DOM_Node findVariable(const string& name, const Patch* patch,
			 int matl, double time, XMLURL& url);

   static DebugStream dbg;
};

template<class T>
void DataArchive::query( ParticleVariable< T >& var, const std::string& name,
			 int matlIndex,	const Patch* patch, double time )
{
   double tstart = Time::currentSeconds();
   XMLURL url;
   DOM_Node vnode = findVariable(name, patch, matlIndex, time, url);
   if(vnode == 0){
      cerr << "VARIABLE NOT FOUND: " << name << ", index " << matlIndex << ", patch " << patch->getID() << ", time " << time << '\n';
      throw InternalError("Variable not found");
   }
   DOM_NamedNodeMap attributes = vnode.getAttributes();
   DOM_Node typenode = attributes.getNamedItem("type");
   if(typenode == 0)
      throw InternalError("Variable doesn't have a type");
   string type = toString(typenode.getNodeValue());
   const TypeDescription* td = TypeDescription::lookupType(type);
   ASSERT(td == ParticleVariable<T>::getTypeDescription());
   int numParticles;
   if(!get(vnode, "numParticles", numParticles))
      throw InternalError("Cannot get numParticles");
   ParticleSubset* psubset = scinew ParticleSubset(scinew ParticleSet(numParticles),
						   true, matlIndex, patch);
   var.allocate(psubset);
   long start;
   if(!get(vnode, "start", start))
      throw InternalError("Cannot get start");
   long end;
   if(!get(vnode, "end", end))
      throw InternalError("Cannot get end");
   string filename;
   if(!get(vnode, "filename", filename))
      throw InternalError("Cannot get filename");
   XMLURL dataurl(url, filename.c_str());
   if(dataurl.getProtocol() != XMLURL::File)
      throw InternalError(string("Cannot read over: ")
			  +toString(dataurl.getProtocolName()));
   string datafile(toString(dataurl.getPath()));

   int fd = open(datafile.c_str(), O_RDONLY);
   if(fd == -1)
      throw ErrnoException("DataArchive::query (open call)", errno);
   off64_t ls = lseek64(fd, start, SEEK_SET);
   if(ls == -1)
      throw ErrnoException("DataArchive::query (lseek64 call)", errno);

   InputContext ic(fd, start);
   var.read(ic);
   ASSERTEQ(end, ic.cur);
   int s = close(fd);
   if(s == -1)
      throw ErrnoException("DataArchive::query (read call)", errno);
   dbg << "DataArchive::query(ParticleVariable) completed in " << Time::currentSeconds()-tstart << " seconds\n";
}
   
template<class T>
void DataArchive::query(ParticleVariable< T >& var, const std::string& name,
			int matlIndex, particleId id,
			double min, double max)
{
   cerr << "DataArchive::query not finished\n";
}

template<class T>
void DataArchive::query( NCVariable< T >& var, const std::string& name,
			 int matlIndex, const Patch* patch, double time )
{
   double tstart = Time::currentSeconds();
   XMLURL url;
   DOM_Node vnode = findVariable(name, patch, matlIndex, time, url);
   if(vnode == 0){
      cerr << "VARIABLE NOT FOUND: " << name << ", index " << matlIndex << ", patch " << patch->getID() << ", time " << time << '\n';
      throw InternalError("Variable not found");
   }
   DOM_NamedNodeMap attributes = vnode.getAttributes();
   DOM_Node typenode = attributes.getNamedItem("type");
   if(typenode == 0)
      throw InternalError("Variable doesn't have a type");
   string type = toString(typenode.getNodeValue());
   const TypeDescription* td = TypeDescription::lookupType(type);
//   ASSERT(td == NCVariable<T>::getTypeDescription());
   var.allocate(patch->getNodeLowIndex(), patch->getNodeHighIndex());
   long start;
   if(!get(vnode, "start", start))
      throw InternalError("Cannot get start");
   long end;
   if(!get(vnode, "end", end))
      throw InternalError("Cannot get end");
   string filename;
   if(!get(vnode, "filename", filename))
      throw InternalError("Cannot get filename");
   XMLURL dataurl(url, filename.c_str());
   if(dataurl.getProtocol() != XMLURL::File)
      throw InternalError(string("Cannot read over: ")
			  +toString(dataurl.getProtocolName()));
   string datafile(toString(dataurl.getPath()));

   int fd = open(datafile.c_str(), O_RDONLY);
   if(fd == -1)
      throw ErrnoException("DataArchive::query (open call)", errno);
   off64_t ls = lseek64(fd, start, SEEK_SET);
   if(ls == -1)
      throw ErrnoException("DataArchive::query (lseek64 call)", errno);

   InputContext ic(fd, start);
   var.read(ic);
   ASSERTEQ(end, ic.cur);
   int s = close(fd);
   if(s == -1)
      throw ErrnoException("DataArchive::query (read call)", errno);
   dbg << "DataArchive::query(NCVariable) completed in " << Time::currentSeconds()-tstart << " seconds\n";
}

template<class T>
void DataArchive::query( NCVariable< T >&, const std::string& name, int matlIndex,
			const IntVector& index,
			double min, double max)
{
   cerr << "DataArchive::query not finished\n";
}

template<class T>
void DataArchive::query( CCVariable< T >& var, const std::string& name,
			 int matlIndex, const Patch* patch, double time )
{
   double tstart = Time::currentSeconds();
   XMLURL url;
   DOM_Node vnode = findVariable(name, patch, matlIndex, time, url);
   if(vnode == 0){
      cerr << "VARIABLE NOT FOUND: " << name << ", index " << matlIndex << ", patch " << patch->getID() << ", time " << time << '\n';
      throw InternalError("Variable not found");
   }
   DOM_NamedNodeMap attributes = vnode.getAttributes();
   DOM_Node typenode = attributes.getNamedItem("type");
   if(typenode == 0)
      throw InternalError("Variable doesn't have a type");
   string type = toString(typenode.getNodeValue());
   const TypeDescription* td = TypeDescription::lookupType(type);
   ASSERT(td == CCVariable<T>::getTypeDescription());
   var.allocate(patch->getCellLowIndex(), patch->getCellHighIndex());
   long start;
   if(!get(vnode, "start", start))
      throw InternalError("Cannot get start");
   long end;
   if(!get(vnode, "end", end))
      throw InternalError("Cannot get end");
   string filename;
   if(!get(vnode, "filename", filename))
      throw InternalError("Cannot get filename");
   XMLURL dataurl(url, filename.c_str());
   if(dataurl.getProtocol() != XMLURL::File)
      throw InternalError(string("Cannot read over: ")
			  +toString(dataurl.getProtocolName()));
   string datafile(toString(dataurl.getPath()));

   int fd = open(datafile.c_str(), O_RDONLY);
   if(fd == -1)
      throw ErrnoException("DataArchive::query (open call)", errno);
   off64_t ls = lseek64(fd, start, SEEK_SET);
   if(ls == -1)
      throw ErrnoException("DataArchive::query (lseek64 call)", errno);

   InputContext ic(fd, start);
   var.read(ic);
   ASSERTEQ(end, ic.cur);
   int s = close(fd);
   if(s == -1)
      throw ErrnoException("DataArchive::query (read call)", errno);
   dbg << "DataArchive::query(CCVariable) completed in " << Time::currentSeconds()-tstart << " seconds\n";
}

template<class T>
void DataArchive::query( CCVariable< T >&, const std::string& name, int matlIndex,
			const IntVector& index,
			double min, double max)
{
   cerr << "DataArchive::query not finished\n";
}

template<class T>
void DataArchive::query(std::vector<T>& values, const std::string& name,
    	    	    	int matlIndex, IntVector loc,
			double startTime, double endTime)
{
    double call_start = Time::currentSeconds();

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
    while ((ts < d_tstimes.size()) && (startTime > d_tstimes[ts]))
    	ts++;

    for ( ; (ts < d_tstimes.size()) && (d_tstimes[ts] < endTime); ts++) {
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
    	    	    if ((*iter)->containsCell(loc))
    	    	    	patch = *iter;
    	    	}
    	    	break;
	    
	    case TypeDescription::NCVariable:
	    	// Unfortunately, this const cast hack is necessary.
    	    	patch = ((LevelP)level)->getPatchFromPoint(level->getNodePosition(loc));
		break;
	    }
	}
	if (patch == NULL)
	    throw InternalError("Couldn't find patch containing location");

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
	}
    }

    dbg << "DataArchive::query(values) completed in "
        << (Time::currentSeconds() - call_start) << " seconds\n";
}

} // end namespace Uintah

//
// $Log$
// Revision 1.11  2000/09/15 22:08:34  witzel
// Changed the variable hash map structure so that it only parses data xml
// files for a timestep after that timestep has be queried (instead of
// parsing all of them at first).   Also added d_lastFoundIt members to
// TimeHashMaps and PatchHashMaps to speed things up when multiple queries
// are made to the same timestep and/or patch (doesn't really seem to have
// much of an affect, but I'll keep it for now anyway).
//
// Revision 1.10  2000/09/14 23:59:06  witzel
// Changed findVariable method to make it much more efficient and not
// have to search through xml files over and over again.  The first
// time it is called it creates a data structure with hash tables to
// speed up variable searches on subsequent calls.
//
// Revision 1.9  2000/08/12 23:29:18  jehall
// Added a DataArchive query for tracking the value of a variable at a
// node/cell across multiple timesteps.
//
// Revision 1.8  2000/07/11 19:44:50  kuzimmer
// commented out line 284:  ASSERT(td == NCVariable<T>::getTypeDescription()); This was failing because two NCVariable<double> typeDescriptors were getting created.  This should not happen, so a bug report has been filed
//
// Revision 1.7  2000/06/27 18:28:35  bigler
// Steve did some fixing up and moving around
//
// Revision 1.6  2000/06/15 21:57:22  sparker
// Added multi-patch support (bugzilla #107)
// Changed interface to datawarehouse for particle data
// Particles now move from patch to patch
//
// Revision 1.5  2000/05/30 20:19:40  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.4  2000/05/21 08:19:11  sparker
// Implement NCVariable read
// Do not fail if variable type is not known
// Added misc stuff to makefiles to remove warnings
//
// Revision 1.3  2000/05/20 08:09:36  sparker
// Improved TypeDescription
// Finished I/O
// Use new XML utility libraries
//
// Revision 1.2  2000/05/20 02:34:56  kuzimmer
// Multiple changes for new vis tools and DataArchive
//
// Revision 1.1  2000/05/18 16:01:30  sparker
// Add data archive interface
//
//

#endif

