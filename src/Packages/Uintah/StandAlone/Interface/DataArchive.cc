#include <Uintah/Interface/DataArchive.h>
#include <Uintah/Grid/Grid.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Interface/InputContext.h>
#include <SCICore/Exceptions/InternalError.h>
#include <SCICore/Util/Assert.h>
#include <SCICore/Thread/Time.h>
#include <PSECore/XMLUtil/SimpleErrorHandler.h>
#include <PSECore/XMLUtil/XMLUtil.h>
#include <SCICore/Util/DebugStream.h>
#include <iostream>
#include <fstream>
#include <sax/SAXException.hpp>
#include <sax/SAXParseException.hpp>
#include <sax/ErrorHandler.hpp>
#include <fcntl.h>
#include <unistd.h>
#include <sys/param.h>

using namespace Uintah;
using namespace std;
using namespace SCICore::Exceptions;
using namespace PSECore::XMLUtil;
using SCICore::Thread::Time;
using SCICore::Util::DebugStream;

DebugStream DataArchive::dbg("DataArchive", false);

DataArchive::DataArchive(const std::string& filebase)
   : d_filebase(filebase), d_varHashMaps(NULL), d_lock("DataArchive lock")
{
   have_timesteps=false;
   string index(filebase+"/index.xml");
   XMLCh* tmpRel = XMLString::transcode(index.c_str());
   d_base.setURL(tmpRel);
   delete[] tmpRel;

   if(d_base.isRelative()){
      char path[MAXPATHLEN];
      string url = string("file://")+getwd(path)+"/.";
      d_base.makeRelativeTo(url.c_str());
      if(d_base.isRelative())
	 cerr << "base is still relative!\n";
   }
   
   DOMParser parser;
   parser.setDoValidation(false);

   SimpleErrorHandler handler;
   parser.setErrorHandler(&handler);

   cout << "Parsing " << toString(d_base.getURLText()) << endl;
   parser.parse(d_base.getURLText());

   if(handler.foundError)
      throw InternalError("Error reading file: "+toString(d_base.getURLText()));

   d_indexDoc = parser.getDocument();
}


DataArchive::~DataArchive()
{
  delete d_varHashMaps;
}

void DataArchive::queryTimesteps( std::vector<int>& index,
				 std::vector<double>& times )
{
   double start = Time::currentSeconds();
   if(!have_timesteps){
      d_lock.lock();
      if(!have_timesteps){
	 DOM_Node ts = findNode("timesteps", d_indexDoc.getDocumentElement());
	 if(ts == 0)
	    throw InternalError("timesteps node not found in index.xml");
	 for(DOM_Node t = ts.getFirstChild(); t != 0; t = t.getNextSibling()){
	    if(t.getNodeType() == DOM_Node::ELEMENT_NODE){
	       DOM_NamedNodeMap attributes = t.getAttributes();
	       DOM_Node tsfile = attributes.getNamedItem("href");
	       if(tsfile == 0)
		  throw InternalError("timestep href not found");
	    
	       DOMString href_name = tsfile.getNodeValue();
	       XMLURL url(d_base, toString(href_name).c_str());
	       DOMParser parser;
	       parser.setDoValidation(false);
	    
	       SimpleErrorHandler handler;
	       parser.setErrorHandler(&handler);
	    
	       //cerr << "reading: " << toString(url.getURLText()) << '\n';
	       parser.parse(url.getURLText());
	       if(handler.foundError)
		  throw InternalError("Cannot read timestep file");
	    
	       DOM_Node top = parser.getDocument().getDocumentElement();
	       d_tstop.push_back(top);
	       d_tsurl.push_back(url);
	       DOM_Node time = findNode("Time", top);
	       if(time == 0)
		  throw InternalError("Cannot find Time block");
	    
	       int timestepNumber;
	       if(!get(time, "timestepNumber", timestepNumber))
		  throw InternalError("Cannot find timestepNumber");
	       double currentTime;
	       if(!get(time, "currentTime", currentTime))
		  throw InternalError("Cannot find currentTime");
	       d_tsindex.push_back(timestepNumber);
	       d_tstimes.push_back(currentTime);
	    }
	 }
	 have_timesteps=true;
      }
      d_lock.unlock();
   }
   index=d_tsindex;
   times=d_tstimes;
   dbg << "DataArchive::queryTimesteps completed in " << Time::currentSeconds()-start << " seconds\n";
}

DOM_Node DataArchive::getTimestep(double searchtime, XMLURL& found_url)
{
   if(!have_timesteps){
      vector<int> index;
      vector<double> times;
      queryTimesteps(index, times);
      // Will build d_ts* as a side-effect...
   }
   int i;
   for(i=0;i<(int)d_tstimes.size();i++)
      if(searchtime == d_tstimes[i])
	 break;
   if(i == (int)d_tstimes.size())
      return DOM_Node();
   found_url = d_tsurl[i];
   return d_tstop[i];
}

DOM_Node DataArchive::findVariable(const string& name, const Patch* patch,
				   int matl, double time, XMLURL& url)
{
  if (d_varHashMaps == NULL) {
    vector<int> indices;
    vector<double> times;
    queryTimesteps(indices, times);
    d_varHashMaps = scinew TimeHashMaps(times, d_tsurl, d_tstop);
  }
  
  return d_varHashMaps->findVariable(name, patch, matl, time, url);
}

GridP DataArchive::queryGrid( double time )
{
   double start = Time::currentSeconds();
   XMLURL url;
   d_lock.lock();
   DOM_Node top = getTimestep(time, url);
   DOM_Node gridnode = findNode("Grid", top);
   if(gridnode == 0)
      throw InternalError("Cannot find Grid in timestep");
   int numLevels = -1234;
   GridP grid = scinew Grid;
   for(DOM_Node n = gridnode.getFirstChild(); n != 0; n=n.getNextSibling()){
      if(n.getNodeName().equals(DOMString("numLevels"))){
	 if(!get(n, numLevels))
	    throw InternalError("Error parsing numLevels");
      } else if(n.getNodeName().equals(DOMString("Level"))){
	 Point anchor;
	 if(!get(n, "anchor", anchor))
	    throw InternalError("Error parsing level anchor point");
	 Vector dcell;
	 if(!get(n, "cellspacing", dcell))
	    throw InternalError("Error parsing level cellspacing");
	 LevelP level = scinew Level(grid.get_rep(), anchor, dcell);
	 int numPatches = -1234;
	 long totalCells = 0;
	 for(DOM_Node r = n.getFirstChild(); r != 0; r=r.getNextSibling()){
	    if(r.getNodeName().equals("numPatches") ||
	       r.getNodeName().equals("numRegions")){
	       if(!get(r, numPatches))
		  throw InternalError("Error parsing numRegions");
	    } else if(r.getNodeName().equals("totalCells")){
	       if(!get(r, totalCells))
		  throw InternalError("Error parsing totalCells");
	    } else if(r.getNodeName().equals("Patch") ||
		      r.getNodeName().equals("Region")){
	       int id;
	       if(!get(r, "id", id))
		  throw InternalError("Error parsing patch id");
	       IntVector lowIndex;
	       if(!get(r, "lowIndex", lowIndex))
		  throw InternalError("Error parsing patch lowIndex");
	       IntVector highIndex;
	       if(!get(r, "highIndex", highIndex))
		  throw InternalError("Error parsing patch highIndex");
	       long totalCells;
	       if(!get(r, "totalCells", totalCells))
		  throw InternalError("Error parsing patch total cells");
	       Patch* r = level->addPatch(lowIndex, highIndex, id);
	       ASSERTEQ(r->totalCells(), totalCells);
	    } else if(r.getNodeName().equals("anchor")
		      || r.getNodeName().equals("cellspacing")){
	       // Nothing - handled above
	    } else if(r.getNodeType() != DOM_Node::TEXT_NODE){
	       cerr << "WARNING: Unknown level data: " << ::toString(n.getNodeName()) << '\n';
	    }
	 }
	 ASSERTEQ(level->numPatches(), numPatches);
	 ASSERTEQ(level->totalCells(), totalCells);
	 level->finalizeLevel();
	 grid->addLevel(level);
      } else if(n.getNodeType() != DOM_Node::TEXT_NODE){
	 cerr << "WARNING: Unknown grid data: " << toString(n.getNodeName()) << '\n';
      }
   }
   d_lock.unlock();
   grid->performConsistencyCheck();
   ASSERTEQ(grid->numLevels(), numLevels);
   dbg << "DataArchive::queryGrid completed in " << Time::currentSeconds()-start << " seconds\n";
   return grid;
}

void DataArchive::queryLifetime( double& min, double& max, particleId id)
{
   cerr << "DataArchive::lifetime not finished\n";
}

void DataArchive::queryLifetime( double& min, double& max, const Patch* patch)
{
   cerr << "DataArchive::lifetime not finished\n";
}

void DataArchive::queryVariables( vector<string>& names,
				 vector<const TypeDescription*>& types)
{
   double start = Time::currentSeconds();
   d_lock.lock();
   DOM_Node vars = findNode("variables", d_indexDoc.getDocumentElement());
   if(vars == 0)
      throw InternalError("variables section not found\n");
   for(DOM_Node n = vars.getFirstChild(); n != 0; n = n.getNextSibling()){
      if(n.getNodeName().equals("variable")){
	 DOM_NamedNodeMap attributes = n.getAttributes();
	 DOM_Node type = attributes.getNamedItem("type");
	 if(type == 0)
	    throw InternalError("Variable type not found");
	 string type_name = toString(type.getNodeValue());
	 const TypeDescription* td = TypeDescription::lookupType(type_name);
	 if(!td){
	    static TypeDescription* unknown_type = 0;
	    if(!unknown_type)
	       unknown_type = scinew TypeDescription(TypeDescription::Unknown,
						     "-- unknown type --",
						     false, -1);
	    td = unknown_type;
	 }
	 types.push_back(td);
	 DOM_Node name = attributes.getNamedItem("name");
	 if(name == 0)
	    throw InternalError("Variable name not found");
	 names.push_back(toString(name.getNodeValue()));
      } else if(n.getNodeType() != DOM_Node::TEXT_NODE){
	 cerr << "WARNING: Unknown variable data: " << toString(n.getNodeName()) << '\n';
      }
   }
   d_lock.unlock();
   dbg << "DataArchive::queryVariables completed in " << Time::currentSeconds()-start << " seconds\n";
}

DataArchive::TimeHashMaps::TimeHashMaps(const vector<double>& tsTimes,
					const vector<XMLURL>& tsUrls,
					const vector<DOM_Node>& tsTopNodes)
{
   ASSERTL3(tsTimes.size() == tsTopNodes.size());
   ASSERTL3(tsUrls.size() == tsTopNodes.size());

   for (int i = 0; i < (int)tsTimes.size(); i++)
     d_patchHashMaps[tsTimes[i]].init(tsUrls[i], tsTopNodes[i]);
   
   d_lastFoundIt = d_patchHashMaps.end();
}

DOM_Node DataArchive::TimeHashMaps::findVariable(const string& name,
						 const Patch* patch, int matl,
						 double time,
						 XMLURL& foundUrl)
{
  // assuming nearby queries will often be made sequentially,
  // checking the lastFound can reduce overall query times.
  if ((d_lastFoundIt != d_patchHashMaps.end()) &&
      ((*d_lastFoundIt).first == time))
    return (*d_lastFoundIt).second.findVariable(name, patch, matl, foundUrl);
  map<double, PatchHashMaps>::iterator foundIt =
    d_patchHashMaps.find(time);
  if (foundIt != d_patchHashMaps.end()) {
    d_lastFoundIt = foundIt;
    return (*foundIt).second.findVariable(name, patch, matl, foundUrl);
  }
  return DOM_Node();
}

DataArchive::PatchHashMaps::PatchHashMaps()
  : d_matHashMaps(),
    d_lastFoundIt(d_matHashMaps.end()),
    d_isParsed(false)
{
}

void DataArchive::PatchHashMaps::init(XMLURL tsUrl, DOM_Node tsTopNode)
{
  d_isParsed = false;
  // grab the data xml files from the timestep xml file
  ASSERTL3(tsTopNode != 0);
  DOM_Node datanode = findNode("Data", tsTopNode);
  if(datanode == 0)
    throw InternalError("Cannot find Data in timestep");
  for(DOM_Node n = datanode.getFirstChild(); n != 0; n=n.getNextSibling()){
    if(n.getNodeName().equals(DOMString("Datafile"))){
      DOM_NamedNodeMap attributes = n.getAttributes();
      DOM_Node datafile = attributes.getNamedItem("href");
      if(datafile == 0)
	throw InternalError("timestep href not found");
      
      DOMString href_name = datafile.getNodeValue();
      XMLURL url(tsUrl, toString(href_name).c_str());
      d_xmlUrls.push_back(url);
    }
    else if(n.getNodeType() != DOM_Node::TEXT_NODE){
      cerr << "WARNING: Unknown element in Data section: " << toString(n.getNodeName()) << '\n';
    }
  }
}

void DataArchive::PatchHashMaps::parse()
{
  for (list<XMLURL>::iterator urlIt = d_xmlUrls.begin();
       urlIt != d_xmlUrls.end(); urlIt++) {
    DOMParser parser;
    parser.setDoValidation(false);
    
    SimpleErrorHandler handler;
    parser.setErrorHandler(&handler);
    
    //cerr << "reading: " << toString(url.getURLText()) << '\n';
    parser.parse((*urlIt).getURLText());
    if(handler.foundError)
      throw InternalError("Cannot read timestep file");
    
    DOM_Node top = parser.getDocument().getDocumentElement();
    for(DOM_Node r = top.getFirstChild(); r != 0; r=r.getNextSibling()){
      if(r.getNodeName().equals(DOMString("Variable"))){
	string varname;
	if(!get(r, "variable", varname))
	  throw InternalError("Cannot get variable name");
	
	int patchid;
	if(!get(r, "patch", patchid) && !get(r, "region", patchid))
	  throw InternalError("Cannot get patch id");
	
	int index;
	if(!get(r, "index", index))
	  throw InternalError("Cannot get index");
	
	add(varname, patchid, index, r, &d_xmlUrls.back());
      } else if(r.getNodeType() != DOM_Node::TEXT_NODE){
	cerr << "WARNING: Unknown element in Variables section: " << toString(r.getNodeName()) << '\n';
      }
    }
  }
  
  d_isParsed = true;
  d_lastFoundIt = d_matHashMaps.end();
}

DOM_Node DataArchive::PatchHashMaps::findVariable(const string& name,
						  const Patch* patch,
						  int matl,
						  XMLURL& foundUrl)
{
  if (!d_isParsed) parse(); // parse on demand
  int patchid = patch->getID();
  
  // assuming nearby queries will often be made sequentially,
  // checking the lastFound can reduce overall query times.
  if ((d_lastFoundIt != d_matHashMaps.end()) &&
      ((*d_lastFoundIt).first == patchid))
    return (*d_lastFoundIt).second.findVariable(name, matl, foundUrl);

  map<int, MaterialHashMaps>::iterator foundIt =
    d_matHashMaps.find(patchid);
  
  if (foundIt != d_matHashMaps.end()) {
    d_lastFoundIt = foundIt;
    return (*foundIt).second.findVariable(name, matl, foundUrl);
  }
  return DOM_Node();  
}

DOM_Node DataArchive::MaterialHashMaps::findVariable(const string& name,
						     int matl,
						     XMLURL& foundUrl)
{
  if (matl < (int)d_varHashMaps.size()) {
    VarHashMap& hashMap = d_varHashMaps[matl];
    VarHashMapIterator foundIt = hashMap.find(name.c_str());
    
    if (foundIt !=  hashMap.end()) {
      foundUrl = *(*foundIt).second.second;
      return (*foundIt).second.first;
    }
  }
  return DOM_Node();  
}

void DataArchive::MaterialHashMaps::add(const string& name, int matl,
					DOM_Node varNode, XMLURL* pUrl)
{
  d_varNames.push_back(name);
  const char* var_name = d_varNames.back().c_str();
  if (matl >= (int)d_varHashMaps.size())
    d_varHashMaps.resize(matl + 1);
  pair<DOM_Node, XMLURL*> value(varNode, pUrl);
  pair<const char*, pair<DOM_Node, XMLURL*> > valkeypair(var_name, value);
  if (d_varHashMaps[matl].insert(valkeypair).second == false) {
    cerr << "Duplicate variable name: " << name << endl;
  }
}

int DataArchive::queryNumMaterials( const string& name, const Patch* patch, double time)
{
   double start = Time::currentSeconds();
   int i=-1;
   DOM_Node rnode;
   do {
      i++;
      XMLURL url;
      rnode = findVariable(name, patch, i, time, url);
   } while(rnode != 0);
   dbg << "DataArchive::queryNumMaterials completed in " << Time::currentSeconds()-start << " seconds\n";
   return i;
}


//
// $Log$
// Revision 1.11.2.1  2000/10/26 10:06:15  moulding
// merge HEAD into FIELD_REDESIGN
//
// Revision 1.12  2000/10/04 19:48:01  kuzimmer
// added some locks since the xerces stuff appears not to be thread safe
//
// Revision 1.11  2000/09/25 20:39:14  sparker
// Quiet g++ compiler warnings
//
// Revision 1.10  2000/09/15 22:08:34  witzel
// Changed the variable hash map structure so that it only parses data xml
// files for a timestep after that timestep has be queried (instead of
// parsing all of them at first).   Also added d_lastFoundIt members to
// TimeHashMaps and PatchHashMaps to speed things up when multiple queries
// are made to the same timestep and/or patch (doesn't really seem to have
// much of an affect, but I'll keep it for now anyway).
//
// Revision 1.9  2000/09/14 23:59:06  witzel
// Changed findVariable method to make it much more efficient and not
// have to search through xml files over and over again.  The first
// time it is called it creates a data structure with hash tables to
// speed up variable searches on subsequent calls.
//
// Revision 1.8  2000/07/27 22:39:52  sparker
// Implemented MPIScheduler
// Added associated support
//
// Revision 1.7  2000/06/27 18:28:36  bigler
// Steve did some fixing up and moving around
//
// Revision 1.6  2000/06/15 21:57:21  sparker
// Added multi-patch support (bugzilla #107)
// Changed interface to datawarehouse for particle data
// Particles now move from patch to patch
//
// Revision 1.5  2000/05/31 18:01:34  sparker
// More region backwards compatibility
//
// Revision 1.4  2000/05/31 03:11:12  sparker
// Made reader backwards compatible with pre-patch uda files
//
// Revision 1.3  2000/05/30 20:19:40  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.2  2000/05/21 08:19:11  sparker
// Implement NCVariable read
// Do not fail if variable type is not known
// Added misc stuff to makefiles to remove warnings
//
// Revision 1.1  2000/05/20 08:09:35  sparker
// Improved TypeDescription
// Finished I/O
// Use new XML utility libraries
//
//
