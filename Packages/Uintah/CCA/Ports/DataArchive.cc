#include <Packages/Uintah/CCA/Ports/DataArchive.h>
#include <Packages/Uintah/CCA/Ports/InputContext.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/UnknownVariable.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Core/Util/NotFinished.h>

#include <Dataflow/XMLUtil/SimpleErrorHandler.h>
#include <Dataflow/XMLUtil/XMLUtil.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Util/Assert.h>
#include <Core/Thread/Time.h>
#include <Core/Util/DebugStream.h>

#include <sax/SAXException.hpp>
#include <sax/SAXParseException.hpp>
#include <sax/ErrorHandler.hpp>

#include <sys/param.h>

#include <iostream>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>

using namespace std;

using namespace Uintah;
using namespace SCIRun;

DebugStream DataArchive::dbg("DataArchive", false);

DataArchive::DataArchive(const std::string& filebase,
			 int processor /* =0 */, int numProcessors /* =1 */)
  : d_filebase(filebase), d_varHashMaps(NULL),
    d_processor(processor), d_numProcessors(numProcessors),
    d_lock("DataArchive lock")
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

void
DataArchive::queryTimesteps( std::vector<int>& index,
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

DOM_Node
DataArchive::getTimestep(double searchtime, XMLURL& found_url)
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

DOM_Node
DataArchive::findVariable(const string& name, const Patch* patch,
			  int matl, double time, XMLURL& url)
{
  return getTopLevelVarHashMaps()->findVariable(name, patch, matl, time, url);
}

GridP
DataArchive::queryGrid( double time )
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
	  Patch* r = level->addPatch(lowIndex, highIndex,lowIndex,
				     highIndex,id);
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

void
DataArchive::queryLifetime( double& /*min*/, double& /*max*/,
			    particleId /*id*/)
{
  cerr << "DataArchive::lifetime not finished\n";
}

void
DataArchive::queryLifetime( double& /*min*/, double& /*max*/,
			    const Patch* /*patch*/)
{
  cerr << "DataArchive::lifetime not finished\n";
}

void
DataArchive::queryVariables( vector<string>& names,
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

void
DataArchive::query( Variable& var, const std::string& name,
		    int matlIndex, const Patch* patch, double time )
{
  double tstart = Time::currentSeconds();
  XMLURL url;
  d_lock.lock();  
  DOM_Node vnode = findVariable(name, patch, matlIndex, time, url);
  d_lock.unlock();
  if(vnode == 0){
    cerr << "VARIABLE NOT FOUND: " << name << ", index " << matlIndex << ", patch " << patch->getID() << ", time " << time << '\n';
    throw InternalError("Variable not found");
  }
  query(var, vnode, url, matlIndex, patch);
  dbg << "DataArchive::query() completed in "
      << Time::currentSeconds()-tstart << " seconds\n";
}

void
DataArchive::query( Variable& var, DOM_Node vnode, XMLURL url,
		    int matlIndex, const Patch* patch )
{
  d_lock.lock();
  DOM_NamedNodeMap attributes = vnode.getAttributes();
  DOM_Node typenode = attributes.getNamedItem("type");
  if(typenode == 0)
    throw InternalError("Variable doesn't have a type");
  string type = toString(typenode.getNodeValue());
  const TypeDescription* td = var.virtualGetTypeDescription();
  ASSERT(td->getName() == type);
  
  if (td->getType() == TypeDescription::ParticleVariable) {
    int numParticles;
    if(!get(vnode, "numParticles", numParticles))
      throw InternalError("Cannot get numParticles");
    ParticleSubset* psubset = scinew
      ParticleSubset(scinew ParticleSet(numParticles), true,
		     matlIndex, patch);
    (dynamic_cast<ParticleVariableBase*>(&var))->allocate(psubset);
  }
  else if (td->getType() != TypeDescription::ReductionVariable)
    var.allocate(patch);
  
  long start;
  if(!get(vnode, "start", start))
    throw InternalError("Cannot get start");
  long end;
  if(!get(vnode, "end", end))
    throw InternalError("Cannot get end");
  string filename;  
  if(!get(vnode, "filename", filename))
    throw InternalError("Cannot get filename");
  string compressionMode;  
  if(!get(vnode, "compression", compressionMode))
    compressionMode = "";
  
  XMLURL dataurl(url, filename.c_str());
  if(dataurl.getProtocol() != XMLURL::File)
    throw InternalError(string("Cannot read over: ")
			+toString(dataurl.getPath()));
  string datafile(toString(dataurl.getPath()));
  
  int fd = open(datafile.c_str(), O_RDONLY);
  if(fd == -1)
    throw ErrnoException("DataArchive::query (open call)", errno);
#ifdef __sgi
  off64_t ls = lseek64(fd, start, SEEK_SET);
#else
  off_t ls = lseek(fd, start, SEEK_SET);
#endif
  if(ls == -1)
    throw ErrnoException("DataArchive::query (lseek64 call)", errno);
  
  InputContext ic(fd, start);
  var.read(ic, end, compressionMode);
  ASSERTEQ(end, ic.cur);
  int s = close(fd);
  if(s == -1)
    throw ErrnoException("DataArchive::query (read call)", errno);
  d_lock.unlock();  
}

void 
DataArchive::findPatchAndIndex(GridP grid, Patch*& patch, particleIndex& idx,
			       long64 particleID, int matlIndex,
			       double time)
{
  Patch *local = patch;
  if( patch != NULL ){
    ParticleVariable<long64> var;
    query(var, "p.particleID", matlIndex, patch, time);
    //  cerr<<"var["<<idx<<"] = "<<var[idx]<<endl;
    if( var[idx] == particleID )
      return;
    else {
      ParticleSubset* subset = var.getParticleSubset();
      for(ParticleSubset::iterator p_iter = subset->begin();
	  p_iter != subset->end(); p_iter++){
	if( var[*p_iter] == particleID){
	  idx = *p_iter;
	  return;
	}
      }
    }
  }
  patch = NULL;
  for (int level_nr = 0;
       (level_nr < grid->numLevels()) && (patch == NULL); level_nr++) {
    
    const LevelP level = grid->getLevel(level_nr);
    
    for (Level::const_patchIterator iter = level->patchesBegin();
	 (iter != level->patchesEnd()) && (patch == NULL); iter++) {
      if( *iter == local ) continue;
      ParticleVariable<long64> var;
      query(var, "p.particleID", matlIndex, *iter, time);
      ParticleSubset* subset = var.getParticleSubset();
      for(ParticleSubset::iterator p_iter = subset->begin();
	  p_iter != subset->end(); p_iter++){
	if( var[*p_iter] == particleID){
	  patch = *iter;
	  idx = *p_iter;
	  //	  cerr<<"var["<<*p_iter<<"] = "<<var[*p_iter]<<endl;
	  break;
	}
      }
      
      if( patch != NULL )
	break;
    }
  }
}

void
DataArchive::restartInitialize(int& timestep, const GridP& grid,
			       DataWarehouse* dw, double* pTime, double* pDelt)
{
  vector<int> indices;
  vector<double> times;
  queryTimesteps(indices, times);
   
  unsigned int i = 0;

  if (timestep == -1 && indices.size() > 0) {
    i = (unsigned int)(indices.size() - 1); 
  }
  else {
    for (i = 0; i < indices.size(); i++)
      if (indices[i] == timestep)
	break;
  }

  if (i == indices.size()) {
    // timestep not found
    ostringstream message;
    message << "Timestep " << timestep << " not found";
    throw InternalError(message.str());
  }

  *pTime = times[i];
  timestep = indices[i];
   
  ASSERTL3(indices.size() == d_tstop.size());
  ASSERTL3(d_tsurl.size() == d_tstop.size());

  PatchHashMaps patchMap;
  patchMap.init(d_tsurl[i], d_tstop[i], d_processor, d_numProcessors);

  DOM_Node timeBlock = findNode("Time", d_tstop[i]);
  if (!get(timeBlock, "delt", *pDelt))
    *pDelt = 0;
  
  // iterator through all patch, initializing on each patch
  // (perhaps not the most efficient, but this is only initialization)
  for (i = 0; i < (unsigned int)grid->numLevels(); i++) {
    LevelP level = grid->getLevel(i);
    list<Patch*> patches;
    patches.push_back(NULL); // add NULL patch for dealing with globals
      
    Level::patchIterator levelPatchIter = level->patchesBegin();
    for ( ; levelPatchIter != level->patchesEnd(); levelPatchIter++)
      patches.push_back(*levelPatchIter);

    for (list<Patch*>::iterator patchIter = patches.begin();
	 patchIter != patches.end(); patchIter++)
      {
	Patch* patch = *patchIter;
	const MaterialHashMaps* matlMap = patchMap.findPatchData(patch);
	if (matlMap != NULL) {
	  const vector<VarHashMap>& matVec = matlMap->getVarHashMaps();
	  for (int matl = -1; matl < (int)matVec.size()-1; matl++) {
	    const VarHashMap& hashMap = matVec[matl+1];
	    VarHashMapIterator varIter(const_cast<VarHashMap*>(&hashMap));
	    for (varIter.first(); varIter.ok(); ++varIter) {
	      VarLabel* label = VarLabel::find(varIter.get_key());
	      if (label == NULL)
		throw UnknownVariable(label->getName(), patch, matl,
				      "on DataArchive::scheduleRestartInitialize");

	      initVariable(patch, dw, label,
			   matl, varIter.get_data());
	    }
	  }
	}
      }
  }
}

bool DataArchive::queryRestartTimestep(int& timestep)
{
  DOM_Node restartNode = findNode("restart", d_indexDoc.getDocumentElement());
  if (restartNode == 0) {
    DOM_Node restartsNode = findNode("restarts", d_indexDoc.getDocumentElement());
    if (restartsNode == 0)
      return false;
    
    restartNode = findNode("restart", restartsNode);
    if (restartNode == 0)
      return false;

    // get the last restart tag in the restarts list
    while (findNextNode("restart", restartNode) != 0)
      restartNode = findNextNode("restart", restartNode);
  }
  
  DOM_NamedNodeMap attributes = restartNode.getAttributes();
  DOM_Node timestepNode = attributes.getNamedItem("timestep");
  if (timestepNode == 0)
    return false;
  char*s = timestepNode.getNodeValue().transcode();
  timestep = atoi(s);
  delete[] s;
  return true;
}

void
DataArchive::initVariable(const Patch* patch,
			  DataWarehouse* dw,
			  VarLabel* label, int matl,
			  pair<DOM_Node, XMLURL> dataRef)
{
  Variable* var = label->typeDescription()->createInstance();
  DOM_Node vnode = dataRef.first;
  XMLURL url = dataRef.second;
  query(*var, vnode, url, matl, patch);

  ParticleVariableBase* particles;
  if ((particles = dynamic_cast<ParticleVariableBase*>(var))) {
    if (!dw->haveParticleSubset(matl, patch)) {
      int numParticles = particles->getParticleSubset()->numParticles(); 
      dw->createParticleSubset(numParticles, matl, patch);
    }
  }

  dw->put(var, label, matl, patch);
}

DataArchive::TimeHashMaps::TimeHashMaps(const vector<double>& tsTimes,
					const vector<XMLURL>& tsUrls,
					const vector<DOM_Node>& tsTopNodes,
					int processor, int numProcessors)
{
  ASSERTL3(tsTimes.size() == tsTopNodes.size());
  ASSERTL3(tsUrls.size() == tsTopNodes.size());

  for (int i = 0; i < (int)tsTimes.size(); i++)
    d_patchHashMaps[tsTimes[i]].init(tsUrls[i], tsTopNodes[i],
				     processor, numProcessors);
   
  d_lastFoundIt = d_patchHashMaps.end();
}

inline DOM_Node
DataArchive::TimeHashMaps::findVariable(const string& name,
					const Patch* patch, int matl,
					double time, XMLURL& foundUrl)
{
  PatchHashMaps* timeData = findTimeData(time);
  return (timeData == NULL) ? DOM_Node() :
    timeData->findVariable(name, patch, matl, foundUrl);
}

inline DataArchive::MaterialHashMaps*
DataArchive::TimeHashMaps::findPatchData(double time, const Patch* patch)
{
  PatchHashMaps* timeData = findTimeData(time);
  return (timeData == NULL) ? NULL : timeData->findPatchData(patch);
}

DataArchive::PatchHashMaps*
DataArchive::TimeHashMaps::findTimeData(double time)
{
  // assuming nearby queries will often be made sequentially,
  // checking the lastFound can reduce overall query times.
  if ((d_lastFoundIt != d_patchHashMaps.end()) &&
      ((*d_lastFoundIt).first == time))
    return &(*d_lastFoundIt).second;
  map<double, PatchHashMaps>::iterator foundIt =
    d_patchHashMaps.find(time);
  if (foundIt != d_patchHashMaps.end()) {
    d_lastFoundIt = foundIt;
    return &(*foundIt).second;
  }
  return NULL;
}




DataArchive::PatchHashMaps::PatchHashMaps()
  : d_matHashMaps(),
    d_lastFoundIt(d_matHashMaps.end()),
    d_isParsed(false)
{
}

void DataArchive::PatchHashMaps::init(XMLURL tsUrl, DOM_Node tsTopNode,
				      int processor, int numProcessors)
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
      DOM_Node procNode = attributes.getNamedItem("proc");
      if (procNode != NULL) {
	char* s = procNode.getNodeValue().transcode();
	int proc = atoi(s);
	delete[] s;
	if ((proc % numProcessors) != processor)
	  continue;
      }
	 
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
    
    //cerr << "reading: " << toString(urlIt->getURLText()) << '\n';
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
	
	add(varname, patchid, index, r, *urlIt);
      } else if(r.getNodeType() != DOM_Node::TEXT_NODE){
	cerr << "WARNING: Unknown element in Variables section: " << toString(r.getNodeName()) << '\n';
      }
    }
  }
  
  d_isParsed = true;
  d_lastFoundIt = d_matHashMaps.end();
}

inline DOM_Node
DataArchive::PatchHashMaps::findVariable(const string& name,
					 const Patch* patch,
					 int matl,
					 XMLURL& foundUrl)
{
  MaterialHashMaps* patchData = findPatchData(patch);
  return (patchData == NULL) ? DOM_Node() :
    patchData->findVariable(name, matl, foundUrl);
}

DataArchive::MaterialHashMaps*
DataArchive::PatchHashMaps::findPatchData(const Patch* patch)
{
  if (!d_isParsed) parse(); // parse on demand
  int patchid = (patch ? patch->getID() : -1);
  
  // assuming nearby queries will often be made sequentially,
  // checking the lastFound can reduce overall query times.
  if ((d_lastFoundIt != d_matHashMaps.end()) &&
      ((*d_lastFoundIt).first == patchid))
    return &(*d_lastFoundIt).second;

  map<int, MaterialHashMaps>::iterator foundIt =
    d_matHashMaps.find(patchid);
  
  if (foundIt != d_matHashMaps.end()) {
    d_lastFoundIt = foundIt;
    return &(*foundIt).second;
  }
  return NULL;  
}

DOM_Node DataArchive::MaterialHashMaps::findVariable(const string& name,
						     int matl,
						     XMLURL& foundUrl)
{
  matl++; // allows for matl=-1 for universal variables
 
  if (matl < (int)d_varHashMaps.size()) {
    VarHashMap& hashMap = d_varHashMaps[matl];
    pair<DOM_Node, XMLURL> found;
    if (hashMap.lookup(name, found)) {
      foundUrl = found.second;
      return found.first;
    }
  }
  return DOM_Node();  
}

void DataArchive::MaterialHashMaps::add(const string& name, int matl,
					DOM_Node varNode, XMLURL url)
{
  matl++; // allows for matl=-1 for universal variables
   
  if (matl >= (int)d_varHashMaps.size())
    d_varHashMaps.resize(matl + 1);
  pair<DOM_Node, XMLURL> value(varNode, url);
  pair<DOM_Node, XMLURL> dummy;
  if (d_varHashMaps[matl].lookup(name, dummy) == 1)
    cerr << "Duplicate variable name: " << name << endl;
  else
    d_varHashMaps[matl].insert(name, value);
}

ConsecutiveRangeSet DataArchive::queryMaterials( const string& name,
						 const Patch* patch,
						 double time)
{
  double start = Time::currentSeconds();
  d_lock.lock();

  MaterialHashMaps* matlVarHashMaps =
    getTopLevelVarHashMaps()->findPatchData(time, patch);

  if (matlVarHashMaps == NULL) {
    ostringstream msg;
    msg << "Cannot find data for time = " << time << ", patch = " <<
      (patch ? patch->getID() : -1);
    throw InternalError(msg.str());
  }
   
  ConsecutiveRangeSet result;
  int numMatls = (int)matlVarHashMaps->getVarHashMaps().size() - 1;
  for (int matl = -1; matl < numMatls; matl++) {
    XMLURL url;
    if (matlVarHashMaps->findVariable(name, matl, url) != 0)
      result.addInOrder(matl);
  }
  
  d_lock.unlock();
  dbg << "DataArchive::queryMaterials completed in " << Time::currentSeconds()-start << " seconds\n";

  return result;
}

int DataArchive::queryNumMaterials(const Patch* patch, double time)
{
  double start = Time::currentSeconds();

  MaterialHashMaps* matlVarHashMaps =
    getTopLevelVarHashMaps()->findPatchData(time, patch);

  if (matlVarHashMaps == NULL) {
    ostringstream msg;
    msg << "Cannot find data for time = " << time << ", patch = " <<
      (patch ? patch->getID() : -1);
    throw InternalError(msg.str());
  }

  dbg << "DataArchive::queryNumMaterials completed in " << Time::currentSeconds()-start << " seconds\n";

  return (int)matlVarHashMaps->getVarHashMaps().size() - 1 /* the other 1 is
							      for globals */;
}
