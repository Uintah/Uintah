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

#include <xercesc/sax/SAXException.hpp>
#include <xercesc/sax/SAXParseException.hpp>
#include <xercesc/sax/ErrorHandler.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/util/XMLString.hpp>

#include <sys/param.h>

#include <iostream>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>

using namespace std;

using namespace Uintah;
using namespace SCIRun;

DebugStream DataArchive::dbg("DataArchive", false);

bool DataArchive::cacheOnlyCurrentTimestep = false;

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
  
  XercesDOMParser* parser = new XercesDOMParser;
  parser->setDoValidation(false);
  
  SimpleErrorHandler handler;
  parser->setErrorHandler(&handler);
  
  cout << "Parsing " << XMLString::transcode(d_base.getURLText()) << endl;
  parser->parse(d_base.getURLText());
  
  if(handler.foundError)
    throw InternalError("Error reading file: "+ string(XMLString::transcode(d_base.getURLText())));
  
  d_indexDoc = dynamic_cast<DOMDocument*>(parser->getDocument()->cloneNode(true));

  delete parser;
  d_swapBytes = queryEndianness() != SCIRun::endianness();
  d_nBytes = queryNBits() / 8;
}


DataArchive::~DataArchive()
{
  delete d_varHashMaps;
  d_indexDoc->release();


  // need to delete the nodes
  int size = d_tstop.size();
  for (int i = 0; i < size; i++) {
    d_tstop[i]->getOwnerDocument()->release();
  }
}

string DataArchive::queryEndianness()
{
  string ret;
  d_lock.lock();
  const DOMNode* meta = findNode("Meta", d_indexDoc->getDocumentElement());
  if( meta == 0 )
    throw InternalError("Meta node not found in index.xml");
  const DOMNode* endian_node = findNode("endianness", meta);
  if( endian_node == 0 ){
    cout<<"\nXML Warning: endianness node not found.\n"<<
      "Assuming data was created on a " << SCIRun::endianness() << " machine.\n"<<
      "To eliminate this message and express the correct\n"<<
      "endianess, please add either\n"<<
      "\t<endianness>little_endian</endianness>\n"<<
      "or\n\t<endianness>big_endian</endianness>\n"<<
      "to the <Meta> section of the index.xml file.\n\n";
    ret = string(SCIRun::endianness());
    d_lock.unlock();
    return ret;
  }
  DOMNode* child = endian_node->getFirstChild();
  //DOMString endian = child->getNodeValue();
  ret = string(XMLString::transcode(child->getNodeValue()));
  d_lock.unlock();
  return ret;
}

int DataArchive::queryNBits()
{
  int ret;
  d_lock.lock();
  const DOMNode* meta = findNode("Meta", d_indexDoc->getDocumentElement());
  if( meta == 0 )
    throw InternalError("Meta node not found in index.xml");
  const DOMNode* nBits_node = findNode("nBits", meta);
  if( nBits_node == 0 ){
    cout<<"\nXML Warning: nBits node not found.\n"<<
      "Assuming data was created using " << sizeof(unsigned long) * 8 << " bits.\n"
      "To eliminate this message and express the correct\n"<<
      "number of bits, please add either\n"<<
      "\t<nBits>32</nBits>\n"<<
      "or\n\t<nBits>64</nBits>\n"<<
      "to the <Meta> section of the index.xml file.\n\n";
    d_lock.unlock();
    return sizeof(unsigned long) * 8;
  }
  DOMNode* child = nBits_node->getFirstChild();
  //DOMString nBits = child->getNodeValue();
  ret = atoi(XMLString::transcode(child->getNodeValue()));
  d_lock.unlock();
  return ret;
}

void
DataArchive::queryTimesteps( std::vector<int>& index,
			     std::vector<double>& times )
{
  double start = Time::currentSeconds();
  if(!have_timesteps){
    d_lock.lock();
    if(!have_timesteps){
      const DOMNode* ts = findNode("timesteps", d_indexDoc->getDocumentElement());
      if(ts == 0)
	throw InternalError("timesteps node not found in index.xml");
      for(DOMNode* t = ts->getFirstChild(); t != 0; t = t->getNextSibling()){
	if(t->getNodeType() == DOMNode::ELEMENT_NODE){
	  DOMNamedNodeMap* attributes = t->getAttributes();
	  DOMNode* tsfile = attributes->getNamedItem(XMLString::transcode("href"));
	  if(tsfile == 0)
	    throw InternalError("timestep href not found");
	  
	  //DOMString href_name = tsfile->getNodeValue();
	  XMLURL url(d_base, XMLString::transcode(tsfile->getNodeValue()));
	  XercesDOMParser* parser = new XercesDOMParser;
	  parser->setDoValidation(false);
	  
	  SimpleErrorHandler handler;
	  parser->setErrorHandler(&handler);
	  
	  //cerr << "reading: " << XMLString::transcode(url.getURLText()) << '\n';
	  parser->parse(url.getURLText());
	  if(handler.foundError)
	    throw InternalError("Cannot read timestep file");
	  
	  DOMNode* top = dynamic_cast<DOMDocument*>(parser->getDocument()->cloneNode(true))->getDocumentElement();
	  
	  delete parser;
	  d_tstop.push_back(top);
	  d_tsurl.push_back(url);
	  const DOMNode* time = findNode("Time", top);
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

DOMNode*
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
    return 0; //I'm not sure about this
  found_url = d_tsurl[i];
  return d_tstop[i];
}

DOMNode*
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
  DOMNode* top = getTimestep(time, url);
  if (top == 0)
    throw InternalError("Cannot find Grid in timestep");
  const DOMNode* gridnode = findNode("Grid", top);
  if(gridnode == 0)
    throw InternalError("Cannot find Grid in timestep");
  int numLevels = -1234;
  GridP grid = scinew Grid;
  for(DOMNode* n = gridnode->getFirstChild(); n != 0; n=n->getNextSibling()){
    if(strcmp(XMLString::transcode(n->getNodeName()), "numLevels") == 0) {
      if(!get(n, numLevels))
	throw InternalError("Error parsing numLevels");
    } else if(strcmp(XMLString::transcode(n->getNodeName()),"Level") == 0){
      Point anchor;
      if(!get(n, "anchor", anchor))
	throw InternalError("Error parsing level anchor point");
      Vector dcell;
      if(!get(n, "cellspacing", dcell))
	throw InternalError("Error parsing level cellspacing");
      int id;
      if(!get(n, "id", id)){
	static bool warned_once=false;
	if(!warned_once){
	  cerr << "WARNING: Data archive does not have level ID\n";
	  cerr << "This is okay, as long as you aren't trying to do AMR\n";
	}
	warned_once=true;
	id=-1;
      }
      LevelP level = grid->addLevel(anchor, dcell, id);
      int numPatches = -1234;
      long totalCells = 0;
      IntVector periodicBoundaries(0, 0, 0);      
      for(DOMNode* r = n->getFirstChild(); r != 0; r=r->getNextSibling()){
	if(strcmp(XMLString::transcode(r->getNodeName()),"numPatches") == 0 ||
	   strcmp(XMLString::transcode(r->getNodeName()),"numRegions") == 0){
	  if(!get(r, numPatches))
	    throw InternalError("Error parsing numRegions");
	} else if(strcmp(XMLString::transcode(r->getNodeName()),"totalCells") == 0){
	  if(!get(r, totalCells))
	    throw InternalError("Error parsing totalCells");
	} else if(strcmp(XMLString::transcode(r->getNodeName()),"Patch") == 0 ||
		  strcmp(XMLString::transcode(r->getNodeName()),"Region") == 0){
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
	  Patch* patch = level->addPatch(lowIndex, highIndex,lowIndex,
				     highIndex,id);
	  ASSERTEQ(patch->totalCells(), totalCells);
	} else if(strcmp(XMLString::transcode(r->getNodeName()),"anchor") == 0
		  || strcmp(XMLString::transcode(r->getNodeName()),"cellspacing") == 0
		  || strcmp(XMLString::transcode(r->getNodeName()),"id") == 0){
	  // Nothing - handled above
	} else if(strcmp(XMLString::transcode(r->getNodeName()),"periodic") == 0) {
	  if(!get(n, "periodic", periodicBoundaries))
	    throw InternalError("Error parsing periodoc");
	} else if(r->getNodeType() != DOMNode::TEXT_NODE){
	  cerr << "WARNING: Unknown level data: " << ::XMLString::transcode(r->getNodeName()) << '\n';
	}
      }
      ASSERTEQ(level->numPatches(), numPatches);
      ASSERTEQ(level->totalCells(), totalCells);
      
      if(periodicBoundaries != IntVector(0, 0, 0)){
	level->finalizeLevel(periodicBoundaries.x() != 0,
			     periodicBoundaries.y() != 0,
			     periodicBoundaries.z() != 0);
      }
      else {
	level->finalizeLevel();
      }
    } else if(n->getNodeType() != DOMNode::TEXT_NODE){
      cerr << "WARNING: Unknown grid data: " << XMLString::transcode(n->getNodeName()) << '\n';
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
  const DOMNode* vars = findNode("variables", d_indexDoc->getDocumentElement());
  if(vars == 0)
    throw InternalError("variables section not found\n");
  queryVariables(vars, names, types);

  d_lock.unlock();
  dbg << "DataArchive::queryVariables completed in " << Time::currentSeconds()-start << " seconds\n";
}

void
DataArchive::queryGlobals( vector<string>& names,
			   vector<const TypeDescription*>& types)
{
  double start = Time::currentSeconds();
  d_lock.lock();
  const DOMNode* vars = findNode("globals", d_indexDoc->getDocumentElement());
  if(vars == 0)
    return;
  queryVariables(vars, names, types);

  d_lock.unlock();

  dbg << "DataArchive::queryGlobals completed in " << Time::currentSeconds()-start << " seconds\n";   
}

void
DataArchive::queryVariables(const DOMNode* vars, vector<string>& names,
			    vector<const TypeDescription*>& types)
{
  for(DOMNode* n = vars->getFirstChild(); n != 0; n = n->getNextSibling()){
    if(strcmp(XMLString::transcode(n->getNodeName()),"variable") == 0){
      DOMNamedNodeMap* attributes = n->getAttributes();
      DOMNode* type = attributes->getNamedItem(XMLString::transcode("type"));
      if(type == 0)
	throw InternalError("Variable type not found");
      string type_name = XMLString::transcode(type->getNodeValue());
      const TypeDescription* td = TypeDescription::lookupType(type_name);
      if(!td){
	static TypeDescription* unknown_type = 0;
	if(!unknown_type)
	  unknown_type = scinew TypeDescription(TypeDescription::Unknown,
						"-- unknown type --",
						false, MPI_Datatype(-1));
	td = unknown_type;
      }
      types.push_back(td);
      DOMNode* name = attributes->getNamedItem(XMLString::transcode("name"));
      if(name == 0)
	throw InternalError("Variable name not found");
      names.push_back(XMLString::transcode(name->getNodeValue()));
    } else if(n->getNodeType() != DOMNode::TEXT_NODE){
      cerr << "WARNING: Unknown variable data: " << XMLString::transcode(n->getNodeName()) << '\n';
    }
  }
}

void
DataArchive::query( Variable& var, const std::string& name,
		    int matlIndex, const Patch* patch, double time )
{
  double tstart = Time::currentSeconds();
  XMLURL url;
  d_lock.lock();  
  DOMNode* vnode = findVariable(name, patch, matlIndex, time, url);
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
DataArchive::query( Variable& var, DOMNode* vnode, XMLURL url,
		    int matlIndex, const Patch* patch )
{
  d_lock.lock();
  DOMNamedNodeMap* attributes = vnode->getAttributes();
  DOMNode* typenode = attributes->getNamedItem(XMLString::transcode("type"));
  if(typenode == 0)
    throw InternalError("Variable doesn't have a type");
  string type = XMLString::transcode(typenode->getNodeValue());
  const TypeDescription* td = var.virtualGetTypeDescription();
  ASSERT(td->getName() == type);
  
  if (td->getType() == TypeDescription::ParticleVariable) {
    int numParticles;
    if(!get(vnode, "numParticles", numParticles))
      throw InternalError("Cannot get numParticles");
    psetDBType::key_type key(matlIndex, patch);
    ParticleSubset* psubset = 0;
    psetDBType::iterator psetIter = d_psetDB.find(key);
    if(psetIter != d_psetDB.end()) {
      psubset = (*psetIter).second.get_rep();
    }
    if (psubset == 0 || psubset->numParticles() != numParticles)
    {
     d_psetDB[key] = psubset =
       scinew ParticleSubset(scinew ParticleSet(numParticles), true,
			     matlIndex, patch);
    }
    (static_cast<ParticleVariableBase*>(&var))->allocate(psubset);
//      (dynamic_cast<ParticleVariableBase*>(&var))->allocate(psubset);
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
			+XMLString::transcode(dataurl.getPath()));
  string datafile(XMLString::transcode(dataurl.getPath()));
  
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
  var.read(ic, end, d_swapBytes, d_nBytes, compressionMode);
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
  unsigned int i = 0;  
  vector<int> indices;
  vector<double> times;
  queryTimesteps(indices, times);

  vector<string> names;
  vector< const TypeDescription *> typeDescriptions;
  queryVariables(names, typeDescriptions);
  queryGlobals(names, typeDescriptions);  
  
  map<string, VarLabel*> varMap;
  for (i = 0; i < names.size(); i++) {
    varMap[names[i]] = VarLabel::find(names[i]);
  }
    

  if (timestep == 0) {
    i = 0; // timestep == 0 means use the first timestep
  }
  else if (timestep == -1 && indices.size() > 0) {
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

  const DOMNode* timeBlock = findNode("Time", d_tstop[i]);
  if (!get(timeBlock, "delt", *pDelt))
    *pDelt = 0;
  
  // iterate through all patch, initializing on each patch
  // (perhaps not the most efficient, but this is only initialization)
  for (i = 0; i < (unsigned int)grid->numLevels(); i++) {
    LevelP level = grid->getLevel(i);
    list<Patch*> patches;
    patches.push_back(NULL); // add NULL patch for dealing with globals
      
    Level::patchIterator levelPatchIter = level->patchesBegin();
    for ( ; levelPatchIter != level->patchesEnd(); levelPatchIter++)
      patches.push_back(*levelPatchIter);

    map<string, VarLabel*>::iterator labelIter;    
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
	      // skip if the variable isn't in the top level variable list
	      // (this is useful for manually editting the index.xml to
	      // remove variables when combining patches)
	      labelIter = varMap.find(varIter.get_key());
	      if (labelIter != varMap.end()) {
		VarLabel* label = labelIter->second;
		if (label == 0) {
		  throw UnknownVariable(varIter.get_key(), dw->getID(), patch, matl,
					"on DataArchive::scheduleRestartInitialize");
		}
		else {
		  initVariable(patch, dw, labelIter->second,
			       matl, varIter.get_data());
		}
	      }
	    }
	  }
	}
      }
  }
}

bool DataArchive::queryRestartTimestep(int& timestep)
{
  DOMNode* restartNode = 
    const_cast<DOMNode*>(findNode("restart", d_indexDoc->getDocumentElement()));
  if (restartNode == 0) {
    DOMNode* restartsNode = 
      const_cast<DOMNode*>(findNode("restarts", d_indexDoc->getDocumentElement()));
    if (restartsNode == 0)
      return false;
    
    restartNode = const_cast<DOMNode*>(findNode("restart", restartsNode));
    if (restartNode == 0)
      return false;

    // get the last restart tag in the restarts list
    while (findNextNode("restart", restartNode) != 0)
      restartNode = const_cast<DOMNode*>(findNextNode("restart", restartNode));
  }
  
  DOMNamedNodeMap* attributes = restartNode->getAttributes();
  DOMNode* timestepNode = attributes->getNamedItem(XMLString::transcode("timestep"));
  if (timestepNode == 0)
    return false;
  char*s = XMLString::transcode(timestepNode->getNodeValue());
  timestep = atoi(s);
  delete[] s;
  return true;
}

void
DataArchive::initVariable(const Patch* patch,
			  DataWarehouse* dw,
			  VarLabel* label, int matl,
			  pair<DOMNode*, XMLURL> dataRef)
{
  Variable* var = label->typeDescription()->createInstance();
  DOMNode* vnode = dataRef.first;
  XMLURL url = dataRef.second;
  query(*var, vnode, url, matl, patch);

  ParticleVariableBase* particles;
  if ((particles = dynamic_cast<ParticleVariableBase*>(var))) {
    if (!dw->haveParticleSubset(matl, patch)) {
      cerr << "Saved ParticleSubset on matl " << matl << " patch " << patch << endl;
      dw->saveParticleSubset(matl, patch, particles->getParticleSubset());
    }
    else {
      ASSERTEQ(dw->getParticleSubset(matl, patch),
	       particles->getParticleSubset());
    }
  }

  dw->put(var, label, matl, patch); 
  delete var; // should have been cloned when it was put
}

DataArchive::TimeHashMaps::TimeHashMaps(const vector<double>& tsTimes,
					const vector<XMLURL>& tsUrls,
					const vector<DOMNode*>& tsTopNodes,
					int processor, int numProcessors)
{
  ASSERTL3(tsTimes.size() == tsTopNodes.size());
  ASSERTL3(tsUrls.size() == tsTopNodes.size());

  for (int i = 0; i < (int)tsTimes.size(); i++)
    d_patchHashMaps[tsTimes[i]].init(tsUrls[i], tsTopNodes[i],
				     processor, numProcessors);
   
  d_lastFoundIt = d_patchHashMaps.end();
}

inline DOMNode*
DataArchive::TimeHashMaps::findVariable(const string& name,
					const Patch* patch, int matl,
					double time, XMLURL& foundUrl)
{
  PatchHashMaps* timeData = findTimeData(time);
  return (timeData == NULL) ? NULL :
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
    if (DataArchive::cacheOnlyCurrentTimestep) {
      // Only caching the current timestep.
      // purge the last accessed timestep, since this timestep is differen.t
      if (d_lastFoundIt != d_patchHashMaps.end())
	(*d_lastFoundIt).second.purgeCache();
    }
    d_lastFoundIt = foundIt;
    return &(*foundIt).second;
  }
  return NULL;
}

void DataArchive::TimeHashMaps::purgeTimeData(double time)
{
  // purge a timestep's cachine
  PatchHashMaps* timeData = findTimeData(time);
  if (timeData != 0) {
    timeData->purgeCache();
  }
}

DataArchive::PatchHashMaps::PatchHashMaps()
  : d_matHashMaps(),
    d_lastFoundIt(d_matHashMaps.end()),
    d_isParsed(false)
{
}

void DataArchive::PatchHashMaps::init(XMLURL tsUrl, DOMNode* tsTopNode,
				      int processor, int numProcessors)
{
  d_isParsed = false;
  // grab the data xml files from the timestep xml file
  ASSERTL3(tsTopNode != 0);
  const DOMNode* datanode = findNode("Data", tsTopNode);
  if(datanode == 0)
    throw InternalError("Cannot find Data in timestep");
  for(DOMNode* n = datanode->getFirstChild(); n != 0; n=n->getNextSibling()){
    if(strcmp(XMLString::transcode(n->getNodeName()),"Datafile") == 0){
      DOMNamedNodeMap* attributes = n->getAttributes();
      DOMNode* procNode = attributes->getNamedItem(XMLString::transcode("proc"));
      if (procNode != NULL) {
	char* s = XMLString::transcode(procNode->getNodeValue());
	int proc = atoi(s);
	delete[] s;
	if ((proc % numProcessors) != processor)
	  continue;
      }
	 
      DOMNode* datafile = attributes->getNamedItem(XMLString::transcode("href"));
      if(datafile == 0)
	throw InternalError("timestep href not found");
      //DOMString href_name = datafile.getNodeValue();
      XMLURL url(tsUrl, XMLString::transcode(datafile->getNodeValue()));
      d_xmlUrls.push_back(url);
    }
    else if(n->getNodeType() != DOMNode::TEXT_NODE){
      cerr << "WARNING: Unknown element in Data section: " << XMLString::transcode(n->getNodeName()) << '\n';
    }
  }
}

void DataArchive::PatchHashMaps::purgeCache()
{
  d_matHashMaps.clear();
  d_isParsed = false;

  for (int i = 0; i < docs.size(); i++)
    docs[i]->release();
  docs.clear();
}

void DataArchive::PatchHashMaps::parse()
{
  for (list<XMLURL>::iterator urlIt = d_xmlUrls.begin();
       urlIt != d_xmlUrls.end(); urlIt++) {
    XercesDOMParser *parser = new XercesDOMParser;
    parser->setDoValidation(false);
    
    SimpleErrorHandler handler;
    parser->setErrorHandler(&handler);
    
    //cerr << "reading: " << XMLString::transcode(urlIt->getURLText()) << '\n';
    parser->parse((*urlIt).getURLText());
    if(handler.foundError)
      throw InternalError("Cannot read timestep file");
    
    DOMDocument* doc = dynamic_cast<DOMDocument*>(parser->getDocument()->cloneNode(true));
    DOMNode* top = doc->getDocumentElement();

    docs.push_back(doc);
    
    delete parser;
    for(DOMNode* r = top->getFirstChild(); r != 0; r=r->getNextSibling()){
      if(strcmp(XMLString::transcode(r->getNodeName()),"Variable") == 0){
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
      } else if(r->getNodeType() != DOMNode::TEXT_NODE){
	cerr << "WARNING: Unknown element in Variables section: " << XMLString::transcode(r->getNodeName()) << '\n';
      }
    }

    //top->getOwnerDocument()->release();
  }
  
  d_isParsed = true;
  d_lastFoundIt = d_matHashMaps.end();
}

inline DOMNode*
DataArchive::PatchHashMaps::findVariable(const string& name,
					 const Patch* patch,
					 int matl,
					 XMLURL& foundUrl)
{
  MaterialHashMaps* patchData = findPatchData(patch);
  return (patchData == NULL) ? NULL :
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

DOMNode* DataArchive::MaterialHashMaps::findVariable(const string& name,
						     int matl,
						     XMLURL& foundUrl)
{
  matl++; // allows for matl=-1 for universal variables
 
  if (matl < (int)d_varHashMaps.size()) {
    VarHashMap& hashMap = d_varHashMaps[matl];
    pair<DOMNode*, XMLURL> found;
    if (hashMap.lookup(name, found)) {
      foundUrl = found.second;
      return found.first;
    }
  }
  return NULL;  
}

void DataArchive::MaterialHashMaps::add(const string& name, int matl,
					DOMNode* varNode, XMLURL url)
{
  matl++; // allows for matl=-1 for universal variables
   
  if (matl >= (int)d_varHashMaps.size())
    d_varHashMaps.resize(matl + 1);
  pair<DOMNode*, XMLURL> value(varNode, url);
  pair<DOMNode*, XMLURL> dummy;
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
