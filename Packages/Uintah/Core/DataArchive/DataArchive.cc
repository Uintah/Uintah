#include <Packages/Uintah/Core/DataArchive/DataArchive.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/CCA/Components/ProblemSpecification/ProblemSpecReader.h>
#include <Packages/Uintah/CCA/Ports/InputContext.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/UnknownVariable.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Core/Util/NotFinished.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Util/Assert.h>
#include <Core/Thread/Time.h>
#include <Core/Util/DebugStream.h>

#include <xercesc/sax/SAXException.hpp>
#include <xercesc/sax/SAXParseException.hpp>
#include <xercesc/sax/ErrorHandler.hpp>
#include <xercesc/util/PlatformUtils.hpp>

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

  try {
    XMLPlatformUtils::Initialize();
  } catch(const XMLException& toCatch) {
    char* ch = XMLString::transcode(toCatch.getMessage());
    string ex("XML Exception: " + string(ch));
    delete [] ch;
    throw ProblemSetupException(ex);
  }


  have_timesteps=false;
  string index(filebase+"/index.xml");
  const XMLCh* tmpRel = XMLString::transcode(index.c_str());
  d_base.setURL(tmpRel);
  delete [] tmpRel;
  
  if(d_base.isRelative()){
    char path[MAXPATHLEN];
    string url = string("file://")+getwd(path)+"/.";
    d_base.makeRelativeTo(url.c_str());
    if(d_base.isRelative())
      cerr << "base is still relative!\n";
  }
  
  char* urltext = XMLString::transcode(d_base.getURLText());
  ProblemSpecReader psr(urltext);
  cout << "Parsing " << urltext << endl;
  delete [] urltext;

  d_indexDoc = psr.readInputFile();
  
  d_swapBytes = queryEndianness() != SCIRun::endianness();
  d_nBytes = queryNBits() / 8;
}


DataArchive::~DataArchive()
{
  delete d_varHashMaps;
  d_indexDoc->releaseDocument();


  // need to delete the nodes
  int size = static_cast<int>(d_tstop.size());
  for (int i = 0; i < size; i++) {
    d_tstop[i]->releaseDocument();
  }
}

string DataArchive::queryEndianness()
{
  string ret;
  d_lock.lock();
  ProblemSpecP meta = d_indexDoc->findBlock("Meta");
  if( meta == 0)
    throw InternalError("Meta node not found in index.xml");
  ProblemSpecP endian_node = meta->findBlock("endianness");
  if( endian_node== 0 ){
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

  ret = endian_node->getFirstChild()->getNodeValue();
  d_lock.unlock();
  return ret;
}

int DataArchive::queryNBits()
{
  int ret;
  d_lock.lock();
  ProblemSpecP meta = d_indexDoc->findBlock("Meta");
  if( meta == 0 )
    throw InternalError("Meta node not found in index.xml");
  ProblemSpecP nBits_node = meta->findBlock("nBits");
  if( nBits_node == 0){
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

  ret = atoi(nBits_node->getFirstChild()->getNodeValue().c_str());
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
      ProblemSpecP ts = d_indexDoc->findBlock("timesteps");
      if(ts == 0)
	throw InternalError("timesteps node not found in index.xml");
      for(ProblemSpecP t = ts->getFirstChild(); t != 0; t = t->getNextSibling()){
	if(t->getNodeType() == ProblemSpec::ELEMENT_NODE){
	  map<string,string> attributes;
	  t->getAttributes(attributes);
	  string tsfile = attributes["href"];
	  if(tsfile == "")
	    throw InternalError("timestep href not found");
	  
	  XMLURL url(d_base, tsfile.c_str());

	  char* urltext = XMLString::transcode(url.getURLText());
	  ProblemSpecReader psr(urltext);
	  delete [] urltext;

	  ProblemSpecP top = psr.readInputFile();
	  
	  d_tstop.push_back(top);
	  d_tsurl.push_back(url);
	  ProblemSpecP time = top->findBlock("Time");
	  if(time == 0)
	    throw InternalError("Cannot find Time block");
	  
	  int timestepNumber;
	  if(!time->get("timestepNumber", timestepNumber))
	    throw InternalError("Cannot find timestepNumber");

	  double currentTime;
	  if(!time->get("currentTime", currentTime))
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

ProblemSpecP
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
    return 0; 
  found_url = d_tsurl[i];
  return d_tstop[i];
}

ProblemSpecP
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
  ProblemSpecP top = getTimestep(time, url);
  if (top == 0)
    throw InternalError("Cannot find Grid in timestep");
  ProblemSpecP gridnode = top->findBlock("Grid");
  if(gridnode == 0)
    throw InternalError("Cannot find Grid in timestep");
  int numLevels = -1234;
  GridP grid = scinew Grid;
  for(ProblemSpecP n = gridnode->getFirstChild(); n != 0; n=n->getNextSibling()){
    if(n->getNodeName() == "numLevels") {
      if(!n->get(numLevels))
	throw InternalError("Error parsing numLevels");
    } else if(n->getNodeName() == "Level"){
      Point anchor;
      if(!n->get("anchor", anchor))
	throw InternalError("Error parsing level anchor point");
      Vector dcell;
      if(!n->get("cellspacing", dcell))
	throw InternalError("Error parsing level cellspacing");
      int id;
      if(!n->get("id", id)){
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
      for(ProblemSpecP r = n->getFirstChild(); r != 0; r=r->getNextSibling()){
	if(r->getNodeName() == "numPatches" ||
	   r->getNodeName() == "numRegions") {
	  if(!r->get(numPatches))
	    throw InternalError("Error parsing numRegions");
	} else if(r->getNodeName() == "totalCells") {
	  if(!r->get(totalCells))
	    throw InternalError("Error parsing totalCells");
	} else if(r->getNodeName() == "Patch" ||
		  r->getNodeName() == "Region") {
	  int id;
	  if(!r->get("id", id))
	    throw InternalError("Error parsing patch id");
	  IntVector lowIndex;
	  if(!r->get("lowIndex", lowIndex))
	    throw InternalError("Error parsing patch lowIndex");
	  IntVector highIndex;
	  if(!r->get("highIndex", highIndex))
	    throw InternalError("Error parsing patch highIndex");
	  long totalCells;
	  if(!r->get("totalCells", totalCells))
	    throw InternalError("Error parsing patch total cells");
	  Patch* patch = level->addPatch(lowIndex, highIndex,lowIndex,
				     highIndex,id);
	  ASSERTEQ(patch->totalCells(), totalCells);
	} else if(r->getNodeName() == "anchor"
		  || r->getNodeName() == "cellspacing"
		  || r->getNodeName() == "id") {
	  // Nothing - handled above
	} else if(r->getNodeName() == "periodic") {
	  if(!n->get("periodic", periodicBoundaries))
	    throw InternalError("Error parsing periodoc");
	} else if(r->getNodeType() != ProblemSpec::TEXT_NODE){
	  cerr << "WARNING: Unknown level data: " << r->getNodeName() << '\n';
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
    } else if(n->getNodeType() != ProblemSpec::TEXT_NODE){
      cerr << "WARNING: Unknown grid data: " << n->getNodeName() << '\n';
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
  ProblemSpecP vars = d_indexDoc->findBlock("variables");
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
  ProblemSpecP vars = d_indexDoc->findBlock("globals");
  if(vars == 0)
    return;
  queryVariables(vars, names, types);

  d_lock.unlock();

  dbg << "DataArchive::queryGlobals completed in " << Time::currentSeconds()-start << " seconds\n";   
}

void
DataArchive::queryVariables(ProblemSpecP vars, vector<string>& names,
			    vector<const TypeDescription*>& types)
{
  for(ProblemSpecP n = vars->getFirstChild(); n != 0; n = n->getNextSibling()){
    if(n->getNodeName() == "variable") {
      map<string,string> attributes;
      n->getAttributes(attributes);

      string type = attributes["type"];
      if(type == "")
	throw InternalError("Variable type not found");
      const TypeDescription* td = TypeDescription::lookupType(type);
      if(!td){
	static TypeDescription* unknown_type = 0;
	if(!unknown_type)
	  unknown_type = scinew TypeDescription(TypeDescription::Unknown,
						"-- unknown type --",
						false, MPI_Datatype(-1));
	td = unknown_type;
      }
      types.push_back(td);
      string name = attributes["name"];
      if(name == "")
	throw InternalError("Variable name not found");
      names.push_back(name);
    } else if(n->getNodeType() != ProblemSpec::TEXT_NODE){
      cerr << "WARNING: Unknown variable data: " << n->getNodeName() << '\n';
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
  ProblemSpecP vnode = findVariable(name, patch, matlIndex, time, url);
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
DataArchive::query( Variable& var, ProblemSpecP vnode, XMLURL url,
		    int matlIndex, const Patch* patch )
{
  d_lock.lock();
  map<string,string> attributes;
  vnode->getAttributes(attributes);

  string type = attributes["type"];
  if(type == "")
    throw InternalError("Variable doesn't have a type");
  const TypeDescription* td = var.virtualGetTypeDescription();
  ASSERT(td->getName() == type);
  
  if (td->getType() == TypeDescription::ParticleVariable) {
    int numParticles;
    if(!vnode->get("numParticles", numParticles))
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
			     matlIndex, patch, 0);
    }
    (static_cast<ParticleVariableBase*>(&var))->allocate(psubset);
//      (dynamic_cast<ParticleVariableBase*>(&var))->allocate(psubset);
  }
  else if (td->getType() != TypeDescription::ReductionVariable)
    var.allocate(patch);
  
  long start;
  if(!vnode->get("start", start))
    throw InternalError("Cannot get start");
  long end;
  if(!vnode->get("end", end))
    throw InternalError("Cannot get end");
  string filename;  
  if(!vnode->get("filename", filename))
    throw InternalError("Cannot get filename");
  string compressionMode;  
  if(!vnode->get("compression", compressionMode))
    compressionMode = "";
  
  XMLURL dataurl(url, filename.c_str());
  if(dataurl.getProtocol() != XMLURL::File) {
    char* urlpath = XMLString::transcode(dataurl.getPath());
    throw InternalError(string("Cannot read over: ")+urlpath);
  }
  char* urlpath = XMLString::transcode(dataurl.getPath());
  string datafile(urlpath);
  delete [] urlpath;
  
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
    if( idx < var.getParticleSet()->numParticles() && var[idx] == particleID )
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

  ProblemSpecP timeBlock = d_tstop[i]->findBlock("Time");
  if (!timeBlock->get("delt", *pDelt))
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
  ProblemSpecP restartNode = d_indexDoc->findBlock("restart");
  if (restartNode == 0) {
    ProblemSpecP restartsNode = d_indexDoc->findBlock("restarts");
    if (restartsNode == 0)
      return false;
    
    restartNode = restartsNode->findBlock("restart");
    if (restartNode == 0)
      return false;

    // get the last restart tag in the restarts list
    while (restartNode->findNextBlock("restart") != 0)
      restartNode = restartNode->findNextBlock("restart");
  }
  
  map<string,string> attributes;
  restartNode->getAttributes(attributes);
  string ts = attributes["timestep"];
  if (ts == "")
    return false;
  timestep = atoi(ts.c_str());
  return true;
}

void
DataArchive::initVariable(const Patch* patch,
			  DataWarehouse* dw,
			  VarLabel* label, int matl,
			  pair<ProblemSpecP, XMLURL> dataRef)
{
  Variable* var = label->typeDescription()->createInstance();
  ProblemSpecP vnode = dataRef.first;
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
					const vector<ProblemSpecP>& tsTopNodes,
					int processor, int numProcessors)
{
  ASSERTL3(tsTimes.size() == tsTopNodes.size());
  ASSERTL3(tsUrls.size() == tsTopNodes.size());

  for (int i = 0; i < (int)tsTimes.size(); i++)
    d_patchHashMaps[tsTimes[i]].init(tsUrls[i], tsTopNodes[i],
				     processor, numProcessors);
   
  d_lastFoundIt = d_patchHashMaps.end();
}

inline ProblemSpecP
DataArchive::TimeHashMaps::findVariable(const string& name,
					const Patch* patch, int matl,
					double time, XMLURL& foundUrl)
{
  PatchHashMaps* timeData = findTimeData(time);
  return (timeData == NULL) ? scinew ProblemSpec(0) :
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

DataArchive::PatchHashMaps::~PatchHashMaps() {
  int size = static_cast<int>(docs.size());
  for (int i = 0; i < size; i++) {
    docs[i]->releaseDocument();
  }
}

void DataArchive::PatchHashMaps::init(XMLURL tsUrl, ProblemSpecP tsTopNode,
				      int processor, int numProcessors)
{
  d_isParsed = false;
  // grab the data xml files from the timestep xml file
  ASSERTL3(tsTopNode != 0);
  ProblemSpecP datanode = tsTopNode->findBlock("Data");
  if(datanode == 0)
    throw InternalError("Cannot find Data in timestep");
  for(ProblemSpecP n = datanode->getFirstChild(); n != 0; n=n->getNextSibling()){
    if(n->getNodeName() == "Datafile") {
      map<string,string> attributes;
      n->getAttributes(attributes);
      string proc = attributes["proc"];
      if (proc != "") {
	int procnum = atoi(proc.c_str());
	if ((procnum % numProcessors) != processor)
	  continue;
      }
	 
      string datafile = attributes["href"];
      if(datafile == "")
	throw InternalError("timestep href not found");
      XMLURL url(tsUrl, datafile.c_str());
      d_xmlUrls.push_back(url);
    }
    else if(n->getNodeType() != ProblemSpec::TEXT_NODE){
      cerr << "WARNING: Unknown element in Data section: " << n->getNodeName() << '\n';
    }
  }
}

void DataArchive::PatchHashMaps::purgeCache()
{
  d_matHashMaps.clear();
  d_isParsed = false;

  for (unsigned int i = 0; i < docs.size(); i++)
    docs[i]->releaseDocument();
  docs.clear();
}

void DataArchive::PatchHashMaps::parse()
{
  for (list<XMLURL>::iterator urlIt = d_xmlUrls.begin();
       urlIt != d_xmlUrls.end(); urlIt++) {

    char* urltext = XMLString::transcode(urlIt->getURLText());
    //cerr << "reading: " << to_char_ptr(urlIt->getURLText()) << '\n';

    ProblemSpecReader psr(urltext);
    delete [] urltext;

    ProblemSpecP top = psr.readInputFile();
    docs.push_back(top);

    for(ProblemSpecP r = top->getFirstChild(); r != 0; r=r->getNextSibling()){
      if(r->getNodeName() == "Variable") {
	string varname;
	if(!r->get("variable", varname))
	  throw InternalError("Cannot get variable name");
	
	int patchid;
	if(!r->get("patch", patchid) && !r->get("region", patchid))
	  throw InternalError("Cannot get patch id");
	
	int index;
	if(!r->get("index", index))
	  throw InternalError("Cannot get index");
	
	add(varname, patchid, index, r, *urlIt);
      } else if(r->getNodeType() != ProblemSpec::TEXT_NODE){
	cerr << "WARNING: Unknown element in Variables section: " << r->getNodeName() << '\n';
      }
    }

  }
  
  d_isParsed = true;
  d_lastFoundIt = d_matHashMaps.end();
}

inline ProblemSpecP
DataArchive::PatchHashMaps::findVariable(const string& name,
					 const Patch* patch,
					 int matl,
					 XMLURL& foundUrl)
{
  MaterialHashMaps* patchData = findPatchData(patch);
  return (patchData == NULL) ? scinew ProblemSpec(0) :
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

ProblemSpecP DataArchive::MaterialHashMaps::findVariable(const string& name,
						     int matl,
						     XMLURL& foundUrl)
{
  matl++; // allows for matl=-1 for universal variables
 
  if (matl < (int)d_varHashMaps.size()) {
    VarHashMap& hashMap = d_varHashMaps[matl];
    pair<ProblemSpecP, XMLURL> found;
    if (hashMap.lookup(name, found)) {
      foundUrl = found.second;
      return found.first;
    }
  }
  return NULL;  
}

void DataArchive::MaterialHashMaps::add(const string& name, int matl,
					ProblemSpecP varNode, XMLURL url)
{
  matl++; // allows for matl=-1 for universal variables
   
  if (matl >= (int)d_varHashMaps.size())
    d_varHashMaps.resize(matl + 1);
  pair<ProblemSpecP, XMLURL> value(varNode, url);
  pair<ProblemSpecP, XMLURL> dummy;
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
