#include <Packages/Uintah/Core/DataArchive/DataArchive.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/CCA/Components/ProblemSpecification/ProblemSpecReader.h>
#include <Packages/Uintah/CCA/Ports/InputContext.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/UnknownVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/VarLabel.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Math/MiscMath.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Util/Assert.h>
#include <Core/Thread/Time.h>
#include <Core/Util/DebugStream.h>
#include <Core/Containers/OffsetArray1.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <fcntl.h>

#ifdef _WIN32
#  include <io.h>
#else
#  include <sys/param.h>
#  include <unistd.h>
#endif

using namespace std;

using namespace Uintah;
using namespace SCIRun;

DebugStream DataArchive::dbg("DataArchive", false);

DataArchive::DataArchive(const std::string& filebase,
                         int processor /* = 0 */, int numProcessors /* = 1 */,
                         bool verbose /* = true */ ) :
  ref_cnt(0), lock("DataArchive ref_cnt lock"),
  d_filebase(filebase), 
  d_cell_scale( Vector(1.0,1.0,1.0) ),
  d_processor(processor), d_numProcessors(numProcessors),
  default_cache_size(10), timestep_cache_size(10),
  d_lock("DataArchive lock")
{

  string index(filebase+"/index.xml");
  
  ProblemSpecReader psr(index.c_str());
  if( verbose && processor == 0) {
    cerr << "Parsing " << index << endl;
  }

  d_indexDoc = psr.readInputFile();

  d_globalEndianness = "";
  d_globalNumBits = -1;
  queryEndiannessAndBits(d_indexDoc, d_globalEndianness, d_globalNumBits);
}


DataArchive::~DataArchive()
{
  d_indexDoc->releaseDocument();
}

// static, so can be called from either DataArchive or TimeData
void
DataArchive::queryEndiannessAndBits(ProblemSpecP doc, string& endianness, int& numBits)
{
  ProblemSpecP meta = doc->findBlock("Meta");

  if (meta == 0) {
    return;
  }

  ProblemSpecP endian_node = meta->findBlock("endianness");
  if (endian_node) {
    endianness = endian_node->getNodeValue();
  }

  ProblemSpecP nBits_node = meta->findBlock("nBits");
  if( nBits_node) {
    numBits = atoi(nBits_node->getNodeValue().c_str());
  }
}

void
DataArchive::queryTimesteps( std::vector<int>& index,
                             std::vector<double>& times )
{
  double start = Time::currentSeconds();
  if(d_timeData.size() == 0){
    d_lock.lock();
    if(d_timeData.size() == 0){
      ProblemSpecP ts = d_indexDoc->findBlock("timesteps");
      if(ts == 0)
        throw InternalError("DataArchive::queryTimestepstimes:steps node not found in index.xml",
                            __FILE__, __LINE__);
      for(ProblemSpecP t = ts->getFirstChild(); t != 0; t = t->getNextSibling()){
        if(t->getNodeType() == ProblemSpec::ELEMENT_NODE){
          map<string,string> attributes;
          t->getAttributes(attributes);
          string tsfile = attributes["href"];
          if(tsfile == "")
            throw InternalError("DataArchive::queryTimesteps:timestep href not found",
                                  __FILE__, __LINE__);
          

          int timestepNumber;
          double currentTime;
          string ts = d_filebase + "/" + tsfile;
          ProblemSpecP timestepDoc = 0;

          if(attributes["time"] == "") {
            // This block if for earlier versions of the index.xml file that do not
            // contain time information as attributes of the timestep field.

            ProblemSpecReader psr(ts.c_str());
            
            timestepDoc = psr.readInputFile();
            
            ProblemSpecP time = timestepDoc->findBlock("Time");
            if(time == 0)
              throw InternalError("DataArchive::queryTimesteps:Cannot find Time block",
                                  __FILE__, __LINE__);
            
            if(!time->get("timestepNumber", timestepNumber))
              throw InternalError("DataArchive::queryTimesteps:Cannot find timestepNumber",
                                  __FILE__, __LINE__);
            
            if(!time->get("currentTime", currentTime))
              throw InternalError("DataArchive::queryTimesteps:Cannot find currentTime",
                                  __FILE__, __LINE__);
          } else {
            // This block will read delt and time info from the index.xml file instead of
            // opening every single timestep.xml file to get this information
            istringstream timeVal(attributes["time"]);
            istringstream timestepVal(t->getNodeValue());

            timeVal >> currentTime;
            timestepVal >> timestepNumber;

          }

          d_tsindex.push_back(timestepNumber);
          d_tstimes.push_back(currentTime);
          d_timeData.push_back(TimeData(this, timestepDoc, ts));
        }
      }
    }
    d_lock.unlock();
  }
  index=d_tsindex;
  times=d_tstimes;
  dbg << "DataArchive::queryTimesteps completed in " << Time::currentSeconds()-start << " seconds\n";
}

DataArchive::TimeData& 
DataArchive::getTimeData(int index)
{
  ASSERTRANGE(index, 0, d_timeData.size());
  TimeData& td = d_timeData[index];
  if (!td.d_initialized)
    td.init();

  list<int>::iterator is_cached = std::find(d_lastNtimesteps.begin(), d_lastNtimesteps.end(), index);
  if (is_cached != d_lastNtimesteps.end()) {
    // It's in the list, so yank it in preperation for putting it at
    // the top of the list.
    dbg << "Already cached, putting at top of list.\n";
    d_lastNtimesteps.erase(is_cached);
  } else {
    dbg << "Not in list.\n";
    // Not in the list.  If the list is maxed out, purge the cache
    // of the last item by removing it from the list.  If
    // timestep_cache_size is <= 0, there is an unlimited size to
    // the cache, so don't purge.
    dbg << "timestep_cache_size = "<<timestep_cache_size<<", d_lastNtimesteps.size() = "<<d_lastNtimesteps.size()<<"\n";
    if (timestep_cache_size > 0 && (int)(d_lastNtimesteps.size()) >= timestep_cache_size) {
      int cacheTimestep = d_lastNtimesteps.back();
      d_lastNtimesteps.pop_back();
      dbg << "Making room.  Purging index "<< cacheTimestep <<"\n";
      d_timeData[cacheTimestep].purgeCache();
    }
  }
  // Finally insert our new candidate at the top of the list.
  d_lastNtimesteps.push_front(index);

  return td;
}

int
DataArchive::queryPatchwiseProcessor(const Patch* patch, int index )
{
  TimeData& timedata = getTimeData(index);
  d_lock.lock();

  int proc = timedata.d_patchInfo[patch->getLevel()->
                                  getIndex()][patch->getLevelIndex()].proc;

  d_lock.unlock();
  return proc;
}

int DataArchive::queryNumProcs(int index)
{
  ProblemSpecP ps=getTimestepDoc(index);
  ProblemSpecP meta=ps->findBlock("Meta");
  int procs=-1;
  if(meta!=0)
  {
    meta->get("numProcs",procs);
  }
  return procs;
}

GridP
DataArchive::queryGrid( int index, const ProblemSpec* ups)
{
  double start = Time::currentSeconds();
  d_lock.lock();

  TimeData& timedata = getTimeData(index);
  if (timedata.d_grid != 0) {
    d_lock.unlock();
    return timedata.d_grid;
  }

  timedata.d_patchInfo.clear();
  timedata.d_matlInfo.clear();

  const ProblemSpecP top = timedata.d_tstop;
  if (top == 0)
    throw InternalError("DataArchive::queryGrid:Cannot find Grid in timestep",
                        __FILE__, __LINE__);
  ProblemSpecP gridnode = top->findBlock("Grid");
  if(gridnode == 0)
    throw InternalError("DataArchive::queryGrid:Cannot find Grid in timestep",
                        __FILE__, __LINE__);
  int numLevels = -1234;
  GridP grid = scinew Grid;
  int levelIndex = -1;
  for(ProblemSpecP n = gridnode->getFirstChild(); n != 0; n=n->getNextSibling()){
    if(n->getNodeName() == "numLevels") {
      if(!n->get(numLevels))
        throw InternalError("DataArchive::queryGrid:Error parsing numLevels",
                            __FILE__, __LINE__);
    } else if(n->getNodeName() == "Level"){
      Point anchor;
      if(!n->get("anchor", anchor))
        throw InternalError("DataArchive::queryGrid:DataArchive::queryGrid:Error parsing level anchor point",
                            __FILE__, __LINE__);
      Vector dcell;
      OffsetArray1<double> faces[3];
      bool have_stretch = false;
      if(!n->get("cellspacing", dcell)) {
        // stretched grids
        bool have_axis[3] = {false, false, false};
        for (ProblemSpecP stretch = n->findBlock("StretchPositions"); stretch != 0; stretch = stretch->findNextBlock("StretchPositions")) {
          map<string,string> attributes;
          stretch->getAttributes(attributes);
          string mapstring;
          int axis, low, high;

          mapstring = attributes["axis"];
          if (mapstring == "") 
            throw InternalError("DataArchive::queryGrid:DataArchive::queryGrid:Error parsing StretchPositions axis",
                            __FILE__, __LINE__);
          axis = atoi(mapstring.c_str());
          mapstring = attributes["low"];
          if (mapstring == "") 
            throw InternalError("DataArchive::queryGrid:DataArchive::queryGrid:Error parsing StretchPositions low index",
                            __FILE__, __LINE__);
          low = atoi(mapstring.c_str());
          mapstring = attributes["high"];
          if (mapstring == "") 
            throw InternalError("DataArchive::queryGrid:DataArchive::queryGrid:Error parsing StretchPositions high index",
                            __FILE__, __LINE__);
          high = atoi(mapstring.c_str());

          if (have_axis[axis])
            throw InternalError("DataArchive::queryGrid:DataArchive::queryGrid:StretchPositions already defined for axis",
                            __FILE__, __LINE__);
          have_axis[axis] = true;

          faces[axis].resize(low, high);
          int index = low;
          for (ProblemSpecP pos = stretch->findBlock("pos"); pos != 0; pos = pos->findNextBlock("pos")) {
            pos->get(faces[axis][index]);
            index++;
          }
          ASSERTEQ(index, high);
        }
        if (have_axis[0] == false && have_axis[1] == false && have_axis[2] == false)
          throw InternalError("DataArchive::queryGrid:Error parsing level cellspacing",
                              __FILE__, __LINE__);
        else if (have_axis[0] == false || have_axis[1] == false || have_axis[2] == false)
          throw InternalError("DataArchive::queryGrid:Error parsing stretch grid axes",
                              __FILE__, __LINE__);
        have_stretch = true;
      }
      if (!have_stretch)
        dcell *= d_cell_scale;
      IntVector extraCells(0,0,0);
      n->get("extraCells", extraCells);
      
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
      level->setExtraCells(extraCells);
      if (have_stretch) {
        level->setStretched((Grid::Axis)0, faces[0]);
        level->setStretched((Grid::Axis)1, faces[1]);
        level->setStretched((Grid::Axis)2, faces[2]);
      }
      levelIndex++;
      timedata.d_patchInfo.push_back(vector<PatchData>());
      timedata.d_matlInfo.push_back(vector<bool>());

      int numPatches = -1234;
      long totalCells = 0;
      IntVector periodicBoundaries(0, 0, 0);      
      for(ProblemSpecP r = n->getFirstChild(); r != 0; r=r->getNextSibling()){
        if(r->getNodeName() == "numPatches" ||
           r->getNodeName() == "numRegions") {
          if(!r->get(numPatches))
            throw InternalError("DataArchive::queryGrid:Error parsing numRegions",
                                __FILE__, __LINE__);
        } else if(r->getNodeName() == "totalCells") {
          if(!r->get(totalCells))
            throw InternalError("DataArchive::queryGrid:Error parsing totalCells",
                                __FILE__, __LINE__);
        } else if(r->getNodeName() == "Patch" ||
                  r->getNodeName() == "Region") {
          int id;
          if(!r->get("id", id))
            throw InternalError("DataArchive::queryGrid:Error parsing patch id",
                                __FILE__, __LINE__);
          int proc = -1;
          IntVector lowIndex;
          if(!r->get("lowIndex", lowIndex))
            throw InternalError("DataArchive::queryGrid:Error parsing patch lowIndex",
                                __FILE__, __LINE__);
          IntVector highIndex;
          if(!r->get("highIndex", highIndex))
            throw InternalError("DataArchive::queryGrid:Error parsing patch highIndex",
                                __FILE__, __LINE__);
          IntVector inLowIndex = lowIndex;
          IntVector inHighIndex = highIndex;
          r->get("interiorLowIndex", inLowIndex);
          r->get("interiorHighIndex", inHighIndex);
          long totalCells;
          if(!r->get("totalCells", totalCells))
            throw InternalError("DataArchive::queryGrid:Error parsing patch total cells",
                                __FILE__, __LINE__);
          Patch* patch = level->addPatch(lowIndex, highIndex,inLowIndex, inHighIndex,id);
          ASSERTEQ(patch->totalCells(), totalCells);
          PatchData pi;
          r->get("proc", pi.proc); // defaults to -1 if not available
          timedata.d_patchInfo[levelIndex].push_back(pi);
        } else if(r->getNodeName() == "anchor"
                  || r->getNodeName() == "cellspacing"    // This could use a comment or 2 --Todd
                  || r->getNodeName() == "id"
                  || r->getNodeName() == "extraCells") {
          // Nothing - handled above
        } else if(r->getNodeName() == "periodic") {
          if(!n->get("periodic", periodicBoundaries))
            throw InternalError("DataArchive::queryGrid:Error parsing periodoc", __FILE__, __LINE__);
        } else if(r->getNodeType() != ProblemSpec::TEXT_NODE){
          //cerr << "DataArchive::queryGrid:WARNING: Unknown level data: " << r->getNodeName() << '\n';
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
      if (ups) {
        // this is not necessary on non-restarts.
        ProblemSpecP grid_ps = ups->findBlock("Grid");
        level->assignBCS(grid_ps);
       }

    } else if(n->getNodeType() != ProblemSpec::TEXT_NODE){
      //cerr << "DataArchive::queryGrid:WARNING: Unknown grid data: " << n->getNodeName() << '\n';
    }
  }
  
  d_lock.unlock();
  grid->performConsistencyCheck();

  timedata.d_grid = grid;

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
                             vector<const Uintah::TypeDescription*>& types)
{
  double start = Time::currentSeconds();
  d_lock.lock();
  ProblemSpecP vars = d_indexDoc->findBlock("variables");
  if(vars == 0)
    throw InternalError("DataArchive::queryVariables:variables section not found\n",
                        __FILE__, __LINE__);
  queryVariables(vars, names, types);

  d_lock.unlock();
  dbg << "DataArchive::queryVariables completed in " << Time::currentSeconds()-start << " seconds\n";
}

void
DataArchive::queryGlobals( vector<string>& names,
                           vector<const Uintah::TypeDescription*>& types)
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
                            vector<const Uintah::TypeDescription*>& types)
{
  for(ProblemSpecP n = vars->getFirstChild(); n != 0; n = n->getNextSibling()){
    if(n->getNodeName() == "variable") {
      map<string,string> attributes;
      n->getAttributes(attributes);

      string type = attributes["type"];
      if(type == "")
        throw InternalError("DataArchive::queryVariables:Variable type not found",
                            __FILE__, __LINE__);
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
        throw InternalError("DataArchive::queryVariables:Variable name not found",
                            __FILE__, __LINE__);
      names.push_back(name);
    } else if(n->getNodeType() != ProblemSpec::TEXT_NODE){
      cerr << "DataArchive::queryVariables:WARNING: Unknown variable data: " << n->getNodeName() << '\n';
    }
  }
}

void
DataArchive::query( Variable& var, const std::string& name, int matlIndex, 
                    const Patch* patch, int index, DataFileInfo* dfi /* = 0 */)
{
  double tstart = Time::currentSeconds();
  string url;

#ifndef _WIN32
  const char* tag = AllocatorSetDefaultTag("QUERY");
#endif

  TimeData& timedata = getTimeData(index);
  ASSERT(timedata.d_initialized);
  // make sure info for this patch gets parsed from p*****.xml.
  d_lock.lock();  
  timedata.parsePatch(patch);
  d_lock.unlock();  

  VarData& varinfo = timedata.d_varInfo[name];
  string dataurl;
  int patchid;
  if (patch) {
    PatchData& patchinfo = timedata.d_patchInfo[patch->getLevel()->getIndex()][patch->getLevelIndex()];
    ASSERT(patchinfo.parsed);
    patchid = patch->getRealPatch()->getID();

    ostringstream ostr;
    // append l#/datafilename to the directory
    ostr << timedata.d_tsurldir << "l" << patch->getLevel()->getIndex() << "/" << patchinfo.datafilename;
    dataurl = ostr.str();
  }
  else {
    // reference reduction file 'global.data' will a null patch
    patchid = -1;
    dataurl = timedata.d_tsurldir + timedata.d_globaldata;
  }

  // on a call from restartInitialize, we already have the information from the dfi,
  // otherwise get it from the hash table info
  DataFileInfo datafileinfo;
  if (!dfi) {
    // if this is a virtual patch, grab the real patch, but only do that here - in the next query, we want
    // the data to be returned in the virtual coordinate space
    if (!timedata.d_datafileInfo.lookup(VarnameMatlPatch(name, matlIndex, patchid), datafileinfo)) {
      cerr << "VARIABLE NOT FOUND: " << name << ", material index " << matlIndex << ", patch " << patch->getID() << ", time index " << index << "\nPlease make sure the correct material index is specified\n";
      throw InternalError("DataArchive::query:Variable not found",
                          __FILE__, __LINE__);
    }
    dfi = &datafileinfo;
  }
  const TypeDescription* td = var.virtualGetTypeDescription();
  ASSERT(td->getName() == varinfo.type);
  
  if (td->getType() == TypeDescription::ParticleVariable) {
    if(dfi->numParticles == -1)
      throw InternalError("DataArchive::query:Cannot get numParticles",
                          __FILE__, __LINE__);
    psetDBType::key_type key(matlIndex, patch);
    ParticleSubset* psubset = 0;
    psetDBType::iterator psetIter = d_psetDB.find(key);
    if(psetIter != d_psetDB.end()) {
      psubset = (*psetIter).second.get_rep();
    }
    if (psubset == 0 || psubset->numParticles() != dfi->numParticles)
    {
     d_psetDB[key] = psubset =
       scinew ParticleSubset(scinew ParticleSet(dfi->numParticles), true,
                             matlIndex, patch, 0);
    }
    (static_cast<ParticleVariableBase*>(&var))->allocate(psubset);
//      (dynamic_cast<ParticleVariableBase*>(&var))->allocate(psubset);
  }
  else if (td->getType() != TypeDescription::ReductionVariable) {
    var.allocate(patch, varinfo.boundaryLayer);
  }
  
#ifdef _WIN32
  int fd = open(dataurl.c_str(), O_RDONLY|O_BINARY);
#else
  int fd = open(dataurl.c_str(), O_RDONLY);
#endif
  if(fd == -1) {
    cerr << "Error opening file: " << dataurl.c_str() << ", errno=" << errno << '\n';
    throw ErrnoException("DataArchive::query (open call)", errno, __FILE__, __LINE__);
  }
#ifdef __sgi
  off64_t ls = lseek64(fd, dfi->start, SEEK_SET);
#else
  off_t ls = lseek(fd, dfi->start, SEEK_SET);
#endif
  if(ls == -1) {
    cerr << "Error lseek - file: " << dataurl.c_str() << ", errno=" << errno << '\n';
    throw ErrnoException("DataArchive::query (lseek call)", errno, __FILE__, __LINE__);
  }
  InputContext ic(fd, dataurl.c_str(), dfi->start);
  double starttime = Time::currentSeconds();
  var.read(ic, dfi->end, timedata.d_swapBytes, timedata.d_nBytes, varinfo.compression);

  dbg << "DataArchive::query: time to read raw data: "<<Time::currentSeconds() - starttime<<endl;
  ASSERTEQ(dfi->end, ic.cur);
  int s = close(fd);
  if(s == -1) {
    cerr << "Error closing file: " << dataurl.c_str() << ", errno=" << errno << '\n';
    throw ErrnoException("DataArchive::query (close call)", errno, __FILE__, __LINE__);
  }

#ifndef _WIN32
  AllocatorSetDefaultTag(tag);
#endif
  dbg << "DataArchive::query() completed in "
      << Time::currentSeconds()-tstart << " seconds\n";
}


void 
DataArchive::findPatchAndIndex(GridP grid, Patch*& patch, particleIndex& idx,
                               long64 particleID, int matlIndex, int levelIndex,
                               int index)
{
  Patch *local = patch;
  if( patch != NULL ){
    ParticleVariable<long64> var;
    query(var, "p.particleID", matlIndex, patch, index);
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
//   for (int level_nr = 0;
//        (level_nr < grid->numLevels()) && (patch == NULL); level_nr++) {
    
//     const LevelP level = grid->getLevel(level_nr);
    const LevelP level = grid->getLevel(levelIndex);
    
    for (Level::const_patchIterator iter = level->patchesBegin();
         (iter != level->patchesEnd()) && (patch == NULL); iter++) {
      if( *iter == local ) continue;
      ParticleVariable<long64> var;
      query(var, "p.particleID", matlIndex, *iter, index);
      ParticleSubset* subset = var.getParticleSubset();
      for(ParticleSubset::iterator p_iter = subset->begin();
          p_iter != subset->end(); p_iter++){
        if( var[*p_iter] == particleID){
          patch = *iter;
          idx = *p_iter;
          //      cerr<<"var["<<*p_iter<<"] = "<<var[*p_iter]<<endl;
          break;
        }
      }
      
      if( patch != NULL )
        break;
    }
//  }
}

void
DataArchive::restartInitialize(int index, const GridP& grid, DataWarehouse* dw,
                               LoadBalancer* lb, double* pTime)
{
  vector<int> indices;
  vector<double> times;
  queryTimesteps(indices, times);

  vector<string> names;
  vector< const TypeDescription *> typeDescriptions;
  queryVariables(names, typeDescriptions);
  queryGlobals(names, typeDescriptions);  
  
  map<string, VarLabel*> varMap;
  for (unsigned i = 0; i < names.size(); i++) {
    varMap[names[i]] = VarLabel::find(names[i]);
  }

  TimeData& timedata = getTimeData(index);

  *pTime = times[index];

  if (lb)
    lb->restartInitialize(timedata.d_tstop, timedata.d_tsurl, grid);

  // set here instead of the SimCont because we need the DW ID to be set 
  // before saving particle subsets
  dw->setID( indices[index]);
  
  // make sure to load all the data so we can iterate through it 
  for (int l = 0; l < grid->numLevels(); l++) {
    LevelP level = grid->getLevel(l);
    for (int p = 0; p < level->numPatches(); p++) {
      const Patch* patch = level->getPatch(p);
      if (lb->getPatchwiseProcessorAssignment(patch) == d_processor)
        timedata.parsePatch(patch);
    }
  }

  // iterate through all entries in the VarData hash table, and loading the 
  // variables if that data belongs on this processor
  VarHashMapIterator iter(&timedata.d_datafileInfo);
  iter.first();
  for (iter; iter.ok(); ++iter) {
    VarnameMatlPatch& key = iter.get_key();
    DataFileInfo& data = iter.get_data();

    // get the Patch from the Patch ID (ID of -1 = NULL - for reduction vars)
    const Patch* patch = key.patchid_ == -1 ? NULL : grid->getPatchByID(key.patchid_, 0);
    int matl = key.matlIndex_;

    VarLabel* label = varMap[key.name_];
    if (label == 0) {
      throw UnknownVariable(key.name_, dw->getID(), patch, matl,
                            "on DataArchive::scheduleRestartInitialize",
                            __FILE__, __LINE__);
    }

    if (!patch || !lb || lb->getPatchwiseProcessorAssignment(patch) == d_processor) {
      Variable* var = label->typeDescription()->createInstance();
      query(*var, key.name_, matl, patch, index, &data);

      ParticleVariableBase* particles;
      if ((particles = dynamic_cast<ParticleVariableBase*>(var))) {
        if (!dw->haveParticleSubset(matl, patch)) {
          dw->saveParticleSubset(particles->getParticleSubset(), matl, patch);
        }
        else {
          ASSERTEQ(dw->getParticleSubset(matl, patch), particles->getParticleSubset());
        }
      }
      dw->put(var, label, matl, patch); 
      delete var; // should have been cloned when it was put
    }
  }
}

bool
DataArchive::queryRestartTimestep(int& timestep)
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

// We want to cache at least a single timestep, so that we don't have
// to reread the timestep for every patch queried.  This sets the
// cache size to one, so that this condition is held.
void
DataArchive::turnOffXMLCaching() {
  setTimestepCacheSize(1);
}

// Sets the number of timesteps to cache back to the default_cache_size
void
DataArchive::turnOnXMLCaching() {
  setTimestepCacheSize(default_cache_size);
}

// Sets the timestep cache size to whatever you want.  This is useful
// if you want to override the default cache size determined by
// TimeHashMaps.
void
DataArchive::setTimestepCacheSize(int new_size) {
  d_lock.lock();
  // Now we need to reduce the size
  int current_size = (int)d_lastNtimesteps.size();
  dbg << "current_size = "<<current_size<<"\n";
  if (timestep_cache_size >= current_size)
    // everything's fine
    d_lock.unlock();
    return;

  int kill_count = current_size - timestep_cache_size;
  dbg << "kill_count = "<<kill_count<<"\n";
  for(int i = 0; i < kill_count; i++) {
    int cacheTimestep = d_lastNtimesteps.back();
    dbg << "Making room.  Purging time index "<< cacheTimestep <<"\n";

    d_lastNtimesteps.pop_back();
    d_timeData[cacheTimestep].purgeCache();
  }
  d_lock.unlock();
}

DataArchive::TimeData::TimeData(DataArchive* da, ProblemSpecP timestepDoc, string timestepURL) :
  da(da), d_tstop(timestepDoc), d_tsurl(timestepURL), d_initialized(false)
{
  d_tsurldir = timestepURL.substr(0, timestepURL.find_last_of('/')+1);
}

DataArchive::TimeData::~TimeData()
{
  purgeCache();
}

void
DataArchive::TimeData::init()
{
  d_initialized=true;
  //  cerr << "PatchHashMaps["<<time<<"]::init\n";
  // grab the data xml files from the timestep xml file
  if (d_tstop == 0) {
    ProblemSpecReader psr(d_tsurl.c_str());
    d_tstop = psr.readInputFile();
  }

  //handle endianness and number of bits
  string endianness = da->d_globalEndianness;
  int numbits = da->d_globalNumBits;
  DataArchive::queryEndiannessAndBits(d_tstop, endianness, numbits);

  static bool endian_warned = false;
  static bool bits_warned = false;

  if (endianness == "") {
    endianness = string(SCIRun::endianness());
    if (!endian_warned) {
      endian_warned = true;
      cout<<"\nXML Warning: endianness node not found.\n"<<
        "Assuming data was created on a " << SCIRun::endianness() << " machine.\n"<<
        "To eliminate this message and express the correct\n"<<
        "endianess, please add either\n"<<
        "\t<endianness>little_endian</endianness>\n"<<
        "or\n\t<endianness>big_endian</endianness>\n"<<
        "to the <Meta> section of the index.xml file.\n\n";
    }
  }
  if (numbits == -1) {
    numbits = sizeof(unsigned long) * 8;
    if (!bits_warned) {
      cout<<"\nXML Warning: nBits node not found.\n"<<
        "Assuming data was created using " << sizeof(unsigned long) * 8 << " bits.\n"
        "To eliminate this message and express the correct\n"<<
        "number of bits, please add either\n"<<
        "\t<nBits>32</nBits>\n"<<
        "or\n\t<nBits>64</nBits>\n"<<
        "to the <Meta> section of the index.xml file.\n\n";
    }
  }

  d_swapBytes = endianness != string(SCIRun::endianness());
  d_nBytes = numbits / 8;

  ProblemSpecP datanode = d_tstop->findBlock("Data");
  if(datanode == 0)
    throw InternalError("Cannot find Data in timestep", __FILE__, __LINE__);
  for(ProblemSpecP n = datanode->getFirstChild(); n != 0; n=n->getNextSibling()){
    if(n->getNodeName() == "Datafile") {
      map<string,string> attributes;
      n->getAttributes(attributes);
      string proc = attributes["proc"];
      /* - Remove this check for restarts.  We need to accurately
         determine which patch goes on which proc, and for the moment
         we need to be able to parse all pxxxx.xml files.  --BJW  
      if (proc != "") {
        int procnum = atoi(proc.c_str());
        if ((procnum % numProcessors) != processor)
          continue;
      }
      */
      string datafile = attributes["href"];
      if(datafile == "")
        throw InternalError("timestep href not found", __FILE__, __LINE__);

      if (datafile == "global.xml") {
        parseFile(d_tsurldir + datafile, -1, -1);
      }
      else {

        // get level info out of the xml file: should be lX/pxxxxx.xml
        int level = 0;
        int start = datafile.find_first_of("l",0, datafile.length()-3);
        int end = datafile.find_first_of("/");
        if (start != string::npos && end != string::npos && end > start && end-start <= 2)
          level = atoi(datafile.substr(start+1, end-start).c_str());

        if (level >= d_xmlUrls.size()) {
          d_xmlUrls.resize(level+1);
          d_xmlParsed.resize(level+1);
        }

        string url = d_tsurldir + datafile;
        d_xmlUrls[level].push_back(url);
        d_xmlParsed[level].push_back(false);
      }
    }
    else if(n->getNodeType() != ProblemSpec::TEXT_NODE){
      cerr << "WARNING: Unknown element in Data section: " << n->getNodeName() << '\n';
    }
  }
}

void
DataArchive::TimeData::purgeCache()
{
  d_grid = 0;
  d_tstop = 0;

  d_datafileInfo.remove_all();
  d_patchInfo.clear(); 
  d_varInfo.clear();
  d_xmlUrls.clear();
  d_xmlParsed.clear();
  d_initialized = false;
}

// This is the function that parses the p*****.xml file for a single processor.
void
DataArchive::TimeData::parseFile(string urlIt, int levelNum, int basePatch)
{
  // parse the file
  ProblemSpecReader psr(urlIt);
  ProblemSpecP top = psr.readInputFile();
  
  // materials are the same for all patches on a level - don't parse them for more than one file
  bool addMaterials = levelNum >= 0 && d_matlInfo[levelNum].size() == 0;

  for(ProblemSpecP vnode = top->getFirstChild(); vnode != 0; vnode=vnode->getNextSibling()){
    if(vnode->getNodeName() == "Variable") {
      string varname;
      if(!vnode->get("variable", varname))
        throw InternalError("Cannot get variable name", __FILE__, __LINE__);
      
      int patchid;
      if(!vnode->get("patch", patchid) && !vnode->get("region", patchid))
        throw InternalError("Cannot get patch id", __FILE__, __LINE__);
      
      int index;
      if(!vnode->get("index", index))
        throw InternalError("Cannot get index", __FILE__, __LINE__);
      
      if (addMaterials) {
        // set the material to existing.  index+1 to use matl -1
        if (index+1 >= d_matlInfo[levelNum].size())
          d_matlInfo[levelNum].resize(index+2);
        d_matlInfo[levelNum][index] = true;
      }

      map<string,string> attributes;
      vnode->getAttributes(attributes);

      string type = attributes["type"];
      if(type == "")
        throw InternalError("DataArchive::query:Variable doesn't have a type",
                            __FILE__, __LINE__);
      long start;
      if(!vnode->get("start", start))
        throw InternalError("DataArchive::query:Cannot get start", __FILE__, __LINE__);
      long end;
      if(!vnode->get("end", end))
        throw InternalError("DataArchive::query:Cannot get end",
                            __FILE__, __LINE__);
      string filename;  
      if(!vnode->get("filename", filename))
        throw InternalError("DataArchive::query:Cannot get filename",
                            __FILE__, __LINE__);

      // not required
      string compressionMode = "";  
      IntVector boundary(0,0,0);
      int numParticles = -1;

      vnode->get("compression", compressionMode);      
      vnode->get("boundaryLayer", boundary);
      vnode->get("numParticles", numParticles);

      if (d_varInfo.find(varname) == d_varInfo.end()) {
        VarData& varinfo = d_varInfo[varname];
        varinfo.type = type;
        varinfo.compression = compressionMode;
        varinfo.boundaryLayer = boundary;
      }

      if (levelNum == -1) { // global file (reduction vars)
        d_globaldata = filename;
      }
      else {
        ASSERTRANGE(patchid-basePatch, 0, d_patchInfo[levelNum].size());
        PatchData& patchinfo = d_patchInfo[levelNum][patchid-basePatch];
        if (!patchinfo.parsed) {
          patchinfo.parsed = true;
          patchinfo.datafilename = filename;
        }
      }
      VarnameMatlPatch vmp(varname, index, patchid);
      DataFileInfo dummy;

      if (d_datafileInfo.lookup(vmp, dummy) == 1) {
        //cerr << "Duplicate variable name: " << name << endl;
      }
      else {
        DataFileInfo dfi(start, end, numParticles);
        d_datafileInfo.insert(vmp, dfi);
      }
    } else if(vnode->getNodeType() != ProblemSpec::TEXT_NODE){
      cerr << "WARNING: Unknown element in Variables section: " << vnode->getNodeName() << '\n';
    }
  }
  top->releaseDocument();
}



void
DataArchive::TimeData::parsePatch(const Patch* patch)
{
  ASSERT(d_grid != 0);
  if (!patch) return;
  // make sure the data for this patch has been processed.
  // Return straightaway if we have parsed this patch
  int levelIndex = patch->getLevel()->getIndex(); 
  int levelBasePatchID = patch->getLevel()->getPatch(0)->getID();
  int patchIndex = patch->getLevelIndex();

  PatchData& patchinfo = d_patchInfo[levelIndex][patchIndex];
  if (patchinfo.parsed)
    return;

  //If this is a newer uda, the patch info in the grid will store the processor where the data is
  if (patchinfo.proc != -1) {
    ostringstream file;
    file << d_tsurldir << "l" << (int) patch->getLevel()->getIndex() << "/p" << setw(5) << setfill('0') << (int) patchinfo.proc << ".xml";
    parseFile(file.str(), levelIndex, levelBasePatchID);
  }

  // Try making a guess as to the processor.  First go is to try
  // the processor of the same index as the patch.  Many datasets
  // have only one patch per processor, so this is a reasonable
  // first attempt.  Future attemps could perhaps be smarter.
  if (!patchinfo.parsed && !d_xmlParsed[levelIndex][patchIndex]) {
    parseFile(d_xmlUrls[levelIndex][patchIndex], levelIndex, levelBasePatchID);
    d_xmlParsed[levelIndex][patchIndex] = true;
  }

  // failed the guess - parse the entire dataset for this level
  if (!patchinfo.parsed) {
    for (unsigned proc = 0; proc < d_xmlUrls[levelIndex].size(); proc++) {
      parseFile(d_xmlUrls[levelIndex][proc], levelIndex, levelBasePatchID);
      d_xmlParsed[levelIndex][proc] = true;
    }
  }

}


ConsecutiveRangeSet
DataArchive::queryMaterials( const string& varname,
                             const Patch* patch,
                             int index )
{
  double start = Time::currentSeconds();
  d_lock.lock();

  TimeData& timedata = getTimeData(index);
  timedata.parsePatch(patch);

  ConsecutiveRangeSet matls;

  for (unsigned i = 0; i < timedata.d_matlInfo[patch->getLevel()->getIndex()].size(); i++) {
    // i-1, since the matlInfo is adjusted to allow -1 as entries
    VarnameMatlPatch vmp(varname, i-1, patch->getID());
    DataFileInfo dummy;

    if (timedata.d_datafileInfo.lookup(vmp, dummy) == 1)
      matls.addInOrder(i-1);

  }

  d_lock.unlock();
  dbg << "DataArchive::queryMaterials completed in " << Time::currentSeconds()-start << " seconds\n";

  return matls;
}

int
DataArchive::queryNumMaterials(const Patch* patch, int index)
{
  double start = Time::currentSeconds();

  d_lock.lock();
  TimeData& timedata = getTimeData(index);
  timedata.parsePatch(patch);

  int numMatls;

  for (unsigned i = 0; i < timedata.d_matlInfo[patch->getLevel()->getIndex()].size(); i++) 
    if (timedata.d_matlInfo[patch->getLevel()->getIndex()][i]) 
      numMatls++;

  dbg << "DataArchive::queryNumMaterials completed in " << Time::currentSeconds()-start << " seconds\n";

  d_lock.unlock();
  return numMatls;
}

