/*
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
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

#include <Core/DataArchive/DataArchive.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <CCA/Components/ProblemSpecification/ProblemSpecReader.h>
#include <CCA/Ports/InputContext.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/LoadBalancer.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/UnknownVariable.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Level.h>
#include <Core/Math/MiscMath.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Util/Assert.h>
#include <Core/Thread/Time.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/XMLUtils.h>
#include <Core/Containers/OffsetArray1.h>

#include <fstream>
#include <iomanip>
#include <iostream>

#include <fcntl.h>
#include <stdlib.h>
#include <sys/param.h>
#include <unistd.h>

using namespace std;
using namespace Uintah;
using namespace SCIRun;

DebugStream DataArchive::dbg("DataArchive", false);

DataArchive::DataArchive( const string & filebase,
                                int      processor     /* = 0 */,
                                int      numProcessors /* = 1 */,
                                bool     verbose       /* = true */ ) :
  ref_cnt(0), lock("DataArchive ref_cnt lock"),
  timestep_cache_size(10), default_cache_size(10), 
  d_filebase(filebase), 
  d_cell_scale( Vector(1.0,1.0,1.0) ),
  d_processor(processor),
  d_numProcessors(numProcessors), d_lock("DataArchive lock"),
  d_particlePositionName("p.x"),
  d_lb( NULL )
{
  if( d_filebase == "" ) {
    throw InternalError("DataArchive::DataArchive 'filebase' cannot be empty (\"\").", __FILE__, __LINE__);
  }

  while( d_filebase[ d_filebase.length() - 1 ] == '/' ) {
    // Remove '/' from the end of the filebase (if there is one).
    d_filebase = d_filebase.substr( 0, filebase.length() - 1 );
  }

  string index = d_filebase + "/index.xml";
  if( verbose ) {
    proc0cout << "Parsing " << index << ".\n";
  }

  d_indexFile = fopen( index.c_str(), "r" ); // Was: ProblemSpecReader().readInputFile( index );
  if( d_indexFile == NULL ) {
    throw InternalError( "DataArchive::DataArchive() failed to open index xml file.", __FILE__, __LINE__ );
  }

  d_globalEndianness = "";
  d_globalNumBits = -1;
  queryEndiannessAndBits( d_indexFile, d_globalEndianness, d_globalNumBits );

  queryParticlePositionName( d_indexFile );
}

DataArchive::~DataArchive()
{
  // The d_createdVarLabels member variable, is used to keep track of
  // the VarLabels (for each of the data fields found in the data
  // archive we are reading data out of) for which a varLabel does not
  // already exist.  Now that we have read in the data, we no longer
  // need these temporary VarLabels, so delete them to avoid a memory
  // leak.  Note, most of these VarLabels will be 're-created' by
  // individual components when they go to access their data.

  map<string, VarLabel*>::iterator vm_iter = d_createdVarLabels.begin();
  for( ; vm_iter != d_createdVarLabels.end(); vm_iter++ ) {
    VarLabel::destroy( vm_iter->second );
  }
}

void
DataArchive::queryParticlePositionName( FILE * doc )
{
  rewind( doc );

  while( true ) {

    string line = UintahXML::getLine( doc );
    if( line == "" ) {
      return;
    }
    else if( line.compare( 0, 18, "<ParticlePosition>" ) == 0 ) {
      vector<string> pieces = UintahXML::splitXMLtag( line );
      d_particlePositionName = pieces[1];
      return;
    }
  }
}


// Static, so can be called from either DataArchive or TimeData.
void
DataArchive::queryEndiannessAndBits(  FILE * doc, string & endianness, int & numBits )
{
  rewind( doc );
  bool found = ProblemSpec::findBlock( "<Meta>", doc );

  if( found ) {
    while( true ) {

      string line = UintahXML::getLine( doc );
      if( line == "" || line == "</Meta>" ) {
        return;
      }
      else {
        vector<string> pieces = UintahXML::splitXMLtag( line );
        if( pieces[0] == "<endianness>" ) {
          endianness = pieces[1];
        }
        else if( pieces[0] == "<nBits>" ) {
          numBits = atoi( pieces[1].c_str() );
        }
      }
    }
  }
}

void
DataArchive::queryTimesteps( vector<int>    & index,
                             vector<double> & times )
{
  double start = Time::currentSeconds();
  if( d_timeData.size() == 0 ){
    d_lock.lock();

    if( d_timeData.size() == 0 ){

      rewind( d_indexFile );
      bool found = ProblemSpec::findBlock( "<timesteps>", d_indexFile );

      if( !found ) {
        throw InternalError( "DataArchive::queryTimesteps 'timesteps' node not found in index.xml", __FILE__, __LINE__ );
      }

      while( true ) {

        string line = UintahXML::getLine( d_indexFile );
        if( line == "" || line == "</timesteps>" ) {
          break;
        }
        else if( line.compare( 0, 10, "<timestep " ) == 0 ) {

          ProblemSpec ts_doc( line );

          map<string,string> attributes;
          ts_doc.getAttributes( attributes );
          string tsfile = attributes[ "href" ];
          if( tsfile == "" ) {
            throw InternalError("DataArchive::queryTimesteps:timestep href not found", __FILE__, __LINE__);
          }

          int          timestepNumber;
          double       currentTime;
          string       ts_path_and_filename = d_filebase + "/" + tsfile; // Usually '.../timestep.xml'
          ProblemSpecP timestepDoc = 0;

          if( attributes["time"] == "" ) {
            // This block is for earlier versions of the index.xml file that did not
            // contain time information as an attribute of the timestep field.
            throw InternalError( "DataArchive::queryTimesteps - Error, UDA is too old to read...", __FILE__, __LINE__ );
          }
          else {
            // This block will read delt and time info from the index.xml file instead of
            // opening every single timestep.xml file to get this information
            istringstream timeVal( attributes["time"] );
            istringstream timestepVal( ts_doc.getNodeValue() );

            timeVal >> currentTime;
            timestepVal >> timestepNumber;

            if( !timeVal || !timestepVal ) {
              printf( "WARNING: DataArchive.cc: stringstream failed...\n" );
            }

          }

          d_ts_indices.push_back( timestepNumber );
          d_ts_times.push_back( currentTime );
          d_timeData.push_back( TimeData( this, ts_path_and_filename ) );
        }
      } // end while
    }
    d_lock.unlock();
  }
  index = d_ts_indices;
  times = d_ts_times;
  dbg << "DataArchive::queryTimesteps completed in " << Time::currentSeconds()-start << " seconds\n";
}

DataArchive::TimeData &
DataArchive::getTimeData( int index )
{
  ASSERTRANGE(index, 0, (int)d_timeData.size());

  TimeData & td = d_timeData[ index ];
  if( !td.d_initialized ) {
    td.init();
  }


  list<int>::iterator is_cached = find( d_lastNtimesteps.begin(), d_lastNtimesteps.end(), index );

  if( is_cached != d_lastNtimesteps.end() ) {
    // It's in the list, so yank it in preperation for putting it at the top of the list.
    dbg << "Already cached, putting at top of list.\n";
    d_lastNtimesteps.erase( is_cached );
  }
  else {
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
  d_lastNtimesteps.push_front( index );
  return td;
}

int
DataArchive::queryPatchwiseProcessor( const Patch * patch, const int index )
{
  d_lock.lock();
  TimeData & timedata = getTimeData( index );

  int proc = timedata.d_patchInfo[ patch->getLevel()->getIndex() ][ patch->getLevelIndex() ].proc;
  d_lock.unlock();
  return proc;
}

GridP
DataArchive::queryGrid( int index, const ProblemSpecP ups /* = NULL */ )
{
  // The following variable along with d_cell_scale is necessary to allow the 
  // UdaScale module work.  Small domains are a problem for the SCIRun widgets
  // so UdaScale allows the user increase the domain by setting the 
  // d_cell_scale. The next call to this function will use the new scaling.
  // This can be removed if SCIRun is no longer used for visualization.
  static Vector old_cell_scale(1.0,1.0,1.0);  

  d_lock.lock();

  double     start    = Time::currentSeconds();
  TimeData & timedata = getTimeData( index );

  timedata.d_patchInfo.clear();
  timedata.d_matlInfo.clear();

  FILE * fp = fopen( timedata.d_ts_path_and_filename.c_str(), "r" );

  if( fp == NULL ) {
    throw InternalError("DataArchive::queryGrid() failed to open input file.\n", __FILE__, __LINE__);
  }

  GridP grid = scinew Grid;

  vector< vector<int> > procMap; // One vector<int> per level.

  grid->readLevelsFromFile( fp, procMap );

  fclose( fp );

  // Check to see if the grid has already been reconstructed and that
  // the cell scaling has not changed. Cell scale check can be removed
  // if SCIRun is no longer used for visualization
  if (timedata.d_grid != 0  &&  old_cell_scale == d_cell_scale) {
    d_lock.unlock();
    return timedata.d_grid;
  } 

  // update the static variable old_cell_scale if the cell scale has changed.
  // Can be removed if SCIRun is no longer used for visualization.
  if( old_cell_scale != d_cell_scale ){
    old_cell_scale = d_cell_scale;
  }

  if( ups ) { // 'ups' is non-null only for restarts.

    ProblemSpecP grid_ps = ups->findBlock( "Grid" );
    grid->assignBCS( grid_ps, NULL );
  }

  for( unsigned int levelIndex = 0; levelIndex < grid->numLevels(); levelIndex++ ) {

    // Initialize timedata with empty vectors for this level:
    timedata.d_patchInfo.push_back(vector<PatchData>());
    timedata.d_matlInfo.push_back(vector<bool>());

    // Now pull out the patch processor information that we got during the grid creation
    // and put it in the timedata struct.
    vector<int> & procMapForLevel = procMap[ levelIndex ];

    for( vector<int>::iterator iter = procMapForLevel.begin(); iter != procMapForLevel.end(); ++iter ) {
      PatchData pi;
      pi.proc = *iter;
      timedata.d_patchInfo[ levelIndex ].push_back(pi);
    }
    
  }

  timedata.d_grid = grid;

  d_lock.unlock();
  grid->performConsistencyCheck();

  timedata.d_grid = grid;

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
DataArchive::queryVariables( vector<string>                         & names,
                             vector<const Uintah::TypeDescription*> & types )
{
  double start = Time::currentSeconds();
  d_lock.lock();

  rewind( d_indexFile ); // Start at beginning of file.
  bool found = ProblemSpec::findBlock( "<variables>", d_indexFile );

  if( !found ) {
    throw InternalError("DataArchive::queryVariables:variables section not found\n", __FILE__, __LINE__);
  }
  queryVariables( d_indexFile, names, types );

  d_lock.unlock();
  dbg << "DataArchive::queryVariables completed in " << Time::currentSeconds()-start << " seconds\n";
}

void
DataArchive::queryGlobals( vector<string>                         & names,
                           vector<const Uintah::TypeDescription*> & types )
{
  double start = Time::currentSeconds();

  d_lock.lock();
  
  rewind( d_indexFile ); // Start looking from the top of the file. 

  bool result = ProblemSpec::findBlock( "<globals>", d_indexFile );

  if( !result ) {
    return;
  }
  queryVariables( d_indexFile, names, types, true );

  d_lock.unlock();

  dbg << "DataArchive::queryGlobals completed in " << Time::currentSeconds()-start << " seconds\n";   
}

void
DataArchive::queryVariables( FILE                                   * fp,
                             vector<string>                         & names,
                             vector<const Uintah::TypeDescription*> & types,
                             bool                                     globals /* = false */ )
{
  // Assuming that fp points to the line following "<variables>"...

  string end_block;
  if( globals ) {
    end_block = "</globals>";
  }
  else {
    end_block = "</variables>";
  }

  while( true ) {

    string line = UintahXML::getLine( d_indexFile );
    if( line == "" || line == end_block ) {
      break;
    }
    else if( line.compare( 0, 10, "<variable " ) == 0 ) {
          
      ProblemSpec ts_doc( line );

      map<string,string> attributes;
      ts_doc.getAttributes( attributes );
      string the_type = attributes[ "type" ];
      if( the_type == "" ) {
        throw InternalError("DataArchive::queryVariables() - 'type' not found", __FILE__, __LINE__);
      }

      const TypeDescription* td = TypeDescription::lookupType( the_type );

      if( !td ){
        static TypeDescription* unknown_type = 0;
        if( !unknown_type ) {
          unknown_type = scinew TypeDescription( TypeDescription::Unknown, "-- unknown type --", false, MPI_Datatype(-1) );
        }
        td = unknown_type;
      }

      types.push_back( td );
      string name = attributes[ "name" ];
      if(name == "") {
        throw InternalError( "DataArchive::queryVariables() - 'name' not found", __FILE__, __LINE__ );
      }
      names.push_back( name );
    }
    else {
      throw InternalError( "DataArchive::queryVariables() - bad data in variables block.", __FILE__, __LINE__ );
    }
  }
}

void
DataArchive::query(       Variable     & var,
                    const string       & name,
                    const int            matlIndex, 
                    const Patch        * patch,
                    const int            index,
                          DataFileInfo * dfi /* = 0 */ )
{
  double tstart = Time::currentSeconds();

#if !defined( DISABLE_SCI_MALLOC )
  const char* tag = AllocatorSetDefaultTag("QUERY");
#endif

  d_lock.lock();
  TimeData& timedata = getTimeData(index);
  d_lock.unlock();

  ASSERT(timedata.d_initialized);
  // make sure info for this patch gets parsed from p*****.xml.
  d_lock.lock();  
  timedata.parsePatch(patch);
  d_lock.unlock();  

  VarData & varinfo = timedata.d_varInfo[name];
  string    data_filename;
  int       patchid;

  if( Parallel::getMPIRank() == 1000 ) {
    cout << "patch is " << patch << "\n";
  }

  if (patch) {
    // we need to use the real_patch (in case of periodic boundaries) to get the data, but we need the
    // passed in patch to allocate the patch to the proper virtual region... (see var.allocate below)
    const Patch* real_patch = patch->getRealPatch();
    PatchData& patchinfo = timedata.d_patchInfo[real_patch->getLevel()->getIndex()][real_patch->getLevelIndex()];
    ASSERT(patchinfo.parsed);
    patchid = real_patch->getID();

    ostringstream ostr;
    // append l#/datafilename to the directory
    ostr << timedata.d_ts_directory << "l" << patch->getLevel()->getIndex() << "/" << patchinfo.datafilename;
    data_filename = ostr.str();
  }
  else {
    // reference reduction file 'global.data' will a null patch
    patchid = -1;
    data_filename = timedata.d_ts_directory + timedata.d_globaldata;
  }

  // On a call from restartInitialize, we already have the information from the dfi,
  // otherwise get it from the hash table info.
  DataFileInfo datafileinfo;
  if( !dfi ) {
    // If this is a virtual patch, grab the real patch, but only do that here - in the next query, we want
    // the data to be returned in the virtual coordinate space.
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
    if(dfi->numParticles == -1) {
      throw InternalError( "DataArchive::query:Cannot get numParticles", __FILE__, __LINE__ );
    }
    if (patch->isVirtual()) {
      throw InternalError("DataArchive::query: Particle query on virtual patches "
                          "not finished.  We need to adjust the particle positions to virtual space...", __FILE__, __LINE__);
    }
    psetDBType::key_type   key( matlIndex, patch );
    ParticleSubset       * psubset  = 0;
    psetDBType::iterator   psetIter = d_psetDB.find(key);
    if(psetIter != d_psetDB.end()) {
      psubset = (*psetIter).second.get_rep();
    }
    if (psubset == 0 || (int)psubset->numParticles() != dfi->numParticles)
    {
      psubset = scinew ParticleSubset(dfi->numParticles, matlIndex, patch);
      //      cout << "numParticles: " << dfi->numParticles << "\n";
      //      cout << "d_pset size: " << d_psetDB.size() << "\n";
      //      cout << "1. key is: " << key.first << "\n";
      //      cout << "2. key is: " << key.second << "\n";
      d_psetDB[key] = psubset;
    }
    (static_cast<ParticleVariableBase*>(&var))->allocate(psubset);
//      (dynamic_cast<ParticleVariableBase*>(&var))->allocate(psubset);
  }
  else if (td->getType() != TypeDescription::ReductionVariable) {
    var.allocate( patch, varinfo.boundaryLayer );
  }
  
  int fd = open( data_filename.c_str(), O_RDONLY );
  if(fd == -1) {
    cerr << "Error opening file: " << data_filename.c_str() << ", errno=" << errno << '\n';
    throw ErrnoException("DataArchive::query (open call)", errno, __FILE__, __LINE__);
  }
  off_t ls = lseek( fd, dfi->start, SEEK_SET );

  if( ls == -1 ) {
    cerr << "Error lseek - file: " << data_filename.c_str() << ", errno=" << errno << '\n';
    throw ErrnoException("DataArchive::query (lseek call)", errno, __FILE__, __LINE__);
  }
  InputContext ic( fd, data_filename.c_str(), dfi->start );
  double       starttime = Time::currentSeconds();

  if( Parallel::getMPIRank() == 1000 ) {
    if( patch ) { cout << "reading in data for patch: " << patch->getID() << "\n"; }
    else { cout << "no patch at this point\n"; }
    cout << "data file is : " << data_filename << "\n";
  }

  var.read( ic, dfi->end, timedata.d_swapBytes, timedata.d_nBytes, varinfo.compression );

  dbg << "DataArchive::query: time to read raw data: "<<Time::currentSeconds() - starttime<<endl;
  ASSERTEQ( dfi->end, ic.cur );
  int result = close( fd );
  if( result == -1 ) {
    cerr << "Error closing file: " << data_filename.c_str() << ", errno=" << errno << '\n';
    throw ErrnoException("DataArchive::query (close call)", errno, __FILE__, __LINE__);
  }

#if !defined( DISABLE_SCI_MALLOC )
  AllocatorSetDefaultTag(tag);
#endif
  dbg << "DataArchive::query() completed in " << Time::currentSeconds()-tstart << " seconds\n";
}

void
DataArchive::query(       Variable       & var,
                    const string         & name,
                    const int              matlIndex, 
                    const Patch          * patch,
                    const int              timeIndex,
                    const Ghost::GhostType gt,
                    const int              ngc )
{
  if( ngc == 0 ) {
    query( var, name, matlIndex, patch, timeIndex, 0 );
  }
  else {
    d_lock.lock();
    TimeData & td = getTimeData( timeIndex );
    d_lock.unlock();
    td.parsePatch(patch); // make sure vars is actually populated
    if (td.d_varInfo.find(name) != td.d_varInfo.end()) {
      VarData& varinfo = td.d_varInfo[name];
      const TypeDescription* type = TypeDescription::lookupType(varinfo.type);
      IntVector low, high;
      patch->computeVariableExtents(type->getType(), varinfo.boundaryLayer, gt, ngc, low, high);
      queryRegion(var, name, matlIndex, patch->getLevel(), timeIndex, low, high);
    }
    else {
      cerr << "VARIABLE NOT FOUND: " << name << ", material index " << matlIndex << ", patch " << patch->getID() << ", time index " 
           << timeIndex << "\nPlease make sure the correct material index is specified\n";
      throw InternalError("DataArchive::query:Variable not found",
                          __FILE__, __LINE__);
    }
  }
}

void
DataArchive::queryRegion(       Variable  & var,
                          const string    & name,
                          const int         matlIndex, 
                          const Level     * level,
                          const int         timeIndex,
                          const IntVector & low,
                          const IntVector & high)
{
  // NOTE - this is not going to do error checking like making sure the entire volume is filled.  
  //        We'll assume that if there were bad regions, they would have been caught in the simulation.
  GridVariableBase* gridvar = dynamic_cast<GridVariableBase*>(&var);
  ASSERT(gridvar);
  gridvar->allocate( low, high );

  d_lock.lock();
  TimeData & td = getTimeData( timeIndex );
  d_lock.unlock();
  const TypeDescription* type = 0;
  Patch::VariableBasis basis = Patch::NodeBased; // not sure if this is a reasonable default...
  Patch::selectType patches;
  
  level->selectPatches( low, high, patches );
  for(int i=0;i<patches.size();i++){
    const Patch* patch = patches[i];
    
    if (type == 0) {
      td.parsePatch(patch); // make sure varInfo is loaded
      VarData& varinfo = td.d_varInfo[name];
      type = TypeDescription::lookupType(varinfo.type);
      basis = Patch::translateTypeToBasis(type->getType(), false);
    }
    IntVector l, h;

    l = Max( patch->getExtraLowIndex( basis, IntVector(0, 0, 0)), low );
    h = Min( patch->getExtraHighIndex(basis, IntVector(0, 0, 0)), high );
    if( l.x() >= h.x() || l.y() >= h.y() || l.z() >= h.z() ) {
      continue;
    }
    GridVariableBase* tmpVar = gridvar->cloneType();
    query( *tmpVar, name, matlIndex, patch, timeIndex );

    if (patch->isVirtual()) {
      // if patch is virtual, it is probable a boundary layer/extra cell that has been requested (from AMR)
      // let Bryan know if this doesn't work.  We need to adjust the source but not the dest by the virtual offset
      tmpVar->offset(patch->getVirtualOffset());
    }
    try {
      gridvar->copyPatch(tmpVar, l, h);
    } catch (InternalError& e) {
      cout << " Bad range: " << low << " " << high << ", patch intersection: " << l << " " << h 
           << " actual patch " << patch->getLowIndex(basis) << " " << patch->getHighIndex(basis) 
           << " var range: "  << tmpVar->getLow() << " " << tmpVar->getHigh() << endl;
      throw e;
    }
    delete tmpVar;
  }
}


void 
DataArchive::findPatchAndIndex( const GridP            grid,
                                      Patch         *& patch,
                                      particleIndex  & idx,
                                const long64           particleID,
                                const int              matlIndex,
                                const int              levelIndex,
                                const int              index)
{
  Patch *local = patch;
  if( patch != NULL ){
    ParticleVariable<long64> var;
    query( var, "p.particleID", matlIndex, patch, index );
    //  cerr<<"var["<<idx<<"] = "<<var[idx]<<endl;
    if( idx < (int)var.getParticleSubset()->numParticles() && var[idx] == particleID ) {
      return;
    }
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
      query( var, "p.particleID", matlIndex, *iter, index );
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
DataArchive::restartInitialize( const int             index,
                                const GridP         & grid,
                                      DataWarehouse * dw,
                                      LoadBalancer  * lb,
                                      double        * pTime )
{
  d_lb = lb;

  vector<int>    indices;
  vector<double> times;
  queryTimesteps( indices, times );

  vector<string>                   names;
  vector< const TypeDescription *> typeDescriptions;
  queryVariables( names, typeDescriptions );
  queryGlobals(   names, typeDescriptions );  
  
  map<string, VarLabel*> varMap;

  for (unsigned i = 0; i < names.size(); i++) {
    VarLabel * vl = VarLabel::find(names[i]);
    if( vl == NULL ) {
      // proc0cout << "Warning, VarLabel for " << names[i] << " was not found... attempting to create.\n"
      //           << "However, it is possible that this may cause problems down the road...\n";
      // ***** THIS ASSUMES A SINGLE GHOST CELL ***** BE CAREFUL ********
      // check if we have extracells specified. This affects Wasatch only and should have no impact on other components.
      // const bool hasExtraCells = (grid->getPatchByID(0,0)->getExtraCells() != SCIRun::IntVector(0,0,0));
      // if extracells are specified, then create varlabels that are consistent with Wasatch varlabels.
      vl = VarLabel::create( names[i], typeDescriptions[i], IntVector(0,0,0) );

      // At the end of this routine, we will need to delete the VarLabels that we create here in
      // order to avoid a memory leak.
      d_createdVarLabels[names[i]] = vl;
    }
    varMap[names[i]] = vl;
  }

  TimeData& timedata = getTimeData( index );

  *pTime = times[ index ];

  if( lb ) {
    lb->restartInitialize( this, index, timedata.d_ts_path_and_filename, grid );
  }

  // set here instead of the SimCont because we need the DW ID to be set 
  // before saving particle subsets 
  dw->setID( indices[index] );
  
  // Make sure to load all the data so we can iterate through it.
  for( unsigned int l = 0; l < grid->numLevels(); l++ ) {
    LevelP level = grid->getLevel( l );
    for( int p = 0; p < level->numPatches(); p++ ) {
      const Patch* patch = level->getPatch( p );
      if( lb->getPatchwiseProcessorAssignment( patch ) == d_processor ) {

        timedata.parsePatch( patch );
      }
    }
  }

  // Iterate through all entries in the VarData hash table, and load the 
  // variables if that data belongs on this processor.

  for( VarHashMapIterator iter( &timedata.d_datafileInfo ); iter.ok(); ++iter ) {
    VarnameMatlPatch & key  = iter.get_key();
    DataFileInfo     & data = iter.get_data();

    // get the Patch from the Patch ID (ID of -1 = NULL - for reduction vars)
    const Patch* patch = key.patchid_ == -1 ? NULL : grid->getPatchByID( key.patchid_, 0 );
    int matl = key.matlIndex_;

    VarLabel* label = varMap[key.name_];
    if (label == 0) {
      throw UnknownVariable( key.name_, dw->getID(), patch, matl,
                             "on DataArchive::scheduleRestartInitialize",
                             __FILE__, __LINE__ );
    }

    if( !patch || !lb || lb->getPatchwiseProcessorAssignment( patch ) == d_processor ) {

      Variable * var = label->typeDescription()->createInstance();

      query( *var, key.name_, matl, patch, index, &data );

      ParticleVariableBase* particles;
      if ((particles = dynamic_cast<ParticleVariableBase*>(var))) {
        if (!dw->haveParticleSubset(matl, patch)) {
          dw->saveParticleSubset(particles->getParticleSubset(), matl, patch);
        }
        else {
          ASSERTEQ(dw->getParticleSubset(matl, patch), particles->getParticleSubset());
        }
      }
      dw->put( var, label, matl, patch ); 
      delete var; // should have been cloned when it was put
    }
  }
} // end restartInitialize()


//______________________________________________________________________
//  This method is a specialization of restartInitialize().
//  It's only used by the reduceUda component
void
DataArchive::reduceUda_ReadUda( const ProcessorGroup * pg,
                                const int              timeIndex, 
                                const GridP          & grid,
                                const PatchSubset    * patches,
                                      DataWarehouse  * dw,
                                      LoadBalancer   * lb )
{
  vector<int>    timesteps;
  vector<double> times;
  vector<string> names;
  vector< const TypeDescription *> typeDescriptions;
  
  queryTimesteps(timesteps, times);
  queryVariables(names, typeDescriptions);
  queryGlobals(  names, typeDescriptions);  
  
  // create varLabels if they don't already exist
  map<string, VarLabel*> varMap;
  for (unsigned i = 0; i < names.size(); i++) {
    VarLabel * vl = VarLabel::find(names[i]);
    
    if( vl == NULL ) {
      vl = VarLabel::create( names[i], typeDescriptions[i], IntVector(0,0,0) );
      d_createdVarLabels[names[i]] = vl;
    }
    
    varMap[names[i]] = vl;
  }

  TimeData& timedata = getTimeData( timeIndex );

  // set here instead of the SimCont because we need the DW ID to be set 
  // before saving particle subsets
  dw->setID( timesteps[timeIndex] );
  
  // make sure to load all the data so we can iterate through it 
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    timedata.parsePatch( patch );
  }

  //__________________________________
  // Iterate through all entries in the VarData hash table, and load the variables.
  
  // VarHashMapIterator iter(&timedata.d_datafileInfo);
  // iter.first(); // What is this?  Delete it?  Fix me!
  
  for( VarHashMapIterator iter(&timedata.d_datafileInfo); iter.ok(); ++iter) {
    VarnameMatlPatch& key = iter.get_key();
    DataFileInfo& data    = iter.get_data();

    // get the Patch from the Patch ID (ID of -1 = NULL - for reduction vars)
    const Patch* patch = key.patchid_ == -1 ? NULL : grid->getPatchByID(key.patchid_, 0);
    int matl = key.matlIndex_;

    VarLabel* label = varMap[ key.name_ ];

    if (label == 0) {
      continue;
    }
    
    // If this proc does not own this patch
    // then ignore the variable 
    int proc = lb->getPatchwiseProcessorAssignment(patch);
    if ( proc != pg->myrank() ) {
      continue;
    }

    // Put the data in the DataWarehouse.
    Variable* var = label->typeDescription()->createInstance();
    query( *var, key.name_, matl, patch, timeIndex, &data );

    ParticleVariableBase* particles;
    if ( (particles = dynamic_cast<ParticleVariableBase*>(var)) ) {
      if ( !dw->haveParticleSubset(matl, patch) ) {
        dw->saveParticleSubset(particles->getParticleSubset(), matl, patch);
      } else {
        ASSERTEQ( dw->getParticleSubset(matl, patch), particles->getParticleSubset() );
      }
    }

    dw->put( var, label, matl, patch );
    delete var;
  }

} // end reduceUda_ReadUda()

//______________________________________________________________________
//
bool
DataArchive::queryRestartTimestep( int & timestep )
{
  // FYI: There was a bug that caused the <restarts> in UDAs to look like this:
  //
  //    <restarts/>
  //    <restart from="advect.uda.001" timestep="22"/>
  // 
  // I fixed this recently and now they look like this.
  //    <restarts>
  //       <restart from="advect.uda.001" timestep="22"/>
  //    </restarts>
  //
  // However, to handle both cases, instead of looking for the <restarts> block
  // I just search the entire file for "<restart ...".  This is fine because we
  // create and define this file, so there would never be another "<restart ...>"
  // anywhere else... I hope.

  ProblemSpec * restart_ps = NULL;

  rewind( d_indexFile ); // Start parsing from top of file.
  while( true ) {

    string line = UintahXML::getLine( d_indexFile );
    if( line == "" ) {
      break;
    }
    else if( line.compare( 0, 9, "<restart " ) == 0 ) {
      
      if( restart_ps ) {
        delete restart_ps;
      }
      restart_ps = scinew ProblemSpec( line );
    }
  }

  if( restart_ps != NULL ) {

    // Found (the last) "<restart " node.

    map<string,string> attributes;
    restart_ps->getAttributes( attributes );
    string ts_num_str = attributes[ "timestep" ];
    if( ts_num_str == "" ) {
      throw InternalError("DataArchive::queryVariables() - 'timestep' not found", __FILE__, __LINE__);
    }
    timestep = atoi( ts_num_str.c_str() );
    delete restart_ps;
    return true;
  }
  else {
    return false;
  }

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
DataArchive::setTimestepCacheSize( int new_size ) {
  d_lock.lock();
  // Now we need to reduce the size
  int current_size = (int)d_lastNtimesteps.size();
  dbg << "current_size = "<<current_size<<"\n";
  if (timestep_cache_size >= current_size) {
    // everything's fine
    d_lock.unlock();
    return;
  }

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

DataArchive::TimeData::TimeData( DataArchive * da, const string & timestepPathAndFilename ) :
  d_initialized( false ), d_ts_path_and_filename( timestepPathAndFilename ), d_parent_da( da )
{
  d_ts_directory = timestepPathAndFilename.substr( 0, timestepPathAndFilename.find_last_of('/') + 1 );
}

DataArchive::TimeData::~TimeData()
{
  purgeCache();
}

void
DataArchive::TimeData::init()
{
  d_initialized = true;

  // Pull the list of data xml files from the timestep.xml file.

  FILE * ts_file = fopen( d_ts_path_and_filename.c_str(), "r" );

  if( ts_file == NULL ) {
    // FIXME: add more info to exception.
    throw ProblemSetupException( "Failed to open timestep file.", __FILE__, __LINE__ );    
  }

  // Handle endianness and number of bits
  string endianness = d_parent_da->d_globalEndianness;
  int    numbits    = d_parent_da->d_globalNumBits;

  DataArchive::queryEndiannessAndBits( ts_file, endianness, numbits );

  if (endianness == "" || numbits == -1 ) {
    // This will only happen on a very old UDA.
    throw ProblemSetupException( "endianness and/or numbits missing", __FILE__, __LINE__ );
  }

  d_swapBytes = endianness != string(SCIRun::endianness());
  d_nBytes    = numbits / 8;

  bool found = ProblemSpec::findBlock( "<Data>", ts_file );

  if( !found ) {
    throw InternalError( "Cannot find <Data> in timestep file", __FILE__, __LINE__ );
  }

  bool done = false;
  while( !done ) {

    string line = UintahXML::getLine( ts_file );
    if( line == "" || line == "</Data>" ) {
      done = true;
    }
    else if( line.compare( 0, 10, "<Datafile " ) == 0 ) {

      ProblemSpec ts_doc( line );

      map<string,string> attributes;
      ts_doc.getAttributes( attributes );
      string datafile = attributes[ "href" ];
      if( datafile == "" ) {
        throw InternalError("DataArchive::TimeData::init() - 'href' not found", __FILE__, __LINE__);
      }
      string proc = attributes["proc"];

      // WARNING: QWERTY: READ THIS Dav... 

      /* - Remove this check for restarts.  We need to accurately
         determine which patch goes on which proc, and for the moment
         we need to be able to parse all pxxxx.xml files.  --BJW  
         if (proc != "") {
             int procnum = atoi(proc.c_str());
             if ((procnum % numProcessors) != processor) {
                  continue;
             }
          }
      */
      if( datafile == "global.xml" ) {
        // Assuming that global.xml will always be small and thus using normal
        // xml lib parsing...
        parseFile( d_ts_directory + datafile, -1, -1 );
      }
      else {

        // Get the level info out of the xml file: should be lX/pxxxxx.xml.
        unsigned level = 0;
        string::size_type start = datafile.find_first_of("l",0, datafile.length()-3);
        string::size_type end = datafile.find_first_of("/");
        if (start != string::npos && end != string::npos && end > start && end-start <= 2) {
          level = atoi(datafile.substr(start+1, end-start).c_str());
        }
        
        if( level >= d_xmlFilenames.size() ) {
          d_xmlFilenames.resize( level +1 );
          d_xmlParsed.resize(    level + 1 );
        }

        string filename = d_ts_directory + datafile;
        d_xmlFilenames[ level ].push_back( filename );
        d_xmlParsed[    level ].push_back( false );
      }
    }
    else {
      throw InternalError("DataArchive::TimeData::init() - bad line in <Data> block...", __FILE__, __LINE__);
    }
  } // end while()

  fclose( ts_file );
}

void
DataArchive::TimeData::purgeCache()
{
  d_grid  = 0;

  d_datafileInfo.remove_all();
  d_patchInfo.clear(); 
  d_varInfo.clear();
  d_xmlFilenames.clear();
  d_xmlParsed.clear();
  d_initialized = false;
}

// This is the function that parses the p*****.xml file for a single processor.
void
DataArchive::TimeData::parseFile( const string & filename, int levelNum, int basePatch )
{
  // Parse the file.
  ProblemSpecP top = ProblemSpecReader().readInputFile( filename );
  
  // Materials are the same for all patches on a level - only parse them from one file.
  bool addMaterials = levelNum >= 0 && d_matlInfo[levelNum].size() == 0;

  for( ProblemSpecP vnode = top->getFirstChild(); vnode != 0; vnode=vnode->getNextSibling() ){
    if(vnode->getNodeName() == "Variable") {
      string varname;
      if( !vnode->get("variable", varname) ) {
        throw InternalError("Cannot get variable name", __FILE__, __LINE__);
      }
      
      int patchid;
      if(!vnode->get("patch", patchid) && !vnode->get("region", patchid)) {
        throw InternalError("Cannot get patch id", __FILE__, __LINE__);
      }
      
      int index;
      if(!vnode->get("index", index)) {
        throw InternalError("Cannot get index", __FILE__, __LINE__);
      }
      
      if (addMaterials) {
        // set the material to existing.  index+1 to use matl -1
        if (index+1 >= (int)d_matlInfo[levelNum].size()) {
          d_matlInfo[levelNum].resize(index+2);
        }
        d_matlInfo[levelNum][index] = true;
      }

      map<string,string> attributes;
      vnode->getAttributes(attributes);

      string type = attributes["type"];
      if( type == "" ) {
        throw InternalError("DataArchive::query:Variable doesn't have a type", __FILE__, __LINE__);
      }
      long start;
      if( !vnode->get("start", start) ) {
        throw InternalError("DataArchive::query:Cannot get start", __FILE__, __LINE__);
      }
      long end;
      if( !vnode->get("end", end) ) {
        throw InternalError("DataArchive::query:Cannot get end", __FILE__, __LINE__);
      }
      string filename;  
      if( !vnode->get("filename", filename) ) {
        throw InternalError("DataArchive::query:Cannot get filename", __FILE__, __LINE__);
      }

      // Not required
      string    compressionMode = "";  
      IntVector boundary(0,0,0);
      int       numParticles = -1;

      vnode->get( "compression", compressionMode );
      vnode->get( "boundaryLayer", boundary );
      vnode->get( "numParticles", numParticles );

      if( d_varInfo.find(varname) == d_varInfo.end() ) {
        VarData& varinfo = d_varInfo[varname];
        varinfo.type = type;
        varinfo.compression = compressionMode;
        varinfo.boundaryLayer = boundary;
      }
      else if (compressionMode != "") {
        // For particles variables of size 0, the uda doesn't say it
        // has a compressionMode...  (FYI, why is this?  Because it is
        // ambiguous... if there is no data, is it compressed?)
        //
        // To the best of my understanding, we only look at the variables stats
        // the first time we encounter it... even if there are multiple materials.
        // So we run into a problem is the variable has 0 data the first time it
        // is looked at... The problem there is that it doesn't mark it as being
        // compressed, and therefore the next time we see that variable (eg, in
        // another material) we (used to) assume it was not compressed... the 
        // following lines compenstate for this problem:
        VarData& varinfo = d_varInfo[varname];
        varinfo.compression = compressionMode;
      }

      if (levelNum == -1) { // global file (reduction vars)
        d_globaldata = filename;
      }
      else {
        ASSERTRANGE(patchid-basePatch, 0, (int)d_patchInfo[levelNum].size());

        PatchData& patchinfo = d_patchInfo[levelNum][patchid-basePatch];
        if (!patchinfo.parsed) {
          patchinfo.parsed = true;
          patchinfo.datafilename = filename;
        }
      }
      VarnameMatlPatch vmp(varname, index, patchid);
      DataFileInfo     dummy;

      if (d_datafileInfo.lookup(vmp, dummy) == 1) {
        //cerr << "Duplicate variable name: " << name << endl;
      }
      else {
        DataFileInfo dfi(start, end, numParticles);
        d_datafileInfo.insert(vmp, dfi);
      }
    }
    else if( vnode->getNodeType() != ProblemSpec::TEXT_NODE ) {
      cerr << "WARNING: Unknown element in Variables section: " << vnode->getNodeName() << '\n';
    }
  }
} // end TimeData::parseFile()

void
DataArchive::TimeData::parsePatch( const Patch * patch )
{
  ASSERT(d_grid != 0);
  if( !patch ) { 
    proc0cout << "parsePatch called with null patch....\n";
    return;
  }

  const Patch * real_patch = patch->getRealPatch();

  // Make sure the data for this patch has been processed.  Return straightaway if we have already parsed this patch.
  //
  int levelIndex       = real_patch->getLevel()->getIndex(); 
  int levelBasePatchID = real_patch->getLevel()->getPatch(0)->getID();
  int patchIndex       = real_patch->getLevelIndex();

  PatchData & patchinfo = d_patchInfo[levelIndex][patchIndex];
  if( patchinfo.parsed ) {
    return;
  }

  // If this is a newer uda, the patch info in the grid will store the processor where the data is.
  if( patchinfo.proc != -1 ) {
    ostringstream file;
    file << d_ts_directory << "l" << (int) real_patch->getLevel()->getIndex() << "/p" << setw(5) << setfill('0') << (int) patchinfo.proc << ".xml";
    parseFile( file.str(), levelIndex, levelBasePatchID );
  }

  // Try making a guess as to the processor.  First go is to try the processor of the same index as the patch.  Many datasets
  // have only one patch per processor, so this is a reasonable first attempt.  Future attemps could perhaps be smarter.

  if (!patchinfo.parsed && patchIndex < (int)d_xmlParsed[levelIndex].size() && !d_xmlParsed[levelIndex][patchIndex]) {
    parseFile( d_xmlFilenames[levelIndex][patchIndex], levelIndex, levelBasePatchID );
    d_xmlParsed[levelIndex][patchIndex] = true;
  }

  // failed the guess - parse the entire dataset for this level
  if (!patchinfo.parsed) {
    for (unsigned proc = 0; proc < d_xmlFilenames[levelIndex].size(); proc++) {
      parseFile(d_xmlFilenames[levelIndex][proc], levelIndex, levelBasePatchID);
      d_xmlParsed[levelIndex][proc] = true;
    }
  }
}

// Parses the timestep xml file for <oldDelt>
//
double
DataArchive::getOldDelt( int restart_index )
{
  TimeData& timedata = getTimeData( restart_index );
  FILE * fp = fopen( timedata.d_ts_path_and_filename.c_str(), "r" );
  if( fp == NULL ) {
    throw InternalError("DataArchive::setOldDelt() failed open datafile.", __FILE__, __LINE__);
  }
  // Note, old UDAs had a <delt> flag, but that was deprecated long ago in favor of the <oldDelt>
  // flag which is what we are going to look for here.

  while( true ) {

    string line = UintahXML::getLine( fp );

    if( line == "" ) {
      fclose( fp );
      throw InternalError("DataArchive::setOldDelt() failed to find <oldDelt>.", __FILE__, __LINE__);
    }
    else if( line.compare( 0, 9, "<oldDelt>" ) == 0 ) {
      vector<string> pieces = UintahXML::splitXMLtag( line );

      fclose( fp );
      return atof( pieces[1].c_str() );
    }
  }

}

// Parses the timestep xml file and skips the <Meta>, <Grid>, and <Data> sections, returning 
// everything else.  This function assumes that the timestep.xml file was created by us and
// is in the correct order - in other words, anything after </Data> is component related,
// and everything before it can be removed.
//
ProblemSpecP
DataArchive::getTimestepDocForComponent( int restart_index )
{
  TimeData& timedata = getTimeData( restart_index );
  FILE * fp = fopen( timedata.d_ts_path_and_filename.c_str(), "r" );

  if( fp == NULL ) {
    throw InternalError("DataArchive::getTimespecDocForComponent() failed open datafile.", __FILE__, __LINE__);
  }

  bool found = ProblemSpec::findBlock( "</Data>", fp );

  if( !found ) {
    throw InternalError("DataArchive::getTimespecDocForComponent() failed to find </Data>.", __FILE__, __LINE__);
  }

  string buffer = "<Uintah_timestep>";

  while( true ) {

    string line = UintahXML::getLine( fp );
    
    buffer.append( line );

    if( line == "</Uintah_timestep>" ) {
      break;
    }
  }

  fclose( fp );

  ProblemSpec * result = new ProblemSpec( buffer );

  return result;
}

ConsecutiveRangeSet
DataArchive::queryMaterials( const string & varname,
                             const Patch  * patch,
                                   int      index )
{
  double start = Time::currentSeconds();
  d_lock.lock();

  TimeData& timedata = getTimeData( index );
  timedata.parsePatch(patch);

  ConsecutiveRangeSet matls;

  for (unsigned i = 0; i < timedata.d_matlInfo[patch->getLevel()->getIndex()].size(); i++) {
    // i-1, since the matlInfo is adjusted to allow -1 as entries
    VarnameMatlPatch vmp(varname, i-1, patch->getRealPatch()->getID());
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

  TimeData& timedata = getTimeData( index );

  timedata.parsePatch(patch);

  int numMatls = -1;

  for (unsigned i = 0; i < timedata.d_matlInfo[patch->getLevel()->getIndex()].size(); i++) {
    if (timedata.d_matlInfo[patch->getLevel()->getIndex()][i]) {
      numMatls++;
    }
  }

  dbg << "DataArchive::queryNumMaterials completed in " << Time::currentSeconds()-start << " seconds\n";

  d_lock.unlock();
  return numMatls;
}


//______________________________________________________________________
//    Does this variable exist on this patch at this timestep
bool
DataArchive::exists( const string& varname,
                     const Patch* patch,
                     const int timeStep )
{
  d_lock.lock();

  TimeData& timedata = getTimeData(timeStep);
  timedata.parsePatch(patch);

  int levelIndex = patch->getLevel()->getIndex();
  
  for (unsigned i = 0; i < timedata.d_matlInfo[levelIndex].size(); i++) {
    // i-1, since the matlInfo is adjusted to allow -1 as entries
    VarnameMatlPatch vmp( varname, i-1, patch->getRealPatch()->getID() );
    DataFileInfo dummy;

    if (timedata.d_datafileInfo.lookup(vmp, dummy) == 1){
      d_lock.unlock();
      return true;
    }
  }
  d_lock.unlock();

  return false;
}
