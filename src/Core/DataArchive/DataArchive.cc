/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

#include <CCA/Components/ProblemSpecification/ProblemSpecReader.h>
#include <CCA/Ports/InputContext.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <CCA/Ports/DataWarehouse.h>

#if HAVE_PIDX
#  include <CCA/Ports/PIDXOutputContext.h>
#endif

#include <Core/Containers/OffsetArray1.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/UnknownVariable.h>
#include <Core/Grid/Variables/StaticInstantiate.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Math/MiscMath.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Util/Assert.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/StringUtil.h>
#include <Core/Util/XMLUtils.h>

#include <libxml/xmlreader.h>

#include <iostream>
#include <sstream>

#include <iomanip>
#include <fstream>
#include <fcntl.h>

#include <sys/param.h>
#include <unistd.h>

using namespace std;
using namespace Uintah;

//______________________________________________________________________
// Initialize class static variables:

DebugStream DataArchive::dbg( "DataArchive", false );
bool        DataArchive::d_types_initialized = false;

//______________________________________________________________________
//

DataArchive::DataArchive( const string & filebase,
                          const int      processor     /* = 0 */,
                          const int      numProcessors /* = 1 */,
                          const bool     verbose       /* = true */ ) :
  timestep_cache_size(10),
  default_cache_size(10),
  d_filebase(filebase),
  d_cell_scale( Vector(1.0,1.0,1.0) ),
  d_processor(processor),
  d_numProcessors(numProcessors),
  d_particlePositionName("p.x")
{
#ifdef STATIC_BUILD
  if( !d_types_initialized ) {
    d_types_initialized = true;
    // For static builds, sometimes the Uintah types (CCVariable, etc) do not get automatically
    // registered to the Uintah type system... this call forces that to happen.
    proc0cout << "Loading Uintah var types into type system (static build).\n";
    instantiateVariableTypes();
  }
#endif  

  if( d_filebase == "" ) {
    throw InternalError( "DataArchive::DataArchive 'filebase' cannot be empty (\"\").", __FILE__, __LINE__ );
  }

  while( d_filebase[ d_filebase.length() - 1 ] == '/' ) {
    // Remove '/' from the end of the filebase (if there is one).
    d_filebase = d_filebase.substr( 0, filebase.length() - 1 );
  }

  string index = d_filebase + "/index.xml";
  // if( verbose ) {
  //   proc0cout << "Parsing " << index << "\n";
  //}

  d_indexFile = fopen( index.c_str(), "r" ); // Was: ProblemSpecReader().readInputFile( index );
  if( d_indexFile == nullptr ) {
    throw InternalError( "DataArchive::DataArchive() failed to open index xml file.", __FILE__, __LINE__ );
  }

  d_outputFileFormat = UDA;
  d_globalEndianness = "";
  d_globalNumBits    = -1;
  queryEndiannessAndBits( d_indexFile, d_globalEndianness, d_globalNumBits );

  queryParticlePositionName( d_indexFile );

  queryOutputFormat( d_indexFile );

}
//______________________________________________________________________
//
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
//______________________________________________________________________
//
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


//______________________________________________________________________
//
void
DataArchive::queryOutputFormat( FILE * doc )
{
  rewind( doc );

  while( true ) {

    string line = UintahXML::getLine( doc );
    if( line == "" ) {
      return;
    }
    else if( line.compare( 0, 14, "<outputFormat>" ) == 0 ) {
      vector<string> pieces = UintahXML::splitXMLtag( line );

      const string format = string_tolower( pieces[1] );

      if( format == "pidx" ) {
        d_outputFileFormat = PIDX;
      }
      else {
        d_outputFileFormat = UDA;
      }
      return;
    }
  }
}


//______________________________________________________________________
//
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
//______________________________________________________________________
//
void
DataArchive::queryTimesteps( vector<int>    & index,
                             vector<double> & times )
{
  Timers::Simple timer;
  timer.start();

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
            throw InternalError( "DataArchive::queryTimesteps:timestep href not found", __FILE__, __LINE__ );
          }

          int          timestepNumber;
          double       currentTime;
	  // Usually '.../timestep.xml'
          string       ts_path_and_filename = d_filebase + "/" + tsfile;
          ProblemSpecP timestepDoc = 0;

	  string::size_type deliminator_index = tsfile.find("/");
	  string tnumber( tsfile, 0, deliminator_index );
	  // Usually '.../grid.xml'
	  string       grid_path_and_filename = d_filebase + "/" + tnumber + "/" + "grid.xml";

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

  dbg << "DataArchive::queryTimesteps completed in " << timer().seconds()
      << " seconds\n";
}
//______________________________________________________________________
//
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
//______________________________________________________________________
//
int
DataArchive::queryPatchwiseProcessor( const Patch * patch, const int index )
{
  d_lock.lock();
  TimeData & timedata = getTimeData( index );

  int proc = timedata.d_patchInfo[ patch->getLevel()->getIndex() ][ patch->getLevelIndex() ].proc;
  d_lock.unlock();
  return proc;
}
//______________________________________________________________________
//
GridP
DataArchive::queryGrid( int index, const ProblemSpecP & ups /* = nullptr */, bool assignBCs /* = true */ )
{
  Timers::Simple timer;
  timer.start();

  // The following variable along with d_cell_scale is necessary to allow the
  // UdaScale module work.  Small domains are a problem for the SCIRun widgets
  // so UdaScale allows the user increase the domain by setting the
  // d_cell_scale. The next call to this function will use the new scaling.
  // This can be removed if SCIRun is no longer used for visualization.
  static Vector old_cell_scale(1.0,1.0,1.0);

  d_lock.lock();

  TimeData & timedata = getTimeData( index );

  //  FILE* fp = 0;

  // Based on the timestep path and file name (eg: .../timestep.xml), we need
  // to cut off the associated path so that we can find the path to grid.xml.
  string::size_type path_length = timedata.d_ts_path_and_filename.rfind( "/" ) + 1;
  string path( timedata.d_ts_path_and_filename, 0, path_length );
  string grid_filename = path + "grid.xml";

  FILE * fp_grid = fopen( grid_filename.c_str(), "r" );

  // Check if the grid.xml is present, and use that, if it isn't, then use the grid information
  // that is stored in timestep.xml.

  bool grid_xml_is_binary = false;
  if ( fp_grid == nullptr ) {
    // Could not open grid.xml, just go with timestep.xml.
    fp_grid = fopen( timedata.d_ts_path_and_filename.c_str(), "r" );
  }
  else {
    // Determine if the grid is written in ASCII xml, or in binary.
    unsigned int marker = -1;
    fread( &marker, sizeof( marker ), 1, fp_grid );

    if( marker == GRID_MAGIC_NUMBER ) {
      grid_xml_is_binary = true;
    }
    else {
      // FIXME: do we need to reset the file pointer here?
    }
  }

  if( fp_grid == nullptr ) {
    throw InternalError( "DataArchive::queryGrid() failed to open input file.\n", __FILE__, __LINE__ );
  }

  GridP                 grid = scinew Grid();
  vector< vector<int> > procMap; // One vector<int> per level.

  if( grid_xml_is_binary ) {
    grid->readLevelsFromFileBinary( fp_grid, procMap );
  }
  else {
    grid->readLevelsFromFile( fp_grid, procMap );
  }

  fclose( fp_grid );

  // Check to see if the grid has already been reconstructed and that
  // the cell scaling has not changed. Cell scale check can be removed
  // if SCIRun is no longer used for visualization
  if (timedata.d_grid != nullptr  &&  old_cell_scale == d_cell_scale) {
    d_lock.unlock();
    return timedata.d_grid;
  }

  // update the static variable old_cell_scale if the cell scale has changed.
  // Can be removed if SCIRun is no longer used for visualization.
  if( old_cell_scale != d_cell_scale ){
    old_cell_scale = d_cell_scale;
  }

  if( ups && assignBCs ) { // 'ups' is non-null only for restarts.

    ProblemSpecP grid_ps = ups->findBlock( "Grid" );
    grid->assignBCS( grid_ps, nullptr );
  }

  timedata.d_patchInfo.clear();
  timedata.d_matlInfo.clear();

  for( int levelIndex = 0; levelIndex < grid->numLevels(); levelIndex++ ) {

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

  dbg << "DataArchive::queryGrid completed in " << timer().seconds()
      << " seconds\n";

  return grid;

} // end queryGrid()

//______________________________________________________________________
//
void
DataArchive::queryLifetime( double& /*min*/, double& /*max*/,
                            particleId /*id*/)
{
  cerr << "DataArchive::lifetime not finished\n";
}
//______________________________________________________________________
//
void
DataArchive::queryLifetime( double& /*min*/, double& /*max*/,
                            const Patch* /*patch*/)
{
  cerr << "DataArchive::lifetime not finished\n";
}
//______________________________________________________________________
//
void
DataArchive::queryVariables( vector<string>                         & names,
                             vector<const Uintah::TypeDescription*> & types )
{
  Timers::Simple timer;
  timer.start();

  d_lock.lock();

  rewind( d_indexFile ); // Start at beginning of file.
  bool found = ProblemSpec::findBlock( "<variables>", d_indexFile );

  if( !found ) {
    throw InternalError( "DataArchive::queryVariables:variables section not found", __FILE__, __LINE__ );
  }

  queryVariables( d_indexFile, names, types );

  d_lock.unlock();

  dbg << "DataArchive::queryVariables completed in " << timer().seconds()
      << " seconds\n";
}
//______________________________________________________________________
//
void
DataArchive::queryGlobals( vector<string>                         & names,
                           vector<const Uintah::TypeDescription*> & types )
{
  Timers::Simple timer;
  timer.start();

  d_lock.lock();

  rewind( d_indexFile ); // Start looking from the top of the file.

  bool result = ProblemSpec::findBlock( "<globals>", d_indexFile );

  if( !result ) {
    return;
  }

  queryVariables( d_indexFile, names, types, true );

  d_lock.unlock();

  dbg << "DataArchive::queryGlobals completed in " << timer().seconds()
      << " seconds\n";
}
//______________________________________________________________________
//
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
        throw InternalError( "DataArchive::queryVariables() - 'type' not found", __FILE__, __LINE__ );
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
//______________________________________________________________________
//
void
DataArchive::query(       Variable     & var,
                    const string       & name,
                    const int            matlIndex,
                    const Patch        * patch,
                    const int            timeIndex,
                          DataFileInfo * dfi /* = 0 */ )
{
  Timers::Simple timer;
  timer.start();

#if !defined( DISABLE_SCI_MALLOC )
  const char* tag = AllocatorSetDefaultTag("QUERY");
#endif

  d_lock.lock();
  TimeData& timedata = getTimeData(timeIndex);
  d_lock.unlock();

  ASSERT(timedata.d_initialized);
  // make sure info for this patch gets parsed from p*****.xml.
  d_lock.lock();
  timedata.parsePatch( patch );
  d_lock.unlock();

  VarData & varinfo = timedata.d_varInfo[name];
  string    data_filename;
  int       patchid;
  int       varType = BLANK;

  if (patch) {
    varType = PATCH_VAR;
    // we need to use the real_patch (in case of periodic boundaries) to get the data, but we need the
    // passed in patch to allocate the patch to the proper virtual region... (see var.allocate below)
    const Patch* real_patch = patch->getRealPatch();
    int levelIndex          = real_patch->getLevel()->getIndex();
    int patchIndex          = real_patch->getLevelIndex();

    PatchData& patchinfo = timedata.d_patchInfo[levelIndex][patchIndex];
    ASSERT(patchinfo.parsed);
    patchid = real_patch->getID();

    ostringstream ostr;
    // append l#/datafilename to the directory
    ostr << timedata.d_ts_directory << "l" << patch->getLevel()->getIndex() << "/" << patchinfo.datafilename;
    data_filename = ostr.str();
  }
  else {
    varType = REDUCTION_VAR;
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

    vector<VarnameMatlPatch>::iterator iter = std::find( timedata.d_datafileInfoIndex.begin(), timedata.d_datafileInfoIndex.end(), VarnameMatlPatch(name, matlIndex, patchid ) );
    if( iter == timedata.d_datafileInfoIndex.end() ) { // Previously used the hashmap lookup( timedata.d_datafileInfo.lookup() )
      cerr << "VARIABLE NOT FOUND: " << name 
           << ", material index " << matlIndex 
           << ", Level " << patch->getLevel()->getIndex() 
           << ", patch " << patch->getID() 
           << ", time index " << timeIndex << "\n";

      throw InternalError("DataArchive::query:Variable not found", __FILE__, __LINE__);
    }
    
    int pos = std::distance( timedata.d_datafileInfoIndex.begin(), iter );
    dfi = &timedata.d_datafileInfoValue[ pos ];
  }

  const TypeDescription* td = var.virtualGetTypeDescription();
  ASSERT(td->getName() == varinfo.type);

  //__________________________________
  // Allocate memory for grid or particle variables
  if (td->getType() == TypeDescription::ParticleVariable) {

    if(dfi->numParticles == -1) {
      throw InternalError( "DataArchive::query:Cannot get numParticles", __FILE__, __LINE__ );
    }
    if (patch->isVirtual()) {
      throw InternalError( "DataArchive::query: Particle query on virtual patches "
                           "not finished.  We need to adjust the particle positions to virtual space...", __FILE__, __LINE__ );
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

  //__________________________________
  // open data file Standard Uda Format
  if( d_outputFileFormat == UDA || varType == REDUCTION_VAR) {
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

    // read in the variable
    InputContext ic( fd, data_filename.c_str(), dfi->start );

    Timers::Simple read_timer;
    timer.start();

    var.read( ic, dfi->end, timedata.d_swapBytes, timedata.d_nBytes, varinfo.compression );

    dbg << "DataArchive::query: time to read raw data: "
	<< read_timer().seconds() << " seconds\n";

    ASSERTEQ( dfi->end, ic.cur );

    int result = close( fd );
    if( result == -1 ) {
      cerr << "Error closing file: " << data_filename.c_str() << ", errno=" << errno << '\n';
      throw ErrnoException("DataArchive::query (close call)", errno, __FILE__, __LINE__);
    }
  }

  //__________________________________
  //  bulletproofing
  if( isPIDXEnabled() == false && d_outputFileFormat == PIDX ){
    ostringstream error;
    error << "\nERROR DataArchive::query()\n"
          << "The uda you are trying to open was written using the PIDX file format.\n"
          << "You must configure and compile with PIDX enabled.";
    throw InternalError( error.str() , __FILE__, __LINE__ );
  }


  #if HAVE_PIDX
  //__________________________________
  //   open PIDX
  //  TO DO:
  //    - do we need  calls to PIDX_get_variable_count() PIDX_get_dims()??

  if( d_outputFileFormat == PIDX && varType == PATCH_VAR ) {

    PIDXOutputContext pidx;
    const Level* level = patch->getLevel();

    //__________________________________
    // define the level extents for this variable type
    IntVector lo;
    IntVector hi;
    level->findCellIndexRange(lo,hi);
    PIDX_point level_size;
    pidx.setLevelExtents( "DataArchive::query()", lo, hi, level_size );

    //__________________________________
    // define patch extents
    PIDX_point patchOffset;
    PIDX_point patchSize;
    PIDXOutputContext::patchExtents patchExts;

    pidx.setPatchExtents( "DataArchive::query()", patch, level, varinfo.boundaryLayer,
                         td, patchExts, patchOffset, patchSize );

    if (dbg.active() && isProc0_macro ){
      patchExts.print(cout);
    }
    //__________________________________
    //  Creating access
    PIDX_access access;
    PIDX_create_access(&access);
    
    if( Parallel::usingMPI() ) {
      MPI_Comm comm = Parallel::getRootProcessorGroup()->getComm();
      PIDX_set_mpi_access(access, comm);
    }
    //__________________________________
    //  Open idx file
    ostringstream levelPath;
    levelPath << timedata.d_ts_directory << "l" << level->getIndex() << "/";  // uda/timestep/level/
    string idxFilename = (levelPath.str() + varinfo.filename );               // be careful, the patch.datafilename != varinfo.filename
                                                                              // varinfo.filename == <CCVars.idx, SFC*Vars.idx....>
    PIDX_file idxFile;      // IDX file descriptor
    PIDX_point global_size;

    int ret = PIDX_file_open(idxFilename.c_str(), PIDX_MODE_RDONLY, access, global_size, &idxFile);
    pidx.checkReturnCode( ret,"DataArchive::query() - PIDX_file_open failure", __FILE__, __LINE__);

    //__________________________________
    //  Extra Calls that _MAY_ be needed
    //PIDX_point global_size;
    //ret = PIDX_get_dims(idxFile, global_size);          // returns the levelSize  Is this needed?
    //pidx.checkReturnCode( ret,"DataArchive::query() - PIDX_get_dims failure", __FILE__, __LINE__);

    int variable_count = 0;             ///< Number of fields in PIDX file
    ret = PIDX_get_variable_count(idxFile, &variable_count);
    pidx.checkReturnCode( ret,"DataArchive::query() - PIDX_get_variable_count failure", __FILE__, __LINE__);

    //int me;
    //PIDX_get_current_time_step(idxFile, &me);
    //cout << " PIDX file has currentl timestep: " << me << endl;

    //__________________________________
    //  set locations in PIDX file for querying variable
    int timestep = d_ts_indices[timeIndex];
    ret = PIDX_set_current_time_step(idxFile, timestep);
    pidx.checkReturnCode(ret, "DataArchive::query() - PIDX_set_current_time_step failure", __FILE__, __LINE__);

    int varIndex = dfi->start;
    ret = PIDX_set_current_variable_index(idxFile, varIndex);
    pidx.checkReturnCode(ret, "DataArchive::query() - PIDX_set_current_variable_index failure", __FILE__, __LINE__);

    //__________________________________
    // read IDX file for variable desc
    PIDX_variable varDesc;
    ret = PIDX_get_current_variable(idxFile, &varDesc);
    pidx.checkReturnCode(ret, "DataArchive::query() - PIDX_get_current_variable failure", __FILE__, __LINE__);

    int values_per_sample = varDesc->vps;

    int bits_per_sample = 0;
    ret = PIDX_default_bits_per_datatype(varDesc->type_name, &bits_per_sample);
    pidx.checkReturnCode(ret, "DataArchive::query() - PIDX_default_bits_per_datatype failure", __FILE__, __LINE__);

    //__________________________________
    // Allocate memory and read in data from PIDX file  Need to use patch_buffer !!!
    unsigned char *dataPIDX;
    size_t arraySize = (bits_per_sample/8) * patchExts.totalCells_EC  * values_per_sample;
    dataPIDX = (unsigned char*)malloc( arraySize );
    memset( dataPIDX, 0, arraySize);

    // debugging
    if (dbg.active() ){
      proc0cout << "Query:  filename: " << idxFilename << "\n"
                << "    " << name
                << " timestep: " << timestep
                << " matlIndex: " <<  matlIndex
                << " patchID: "   << patchid
                << " level: "     << level->getIndex() << endl;
      proc0cout << "PIDX query: \n"
                << "    " << varDesc->var_name
                << " type_name: " << varDesc->type_name
                << " varIndex: " << varIndex
                << " values_per_sample: " << varDesc->vps
                << " bits_per_sample: "<< bits_per_sample
                << " arraySize " << arraySize << endl;
    }

    ret = PIDX_variable_read_data_layout(varDesc, patchOffset, patchSize, dataPIDX, PIDX_row_major);
    pidx.checkReturnCode(ret, "DataArchive::query() - PIDX_variable_read_data_layout failure", __FILE__, __LINE__);

    //__________________________________
    // close idx file and access
    ret = PIDX_close( idxFile );
    pidx.checkReturnCode(ret, "DataArchive::query() - PIDX_close failure", __FILE__, __LINE__);

    ret = PIDX_close_access( access );
    pidx.checkReturnCode(ret, "DataArchive::query() - PIDX_close_access failure", __FILE__, __LINE__);


    //__________________________________
    // debugging
    if (dbg.active() ){
      pidx.printBufferWrap("DataArchive::query    AFTER  close",
                           td->getSubType()->getType(),
                           varDesc->vps,
                           patchExts.lo_EC, patchExts.hi_EC,
                           dataPIDX,
                           arraySize );
    }
    //__________________________________
    // now move the dataPIDX buffer into the array3 variable
    var.readPIDX( dataPIDX,  arraySize, timedata.d_swapBytes );
    free( dataPIDX );
  }
  #endif

#if !defined( DISABLE_SCI_MALLOC )
  AllocatorSetDefaultTag(tag);
#endif

  dbg << "DataArchive::query() completed in " << timer().seconds()
      << " seconds\n";
}
//______________________________________________________________________
//
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
    td.parsePatch( patch ); // make sure vars is actually populated
    if (td.d_varInfo.find(name) != td.d_varInfo.end()) {
      VarData& varinfo = td.d_varInfo[name];
      const TypeDescription* type = TypeDescription::lookupType(varinfo.type);
      IntVector low, high;
      patch->computeVariableExtents(type->getType(), varinfo.boundaryLayer, gt, ngc, low, high);
      queryRegion(var, name, matlIndex, patch->getLevel(), timeIndex, low, high);
    }
    else {
      cerr << "VARIABLE NOT FOUND: " << name 
           << ", material index " << matlIndex 
           << ", Level " << patch->getLevel()->getIndex() 
           << ", patch " << patch->getID() 
           << ", time index " << timeIndex << "\n";
      throw InternalError( "DataArchive::query:Variable not found", __FILE__, __LINE__ );
    }
  }
}
//______________________________________________________________________
//
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
      td.parsePatch( patch ); // make sure varInfo is loaded
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
//______________________________________________________________________
//
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
  if( patch != nullptr ){
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
  patch = nullptr;
//   for (int level_nr = 0;
//        (level_nr < grid->numLevels()) && (patch == nullptr); level_nr++) {

//     const LevelP level = grid->getLevel(level_nr);
    const LevelP level = grid->getLevel(levelIndex);

    for (Level::const_patch_iterator iter = level->patchesBegin();
         (iter != level->patchesEnd()) && (patch == nullptr); iter++) {
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

      if( patch != nullptr )
        break;
    }
//  }
}
//______________________________________________________________________
//
void
DataArchive::restartInitialize( const int                index,
                                const GridP            & grid,
                                      DataWarehouse    * dw,
                                      LoadBalancerPort * lb,
                                      double           * pTime )
{
  vector<int>    ts_indices;
  vector<double> times;
  queryTimesteps( ts_indices, times );

  vector<string>                   names;
  vector< const TypeDescription *> typeDescriptions;
  queryVariables( names, typeDescriptions );
  queryGlobals(   names, typeDescriptions );

  map<string, VarLabel*> varMap;

  for (unsigned i = 0; i < names.size(); i++) {
    VarLabel * vl = VarLabel::find(names[i]);
    if( vl == nullptr ) {
      // proc0cout << "Warning, VarLabel for " << names[i] << " was not found... attempting to create.\n"
      //           << "However, it is possible that this may cause problems down the road...\n";
      // ***** THIS ASSUMES A SINGLE GHOST CELL ***** BE CAREFUL ********
      // check if we have extracells specified. This affects Wasatch only and should have no impact on other components.
      // const bool hasExtraCells = (grid->getPatchByID(0,0)->getExtraCells() != Uintah::IntVector(0,0,0));
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
  dw->setID( ts_indices[index] );

  // Make sure to load all the data so we can iterate through it.
  for( int l = 0; l < grid->numLevels(); l++ ) {
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


  //VarHashMapIterator iter( &timedata.d_datafileInfo );
  vector<VarnameMatlPatch>::iterator iter;

  for( iter = timedata.d_datafileInfoIndex.begin(); iter != timedata.d_datafileInfoIndex.end(); ++iter ) {

    int pos = std::distance( timedata.d_datafileInfoIndex.begin(), iter );
    VarnameMatlPatch & key  = *iter;
    DataFileInfo     & data = timedata.d_datafileInfoValue[ pos ];

    // Get the Patch from the Patch ID (ID of -1 = nullptr - for reduction vars)
    const Patch* patch = key.patchid_ == -1 ? nullptr : grid->getPatchByID( key.patchid_, 0 );
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
DataArchive::reduceUda_ReadUda( const ProcessorGroup   * pg,
                                const int                timeIndex,
                                const GridP            & grid,
                                const PatchSubset      * patches,
                                      DataWarehouse    * dw,
                                      LoadBalancerPort * lb )
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

    if( vl == nullptr ) {
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

  for( vector<VarnameMatlPatch>::iterator iter = timedata.d_datafileInfoIndex.begin(); iter != timedata.d_datafileInfoIndex.end(); ++iter ) {

    int                pos  = std::distance( timedata.d_datafileInfoIndex.begin(), iter );
    VarnameMatlPatch & key  = *iter;
    DataFileInfo     & data = timedata.d_datafileInfoValue[ pos ];

    // Get the Patch from the Patch ID (ID of -1 = nullptr - for reduction vars)
    const Patch* patch = key.patchid_ == -1 ? nullptr : grid->getPatchByID(key.patchid_, 0);
    int matl = key.matlIndex_;

    VarLabel* label = varMap[ key.name_ ];

    if ( label == nullptr ) {
      continue;
    }

    // If this process does not own this patch, then ignore the variable...
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

  ProblemSpec * restart_ps = nullptr;

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

  if( restart_ps != nullptr ) {

    // Found (the last) "<restart " node.

    map<string,string> attributes;
    restart_ps->getAttributes( attributes );
    string ts_num_str = attributes[ "timestep" ];
    if( ts_num_str == "" ) {
      throw InternalError( "DataArchive::queryVariables() - 'timestep' not found", __FILE__, __LINE__ );
    }
    timestep = atoi( ts_num_str.c_str() );
    delete restart_ps;
    return true;
  }
  else {
    return false;
  }
}

//______________________________________________________________________
// We want to cache at least a single timestep, so that we don't have
// to reread the timestep for every patch queried.  This sets the
// cache size to one, so that this condition is held.
void
DataArchive::turnOffXMLCaching() {
  setTimestepCacheSize(1);
}

//______________________________________________________________________
// Sets the number of timesteps to cache back to the default_cache_size
void
DataArchive::turnOnXMLCaching() {
  setTimestepCacheSize(default_cache_size);
}

//______________________________________________________________________
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
//______________________________________________________________________
//
void
DataArchive::TimeData::init()
{
  d_initialized = true;

  // Pull the list of data xml files from the timestep.xml file.

  FILE * ts_file   = fopen( d_ts_path_and_filename.c_str(), "r" );

  if( ts_file == nullptr ) {
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

  d_swapBytes = endianness != string(Uintah::endianness());
  d_nBytes    = numbits / 8;

  bool found = false;

  // Based on the timestep path and file name (eg: .../timestep.xml), we need
  // to cut off the associated path so that we can find the path to data.xml.
  string::size_type path_length = d_ts_path_and_filename.rfind( "/" ) + 1;
  string path( d_ts_path_and_filename, 0, path_length );
  string data_filename = path + "data.xml";

  FILE * data_file = fopen( data_filename.c_str(), "r" );

  string looked_in = data_filename;

  if ( data_file != nullptr ) {
    // If the data.xml file exists, look in it.
    found = ProblemSpec::findBlock( "<Data>", data_file );
  }
  else {
    // Otherwise, look in the original timestep.xml file.
    found = ProblemSpec::findBlock( "<Data>", ts_file );
    looked_in = d_ts_path_and_filename;
  }

  if( !found ) {
    throw InternalError( "Cannot find <Data> in " + looked_in, __FILE__, __LINE__ );
  }

  bool done = false;
  while( !done ) {

    string line = "";
    if ( data_file != nullptr ) {
      line = UintahXML::getLine( data_file );
    }
    else {
      line = UintahXML::getLine( ts_file );
    }
    if( line == "" || line == "</Data>" ) {
      done = true;
    }
    else if( line.compare( 0, 10, "<Datafile " ) == 0 ) {

      ProblemSpec ts_doc( line );

      map<string,string> attributes;
      ts_doc.getAttributes( attributes );
      string datafile = attributes[ "href" ];
      if( datafile == "" ) {
        throw InternalError( "DataArchive::TimeData::init() - 'href' not found", __FILE__, __LINE__ );
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
        // Assuming that global.xml will always be small and thus using normal xml lib parsing...
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
      throw InternalError( "DataArchive::TimeData::init() - bad line in <Data> block...", __FILE__, __LINE__ );
    }
  } // end while()

  fclose( ts_file );

  if ( data_file ) {
    fclose( data_file );
  }

} // end init()
//______________________________________________________________________
//
void
DataArchive::TimeData::purgeCache()
{
  d_grid = 0;

  d_datafileInfoIndex.clear();
  d_datafileInfoValue.clear();
  
  d_patchInfo.clear();
  d_varInfo.clear();
  d_xmlFilenames.clear();
  d_xmlParsed.clear();
  d_initialized = false;
}

//______________________________________________________________________
// This is the function that parses the p*****.xml file for a single processor.
void
DataArchive::TimeData::parseFile( const string & filename, int levelNum, int basePatch )
{
  // Parse the file.
  ProblemSpecP top = ProblemSpecReader().readInputFile( filename );

  // Materials are the same for all patches on a level - only parse them from one file.
  bool addMaterials = levelNum >= 0 && d_matlInfo[levelNum].size() == 0;

  for( ProblemSpecP vnode = top->getFirstChild(); vnode != nullptr; vnode=vnode->getNextSibling() ){
    if(vnode->getNodeName() == "Variable") {
      string varname;
      if( !vnode->get("variable", varname) ) {
        throw InternalError( "Cannot get variable name", __FILE__, __LINE__ );
      }

      int patchid;
      if(!vnode->get("patch", patchid) && !vnode->get("region", patchid)) {
        throw InternalError( "Cannot get patch id", __FILE__, __LINE__ );
      }

      int index;
      if(!vnode->get("index", index)) {
        throw InternalError( "Cannot get index", __FILE__, __LINE__ );
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
        throw InternalError( "DataArchive::query:Variable doesn't have a type", __FILE__, __LINE__ );
      }
      long start;
      if( !vnode->get("start", start) ) {
        throw InternalError( "DataArchive::query:Cannot get start", __FILE__, __LINE__ );
      }
      long end;
      if( !vnode->get("end", end) ) {
        throw InternalError( "DataArchive::query:Cannot get end", __FILE__, __LINE__ );
      }
      string filename;
      if( !vnode->get("filename", filename) ) {
        throw InternalError( "DataArchive::query:Cannot get filename", __FILE__, __LINE__ );
      }

      // Not required
      string    compressionMode = "";
      IntVector boundary(0,0,0);
      int       numParticles = -1;

      vnode->get( "compression", compressionMode );
      vnode->get( "boundaryLayer", boundary );
      vnode->get( "numParticles", numParticles );

      if( d_varInfo.find(varname) == d_varInfo.end() ) {
        VarData& varinfo      = d_varInfo[varname];
        varinfo.type          = type;
        varinfo.compression   = compressionMode;
        varinfo.boundaryLayer = boundary;
        varinfo.filename      = filename;
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
        ASSERTRANGE( patchid-basePatch, 0, (int)d_patchInfo[levelNum].size() );

        PatchData& patchinfo = d_patchInfo[levelNum][patchid-basePatch];
        if (!patchinfo.parsed) {
          patchinfo.parsed = true;
          patchinfo.datafilename = filename;
        }
      }
      VarnameMatlPatch vmp(varname, index, patchid);
      DataFileInfo     dummy;

      if( std::find( d_datafileInfoIndex.begin(), d_datafileInfoIndex.end(), vmp ) != d_datafileInfoIndex.end() ) {
        // cerr << "Duplicate variable name: " << name << endl;
      }
      else {
        DataFileInfo dfi( start, end, numParticles );
        d_datafileInfoIndex.push_back( vmp );
        d_datafileInfoValue.push_back( dfi );
      }
    }
    else if( vnode->getNodeType() != ProblemSpec::TEXT_NODE ) {
      cerr << "WARNING: Unknown element in Variables section: " << vnode->getNodeName() << '\n';
    }
  }
} // end TimeData::parseFile()

//______________________________________________________________________
//
void
DataArchive::TimeData::parsePatch( const Patch * patch )
{
  ASSERT( d_grid != nullptr );

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

  // If this is a newer uda, the patch info in the grid will store the
  // processor where the data is.
  if( patchinfo.proc != -1 ) {
    ostringstream file;
    file << d_ts_directory << "l" << (int) real_patch->getLevel()->getIndex() << "/p" << setw(5) << setfill('0') << (int) patchinfo.proc << ".xml";
    parseFile( file.str(), levelIndex, levelBasePatchID );

    // ARS - Commented out because the failure occurs regardless if
    // the l0 refence is present or not.
    
    // if( !patchinfo.parsed )
    // {
    //   throw InternalError( "DataArchive::parsePatch() - found patch processor "
    // 			   "id but could find the data in the coresponding "
    // 			   "processor data file. Check for zero length "
    // 			   "processor data files and remove their reference "
    // 			   "from the timestep.xml via this script: "
    // 			   "sed -i.bak '/Datafile href=\"l0/d' t*/timestep.xml.",
    // 			   __FILE__, __LINE__ );
    // }
  }
  else
  {
    // Try making a guess as to the processor.  First go is to try the
    // processor of the same index as the patch.  Many datasets have
    // only one patch per processor, so this is a reasonable first
    // attempt.  Future attemps could perhaps be smarter.
    if (!patchinfo.parsed && patchIndex < (int)d_xmlParsed[levelIndex].size() && !d_xmlParsed[levelIndex][patchIndex]) {
      parseFile( d_xmlFilenames[levelIndex][patchIndex], levelIndex, levelBasePatchID );
      d_xmlParsed[levelIndex][patchIndex] = true;
    }

    // Failed the guess - parse the entire dataset for this level
    if ( !patchinfo.parsed ) {
      for (unsigned proc = 0; proc < d_xmlFilenames[levelIndex].size(); proc++) {
	parseFile( d_xmlFilenames[levelIndex][proc], levelIndex, levelBasePatchID );
	d_xmlParsed[levelIndex][proc] = true;
      }
    }
  }
}

//______________________________________________________________________
// Parses the timestep xml file for <oldDelt>
//
double
DataArchive::getOldDelt( int restart_index )
{
  TimeData& timedata = getTimeData( restart_index );
  FILE * fp = fopen( timedata.d_ts_path_and_filename.c_str(), "r" );
  if( fp == nullptr ) {
    throw InternalError( "DataArchive::setOldDelt() failed open datafile.", __FILE__, __LINE__ );
  }
  // Note, old UDAs had a <delt> flag, but that was deprecated long ago in favor of the <oldDelt>
  // flag which is what we are going to look for here.

  while( true ) {

    string line = UintahXML::getLine( fp );

    if( line == "" ) {
      fclose( fp );
      throw InternalError( "DataArchive::setOldDelt() failed to find <oldDelt>.", __FILE__, __LINE__ );
    }
    else if( line.compare( 0, 9, "<oldDelt>" ) == 0 ) {
      vector<string> pieces = UintahXML::splitXMLtag( line );

      fclose( fp );
      return atof( pieces[1].c_str() );
    }
  }
}

//______________________________________________________________________
// Parses the timestep xml file and skips the <Meta>, <Grid>, and <Data> sections, returning
// everything else.  This function assumes that the timestep.xml file was created by us and
// is in the correct order - in other words, anything after </Data> is component related,
// and everything before it can be removed.
//
// Now that we are using grid.xml for the <Grid> and data.xml for the <Data> sections, this function is
// altered slightly.  Read in from the beginning of the <Uintah_timestep> including the <Meta> section.
//
ProblemSpecP
DataArchive::getTimestepDocForComponent( int restart_index )
{
  TimeData & timedata = getTimeData( restart_index );
  FILE     * fp       = fopen( timedata.d_ts_path_and_filename.c_str(), "r" );

  if( fp == nullptr ) {
    throw InternalError( "DataArchive::getTimespecDocForComponent() failed open datafile.", __FILE__, __LINE__ );
  }

#if 0
  bool found = ProblemSpec::findBlock( "</Data>", fp );
  if (!found) {
    found = ProblemSpec::findBlock( "</Data>", fp_grid );
    cout << "Found </Data> in grid_path filename" << endl;
  }

  if( !found ) {
    throw InternalError( "DataArchive::getTimespecDocForComponent() failed to find </Data>.", __FILE__, __LINE__ );
  }
#endif

  ////  string buffer = "<Uintah_timestep>";
  string buffer = "";

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
//______________________________________________________________________
//
ConsecutiveRangeSet
DataArchive::queryMaterials( const string & varname,
                             const Patch  * patch,
                                   int      index )
{
  Timers::Simple timer;
  timer.start();

  d_lock.lock();

  TimeData& timedata = getTimeData( index );
  timedata.parsePatch( patch );

  ConsecutiveRangeSet matls;

  for (unsigned i = 0; i < timedata.d_matlInfo[patch->getLevel()->getIndex()].size(); i++) {
    // i-1, since the matlInfo is adjusted to allow -1 as entries
    VarnameMatlPatch vmp(varname, i-1, patch->getRealPatch()->getID());
    DataFileInfo dummy;

    if( std::find( timedata.d_datafileInfoIndex.begin(), timedata.d_datafileInfoIndex.end(), vmp ) != timedata.d_datafileInfoIndex.end() ) {
      matls.addInOrder(i-1);
    }
  }

  d_lock.unlock();

  dbg << "DataArchive::queryMaterials completed in " << timer().seconds()
      << " seconds\n";

  return matls;
}
//______________________________________________________________________
//
int
DataArchive::queryNumMaterials(const Patch* patch, int index)
{
  Timers::Simple timer;
  timer.start();

  d_lock.lock();

  TimeData& timedata = getTimeData( index );

  timedata.parsePatch( patch );

  int numMatls = -1;

  for (unsigned i = 0; i < timedata.d_matlInfo[patch->getLevel()->getIndex()].size(); i++) {
    if (timedata.d_matlInfo[patch->getLevel()->getIndex()][i]) {
      numMatls++;
    }
  }

  d_lock.unlock();

  dbg << "DataArchive::queryNumMaterials completed in " << timer().seconds()
      << " seconds\n";

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
  timedata.parsePatch( patch );

  int levelIndex = patch->getLevel()->getIndex();

  for (unsigned i = 0; i < timedata.d_matlInfo[levelIndex].size(); i++) {
    // i-1, since the matlInfo is adjusted to allow -1 as entries
    VarnameMatlPatch vmp( varname, i-1, patch->getRealPatch()->getID() );
    DataFileInfo dummy;

    if( std::find( timedata.d_datafileInfoIndex.begin(), timedata.d_datafileInfoIndex.end(), vmp ) != timedata.d_datafileInfoIndex.end() ) {
      d_lock.unlock();
      return true;
    }
  }

  d_lock.unlock();

  return false;
}
