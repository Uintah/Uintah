/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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
#include <CCA/Ports/PIDXOutputContext.h>

#if HAVE_PIDX

#include <Core/Util/StringUtil.h>
#include <Core/Util/DebugStream.h>
#include <vector>

using namespace std;
using namespace Uintah;

namespace {
  DebugStream dbgPIDX ("PIDXOutputContext", "PIDXOutputContext", "PIDXOutputContext PIDX debug stream", false);
}

//______________________________________________________________________
//                P I D X _ F L A G S   C L A S S
//______________________________________________________________________
//
PIDXOutputContext::PIDX_flags::PIDX_flags()
{
  compressMap[ "NONE" ]         = PIDX_NO_COMPRESSION;
  compressMap[ "CHUNKING" ]     = PIDX_CHUNKING_ONLY;
  compressMap[ "CHUNKING_ZFP" ] = PIDX_CHUNKING_ZFP;

  // Set Defaults
  d_debugOutput = false;

  // Checkpoint Defaults
  d_checkpointFlags.ioType             = PIDX_RAW_IO;
  d_checkpointFlags.compressionType    = PIDX_NO_COMPRESSION;
  d_checkpointFlags.compressionBitrate = 32.0;
  d_checkpointFlags.restructureBoxSize = IntVector( 64, 64, 64 );
  d_checkpointFlags.pipeSize           =  64;
  d_checkpointFlags.partitionCount     = IntVector( 1, 1, 1 );
  d_checkpointFlags.blockSize          =  15;
  d_checkpointFlags.blockCount         = 256;

  // VisIo Defaults
  d_visIoFlags.ioType             = PIDX_RAW_IO;
  d_visIoFlags.compressionType    = PIDX_NO_COMPRESSION;
  d_visIoFlags.compressionBitrate = 32.0;
  d_visIoFlags.restructureBoxSize = IntVector( 64, 64, 64 );
  d_visIoFlags.pipeSize           =  64;
  d_visIoFlags.partitionCount     = IntVector( 1, 1, 1 );
  d_visIoFlags.blockSize          =  15;
  d_visIoFlags.blockCount         = 256;
}

//______________________________________________________________________
//  Utility:  returns the the compression type from input string

unsigned int 
PIDXOutputContext::PIDX_flags::str2CompressType( const std::string & type )
{
  string TYPE = string_toupper( type );  // convert to upper case  

  if( compressMap.find( TYPE ) == compressMap.end() ) {
    ostringstream warn;
    warn << "ERROR:PIDXOutputContext: The compression type (" << TYPE << ") is not supported."
         << " Valid options are: NONE, CHUNKING, CHUNKING_ZFP";
    throw Uintah::InternalError( warn.str(), __FILE__, __LINE__ );
  }
  return compressMap[ TYPE ];
}

//______________________________________________________________________
//   Utility:  returns the name of the compression type
std::string  
PIDXOutputContext::PIDX_flags::getCompressTypeName( const int type )
{
  std::map< std::string, int >::const_iterator it;
  std::string key = "NULL";
  for (it = compressMap.begin(); it!= compressMap.end(); ++it){
    if( it->second == type ) {
      key = it->first;
      return key;
    }
  }
  return key;
}

//______________________________________________________________________
//  Parses the ups file and set flags
void
PIDXOutputContext::PIDX_flags::problemSetup( const ProblemSpecP& DA_ps )
{
  ProblemSpecP pidx_ps = DA_ps->findBlock( "PIDX" );

  if( pidx_ps != nullptr ) {
  
    pidx_ps->getWithDefault( "debugOutput", d_debugOutput, false );

    PIDX_IoFlags * flags[2];
    flags[0] = &d_checkpointFlags;
    flags[1] = &d_visIoFlags;

    for( int i = 0; i <= 1; i++ ) {
      ProblemSpecP flagPs;

      PIDX_IoFlags * flagData = flags[ i ];
      if( i == 0 ) {
        flagPs = pidx_ps->findBlock( "checkpoint" );
      }
      else {
        flagPs = pidx_ps->findBlock( "visIO" );
      }

      if( flagPs == nullptr ) { continue; }

      string type;
      flagPs->get( "compressionType", type );
      flagData->compressionType = str2CompressType( type );

      if(flagPs->findBlock( "idxIo" ) != nullptr){
        flagPs->get( "compressionBitrate", flagData->compressionBitrate );
      }
      
      ProblemSpecP idxIoPS = flagPs->findBlock( "idxIo" );
      ProblemSpecP rawIoPS = flagPs->findBlock( "rawIo" );

      if( idxIoPS != nullptr ) {
        if(flagData->compressionType == PIDX_CHUNKING_ZFP)
          flagData->ioType = PIDX_IDX_IO;
        else 
          flagData->ioType = PIDX_LOCAL_PARTITION_IDX_IO;

        idxIoPS->get( "partitionCount", flagData->partitionCount );
        idxIoPS->get( "idxBlockSize",   flagData->blockSize );
        idxIoPS->get( "idxBlockCount",  flagData->blockCount );
      }
      else { // Raw IO

        flagData->ioType = PIDX_RAW_IO;

        string type;
        //rawIoPS->get( "compressType", type );
        //flagData->compressionType = str2CompressType( type );

        rawIoPS->get( "restructureBoxSize", flagData->restructureBoxSize );
        rawIoPS->get( "pipeSize",           flagData->pipeSize );
      }
    }
  }
  else {
    proc0cout << "Warning: Input .ups file does not have the <DataArchiver->PIDX> settings tag... Using defaults...\n";
  }
}

//______________________________________________________________________
//         P I D X O U T P U T C O N T E X T   C L A S S
//______________________________________________________________________


PIDXOutputContext::PIDXOutputContext()
{
  d_isInitialized = false;
  d_outputDoubleAsFloat = false;
}
//______________________________________________________________________
//
PIDXOutputContext::~PIDXOutputContext() 
{
  if(d_isInitialized){
    PIDX_close_access(this->access);
  }
}

//______________________________________________________________________
//  returns a vector of supported variables types
vector< Uintah::TypeDescription::Type >
PIDXOutputContext::getSupportedVariableTypes(){
    
  vector<TypeDescription::Type> GridVarTypes;
  GridVarTypes.empty();
  GridVarTypes.push_back( TypeDescription::CCVariable );        // This is where you define what types are supported.
  GridVarTypes.push_back( TypeDescription::SFCXVariable );
  GridVarTypes.push_back( TypeDescription::SFCYVariable );
  GridVarTypes.push_back( TypeDescription::SFCZVariable );
  GridVarTypes.push_back( TypeDescription::ParticleVariable );
  GridVarTypes.push_back( TypeDescription::NCVariable );
  return GridVarTypes;
}
 
//______________________________________________________________________
//  This returns the directory name associated with each data type
string
PIDXOutputContext::getDirectoryName(TypeDescription::Type TD)
{
  switch ( TD ){
    case TypeDescription::CCVariable:       return "CCVars";           break;
    case TypeDescription::SFCXVariable:     return "SFCXVars";         break;
    case TypeDescription::SFCYVariable:     return "SFCYVars";         break;
    case TypeDescription::SFCZVariable:     return "SFCZVars";         break;
    case TypeDescription::ParticleVariable: return "ParticleVars";     break;
    case TypeDescription::NCVariable:       return "NCVars";           break;
    default:
      throw Uintah::InternalError("  PIDXOutputContext::getDirectoryName type description not supported", __FILE__, __LINE__);
  }
}
//______________________________________________________________________
//    Logic for determinine the size of the box
void
PIDXOutputContext::computeBoxSize( const PatchSubset* patches, 
                                   const PIDX_flags flags,
                                   PIDX_point& newBox )
{
  ASSERT(patches->size() != 0);
  ASSERT(patches->get(0) != 0);
    
  //const Patch* patch = patches->get(0);
  const Level* level = patches->get(0)->getLevel();

  //__________________________________
  //  compute a patch size over all patches on this level
  IntVector nCells = level->nCellsPatch_max();
  int nPatchCells  = nCells.x() * nCells.y() * nCells.z();

  IntVector box( 64, 64, 64 );    // default value

 #if 0

  const int cubed16  = 16*16*16;
  const int cubed32  = 32*32*32;
  const int cubed64  = 64*64*64;
  const int cubed128 = 128*128*128;
  const int cubed256 = 256*256*256;
  
 //__________________________________
  // logic for adjusting the box
  if (nPatchCells <=  cubed16) {
    
  } else if (nPatchCells >  cubed16  && nPatchCells <= cubed32) {
    box = IntVector(32,32,32);
  } else if ( nPatchCells >  cubed32  && nPatchCells <= cubed64 ) {
    box = IntVector(64,64,64);
  } else if ( nPatchCells >  cubed64  && nPatchCells <= cubed128 ){
    box = IntVector(128,128,128);
  } else if ( nPatchCells >  cubed128 && nPatchCells <= cubed256 ){
    box = IntVector(256,256,256);
  }
  
  //__________________________________
  //  override the logic if user specifies somthing
  if ( flags.d_outputPatchSize != IntVector( -9, -9, -9 ) ) {
    box = flags.d_outputPatchSize;
  }
#endif
  
  if ( flags.d_debugOutput ) {
    cout << Parallel::getMPIRank() << " PIDX outputPatchSize: Level: "<< level->getIndex() << " box: " << box  
         << " Patchsize: " << nCells << " nPatchCells: " << nPatchCells  << "\n";
  }
  //PIDX_set_point( newBox, box.x(), box.y(), box.z());
  //  PIDX_set_point( newBox, 64, 64, 64 );
  box = IntVector( (int)pow(2, (int)ceil(log(nCells.x())/log(2))) * 2,
                   (int)pow(2, (int)ceil(log(nCells.y())/log(2))) * 2,
                   (int)pow(2, (int)ceil(log(nCells.z())/log(2))) * 2 );
  PIDX_set_point( newBox, box.x(), box.y(), box.z() );

}



//______________________________________________________________________
//
void
PIDXOutputContext::initialize( const string       & filename, 
                               const unsigned int   timeStep,
                                     MPI_Comm       comm,
                                     PIDX_flags     flags,
			             PIDX_point     dim,
                               const int            typeOutput )
{
  if(dbgPIDX.active())
    dbgPIDX << "PIDXOutputContext::initialize()\n";

  this->filename = filename;
  this->timestep = timeStep;
  string desc = "PIDXOutputContext::initialize";
  //__________________________________
  //
  int rc = PIDX_create_access(&(this->access));
  checkReturnCode( rc, desc + " - PIDX_create_access", __FILE__, __LINE__);
  
  if( comm != MPI_COMM_NULL ){
    PIDX_set_mpi_access( this->access, comm );
    checkReturnCode( rc, desc + " - PIDX_set_mpi_access", __FILE__, __LINE__);
  }
  
  PIDX_file_create( filename.c_str(), PIDX_MODE_CREATE, access, dim, &(this->file) );
  checkReturnCode( rc, desc + " - PIDX_file_create", __FILE__, __LINE__);
  
  PIDX_IoFlags & ioFlags = flags.d_visIoFlags;
  if( typeOutput == CHECKPOINT ){
    ioFlags = flags.d_checkpointFlags;
  }

  PIDX_set_io_mode( this->file, ioFlags.ioType );

  PIDX_set_compression_type( this->file, ioFlags.compressionType );

  checkReturnCode( rc, desc + " - PIDX_set_compression_type", __FILE__, __LINE__);

  if( ioFlags.ioType == PIDX_RAW_IO ) {
    PIDX_point rbox;
    int ret = PIDX_set_point( rbox,
                              ioFlags.restructureBoxSize[ 0 ],
                              ioFlags.restructureBoxSize[ 1 ],
                              ioFlags.restructureBoxSize[ 2 ] );
    checkReturnCode( ret,desc + " - PIDX_set_point restructure box failure", __FILE__, __LINE__ );
    PIDX_set_restructuring_box( file, rbox );
    checkReturnCode( rc, desc + " - checkpoint PIDX_set_restructuring_box", __FILE__, __LINE__);
  }

  PIDX_set_variable_pile_length( file, ioFlags.pipeSize );
  checkReturnCode( rc, desc + " - checkpoint PIDX_set_variable_pile_length", __FILE__, __LINE__);

  if( ioFlags.ioType == PIDX_LOCAL_PARTITION_IDX_IO ) {

    PIDX_set_block_size( this->file,  ioFlags.blockSize );
    checkReturnCode( rc, desc + " - PIDX_set_block_size", __FILE__, __LINE__);
  
    PIDX_set_block_count( this->file, ioFlags.blockCount );
    checkReturnCode( rc, desc + " - PIDX_set_block_count", __FILE__, __LINE__);

    PIDX_set_partition_count( this->file, ioFlags.partitionCount[0], ioFlags.partitionCount[1], ioFlags.partitionCount[2] );
  }


  // FIXME: The 1 below represents the 1st timestep... but if we begin output on another timestep, this should be changed...
  //PIDX_set_cache_time_step( this->file, 1 );
  //checkReturnCode( rc, desc + " - PIDX_enable_idx_io", __FILE__, __LINE__ );
  
  PIDX_set_first_time_step( this->file, timeStep );

  PIDX_set_current_time_step( this->file, timeStep );
  checkReturnCode( rc, desc + " - PIDX_set_current_time_step", __FILE__, __LINE__);

  d_isInitialized = true;
}

void
PIDXOutputContext::initializeParticles( const string       & filename, 
                                        const unsigned int   timeStep,
                                              MPI_Comm       comm,
                                              PIDX_point     dim,
                                        const int            typeOutput )
{
  cout << "PIDXOutputContext::initializeParticles()\n";

  this->filename = filename;
  this->timestep = timeStep;
  string desc = "PIDXOutputContext::initialize";
  //__________________________________
  //
  int rc = PIDX_create_access(&(this->access));
  checkReturnCode( rc, desc + " - PIDX_create_access", __FILE__, __LINE__);
  
  if( comm != MPI_COMM_NULL ){
    PIDX_set_mpi_access( this->access, comm );
    checkReturnCode( rc, desc + " - PIDX_set_mpi_access", __FILE__, __LINE__);
  }
  
  PIDX_file_create( filename.c_str(), PIDX_MODE_CREATE, access, dim, &(this->file) );
  checkReturnCode( rc, desc + " - PIDX_file_create", __FILE__, __LINE__);
  
  PIDX_set_io_mode( this->file, PIDX_PARTICLE_IO );

  PIDX_set_first_time_step( this->file, timeStep );
  
  PIDX_set_current_time_step( this->file, timeStep );
  checkReturnCode( rc, desc + " - PIDX_set_current_time_step", __FILE__, __LINE__);

  d_isInitialized = true;
}


//______________________________________________________________________
//  
void
PIDXOutputContext::setPatchExtents( const string          & desc, 
                                    const Patch           * patch,
                                    const Level           * level,
                                    const IntVector       & boundaryLayer,
                                    const TypeDescription * TD,
                                          patchExtents    & pExtents,
                                          PIDX_point      & patchOffset,
                                          PIDX_point      & patchSize ) const
{
   // Compute the extents of this variable (CCVariable, SFC(*)Variable...etc).
   IntVector hi_EC;
   IntVector lo_EC;
   patch->computeVariableExtents( TD->getType(), boundaryLayer, Ghost::None, 0, lo_EC, hi_EC );

   IntVector nCells_EC    = hi_EC - lo_EC;
   int totalCells_EC      = nCells_EC.x() * nCells_EC.y() * nCells_EC.z();
   
   IntVector offset       = level->getExtraCells();
   IntVector pOffset      = lo_EC + offset;           // pidx array indexing starts at 0, must shift by nExtraCells
   
   pExtents.lo_EC         = lo_EC;      // for readability
   pExtents.hi_EC         = hi_EC;
   pExtents.patchSize     = nCells_EC;
   pExtents.patchOffset   = pOffset;
   pExtents.totalCells_EC = totalCells_EC;

   int rc = PIDX_set_point(patchOffset,    pOffset.x(),    pOffset.y(), pOffset.z());
   checkReturnCode( rc, desc + " - PIDX_set_point_5D failure",__FILE__, __LINE__);

   rc = PIDX_set_point(patchSize, nCells_EC.x(), nCells_EC.y(), nCells_EC.z());
   checkReturnCode( rc, desc + "- PIDX_set_point_5D failure",__FILE__, __LINE__);
}

//______________________________________________________________________
//  

void
PIDXOutputContext::setLevelExtents( const string     & desc, 
                                          IntVector    lo,
                                          IntVector    hi,
                                          PIDX_point & level_size )
{                                                                             
  d_levelExtents[0] = hi[0] - lo[0] ;                                                                  
  d_levelExtents[1] = hi[1] - lo[1] ;                                                                  
  d_levelExtents[2] = hi[2] - lo[2] ;                                                                  

  int ret = PIDX_set_point(level_size, d_levelExtents[0], d_levelExtents[1], d_levelExtents[2]);  
  checkReturnCode( ret,desc+" - PIDX_set_point_5D failure", __FILE__, __LINE__);                     
}

//______________________________________________________________________
//  

void
PIDXOutputContext::checkReturnCode( const int      rc,
                                    const string   warn,
                                    const char   * file, 
                                    const int      line )
{
  if ( rc != PIDX_success ) {
    throw InternalError( warn, file, line );
  }
}


//______________________________________________________________________
//

void
PIDXOutputContext::hardWireBufferValues(       unsigned char * patchBuffer, 
                                         const patchExtents    patchExts,
                                         const size_t          arraySize,
                                         const int             samples_per_value )
{ 
  IntVector lo_EC = patchExts.lo_EC;
  IntVector hi_EC = patchExts.hi_EC;  
  
  double* buffer = (double*)malloc( arraySize );
  memcpy( buffer, patchBuffer, arraySize );
  IntVector levelExts = getLevelExtents();
  Vector origin = levelExts.asVector()/2.0;

  int c = 0;
  for (int k=lo_EC.z(); k<hi_EC.z(); k++){
    for (int j=lo_EC.y(); j<hi_EC.y(); j++){
      for (int i=lo_EC.x(); i<hi_EC.x(); i++){
        for ( int s = 0; s < samples_per_value; ++s ){
          IntVector here(i,j,k);
          Vector diff = here.asVector() - origin;
          double R = diff.length();
          buffer[c] = R;                         // Add function here for 
          c++;
        }
      }
    }
  }
    
  memcpy(patchBuffer,buffer, arraySize );
  free(buffer);
}


//______________________________________________________________________
//
void
PIDXOutputContext::printBufferWrap( const string                & desc,
                                    const TypeDescription::Type   TD,
                                    const int                     samples_per_value,
                                    const IntVector             & lo_EC,
                                    const IntVector             & hi_EC,
                                    const unsigned char         * dataPIDX,
                                    const size_t                  arraySize ) const
{
  if (TD == TypeDescription::int_type) {
    printBuffer<int>( desc, "%3i ", samples_per_value, lo_EC, hi_EC,dataPIDX, arraySize );
  }
  else {
    printBuffer<int>( desc, "%5.3f ", samples_per_value, lo_EC, hi_EC,dataPIDX, arraySize );
  }
}


//______________________________________________________________________
//
template<class T>
void
PIDXOutputContext::printBuffer( const string        & desc,
                                const string        & format,
                                const int             samples_per_value,
                                const IntVector     & lo_EC,
                                const IntVector     & hi_EC,
                                const unsigned char * dataPIDX,
                                const size_t          arraySize ) const
{
  cout << "__________________________________\n";
  cout << desc << endl;
  T* buffer = (T*)malloc( arraySize );
  memcpy( buffer, dataPIDX, arraySize );

  int c = 0;
  for (int k=lo_EC.z(); k<hi_EC.z(); k++){
    for (int j=lo_EC.y(); j<hi_EC.y(); j++){
      for (int i=lo_EC.x(); i<hi_EC.x(); i++){
        printf( " [%2i,%2i,%2i] ", i,j,k);
        for ( int s = 0; s < samples_per_value; ++s ){
          printf( format.c_str(),buffer[c]);
          c++;
        }
      }
      cout << "\n";
    }
    cout << "\n";
  }  
  cout << "\n__________________________________\n";
  cout << "\n";
  free( buffer );
}

// explicit instantiations
template void PIDXOutputContext::printBuffer<int>( const string        & desc,
                                                   const string        & format,
                                                   const int             samples_per_value,
                                                   const IntVector     & lo_EC,
                                                   const IntVector     & hi_EC,
                                                   const unsigned char * dataPIDX,
                                                   const size_t          arraySize ) const;

template void PIDXOutputContext::printBuffer<float>( const string        & desc,
                                                     const string        & format,
                                                     const int             samples_per_value,
                                                     const IntVector     & lo_EC,
                                                     const IntVector     & hi_EC,
                                                     const unsigned char * dataPIDX,
                                                     const size_t          arraySize ) const;

template void PIDXOutputContext::printBuffer<double>( const string        & desc,
                                                      const string        & format,
                                                      const int             samples_per_value,
                                                      const IntVector     & lo_EC,
                                                      const IntVector     & hi_EC,
                                                      const unsigned char * dataPIDX,
                                                      const size_t          arraySize ) const;
#endif
