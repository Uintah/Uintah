/*
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
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
#include <PIDX_compression.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Util/StringUtil.h>
#include <iostream>
#include <vector>

using namespace std;
using namespace SCIRun;
using namespace Uintah;

//______________________________________________________________________
//                P I D X _ F L A G S   C L A S S
//______________________________________________________________________
//
PIDXOutputContext::PIDX_flags::PIDX_flags()
{
  compressMap["NONE"]              = PIDX_NO_COMPRESSION;
  compressMap["CHUNKING"]          = PIDX_CHUNKING_ONLY;
  compressMap["PIDX_CHUNKING_ZFP"] = PIDX_CHUNKING_ZFP;
  
  outputRawIO    = false;
  debugOutput    = false;
  combinePatches = IntVector(-9,-9,-9);
  compressionType = PIDX_NO_COMPRESSION;
}


PIDXOutputContext::PIDX_flags::~PIDX_flags(){};


//______________________________________________________________________
//  Utility:  returns the the compression type from input string
unsigned int 
PIDXOutputContext::PIDX_flags::str2CompressType( const std::string& me)
{
  string ME = string_toupper( me );  // convert to upper case  
  if ( compressMap.count(ME) == 0){
    ostringstream warn;
    warn<< "ERROR:PIDXOutput:: the compression type ("<<ME<<") is not supported."
        << " Valid options are: NONE, CHUNKING, CHUNKING_ZFP";
    throw SCIRun::InternalError(warn.str(), __FILE__, __LINE__);
  }
  return compressMap[ME];
}

//______________________________________________________________________
//   Utility:  returns the name of the compression type
std::string  
PIDXOutputContext::PIDX_flags::getCompressTypeName( const int me )
{
  std::map< std::string, int >::const_iterator it;
  std::string key = "NULL";
  for (it = compressMap.begin(); it!= compressMap.end(); ++it){
    if( it->second == me){
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
  ProblemSpecP pidx_ps = DA_ps->findBlock("PIDX");

  if( pidx_ps != 0 ) {
  
    string me;
    pidx_ps->getWithDefault( "compressionType",  me, "NONE");
    
    compressionType = str2CompressType( me );
    pidx_ps->get( "debugOutput",     debugOutput );

    pidx_ps->get( "outputRawIO",     outputRawIO );

    if( outputRawIO ){
      pidx_ps->get( "combinePatches", combinePatches );
    }
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
  GridVarTypes.push_back(TypeDescription::CCVariable );        // This is where you define what types are supported.
  GridVarTypes.push_back(TypeDescription::SFCXVariable );
  GridVarTypes.push_back(TypeDescription::SFCYVariable );
  GridVarTypes.push_back(TypeDescription::SFCZVariable );
  return GridVarTypes;
}
 
//______________________________________________________________________
//  This returns the directory name associated with each data type
string
PIDXOutputContext::getDirectoryName(TypeDescription::Type TD)
{
  switch ( TD ){
      case TypeDescription::CCVariable:
        return "CCVars";  
        break;
      case TypeDescription::SFCXVariable:
        return "SFCXVars";
        break;
      case TypeDescription::SFCYVariable:
        return "SFCYVars";
        break;
      case TypeDescription::SFCZVariable:
        return "SFCZVars";
        break;
      default:
         throw SCIRun::InternalError("  PIDXOutputContext::getDirectoryName type description not supported", __FILE__, __LINE__);
  }
} 
//______________________________________________________________________
//
void
PIDXOutputContext::initialize( string filename, 
                               unsigned int timeStep,
                               MPI_Comm comm,
                               PIDX_flags flags)
{
    
  this->filename = filename;
  this->timestep = timeStep;
  this->comm = comm; 

  PIDX_create_access(&(this->access));
  
  if(comm != NULL){
    PIDX_set_mpi_access( this->access, this->comm );
  }
  
  PIDX_file_create( filename.c_str(), PIDX_MODE_CREATE, access, &(this->file) );
  
  if ( flags.debugOutput ){
    PIDX_debug_output( (this->file) );
  }

  if ( flags.outputRawIO ){
    PIDX_enable_raw_io(this->file);  //Possible performance improvement at low core counts
  }
  
  int64_t restructured_box_size[5] = {64, 64, 64, 1, 1};
  PIDX_set_restructuring_box(file, restructured_box_size);

  //PIDX_set_resolution(this->file, 0, 2);
  
  PIDX_set_current_time_step(this->file, timeStep);
  PIDX_set_block_size(this->file, 16);
  PIDX_set_block_count(this->file, 128);
    
  
                                       // Need to add logic so set compression to none on a checkpoint
  // compression settings
  PIDX_set_compression_type(this->file, flags.compressionType);
//  PIDX_set_lossy_compression_bit_rate(this->file, 8);

  d_isInitialized = true;
}

//______________________________________________________________________
//  
void
PIDXOutputContext::setPatchExtents( string desc, 
                                    const Patch* patch,
                                    const Level* level,
                                    const IntVector& boundaryLayer,
                                    const TypeDescription* TD,
                                    patchExtents& pExtents,
                                    PIDX_point& patchOffset,
                                    PIDX_point& patchSize )
{

   // compute the extents of this variable (CCVariable, SFC(*)Variable...etc)
   IntVector hi_EC;
   IntVector lo_EC;
   patch->computeVariableExtents(TD->getType(), boundaryLayer, Ghost::None, 0, lo_EC, hi_EC);

   IntVector nCells_EC    = hi_EC - lo_EC;
   int totalCells_EC      = nCells_EC.x() * nCells_EC.y() * nCells_EC.z();
   
   IntVector offset       = level->getExtraCells();
   IntVector pOffset      = lo_EC + offset;           // pidx array indexing starts at 0, must shift by nExtraCells
   
   pExtents.lo_EC         = lo_EC;      // for readability
   pExtents.hi_EC         = hi_EC;
   pExtents.patchSize     = nCells_EC;
   pExtents.patchOffset   = pOffset;
   pExtents.totalCells_EC = totalCells_EC;

   int rc = PIDX_set_point_5D(patchOffset,    pOffset.x(),    pOffset.y(), pOffset.z(),   0, 0);
   checkReturnCode( rc, desc + " - PIDX_set_point_5D failure",__FILE__, __LINE__);

   rc = PIDX_set_point_5D(patchSize, nCells_EC.x(), nCells_EC.y(), nCells_EC.z(), 1, 1);
   checkReturnCode( rc, desc + "- PIDX_set_point_5D failure",__FILE__, __LINE__);
}

//______________________________________________________________________
//  
void
PIDXOutputContext::setLevelExtents( string desc, 
                                    IntVector lo,
                                    IntVector hi,
                                    PIDX_point& level_size )
{                                                                             
  d_levelExtents[0] = hi[0] - lo[0] ;                                                                  
  d_levelExtents[1] = hi[1] - lo[1] ;                                                                  
  d_levelExtents[2] = hi[2] - lo[2] ;                                                                  

  int ret = PIDX_set_point_5D(level_size, d_levelExtents[0], d_levelExtents[1], d_levelExtents[2], 1, 1);  
  checkReturnCode( ret,desc+" - PIDX_set_point_5D failure", __FILE__, __LINE__);                     
}

//______________________________________________________________________
//  

void
PIDXOutputContext::checkReturnCode( const int rc,
                                    const string warn,
                                    const char* file, 
                                    int line)
{
  if (rc != PIDX_success){
    throw InternalError(warn, file, line);
  }
}


//______________________________________________________________________
//
void
PIDXOutputContext::hardWireBufferValues(unsigned char* patchBuffer, 
                                        const patchExtents patchExts,
                                        const size_t arraySize,
                                        const int samples_per_value )
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
PIDXOutputContext::printBufferWrap( const string&   desc,
                                    const TypeDescription::Type TD,
                                    int             samples_per_value,
                                    IntVector     & lo_EC,
                                    IntVector     & hi_EC,
                                    unsigned char * dataPIDX,
                                    size_t          arraySize )
{
  if (TD == TypeDescription::int_type) {
    printBuffer<int>( desc, "%3i ", samples_per_value, lo_EC, hi_EC,dataPIDX, arraySize );
  } else {
    printBuffer<int>( desc, "%5.3f ", samples_per_value, lo_EC, hi_EC,dataPIDX, arraySize );
  }
}


//______________________________________________________________________
//
template<class T>
void
PIDXOutputContext::printBuffer( const string        & desc,
                                const string        & format,
                                      int             samples_per_value,
                                      IntVector     & lo_EC,
                                      IntVector     & hi_EC,
                                      unsigned char * dataPIDX,
                                      size_t          arraySize )
{
  cout << "__________________________________ " << endl;
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
      printf("\n");
    }
    printf("\n");
  }  
  cout << "\n__________________________________ " << endl;
  printf("\n");
  free(buffer);
}

// explicit instantiations
template void PIDXOutputContext::printBuffer<int>( const string& desc,
                                                   const string& format,
                                                   int samples_per_value,
                                                   IntVector& lo_EC,
                                                   IntVector& hi_EC,
                                                   unsigned char* dataPIDX,
                                                   size_t arraySize );

template void PIDXOutputContext::printBuffer<float>( const string & desc,
                                                     const string& format,
                                                     int samples_per_value,
                                                     IntVector& lo_EC,
                                                     IntVector& hi_EC,
                                                     unsigned char* dataPIDX,
                                                     size_t arraySize );

template void PIDXOutputContext::printBuffer<double>( const string & desc,
                                                      const string& format,
                                                      int samples_per_value,
                                                      IntVector& lo_EC,
                                                      IntVector& hi_EC,
                                                      unsigned char* dataPIDX,
                                                      size_t arraySize );

#endif // HAVE_PIDX
