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
#include "PIDXOutputContext.h"
#include <Core/Exceptions/InternalError.h>
#if HAVE_PIDX

namespace Uintah 
{

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
  std::vector<TypeDescription::Type>
  PIDXOutputContext::getSupportedVariableTypes(){
    
    std::vector<TypeDescription::Type> GridVarTypes;
    GridVarTypes.empty();
    GridVarTypes.push_back(TypeDescription::CCVariable );        // This is where you define what types are supported.
    GridVarTypes.push_back(TypeDescription::SFCXVariable );
    GridVarTypes.push_back(TypeDescription::SFCYVariable );
    GridVarTypes.push_back(TypeDescription::SFCZVariable );
    return GridVarTypes;
  }
 
  //______________________________________________________________________
  //  This returns the directory name associated with each data type
  std::string
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
  void PIDXOutputContext::initialize(std::string filename, unsigned int timeStep, int globalExtents[3] ,MPI_Comm comm)
  {


    this->filename = filename;
    this->timestep = timeStep;
    this->comm = comm; 


    PIDX_point global_bounding_box;
    PIDX_create_access(&(this->access));
    PIDX_set_mpi_access(this->access, this->comm);

    
    PIDX_set_point_5D(global_bounding_box, globalExtents[0], globalExtents[1], globalExtents[2], 1, 1);


    PIDX_file_create(filename.c_str(), PIDX_MODE_CREATE, access, &(this->file));

    int64_t restructured_box_size[5] = {64, 64, 64, 1, 1};
    PIDX_set_restructuring_box(file, restructured_box_size);

    //PIDX_set_resolution(this->file, 0, 2);
    PIDX_set_dims(this->file, global_bounding_box);
    PIDX_set_current_time_step(this->file, timeStep);
    PIDX_set_block_size(this->file, 16);
    PIDX_set_block_count(this->file, 128);
    
    //PIDX_set_compression_type(this->file, PIDX_CHUNKING_ZFP);
    //PIDX_set_lossy_compression_bit_rate(this->file, 8);
    d_isInitialized = true;
  }


} // end namespace Uintah

#endif // HAVE_PIDX
