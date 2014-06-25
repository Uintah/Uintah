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

/* GPU DataWarehouse device & host access*/

#ifndef GPU_DW_H
#define GPU_DW_H

#include <sci_defs/cuda_defs.h>
#include <Core/Grid/Variables/GPUVariable.h>
#include <Core/Grid/Variables/GPUGridVariable.h>
#include <Core/Grid/Variables/GPUReductionVariable.h>
#include <Core/Grid/Variables/GPUPerPatch.h>

#define MAX_ITEM   100
#define MAX_LVITEM 10
#define MAX_NAME   20

namespace Uintah {

class GPUDataWarehouse {

public:

           GPUDataWarehouse() { d_numItems=0; d_device_copy=NULL; d_device_id=0; d_debug=true; d_dirty=true; }
  virtual ~GPUDataWarehouse() {}

  HOST_DEVICE void get(const GPUGridVariableBase& var, char const* label, int patchID, int matlIndex);
  HOST_DEVICE void get(const GPUReductionVariableBase& var, char const* label, int patchID, int matlIndex);
  HOST_DEVICE void get(const GPUPerPatchBase& var, char const* label, int patchID, int matlIndex);

  HOST_DEVICE void getModifiable(GPUGridVariableBase& var, char const* label, int patchID, int matlIndex);
  HOST_DEVICE void getModifiable(GPUReductionVariableBase& var, char const* label, int patchID, int matlIndex);
  HOST_DEVICE void getModifiable(GPUPerPatchBase& var, char const* label, int patchID, int matlIndex);

  HOST_DEVICE void put(GPUGridVariableBase& var, char const* label, int patchID, int matlIndex, bool overWrite=false);
  HOST_DEVICE void put(GPUReductionVariableBase& var, char const* label, int patchID, int matlIndex, bool overWrite=false);
  HOST_DEVICE void put(GPUPerPatchBase& var, char const* label, int patchID, int matlIndex, bool overWrite=false);

  HOST_DEVICE void allocateAndPut(GPUGridVariableBase& var, char const* label, int patchID, int matlIndex, int3 low, int3 high);
  HOST_DEVICE void allocateAndPut(GPUReductionVariableBase& var, char const* label, int patchID, int matlIndex);
  HOST_DEVICE void allocateAndPut(GPUPerPatchBase& var, char const* label, int patchID, int matlIndex);

  //______________________________________________________________________
  // GPU DataWarehouse support methods
  HOST_DEVICE bool exist(char const* name, int patchID, int matlIndex);
  HOST_DEVICE bool remove(char const* name, int patchID, int matlIndex);
  HOST_DEVICE void init_device(int id);
  HOST_DEVICE void syncto_device(); 
  HOST_DEVICE void clear();
  HOST_DEVICE GPUDataWarehouse* getdevice_ptr(){return d_device_copy;};
  HOST_DEVICE void setDebug(bool s){d_debug=s;}
  
private:

  HOST_DEVICE void printGetError(const char* msg, char const* label, int patchID, int matlIndex);


  int d_numItems;

  struct dataItem {   // flat array
    char       label[MAX_NAME];
    int        domainID;
    int        matlIndex;
    int3       var_offset;  
    int3       var_size;
    void*      var_ptr;
  };

  bool d_dirty;
  dataItem d_varDB[MAX_ITEM];
  dataItem d_levelDB[MAX_LVITEM];
  GPUDataWarehouse*  d_device_copy;  // in-device copy location
  int d_device_id;
  bool d_debug;

  HOST_DEVICE dataItem* getItem(char const* label, int patchID, int matlIndex);
};


} //end namespace Uintah

#endif
