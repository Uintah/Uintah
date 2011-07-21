/*
 
 The MIT License
 
 Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
 Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
 University of Utah.
 
 License for the specific language governing rights and limitations under
 Permission is hereby granted, free of charge, to any person obtaining a 
 copy of this software and associated documentation files (the "Software"),
 to deal in the Software without restriction, including without limitation 
 the rights to use, copy, modify, merge, publish, distribute, sublicense, 
 and/or sell copies of the Software, and to permit persons to whom the 
 Software is furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included 
 in all copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
 OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
 THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
 FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
 DEALINGS IN THE SOFTWARE.
 
 */


#ifndef UINTAH_HOMEBREW_CUDADevice_H
#define UINTAH_HOMEBREW_CUDADevice_H


#include <Core/Grid/Task.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/Grid.h>
#include <Core/Geometry/IntVector.h>
#include <vector>
#include <list>
#include <map>
#include <CUDA/CUDA.h>

namespace Uintah {
    
    
    
    /**************************************
     
     CLASS
     
     CUDADevice
     
     
     
     GENERAL INFORMATION
     
     TaskGraph.h
     
     Joseph R. Peterson
     Department of Chemistry
     University of Utah
     
     Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
     
     Copyright (C) 2011 SCI Group
     
     
     
     KEYWORDS
     CUDA Device
     
     
     
     DESCRIPTION
     
     CUDADevice holds the information regarding a CUDA device used within the simulation.
     Specifically, it holds a device pointer, as well as information regarding the memory 
     of the device, the compute capability, threads per block, max dimensions of blocks, 
     max dimensions of a grid, etc.  Basically it wraps the functionality of cudaDeviceProp.
     
     WARNING
     
     ****************************************/
    class SchedulerCommon;
    
    class CUDADevice {
    public:
        /// @brief Default constructor
        /// @param devprop the device property structure
        CUDADevice(int);
        
        /// @brief Copy Constructor
        /// @param tocopy the CUDA device to copy
        CUDADevice(CUDADevice &);
        
        /// @brief the destructor
        ~CUDADevice();
        
        
        
        /////////////////////////////////////
        // Functions regarding device pointer
        /////////////////////////////////////
        /// @brief gets the number of the cuda device (essentially its pointer)
        int getDevicePtr()
        {
            return devicePointer;
        }
        
        
        /////////////////////////////
        // Functions regarding Memory
        /////////////////////////////
        /// @brief gets the total global memory for the device
        int maxGlobalMemory()
        {
            return devprop.totalGlobalMem;
        }
        
        /// @brief gets the total constant memory for the device
        int maxConstantMemory()
        {
            return devprop.totalConstMem;
        }
        
        /// @brief gets the total texture memory on the device
        int maxTextureMemory()
        {
            return devprop.maxTexture1D;
        }
        
        /// @brief gets the shared memory per block
        int sharedMemoryPerBlock()
        {
            return devprop.sharedMemPerBlock;
        }
        
        /// @brief gets the memory bus width for the device
        int memoryBusWidth()
        {
            return devprop.memoryBusWidth();
        }
        
        
        ////////////////////////////////////////////////
        // Functions regarding blocks, threads and grids
        ////////////////////////////////////////////////
        /// @brief gets the warp size for the device
        int warpSize()
        {
            return devprop.warpSize;
        }
        
        /// @brief gets the maximum threads available on the device
        int maxThreads()
        {
            return devprop.maxThreadsPerMultiProcessor;
        }
        
        /// @brief gets maximum threads a block can have
        int maxThreadsPerBlock()
        {
            return devprop.maxThreadsPerBlock;
        }
        
        /// @brief gets the grid size as an IntVector for x,y,z grid sizes
        IntVector maxGridSize()
        {
            return IntVector(devprop.maxGridSize[0],devprop.maxGridSize[1],devprop.maxGridSize[2]);
        }
        
        /// @brief gets the total number of kernels that can be run concurrently
        int maxConcurrentKernels()
        {
            return devprop.concurrentKernels;
        }
        
    private:
        
        cudaDeviceProp devprop;
        int devicePointer;
        
    };
} // End namespace Uintah

#endif
