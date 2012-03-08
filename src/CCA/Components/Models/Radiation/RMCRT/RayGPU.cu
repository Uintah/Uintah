/*

 The MIT License

 Copyright (c) 1997-2012 Center for the Simulation of Accidental Fires and
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


//----- RayGPU.cu ----------------------------------------------
#include <CCA/Components/Models/Radiation/RMCRT/MersenneTwister.h>
#include <CCA/Components/Models/Radiation/RMCRT/Ray.h>
#include <Core/Grid/DbgOutput.h>

#include <sci_defs/cuda_defs.h>

using namespace Uintah;
using namespace std;

static DebugStream dbg("RAY_GPU",       false);
static DebugStream dbg2("RAY_GPU_DEBUG",false);
static DebugStream dbg_BC("RAY_GPU_BC", false);


//---------------------------------------------------------------------------
// Method: The GPU ray tracer
//---------------------------------------------------------------------------
void Ray::rayTraceGPU( const ProcessorGroup* pc,
                       const PatchSubset* patches,
                       const MaterialSubset* matls,
                       DataWarehouse* old_dw,
                       DataWarehouse* new_dw,
                       int device,
                       bool modifies_divQ,
                       Task::WhichDW which_abskg_dw,
                       Task::WhichDW which_sigmaT4_dw )
{
  cudaSetDevice(device);
  initMTRandGPU();

  // setup for and invoke RT GPU kernel

}

void Ray::initMTRandGPU()
{
  // stub
}

//---------------------------------------------------------------------------
// Method:
//---------------------------------------------------------------------------
inline bool Ray::containsCellGPU(const dim3 &low,
                                 const dim3 &high,
                                 const dim3 &cell,
                                 const int &face)
{
  // stub
  return false;
}

//---------------------------------------------------------------------------
// Function: The GPU ray tracer kernel
//---------------------------------------------------------------------------
__global__ void rayTraceKernel()
{
  // stub
}

