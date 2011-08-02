/*
 
 The MIT License
 
 Copyright (c) 1997-2010 Center for the Simulation of Accidental Fires and 
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

#include <sci_defs/cuda_defs.h>

#ifdef HAVE_CUDA

#include <Core/Grid/Task.h>
#include <Core/Grid/CUDATask.h>
#include <Core/Disclosure/TypeDescription.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Grid.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Parallel/Parallel.h>
#include <set>


using namespace Uintah;
using namespace SCIRun;

CUDATask::~CUDATask()
{
}

//__________________________________
void
CUDATask::doit (const ProcessorGroup* pc,
            const PatchSubset* patches,
            const MaterialSubset* matls,
            vector < DataWarehouseP > &dws,
            int dev,
            CUDADevice *devprop)
{
    DataWarehouse* fromDW = mapDataWarehouse(Task::OldDW, dws);
    DataWarehouse* toDW = mapDataWarehouse(Task::NewDW, dws);
    if(d_action2)
    { 
        if(dynamic_cast<CUDAActionBase *>(d_action2))
        {    
            devprop->incrementRunningKernels();
            dynamic_cast<CUDAActionBase *>(d_action2)->doit(pc, patches, matls, fromDW, toDW, dev, devprop);
        }
        else
            throw new InternalError("ERROR:CUDATask:doit(): Failed to cast the action to a CUDAAction.  Bailing out.",__FILE__, __LINE__);
        
    }
}


#endif
