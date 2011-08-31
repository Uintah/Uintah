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

#include <CCA/Components/Examples/PoissonGPU1.h>
#include <CCA/Components/Examples/ExamplesLabel.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/CUDATask.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/SimpleMaterial.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Core/Grid/BoundaryConditions/BoundCond.h>
#include <CCA/Components/Schedulers/CUDADevice.h>

#include <sci_defs/cuda_defs.h>

using namespace Uintah;

PoissonGPU1::PoissonGPU1(const ProcessorGroup* myworld) :
UintahParallelComponent(myworld)
{

  phi_label = VarLabel::create("phi", NCVariable<double>::getTypeDescription());
  residual_label = VarLabel::create("residual",
      sum_vartype::getTypeDescription());
}

PoissonGPU1::~PoissonGPU1()
{
  VarLabel::destroy( phi_label);
  VarLabel::destroy( residual_label);
}
//______________________________________________________________________
//
void
PoissonGPU1::problemSetup(const ProblemSpecP& params,
    const ProblemSpecP& restart_prob_spec, GridP& /*grid*/,
    SimulationStateP& sharedState)
{
  sharedState_ = sharedState;
  ProblemSpecP poisson = params->findBlock("Poisson");

  poisson->require("delt", delt_);

  mymat_ = scinew SimpleMaterial();

  sharedState->registerSimpleMaterial(mymat_);
}
//______________________________________________________________________
//
void
PoissonGPU1::scheduleInitialize(const LevelP& level, SchedulerP& sched)
{
  Task* task = scinew Task("PoissonGPU1::initialize",
      this, &PoissonGPU1::initialize);

  task->computes(phi_label);
  task->computes(residual_label);
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}
//______________________________________________________________________
//
void
PoissonGPU1::scheduleComputeStableTimestep(const LevelP& level,
    SchedulerP& sched)
{
  Task* task = scinew Task("PoissonGPU1::computeStableTimestep",
      this, &PoissonGPU1::computeStableTimestep);

  task->requires(Task::NewDW, residual_label);
  task->computes(sharedState_->get_delt_label(), level.get_rep());
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}
//______________________________________________________________________
//
void
PoissonGPU1::scheduleTimeAdvance(const LevelP& level, SchedulerP& sched)
{
  Task* task = scinew Task("PoissonGPU1::timeAdvance",
      this, &PoissonGPU1::timeAdvance);
//  CUDATask* task = scinew CUDATask("PoissonGPU1::timeAdvance",
//      this, &PoissonGPU1::timeAdvance);

  task->requires(Task::OldDW, phi_label, Ghost::AroundNodes, 1);
  task->computes(phi_label);
  task->computes(residual_label);
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}
//______________________________________________________________________
//
void
PoissonGPU1::computeStableTimestep(const ProcessorGroup* pg,
    const PatchSubset* patches, const MaterialSubset* /*matls*/,
    DataWarehouse*, DataWarehouse* new_dw)
{

  if (pg->myrank() == 0)
  {
    sum_vartype residual;
    new_dw->get(residual, residual_label);
    cerr << "Residual=" << residual << '\n';
  }
  new_dw->put(delt_vartype(delt_), sharedState_->get_delt_label(), getLevel(patches));
}
//______________________________________________________________________
//
void
PoissonGPU1::initialize(const ProcessorGroup*, const PatchSubset* patches,
    const MaterialSubset* matls, DataWarehouse* /*old_dw*/,
    DataWarehouse* new_dw)
{
  int matl = 0;
  for (int p = 0; p < patches->size(); p++)
  {
    const Patch* patch = patches->get(p);

    NCVariable<double> phi;
    new_dw->allocateAndPut(phi, phi_label, matl, patch);
    phi.initialize(0.0);

    for (Patch::FaceType face = Patch::startFace; face <= Patch::endFace; face
        = Patch::nextFace(face))
    {
      if (patch->getBCType(face) == Patch::None)
      {
        int numChildren = patch->getBCDataArray(face)->getNumberChildren(
            matl);
        for (int child = 0; child < numChildren; child++)
        {
          Iterator nbound_ptr, nu;

          const BoundCondBase* bcb = patch->getArrayBCValues(face, matl, "Phi", nu, nbound_ptr, child);
          const BoundCond<double>* bc = dynamic_cast<const BoundCond<double>*> (bcb);
          double value = bc->getValue();
          for (nbound_ptr.reset(); !nbound_ptr.done(); nbound_ptr++)
          {
            phi[*nbound_ptr] = value;
          }
          delete bcb;
        }
      }
    }

    new_dw->put(sum_vartype(-1), residual_label);
  }
}

//______________________________________________________________________
//
// @brief A kernel that applies the stencil used in timeAdvance(...)
// @param domainSize a three component vector that gives the size of the domain as (x,y,z)
// @param domainLower a three component vector that gives the lower corner of the workarea as (x,y,z)
// @param residual the residual calculated by this individual kernel 
// @param oldphi pointer to the source phi allocated on the device
// @param newphi pointer to the sink phi allocated on the device
__global__ void timeAdvanceKernel(uint3 domainSize, uint3 domainLower, double *phi, double *newphi,
    double *residual) {

//  __shared__ double[] residual_device;
  // calculate the indices
//  int indxX = domainLower.x + blockDim.x * blockIdx.x + threadIdx.x;
//  int indxY = domainLower.y + blockDim.y * blockIdx.y + threadIdx.y;
//  int indxZ = domainLower.z + blockDim.z * blockIdx.z + threadIdx.z;
  int tidX = blockDim.x * blockIdx.x + threadIdx.x;
  int tidY = blockDim.y * blockIdx.y + threadIdx.y;
  int tidZ = blockDim.z * blockIdx.z + threadIdx.z;

  int dz = domainSize.x - 1;
  int dy = domainSize.y - 1;
  int dx = domainSize.z - 1;
  int offset = 1;

// calculate the offset in the dw representation
  int tid = tidX + ((dx + offset) * tidY + (dx + offset) * (dy + offset) * tidZ);

  newphi[tid] = (1.0 / 6.0)
      * (phi[tid - offset] + phi[tid + offset] + phi[tid - (dx + offset)]
          + phi[tid + dx + offset] + phi[tid - (dx + offset) * (offset + dy)]
          + phi[tid + (offset + dx) * (offset + dy)]);

  double diff = newphi[tid] - phi[tid];
  // this will cause a race condition. what we need is a scan to compute this
  // in conjunction with atomicAdd() and __shared__ double[] residual_device;
  *residual += diff * diff;
}

//______________________________________________________________________
//
void
PoissonGPU1::timeAdvance(const ProcessorGroup*,
                         const PatchSubset* patches,
                         const MaterialSubset* matls,
                         DataWarehouse* old_dw,
                         DataWarehouse* new_dw) //,
//                         int deviceID = 0,
//                         CUDADevice *deviceProperties = NULL)
{
  //
  int matl = 0;
  int previousPatchSize = 0;// this is to see if we need to release and reallocate between computations
  int size = 0;

  // declare device and host memory
  double* newphi_device;
  double* phi_device;
  double* phi_host;
  double* newphi_host;

  // find the "best" device for cudaSetDevice()
  int num_devices, device;
  cudaGetDeviceCount(&num_devices);
  if (num_devices > 1) {
    int max_multiprocessors = 0, max_device = 0;
    for (device = 0; device < num_devices; device++) {
      cudaDeviceProp properties;
      cudaGetDeviceProperties(&properties, device);
      if (max_multiprocessors < properties.multiProcessorCount) {
        max_multiprocessors = properties.multiProcessorCount;
        max_device = device;
      }
    }
    cudaSetDevice(max_device);
  }

  // Do time steps
  for (int p = 0; p < patches->size(); p++)
  {
    const Patch* patch = patches->get(p);
    constNCVariable<double> phi;
    old_dw->get(phi, phi_label, matl, patch, Ghost::AroundNodes, 1);

    NCVariable<double> newphi;
    new_dw->allocateAndPut(newphi, phi_label, matl, patch);
    newphi.copyPatch(phi, newphi.getLowIndex(), newphi.getHighIndex());

    double residual = 0;
    IntVector l = patch->getNodeLowIndex();
    IntVector h = patch->getNodeHighIndex();
    IntVector s = h-l;
    int xdim = s.x(), ydim = s.y(), zdim = s.z();
    size = xdim * ydim * zdim * sizeof(double);

    l += IntVector(patch->getBCType(Patch::xminus) == Patch::Neighbor?0:1,
        patch->getBCType(Patch::yminus) == Patch::Neighbor?0:1,
        patch->getBCType(Patch::zminus) == Patch::Neighbor?0:1);
    h -= IntVector(patch->getBCType(Patch::xplus) == Patch::Neighbor?0:1,
        patch->getBCType(Patch::yplus) == Patch::Neighbor?0:1,
        patch->getBCType(Patch::zplus) == Patch::Neighbor?0:1);

    // check if we need to reallocate
    if (size != previousPatchSize)
    {
      if (previousPatchSize != 0)
      {
        cudaFree(phi_device);
        cudaFree(newphi_device);
      }
      cudaMalloc(&phi_device, size);
      cudaMalloc(&newphi_device, size);
    }

    //__________________________________
    //  Memory Allocation
    phi_host = (double*) phi.getWindow()->getData()->getPointer();
    newphi_host = (double*) newphi.getWindow()->getData()->getPointer();

    // allocate space on the device
    // TODO
    // Fix this so when we have >= CCv2.0 we can use pinned host mem for phi
    cudaMemcpy(phi_device, phi_host, size, cudaMemcpyHostToDevice);
    cudaMemcpy(newphi_device, newphi_host, size, cudaMemcpyHostToDevice);

    uint3 domainSize = make_uint3(xdim, ydim, zdim);
    uint3 domainLower = make_uint3(l.x(), l.y(), l.z());
    int totalBlocks = size / (sizeof(double) * xdim * ydim * zdim);
    dim3 threadsPerBlock(xdim, ydim, zdim);

    if (size % (totalBlocks) != 0) {
      totalBlocks++;
    }

    // launch kernel
    timeAdvanceKernel<<< totalBlocks, threadsPerBlock >>>(domainSize, domainLower, phi_device, newphi_device, &residual);

    cudaDeviceSynchronize();
    cudaMemcpy(newphi_host, newphi_device, size, cudaMemcpyDeviceToHost);

    // now store residual that was device calculated
    new_dw->put(sum_vartype(residual), residual_label);

    //    //__________________________________
    //    //  3D-Pointer Stencil operation for reference
    //    double*** phi_data = (double***) phi.getWindow()->getData()->get3DPointer();
    //    double*** newphi_data = (double***) newphi.getWindow()->getData()->get3DPointer();
    //    double diff;
    //
    //    int zlen = s.z()-1;
    //    int ylen = s.y()-1;
    //    int xlen = s.x()-1;
    //    for (int i = 1; i < zlen; i++) {
    //      for (int j = 1; j < ylen; j++) {
    //        for (int k = 1; k < xlen; k++) {
    //
    //          double xminus = phi_data[i-1][j][k];
    //          double xplus  = phi_data[i+1][j][k];
    //          double yminus = phi_data[i][j-1][k];
    //          double yplus  = phi_data[i][j+1][k];
    //          double zminus = phi_data[i][j][k-1];
    //          double zplus  = phi_data[i][j][k+1];
    //
    //          newphi_data[i][j][k] = (1./6) * (xminus + xplus + yminus + yplus  + zminus + zplus);
    //
    //          diff = newphi_data[i][j][k] - phi_data[i][j][k];
    //          residual += diff * diff;
    //        }
    //      }
    //    }
    //    new_dw->put(sum_vartype(residual), residual_label);
  }

  // free up allocated memory
  cudaFree(phi_device);
  cudaFree(newphi_device);
}
