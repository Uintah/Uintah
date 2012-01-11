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

#include <CCA/Components/Examples/PoissonGPU1.h>
#include <CCA/Components/Examples/ExamplesLabel.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/SimpleMaterial.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Core/Grid/BoundaryConditions/BoundCond.h>

#include <sci_defs/cuda_defs.h>

using namespace Uintah;

PoissonGPU1::PoissonGPU1(const ProcessorGroup* myworld) : UintahParallelComponent(myworld) {

  phi_label = VarLabel::create("phi", NCVariable<double>::getTypeDescription());
  residual_label = VarLabel::create("residual", sum_vartype::getTypeDescription());
}

PoissonGPU1::~PoissonGPU1() {
  VarLabel::destroy(phi_label);
  VarLabel::destroy(residual_label);
}
//______________________________________________________________________
//
void PoissonGPU1::problemSetup(const ProblemSpecP& params,
                               const ProblemSpecP& restart_prob_spec,
                               GridP& /*grid*/,
                               SimulationStateP& sharedState) {
  sharedState_ = sharedState;
  ProblemSpecP poisson = params->findBlock("Poisson");

  poisson->require("delt", delt_);

  mymat_ = scinew
  SimpleMaterial();

  sharedState->registerSimpleMaterial(mymat_);
}
//______________________________________________________________________
//
void PoissonGPU1::scheduleInitialize(const LevelP& level,
                                     SchedulerP& sched) {
  Task * task = scinew
  Task("PoissonGPU1::initialize", this, &PoissonGPU1::initialize);

  task->computes(phi_label);
  task->computes(residual_label);
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}
//______________________________________________________________________
//
void PoissonGPU1::scheduleComputeStableTimestep(const LevelP& level,
                                                SchedulerP& sched) {
  Task * task = scinew
  Task("PoissonGPU1::computeStableTimestep", this, &PoissonGPU1::computeStableTimestep);

  task->requires(Task::NewDW, residual_label);
  task->computes(sharedState_->get_delt_label(), level.get_rep());
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}
//______________________________________________________________________
//
void PoissonGPU1::scheduleTimeAdvance(const LevelP& level,
                                      SchedulerP& sched) {
  Task * task = scinew
  Task("PoissonGPU1::timeAdvance", this, &PoissonGPU1::timeAdvance);
//  CUDATask* task = scinew CUDATask("PoissonGPU1::timeAdvance",
//      this, &PoissonGPU1::timeAdvance);

  task->requires(Task::OldDW, phi_label, Ghost::AroundNodes, 1);
  task->computes(phi_label);
  task->computes(residual_label);
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}
//______________________________________________________________________
//
void PoissonGPU1::computeStableTimestep(const ProcessorGroup* pg,
                                        const PatchSubset* patches,
                                        const MaterialSubset* /*matls*/,
                                        DataWarehouse*,
                                        DataWarehouse* new_dw) {

  if (pg->myrank() == 0) {
    sum_vartype residual;
    new_dw->get(residual, residual_label);
    std::cerr << "Residual=" << residual << '\n';
  }
  new_dw->put(delt_vartype(delt_), sharedState_->get_delt_label(), getLevel(patches));
}
//______________________________________________________________________
//
void PoissonGPU1::initialize(const ProcessorGroup*,
                             const PatchSubset* patches,
                             const MaterialSubset* matls,
                             DataWarehouse* /*old_dw*/,
                             DataWarehouse* new_dw) {
  int matl = 0;
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);

    NCVariable<double> phi;
    new_dw->allocateAndPut(phi, phi_label, matl, patch);
    phi.initialize(0.0);

    for (Patch::FaceType face = Patch::startFace; face <= Patch::endFace;
        face = Patch::nextFace(face)) {
      if (patch->getBCType(face) == Patch::None) {
        int numChildren = patch->getBCDataArray(face)->getNumberChildren(matl);
        for (int child = 0; child < numChildren; child++) {
          Iterator nbound_ptr, nu;
          const BoundCondBase* bcb = patch->getArrayBCValues(face, matl, "Phi", nu, nbound_ptr, child);
          const BoundCond<double>* bc = dynamic_cast<const BoundCond<double>*>(bcb);
          double value = bc->getValue();
          for (nbound_ptr.reset(); !nbound_ptr.done(); nbound_ptr++) {
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
// @param domainLower a three component vector that gives the lower corner of the work area as (x,y,z)
// @param ghostLayers the number of layers of ghost cells
// @param phi pointer to the source phi allocated on the device
// @param newphi pointer to the sink phi allocated on the device
// @param residual the residual calculated by this individual kernel 
__global__ void timeAdvanceKernel(uint3 domainSize,
                                  uint3 domainLower,
                                  int ghostLayers,
                                  double *phi,
                                  double *newphi,
                                  double *residual) {

// calculate the thread indices
  int tidX = blockDim.x * blockIdx.x + threadIdx.x;
  int tidY = blockDim.y * blockIdx.y + threadIdx.y;
  //  int tidZ = blockDim.z * blockIdx.z + threadIdx.z;

  int num_slices = domainSize.z - ghostLayers;
  int dx = domainSize.x;
  int dy = domainSize.y;

  if (tidX < (dx - ghostLayers) && tidY < (dy - ghostLayers) && tidX > 0 && tidY > 0) {
    for (int slice = ghostLayers; slice < num_slices; slice++) {
      int idx = (slice) + (tidY * dx) + (tidX * dx * dy);

      int xminus = (slice - 1) + (tidY * dx) + (tidX * dx * dy);
      int xplus  = (slice + 1) + (tidY * dx) + (tidX * dx * dy);
      int yminus = slice + ((tidY - 1) * dx) + (tidX * dx * dy);
      int yplus  = slice + ((tidY + 1) * dx) + (tidX * dx * dy);
      int zminus = slice + (tidY * dx) + ((tidX - 1) * dx * dy);
      int zplus  = slice + (tidY * dx) + ((tidX + 1) * dx * dy);

      newphi[idx] =
               (1.0 / 6.0)
                 * (phi[xminus]
                  + phi[xplus]
                  + phi[yminus]
                  + phi[yplus]
                  + phi[zminus]
                  + phi[zplus]);

    }
  }
}

//______________________________________________________________________
//
void PoissonGPU1::timeAdvance(const ProcessorGroup*,
                              const PatchSubset* patches,
                              const MaterialSubset* matls,
                              DataWarehouse* old_dw,
                              DataWarehouse* new_dw) {

  int matl = 0;
  int previousPatchSize = 0;// this is to see if we need to release and reallocate between computations
  int size = 0;
  int ghostLayers = 1;

  // declare device and host memory
  double* newphi_device;
  double* phi_device;
  double* phi_host;
  double* newphi_host;
//
//  // find the "best" device for cudaSetDevice()
//  int num_devices, device;
//  cudaGetDeviceCount(&num_devices);
//  if (num_devices > 1) {
//    int max_multiprocessors = 0, max_device = 0;
//    for (device = 0; device < num_devices; device++) {
//      cudaDeviceProp properties;
//      cudaGetDeviceProperties(&properties, device);
//      if (max_multiprocessors < properties.multiProcessorCount) {
//        max_multiprocessors = properties.multiProcessorCount;
//        max_device = device;
//      }
//    }
//    cudaSetDevice(max_device);
//  }
//  // Do time steps
//  int numPatches = patches->size();
//  for (int p = 0; p < numPatches; p++) {
//    const Patch* patch = patches->get(p);
//    constNCVariable<double> phi;
//    old_dw->get(phi, phi_label, matl, patch, Ghost::AroundNodes, ghostLayers);
//
//    NCVariable<double> newphi;
//    new_dw->allocateAndPut(newphi, phi_label, matl, patch);
//    newphi.copyPatch(phi, newphi.getLowIndex(), newphi.getHighIndex());
//
//    double residual = 0;
//    IntVector l = patch->getNodeLowIndex();
//    IntVector h = patch->getNodeHighIndex();
//    int xdim = h.x() + 1, ydim = h.y() + 1, zdim = h.z() + 1;
//    size = xdim * ydim * zdim * sizeof(double);
//
//    l += IntVector(patch->getBCType(Patch::xminus) == Patch::Neighbor ? 0 : 1,
//        patch->getBCType(Patch::yminus) == Patch::Neighbor ? 0 : 1,
//        patch->getBCType(Patch::zminus) == Patch::Neighbor ? 0 : 1);
//    h -= IntVector(patch->getBCType(Patch::xplus) == Patch::Neighbor ? 0 : 1,
//        patch->getBCType(Patch::yplus) == Patch::Neighbor ? 0 : 1,
//        patch->getBCType(Patch::zplus) == Patch::Neighbor ? 0 : 1);
//
//    // check if we need to reallocate
//    if (size != previousPatchSize) {
//      if (previousPatchSize != 0) {
//        cudaFree(phi_device);
//        cudaFree(newphi_device);
//      }
//      cudaMalloc(&phi_device, size);
//      cudaMalloc(&newphi_device, size);
//    }
//
//    //__________________________________
//    //  Memory Allocation
//    phi_host    = (double*)phi.getWindow()->getData()->getPointer();
//    newphi_host = (double*)newphi.getWindow()->getData()->getPointer();
//
//    // allocate space on the device
//    // TODO
//    // Fix this so when we have >= CCv2.0 we can use pinned host mem for phi
//    cudaMemcpy(phi_device,    phi_host,    size, cudaMemcpyHostToDevice);
//    cudaMemcpy(newphi_device, newphi_host, size, cudaMemcpyHostToDevice);
//
//    uint3 domainSize = make_uint3(xdim, ydim, zdim);
//    uint3 domainLower = make_uint3(l.x(), l.y(), l.z());
//    int totalBlocks = size / (sizeof(double) * xdim * ydim * zdim);
//    dim3 threadsPerBlock(xdim, ydim, zdim);
//
//    if (size % (totalBlocks) != 0) {
//      totalBlocks++;
//    }
//    // launch kernel
//    timeAdvanceKernel<<< totalBlocks, threadsPerBlock >>>(domainSize, domainLower, ghostLayers, phi_device, newphi_device, &residual);
//
//    cudaDeviceSynchronize();
//    cudaMemcpy(newphi_host, newphi_device, size, cudaMemcpyDeviceToHost);
//
//
//    new_dw->put(sum_vartype(residual), residual_label);
//    //__________________________________
//    //  3D-Pointer Stencil
//    double*** phi_data = (double***)phi.getWindow()->getData()->get3DPointer();
//    double*** newphi_data = (double***)newphi.getWindow()->getData()->get3DPointer();
//
//    int zhigh = h.z();
//    int yhigh = h.y();
//    int xhigh = h.x();
//
//    for (int i = l.z(); i < zhigh; i++) {
//      for (int j = l.y(); j < yhigh; j++) {
//        for (int k = l.x(); k < xhigh; k++) {
//
//          double xminus = phi_data[i-1][j][k];
//          double xplus  = phi_data[i+1][j][k];
//          double yminus = phi_data[i][j-1][k];
//          double yplus  = phi_data[i][j+1][k];
//          double zminus = phi_data[i][j][k-1];
//          double zplus  = phi_data[i][j][k+1];
//
//          newphi_data[i][j][k] = (1. / 6) * (xminus + xplus + yminus + yplus + zminus + zplus);
//
//          double diff = newphi_data[i][j][k] - phi_data[i][j][k];
//          residual += diff * diff;
//        }
//      }
//    }
//    new_dw->put(sum_vartype(residual), residual_label);

  // Do time steps
  int numPatches = patches->size();
  int z = 0;
  for (int p = 0; p < numPatches; p++) {
    const Patch* patch = patches->get(p);
    constNCVariable<double> phi;
    old_dw->get(phi, phi_label, matl, patch, Ghost::AroundNodes, ghostLayers);

    NCVariable<double> newphi;
    new_dw->allocateAndPut(newphi, phi_label, matl, patch);
    newphi.copyPatch(phi, newphi.getLowIndex(), newphi.getHighIndex());

    double residual = 0;
    IntVector l = patch->getNodeLowIndex();
    IntVector h = patch->getNodeHighIndex();
    IntVector s = h - l;
    int xdim = s.x(), ydim = s.y(), zdim = s.z();
    size = xdim * ydim * zdim * sizeof(double);

    l += IntVector(patch->getBCType(Patch::xminus) == Patch::Neighbor ? 0 : 1,
        patch->getBCType(Patch::yminus) == Patch::Neighbor ? 0 : 1,
        patch->getBCType(Patch::zminus) == Patch::Neighbor ? 0 : 1);
    h -= IntVector(patch->getBCType(Patch::xplus) == Patch::Neighbor ? 0 : 1,
        patch->getBCType(Patch::yplus) == Patch::Neighbor ? 0 : 1,
        patch->getBCType(Patch::zplus) == Patch::Neighbor ? 0 : 1);


    //__________________________________
    // 1D-Pointer Stencil
    double *phi_data = (double*)phi.getWindow()->getData()->getPointer();
    double *newphi_data = (double*)newphi.getWindow()->getData()->getPointer();

    int zhigh = h.z();
    int yhigh = h.y();
    int xhigh = h.x();
    int ghostLayers = 1;
    int ystride = ydim;
    int xstride = xdim;

    int lowx = l.x();
    int lowy = l.y();
    int lowz = l.z();

    for (int i = lowz; i < zhigh; i++) {
      for (int j = lowy; j < yhigh; j++) {
        for (int k = lowx; k < xhigh; k++) {

          // For an array of [ A ][ B ][ C ], we can index it thus:
          // (a * B * C) + (b * C) + (c * 1)
          int idx = i + (j * xstride) + (k * xstride * ystride);

          int xminus = (i - 1) + (j * xstride) + (k * xstride * ystride);
          int xplus  = (i + 1) + (j * xstride) + (k * xstride * ystride);
          int yminus = i + ((j - 1) * xstride) + (k * xstride * ystride);
          int yplus  = i + ((j + 1) * xstride) + (k * xstride * ystride);
          int zminus = i + (j * xstride) + ((k - 1) * xstride * ystride);
          int zplus  = i + (j * xstride) + ((k + 1) * xstride * ystride);

          newphi_data[idx] = (1. / 6) * (phi_data[xminus] + phi_data[xplus] + phi_data[yminus]
              + phi_data[yplus] + phi_data[zminus] + phi_data[zplus]);

          double diff = newphi_data[idx] - phi_data[idx];
          residual += diff * diff;
        }
      }
    }
    new_dw->put(sum_vartype(residual), residual_label);
    z++;
  }  // end patch for loop

  // free up allocated memory
  //cudaFree(phi_device);
  //cudaFree(newphi_device);
}
