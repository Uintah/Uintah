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
  CUDATask* task = scinew CUDATask("PoissonGPU1::timeAdvance",
      this, &PoissonGPU1::timeAdvance);

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
  std::cout << "In computeStableTimestep" << std::endl;


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
  std::cout << "In initialize" << std::endl;


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

                  const BoundCondBase* bcb = patch->getArrayBCValues(face,
                      matl, "Phi", nu, nbound_ptr, child);

                  const BoundCond<double>* bc = dynamic_cast<const BoundCond<
                      double>*> (bcb);
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
__global__ void
timeAdvanceKernel(uint3 domainSize, uint3 domainLower, double *oldphi, double *newphi)
{
  // calculate the indices
  int indxX = domainLower.x + blockDim.x * blockIdx.x + threadIdx.x;
  int indxY = domainLower.y + blockDim.y * blockIdx.y + threadIdx.y;
  int indxZ = domainLower.z + blockDim.z * blockIdx.z + threadIdx.z;

//  // Do not perform calculation if in a ghost cell or outside the domain
//  if ((indxX < domainSize.x && indxX > 0)
//      && (indxY < domainSize.y && indxY > 0) && (indxZ < domainSize.z && indxZ
//      > 0))
//    {
//      // calculate the offset in the dw representation
//      int baseIdx = domainSize.x * (indxZ * domainSize.y + indxY + 1);
//
//      // compute the stencil
//      // FIXME - domainSize is not a class type. Need to get access to underlying IntVectors
//      newphi[baseIdx] = (1.0 / 6.0) * (oldphi[baseIdx + 1]
//          + oldphi[baseIdx - 1] + oldphi[baseIdx + baseIdx] + oldphi[baseIdx
//          - domainSize.y] + oldphi[baseIdx + domainSize.z * domainSize.y]
//          + oldphi[baseIdx - domainSize.z * domainSize.y]);
//
//      // compute the residual
//      double diff = newphi[baseIdx] - oldphi[baseIdx];
//      *residual += diff * diff; // THIS LINE IS WRONG--NEED SOME SORT OF LOCKING MAYBE
//    }
}

//______________________________________________________________________
//
void
PoissonGPU1::timeAdvance(const ProcessorGroup*, const PatchSubset* patches,
    const MaterialSubset* matls, DataWarehouse* old_dw, DataWarehouse* new_dw,
    int deviceID = 0, CUDADevice *deviceProperties = NULL)
{

  std::cout << "In timeAdvance" << std::endl;

  int matl = 0;
  int previousPatchSize = 0; // this is to see if we need to release and reallocate between computations
  int size = 0;

  // device memory
  double * phinew;
  double * phiold;

  // set CUDA device
  // cudaThreadExit();
  cudaSetDevice(deviceID);

  double residual = 0;

  for (int p = 0; p < patches->size(); p++)
    {
      const Patch* patch = patches->get(p);
      constNCVariable<double> phi;

      old_dw->get(phi, phi_label, matl, patch, Ghost::AroundNodes, 1);
      NCVariable<double> newphi;

      new_dw->allocateAndPut(newphi, phi_label, matl, patch);
      newphi.copyPatch(phi, newphi.getLowIndex(), newphi.getHighIndex());

      residual = 0;

      IntVector l = patch->getNodeLowIndex();
      IntVector h = patch->getNodeHighIndex();

      l += IntVector(
          patch->getBCType(Patch::xminus) == Patch::Neighbor ? 0 : 1,
          patch->getBCType(Patch::yminus) == Patch::Neighbor ? 0 : 1,
          patch->getBCType(Patch::zminus) == Patch::Neighbor ? 0 : 1);
      h -= IntVector(patch->getBCType(Patch::xplus) == Patch::Neighbor ? 0 : 1,
          patch->getBCType(Patch::yplus) == Patch::Neighbor ? 0 : 1,
          patch->getBCType(Patch::zplus) == Patch::Neighbor ? 0 : 1);

      //__________________________________
      //  Stencil 

      //// Memory Allocation ////
      ///////////////////////////
      // USE PINNING INSTEAD
      int size = (h.x() - l.x()) * (h.y() - l.y()) * (h.z() - l.z())
          * sizeof(double);
      // check if we need to reallocate
      if (size != previousPatchSize)
        {
          if (previousPatchSize != 0)
            {
              cudaFree(phinew);
              cudaFree(phiold);
            }
          cudaMalloc(&phiold, size);
          cudaMalloc(&phinew, size);
        }
      // host pointers
      /*
      double *oldhostmem = NULL; // point this at DW representation
      double *newhostmem = NULL; // point this at DW representation

      cudaMemcpy(phiold, oldhostmem, size, cudaMemcpyHostToDevice);
      cudaMemcpy(phinew, newhostmem, size, cudaMemcpyHostToDevice);
      */
      //// Kernel Execution ////
      //////////////////////////
      uint3 domainSize = make_uint3(h.x() - l.x(), h.y() - l.y(), h.z() - l.z());
      uint3 domainLower = make_uint3(l.x(), l.y(), l.z());

      int tx = 8;
      int ty = 8;
      int tz = 8;
      int totalBlocks = size / (sizeof(double) * tx * ty * tz);
      
      if (size % size / (sizeof(double) * tx * ty * tz) != 0) {
        totalBlocks++;
      }
      
      dim3 threadsPerBlock(tx, ty, tz);

      // launch kernel
      std::cout << "Prior to kernel." << std::endl;
      //timeAdvanceKernel<<<totalBlocks, threadsPerBlock>>>(domainSize, domainLower, phiold, phinew);
      std::cout << "Post kernel." << std::endl;

      //cudaThreadSynchronize();
      sleep(1000);

      // Memory Deallocation ////
      ///////////////////////////
      // USE PINNING INSTEAD
      //cudaMemcpy(newhostmem, phinew, size, cudaMemcpyDeviceToHost);

      new_dw->put(sum_vartype(residual), residual_label);
    }

  // final device free
  cudaFree(phiold);
  cudaFree(phinew);
}
