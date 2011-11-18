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



#include <CCA/Components/Examples/HeatEquationGPU.h>
#include <CCA/Components/Examples/ExamplesLabel.h>
#include <CCA/Ports/LoadBalancer.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/SimpleMaterial.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Malloc/Allocator.h>

#include <sci_defs/cuda_defs.h>

using namespace Uintah;

HeatEquationGPU::HeatEquationGPU(const ProcessorGroup* myworld)
  : UintahParallelComponent(myworld)
{
  temperature_label = VarLabel::create("temperature", 
                               NCVariable<double>::getTypeDescription());
  residual_label = VarLabel::create("residual", 
                                    sum_vartype::getTypeDescription());
}

HeatEquationGPU::~HeatEquationGPU()
{
  VarLabel::destroy(temperature_label);
  VarLabel::destroy(residual_label);
}

void HeatEquationGPU::problemSetup(const ProblemSpecP& params,
                            const ProblemSpecP& restart_prob_spec,
                            GridP&, SimulationStateP& sharedState)
{
  sharedState_ = sharedState;
  ProblemSpecP heateqn = params->findBlock("HeatEquationGPU");
  heateqn->require("delt", delt_);
  heateqn->require("maxresidual", maxresidual_);
  mymat_ = scinew SimpleMaterial();
  sharedState->registerSimpleMaterial(mymat_);
}
 
void HeatEquationGPU::scheduleInitialize(const LevelP& level,
			       SchedulerP& sched)
{
  Task* task = scinew Task("initialize",
			   this, &HeatEquationGPU::initialize);
  task->computes(temperature_label);
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}
 
void HeatEquationGPU::scheduleComputeStableTimestep(const LevelP& level,
					  SchedulerP& sched)
{
  Task* task = scinew Task("computeStableTimestep",
			   this, &HeatEquationGPU::computeStableTimestep);
  task->computes(sharedState_->get_delt_label(),level.get_rep());
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}

void
HeatEquationGPU::scheduleTimeAdvance( const LevelP& level, SchedulerP& sched)
{
  Task* task = scinew Task("timeAdvance",
			   this, &HeatEquationGPU::timeAdvance);

  task->requires(Task::OldDW, temperature_label, Ghost::AroundNodes, 1);
  task->computes(temperature_label);
  task->computes(residual_label);
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());

}

void HeatEquationGPU::computeStableTimestep(const ProcessorGroup*,
				  const PatchSubset* patches,
				  const MaterialSubset*,
				  DataWarehouse*, DataWarehouse* new_dw)
{
  new_dw->put(delt_vartype(delt_), sharedState_->get_delt_label(),getLevel(patches));
}

void HeatEquationGPU::initialize(const ProcessorGroup*,
		       const PatchSubset* patches,
		       const MaterialSubset* matls,
		       DataWarehouse*, DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);

      NCVariable<double> temperature;
      new_dw->allocateAndPut(temperature, temperature_label, matl, patch);
      temperature.initialize(0);

      if(patch->getBCType(Patch::xminus) != Patch::Neighbor){
	IntVector l,h;
	patch->getFaceNodes(Patch::xminus, 0, l, h);

	for(NodeIterator iter(l,h); !iter.done(); iter++)
	  temperature[*iter]=1;
      }
    }
  }
}

void HeatEquationGPU::timeAdvance(const ProcessorGroup* pg,
                               const PatchSubset* patches,
                               const MaterialSubset* matls,
                               DataWarehouse* old_dw, DataWarehouse* new_dw)
{

  // timing and GPU variables'
  clock_t before = 0.0;
  clock_t after = 0.0;
  int gpuDeviceNum = -1;
  char name[MPI_MAX_PROCESSOR_NAME]; // max proc. name length of 256
  int length;

  // get total devices
  int device_count = 0;
  CUDA_SAFE_CALL(cudaGetDeviceCount(&device_count));


  // start timing
  before = clock();

  gpuDeviceNum = rank % PPN;
  unsigned int timer = 0;
  double gpuTime = 0.0;

  /*
   * NOTE:
   * All kernel launches are asynchronous, as are memory-copy functions with the Async suffix on their names.
   * Therefore, to accurately measure the elapsed time for a particular call or sequence of CUDA calls, it is
   * necessary to synchronize the CPU thread with the GPU by calling cudaThreadSynchronize() immediately before
   * starting and stopping the CPU timer. 
   */
  CUT_SAFE_CALL(cutCreateTimer(&timer))
  CUDA_SAFE_CALL(cudaThreadSynchronize());
  CUT_SAFE_CALL(cutResetTimer(timer));

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);
      constNCVariable<double> temperature;

      old_dw->get(temperature, temperature_label, matl, patch, 
                  Ghost::AroundNodes, 1);

      NCVariable<double> newtemperature;

      new_dw->allocateAndPut(newtemperature, temperature_label, matl, patch);
      newtemperature.copyPatch(temperature, newtemperature.getLow(), 
                               newtemperature.getHigh());

      double residual=0;
      IntVector l = patch->getNodeLowIndex();
      IntVector h = patch->getNodeHighIndex(); 

      l += IntVector(patch->getBCType(Patch::xminus) == Patch::Neighbor?0:1,
		     patch->getBCType(Patch::yminus) == Patch::Neighbor?0:1,
		     patch->getBCType(Patch::zminus) == Patch::Neighbor?0:1);
      h -= IntVector(patch->getBCType(Patch::xplus) == Patch::Neighbor?0:1,
		     patch->getBCType(Patch::yplus) == Patch::Neighbor?0:1,
		     patch->getBCType(Patch::zplus) == Patch::Neighbor?0:1);

      delt_vartype dt;
      old_dw->get(dt, sharedState_->get_delt_label());
      Vector dx = patch->getLevel()->dCell();
      Vector diffusion_number(1./(dx.x()*dx.x()), 1./(dx.y()*dx.y()),
                              1./(dx.z()*dx.z()));
      
      double k = .5;

      cout << "dx = " << dx << endl;
      diffusion_number = diffusion_number* k*dt;
      cout << "diffusion_number = " << diffusion_number << endl;


      // Find the number of blocks to launch
      double bx = h.x() - l.x();
      double by = h.y() - l.y();
      double bz = h.z() - l.z();
      // go with a default threadblock size of 512
      int xsize = 8;
      int ysize = 8;
      int zsize = 8;
      int bbx = (int)ceil(bx/xsize);
      int bby = (int)ceil(by/ysize);
      int bbz = (int)ceil(bz/zsize); 
      
      int totalBlocks = bbx*bby*bbz;

      // Beginning timing of the GPU code
      CUT_SAFE_CALL(cutStartTimer(timer));

      // Get the flat memory from the Node iterator
      int memSize = bx*by*bz*sizeof(double);	// get the total size of the memory needed on the device--JOE add a check here for greater than device memory size
      double *hostMemory = (double*)malloc(memSize);
      double *host2Memory = (double*)malloc(memSize);
      double *host3Memory = (double*)malloc(memSize);
      double *deviceMemory;

      // LATER LETS USE PINNED OR STREAMING MEMORY TO PREVENT THIS
      //  ALSO NEED TO EXPOSE UNDERLYING MEMORY OF THE NCVARIABLE
      for(int i = 0; i < memSize; i++)
      {
         hostMemory[i] = temperature[i];
      }

      CUDA_SAFE_CALL(cudaMalloc((void**) &deviceMem, memSize*3));
      CUDA_SAFE_CALL(cudaMemcpy(deviceMemory, hostMemory, memSize, cudaMemcpyHostToDevice));
      CUDA_SAFE_CALL(cudaMemcpy(deviceMemory, host3Memory, memSize, cudaMemcpyHostToDevice)); // create memory for the residuals
      // Call the temperature kernel

      heat_kernel<<<totalBlocks, xsize*ysize*zsize>>>(hostMemory, host2Memory, host3Memory, 
                                                      xsize,ysize,zsize, 
                                                      bx, by, bz,
                                                      h.x(), h.y(), h.z());
       
      // Get memory back and put it in the nodes
      CUDA_SAFE_CALL(cudaMemcpy(host2Memory, deviceMemory, memSize, cudaMemcpyDeviceToHost));
      CUDA_SAFE_CALL(cudaMemcpy(host3Memory, deviceMemory, memSize, cudaMemcpyDeviceToHost));	// copy back the residuals (must be a better way to do this)

      CUDA_SAFE_CALL(cudaFree(deviceMem));

      // calculate and report time on GPU (per MPI process)
      CUT_SAFE_CALL(cutStopTimer(timer));
      gpuTime = cutGetTimerValue(timer);

      // reduce all the residuals NOTE: there must be a better way to do this using CUDA
      for(int i = 0; i < memSize; i++)
      {
        residual += host3Memory[i];
        newtemperature[i] = host2Memory[i]; // LATER LETS USE PINNED OR STREAMING MEMORY TO PREVENT THIS
      }
      free(hostMemory);
      free(host2Memory);
      free(host3Memory);

      new_dw->put(sum_vartype(residual), residual_label);
    }
  }

  // end timing
  after = clock();
  total_time = (float) (after - before) / CLOCKS_PER_SEC;
}

/// @brief Computes the heat kernel for each block via a 3D blocking kernel
///
/// 
/// @param heatold the pointer to heats to compute with
/// @param heatnew the pointer to heats to compute to
/// @param residual a place to store the residual?
///
/// @param dx the size of the threadblock (and thus memory) in x
/// @param dy the size of the threadblock in y
/// @param dz the size of the threadblock in z
///
/// @param delx the delx of the memory
/// @param dely the dely of the memory
/// @param delz the delz of the memory
/// @param hx the highest index in x
/// @param hy the highest index in y
/// @param hz the highest index in z
__global__ void heat_kernel(double *heatold, double *heatnew, double *residual, double dx, double dy, double dz, int delx, int dely, int delz, double hx, double hy, double hz)
{
  // get the indicies for offset in memory
  int i = threadIdx.x;
  int j = threadIdx.y;
  int k = threadIdx.z;  
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int bz = blockIdx.z;

  int baseX = bx*dx + i;
  int baseY = by*dy + j;
  int baseZ = bz*dz + k;

  // compute the heat and residual
  // check if we are on the last block
  if(baseX > hx || baseY > hy || baseZ > hz) 
  {
     
  } else {
    double base = baseX+delY*baseY+delZ*baseZ;

    heatnew[base] = 1.0/6.0*(heatold[baseX-1+delY*baseY+delZ*baseZ]+heatold[baseX+1+delY*baseY+delZ*baseZ]
                            +heatold[baseX+delY*(baseY-1)+delZ*baseZ]+heatold[baseX+delY*(baseY+1)+delZ*baseZ]
                            +heatold[baseX+delY*baseY+delZ*(baseZ-1)]+heatold[baseX+delY*baseY+delZ*(baseZ+1)]);
    residual[base] = (heatnew[base]-heatold[base])*(heatnew[base]-heatold[base]);
  } 
}

