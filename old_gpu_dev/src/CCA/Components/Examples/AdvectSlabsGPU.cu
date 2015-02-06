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



#include <CCA/Components/Examples/AdvectSlabsGPU.h>
#include <CCA/Components/Examples/ExamplesLabel.h>
#include <CCA/Ports/LoadBalancer.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/SimpleMaterial.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/Endian.h>
#include <Core/Util/FancyAssert.h>

using namespace Uintah;

AdvectSlabsGPU::AdvectSlabsGPU(const ProcessorGroup* myworld)
: UintahParallelComponent(myworld)
{
  mass_label = VarLabel::create("mass", 
                                CCVariable<double>::getTypeDescription());
  massAdvected_label = VarLabel::create("massAdvected", 
                                        CCVariable<double>::getTypeDescription());

  //__________________________________
  //  outflux/influx slabs
  OF_slab[RIGHT] = RIGHT;         IF_slab[RIGHT]  = LEFT;
  OF_slab[LEFT]  = LEFT;          IF_slab[LEFT]   = RIGHT;
  OF_slab[TOP]   = TOP;           IF_slab[TOP]    = BOTTOM;
  OF_slab[BOTTOM]= BOTTOM;        IF_slab[BOTTOM] = TOP;
  OF_slab[FRONT] = FRONT;         IF_slab[FRONT]  = BACK;
  OF_slab[BACK]  = BACK;          IF_slab[BACK]   = FRONT;

  // Slab adjacent cell
  S_ac[RIGHT]  =  IntVector( 1, 0, 0);
  S_ac[LEFT]   =  IntVector(-1, 0, 0);
  S_ac[TOP]    =  IntVector( 0, 1, 0);
  S_ac[BOTTOM] =  IntVector( 0,-1, 0);
  S_ac[FRONT]  =  IntVector( 0, 0, 1);
  S_ac[BACK]   =  IntVector( 0, 0,-1);
}

AdvectSlabsGPU::~AdvectSlabsGPU()
{

}

void AdvectSlabsGPU::problemSetup(const ProblemSpecP& params,
                                  const ProblemSpecP& restart_prob_spec,
                                  GridP&, SimulationStateP& sharedState)
{
  sharedState_ = sharedState;
  ProblemSpecP ps = params->findBlock("AdvectSlabsGPU");
  ps->require("delt", delt_);
  mymat_ = scinew SimpleMaterial();
  sharedState->registerSimpleMaterial(mymat_);
}

void AdvectSlabsGPU::scheduleInitialize(const LevelP& level,
                                        SchedulerP& sched)
{
  Task* task = scinew Task("initialize",
                           this, &AdvectSlabsGPU::initialize);
  task->computes(mass_label);
  task->computes(massAdvected_label);
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}

void AdvectSlabsGPU::scheduleComputeStableTimestep(const LevelP& level,
                                                   SchedulerP& sched)
{
  Task* task = scinew Task("computeStableTimestep",
                           this, &AdvectSlabsGPU::computeStableTimestep);
  task->computes(sharedState_->get_delt_label(),level.get_rep());
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}

void
AdvectSlabsGPU::scheduleTimeAdvance( const LevelP& level, SchedulerP& sched)
{
  Task* task = scinew Task("timeAdvance",
                           this, &AdvectSlabsGPU::timeAdvance);

  task->requires(Task::OldDW, mass_label, Ghost::AroundCells, 1);
  task->computes(mass_label);
  task->computes(massAdvected_label);
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());

}

void AdvectSlabsGPU::computeStableTimestep(const ProcessorGroup*,
                                           const PatchSubset* patches,
                                           const MaterialSubset*,
                                           DataWarehouse*, DataWarehouse* new_dw)
{
  new_dw->put(delt_vartype(delt_), sharedState_->get_delt_label(),getLevel(patches));
}

void AdvectSlabsGPU::initialize(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse*old_dw, DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    new_dw->allocateTemporary(d_OFS, patch, Ghost::AroundCells,1);
    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);

      CCVariable<double> mass, massAd;
      new_dw->allocateAndPut(mass,   mass_label,         matl, patch, Ghost::AroundCells, 1);
      new_dw->allocateAndPut(massAd, massAdvected_label, matl, patch, Ghost::AroundCells, 1);
      mass.initialize(0.0);
      massAd.initialize(0.0);

      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++)
      {
        // set initial value for fluxes
        for(int face = TOP; face <= BACK; face++ )  {
          d_OFS[*iter].d_fflux[face]= 1;
        }
        // set up the initial mass
        mass[*iter]=1;
      }
    }
  }
}

/// KERNEL FOR TIME ADVANCE
// @brief A kernel that applies the stencil used in timeAdvance(...)
// @param domainSize a three component vector that gives the size of the domain as (x,y,z)
// @param domainLower a three component vector that gives the lower corner of the work area as (x,y,z)
// @param ghostLayers the number of layers of ghost cells
// @param mass pointer to the source mass allocated on the device
// @param massAd pointer to the  massAdvected allocated on the device
// @param OFS pointer to an array of 6 element double arrays
// @param invol inverse of the volume of a single cell
__global__ void timeAdvanceKernelAdvectSlabs(uint3 domainSize,
                                             uint3 domainLower,
                                             int ghostLayers,
                                             double *mass,
                                             double *massAd,
                                             double *newMass,
                                             double *ofs,
                                             double invol) {

  // calculate the thread indices
  int tidX = blockDim.x * blockIdx.x + threadIdx.x;
  int tidY = blockDim.y * blockIdx.y + threadIdx.y;

 
  // Indexing
  int num_slices = domainSize.z - ghostLayers;
  int dx = domainSize.x;
  int dy = domainSize.y;
  int dxOFS = dx+1;
  int dyOFS = dy+1;

  double q_face_flux[6];
  double faceVol[6];

  if (tidX < (dx-1) && tidY < (dy-1) && tidX > 0 && tidY > 0) {
    int tidYminus = tidY - 1;
    int tidYplus  = tidY + 1;
    int tidXminus = tidX - 1;
    int tidXplus  = tidX + 1;
    for (int slice = ghostLayers; slice < num_slices; slice++) {
      // Variables needed for each cell
      double sum_q_face_flux = 0.0;
      int cell2 = INDEX3D(dx,dy, tidX,tidY, slice);
      int cell  = 6*cell2; //INDEX3D(dxOFS,dyOFS, tidX,tidY, slice);
      double influxVol  = 0.0;
      double outfluxVol = 0.0;
      double massCell   = mass[cell2];
      int adjCell;

      // Schematic of layout
      // enum FACE {TOP(0), BOTTOM(1), RIGHT(2), LEFT(3), FRONT(4), BACK(5)};
      //__________________________________
      //  outflux/influx slabs
      //  OF_slab[RIGHT] = RIGHT;         IF_slab[RIGHT]  = LEFT;
      //  OF_slab[LEFT]  = LEFT;          IF_slab[LEFT]   = RIGHT;
      //  OF_slab[TOP]   = TOP;           IF_slab[TOP]    = BOTTOM;
      //  OF_slab[BOTTOM]= BOTTOM;        IF_slab[BOTTOM] = TOP;
      //  OF_slab[FRONT] = FRONT;         IF_slab[FRONT]  = BACK;
      //  OF_slab[BACK]  = BACK;          IF_slab[BACK]   = FRONT;

      // Corresponding CPU code
      // double outfluxVol = d_OFS[c ].d_fflux[OF_slab[f]];
      // double influxVol  = d_OFS[ac].d_fflux[IF_slab[f]];


      // Unrolled 'for' loop
      // Above
      adjCell    = INDEX3D(dxOFS,dyOFS, tidX, tidYplus, slice);
      outfluxVol = ofs[cell];
      influxVol  = ofs[(6*adjCell+1)];

      q_face_flux[0]   = mass[adjCell]*influxVol - massCell*outfluxVol;
      faceVol[0]       = outfluxVol + influxVol;
      sum_q_face_flux += q_face_flux[0];

      // Below
      adjCell    = INDEX3D(dxOFS,dyOFS, tidX, tidYminus, slice);
      outfluxVol = ofs[(cell+1)];
      influxVol  = ofs[(6*adjCell)];

      q_face_flux[1]   = mass[adjCell]*influxVol - massCell*outfluxVol;
      faceVol[1]       = outfluxVol + influxVol;
      sum_q_face_flux += q_face_flux[1];

      // Right
      adjCell    = INDEX3D(dxOFS,dyOFS, tidXplus, tidY, slice);
      outfluxVol = ofs[(cell+2)];
      influxVol  = ofs[(6*adjCell+3)];

      q_face_flux[2]   = mass[adjCell]*influxVol - massCell*outfluxVol;
      faceVol[2]       = outfluxVol + influxVol;
      sum_q_face_flux += q_face_flux[2];

      // Left
      adjCell    = INDEX3D(dxOFS,dyOFS, tidXminus, tidY, slice);
      outfluxVol = ofs[(cell+3)];
      influxVol  = ofs[(6*adjCell+2)];

      q_face_flux[3]   = mass[adjCell]*influxVol - massCell*outfluxVol;
      faceVol[3]       = outfluxVol + influxVol;
      sum_q_face_flux += q_face_flux[3];

      // Front
      adjCell    = INDEX3D(dxOFS,dyOFS, tidX, tidY, (slice-1));
      outfluxVol = ofs[(cell+4)];
      influxVol  = ofs[(6*adjCell+5)];

      q_face_flux[4]   = mass[adjCell]*influxVol - massCell*outfluxVol;
      faceVol[4]       = outfluxVol + influxVol;
      sum_q_face_flux += q_face_flux[4];

      // Back
      adjCell    = INDEX3D(dxOFS,dyOFS, tidX, tidY, (slice+1));
      outfluxVol = ofs[(cell+5)];
      influxVol  = ofs[(6*adjCell+4)];

      q_face_flux[5]   = mass[adjCell]*influxVol - massCell*outfluxVol;
      faceVol[5]       = outfluxVol + influxVol;
      sum_q_face_flux += q_face_flux[5];

      // Sum all the Advected masses and adjust the new mass
      massAd[cell2]  = sum_q_face_flux*invol;
      newMass[cell2] = massCell - massAd[cell2];
    }
  }
}

void AdvectSlabsGPU::timeAdvance(const ProcessorGroup* pg,
                                 const PatchSubset* patches,
                                 const MaterialSubset* matls,
                                 DataWarehouse* old_dw, DataWarehouse* new_dw)
{
  int previousPatchSize = 0;// this is to see if we need to release and reallocate between computations
  int size = 0;
  int ghostLayers = 1;

  // declare device and host memory
  double* mass_device;
  double* newMass_device;
  double* massAd_device;
  double* ofs_device;
  double* massAd_host;
  double* mass_host;
  double* newMass_host;
  double* ofs_host;

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

  struct fflux ff;
  for(int p=0;p<patches->size();p++) {
   // std::cout << "prev patch size: " << previousPatchSize << std::endl;
    const Patch* patch = patches->get(p);
    Vector dx = patch->dCell();
    double invvol = 1.0/(dx.x() * dx.y() * dx.z());

    d_OFS.initialize(ff);
    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);

      // variable to get
      constCCVariable<double> mass;
      CCVariable<double>      mass2;
      CCVariable<double>      massAd;

      old_dw->get(mass, mass_label, matl, patch, Ghost::AroundCells, 1);
      new_dw->allocateAndPut(mass2, mass_label, matl, patch, Ghost::AroundCells, 1 );
      new_dw->allocateAndPut(massAd, massAdvected_label, matl, patch, Ghost::AroundCells, 1 );


      // Here the extents of the patch are extracted and the size of the domain is memory
      // needed is calculated.  Any memory allocation occur here.
      IntVector l = patch->getNodeLowIndex();
      IntVector h = patch->getNodeHighIndex();

      // Sizes //
      // Size of mass and advectedMass variables
      IntVector s = h - l;
      int xdim = s.x(), ydim = s.y(), zdim = s.z();
      size = xdim * ydim * zdim * sizeof(double);
      // Size of the OFS variable
      IntVector ofsSize = d_OFS.getWindow()->getData()->size();
      int sizeOFS = 6*ofsSize.x()*ofsSize.y()*ofsSize.z()*sizeof(double);

      l += IntVector(patch->getBCType(Patch::xminus) == Patch::Neighbor ? 0 : 1,
                     patch->getBCType(Patch::yminus) == Patch::Neighbor ? 0 : 1,
                     patch->getBCType(Patch::zminus) == Patch::Neighbor ? 0 : 1);
      h -= IntVector(patch->getBCType(Patch::xplus) == Patch::Neighbor ? 0 : 1,
                     patch->getBCType(Patch::yplus) == Patch::Neighbor ? 0 : 1,
                     patch->getBCType(Patch::zplus) == Patch::Neighbor ? 0 : 1);

      // check if we need to reallocate
      if (size != previousPatchSize) {
        if (previousPatchSize != 0) {
          cudaFree(mass_device);
          cudaFree(newMass_device);
          cudaFree(massAd_device);
          cudaFree(ofs_device);
        }
        cudaMalloc(&mass_device, size);
        // Kernel error checking
        cudaError_t error = cudaGetLastError();
        if(error!=cudaSuccess) {
          fprintf(stderr,"ERROR1: %s\n", cudaGetErrorString(error) );
          exit(-1);
        }
        cudaMalloc(&massAd_device, size);
        // Kernel error checking
        error = cudaGetLastError();
        if(error!=cudaSuccess) {
          fprintf(stderr,"ERROR3: %s\n", cudaGetErrorString(error) );
          exit(-1);
        }
        cudaMalloc(&ofs_device, sizeOFS);
        // Kernel error checking
        error = cudaGetLastError();
        if(error!=cudaSuccess) {
          fprintf(stderr,"ERROR3: %s\n", cudaGetErrorString(error) );
          exit(-1);
        }
        cudaMalloc(&newMass_device, size);
        // Kernel error checking
        error = cudaGetLastError();
        if(error!=cudaSuccess) {
          fprintf(stderr,"ERROR3: %s\n", cudaGetErrorString(error) );
          exit(-1);
        }
      }

      //__________________________________
      //  Memory Allocation
      mass_host    = (double*)mass.getWindow()->getData()->getPointer();
      newMass_host = (double*)mass2.getWindow()->getData()->getPointer();
      massAd_host  = (double*)massAd.getWindow()->getData()->getPointer();
      ofs_host     = (double*)d_OFS.getWindow()->getData()->getPointer();
#ifdef GJDKLAFJKDLASJCLKDMSALKCJDASKLNGDA
      std::cout << "Dims of OFS: (" << ofsSize.x() <<","<< ofsSize.y() << "," << ofsSize.z() << ")" << std::endl;
      std::cout << "Size: " << ofsSize.x()*ofsSize.y()*ofsSize.z() << std::endl;
      std::cout << "Size of OFS: " <<  ofsSize.x()*ofssize.y()*ofsSize.z()* sizeof(struct fflux) << std::endl;
      std::cout << "Size: " << size/sizeof(double) << " x6= " << 6*size << std::endl;
#endif
      // allocate space on the device
      cudaMemcpy(mass_device, mass_host, size,    cudaMemcpyHostToDevice);
      //cudaMemcpy(ofs_device,  ofs_host,  sizeOFS, cudaMemcpyHostToDevice);


      uint3 domainSize  = make_uint3(xdim, ydim, zdim);
      uint3 domainLower = make_uint3(l.x(), l.y(), l.z());
      // Threads per block must be power of 2 in each direction.  Here
      //  8 is chosen as a test value in the x and y and 1 in the z,
      //  as each of these (x,y) threads streams through the z direction.
      dim3 threadsPerBlock(8, 8, 1);

      // Set up the number of blocks of threads in each direction accounting for any
      //  non-power of 8 end pieces.
      int xBlocks = xdim / 8;
      if( xdim % 8 != 0)
      {
        xBlocks++;
      }
      int yBlocks = ydim / 8;
      if( ydim % 8 != 0)
      {
        yBlocks++;
      }
      dim3 totalBlocks(xBlocks,yBlocks);

      // launch kernel
      timeAdvanceKernelAdvectSlabs<<< totalBlocks, threadsPerBlock >>>(domainSize, domainLower, ghostLayers, mass_device,  massAd_device, newMass_device, ofs_device, invvol);

      // Kernel error checking
      cudaError_t error = cudaGetLastError();
      if(error!=cudaSuccess) {
        fprintf(stderr,"ERROR5: %s\n", cudaGetErrorString(error) );
        exit(-1);
      }

      cudaDeviceSynchronize();
      cudaMemcpy(newMass_host, newMass_device, size, cudaMemcpyDeviceToHost);
      //cudaMemcpy(massAd_host,  massAd_device,  size, cudaMemcpyDeviceToHost);
    }
  }

  // free up allocated memory
  cudaFree(mass_device);
  cudaFree(massAd_device);
  cudaFree(newMass_device);
  cudaFree(ofs_device);
}
