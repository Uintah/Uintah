/*
 * The MIT License
 *
 * Copyright (c) 1997-2025 The University of Utah
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

#ifndef UINTAH_HOMEBREW_DataWarehouse_H
#define UINTAH_HOMEBREW_DataWarehouse_H

#include <Core/Util/Handle.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/Variables/constGridVariable.h>
#include <Core/Grid/Ghost.h>
#include <Core/Util/RefCounted.h>
#include <Core/Grid/Variables/ParticleVariableBase.h>
#include <Core/Grid/Variables/ReductionVariableBase.h>
#include <Core/Grid/Variables/PerPatchBase.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/SoleVariableBase.h>
#include <Core/Grid/Variables/VarLabelMatl.h>
#include <Core/Grid/Variables/VarLabelMatlMemspace.h>
#include <Core/Grid/Task.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <CCA/Ports/SchedulerP.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Vector.h>

#include <sci_defs/gpu_defs.h>

#if defined(KOKKOS_USING_GPU)
#include <CCA/Components/Schedulers/GPUDataWarehouse.h>
#endif

#include <iosfwd>

namespace Uintah {

class Level;
class OutputContext;
#if HAVE_PIDX
class PIDXOutputContext;
#endif
class ProcessorGroup;
class VarLabel;
class Task;

/**************************************

CLASS
   DataWarehouse

   Short description:
   Abstract class that currently only inherits from OnDemandDW

GENERAL INFORMATION

   DataWarehouse.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)


KEYWORDS
   DataWarehouse

DESCRIPTION
   Long description...

WARNING

****************************************/

typedef int atomicDataStatus;
//    0                   1                   2                   3
//    0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
//   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//   |    16-bit reference counter   |  unused     |U|S|D|V|A|V|C|A|A|
//   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

//left sixteen bits is a 16-bit integer reference counter.

//Not allocated/Invalid = If the value is 0x00000000

//Allocating                = bit 31 - 0x00000001
//Allocated                 = bit 30 - 0x00000002
//Copying in                = bit 29 - 0x00000004
//Valid                     = bit 28 - 0x00000008
//Awaiting ghost data       = bit 27 - 0x00000010
//Valid with ghost cells    = bit 26 - 0x00000020
//Deallocating              = bit 25 - 0x00000040
//Superpatch                = bit 24 - 0x00000080
//Unknown                   = bit 23 - 0x00000100

class DataWarehouse : public RefCounted {

public:
  virtual ~DataWarehouse();

  virtual bool exists(const VarLabel*, int matlIndex, const Patch*) const = 0;
  virtual bool exists(const VarLabel*, int matlIndex, const Level*) const = 0;

  // Returns a (const) pointer to the grid.  This pointer can then be
  // used to (for example) get the number of levels in the grid.
  virtual const Grid * getGrid() = 0;

  // Generic put and allocate, passing Variable as a pointer rather than
  // by reference to avoid ambiguity with other put overloaded methods.
  virtual void put(Variable*, const VarLabel*, int matlIndex,
                   const Patch*) = 0;

  virtual void scrub( const VarLabel * label
                    ,       int        matlIndex = -1
                    , const void     * domain = nullptr) = 0;

  // Reduction Variables
  virtual void get(ReductionVariableBase&, const VarLabel*,
                   const Level* level = 0, int matlIndex = -1) = 0;

  virtual std::map<int,double> get_sum_vartypeD( const VarLabel * label,
                                                 const MaterialSubset * ) = 0;

  virtual std::map<int,Vector> get_sum_vartypeV( const VarLabel * label,
                                                 const MaterialSubset * ) = 0;

  virtual void put(const ReductionVariableBase&, const VarLabel*,
                   const Level* level = 0, int matlIndex = -1) = 0;

  virtual void put_sum_vartype( std::map<int,Vector>,
                                const VarLabel *,
                                const MaterialSubset *) = 0;

  virtual void put_sum_vartype( std::map<int,double>,
                                const VarLabel *,
                                const MaterialSubset *) = 0;

  virtual void override(const ReductionVariableBase&, const VarLabel*,
                        const Level* level = 0, int matlIndex = -1) = 0;
  virtual void print(std::ostream& intout, const VarLabel* label,
                     const Level* level, int matlIndex = -1) = 0;

  // Sole Variables
  virtual bool exists(const VarLabel*) const = 0;
  virtual void get(SoleVariableBase&, const VarLabel*,
                   const Level* level = 0, int matlIndex = -1) = 0;
  virtual void put(const SoleVariableBase&, const VarLabel*,
                   const Level* level = 0, int matlIndex = -1) = 0;

  virtual void override(const SoleVariableBase&, const VarLabel*,
                        const Level* level = 0, int matlIndex = -1) = 0;

  virtual void doReserve() = 0;

  // Particle Variables
  // Changed way PS's were stored from ghost info to low-high range.
  // we can still keep the getPS function API the same though to not annoy
  // everybody -- BJW, May 05
  virtual ParticleSubset* createParticleSubset( particleIndex numParticles,
                                                int matlIndex, const Patch*,
                                                IntVector low = IntVector(0,0,0),
                                                IntVector high = IntVector(0,0,0) ) = 0;

  virtual void deleteParticleSubset( ParticleSubset* psubset ) = 0;

  virtual void saveParticleSubset(ParticleSubset* psubset,
                                  int matlIndex, const Patch*,
                                  IntVector low = IntVector(0,0,0),
                                  IntVector high = IntVector(0,0,0)) = 0;
  virtual bool haveParticleSubset(int matlIndex, const Patch*,
                                  IntVector low = IntVector(0,0,0),
                                  IntVector high = IntVector(0,0,0), bool exact = false) = 0;
  virtual ParticleSubset* getParticleSubset(int matlIndex, const Patch*,
                                            IntVector low, IntVector high) = 0;
  virtual ParticleSubset* getParticleSubset(int matlIndex, const Patch*) = 0;
  virtual ParticleSubset* getDeleteSubset(int matlIndex, const Patch*) = 0;
  virtual std::map<const VarLabel*, ParticleVariableBase*>* getNewParticleState(int matlIndex, const Patch*) = 0;
  virtual ParticleSubset* getParticleSubset(int matlIndex, const Patch*,
                                            Ghost::GhostType,
                                            int numGhostCells,
                                            const VarLabel* posvar) = 0;
  virtual ParticleSubset* getParticleSubset(int matlIndex, IntVector low, IntVector high,
                                            const Patch* relPatch,
                                            const VarLabel* posvar, const Level* level=0) = 0;
  virtual void allocateTemporary(ParticleVariableBase&,
                                 ParticleSubset*) = 0;
  virtual void allocateAndPut(ParticleVariableBase&, const VarLabel*,
                              ParticleSubset*) = 0;
  virtual void get(constParticleVariableBase&, const VarLabel*,
                   ParticleSubset*) = 0;
  virtual void get(constParticleVariableBase&, const VarLabel*,
                   int matlIndex, const Patch* patch) = 0;
  virtual void getModifiable(ParticleVariableBase&, const VarLabel*,
                             ParticleSubset*) = 0;
  virtual void put(ParticleVariableBase&, const VarLabel*,
                   bool replace = false) = 0;


  virtual void getCopy(ParticleVariableBase&, const VarLabel*, ParticleSubset*) = 0;
  virtual void copyOut(ParticleVariableBase&, const VarLabel*, ParticleSubset*) = 0;

  virtual void print() = 0;
  virtual void clear() = 0;


  virtual ParticleVariableBase* getParticleVariable(const VarLabel*,
                                                    ParticleSubset*) = 0;
  virtual ParticleVariableBase*
  getParticleVariable(const VarLabel*, int matlIndex, const Patch*) = 0;

  // Generic grid based variables
  virtual void get( constGridVariableBase& var,
                    const VarLabel* label, int matlIndex, const Patch* patch,
                    Ghost::GhostType gtype, int numGhostCells ) = 0;

  virtual void getModifiable( GridVariableBase& var,
                              const VarLabel* label, int matlIndex, const Patch* patch, Ghost::GhostType gtype=Ghost::None, int numGhostCells=0 ) = 0;

  virtual void allocateTemporary( GridVariableBase& var, const Patch* patch,
                                  Ghost::GhostType gtype = Ghost::None, int numGhostCells = 0 ) = 0;
//                                const IntVector& boundaryLayer ) = 0;
//                                const IntVector& boundaryLayer = IntVector(0,0,0)) = 0;

  virtual void allocateAndPut( GridVariableBase& var,
                               const VarLabel* label, int matlIndex,
                               const Patch* patch, Ghost::GhostType gtype = Ghost::None,
                               int numGhostCells = 0 ) = 0;

  virtual void put(GridVariableBase& var, const VarLabel* label, int matlIndex, const Patch* patch,
            bool replace = false) = 0;

  // Returns the constGridVariable for all patches on the level
  virtual void getLevel( constGridVariableBase&,
                         const VarLabel*,
                         int matlIndex,
                         const Level* level) = 0;

  virtual void getRegion(constGridVariableBase&, const VarLabel*,
                          int matlIndex, const Level* level,
                          const IntVector& low, const IntVector& high,
                          bool useBoundaryCells = true) = 0;

  // Copy out of the warehouse into an allocated variable.
  virtual void copyOut(GridVariableBase& var, const VarLabel* label, int matlIndex,
               const Patch* patch, Ghost::GhostType gtype = Ghost::None,
               int numGhostCells = 0) = 0;

  // Makes var a copy of the specified warehouse data, allocating it
  // to the appropriate size first.
  virtual void getCopy(GridVariableBase& var, const VarLabel* label, int matlIndex,
               const Patch* patch, Ghost::GhostType gtype = Ghost::None,
               int numGhostCells = 0) = 0;


  // PerPatch Variables
  virtual void get(PerPatchBase&, const VarLabel*,
                   int matlIndex, const Patch*) = 0;
  virtual void put(PerPatchBase&, const VarLabel*,
                   int matlIndex, const Patch*, bool replace = false) = 0;

  // This is so we can get reduction information for regridding
  virtual void getVarLabelMatlLevelTriples(std::vector<VarLabelMatl<Level> >& vars ) const = 0;

  // Remove particles that are no longer relevant
  virtual void deleteParticles(ParticleSubset* delset) = 0;

  // Add particles
  virtual void addParticles(const Patch* patch, int matlIndex,
                            std::map<const VarLabel*, ParticleVariableBase*>* addedstate) = 0;

  // Move stuff to a different data Warehouse
  virtual void transferFrom(DataWarehouse*, const VarLabel*,
          const PatchSubset*, const MaterialSubset*) = 0;

  virtual void transferFrom(DataWarehouse*, const VarLabel*,
          const PatchSubset*, const MaterialSubset*,
                            bool replace) = 0;

  virtual void transferFrom(DataWarehouse*, const VarLabel*,
                            const PatchSubset*, const MaterialSubset*,
                            bool replace, const PatchSubset*) = 0;

  // An overloaded version of transferFrom.  GPU transfers need a stream, and a
  // stream is found in a detailedTask object.
//  virtual void transferFrom(DataWarehouse*, const VarLabel*,
//                            const PatchSubset*, const MaterialSubset*, ExecutionObject& execObj,
//                            bool replace, const PatchSubset*) = 0;

  virtual size_t emit(OutputContext&, const VarLabel* label,
                    int matlIndex, const Patch* patch) = 0;

#if HAVE_PIDX
  virtual void emitPIDX(PIDXOutputContext&,
                        const VarLabel* label,
                        int matlIndex,
                        const Patch* patch,
                        unsigned char* buffer,
                        size_t bufferSize) = 0;
#endif

  // Scrubbing
  enum ScrubMode {
    ScrubNone,
    ScrubComplete,
    ScrubNonPermanent
  };

  virtual ScrubMode setScrubbing(ScrubMode) = 0;

  // For related datawarehouses
  virtual DataWarehouse* getOtherDataWarehouse(Task::WhichDW) = 0;

  // For the schedulers
  virtual bool isFinalized() const = 0;
  virtual void finalize() = 0;
  virtual void unfinalize() = 0;
  virtual void refinalize() = 0;

  // Returns the generation number (id) of this data warehouse.  Id's
  // start at 0. Each subsequent DW's id is one greater.  SetID should
  // only be called by the SimulationController (and only once) if the
  // DW is being used for a restarted simulation.  This allows the DW
  // generation number to be kept in sync with the actual number of
  // timesteps for the restarted simulation.
  int  getID() const { return d_generation; }
  void setID( int id ) { d_generation = id; }

  // For timestep abort/recompute
  virtual bool abortTimeStep() = 0;
  virtual bool recomputeTimeStep() = 0;

  virtual void reduceMPI(const VarLabel* label, const Level* level,
          const MaterialSubset* matls, int nComm) = 0;

#if defined(KOKKOS_USING_GPU)
  GPUDataWarehouse* getGPUDW(int i) const { return d_gpuDWs[i]; }
  GPUDataWarehouse* getGPUDW()      const { return d_gpuDWs[0]; }
#endif

protected:
  DataWarehouse( const ProcessorGroup* myworld,
                 Scheduler* scheduler,
                 int generation );
  // These two things should be removed from here if possible - Steve
  const ProcessorGroup* d_myworld;
  Scheduler* d_scheduler;

  // Generation should be const, but is not as during a restart, the
  // generation number of the first DW is updated from 0 (the default
  // for the first DW) to the correct generation number based on how
  // many previous time steps had taken place before the restart.
  int d_generation;

#if defined(KOKKOS_USING_GPU)
  std::vector<GPUDataWarehouse*> d_gpuDWs;
#endif

private:
  DataWarehouse(const DataWarehouse&);
  DataWarehouse& operator=(const DataWarehouse&);
};

} // End namespace Uintah

#endif
