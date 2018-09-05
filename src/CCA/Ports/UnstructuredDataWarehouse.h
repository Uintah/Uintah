/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

#ifndef UINTAH_HOMEBREW_UnstructuredDataWarehouse_H
#define UINTAH_HOMEBREW_UnstructuredDataWarehouse_H

#include <Core/Util/Handle.h>
#include <Core/Grid/UnstructuredGridP.h>
#include <Core/Grid/Variables/constUnstructuredGridVariable.h>
#include <Core/Grid/Ghost.h>
#include <Core/Util/RefCounted.h>
#include <Core/Grid/Variables/UnstructuredParticleVariableBase.h>
#include <Core/Grid/Variables/UnstructuredReductionVariableBase.h>
#include <Core/Grid/Variables/UnstructuredPerPatchBase.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/UnstructuredSoleVariableBase.h>
#include <Core/Grid/Variables/UnstructuredVarLabelMatl.h>
#include <Core/Grid/UnstructuredTask.h>
#include <CCA/Ports/UnstructuredDataWarehouseP.h>
#include <CCA/Ports/UnstructuredSchedulerP.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Vector.h>
#include <sci_defs/cuda_defs.h>
#ifdef HAVE_CUDA
#include <CCA/Components/Schedulers/GPUDataWarehouse.h>
#endif

#include <iosfwd>


namespace Uintah {

class UnstructuredLevel;
class OutputContext;
#if HAVE_PIDX
class PIDXOutputContext;
#endif
class ProcessorGroup;
class UnstructuredVarLabel;
class UnstructuredTask;

/**************************************
	
CLASS
   UnstructuredDataWarehouse
	
   Short description...
	
GENERAL INFORMATION
	
   UnstructuredDataWarehouse.h
	
   Steven G. Parker
   Department of Computer Science
   University of Utah
	
   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
	
   	
KEYWORDS
   UnstructuredDataWarehouse
	
DESCRIPTION
   Long description...
	
WARNING
	
****************************************/
      
class UnstructuredDataWarehouse : public RefCounted {

public:
  virtual ~UnstructuredDataWarehouse();
      
  virtual bool exists(const UnstructuredVarLabel*, int matlIndex, const UnstructuredPatch*) const = 0;
  virtual bool exists(const UnstructuredVarLabel*, int matlIndex, const UnstructuredLevel*) const = 0;

  virtual UnstructuredReductionVariableBase* getReductionVariable( const UnstructuredVarLabel*,
						       int matlIndex,
						       const UnstructuredLevel* ) const = 0;
  
  // Returns a (const) pointer to the grid.  This pointer can then be
  // used to (for example) get the number of levels in the grid.
  virtual const UnstructuredGrid * getGrid() = 0;

  // Generic put and allocate, passing Variable as a pointer rather than
  // by reference to avoid ambiguity with other put overloaded methods.
  virtual void put(UnstructuredVariable*, const UnstructuredVarLabel*, int matlIndex,
		   const UnstructuredPatch*) = 0;

  // Reduction Variables
  virtual void get(UnstructuredReductionVariableBase&, const UnstructuredVarLabel*,
		   const UnstructuredLevel* level = 0, int matlIndex = -1) = 0;
  virtual void put(const UnstructuredReductionVariableBase&, const UnstructuredVarLabel*,
		   const UnstructuredLevel* level = 0, int matlIndex = -1) = 0;

  virtual void override(const UnstructuredReductionVariableBase&, const UnstructuredVarLabel*,
			const UnstructuredLevel* level = 0, int matlIndex = -1) = 0;
  virtual void print(std::ostream& intout, const UnstructuredVarLabel* label,
		     const UnstructuredLevel* level, int matlIndex = -1) = 0;

  // Sole Variables
  virtual bool exists(const UnstructuredVarLabel*) const = 0;
  virtual void get(UnstructuredSoleVariableBase&, const UnstructuredVarLabel*,
		   const UnstructuredLevel* level = 0, int matlIndex = -1) = 0;
  virtual void put(const UnstructuredSoleVariableBase&, const UnstructuredVarLabel*,
		   const UnstructuredLevel* level = 0, int matlIndex = -1) = 0;

  virtual void override(const UnstructuredSoleVariableBase&, const UnstructuredVarLabel*,
			const UnstructuredLevel* level = 0, int matlIndex = -1) = 0;

  virtual void doReserve() = 0; 

  // Particle Variables
  // changed way PS's were stored from ghost info to low-high range.
  // we can still keep the getPS function API the same though to not annoy
  // everybody -- BJW, May 05
  virtual UnstructuredParticleSubset* createParticleSubset(  particleIndex numParticles,
                                                 int matlIndex, const UnstructuredPatch*,
                                                 IntVector low = IntVector(0,0,0),
                                                 IntVector high = IntVector(0,0,0) ) = 0;
  virtual void saveParticleSubset(UnstructuredParticleSubset* psubset,
                                  int matlIndex, const UnstructuredPatch*,
                                  IntVector low = IntVector(0,0,0),
                                  IntVector high = IntVector(0,0,0)) = 0;
  virtual bool haveParticleSubset(int matlIndex, const UnstructuredPatch*,
                                  IntVector low = IntVector(0,0,0),
                                  IntVector high = IntVector(0,0,0), bool exact = false) = 0;
  virtual UnstructuredParticleSubset* getParticleSubset(int matlIndex, const UnstructuredPatch*,
                                            IntVector low, IntVector high) = 0;
  virtual UnstructuredParticleSubset* getParticleSubset(int matlIndex, const UnstructuredPatch*) = 0;
  virtual UnstructuredParticleSubset* getDeleteSubset(int matlIndex, const UnstructuredPatch*) = 0;
  virtual std::map<const UnstructuredVarLabel*, UnstructuredParticleVariableBase*>* getNewParticleState(int matlIndex, const UnstructuredPatch*) = 0;
  virtual UnstructuredParticleSubset* getParticleSubset(int matlIndex, const UnstructuredPatch*,
					    Ghost::GhostType,
					    int numGhostCells,
					    const UnstructuredVarLabel* posvar) = 0;
  virtual UnstructuredParticleSubset* getParticleSubset(int matlIndex, IntVector low, IntVector high,
                                            const UnstructuredPatch* relPatch,
                                            const UnstructuredVarLabel* posvar, const UnstructuredLevel* level=0) = 0;
  virtual void allocateTemporary(UnstructuredParticleVariableBase&,
				 UnstructuredParticleSubset*) = 0;
  virtual void allocateAndPut(UnstructuredParticleVariableBase&, const UnstructuredVarLabel*,
			      UnstructuredParticleSubset*) = 0;
  virtual void get(constUnstructuredParticleVariableBase&, const UnstructuredVarLabel*,
		   UnstructuredParticleSubset*) = 0;
  virtual void get(constUnstructuredParticleVariableBase&, const UnstructuredVarLabel*,
		   int matlIndex, const UnstructuredPatch* patch) = 0;
  virtual void getModifiable(UnstructuredParticleVariableBase&, const UnstructuredVarLabel*,
			     UnstructuredParticleSubset*) = 0;
  virtual void put(UnstructuredParticleVariableBase&, const UnstructuredVarLabel*,
		   bool replace = false) = 0;


  virtual void getCopy(UnstructuredParticleVariableBase&, const UnstructuredVarLabel*, UnstructuredParticleSubset*) = 0;
  virtual void copyOut(UnstructuredParticleVariableBase&, const UnstructuredVarLabel*, UnstructuredParticleSubset*) = 0;

  virtual void print() = 0;
  virtual void clear() = 0;


  virtual UnstructuredParticleVariableBase* getParticleVariable(const UnstructuredVarLabel*,
						    UnstructuredParticleSubset*) = 0;
  virtual UnstructuredParticleVariableBase*
  getParticleVariable(const UnstructuredVarLabel*, int matlIndex, const UnstructuredPatch*) = 0;

  // Generic grid based variables

  virtual void get( constUnstructuredGridVariableBase& var,
                    const UnstructuredVarLabel* label, int matlIndex, const UnstructuredPatch* patch,
                    Ghost::GhostType gtype, int numGhostCells ) = 0;

  virtual void getModifiable( UnstructuredGridVariableBase& var,
                              const UnstructuredVarLabel* label, int matlIndex, const UnstructuredPatch* patch, Ghost::GhostType gtype=Ghost::None, int numGhostCells=0 ) = 0;

  virtual void allocateTemporary( UnstructuredGridVariableBase& var, const UnstructuredPatch* patch,
                                  Ghost::GhostType gtype = Ghost::None, int numGhostCells = 0 ) = 0;
//                                  const IntVector& boundaryLayer ) = 0;
//                                const IntVector& boundaryLayer = IntVector(0,0,0)) = 0;

  virtual void allocateAndPut( UnstructuredGridVariableBase& var,
                               const UnstructuredVarLabel* label, int matlIndex,
                               const UnstructuredPatch* patch, Ghost::GhostType gtype = Ghost::None,
                               int numGhostCells = 0 ) = 0;

  virtual void put(UnstructuredGridVariableBase& var, const UnstructuredVarLabel* label, int matlIndex, const UnstructuredPatch* patch,
            bool replace = false) = 0;

  // returns the constUnstructuredGridVariable for all patches on the level
  virtual void getLevel( constUnstructuredGridVariableBase&, 
                         const UnstructuredVarLabel*,
                         int matlIndex, 
                         const UnstructuredLevel* level) = 0;

  virtual void getRegion(constUnstructuredGridVariableBase&, const UnstructuredVarLabel*,
                          int matlIndex, const UnstructuredLevel* level,
                          const IntVector& low, const IntVector& high,
                          bool useBoundaryCells = true) = 0;

  // Copy out of the warehouse into an allocated variable.
  virtual void copyOut(UnstructuredGridVariableBase& var, const UnstructuredVarLabel* label, int matlIndex,
	       const UnstructuredPatch* patch, Ghost::GhostType gtype = Ghost::None,
	       int numGhostCells = 0) = 0;

  // Makes var a copy of the specified warehouse data, allocating it
  // to the appropriate size first.
  virtual void getCopy(UnstructuredGridVariableBase& var, const UnstructuredVarLabel* label, int matlIndex,
	       const UnstructuredPatch* patch, Ghost::GhostType gtype = Ghost::None,
	       int numGhostCells = 0) = 0;
      

  // PerPatch Variables
  virtual void get(UnstructuredPerPatchBase&, const UnstructuredVarLabel*,
		   int matlIndex, const UnstructuredPatch*) = 0;
  virtual void put(UnstructuredPerPatchBase&, const UnstructuredVarLabel*,
		   int matlIndex, const UnstructuredPatch*, bool replace = false) = 0;
     
  // this is so we can get reduction information for regridding
  virtual void getVarLabelMatlLevelTriples(std::vector<UnstructuredVarLabelMatl<UnstructuredLevel> >& vars ) const = 0;

  // Remove particles that are no longer relevant
  virtual void deleteParticles(UnstructuredParticleSubset* delset) = 0;

  // Add particles
  virtual void addParticles(const UnstructuredPatch* patch, int matlIndex,
			    std::map<const UnstructuredVarLabel*, UnstructuredParticleVariableBase*>* addedstate) = 0;

  // Move stuff to a different data Warehouse
  virtual void transferFrom(UnstructuredDataWarehouse*, const UnstructuredVarLabel*,
          const UnstructuredPatchSubset*, const MaterialSubset*) = 0;

  virtual void transferFrom(UnstructuredDataWarehouse*, const UnstructuredVarLabel*,
          const UnstructuredPatchSubset*, const MaterialSubset*,
                            bool replace) = 0;

  virtual void transferFrom(UnstructuredDataWarehouse*, const UnstructuredVarLabel*,
			    const UnstructuredPatchSubset*, const MaterialSubset*,
                            bool replace, const UnstructuredPatchSubset*) = 0;

  //An overloaded version of transferFrom.  GPU transfers need a stream, and a
  //stream is found in a detailedTask object.
  virtual void transferFrom(UnstructuredDataWarehouse*, const UnstructuredVarLabel*,
                            const UnstructuredPatchSubset*, const MaterialSubset*, void * detailedTask,
                            bool replace, const UnstructuredPatchSubset*) = 0;

  virtual size_t emit(OutputContext&, const UnstructuredVarLabel* label,
		    int matlIndex, const UnstructuredPatch* patch) = 0;

#if HAVE_PIDX
  virtual void emitPIDX(PIDXOutputContext&, 
                        const UnstructuredVarLabel* label, 
                        int matlIndex, 
                        const UnstructuredPatch* patch, 
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
  virtual UnstructuredDataWarehouse* getOtherDataWarehouse(UnstructuredTask::WhichDW) = 0;

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

  // For timestep abort/restart
  virtual bool timestepAborted() = 0;
  virtual bool timestepRestarted() = 0;
  virtual void abortTimestep() = 0;
  virtual void restartTimestep() = 0;

  virtual void reduceMPI(const UnstructuredVarLabel* label, const UnstructuredLevel* level,
	  const MaterialSubset* matls, int nComm) = 0;

#ifdef HAVE_CUDA
  GPUDataWarehouse* getGPUDW(int i) const { return d_gpuDWs[i]; }
  GPUDataWarehouse* getGPUDW() const {
    int i;
    CUDA_RT_SAFE_CALL(cudaGetDevice(&i));
    return d_gpuDWs[i]; 
  }
#endif
protected:
  UnstructuredDataWarehouse( const ProcessorGroup* myworld,
		 UnstructuredScheduler* scheduler, 
		 int generation );
  // These two things should be removed from here if possible - Steve
  const ProcessorGroup* d_myworld;
  UnstructuredScheduler* d_scheduler;

  // Generation should be const, but is not as during a restart, the
  // generation number of the first DW is updated from 0 (the default
  // for the first DW) to the correct generation number based on how
  // many previous time steps had taken place before the restart.
  int d_generation;
  
#ifdef HAVE_CUDA
  std::vector<GPUDataWarehouse*> d_gpuDWs;
#endif
private:
  UnstructuredDataWarehouse(const UnstructuredDataWarehouse&);
  UnstructuredDataWarehouse& operator=(const UnstructuredDataWarehouse&);
};

} // End namespace Uintah

#endif
