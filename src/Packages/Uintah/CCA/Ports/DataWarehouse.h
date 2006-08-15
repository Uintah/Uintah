#ifndef UINTAH_HOMEBREW_DataWarehouse_H
#define UINTAH_HOMEBREW_DataWarehouse_H

#include <Packages/Uintah/Core/Util/Handle.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariableBase.h>
#include <Packages/Uintah/Core/Grid/Ghost.h>
#include <Packages/Uintah/Core/Util/RefCounted.h>
#include <Packages/Uintah/Core/Grid/Variables/ParticleVariableBase.h>
#include <Packages/Uintah/Core/Grid/Variables/NCVariableBase.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCXVariableBase.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCYVariableBase.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCZVariableBase.h>
#include <Packages/Uintah/Core/Grid/Variables/ReductionVariableBase.h>
#include <Packages/Uintah/Core/Grid/Variables/PerPatchBase.h>
#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>
#include <Packages/Uintah/Core/Grid/Variables/SoleVariableBase.h>
#include <Packages/Uintah/Core/Grid/Variables/VarLabelMatl.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/CCA/Ports/SchedulerP.h>
#include <Core/Geometry/IntVector.h>

#include <sgi_stl_warnings_off.h>
#include <iosfwd>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {
  class Vector;
}

#include <Packages/Uintah/CCA/Ports/share.h>

namespace Uintah {

class Level;
class OutputContext;
class ProcessorGroup;
class VarLabel;
class Task;

/**************************************
	
CLASS
   DataWarehouse
	
   Short description...
	
GENERAL INFORMATION
	
   DataWarehouse.h
	
   Steven G. Parker
   Department of Computer Science
   University of Utah
	
   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
	
   Copyright (C) 2000 SCI Group
	
KEYWORDS
   DataWarehouse
	
DESCRIPTION
   Long description...
	
WARNING
	
****************************************/
      
class SCISHARE DataWarehouse : public RefCounted {

public:
  virtual ~DataWarehouse();
      
  virtual bool exists(const VarLabel*, int matlIndex, const Patch*) const = 0;

  // Returns a (const) pointer to the grid.  This pointer can then be
  // used to (for example) get the number of levels in the grid.
  virtual const Grid * getGrid() = 0;

  // Generic put and allocate, passing Variable as a pointer rather than
  // by reference to avoid ambiguity with other put overloaded methods.
  virtual void put(Variable*, const VarLabel*, int matlIndex,
		   const Patch*) = 0;
  virtual void allocateAndPutGridVar(Variable*, const VarLabel*, 
				     int matlIndex, const Patch*) = 0;


  // Reduction Variables
  virtual void get(ReductionVariableBase&, const VarLabel*,
		   const Level* level = 0, int matlIndex = -1) = 0;
  virtual void put(const ReductionVariableBase&, const VarLabel*,
		   const Level* level = 0, int matlIndex = -1) = 0;

  virtual void override(const ReductionVariableBase&, const VarLabel*,
			const Level* level = 0, int matlIndex = -1) = 0;
  virtual void print(ostream& intout, const VarLabel* label,
		     const Level* level, int matlIndex = -1) = 0;

  // Sole Variables
  virtual void get(SoleVariableBase&, const VarLabel*,
		   const Level* level = 0, int matlIndex = -1) = 0;
  virtual void put(const SoleVariableBase&, const VarLabel*,
		   const Level* level = 0, int matlIndex = -1) = 0;

  virtual void override(const SoleVariableBase&, const VarLabel*,
			const Level* level = 0, int matlIndex = -1) = 0;


  // Particle Variables
  // changed way PS's were stored from ghost info to low-high range.
  // we can still keep the getPS function API the same though to not annoy
  // everybody -- BJW, May 05
  virtual ParticleSubset* createParticleSubset(particleIndex numParticles,
					       int matlIndex, const Patch*,
                                               IntVector low = IntVector(0,0,0),
                                               IntVector high = IntVector(0,0,0)) = 0;
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
  virtual map<const VarLabel*, ParticleVariableBase*>* getNewParticleState(int matlIndex, const Patch*) = 0;
  virtual ParticleSubset* getParticleSubset(int matlIndex, const Patch*, 
					    Ghost::GhostType, 
					    int numGhostCells,
					    const VarLabel* posvar) = 0;
  virtual void allocateTemporary(ParticleVariableBase&,
				 ParticleSubset*) = 0;
  virtual void allocateAndPut(ParticleVariableBase&, const VarLabel*,
			      ParticleSubset*) = 0;
  virtual void get(constParticleVariableBase&, const VarLabel*,
		   ParticleSubset*) = 0;
  virtual void get(constParticleVariableBase&, const VarLabel*,
		   int matlIndex, const Patch* patch) = 0;
  void getCopy(ParticleVariableBase&, const VarLabel*,
	       ParticleSubset*);
  void copyOut(ParticleVariableBase&, const VarLabel*,
	       ParticleSubset*);
  virtual void getModifiable(ParticleVariableBase&, const VarLabel*,
			     ParticleSubset*) = 0;
  virtual void put(ParticleVariableBase&, const VarLabel*,
		   bool replace = false) = 0;
     
  virtual ParticleVariableBase* getParticleVariable(const VarLabel*,
						    ParticleSubset*) = 0;
  virtual ParticleVariableBase*
  getParticleVariable(const VarLabel*, int matlIndex, const Patch*) = 0;

  // Generic grid based variables

  void copyOutGridData(Variable* var, const VarLabel* label, int matlIndex,
		       const Patch* patch,
		       Ghost::GhostType gtype = Ghost::None,
		       int numGhostCells = 0);
     
     
  // Node Centered (NC) Variables
  virtual void allocateTemporary(NCVariableBase&, const Patch*,
				 Ghost::GhostType = Ghost::None,
				 int numGhostCells = 0,
				 const IntVector& boundaryLayer = IntVector(0,0,0)) = 0;
  virtual void allocateAndPut(NCVariableBase&, const VarLabel*,
			      int matlIndex, const Patch*,
			      Ghost::GhostType = Ghost::None,
			      int numGhostCells = 0) = 0;
  virtual void get(constNCVariableBase&, const VarLabel*, int matlIndex,
		   const Patch*, Ghost::GhostType, int numGhostCells) = 0;
  virtual void getModifiable(NCVariableBase&, const VarLabel*,
			     int matlIndex, const Patch*) = 0;
  virtual void getRegion(constNCVariableBase&, const VarLabel*,
  			 int matlIndex, const Level* level,
  			 const IntVector& low, const IntVector& high,
                         bool useBoundaryCells = true) = 0;
  void copyOut(NCVariableBase& var, const VarLabel* label, int matlIndex,
	       const Patch* patch, Ghost::GhostType gtype = Ghost::None,
	       int numGhostCells = 0)
  { copyOut_template(var, label, matlIndex, patch, gtype, numGhostCells); }
  void getCopy(NCVariableBase& var, const VarLabel* label, int matlIndex,
	       const Patch* patch, Ghost::GhostType gtype = Ghost::None,
	       int numGhostCells = 0)
  { getCopy_template(var, label, matlIndex, patch, gtype, numGhostCells); }
  virtual void put(NCVariableBase&, const VarLabel*,
		   int matlIndex, const Patch*, bool replace = false) = 0;
      
  // Cell Centered (CC) Variables
  virtual void allocateTemporary(CCVariableBase&, const Patch*, 
				 Ghost::GhostType = Ghost::None,
				 int numGhostCells = 0,
				 const IntVector& boundaryLayer = IntVector(0,0,0)) = 0;
  virtual void allocateAndPut(CCVariableBase&, const VarLabel*,
			      int matlIndex, const Patch*, 
			      Ghost::GhostType = Ghost::None,
			      int numGhostCells = 0) = 0;
  virtual void get(constCCVariableBase&, const VarLabel*, int matlIndex,
		   const Patch*, Ghost::GhostType, int numGhostCells) = 0;
  virtual void getModifiable(CCVariableBase&, const VarLabel*,
			     int matlIndex, const Patch*) = 0;
  virtual void getRegion(constCCVariableBase&, const VarLabel*,
  			 int matlIndex, const Level* level,
  			 const IntVector& low, const IntVector& high,
                         bool useBoundaryCells = true) = 0;
  void copyOut(CCVariableBase& var, const VarLabel* label, int matlIndex,
	       const Patch* patch, Ghost::GhostType gtype = Ghost::None,
	       int numGhostCells = 0)
  { copyOut_template(var, label, matlIndex, patch, gtype, numGhostCells); }
  void getCopy(CCVariableBase& var, const VarLabel* label, int matlIndex,
	       const Patch* patch, Ghost::GhostType gtype = Ghost::None,
	       int numGhostCells = 0)
  { getCopy_template(var, label, matlIndex, patch, gtype, numGhostCells); }
  virtual void put(CCVariableBase&, const VarLabel*,
		   int matlIndex, const Patch*, bool replace = false) = 0;

  // Staggered Variables in all three directions (SFCX, SFCY, SFCZ)
  virtual void allocateTemporary(SFCXVariableBase&, const Patch*,
				 Ghost::GhostType = Ghost::None,
				 int numGhostCells = 0,
				 const IntVector& boundaryLayer = IntVector(0,0,0)) = 0;
  virtual void allocateAndPut(SFCXVariableBase&, const VarLabel*,
			      int matlIndex, const Patch*,
			      Ghost::GhostType = Ghost::None,
			      int numGhostCells = 0) = 0;
  virtual void get(constSFCXVariableBase&, const VarLabel*, int matlIndex,
		   const Patch*, Ghost::GhostType, int numGhostCells) = 0;
  virtual void getModifiable(SFCXVariableBase&, const VarLabel*,
			     int matlIndex, const Patch*) = 0;
  virtual void getRegion(constSFCXVariableBase&, const VarLabel*,
  			 int matlIndex, const Level* level,
  			 const IntVector& low, const IntVector& high,
                         bool useBoundaryCells = true) = 0;
  void copyOut(SFCXVariableBase& var, const VarLabel* label, int matlIndex,
	       const Patch* patch, Ghost::GhostType gtype = Ghost::None,
	       int numGhostCells = 0)
  { copyOut_template(var, label, matlIndex, patch, gtype, numGhostCells); }
  void getCopy(SFCXVariableBase& var, const VarLabel* label, int matlIndex,
	       const Patch* patch, Ghost::GhostType gtype = Ghost::None,
	       int numGhostCells = 0)
  { getCopy_template(var, label, matlIndex, patch, gtype, numGhostCells); }
  virtual void put(SFCXVariableBase&, const VarLabel*,
		   int matlIndex, const Patch*, bool replace = false) = 0;

  virtual void allocateTemporary(SFCYVariableBase&, const Patch*,
				 Ghost::GhostType = Ghost::None,
				 int numGhostCells = 0,
				 const IntVector& boundaryLayer = IntVector(0,0,0)) = 0;
  virtual void allocateAndPut(SFCYVariableBase&, const VarLabel*,
			      int matlIndex, const Patch*,
			      Ghost::GhostType = Ghost::None,
			      int numGhostCells = 0) = 0;
  virtual void get(constSFCYVariableBase&, const VarLabel*, int matlIndex,
		   const Patch*, Ghost::GhostType, int numGhostCells) = 0;
  virtual void getModifiable(SFCYVariableBase&, const VarLabel*,
			     int matlIndex, const Patch*) = 0;
  virtual void getRegion(constSFCYVariableBase&, const VarLabel*,
  			 int matlIndex, const Level* level,
  			 const IntVector& low, const IntVector& high,
                         bool useBoundaryCells = true) = 0;
  void copyOut(SFCYVariableBase& var, const VarLabel* label, int matlIndex,
	       const Patch* patch, Ghost::GhostType gtype = Ghost::None,
	       int numGhostCells = 0)
  { copyOut_template(var, label, matlIndex, patch, gtype, numGhostCells); }
  void getCopy(SFCYVariableBase& var, const VarLabel* label, int matlIndex,
	       const Patch* patch, Ghost::GhostType gtype = Ghost::None,
	       int numGhostCells = 0)
  { getCopy_template(var, label, matlIndex, patch, gtype, numGhostCells); }
  virtual void put(SFCYVariableBase&, const VarLabel*,
		   int matlIndex, const Patch*, bool replace = false) = 0;

  virtual void allocateTemporary(SFCZVariableBase&, const Patch*,
				 Ghost::GhostType = Ghost::None,
				 int numGhostCells = 0,
				 const IntVector& boundaryLayer = IntVector(0,0,0)) = 0;
  virtual void allocateAndPut(SFCZVariableBase&, const VarLabel*,
			      int matlIndex, const Patch*,
			      Ghost::GhostType = Ghost::None,
			      int numGhostCells = 0) = 0;
  virtual void get(constSFCZVariableBase&, const VarLabel*, int matlIndex,
		   const Patch*, Ghost::GhostType, int numGhostCells) = 0;
  virtual void getModifiable(SFCZVariableBase&, const VarLabel*,
			     int matlIndex, const Patch*) = 0;
  virtual void getRegion(constSFCZVariableBase&, const VarLabel*,
  			 int matlIndex, const Level* level,
  			 const IntVector& low, const IntVector& high,
                         bool useBoundaryCells = true) = 0;
  void copyOut(SFCZVariableBase& var, const VarLabel* label, int matlIndex,
	       const Patch* patch, Ghost::GhostType gtype = Ghost::None,
	       int numGhostCells = 0)
  { copyOut_template(var, label, matlIndex, patch, gtype, numGhostCells); }
  void getCopy(SFCZVariableBase& var, const VarLabel* label, int matlIndex,
	       const Patch* patch, Ghost::GhostType gtype = Ghost::None,
	       int numGhostCells = 0)
  { getCopy_template(var, label, matlIndex, patch, gtype, numGhostCells); }
  virtual void put(SFCZVariableBase&, const VarLabel*,
		   int matlIndex, const Patch*, bool replace = false) = 0;

  // PerPatch Variables
  virtual void get(PerPatchBase&, const VarLabel*,
		   int matlIndex, const Patch*) = 0;
  virtual void put(PerPatchBase&, const VarLabel*,
		   int matlIndex, const Patch*, bool replace = false) = 0;
     
  // this is so we can get reduction information for regridding
  virtual void getVarLabelMatlLevelTriples(vector<VarLabelMatl<Level> >& vars ) const = 0;

  // Remove particles that are no longer relevant
  virtual void deleteParticles(ParticleSubset* delset) = 0;

  // Add particles
  virtual void addParticles(const Patch* patch, int matlIndex,
			    map<const VarLabel*, ParticleVariableBase*>* addedstate) = 0;

  // Move stuff to a different data Warehouse
  virtual void transferFrom(DataWarehouse*, const VarLabel*,
			    const PatchSubset*, const MaterialSubset*,
                            bool replace = false, const PatchSubset* = 0) = 0;

  virtual void emit(OutputContext&, const VarLabel* label,
		    int matlIndex, const Patch* patch) = 0;

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

  // For timestep abort/restart
  virtual bool timestepAborted() = 0;
  virtual bool timestepRestarted() = 0;
  virtual void abortTimestep() = 0;
  virtual void restartTimestep() = 0;
  
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
     
private:
  // Copy out of the warehouse into an allocated variable. 
  template <class Variable>
  void copyOut_template(Variable& var, const VarLabel* label,
			int matlIndex, const Patch* patch,
			Ghost::GhostType gtype = Ghost::None,
			int numGhostCells = 0);

  // Makes var a copy of the specified warehouse data, allocating it
  // to the appropriate size first.
  template <class Variable>
  void getCopy_template(Variable& var, const VarLabel* label,
			int matlIndex, const Patch* patch,
			Ghost::GhostType gtype = Ghost::None,
			int numGhostCells = 0);

  DataWarehouse(const DataWarehouse&);
  DataWarehouse& operator=(const DataWarehouse&);
};

template <class Variable>
void DataWarehouse::copyOut_template(Variable& var, const VarLabel* label,
				     int matlIndex, const Patch* patch,
				     Ghost::GhostType gtype, int numGhostCells)
{
  constVariableBase<Variable>* constVar = var.cloneConstType();
  this->get(*constVar, label, matlIndex, patch, gtype, numGhostCells);
  var.copyData(&constVar->getBaseRep());
  delete constVar;
}

template <class Variable>
void DataWarehouse::getCopy_template(Variable& var, const VarLabel* label,
				     int matlIndex, const Patch* patch,
				     Ghost::GhostType gtype, int numGhostCells)
{
  constVariableBase<Variable>* constVar = var.cloneConstType();
  this->get(*constVar, label, matlIndex, patch, gtype, numGhostCells);
  var.allocate(&constVar->getBaseRep());
  var.copyData(&constVar->getBaseRep());
  delete constVar;
}

} // End namespace Uintah

#endif
