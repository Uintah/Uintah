#ifndef UINTAH_HOMEBREW_DataWarehouse_H
#define UINTAH_HOMEBREW_DataWarehouse_H

#include <Packages/Uintah/Core/ProblemSpec/Handle.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/CCVariableBase.h>
#include <Packages/Uintah/Core/Grid/Ghost.h>
#include <Packages/Uintah/Core/ProblemSpec/RefCounted.h>
#include <Packages/Uintah/Core/Grid/ParticleVariableBase.h>
#include <Packages/Uintah/Core/Grid/NCVariableBase.h>
#include <Packages/Uintah/Core/Grid/SFCXVariableBase.h>
#include <Packages/Uintah/Core/Grid/SFCYVariableBase.h>
#include <Packages/Uintah/Core/Grid/SFCZVariableBase.h>
#include <Packages/Uintah/Core/Grid/ReductionVariableBase.h>
#include <Packages/Uintah/Core/Grid/PerPatchBase.h>
#include <Packages/Uintah/Core/Grid/ComputeSet.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
//#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/CCA/Ports/SchedulerP.h>
#include <Core/Geometry/IntVector.h>

#include <iosfwd>

namespace SCIRun {
  class Vector;
}

namespace Uintah {

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
      
   class DataWarehouse : public RefCounted {

   public:
      virtual ~DataWarehouse();
      
      virtual bool exists(const VarLabel*, int matlIndex, const Patch*) const =0;

      // Generic put, passing Variable as a pointer rather than by reference
      // to avoid ambiguity with other put overloaded methods.
      virtual void put(Variable*, const VarLabel*, int matlIndex,
		       const Patch*) = 0;
 
      // Reduction Variables
      virtual void allocate(ReductionVariableBase&, const VarLabel*,
			    int matlIndex = -1) = 0;
      virtual void get(ReductionVariableBase&, const VarLabel*,
		       int matlIndex = -1) = 0;
      virtual void put(const ReductionVariableBase&, const VarLabel*,
		       int matlIndex = -1) = 0;
      virtual void override(const ReductionVariableBase&, const VarLabel*,
			    int matlIndex = -1) = 0;

      // Particle Variables
      virtual ParticleSubset* createParticleSubset(particleIndex numParticles,
				        int matlIndex, const Patch*) = 0;
      virtual void saveParticleSubset(int matlIndex, const Patch*,
				      ParticleSubset* psubset) = 0;
      virtual bool haveParticleSubset(int matlIndex, const Patch*) = 0;
      virtual ParticleSubset* getParticleSubset(int matlIndex,
					const Patch*) = 0;
      virtual ParticleSubset* getParticleSubset(int matlIndex,
			 const Patch*, Ghost::GhostType, int numGhostCells,
			 const VarLabel* posvar) = 0;
      virtual void allocate(ParticleVariableBase&, const VarLabel*,
			    ParticleSubset*) = 0;
      virtual void get(constParticleVariableBase&, const VarLabel*,
		       ParticleSubset*) = 0;
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

      // Generic grid based variables
     
      // Node Centered (NC) Variables
      virtual void allocate(NCVariableBase&, const VarLabel*,
			    int matlIndex, const Patch*,
			    Ghost::GhostType = Ghost::None,
			    int numGhostCells = 0) = 0;
      virtual void get(constNCVariableBase&, const VarLabel*, int matlIndex,
		       const Patch*, Ghost::GhostType, int numGhostCells) = 0;
      virtual void getModifiable(NCVariableBase&, const VarLabel*,
				 int matlIndex, const Patch*) = 0;
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
      virtual void allocate(CCVariableBase&, const VarLabel*,
			    int matlIndex, const Patch*, 
			    Ghost::GhostType = Ghost::None,
			    int numGhostCells = 0) = 0;
      virtual void get(constCCVariableBase&, const VarLabel*, int matlIndex,
		       const Patch*, Ghost::GhostType, int numGhostCells) = 0;
      virtual void getModifiable(CCVariableBase&, const VarLabel*,
				 int matlIndex, const Patch*) = 0;
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
      virtual void allocate(SFCXVariableBase&, const VarLabel*,
			    int matlIndex, const Patch*,
			    Ghost::GhostType = Ghost::None,
			    int numGhostCells = 0) = 0;
      virtual void get(constSFCXVariableBase&, const VarLabel*, int matlIndex,
		       const Patch*, Ghost::GhostType, int numGhostCells) = 0;
      virtual void getModifiable(SFCXVariableBase&, const VarLabel*,
				 int matlIndex, const Patch*) = 0;
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

      virtual void allocate(SFCYVariableBase&, const VarLabel*,
			    int matlIndex, const Patch*,
			    Ghost::GhostType = Ghost::None,
			    int numGhostCells = 0) = 0;
      virtual void get(constSFCYVariableBase&, const VarLabel*, int matlIndex,
		       const Patch*, Ghost::GhostType, int numGhostCells) = 0;
      virtual void getModifiable(SFCYVariableBase&, const VarLabel*,
				 int matlIndex, const Patch*) = 0;
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

      virtual void allocate(SFCZVariableBase&, const VarLabel*,
			    int matlIndex, const Patch*,
			    Ghost::GhostType = Ghost::None,
			    int numGhostCells = 0) = 0;
      virtual void get(constSFCZVariableBase&, const VarLabel*, int matlIndex,
		       const Patch*, Ghost::GhostType, int numGhostCells) = 0;
      virtual void getModifiable(SFCZVariableBase&, const VarLabel*,
				 int matlIndex, const Patch*) = 0;
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
     
      // Remove particles that are no longer relevant
      virtual void deleteParticles(ParticleSubset* delset) = 0;


      virtual void emit(OutputContext&, const VarLabel* label,
			int matlIndex, const Patch* patch) = 0;

      virtual void print(ostream& intout, const VarLabel* label,
			 int matlIndex = -1) = 0;

      // For the schedulers
      virtual bool isFinalized() const = 0;
      virtual bool exists(const VarLabel*, const Patch*) const = 0;
      virtual void finalize() = 0;

      // For sanity checking.
      // Must be called by the thread that will run the test.
      virtual void setCurrentTask(const Task* task) = 0;
      virtual void checkTasksAccesses(const PatchSubset* patches,
				      const MaterialSubset* matls) = 0;
     
      int getID() const {
	 return d_generation;
      }

      static bool show_warnings;     
   protected:
      DataWarehouse(const ProcessorGroup* myworld,
		    const Scheduler* scheduler, int generation);
      // These two things should be removed from here if possible - Steve
      const ProcessorGroup* d_myworld;
      const Scheduler* d_scheduler;
      const int d_generation;
     
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
