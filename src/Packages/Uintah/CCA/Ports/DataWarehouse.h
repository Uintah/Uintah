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
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
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
      virtual void put(const Variable*, const VarLabel*, int matlIndex,
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
      virtual void get(ParticleVariableBase&, const VarLabel*,
		       ParticleSubset*) = 0;
      virtual void put(const ParticleVariableBase&, const VarLabel*,
		       bool replace = false) = 0;
      inline void modify(const ParticleVariableBase& var,
			 const VarLabel* label)
      { put(var, label, true); }
     
      virtual ParticleVariableBase* getParticleVariable(const VarLabel*,
							ParticleSubset*) = 0;
      
      // Node Centered (NC) Variables
      virtual void allocate(NCVariableBase&, const VarLabel*,
			    int matlIndex, const Patch*,
			    const IntVector gc = IntVector(0,0,0)) = 0;
      virtual void get(NCVariableBase&, const VarLabel*, int matlIndex,
		       const Patch*, Ghost::GhostType, int numGhostCells) = 0;
      virtual void put(const NCVariableBase&, const VarLabel*,
		       int matlIndex, const Patch*, bool replace = false) = 0;
      inline void modify(const NCVariableBase& var, const VarLabel* label,
		       int matlIndex, const Patch* patch)
      { put(var, label, matlIndex, patch, true); }
      
      // Cell Centered (CC) Variables
      virtual void allocate(CCVariableBase&, const VarLabel*,
			    int matlIndex, const Patch*, 
			    const IntVector gc = IntVector(0,0,0)) = 0;
      virtual void get(CCVariableBase&, const VarLabel*, int matlIndex,
		       const Patch*, Ghost::GhostType, int numGhostCells) = 0;
      virtual void put(const CCVariableBase&, const VarLabel*,
		       int matlIndex, const Patch*, bool replace = false) = 0;
      inline void modify(const CCVariableBase& var, const VarLabel* label,
			  int matlIndex, const Patch* patch)
      { put(var, label, matlIndex, patch, true); }

      // Staggered Variables in all three directions (SFCX, SFCY, SFCZ)
      virtual void allocate(SFCXVariableBase&, const VarLabel*,
			    int matlIndex, const Patch*) = 0;
      virtual void get(SFCXVariableBase&, const VarLabel*, int matlIndex,
		       const Patch*, Ghost::GhostType, int numGhostCells) = 0;
      virtual void put(const SFCXVariableBase&, const VarLabel*,
		       int matlIndex, const Patch*, bool replace = false) = 0;
      inline void modify(const SFCXVariableBase& var, const VarLabel* label,
			  int matlIndex, const Patch* patch)
      { put(var, label, matlIndex, patch, true); }

      virtual void allocate(SFCYVariableBase&, const VarLabel*,
			    int matlIndex, const Patch*) = 0;
      virtual void get(SFCYVariableBase&, const VarLabel*, int matlIndex,
		       const Patch*, Ghost::GhostType, int numGhostCells) = 0;
      virtual void put(const SFCYVariableBase&, const VarLabel*,
		       int matlIndex, const Patch*, bool replace = false) = 0;
      inline void modify(const SFCYVariableBase& var, const VarLabel* label,
			 int matlIndex, const Patch* patch)
      { put(var, label, matlIndex, patch, true); }

      virtual void allocate(SFCZVariableBase&, const VarLabel*,
			    int matlIndex, const Patch*) = 0;
      virtual void get(SFCZVariableBase&, const VarLabel*, int matlIndex,
		       const Patch*, Ghost::GhostType, int numGhostCells) = 0;
      virtual void put(const SFCZVariableBase&, const VarLabel*,
		       int matlIndex, const Patch*, bool replace = false) = 0;
      inline void modify(const SFCZVariableBase& var, const VarLabel* label,
			  int matlIndex, const Patch* patch)
      { put(var, label, matlIndex, patch, true); }

      // PerPatch Variables
      virtual void get(PerPatchBase&, const VarLabel*,
				int matlIndex, const Patch*) = 0;
      virtual void put(const PerPatchBase&, const VarLabel*,
		       int matlIndex, const Patch*, bool replace = false) = 0;
      inline void modify(const PerPatchBase& var, const VarLabel* label,
			  int matlIndex, const Patch* patch)
      { put(var, label, matlIndex, patch, true); }
     
      // Remove particles that are no longer relevant
      virtual void deleteParticles(ParticleSubset* delset) = 0;


      virtual void emit(OutputContext&, const VarLabel* label,
			int matlIndex, const Patch* patch) const = 0;

      virtual void print(ostream& intout, const VarLabel* label,
			 int matlIndex = -1) const = 0;

      // For the schedulers
      virtual bool isFinalized() const = 0;
      virtual bool exists(const VarLabel*, const Patch*) const = 0;
      virtual void finalize() = 0;

      int getID() const {
	 return d_generation;
      }
   protected:
      DataWarehouse(const ProcessorGroup* myworld, int generation);
      // These two things should be removed from here if possible - Steve
      const ProcessorGroup* d_myworld;
      const int d_generation;

      
   private:
      
      DataWarehouse(const DataWarehouse&);
      DataWarehouse& operator=(const DataWarehouse&);
   };
} // End namespace Uintah

#endif
