
#ifndef UINTAH_HOMEBREW_ParticleVariableBase_H
#define UINTAH_HOMEBREW_ParticleVariableBase_H

#include <Packages/Uintah/Core/Grid/ParticleSubset.h>
#include <Packages/Uintah/Core/Grid/Variable.h>
#include <Packages/Uintah/Core/Grid/constVariable.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
  class BufferInfo;
  class OutputContext;
  class ParticleSubset;
  class ParticleSet;
  class Patch;
  class ProcessorGroup;
  class TypeDescription;

/**************************************

CLASS
   ParticleVariableBase
   
   Short description...

GENERAL INFORMATION

   ParticleVariableBase.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   ParticleVariableBase

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  typedef constVariableBase<ParticleVariableBase> constParticleVariableBase;

   class ParticleVariableBase : public Variable {
   public:
      
      virtual ~ParticleVariableBase();
      virtual void copyPointer(ParticleVariableBase&) = 0;
      
      //////////
      // Insert Documentation Here:
      virtual const ParticleVariableBase* clone() const = 0;
      virtual ParticleVariableBase* clone() = 0;     
      virtual const ParticleVariableBase* cloneSubset(ParticleSubset*) const = 0;
      virtual ParticleVariableBase* cloneSubset(ParticleSubset*) = 0;

      // Make a new default object of the base class.
      virtual ParticleVariableBase* cloneType() const = 0;
      virtual constParticleVariableBase* cloneConstType() const = 0;
     
      virtual void copyData(const ParticleVariableBase* src) = 0;

      virtual void allocate(const Patch*) = 0; // will throw an InternalError
      virtual void allocate(ParticleSubset*) = 0;
      virtual void gather(ParticleSubset* dest,
			  std::vector<ParticleSubset*> subsets,
			  std::vector<ParticleVariableBase*> srcs,
			  particleIndex extra = 0) = 0;
      virtual void gather(ParticleSubset* dest,
			  std::vector<ParticleSubset*> subsets,
			  std::vector<ParticleVariableBase*> srcs,
			  const std::vector<const Patch*>& srcPatches,
			  particleIndex extra = 0) = 0;
      virtual void unpackMPI(void* buf, int bufsize, int* bufpos,
			     const ProcessorGroup* pg,
			     ParticleSubset* pset) = 0;
      virtual void packMPI(void* buf, int bufsize, int* bufpos,
			   const ProcessorGroup* pg,
			   ParticleSubset* pset) = 0;
      virtual void packMPI(void* buf, int bufsize, int* bufpos,
			   const ProcessorGroup* pg,
			   ParticleSubset* pset, const Patch* forPatch) = 0;
      virtual void packsizeMPI(int* bufpos,
			       const ProcessorGroup* pg,
			       ParticleSubset* pset) = 0;

      //////////
      // Insert Documentation Here:
      ParticleSubset* getParticleSubset() const {
	 return d_pset;
      }

      //////////
      // Insert Documentation Here:
      ParticleSet* getParticleSet() const {
	 return d_pset->getParticleSet();
      }
      
      virtual void* getBasePointer() const = 0;
      void getMPIBuffer(BufferInfo& buffer, ParticleSubset* sendset);
      virtual const TypeDescription* virtualGetTypeDescription() const = 0;
     virtual RefCounted* getRefCounted() = 0;
     virtual void getSizeInfo(string& elems, unsigned long& totsize,
			      void*& ptr) const = 0;
   protected:
      ParticleVariableBase(const ParticleVariableBase&);
      ParticleVariableBase(ParticleSubset* pset);
      ParticleVariableBase& operator=(const ParticleVariableBase&);
      
      ParticleSubset*  d_pset;

   private:
   };

} // End namespace Uintah

#endif
