
#ifndef UINTAH_HOMEBREW_NCVariableBase_H
#define UINTAH_HOMEBREW_NCVariableBase_H

#include <Packages/Uintah/Core/Grid/Variable.h>

namespace SCIRun {
  class IntVector;
}

namespace Uintah {
  using namespace SCIRun;

  class BufferInfo;
  class OutputContext;
  class Patch;

/**************************************

CLASS
   NCVariableBase
   
   Short description...

GENERAL INFORMATION

   NCVariableBase.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   NCVariableBase

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class NCVariableBase : public Variable {
   public:
      
      virtual ~NCVariableBase();
      
      virtual void copyPointer(const NCVariableBase&) = 0;

      virtual void rewindow(const IntVector& low, const IntVector& high) = 0;
      
      //////////
      // Insert Documentation Here:
      virtual NCVariableBase* clone() const = 0;

      virtual void allocate(const Patch* patch) = 0;
      virtual void allocate(const IntVector& lowIndex,
			    const IntVector& highIndex) = 0;
      virtual void copyPatch(const NCVariableBase* src,
			      const IntVector& lowIndex,
			      const IntVector& highIndex) = 0;
      virtual void* getBasePointer() = 0;
     void getMPIBuffer(BufferInfo& buffer,
		       const IntVector& low, const IntVector& high);
      virtual const TypeDescription* virtualGetTypeDescription() const = 0;
      virtual void getSizes(IntVector& low, IntVector& high,
			    IntVector& siz, IntVector& strides) const = 0;
     virtual RefCounted* getRefCounted() = 0;
   protected:
      NCVariableBase(const NCVariableBase&);
      NCVariableBase();
      
   private:
      NCVariableBase& operator=(const NCVariableBase&);
   };
} // End namespace Uintah

#endif
