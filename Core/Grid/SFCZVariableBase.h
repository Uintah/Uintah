
#ifndef UINTAH_HOMEBREW_SFCZVariableBase_H
#define UINTAH_HOMEBREW_SFCZVariableBase_H

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
   SFCZVariableBase
   
   Short description...

GENERAL INFORMATION

   SFCZVariableBase.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   SFCZVariableBase

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class SFCZVariableBase : public Variable {
   public:
      
      virtual ~SFCZVariableBase();
      
      virtual void copyPointer(const SFCZVariableBase&) = 0;
      
      virtual void rewindow(const IntVector& low, const IntVector& high) = 0;

      //////////
      // Insert Documentation Here:
      virtual SFCZVariableBase* clone() const = 0;

      virtual void allocate(const Patch*) = 0;
      virtual void allocate(const IntVector& lowIndex,
			    const IntVector& highIndex) = 0;
      virtual void copyPatch(SFCZVariableBase* src,
			      const IntVector& lowIndex,
			      const IntVector& highIndex) = 0;
      void getMPIBuffer(BufferInfo& buffer,
			const IntVector& low, const IntVector& high);
      virtual void* getBasePointer() = 0;
      virtual const TypeDescription* virtualGetTypeDescription() const = 0;
      virtual void getSizes(IntVector& low, IntVector& high, IntVector& siz) const = 0;
      virtual void getSizes(IntVector& low, IntVector& high,
			    IntVector& siz, IntVector& strides) const = 0;

     virtual RefCounted* getRefCounted() = 0;
   protected:
      SFCZVariableBase(const SFCZVariableBase&);
      SFCZVariableBase();
      
   private:
      SFCZVariableBase& operator=(const SFCZVariableBase&);
   };

} // End namespace Uintah

#endif
