
#ifndef UINTAH_HOMEBREW_CCVariableBase_H
#define UINTAH_HOMEBREW_CCVariableBase_H

#include <Packages/Uintah/Core/Grid/Variable.h>

namespace SCIRun {
  class IntVector;
}

namespace Uintah {
  using SCIRun::IntVector;

  class BufferInfo;
  class OutputContext;
  class Patch;

/**************************************

CLASS
   CCVariableBase
   
   Short description...

GENERAL INFORMATION

   CCVariableBase.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   CCVariableBase

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class CCVariableBase : public Variable {
   public:
      
      virtual ~CCVariableBase();
      
      virtual void copyPointer(const CCVariableBase&) = 0;
      
      virtual void rewindow(const IntVector& low, const IntVector& high) = 0;

      //////////
      // Insert Documentation Here:
      virtual CCVariableBase* clone() const = 0;

      virtual void allocate(const Patch* patch) = 0;
      virtual void allocate(const IntVector& lowIndex,
			    const IntVector& highIndex) = 0;
      virtual void copyPatch(CCVariableBase* src,
			      const IntVector& lowIndex,
			      const IntVector& highIndex) = 0;
      virtual void* getBasePointer() = 0;
     void getMPIBuffer(BufferInfo& buffer,
		       const IntVector& low, const IntVector& high);
      virtual const TypeDescription* virtualGetTypeDescription() const = 0;
      virtual void getSizes(IntVector& low, IntVector& high, IntVector& siz) const = 0;
      virtual void getSizes(IntVector& low, IntVector& high,
			    IntVector& siz, IntVector& strides) const = 0;
   protected:
      CCVariableBase(const CCVariableBase&);
      CCVariableBase();
      
   private:
      CCVariableBase& operator=(const CCVariableBase&);
   };
} // End namespace Uintah

#endif
