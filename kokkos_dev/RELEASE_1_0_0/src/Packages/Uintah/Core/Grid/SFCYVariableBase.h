
#ifndef UINTAH_HOMEBREW_SFCYVariableBase_H
#define UINTAH_HOMEBREW_SFCYVariableBase_H

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
   SFCYVariableBase
   
   Short description...

GENERAL INFORMATION

   SFCYVariableBase.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   SFCYVariableBase

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class SFCYVariableBase : public Variable {
   public:
      
      virtual ~SFCYVariableBase();
      
      virtual void copyPointer(const SFCYVariableBase&) = 0;
      
      virtual void rewindow(const IntVector& low, const IntVector& high) = 0;

      //////////
      // Insert Documentation Here:
      virtual SFCYVariableBase* clone() const = 0;

      virtual void allocate(const Patch*) = 0;
      virtual void allocate(const IntVector& lowIndex,
			    const IntVector& highIndex) = 0;
      virtual void copyPatch(SFCYVariableBase* src,
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
      SFCYVariableBase(const SFCYVariableBase&);
      SFCYVariableBase();
      
   private:
      SFCYVariableBase& operator=(const SFCYVariableBase&);
   };

} // End namespace Uintah

#endif
