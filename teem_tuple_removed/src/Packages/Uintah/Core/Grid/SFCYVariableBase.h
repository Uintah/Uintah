
#ifndef UINTAH_HOMEBREW_SFCYVariableBase_H
#define UINTAH_HOMEBREW_SFCYVariableBase_H

#include <Packages/Uintah/Core/Grid/Variable.h>
#include <Packages/Uintah/Core/Grid/constVariable.h>

namespace SCIRun {
  class IntVector;
}

namespace Uintah {

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

  class SFCYVariableBase;
  typedef constVariableBase<SFCYVariableBase> constSFCYVariableBase;

   class SFCYVariableBase : public Variable {
   public:
      
      virtual ~SFCYVariableBase();
      
      virtual void copyPointer(SFCYVariableBase&) = 0;
      
      virtual bool rewindow(const IntVector& low, const IntVector& high) = 0;
      virtual void offsetGrid(const IntVector& /*offset*/) = 0;
     
      //////////
      // Insert Documentation Here:
      virtual SFCYVariableBase* clone() = 0;
      virtual const SFCYVariableBase* clone() const = 0;     

      // Make a new default object of the base class.
      virtual SFCYVariableBase* cloneType() const = 0;
      virtual constSFCYVariableBase* cloneConstType() const = 0;

      // Clones the type with a variable having the given extents
      // but with null data -- good as a place holder.
      virtual SFCYVariableBase* makePlaceHolder(const IntVector & low,
						const IntVector & high) const = 0;
     
      virtual void allocate(const Patch*) = 0;
      virtual void allocate(const IntVector& lowIndex,
			    const IntVector& highIndex) = 0;
      virtual void allocate(const SFCYVariableBase* src) = 0;
      virtual void copyPatch(const SFCYVariableBase* src,
			     const IntVector& lowIndex,
			     const IntVector& highIndex) = 0;
      virtual void copyData(const SFCYVariableBase* src) = 0;

      virtual void* getBasePointer() const = 0;
     void getMPIBuffer(BufferInfo& buffer,
			const IntVector& low, const IntVector& high);
      virtual const TypeDescription* virtualGetTypeDescription() const = 0;
      virtual void getSizes(IntVector& low, IntVector& high,
			    IntVector& siz) const = 0;
      virtual void getSizes(IntVector& low, IntVector& high,
			    IntVector& dataLow, IntVector& siz,
			    IntVector& strides) const = 0;
      virtual void getSizeInfo(string& elems, unsigned long& totsize,
			       void*& ptr) const = 0;
      virtual IntVector getLow() = 0;
      virtual IntVector getHigh() = 0;
     virtual RefCounted* getRefCounted() = 0;
   protected:
      SFCYVariableBase(const SFCYVariableBase&);
      SFCYVariableBase();
      
   private:
      SFCYVariableBase& operator=(const SFCYVariableBase&);
   };

} // End namespace Uintah

#endif
