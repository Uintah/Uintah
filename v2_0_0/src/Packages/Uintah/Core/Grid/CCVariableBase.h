
#ifndef UINTAH_HOMEBREW_CCVariableBase_H
#define UINTAH_HOMEBREW_CCVariableBase_H

#include <Packages/Uintah/Core/Grid/Variable.h>
#include <Packages/Uintah/Core/Grid/constVariable.h>

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

  class CCVariableBase;
  typedef constVariableBase<CCVariableBase> constCCVariableBase;

   class CCVariableBase : public Variable {
   public:
      
      virtual ~CCVariableBase();
      
      virtual void copyPointer(CCVariableBase&) = 0;
      
      virtual bool rewindow(const IntVector& low, const IntVector& high) = 0;
      virtual void offsetGrid(const IntVector& /*offset*/) = 0;
     
      //////////
      // Insert Documentation Here:
      virtual CCVariableBase* clone() = 0;
      virtual const CCVariableBase* clone() const = 0;

      // Make a new default object of the base class.
      virtual CCVariableBase* cloneType() const = 0;
      virtual constCCVariableBase* cloneConstType() const = 0;

      // Clones the type with a variable having the given extents
      // but with null data -- good as a place holder.
      virtual CCVariableBase* makePlaceHolder(const IntVector & low,
					      const IntVector & high) const = 0;
     
      virtual void allocate(const Patch* patch) = 0;
      virtual void allocate(const IntVector& lowIndex,
			    const IntVector& highIndex) = 0;
      virtual void allocate(const CCVariableBase* src) = 0;
      virtual void copyPatch(const CCVariableBase* src,
			     const IntVector& lowIndex,
			     const IntVector& highIndex) = 0;
      virtual void copyData(const CCVariableBase* src) = 0;

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
      CCVariableBase(const CCVariableBase&);
      CCVariableBase();
      
   private:
      CCVariableBase& operator=(const CCVariableBase&);
   };
} // End namespace Uintah

#endif
