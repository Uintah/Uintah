
#ifndef UINTAH_HOMEBREW_NCVariableBase_H
#define UINTAH_HOMEBREW_NCVariableBase_H

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

  class NCVariableBase;
  typedef constVariableBase<NCVariableBase> constNCVariableBase;

   class NCVariableBase : public Variable {
   public:
      
      virtual ~NCVariableBase();
      
      virtual void copyPointer(NCVariableBase&) = 0;

      virtual bool rewindow(const IntVector& low, const IntVector& high) = 0;
      virtual void offsetGrid(const IntVector& /*offset*/) = 0;
     
      //////////
      // Insert Documentation Here:
      virtual NCVariableBase* clone() = 0;
      virtual const NCVariableBase* clone() const = 0;     

      // Make a new default object of the base class.
      virtual NCVariableBase* cloneType() const = 0;
      virtual constNCVariableBase* cloneConstType() const = 0;

      // Clones the type with a variable having the given extents
      // but with null data -- good as a place holder.
      virtual NCVariableBase* makePlaceHolder(const IntVector & low,
					      const IntVector & high) const = 0;     

      virtual void allocate(const Patch* patch) = 0;
      virtual void allocate(const IntVector& lowIndex,
			    const IntVector& highIndex) = 0;
      virtual void allocate(const NCVariableBase* src) = 0;
      virtual void copyPatch(const NCVariableBase* src,
			     const IntVector& lowIndex,
			     const IntVector& highIndex) = 0;
      virtual void copyData(const NCVariableBase* src) = 0;


      virtual void* getBasePointer() const = 0;
      void getMPIBuffer(BufferInfo& buffer,
		       const IntVector& low, const IntVector& high);
      virtual const TypeDescription* virtualGetTypeDescription() const = 0;
      virtual void getSizes(IntVector& low, IntVector& high,
			    IntVector& dataLow, IntVector& siz,
			    IntVector& strides) const = 0;
      virtual void getSizeInfo(string& elems, unsigned long& totsize,
			  void*& ptr) const = 0;
      virtual IntVector getLow() = 0;
      virtual IntVector getHigh() = 0;
     virtual RefCounted* getRefCounted() = 0;
   protected:
      NCVariableBase(const NCVariableBase&);
      NCVariableBase();
      
   private:
      NCVariableBase& operator=(const NCVariableBase&);
   };
} // End namespace Uintah

#endif
