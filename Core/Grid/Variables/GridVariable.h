#ifndef UINTAH_HOMEBREW_GRIDVARIABLE_H
#define UINTAH_HOMEBREW_GRIDVARIABLE_H

#include <Packages/Uintah/Core/Grid/Variables/Variable.h>
#include <Packages/Uintah/Core/Parallel/BufferInfo.h>
namespace Uintah {

  using SCIRun::IntVector;

  /**************************************

CLASS
   GridVariable
   
   Short description...

GENERAL INFORMATION

   GridVariable.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   GridVariable

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class GridVariable : public Variable {
  public:
    GridVariable() {}
    virtual ~GridVariable() {};
      
    virtual bool rewindow(const IntVector& low, const IntVector& high) = 0;
    virtual void offset(const IntVector& offset) = 0;

    virtual GridVariable* cloneType() const = 0;
    virtual void allocate(const IntVector& lowIndex, const IntVector& highIndex) = 0;
    virtual void allocate(const Patch* patch, const IntVector& boundary) = 0;
    virtual void allocate(const GridVariable* src) = 0;
    
    virtual void getMPIBuffer(BufferInfo& buffer,
                              const IntVector& low, const IntVector& high) = 0;

    // Clones the type with a variable having the given extents
    // but with null data -- good as a place holder.
    virtual GridVariable* makePlaceHolder(const IntVector & low,
                                            const IntVector & high) const = 0;
    
    virtual void getSizes(IntVector& low, IntVector& high,
                          IntVector& siz) const = 0;
    virtual void getSizes(IntVector& low, IntVector& high,
			    IntVector& dataLow, IntVector& siz,
			    IntVector& strides) const = 0;
    //////////
    // Insert Documentation Here:
    virtual void copyPatch(const GridVariable* src,
			   const IntVector& lowIndex,
			   const IntVector& highIndex) = 0;
    
    virtual void copyData(const GridVariable* src) = 0;
    
    virtual void* getBasePointer() const = 0;
    virtual IntVector getLow() = 0;
    virtual IntVector getHigh() = 0;

    virtual const GridVariable* clone() const = 0;
    virtual GridVariable* clone() = 0;

  };

} // namespace Uintah
#endif
