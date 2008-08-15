#ifndef UINTAH_HOMEBREW_GRIDVARIABLE_H
#define UINTAH_HOMEBREW_GRIDVARIABLE_H

#include <Packages/Uintah/Core/Grid/Variables/Variable.h>
#include <Packages/Uintah/Core/Parallel/BufferInfo.h>
#include <Packages/Uintah/Core/Grid/uintahshare.h>
#include <Core/Geometry/IntVector.h>
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

  class UINTAHSHARE GridVariableBase : public Variable {
  public:
    virtual ~GridVariableBase() {}
      
    virtual bool rewindow(const IntVector& low, const IntVector& high) = 0;
    virtual void offset(const IntVector& offset) = 0;

    virtual GridVariableBase* cloneType() const = 0;

    using Variable::allocate; // Quiets PGI compiler warning about hidden virtual function...
    virtual void allocate(const IntVector& lowIndex, const IntVector& highIndex) = 0;
    virtual void allocate(const GridVariableBase* src) { allocate(src->getLow(), src->getHigh()); }
    
    virtual void getMPIBuffer(BufferInfo& buffer,
                              const IntVector& low, const IntVector& high);

    virtual void getSizes(IntVector& low, IntVector& high,
                          IntVector& siz) const = 0;
    virtual void getSizes(IntVector& low, IntVector& high,
			    IntVector& dataLow, IntVector& siz,
			    IntVector& strides) const = 0;
    //////////
    // Insert Documentation Here:
    virtual void copyPatch(const GridVariableBase* src,
			   const IntVector& lowIndex,
			   const IntVector& highIndex) = 0;
    
    virtual void copyData(const GridVariableBase* src) = 0;
    
    virtual void* getBasePointer() const = 0;
    virtual IntVector getLow() const = 0;
    virtual IntVector getHigh() const = 0;

    virtual const GridVariableBase* clone() const = 0;
    virtual GridVariableBase* clone() = 0;

  protected:
    GridVariableBase() {}
    GridVariableBase(const GridVariableBase&);
  private:
    GridVariableBase& operator=(const GridVariableBase&);    
  };

} // namespace Uintah
#endif
