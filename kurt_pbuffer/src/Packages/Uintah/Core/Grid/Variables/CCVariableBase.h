
#ifndef UINTAH_HOMEBREW_CCVariableBase_H
#define UINTAH_HOMEBREW_CCVariableBase_H

#include <Packages/Uintah/Core/Grid/Variables/GridVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/constVariable.h>

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

   class CCVariableBase : public GridVariable {
   public:
      
      virtual ~CCVariableBase();
      
      // Make a new default object of the base class.
      virtual constCCVariableBase* cloneConstType() const = 0;

      virtual void getMPIBuffer(BufferInfo& buffer,
                                const IntVector& low, const IntVector& high);

   protected:
      CCVariableBase(const CCVariableBase&);
      CCVariableBase();
      
   private:
      CCVariableBase& operator=(const CCVariableBase&);
   };
} // End namespace Uintah

#endif
