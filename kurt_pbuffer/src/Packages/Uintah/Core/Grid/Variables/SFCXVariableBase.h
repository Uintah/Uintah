
#ifndef UINTAH_HOMEBREW_SFCXVariableBase_H
#define UINTAH_HOMEBREW_SFCXVariableBase_H

#include <Packages/Uintah/Core/Grid/Variables/GridVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/constVariable.h>

namespace SCIRun {
  class IntVector;
}

namespace Uintah {

  class BufferInfo;
  class OutputContext;
  class Patch;

/**************************************

CLASS
   SFCXVariableBase
   
   Short description...

GENERAL INFORMATION

   SFCXVariableBase.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   SFCXVariableBase

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class SFCXVariableBase;
  typedef constVariableBase<SFCXVariableBase> constSFCXVariableBase;

   class SFCXVariableBase : public GridVariable {
   public:
      
      virtual ~SFCXVariableBase();
      
      virtual constSFCXVariableBase* cloneConstType() const = 0;     

      virtual void getMPIBuffer(BufferInfo& buffer,
                                const IntVector& low, const IntVector& high);
   protected:
      SFCXVariableBase(const SFCXVariableBase&);
      SFCXVariableBase();
      
   private:
      SFCXVariableBase& operator=(const SFCXVariableBase&);
   };
} // End namespace Uintah

#endif
