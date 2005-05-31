
#ifndef UINTAH_HOMEBREW_SFCYVariableBase_H
#define UINTAH_HOMEBREW_SFCYVariableBase_H

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

   class SFCYVariableBase : public GridVariable {
   public:
      
      virtual ~SFCYVariableBase();
      
      virtual constSFCYVariableBase* cloneConstType() const = 0;

      virtual void getMPIBuffer(BufferInfo& buffer,
                                const IntVector& low, const IntVector& high);
   protected:
      SFCYVariableBase(const SFCYVariableBase&);
      SFCYVariableBase();
      
   private:
      SFCYVariableBase& operator=(const SFCYVariableBase&);
   };

} // End namespace Uintah

#endif
