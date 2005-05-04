
#ifndef UINTAH_HOMEBREW_SFCZVariableBase_H
#define UINTAH_HOMEBREW_SFCZVariableBase_H

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
   SFCZVariableBase
   
   Short description...

GENERAL INFORMATION

   SFCZVariableBase.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   SFCZVariableBase

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class SFCZVariableBase;
  typedef constVariableBase<SFCZVariableBase> constSFCZVariableBase;

   class SFCZVariableBase : public GridVariable {
   public:
      
      virtual ~SFCZVariableBase();
      
      virtual constSFCZVariableBase* cloneConstType() const = 0;

      virtual void getMPIBuffer(BufferInfo& buffer,
                                const IntVector& low, const IntVector& high);
   protected:
      SFCZVariableBase(const SFCZVariableBase&);
      SFCZVariableBase();
      
   private:
      SFCZVariableBase& operator=(const SFCZVariableBase&);
   };

} // End namespace Uintah

#endif
