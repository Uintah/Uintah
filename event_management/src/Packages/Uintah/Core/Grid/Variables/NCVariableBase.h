
#ifndef UINTAH_HOMEBREW_NCVariableBase_H
#define UINTAH_HOMEBREW_NCVariableBase_H

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

   class NCVariableBase : public GridVariable {
   public:
      
      virtual ~NCVariableBase();
      
      virtual constNCVariableBase* cloneConstType() const = 0;

      virtual void getMPIBuffer(BufferInfo& buffer,
                                const IntVector& low, const IntVector& high);

   protected:
      NCVariableBase(const NCVariableBase&);
      NCVariableBase();
      
   private:
      NCVariableBase& operator=(const NCVariableBase&);
   };
} // End namespace Uintah

#endif
