
#ifndef UINTAH_HOMEBREW_FCVariableBase_H
#define UINTAH_HOMEBREW_FCVariableBase_H

#include <Packages/Uintah/Core/Grid/Variable.h>
#include <mpi.h>

namespace SCIRun {
  class IntVector;
}

namespace Uintah {

class OutputContext;

/**************************************

CLASS
   FCVariableBase
   
   Short description...

GENERAL INFORMATION

   FCVariableBase.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of AFCidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   FCVariableBase

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class FCVariableBase : public Variable {
   public:
      
      virtual ~FCVariableBase();
      
      virtual void copyPointer(const FCVariableBase&) = 0;
      
      //////////
      // Insert Documentation Here:
      virtual FCVariableBase* clone() const = 0;

      virtual void allocate(const IntVector& lowIndex,
			    const IntVector& highIndex) = 0;
      virtual void copyPatch(FCVariableBase* src,
			      const IntVector& lowIndex,
			      const IntVector& highIndex) = 0;
   protected:
      FCVariableBase(const FCVariableBase&);
      FCVariableBase();
      
   private:
      FCVariableBase& operator=(const FCVariableBase&);
   };
} // End namespace Uintah

#endif
