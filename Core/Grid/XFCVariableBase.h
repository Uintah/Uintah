
#ifndef UINTAH_HOMEBREW_XFCVariableBase_H
#define UINTAH_HOMEBREW_XFCVariableBase_H

#include <Packages/Uintah/Core/Grid/Variable.h>
#include <mpi.h>

namespace SCIRun {
  class IntVector;
}

namespace Uintah {

using namespace SCIRun;

class OutputContext;

/**************************************

CLASS
   XFCVariableBase
   
   Short description...

GENERAL INFORMATION

   XFCVariableBase.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of AFCidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   XFCVariableBase

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class XFCVariableBase : public Variable {
   public:
      
      virtual ~XFCVariableBase();
      
      virtual void copyPointer(const XFCVariableBase&) = 0;
      
      //////////
      // Insert Documentation Here:
      virtual XFCVariableBase* clone() const = 0;

      virtual void allocate(const IntVector& lowIndex,
			    const IntVector& highIndex) = 0;
      virtual void copyPatch(XFCVariableBase* src,
			      const IntVector& lowIndex,
			      const IntVector& highIndex) = 0;
      virtual void emit(OutputContext&) = 0;
   protected:
      XFCVariableBase(const XFCVariableBase&);
      XFCVariableBase();
      
   private:
      XFCVariableBase& operator=(const XFCVariableBase&);
   };
} // End namespace Uintah

#endif
