
#ifndef UINTAH_HOMEBREW_YFCVariableBase_H
#define UINTAH_HOMEBREW_YFCVariableBase_H

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
   YFCVariableBase
   
   Short description...

GENERAL INFORMATION

   YFCVariableBase.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of AFCidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   YFCVariableBase

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class YFCVariableBase : public Variable {
   public:
      
      virtual ~YFCVariableBase();
      
      virtual void copyPointer(const YFCVariableBase&) = 0;
      
      //////////
      // Insert Documentation Here:
      virtual YFCVariableBase* clone() const = 0;

      virtual void allocate(const IntVector& lowIndex,
			    const IntVector& highIndex) = 0;
      virtual void copyPatch(YFCVariableBase* src,
			      const IntVector& lowIndex,
			      const IntVector& highIndex) = 0;
      virtual void emit(OutputContext&) = 0;
   protected:
      YFCVariableBase(const YFCVariableBase&);
      YFCVariableBase();
      
   private:
      YFCVariableBase& operator=(const YFCVariableBase&);
   };
} // End namespace Uintah

#endif
