
#ifndef UINTAH_HOMEBREW_ZFCVariableBase_H
#define UINTAH_HOMEBREW_ZFCVariableBase_H

#include <Uintah/Grid/Variable.h>
#include <mpi.h>

namespace SCICore {
   namespace Geometry {
      class IntVector;
   }
}

namespace Uintah {
   class OutputContext;
   class Patch;
   using SCICore::Geometry::IntVector;

/**************************************

CLASS
   ZFCVariableBase
   
   Short description...

GENERAL INFORMATION

   ZFCVariableBase.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of AFCidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   ZFCVariableBase

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class ZFCVariableBase : public Variable {
   public:
      
      virtual ~ZFCVariableBase();
      
      virtual void copyPointer(const ZFCVariableBase&) = 0;
      
      //////////
      // Insert Documentation Here:
      virtual ZFCVariableBase* clone() const = 0;

      virtual void allocate(const IntVector& lowIndex,
			    const IntVector& highIndex) = 0;
      virtual void copyPatch(ZFCVariableBase* src,
			      const IntVector& lowIndex,
			      const IntVector& highIndex) = 0;
      virtual void emit(OutputContext&) = 0;
   protected:
      ZFCVariableBase(const ZFCVariableBase&);
      ZFCVariableBase();
      
   private:
      ZFCVariableBase& operator=(const ZFCVariableBase&);
   };

} // end namespace Uintah

//
// $Log$
// Revision 1.1  2000/11/28 03:47:26  jas
// Added FCVariables for the specific faces X,Y,and Z.
//
//

#endif
