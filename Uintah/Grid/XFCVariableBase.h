
#ifndef UINTAH_HOMEBREW_XFCVariableBase_H
#define UINTAH_HOMEBREW_XFCVariableBase_H

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

} // end namespace Uintah

//
// $Log$
// Revision 1.1  2000/11/28 03:47:26  jas
// Added FCVariables for the specific faces X,Y,and Z.
//
//

#endif
