
#ifndef UINTAH_HOMEBREW_SFCYVariableBase_H
#define UINTAH_HOMEBREW_SFCYVariableBase_H

#include <Uintah/Grid/Variable.h>

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

   class SFCYVariableBase : public Variable {
   public:
      
      virtual ~SFCYVariableBase();
      
      virtual void copyPointer(const SFCYVariableBase&) = 0;
      
      //////////
      // Insert Documentation Here:
      virtual SFCYVariableBase* clone() const = 0;

      virtual void allocate(const IntVector& lowIndex,
			    const IntVector& highIndex) = 0;
      virtual void copyPatch(SFCYVariableBase* src,
			      const IntVector& lowIndex,
			      const IntVector& highIndex) = 0;
      virtual void emit(OutputContext&) = 0;
   protected:
      SFCYVariableBase(const SFCYVariableBase&);
      SFCYVariableBase();
      
   private:
      SFCYVariableBase& operator=(const SFCYVariableBase&);
   };

} // end namespace Uintah

//
// $Log$
// Revision 1.2  2000/07/27 22:39:50  sparker
// Implemented MPIScheduler
// Added associated support
//
// Revision 1.1  2000/06/27 23:18:18  rawat
// implemented Staggered cell variables. Modified Patch.cc to get ghostcell
// and staggered cell indexes.
//
//

#endif
