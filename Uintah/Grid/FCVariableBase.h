
#ifndef UINTAH_HOMEBREW_FCVariableBase_H
#define UINTAH_HOMEBREW_FCVariableBase_H

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

   class FCVariableBase {
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
      virtual void emit(OutputContext&) = 0;
   protected:
      FCVariableBase(const FCVariableBase&);
      FCVariableBase();
      
   private:
      FCVariableBase& operator=(const FCVariableBase&);
   };

} // end namespace Uintah

//
// $Log$
// Revision 1.1  2000/06/14 21:59:35  jas
// Copied CCVariable stuff to make FCVariables.  Implementation is not
// correct for the actual data storage and iteration scheme.
//
//

#endif
