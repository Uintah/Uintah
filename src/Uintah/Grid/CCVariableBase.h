
#ifndef UINTAH_HOMEBREW_CCVariableBase_H
#define UINTAH_HOMEBREW_CCVariableBase_H

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
   CCVariableBase
   
   Short description...

GENERAL INFORMATION

   CCVariableBase.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   CCVariableBase

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class CCVariableBase {
   public:
      
      virtual ~CCVariableBase();
      
      virtual void copyPointer(const CCVariableBase&) = 0;
      
      //////////
      // Insert Documentation Here:
      virtual CCVariableBase* clone() const = 0;

      virtual void allocate(const IntVector& lowIndex,
			    const IntVector& highIndex) = 0;
      virtual void copyPatch(CCVariableBase* src,
			      const IntVector& lowIndex,
			      const IntVector& highIndex) = 0;
      virtual void emit(OutputContext&) = 0;
   protected:
      CCVariableBase(const CCVariableBase&);
      CCVariableBase();
      
   private:
      CCVariableBase& operator=(const CCVariableBase&);
   };

} // end namespace Uintah

//
// $Log$
// Revision 1.3  2000/05/31 04:01:51  rawat
// partially completed CCVariable implementation
//
// Revision 1.2  2000/05/30 20:19:28  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.1  2000/05/11 20:12:11  dav
// Added CCVaraibleBase
//
//

#endif
