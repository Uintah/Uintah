
#ifndef UINTAH_HOMEBREW_NCVariableBase_H
#define UINTAH_HOMEBREW_NCVariableBase_H

namespace SCICore {
   namespace Geometry {
      class IntVector;
   }
}

namespace Uintah {
   class OutputContext;
   class Region;
   using SCICore::Geometry::IntVector;

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

   class NCVariableBase {
   public:
      
      virtual ~NCVariableBase();
      
      virtual void copyPointer(const NCVariableBase&) = 0;
      
      //////////
      // Insert Documentation Here:
      virtual NCVariableBase* clone() const = 0;

      virtual void allocate(const IntVector& lowIndex,
			    const IntVector& highIndex) = 0;
      virtual void copyRegion(NCVariableBase* src,
			      const IntVector& lowIndex,
			      const IntVector& highIndex) = 0;
      virtual void emit(OutputContext&) = 0;
   protected:
      NCVariableBase(const NCVariableBase&);
      NCVariableBase();
      
   private:
      NCVariableBase& operator=(const NCVariableBase&);
   };

} // end namespace Uintah

//
// $Log$
// Revision 1.5  2000/05/15 19:39:48  sparker
// Implemented initial version of DataArchive (output only so far)
// Other misc. cleanups
//
// Revision 1.4  2000/05/10 20:03:00  sparker
// Added support for ghost cells on node variables and particle variables
//  (work for 1 patch but not debugged for multiple)
// Do not schedule fracture tasks if fracture not enabled
// Added fracture directory to MPM sub.mk
// Be more uniform about using IntVector
// Made regions have a single uniform index space - still needs work
//
// Revision 1.3  2000/05/02 06:07:22  sparker
// Implemented more of DataWarehouse and SerialMPM
//
// Revision 1.2  2000/04/26 06:48:50  sparker
// Streamlined namespaces
//
// Revision 1.1  2000/04/20 20:09:22  jas
// I don't know what these do, but Steve says we need them.
//
// Revision 1.1  2000/04/19 05:26:15  sparker
// Implemented new problemSetup/initialization phases
// Simplified DataWarehouse interface (not finished yet)
// Made MPM get through problemSetup, but still not finished
//
//

#endif
