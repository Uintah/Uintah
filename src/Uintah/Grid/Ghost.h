#ifndef UINTAH_HOMEBREW_Ghost_H
#define UINTAH_HOMEBREW_Ghost_H

namespace Uintah {
   
   /**************************************
     
     CLASS
       Ghost
      
       Short Description...
      
     GENERAL INFORMATION
      
       Ghost.h
      
       Steven G. Parker
       Department of Computer Science
       University of Utah
      
       Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
       Copyright (C) 2000 SCI Group
      
     KEYWORDS
       Ghost
      
     DESCRIPTION
       Long description...
      
     WARNING
      
     ****************************************/
    
   class Ghost {
   public:
      enum GhostType {
	 None,
	 AroundNodes,
	 AroundCells,
	 AroundXFaces,
	 AroundYFaces,
	 AroundZFaces,
	 AroundAllFaces
      };
      
   private:
      Ghost();
      Ghost(const Ghost&);
      Ghost& operator=(const Ghost&);
      
#ifdef __GNUG__
   public:
      // Keep gcc quiet, to avoid:
      // warning: all member functions in class `Uintah::Ghost' are private
      Ghost(Ghost*);
#endif
   };
} // end namespace Uintah

//
// $Log$
// Revision 1.2  2000/09/25 20:37:42  sparker
// Quiet g++ compiler warnings
// Work around g++ compiler bug instantiating vector<NCVariable<Vector> >
// Added computeVariableExtents to (eventually) simplify data warehouses
//
// Revision 1.1  2000/05/10 20:02:59  sparker
// Added support for ghost cells on node variables and particle variables
//  (work for 1 patch but not debugged for multiple)
// Do not schedule fracture tasks if fracture not enabled
// Added fracture directory to MPM sub.mk
// Be more uniform about using IntVector
// Made regions have a single uniform index space - still needs work
//
//

#endif
