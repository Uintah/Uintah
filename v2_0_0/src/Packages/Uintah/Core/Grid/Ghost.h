#ifndef UINTAH_HOMEBREW_Ghost_H
#define UINTAH_HOMEBREW_Ghost_H

#include <Core/Util/Assert.h>
#include <Core/Geometry/IntVector.h>
#undef None

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
	 AroundFacesX,
	 AroundFacesY,
	 AroundFacesZ,
	 AroundFaces,
	 numGhostTypes // 7
      };

     static SCIRun::IntVector getGhostTypeDir(Ghost::GhostType gtype)
     {
       int index = (int)gtype;
       CHECKARRAYBOUNDS(index, 0, numGhostTypes);
       return directions[index];
     }

     static std::string getGhostTypeName(Ghost::GhostType gtype)
     {
       int index = (int)gtype;
       CHECKARRAYBOUNDS(index, 0, numGhostTypes);
       return names[index];
     }
   private:
      Ghost();
      Ghost(const Ghost&);
      Ghost& operator=(const Ghost&);

     static SCIRun::IntVector directions[numGhostTypes];
     static std::string names[numGhostTypes];
#ifdef __GNUG__
   public:
      // Keep gcc quiet, to avoid:
      // warning: all member functions in class `Packages/Uintah::Ghost' are private
      Ghost(Ghost*);
#endif
   };
} // End namespace Uintah

#endif
