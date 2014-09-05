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
      // warning: all member functions in class `Packages/Uintah::Ghost' are private
      Ghost(Ghost*);
#endif
   };
} // End namespace Uintah

#endif
