#ifndef UINTAH_HOMEBREW_SendState_H
#define UINTAH_HOMEBREW_SendState_H

#include <map>

namespace Uintah {
   class Patch;
   class ParticleSubset;
   /**************************************
     
     CLASS
       SendState
      
       Short Description...
      
     GENERAL INFORMATION
      
       SendState.h
      
       Steven G. Parker
       Department of Computer Science
       University of Utah
      
       Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
       Copyright (C) 2000 SCI Group
      
     KEYWORDS
       SendState
      
     DESCRIPTION
       Long description...
      
     WARNING
      
     ****************************************/
    
   class SendState {
   public:
      SendState();
      ~SendState();
      std::map<std::pair<std::pair<const Patch*, int>, int>, ParticleSubset*> d_sendSubsets;

   private:
      SendState(const SendState&);
      SendState& operator=(const SendState&);
      
   };
} // end namespace Uintah

//
// $Log$
// Revision 1.2  2000/12/10 09:06:11  sparker
// Merge from csafe_risky1
//
// Revision 1.1.2.2  2000/10/02 17:33:39  sparker
// Fixed boundary particles code for multiple materials
// Free ParticleSubsets used for boundary particle sends
//
// Revision 1.1.2.1  2000/10/02 15:02:45  sparker
// Send only boundary particles
//
//

#endif

