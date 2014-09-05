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
} // End namespace Uintah

#endif

