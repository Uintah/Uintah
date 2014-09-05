#ifndef UINTAH_HOMEBREW_SendState_H
#define UINTAH_HOMEBREW_SendState_H

#include <map>

namespace Uintah {
  using namespace std;
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
    ParticleSubset* find_sendset(const Patch*, int, int) const;
    void add_sendset(const Patch*, int, int, ParticleSubset*);
    
  private:
    typedef map<pair<pair<const Patch*, int>, int>, ParticleSubset*> maptype;
    maptype sendSubsets;
    SendState(const SendState&);
    SendState& operator=(const SendState&);
   };
} // End namespace Uintah

#endif

