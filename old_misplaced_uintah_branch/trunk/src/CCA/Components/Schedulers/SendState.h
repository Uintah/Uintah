#ifndef UINTAH_HOMEBREW_SendState_H
#define UINTAH_HOMEBREW_SendState_H

#include <map>
#include <Core/Grid/Ghost.h>
#include <Core/Grid/Variables/PSPatchMatlGhost.h>

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
    ParticleSubset* find_sendset(int dest, const Patch*, int matl, 
                                 IntVector low, IntVector high, int dwid = 0) const;
    void add_sendset(ParticleSubset* pset, int dest, const Patch*, int matl,  
                     IntVector low, IntVector high, int dwid = 0);

    void print();
  private:
    typedef map<pair<PSPatchMatlGhost, int>, ParticleSubset*> maptype;
    maptype sendSubsets;
    SendState(const SendState&);
    SendState& operator=(const SendState&);
   };
} // End namespace Uintah

#endif

