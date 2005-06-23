
#ifndef UINTAH_HOMEBREW_PSPatchMatlGhost_H
#define UINTAH_HOMEBREW_PSPatchMatlGhost_H

#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Core/Geometry/IntVector.h>

using SCIRun::IntVector;

namespace Uintah {


    /**************************************
      
      struct
        PSPatchMatlGhost
      
        Patch, Material, Ghost info
	
      
      GENERAL INFORMATION
      
        PSPatchMatlGhost.h
      
        Bryan Worthen
        Department of Computer Science
        University of Utah
      
        Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
        Copyright (C) 2002 SCI Group
      
      KEYWORDS
        Patch, Material, Ghost
      
      DESCRIPTION
        Has all the important information for identifying a particle subset
          patch, material, and ghost properties
      
      WARNING
      
      ****************************************/

struct PSPatchMatlGhost {
  PSPatchMatlGhost(const Patch* patch, int matl, 
                   IntVector low, IntVector high, int dwid)
    : patch_(patch), matl_(matl), low_(low), high_(high), dwid_(dwid)
  {}
  PSPatchMatlGhost(const PSPatchMatlGhost& copy)
    : patch_(copy.patch_), matl_(copy.matl_), low_(copy.low_), 
       high_(copy.high_), dwid_(copy.dwid_)
  {}
  
  bool operator<(const PSPatchMatlGhost& other) const;
  const Patch* patch_;
  int matl_;
  IntVector low_;
  IntVector high_;
  int dwid_;
};  

} // End namespace Uintah

#endif
