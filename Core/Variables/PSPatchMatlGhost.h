
#ifndef UINTAH_HOMEBREW_PSPatchMatlGhost_H
#define UINTAH_HOMEBREW_PSPatchMatlGhost_H

#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Ghost.h>

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
                   Ghost::GhostType gt = Ghost::None, int ngc = 0, int dwid = 0)
    : patch_(patch), matl_(matl), gt_(gt), numGhostCells_(ngc), dwid_(dwid)
  {}
  PSPatchMatlGhost(const PSPatchMatlGhost& copy)
    : patch_(copy.patch_), matl_(copy.matl_), gt_(copy.gt_), 
       numGhostCells_(copy.numGhostCells_), dwid_(copy.dwid_)
  {}
  
  bool operator<(const PSPatchMatlGhost& other) const;
  const Patch* patch_;
  int matl_;
  Ghost::GhostType gt_;
  int numGhostCells_;
  int dwid_;
};  

} // End namespace Uintah

#endif
