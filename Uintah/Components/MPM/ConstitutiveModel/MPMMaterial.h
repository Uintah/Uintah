#ifndef __MPM_MATERIAL_H__
#define __MPM_MATERIAL_H__

#include <Uintah/Interface/DataWarehouseP.h>
#include "Material.h"
class ConstitutiveModel;

/**************************************

CLASS
   MPMMaterial

   Short description...

GENERAL INFORMATION

   MPMMaterial.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)

   Copyright (C) 2000 SCI Group

KEYWORDS
   MPM_Material

DESCRIPTION
   Long description...

WARNING

****************************************/

class MPMMaterial : public Material {
public:
  MPMMaterial(int dwi, int vfi, ConstitutiveModel *cm);

  ~MPMMaterial();


  //////////
  // Return correct constitutive model pointer for this material
  ConstitutiveModel * getConstitutiveModel();

  //////////
  // Return index associated with this material's
  // location in the data warehouse
  int getDWIndex();

  //////////
  // Return index associated with this material's
  // velocity field
  int getVFIndex();

private:

  // Index associated with this material's spot in the DW
  int d_dwindex;
  // Index associated with this material's velocity field
  int d_vfindex;
  // Specific constitutive model associated with this material
  ConstitutiveModel *d_cm;

  // Prevent copying of this class
  // copy constructor
  MPMMaterial(const MPMMaterial &mpmm);
  MPMMaterial& operator=(const MPMMaterial &mpmm);

};

#endif // __MPM_MATERIAL_H__

// $Log$
// Revision 1.1  2000/03/24 00:45:43  guilkey
// Added MPMMaterial class, as well as a skeleton Material class, from
// which MPMMaterial is inherited.  The Material class will be filled in
// as it's mission becomes better identified.
//

