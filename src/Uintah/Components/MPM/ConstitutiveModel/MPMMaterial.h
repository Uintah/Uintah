#ifndef __MPM_MATERIAL_H__
#define __MPM_MATERIAL_H__

#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Grid/Material.h>

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

private:

  // Specific constitutive model associated with this material
  ConstitutiveModel *d_cm;

  // Prevent copying of this class
  // copy constructor
  MPMMaterial(const MPMMaterial &mpmm);
  MPMMaterial& operator=(const MPMMaterial &mpmm);

};

#endif // __MPM_MATERIAL_H__

// $Log$
// Revision 1.3  2000/04/14 02:19:42  jas
// Now using the ProblemSpec for input.
//
// Revision 1.2  2000/03/30 18:31:22  guilkey
// Moved Material base class to Grid directory.  Modified MPMMaterial
// and sub.mk to coincide with these changes.
//
// Revision 1.1  2000/03/24 00:45:43  guilkey
// Added MPMMaterial class, as well as a skeleton Material class, from
// which MPMMaterial is inherited.  The Material class will be filled in
// as it's mission becomes better identified.
//

