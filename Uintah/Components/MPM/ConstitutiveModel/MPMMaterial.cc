//QUESTION FOR STEVE OR DAVE:
//Where is the Material information stored?  Does it go into the
//data warehouse?  Also, MPMMaterial stores a pointer to a CM,
//does that CM need to be stored in the DW or does it just hang
//out in the ether?

//  MPMMaterial.cc
//

#include "MPMMaterial.h"
#include "ConstitutiveModel.h"

MPMMaterial::MPMMaterial(int dwi, int vfi, ConstitutiveModel *cm)
{
  // Constructor

  d_dwindex = dwi;
  d_vfindex = vfi;
  d_cm = cm;

}

MPMMaterial::~MPMMaterial()
{
  // Destructor
}

int MPMMaterial::getDWIndex()
{
  // Return this material's index into the data warehouse
  return d_dwindex;
}

int MPMMaterial::getVFIndex()
{
  // Return this material's index for velocity field
  return d_vfindex;
}

ConstitutiveModel * MPMMaterial::getConstitutiveModel()
{
  // Return the pointer to the constitutive model associated
  // with this material

  return d_cm;
}
