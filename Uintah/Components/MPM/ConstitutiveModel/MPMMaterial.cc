//  MPMMaterial.cc
//

#include "MPMMaterial.h"
#include "ConstitutiveModel.h"

using namespace Uintah::Components;

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

ConstitutiveModel * MPMMaterial::getConstitutiveModel()
{
  // Return the pointer to the constitutive model associated
  // with this material

  return d_cm;
}
