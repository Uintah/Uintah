//  Material.cc
//

#include "Material.h"

int Material::getDWIndex()
{
  // Return this material's index into the data warehouse
  return d_dwindex;
}

int Material::getVFIndex()
{
  // Return this material's index for velocity field
  return d_vfindex;
}
