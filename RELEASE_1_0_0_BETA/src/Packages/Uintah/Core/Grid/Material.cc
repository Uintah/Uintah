//  Material.cc

#include <Packages/Uintah/Core/Grid/Material.h>

using namespace Uintah;

int Material::getDWIndex() const
{
  // Return this material's index into the data warehouse
  return d_dwindex;
}

int Material::getVFIndex() const
{
  // Return this material's index for velocity field
  return d_vfindex;
}

void Material::setDWIndex(int idx)
{
   d_dwindex = idx;
}

void Material::setVFIndex(int idx)
{
   d_vfindex = idx;
}
