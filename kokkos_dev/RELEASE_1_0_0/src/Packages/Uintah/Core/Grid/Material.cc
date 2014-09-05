//  Material.cc

#include <Packages/Uintah/Core/Grid/Material.h>

using namespace Uintah;

Material::Material()
{
  thismatl=0;
}

Material::~Material()
{
  if(thismatl && thismatl->removeReference())
    delete thismatl;
}

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
   ASSERT(!thismatl);
   thismatl = new MaterialSubset(); // scinew not used here because it
                                    // triggers some problem in g++ 2.95
   thismatl->addReference();
   thismatl->add(idx);
}

void Material::setVFIndex(int idx)
{
   d_vfindex = idx;
}
