//  Material.cc

#include <Packages/Uintah/Core/Grid/Material.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <string>
#include <iostream>
using namespace Uintah;


Material::Material()
{
  thismatl=0;
  haveName = false;
}

Material::Material(ProblemSpecP& ps)
{
  
  thismatl=0;

  // Look for the name attribute
  if(ps->getAttribute("name", name))
    haveName = true;
  else
    haveName = false;
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
   thismatl = scinew MaterialSubset(); // scinew not used here because it
                                       // triggers some problem in g++ 2.95
   thismatl->addReference();
   thismatl->add(idx);
}

void Material::setVFIndex(int idx)
{
   d_vfindex = idx;
}

double Material::getThermalConductivity() const
{
  return d_thermalConductivity;
}

double Material::getSpecificHeat() const
{
  return d_specificHeat;
}

double Material::getHeatTransferCoefficient() const
{
  return d_heatTransferCoefficient;
}

bool Material::getIncludeFlowWork() const
{
  return d_includeFlowWork;
}
