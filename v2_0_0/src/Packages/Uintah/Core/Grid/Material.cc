//  Material.cc

#include <Packages/Uintah/Core/Grid/Material.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <string>
#include <iostream>
using namespace Uintah;


Material::Material()
{
  thismatl=0;
}

Material::Material(ProblemSpecP& ps)
{
  
  thismatl=0;

  // Look for the Rx attribute
  std::string rx_product;
  if (!ps->getAttribute("Rx",rx_product))
    Material::d_rx_prod = Material::none;
  else {
    if (rx_product == "product")
      Material::d_rx_prod = Material::product;
    else if (rx_product == "reactant")
      Material::d_rx_prod = Material::reactant;
  }

  if (Material::d_rx_prod == Material::product)
    std::cerr << "Material is a product" << std::endl;
  if (Material::d_rx_prod == Material::reactant)
    std::cerr << "Material is a reactant" << std::endl;
  if (Material::d_rx_prod == Material::none)
    std::cerr << "Material is not specified product/reactant" << std::endl;

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

Material::RX_Prod Material::getRxProduct()
{
  return d_rx_prod;
}

Burn* Material::getBurnModel()
{
  return 0;
}
