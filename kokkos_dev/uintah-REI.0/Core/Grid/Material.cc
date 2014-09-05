//  Material.cc

#include <Packages/Uintah/Core/Grid/Material.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <string>
#include <iostream>
#include <sstream>

using namespace Uintah;


Material::Material()
{
  thismatl=0;
  haveName = false;
  name="";
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

ProblemSpecP Material::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP mat = 0;
  if (haveName) {
    mat = ps->appendChild("material");
    mat->setAttribute("name",name);
  } else {
    mat = ps->appendChild("material");
  }

  std::stringstream strstream;
  strstream << getDWIndex();
  string index_val = strstream.str();
  mat->setAttribute("index",index_val);
  return mat;
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
   thismatl = scinew MaterialSubset(); 
                                       
   thismatl->addReference();
   thismatl->add(idx);
}

void Material::setVFIndex(int idx)
{
   d_vfindex = idx;
}

void Material::registerParticleState(SimulationState* ss)
{
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
