#include "Material.h"
#include <Uintah/Components/MPM/ConstitutiveModel/ConstitutiveModelFactory.h>

using namespace Uintah::Components;

Material::Material(){}
Material::~Material(){}

void Material::addMaterial(ProblemSpecP mat_ps)
{
  mat_ps->require("density",d_density);
  mat_ps->require("material_type",d_mat_type);
  mat_ps->require("toughness",d_toughness);
  mat_ps->require("thermal_conductivity",d_thermal_cond);
  mat_ps->require("specific_heat",d_spec_heat);
 
  ConstitutiveModelFactory::readParameters(mat_ps,d_mat_type,d_mat_properties);
}

double Material::getDensity() const 
{ 
  return d_density; 
}

double Material::getToughness() const
{
  return d_toughness;
}

double Material::getThermalConductivity() const
{
  return d_thermal_cond;
}

double Material::getSpecificHeat() const
{
  return d_spec_heat;
}


std::string Material::getMaterialType() const 
{ 
  return d_mat_type; 
}

void Material::getMatProps(double matprop[10]) const
{

  for(int i=0;i<10;i++){
	matprop[i] = d_mat_properties[i];
  }

  return;

}



// $Log$
// Revision 1.4  2000/04/19 05:26:08  sparker
// Implemented new problemSetup/initialization phases
// Simplified DataWarehouse interface (not finished yet)
// Made MPM get through problemSetup, but still not finished
//
// Revision 1.3  2000/04/14 02:05:46  jas
// Subclassed out the GeometryPiece into 4 types: Box,Cylinder,Sphere, and
// Tri.  This made the GeometryObject class simpler since many of the
// methods are now relegated to the GeometryPiece subclasses.
//
// Revision 1.2  2000/03/20 17:17:15  sparker
// Made it compile.  There are now several #idef WONT_COMPILE_YET statements.
//
// Revision 1.1  2000/03/14 22:36:05  jas
// Readded geometry specification source files.
//
// Revision 1.1  2000/02/27 07:48:41  sparker
// Homebrew code all compiles now
// First step toward PSE integration
// Added a "Standalone Uintah Simulation" (sus) executable
// MPM does NOT run yet
//
// Revision 1.1  2000/02/24 06:12:17  sparker
// Imported homebrew code
//
// Revision 1.1  2000/01/24 22:48:53  sparker
// Stuff may actually work someday...
//
// Revision 1.1  1999/06/14 06:23:41  cgl
// - src/mpm/Makefile modified to work for IRIX64 or Linux
// - src/grid/Grid.cc added length to character array, since it
// 	was only 4 long, but was being sprintf'd with a 4 character
// 	number, leaving no room for the terminating 0.
// - added smpm directory. to house the samrai version of mpm.
//
// Revision 1.7  1999/05/30 02:10:49  cgl
// The stand-alone version of ConstitutiveModel and derived classes
// are now more isolated from the rest of the code.  A new class
// ConstitutiveModelFactory has been added to handle all of the
// switching on model type.  Between the ConstitutiveModelFactory
// class functions and a couple of new virtual functions in the
// ConstitutiveModel class, new models can be added without any
// source modifications to any classes outside of the constitutive_model
// directory.  See csafe/Uintah/src/CD/src/constitutive_model/HOWTOADDANEWMODEL
// for updated details on how to add a new model.
//
// --cgl
//
// Revision 1.6  1999/05/24 20:58:35  guilkey
// Added a new constitutive model, and tried to make it easier for
// others to add new models in the future.
//
// Revision 1.5  1999/02/25 22:32:33  guilkey
// Fixed some functions associated with the HyperElastic constitutive model.
//
// Revision 1.4  1999/01/26 21:53:34  campbell
// Added logging capabilities
//
