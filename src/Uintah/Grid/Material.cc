#include "Material.h"
#include "ConstitutiveModelFactory.h"
#include <fstream>
using std::ifstream;

Material::Material(){}
Material::~Material(){}

void Material::addMaterial(ifstream &filename)
{
  filename >> Density >> MatType;
  ConstitutiveModelFactory::readParameters(filename, MatType, mp);
}

double Material::getDensity() const { return Density; }

int Material::getMaterialType() const { return MatType; }

void Material::getMatProps(double matprop[10]) const
{

  for(int i=0;i<10;i++){
	matprop[i] = mp[i];
  }

  return;

}

// $Log$
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
