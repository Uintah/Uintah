#include <unistd.h>
#include "ConstitutiveModelFactory.h"

#ifdef __GNUG__
#include "BoundedArray.cc"
#include "Matrix.cc"
template class BoundedArray<double>;
template double operator*(BoundedArray<double> const &, BoundedArray<double> const &);
#endif

using std::string;

int main(int argc, char **argv)
{
  string filename = "materials.dat";
  int    c;
  while ((c = getopt(argc, argv,"f:h")) != EOF)
    {
      switch (c)
	{
	case 'f': filename = optarg; break;
	case 'h':
	default:
	  cout << "usage:: " << argv[0] << " [-f filename] [-h]" << endl;
	  exit(0);
	  break;
	}
    }


  ofstream f(filename.c_str());
  if(!f)
    {
      cerr << "Unable to open \"" << filename << "\"" << endl;
      exit(1);
    }
  int      i;
  ConstitutiveModel *elastic     = scinew ElasticConstitutiveModel;
  ConstitutiveModel *mooney      = scinew CompMooneyRivlin;
  ConstitutiveModel *neohook     = scinew CompNeoHook;
  ConstitutiveModel *neohookplas = scinew CompNeoHookPlas;
  ConstitutiveModel *hypelastdam = scinew HyperElasticDamage;
  ConstitutiveModel *viselastdam = scinew ViscoElasticDamage;
  // add new Model's variable initalization here
  
  ConstitutiveModel *materials[ConstitutiveModelFactory::CM_MAX-1] =
  {
    elastic,
    mooney,
    neohook,
    neohookplas,
    hypelastdam,
    viselastdam,
    // insert new Model's variable here
  };
  
  f << (ConstitutiveModelFactory::CM_MAX - 1)
    << " # number of material types" << endl;
  for(i = ConstitutiveModelFactory::CM_NULL + 1;
      i < ConstitutiveModelFactory::CM_MAX;
      i++)
    {
      f << materials[i-1]->getType()
	<< " # material type followed by name" << endl;
      f << materials[i-1]->getName() << endl;
      f << materials[i-1]->getNumParameters()
	<< " # number of parameters for material type, followed by names"
	<< endl;
      materials[i-1]->printParameterNames(f);
    }
  
  f.close();
  return (0);
}

