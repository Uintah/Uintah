#include "ConstitutiveModelFactory.h"
#include <fstream>
#include <iostream>
using std::cerr;
using std::ifstream;
using std::ofstream;

#ifdef WONT_COMPILE_YET

void ConstitutiveModelFactory::readParameters(ifstream& in, int mat_type,
					      double *p_array)
{
  switch(mat_type)
    {
    case CM_ELASTIC:
      ElasticConstitutiveModel::readParameters(in, p_array);
      break;
    case CM_MOONEY_RIVLIN:
      CompMooneyRivlin::readParameters(in, p_array);
      break;
    case CM_NEO_HOOK:
      CompNeoHook::readParameters(in, p_array);
      break;
    case CM_NEO_HOOK_PLAS:
      CompNeoHookPlas::readParameters(in, p_array);
      break;
    case CM_HYPER_ELASTIC_DAMAGE:
      HyperElasticDamage::readParameters(in, p_array);
      break;
    case CM_VISCOELASTIC_DAMAGE:
      ViscoElasticDamage::readParameters(in, p_array);
      break;
    default:
      cerr << "Unknown Material Type R (" << mat_type << ") aborting\n";
      exit(1);
      break;
    }
}

void ConstitutiveModelFactory::writeParameters(ofstream& out, int mat_type,
					       double *p_array)
{
  out << mat_type << " ";
  switch(mat_type)
    {
    case CM_ELASTIC:
      ElasticConstitutiveModel::writeParameters(out, p_array);
      break;
    case CM_MOONEY_RIVLIN:
      CompMooneyRivlin::writeParameters(out, p_array);
      break;
    case CM_NEO_HOOK:
      CompNeoHook::writeParameters(out, p_array);
      break;
    case CM_NEO_HOOK_PLAS:
      CompNeoHookPlas::writeParameters(out, p_array);
      break;
    case CM_HYPER_ELASTIC_DAMAGE:
      HyperElasticDamage::writeParameters(out, p_array);
      break;
    case CM_VISCOELASTIC_DAMAGE:
      ViscoElasticDamage::writeParameters(out, p_array);
      break;
    default:
      cerr << "Unknown Material Type W (" << mat_type << ") aborting\n";
      exit(1);
      break;
    }
}

ConstitutiveModel* ConstitutiveModelFactory::readParametersAndCreate(ifstream& in)
{
  int mat_type;
  in >> mat_type;
  switch(mat_type)
    {
    case CM_ELASTIC:
      return(ElasticConstitutiveModel::readParametersAndCreate(in));
    case CM_MOONEY_RIVLIN:
      return(CompMooneyRivlin::readParametersAndCreate(in));
    case CM_NEO_HOOK:
      return(CompNeoHook::readParametersAndCreate(in));
    case CM_NEO_HOOK_PLAS:
      return(CompNeoHookPlas::readParametersAndCreate(in));
    case CM_HYPER_ELASTIC_DAMAGE:
      return(HyperElasticDamage::readParametersAndCreate(in));
    case CM_VISCOELASTIC_DAMAGE:
      return(ViscoElasticDamage::readParametersAndCreate(in));
    default:
      cerr << "Unknown Material Type RaC (" << mat_type << ") aborting\n";
      exit(1);
      break;
    }
  return(0);
}

ConstitutiveModel* ConstitutiveModelFactory::readRestartParametersAndCreate(ifstream& in)
{
  int mat_type;
  in >> mat_type;
  switch(mat_type)
    {
    case CM_ELASTIC:
      return(ElasticConstitutiveModel::readRestartParametersAndCreate(in));
    case CM_MOONEY_RIVLIN:
      return(CompMooneyRivlin::readRestartParametersAndCreate(in));
    case CM_NEO_HOOK:
      return(CompNeoHook::readRestartParametersAndCreate(in));
    case CM_NEO_HOOK_PLAS:
      return(CompNeoHookPlas::readRestartParametersAndCreate(in));
    case CM_HYPER_ELASTIC_DAMAGE:
      return(HyperElasticDamage::readRestartParametersAndCreate(in));
    case CM_VISCOELASTIC_DAMAGE:
      return(ViscoElasticDamage::readRestartParametersAndCreate(in));
    default:
      cerr << "Unknown Material Type (" << mat_type << ") aborting\n";
      exit(1);
      break;
    }
  return(0);
}

ConstitutiveModel* ConstitutiveModelFactory::create(int mat_type,
						    double *p_array)
{
  switch(mat_type)
    {
    case CM_ELASTIC:
      return(ElasticConstitutiveModel::create(p_array));
    case CM_MOONEY_RIVLIN:
      return(CompMooneyRivlin::create(p_array));
    case CM_NEO_HOOK:
      return(CompNeoHook::create(p_array));
    case CM_NEO_HOOK_PLAS:
      return(CompNeoHookPlas::create(p_array));
    case CM_HYPER_ELASTIC_DAMAGE:
      return(HyperElasticDamage::create(p_array));
    case CM_VISCOELASTIC_DAMAGE:
      return(ViscoElasticDamage::create(p_array));
    default:
      cerr << "Unknown Material Type c (" << mat_type << ") aborting\n";
      exit(1);
      break;
    }
  return(0);
}
#endif
