#include "ConstitutiveModelFactory.h"
#include "ElasticConstitutiveModel.h"
#include "CompMooneyRivlin.h"
#include "CompNeoHook.h"
#include "CompNeoHookPlas.h"
#include "HyperElasticDamage.h"
#include "ViscoElasticDamage.h"
#include <fstream>
#include <iostream>
#include <string>
using std::cerr;
using std::ifstream;
using std::ofstream;

using namespace Uintah::Components;


void ConstitutiveModelFactory::readParameters(ProblemSpecP ps, 
					      std::string mat_type,
					      double *p_array)
{
  if (mat_type == "elastic")
    ElasticConstitutiveModel::readParameters(ps, p_array);
     
  else if (mat_type == "comp_mooney_rivlin")
    CompMooneyRivlin::readParameters(ps, p_array);

  else if (mat_type ==  "comp_neo_hook")
    CompNeoHook::readParameters(ps, p_array);

  else if (mat_type == "comp_neo_hook_plastic")
    CompNeoHookPlas::readParameters(ps, p_array);
 
  else if (mat_type == "hyper_elastic_damage")
    HyperElasticDamage::readParameters(ps, p_array);
   
  else if (mat_type == "visco_elastic_damage")
    ViscoElasticDamage::readParameters(ps, p_array);
   
  else {
      cerr << "Unknown Material Type R (" << mat_type << ") aborting\n";
      exit(1);
  }
}

#ifdef WONT_COMPILE_YET  

ConstitutiveModel* ConstitutiveModelFactory::readParametersAndCreate(
					     ProblemSpecP ps,
					     std::string mat_type)
{
  if (mat_type == "elastic")
    return(ElasticConstitutiveModel::readParametersAndCreate(ps));
 
  else if (mat_type =="comp_mooney_rivlin")
    return(CompMooneyRivlin::readParametersAndCreate(ps));

  else if (mat_type == "comp_neo_hook")
    return(CompNeoHook::readParametersAndCreate(ps));

  else if (mat_type == "comp_neo_hook_plastic")
    return(CompNeoHookPlas::readParametersAndCreate(ps));

  else if (mat_type == "hyper_elastic_damage")
    return(HyperElasticDamage::readParametersAndCreate(ps));

  else if (mat_type == "visco_elastic_damage")
    return(ViscoElasticDamage::readParametersAndCreate(ps));
   
  else {
      cerr << "Unknown Material Type RaC (" << mat_type << ") aborting\n";
      exit(1);
  }
  return(0);
}

ConstitutiveModel* ConstitutiveModelFactory::readRestartParametersAndCreate(
					     ProblemSpecP ps,
					     std::string mat_type)
{
 
  if (mat_type == "elastic")
    return(ElasticConstitutiveModel::readRestartParametersAndCreate(ps));

  else if (mat_type == "comp_mooney_rivlin")
    return(CompMooneyRivlin::readRestartParametersAndCreate(ps));

  else if (mat_type == "comp_neo_hook")
    return(CompNeoHook::readRestartParametersAndCreate(ps));

  else if (mat_type == "comp_neo_hook_plastic")
    return(CompNeoHookPlas::readRestartParametersAndCreate(ps));

  else if (mat_type == "hyper_elastic_damage")
    return(HyperElasticDamage::readRestartParametersAndCreate(ps));

  else if (mat_type == "visco_elastic_damage")
    return(ViscoElasticDamage::readRestartParametersAndCreate(ps));
  
  else {
      cerr << "Unknown Material Type (" << mat_type << ") aborting\n";
      exit(1);
  }
  return(0);
}

ConstitutiveModel* ConstitutiveModelFactory::create(std::string mat_type,
						    double *p_array)
{
  if (mat_type == "elastic")
    return(ElasticConstitutiveModel::create(p_array));

  else if (mat_type == "comp_mooney_rivlin")
    return(CompMooneyRivlin::create(p_array));

  else if (mat_type == "comp_neo_hook")
    return(CompNeoHook::create(p_array));

  else if (mat_type == "comp_neo_hook_plastic")
    return(CompNeoHookPlas::create(p_array));

  else if (mat_type == "hyper_elastic_damage")
    return(HyperElasticDamage::create(p_array));

  else if (mat_type == "visco_elastic_damage")
    return(ViscoElasticDamage::create(p_array));
   
  else {
    cerr << "Unknown Material Type c (" << mat_type << ") aborting\n";
    exit(1);
  }
  return(0);
}

#endif


