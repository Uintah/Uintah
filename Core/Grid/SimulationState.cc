
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Grid/ReductionVariable.h>
#include <Packages/Uintah/Core/Grid/Material.h>
#include <Packages/Uintah/Core/Grid/SimpleMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesMaterial.h>
#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <Packages/Uintah/Core/Grid/Reductions.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Malloc/Allocator.h>

using namespace Uintah;

SimulationState::SimulationState(ProblemSpecP &ps)
{
   VarLabel* nonconstDelt = 
     VarLabel::create("delT", delt_vartype::getTypeDescription() );
// ReductionVariable<double, Reductions::Min<double> >::getTypeDescription());
   nonconstDelt->allowMultipleComputes();
   delt_label = nonconstDelt;

   refineFlag_label = VarLabel::create("refineFlag",
				       CCVariable<int>::getTypeDescription());
   d_ref_press = 0.0;
   d_elapsed_time = 0.0;

  // Get the physical constants that are shared between codes.
  // For now it is just gravity.

  ProblemSpecP phys_cons_ps = ps->findBlock("PhysicalConstants");
  if(phys_cons_ps){
    phys_cons_ps->require("gravity",d_gravity);
    phys_cons_ps->get("reference_pressure",d_ref_press);
  } else {
    d_gravity=Vector(0,0,0);
    d_ref_press=0;
  }

  all_mpm_matls = 0;
  all_ice_matls = 0;
  all_arches_matls = 0;
}

void SimulationState::registerMaterial(Material* matl)
{
   matl->setDWIndex((int)matls.size());
   matls.push_back(matl);

   if(matl->hasName())
     named_matls[matl->getName()] = matl;
}

void SimulationState::registerMPMMaterial(MPMMaterial* matl)
{
   mpm_matls.push_back(matl);
   registerMaterial(matl);
}

void SimulationState::registerArchesMaterial(ArchesMaterial* matl)
{
   arches_matls.push_back(matl);
   registerMaterial(matl);
}

void SimulationState::registerICEMaterial(ICEMaterial* matl)
{
   ice_matls.push_back(matl);
   registerMaterial(matl);
}

void SimulationState::registerSimpleMaterial(SimpleMaterial* matl)
{
  simple_matls.push_back(matl);
  registerMaterial(matl);
}

void SimulationState::finalizeMaterials()
{
  all_mpm_matls = scinew MaterialSet();
  all_mpm_matls->addReference();
  vector<int> tmp_mpm_matls(mpm_matls.size());
  for(int i=0;i<(int)mpm_matls.size();i++)
    tmp_mpm_matls[i] = mpm_matls[i]->getDWIndex();
  all_mpm_matls->addAll(tmp_mpm_matls);
  
  all_arches_matls = scinew MaterialSet();
  all_arches_matls->addReference();
  vector<int> tmp_arches_matls(arches_matls.size());
  for (int i = 0; i<(int)arches_matls.size();i++)
    tmp_arches_matls[i] = arches_matls[i]->getDWIndex();
  all_arches_matls->addAll(tmp_arches_matls);

  all_ice_matls = scinew MaterialSet();
  all_ice_matls->addReference();
  vector<int> tmp_ice_matls(ice_matls.size());
  for(int i=0;i<(int)ice_matls.size();i++)
    tmp_ice_matls[i] = ice_matls[i]->getDWIndex();
  all_ice_matls->addAll(tmp_ice_matls);

  all_matls = scinew MaterialSet();
  all_matls->addReference();
  vector<int> tmp_matls(matls.size());
  for(int i=0 ;i<(int)matls.size();i++)
    tmp_matls[i] = matls[i]->getDWIndex();
  all_matls->addAll(tmp_matls);

  allInOneMatl = scinew MaterialSubset();
  allInOneMatl->addReference();
  // a material that represents all materials 
  // (i.e. summed over all materials -- the whole enchilada)
  allInOneMatl->add((int)matls.size());
}

int SimulationState::getNumVelFields() const {
  int num_vf=0;
  for (int i = 0; i < (int)matls.size(); i++) {
    num_vf = Max(num_vf,matls[i]->getVFIndex());
  }
  return num_vf+1;
}

SimulationState::~SimulationState()
{
  VarLabel::destroy(delt_label);
  VarLabel::destroy(refineFlag_label);
  for (int i = 0; i < (int)matls.size(); i++)
    delete matls[i];

  if(all_matls && all_matls->removeReference())
    delete all_matls;
  
  if(all_mpm_matls && all_mpm_matls->removeReference())
    delete all_mpm_matls;

  if (all_arches_matls && all_arches_matls->removeReference())
    delete all_arches_matls;

  if(all_ice_matls && all_ice_matls->removeReference())
    delete all_ice_matls;

  if (allInOneMatl && allInOneMatl->removeReference()) {
    delete allInOneMatl;
  }
}

const MaterialSet* SimulationState::allMPMMaterials() const
{
  ASSERT(all_mpm_matls != 0);
  return all_mpm_matls;
}

const MaterialSet* SimulationState::allArchesMaterials() const
{
  ASSERT(all_arches_matls != 0);
  return all_arches_matls;
}

const MaterialSet* SimulationState::allICEMaterials() const
{
  ASSERT(all_ice_matls != 0);
  return all_ice_matls;
}

const MaterialSet* SimulationState::allMaterials() const
{
  ASSERT(all_matls != 0);
  return all_matls;
}

Material* SimulationState::getMaterialByName(const std::string& name) const
{
  map<string, Material*>::const_iterator iter = named_matls.find(name);
  if(iter == named_matls.end())
    return 0;
  return iter->second;
}

Material* SimulationState::parseAndLookupMaterial(ProblemSpecP& params,
						  const std::string& name) const
{
  string matlname;
  if(!params->get(name, matlname))
    throw ProblemSetupException("Cannot find material section");
  Material* result = getMaterialByName(matlname);
  if(!result){
    int matlidx;
    if(!params->get(name, matlidx))
      throw ProblemSetupException("Cannot find material called "+matlname);
    if(matlidx < 0 || matlidx >= static_cast<int>(matls.size()))
      throw ProblemSetupException("Invalid material: "+to_string(matlidx));
    result = matls[matlidx];
  }
  return result;
}
