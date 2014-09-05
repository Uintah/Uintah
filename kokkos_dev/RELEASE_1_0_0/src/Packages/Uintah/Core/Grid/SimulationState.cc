
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/ReductionVariable.h>
#include <Packages/Uintah/Core/Grid/Material.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <Packages/Uintah/Core/Grid/Reductions.h>
#include <Core/Malloc/Allocator.h>

using namespace Uintah;

SimulationState::SimulationState(ProblemSpecP &ps)
{
   VarLabel* nonconstDelt = scinew VarLabel("delT",
    ReductionVariable<double, Reductions::Min<double> >::getTypeDescription());
   nonconstDelt->allowMultipleComputes();
   delt_label = nonconstDelt;
   d_mpm_cfd = false;
   d_ref_press = 0.0;

  // Get the physical constants that are shared between codes.
  // For now it is just gravity.

  ProblemSpecP phys_cons_ps = ps->findBlock("PhysicalConstants");
  phys_cons_ps->require("gravity",d_gravity);
  phys_cons_ps->get("reference_pressure",d_ref_press);

  all_mpm_matls = 0;
  all_ice_matls = 0;
}

void SimulationState::registerMaterial(Material* matl)
{
   matl->setDWIndex((int)matls.size());
   matls.push_back(matl);
}

void SimulationState::registerMPMMaterial(MPMMaterial* matl)
{
   mpm_matls.push_back(matl);
   registerMaterial(matl);
}

void SimulationState::registerICEMaterial(ICEMaterial* matl)
{
   ice_matls.push_back(matl);
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
  
  all_ice_matls = scinew MaterialSet();
  all_ice_matls->addReference();
  vector<int> tmp_ice_matls(ice_matls.size());
  for(int i=0;i<(int)ice_matls.size();i++)
    tmp_ice_matls[i] = ice_matls[i]->getDWIndex();
  all_ice_matls->addAll(tmp_ice_matls);

  all_matls = scinew MaterialSet();
  all_matls->addReference();
  vector<int> tmp_matls(matls.size());
  for(int i=0;i<(int)matls.size();i++)
    tmp_matls[i] = matls[i]->getDWIndex();
  all_matls->addAll(tmp_matls);
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
  delete delt_label;

  for (int i = 0; i < (int)matls.size(); i++)
    delete matls[i];
  if(all_mpm_matls && all_mpm_matls->removeReference())
    delete all_mpm_matls;
    
  if(all_ice_matls && all_ice_matls->removeReference())
    delete all_ice_matls;
}

const MaterialSet* SimulationState::allMPMMaterials() const
{
  ASSERT(all_mpm_matls != 0);
  return all_mpm_matls;
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
