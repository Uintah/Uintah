#include <Packages/Uintah/CCA/Components/SwitchingCriteria/PBXTemperature.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Variables/NCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Labels/MPMLabel.h>
#include <Packages/Uintah/Core/Labels/MPMICELabel.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <string>
#include <iostream>

using namespace std;

using namespace Uintah;

PBXTemperature::PBXTemperature(ProblemSpecP& ps)
{
  ps->require("material",   d_material);
  ps->require("temperature",d_temperature);

  cout << "material = " << d_material << endl;
  cout << "temperature = " << d_temperature << endl;

  Mlb  = scinew MPMLabel();
  MIlb = scinew MPMICELabel();
}

PBXTemperature::~PBXTemperature()
{
  delete Mlb;
  delete MIlb;
}

void PBXTemperature::problemSetup(const ProblemSpecP& ps, 
                                  SimulationStateP& state)
{
  d_sharedState = state;
}

void PBXTemperature::scheduleSwitchTest(const LevelP& level, SchedulerP& sched)
{
  Task* t = scinew Task("switchTest", this, &PBXTemperature::switchTest);

  MaterialSubset* one_matl = scinew MaterialSubset();
  one_matl->add(0);
  one_matl->addReference();

  t->requires(Task::NewDW, Mlb->gMassLabel,       Ghost::AroundCells, 1);
  t->requires(Task::NewDW, Mlb->gTemperatureLabel,Ghost::AroundCells, 1);
  t->requires(Task::OldDW, MIlb->NC_CCweightLabel,one_matl,
                                                  Ghost::AroundCells, 1);

  t->computes(d_sharedState->get_switch_label(), level.get_rep());

  sched->addTask(t, level->eachPatch(),d_sharedState->allMaterials());

  if (one_matl->removeReference())
    delete one_matl; // shouln't happen, but...
}

void PBXTemperature::switchTest(const ProcessorGroup* group,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw)
{
  double sw = 0;

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
//    if (cout_doing.active()) {
//      cout_doing << "Doing PBXTemperature::switchTest on patch "
//                 <<  patch->getID() <<"\t " << "\n";
//    }
                                                                                
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(d_material);
    int indx = mpm_matl->getDWIndex();

    constNCVariable<double> gmass, gtemperature;
    constNCVariable<double> NC_CCweight;

    Ghost::GhostType  gac = Ghost::AroundCells;
                                                                                
    new_dw->get(gmass,        Mlb->gMassLabel,        indx, patch,gac, 1);
    new_dw->get(gtemperature, Mlb->gTemperatureLabel, indx, patch,gac, 1);
    old_dw->get(NC_CCweight, MIlb->NC_CCweightLabel,  0,    patch,gac, 1);
    IntVector nodeIdx[8];

    for(CellIterator iter =patch->getCellIterator();!iter.done();iter++){
      IntVector c = *iter;
      patch->findNodesFromCell(*iter,nodeIdx);

      double Temp_CC_mpm = 0.0;
      double cmass = 1.e-100;

      double MaxMass = d_SMALL_NUM;
      double MinMass = 1.0/d_SMALL_NUM;

      for (int in=0;in<8;in++){
        double NC_CCw_mass = NC_CCweight[nodeIdx[in]] * gmass[nodeIdx[in]];
        MaxMass = std::max(MaxMass,NC_CCw_mass);
        MinMass = std::min(MinMass,NC_CCw_mass);
        cmass    += NC_CCw_mass;
        Temp_CC_mpm += gtemperature[nodeIdx[in]] * NC_CCw_mass;
      }
      Temp_CC_mpm /= cmass;


      if ( (MaxMass-MinMass)/MaxMass > 0.4            //--------------KNOB 1
        && (MaxMass-MinMass)/MaxMass < 1.0
        &&  MaxMass > d_TINY_RHO){
        if(Temp_CC_mpm >= d_temperature){
         sw=1;
         break;
        }
      }
    }

  }

  max_vartype switch_condition(sw);
  new_dw->put(switch_condition,d_sharedState->get_switch_label(),getLevel(patches));
}
