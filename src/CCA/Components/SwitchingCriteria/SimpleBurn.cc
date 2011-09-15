/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#include <CCA/Components/SwitchingCriteria/SimpleBurn.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Labels/MPMICELabel.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Parallel/Parallel.h>
#include <string>
#include <iostream>

using namespace std;
using namespace Uintah;
static DebugStream dbg("SWITCHER", false);

SimpleBurnCriteria::SimpleBurnCriteria(ProblemSpecP& ps)
{
  ps->require("reactant_material",   d_material);
  ps->require("ThresholdTemperature",d_temperature);

  proc0cout << "Switching criteria:  \tSimpleBurn, reactant matl: " 
            << d_material << " Threshold tempterature " << d_temperature << endl;

  Mlb  = scinew MPMLabel();
  MIlb = scinew MPMICELabel();
}

SimpleBurnCriteria::~SimpleBurnCriteria()
{
  delete Mlb;
  delete MIlb;
}
//__________________________________
//
void SimpleBurnCriteria::problemSetup(const ProblemSpecP& ps, 
                                  const ProblemSpecP& restart_prob_spec, 
                                  SimulationStateP& state)
{
  d_sharedState = state;
}
//__________________________________
//
void SimpleBurnCriteria::scheduleSwitchTest(const LevelP& level, SchedulerP& sched)
{
  printSchedule(level,dbg,"Switching Criteria:SimpleBurnCriteria::scheduleSwitchTest");
  Task* t = scinew Task("switchTest", this, &SimpleBurnCriteria::switchTest);

  MaterialSubset* one_matl = scinew MaterialSubset();
  one_matl->add(0);
  one_matl->addReference();
  
  Ghost::GhostType  gac = Ghost::AroundCells;
  
  if(level->hasFinerLevel() == false){  // only on finest level
    t->requires(Task::NewDW, Mlb->gMassLabel,                gac, 1);
    t->requires(Task::NewDW, Mlb->gTemperatureLabel,one_matl,gac, 1);
    t->requires(Task::OldDW, Mlb->NC_CCweightLabel, one_matl,gac, 1);
  }

  t->computes(d_sharedState->get_switch_label());

  sched->addTask(t, level->eachPatch(),d_sharedState->allMaterials());

  if (one_matl->removeReference()){
    delete one_matl;
  }
}
//______________________________________________________________________
//  This task uses similar logic in the HEChem/simpleBurn.cc
//  to determine if the burning criteria has been reached.
void SimpleBurnCriteria::switchTest(const ProcessorGroup* group,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw)
{
  double timeToSwitch = 0;
  const Level* level = getLevel(patches);
  
  if (level->hasFinerLevel() == false){  // only on finest level

    for(int p=0;p<patches->size();p++){
      const Patch* patch = patches->get(p);
      printTask(patches, patch,dbg,"Doing SimpleBurnCriteria::switchTest");

      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(d_material);
      int indx = mpm_matl->getDWIndex();

      constNCVariable<double> gmass, gTempAllMatls;
      constNCVariable<double> NC_CCweight;
      Ghost::GhostType  gac = Ghost::AroundCells;

      new_dw->get(gmass,        Mlb->gMassLabel,        indx, patch,gac, 1);
      new_dw->get(gTempAllMatls,Mlb->gTemperatureLabel, 0,    patch,gac, 1);
      old_dw->get(NC_CCweight,  Mlb->NC_CCweightLabel,  0,    patch,gac, 1);

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
          Temp_CC_mpm += gTempAllMatls[nodeIdx[in]] * NC_CCw_mass;
        }
        Temp_CC_mpm /= cmass;
        
        double ratio = (MaxMass-MinMass)/MaxMass ;

        if ( ratio > 0.4 && ratio < 1.0 &&  MaxMass > d_TINY_RHO){
          if(Temp_CC_mpm >= d_temperature){
           timeToSwitch=1;
           cout << " \n"
                << " The simpleBurn switching criteria is satisfied in cell " << c
                << " (MaxMass-MinMass)/MaxMass = " << ratio
                << ", temp_CC_mpm= " << Temp_CC_mpm << "\n"<< endl;
           break;
          }
        }
      } // iter
    }  //patches
  } // finest Level

  max_vartype switch_condition(timeToSwitch);

  const Level* allLevels = 0;
  new_dw->put(switch_condition,d_sharedState->get_switch_label(), allLevels);
}
