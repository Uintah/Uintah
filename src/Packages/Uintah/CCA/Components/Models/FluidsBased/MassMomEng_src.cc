/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
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



#include <Packages/Uintah/CCA/Components/Models/FluidsBased/MassMomEng_src.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Material.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Labels/MPMICELabel.h>
#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <iostream>

using namespace Uintah;
using namespace std;

MassMomEng_src::MassMomEng_src(const ProcessorGroup* myworld, ProblemSpecP& params)
  : ModelInterface(myworld), params(params)
{
  mymatls = 0;
  MIlb  = scinew MPMICELabel();
  totalMass_srcLabel = 0;
  totalIntEng_srcLabel = 0;
}

MassMomEng_src::~MassMomEng_src()
{
  delete MIlb;
  if(mymatls && mymatls->removeReference())
    delete mymatls;

  if(0!=totalMass_srcLabel)
    VarLabel::destroy(totalMass_srcLabel);
  
  if(0!=totalIntEng_srcLabel)
    VarLabel::destroy(totalIntEng_srcLabel);
}



//______________________________________________________________________
void MassMomEng_src::problemSetup(GridP&, SimulationStateP& sharedState,
			     ModelSetup* )
{
  d_matl = sharedState->parseAndLookupMaterial(params, "Material");
  params->require("rate", d_rate);

  vector<int> m(1);
  m[0] = d_matl->getDWIndex();
  mymatls = scinew MaterialSet();
  mymatls->addAll(m);
  mymatls->addReference();   
  
  totalMass_srcLabel  = VarLabel::create( "TotalMass_src",
                                        sum_vartype::getTypeDescription() );

  totalIntEng_srcLabel  = VarLabel::create("TotalIntEng_src",
                                        sum_vartype::getTypeDescription() );
}

//______________________________________________________________________
void MassMomEng_src::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP model_ps = ps->appendChild("Model");
  model_ps->setAttribute("type","MassMomEng_src");
  model_ps->appendElement("Material",d_matl->getName());
  model_ps->appendElement("rate",     d_rate );
}
 
//______________________________________________________________________
void MassMomEng_src::scheduleInitialize(SchedulerP&,
				   const LevelP& level,
				   const ModelInfo*)
{
  // None necessary...
}

//______________________________________________________________________     
void MassMomEng_src::scheduleComputeStableTimestep(SchedulerP&,
					      const LevelP&,
					      const ModelInfo*)
{
  // None necessary...
}

//__________________________________      
void MassMomEng_src::scheduleComputeModelSources(SchedulerP& sched,
				                const LevelP& level,
				                const ModelInfo* mi)
{
  Task* t = scinew Task("MassMomEng_src::computeModelSources",this, 
                        &MassMomEng_src::computeModelSources, mi);
  t->modifies(mi->modelMass_srcLabel);
  t->modifies(mi->modelMom_srcLabel);
  t->modifies(mi->modelEng_srcLabel);
  t->modifies(mi->modelVol_srcLabel);

  t->computes(MassMomEng_src::totalMass_srcLabel);
  t->computes(MassMomEng_src::totalIntEng_srcLabel);
  
  t->requires( Task::OldDW, mi->delT_Label);
  sched->addTask(t, level->eachPatch(), mymatls);
}

//__________________________________
void MassMomEng_src::computeModelSources(const ProcessorGroup*, 
			                    const PatchSubset* patches,
			                    const MaterialSubset* matls,
			                    DataWarehouse* old_dw,
			                    DataWarehouse* new_dw,
			                    const ModelInfo* mi)
{
  delt_vartype delT;
  old_dw->get(delT, mi->delT_Label);
  double dt = delT;

  int indx = d_matl->getDWIndex();
  double totalMass_src = 0.0;
  double totalIntEng_src = 0.0;
  
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);  
    CCVariable<double> mass_src;
    CCVariable<Vector> mom_src;
    CCVariable<double> eng_src;
    CCVariable<double> sp_vol_src;
    
    new_dw->getModifiable(mass_src,   mi->modelMass_srcLabel, indx, patch);
    new_dw->getModifiable(mom_src,    mi->modelMom_srcLabel,  indx, patch);
    new_dw->getModifiable(eng_src,    mi->modelEng_srcLabel,  indx, patch);
    new_dw->getModifiable(sp_vol_src, mi->modelVol_srcLabel,  indx, patch);

    //__________________________________
    //  Do some work
    for(CellIterator iter = patch->getExtraCellIterator__New(); !iter.done(); iter++){
      IntVector c = *iter;
    }
    new_dw->put(sum_vartype(totalMass_src),  MassMomEng_src::totalMass_srcLabel);
    new_dw->put(sum_vartype(totalIntEng_src),MassMomEng_src::totalIntEng_srcLabel);
  }
}
//______________________________________________________________________  
   
void MassMomEng_src::scheduleModifyThermoTransportProperties(SchedulerP&,
                                                        const LevelP&,
                                                        const MaterialSet*)
{
  // do nothing      
}
void MassMomEng_src::computeSpecificHeat(CCVariable<double>&,
                                    const Patch*,   
                                    DataWarehouse*, 
                                    const int)      
{
  //do nothing
}
//______________________________________________________________________
//
void MassMomEng_src::scheduleErrorEstimate(const LevelP&,
                                      SchedulerP&)
{
  // Not implemented yet
}
//__________________________________
void MassMomEng_src::scheduleTestConservation(SchedulerP&,
                                         const PatchSet*,
                                         const ModelInfo*)
{
  // Not implemented yet
}
