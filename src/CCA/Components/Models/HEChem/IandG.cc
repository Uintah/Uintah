/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <CCA/Components/Models/HEChem/IandG.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Labels/ICELabel.h>
#include <CCA/Components/ICE/ICEMaterial.h>
#include <CCA/Components/ICE/BoundaryCond.h>
#include <iostream>
#include <Core/Util/DebugStream.h>

using namespace Uintah;
using namespace std;
//__________________________________
//  setenv SCI_DEBUG "MODELS_DOING_COUT:+"
//  MODELS_DOING_COUT:   dumps when tasks are scheduled and performed
static DebugStream cout_doing("MODELS_DOING_COUT", false);

IandG::IandG(const ProcessorGroup* myworld, ProblemSpecP& params)
  : ModelInterface(myworld), params(params)
{
  mymatls = 0;
  Ilb  = scinew ICELabel();
  //__________________________________
  //  diagnostic labels
  reactedFractionLabel   = VarLabel::create("IandG:F",
                     CCVariable<double>::getTypeDescription());
                     
  IandGterm1Label   = VarLabel::create("IandG:term1",
                     CCVariable<double>::getTypeDescription());
  IandGterm2Label   = VarLabel::create("IandG:term2",
                     CCVariable<double>::getTypeDescription());
  IandGterm3Label   = VarLabel::create("IandG:term3",
                     CCVariable<double>::getTypeDescription());
                     
}

IandG::~IandG()
{
  delete Ilb;

  VarLabel::destroy(reactedFractionLabel);
  VarLabel::destroy(IandGterm1Label);
  VarLabel::destroy(IandGterm2Label);
  VarLabel::destroy(IandGterm3Label);
  
  if(mymatls && mymatls->removeReference())
    delete mymatls;
}

void IandG::problemSetup(GridP&, SimulationStateP& sharedState,
                             ModelSetup*)
{
  ProblemSpecP IG_ps = params->findBlock("IandG");
  d_sharedState = sharedState;
  matl0 = sharedState->parseAndLookupMaterial(IG_ps, "fromMaterial");
  matl1 = sharedState->parseAndLookupMaterial(IG_ps, "toMaterial");
  IG_ps->require("I",  d_I);
  IG_ps->require("G1", d_G1);
  IG_ps->require("G2", d_G2);
  IG_ps->require("a",  d_a);
  IG_ps->require("b",  d_b);
  IG_ps->require("c",  d_c);
  IG_ps->require("d",  d_d);
  IG_ps->require("e",  d_e);
  IG_ps->require("g",  d_g);
  IG_ps->require("x",  d_x);
  IG_ps->require("y",  d_y);
  IG_ps->require("z",  d_z);
  IG_ps->require("Figmax", d_Figmax);
  IG_ps->require("FG1max", d_FG1max);
  IG_ps->require("FG2min", d_FG2min);
  IG_ps->require("rho0",   d_rho0);
  IG_ps->require("E0",     d_E0);
  IG_ps->require("ThresholdPressure",   d_threshold_pressure);


  //__________________________________
  //  define the materialSet
  mymatls = scinew MaterialSet();

  vector<int> m;
  m.push_back(0);                       // needed for the pressure and NC_CCWeight
  m.push_back(matl0->getDWIndex());
  m.push_back(matl1->getDWIndex());

  mymatls->addAll_unique(m);            // elimiate duplicate entries
  mymatls->addReference(); 
}

void IandG::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP model_ps = ps->appendChild("Model");
  model_ps->setAttribute("type","IandG");

  ProblemSpecP IG_ps = ps->appendChild("IandG");
  IG_ps->appendElement("fromMaterial",matl0->getName());
  IG_ps->appendElement("toMaterial",matl1->getName());
  IG_ps->appendElement("I",  d_I);
  IG_ps->appendElement("G1", d_G1);
  IG_ps->appendElement("G2", d_G2);
  IG_ps->appendElement("a",  d_a);
  IG_ps->appendElement("b",  d_b);
  IG_ps->appendElement("c",  d_c);
  IG_ps->appendElement("d",  d_d);
  IG_ps->appendElement("e",  d_e);
  IG_ps->appendElement("g",  d_g);
  IG_ps->appendElement("x",  d_x);
  IG_ps->appendElement("y",  d_y);
  IG_ps->appendElement("z",  d_z);
  IG_ps->appendElement("Figmax", d_Figmax);
  IG_ps->appendElement("FG1max", d_FG1max);
  IG_ps->appendElement("FG2min", d_FG2min);
  IG_ps->appendElement("rho0",   d_rho0);
  IG_ps->appendElement("E0",     d_E0);
  IG_ps->appendElement("ThresholdPressure",   d_threshold_pressure);
}


//______________________________________________________________________
//     
void IandG::scheduleInitialize(SchedulerP&,
                               const LevelP&,
                               const ModelInfo*)
{
  // None necessary...
}
//______________________________________________________________________
//      
void IandG::scheduleComputeStableTimestep(SchedulerP&,
                                          const LevelP&,
                                          const ModelInfo*)
{
  // None necessary...
}

//______________________________________________________________________
//     
void IandG::scheduleComputeModelSources(SchedulerP& sched,
                                        const LevelP& level,
                                        const ModelInfo* mi)
{
  Task* t = scinew Task("IandG::computeModelSources", this, 
                        &IandG::computeModelSources, mi);
  cout_doing << "IandG::scheduleComputeModelSources "<<  endl;  
  
  Ghost::GhostType  gn  = Ghost::None;
  const MaterialSubset* react_matl = matl0->thisMaterial();
  const MaterialSubset* prod_matl  = matl1->thisMaterial();
  MaterialSubset* one_matl     = scinew MaterialSubset();
  one_matl->add(0);
  one_matl->addReference();
  MaterialSubset* press_matl   = one_matl;
  
  t->requires( Task::OldDW, mi->delT_Label,        level.get_rep());
  //__________________________________
  // Products
  t->requires(Task::NewDW,  Ilb->rho_CCLabel,      prod_matl, gn);
  
  //__________________________________
  // Reactants
  t->requires(Task::NewDW, Ilb->sp_vol_CCLabel,    react_matl, gn);
  t->requires(Task::OldDW, Ilb->vel_CCLabel,       react_matl, gn);
  t->requires(Task::OldDW, Ilb->temp_CCLabel,      react_matl, gn);
  t->requires(Task::NewDW, Ilb->rho_CCLabel,       react_matl, gn);
  t->requires(Task::NewDW, Ilb->specific_heatLabel,react_matl, gn);
  
  t->requires(Task::NewDW, Ilb->press_equil_CCLabel, press_matl,gn);
  t->computes(reactedFractionLabel, react_matl);
  t->computes(IandGterm1Label, react_matl);
  t->computes(IandGterm2Label, react_matl);
  t->computes(IandGterm3Label, react_matl);

  t->modifies(mi->modelMass_srcLabel);
  t->modifies(mi->modelMom_srcLabel);
  t->modifies(mi->modelEng_srcLabel);
  t->modifies(mi->modelVol_srcLabel); 
  sched->addTask(t, level->eachPatch(), mymatls);

  if (one_matl->removeReference())
    delete one_matl;
}

//______________________________________________________________________
//
void IandG::computeModelSources(const ProcessorGroup*, 
                         const PatchSubset* patches,
                         const MaterialSubset*,
                         DataWarehouse* old_dw,
                         DataWarehouse* new_dw,
                         const ModelInfo* mi)
{
  delt_vartype delT;
  old_dw->get(delT, mi->delT_Label);

  int m0 = matl0->getDWIndex();
  int m1 = matl1->getDWIndex();
 
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);  
    
    cout_doing << "Doing computeModelSources on patch "<< patch->getID()
               <<"\t\t\t\t  IandG" << endl;
    CCVariable<double> mass_src_0, mass_src_1, mass_0;
    CCVariable<Vector> momentum_src_0, momentum_src_1;
    CCVariable<double> energy_src_0, energy_src_1;
    CCVariable<double> sp_vol_src_0, sp_vol_src_1;

    new_dw->getModifiable(mass_src_0,    mi->modelMass_srcLabel,  m0,patch);
    new_dw->getModifiable(momentum_src_0,mi->modelMom_srcLabel,   m0,patch);
    new_dw->getModifiable(energy_src_0,  mi->modelEng_srcLabel,   m0,patch);
    new_dw->getModifiable(sp_vol_src_0,  mi->modelVol_srcLabel,   m0,patch);

    new_dw->getModifiable(mass_src_1,    mi->modelMass_srcLabel,  m1,patch);
    new_dw->getModifiable(momentum_src_1,mi->modelMom_srcLabel,   m1,patch);
    new_dw->getModifiable(energy_src_1,  mi->modelEng_srcLabel,   m1,patch);
    new_dw->getModifiable(sp_vol_src_1,  mi->modelVol_srcLabel,   m1,patch);

    constCCVariable<double> press_CC, cv_reactant;
    constCCVariable<double> rctTemp,rctRho,rctSpvol,prodRho;
    constCCVariable<Vector> rctvel_CC;
    CCVariable<double> Fr;
    CCVariable<double> term1,term2,term3;
            
    Vector dx = patch->dCell();
    double cell_vol = dx.x()*dx.y()*dx.z();
    Ghost::GhostType  gn  = Ghost::None;    
   
    //__________________________________
    // Reactant data
    old_dw->get(rctTemp,       Ilb->temp_CCLabel,      m0,patch,gn, 0);
    old_dw->get(rctvel_CC,     Ilb->vel_CCLabel,       m0,patch,gn, 0);
    new_dw->get(rctRho,        Ilb->rho_CCLabel,       m0,patch,gn, 0);
    new_dw->get(rctSpvol,      Ilb->sp_vol_CCLabel,    m0,patch,gn, 0);
    new_dw->get(cv_reactant,   Ilb->specific_heatLabel,m0,patch,gn, 0);
    new_dw->allocateAndPut(Fr,reactedFractionLabel,m0,patch);
    new_dw->allocateAndPut(term1, IandGterm1Label, m0,patch);
    new_dw->allocateAndPut(term2, IandGterm2Label, m0,patch);
    new_dw->allocateAndPut(term3, IandGterm3Label, m0,patch);
    Fr.initialize(0.);
    term1.initialize(0.);
    term2.initialize(0.);
    term3.initialize(0.);

    //__________________________________
    // Product Data, 
    new_dw->get(prodRho,       Ilb->rho_CCLabel,        m1, patch,gn, 0);

    //__________________________________
    //   Misc.
    new_dw->get(press_CC,      Ilb->press_equil_CCLabel,0,  patch,gn, 0);
  

    for (CellIterator iter = patch->getCellIterator();!iter.done();iter++){
      IntVector c = *iter;
      if (press_CC[c] > d_threshold_pressure){
        //__________________________________
        // Insert Burn Model Here
        double burnedMass = 0.;
        double delF = 0;
        double F = prodRho[c]/(rctRho[c]+prodRho[c]);
        if(F >= 0. && F < d_Figmax){
         term1[c]=d_I*pow((1.-F),d_b)*pow((1./(rctSpvol[c]*d_rho0)-1.-d_a),d_x);
         delF += term1[c];
        }
        if(F >= 0. && F < d_FG1max){
          term2[c] = d_G1*pow((1.-F),d_c)*pow(F,d_d)*pow(press_CC[c],d_y);
          delF += term2[c];
        }
        if(F >= d_FG2min && F < 0.99){
          term3[c] = d_G2*pow((1.-F),d_e)*pow(F,d_g)*pow(press_CC[c],d_z);
          delF += term3[c];
        }
        delF*=delT;
        Fr[c] = F;
        double rctMass = rctRho[c]*cell_vol;
        double prdMass = prodRho[c]*cell_vol;
        burnedMass = min(delF*(prdMass+rctMass), rctMass);
        burnedMass = min(burnedMass, .2*d_rho0*cell_vol);

        //__________________________________
        // conservation of mass, momentum and energy                           
        mass_src_0[c] -= burnedMass;
        mass_src_1[c] += burnedMass;
           
        Vector momX        = rctvel_CC[c] * burnedMass;
        momentum_src_0[c] -= momX;
        momentum_src_1[c] += momX;

        double energyX   = cv_reactant[c]*rctTemp[c]*burnedMass; 
        double releasedHeat = burnedMass * d_E0;
        energy_src_0[c] -= energyX;
        energy_src_1[c] += energyX + releasedHeat;

        double createdVolx  = burnedMass * rctSpvol[c];
        sp_vol_src_0[c] -= createdVolx;
        sp_vol_src_1[c] += createdVolx;
      }  // if (pressure)
    }  // cell iterator  

    //__________________________________
    //  set symetric BC
    setBC(mass_src_0, "set_if_sym_BC",patch, d_sharedState, m0, new_dw);
    setBC(mass_src_1, "set_if_sym_BC",patch, d_sharedState, m1, new_dw);
    setBC(term1, "set_if_sym_BC",patch, d_sharedState, m0, new_dw);
    setBC(term2, "set_if_sym_BC",patch, d_sharedState, m0, new_dw);
    setBC(term3, "set_if_sym_BC",patch, d_sharedState, m0, new_dw);
    setBC(Fr,    "set_if_sym_BC",patch, d_sharedState, m0, new_dw);
  }
}
//______________________________________________________________________
//
void IandG::scheduleModifyThermoTransportProperties(SchedulerP&,
                                                    const LevelP&,         
                                                    const MaterialSet*)    
{
  // do nothing      
}
void IandG::computeSpecificHeat(CCVariable<double>&,
                                const Patch*,   
                                DataWarehouse*, 
                                const int)      
{
  //do nothing
}
//______________________________________________________________________
//
void IandG::scheduleErrorEstimate(const LevelP&,
                                  SchedulerP&)
{
  // Not implemented yet
}
//__________________________________
void IandG::scheduleTestConservation(SchedulerP&,
                                     const PatchSet*,                
                                     const ModelInfo*)               
{
  // Not implemented yet
}

