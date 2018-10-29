/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


#include <CCA/Components/Models/HEChem/Simple_Burn.h>
#include <CCA/Components/Models/HEChem/Common.h>

#include <CCA/Components/ICE/Core/ICELabel.h>
#include <CCA/Components/ICE/CustomBCs/BoundaryCond.h>
#include <CCA/Components/MPM/Core/MPMLabel.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>
#include <CCA/Components/MPMICE/Core/MPMICELabel.h>

#include <CCA/Ports/Scheduler.h>

#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Util/DebugStream.h>

#include <iostream>

using namespace Uintah;
using namespace std;
     
#define d_SMALL_NUM 1e-100
#define d_TINY_RHO 1e-12

//__________________________________
//  setenv SCI_DEBUG "MPMICE_NORMAL_COUT:+,MODELS_DOING_COUT:+"
//  MODELS_DOING_COUT:   dumps when tasks are scheduled and performed
static DebugStream cout_doing("MODELS_DOING_COUT", false);

Simple_Burn::Simple_Burn(const ProcessorGroup* myworld, 
                         const MaterialManagerP& materialManager,
                         const ProblemSpecP& params,
                         const ProblemSpecP& prob_spec)
  : HEChemModel(myworld, materialManager),
    d_params(params), d_prob_spec(prob_spec)
{
  mymatls = 0;
  Mlb  = scinew MPMLabel();
  Ilb  = scinew ICELabel();
  MIlb = scinew MPMICELabel();
  
  d_saveConservedVars = scinew saveConservedVars();
  //__________________________________
  //  diagnostic labels
  onSurfaceLabel   = VarLabel::create("Simple_Burn::onSurface",
                     CCVariable<double>::getTypeDescription());
                     
  surfaceTempLabel = VarLabel::create("Simple_Burn::surfaceTemp",
                     CCVariable<double>::getTypeDescription());
                     
  totalMassBurnedLabel  = VarLabel::create( "totalMassBurned",
                     sum_vartype::getTypeDescription() );
                     
  totalHeatReleasedLabel= VarLabel::create( "totalHeatReleased",
                     sum_vartype::getTypeDescription() );
                
}
//______________________________________________________________________
//
Simple_Burn::~Simple_Burn()
{
  delete Ilb;
  delete Mlb;
  delete MIlb;
  delete d_saveConservedVars;
  
  VarLabel::destroy(surfaceTempLabel);
  VarLabel::destroy(onSurfaceLabel);
  VarLabel::destroy(totalMassBurnedLabel);
  VarLabel::destroy(totalHeatReleasedLabel);
  
  if(mymatls && mymatls->removeReference())
    delete mymatls;
}
//______________________________________________________________________
//
void Simple_Burn::problemSetup(GridP&,
                                const bool isRestart)
{
  ProblemSpecP SB_ps = d_params->findBlock("Simple_Burn");
  SB_ps->require("ThresholdTemp",    d_thresholdTemp);
  SB_ps->require("ThresholdPressure",d_thresholdPress);
  
  matl0 = m_materialManager->parseAndLookupMaterial(SB_ps, "fromMaterial");
  matl1 = m_materialManager->parseAndLookupMaterial(SB_ps, "toMaterial");
  
  SB_ps->require("Enthalpy",         d_Enthalpy);
  SB_ps->require("BurnCoeff",        d_BurnCoeff);
  SB_ps->require("refPressure",      d_refPress);

  //__________________________________
  //  define the materialSet
  mymatls = scinew MaterialSet();

  vector<int> m;
  m.push_back(0);                                // needed for the pressure and NC_CCWeight
  m.push_back(matl0->getDWIndex());
  m.push_back(matl1->getDWIndex());

  mymatls->addAll_unique(m);                    // elimiate duplicate entries
  mymatls->addReference();
  
  //__________________________________
  //  Are we saving the total burned mass and total burned energy
  ProblemSpecP DA_ps = d_prob_spec->findBlock("DataArchiver");
  for (ProblemSpecP child = DA_ps->findBlock("save"); child != nullptr; child = child->findNextBlock("save")) {
    map<string,string> var_attr;
    child->getAttributes(var_attr);
    if (var_attr["label"] == "totalMassBurned"){
      d_saveConservedVars->mass  = true;
    }
    if (var_attr["label"] == "totalHeatReleased"){
      d_saveConservedVars->energy = true;
    }
  }
}
//______________________________________________________________________
//
void Simple_Burn::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP model_ps = ps->appendChild("Model");
  model_ps->setAttribute("type","Simple_Burn");
  ProblemSpecP SB_ps = model_ps->appendChild("Simple_Burn");
  
  SB_ps->appendElement("ThresholdTemp",     d_thresholdTemp);
  SB_ps->appendElement("ThresholdPressure", d_thresholdPress);
  SB_ps->appendElement("fromMaterial",      matl0->getName());
  SB_ps->appendElement("toMaterial",        matl1->getName());
  SB_ps->appendElement("Enthalpy",          d_Enthalpy);
  SB_ps->appendElement("BurnCoeff",         d_BurnCoeff);
  SB_ps->appendElement("refPressure",       d_refPress);
}

//______________________________________________________________________
//     
void Simple_Burn::scheduleInitialize(SchedulerP&,
                                     const LevelP&)
{
  // None necessary...
}
//______________________________________________________________________
//      
void Simple_Burn::scheduleComputeStableTimeStep(SchedulerP&,
                                              const LevelP&)
{
  // None necessary...
}

//______________________________________________________________________
//     
void Simple_Burn::scheduleComputeModelSources(SchedulerP& sched,
                                                  const LevelP& level)
{
  if (level->hasFinerLevel()){  // only on finest level
    return;
  }  
 
  Task* t = scinew Task("Simple_Burn::computeModelSources",this, 
                        &Simple_Burn::computeModelSources);
                                            
  cout_doing << "SIMPLE_BURN::scheduleComputeModelSources "<<  endl;  
  
  Ghost::GhostType  gac = Ghost::AroundCells;  
  Ghost::GhostType  gn  = Ghost::None;
  const MaterialSubset* react_matl = matl0->thisMaterial();
  const MaterialSubset* prod_matl  = matl1->thisMaterial();
  MaterialSubset* one_matl     = scinew MaterialSubset();
  one_matl->add(0);
  one_matl->addReference();
  MaterialSubset* press_matl   = one_matl;

  // Used for getting temperature and volume fraction for all materials for
  //  for burning criteria
  const MaterialSet* all_matls = m_materialManager->allMaterials();
  const MaterialSubset* all_matls_sub = all_matls->getUnion();  
  Task::MaterialDomainSpec oms = Task::OutOfDomain;  //outside of mymatl set.
  t->requires(Task::OldDW, Ilb->temp_CCLabel,      all_matls_sub, oms, gac,1);
  t->requires(Task::NewDW, Ilb->vol_frac_CCLabel,  all_matls_sub, oms, gac,1);

  t->requires( Task::OldDW, Ilb->timeStepLabel );
  t->requires( Task::OldDW, Ilb->delTLabel,       level.get_rep());
  //__________________________________
  // Products
  t->requires(Task::OldDW,  Ilb->temp_CCLabel,    prod_matl, gn);       
  t->requires(Task::NewDW,  Ilb->vol_frac_CCLabel,prod_matl, gn);       
  t->requires(Task::NewDW,  Ilb->TempX_FCLabel,   prod_matl, gac,2);    
  t->requires(Task::NewDW,  Ilb->TempY_FCLabel,   prod_matl, gac,2);    
  t->requires(Task::NewDW,  Ilb->TempZ_FCLabel,   prod_matl, gac,2);
    
  t->requires(Task::NewDW,  Ilb->press_equil_CCLabel, press_matl,gn);
  t->requires(Task::OldDW,  Mlb->NC_CCweightLabel,   one_matl,  gac, 1);
  
  //__________________________________
  // Reactants
  t->requires(Task::NewDW, Ilb->sp_vol_CCLabel,   react_matl, gn);
  t->requires(Task::NewDW, MIlb->vel_CCLabel,     react_matl, gn);
  t->requires(Task::NewDW, MIlb->temp_CCLabel,    react_matl, gn);
  t->requires(Task::NewDW, MIlb->cMassLabel,      react_matl, gn);
  t->requires(Task::NewDW, Mlb->gMassLabel,       react_matl, gac,1);  

  t->computes(Simple_Burn::onSurfaceLabel,     one_matl);
  t->computes(Simple_Burn::surfaceTempLabel,   one_matl);
  
  if(d_saveConservedVars->mass ){
    t->computes(Simple_Burn::totalMassBurnedLabel);
  }
  if(d_saveConservedVars->energy){
    t->computes(Simple_Burn::totalHeatReleasedLabel);
  }
  
  
  t->modifies(Ilb->modelMass_srcLabel);
  t->modifies(Ilb->modelMom_srcLabel);
  t->modifies(Ilb->modelEng_srcLabel);
  t->modifies(Ilb->modelVol_srcLabel); 
  sched->addTask(t, level->eachPatch(), mymatls);

  if (one_matl->removeReference())
    delete one_matl;
}

//______________________________________________________________________
//
void Simple_Burn::computeModelSources(const ProcessorGroup*, 
                                         const PatchSubset* patches,
                                         const MaterialSubset*,
                                         DataWarehouse* old_dw,
                                         DataWarehouse* new_dw)
{
  timeStep_vartype timeStep;
  old_dw->get(timeStep, Ilb->timeStepLabel );

  bool isNotInitialTimeStep = (timeStep > 0);

  delt_vartype delT;
  old_dw->get(delT, Ilb->delTLabel,getLevel(patches));

  int m0 = matl0->getDWIndex();
  int m1 = matl1->getDWIndex();
  double totalBurnedMass = 0;
  double totalHeatReleased = 0;
 
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);  
    
    cout_doing << "Doing computeModelSources on patch "<< patch->getID()
               <<"\t\t\t\t  Simple_Burn" << endl;
    CCVariable<double> mass_src_0, mass_src_1;
    CCVariable<Vector> momentum_src_0, momentum_src_1;
    CCVariable<double> energy_src_0, energy_src_1;
    CCVariable<double> sp_vol_src_0, sp_vol_src_1;
    CCVariable<double> onSurface, surfaceTemp;
    
    new_dw->getModifiable(mass_src_0,    Ilb->modelMass_srcLabel,  m0,patch);
    new_dw->getModifiable(momentum_src_0,Ilb->modelMom_srcLabel,   m0,patch);
    new_dw->getModifiable(energy_src_0,  Ilb->modelEng_srcLabel,   m0,patch);
    new_dw->getModifiable(sp_vol_src_0,  Ilb->modelVol_srcLabel,   m0,patch);

    new_dw->getModifiable(mass_src_1,    Ilb->modelMass_srcLabel,  m1,patch);
    new_dw->getModifiable(momentum_src_1,Ilb->modelMom_srcLabel,   m1,patch);
    new_dw->getModifiable(energy_src_1,  Ilb->modelEng_srcLabel,   m1,patch);
    new_dw->getModifiable(sp_vol_src_1,  Ilb->modelVol_srcLabel,   m1,patch);
 
    constCCVariable<double> press_CC,gasTemp,gasVol_frac;
    constCCVariable<double> solidTemp,solidMass,solidSp_vol;

    constNCVariable<double> NC_CCweight,NCsolidMass;
    constSFCXVariable<double> gasTempX_FC;
    constSFCYVariable<double> gasTempY_FC;
    constSFCZVariable<double> gasTempZ_FC;
    constCCVariable<Vector> vel_CC;
    
    Vector dx = patch->dCell();
    Ghost::GhostType  gn  = Ghost::None;    
    Ghost::GhostType  gac = Ghost::AroundCells;   
   
    //__________________________________
    // Reactant data
    new_dw->get(solidTemp,       MIlb->temp_CCLabel, m0,patch,gn, 0);
    new_dw->get(solidMass,       MIlb->cMassLabel,   m0,patch,gn, 0);
    new_dw->get(solidSp_vol,     Ilb->sp_vol_CCLabel,m0,patch,gn, 0);
    new_dw->get(vel_CC,          MIlb->vel_CCLabel,  m0,patch,gn, 0);
    new_dw->get(NCsolidMass,     Mlb->gMassLabel,    m0,patch,gac,1);

    //__________________________________
    // Product Data, 
    new_dw->get(gasTempX_FC,      Ilb->TempX_FCLabel,m1,patch,gac,2);
    new_dw->get(gasTempY_FC,      Ilb->TempY_FCLabel,m1,patch,gac,2);
    new_dw->get(gasTempZ_FC,      Ilb->TempZ_FCLabel,m1,patch,gac,2);
    old_dw->get(gasTemp,          Ilb->temp_CCLabel, m1,patch,gn, 0);
    new_dw->get(gasVol_frac,      Ilb->vol_frac_CCLabel,  m1, patch,gn, 0);
    //__________________________________
    //   Misc.
    new_dw->get(press_CC,         Ilb->press_equil_CCLabel,0,  patch,gn, 0);
    old_dw->get(NC_CCweight,     Mlb->NC_CCweightLabel,  0,   patch,gac,1);   
  
    new_dw->allocateAndPut(onSurface,  Simple_Burn::onSurfaceLabel,   0, patch);
    new_dw->allocateAndPut(surfaceTemp,Simple_Burn::surfaceTempLabel, 0, patch);
 
    IntVector nodeIdx[8];
    
    MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM", m0);
    double cv_solid = mpm_matl->getSpecificHeat();
   

    // Get all Temperatures for burning check
    int numAllMatls = m_materialManager->getNumMatls();
    std::vector<constCCVariable<double> >  vol_frac_CC(numAllMatls);
    std::vector<constCCVariable<double> >  temp_CC(numAllMatls);
    for (int m = 0; m < numAllMatls; m++) {
      Material* matl = m_materialManager->getMaterial(m);
      int indx = matl->getDWIndex();
      old_dw->get(temp_CC[m],       MIlb->temp_CCLabel,    indx, patch, gac, 1);
      new_dw->get(vol_frac_CC[m],   Ilb->vol_frac_CCLabel, indx, patch, gac, 1);
    }

 
    for (CellIterator iter = patch->getCellIterator();!iter.done();iter++){
      IntVector c = *iter;

     //__________________________________
     // Find if the cell contains surface:
      patch->findNodesFromCell(*iter,nodeIdx);
      double MaxMass = d_SMALL_NUM;
      double MinMass = 1.0/d_SMALL_NUM;                 
      for (int nN=0; nN<8; nN++) {
        MaxMass = std::max(MaxMass,NC_CCweight[nodeIdx[nN]]*
                                   NCsolidMass[nodeIdx[nN]]);
        MinMass = std::min(MinMass,NC_CCweight[nodeIdx[nN]]*
                                   NCsolidMass[nodeIdx[nN]]); 
      }               


      //===============================================
      //If you change the burning criteria logic you must also modify
      //CCA/Components/SwitchCriteria
      //===============================================
      if ( (MaxMass-MinMass)/MaxMass > 0.4            //--------------KNOB 1
        && (MaxMass-MinMass)/MaxMass < 1.0
        &&  MaxMass > d_TINY_RHO){

        //__________________________________
        //  Determine the temperature
        //  to use in burn model
        double Temp = 0;

        // Check if any material occupies more than 20 % of cell and has temperature larger than
        //  ignition temperature, use that temperature as the criteria for burning
        for (int m = 0; m < numAllMatls; m++){
          if(vol_frac_CC[m][c] > 0.2 && temp_CC[m][c] > d_thresholdTemp && temp_CC[m][c] > Temp){
            Temp = temp_CC[m][c];
          }
        }

        surfaceTemp[c] = Temp;

        Vector rhoGradVector = computeDensityGradientVector(nodeIdx, NCsolidMass, NC_CCweight, dx);
        double surfArea = computeSurfaceArea(rhoGradVector, dx);  
        onSurface[c] = surfArea; // debugging var

        //__________________________________
        //  Simple Burn Model
        double burnedMass = 0.0;
        if ((Temp > d_thresholdTemp) && (press_CC[c] > d_thresholdPress)) {
          burnedMass = delT *surfArea * d_BurnCoeff 
                       * pow((press_CC[c]/d_refPress),0.778);
        }
        if(burnedMass > solidMass[c]){
          burnedMass = solidMass[c];
        }

        //__________________________________
        // conservation of mass, momentum and energy                           
        mass_src_0[c] -= burnedMass;
        mass_src_1[c] += burnedMass;
        totalBurnedMass += burnedMass;
           
        Vector momX        = vel_CC[c] * burnedMass;
        momentum_src_0[c] -= momX;
        momentum_src_1[c] += momX;

        double energyX   = cv_solid*solidTemp[c]*burnedMass; 
        double releasedHeat = burnedMass * d_Enthalpy;
        energy_src_0[c] -= energyX;
        energy_src_1[c] += energyX + releasedHeat;
        totalHeatReleased += releasedHeat;

        double createdVolx  = burnedMass * solidSp_vol[c];
        sp_vol_src_0[c] -= createdVolx;
        sp_vol_src_1[c] += createdVolx;
      }  // if (maxMass-MinMass....)
    }  // cell iterator  

    //__________________________________
    //  set symetric BC
    setBC(mass_src_0, "set_if_sym_BC",patch, m_materialManager, m0, new_dw, isNotInitialTimeStep);
    setBC(mass_src_1, "set_if_sym_BC",patch, m_materialManager, m1, new_dw, isNotInitialTimeStep);
   
  }
  //__________________________________
  //save total quantities
  if(d_saveConservedVars->mass ){
   new_dw->put(sum_vartype(totalBurnedMass),  Simple_Burn::totalMassBurnedLabel);
  }
  if(d_saveConservedVars->energy){
   new_dw->put(sum_vartype(totalHeatReleased),Simple_Burn::totalHeatReleasedLabel);
  }
  
}
