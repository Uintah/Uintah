/*

The MIT License

Copyright (c) 1997-2010 Center for the Simulation of Accidental Fires and 
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



#include <CCA/Components/Models/HEChem/DDT1.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Labels/ICELabel.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Labels/MPMICELabel.h>
#include <CCA/Components/ICE/ICEMaterial.h>
#include <CCA/Components/ICE/BoundaryCond.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <iostream>
#include <Core/Util/DebugStream.h>

using namespace Uintah;
using namespace std;
//__________________________________
//  setenv SCI_DEBUG "MODELS_DOING_COUT:+"
//  MODELS_DOING_COUT:   dumps when tasks are scheduled and performed
static DebugStream cout_doing("MODELS_DOING_COUT", false);

const double DDT1::EPSILON   = 1e-6;   /* stop epsilon for Bisection-Newton method */

DDT1::DDT1(const ProcessorGroup* myworld,
                         ProblemSpecP& params,
                         const ProblemSpecP& prob_spec)
  : ModelInterface(myworld), d_prob_spec(prob_spec), d_params(params)
{
  mymatls = 0;
  Ilb  = scinew ICELabel();
  MIlb = scinew MPMICELabel();
  Mlb  = scinew MPMLabel();
  //__________________________________
  //  diagnostic labels JWL++
  reactedFractionLabel   = VarLabel::create("F",
                                      CCVariable<double>::getTypeDescription());
                     
  delFLabel   = VarLabel::create("delF",
                                      CCVariable<double>::getTypeDescription());

  detonatingLabel = VarLabel::create("detonating",
                                      CCVariable<double>::getTypeDescription());
  //__________________________________
  //  diagnostic labels Simple Burn    
  d_saveConservedVars = scinew saveConservedVars();
  BurningCellLabel = VarLabel::create("DDT1.BurningCell",
                                      CCVariable<double>::getTypeDescription());
  TsLabel          = VarLabel::create("DDT1.SurfTemp",
                                      CCVariable<double>::getTypeDescription());
  numPPCLabel      = VarLabel::create("DDT1.numPPC",
                                      CCVariable<double>::getTypeDescription());
  surfaceTempLabel      = VarLabel::create("DDT1.surfaceTemp",
                                      CCVariable<double>::getTypeDescription());
  burningLabel = VarLabel::create("burning",
                                      CCVariable<double>::getTypeDescription());
  onSurfaceLabel = VarLabel::create("DDT1.onSurface",
                                      CCVariable<double>::getTypeDescription());
    
  totalMassBurnedLabel  = VarLabel::create( "totalMassBurned",
                                             sum_vartype::getTypeDescription() );
  totalHeatReleasedLabel= VarLabel::create( "totalHeatReleased",
                                             sum_vartype::getTypeDescription() );
}

DDT1::~DDT1()
{
  delete Ilb;
  delete MIlb;
  delete Mlb;

  // JWL++
  VarLabel::destroy(reactedFractionLabel);
  VarLabel::destroy(delFLabel);
  VarLabel::destroy(detonatingLabel);
  // Simple Burn
  VarLabel::destroy(BurningCellLabel);
  VarLabel::destroy(burningLabel);
  VarLabel::destroy(TsLabel);
  VarLabel::destroy(numPPCLabel);
  VarLabel::destroy(surfaceTempLabel);
  VarLabel::destroy(onSurfaceLabel);
  VarLabel::destroy(totalMassBurnedLabel);
  VarLabel::destroy(totalHeatReleasedLabel);

  if(mymatls && mymatls->removeReference())
    delete mymatls;
}

void DDT1::problemSetup(GridP&, SimulationStateP& sharedState,
			     ModelSetup*)
{
  d_sharedState = sharedState;
  // Required for JWL++
  d_params->require("ThresholdPressureJWL",   d_threshold_pressure);
  d_params->require("fromMaterial",fromMaterial);
  d_params->require("toMaterial",toMaterial);
  d_params->require("G",    d_G);
  d_params->require("b",    d_b);
  d_params->require("E0",   d_E0);
  d_params->require("rho0", d_rho0);

  d_params->require("IdealGasConst",     R );
  d_params->require("PreExpCondPh",      Ac);
  d_params->require("ActEnergyCondPh",   Ec);
  d_params->require("PreExpGasPh",       Bg);
  d_params->require("CondPhaseHeat",     Qc);
  d_params->require("GasPhaseHeat",      Qg);
  d_params->require("HeatConductGasPh",  Kg);
  d_params->require("HeatConductCondPh", Kc);
  d_params->require("SpecificHeatBoth",  Cp);
  d_params->require("MoleWeightGasPh",   MW);
  d_params->require("BoundaryParticles", BP);
  d_params->require("ThresholdPressure", ThresholdPressure);
  d_params->require("IgnitionTemp",      ignitionTemp);
 
  matl0 = sharedState->parseAndLookupMaterial(d_params, "fromMaterial");
  matl1 = sharedState->parseAndLookupMaterial(d_params, "toMaterial");

   
  /* initialize constants */
  CC1 = Ac*R*Kc/Ec/Cp;        
  CC2 = Qc/Cp/2;              
  CC3 = 4*Kg*Bg*MW*MW/Cp/R/R;  
  CC4 = Qc/Cp;                
  CC5 = Qg/Cp;                
    
  //__________________________________
  //  Are we saving the total burned mass and total burned energy
  ProblemSpecP DA_ps = d_prob_spec->findBlock("DataArchiver");
  for (ProblemSpecP child = DA_ps->findBlock("save");
       child != 0;
       child = child->findNextBlock("save") ){
       map<string,string> var_attr;
       child->getAttributes(var_attr);
       if (var_attr["label"] == "totalMassBurned"){
           d_saveConservedVars->mass  = true;
       }
       if (var_attr["label"] == "totalHeatReleased"){
           d_saveConservedVars->energy = true;
       }
  }
    
  /*  define the materialSet */
  vector<int> m_tmp(2);
  m_tmp[0] = matl0->getDWIndex();
  m_tmp[1] = matl1->getDWIndex();
  mymatls = scinew MaterialSet();            

  if( m_tmp[0] != 0 && m_tmp[1] != 0){
      vector<int> m(3);
      m[0] = 0; /* needed for the pressure and NC_CCWeight */ 
      m[1] = m_tmp[0];
      m[2] = m_tmp[1];
      mymatls->addAll(m);
  }else{
      vector<int> m(2);
      m[0] = m_tmp[0];
      m[1] = m_tmp[1];
      mymatls->addAll(m);
  }
  mymatls->addReference();
}

void DDT1::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP model_ps = ps->appendChild("Model");
  model_ps->setAttribute("type","DDT1");

  // JWL++ inputs
  model_ps->appendElement("ThresholdPressureJWL",d_threshold_pressure);
  model_ps->appendElement("fromMaterial",fromMaterial);
  model_ps->appendElement("toMaterial",toMaterial);
  model_ps->appendElement("G",    d_G);
  model_ps->appendElement("b",    d_b);
  model_ps->appendElement("E0",   d_E0);
  model_ps->appendElement("rho0", d_rho0);
  model_ps->appendElement("ThresholdTemp",     ignitionTemp);
  model_ps->appendElement("ThresholdPressureSB", ThresholdPressure);
  model_ps->appendElement("fromMaterial",      matl0->getName());

  // Steady Burn inputs
  model_ps->appendElement("IdealGasConst",     R );
  model_ps->appendElement("PreExpCondPh",      Ac);
  model_ps->appendElement("ActEnergyCondPh",   Ec);
  model_ps->appendElement("PreExpGasPh",       Bg);
  model_ps->appendElement("CondPhaseHeat",     Qc);
  model_ps->appendElement("GasPhaseHeat",      Qg);
  model_ps->appendElement("HeatConductGasPh",  Kg);
  model_ps->appendElement("HeatConductCondPh", Kc);
  model_ps->appendElement("SpecificHeatBoth",  Cp);
  model_ps->appendElement("MoleWeightGasPh",   MW);
  model_ps->appendElement("BoundaryParticles", BP);
  model_ps->appendElement("ThresholdPressure", ThresholdPressure);
  model_ps->appendElement("IgnitionTemp",      ignitionTemp);
}


//______________________________________________________________________
//     
void DDT1::scheduleInitialize(SchedulerP& sched,
                               const LevelP& level,
                               const ModelInfo*)
{
  printSchedule(level,"DDT1::scheduleInitialize\t\t\t");
  Task* t = scinew Task("DDT1::initialize", this, &DDT1::initialize);
  const MaterialSubset* react_matl = matl0->thisMaterial();
  t->computes(reactedFractionLabel, react_matl);
  t->computes(burningLabel,         react_matl);
  t->computes(TsLabel,              react_matl);
  t->computes(BurningCellLabel,     react_matl);
  t->computes(surfaceTempLabel,     react_matl);
  sched->addTask(t, level->eachPatch(), mymatls);
}

//______________________________________________________________________
//
void DDT1::initialize(const ProcessorGroup*,
                             const PatchSubset* patches,
                             const MaterialSubset* /*matls*/,
                             DataWarehouse*,
                             DataWarehouse* new_dw){
  int m0 = matl0->getDWIndex();
  for(int p=0;p<patches->size();p++) {
    const Patch* patch = patches->get(p);
    cout_doing << "Doing Initialize on patch " << patch->getID()<< "\t\t\t DDT1" << endl;
    
    // This section is needed for outputting F, burn, Ts on initial timestep
    CCVariable<double> F, burn, Ts, surfTemp;
    new_dw->allocateAndPut(F,    reactedFractionLabel, m0, patch);
    new_dw->allocateAndPut(burn, BurningCellLabel,     m0, patch);
    new_dw->allocateAndPut(Ts,   TsLabel,              m0, patch);
    Ts.initialize(0.0);
    F.initialize(0.0);
    burn.initialize(0.0);
  }
}

//______________________________________________________________________
//      
void DDT1::scheduleComputeStableTimestep(SchedulerP&,
                                          const LevelP&,
                                          const ModelInfo*)
{
  // None necessary...
}

//______________________________________________________________________
//     
void DDT1::scheduleComputeModelSources(SchedulerP& sched,
                                       const LevelP& level,
                                       const ModelInfo* mi)
{
    Task* t = scinew Task("DDT1::computeModelSources", this, 
                          &DDT1::computeModelSources, mi);
    cout_doing << "DDT1::scheduleComputeModelSources "<<  endl;  
    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gn  = Ghost::None;
    const MaterialSubset* react_matl = matl0->thisMaterial();
    const MaterialSubset* prod_matl  = matl1->thisMaterial();
    MaterialSubset* one_matl     = scinew MaterialSubset();
    one_matl->add(0);
    one_matl->addReference();
  
    const MaterialSet* all_matls = d_sharedState->allMaterials();
    const MaterialSubset* all_matls_sub = all_matls->getUnion();
    Task::DomainSpec oms = Task::OutOfDomain;

    // Particle Per Cell Task
    Task* t1 = scinew Task("DDT1::computeNumPPC", this, 
                           &DDT1::computeNumPPC, mi);
      
    printSchedule(level,"DDT1::scheduleComputeNumPPC\t\t\t");  
      
    t1->requires(Task::OldDW, Mlb->pXLabel,          react_matl, gn);
    t1->computes(numPPCLabel, react_matl);
      
    sched->addTask(t1, level->eachPatch(), mymatls);
      
    //__________________________________
    // Requires
    //__________________________________
    t->requires( Task::OldDW, mi->delT_Label,       level.get_rep());
    t->requires(Task::OldDW, Ilb->temp_CCLabel,     all_matls_sub, oms, gac,1);
    t->requires(Task::NewDW, Ilb->vol_frac_CCLabel, all_matls_sub, oms, gac,1);
    

    //__________________________________
    // Products
    t->requires(Task::NewDW,  Ilb->rho_CCLabel,     prod_matl, gn);
    t->requires(Task::OldDW,  Ilb->temp_CCLabel,    prod_matl, gac,1);       
    t->requires(Task::NewDW,  Ilb->vol_frac_CCLabel,prod_matl, gn);       
    t->requires(Task::NewDW,  Ilb->TempX_FCLabel,   prod_matl, gac,2);    
    t->requires(Task::NewDW,  Ilb->TempY_FCLabel,   prod_matl, gac,2);    
    t->requires(Task::NewDW,  Ilb->TempZ_FCLabel,   prod_matl, gac,2);
    t->requires(Task::NewDW,  Ilb->press_equil_CCLabel, one_matl,  gac,1);
    t->requires(Task::OldDW,  MIlb->NC_CCweightLabel,   one_matl,  gac, 1);
        
  
    //__________________________________
    // Reactants
    t->requires(Task::NewDW, Ilb->sp_vol_CCLabel,   react_matl, gn);
    t->requires(Task::OldDW, MIlb->vel_CCLabel,     react_matl, gn);
    t->requires(Task::OldDW, MIlb->temp_CCLabel,    react_matl, gac,1);
    t->requires(Task::NewDW, MIlb->temp_CCLabel,    react_matl, gac,1);
    t->requires(Task::NewDW, Ilb->rho_CCLabel,      react_matl, gn);
    t->requires(Task::OldDW, Mlb->pXLabel,          react_matl, gn);
    t->requires(Task::NewDW, MIlb->cMassLabel,      react_matl, gn);
    t->requires(Task::NewDW, Mlb->gMassLabel,       react_matl, gac,1); 
    t->requires(Task::OldDW, reactedFractionLabel,  react_matl, gn); 
    t->requires(Task::NewDW, numPPCLabel,           react_matl, gac,1);
    
    //__________________________________
    // Computes
    //__________________________________
    t->computes(reactedFractionLabel, react_matl);
    t->computes(delFLabel,            react_matl);
    t->computes(burningLabel,         react_matl);
    t->computes(BurningCellLabel,         react_matl);
    t->computes(detonatingLabel,      react_matl);
    t->computes(onSurfaceLabel,    react_matl);
    t->computes(surfaceTempLabel,  react_matl);
    t->computes(TsLabel,          react_matl);

    //__________________________________
    // Conserved Variables
    //__________________________________
    if(d_saveConservedVars->mass ){
        t->computes(DDT1::totalMassBurnedLabel);
    }
    if(d_saveConservedVars->energy){
        t->computes(DDT1::totalHeatReleasedLabel);
    }
    
    //__________________________________
    // Modifies  
    //__________________________________
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

void DDT1::computeNumPPC(const ProcessorGroup*, 
                                const PatchSubset* patches,
                                const MaterialSubset* /*matls*/,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw,
                                const ModelInfo* mi)
{
    int m0 = matl0->getDWIndex(); /* reactant material */
    
    /* Patch Iteration */
    for(int p=0;p<patches->size();p++){
        const Patch* patch = patches->get(p);  
        printTask(patches,patch,"Doing computeNumPPC\t\t\t\t");
        
        /* Indicating how many particles a cell contains */
        ParticleSubset* pset = old_dw->getParticleSubset(m0, patch);
        
        constParticleVariable<Point>  px;
        old_dw->get(px, Mlb->pXLabel, pset);
        
        /* Indicating cells containing how many particles */
        CCVariable<double> pFlag;
        new_dw->allocateAndPut(pFlag,       numPPCLabel,      m0, patch);
        pFlag.initialize(0.0);
        
        /* count how many reactant particles in each cell */
        for(ParticleSubset::iterator iter=pset->begin(), iter_end=pset->end();
            iter != iter_end; iter++){
            particleIndex idx = *iter;
            IntVector c;
            patch->findCell(px[idx],c);
            pFlag[c] += 1.0;
        }    
        setBC(pFlag, "zeroNeumann", patch, d_sharedState, m0, new_dw);
    }
    
}

void DDT1::computeModelSources(const ProcessorGroup*, 
                                const PatchSubset* patches,
                                const MaterialSubset*,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw,
                                const ModelInfo* mi)
{
  delt_vartype delT;
  const Level* level = getLevel(patches);
  old_dw->get(delT, mi->delT_Label, level);

 
  int m0 = matl0->getDWIndex();
  int m1 = matl1->getDWIndex();
  double totalBurnedMass = 0;
  double totalHeatReleased = 0;
 
  double massConv = 0;


  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);  
    
    cout_doing << "Doing computeModelSources on patch "<< patch->getID()
               <<"\t\t\t\t  DDT1" << endl;

    // Variable to modify
    CCVariable<double> mass_src_0, mass_src_1, mass_0;
    CCVariable<Vector> momentum_src_0, momentum_src_1;
    CCVariable<double> energy_src_0, energy_src_1;
    CCVariable<double> sp_vol_src_0, sp_vol_src_1;
    CCVariable<double> onSurface, surfaceTemp;

    new_dw->getModifiable(mass_src_0,    mi->modelMass_srcLabel,  m0,patch);
    new_dw->getModifiable(momentum_src_0,mi->modelMom_srcLabel,   m0,patch);
    new_dw->getModifiable(energy_src_0,  mi->modelEng_srcLabel,   m0,patch);
    new_dw->getModifiable(sp_vol_src_0,  mi->modelVol_srcLabel,   m0,patch);

    new_dw->getModifiable(mass_src_1,    mi->modelMass_srcLabel,  m1,patch);
    new_dw->getModifiable(momentum_src_1,mi->modelMom_srcLabel,   m1,patch);
    new_dw->getModifiable(energy_src_1,  mi->modelEng_srcLabel,   m1,patch);
    new_dw->getModifiable(sp_vol_src_1,  mi->modelVol_srcLabel,   m1,patch);

    // New Variables to store for this timestep
    constCCVariable<double> press_CC, cv_reactant, gasTemp,gasVol_frac,solidTemp,solidMass;
    constCCVariable<double> rctTemp,rctRho,rctSpvol,prodRho, rctFr;
    constCCVariable<Vector> rctvel_CC;
    constCCVariable<double> pFlag;
    CCVariable<double> Fr;
    CCVariable<double> delF;
    CCVariable<double> burning, detonating;
    CCVariable<double> BurningCell, surfTemp;
    constNCVariable<double> NC_CCweight,NCsolidMass;
    constSFCXVariable<double> gasTempX_FC;
    constSFCYVariable<double> gasTempY_FC;
    constSFCZVariable<double> gasTempZ_FC;
	    
    Vector dx = patch->dCell();
    double cell_vol = dx.x()*dx.y()*dx.z();
    Ghost::GhostType  gn  = Ghost::None;
    Ghost::GhostType  gac = Ghost::AroundCells;
   
    //__________________________________
    // Reactant data
    old_dw->get(rctTemp,       MIlb->temp_CCLabel,  m0,patch,gn, 0);
    old_dw->get(rctvel_CC,     MIlb->vel_CCLabel,   m0,patch,gn, 0);
    new_dw->get(rctRho,        Ilb->rho_CCLabel,    m0,patch,gn, 0);
    new_dw->get(rctSpvol,      Ilb->sp_vol_CCLabel, m0,patch,gn, 0);
    new_dw->get(solidTemp,     MIlb->temp_CCLabel,  m0,patch,gac, 1);
    new_dw->get(solidMass,     MIlb->cMassLabel,    m0,patch,gn, 0);
    new_dw->get(NCsolidMass,   Mlb->gMassLabel,     m0,patch,gac,1);
    new_dw->get(pFlag,         numPPCLabel,         m0,patch,gac,1);
    //new_dw->get(cv_reactant,   Ilb->specific_heatLabel,m0,patch,gn, 0);
    old_dw->get(rctFr,         reactedFractionLabel,  m0,patch,gac,0);
    new_dw->allocateAndPut(Fr,   reactedFractionLabel,m0,patch);
    new_dw->allocateAndPut(delF, delFLabel,           m0,patch);
    Fr.initialize(0.);
    delF.initialize(0.);
    //__________________________________
    // Product Data, 
    new_dw->get(prodRho,         Ilb->rho_CCLabel,   m1,patch,gn, 0);
    new_dw->get(gasTempX_FC,      Ilb->TempX_FCLabel,m1,patch,gac,2);
    new_dw->get(gasTempY_FC,      Ilb->TempY_FCLabel,m1,patch,gac,2);
    new_dw->get(gasTempZ_FC,      Ilb->TempZ_FCLabel,m1,patch,gac,2);
    old_dw->get(gasTemp,          Ilb->temp_CCLabel, m1,patch,gn, 0);
    new_dw->get(gasVol_frac,      Ilb->vol_frac_CCLabel,  m1, patch,gn, 0);
    //__________________________________
    //   Misc.
    new_dw->get(press_CC,         Ilb->press_equil_CCLabel,0,  patch,gac, 1);
    old_dw->get(NC_CCweight,      MIlb->NC_CCweightLabel,  0,   patch,gac,1);   
   
    new_dw->allocateAndPut(burning,    burningLabel,           m0, patch,gn,0);  
    new_dw->allocateAndPut(detonating, detonatingLabel,        m0, patch);
    new_dw->allocateAndPut(BurningCell, BurningCellLabel, m0, patch, gn, 0);
    new_dw->allocateAndPut(surfTemp,    TsLabel,          m0, patch, gn, 0);
    burning.initialize(0.);
    detonating.initialize(0.);
    BurningCell.initialize(0.0);
    surfTemp.initialize(0.0);

    new_dw->allocateAndPut(onSurface,  onSurfaceLabel,   m0, patch);
    new_dw->allocateAndPut(surfaceTemp,surfaceTempLabel, m0, patch);
      
    IntVector nodeIdx[8];
      
     
    // Get Temperatures for burning check
    int numAllMatls = d_sharedState->getNumMatls();
    StaticArray<constCCVariable<double> > vol_frac_CC(numAllMatls);
    StaticArray<constCCVariable<double> > temp_CC(numAllMatls);
    for(int m = 0; m < numAllMatls; m++) {
      Material* matl = d_sharedState->getMaterial(m);
      int indx = matl->getDWIndex();
      old_dw->get(temp_CC[m],     MIlb->temp_CCLabel,    indx, patch, gac, 1);
      new_dw->get(vol_frac_CC[m], Ilb->vol_frac_CCLabel, indx, patch, gac, 1);
    }

    for (CellIterator iter = patch->getCellIterator();!iter.done();iter++){
      IntVector c = *iter;
      // Copy old Fr in case pressure threshold is not met
      Fr[c]=rctFr[c];
      // JWL++ Model For explosions
      if (press_CC[c] > d_threshold_pressure){
        // Flag for detonating
        detonating[c] = 1;
        double burnedMass;

        double F = prodRho[c]/(rctRho[c]+prodRho[c]);
        if(F >= 0. && F < .99){
          delF[c] = d_G*pow(press_CC[c],d_b)*(1.-F);
        }
        delF[c]*=delT;
        Fr[c] = F;
        double rctMass = rctRho[c]*cell_vol;
        double prdMass = prodRho[c]*cell_vol;
        burnedMass = min(delF[c]*(prdMass+rctMass), rctMass);
        burnedMass = min(burnedMass, .2*d_rho0*cell_vol);

        //__________________________________
        // conservation of mass, momentum and energy                           
        mass_src_0[c] -= burnedMass;
        mass_src_1[c] += burnedMass;
        massConv += burnedMass;           

        Vector momX        = rctvel_CC[c] * burnedMass;
        momentum_src_0[c] -= momX;
        momentum_src_1[c] += momX;
      
        double energyX   = dynamic_cast<const MPMMaterial*>(matl0)->getSpecificHeat()*rctTemp[c]*burnedMass;//*cv_reactant[c]; 
        double releasedHeat = burnedMass * d_E0;
        energy_src_0[c] -= energyX;
        energy_src_1[c] += energyX + releasedHeat;

        double createdVolx  = burnedMass * rctSpvol[c];
        sp_vol_src_0[c] -= createdVolx;
        sp_vol_src_1[c] += createdVolx;
      } else if(press_CC[c] < d_threshold_pressure && press_CC[c] > ThresholdPressure) {
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
            /* test whether the current cell satisfies burning criteria */
            bool   burning = 0;
            double maxProductVolFrac  = -1.0;
            double maxReactantVolFrac = -1.0;
            double productPress = 0.0;
            double Tzero = 0.0;
            double temp_vf = 0.0;      
            /*if( (MaxMass-MinMass)/MaxMass>0.4 && (MaxMass-MinMass)/MaxMass<1.0 && pFlag[c]>0 ){ */
            if( MinMass/MaxMass<0.7 && pFlag[c]>0 ){ 
                /* near interface and containing particles */
                for(int i = -1; i<=1; i++){
                    for(int j = -1; j<=1; j++){
                        for(int k = -1; k<=1; k++){
                            IntVector cell = c + IntVector(i,j,k);
                            /* Search for Tzero from max_vol_frac reactant cell */
                            temp_vf = vol_frac_CC[m0][cell]; 
                            if( temp_vf > maxReactantVolFrac ){
                                maxReactantVolFrac = temp_vf;
                                Tzero = solidTemp[cell];
                            }//endif
                            
                            /* Search for pressure from max_vol_frac product cell */
                            temp_vf = vol_frac_CC[m1][cell]; 
                            if( temp_vf > maxProductVolFrac ){
                                maxProductVolFrac = temp_vf;
                                productPress = press_CC[cell];
                            }//endif
                            
                            if(burning == 0 && pFlag[cell] <= BP){
                                for (int m = 0; m < numAllMatls; m++){
                                    if(vol_frac_CC[m][cell] > 0.2 && temp_CC[m][cell] > ignitionTemp){
                                        burning = 1;
                                        break;
                                    }
                                }
                            }//endif
                            
                        }//end 3rd for
                    }//end 2nd for
                }//end 1st for
            }//endif
            if(burning == 1 && productPress >= ThresholdPressure){
                BurningCell[c]=1.0;
                
                Vector rhoGradVector = computeDensityGradientVector(nodeIdx,
                                                                    NCsolidMass, NC_CCweight,dx);
                
                double surfArea = computeSurfaceArea(rhoGradVector, dx); 
                double Tsurf = 850.0;  // initial guess for the surface temperature.
                
                double burnedMass = computeBurnedMass(Tzero, Tsurf, productPress,
                                                      rctSpvol[c], surfArea, delT,
                                                      solidMass[c]);
                
                surfTemp[c] = Tsurf;
                
                /* conservation of mass, momentum and energy   */
                mass_src_0[c]   -= burnedMass;
                mass_src_1[c]   += burnedMass;
                totalBurnedMass += burnedMass;
                
                Vector momX = rctvel_CC[c] * burnedMass;
                momentum_src_0[c]  -= momX;
                momentum_src_1[c]  += momX;
                
                double energyX   = Cp*solidTemp[c]*burnedMass; 
                double releasedHeat = burnedMass * (Qc + Qg);
                energy_src_0[c]   -= energyX;
                energy_src_1[c]   += energyX + releasedHeat;
                totalHeatReleased += releasedHeat;
                
                double createdVolx = burnedMass * rctSpvol[c];
                sp_vol_src_0[c]  -= createdVolx;
                sp_vol_src_1[c]  += createdVolx;
	    }
        }  // if (maxMass-MinMass....)
      }// if (pressure)
    }  // cell iterator  
    //__________________________________
    //  set symetric BC
    setBC(mass_src_0, "set_if_sym_BC",patch, d_sharedState, m0, new_dw);
    setBC(mass_src_1, "set_if_sym_BC",patch, d_sharedState, m1, new_dw);
    setBC(delF,       "set_if_sym_BC",patch, d_sharedState, m0, new_dw);
    setBC(Fr,         "set_if_sym_BC",patch, d_sharedState, m0, new_dw);
  }
  //__________________________________
  //save total quantities
  if(d_saveConservedVars->mass ){
      new_dw->put(sum_vartype(totalBurnedMass),  DDT1::totalMassBurnedLabel);
  }
  if(d_saveConservedVars->energy){
      new_dw->put(sum_vartype(totalHeatReleased),DDT1::totalHeatReleasedLabel);
  }
}
    
    
//______________________________________________________________________
double DDT1::computeSurfaceArea(Vector &rhoGradVector, Vector &dx){
    double delX = dx.x();
    double delY = dx.y();
    double delZ = dx.z();
    double rgvX = fabs(rhoGradVector.x());
    double rgvY = fabs(rhoGradVector.y());
    double rgvZ = fabs(rhoGradVector.z());
        
    double max = rgvX;
    if(rgvY > max)   max = rgvY;
    if(rgvZ > max)   max = rgvZ;
        
    double coeff = pow(1.0/max, 1.0/3.0);
        
    double TmpX = delX*rgvX;
    double TmpY = delY*rgvY;
    double TmpZ = delZ*rgvZ;
        
    return delX*delY*delZ / (TmpX+TmpY+TmpZ) * coeff; 
}

//______________________________________________________________________    
//
Vector DDT1::computeDensityGradientVector(IntVector *nodeIdx, 
                                                constNCVariable<double> &NCsolidMass, 
                                                constNCVariable<double> &NC_CCweight, 
                                                Vector &dx){
    double gradRhoX = 0.25 * (
                                (NCsolidMass[nodeIdx[0]]*NC_CCweight[nodeIdx[0]]+
                                NCsolidMass[nodeIdx[1]]*NC_CCweight[nodeIdx[1]]+
                                NCsolidMass[nodeIdx[2]]*NC_CCweight[nodeIdx[2]]+
                                NCsolidMass[nodeIdx[3]]*NC_CCweight[nodeIdx[3]])
                                -
                                (NCsolidMass[nodeIdx[4]]*NC_CCweight[nodeIdx[4]]+
                                NCsolidMass[nodeIdx[5]]*NC_CCweight[nodeIdx[5]]+
                                NCsolidMass[nodeIdx[6]]*NC_CCweight[nodeIdx[6]]+
                                NCsolidMass[nodeIdx[7]]*NC_CCweight[nodeIdx[7]])
                                )/dx.x();
        
    double gradRhoY = 0.25 * (
                                (NCsolidMass[nodeIdx[0]]*NC_CCweight[nodeIdx[0]]+
                                NCsolidMass[nodeIdx[1]]*NC_CCweight[nodeIdx[1]]+
                                NCsolidMass[nodeIdx[4]]*NC_CCweight[nodeIdx[4]]+
                                NCsolidMass[nodeIdx[5]]*NC_CCweight[nodeIdx[5]])
                                -
                                (NCsolidMass[nodeIdx[2]]*NC_CCweight[nodeIdx[2]]+
                                NCsolidMass[nodeIdx[3]]*NC_CCweight[nodeIdx[3]]+
                                NCsolidMass[nodeIdx[6]]*NC_CCweight[nodeIdx[6]]+
                                NCsolidMass[nodeIdx[7]]*NC_CCweight[nodeIdx[7]])
                                )/dx.y();
        
    double gradRhoZ = 0.25 * (
                                (NCsolidMass[nodeIdx[1]]*NC_CCweight[nodeIdx[1]]+
                                NCsolidMass[nodeIdx[3]]*NC_CCweight[nodeIdx[3]]+
                                NCsolidMass[nodeIdx[5]]*NC_CCweight[nodeIdx[5]]+
                                NCsolidMass[nodeIdx[7]]*NC_CCweight[nodeIdx[7]])
                                -
                                (NCsolidMass[nodeIdx[0]]*NC_CCweight[nodeIdx[0]]+
                                NCsolidMass[nodeIdx[2]]*NC_CCweight[nodeIdx[2]]+
                                NCsolidMass[nodeIdx[4]]*NC_CCweight[nodeIdx[4]]+
                                NCsolidMass[nodeIdx[6]]*NC_CCweight[nodeIdx[6]])
                                )/dx.z();
        
    double absGradRho = sqrt(gradRhoX*gradRhoX + gradRhoY*gradRhoY + gradRhoZ*gradRhoZ );
        
    return Vector(gradRhoX/absGradRho, gradRhoY/absGradRho, gradRhoZ/absGradRho);
}
    
    
    
//______________________________________________________________________
//
void DDT1::scheduleModifyThermoTransportProperties(SchedulerP&,
                                                    const LevelP&,
                                                    const MaterialSet*)
{
  // do nothing      
}
void DDT1::computeSpecificHeat(CCVariable<double>&,
                                const Patch*,   
                                DataWarehouse*, 
                                const int)      
{
  //do nothing
}
//______________________________________________________________________
//
void DDT1::scheduleErrorEstimate(const LevelP&,
                                  SchedulerP&)
{
  // Not implemented yet
}
//__________________________________
void DDT1::scheduleTestConservation(SchedulerP&,
                                     const PatchSet*,                      
                                     const ModelInfo*)                     
{
  // Not implemented yet
}

//______________________________________________________________________
//
void DDT1::printSchedule(const LevelP& level,
                                const string& where){
  if (cout_doing.active()){
    cout_doing << d_myworld->myrank() << " "
               << where << "L-"
               << level->getIndex()<< endl;
  }
}
    
//______________________________________________________________________
void DDT1::printTask(const PatchSubset* patches,
                            const Patch* patch,
                            const string& where){
    if (cout_doing.active()){
        cout_doing << d_myworld->myrank() << " " 
        << where << " DDT1 L-"
        << getLevel(patches)->getIndex()
        << " patch " << patch->getGridIndex()<< endl;
    }  
}

/****************************************************************************/
/******************* Bisection Newton Solver ********************************/
/****************************************************************************/
double DDT1::computeBurnedMass(double To, double& Ts, double P, double Vc, double surfArea, 
                                    double delT, double solidMass){  
    UpdateConstants(To, P, Vc);
    Ts = BisectionNewton(Ts);
    double m =  m_Ts(Ts);
    double burnedMass = delT * surfArea * m;
    if (burnedMass + MIN_MASS_IN_A_CELL > solidMass) 
        burnedMass = solidMass - MIN_MASS_IN_A_CELL;  
    return burnedMass;
}
    
//______________________________________________________________________
void DDT1::UpdateConstants(double To, double P, double Vc){
    /* CC1 = Ac*R*Kc/Ec/Cp        */
    /* CC2 = Qc/Cp/2              */
    /* CC3 = 4*Kg*Bg*W*W/Cp/R/R;  */
    /* CC4 = Qc/Cp                */
    /* CC5 = Qg/Cp                */
    /* Vc = Condensed Phase Specific Volume */
    
    C1 = CC1 / Vc; 
    C2 = To + CC2; 
    C3 = CC3 * P*P;
    C4 = To + CC4; 
    C5 = CC5 * C3; 
    
    Tmin = C4;
    double Tsmax = Ts_max();
    if (Tmin < Tsmax)
        Tmax =  F_Ts(Tsmax);
    else
        Tmax = F_Ts(Tmin);
    
    IL = Tmin;
    IR = Tmax;
}
    
    
/***   
 ***   Ts = F_Ts(Ts) = Ts_m(m_Ts(Ts))                                              
 ***   f_Ts(Ts) = C4 + C5/(sqrt(m^2+C3) + m)^2 
 ***
 ***   Solve for diff(f_Ts(Ts))=0 
 ***   Ts_max = C2 - Ec/2R + sqrt(4*R^2*C2^2+Ec^2)/2R
 ***   f_Ts_max = f_Ts(Ts_max)
 ***/
double DDT1::F_Ts(double Ts){
    return Ts_m(m_Ts(Ts));
}
    
double DDT1::m_Ts(double Ts){
    return sqrt( C1*Ts*Ts/(Ts-C2)*exp(-Ec/R/Ts) );
}
    
double DDT1::Ts_m(double m){
    double deno = sqrt(m*m+C3)+m;
    return C4 + C5/(deno*deno);
}
    
/* the function value for the zero finding problem */
double DDT1::Func(double Ts){
    return Ts - F_Ts(Ts);
}
    
/* dFunc/dTs */
double DDT1::Deri(double Ts){
    double m = m_Ts(Ts);
    double K1 = Ts-C2;
    double K2 = sqrt(m*m+C3);
    double K3 = (R*Ts*(K1-C2)+Ec*K1)*m*C5;
    double K4 = (K2+m)*(K2+m)*K1*K2*R*Ts*Ts;
    return 1.0 + K3/K4;
}
    
/* F_Ts(Ts_max) is the max of F_Ts function */
double DDT1::Ts_max(){
    return 0.5*(2.0*R*C2 - Ec + sqrt(4.0*R*R*C2*C2+Ec*Ec))/R;
} 
    
void DDT1::SetInterval(double f, double Ts){  
    /* IL <= 0,  IR >= 0 */
    if(f < 0)  
        IL = Ts;
    else if(f > 0)
        IR = Ts;
    else if(f ==0){
        IL = Ts;
        IR = Ts; 
    }
}
    
/* Bisection - Newton Method */
double DDT1::BisectionNewton(double Ts){  
    double y = 0;
    double df_dTs = 0;
    double delta_old = 0;
    double delta_new = 0;
    
    int iter = 0;
    if(Ts>Tmax || Ts<Tmin)
        Ts = (Tmin+Tmax)/2;
    
    while(1){
        iter++;
        y = Func(Ts);
        SetInterval(y, Ts);
        
        if(fabs(y)<EPSILON)
            return Ts;
        
        delta_new = 1e100;
        while(1){
            if(iter>100){
                cout<<"Not converging after 100 iterations in DDT1.cc."<<endl;
                exit(1);
            }
                
            df_dTs = Deri(Ts);
            if(df_dTs==0) 
                break;
            
            delta_old = delta_new;
            delta_new = -y/df_dTs; //Newton Step
            Ts += delta_new;
            y = Func(Ts);
            
            if(fabs(y)<EPSILON)
                return Ts;
            
            if(Ts<IL || Ts>IR || fabs(delta_new)>fabs(delta_old*0.7))
                break;
            
            iter++; 
            SetInterval(y, Ts);  
        }
        
        Ts = (IL+IR)/2.0; //Bisection Step
    }
}
