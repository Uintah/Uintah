
#include <Packages/Uintah/CCA/Components/ICE/AMRICE.h>
#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <Packages/Uintah/CCA/Components/Regridder/PerPatchVars.h>
#include <Packages/Uintah/CCA/Ports/SolverInterface.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/PerPatch.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Math/FastMatrix.h>
#include <Core/Math/MiscMath.h>
#include <Core/Util/DebugStream.h>
#include <iomanip>

#define SPEW 0
//#undef SPEW

//#define BRYAN

using namespace Uintah;
using namespace std;
static DebugStream cout_doing("AMRICE_DOING_COUT", false);

//setenv SCI_DEBUG AMR:+ you can see the new grid it creates


AMRICE::AMRICE(const ProcessorGroup* myworld)
  : ICE(myworld, true)
{
}

AMRICE::~AMRICE()
{
}
//___________________________________________________________________
void AMRICE::problemSetup(const ProblemSpecP& params, 
                          const ProblemSpecP& materials_ps, 
                          GridP& grid, SimulationStateP& sharedState)
{
  cout_doing << d_myworld->myrank() 
             << " Doing problemSetup  \t\t\t AMRICE" << '\n';
             
  ICE::problemSetup(params, materials_ps,grid, sharedState);
  ProblemSpecP cfd_ps = params->findBlock("CFD");
  ProblemSpecP ice_ps = cfd_ps->findBlock("ICE");
  ProblemSpecP amr_ps = ice_ps->findBlock("AMR");
  if(!amr_ps){
    string warn;
    warn ="\n INPUT FILE ERROR:\n <AMR>  block not found inside of <ICE> block \n";
    throw ProblemSetupException(warn, __FILE__, __LINE__);
    
  }
  ProblemSpecP refine_ps = amr_ps->findBlock("Refinement_Criteria_Thresholds");
  if(!refine_ps ){
    string warn;
    warn ="\n INPUT FILE ERROR:\n <Refinement_Criteria_Thresholds> "
         " block not found inside of <ICE> block \n";
    throw ProblemSetupException(warn, __FILE__, __LINE__);
  }
  amr_ps->require( "orderOfInterpolation", d_orderOfInterpolation);
  amr_ps->getWithDefault( "regridderTest", d_regridderTest,     false);
  amr_ps->getWithDefault( "do_Refluxing",  d_doRefluxing,       true);
  amr_ps->getWithDefault( "useLockStep",   d_useLockStep,       false);
  refine_ps->getWithDefault("Density",     d_rho_threshold,     1e100);
  refine_ps->getWithDefault("Temperature", d_temp_threshold,    1e100);
  refine_ps->getWithDefault("Pressure",    d_press_threshold,   1e100);
  refine_ps->getWithDefault("VolumeFrac",  d_vol_frac_threshold,1e100);
  refine_ps->getWithDefault("Velocity",    d_vel_threshold,     1e100);
  
  //__________________________________
  // bullet proofing
  int maxLevel = grid->numLevels();
  
  for (int i=0; i< maxLevel; i++){
     double trr = grid->getLevel(i)->timeRefinementRatio();

    if( d_useLockStep && trr != 1){
      string warn;
      warn ="\n INPUT FILE ERROR:\n To use the lockstep algorithm you must specify \n<Grid> \n  <time_refinement_ratio> 1 </time_refinement_ratio> \n</Grid>";
      throw ProblemSetupException(warn, __FILE__, __LINE__);
    }
  }
  
}
//___________________________________________________________________              
void AMRICE::scheduleInitialize(const LevelP& level,
                                  SchedulerP& sched)
{
  cout_doing << d_myworld->myrank() 
             << " AMRICE::scheduleInitialize \t\tL-"<<level->getIndex()<< '\n';
  ICE::scheduleInitialize(level, sched);
}
//___________________________________________________________________
void AMRICE::initialize(const ProcessorGroup*,
                           const PatchSubset*, const MaterialSubset*,
                           DataWarehouse*, DataWarehouse*)
{
}
/*___________________________________________________________________
 Function~  AMRICE::addRefineDependencies--
 Purpose:  
_____________________________________________________________________*/
void AMRICE::addRefineDependencies(Task* task, 
                                   const VarLabel* var,
                                   Task::DomainSpec DS,
                                   const MaterialSubset* matls,
                                   int step, 
                                   int nsteps)
{
  cout_dbg << d_myworld->myrank() << " \t addRefineDependencies (" << var->getName()
           << ")"<< endl;
  ASSERTRANGE(step, 0, nsteps+1);
  
  Ghost::GhostType  gac = Ghost::AroundCells;
  if(step != nsteps) {
    cout_dbg << " requires from CoarseOldDW ";
    task->requires(Task::CoarseOldDW, var, 0, Task::CoarseLevel, matls, DS, gac, 1);
  }
  if(step != 0) {
    cout_dbg << " requires from CoarseNewDW ";
    task->requires(Task::CoarseNewDW, var, 0, Task::CoarseLevel, matls, DS, gac, 1);
  }

  task->modifies(var, matls, DS);

  cout_dbg <<""<<endl;
}
/*___________________________________________________________________
 Function~  AMRICE::scheduleRefineInterface--
 Purpose:  
_____________________________________________________________________*/
void AMRICE::scheduleRefineInterface(const LevelP& fineLevel,
                                     SchedulerP& sched,
                                     int step, 
                                     int nsteps)
{
  if(fineLevel->getIndex() > 0  && doICEOnLevel(fineLevel->getIndex(), fineLevel->getGrid()->numLevels())){
    cout_doing << d_myworld->myrank() << " AMRICE::scheduleRefineInterface \t\t\tL-" 
               << fineLevel->getIndex() << " progressVar "<< (double)step/(double)nsteps <<'\n';
               
    double subCycleProgress = double(step)/double(nsteps);
    
    ostringstream str;
    str << "AMRICE::refineCoarseFineInterface" << " " << step << " " << nsteps;
    Task* task = scinew Task(str.str().c_str(), 
                       this, &AMRICE::refineCoarseFineInterface, 
                       subCycleProgress);
  
  
    Task::DomainSpec ND   = Task::NormalDomain;
    Task::DomainSpec oims = Task::OutOfDomain;  //outside of ice matlSet.
    const MaterialSet* all_matls = d_sharedState->allMaterials();
    const MaterialSubset* all_matls_sub = all_matls->getUnion();
    
    addRefineDependencies(task, lb->press_CCLabel, oims,d_press_matl, step, nsteps);
    addRefineDependencies(task, lb->rho_CCLabel,   ND,  all_matls_sub,step, nsteps);
    addRefineDependencies(task, lb->sp_vol_CCLabel,ND,  all_matls_sub,step, nsteps);
    addRefineDependencies(task, lb->temp_CCLabel,  ND,  all_matls_sub,step, nsteps);
    addRefineDependencies(task, lb->vel_CCLabel,   ND,  all_matls_sub,step, nsteps);
    
    //__________________________________
    // Model Variables.
    if(d_modelSetup && d_modelSetup->tvars.size() > 0){
      vector<TransportedVariable*>::iterator iter;

      for(iter = d_modelSetup->tvars.begin();
         iter != d_modelSetup->tvars.end(); iter++){
        TransportedVariable* tvar = *iter;
        addRefineDependencies(task, tvar->var,ND,  all_matls_sub, step, nsteps);
      }
    }
    sched->addTask(task, fineLevel->eachPatch(), d_sharedState->allICEMaterials());
  }
}
/*______________________________________________________________________
 Function~  AMRICE::refineCoarseFineInterface
 Purpose~   
______________________________________________________________________*/
void AMRICE::refineCoarseFineInterface(const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset*,
                                       DataWarehouse* fine_old_dw,
                                       DataWarehouse* fine_new_dw,
                                       const double subCycleProgress)
{
  const Level* level = getLevel(patches);
  if(level->getIndex() > 0){     
    cout_doing << d_myworld->myrank() 
               << " Doing refineCoarseFineInterface"<< "\t\t\t AMRICE L-" 
               << level->getIndex() << " Patches: " << *patches << " progressVar " << subCycleProgress<<endl;
    int  numMatls = d_sharedState->getNumICEMatls();
    bool dbg_onOff = cout_dbg.active();      // is cout_dbg switch on or off
      
    for(int p=0;p<patches->size();p++){
      const Patch* patch = patches->get(p);
      
      for (int m = 0; m < numMatls; m++) {
        ICEMaterial* matl = d_sharedState->getICEMaterial(m);
        int indx = matl->getDWIndex();    
        CCVariable<double> press_CC, rho_CC, sp_vol_CC, temp_CC;
        CCVariable<Vector> vel_CC;


        fine_new_dw->getModifiable(press_CC, lb->press_CCLabel,  0,   patch); 
        fine_new_dw->getModifiable(rho_CC,   lb->rho_CCLabel,    indx,patch);
        fine_new_dw->getModifiable(sp_vol_CC,lb->sp_vol_CCLabel, indx,patch);
        fine_new_dw->getModifiable(temp_CC,  lb->temp_CCLabel,   indx,patch);
        fine_new_dw->getModifiable(vel_CC,   lb->vel_CCLabel,    indx,patch);

        //__________________________________
        //  Print Data 
        if(switchDebug_AMR_refineInterface){
          ostringstream desc;     
          desc << "TOP_refineInterface_Mat_" << indx << "_patch_"
               << patch->getID()<< " step " << subCycleProgress;

          printData(indx, patch,   1, desc.str(), "press_CC",    press_CC); 
          printData(indx, patch,   1, desc.str(), "rho_CC",      rho_CC);
          printData(indx, patch,   1, desc.str(), "sp_vol_CC",   sp_vol_CC);
          printData(indx, patch,   1, desc.str(), "Temp_CC",     temp_CC);
          printVector(indx, patch, 1, desc.str(), "vel_CC", 0,   vel_CC);
        }

        refineCoarseFineBoundaries(patch, press_CC, fine_new_dw, 
                                   lb->press_CCLabel,  0,   subCycleProgress);

        refineCoarseFineBoundaries(patch, rho_CC,   fine_new_dw, 
                                   lb->rho_CCLabel,    indx,subCycleProgress);

        refineCoarseFineBoundaries(patch, sp_vol_CC,fine_new_dw,
                                   lb->sp_vol_CCLabel, indx,subCycleProgress);

        refineCoarseFineBoundaries(patch, temp_CC,  fine_new_dw,
                                   lb->temp_CCLabel,   indx,subCycleProgress);

        refineCoarseFineBoundaries(patch, vel_CC,   fine_new_dw,
                                   lb->vel_CCLabel,    indx,subCycleProgress);
       //__________________________________
       //    Model Variables                     
       if(d_modelSetup && d_modelSetup->tvars.size() > 0){
         vector<TransportedVariable*>::iterator t_iter;
          for( t_iter  = d_modelSetup->tvars.begin();
               t_iter != d_modelSetup->tvars.end(); t_iter++){
            TransportedVariable* tvar = *t_iter;

            if(tvar->matls->contains(indx)){
              string Labelname = tvar->var->getName();
              CCVariable<double> q_CC;
              fine_new_dw->getModifiable(q_CC, tvar->var, indx, patch);
              
              if(switchDebug_AMR_refineInterface){ 
                string name = tvar->var->getName();
                printData(indx, patch, 1, "TOP_refineInterface", Labelname, q_CC);
              }              
              
              refineCoarseFineBoundaries(patch, q_CC, fine_new_dw,
                                          tvar->var,    indx,subCycleProgress);
              
              if(switchDebug_AMR_refineInterface){ 
                string name = tvar->var->getName();
                printData(indx, patch, 1, "BOT_refineInterface", Labelname, q_CC);
              }
            }
          }
        }                                     
                            

        //__________________________________
        //  Print Data 
        if(switchDebug_AMR_refineInterface){
          ostringstream desc;    
          desc << "BOT_refineInterface_Mat_" << indx << "_patch_"
               << patch->getID()<< " step " << subCycleProgress;
          printData(indx, patch,   1, desc.str(), "press_CC",  press_CC);
          printData(indx, patch,   1, desc.str(), "rho_CC",    rho_CC);
          printData(indx, patch,   1, desc.str(), "sp_vol_CC", sp_vol_CC);
          printData(indx, patch,   1, desc.str(), "Temp_CC",   temp_CC);
          printVector(indx, patch, 1, desc.str(), "vel_CC", 0, vel_CC);
        }
      }
    }
    cout_dbg.setActive(dbg_onOff);  // reset on/off switch for cout_dbg
  }             
}


/*___________________________________________________________________
 Function~  AMRICE::refineCoarseFineBoundaries--    D O U B L E  
_____________________________________________________________________*/
void AMRICE::refineCoarseFineBoundaries(const Patch* patch,
                                        CCVariable<double>& val,
                                        DataWarehouse* fine_new_dw,
                                        const VarLabel* label,
                                        int matl,
                                        double subCycleProgress_var)
{
  const Level* level = patch->getLevel();
  const Level* coarseLevel = level->getCoarserLevel().get_rep();

  cout_dbg << "\t refineCoarseFineBoundaries ("<<label->getName() << ") \t" 
           << " subCycleProgress_var " << subCycleProgress_var
           << " Level-" << level->getIndex()<< '\n';
  DataWarehouse* coarse_old_dw = 0;
  DataWarehouse* coarse_new_dw = 0;
  
  if (subCycleProgress_var != 1.0){
    coarse_old_dw = fine_new_dw->getOtherDataWarehouse(Task::CoarseOldDW);
  }
  if (subCycleProgress_var != 0.0){
    coarse_new_dw = fine_new_dw->getOtherDataWarehouse(Task::CoarseNewDW);
  }
  
  refine_CF_interfaceOperator<double>
    (patch, level, coarseLevel, val, label, subCycleProgress_var, matl,
     fine_new_dw, coarse_old_dw, coarse_new_dw);
}
/*___________________________________________________________________
 Function~  AMRICE::refineCoarseFineBoundaries--    V E C T O R 
_____________________________________________________________________*/
void AMRICE::refineCoarseFineBoundaries(const Patch* patch,
                                        CCVariable<Vector>& val,
                                        DataWarehouse* fine_new_dw,
                                        const VarLabel* label,
                                        int matl,
                                        double subCycleProgress_var)
{
  const Level* level = patch->getLevel();
  const Level* coarseLevel = level->getCoarserLevel().get_rep();

  cout_dbg << "\t refineCoarseFineBoundaries ("<<label->getName() << ") \t" 
           << " subCycleProgress_var " << subCycleProgress_var
           << " Level-" << level->getIndex()<< '\n';
  DataWarehouse* coarse_old_dw = 0;
  DataWarehouse* coarse_new_dw = 0;
  
  if (subCycleProgress_var != 1.0){
    coarse_old_dw = fine_new_dw->getOtherDataWarehouse(Task::CoarseOldDW);
  }
  if (subCycleProgress_var != 0.0){
    coarse_new_dw = fine_new_dw->getOtherDataWarehouse(Task::CoarseNewDW);
  }

  refine_CF_interfaceOperator<Vector>
    (patch, level, coarseLevel, val, label, subCycleProgress_var, matl,
     fine_new_dw, coarse_old_dw, coarse_new_dw);
}

/*___________________________________________________________________
 Function~  AMRICE::scheduleSetBC_FineLevel--
 Purpose:  
_____________________________________________________________________*/
void AMRICE::scheduleSetBC_FineLevel(const PatchSet* patches,
                                     SchedulerP& sched) 
{
  const Level* fineLevel = getLevel(patches);
  int L_indx = fineLevel->getIndex();
  
  if(L_indx > 0  && doICEOnLevel(L_indx, fineLevel->getGrid()->numLevels())){
    cout_doing << d_myworld->myrank() << " AMRICE::scheduleSetBC_FineLevel \t\t\tL-" 
               << L_indx <<" P-" << *patches << '\n';
    
    Task* t;
    t = scinew Task("AMRICE::setBC_FineLevel", this, &AMRICE::setBC_FineLevel);
    Task::DomainSpec oims = Task::OutOfDomain;  //outside of ice matlSet.
    Ghost::GhostType  gn = Ghost::None;
    Task::DomainSpec ND   = Task::NormalDomain;

    // need to interpolate these intermediate values
    t->requires(Task::NewDW, lb->gammaLabel,        0, Task::CoarseLevel, 0, ND, gn,0);
    t->requires(Task::NewDW, lb->specific_heatLabel,0, Task::CoarseLevel, 0, ND, gn,0);
    t->requires(Task::NewDW, lb->vol_frac_CCLabel,  0, Task::CoarseLevel, 0, ND, gn,0);
    

    t->modifies(lb->press_CCLabel, d_press_matl, oims);     
    t->modifies(lb->rho_CCLabel);
    t->modifies(lb->sp_vol_CCLabel);
    t->modifies(lb->temp_CCLabel);
    t->modifies(lb->vel_CCLabel);
    t->computes(lb->gammaLabel);
    t->computes(lb->specific_heatLabel);
    t->computes(lb->vol_frac_CCLabel);
    
    //__________________________________
    // Model Variables.
    if(d_modelSetup && d_modelSetup->tvars.size() > 0){
      vector<TransportedVariable*>::iterator iter;

      for(iter = d_modelSetup->tvars.begin();
         iter != d_modelSetup->tvars.end(); iter++){
        TransportedVariable* tvar = *iter;
        t->modifies(tvar->var);
      }
    }
    sched->addTask(t, patches, d_sharedState->allMaterials());
  }
}

/*______________________________________________________________________
 Function~  AMRICE::setBC_FineLevel
 Purpose~   set the boundary conditions on the fine level at the edge
 of the computational domain
______________________________________________________________________*/
void AMRICE::setBC_FineLevel(const ProcessorGroup*,
                             const PatchSubset* patches,          
                             const MaterialSubset*,               
                             DataWarehouse* fine_old_dw,          
                             DataWarehouse* fine_new_dw)             
{
  const Level* fineLevel = getLevel(patches);
  const Level* coarseLevel = fineLevel->getCoarserLevel().get_rep();
  
  if(fineLevel->getIndex() > 0){     
    cout_doing << d_myworld->myrank() 
               << " Doing setBC_FineLevel"<< "\t\t\t\t AMRICE L-" 
               << fineLevel->getIndex() << " Patches: " << *patches <<endl;
               
    int  numMatls = d_sharedState->getNumMatls();
    bool dbg_onOff = cout_dbg.active();      // is cout_dbg switch on or off
      
    for(int p=0;p<patches->size();p++){
      const Patch* patch = patches->get(p);
      StaticArray<CCVariable<double> > sp_vol_CC(numMatls);
      StaticArray<constCCVariable<double> > sp_vol_const(numMatls);
      
      
      for (int m = 0; m < numMatls; m++) {
        ICEMaterial* matl = d_sharedState->getICEMaterial(m);
        int indx = matl->getDWIndex(); 
        CCVariable<double> rho_CC, temp_CC, cv, gamma,vol_frac;
        CCVariable<Vector> vel_CC;
        
        fine_new_dw->getModifiable(sp_vol_CC[m],lb->sp_vol_CCLabel,    indx,patch);
        fine_new_dw->getModifiable(rho_CC,      lb->rho_CCLabel,       indx,patch);
        fine_new_dw->getModifiable(temp_CC,     lb->temp_CCLabel,      indx,patch);
        fine_new_dw->getModifiable(vel_CC,      lb->vel_CCLabel,       indx,patch);
        fine_new_dw->allocateAndPut(gamma,      lb->gammaLabel,        indx,patch);
        fine_new_dw->allocateAndPut(cv,         lb->specific_heatLabel,indx,patch);
        fine_new_dw->allocateAndPut(vol_frac,   lb->vol_frac_CCLabel,  indx,patch);
        

        //__________________________________
        // interpolate the intermediate variables (cv, gamma,vol_frac)
        // to the finer level along the boundary edge
        // Assumption: cv, gamma, vol_frac on the coarest level are accurate enough
        cv.initialize(d_EVIL_NUM);
        gamma.initialize(d_EVIL_NUM);
        vol_frac.initialize(d_EVIL_NUM);

        IntVector refineRatio = fineLevel->getRefinementRatio();
        DataWarehouse* coarse_new_dw = fine_new_dw->getOtherDataWarehouse(Task::CoarseNewDW);
        
        int orderOfInterpolation = 0;
        //__________________________________
        // Iterate over fine level boundary faces
        vector<Patch::FaceType>::const_iterator iter;  
        for (iter  = patch->getBoundaryFaces()->begin(); 
             iter != patch->getBoundaryFaces()->end(); ++iter){
          Patch::FaceType face = *iter;
          cout_dbg << " Setting BC on Face " << face << " patch " << patch->getGridIndex() << " Level " << fineLevel->getIndex() << endl;
          //__________________________________
          // fine level hi & lo cell iter limits
          // coarselevel hi and low index
          IntVector cl, ch, fl, fh;
          getCoarseFineFaceRange(patch, coarseLevel, face, orderOfInterpolation, cl, ch, fl, fh);

          constCCVariable<double> cv_coarse, gamma_coarse, vol_frac_coarse;
          coarse_new_dw->getRegion(cv_coarse,      lb->specific_heatLabel, indx, coarseLevel,cl, ch);
          coarse_new_dw->getRegion(gamma_coarse,   lb->gammaLabel,         indx, coarseLevel,cl, ch);
          coarse_new_dw->getRegion(vol_frac_coarse,lb->vol_frac_CCLabel,   indx, coarseLevel,cl, ch);

          selectInterpolator(cv_coarse,       orderOfInterpolation, coarseLevel, 
                             fineLevel, refineRatio, fl,fh, cv);
                              
          selectInterpolator(gamma_coarse,    orderOfInterpolation, coarseLevel, 
                             fineLevel, refineRatio, fl,fh, gamma);
                             
          selectInterpolator(vol_frac_coarse, orderOfInterpolation, coarseLevel, 
                             fineLevel, refineRatio, fl,fh, vol_frac);     
        } // boundary face loop
        
#if 0        
        // Worry about this later
        // the problem is that you don't know have delT for the finer level at this point in the cycle
        preprocess_CustomBCs("setBC_FineLevel",fine_old_dw, fine_new_dw, lb,  patch, 999,
                       d_customBC_var_basket);
#endif

        constCCVariable<double> placeHolder;

        
        setBC(rho_CC, "Density",  placeHolder, placeHolder,
              patch,d_sharedState, indx, fine_new_dw, d_customBC_var_basket);

        setBC(vel_CC, "Velocity", 
              patch,d_sharedState, indx, fine_new_dw, d_customBC_var_basket);       

        setBC(temp_CC,"Temperature",gamma, cv,
              patch,d_sharedState, indx, fine_new_dw, d_customBC_var_basket);

        setSpecificVolBC(sp_vol_CC[m], "SpecificVol", false,rho_CC,vol_frac,
                         patch,d_sharedState, indx);
                         
        sp_vol_const[m] = sp_vol_CC[m];  // needed by pressure BC
                         
#if 0
        // worry about this later
        delete_CustomBCs(d_customBC_var_basket);
#endif
        //__________________________________
        //    Model Variables                     
        if(d_modelSetup && d_modelSetup->tvars.size() > 0){
          vector<TransportedVariable*>::iterator t_iter;
          for( t_iter  = d_modelSetup->tvars.begin();
               t_iter != d_modelSetup->tvars.end(); t_iter++){
            TransportedVariable* tvar = *t_iter;

            if(tvar->matls->contains(indx)){
              string Labelname = tvar->var->getName();
              CCVariable<double> q_CC;
              fine_new_dw->getModifiable(q_CC, tvar->var, indx, patch);
          
              setBC(q_CC, Labelname,  patch, d_sharedState, indx, fine_new_dw);

              if(switchDebug_AMR_refineInterface){
                printData(indx, patch, 1, "BOT_setBC_FineLevel", Labelname, q_CC);
              }
            }
          }
        }
        
        //__________________________________
        //  Print Data 
        if(switchDebug_AMR_refine){
          ostringstream desc;    
          desc << "BOT_setBC_FineLevel_Mat_" << indx << "_patch_"<< patch->getID();
          printData(indx, patch,   1, desc.str(), "rho_CC",    rho_CC);
          printData(indx, patch,   1, desc.str(), "sp_vol_CC", sp_vol_CC[m]);
          printData(indx, patch,   1, desc.str(), "Temp_CC",   temp_CC);
          printVector(indx, patch, 1, desc.str(), "vel_CC", 0, vel_CC);
        }
      } // matl loop
      
      //__________________________________
      //  Pressure boundary condition
      CCVariable<double> press_CC;
      StaticArray<CCVariable<double> > placeHolder(0);
      
      fine_new_dw->getModifiable(press_CC, lb->press_CCLabel, 0, patch);
      
      setBC(press_CC, placeHolder, sp_vol_const, d_surroundingMatl_indx,
            "sp_vol", "Pressure", patch , d_sharedState, 0, fine_new_dw, 
            d_customBC_var_basket);
      
      if(switchDebug_AMR_refine){
        ostringstream desc;    
        desc << "BOT_setBC_FineLevel_Mat_" << 0 << "_patch_"<< patch->getID();
        printData(0, patch, 1, desc.str(), "press_CC", press_CC);
      }      
    }  // patches loop
    cout_dbg.setActive(dbg_onOff);  // reset on/off switch for cout_dbg
  }             
}


/*___________________________________________________________________
 Function~  AMRICE::scheduleRefine--  
_____________________________________________________________________*/
void AMRICE::scheduleRefine(const PatchSet* patches,
                            SchedulerP& sched)
{
  const Level* fineLevel = getLevel(patches);
  int L_indx = fineLevel->getIndex();
  
  if(L_indx > 0  && doICEOnLevel(L_indx, fineLevel->getGrid()->numLevels())){
    
    cout_doing << d_myworld->myrank() 
               << " AMRICE::scheduleRefine\t\t\t\tL-" 
               <<  L_indx << " P-" << *patches << '\n';
    Task* task = scinew Task("AMRICE::refine",this, &AMRICE::refine);
    
    MaterialSubset* subset = scinew MaterialSubset;
    
    subset->add(0);
    Ghost::GhostType  gac = Ghost::AroundCells;
        
    task->requires(Task::NewDW, lb->press_CCLabel,
                   0, Task::CoarseLevel, subset, Task::OutOfDomain, gac,1);
    
    task->requires(Task::NewDW, lb->rho_CCLabel,
                   0, Task::CoarseLevel, 0, Task::NormalDomain, gac,1);
    
    task->requires(Task::NewDW, lb->sp_vol_CCLabel,
                   0, Task::CoarseLevel, 0, Task::NormalDomain, gac,1);
    
    task->requires(Task::NewDW, lb->temp_CCLabel,
                   0, Task::CoarseLevel, 0, Task::NormalDomain, gac,1);
    
    task->requires(Task::NewDW, lb->vel_CCLabel,
                   0, Task::CoarseLevel, 0, Task::NormalDomain, gac,1);
    
    //__________________________________
    // Model Variables.
    if(d_modelSetup && d_modelSetup->tvars.size() > 0){
      vector<TransportedVariable*>::iterator iter;
      
      for(iter = d_modelSetup->tvars.begin();
          iter != d_modelSetup->tvars.end(); iter++){
        TransportedVariable* tvar = *iter;
        task->requires(Task::NewDW, tvar->var,
                       0, Task::CoarseLevel, 0, Task::NormalDomain, gac,1);
        task->computes(tvar->var);
      }
    }
    
    task->computes(lb->press_CCLabel, subset, Task::OutOfDomain);
    task->computes(lb->rho_CCLabel);
    task->computes(lb->sp_vol_CCLabel);
    task->computes(lb->temp_CCLabel);
    task->computes(lb->vel_CCLabel);
    sched->addTask(task, patches, d_sharedState->allICEMaterials());
    
    //__________________________________
    // Sub Task 
    scheduleSetBC_FineLevel(patches, sched);
  }
}

/*___________________________________________________________________
 Function~  AMRICE::Refine--  
_____________________________________________________________________*/
void AMRICE::refine(const ProcessorGroup*,
                    const PatchSubset* patches,
                    const MaterialSubset* matls,
                    DataWarehouse*,
                    DataWarehouse* new_dw)
{
  
  const Level* fineLevel = getLevel(patches);
  const Level* coarseLevel = fineLevel->getCoarserLevel().get_rep();
  
  cout_doing << d_myworld->myrank() 
             << " Doing refine \t\t\t\t\t AMRICE L-"<< fineLevel->getIndex();
  IntVector rr(fineLevel->getRefinementRatio());
  double invRefineRatio = 1./(rr.x()*rr.y()*rr.z());
  
  for(int p=0;p<patches->size();p++){  
    const Patch* finePatch = patches->get(p);
    cout_doing << "  patch " << finePatch->getID()<< endl;
    
    Level::selectType coarsePatches;
    finePatch->getCoarseLevelPatches(coarsePatches);
    
    // bullet proofing
    iteratorTest(finePatch, fineLevel, coarseLevel, new_dw);

    //testInterpolators<double>(new_dw,d_orderOfInterpolation,coarseLevel,fineLevel,finePatch);

   
    // refine pressure
    CCVariable<double> press_CC;
    new_dw->allocateAndPut(press_CC, lb->press_CCLabel, 0, finePatch);
    press_CC.initialize(d_EVIL_NUM);
    CoarseToFineOperator<double>(press_CC,  lb->press_CCLabel,0, new_dw, 
                         invRefineRatio, finePatch, fineLevel, coarseLevel);

    for(int m = 0;m<matls->size();m++){
      int indx = matls->get(m);
      CCVariable<double> rho_CC, temp, sp_vol_CC;
      CCVariable<Vector> vel_CC;
      
      new_dw->allocateAndPut(rho_CC,   lb->rho_CCLabel,    indx, finePatch);
      new_dw->allocateAndPut(sp_vol_CC,lb->sp_vol_CCLabel, indx, finePatch);
      new_dw->allocateAndPut(temp,     lb->temp_CCLabel,   indx, finePatch);
      new_dw->allocateAndPut(vel_CC,   lb->vel_CCLabel,    indx, finePatch);  
      
      rho_CC.initialize(d_EVIL_NUM);
      sp_vol_CC.initialize(d_EVIL_NUM);
      temp.initialize(d_EVIL_NUM);
      vel_CC.initialize(Vector(d_EVIL_NUM));

      // refine  
      CoarseToFineOperator<double>(rho_CC,    lb->rho_CCLabel,  indx, new_dw, 
                         invRefineRatio, finePatch, fineLevel, coarseLevel);      

      CoarseToFineOperator<double>(sp_vol_CC, lb->sp_vol_CCLabel,indx, new_dw, 
                         invRefineRatio, finePatch, fineLevel, coarseLevel);

      CoarseToFineOperator<double>(temp,      lb->temp_CCLabel, indx, new_dw, 
                         invRefineRatio, finePatch, fineLevel, coarseLevel);
       
      CoarseToFineOperator<Vector>( vel_CC,   lb->vel_CCLabel,  indx, new_dw, 
                         invRefineRatio, finePatch, fineLevel, coarseLevel);
      //__________________________________
      //    Model Variables                     
      if(d_modelSetup && d_modelSetup->tvars.size() > 0){
        vector<TransportedVariable*>::iterator t_iter;
        for( t_iter  = d_modelSetup->tvars.begin();
             t_iter != d_modelSetup->tvars.end(); t_iter++){
          TransportedVariable* tvar = *t_iter;

          if(tvar->matls->contains(indx)){
            CCVariable<double> q_CC;
            new_dw->allocateAndPut(q_CC, tvar->var, indx, finePatch);
            
            q_CC.initialize(d_EVIL_NUM);
            
            CoarseToFineOperator<double>(q_CC, tvar->var, indx, new_dw, 
                       invRefineRatio, finePatch, fineLevel, coarseLevel);
                       
            if(switchDebug_AMR_refine){
              ostringstream desc; 
              string name = tvar->var->getName();
              printData(indx, finePatch, 1, "Refine_task", name, q_CC);
            }                 
          } 
        }
      }    
      
      //__________________________________
      //  Print Data
      if(switchDebug_AMR_refine){ 
      ostringstream desc;     
        desc << "BOT_Refine_Mat_" << indx << "_patch_"<< finePatch->getID();
        printData(indx, finePatch,   1, desc.str(), "press_CC",  press_CC); 
        printData(indx, finePatch,   1, desc.str(), "rho_CC",    rho_CC);
        printData(indx, finePatch,   1, desc.str(), "sp_vol_CC", sp_vol_CC);
        printData(indx, finePatch,   1, desc.str(), "Temp_CC",   temp);
        printVector(indx, finePatch, 1, desc.str(), "vel_CC", 0, vel_CC);
      }
    }
  }  // course patch loop 
}

/*_____________________________________________________________________
 Function~  AMRICE::iteratorTest--
 Purpose~   Verify that the all of the fine level cells will be accessed 
_____________________________________________________________________*/
void AMRICE::iteratorTest(const Patch* finePatch,
                          const Level* fineLevel,
                          const Level* coarseLevel,
                          DataWarehouse* new_dw)
{
  Level::selectType coarsePatches;
  finePatch->getCoarseLevelPatches(coarsePatches); 
  IntVector fl = finePatch->getCellLowIndex();
  IntVector fh = finePatch->getCellHighIndex();
  
  CCVariable<double> hitCells;
  new_dw->allocateTemporary(hitCells, finePatch);
  hitCells.initialize(d_EVIL_NUM);
  IntVector lo(0,0,0);
  IntVector hi(0,0,0);
  
  
  for(int i=0;i<coarsePatches.size();i++){
    const Patch* coarsePatch = coarsePatches[i];
    // iterator should hit the cells over the intersection of the fine and coarse patches

    IntVector cl = coarsePatch->getLowIndex();
    IntVector ch = coarsePatch->getHighIndex();
         
    IntVector fl_tmp = coarseLevel->mapCellToFiner(cl);
    IntVector fh_tmp = coarseLevel->mapCellToFiner(ch);
    
    lo = Max(fl, fl_tmp);
    hi = Min(fh, fh_tmp);
    
    for(CellIterator iter(lo,hi); !iter.done(); iter++){
      IntVector c = *iter;
      hitCells[c] = 1.0;
    }
#if 0
    cout << " coarsePatch.size() " << coarsePatches.size() 
         << " coarsePatch " << coarsePatch->getID()
         << " finePatch " << finePatch->getID() 
         << " fineLevel: fl " << fl << " fh " << fh
         << " coarseLevel: cl " << cl << " ch " << ch 
         << " final Iterator: " << lo << " " << hi << endl;
#endif    
    
  }
  
  //____ B U L L E T   P R O O F I N G_______ 
  // All cells must be initialized at this point
  IntVector badCell;
  CellIterator iter(lo,hi);
  if( isEqual<double>(d_EVIL_NUM,iter,hitCells, badCell) ){
  
    IntVector c_badCell = fineLevel->mapCellToCoarser(badCell);
    const Patch* patch = coarseLevel->selectPatchForCellIndex(c_badCell);
    
    ostringstream warn;
    warn <<"ERROR AMRICE::Refine Task:iteratorTest "
         << "detected an fine level cell that won't get initialized "
         << badCell << " Patch " << finePatch->getID() 
         << " Level idx "<<fineLevel->getIndex()<<"\n "
         << "The underlying coarse cell "<< c_badCell 
         << " belongs to coarse level patch " << patch->getID() << "\n";
    throw InvalidValue(warn.str(), __FILE__, __LINE__);
  }  
}


/*___________________________________________________________________
 Function~  AMRICE::scheduleCoarsen--  
_____________________________________________________________________*/
void AMRICE::scheduleCoarsen(const LevelP& coarseLevel,
                               SchedulerP& sched)
{
  const Level* fineLevel = coarseLevel->getFinerLevel().get_rep();
  if(!doICEOnLevel(fineLevel->getIndex(), fineLevel->getGrid()->numLevels()))
    return;
    
  Ghost::GhostType  gn = Ghost::None; 
  cout_doing << d_myworld->myrank() 
             << " AMRICE::scheduleCoarsen\t\t\t\tL-" 
             << fineLevel->getIndex()<< "->"<<coarseLevel->getIndex()<<endl; 
             
  Task* task = scinew Task("AMRICE::coarsen",this, &AMRICE::coarsen);

  Task::DomainSpec ND   = Task::NormalDomain;                            
  Task::DomainSpec oims = Task::OutOfDomain;  //outside of ice matlSet.  
  const MaterialSet* all_matls = d_sharedState->allMaterials();          
  const MaterialSubset* all_matls_sub = all_matls->getUnion();           

  task->requires(Task::NewDW, lb->press_CCLabel,
               0, Task::FineLevel,  d_press_matl,oims, gn, 0);
                 
  task->requires(Task::NewDW, lb->rho_CCLabel,
               0, Task::FineLevel,  all_matls_sub,ND, gn, 0);
               
  task->requires(Task::NewDW, lb->sp_vol_CCLabel,
               0, Task::FineLevel,  all_matls_sub,ND, gn, 0);
  
  task->requires(Task::NewDW, lb->temp_CCLabel,
               0, Task::FineLevel,  all_matls_sub,ND, gn, 0);
  
  task->requires(Task::NewDW, lb->vel_CCLabel,
               0, Task::FineLevel,  all_matls_sub,ND, gn, 0);
               
  task->requires(Task::NewDW, lb->specific_heatLabel,
               0, Task::FineLevel,  all_matls_sub,ND, gn, 0);

  //__________________________________
  // Model Variables.
  if(d_modelSetup && d_modelSetup->tvars.size() > 0){
    vector<TransportedVariable*>::iterator iter;

    for(iter = d_modelSetup->tvars.begin();
       iter != d_modelSetup->tvars.end(); iter++){
      TransportedVariable* tvar = *iter;
      task->requires(Task::NewDW, tvar->var,
                  0, Task::FineLevel,all_matls_sub,ND, gn, 0);
      task->modifies(tvar->var);
    }
  }
  
  task->modifies(lb->press_CCLabel, d_press_matl, oims);
  task->modifies(lb->rho_CCLabel);
  task->modifies(lb->sp_vol_CCLabel);
  task->modifies(lb->temp_CCLabel);
  task->modifies(lb->vel_CCLabel);
  task->requires(Task::NewDW, lb->specific_heatLabel, gn, 0);

  sched->addTask(task, coarseLevel->eachPatch(), d_sharedState->allICEMaterials()); 
  
  //__________________________________
  // schedule refluxing
  if(d_doRefluxing){
    scheduleReflux_computeCorrectionFluxes(coarseLevel, sched);
    scheduleReflux_applyCorrection(coarseLevel, sched);
  }
}

/*___________________________________________________________________
 Function~  AMRICE::Coarsen--  
_____________________________________________________________________*/
void AMRICE::coarsen(const ProcessorGroup*,
                     const PatchSubset* patches,
                     const MaterialSubset* matls,
                     DataWarehouse*,
                     DataWarehouse* new_dw)
{
  
  const Level* coarseLevel = getLevel(patches);
  const Level* fineLevel = coarseLevel->getFinerLevel().get_rep();
  cout_doing << d_myworld->myrank()
             << " Doing coarsen \t\t\t\t\t AMRICE L-" 
             <<fineLevel->getIndex()<< "->"<<coarseLevel->getIndex();
  
  bool dbg_onOff = cout_dbg.active();      // is cout_dbg switch on or off
  Ghost::GhostType  gn = Ghost::None;
  
  
  for(int p=0;p<patches->size();p++){  
    const Patch* coarsePatch = patches->get(p);
    cout_doing <<"  patch " << coarsePatch->getID()<< endl;
    
    for(int m = 0;m<matls->size();m++){
      int indx = matls->get(m);

      constCCVariable<double> cv;
      CCVariable<double> rho_CC, temp, sp_vol_CC;
      CCVariable<Vector> vel_CC;

      new_dw->get(cv,                 lb->specific_heatLabel, indx, coarsePatch, gn,0);
      new_dw->getModifiable(rho_CC,   lb->rho_CCLabel,        indx, coarsePatch);
      new_dw->getModifiable(sp_vol_CC,lb->sp_vol_CCLabel,     indx, coarsePatch);
      new_dw->getModifiable(temp,     lb->temp_CCLabel,       indx, coarsePatch);
      new_dw->getModifiable(vel_CC,   lb->vel_CCLabel,        indx, coarsePatch);  
      
      // coarsen         
      fineToCoarseOperator<double>(rho_CC,    rho_CC, cv, "mass", 
                         lb->rho_CCLabel, indx, new_dw, 
                         coarsePatch, coarseLevel, fineLevel);      

      fineToCoarseOperator<double>(sp_vol_CC, rho_CC, cv, "sp_vol",
                         lb->sp_vol_CCLabel,indx, new_dw, 
                         coarsePatch, coarseLevel, fineLevel);

      fineToCoarseOperator<double>(temp,      rho_CC, cv, "energy",   
                         lb->temp_CCLabel, indx, new_dw, 
                         coarsePatch, coarseLevel, fineLevel);
       
      fineToCoarseOperator<Vector>( vel_CC,   rho_CC, cv, "momentum",   
                         lb->vel_CCLabel,  indx, new_dw, 
                         coarsePatch, coarseLevel, fineLevel);
      
      //__________________________________
      // pressure
      if( indx == 0){
          // pressure
        CCVariable<double> press_CC;                  
        new_dw->getModifiable(press_CC, lb->press_CCLabel,  0,    coarsePatch);
        fineToCoarseOperator<double>(press_CC,  rho_CC, cv, "pressure",
                         lb->press_CCLabel, 0,   new_dw, 
                         coarsePatch, coarseLevel, fineLevel);
      }                   
                         
                         
      //__________________________________
      //    Model Variables                     
      if(d_modelSetup && d_modelSetup->tvars.size() > 0){
        vector<TransportedVariable*>::iterator t_iter;
        for( t_iter  = d_modelSetup->tvars.begin();
            t_iter != d_modelSetup->tvars.end(); t_iter++){
          TransportedVariable* tvar = *t_iter;

          if(tvar->matls->contains(indx)){
            CCVariable<double> q_CC;
            new_dw->getModifiable(q_CC, tvar->var, indx, coarsePatch);
            fineToCoarseOperator<double>(q_CC, rho_CC, cv, "scalar", 
                       tvar->var, indx, new_dw, 
                       coarsePatch, coarseLevel, fineLevel);
            
            if(switchDebug_AMR_refine){  
              string name = tvar->var->getName();
              printData(indx, coarsePatch, 1, "coarsen_models", name, q_CC);
            }                 
          }
        }
      }    

      //__________________________________
      //  Print Data 
      if(switchDebug_AMR_refine){
        ostringstream desc;     
        desc << "coarsen_Mat_" << indx << "_patch_"<< coarsePatch->getID();
       // printData(indx, coarsePatch,   1, desc.str(), "press_CC",    press_CC);
        printData(indx, coarsePatch,   1, desc.str(), "rho_CC",      rho_CC);
        printData(indx, coarsePatch,   1, desc.str(), "sp_vol_CC",   sp_vol_CC);
        printData(indx, coarsePatch,   1, desc.str(), "Temp_CC",     temp);
        printVector(indx, coarsePatch, 1, desc.str(), "vel_CC", 0,   vel_CC);
      }
    }
  }  // course patch loop 
  cout_dbg.setActive(dbg_onOff);  // reset on/off switch for cout_dbg
}
/*___________________________________________________________________
 Function~  AMRICE::scheduleReflux_computeCorrectionFluxes--  
_____________________________________________________________________*/
void AMRICE::scheduleReflux_computeCorrectionFluxes(const LevelP& coarseLevel,
                                                    SchedulerP& sched)
{
  const Level* fineLevel = coarseLevel->getFinerLevel().get_rep(); 
  cout_doing << d_myworld->myrank() 
             << " AMRICE::scheduleReflux_computeCorrectionFluxes\tL-" 
             << fineLevel->getIndex() << "->"<< coarseLevel->getIndex()<< endl;
             
  Task* task = scinew Task("AMRICE::reflux_computeCorrectionFluxes",
                           this, &AMRICE::reflux_computeCorrectionFluxes);
  
  Ghost::GhostType gn  = Ghost::None;
  Ghost::GhostType gac = Ghost::AroundCells;
  task->requires(Task::NewDW, lb->rho_CCLabel,        gac, 1);
  task->requires(Task::NewDW, lb->specific_heatLabel, gac, 1);
  
  //__________________________________
  // Fluxes from the fine level            
                                      // MASS
  task->requires(Task::NewDW, lb->mass_X_FC_fluxLabel,
               0,Task::FineLevel, 0, Task::NormalDomain, gn, 0);
  task->requires(Task::NewDW, lb->mass_Y_FC_fluxLabel,
               0,Task::FineLevel, 0, Task::NormalDomain, gn, 0);
  task->requires(Task::NewDW, lb->mass_Z_FC_fluxLabel,
               0,Task::FineLevel, 0, Task::NormalDomain, gn, 0);
                                      // MOMENTUM
  task->requires(Task::NewDW, lb->mom_X_FC_fluxLabel,
               0,Task::FineLevel, 0, Task::NormalDomain, gn, 0);
  task->requires(Task::NewDW, lb->mom_Y_FC_fluxLabel,
               0,Task::FineLevel, 0, Task::NormalDomain, gn, 0);
  task->requires(Task::NewDW, lb->mom_Z_FC_fluxLabel,
               0,Task::FineLevel, 0, Task::NormalDomain, gn, 0);
                                      // INT_ENG
  task->requires(Task::NewDW, lb->int_eng_X_FC_fluxLabel,
               0,Task::FineLevel, 0, Task::NormalDomain, gn, 0);
  task->requires(Task::NewDW, lb->int_eng_Y_FC_fluxLabel,
               0,Task::FineLevel, 0, Task::NormalDomain, gn, 0);
  task->requires(Task::NewDW, lb->int_eng_Z_FC_fluxLabel,
               0,Task::FineLevel, 0, Task::NormalDomain, gn, 0);
                                      // SPECIFIC VOLUME
  task->requires(Task::NewDW, lb->sp_vol_X_FC_fluxLabel,
               0,Task::FineLevel, 0, Task::NormalDomain, gn, 0);
  task->requires(Task::NewDW, lb->sp_vol_Y_FC_fluxLabel,
               0,Task::FineLevel, 0, Task::NormalDomain, gn, 0);
  task->requires(Task::NewDW, lb->sp_vol_Z_FC_fluxLabel,
               0,Task::FineLevel, 0, Task::NormalDomain, gn, 0);             

  //__________________________________
  // Model Variables.
  if(d_modelSetup && d_modelSetup->d_reflux_vars.size() > 0){
    vector<AMR_refluxVariable*>::iterator iter;
    for( iter  = d_modelSetup->d_reflux_vars.begin();
         iter != d_modelSetup->d_reflux_vars.end(); iter++){
      AMR_refluxVariable* rvar = *iter;
      
      task->requires(Task::NewDW, rvar->var_X_FC_flux,
                  0, Task::FineLevel, 0, Task::NormalDomain, gn, 0);
      task->requires(Task::NewDW, rvar->var_Y_FC_flux,
                  0, Task::FineLevel, 0, Task::NormalDomain, gn, 0);
      task->requires(Task::NewDW, rvar->var_Z_FC_flux,
                  0, Task::FineLevel, 0, Task::NormalDomain, gn, 0);
                  
      task->modifies(rvar->var_X_FC_flux);
      task->modifies(rvar->var_Y_FC_flux);
      task->modifies(rvar->var_Z_FC_flux);
    }
  }

  task->modifies(lb->mass_X_FC_fluxLabel);
  task->modifies(lb->mass_Y_FC_fluxLabel);
  task->modifies(lb->mass_Z_FC_fluxLabel);
  
  task->modifies(lb->mom_X_FC_fluxLabel);
  task->modifies(lb->mom_Y_FC_fluxLabel);
  task->modifies(lb->mom_Z_FC_fluxLabel);
  
  task->modifies(lb->int_eng_X_FC_fluxLabel);
  task->modifies(lb->int_eng_Y_FC_fluxLabel);
  task->modifies(lb->int_eng_Z_FC_fluxLabel); 
  
  task->modifies(lb->sp_vol_X_FC_fluxLabel);
  task->modifies(lb->sp_vol_Y_FC_fluxLabel);
  task->modifies(lb->sp_vol_Z_FC_fluxLabel);
  
  sched->addTask(task, coarseLevel->eachPatch(), d_sharedState->allICEMaterials()); 
}
/*___________________________________________________________________
 Function~  AMRICE::Reflux_computeCorrectionFluxesFluxes--  
_____________________________________________________________________*/
void AMRICE::reflux_computeCorrectionFluxes(const ProcessorGroup*,
                                            const PatchSubset* coarsePatches,
                                            const MaterialSubset* matls,
                                            DataWarehouse*,
                                            DataWarehouse* new_dw)
{
  const Level* coarseLevel = getLevel(coarsePatches);
  const Level* fineLevel = coarseLevel->getFinerLevel().get_rep();
  
  cout_doing << d_myworld->myrank() 
             << " Doing reflux_computeCorrectionFluxes \t\t\t AMRICE L-"
             <<fineLevel->getIndex()<< "->"<< coarseLevel->getIndex();
  
  bool dbg_onOff = cout_dbg.active();      // is cout_dbg switch on or off
  
  for(int c_p=0;c_p<coarsePatches->size();c_p++){  
    const Patch* coarsePatch = coarsePatches->get(c_p);
    
    cout_doing <<"  patch " << coarsePatch->getID()<< endl;

    for(int m = 0;m<matls->size();m++){
      int indx = matls->get(m);
      constCCVariable<double> cv, rho_CC;

      Ghost::GhostType  gac = Ghost::AroundCells;
      new_dw->get(rho_CC,lb->rho_CCLabel,       indx,coarsePatch, gac,1);
      new_dw->get(cv,    lb->specific_heatLabel,indx,coarsePatch, gac,1);
      
      Level::selectType finePatches;
      coarsePatch->getFineLevelPatches(finePatches);
      
      for(int i=0; i < finePatches.size();i++){  
        const Patch* finePatch = finePatches[i];       
        //__________________________________
        //   compute the correction
        if(finePatch->hasCoarseFineInterfaceFace() ){
 
          refluxOperator_computeCorrectionFluxes<double>(rho_CC,  cv, "mass",   indx, 
                        coarsePatch, finePatch, coarseLevel, fineLevel,new_dw);

          refluxOperator_computeCorrectionFluxes<double>( rho_CC, cv, "sp_vol", indx, 
                        coarsePatch, finePatch, coarseLevel, fineLevel,new_dw);

          refluxOperator_computeCorrectionFluxes<Vector>( rho_CC, cv, "mom",    indx, 
                        coarsePatch, finePatch, coarseLevel, fineLevel,new_dw);

          refluxOperator_computeCorrectionFluxes<double>( rho_CC, cv, "int_eng", indx, 
                        coarsePatch, finePatch, coarseLevel, fineLevel,new_dw);
          //__________________________________
          //    Model Variables
          if(d_modelSetup && d_modelSetup->d_reflux_vars.size() > 0){
            vector<AMR_refluxVariable*>::iterator iter;
            for( iter  = d_modelSetup->d_reflux_vars.begin();
                 iter != d_modelSetup->d_reflux_vars.end(); iter++){
              AMR_refluxVariable* r_var = *iter;

              if(r_var->matls->contains(indx)){
                string var_name = r_var->var_CC->getName();
                refluxOperator_computeCorrectionFluxes<double>(rho_CC, cv, var_name, indx, 
                              coarsePatch, finePatch, coarseLevel, fineLevel,new_dw);
              }
            }
          }
        }
      }  // finePatch loop 


      //__________________________________
      //  Print Data
      if(switchDebug_AMR_reflux){ 
        ostringstream desc;     
        desc << "RefluxComputeCorrectonFluxes_Mat_" << indx << "_patch_"<< coarsePatch->getID();
        // need to add something here
      }
    }  // matl loop
  }  // course patch loop
  
  cout_dbg.setActive(dbg_onOff);  // reset on/off switch for cout_dbg
}

/*___________________________________________________________________
 Function~  ICE::refluxCoarseLevelIterator--  
 Purpose:  returns the iterator and face-centered offset that the coarse 
           level uses to do refluxing.  THIS IS COMPILCATED AND CONFUSING
_____________________________________________________________________*/
void ICE::refluxCoarseLevelIterator(Patch::FaceType patchFace,
                               const Patch* coarsePatch,
                               const Patch* finePatch,
                               const Level* fineLevel,
                               CellIterator& iter,
                               IntVector& coarse_FC_offset,
                               bool& isRight_CP_FP_pair)
{
  CellIterator f_iter=finePatch->getFaceCellIterator(patchFace, "alongInteriorFaceCells");

  // find the intersection of the fine patch face iterator and underlying coarse patch
  IntVector f_lo_face = f_iter.begin();                 // fineLevel face indices   
  IntVector f_hi_face = f_iter.end();

  f_lo_face = fineLevel->mapCellToCoarser(f_lo_face);     
  f_hi_face = fineLevel->mapCellToCoarser(f_hi_face);

  IntVector c_lo_patch = coarsePatch->getLowIndex(); 
  IntVector c_hi_patch = coarsePatch->getHighIndex();

  IntVector l = Max(f_lo_face, c_lo_patch);             // intersection
  IntVector h = Min(f_hi_face, c_hi_patch);

  //__________________________________
  // Offset for the coarse level iterator
  // shift l & h,   1 cell for x+, y+, z+ finePatchfaces
  // shift l only, -1 cell for x-, y-, z- finePatchfaces
  // 
  // Offset for the coarse_face center (c_FC_offset)
  // shift 1 cell   for x-, y-, z- finePatch faces
  // shift 0 cells  for x+, y+, z+ finePatchFaces

  string name = finePatch->getFaceName(patchFace);
  IntVector offset = finePatch->faceDirection(patchFace);
  coarse_FC_offset = IntVector(0,0,0);

  if(name == "xminus" || name == "yminus" || name == "zminus"){
    l += offset;
    coarse_FC_offset = -offset;
  }
  if(name == "xplus" || name == "yplus" || name == "zplus"){
    l += offset;
    h += offset;
  }

  l = Max(l, coarsePatch->getLowIndex());
  h = Min(h, coarsePatch->getHighIndex());
  
  iter=CellIterator(l,h);
  isRight_CP_FP_pair = false;
  if ( coarsePatch->containsCell(l + coarse_FC_offset) ){
    isRight_CP_FP_pair = true;
  }
  
  if (cout_dbg.active()) {
    cout_dbg << "refluxCoarseLevelIterator: face "<< patchFace
             << " finePatch " << finePatch->getID()
             << " coarsePatch " << coarsePatch->getID()
             << " [CellIterator at " << iter.begin() << " of " << iter.end() 
             << "] coarse_FC_offset " << coarse_FC_offset 
             << " does this coarse patch own the face centered variable " 
             << isRight_CP_FP_pair << endl; 
  }
}


/*___________________________________________________________________
 Function~  AMRICE::scheduleReflux_applyCorrection--  
_____________________________________________________________________*/
void AMRICE::scheduleReflux_applyCorrection(const LevelP& coarseLevel,
                                            SchedulerP& sched)
{
  const Level* fineLevel = coarseLevel->getFinerLevel().get_rep();
  cout_doing << d_myworld->myrank() 
             << " AMRICE::scheduleReflux_applyCorrectionFluxes\t\tL-" 
             << fineLevel->getIndex() << "->"<< coarseLevel->getIndex()<< endl;
             
  Task* task = scinew Task("AMRICE::reflux_applyCorrectionFluxes",
                          this, &AMRICE::reflux_applyCorrectionFluxes);
  
  Ghost::GhostType gn = Ghost::None;
  Ghost::GhostType  gac = Ghost::AroundCells;
  
    
  //__________________________________
  // coarse grid solution variables after advection               
  task->requires(Task::NewDW, lb->rho_CCLabel,     gn, 0);
  task->requires(Task::NewDW, lb->vel_CCLabel,     gn, 0);
  task->requires(Task::NewDW, lb->sp_vol_CCLabel,  gn, 0);
  task->requires(Task::NewDW, lb->temp_CCLabel,    gn, 0);                         
  //__________________________________
  // Correction fluxes  from the coarse level            
                                      // MASS
  task->requires(Task::NewDW, lb->mass_X_FC_fluxLabel, gac, 1);  
  task->requires(Task::NewDW, lb->mass_Y_FC_fluxLabel, gac, 1);  
  task->requires(Task::NewDW, lb->mass_Z_FC_fluxLabel, gac, 1);  
                                      // MOMENTUM
  task->requires(Task::NewDW, lb->mom_X_FC_fluxLabel,  gac, 1);  
  task->requires(Task::NewDW, lb->mom_Y_FC_fluxLabel,  gac, 1);  
  task->requires(Task::NewDW, lb->mom_Z_FC_fluxLabel,  gac, 1);  
                                      // INT_ENG
  task->requires(Task::NewDW, lb->int_eng_X_FC_fluxLabel,gac, 1);    
  task->requires(Task::NewDW, lb->int_eng_Y_FC_fluxLabel,gac, 1);    
  task->requires(Task::NewDW, lb->int_eng_Z_FC_fluxLabel,gac, 1);    
                                      // SPECIFIC VOLUME
  task->requires(Task::NewDW, lb->sp_vol_X_FC_fluxLabel, gac, 1);    
  task->requires(Task::NewDW, lb->sp_vol_Y_FC_fluxLabel, gac, 1);    
  task->requires(Task::NewDW, lb->sp_vol_Z_FC_fluxLabel, gac, 1);               


  //__________________________________
  // Model Variables.
  if(d_modelSetup && d_modelSetup->d_reflux_vars.size() > 0){
    vector<AMR_refluxVariable*>::iterator iter;
    for( iter  = d_modelSetup->d_reflux_vars.begin();
         iter != d_modelSetup->d_reflux_vars.end(); iter++){
      AMR_refluxVariable* rvar = *iter;
      
      task->requires(Task::NewDW, rvar->var_X_FC_flux, gac, 1);    
      task->requires(Task::NewDW, rvar->var_Y_FC_flux, gac, 1);    
      task->requires(Task::NewDW, rvar->var_Z_FC_flux, gac, 1);
      task->modifies(rvar->var_CC);
    }
  }

  task->modifies(lb->rho_CCLabel);
  task->modifies(lb->sp_vol_CCLabel);
  task->modifies(lb->temp_CCLabel);
  task->modifies(lb->vel_CCLabel);

  sched->addTask(task, coarseLevel->eachPatch(), d_sharedState->allICEMaterials()); 
}
/*___________________________________________________________________
 Function~  AMRICE::Reflux_applyCorrectionFluxes--  
_____________________________________________________________________*/
void AMRICE::reflux_applyCorrectionFluxes(const ProcessorGroup*,
                                          const PatchSubset* coarsePatches,
                                          const MaterialSubset* matls,
                                          DataWarehouse*,
                                          DataWarehouse* new_dw)
{
  const Level* coarseLevel = getLevel(coarsePatches);
  const Level* fineLevel = coarseLevel->getFinerLevel().get_rep();
  
  cout_doing << d_myworld->myrank() 
             << " Doing reflux_applyCorrectionFluxes \t\t\t AMRICE L-"
             <<fineLevel->getIndex()<< "->"<< coarseLevel->getIndex();
  
  bool dbg_onOff = cout_dbg.active();      // is cout_dbg switch on or off
  
  for(int c_p=0;c_p<coarsePatches->size();c_p++){  
    const Patch* coarsePatch = coarsePatches->get(c_p);
    cout_doing << "  patch " << coarsePatch->getID()<< endl;
    
    for(int m = 0;m<matls->size();m++){
      int indx = matls->get(m);     
      CCVariable<double> rho_CC, temp, sp_vol_CC;
      constCCVariable<double> cv;
      CCVariable<Vector> vel_CC;

      Ghost::GhostType  gn  = Ghost::None;
      new_dw->getModifiable(rho_CC,   lb->rho_CCLabel,    indx, coarsePatch);
      new_dw->getModifiable(sp_vol_CC,lb->sp_vol_CCLabel, indx, coarsePatch);
      new_dw->getModifiable(temp,     lb->temp_CCLabel,   indx, coarsePatch);
      new_dw->getModifiable(vel_CC,   lb->vel_CCLabel,    indx, coarsePatch);
      new_dw->get(cv,                 lb->specific_heatLabel,indx,coarsePatch, gn,0);
      
      Level::selectType finePatches;
      coarsePatch->getFineLevelPatches(finePatches); 
      
      for(int i=0; i < finePatches.size();i++){  
        const Patch* finePatch = finePatches[i];        
//      cout_doing << d_myworld->myrank() << "  coarsePatch " << coarsePatch->getID() <<" finepatch " << finePatch->getID()<< endl;

        //__________________________________
        // Apply the correction
        if(finePatch->hasCoarseFineInterfaceFace() ){

          refluxOperator_applyCorrectionFluxes<double>(rho_CC,    "mass",  indx, 
                        coarsePatch, finePatch, coarseLevel, fineLevel,new_dw);

          refluxOperator_applyCorrectionFluxes<double>(sp_vol_CC, "sp_vol",  indx, 
                        coarsePatch, finePatch, coarseLevel, fineLevel,new_dw);

          refluxOperator_applyCorrectionFluxes<Vector>(vel_CC, "mom",     indx, 
                        coarsePatch, finePatch, coarseLevel, fineLevel,new_dw);

          refluxOperator_applyCorrectionFluxes<double>(temp,  "int_eng", indx, 
                        coarsePatch, finePatch, coarseLevel, fineLevel,new_dw);
          //__________________________________
          //    Model Variables
          if(d_modelSetup && d_modelSetup->d_reflux_vars.size() > 0){
            vector<AMR_refluxVariable*>::iterator iter;
            for( iter  = d_modelSetup->d_reflux_vars.begin();
                 iter != d_modelSetup->d_reflux_vars.end(); iter++){
              AMR_refluxVariable* r_var = *iter;

              if(r_var->matls->contains(indx)){
                CCVariable<double> q_CC;
                string var_name = r_var->var_CC->getName();
                new_dw->getModifiable(q_CC,  r_var->var_CC, indx, coarsePatch);

                refluxOperator_applyCorrectionFluxes<double>(q_CC, var_name, indx, 
                              coarsePatch, finePatch, coarseLevel, fineLevel,new_dw);

                if(switchDebug_AMR_refine){
                  string name = r_var->var_CC->getName();
                  printData(indx, coarsePatch, 1, "coarsen_models", name, q_CC);
                }
              }
            }
          }
        }  // patch has a coarseFineInterface
      }  // finePatch loop 
  
      //__________________________________
      //  Print Data
      if(switchDebug_AMR_reflux){ 
        ostringstream desc;     
        desc << "Reflux_applyCorrection_Mat_" << indx << "_patch_"<< coarsePatch->getID();
        printData(indx, coarsePatch,   1, desc.str(), "rho_CC",   rho_CC);
        printData(indx, coarsePatch,   1, desc.str(), "sp_vol_CC",sp_vol_CC);
        printData(indx, coarsePatch,   1, desc.str(), "Temp_CC",  temp);
        printVector(indx, coarsePatch, 1, desc.str(), "vel_CC", 0,vel_CC);
      }
    }  // matl loop
  }  // course patch loop 
  cout_dbg.setActive(dbg_onOff);  // reset on/off switch for cout_dbg
}

/*_____________________________________________________________________
 Function~  AMRICE::scheduleInitialErrorEstimate--
______________________________________________________________________*/
void AMRICE::scheduleInitialErrorEstimate(const LevelP& coarseLevel,
                                          SchedulerP& sched)
{
#if 0
  scheduleErrorEstimate(coarseLevel, sched);
#endif
}

/*_____________________________________________________________________
 Function~  AMRICE::scheduleErrorEstimate--
______________________________________________________________________*/
void AMRICE::scheduleErrorEstimate(const LevelP& coarseLevel,
                                   SchedulerP& sched)
{
  if(!doICEOnLevel(coarseLevel->getIndex()+1, coarseLevel->getGrid()->numLevels()))
    return;
  cout_doing << d_myworld->myrank() 
             << " AMRICE::scheduleErrorEstimate \t\t\tL-" 
             << coarseLevel->getIndex() << '\n';
  
  Task* t = scinew Task("AMRICE::errorEstimate", 
                  this, &AMRICE::errorEstimate, false);  
  
  Ghost::GhostType  gac  = Ghost::AroundCells; 
  Task::DomainSpec oims = Task::OutOfDomain;  //outside of ice matlSet.
                  
  t->requires(Task::NewDW, lb->rho_CCLabel,       gac, 1);
  t->requires(Task::NewDW, lb->temp_CCLabel,      gac, 1);
  t->requires(Task::NewDW, lb->vel_CCLabel,       gac, 1);
  t->requires(Task::NewDW, lb->vol_frac_CCLabel,  gac, 1);
  t->requires(Task::NewDW, lb->press_CCLabel,    d_press_matl,oims,gac, 1);
  
  t->computes(lb->mag_grad_rho_CCLabel);
  t->computes(lb->mag_grad_temp_CCLabel);
  t->computes(lb->mag_div_vel_CCLabel);
  t->computes(lb->mag_grad_vol_frac_CCLabel);
  t->computes(lb->mag_grad_press_CCLabel);
  
  t->modifies(d_sharedState->get_refineFlag_label(),      d_sharedState->refineFlagMaterials(), oims);
  t->modifies(d_sharedState->get_refinePatchFlag_label(), d_sharedState->refineFlagMaterials(), oims);
  
  sched->addTask(t, coarseLevel->eachPatch(), d_sharedState->allICEMaterials());
  
  //__________________________________
  // Models
  for(vector<ModelInterface*>::iterator iter = d_models.begin();
     iter != d_models.end(); iter++){
    (*iter)->scheduleErrorEstimate(coarseLevel, sched);;
  }
}
/*_____________________________________________________________________ 
Function~  AMRICE::compute_Mag_gradient--
Purpose~   computes the magnitude of the gradient/divergence of q_CC.
           First order central difference.
______________________________________________________________________*/
void AMRICE::compute_Mag_gradient( constCCVariable<double>& q_CC,
                                    CCVariable<double>& mag_grad_q_CC,
                                    const Patch* patch) 
{                  
  Vector dx = patch->dCell(); 
  for(CellIterator iter = patch->getCellIterator();!iter.done();iter++){
    IntVector c = *iter;
    IntVector r = c;
    IntVector l = c;
    Vector grad_q_CC;
    for(int dir = 0; dir <3; dir ++ ) { 
      double inv_dx = 0.5 /dx[dir];
      r[dir] += 1;
      l[dir] -= 1;
      grad_q_CC[dir] = (q_CC[r] - q_CC[l])*inv_dx;
    }
    mag_grad_q_CC[c] = grad_q_CC.length();
  }
}
//______________________________________________________________________
//          vector version
void AMRICE::compute_Mag_Divergence( constCCVariable<Vector>& q_CC,
                                    CCVariable<double>& mag_div_q_CC,
                                    const Patch* patch) 
{                  
  Vector dx = patch->dCell(); 
  

  for(CellIterator iter = patch->getCellIterator();!iter.done();iter++){
    IntVector c = *iter;
    IntVector r = c;
    IntVector l = c;
    Vector Divergence_q_CC;
    for(int dir = 0; dir <3; dir ++ ) { 
      double inv_dx = 0.5 /dx[dir];
      r[dir] += 1;
      l[dir] -= 1;
      Divergence_q_CC[dir]=(q_CC[r][dir] - q_CC[l][dir])*inv_dx;
    }
    mag_div_q_CC[c] = Divergence_q_CC.length();
  }
}
/*_____________________________________________________________________
 Function~  AMRICE::set_refinementFlags
______________________________________________________________________*/         
void AMRICE::set_refineFlags( CCVariable<double>& mag_grad_q_CC,
                              double threshold,
                              CCVariable<int>& refineFlag,
                              PerPatch<PatchFlagP>& refinePatchFlag,
                              const Patch* patch) 
{                  
  PatchFlag* refinePatch = refinePatchFlag.get().get_rep();
  for(CellIterator iter = patch->getCellIterator();!iter.done();iter++){
    IntVector c = *iter;
    if( mag_grad_q_CC[c] > threshold){
      refineFlag[c] = true;
      refinePatch->set();
    }
  }
}
/*_____________________________________________________________________
 Function~  AMRICE::errorEstimate--
______________________________________________________________________*/
void
AMRICE::errorEstimate(const ProcessorGroup*,
                      const PatchSubset* patches,
                      const MaterialSubset* /*matls*/,
                      DataWarehouse*,
                      DataWarehouse* new_dw,
                      bool /*initial*/)
{
  const Level* level = getLevel(patches);
  cout_doing << d_myworld->myrank() 
             << " Doing errorEstimate \t\t\t\t\t AMRICE L-"<< level->getIndex();
             
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    
    cout_doing << " patch " << patch->getID()<< endl;
    Ghost::GhostType  gac  = Ghost::AroundCells;
    const VarLabel* refineFlagLabel = d_sharedState->get_refineFlag_label();
    const VarLabel* refinePatchLabel= d_sharedState->get_refinePatchFlag_label();
    
    CCVariable<int> refineFlag;
    new_dw->getModifiable(refineFlag, refineFlagLabel, 0, patch);      

    PerPatch<PatchFlagP> refinePatchFlag;
    new_dw->get(refinePatchFlag, refinePatchLabel, 0, patch);


    //__________________________________
    //  PRESSURE      
    constCCVariable<double> press_CC;
    CCVariable<double> mag_grad_press_CC;
    
    new_dw->get(press_CC, lb->press_CCLabel, 0,patch,gac,1);
    new_dw->allocateAndPut(mag_grad_press_CC,
                       lb->mag_grad_press_CCLabel,  0,patch);
    mag_grad_press_CC.initialize(0.0);
    
    compute_Mag_gradient(press_CC, mag_grad_press_CC, patch);
    set_refineFlags( mag_grad_press_CC, d_press_threshold,refineFlag, 
                            refinePatchFlag, patch);
    //__________________________________
    //  RHO, TEMP, VEL_CC, VOL_FRAC
    int numICEMatls = d_sharedState->getNumICEMatls();
    for(int m=0;m < numICEMatls;m++){
      Material* matl = d_sharedState->getICEMaterial( m );
      int indx = matl->getDWIndex();
              
      constCCVariable<double> rho_CC, temp_CC, vol_frac_CC;
      constCCVariable<Vector> vel_CC;
      CCVariable<double> mag_grad_rho_CC, mag_grad_temp_CC, mag_grad_vol_frac_CC;
      CCVariable<double> mag_div_vel_CC;
      
      new_dw->get(rho_CC,      lb->rho_CCLabel,      indx,patch,gac,1);
      new_dw->get(temp_CC,     lb->temp_CCLabel,     indx,patch,gac,1);
      new_dw->get(vel_CC,      lb->vel_CCLabel,      indx,patch,gac,1);
      new_dw->get(vol_frac_CC, lb->vol_frac_CCLabel, indx,patch,gac,1);

      new_dw->allocateAndPut(mag_grad_rho_CC,     
                         lb->mag_grad_rho_CCLabel,     indx,patch);
      new_dw->allocateAndPut(mag_grad_temp_CC,    
                         lb->mag_grad_temp_CCLabel,    indx,patch);
      new_dw->allocateAndPut(mag_div_vel_CC, 
                         lb->mag_div_vel_CCLabel,      indx,patch);
      new_dw->allocateAndPut(mag_grad_vol_frac_CC,
                         lb->mag_grad_vol_frac_CCLabel,indx,patch);
                         
      mag_grad_rho_CC.initialize(0.0);
      mag_grad_temp_CC.initialize(0.0);
      mag_div_vel_CC.initialize(0.0);
      mag_grad_vol_frac_CC.initialize(0.0);
      
      //__________________________________
      // compute the gradients and set the refinement flags
                                        // Density
      compute_Mag_gradient(rho_CC,      mag_grad_rho_CC,      patch); 
      set_refineFlags( mag_grad_rho_CC, d_rho_threshold,refineFlag, 
                            refinePatchFlag, patch);
      
                                        // Temperature
      compute_Mag_gradient(temp_CC,      mag_grad_temp_CC,     patch); 
      set_refineFlags( mag_grad_temp_CC, d_temp_threshold,refineFlag, 
                            refinePatchFlag, patch);
      
                                        // Vol Fraction
      compute_Mag_gradient(vol_frac_CC,  mag_grad_vol_frac_CC, patch); 
      set_refineFlags( mag_grad_vol_frac_CC, d_vol_frac_threshold,refineFlag, 
                            refinePatchFlag, patch);
      
                                        // Velocity
      compute_Mag_Divergence(vel_CC,      mag_div_vel_CC,  patch); 
      set_refineFlags( mag_div_vel_CC, d_vel_threshold,refineFlag, 
                            refinePatchFlag, patch);
    }  // matls
    
    //______________________________________________________________________
    //    Hardcoding to move the error flags around every nTimeSteps
    if(d_regridderTest){
      double nTimeSteps = 5;  // how ofter to move the error flags
      const Level* level = getLevel(patches);
      int dw = d_sharedState->getCurrentTopLevelTimeStep();
      double timeToMove = fmod((double)dw, nTimeSteps);
      static int counter = 0;

      IntVector lo = patch->getInteriorCellLowIndex();
      IntVector hi = patch->getInteriorCellHighIndex() - IntVector(1,1,1);

      if (level->getIndex() == 0 ){
        //__________________________________
        // counter to move the error flag around
        if(timeToMove == 0){
          counter += 1;
          if (counter == 8) {
            counter = 0;
          }
        }
        //__________________________________
        //  find the 8 corner cells of level 0
        vector<IntVector> corners;

        for(int k = 0; k< 2; k++){
          for(int j = 0; j< 2; j++){
            for(int i = 0; i< 2; i++){
              int x = (i) * lo.x() + (1-i)*hi.x();
              int y = (j) * lo.y() + (1-j)*hi.y();
              int z = (k) * lo.z() + (1-k)*hi.z();
              corners.push_back(IntVector(x,y,z));     
            }
          }
        }
        if(timeToMove == 0){
          cout << "RegridderTest:  moving the error flag to "
               << corners[counter]<< " on level " 
               << level->getIndex() <<endl; 
        }
        //__________________________________
        //  Set the refinement flag      
        PatchFlag* refinePatch = refinePatchFlag.get().get_rep();
        refineFlag[corners[counter]] = true;
        refinePatch->set();
      }
      
      //__________________________________
      //  Levels other than 0
      if(level->getIndex() && timeToMove == 0){
        Vector randNum(drand48());
        
        IntVector diff = hi - lo;
        Vector here = randNum * diff.asVector();
        int i = RoundUp(here.x());
        int j = RoundUp(here.y());
        int k = RoundUp(here.z());
        
        IntVector twk(i,j,k);
        IntVector c = twk + lo;
        
        c = Max(c, lo+IntVector(2,2,2));
        c = Min(c, hi-IntVector(2,2,2));
        
        cout << "RegridderTest:  moving the error flag to "
               << c << " on level " 
               << level->getIndex() <<endl;
    
        //  Set the refinement flag      
        PatchFlag* refinePatch = refinePatchFlag.get().get_rep();
        refineFlag[c] = true;
        refinePatch->set();        
      }
    }  // regridderTest
    
  }  // patches
}
