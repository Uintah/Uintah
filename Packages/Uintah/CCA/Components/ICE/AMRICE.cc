
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
#include <Packages/Uintah/Core/Math/Rand48.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Math/MiscMath.h>
#include <Core/Util/DebugStream.h>
#include <iomanip>

#define SPEW 0
//#undef SPEW


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
  ProblemSpecP ice_ps;
  ProblemSpecP amr_ps = params->findBlock("AMR");
  if (amr_ps)
    ice_ps = amr_ps->findBlock("ICE");
  if(!ice_ps){
    string warn;
    warn ="\n INPUT FILE ERROR:\n <ICE>  block not found inside of <AMR> block \n";
    throw ProblemSetupException(warn, __FILE__, __LINE__);
    
  }
  ProblemSpecP refine_ps = ice_ps->findBlock("Refinement_Criteria_Thresholds");
  if(!refine_ps ){
    string warn;
    warn ="\n INPUT FILE ERROR:\n <Refinement_Criteria_Thresholds> "
         " block not found inside of <ICE> block \n";
    throw ProblemSetupException(warn, __FILE__, __LINE__);
  }
  ice_ps->require( "orderOfInterpolation", d_orderOfInterpolation);
  ice_ps->getWithDefault( "regridderTest", d_regridderTest,     false);
  ice_ps->getWithDefault( "do_Refluxing",  d_doRefluxing,       true);
  ice_ps->getWithDefault( "useLockStep",   d_useLockStep,       false);
  
  //__________________________________
  // Pull out the refinement threshold criteria 
  for (ProblemSpecP var_ps = refine_ps->findBlock("Variable");var_ps != 0; 
                    var_ps = var_ps->findNextBlock("Variable")) {
    thresholdVar data; 
    string name, value, matl;
        
    map<string,string> input;
    var_ps->getAttributes(input);
    name  = input["name"];
    value = input["value"];
    matl  = input["matl"];
    
    stringstream n_ss(name);
    stringstream v_ss(value);
    stringstream m_ss(matl);
    
    n_ss >> data.name;
    v_ss >> data.value;
    m_ss >> data.matl;
    
    int numMatls = d_sharedState->getNumMatls();

    //__________________________________
    //  bulletproofing    
    VarLabel* label = VarLabel::find(name);
    
    if(label == NULL){
      throw ProblemSetupException("The threshold variable name("+name+") could not be found",
                                   __FILE__, __LINE__);
    }

    if( data.name != "rho_CC"      && data.name != "temp_CC" && 
        data.name != "vol_frac_CC" && data.name != "vel_CC" &&
        data.name != "press_CC"){
      ostringstream warn;
      warn <<"\n INPUT FILE ERROR:\n The threshold variable name ("<< name <<") is not valid\n";
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }

    if( data.value < 0){
      ostringstream warn;
      warn <<"\n INPUT FILE ERROR:\n The threshold value ("<< value <<") cannot be negative\n";
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }
    
    if( (data.matl < 0 || data.matl > numMatls) && matl != "all"){
      ostringstream warn;
      warn <<"\n INPUT FILE ERROR:\n The threshold material ("<< matl <<") is not valid\n"
           << " select any material < total number of materials or 'all'";
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }
    if( data.name == "Pressure"){  // ignore what the user input, it's always 0
     data.matl = 0;
    }
    
    //__________________________________
    // if using "all" matls 
    if(matl == "all"){
      for (int m = 0; m < numMatls; m++){
        data.matl = m;
        d_thresholdVars.push_back(data);
      }
      
    }else{
      d_thresholdVars.push_back(data);
    }
  }
  
  //__________________________________
  // More bullet proofing
  int maxLevel = grid->numLevels();
  
  for (int i=0; i< maxLevel; i++){
     double trr = d_sharedState->timeRefinementRatio();

    if( d_useLockStep && trr != 1){
      string warn;
      warn ="\n INPUT FILE ERROR:\n To use the lockstep algorithm you must specify \n<AMR> \n  <time_refinement_ratio> 1 </time_refinement_ratio> \n</AMR>";
      throw ProblemSetupException(warn, __FILE__, __LINE__);
    }
  }
  
  //__________________________________
  //manual manipulate the scheduling of copy data
  Scheduler* sched = dynamic_cast<Scheduler*>(getPort("scheduler"));

  // we need these for AMRICE::refine
  sched->scheduleDataCopyVar("specific_heat");
  sched->scheduleDataCopyVar("gamma");
  sched->scheduleDataCopyVar("vol_frac_CC");
  sched->scheduleDataCopyVar("sp_vol_CC");
  sched->scheduleDataCopyVar("temp_CC");

  //We need these variables from OldDW to use between tasks, but do not
  // schedule datacopy
  
  sched->scheduleNotDataCopyVar("mass_X_FC_flux");
  sched->scheduleNotDataCopyVar("mass_Y_FC_flux");
  sched->scheduleNotDataCopyVar("mass_Z_FC_flux");
  sched->scheduleNotDataCopyVar("mom_X_FC_flux");
  sched->scheduleNotDataCopyVar("mom_Y_FC_flux");
  sched->scheduleNotDataCopyVar("mom_Z_FC_flux");
  sched->scheduleNotDataCopyVar("sp_vol_X_FC_flux");
  sched->scheduleNotDataCopyVar("sp_vol_Y_FC_flux");
  sched->scheduleNotDataCopyVar("sp_vol_Z_FC_flux");
  sched->scheduleNotDataCopyVar("int_eng_X_FC_flux");
  sched->scheduleNotDataCopyVar("int_eng_Y_FC_flux");
  sched->scheduleNotDataCopyVar("int_eng_Z_FC_flux");
  sched->scheduleNotDataCopyVar("vol_frac_X_FC_flux");
  sched->scheduleNotDataCopyVar("vol_frac_Y_FC_flux");
  sched->scheduleNotDataCopyVar("vol_frac_Z_FC_flux");
  
  //__________________________________
  // MODELS
  vector<AMR_refluxVariable*>::iterator iter;
  for( iter  = d_modelSetup->d_reflux_vars.begin();
       iter != d_modelSetup->d_reflux_vars.end(); iter++){
    AMR_refluxVariable* rvar = *iter;
    string varLabelX = rvar->var_X_FC_flux->getName();
    string varLabelY = rvar->var_Y_FC_flux->getName();
    string varLabelZ = rvar->var_Z_FC_flux->getName();
    sched->scheduleNotDataCopyVar(varLabelX);
    sched->scheduleNotDataCopyVar(varLabelY);
    sched->scheduleNotDataCopyVar(varLabelZ);
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
                                   bool needCoarseOld,
                                   bool needCoarseNew)
{
  cout_dbg << d_myworld->myrank() << " \t addRefineDependencies (" << var->getName()
           << ")"<< endl;
  
  Ghost::GhostType  gac = Ghost::AroundCells;
  bool  fat = true;  // data possibly needed (F)rom (A)nother (T)askgraph
  
  if(needCoarseOld) {
    cout_dbg << " requires from CoarseOldDW ";
    task->requires(Task::CoarseOldDW, var, 0, 
                   Task::CoarseLevel, matls, DS, gac, 1);
  }
  if(needCoarseNew) {
    cout_dbg << " requires from CoarseNewDW ";
    task->requires(Task::CoarseNewDW, var, 0, 
                   Task::CoarseLevel, matls, DS, gac, 1, fat);
  }

  task->modifies(var, matls, DS, fat);

  cout_dbg <<""<<endl;
}
/*___________________________________________________________________
 Function~  AMRICE::scheduleRefineInterface--
 Purpose:  
_____________________________________________________________________*/
void AMRICE::scheduleRefineInterface(const LevelP& fineLevel,
                                     SchedulerP& sched,
                                     bool needCoarseOld, 
                                     bool needCoarseNew)
{
  if(fineLevel->getIndex() > 0 ){
    cout_doing << d_myworld->myrank() << " AMRICE::scheduleRefineInterface \t\t\tL-" 
               << fineLevel->getIndex() 
               << " coarseOld: " << needCoarseOld 
               << " coarseNew: " << needCoarseNew << endl;
               
    Task* task = scinew Task("AMRICE::refineCoarseFineInterface", 
                       this, &AMRICE::refineCoarseFineInterface);
  
  
    Task::DomainSpec ND   = Task::NormalDomain;
    Task::DomainSpec oims = Task::OutOfDomain;  //outside of ice matlSet.
    const MaterialSet* all_matls = d_sharedState->allMaterials();
    const MaterialSubset* all_matls_sub = all_matls->getUnion();
    const MaterialSubset* ice_matls = d_sharedState->allICEMaterials()->getUnion();

    addRefineDependencies(task, lb->press_CCLabel, oims,d_press_matl, needCoarseOld, needCoarseNew);
    addRefineDependencies(task, lb->rho_CCLabel,   ND,  ice_matls,    needCoarseOld, needCoarseNew);
    addRefineDependencies(task, lb->sp_vol_CCLabel,ND,  all_matls_sub,needCoarseOld, needCoarseNew);
    addRefineDependencies(task, lb->temp_CCLabel,  ND,  all_matls_sub,needCoarseOld, needCoarseNew);
    addRefineDependencies(task, lb->vel_CCLabel,   ND,  ice_matls,    needCoarseOld, needCoarseNew);
    
    //__________________________________
    // Model Variables.
    if(d_modelSetup && d_modelSetup->tvars.size() > 0){
      vector<TransportedVariable*>::iterator iter;

      for(iter = d_modelSetup->tvars.begin();
         iter != d_modelSetup->tvars.end(); iter++){
        TransportedVariable* tvar = *iter;
        addRefineDependencies(task, tvar->var,ND,  all_matls_sub, needCoarseOld, needCoarseNew);
      }
    }
    sched->addTask(task, fineLevel->eachPatch(), all_matls);
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
                                       DataWarehouse* fine_new_dw)
{
  double subCycleProgress = getSubCycleProgress(fine_new_dw);
  const Level* level = getLevel(patches);
  if(level->getIndex() > 0){     
    cout_doing << d_myworld->myrank() 
               << " Doing refineCoarseFineInterface"<< "\t\t\t AMRICE L-" 
               << level->getIndex() << " Patches: " << *patches << " progressVar " << subCycleProgress
               << endl;
    int  numMatls = d_sharedState->getNumMatls();
    bool dbg_onOff = cout_dbg.active();      // is cout_dbg switch on or off
      
    for(int p=0;p<patches->size();p++){
      const Patch* patch = patches->get(p);

      //__________________________________
      //pressure
      CCVariable<double> press_CC;
      fine_new_dw->getModifiable(press_CC, lb->press_CCLabel,  0,   patch);
      if(switchDebug_AMR_refineInterface){
        ostringstream desc;
        desc << "TOP_refineInterface_patch_"
               << patch->getID()<< " step " << subCycleProgress;
        printData(0, patch,   1, desc.str(), "press_CC",    press_CC);
      }
      refineCoarseFineBoundaries(patch, press_CC, fine_new_dw,
                                 lb->press_CCLabel,  0,   subCycleProgress);

      for (int m = 0; m < numMatls; m++) {
        Material* matl = d_sharedState->getMaterial(m);
        ICEMaterial* ice_matl;
        ice_matl = dynamic_cast<ICEMaterial*>(matl);
        int indx = matl->getDWIndex();    
        CCVariable<double> rho_CC, sp_vol_CC, temp_CC;
        CCVariable<Vector> vel_CC;

        if(ice_matl){
          fine_new_dw->getModifiable(rho_CC,   lb->rho_CCLabel,    indx,patch);
          fine_new_dw->getModifiable(vel_CC,   lb->vel_CCLabel,    indx,patch);
        }
        fine_new_dw->getModifiable(temp_CC,  lb->temp_CCLabel,   indx,patch);
        fine_new_dw->getModifiable(sp_vol_CC,lb->sp_vol_CCLabel, indx,patch);

        //__________________________________
        //  Print Data 
        if(switchDebug_AMR_refineInterface){
          ostringstream desc;     
          desc << "TOP_refineInterface_Mat_" << indx << "_patch_"
               << patch->getID()<< " step " << subCycleProgress;

          if(ice_matl){
            printVector(indx, patch, 1, desc.str(), "vel_CC", 0,   vel_CC);
            printData(indx, patch,   1, desc.str(), "rho_CC",      rho_CC);
          }
          printData(indx, patch,   1, desc.str(), "sp_vol_CC",   sp_vol_CC);
          printData(indx, patch,   1, desc.str(), "Temp_CC",     temp_CC);
        }

        refineCoarseFineBoundaries(patch, sp_vol_CC,fine_new_dw,
                                   lb->sp_vol_CCLabel, indx,subCycleProgress);

        refineCoarseFineBoundaries(patch, temp_CC,  fine_new_dw,
                                   lb->temp_CCLabel,   indx,subCycleProgress);
        if(ice_matl){
          refineCoarseFineBoundaries(patch, rho_CC,   fine_new_dw, 
                                     lb->rho_CCLabel,    indx,subCycleProgress);

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
                  printData(indx,patch,1,"TOP_refineInterface",Labelname,q_CC);
                }

                refineCoarseFineBoundaries(patch, q_CC, fine_new_dw,
                                           tvar->var,    indx,subCycleProgress);

                if(switchDebug_AMR_refineInterface){ 
                  string name = tvar->var->getName();
                  printData(indx, patch,1,"BOT_refineInterface",Labelname,q_CC);
                }
              } // if
            }  // for
          }   // if models
        }   // ice_matl

        //__________________________________
        //  Print Data 
        if(switchDebug_AMR_refineInterface){
          ostringstream desc;    
          desc << "BOT_refineInterface_Mat_" << indx << "_patch_"
               << patch->getID()<< " step " << subCycleProgress;
          if(ice_matl){
            printData(  indx, patch, 1, desc.str(), "rho_CC",    rho_CC);
            printVector(indx, patch, 1, desc.str(), "vel_CC", 0, vel_CC);
          }
          printData(indx, patch,   1, desc.str(), "sp_vol_CC", sp_vol_CC);
          printData(indx, patch,   1, desc.str(), "Temp_CC",   temp_CC);
        } // dbg
      }  // matls
    }   // patches
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
  
  if(L_indx > 0 ){
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
    
    const MaterialSubset* all_matls = d_sharedState->allMaterials()->getUnion();


    t->modifies(lb->press_CCLabel, d_press_matl, oims);     
    t->modifies(lb->rho_CCLabel);
    t->modifies(lb->sp_vol_CCLabel);
    t->modifies(lb->temp_CCLabel);
    t->modifies(lb->vel_CCLabel);

    // we really only do the ice matls, but we need to tell CopyData to do the right thing
    t->computes(lb->gammaLabel, all_matls, oims);
    t->computes(lb->specific_heatLabel, all_matls, oims);
    t->computes(lb->vol_frac_CCLabel, all_matls, oims);
    
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
    sched->addTask(t, patches, d_sharedState->allICEMaterials());
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
               
    int  numICEMatls = d_sharedState->getNumICEMatls();
    bool dbg_onOff = cout_dbg.active();      // is cout_dbg switch on or off
      
    for(int p=0;p<patches->size();p++){
      const Patch* patch = patches->get(p);
      StaticArray<CCVariable<double> > sp_vol_CC(numICEMatls);
      StaticArray<constCCVariable<double> > sp_vol_const(numICEMatls);
      
      
      for (int m = 0; m < numICEMatls; m++) {
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
  
  if(L_indx > 0 ){
    
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
  Ghost::GhostType  gn = Ghost::None; 
  cout_doing << d_myworld->myrank() 
             << " AMRICE::scheduleCoarsen\t\t\t\tL-" 
             << fineLevel->getIndex()<< "->"<<coarseLevel->getIndex()<<endl; 
             
  Task* task = scinew Task("AMRICE::coarsen",this, &AMRICE::coarsen);

  Task::DomainSpec ND   = Task::NormalDomain;                            
  Task::DomainSpec oims = Task::OutOfDomain;  //outside of ice matlSet.  
  const MaterialSet* all_matls = d_sharedState->allMaterials();
  const MaterialSet* ice_matls = d_sharedState->allICEMaterials();
          
  const MaterialSubset* all_matls_sub = all_matls->getUnion();
  const PatchSet* patch_set = coarseLevel->eachPatch();
       
  bool  fat = true;  // possibly (F)rom (A)nother (T)askgraph
  
  task->requires(Task::NewDW, lb->press_CCLabel,
               0, Task::FineLevel,  d_press_matl,oims, gn, 0, fat);

  task->requires(Task::NewDW, lb->mass_advLabel,
               0, Task::FineLevel,  all_matls_sub,ND, gn, 0, fat);
  
  task->requires(Task::NewDW, lb->sp_vol_advLabel,
               0, Task::FineLevel,  all_matls_sub,ND, gn, 0, fat);
  
  task->requires(Task::NewDW, lb->eng_advLabel,
               0, Task::FineLevel,  all_matls_sub,ND, gn, 0, fat);
  
  task->requires(Task::NewDW, lb->mom_advLabel,
               0, Task::FineLevel,  all_matls_sub,ND, gn, 0, fat);

  //__________________________________
  // Model Variables.
  if(d_modelSetup && d_modelSetup->tvars.size() > 0){
    vector<TransportedVariable*>::iterator iter;

    for(iter = d_modelSetup->tvars.begin();
       iter != d_modelSetup->tvars.end(); iter++){
      TransportedVariable* tvar = *iter;
      task->requires(Task::NewDW, tvar->var_adv,
                  0, Task::FineLevel,all_matls_sub,ND, gn, 0, fat);
      task->modifies(tvar->var_adv, fat);
    }
  }
  
  task->modifies(lb->press_CCLabel, d_press_matl, oims, fat);
  task->modifies(lb->mass_advLabel,   fat);
  task->modifies(lb->sp_vol_advLabel, fat);
  task->modifies(lb->eng_advLabel,    fat);
  task->modifies(lb->mom_advLabel,    fat);

  sched->addTask(task, patch_set, ice_matls); 
  
  //__________________________________
  // schedule refluxing and bulletproofing
  // the bulletproofing tasks have no computes or requires
  if(d_doRefluxing){
    Task* t;
    Task* t1;
                              
    scheduleReflux_computeCorrectionFluxes(coarseLevel, sched); // compute correction
            
    //__________________________________
    //  initialize the bullet proofing flags
    t = scinew Task("AMRICE::reflux_BP_zero_CFI_cells",this, 
                    &AMRICE::reflux_BP_zero_CFI_cells);
    sched->addTask(t, coarseLevel->eachPatch(), ice_matls);
    
    scheduleReflux_applyCorrection(coarseLevel, sched);       // apply correction
    
    //__________________________________
    // check the bullet proofing flags    
    t = scinew Task("AMRICE::reflux_BP_count_CFI_cells",this, 
                    &AMRICE::reflux_BP_count_CFI_cells);
    
    string desc2 = "applyRefluxCorrection";
    t1 = scinew Task("AMRICE::reflux_BP_check_CFI_cells",this,
                    &AMRICE::reflux_BP_check_CFI_cells, desc2);
                    
    sched->addTask(t,  patch_set, ice_matls);
    sched->addTask(t1, patch_set, ice_matls);
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
  
  for(int p=0;p<patches->size();p++){  
    const Patch* coarsePatch = patches->get(p);
    cout_doing <<"  patch " << coarsePatch->getID()<< endl;
    
    for(int m = 0;m<matls->size();m++){
      int indx = matls->get(m);

      CCVariable<double> mass_adv, eng_adv, sp_vol_adv;
      CCVariable<Vector> mom_adv;

      new_dw->getModifiable(mass_adv,  lb->mass_advLabel,   indx, coarsePatch);
      new_dw->getModifiable(sp_vol_adv,lb->sp_vol_advLabel, indx, coarsePatch);
      new_dw->getModifiable(eng_adv,   lb->eng_advLabel,    indx, coarsePatch);
      new_dw->getModifiable(mom_adv,   lb->mom_advLabel,    indx, coarsePatch);  
      
      // coarsen         
      fineToCoarseOperator<double>(mass_adv,   "conserved", 
                         lb->mass_advLabel,   indx, new_dw, 
                         coarsePatch, coarseLevel, fineLevel);      

      fineToCoarseOperator<double>(sp_vol_adv, "conserved",
                         lb->sp_vol_advLabel, indx, new_dw, 
                         coarsePatch, coarseLevel, fineLevel);

      fineToCoarseOperator<double>(eng_adv,   "conserved",   
                         lb->eng_advLabel,    indx, new_dw, 
                         coarsePatch, coarseLevel, fineLevel);
       
      fineToCoarseOperator<Vector>( mom_adv,  "conserved",   
                         lb->mom_advLabel,    indx, new_dw, 
                         coarsePatch, coarseLevel, fineLevel);
      
      //__________________________________
      // pressure
      if( indx == 0){
          // pressure
        CCVariable<double> press_CC;                  
        new_dw->getModifiable(press_CC, lb->press_CCLabel,  0,    coarsePatch);
        fineToCoarseOperator<double>(press_CC, "non-conserved",
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
            CCVariable<double> q_CC_adv;
            new_dw->getModifiable(q_CC_adv, tvar->var_adv, indx, coarsePatch);
            fineToCoarseOperator<double>(q_CC_adv, "conserved", 
                       tvar->var_adv, indx, new_dw, 
                       coarsePatch, coarseLevel, fineLevel);
            
            if(switchDebug_AMR_coarsen){  
              string name = tvar->var->getName();
              printData(indx, coarsePatch, 1, "coarsen_models", name, q_CC_adv);
            }                 
          }
        }
      }    

      //__________________________________
      //  Print Data 
      if(switchDebug_AMR_coarsen){
        ostringstream desc;     
        desc << "BOT_coarsen_Mat_" << indx << "_patch_"<< coarsePatch->getID();
       // printData(indx, coarsePatch,   1, desc.str(), "press_CC",  press_CC);
        printData(indx, coarsePatch,   1, desc.str(), "mass_adv",    mass_adv);
        printData(indx, coarsePatch,   1, desc.str(), "sp_vol_adv",  sp_vol_adv);
        printData(indx, coarsePatch,   1, desc.str(), "eng_adv",     eng_adv);
        printVector(indx, coarsePatch, 1, desc.str(), "mom_adv", 0,  mom_adv);
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
  Ghost::GhostType gx  = Ghost::AroundFacesX;
  Ghost::GhostType gy  = Ghost::AroundFacesY;
  Ghost::GhostType gz  = Ghost::AroundFacesZ;
  bool  fat = true;  // possibly (F)rom (A)nother (T)askgraph
  
  
  task->requires(Task::NewDW, lb->rho_CCLabel,        gac, 1);
  task->requires(Task::NewDW, lb->specific_heatLabel, gac, 1, fat);
  
  //__________________________________
  // Fluxes from the fine level            
                                      // MASS
  task->requires(Task::NewDW, lb->mass_X_FC_fluxLabel,
               0,Task::FineLevel, 0, Task::NormalDomain, gx, 1, fat);
  task->requires(Task::NewDW, lb->mass_Y_FC_fluxLabel,
               0,Task::FineLevel, 0, Task::NormalDomain, gy, 1, fat);
  task->requires(Task::NewDW, lb->mass_Z_FC_fluxLabel,
               0,Task::FineLevel, 0, Task::NormalDomain, gz, 1, fat);
                                      // MOMENTUM
  task->requires(Task::NewDW, lb->mom_X_FC_fluxLabel,
               0,Task::FineLevel, 0, Task::NormalDomain, gx, 1, fat);
  task->requires(Task::NewDW, lb->mom_Y_FC_fluxLabel,
               0,Task::FineLevel, 0, Task::NormalDomain, gy, 1, fat);
  task->requires(Task::NewDW, lb->mom_Z_FC_fluxLabel,
               0,Task::FineLevel, 0, Task::NormalDomain, gz, 1, fat);
                                      // INT_ENG
  task->requires(Task::NewDW, lb->int_eng_X_FC_fluxLabel,
               0,Task::FineLevel, 0, Task::NormalDomain, gx, 1, fat);
  task->requires(Task::NewDW, lb->int_eng_Y_FC_fluxLabel,
               0,Task::FineLevel, 0, Task::NormalDomain, gy, 1, fat);
  task->requires(Task::NewDW, lb->int_eng_Z_FC_fluxLabel,
               0,Task::FineLevel, 0, Task::NormalDomain, gz, 1, fat);
                                      // SPECIFIC VOLUME
  task->requires(Task::NewDW, lb->sp_vol_X_FC_fluxLabel,
               0,Task::FineLevel, 0, Task::NormalDomain, gx, 1, fat);
  task->requires(Task::NewDW, lb->sp_vol_Y_FC_fluxLabel,
               0,Task::FineLevel, 0, Task::NormalDomain, gy, 1, fat);
  task->requires(Task::NewDW, lb->sp_vol_Z_FC_fluxLabel,
               0,Task::FineLevel, 0, Task::NormalDomain, gz, 1, fat);             

  //__________________________________
  // Model Variables.
  if(d_modelSetup && d_modelSetup->d_reflux_vars.size() > 0){
    vector<AMR_refluxVariable*>::iterator iter;
    for( iter  = d_modelSetup->d_reflux_vars.begin();
         iter != d_modelSetup->d_reflux_vars.end(); iter++){
      AMR_refluxVariable* rvar = *iter;
      
      task->requires(Task::NewDW, rvar->var_X_FC_flux,
                  0, Task::FineLevel, 0, Task::NormalDomain, gx, 1, fat);
      task->requires(Task::NewDW, rvar->var_Y_FC_flux,
                  0, Task::FineLevel, 0, Task::NormalDomain, gy, 1, fat);
      task->requires(Task::NewDW, rvar->var_Z_FC_flux,
                  0, Task::FineLevel, 0, Task::NormalDomain, gz, 1, fat);
                  
      task->modifies(rvar->var_X_FC_flux, fat);
      task->modifies(rvar->var_Y_FC_flux, fat);
      task->modifies(rvar->var_Z_FC_flux, fat);
    }
  }

  task->modifies(lb->mass_X_FC_fluxLabel, fat);
  task->modifies(lb->mass_Y_FC_fluxLabel, fat);
  task->modifies(lb->mass_Z_FC_fluxLabel, fat);
  
  task->modifies(lb->mom_X_FC_fluxLabel, fat);
  task->modifies(lb->mom_Y_FC_fluxLabel, fat);
  task->modifies(lb->mom_Z_FC_fluxLabel, fat);
  
  task->modifies(lb->int_eng_X_FC_fluxLabel, fat);
  task->modifies(lb->int_eng_Y_FC_fluxLabel, fat);
  task->modifies(lb->int_eng_Z_FC_fluxLabel, fat); 
  
  task->modifies(lb->sp_vol_X_FC_fluxLabel, fat);
  task->modifies(lb->sp_vol_Y_FC_fluxLabel, fat);
  task->modifies(lb->sp_vol_Z_FC_fluxLabel, fat);
  
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
  
  //__________________________________
  for(int c_p=0;c_p<coarsePatches->size();c_p++){  
    const Patch* coarsePatch = coarsePatches->get(c_p);
    
    cout_doing <<"  coarsePatches " << *coarsePatches << endl;
    
    for(int m = 0;m<matls->size();m++){
      int indx = matls->get(m);
      constCCVariable<double> cv, rho_CC;

      Ghost::GhostType  gac = Ghost::AroundCells;
      new_dw->get(rho_CC,lb->rho_CCLabel,       indx,coarsePatch, gac,1);
      new_dw->get(cv,    lb->specific_heatLabel,indx,coarsePatch, gac,1);
      
      Level::selectType finePatches;
      coarsePatch->getOtherLevelPatches(1, finePatches, 1); // get with a ghost cell to make sure you get all patches 
      
      
      
      for(int i=0; i < finePatches.size();i++){  
        const Patch* finePatch = finePatches[i]; 
        
        //__________________________________
        //   compute the correction
        // one_zero:  used to increment the CFI counter.
        if(finePatch->hasCoarseFineInterfaceFace() ){
          cout_doing << d_myworld->myrank()
                     << "  coarsePatch " << coarsePatch->getID()
                     <<" finepatch " << finePatch->getID()<< endl;

/*`==========TESTING==========*/
// May want to ripout rho_CC and cv from these calls 
/*===========TESTING==========`*/

          int one_zero = 1;
          refluxOperator_computeCorrectionFluxes<double>(rho_CC,  cv, "mass",   indx, 
                        coarsePatch, finePatch, coarseLevel, fineLevel,new_dw,
                        one_zero);
          
          one_zero = 0;
          refluxOperator_computeCorrectionFluxes<double>( rho_CC, cv, "sp_vol", indx, 
                        coarsePatch, finePatch, coarseLevel, fineLevel,new_dw,
                        one_zero);

          refluxOperator_computeCorrectionFluxes<Vector>( rho_CC, cv, "mom",    indx, 
                        coarsePatch, finePatch, coarseLevel, fineLevel,new_dw,
                        one_zero);

          refluxOperator_computeCorrectionFluxes<double>( rho_CC, cv, "int_eng", indx, 
                        coarsePatch, finePatch, coarseLevel, fineLevel,new_dw,
                        one_zero);
          //__________________________________
          //    Model Variables
          if(d_modelSetup && d_modelSetup->d_reflux_vars.size() > 0){
            vector<AMR_refluxVariable*>::iterator iter;
            for( iter  = d_modelSetup->d_reflux_vars.begin();
                 iter != d_modelSetup->d_reflux_vars.end(); iter++){
              AMR_refluxVariable* r_var = *iter;

              if(r_var->matls->contains(indx)){
                string var_name = r_var->var_adv->getName();
                refluxOperator_computeCorrectionFluxes<double>(rho_CC, cv, var_name, indx, 
                              coarsePatch, finePatch, coarseLevel, fineLevel,new_dw,
                              one_zero);
              }
            }
          }  // model
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
  }  // coarse patch loop 
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
                               bool& isRight_CP_FP_pair,
                               const string& whichTask)
{
  CellIterator f_iter=finePatch->getFaceCellIterator(patchFace, "alongInteriorFaceCells");
  
  ASSERT(whichTask == "computeRefluxCorrection" || whichTask == "applyRefluxCorrection" );

  // find the intersection of the fine patch face iterator and underlying coarse patch
  IntVector dir = finePatch->faceAxes(patchFace);        // face axes
  int p_dir = dir[0];                                    // normal direction 
  IntVector f_lo_face = f_iter.begin();                  // fineLevel face indices   
  IntVector f_hi_face = f_iter.end();

  IntVector l = fineLevel->mapCellToCoarser(f_lo_face);     
  IntVector h = fineLevel->mapCellToCoarser(f_hi_face);

  //__________________________________
  // Offset for the coarse level iterator
  // shift l & h,  1 cell for x+, y+, z+ finePatchfaces
  // shift l & h, -1 cell for x-, y-, z- finePatchfaces
  // 
  // if(minus face){
  //    if(refinement ratio == 1)
  //      shift h 
  //    else 
  //      no shift for h
  // }
  //
  // Offset for the coarse_face center (c_FC_offset)
  // shift 1 cell   for x-, y-, z- finePatch faces
  // shift 0 cells  for x+, y+, z+ finePatchFaces

  string name = finePatch->getFaceName(patchFace);
  IntVector offset = finePatch->faceDirection(patchFace);
  coarse_FC_offset = IntVector(0,0,0);

  if(name == "xminus" || name == "yminus" || name == "zminus"){
    coarse_FC_offset = -offset;
    l += offset;
    
    IntVector rr(fineLevel->getRefinementRatio());
    if(rr[p_dir] == 1){
      h += offset;
    }
  }
  if(name == "xplus" || name == "yplus" || name == "zplus"){
    l += offset;
    h += offset;
  }

  IntVector coarse_Lo = coarsePatch->getLowIndex(); 
  IntVector coarse_Hi = coarsePatch->getHighIndex();
  int y = dir[1];  // tangential directions
  int z = dir[2];  
  
/*`==========TESTING==========*/
#if 0
  if(finePatch->getID() == 583){
    cout << "\nrefluxCoarseLevelIterator " << name << " " << whichTask
         << "\n before      " << l << " " << h
         << "\n finePatch   " << *finePatch
         << "\n coarsePatch " << *coarsePatch
         << "\n coarseLevel " << coarsePatch->getLevel()->getIndex()
         << "\n coarse_Lo   " << coarse_Lo << " coarse_Hi " << coarse_Hi << endl;
  }
#endif
/*===========TESTING==========`*/  
  
  l[y] = Max(l[y], coarse_Lo[y]);  // intersection 
  l[z] = Max(l[z], coarse_Lo[z]);  // only the transerse directions 
  h[y] = Min(h[y], coarse_Hi[y]);
  h[z] = Min(h[z], coarse_Hi[z]);       
      
  iter=CellIterator(l,h);
  
  // Does this iterator exceed the boundaries of the underlying coarse patch?
  // If your computing/applying the reflux correction the conditional is different
  // To understand why you need paper a pencil.  Draw a fine patch over a coarse patch
  // and let the CFI coinside with the boundary  between 2 coarse level patches.
  
  IntVector one(1,1,1);            // subtract of 1 (h). 
  IntVector h_minusOne = h - one;  // h -1 is what we're truely interested in.
  
  isRight_CP_FP_pair = false;
  if ( whichTask == "computeRefluxCorrection"  &&
       coarsePatch->containsCell(l + coarse_FC_offset) && 
       coarsePatch->containsCell(h_minusOne + coarse_FC_offset) && 
       l[y] != h[y] && l[z] != h[z] ){
    isRight_CP_FP_pair = true;
  }
  if (whichTask == "applyRefluxCorrection" && 
       coarsePatch->containsCell(l) && 
       coarsePatch->containsCell(h_minusOne) &&
       l[y] != h[y] && l[z] != h[z] ){
    isRight_CP_FP_pair = true;
  }

  
 /*`==========TESTING==========*/
#if 0
  if(finePatch->getID() == 583){
    cout << " after " << l << " " << h 
         << " coarse_FC_offset " << coarse_FC_offset
         << " isRight_CP_FP_pair " << isRight_CP_FP_pair 
         << "\ncoarsePatch->containsCell(l)                             " << coarsePatch->containsCell(l)
         << "\ncoarsePatch->containsCell(h_minusOne)                    " << coarsePatch->containsCell(h_minusOne)
         << "\ncoarsePatch->containsCell(l + coarse_FC_offset)          "<< coarsePatch->containsCell(l + coarse_FC_offset)
         << "\ncoarsePatch->containsCell(h_minusOne + coarse_FC_offset) " << coarsePatch->containsCell(h_minusOne + coarse_FC_offset)
         << " \nl[y] != h[y] && l[z] != h[z]                            " << (l[y] != h[y] && l[z] != h[z])<< endl;
  }
#endif
/*===========TESTING==========`*/  

  
  //____ B U L L E T   P R O O F I N G----  
  if (isRight_CP_FP_pair ){
    IntVector diff = Abs(l - h);
    if( ( l.x() >= h.x() || l.y() >= h.y() || l.z() >= h.z() ) || diff[p_dir] > 1) {
      ostringstream warn;
      warn << "AMRICE:refluxCoarseLevelIterator : "<< l << " " << h << " "  << name
           << "\n  Error:Either l >= h OR l - h > 1"
           << "\n finelevel   " << fineLevel->getIndex()
           << "\n finePatch   " << *finePatch
           << "\n CoarsePatch " << *coarsePatch\
           << "\n coarseLo    " << coarse_Lo << " coarseHi " << coarse_Hi
           << "\n offset      " << offset
           << " unmodified Iterator " << f_iter.begin() << " " << f_iter.end();
      finePatch->printPatchBCs(warn);

      throw InternalError(warn.str(), __FILE__, __LINE__ ); 
    } 
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
  
  Ghost::GhostType gac = Ghost::AroundCells;
  
    
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
      task->modifies(rvar->var_adv);
    }
  }

  task->modifies(lb->mass_advLabel);
  task->modifies(lb->sp_vol_advLabel);
  task->modifies(lb->eng_advLabel);
  task->modifies(lb->mom_advLabel);

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
      CCVariable<double> mass_adv, eng_adv, sp_vol_adv;
      constCCVariable<double> cv;
      CCVariable<Vector> mom_adv;

      Ghost::GhostType  gn  = Ghost::None;
      new_dw->getModifiable(mass_adv,   lb->mass_advLabel,    indx, coarsePatch);
      new_dw->getModifiable(sp_vol_adv, lb->sp_vol_advLabel,  indx, coarsePatch);
      new_dw->getModifiable(eng_adv,    lb->eng_advLabel,     indx, coarsePatch);
      new_dw->getModifiable(mom_adv,    lb->mom_advLabel,     indx, coarsePatch);
      new_dw->get(cv,                   lb->specific_heatLabel,indx,coarsePatch, gn,0);
      
      Level::selectType finePatches;
      coarsePatch->getOtherLevelPatches(1, finePatches, 1); // get with a ghost cell to make sure you get all patches 
      
      for(int i=0; i < finePatches.size();i++){  
        const Patch* finePatch = finePatches[i];        
        //cout_doing << d_myworld->myrank() << "  coarsePatch " << coarsePatch->getID() <<" finepatch " << finePatch->getID()<< endl;
        //__________________________________
        // Apply the correction
        // one_zero:  used to increment the CFI counter.
        if(finePatch->hasCoarseFineInterfaceFace() ){
          int one_zero = 1;
          refluxOperator_applyCorrectionFluxes<double>(mass_adv,   "mass",    indx, 
                        coarsePatch, finePatch, coarseLevel, fineLevel,new_dw,
                        one_zero);

          one_zero = 0;
          refluxOperator_applyCorrectionFluxes<double>(sp_vol_adv, "sp_vol",  indx, 
                        coarsePatch, finePatch, coarseLevel, fineLevel,new_dw,
                        one_zero);

          refluxOperator_applyCorrectionFluxes<Vector>(mom_adv,     "mom",    indx, 
                        coarsePatch, finePatch, coarseLevel, fineLevel,new_dw,
                        one_zero);

          refluxOperator_applyCorrectionFluxes<double>(eng_adv,     "int_eng",indx, 
                        coarsePatch, finePatch, coarseLevel, fineLevel,new_dw,
                        one_zero);
          //__________________________________
          //    Model Variables
          if(d_modelSetup && d_modelSetup->d_reflux_vars.size() > 0){
            vector<AMR_refluxVariable*>::iterator iter;
            for( iter  = d_modelSetup->d_reflux_vars.begin();
                 iter != d_modelSetup->d_reflux_vars.end(); iter++){
              AMR_refluxVariable* r_var = *iter;

              if(r_var->matls->contains(indx)){
                CCVariable<double> q_CC_adv;
                string var_name = r_var->var_adv->getName();
                new_dw->getModifiable(q_CC_adv,  r_var->var_adv, indx, coarsePatch);

                refluxOperator_applyCorrectionFluxes<double>(q_CC_adv, var_name, indx, 
                              coarsePatch, finePatch, coarseLevel, fineLevel,new_dw,
                              one_zero);

                if(switchDebug_AMR_refine){
                  printData(indx, coarsePatch, 1, "coarsen_models", var_name, q_CC_adv);
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
        printData(indx, coarsePatch,   0, desc.str(), "mass_adv",   mass_adv);
        printData(indx, coarsePatch,   0, desc.str(), "sp_vol_adv", sp_vol_adv);
        printData(indx, coarsePatch,   0, desc.str(), "eng_adv",    eng_adv);
        printVector(indx, coarsePatch, 0, desc.str(), "mom_adv", 0, mom_adv);
      }
    }  // matl loop
  }  // course patch loop
  
  cout_dbg.setActive(dbg_onOff);  // reset on/off switch for cout_dbg
}

/*_____________________________________________________________________
 Function~  AMRICE::reflux_BP_zero_CFI_cells--(bulletproofing)
 Purpose~   Initialze coarse fine interface "marks"
______________________________________________________________________*/
void AMRICE::reflux_BP_zero_CFI_cells(const ProcessorGroup*,
                                      const PatchSubset* coarsePatches,
                                      const MaterialSubset*,
                                       DataWarehouse*,
                                       DataWarehouse*)
{
  const Level* coarseLevel = getLevel(coarsePatches);
  const Level* fineLevel = coarseLevel->getFinerLevel().get_rep();
  
  for(int c_p=0;c_p<coarsePatches->size();c_p++){  
    const Patch* coarsePatch = coarsePatches->get(c_p);
    
    Level::selectType finePatches;
    coarsePatch->getOtherLevelPatches(1, finePatches, 1);

    cout_doing << d_myworld->myrank() 
             << " Doing reflux_BP_zero_CFI_cells \t\t\t AMRICE L-"
             <<fineLevel->getIndex()<< "->"<< coarseLevel->getIndex()<<endl;
             
    for(int p=0;p<finePatches.size();p++){  
      const Patch* finePatch = finePatches[p];
    

      vector<Patch::FaceType>::const_iterator iter;  
      for (iter  = finePatch->getCoarseFineInterfaceFaces()->begin(); 
           iter != finePatch->getCoarseFineInterfaceFaces()->end(); ++iter){
        Patch::FaceType patchFace = *iter;
          
        finePatch->setFaceMark(0, patchFace, 0);
        finePatch->setFaceMark(1, patchFace, 0);
      }
    }  // finePatch loop
  }  // coarsepatch loop
}


/*_____________________________________________________________________
 Function~  AMRICE::reflux_BP_count_CFI_cells--(bulletproofing)
 Purpose~   count up the number of coarse fine interface cells and save 
            that number
______________________________________________________________________*/
void AMRICE::reflux_BP_count_CFI_cells(const ProcessorGroup*,
                                       const PatchSubset* coarsePatches,
                                       const MaterialSubset*,
                                       DataWarehouse*,
                                       DataWarehouse*)
{
  const Level* coarseLevel = getLevel(coarsePatches);
  const Level* fineLevel = coarseLevel->getFinerLevel().get_rep();
  
  for(int c_p=0;c_p<coarsePatches->size();c_p++){  
    const Patch* coarsePatch = coarsePatches->get(c_p);
    
    Level::selectType finePatches;
    coarsePatch->getOtherLevelPatches(1, finePatches, 1);

    cout_doing << d_myworld->myrank() 
             << " Doing reflux_BP_count_CFI_cells \t\t\t AMRICE L-"
             <<fineLevel->getIndex()<< "->"<< coarseLevel->getIndex()<<endl;
             
    for(int p=0;p<finePatches.size();p++){  
      const Patch* finePatch = finePatches[p];
    

      vector<Patch::FaceType>::const_iterator iter;  
      for (iter  = finePatch->getCoarseFineInterfaceFaces()->begin(); 
           iter != finePatch->getCoarseFineInterfaceFaces()->end(); ++iter){
        Patch::FaceType patchFace = *iter;
            
        bool isRight_CP_FP_pair = false;
        CellIterator f_iter(IntVector(-8,-8,-8),IntVector(-9,-9,-9));
        fineLevel_CFI_Iterator(patchFace, coarsePatch, finePatch, f_iter, isRight_CP_FP_pair);
            
            
        if(isRight_CP_FP_pair){
        
          int n_CFI_cells = 0;
          int count = finePatch->getFaceMark(1,patchFace);
          for(; !f_iter.done(); f_iter++) {
            n_CFI_cells +=1;
          }        

          // divide the number of cells 
          IntVector rr= finePatch->getLevel()->getRefinementRatio();
          IntVector dir = finePatch->faceAxes(patchFace);
          int y = dir[1];
          int z = dir[2];
          count += n_CFI_cells/(rr[y] * rr[z]);
          
          finePatch->setFaceMark(1, patchFace, count);
        }  // right cp_fp_pair 
      }  // face loop
    }  // finePatch loop
  }  // coarsepatch loop
}

/*___________________________________________________________________
 Function~  AMRICE::reflux_BP_check_CFI_cells--  (bulletproofing)
 Purpose~   Check if each coarse fine interface cell was "touched"
            during refluxing 
_____________________________________________________________________*/
void AMRICE::reflux_BP_check_CFI_cells(const ProcessorGroup*,
                                       const PatchSubset* coarsePatches,
                                       const MaterialSubset*,
                                       DataWarehouse*,
                                       DataWarehouse* new_dw,
                                       string description)
{
  const Level* coarseLevel = getLevel(coarsePatches);
  const Level* fineLevel = coarseLevel->getFinerLevel().get_rep();
  
  for(int c_p=0;c_p<coarsePatches->size();c_p++){  
    const Patch* coarsePatch = coarsePatches->get(c_p);
    
    Level::selectType finePatches;
    coarsePatch->getOtherLevelPatches(1, finePatches, 1);

    cout_doing << d_myworld->myrank() 
             << " Doing reflux_BP_check_CFI_cells \t\t\t AMRICE L-"
             <<fineLevel->getIndex()<< "->"<< coarseLevel->getIndex() << endl;
             
    for(int p=0;p<finePatches.size();p++){  
      const Patch* finePatch = finePatches[p]; 
         
      if(finePatch->hasCoarseFineInterfaceFace() ){

        vector<Patch::FaceType>::const_iterator iter;  
        for (iter  = finePatch->getCoarseFineInterfaceFaces()->begin(); 
             iter != finePatch->getCoarseFineInterfaceFaces()->end(); ++iter){
          Patch::FaceType patchFace = *iter;
          
          //This makes sure that the processor that "touched" the cell is also
          // going to check it.  Each processor can have a different instance
          // of a patch.
          IntVector dummy;
          bool isRight_CP_FP_pair;
          CellIterator dummy_iter(IntVector(0,0,0),IntVector(0,0,0));
          refluxCoarseLevelIterator( patchFace,coarsePatch, finePatch, fineLevel,
                               dummy_iter, dummy, isRight_CP_FP_pair, description);
                               
          if (isRight_CP_FP_pair){                 
     
            bool isRight_CP_FP_pair = false;
            CellIterator f_iter(IntVector(-8,-8,-8),IntVector(-9,-9,-9));
            fineLevel_CFI_Iterator(patchFace, coarsePatch, finePatch, f_iter, isRight_CP_FP_pair);
            
            if(isRight_CP_FP_pair){

              int n_ice_matls = d_sharedState->getNumICEMatls();
              int n_touched_cells = (finePatch->getFaceMark(0, patchFace) )/n_ice_matls;
              int n_CFI_cells     =  finePatch->getFaceMark(1, patchFace);
              //__________________________________
              //  If the number of "marked" cells/numICEMatls != n_CFI_cells
              // ignore if a timestep restart has already been requested
              bool tsr = new_dw->timestepRestarted();
              
              if ( n_touched_cells != n_CFI_cells && !tsr){
                ostringstream warn;
                warn << d_myworld->myrank() << " AMRICE:refluxing_" << description
                     << " \n CFI face: "
                     << finePatch->getFaceName(patchFace)
                     << " cells were 'touched' "<< n_touched_cells << " times"
                     << " it should have been 'touched' " << n_CFI_cells << " times "
                     << "\n patch " << *finePatch 
                     << "\n finePatchLevel " << finePatch->getLevel()->getIndex()<< endl;
                //cout << warn.str() << endl;
                throw InternalError(warn.str(), __FILE__, __LINE__ );
              }
            }
          }
        }  // face iter
      }  // has CFI
    }  // //finePatches
  }  // coarsePatches
}



/*_____________________________________________________________________
 Function~  AMRICE::scheduleInitialErrorEstimate--
______________________________________________________________________*/
void AMRICE::scheduleInitialErrorEstimate(const LevelP& coarseLevel,
                                          SchedulerP& sched)
{
  scheduleErrorEstimate(coarseLevel, sched);
}

/*_____________________________________________________________________
 Function~  AMRICE::scheduleErrorEstimate--
______________________________________________________________________*/
void AMRICE::scheduleErrorEstimate(const LevelP& coarseLevel,
                                   SchedulerP& sched)
{  
  cout_doing << d_myworld->myrank() 
             << " AMRICE::scheduleErrorEstimate \t\t\tL-" 
             << coarseLevel->getIndex() << '\n';
  bool initial = false;             
  if(d_sharedState->getCurrentTopLevelTimeStep() == 0){
    initial = true;  // during initialization 
  }
  Task* t = scinew Task("AMRICE::errorEstimate", 
                  this, &AMRICE::errorEstimate, initial);  
  
  Ghost::GhostType  gac  = Ghost::AroundCells; 
  Task::DomainSpec oims = Task::OutOfDomain;  //outside of ice matlSet.
  bool  fat = true;  // possibly (F)rom (A)nother (T)askgraph
  

  const MaterialSet* ice_matls = d_sharedState->allICEMaterials();
  const MaterialSet* all_matls = d_sharedState->allMaterials();  
  
  const MaterialSubset* matls_sub;
  const MaterialSubset* ice_matls_sub = ice_matls->getUnion();
  const MaterialSubset* all_matls_sub = all_matls->getUnion();  
  
  // Only require ice_matls during initialization, we don't have *_CC variables for 
  // mpm_matls at this point  
  if(initial){
    matls_sub = ice_matls_sub;
  }else{
    matls_sub = all_matls_sub;
  }
  
                  
  t->requires(Task::NewDW, lb->rho_CCLabel,      matls_sub,  gac, 1, fat);
  t->requires(Task::NewDW, lb->temp_CCLabel,     matls_sub,  gac, 1, fat);
  t->requires(Task::NewDW, lb->vel_CCLabel,      matls_sub,  gac, 1, fat);
  t->requires(Task::NewDW, lb->vol_frac_CCLabel, matls_sub,  gac, 1, fat);
  t->requires(Task::NewDW, lb->press_CCLabel,    d_press_matl,oims,gac, 1, fat);
  
  t->computes(lb->mag_grad_rho_CCLabel);
  t->computes(lb->mag_grad_temp_CCLabel);
  t->computes(lb->mag_div_vel_CCLabel);
  t->computes(lb->mag_grad_vol_frac_CCLabel);
  t->computes(lb->mag_grad_press_CCLabel,d_press_matl);
  
  t->modifies(d_sharedState->get_refineFlag_label(),      d_sharedState->refineFlagMaterials(), oims);
  t->modifies(d_sharedState->get_refinePatchFlag_label(), d_sharedState->refineFlagMaterials(), oims);
  
  sched->addTask(t, coarseLevel->eachPatch(), d_sharedState->allMaterials());
  
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
    Vector grad_q_CC;
    for(int dir = 0; dir <3; dir ++ ) { 
      IntVector r = c;
      IntVector l = c;
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
    Vector Divergence_q_CC;
    for(int dir = 0; dir <3; dir ++ ) { 
      IntVector r = c;
      IntVector l = c;
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
void AMRICE::set_refineFlags( constCCVariable<double>& mag_grad_q_CC,
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
                      bool initial)
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

    //__________________________________
    //  initialize mag_grad for all matls
    int numAllMatls = d_sharedState->getNumMatls();
    StaticArray<CCVariable<double> > mag_grad_rho_CC(numAllMatls);
    StaticArray<CCVariable<double> > mag_grad_temp_CC(numAllMatls);
    StaticArray<CCVariable<double> > mag_grad_vol_frac_CC(numAllMatls);
    StaticArray<CCVariable<double> > mag_div_vel_CC(numAllMatls);
          
    for(int m=0;m < numAllMatls;m++){
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      new_dw->allocateAndPut(mag_grad_rho_CC[indx],     
                         lb->mag_grad_rho_CCLabel,     indx,patch);
      new_dw->allocateAndPut(mag_grad_temp_CC[indx],    
                         lb->mag_grad_temp_CCLabel,    indx,patch);
      new_dw->allocateAndPut(mag_div_vel_CC[indx], 
                         lb->mag_div_vel_CCLabel,      indx,patch);
      new_dw->allocateAndPut(mag_grad_vol_frac_CC[indx],
                         lb->mag_grad_vol_frac_CCLabel,indx,patch);
                         
      mag_grad_rho_CC[indx].initialize(0.0);
      mag_grad_temp_CC[indx].initialize(0.0);
      mag_div_vel_CC[indx].initialize(0.0);
      mag_grad_vol_frac_CC[indx].initialize(0.0);
    }  // matls


    //__________________________________
    //  RHO, TEMP, VEL_CC, VOL_FRAC
    // During initialization only compute for ICE matls
    int numMatls = 0;
    if (initial){
      numMatls = d_sharedState->getNumICEMatls();
    }else{
      numMatls = d_sharedState->getNumMatls();
    }
    
    for(int m=0;m < numMatls;m++){
      
      Material* matl;
      if(initial){
        matl = d_sharedState->getICEMaterial( m );
      }else{
        matl = d_sharedState->getMaterial( m );
      }
      
      int indx = matl->getDWIndex();        
      constCCVariable<double> rho_CC, temp_CC, vol_frac_CC;
      constCCVariable<Vector> vel_CC;
      
      new_dw->get(rho_CC,      lb->rho_CCLabel,      indx,patch,gac,1);
      new_dw->get(temp_CC,     lb->temp_CCLabel,     indx,patch,gac,1);
      new_dw->get(vel_CC,      lb->vel_CCLabel,      indx,patch,gac,1);
      new_dw->get(vol_frac_CC, lb->vol_frac_CCLabel, indx,patch,gac,1);
      
      //__________________________________
      // compute the magnitude of the gradient/divergence
      compute_Mag_gradient(rho_CC,       mag_grad_rho_CC[indx],      patch);
      
      compute_Mag_gradient(temp_CC,      mag_grad_temp_CC[indx],     patch); 
      
      compute_Mag_gradient(vol_frac_CC,  mag_grad_vol_frac_CC[indx], patch);
      
      compute_Mag_Divergence(vel_CC,     mag_div_vel_CC[indx],       patch);
    }  // matls

    //__________________________________
    // Only set the refinement flags for certain materials
    for(int i = 0; i< d_thresholdVars.size(); i++ ){
      thresholdVar data = d_thresholdVars[i];
      string name  = data.name;
      int matl     = data.matl;
      double thresholdValue = data.value;
      VarLabel* mag_grad_qLabel = VarLabel::find("mag_grad_"+name);
      
      if(mag_grad_qLabel==NULL){  // bulletproofing
        throw InternalError("AMRICE::errorEstimate: label(mag_grad_"+name+") not found.",
                            __FILE__, __LINE__);
      }
      constCCVariable<double> mag_grad_q_CC;
      new_dw->get(mag_grad_q_CC, mag_grad_qLabel, matl,patch,Ghost::None,0);
      
      set_refineFlags( mag_grad_q_CC, thresholdValue,refineFlag, refinePatchFlag, patch);
    }
  }  // patches
}
