
#include <Packages/Uintah/CCA/Components/ICE/AMRICE.h>
#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <Packages/Uintah/CCA/Components/Regridder/PerPatchVars.h>
#include <Packages/Uintah/CCA/Ports/SolverInterface.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <Packages/Uintah/Core/Grid/Variables/AMRInterpolate.h>
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

using namespace Uintah;
using namespace std;
static DebugStream cout_doing("AMRICE_DOING_COUT", false);
static DebugStream cout_dbg("AMRICE_DBG", false);
//setenv SCI_DEBUG AMR:+ you can see the new grid it creates


AMRICE::AMRICE(const ProcessorGroup* myworld)
  : ICE(myworld, true)
{
}

AMRICE::~AMRICE()
{
}
//___________________________________________________________________
void AMRICE::problemSetup(const ProblemSpecP& params, GridP& grid,
                            SimulationStateP& sharedState)
{
  cout_doing << d_myworld->myrank() 
             << " Doing problemSetup  \t\t\t AMRICE" << '\n';
             
  ICE::problemSetup(params, grid, sharedState);
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
  if(fineLevel->getIndex() > 0  && doICEOnLevel(fineLevel->getIndex())){
    cout_doing << d_myworld->myrank() << " AMRICE::scheduleRefineInterface \t\t\tL-" 
               << fineLevel->getIndex() << " progressVar "<< (double)step/(double)nsteps <<'\n';
               
    double subCycleProgress = double(step)/double(nsteps);
    
    Task* task = scinew Task("AMRICE::refineCoarseFineInterface", 
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
    
    task->modifies(lb->press_CCLabel, d_press_matl, oims);
    task->modifies(lb->rho_CCLabel);
    task->modifies(lb->sp_vol_CCLabel);
    task->modifies(lb->temp_CCLabel);
    task->modifies(lb->vel_CCLabel);

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
    
    // dummy variable needed to keep the taskgraph in sync
    //task->computes(lb->AMR_SyncTaskgraphLabel); 
    
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
                                       DataWarehouse* fine_new_dw,
                                       double subCycleProgress)
{
  const Level* level = getLevel(patches);
  if(level->getIndex() > 0){     
    cout_doing << d_myworld->myrank() 
               << " Doing refineCoarseFineInterface"<< "\t\t\t AMRICE L-" 
               << level->getIndex() << " progressVar " << subCycleProgress<<endl;
    int  numMatls = d_sharedState->getNumICEMatls();
    bool dbg_onOff = cout_dbg.active();      // is cout_dbg switch on or off
      
    for(int p=0;p<patches->size();p++){
      const Patch* patch = patches->get(p);
      
      // dummy variable needed to keep the taskgraph in sync
      //CCVariable<int> dummy;
      //fine_new_dw->allocateAndPut(dummy, lb->AMR_SyncTaskgraphLabel,0, patch);
      
      
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
              CCVariable<double> q_CC;
              fine_new_dw->getModifiable(q_CC, tvar->var, indx, patch);
              
              if(switchDebug_AMR_refineInterface){ 
                string name = tvar->var->getName();
                printData(indx, patch, 1, "TOP_refineInterface", name, q_CC);
              }              
              
              refineCoarseFineBoundaries(patch, q_CC, fine_new_dw,
                                          tvar->var,    indx,subCycleProgress);
              
              if(switchDebug_AMR_refineInterface){ 
                string name = tvar->var->getName();
                printData(indx, patch, 1, "BOT_refineInterface", name, q_CC);
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
 Function~  AMRICE::refine_CF_interfaceOperator-- 
_____________________________________________________________________*/
template<class varType>
void AMRICE::refine_CF_interfaceOperator(const Patch* patch, 
                                 const Level* fineLevel,
                                 const Level* coarseLevel,
                                 CCVariable<varType>& Q, 
                                 const VarLabel* label,
                                 double subCycleProgress_var, 
                                 int matl, 
                                 DataWarehouse* fine_new_dw,
                                 DataWarehouse* coarse_old_dw,
                                 DataWarehouse* coarse_new_dw)
{
  cout_dbg << *patch << " ";
  patch->printPatchBCs(cout_dbg);
  
  for(Patch::FaceType face = Patch::startFace;
      face <= Patch::endFace; face=Patch::nextFace(face)){

   if(patch->getBCType(face) != Patch::Neighbor) {
      //__________________________________
      // fine level hi & lo cell iter limits
      // coarselevel hi and low index
      CellIterator iter_tmp = patch->getFaceCellIterator(face, "plusEdgeCells");
      IntVector fl = iter_tmp.begin();
      IntVector fh = iter_tmp.end(); 
      IntVector refineRatio = fineLevel->getRefinementRatio();
      IntVector coarseLow  = fineLevel->mapCellToCoarser(fl);
      IntVector coarseHigh = fineLevel->mapCellToCoarser(fh+refineRatio - IntVector(1,1,1));


      IntVector axes = patch->faceAxes(face);
      int P_dir = axes[0];  // principal direction      
        
      //__________________________________
      // enlarge the coarselevel foot print by oneCell
      // x-           x+        y-       y+       z-        z+
      // (-1,0,0)  (1,0,0)  (0,-1,0)  (0,1,0)  (0,0,-1)  (0,0,1)
      IntVector oneCell = patch->faceDirection(face);
      if( face == Patch::xminus || face == Patch::yminus 
                                || face == Patch::zminus) {
        coarseHigh -= oneCell;
      }
      if( face == Patch::xplus || face == Patch::yplus 
                               || face == Patch::zplus) {
        coarseLow  -= oneCell;
      }
      
      //__________________________________
      // for higher order interpolation increase the coarse level foot print
      // by the order of interpolation - 1
      if(d_orderOfInterpolation >= 1){
        IntVector interOrder(1,1,1);
        coarseLow  -= interOrder;
        coarseHigh += interOrder;
      } 
      /*
      //__________________________________
      // If the face is orthogonal to a neighboring
      // patch face increase the coarse level foot print
      IntVector expandCellsLo = IntVector(1,1,1) - patch->neighborsLow();
      IntVector expandCellsHi = IntVector(1,1,1) - patch->neighborsHigh();
         
      // in the face normal direction ignore the cell expansion
      expandCellsLo[P_dir] = 0;
      expandCellsHi[P_dir] = 0;
      
      coarseHigh += expandCellsHi;
      coarseLow  -= expandCellsLo;
      */
      //__________________________________
      // coarseHigh and coarseLow cannot lie outside
      // of the coarselevel index range
      IntVector cl, ch;
      coarseLevel->findCellIndexRange(cl,ch);
      coarseLow   = Max(coarseLow, cl);
      coarseHigh  = Min(coarseHigh, ch); 
    
      cout_dbg<< " face " << face << " refineRatio "<< refineRatio
              << " BC type " << patch->getBCType(face)
              << " FineLevel iterator" << fl << " " << fh 
        /*              << " expandCellsHi " << expandCellsHi
              << " expandCellsLo " << expandCellsLo
        */              << " \t coarseLevel iterator " << coarseLow << " " << coarseHigh<<endl;

      //__________________________________
      // subCycleProgress_var near 1.0 
      //  interpolation using the coarse_new_dw data

      if(subCycleProgress_var > 1-1.e-10){ 
       constCCVariable<varType> q_NewDW;
       coarse_new_dw->getRegion(q_NewDW, label, matl, coarseLevel,
                             coarseLow, coarseHigh);
                             
       selectInterpolator(q_NewDW, d_orderOfInterpolation, coarseLevel, 
                          fineLevel, refineRatio, fl,fh, Q);
      } else {    
                      
      //__________________________________
      // subCycleProgress_var somewhere between 0 or 1
      //  interpolation from both coarse new and old dw 
        constCCVariable<varType> q_OldDW, q_NewDW;
        coarse_old_dw->getRegion(q_OldDW, label, matl, coarseLevel,
                             coarseLow, coarseHigh);
        coarse_new_dw->getRegion(q_NewDW, label, matl, coarseLevel,
                             coarseLow, coarseHigh);
                             
        CCVariable<varType> Q_old, Q_new;
        fine_new_dw->allocateTemporary(Q_old, patch);
        fine_new_dw->allocateTemporary(Q_new, patch);
        
        Q_old.initialize(varType(d_EVIL_NUM));
        Q_new.initialize(varType(d_EVIL_NUM));

        selectInterpolator(q_OldDW, d_orderOfInterpolation, coarseLevel, 
                          fineLevel,refineRatio, fl,fh, Q_old);
                          
        selectInterpolator(q_NewDW, d_orderOfInterpolation, coarseLevel, 
                          fineLevel,refineRatio, fl,fh, Q_new);

        // Linear interpolation in time
        for(CellIterator iter(fl,fh); !iter.done(); iter++){
          IntVector f_cell = *iter;
          Q[f_cell] = (1. - subCycleProgress_var)*Q_old[f_cell] 
                          + subCycleProgress_var *Q_new[f_cell];
        }
      }

    }  // valid face
  }  // face loop
  //____ B U L L E T   P R O O F I N G_______ 
  // All values must be initialized at this point
  if(subCycleProgress_var > 1-1.e-10){  
    IntVector badCell;
    CellIterator iter = patch->getExtraCellIterator();
    if( isEqual<varType>(varType(d_EVIL_NUM),iter,Q, badCell) ){
      ostringstream warn;
      warn <<"ERROR AMRICE::refine_CF_interfaceOperator "
           << "detected an uninitialized variable: "
           << label->getName() << ", cell " << badCell
           << " Q_CC " << Q[badCell] 
           << " Patch " << patch->getID() << " Level idx "
           <<fineLevel->getIndex()<<"\n ";
      throw InvalidValue(warn.str(), __FILE__, __LINE__);
    }
  }  
  cout_dbg.setActive(false);// turn off the switch for cout_dbg
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
  if (subCycleProgress_var != 1.0)
    coarse_old_dw = fine_new_dw->getOtherDataWarehouse(Task::CoarseOldDW);
  if (subCycleProgress_var != 0.0)
    coarse_new_dw = fine_new_dw->getOtherDataWarehouse(Task::CoarseNewDW);
  
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
  if (subCycleProgress_var != 1.0)
    coarse_old_dw = fine_new_dw->getOtherDataWarehouse(Task::CoarseOldDW);
  if (subCycleProgress_var != 0.0)
    coarse_new_dw = fine_new_dw->getOtherDataWarehouse(Task::CoarseNewDW);

  refine_CF_interfaceOperator<Vector>
    (patch, level, coarseLevel, val, label, subCycleProgress_var, matl,
     fine_new_dw, coarse_old_dw, coarse_new_dw);
}

/*___________________________________________________________________
 Function~  AMRICE::scheduleRefine--  
_____________________________________________________________________*/
void AMRICE::scheduleRefine(const PatchSet* patches,
                               SchedulerP& sched)
{
  const Level* fineLevel = getLevel(patches);
  cout_doing << d_myworld->myrank() 
             << " AMRICE::scheduleRefine\t\t\t\tL-" 
             <<  fineLevel->getIndex() << " P-" << *patches << '\n';
  Task* task = scinew Task("refine",this, &AMRICE::refine);

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
  
  // if this is a new level, then we need to schedule compute, otherwise, the copydata will yell at us.
  if (patches == getLevel(patches->getSubset(0))->eachPatch()) {
    task->computes(lb->press_CCLabel);
    task->computes(lb->rho_CCLabel);
    task->computes(lb->sp_vol_CCLabel);
    task->computes(lb->temp_CCLabel);
    task->computes(lb->vel_CCLabel);
  }

  sched->addTask(task, patches, d_sharedState->allMaterials()); 
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
             << " Doing refine \t\t\t\t\t\t AMRICE L-"<< fineLevel->getIndex();
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

/*_____________________________________________________________________
 Function~  AMRICE::CoarseToFineOperator--
 Purpose~   push data from coarse Grid to the fine grid
_____________________________________________________________________*/
template<class T>
void AMRICE::CoarseToFineOperator(CCVariable<T>& q_CC,
                                  const VarLabel* varLabel,
                                  const int indx,
                                  DataWarehouse* new_dw,
                                  const double /*ratio*/,
                                  const Patch* finePatch,
                                  const Level* fineLevel,
                                  const Level* coarseLevel)
{
  IntVector refineRatio = fineLevel->getRefinementRatio();
                       
  // region of fine space that will correspond to the coarse we need to get
  IntVector fl, fh;
  Ghost::GhostType  gac = Ghost::AroundCells;

  finePatch->computeVariableExtents(varLabel->typeDescription()->getType(),
                                    IntVector(0,0,0), gac,1, fl, fh); 
  
  // coarse region we need to get from the dw
  IntVector cl = finePatch->getLevel()->mapCellToCoarser(fl);
  IntVector ch = finePatch->getLevel()->mapCellToCoarser(fh) + refineRatio - IntVector(1,1,1);
  
  //__________________________________
  // for higher order interpolation increase the coarse level foot print
  // by the order of interpolation - 1
//   if(d_orderOfInterpolation >= 2){
//     IntVector one(1,1,1);
//     IntVector interOrder(d_orderOfInterpolation,d_orderOfInterpolation,d_orderOfInterpolation);
//     cl  -= (interOrder - one);
//     ch += (interOrder - one);
//   }

  //__________________________________
  // coarseHigh and coarseLow cannot lie outside
  // of the coarselevel index range
  IntVector cl_tmp, ch_tmp;
  coarseLevel->findCellIndexRange(cl_tmp,ch_tmp);
  cl = Max(cl_tmp, cl);
  ch = Min(ch_tmp, ch);

  // fine region to work over
  IntVector lo = finePatch->getInteriorCellLowIndex();
  IntVector hi = finePatch->getInteriorCellHighIndex();

  cout_dbg <<" coarseToFineOperator: " << varLabel->getName()
           <<" finePatch  "<< finePatch->getID() << " "
           << lo<<" "<< hi<< " fl " << fl << " fh " << fh
           <<" coarseRegion " << cl << " " << ch <<endl;
  
  constCCVariable<T> coarse_q_CC;
  new_dw->getRegion(coarse_q_CC, varLabel, indx, coarseLevel, cl, ch);
  
  
  selectInterpolator(coarse_q_CC, d_orderOfInterpolation, coarseLevel, fineLevel,
                      refineRatio, lo,hi,q_CC);
  
  //____ B U L L E T   P R O O F I N G_______ 
  // All fine patch interior values must be initialized at this point
  IntVector badCell;
  CellIterator iter=finePatch->getCellIterator();
  if( isEqual<T>(T(d_EVIL_NUM),iter,q_CC, badCell) ){
    ostringstream warn;
    warn <<"ERROR AMRICE::Refine Task:CoarseToFineOperator "
         << "detected an uninitialized variable "<< varLabel->getName()
         << " " << badCell << " Patch " << finePatch->getID() 
         << " Level idx "<<fineLevel->getIndex()<<"\n ";
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
  if(!doICEOnLevel(fineLevel->getIndex()))
    return;
    
  Ghost::GhostType  gn = Ghost::None; 
  cout_doing << d_myworld->myrank() 
             << " AMRICE::scheduleCoarsen\t\t\t\tL-" 
             << fineLevel->getIndex()<< "->"<<coarseLevel->getIndex()<<endl; 
             
  Task* task = scinew Task("coarsen",this, &AMRICE::coarsen);

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

  sched->addTask(task, coarseLevel->eachPatch(), all_matls); 
  
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
/*_____________________________________________________________________
 Function~  AMRICE::fineToCoarseOperator--
 Purpose~   averages the interior fine patch data onto the coarse patch
_____________________________________________________________________*/
template<class T>
void AMRICE::fineToCoarseOperator(CCVariable<T>& q_CC,
                                  const CCVariable<double>& rho_CC_coarse,
                                  constCCVariable<double>& cv_coarse,
                                  const string& quantity,
                                  const VarLabel* varLabel,
                                  const int indx,
                                  DataWarehouse* new_dw,
                                  const Patch* coarsePatch,
                                  const Level* coarseLevel,
                                  const Level* fineLevel)
{
  Level::selectType finePatches;
  coarsePatch->getFineLevelPatches(finePatches);
   
  Vector dx_c = coarseLevel->dCell();
  Vector dx_f = fineLevel->dCell();
  double coarseCellVol = dx_c.x()*dx_c.y()*dx_c.z();
  double fineCellVol   = dx_f.x()*dx_f.y()*dx_f.z();
                          
  for(int i=0;i<finePatches.size();i++){
    const Patch* finePatch = finePatches[i];
    
    IntVector fl(finePatch->getInteriorCellLowIndex());
    IntVector fh(finePatch->getInteriorCellHighIndex());
    IntVector cl(fineLevel->mapCellToCoarser(fl));
    IntVector ch(fineLevel->mapCellToCoarser(fh));
    
    cl = Max(cl, coarsePatch->getCellLowIndex());
    ch = Min(ch, coarsePatch->getCellHighIndex());

    // get the region of the fine patch that overlaps the coarse patch
    // we might not have the entire patch in this proc's DW
    fl = coarseLevel->mapCellToFiner(cl);
    fh = coarseLevel->mapCellToFiner(ch);
    if (fh.x() <= fl.x() || fh.y() <= fl.y() || fh.z() <= fl.z()) {
      continue;
    }
    
    constCCVariable<T> fine_q_CC;
    constCCVariable<double> cv_fine, rho_CC_fine;
    new_dw->getRegion(fine_q_CC,  varLabel,               indx, fineLevel, fl, fh);
    new_dw->getRegion(cv_fine,    lb->specific_heatLabel, indx, fineLevel, fl, fh);
    new_dw->getRegion(rho_CC_fine,lb->rho_CCLabel,        indx, fineLevel, fl, fh);
    
    cout_dbg << " fineToCoarseOperator: finePatch "<< fl << " " << fh 
             << " coarsePatch "<< cl << " " << ch << endl;
             
    IntVector refinementRatio = fineLevel->getRefinementRatio();
    
    //__________________________________
    //  switches that modify the equation
    //  depending on what quantity is being coarsened.
    double switch1 = d_EVIL_NUM;    
    double switch2 = d_EVIL_NUM;    
    double switch3 = d_EVIL_NUM;    
    if(quantity == "mass" || quantity == "pressure" || quantity == "sp_vol"){
      switch1 = 1.0;
      switch2 = 0.0;
      switch3 = 0.0;         
    }
    if(quantity == "momentum" || quantity == "scalar"){
      switch1 = 0.0;
      switch2 = 1.0;
      switch3 = 0.0;         
    }
    if(quantity == "energy"){
      switch1 = 0.0;
      switch2 = 0.0;
      switch3 = 1.0;
    }
    
    T zero(0.0);
    // iterate over coarse level cells
    for(CellIterator iter(cl, ch); !iter.done(); iter++){
      IntVector c = *iter;
      T q_CC_tmp(zero);
      IntVector fineStart = coarseLevel->mapCellToFiner(c);
    
      // for each coarse level cell iterate over the fine level cells   
      for(CellIterator inside(IntVector(0,0,0),refinementRatio );
                                          !inside.done(); inside++){
        IntVector fc = fineStart + *inside;
        double mass_fineLevel = rho_CC_fine[fc] * fineCellVol;
        
        q_CC_tmp += fine_q_CC[fc] * switch1 * fineCellVol            
                  + fine_q_CC[fc] * switch2 * mass_fineLevel         
                  + fine_q_CC[fc] * switch3 * mass_fineLevel * cv_fine[fc];
      }
      double mass_CC_coarse = rho_CC_coarse[c] * coarseCellVol;
      double denominator = switch1 * coarseCellVol     
                         + switch2 * mass_CC_coarse
                         + switch3 * mass_CC_coarse * cv_coarse[c];
                         
      q_CC[c] =q_CC_tmp / denominator;
    }
  }
  cout_dbg.setActive(false);// turn off the switch for cout_dbg
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
             
  Task* task = scinew Task("reflux_computeCorrectionFluxes",
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
  
  sched->addTask(task, coarseLevel->eachPatch(), d_sharedState->allMaterials()); 
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
 Function~  AMRICE::refluxCoarseLevelIterator--  
 Purpose:  returns the iterator and face-centered offset that the coarse 
           level uses to do refluxing.  THIS IS COMPILCATED AND CONFUSING
_____________________________________________________________________*/
void AMRICE::refluxCoarseLevelIterator(Patch::FaceType patchFace,
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


/*_____________________________________________________________________
 Function~  AMRICE::refluxOperator_computeCorrectionFluxes--
 Purpose~   
_____________________________________________________________________*/
template<class T>
void AMRICE::refluxOperator_computeCorrectionFluxes( 
                              constCCVariable<double>& rho_CC_coarse,
                              constCCVariable<double>& cv,
                              const string& fineVarLabel,
                              const int indx,
                              const Patch* coarsePatch,
                              const Patch* finePatch,
                              const Level* coarseLevel,
                              const Level* fineLevel,
                              DataWarehouse* new_dw)
{
  // form the fine patch flux label names
  string x_name = fineVarLabel + "_X_FC_flux";
  string y_name = fineVarLabel + "_Y_FC_flux";
  string z_name = fineVarLabel + "_Z_FC_flux";
  
  // grab the varLabels
  VarLabel* xlabel = VarLabel::find(x_name);
  VarLabel* ylabel = VarLabel::find(y_name);
  VarLabel* zlabel = VarLabel::find(z_name);  

  if(xlabel == NULL || ylabel == NULL || zlabel == NULL){
    throw InternalError( "refluxOperator_computeCorrectionFluxes: variable label not found: " 
                          + x_name + " or " + y_name + " or " + z_name, __FILE__, __LINE__);
  }

  constSFCXVariable<T> Q_X_fine_flux;
  constSFCYVariable<T> Q_Y_fine_flux;
  constSFCZVariable<T> Q_Z_fine_flux;
  
  SFCXVariable<T>  Q_X_coarse_flux, Q_X_coarse_flux_org;
  SFCYVariable<T>  Q_Y_coarse_flux, Q_Y_coarse_flux_org;
  SFCZVariable<T>  Q_Z_coarse_flux, Q_Z_coarse_flux_org;
  
  // find the exact range of fine data (so we don't mess up mpi)
  IntVector xfl, xfh, yfl, yfh, zfl, zfh, ch;
  IntVector xcl, xch, ycl, ych, zcl, zch, fh;

  xfl = yfl = zfl = finePatch->getInteriorCellLowIndex();
  xcl = ycl = zcl = coarsePatch->getInteriorCellLowIndex();
  
  xfh = finePatch->getInteriorHighIndex(Patch::XFaceBased);
  yfh = finePatch->getInteriorHighIndex(Patch::YFaceBased);
  zfh = finePatch->getInteriorHighIndex(Patch::ZFaceBased);
  xch = coarsePatch->getInteriorHighIndex(Patch::XFaceBased);
  ych = coarsePatch->getInteriorHighIndex(Patch::YFaceBased);
  zch = coarsePatch->getInteriorHighIndex(Patch::ZFaceBased);

  // Intersection of coarse and fine patches
  xfl = Max(coarseLevel->mapCellToFiner(xcl), xfl);
  yfl = Max(coarseLevel->mapCellToFiner(ycl), yfl);
  zfl = Max(coarseLevel->mapCellToFiner(zcl), zfl);
  xfh = Min(coarseLevel->mapCellToFiner(xch), xfh);
  yfh = Min(coarseLevel->mapCellToFiner(ych), yfh);
  zfh = Min(coarseLevel->mapCellToFiner(zch), zfh);

  // if high == low, then don't bother (there are cases that it will, trust me)
  bool do_x = true;
  bool do_y = true;
  bool do_z = true;

  if (xfl.x() >= xfh.x() || xfl.y() >= xfh.y() || xfl.z() >= xfh.z()) {
    do_x = false;
  }
  if (yfl.x() >= zfh.x() || yfl.y() >= yfh.y() || yfl.z() >= yfh.z()) {
    do_y = false;
  }
  if (zfl.x() >= zfh.x() || zfl.y() >= zfh.y() || zfl.z() >= zfh.z()) {
    do_z = false;
  }

  if (do_x) {
    new_dw->getRegion(Q_X_fine_flux,    xlabel,indx, fineLevel,   xfl,xfh);
    new_dw->allocateTemporary(Q_X_coarse_flux_org, coarsePatch);
    new_dw->getModifiable(Q_X_coarse_flux,  xlabel,indx, coarsePatch);
    Q_X_coarse_flux_org.copyData(Q_X_coarse_flux);
  }
  if (do_y) {
    new_dw->getRegion(Q_Y_fine_flux,    ylabel,indx, fineLevel,   yfl,yfh);
    new_dw->allocateTemporary(Q_Y_coarse_flux_org, coarsePatch);
    new_dw->getModifiable(Q_Y_coarse_flux,  ylabel,indx, coarsePatch);
    Q_Y_coarse_flux_org.copyData(Q_Y_coarse_flux);
  }
  if (do_z) {
    new_dw->getRegion(Q_Z_fine_flux,    zlabel,indx, fineLevel,   zfl,zfh);
    new_dw->allocateTemporary(Q_Z_coarse_flux_org, coarsePatch);
    new_dw->getModifiable(Q_Z_coarse_flux,  zlabel,indx, coarsePatch);
    Q_Z_coarse_flux_org.copyData(Q_Z_coarse_flux);
  }

  Vector dx = coarsePatch->dCell();
  double coarseCellVol = dx.x()*dx.y()*dx.z();

  //__________________________________
  //  switches that modify the denomiator 
  //  depending on which quantity is being refluxed.
  
  double switch1 = 0.0;     // denomiator = cellVol
  double switch2 = 1.0;     //            = mass
  double switch3 = 0.0;     //            = mass * cv
  if(fineVarLabel == "mass"){
    switch1 = 1.0;
    switch2 = 0.0;
    switch3 = 0.0;         
  }
  if(fineVarLabel == "int_eng"){
    switch1 = 0.0;
    switch2 = 0.0;
    switch3 = 1.0;
  }

  //__________________________________
  // Iterate over coarsefine interface faces
  vector<Patch::FaceType>::const_iterator iter;  
  for (iter  = finePatch->getCoarseFineInterfaceFaces()->begin(); 
       iter != finePatch->getCoarseFineInterfaceFaces()->end(); ++iter){
    Patch::FaceType patchFace = *iter;   
    
    if (!do_x && (patchFace == Patch::xminus || patchFace == Patch::xplus))
      continue;
    if (!do_y && (patchFace == Patch::yminus || patchFace == Patch::yplus))
      continue;
    if (!do_z && (patchFace == Patch::zminus || patchFace == Patch::zplus))
      continue;
    
    // find the coarse level iterator along the interface
    IntVector c_FC_offset;
    CellIterator c_iter(IntVector(-8,-8,-8),IntVector(-9,-9,-9));
    bool isRight_CP_FP_pair;
    refluxCoarseLevelIterator( patchFace,coarsePatch, finePatch, fineLevel,
                               c_iter ,c_FC_offset,isRight_CP_FP_pair);

    // eject if this is not the right coarse/fine patch pair
    if (isRight_CP_FP_pair == false ){
      return;
    }
    
    // Offset for the fine cell loop (fineStart)
    // shift (+-refinement ratio) for x-, y-, z- finePatch faces
    // shift 0 cells              for x+, y+, z+ finePatchFaces
    
    string name = finePatch->getFaceName(patchFace);
    IntVector offset = finePatch->faceDirection(patchFace);
    IntVector r_Ratio = fineLevel->getRefinementRatio();
    IntVector f_offset(0,0,0);
    double c_FaceNormal = 0;
    double f_FaceNormal = 0;
    
    if(name == "xminus" || name == "yminus" || name == "zminus"){
      f_offset = r_Ratio * -offset;
      c_FaceNormal = +1;
      f_FaceNormal = -1;
    }
    if(name == "xplus" || name == "yplus" || name == "zplus"){
      c_FaceNormal = -1;
      f_FaceNormal = +1;
    } 

    
/*`==========TESTING==========*/
#if SPEW
  cout << " ------------ refluxOperator_computeCorrectionFluxes " << fineVarLabel<< endl; 
  IntVector half  = (c_iter.end() - c_iter.begin() )/IntVector(2,2,2) + c_iter.begin();
  cout <<name <<  " coarsePatch " << *coarsePatch << endl;
  cout << "      finePatch   " << *finePatch << endl; 
#endif 
/*===========TESTING==========`*/
    //__________________________________
    // Add fine patch face fluxes to the coarse cells
    // c_CC f_CC:    coarse/fine level cell center index
    // c_FC f_FC:    coarse/fine level face center index
    if(patchFace == Patch::xminus || patchFace == Patch::xplus){    // X+ X-
    
      //__________________________________
      // sum all of the fluxes passing from the 
      // fine level to the coarse level
      for(; !c_iter.done(); c_iter++){
         IntVector c_CC = *c_iter;
         IntVector c_FC = c_CC + c_FC_offset;
         
         T sum_fineLevelFlux(0.0);
         IntVector fineStart = coarseLevel->mapCellToFiner(c_CC) + f_offset;
         IntVector rRatio_X(1,r_Ratio.y(), r_Ratio.z()); 
         
         for(CellIterator inside(IntVector(0,0,0),rRatio_X );!inside.done(); inside++){
           IntVector f_FC = fineStart + *inside;
           sum_fineLevelFlux += Q_X_fine_flux[f_FC];
         }
         // Q_CC = mass * q_CC = cellVol * rho * q_CC
         // coeff accounts for the different cell sizes on the different levels
         double coeff = (double)r_Ratio.x() * r_Ratio.y() * r_Ratio.z();
         double mass_CC_coarse = rho_CC_coarse[c_CC] * coarseCellVol;
         double denominator = switch1 * coarseCellVol     
                            + switch2 * mass_CC_coarse
                            + switch3 * mass_CC_coarse * cv[c_CC];          
                            
         Q_X_coarse_flux[c_FC] = ( c_FaceNormal*Q_X_coarse_flux_org[c_FC] + coeff* f_FaceNormal*sum_fineLevelFlux) /denominator;
         
/*`==========TESTING==========*/
#if SPEW
        if (c_CC.y() == half.y() && c_CC.z() == half.z() ) {
          cout << " \t c_CC " << c_CC  << " c_FC " << c_FC << " q_X_FC " << c_FaceNormal*Q_X_coarse_flux_org[c_FC]
               << " sum_fineLevelflux " << coeff* f_FaceNormal *sum_fineLevelFlux
               << " Q_X_coarse_flux_org " << Q_X_coarse_flux_org[c_FC]
               << " correction " << ( c_FaceNormal*Q_X_coarse_flux[c_FC] + coeff* f_FaceNormal *sum_fineLevelFlux) /denominator << endl;
          cout << "" << endl;
        }
#endif 
/*===========TESTING==========`*/
      }
    }
    if(patchFace == Patch::yminus || patchFace == Patch::yplus){    // Y+ Y-
    
      for(; !c_iter.done(); c_iter++){
         IntVector c_CC = *c_iter;
         IntVector c_FC = c_CC + c_FC_offset;
         
         T sum_fineLevelFlux(0.0);
         IntVector fineStart = coarseLevel->mapCellToFiner(c_CC) + f_offset;
         IntVector rRatio_Y(r_Ratio.x(),1, r_Ratio.z()); 
         
         for(CellIterator inside(IntVector(0,0,0),rRatio_Y );!inside.done(); inside++){
           IntVector f_FC = fineStart + *inside;
           sum_fineLevelFlux += Q_Y_fine_flux[f_FC];          
         }
         // Q_CC = mass * q_CC = cellVol * rho * q_CC
         // coeff accounts for the different cell sizes on the different levels
         double coeff = (double)r_Ratio.x() * r_Ratio.y() * r_Ratio.z();
         double mass_CC_coarse = rho_CC_coarse[c_CC] * coarseCellVol;
         double denominator = switch1 * coarseCellVol     
                            + switch2 * mass_CC_coarse
                            + switch3 * mass_CC_coarse * cv[c_CC];
                            
         Q_Y_coarse_flux[c_FC] = (c_FaceNormal*Q_Y_coarse_flux_org[c_FC] + coeff*f_FaceNormal*sum_fineLevelFlux) /denominator;
         
/*`==========TESTING==========*/
#if SPEW
        if (c_CC.x() == half.x() && c_CC.z() == half.z() ) {
          cout << " \t c_CC " << c_CC  << " c_FC " << c_FC << " q_Y_FC " << c_FaceNormal*Q_Y_coarse_flux[c_FC]
               << " sum_fineLevelflux " << coeff* f_FaceNormal *sum_fineLevelFlux
               << " Q_Y_coarse_flux_org " << Q_Y_coarse_flux_org[c_FC]
               << " correction " << ( c_FaceNormal*Q_Y_coarse_flux_org[c_FC] + coeff* f_FaceNormal *sum_fineLevelFlux) /denominator << endl;
          cout << "" << endl;
        }
#endif 
/*===========TESTING==========`*/
      }
    }
    if(patchFace == Patch::zminus || patchFace == Patch::zplus){    // Z+ Z-
    
 
      for(; !c_iter.done(); c_iter++){
         IntVector c_CC = *c_iter;
         IntVector c_FC = c_CC + c_FC_offset;
         
         T sum_fineLevelFlux(0.0);
         IntVector fineStart = coarseLevel->mapCellToFiner(c_CC) + f_offset;
         IntVector rRatio_Z(r_Ratio.x(),r_Ratio.y(), 1); 
         
         for(CellIterator inside(IntVector(0,0,0),rRatio_Z );!inside.done(); inside++){
           IntVector f_FC = fineStart + *inside;
           sum_fineLevelFlux += Q_Z_fine_flux[f_FC];
         }
         // Q_CC = mass * q_CC = cellVol * rho * q_CC
         // coeff accounts for the different cell sizes on the different levels
         double coeff = (double)r_Ratio.x() * r_Ratio.y() * r_Ratio.z();
         double mass_CC_coarse = rho_CC_coarse[c_CC] * coarseCellVol;
         double denominator = switch1 * coarseCellVol     
                            + switch2 * mass_CC_coarse
                            + switch3 * mass_CC_coarse * cv[c_CC];
                            
         Q_Z_coarse_flux[c_FC] = (c_FaceNormal*Q_Z_coarse_flux_org[c_FC] + coeff*f_FaceNormal*sum_fineLevelFlux) /denominator;
         
/*`==========TESTING==========*/
#if SPEW
        if (c_CC.x() == half.x() && c_CC.y() == half.y() ) {
          cout << " \t c_CC " << c_CC  << " c_FC " << c_FC << " q_Z_FC " << c_FaceNormal*Q_Z_coarse_flux[c_FC]
               << " sum_fineLevelflux " << coeff* f_FaceNormal *sum_fineLevelFlux
               << " Q_Z_coarse_flux_org " << Q_Z_coarse_flux_org[c_FC]
               << " correction " << ( c_FaceNormal*Q_Z_coarse_flux_org[c_FC] + coeff* f_FaceNormal *sum_fineLevelFlux) /denominator<< endl;
          cout << "" << endl;
        }
#endif 
/*===========TESTING==========`*/
      }
    }
  }  // coarseFineInterface faces 
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
             
  Task* task = scinew Task("reflux_applyCorrectionFluxes",
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

  sched->addTask(task, coarseLevel->eachPatch(), d_sharedState->allMaterials()); 
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
 Function~  AMRICE::refluxOperator_applyCorrectionFluxes
 Purpose~   
_____________________________________________________________________*/
template<class T>
void AMRICE::refluxOperator_applyCorrectionFluxes(                             
                              CCVariable<T>& q_CC_coarse,
                              const string& varLabel,
                              const int indx,
                              const Patch* coarsePatch,
                              const Patch* finePatch,
                              const Level* coarseLevel,
                              const Level* fineLevel,
                              DataWarehouse* new_dw)
{
  // form the fine patch flux label names
  string x_name = varLabel + "_X_FC_flux";
  string y_name = varLabel + "_Y_FC_flux";
  string z_name = varLabel + "_Z_FC_flux";
  
  // grab the varLabels
  VarLabel* xlabel = VarLabel::find(x_name);
  VarLabel* ylabel = VarLabel::find(y_name);
  VarLabel* zlabel = VarLabel::find(z_name);  

  if(xlabel == NULL || ylabel == NULL || zlabel == NULL){
    throw InternalError( "refluxOperator_applyCorrectionFluxes: variable label not found: " 
                          + x_name + " or " + y_name + " or " + z_name, __FILE__, __LINE__);
  }
  constSFCXVariable<T>  Q_X_coarse_flux;
  constSFCYVariable<T>  Q_Y_coarse_flux;
  constSFCZVariable<T>  Q_Z_coarse_flux;
  
  Ghost::GhostType  gac = Ghost::AroundCells;
  new_dw->get(Q_X_coarse_flux,  xlabel,indx, coarsePatch, gac,1);
  new_dw->get(Q_Y_coarse_flux,  ylabel,indx, coarsePatch, gac,1);
  new_dw->get(Q_Z_coarse_flux,  zlabel,indx, coarsePatch, gac,1); 
  
  //__________________________________
  // Iterate over coarsefine interface faces
  vector<Patch::FaceType>::const_iterator iter;  
  for (iter  = finePatch->getCoarseFineInterfaceFaces()->begin(); 
       iter != finePatch->getCoarseFineInterfaceFaces()->end(); ++iter){
    Patch::FaceType patchFace = *iter;
 
    // determine the iterator for the coarse level.
    IntVector c_FC_offset;
    CellIterator c_iter(IntVector(-8,-8,-8),IntVector(-9,-9,-9));
    bool isRight_CP_FP_pair;
    refluxCoarseLevelIterator( patchFace,coarsePatch, finePatch, fineLevel,
                               c_iter ,c_FC_offset,isRight_CP_FP_pair);

    // eject if this is not the right coarse/fine patch pair
    if (isRight_CP_FP_pair == false ){
      return;
    };

#if 0    
    cout << name << " l " << l << " h " << h << endl;
    cout << " coarsePatch " << *coarsePatch << endl;
    cout << " finePatch " << *finePatch << endl; 
#endif    
    
    //__________________________________
    // Add fine patch face fluxes correction to the coarse cells
    // c_CC:    coarse level cell center index
    // c_FC:    coarse level face center index
    if(patchFace == Patch::xminus || patchFace == Patch::xplus){
      for(; !c_iter.done(); c_iter++){
        IntVector c_CC = *c_iter;
        IntVector c_FC = c_CC + c_FC_offset;                
        q_CC_coarse[c_CC] += Q_X_coarse_flux[c_FC];
      }
    }
    if(patchFace == Patch::yminus || patchFace == Patch::yplus){
      for(; !c_iter.done(); c_iter++){
         IntVector c_CC = *c_iter;
         IntVector c_FC = c_CC + c_FC_offset;
         q_CC_coarse[c_CC] += Q_Y_coarse_flux[c_FC];
      }
    }
    if(patchFace == Patch::zminus || patchFace == Patch::zplus){
      for(; !c_iter.done(); c_iter++){
         IntVector c_CC = *c_iter;
         IntVector c_FC = c_CC + c_FC_offset;             
         q_CC_coarse[c_CC] += Q_Z_coarse_flux[c_FC];
      }
    }
  }  // coarseFineInterface faces
} 
/*_____________________________________________________________________
 Function~  AMRICE::scheduleInitialErrorEstimate--
______________________________________________________________________*/
void AMRICE::scheduleInitialErrorEstimate(const LevelP& /*coarseLevel*/,
                                          SchedulerP& /*sched*/)
{
#if 0
  scheduleErrorEstimate(coarseLevel, sched);
  
  //__________________________________
  // Models
  for(vector<ModelInterface*>::iterator iter = d_models.begin();
     iter != d_models.end(); iter++){
    (*iter)->scheduleErrorEstimate(coarseLevel, sched);;
  }
#endif
}

/*_____________________________________________________________________
 Function~  AMRICE::scheduleErrorEstimate--
______________________________________________________________________*/
void AMRICE::scheduleErrorEstimate(const LevelP& coarseLevel,
                                   SchedulerP& sched)
{
  if(!doICEOnLevel(coarseLevel->getIndex()+1))
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
  
  t->modifies(d_sharedState->get_refineFlag_label(), d_sharedState->refineFlagMaterials());
  t->modifies(d_sharedState->get_refinePatchFlag_label(), d_sharedState->refineFlagMaterials());
  
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
    
    new_dw->get(press_CC, lb->press_CCLabel,    0,patch,gac,1);
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
