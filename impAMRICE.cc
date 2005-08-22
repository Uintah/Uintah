#include <Packages/Uintah/CCA/Components/ICE/ICE.h>
#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/ICE/BoundaryCond.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>

#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Exceptions/ConvergenceFailure.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h> 
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Core/Util/DebugStream.h>
#include <Core/Exceptions/InternalError.h>

using namespace SCIRun;
using namespace Uintah;

static DebugStream cout_doing("ICE_DOING_COUT", false);
static DebugStream cout_dbg("impAMRICE_DBG", false);


/* _____________________________________________________________________
 Function~  ICE::scheduleLockstepTimeAdvance--
_____________________________________________________________________*/
void
ICE::scheduleLockstepTimeAdvance( const GridP& grid, SchedulerP& sched)
{
  int maxLevel = grid->numLevels();
  vector<const PatchSet*> allPatchSets;
  
  const MaterialSet* ice_matls = d_sharedState->allICEMaterials();
  const MaterialSet* mpm_matls = d_sharedState->allMPMMaterials();
  const MaterialSet* all_matls = d_sharedState->allMaterials();  

  //MaterialSubset* one_matl = d_press_matl;
  const MaterialSubset* ice_matls_sub = ice_matls->getUnion();
  const MaterialSubset* mpm_matls_sub = mpm_matls->getUnion();
  double AMR_subCycleProgressVar = 0; 
  
  cout_doing << "--------------------------------------------------------"<< endl;
  cout_doing << "ICE::scheduleLockstepTimeAdvance"<< endl;  
  
  
  //__________________________________
  //
  for(int L = 0; L<maxLevel; L++){
    LevelP level = grid->getLevel(L);
    const PatchSet* patches = level->eachPatch();
    allPatchSets.push_back(level->eachPatch());
    
    if(!doICEOnLevel(level->getIndex())){
      return;
    }

    // for AMR, we need to reset the initial Delt otherwise some unsuspecting level will
    // get the init delt when it didn't compute delt on L0.
    if (d_sharedState->getCurrentTopLevelTimeStep() > 1){
      d_initialDt = 10000.0;
    }


    if(d_turbulence){
      // The turblence model is also called directly from
      // accumlateMomentumSourceSinks.  This method just allows other
      // quantities (such as variance) to be computed
      d_turbulence->scheduleTurbulence1(sched, patches, ice_matls);
    }
    vector<PatchSubset*> maxMach_PSS(Patch::numFaces);
    scheduleMaxMach_on_Lodi_BC_Faces(       sched, level,   ice_matls, 
                                                            maxMach_PSS);

    scheduleComputeThermoTransportProperties(sched, level,  ice_matls);

    scheduleComputePressure(                sched, patches, d_press_matl,
                                                            all_matls);

    if (d_RateForm) {
      schedulecomputeDivThetaVel_CC(        sched, patches, ice_matls_sub,        
                                                            mpm_matls_sub,        
                                                            all_matls);           
    }  

    scheduleComputeTempFC(                   sched, patches, ice_matls_sub,  
                                                             mpm_matls_sub,
                                                             all_matls);    

    scheduleComputeModelSources(             sched, level,   all_matls);

    scheduleUpdateVolumeFraction(            sched, level,   d_press_matl,
                                                             all_matls);


    scheduleComputeVel_FC(                   sched, patches,ice_matls_sub, 
                                                            mpm_matls_sub, 
                                                            d_press_matl,    
                                                            all_matls,     
                                                            false);        

    scheduleAddExchangeContributionToFCVel( sched, patches,ice_matls_sub,
                                                           all_matls,
                                                           false);
  }
//______________________________________________________________________
// MULTI-LEVEL PRESSURE SOLVE AND
#if 0
    if(d_impICE) {        //  I M P L I C I T

      scheduleSetupRHS(                     sched, patches,  one_matl, 
                                                             all_matls,
                                                             false);

      scheduleMultiLevelPressureSolve(       sched, level,   patches,
                                                             one_matl,      
                                                             d_press_matl,    
                                                             ice_matls_sub,  
                                                             mpm_matls_sub, 
                                                             all_matls);

      scheduleComputeDel_P(                   sched,  level, patches,  
                                                             one_matl,
                                                             d_press_matl,
                                                             all_matls);
    }                    






  if(d_doAMR){
    for(int L = maxLevel-1; L> 0; L--){ // from finer to coarser levels
      LevelP coarseLevel = grid->getLevel(L-1);
      scheduleCoarsen_imp_delP(  sched,  coarseLevel,  d_press_matl);
    }
  }
#endif

//______________________________________________________________________

  for(int L = 0; L<maxLevel; L++){
    LevelP level = grid->getLevel(L);
    const PatchSet* patches = level->eachPatch();
    
    if(!doICEOnLevel(level->getIndex())){
      return;
    }    
    if(!d_impICE){         //  E X P L I C I T
      scheduleComputeDelPressAndUpdatePressCC(sched, patches,d_press_matl,     
                                                             ice_matls_sub,  
                                                             mpm_matls_sub,  
                                                             all_matls);     
    }
    
    scheduleComputePressFC(                 sched, patches, d_press_matl,
                                                            all_matls);

    scheduleAccumulateMomentumSourceSinks(  sched, patches, d_press_matl,
                                                            ice_matls_sub,
                                                            mpm_matls_sub,
                                                            all_matls);
    scheduleAccumulateEnergySourceSinks(    sched, patches, ice_matls_sub,
                                                            mpm_matls_sub,
                                                            d_press_matl,
                                                            all_matls);

    scheduleComputeLagrangianValues(        sched, patches, all_matls);

    scheduleAddExchangeToMomentumAndEnergy( sched, patches, ice_matls_sub,
                                                            mpm_matls_sub,
                                                            d_press_matl,
                                                            all_matls);

    scheduleComputeLagrangianSpecificVolume(sched, patches, ice_matls_sub,
                                                            mpm_matls_sub, 
                                                            d_press_matl,
                                                            all_matls);

    scheduleComputeLagrangian_Transported_Vars(sched, patches,
                                                            all_matls);

    scheduleAdvectAndAdvanceInTime(         sched, patches, AMR_subCycleProgressVar,
                                                            ice_matls_sub,
                                                            mpm_matls_sub,
                                                            d_press_matl,
                                                            all_matls);

    scheduleTestConservation(               sched, patches, ice_matls_sub,
                                                            all_matls); 
  }
  //__________________________________
  //  coarsen and refineInterface
  if(d_doAMR){
    for(int L = maxLevel-1; L> 0; L--){ // from finer to coarser levels
      LevelP coarseLevel = grid->getLevel(L-1);
      scheduleCoarsen(coarseLevel, sched);
    }
    for(int L = 1; L<maxLevel; L++){   // from coarser to finer levels
      LevelP fineLevel = grid->getLevel(L);
      scheduleRefineInterface(fineLevel, sched, 1, 1);
    }
  }

#if 0
    if(d_canAddICEMaterial){
      //  This checks to see if the model on THIS patch says that it's
      //  time to add a new material
      scheduleCheckNeedAddMaterial(           sched, level,   all_matls);

      //  This one checks to see if the model on ANY patch says that it's
      //  time to add a new material
      scheduleSetNeedAddMaterialFlag(         sched, level,   all_matls);
    }
#endif
    cout_doing << "---------------------------------------------------------"<<endl;
}




/*___________________________________________________________________
 Function~  ICE::scheduleMultiLevelPressureSolve--
_____________________________________________________________________*/
void ICE::scheduleMultiLevelPressureSolve(  SchedulerP& sched,
                                          const LevelP& level,
                                          const PatchSet*,
                                          const MaterialSubset* one_matl,
                                          const MaterialSubset* press_matl,
                                          const MaterialSubset* ice_matls,
                                          const MaterialSubset* mpm_matls,
                                          const MaterialSet* all_matls)
{
  cout_doing << "ICE::scheduleMultiLevelPressureSolve" << endl;
  
  Task* t = scinew Task("ICE::multiLevelPressureSolve", 
                   this, &ICE::multiLevelPressureSolve,
                   level, sched.get_rep(), ice_matls, mpm_matls);
 
  t->hasSubScheduler();
  Ghost::GhostType  gac = Ghost::AroundCells;  
  Ghost::GhostType  gn  = Ghost::None;
  Task::DomainSpec oims = Task::OutOfDomain;  //outside of ice matlSet.
  //__________________________________
  // The computes and requires when you're looking up 
  // from implicitPressure solve
  // OldDW = ParentOldDW
  // NewDW = ParentNewDW
  //__________________________________
  // common Variables
//  t->requires( Task::OldDW, lb->delTLabel);    AMR
  t->requires( Task::NewDW, lb->vol_frac_CCLabel,   gac,2); 
  t->requires( Task::NewDW, lb->sp_vol_CCLabel,     gac,1);
  t->requires( Task::NewDW, lb->press_equil_CCLabel, press_matl, oims,gac,1);
  t->requires( Task::NewDW, lb->rhsLabel,            one_matl,   oims,gn,0);
  //t->requires( Task::OldDW, lb->initialGuessLabel, one_matl,   oims,gn,0);
  //__________________________________
  // SetupRHS
  if(d_models.size() > 0){  
    t->requires(Task::NewDW,lb->modelMass_srcLabel, gn,0);
  } 
  t->requires( Task::NewDW, lb->speedSound_CCLabel, gn,0);
  t->requires( Task::NewDW, lb->max_RHSLabel);
  
  //__________________________________
  // setup Matrix
  t->requires( Task::NewDW, lb->sp_volX_FCLabel,    gac,1);            
  t->requires( Task::NewDW, lb->sp_volY_FCLabel,    gac,1);            
  t->requires( Task::NewDW, lb->sp_volZ_FCLabel,    gac,1);            
  t->requires( Task::NewDW, lb->vol_fracX_FCLabel,  gac,1);            
  t->requires( Task::NewDW, lb->vol_fracY_FCLabel,  gac,1);            
  t->requires( Task::NewDW, lb->vol_fracZ_FCLabel,  gac,1);            
  t->requires( Task::NewDW, lb->sumKappaLabel,  press_matl,oims,gn,0);         
   
  //__________________________________
  // Update Pressure
  t->requires( Task::NewDW, lb->rho_CCLabel,                       gac,1);            
  t->requires( Task::NewDW, lb->press_CCLabel,   press_matl, oims, gac,1);  
  
  computesRequires_CustomBCs(t, "implicitPressureSolve", lb, ice_matls,
                             d_customBC_var_basket);

  //__________________________________
  // ImplicitVel_FC
  t->requires(Task::OldDW,lb->vel_CCLabel,       ice_matls,  gac,1);    
  t->requires(Task::NewDW,lb->vel_CCLabel,       mpm_matls,  gac,1);

  //__________________________________
  //  what's produced from this task
  t->modifies(lb->press_CCLabel, press_matl,oims);  
  t->modifies(lb->term2Label,    one_matl,  oims);   

  t->modifies(lb->uvel_FCMELabel);
  t->modifies(lb->vvel_FCMELabel);
  t->modifies(lb->wvel_FCMELabel);
  
  t->modifies(lb->vol_fracX_FCLabel);
  t->modifies(lb->vol_fracY_FCLabel);
  t->modifies(lb->vol_fracZ_FCLabel);  
  
  LoadBalancer* loadBal = sched->getLoadBalancer();
  const PatchSet* perproc_patches =  
                        loadBal->createPerProcessorPatchSet(level);
  sched->addTask(t, perproc_patches, all_matls);
}
/*___________________________________________________________________ 
 Function~  ICE::multiLevelPressureSolve-- 
_____________________________________________________________________*/
void ICE::multiLevelPressureSolve(const ProcessorGroup* pg,
		                    const PatchSubset* patch_sub, 
		                    const MaterialSubset*,       
		                    DataWarehouse* ParentOldDW,    
                                  DataWarehouse* ParentNewDW,    
		                    LevelP level,                 
                                  Scheduler* sched,
                                  const MaterialSubset* ice_matls,
                                  const MaterialSubset* mpm_matls)
{
  cout_doing<<"Doing multiLevelPressureSolve "<<"\t\t\t\t ICE \tL-" 
            << level->getIndex()<< endl;

  //__________________________________
  // define Matl sets and subsets
  const MaterialSet* all_matls = d_sharedState->allMaterials();
  MaterialSubset* one_matl    = d_press_matl;
  MaterialSet* press_matlSet  = scinew MaterialSet();
  press_matlSet->add(0);
  press_matlSet->addReference(); 
  
  //__________________________________
  //  turn off parentDW scrubbing
  DataWarehouse::ScrubMode ParentOldDW_scrubmode =
                           ParentOldDW->setScrubbing(DataWarehouse::ScrubNone);
  DataWarehouse::ScrubMode ParentNewDW_scrubmode =
                           ParentNewDW->setScrubbing(DataWarehouse::ScrubNone);
  //__________________________________
  // create a new subscheduler
  SchedulerP subsched = sched->createSubScheduler();
  subsched->setRestartable(true); 
  subsched->initialize(3, 1, ParentOldDW, ParentNewDW);
  subsched->clearMappings();
  subsched->mapDataWarehouse(Task::ParentOldDW, 0);
  subsched->mapDataWarehouse(Task::ParentNewDW, 1);
  subsched->mapDataWarehouse(Task::OldDW, 2);
  subsched->mapDataWarehouse(Task::NewDW, 3);
  
  DataWarehouse* subOldDW = subsched->get_dw(2);
  DataWarehouse* subNewDW = subsched->get_dw(3);

  GridP grid = level->getGrid();
  subsched->advanceDataWarehouse(grid);
  const PatchSet* patch_set = level->eachPatch();

  //__________________________________
  //  Move data from parentOldDW to subSchedNewDW.
  delt_vartype dt;
  subNewDW = subsched->get_dw(3);
  ParentOldDW->get(dt, d_sharedState->get_delt_label());
  subNewDW->put(dt, d_sharedState->get_delt_label());
   
  max_vartype max_RHS_old;
  ParentNewDW->get(max_RHS_old, lb->max_RHSLabel);
  subNewDW->put(   max_RHS_old, lb->max_RHSLabel);
  
  subNewDW->transferFrom(ParentNewDW,lb->press_CCLabel,     patch_sub, d_press_matl); 
  subNewDW->transferFrom(ParentNewDW,lb->rhsLabel,          patch_sub, one_matl);
  //subNewDW->transferFrom(ParentOldDW,lb->initialGuessLabel, patch_sub, one_matl);  
  //__________________________________
  //  Iteration Loop
  max_vartype max_RHS = 1/d_SMALL_NUM;
  double smallest_max_RHS_sofar = max_RHS; 
  int counter = 0;
  bool restart   = false;
  bool recursion = true;
  bool firstIter = true;
  //const VarLabel* whichInitialGuess = lb->initialGuessLabel;
  const VarLabel* whichInitialGuess = NULL;
  
  while( counter < d_max_iter_implicit && max_RHS > d_outer_iter_tolerance) {
  
    //__________________________________
    // schedule the tasks
    subsched->initialize(3, 1, ParentOldDW, ParentNewDW);
   


    scheduleSetupMatrix(    subsched, level,  patch_set,  one_matl, 
                                                          all_matls,
                                                          firstIter);

    solver->scheduleSolve(level, subsched, press_matlSet,
                          lb->matrixLabel,   Task::NewDW,
                          lb->imp_delPLabel, false,
                          lb->rhsLabel,      Task::OldDW,
                          whichInitialGuess, Task::OldDW,
			     solver_parameters);



    scheduleUpdatePressure( subsched,  level, patch_set,  ice_matls,
                                                          mpm_matls, 
                                                          d_press_matl,  
                                                          all_matls);
                                                          
    scheduleImplicitVel_FC( subsched,         patch_set,  ice_matls,
                                                          mpm_matls, 
                                                          d_press_matl, 
                                                          all_matls,
                                                          recursion);

    scheduleSetupRHS(       subsched,         patch_set,  one_matl, 
                                                          all_matls,
                                                          recursion);

    subsched->compile();
    //__________________________________
    //  - move subNewDW to subOldDW
    //  - scrub the subScheduler
    //  - execute the tasks
    subsched->advanceDataWarehouse(grid); 
    subOldDW = subsched->get_dw(2);
    subNewDW = subsched->get_dw(3);
    subOldDW->setScrubbing(DataWarehouse::ScrubComplete);
    subNewDW->setScrubbing(DataWarehouse::ScrubNone);
    
    subsched->execute();
    
    counter ++;
    firstIter = false;
    whichInitialGuess = NULL;
    
    //__________________________________
    // diagnostics
    subNewDW->get(max_RHS,     lb->max_RHSLabel);
    subOldDW->get(max_RHS_old, lb->max_RHSLabel);
    
    if(pg->myrank() == 0) {
      cout << "Outer iteration " << counter
           << " max_rhs before solve "<< max_RHS_old
           << " after solve " << max_RHS<< endl;
    }
    
    //__________________________________
    // restart timestep
                                          //  too many outer iterations
    if (counter > d_iters_before_timestep_restart ){
      restart = true;
      if(pg->myrank() == 0)
        cout <<"\nWARNING: max iterations befor timestep restart reached\n"<<endl;
    }
                                          //  solver has requested a restart
    if (subsched->get_dw(3)->timestepRestarted() ) {
      if(pg->myrank() == 0)
        cout << "\nWARNING: Solver had requested a restart\n" <<endl;
      restart = true;
    }
    
                                           //  solution is diverging
    if(max_RHS < smallest_max_RHS_sofar){
      smallest_max_RHS_sofar = max_RHS;
    }
    if(((max_RHS - smallest_max_RHS_sofar) > 100.0*smallest_max_RHS_sofar) ){
      if(pg->myrank() == 0)
        cout << "\nWARNING: outer interation is diverging now "
             << "restarting the timestep"
             << " Max_RHS " << max_RHS 
             << " smallest_max_RHS_sofar "<< smallest_max_RHS_sofar<< endl;
      restart = true;
    }
    if(restart){
      ParentNewDW->abortTimestep();
      ParentNewDW->restartTimestep();
      return;
    }
  }  // outer iteration loop
  
  //__________________________________
  //  BULLET PROOFING
  if ( (counter == d_max_iter_implicit)   && 
       (max_RHS > d_outer_iter_tolerance) &&
       counter > 1) {
    ostringstream s;
    s <<"ERROR ICE::implicitPressureSolve, the maximum number of outer"
      <<" iterations was reached. \n " 
      << "Try either increasing the max_outer_iterations "
      <<" or decrease outer_iteration_tolerance\n ";
    throw ConvergenceFailure(s.str(),counter,max_RHS,d_outer_iter_tolerance,__FILE__,__LINE__);
  }

  //__________________________________
  // Move products of iteration (only) from sub_new_dw -> parent_new_dw
  subNewDW  = subsched->get_dw(3);
  bool replace = true;
  const MaterialSubset* all_matls_sub = all_matls->getUnion();
  ParentNewDW->transferFrom(subNewDW,         // press
                    lb->press_CCLabel,       patch_sub,  d_press_matl, replace); 
  ParentNewDW->transferFrom(subNewDW,         // term2
                    lb->term2Label,          patch_sub,  one_matl,     replace);    
  ParentNewDW->transferFrom(subNewDW,         // uvel_FC
                    lb->uvel_FCMELabel,      patch_sub,  all_matls_sub,replace); 
  ParentNewDW->transferFrom(subNewDW,         // vvel_FC
                    lb->vvel_FCMELabel,      patch_sub,  all_matls_sub,replace); 
  ParentNewDW->transferFrom(subNewDW,         // wvel_FC
                    lb->wvel_FCMELabel,      patch_sub,  all_matls_sub,replace); 
  ParentNewDW->transferFrom(subNewDW,        // vol_fracX_FC
                    lb->vol_fracX_FCLabel,   patch_sub, all_matls_sub,replace); 
  ParentNewDW->transferFrom(subNewDW,         // vol_fracY_FC
                    lb->vol_fracY_FCLabel,   patch_sub, all_matls_sub,replace); 
  ParentNewDW->transferFrom(subNewDW,         // vol_fracZ_FC
                    lb->vol_fracZ_FCLabel,   patch_sub, all_matls_sub,replace);                 
    
  //__________________________________
  //  Turn scrubbing back on
  ParentOldDW->setScrubbing(ParentOldDW_scrubmode);
  ParentNewDW->setScrubbing(ParentNewDW_scrubmode);  

  //__________________________________
  // clean up memory  
  if(press_matlSet->removeReference()){
    delete press_matlSet;
  }
}




/*______________________________________________________________________
 Function~  ICE::scheduleCoarsen_imp_delP--
 Purpose:  After the implicit pressure solve is performed on all levels 
 you need to project/coarsen the fine level solution onto the coarser levels
 _____________________________________________________________________*/
void ICE::scheduleCoarsen_imp_delP(SchedulerP& sched, 
                                  const LevelP& coarseLevel,
                                  const MaterialSubset* press_matl)
{                                                                          
  cout_doing << "ICE::scheduleCoarsen_imp_delP\t\t\t\tL-" 
             << coarseLevel->getIndex() << endl;

  Task* t = scinew Task("ICE::coarsen_imp_delP",
                  this, &ICE::coarsen_imp_delP);

  Task::DomainSpec oims = Task::OutOfDomain;  //outside of ice matlSet.
  Ghost::GhostType  gn = Ghost::None;

  t->requires(Task::NewDW, lb->press_CCLabel,
              0, Task::FineLevel,  press_matl,oims, gn, 0);

  t->modifies(lb->press_CCLabel, d_press_matl, oims);        

  sched->addTask(t, coarseLevel->eachPatch(), d_sharedState->allICEMaterials());
}

/* _____________________________________________________________________
 Function~  ICE::Coarsen_imp_delP
 _____________________________________________________________________  */
void ICE::coarsen_imp_delP(const ProcessorGroup*,
                          const PatchSubset* coarsePatches,
                          const MaterialSubset* matls,
                          DataWarehouse*,
                          DataWarehouse* new_dw)
{ 
  const Level* coarseLevel = getLevel(coarsePatches);
  const Level* fineLevel = coarseLevel->getFinerLevel().get_rep();
  
  for(int p=0;p<coarsePatches->size();p++){
    const Patch* coarsePatch = coarsePatches->get(p);
    
    cout_doing << "Doing Coarsen_imp_delP on patch "
               << coarsePatch->getID() << "\t ICE \tL-" <<coarseLevel->getIndex()<< endl;
    CCVariable<double> imp_delP;                  
    new_dw->getModifiable(imp_delP, lb->imp_delPLabel, 0, coarsePatch);
   
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

      constCCVariable<double> fine_imp_delP;
      new_dw->getRegion(fine_imp_delP,  lb->imp_delPLabel, 0, fineLevel, fl, fh);

      //cout << " fineToCoarseOperator: finePatch "<< fl << " " << fh 
      //         << " coarsePatch "<< cl << " " << ch << endl;

      IntVector refinementRatio = fineLevel->getRefinementRatio();

      // iterate over coarse level cells
      for(CellIterator iter(cl, ch); !iter.done(); iter++){
        IntVector c = *iter;
        double imp_delP_tmp(0.0);
        IntVector fineStart = coarseLevel->mapCellToFiner(c);

        // for each coarse level cell iterate over the fine level cells   
        for(CellIterator inside(IntVector(0,0,0),refinementRatio );
                                            !inside.done(); inside++){
          IntVector fc = fineStart + *inside;

          imp_delP_tmp += fine_imp_delP[fc] * fineCellVol;
        }
        imp_delP[c] =imp_delP_tmp / coarseCellVol;
      }
    }
  }
}

/* _____________________________________________________________________
 Function~  ICE::zeroMatrixUnderFinePatches
 _____________________________________________________________________  */
void ICE::zeroMatrix_RHS_UnderFinePatches(const PatchSubset* coarsePatches,
                                      DataWarehouse* new_dw)
{ 
  const Level* coarseLevel = getLevel(coarsePatches);
  const Level* fineLevel = coarseLevel->getFinerLevel().get_rep();
  
  for(int p=0;p<coarsePatches->size();p++){
    const Patch* coarsePatch = coarsePatches->get(p);
    
    cout_doing << "Doing CoarsenPressure on patch "
               << coarsePatch->getID() << "\t ICE \tL-" <<coarseLevel->getIndex()<< endl;
    
    CCVariable<Stencil7> A;
    CCVariable<double> rhs;                  
    new_dw->getModifiable(A, lb->matrixLabel, 0, coarsePatch);
    new_dw->getModifiable(rhs,lb->rhsLabel,   0, coarsePatch);
   
    Level::selectType finePatches;
    coarsePatch->getFineLevelPatches(finePatches);

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
      IntVector refinementRatio = fineLevel->getRefinementRatio();

      // iterate over coarse level cells
      for(CellIterator iter(cl, ch); !iter.done(); iter++){
        IntVector c = *iter;
        A[c].e= 0;
        A[c].w= 0; 
        A[c].n= 0; 
        A[c].s= 0; 
        A[c].t= 0; 
        A[c].b= 0;
        A[c].p= 1;
        rhs[c] = 0;
      }
    }
  }
}

/*___________________________________________________________________
 Function~  ICE::matrixCoarseLevelIterator--  
 Purpose:  returns the iterator  THIS IS COMPILCATED AND CONFUSING
_____________________________________________________________________*/
void ICE::matrixCoarseLevelIterator(Patch::FaceType patchFace,
                                       const Patch* coarsePatch,
                                       const Patch* finePatch,
                                       const Level* fineLevel,
                                       CellIterator& iter,
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

  string name = finePatch->getFaceName(patchFace);
  IntVector offset = finePatch->faceDirection(patchFace);

  if(name == "xminus" || name == "yminus" || name == "zminus"){
    l += offset;
  }
  if(name == "xplus" || name == "yplus" || name == "zplus"){
    l += offset;
    h += offset;
  }

  l = Max(l, coarsePatch->getLowIndex());
  h = Min(h, coarsePatch->getHighIndex());
  
  iter=CellIterator(l,h);
  isRight_CP_FP_pair = false;
  if ( coarsePatch->containsCell(l) ){
    isRight_CP_FP_pair = true;
  }
  
  if (cout_dbg.active()) {
    cout_dbg << "refluxCoarseLevelIterator: face "<< patchFace
             << " finePatch " << finePatch->getID()
             << " coarsePatch " << coarsePatch->getID()
             << " [CellIterator at " << iter.begin() << " of " << iter.end() << "] "
             << " does this coarse patch own the face centered variable "
             << isRight_CP_FP_pair << endl; 
  }
}
/*___________________________________________________________________
 Function~  ICE::schedule_matrixBC_CFI_coarsePatch--  
_____________________________________________________________________*/
void ICE::schedule_matrixBC_CFI_coarsePatch(const LevelP& coarseLevel,
                                            SchedulerP& sched)
{
  cout_doing << d_myworld->myrank() 
             << " ICE::schedule_Adjust_matrix_coarseFineInterfaces\t\tL-" 
             << coarseLevel->getIndex() <<endl;
             
  Task* task = scinew Task("matrixBC_CFI_coarsePatch",
                this, &ICE::matrixBC_CFI_coarsePatch);
  
  task->modifies(lb->matrixLabel);

  sched->addTask(task, coarseLevel->eachPatch(), d_sharedState->allMaterials()); 
}
/*___________________________________________________________________
 Function~  ICE::matrixBC_CFI_coarsePatch--
 Purpose~   Along each coarseFine interface (on finePatches)
 set the stencil weight
 
   A.p = beta[c] -
          (A.n + A.s + A.e + A.w + A.t + A.b);
   LHS       
   A.p*delP - (A.e*delP_e + A.w*delP_w + A.n*delP_n + A.s*delP_s 
             + A.t*delP_t + A.b*delP_b )
             
Implementation:  For each coarse patch, loop over the overlapping fine level
patches.  If a fine patch has a CFI then set the stencil weights on the coarse level  
_____________________________________________________________________*/
void ICE::matrixBC_CFI_coarsePatch(const ProcessorGroup*,
                                   const PatchSubset* coarsePatches,
                                   const MaterialSubset* matls,
                                   DataWarehouse*,
                                   DataWarehouse* new_dw)
{
  const Level* coarseLevel = getLevel(coarsePatches);
  const Level* fineLevel = coarseLevel->getFinerLevel().get_rep();
  
  cout_doing << d_myworld->myrank() 
             << " Doing adjust_matrix_coarseFineInterfaces \t\t\t ICE L-"
             <<coarseLevel->getIndex();
  
  for(int c_p=0;c_p<coarsePatches->size();c_p++){  
    const Patch* coarsePatch = coarsePatches->get(c_p);
    cout_doing << "  patch " << coarsePatch->getID()<< endl;
         
    CCVariable<Stencil7> A_coarse;
    new_dw->getModifiable(A_coarse, lb->matrixLabel, 0, coarsePatch);

    Level::selectType finePatches;
    coarsePatch->getFineLevelPatches(finePatches); 

    for(int i=0; i < finePatches.size();i++){  
      const Patch* finePatch = finePatches[i];        

      if(finePatch->hasCoarseFineInterfaceFace() ){

        //__________________________________
        // Iterate over coarsefine interface faces
        vector<Patch::FaceType>::const_iterator iter;  
        for (iter  = finePatch->getCoarseFineInterfaceFaces()->begin(); 
             iter != finePatch->getCoarseFineInterfaceFaces()->end(); ++iter){
          Patch::FaceType patchFace = *iter;

          // determine the iterator on the coarse level.
          CellIterator c_iter(IntVector(-8,-8,-8),IntVector(-9,-9,-9));
          bool isRight_CP_FP_pair;
          matrixCoarseLevelIterator( patchFace,coarsePatch, finePatch, fineLevel,
                                     c_iter ,isRight_CP_FP_pair);

          // eject if this is not the right coarse/fine patch pair
          if (isRight_CP_FP_pair == false ){
            return;
          };   

          // The matix element is opposite
          // of the patch face
          int element = patchFace;
          if(patchFace == Patch::xminus || 
             patchFace == Patch::yminus || 
             patchFace == Patch::zminus){
            element += 1;  // e, n, t 
          }
          if(patchFace == Patch::xplus || 
             patchFace == Patch::yplus || 
             patchFace == Patch::zplus){
            element -= 1;   // w, s, b
          }
          
          for(; !c_iter.done(); c_iter++){
            IntVector c = *c_iter;
            A_coarse[c].p = A_coarse[c].p + A_coarse[c][element];
            A_coarse[c][element] = 0.0;
          }
        }  // coarseFineInterface faces
      }  // patch has a coarseFineInterface
    }  // finePatch loop 

    //__________________________________
    //  Print Data
#if 0
    if(switchDebug_AMR_reflux){ 
      ostringstream desc;     
      desc << "Reflux_applyCorrection_Mat_" <<"_patch_"<< coarsePatch->getID();
      printData(indx, coarsePatch,   1, desc.str(), "rho_CC",   rho_CC);
    }
#endif
  }  // course patch loop 
}

/*___________________________________________________________________
 Function~  ICE::schedule_matrixBC_CFI_finePatch--  
_____________________________________________________________________*/
void ICE::schedule_matrixBC_CFI_finePatch(const LevelP& fineLevel,
                                            SchedulerP& sched)
{
  cout_doing << d_myworld->myrank() 
             << " ICE::schedule_matrixBC_CFI_finePatch\t\tL-" 
             << fineLevel->getIndex() <<endl;
             
  Task* task = scinew Task("matrixBC_CFI_finePatch",
                this, &ICE::matrixBC_CFI_finePatch);
  
  task->modifies(lb->matrixLabel);

  sched->addTask(task, fineLevel->eachPatch(), d_sharedState->allMaterials()); 
}
/*___________________________________________________________________ 
 Function~  matrixBC_CFI_finePatch--      
 Purpose~   Along each coarseFine interface (on finePatches)
 set the stencil weight
 
 Naming convention
      +x -x +y -y +z -z
       e, w, n, s, t, b 
 
   A.p = beta[c] -
          (A.n + A.s + A.e + A.w + A.t + A.b);
   LHS       
   A.p*delP - (A.e*delP_e + A.w*delP_w + A.n*delP_n + A.s*delP_s 
             + A.t*delP_t + A.b*delP_b )
             
 Suppose the x- face then you must add A.w to
 both A.p and set A.w = 0.
___________________________________________________________________*/
void ICE::matrixBC_CFI_finePatch(const ProcessorGroup*,
                                 const PatchSubset* finePatches,
                                 const MaterialSubset*,
                                 DataWarehouse*,
                                 DataWarehouse* new_dw)       
{ 

  for(int p=0;p<finePatches->size();p++){  
    const Patch* finePatch = finePatches->get(p);
      
    if(finePatch->hasCoarseFineInterfaceFace() ){    
      cout_dbg << *finePatch << " ";
      finePatch->printPatchBCs(cout_dbg);
      CCVariable<Stencil7> A;
      new_dw->getModifiable(A,   lb->matrixLabel, 0, finePatch);
      //__________________________________
      // Iterate over coarsefine interface faces
      vector<Patch::FaceType>::const_iterator iter;  
      for (iter  = finePatch->getCoarseFineInterfaceFaces()->begin(); 
           iter != finePatch->getCoarseFineInterfaceFaces()->end(); ++iter){
        Patch::FaceType face = *iter;

        CellIterator f_iter=finePatch->getFaceCellIterator(face, "alongInteriorFaceCells");
        int f = face;

        for(; !f_iter.done(); f_iter++){
          IntVector c = *f_iter;
          A[c].p = A[c].p + A[c][f];
          A[c][f] = 0.0;
        }
      }
    }  // if finePatch has a CFI
  }
}
