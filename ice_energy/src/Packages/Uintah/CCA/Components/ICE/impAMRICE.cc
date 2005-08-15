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

  MaterialSubset* one_matl = d_press_matl;
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

    if(d_impICE) {        //  I M P L I C I T

      scheduleSetupRHS(                     sched, patches,  one_matl, 
                                                             all_matls,
                                                             false);

      scheduleImplicitPressureSolve(         sched, level,   patches,
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

    if(!d_impICE){         //  E X P L I C I T
      scheduleComputeDelPressAndUpdatePressCC(sched, patches,d_press_matl,     
                                                             ice_matls_sub,  
                                                             mpm_matls_sub,  
                                                             all_matls);     
    }
  }
//______________________________________________________________________
// MULTI-LEVEL PRESSURE SOLVE AND COARSEN


  if(d_doAMR){
    for(int L = maxLevel-1; L> 0; L--){ // from finer to coarser levels
      LevelP coarseLevel = grid->getLevel(L-1);
      scheduleCoarsenPressure(  sched,  coarseLevel,  d_press_matl);
    }
  }

//______________________________________________________________________

  for(int L = 0; L<maxLevel; L++){
    LevelP level = grid->getLevel(L);
    const PatchSet* patches = level->eachPatch();
    
    if(!doICEOnLevel(level->getIndex())){
      return;
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



/*______________________________________________________________________
 Function~  ICE::scheduleCoarsenPressure--
 Purpose:  After the implicit pressure solve is performed on all levels 
 you need to project/coarsen the fine level solution onto the coarser level
 _____________________________________________________________________*/
void ICE::scheduleCoarsenPressure(SchedulerP& sched, 
                                  const LevelP& coarseLevel,
                                  const MaterialSubset* press_matl)
{
  //if (d_doAMR &&  d_impICE){ 
  if (d_doAMR){                                                                           
    cout_doing << "ICE::scheduleCoarsenPressure\t\t\t\tL-" 
               << coarseLevel->getIndex() << endl;

    Task* t = scinew Task("ICE::coarsenPressure",
                    this, &ICE::coarsenPressure);

    Task::DomainSpec oims = Task::OutOfDomain;  //outside of ice matlSet.
    Ghost::GhostType  gn = Ghost::None;

    t->requires(Task::NewDW, lb->press_CCLabel,
                0, Task::FineLevel,  press_matl,oims, gn, 0);
                
    t->modifies(lb->press_CCLabel, d_press_matl, oims);        

    sched->addTask(t, coarseLevel->eachPatch(), d_sharedState->allICEMaterials());
  }
}

/* _____________________________________________________________________
 Function~  ICE::CoarsenPressure
 _____________________________________________________________________  */
void ICE::coarsenPressure(const ProcessorGroup*,
                          const PatchSubset* coarsePatches,
                          const MaterialSubset* matls,
                          DataWarehouse*,
                          DataWarehouse* new_dw)
{ 
  const Level* coarseLevel = getLevel(coarsePatches);
  const Level* fineLevel = coarseLevel->getFinerLevel().get_rep();
  
  for(int p=0;p<coarsePatches->size();p++){
    const Patch* coarsePatch = coarsePatches->get(p);
    
    cout_doing << "Doing CoarsenPressure on patch "
               << coarsePatch->getID() << "\t ICE \tL-" <<coarseLevel->getIndex()<< endl;
    CCVariable<double> press_CC, notUsed;                  
    new_dw->getModifiable(press_CC, lb->press_CCLabel, 0, coarsePatch);
   
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

      constCCVariable<double> fine_press_CC;
      new_dw->getRegion(fine_press_CC,  lb->press_CCLabel, 0, fineLevel, fl, fh);

      //cout << " fineToCoarseOperator: finePatch "<< fl << " " << fh 
      //         << " coarsePatch "<< cl << " " << ch << endl;

      IntVector refinementRatio = fineLevel->getRefinementRatio();

      // iterate over coarse level cells
      for(CellIterator iter(cl, ch); !iter.done(); iter++){
        IntVector c = *iter;
        double press_CC_tmp(0.0);
        IntVector fineStart = coarseLevel->mapCellToFiner(c);

        // for each coarse level cell iterate over the fine level cells   
        for(CellIterator inside(IntVector(0,0,0),refinementRatio );
                                            !inside.done(); inside++){
          IntVector fc = fineStart + *inside;

          press_CC_tmp += fine_press_CC[fc] * fineCellVol;
        }
        press_CC[c] =press_CC_tmp / coarseCellVol;
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
