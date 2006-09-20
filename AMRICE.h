#ifndef Packages_Uintah_CCA_Components_ICE_AMRICE_h
#define Packages_Uintah_CCA_Components_ICE_AMRICE_h

#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <Packages/Uintah/Core/Grid/AMR.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Variables/PerPatch.h>
#include <Packages/Uintah/CCA/Components/ICE/ICE.h>
#include <Packages/Uintah/CCA/Components/Regridder/PerPatchVars.h>

#include <Packages/Uintah/CCA/Components/ICE/share.h>

//#define REFLUX_DBG 
#undef REFLUX_DBG
#define is_rightFace_variable(face,var) ( ((face == "xminus" || face == "xplus") && var == "scalar-f") ?1:0  )

namespace Uintah {
  class SCISHARE AMRICE : public ICE{
  public:
    AMRICE(const ProcessorGroup* myworld);
    virtual ~AMRICE();
    
    virtual void problemSetup(const ProblemSpecP& params, 
                              const ProblemSpecP& materials_ps, 
                              GridP& grid, SimulationStateP& sharedState);
                              
    virtual void scheduleInitialize(const LevelP& level,
                                    SchedulerP& sched);
                                    
    virtual void scheduleRefineInterface(const LevelP& fineLevel,
                                         SchedulerP& scheduler,
                                         bool needCoarseOld, 
                                         bool needCoarseNew);
                                         
    virtual void scheduleRefine (const PatchSet* patches, 
                                 SchedulerP& sched); 
    
    virtual void scheduleCoarsen(const LevelP& coarseLevel, 
                                 SchedulerP& sched);


    virtual void scheduleInitialErrorEstimate(const LevelP& coarseLevel,
                                              SchedulerP& sched);
                                               
    virtual void scheduleErrorEstimate(const LevelP& coarseLevel,
                                       SchedulerP& sched);
  protected:
    void refineCoarseFineBoundaries(const Patch* patch,
                                    CCVariable<double>& val,
                                    DataWarehouse* new_dw,
                                    const VarLabel* label,
                                    int matl, 
                                    double factor);
                
    void refineCoarseFineBoundaries(const Patch* patch,
                                    CCVariable<Vector>& val,
                                    DataWarehouse* new_dw,
                                    const VarLabel* label,
                                    int matl, 
                                    double factor);
                                  
    void addRefineDependencies(Task* task, 
                               const VarLabel* var,
                               Task::DomainSpec DS,
                               const MaterialSubset* matls,
                               bool needCoarseOld, 
                               bool needCoarseNew);

  private:

    void initialize(const ProcessorGroup*,
                    const PatchSubset* patches, 
                    const MaterialSubset* matls,
                    DataWarehouse* old_dw, 
                    DataWarehouse* new_dw);
                    
    template<class T>
    void refine_CF_interfaceOperator(const Patch* patch, 
                                     const Level* fineLevel,
                                     const Level* coarseLevel,
                                     CCVariable<T>& Q, 
                                     const VarLabel* label,
                                     double subCycleProgress_var,
                                     int matl, 
                                     DataWarehouse* fine_new_dw,
                                     DataWarehouse* coarse_old_dw,
                                     DataWarehouse* coarse_new_dw);
                    
    void refineCoarseFineInterface(const ProcessorGroup*,
                                   const PatchSubset* patches,
                                   const MaterialSubset* matls,
                                   DataWarehouse*, 
                                   DataWarehouse* new_dw);

    void scheduleSetBC_FineLevel(const PatchSet* patches,
                                 SchedulerP& scheduler);
                                                                    
    void setBC_FineLevel(const ProcessorGroup*,
                         const PatchSubset* patches,              
                         const MaterialSubset*,                   
                         DataWarehouse* fine_old_dw,              
                         DataWarehouse* fine_new_dw);     
                                   
    void iteratorTest(const Patch* finePatch,
                      const Level* fineLevel,
                      const Level* coarseLevel,
                      DataWarehouse* new_dw);
                      
    void refine(const ProcessorGroup*,
                const PatchSubset* patches,
                const MaterialSubset* matls,
                DataWarehouse*,
                DataWarehouse* new_dw);

    template<class T>
    void CoarseToFineOperator(CCVariable<T>& q_CC,
                              const VarLabel* varLabel,
                              const int indx,
                              DataWarehouse* new_dw,
                              double ratio,
                              const Patch* finePatch,
                              const Level* fineLevel,
                              const Level* coarseLevel);
                         
    template<class T>
    void fineToCoarseOperator(CCVariable<T>& q_CC,
                              const string& quantity,
                              const VarLabel* varLabel,
                              const int indx,
                              DataWarehouse* new_dw,
                              const Patch* coarsePatch,
                              const Level* coarseLevel,
                              const Level* fineLevel);
                                  
    void coarsen(const ProcessorGroup*,
                 const PatchSubset* patches,
                 const MaterialSubset* matls,
                 DataWarehouse*, DataWarehouse* new_dw);
 
    //__________________________________
    //    refluxing
    void scheduleReflux_computeCorrectionFluxes(const LevelP& coarseLevel,
                                                SchedulerP& sched);
                                                    
    void reflux_computeCorrectionFluxes(const ProcessorGroup*,
                                        const PatchSubset* coarsePatches,
                                        const MaterialSubset* matls,
                                        DataWarehouse*,
                                        DataWarehouse* new_dw); 
    
    void scheduleReflux_applyCorrection(const LevelP& coarseLevel,
                                        SchedulerP& sched);
                                        
    void reflux_applyCorrectionFluxes(const ProcessorGroup*,
                                      const PatchSubset* coarsePatches,
                                      const MaterialSubset* matls,
                                      DataWarehouse*,
                                      DataWarehouse* new_dw);
    // bullet proofing
    void reflux_BP_zero_CFI_cells(const ProcessorGroup*,
                                      const PatchSubset* finePatches,
                                      const MaterialSubset*,
                                       DataWarehouse*,
                                       DataWarehouse*);
                                       
    void reflux_BP_count_CFI_cells(const ProcessorGroup*,
                                   const PatchSubset* coarsePatches,
                                   const MaterialSubset*,
                                   DataWarehouse*,
                                   DataWarehouse*);
                                                           
    void reflux_BP_check_CFI_cells(const ProcessorGroup*,
                                   const PatchSubset* finePatches,
                                   const MaterialSubset*,
                                   DataWarehouse*,
                                   DataWarehouse*,
                                   string description);
                 
    void errorEstimate(const ProcessorGroup*,
                       const PatchSubset* patches,
                       const MaterialSubset* matls,
                       DataWarehouse*,
                       DataWarehouse* new_dw,
                       bool initial);
                       
    void compute_Mag_gradient( constCCVariable<double>& q_CC,
                               CCVariable<double>& mag_grad_q_CC,                   
                               const Patch* patch);
                               
    void compute_Mag_Divergence( constCCVariable<Vector>& q_CC,
                                 CCVariable<double>& mag_div_q_CC,                   
                                 const Patch* patch);

    void set_refineFlags( constCCVariable<double>& q_CC_grad,
                          double threshold,
                          CCVariable<int>& refineFlag,
                          PerPatch<PatchFlagP>& refinePatchFlag,
                          const Patch* patch);                                                  
    AMRICE(const AMRICE&);
    AMRICE& operator=(const AMRICE&);
    
    //__________________________________
    // refinement criteria threshold knobs
    struct thresholdVar {
      string name;
      int matl;
      double value;
    };
    vector<thresholdVar> d_thresholdVars;
    
    bool d_regridderTest;
    int d_orderOfInterpolation;    
  };

static DebugStream cout_dbg("AMRICE_DBG", false);
/*_____________________________________________________________________
 Function~  ICE::refluxOperator_applyCorrectionFluxes
 Purpose~   
_____________________________________________________________________*/
template<class T>
void ICE::refluxOperator_applyCorrectionFluxes(                             
                              CCVariable<T>& q_CC_coarse,
                              const string& varLabel,
                              const int indx,
                              const Patch* coarsePatch,
                              const Patch* finePatch,
                              const Level* coarseLevel,
                              const Level* fineLevel,
                              DataWarehouse* new_dw,
                              const int one_zero)
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
                               c_iter ,c_FC_offset,isRight_CP_FP_pair,
                               "applyRefluxCorrection");
    
                               
    if (isRight_CP_FP_pair ){  // if the right coarse/fine patch pair

#ifdef REFLUX_DBG    
      string name = finePatch->getFaceName(patchFace);
      IntVector half  = (c_iter.end() - c_iter.begin() )/IntVector(2,2,2) + c_iter.begin();
  
      if(is_rightFace_variable(name,varLabel) ){
        cout << " ------------ refluxOperator_applyCorrectionFluxes " << varLabel<< endl; 
        cout << "coarseLevel iterator " << c_iter.begin() << " " << c_iter.end() << endl;
        cout << finePatch->getFaceName(patchFace)<<  " coarsePatch " << *coarsePatch << endl;
        cout << "      finePatch   " << *finePatch << endl;
      }
#endif 

      //__________________________________
      // Add fine patch face fluxes correction to the coarse cells
      // c_CC:    coarse level cell center index
      // c_FC:    coarse level face center index
      int count = finePatch->getFaceMark(0, patchFace);
      
      if(patchFace == Patch::xminus || patchFace == Patch::xplus){
        for(; !c_iter.done(); c_iter++){
          IntVector c_CC = *c_iter;
          IntVector c_FC = c_CC + c_FC_offset;
          T q_CC_coarse_org = q_CC_coarse[c_CC];
          q_CC_coarse[c_CC] += Q_X_coarse_flux[c_FC];
                
          count += one_zero;                       // keep track of how that face                        
          finePatch->setFaceMark(0,patchFace,count); // has been touched
          
#ifdef REFLUX_DBG
          if (c_CC.y() == half.y() && c_CC.z() == half.z() && is_rightFace_variable(name,varLabel) ) {
            cout << " \t c_CC " << c_CC  << " c_FC " << c_FC 
                 << " q_CC_org " << q_CC_coarse_org
                 << " correction " << Q_X_coarse_flux[c_FC]
                 << " q_CC_corrected " << q_CC_coarse[c_CC] << endl;
            cout << "" << endl;
          }
#endif          
          
        }
      }
      if(patchFace == Patch::yminus || patchFace == Patch::yplus){
        for(; !c_iter.done(); c_iter++){
          IntVector c_CC = *c_iter;
          IntVector c_FC = c_CC + c_FC_offset;
          T q_CC_coarse_org = q_CC_coarse[c_CC];
          q_CC_coarse[c_CC] += Q_Y_coarse_flux[c_FC];
          
          count += one_zero;                              
          finePatch->setFaceMark(0,patchFace,count);
          
#ifdef REFLUX_DBG
          if (c_CC.x() == half.x() && c_CC.z() == half.z() && is_rightFace_variable(name,varLabel) ) {
            cout << " \t c_CC " << c_CC  << " c_FC " << c_FC 
                 << " q_CC_org " << q_CC_coarse_org
                 << " correction " << Q_Y_coarse_flux[c_FC]
                 << " q_CC_corrected " << q_CC_coarse[c_CC] << endl;
            cout << "" << endl;
          }
#endif
        }
      }
      if(patchFace == Patch::zminus || patchFace == Patch::zplus){
        for(; !c_iter.done(); c_iter++){
          IntVector c_CC = *c_iter;
          IntVector c_FC = c_CC + c_FC_offset;             
          T q_CC_coarse_org = q_CC_coarse[c_CC];
          q_CC_coarse[c_CC] += Q_Z_coarse_flux[c_FC];
          
          count += one_zero;                              
          finePatch->setFaceMark(0,patchFace,count);
          
#ifdef REFLUX_DBG
          if (c_CC.x() == half.x() && c_CC.y() == half.y() && is_rightFace_variable(name,varLabel) ) {
            cout << " \t c_CC " << c_CC  << " c_FC " << c_FC 
                 << " q_CC_org " << q_CC_coarse_org
                 << " correction " << Q_Z_coarse_flux[c_FC]
                 << " q_CC_corrected " << q_CC_coarse[c_CC] << endl;
            cout << "" << endl;
          }
#endif
        }
      }
    }  // is the right coarse/fine patch pair
  }  // coarseFineInterface faces
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
  IntVector refineRatio = fineLevel->getRefinementRatio();
  //__________________________________
  // Iterate over coarsefine interface faces
  vector<Patch::FaceType>::const_iterator iter;  
  for (iter  = patch->getCoarseFineInterfaceFaces()->begin(); 
       iter != patch->getCoarseFineInterfaceFaces()->end(); ++iter){
    Patch::FaceType face = *iter;

    //__________________________________
    // fine level hi & lo cell iter limits
    // coarselevel hi and low index

    IntVector cl, ch, fl, fh;
    getCoarseFineFaceRange(patch, coarseLevel, face, d_orderOfInterpolation, cl, ch, fl, fh);

    cout_dbg<< " face " << face << " refineRatio "<< refineRatio
            << " BC type " << patch->getBCType(face)
            << " FineLevel iterator" << fl << " " << fh 
            << " \t coarseLevel iterator " << cl << " " << ch <<endl;

    //__________________________________
    // subCycleProgress_var near 1.0 
    //  interpolation using the coarse_new_dw data
    if(subCycleProgress_var > 1-1.e-10){ 
     constCCVariable<varType> q_NewDW;
     coarse_new_dw->getRegion(q_NewDW, label, matl, coarseLevel,
                           cl, ch);
     selectInterpolator(q_NewDW, d_orderOfInterpolation, coarseLevel, 
                        fineLevel, refineRatio, fl,fh, Q);
    } else {    

    //__________________________________
    // subCycleProgress_var somewhere between 0 or 1
    //  interpolation from both coarse new and old dw 
      constCCVariable<varType> q_OldDW, q_NewDW;
      coarse_old_dw->getRegion(q_OldDW, label, matl, coarseLevel, cl, ch);
      coarse_new_dw->getRegion(q_NewDW, label, matl, coarseLevel, cl, ch);

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
  }  // face loop

  //____ B U L L E T   P R O O F I N G_______ 
  // All values must be initialized at this point
  // Note only check patches that aren't on the edge of the domain
  
  if(subCycleProgress_var > 1-1.e-10 ){  
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
  IntVector cl, ch, fl, fh;
  getCoarseLevelRange(finePatch, coarseLevel, cl, ch, fl, fh, 1);

  cout_dbg <<" coarseToFineOperator: " << varLabel->getName()
           <<" finePatch  "<< finePatch->getID() << " fl " << fl << " fh " << fh
           <<" coarseRegion " << cl << " " << ch <<endl;
  
  constCCVariable<T> coarse_q_CC;
  new_dw->getRegion(coarse_q_CC, varLabel, indx, coarseLevel, cl, ch);
  
  selectInterpolator(coarse_q_CC, d_orderOfInterpolation, coarseLevel, fineLevel,
                     refineRatio, fl, fh,q_CC);
  
  //____ B U L L E T   P R O O F I N G_______ 
  // All fine patch interior values must be initialized at this point
  // ignore BP if a timestep restart has already been requested
  bool tsr = new_dw->timestepRestarted();
  
  IntVector badCell;
  CellIterator iter=finePatch->getCellIterator();
  if( isEqual<T>(T(d_EVIL_NUM),iter,q_CC, badCell) && !tsr ){
    ostringstream warn;
    warn <<"ERROR AMRICE::Refine Task:CoarseToFineOperator "
         << "detected an uninitialized variable "<< varLabel->getName()
         << " " << badCell << " Patch " << finePatch->getID() 
         << " Level idx "<<fineLevel->getIndex()<<"\n ";
    throw InvalidValue(warn.str(), __FILE__, __LINE__);
  }
}


/*_____________________________________________________________________
 Function~  AMRICE::fineToCoarseOperator--
 Purpose~   averages the interior fine patch data onto the coarse patch
_____________________________________________________________________*/
template<class T>
void AMRICE::fineToCoarseOperator(CCVariable<T>& q_CC,
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
                          
  for(int i=0;i<finePatches.size();i++){
    const Patch* finePatch = finePatches[i];

    IntVector cl, ch, fl, fh;
    getFineLevelRange(coarsePatch, finePatch, cl, ch, fl, fh);

    if (fh.x() <= fl.x() || fh.y() <= fl.y() || fh.z() <= fl.z()) {
      continue;
    }
    
    constCCVariable<T> fine_q_CC;
    new_dw->getRegion(fine_q_CC,  varLabel, indx, fineLevel, fl, fh, false);

    cout_dbg << " fineToCoarseOperator: finePatch "<< fl << " " << fh 
             << " coarsePatch "<< cl << " " << ch << endl;
             
    IntVector r_Ratio = fineLevel->getRefinementRatio();
    
    double inv_RR = 1.0;    
    
    if(quantity == "non-conserved"){
      inv_RR = 1.0/( (double)(r_Ratio.x() * r_Ratio.y() * r_Ratio.z()) );
    }

    T zero(0.0);
    // iterate over coarse level cells
    for(CellIterator iter(cl, ch); !iter.done(); iter++){
      IntVector c = *iter;
      T q_CC_tmp(zero);
      IntVector fineStart = coarseLevel->mapCellToFiner(c);
    
      // for each coarse level cell iterate over the fine level cells   
      for(CellIterator inside(IntVector(0,0,0),r_Ratio );
                                          !inside.done(); inside++){
        IntVector fc = fineStart + *inside;        
        q_CC_tmp += fine_q_CC[fc];
      }
                         
      q_CC[c] =q_CC_tmp * inv_RR;
    }
  }
  cout_dbg.setActive(false);// turn off the switch for cout_dbg
}
 

/*_____________________________________________________________________
 Function~  AMRICE::refluxOperator_computeCorrectionFluxes--
 Purpose~  Note this method is needed by AMRICE.cc and impAMRICE.cc thus
 it's part of the ICE object 
_____________________________________________________________________*/
template<class T>
void ICE::refluxOperator_computeCorrectionFluxes( 
                              const string& fineVarLabel,
                              const int indx,
                              const Patch* coarsePatch,
                              const Patch* finePatch,
                              const Level* coarseLevel,
                              const Level* fineLevel,
                              DataWarehouse* new_dw,
                              const int one_zero)
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

  IntVector r_Ratio = fineLevel->getRefinementRatio();
  
  // coeff accounts for the different cell sizes on the different levels
  double coeff = ( (double)r_Ratio.x() * r_Ratio.y() * r_Ratio.z() );
  
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
                               c_iter ,c_FC_offset,isRight_CP_FP_pair,
                               "computeRefluxCorrection");

    // 
    if (isRight_CP_FP_pair){

      // Offset for the fine cell loop (fineStart)
      // shift (+-refinement ratio) for x-, y-, z- finePatch faces
      // shift 0 cells              for x+, y+, z+ finePatchFaces

      string name = finePatch->getFaceName(patchFace);
      IntVector offset = finePatch->faceDirection(patchFace);
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
#ifdef REFLUX_DBG
  IntVector half  = (c_iter.end() - c_iter.begin() )/IntVector(2,2,2) + c_iter.begin();
  if(is_rightFace_variable(name,fineVarLabel)){
    cout << " ------------ refluxOperator_computeCorrectionFluxes " << fineVarLabel<< endl;   
    cout << "coarseLevel iterator " << c_iter.begin() << " " << c_iter.end() << endl;
    cout <<name <<  " coarsePatch " << *coarsePatch << endl;
    cout << "      finePatch   " << *finePatch << endl;
  }
#endif 
/*===========TESTING==========`*/
      //__________________________________
      // Add fine patch face fluxes to the coarse cells
      // c_CC f_CC:    coarse/fine level cell center index
      // c_FC f_FC:    coarse/fine level face center index
      int count = finePatch->getFaceMark(0, patchFace);
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

           Q_X_coarse_flux[c_FC] = ( c_FaceNormal*Q_X_coarse_flux_org[c_FC] + coeff* f_FaceNormal*sum_fineLevelFlux);
           //Q_X_coarse_flux[c_FC] = ( Q_X_coarse_flux_org[c_FC] - coeff* sum_fineLevelFlux);

           count += one_zero;                              
           finePatch->setFaceMark(0,patchFace,count);
/*`==========TESTING==========*/
#ifdef REFLUX_DBG
        if (c_CC.y() == half.y() && c_CC.z() == half.z() && is_rightFace_variable(name,fineVarLabel) ) {
          cout << " \t c_CC " << c_CC  << " c_FC " << c_FC 
               << " coarseLevelFlux " << c_FaceNormal*Q_X_coarse_flux_org[c_FC]
               << " sum_fineLevelflux " << coeff* f_FaceNormal *sum_fineLevelFlux
               << " correction " << Q_X_coarse_flux[c_FC]<< endl;
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

           Q_Y_coarse_flux[c_FC] = (c_FaceNormal*Q_Y_coarse_flux_org[c_FC] + coeff*f_FaceNormal*sum_fineLevelFlux);

           count += one_zero;                       // keep track of how many times              
           finePatch->setFaceMark(0,patchFace,count); // the face has been touched
/*`==========TESTING==========*/
#ifdef REFLUX_DBG
        if (c_CC.x() == half.x() && c_CC.z() == half.z() && is_rightFace_variable(name,fineVarLabel)) {
          cout << " \t c_CC " << c_CC  << " c_FC " << c_FC 
               << " coarseLevelFlux " << c_FaceNormal*Q_Y_coarse_flux_org[c_FC]
               << " sum_fineLevelflux " << coeff* f_FaceNormal *sum_fineLevelFlux
               << " correction " << Q_Y_coarse_flux[c_FC] << endl;
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
                              
           Q_Z_coarse_flux[c_FC] = (c_FaceNormal*Q_Z_coarse_flux_org[c_FC] + coeff*f_FaceNormal*sum_fineLevelFlux);
           
            count += one_zero;                              
           finePatch->setFaceMark(0,patchFace,count);
/*`==========TESTING==========*/
#ifdef REFLUX_DBG
        if (c_CC.x() == half.x() && c_CC.y() == half.y() && is_rightFace_variable(name,fineVarLabel)) {
          cout << " \t c_CC " << c_CC  << " c_FC " << c_FC 
                << " coarseLevelFlu " << c_FaceNormal*Q_Z_coarse_flux_org[c_FC]
               << " sum_fineLevelflux " << coeff* f_FaceNormal *sum_fineLevelFlux
               << " correction " << Q_Z_coarse_flux[c_FC]<< endl;
          cout << "" << endl;
        }
#endif 
/*===========TESTING==========`*/
        }
      }
    }  // is right coarse/fine patch pair
  }  // coarseFineInterface faces 
} // end refluxOperator_computeCorrectionFluxes()



}

#endif
