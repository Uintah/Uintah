/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

#ifndef Packages_Uintah_CCA_Components_ICE_AMRICE_h
#define Packages_Uintah_CCA_Components_ICE_AMRICE_h

#include <Core/Exceptions/InvalidValue.h>
#include <Core/Grid/AMR.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <CCA/Components/ICE/ICE.h>
#include <CCA/Components/Regridder/PerPatchVars.h>

//#define REFLUX_DBG 
#undef REFLUX_DBG
#define is_rightFace_variable(face,var) ( ((face == "xminus" || face == "xplus") && (var == "scalar-f" || var == "vol_frac")) ?1:0  )

namespace Uintah {
  class AMRICE : public ICE{
  public:
    AMRICE(const ProcessorGroup* myworld);
    virtual ~AMRICE();
    
    virtual void problemSetup(const ProblemSpecP& params, 
                              const ProblemSpecP& restart_prob_spec, 
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
                                       
    inline void clearFaceMarks(const int whichMap, const Patch* patch) {
      faceMarks_map[whichMap].erase(patch);
    }
    
  
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
                                  
    void scheduleRefineInterface_Variable(const LevelP& fineLevel,  
                                          SchedulerP& sched,        
                                          const VarLabel* var,      
                                          Task::MaterialDomainSpec DS,      
                                          const MaterialSet* matls, 
                                          bool needCoarseOld,       
                                          bool needCoarseNew);      

                              
    template<class T>
    void refluxOperator_computeCorrectionFluxes( 
                              const string& fineVarLabel,
                              const int indx,
                              const Patch* coarsePatch,
                              const Patch* finePatch,
                              const Level* coarseLevel,
                              const Level* fineLevel,
                              DataWarehouse* new_dw,
                              const int one_zero);
                              
    void refluxCoarseLevelIterator(Patch::FaceType patchFace,
                                   const Patch* coarsePatch,
                                   const Patch* finePatch,
                                   const Level* fineLevel,
                                   CellIterator& iter,
                                   IntVector& coarse_FC_offset,
                                   bool& CP_containsCell,
                                   const string& whichTask);
    template<class T>
    void refluxOperator_applyCorrectionFluxes(                             
                              CCVariable<T>& q_CC_coarse,
                              const string& fineVarLabel,
                              const int indx,
                              const Patch* coarsePatch,
                              const Patch* finePatch,
                              const Level* coarseLevel,
                              const Level* fineLevel,
                              DataWarehouse* new_dw,
                              const int one_zero);

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
            
    template<class T>                
    void refineCoarseFineInterface(const ProcessorGroup*,
                                   const PatchSubset* patches,
                                   const MaterialSubset* matls,
                                   DataWarehouse*, 
                                   DataWarehouse* new_dw,
                                   const VarLabel* variable);

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

    void set_refineFlags( constCCVariable<double>& q_CC_grad,
                          double threshold,
                          CCVariable<int>& refineFlag,
                          PerPatch<PatchFlagP>& refinePatchFlag,
                          const Patch* patch);
                          

    inline int getFaceMark(int whichMap, 
                           const Patch* patch, 
                           Patch::FaceType face)
    {
      ASSERT(whichMap>=0 && whichMap<2);
      return faceMarks_map[whichMap][patch][face];
    };

    inline void setFaceMark(int whichMap, 
                            const Patch* patch, 
                            Patch::FaceType face, 
                            int value) 
    {
      ASSERT(whichMap>=0 && whichMap<2);
      faceMarks_map[whichMap][patch][face]=value;
    };
                                                                  
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
    
    int d_orderOfInterpolation;         // Order of interpolation for interior fine patch
    int d_orderOf_CFI_Interpolation;    // order of interpolation at CFI.

    struct faceMarks {
      int marks[Patch::numFaces];
      int& operator[](Patch::FaceType face)
      {
        return marks[static_cast<int>(face)];
      }
      faceMarks()
      {
        marks[0]=0;
        marks[1]=0;
        marks[2]=0;
        marks[3]=0;
        marks[4]=0;
        marks[5]=0;
      }
    };
    map<const Patch*,faceMarks> faceMarks_map[2];
  };

static DebugStream cout_dbg("AMRICE_DBG", false);
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
                              DataWarehouse* new_dw,
                              const int one_zero)
{
  // form the fine patch flux label names
  string x_name = varLabel + "_X_FC_corr";
  string y_name = varLabel + "_Y_FC_corr";
  string z_name = varLabel + "_Z_FC_corr";
  
  // grab the varLabels
  VarLabel* xlabel = VarLabel::find(x_name);
  VarLabel* ylabel = VarLabel::find(y_name);
  VarLabel* zlabel = VarLabel::find(z_name);  

  if(xlabel == NULL || ylabel == NULL || zlabel == NULL){
    throw InternalError( "refluxOperator_applyCorrectionFluxes: variable label not found: " 
                          + x_name + " or " + y_name + " or " + z_name, __FILE__, __LINE__);
  }
  constSFCXVariable<T>  Q_X_coarse_corr;
  constSFCYVariable<T>  Q_Y_coarse_corr;
  constSFCZVariable<T>  Q_Z_coarse_corr;
  
  Ghost::GhostType  gac = Ghost::AroundCells;
  new_dw->get(Q_X_coarse_corr,  xlabel,indx, coarsePatch, gac,1);
  new_dw->get(Q_Y_coarse_corr,  ylabel,indx, coarsePatch, gac,1);
  new_dw->get(Q_Z_coarse_corr,  zlabel,indx, coarsePatch, gac,1); 
  
  //__________________________________
  // Iterate over coarsefine interface faces
  vector<Patch::FaceType> cf;
  finePatch->getCoarseFaces(cf);
  vector<Patch::FaceType>::const_iterator iter;  
  for (iter  = cf.begin(); iter != cf.end(); ++iter){
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
      int count = getFaceMark(0, finePatch, patchFace);
      
      if(patchFace == Patch::xminus || patchFace == Patch::xplus){
        for(; !c_iter.done(); c_iter++){
          IntVector c_CC = *c_iter;
          IntVector c_FC = c_CC + c_FC_offset;
#ifdef REFLUX_DBG
          T q_CC_coarse_org = q_CC_coarse[c_CC];
#endif          
          q_CC_coarse[c_CC] += Q_X_coarse_corr[c_FC];
                
          count += one_zero;                       // keep track of how that face                        
          setFaceMark(0, finePatch, patchFace, count); // has been touched
          
#ifdef REFLUX_DBG
          if (c_CC.y() == half.y() && c_CC.z() == half.z() && is_rightFace_variable(name,varLabel) ) {
            cout << " \t c_CC " << c_CC  << " c_FC " << c_FC 
                 << " q_CC_org " << q_CC_coarse_org
                 << " correction " << Q_X_coarse_corr[c_FC]
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
#ifdef REFLUX_DBG
          T q_CC_coarse_org = q_CC_coarse[c_CC];
#endif
          q_CC_coarse[c_CC] += Q_Y_coarse_corr[c_FC];
          
          count += one_zero;                              
          setFaceMark(0, finePatch, patchFace, count);
          
#ifdef REFLUX_DBG
          if (c_CC.x() == half.x() && c_CC.z() == half.z() && is_rightFace_variable(name,varLabel) ) {
            cout << " \t c_CC " << c_CC  << " c_FC " << c_FC 
                 << " q_CC_org " << q_CC_coarse_org
                 << " correction " << Q_Y_coarse_corr[c_FC]
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
#ifdef REFLUX_DBG
          T q_CC_coarse_org = q_CC_coarse[c_CC];
#endif
          q_CC_coarse[c_CC] += Q_Z_coarse_corr[c_FC];
          
          count += one_zero;                              
          setFaceMark(0, finePatch, patchFace, count);
          
#ifdef REFLUX_DBG
          if (c_CC.x() == half.x() && c_CC.y() == half.y() && is_rightFace_variable(name,varLabel) ) {
            cout << " \t c_CC " << c_CC  << " c_FC " << c_FC 
                 << " q_CC_org " << q_CC_coarse_org
                 << " correction " << Q_Z_coarse_corr[c_FC]
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
 Function~  AMRrefine_CF_interfaceOperator-- 
_____________________________________________________________________*/
template<class varType>
void AMRICE::refine_CF_interfaceOperator(const Patch* finePatch, 
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
  cout_dbg << *finePatch << " ";
  finePatch->printPatchBCs(cout_dbg);
  IntVector refineRatio = fineLevel->getRefinementRatio();
  //__________________________________
  // Iterate over coarsefine interface faces
  vector<Patch::FaceType> cf;
  finePatch->getCoarseFaces(cf);
  vector<Patch::FaceType>::const_iterator iter;  
  for (iter  = cf.begin(); iter != cf.end(); ++iter){
    Patch::FaceType face = *iter;

    //__________________________________
    // fine level hi & lo cell iter limits
    // coarselevel hi and low index

    IntVector cl, ch, fl, fh;
    getCoarseFineFaceRange(finePatch, coarseLevel, face, Patch::ExtraPlusEdgeCells, 
                           d_orderOf_CFI_Interpolation, cl, ch, fl, fh);

    cout_dbg<< " face " << face << " refineRatio "<< refineRatio
            << " BC type " << finePatch->getBCType(face)
            << " FineLevel iterator" << fl << " " << fh 
            << " \t coarseLevel iterator " << cl << " " << ch <<endl;

    //__________________________________
    // subCycleProgress_var near 1.0 
    //  interpolation using the coarse_new_dw data
    if(subCycleProgress_var > 1-1.e-10){ 
     constCCVariable<varType> q_NewDW;
     coarse_new_dw->getRegion(q_NewDW, label, matl, coarseLevel,
                           cl, ch);
     select_CFI_Interpolator(q_NewDW, d_orderOf_CFI_Interpolation, coarseLevel, 
                             fineLevel, refineRatio, fl,fh, finePatch, face, Q);
                        
    } else {    

    //__________________________________
    // subCycleProgress_var somewhere between 0 or 1
    //  interpolation from both coarse new and old dw 
      constCCVariable<varType> q_OldDW, q_NewDW;
      coarse_old_dw->getRegion(q_OldDW, label, matl, coarseLevel, cl, ch);
      coarse_new_dw->getRegion(q_NewDW, label, matl, coarseLevel, cl, ch);

      CCVariable<varType> Q_old, Q_new;
      fine_new_dw->allocateTemporary(Q_old, finePatch);
      fine_new_dw->allocateTemporary(Q_new, finePatch);
      
      if(d_orderOf_CFI_Interpolation != 2){
        Q_old.initialize(varType(d_EVIL_NUM));
        Q_new.initialize(varType(d_EVIL_NUM));
      } else {               // colella's quadradic interpolator requires  
        Q_old.copyData(Q);   // that data exists on the fine level.
        Q_new.copyData(Q);
      }

      select_CFI_Interpolator(q_OldDW, d_orderOf_CFI_Interpolation, coarseLevel, 
                        fineLevel,refineRatio, fl,fh, finePatch, face, Q_old);

      select_CFI_Interpolator(q_NewDW, d_orderOf_CFI_Interpolation, coarseLevel, 
                        fineLevel,refineRatio, fl,fh, finePatch, face, Q_new);

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
    CellIterator iter = finePatch->getExtraCellIterator();
    if( isEqual<varType>(varType(d_EVIL_NUM),iter,Q, badCell) ){
      ostringstream warn;
      warn <<"ERROR AMRICE::refine_CF_interfaceOperator "
           << "detected an uninitialized variable: "
           << label->getName() << ", cell " << badCell
           << " Q_CC " << Q[badCell] 
           << " Patch " << finePatch->getID() << " Level idx "
           <<fineLevel->getIndex()<<"\n ";
      throw InvalidValue(warn.str(), __FILE__, __LINE__);
    }
  }

  cout_dbg.setActive(false);// turn off the switch for cout_dbg
}

/*_____________________________________________________________________
 Method~  AMRICE::CoarseToFineOperator--
 Purpose~   push data from coarse Grid to the fine grid
 
This method initializes the variables on all patches that the regridder
creates.  The BNR and Hierarchical regridders will create patches that 
are partially filled with old data.  We don't
want to overwrite these data, thus only use the tiled regridder
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
  IntVector bl(0,0,0);  // boundary layer or padding
  int nghostCells = 1;
  bool returnExclusiveRange=true;
  getCoarseLevelRange(finePatch, coarseLevel, cl, ch, fl, fh, bl, 
                      nghostCells, returnExclusiveRange);

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
 Function~  AMRICE::refluxOperator_computeCorrectionFluxes--
 Purpose~  Note this method is needed by AMRICE.cc and impAMRICE.cc thus
 it's part of the ICE object 
_____________________________________________________________________*/
template<class T>
void AMRICE::refluxOperator_computeCorrectionFluxes( 
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
  string x_name = fineVarLabel + "_X_FC_";
  string y_name = fineVarLabel + "_Y_FC_";
  string z_name = fineVarLabel + "_Z_FC_";
  
  // grab the varLabels
  const VarLabel* xFluxLabel = VarLabel::find(x_name + "flux");
  const VarLabel* yFluxLabel = VarLabel::find(y_name + "flux");
  const VarLabel* zFluxLabel = VarLabel::find(z_name + "flux");
  
  const VarLabel* xCorrLabel = VarLabel::find(x_name + "corr");
  const VarLabel* yCorrLabel = VarLabel::find(y_name + "corr");
  const VarLabel* zCorrLabel = VarLabel::find(z_name + "corr"); 

  if(xFluxLabel == NULL || yFluxLabel == NULL || zFluxLabel == NULL){
    throw InternalError( "refluxOperator_computeCorrectionFluxes: variable label not found: " 
                          + x_name + " or " + y_name + " or " + z_name, __FILE__, __LINE__);
  }

  constSFCXVariable<T> Q_X_fine_flux, Q_X_coarse_flux;
  constSFCYVariable<T> Q_Y_fine_flux, Q_Y_coarse_flux;
  constSFCZVariable<T> Q_Z_fine_flux, Q_Z_coarse_flux;
  
  SFCXVariable<T> Q_X_coarse_corr;
  SFCYVariable<T> Q_Y_coarse_corr;
  SFCZVariable<T> Q_Z_coarse_corr;
  
  // find the exact range of fine data (so we don't mess up mpi)
  IntVector xfl, xfh, yfl, yfh, zfl, zfh, ch;
  IntVector xcl, xch, ycl, ych, zcl, zch, fh;

  xfl = yfl = zfl = finePatch->getCellLowIndex();
  xcl = ycl = zcl = coarsePatch->getCellLowIndex();
  
  xfh = finePatch->getHighIndex(Patch::XFaceBased);
  yfh = finePatch->getHighIndex(Patch::YFaceBased);
  zfh = finePatch->getHighIndex(Patch::ZFaceBased);
  xch = coarsePatch->getHighIndex(Patch::XFaceBased);
  ych = coarsePatch->getHighIndex(Patch::YFaceBased);
  zch = coarsePatch->getHighIndex(Patch::ZFaceBased);

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


  Ghost::GhostType gx  = Ghost::AroundFacesX;
  Ghost::GhostType gy  = Ghost::AroundFacesY;
  Ghost::GhostType gz  = Ghost::AroundFacesZ;
  if (do_x) {
    new_dw->getRegion(     Q_X_fine_flux,   xFluxLabel, indx, fineLevel, xfl,xfh);
    new_dw->get(           Q_X_coarse_flux, xFluxLabel, indx, coarsePatch, gx,1);
    new_dw->allocateAndPut(Q_X_coarse_corr, xCorrLabel, indx, coarsePatch);
    Q_X_coarse_corr.initialize(T(0));
  }
  if (do_y) {
    new_dw->getRegion(     Q_Y_fine_flux,   yFluxLabel, indx, fineLevel, yfl,yfh);
    new_dw->get(           Q_Y_coarse_flux, yFluxLabel, indx, coarsePatch, gy,1);
    new_dw->allocateAndPut(Q_Y_coarse_corr, yCorrLabel, indx, coarsePatch);
    Q_Y_coarse_corr.initialize(T(0));
  }
  if (do_z) {
    new_dw->getRegion(     Q_Z_fine_flux,   zFluxLabel, indx, fineLevel, zfl,zfh);
    new_dw->get(           Q_Z_coarse_flux, zFluxLabel, indx, coarsePatch, gz,1);
    new_dw->allocateAndPut(Q_Z_coarse_corr, zCorrLabel, indx, coarsePatch);
    Q_Z_coarse_corr.initialize(T(0));
  }

  IntVector r_Ratio = fineLevel->getRefinementRatio();
  
  // number of sub cycles
  double nSubCycles = 1;
  if(!d_sharedState->isLockstepAMR()){
    nSubCycles = (double)fineLevel->getRefinementRatioMaxDim();
  }

  //__________________________________
  // Iterate over coarsefine interface faces
  vector<Patch::FaceType> cf;
  finePatch->getCoarseFaces(cf);
  vector<Patch::FaceType>::const_iterator iter;  
  for (iter  = cf.begin(); iter != cf.end(); ++iter){
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
    cout << "nSubCycles: " << nSubCycles << endl;
  }
#endif 
/*===========TESTING==========`*/
      //__________________________________
      // Add fine patch face fluxes to the coarse cells
      // c_CC f_CC:    coarse/fine level cell center index
      // c_FC f_FC:    coarse/fine level face center index
      int count = getFaceMark(0, finePatch, patchFace);
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

           Q_X_coarse_corr[c_FC] = (c_FaceNormal*Q_X_coarse_flux[c_FC] 
                                 + (f_FaceNormal*sum_fineLevelFlux)/nSubCycles);
           //Q_X_coarse_flux[c_FC] = ( Q_X_coarse_flux_org[c_FC] - coeff* sum_fineLevelFlux);

           count += one_zero;                              
           setFaceMark(0, finePatch, patchFace,count);
/*`==========TESTING==========*/
#ifdef REFLUX_DBG
        if (c_CC.y() == half.y() && c_CC.z() == half.z() && is_rightFace_variable(name,fineVarLabel) ) {
          cout << " \t c_CC " << c_CC  << " c_FC " << c_FC 
               << " coarseLevelFlux " << c_FaceNormal*Q_X_coarse_flux[c_FC]
               << " sum_fineLevelflux " << (f_FaceNormal *sum_fineLevelFlux)/nSubCycles
               << " correction " << Q_X_coarse_corr[c_FC]<< endl;
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

           Q_Y_coarse_corr[c_FC] = (c_FaceNormal*Q_Y_coarse_flux[c_FC] 
                                 + (f_FaceNormal*sum_fineLevelFlux)/nSubCycles);

           count += one_zero;                       // keep track of how many times              
           setFaceMark(0, finePatch, patchFace, count); // the face has been touched
/*`==========TESTING==========*/
#ifdef REFLUX_DBG
        if (c_CC.x() == half.x() && c_CC.z() == half.z() && is_rightFace_variable(name,fineVarLabel)) {
          cout << " \t c_CC " << c_CC  << " c_FC " << c_FC 
               << " coarseLevelFlux " << c_FaceNormal*Q_Y_coarse_flux[c_FC]
               << " sum_fineLevelflux " << (f_FaceNormal *sum_fineLevelFlux)/nSubCycles
               << " correction " << Q_Y_coarse_corr[c_FC] << endl;
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
                              
           Q_Z_coarse_corr[c_FC] = (c_FaceNormal*Q_Z_coarse_flux[c_FC] 
                                 + (f_FaceNormal*sum_fineLevelFlux)/nSubCycles);
           
            count += one_zero;                              
            setFaceMark(0, finePatch, patchFace,count);
/*`==========TESTING==========*/
#ifdef REFLUX_DBG
        if (c_CC.x() == half.x() && c_CC.y() == half.y() && is_rightFace_variable(name,fineVarLabel)) {
          cout << " \t c_CC " << c_CC  << " c_FC " << c_FC 
                << " coarseLevelFlu " << c_FaceNormal*Q_Z_coarse_flux[c_FC]
               << " sum_fineLevelflux " << (f_FaceNormal *sum_fineLevelFlux)/nSubCycles
               << " correction " << Q_Z_coarse_corr[c_FC]<< endl;
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
