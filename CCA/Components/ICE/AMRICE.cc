#include <Packages/Uintah/CCA/Components/ICE/AMRICE.h>
#include <Packages/Uintah/CCA/Components/ICE/ICELabel.h>
#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <Packages/Uintah/CCA/Ports/SolverInterface.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/PerPatch.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Core/Util/DebugStream.h>
#include <iomanip>

using namespace Uintah;
using namespace std;
static DebugStream cout_doing("ICE_DOING_COUT", false);
static DebugStream cout_dbg("AMRICE_DBG", false);

AMRICE::AMRICE(const ProcessorGroup* myworld)
  : ICE(myworld)
{
}

AMRICE::~AMRICE()
{
}
//___________________________________________________________________
void AMRICE::problemSetup(const ProblemSpecP& params, GridP& grid,
                            SimulationStateP& sharedState)
{
  cout_doing << "Doing problemSetup  \t\t\t AMRICE" << '\n';
  ICE::problemSetup(params, grid, sharedState);
}
//___________________________________________________________________              
void AMRICE::scheduleInitialize(const LevelP& level,
                                  SchedulerP& sched)
{
  cout_doing << "AMRICE::scheduleInitialize \t\tL-" << level->getIndex() << '\n';
  ICE::scheduleInitialize(level, sched);
}
//___________________________________________________________________
void AMRICE::initialize(const ProcessorGroup*,
                           const PatchSubset* patches, const MaterialSubset* matls,
                           DataWarehouse* old_dw, DataWarehouse* new_dw)
{
}
/*___________________________________________________________________
 Function~  AMRICE::scheduleRefineInterface--  
_____________________________________________________________________*/
void AMRICE::scheduleRefineInterface(const LevelP& fineLevel,
                                     SchedulerP& sched,
                                     int step, 
                                     int nsteps)
{
  if(fineLevel->getIndex() > 0){
    cout_doing << "AMRICE::scheduleRefineInterface \t\tL-" 
               << fineLevel->getIndex() << '\n';
               
    double subCycleProgress = double(step)/double(nsteps);
    
    Task* task = scinew Task("AMRICE::refineInterface", 
                       this, &AMRICE::refineInterface, subCycleProgress);   
  
    addRefineDependencies(task, lb->press_CCLabel,    step, nsteps);
    addRefineDependencies(task, lb->rho_CCLabel,      step, nsteps);
    addRefineDependencies(task, lb->sp_vol_CCLabel,   step, nsteps);
    addRefineDependencies(task, lb->temp_CCLabel,     step, nsteps);
    addRefineDependencies(task, lb->vel_CCLabel,      step, nsteps);
    if(d_usingLODI) {
      addRefineDependencies(task,lb->vol_frac_CCLabel,step, nsteps);
    }
    
    //__________________________________
    // Model Variables.
    if(d_modelSetup && d_modelSetup->tvars.size() > 0){
      vector<TransportedVariable*>::iterator iter;

      for(iter = d_modelSetup->tvars.begin();
         iter != d_modelSetup->tvars.end(); iter++){
        TransportedVariable* tvar = *iter;
        addRefineDependencies(task, tvar->var, step, nsteps);
      }
    }
    
    const MaterialSet* ice_matls = d_sharedState->allICEMaterials();
    sched->addTask(task, fineLevel->eachPatch(), ice_matls);
  }
}
/*______________________________________________________________________
 Function~  AMRICE::refineInterface
 Purpose~   
______________________________________________________________________*/
void AMRICE::refineInterface(const ProcessorGroup*,
                             const PatchSubset* patches,
                             const MaterialSubset*,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw,
                             double subCycleProgress)
{
  const Level* level = getLevel(patches);
  if(level->getIndex() > 0){     
    cout_doing << "Doing refineInterface"<< "\t\t\t\t\t AMRICE L-" 
               << level->getIndex() << endl;
    int  numMatls = d_sharedState->getNumICEMatls();
    Ghost::GhostType  gac = Ghost::AroundCells;
    
    for(int p=0;p<patches->size();p++){
      const Patch* patch = patches->get(p);
      
      for (int m = 0; m < numMatls; m++) {
        ICEMaterial* matl = d_sharedState->getICEMaterial(m);
        int indx = matl->getDWIndex();    
        constCCVariable<double> press_CC, rho_CC, sp_vol_CC, temp_CC;
        constCCVariable<Vector> vel_CC;

        old_dw->get(press_CC, lb->press_CCLabel,  indx,patch, gac,1);
        old_dw->get(rho_CC,   lb->rho_CCLabel,    indx,patch, gac,1);
        old_dw->get(sp_vol_CC,lb->sp_vol_CCLabel, indx,patch, gac,1);
        old_dw->get(temp_CC,  lb->temp_CCLabel,   indx,patch, gac,1);
        old_dw->get(vel_CC,   lb->vel_CCLabel,    indx,patch, gac,1);

        refineBoundaries(patch, press_CC.castOffConst(), new_dw, 
                            lb->press_CCLabel,  indx,subCycleProgress);

        refineBoundaries(patch, rho_CC.castOffConst(),new_dw, 
                            lb->rho_CCLabel,    indx,subCycleProgress);

        refineBoundaries(patch, sp_vol_CC.castOffConst(),new_dw,
                            lb->sp_vol_CCLabel, indx,subCycleProgress);

        refineBoundaries(patch, temp_CC.castOffConst(),new_dw,
                            lb->temp_CCLabel,   indx,subCycleProgress);

        refineBoundaries(patch, vel_CC.castOffConst(),new_dw,
                            lb->vel_CCLabel,    indx,subCycleProgress);
       //__________________________________
       //    Model Variables                     
       if(d_modelSetup && d_modelSetup->tvars.size() > 0){
         vector<TransportedVariable*>::iterator t_iter;
          for( t_iter  = d_modelSetup->tvars.begin();
               t_iter != d_modelSetup->tvars.end(); t_iter++){
            TransportedVariable* tvar = *t_iter;

            if(tvar->matls->contains(indx)){
              constCCVariable<double> q_CC;
              old_dw->get(q_CC, tvar->var, indx, patch, gac, 1);
              refineBoundaries(patch, q_CC.castOffConst(),new_dw,
                               tvar->var,    indx,subCycleProgress);
             #if 0  
               string name = tvar->var->getName();
               printData(indx, patch, 1, "refineInterface_models", name, q_CC);
             #endif                 
            }
          }
        }                                     
                            
#if 0
        //__________________________________
        //  Print Data 
        ostringstream desc;     
        desc << "refineInterface_Mat_" << indx << "_patch_"<< patch->getID();
        printData(indx, patch,   1, desc.str(), "press_CC",    press_CC);
        printData(indx, patch,   1, desc.str(), "rho_CC",      rho_CC);
        printData(indx, patch,   1, desc.str(), "sp_vol_CC",   sp_vol_CC);
        printData(indx, patch,   1, desc.str(), "Temp_CC",     temp_CC);
        printVector(indx, patch, 1, desc.str(), "vel_CC", 0,   vel_CC);
#endif
      }
    }
  }             
}
/*___________________________________________________________________
 Function~  AMRICE::linearInterpolationWeights--
_____________________________________________________________________*/
inline void linearInterpolationWeights(const IntVector& idx, 
                                    Vector& w,
                                    IntVector& cidx,
                                    const IntVector& coarseHigh,
                                    const Level* level,
                                    Patch::FaceType face)
{
   cidx = level->interpolateCellToCoarser(idx, w);
   if(cidx.x()+1 >= coarseHigh.x()){
     cidx.x(cidx.x()-1);
     w.x(1);
   }
   if(cidx.y()+1 >= coarseHigh.y()){
     cidx.y(cidx.y()-1);
     w.y(1);
   }
   if(cidx.z()+1 >= coarseHigh.z()){
     cidx.z(cidx.z()-1);
     w.z(1);
   }
   switch(face){
   case Patch::xminus:
     w.x(0);
     break;
   case Patch::xplus:
     w.x(1);
     break;
   case Patch::yminus:
     w.y(0);
     break;
   case Patch::yplus:
     w.y(1);
     break;
   case Patch::zminus:
     w.z(0);
     break;
   case Patch::zplus:
     w.z(1);
     break;
   default:
     break;
   }
}
#if 0
/*___________________________________________________________________
 Function~  AMRICE::linearInterpolation--
_____________________________________________________________________*/
template<class T>
  void linearInterpolation(ArrayType& q_CC,
                               const Vector w,
                               const IntVector cidx,
                                T& x0)
{
  x0 = q_CC[cidx+IntVector(0,0,0)]*(1-w.x())*(1-w.y())*(1-w.z())
     + q_CC[cidx+IntVector(1,0,0)]*   w.x() *(1-w.y())*(1-w.z())
     + q_CC[cidx+IntVector(0,1,0)]*(1-w.x())*   w.y() *(1-w.z())
     + q_CC[cidx+IntVector(1,1,0)]*   w.x() *   w.y() *(1-w.z())
     + q_CC[cidx+IntVector(0,0,1)]*(1-w.x())*(1-w.y())*   w.z()
     + q_CC[cidx+IntVector(1,0,1)]*   w.x() *(1-w.y())*   w.z()
     + q_CC[cidx+IntVector(0,1,1)]*(1-w.x())*   w.y() *   w.z()
     + q_CC[cidx+IntVector(1,1,1)]*   w.x() *   w.y() *   w.z();
}
#endif

/*___________________________________________________________________
 Function~  AMRICE::refineFaces--   scalar vesion
_____________________________________________________________________*/
template<class ArrayType, class constArrayType>
void refineFaces(const Patch* patch, 
                const Level* level,
                const Level* coarseLevel, 
                const IntVector& dir,
                Patch::FaceType lowFace, 
                Patch::FaceType highFace,
                ArrayType& Q, 
                const VarLabel* label,
                double subCycleProgress_var, 
                int matl, 
                DataWarehouse* coarse_old_dw,
                DataWarehouse* coarse_new_dw, 
                Patch::VariableBasis basis)
{
  for(Patch::FaceType face = Patch::startFace;
      face <= Patch::endFace; face=Patch::nextFace(face)){
    if(patch->getBCType(face) != Patch::Coarse)
      continue;

    {
     //__________________________________
     // fine level hi & lo cell iter limits
     // coarselevel hi and low index
      CellIterator iter_tmp = patch->getFaceCellIterator(face, "plusEdgeCells");
      IntVector l = iter_tmp.begin();
      IntVector h = iter_tmp.end(); 
      IntVector refineRatio = level->getRefinementRatio();
      IntVector coarseLow  = level->mapCellToCoarser(l);
      IntVector coarseHigh = level->mapCellToCoarser(h+refineRatio-IntVector(1,1,1));

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
        coarseLow -= oneCell;
      }

      cout_dbg << "face " << face << " FineLevel " << iter_tmp;
      cout_dbg << "  coarseLow " << coarseLow << " coarse high " << coarseHigh;
      cout_dbg << "  refine Patch " << *patch << endl;
      
      //__________________________________
      //   subCycleProgress_var  = 0
      if(subCycleProgress_var < 1.e-10){
       constArrayType q_OldDW;
       coarse_old_dw->getRegion(q_OldDW, label, matl, coarseLevel,
                             coarseLow, coarseHigh);
                             
       for(CellIterator iter(l,h); !iter.done(); iter++){
         IntVector idx = *iter;
         IntVector cidx;
         Vector w;
        
         linearInterpolationWeights( idx,w, cidx, coarseHigh, level, face);
         //_________________
         //  interpolation, using the coarse old_DW
         double x0 = q_OldDW[cidx+IntVector(0,0,0)]*(1-w.x())*(1-w.y())*(1-w.z())
           + q_OldDW[cidx+IntVector(1,0,0)]*   w.x() *(1-w.y())*(1-w.z())
           + q_OldDW[cidx+IntVector(0,1,0)]*(1-w.x())*   w.y() *(1-w.z())
           + q_OldDW[cidx+IntVector(1,1,0)]*   w.x() *   w.y() *(1-w.z())
           + q_OldDW[cidx+IntVector(0,0,1)]*(1-w.x())*(1-w.y())*   w.z()
           + q_OldDW[cidx+IntVector(1,0,1)]*   w.x() *(1-w.y())*   w.z()
           + q_OldDW[cidx+IntVector(0,1,1)]*(1-w.x())*   w.y() *   w.z()
           + q_OldDW[cidx+IntVector(1,1,1)]*   w.x() *   w.y() *   w.z();     
           
         Q[idx] = x0;
         
       }  // cell iterator
      } else if(subCycleProgress_var > 1-1.e-10){  // subCycleProgress_var near 1.0
       constArrayType q_NewDW;
       coarse_new_dw->getRegion(q_NewDW, label, matl, coarseLevel,
                             coarseLow, coarseHigh);
                             
       for(CellIterator iter(l,h); !iter.done(); iter++){
         IntVector idx = *iter;
         IntVector cidx;
         Vector w;
        
         linearInterpolationWeights( idx,w, cidx, coarseHigh, level, face);
         //_________________
         //  interpolation using the coarse_new_dw data
         double x1 = q_NewDW[cidx+IntVector(0,0,0)]*(1-w.x())*(1-w.y())*(1-w.z())
           + q_NewDW[cidx+IntVector(1,0,0)]*   w.x() *(1-w.y())*(1-w.z())
           + q_NewDW[cidx+IntVector(0,1,0)]*(1-w.x())*   w.y() *(1-w.z())
           + q_NewDW[cidx+IntVector(1,1,0)]*   w.x() *   w.y() *(1-w.z())
           + q_NewDW[cidx+IntVector(0,0,1)]*(1-w.x())*(1-w.y())*   w.z()
           + q_NewDW[cidx+IntVector(1,0,1)]*   w.x() *(1-w.y())*   w.z()
           + q_NewDW[cidx+IntVector(0,1,1)]*(1-w.x())*   w.y() *   w.z()
           + q_NewDW[cidx+IntVector(1,1,1)]*   w.x() *   w.y() *   w.z();

         Q[idx] = x1;
       }  // cell iterator
      } else {                    // subCycleProgress_var neither 0 or 1 
       constArrayType q_OldDW, q_NewDW;
       coarse_old_dw->getRegion(q_OldDW, label, matl, coarseLevel,
                             coarseLow, coarseHigh);
       coarse_new_dw->getRegion(q_NewDW, label, matl, coarseLevel,
                             coarseLow, coarseHigh);
                             
      for(CellIterator iter(l,h); !iter.done(); iter++){
         IntVector idx = *iter;
         IntVector cidx;
         Vector w;
        
         linearInterpolationWeights( idx,w, cidx, coarseHigh, level, face);
         //_________________
         //  interpolation from both coarse new and old dw
         // coarse_old_dw data
         double x0 = q_OldDW[cidx+IntVector(0,0,0)]*(1-w.x())*(1-w.y())*(1-w.z())           
           + q_OldDW[cidx+IntVector(1,0,0)]*   w.x() *(1-w.y())*(1-w.z())
           + q_OldDW[cidx+IntVector(0,1,0)]*(1-w.x())*   w.y() *(1-w.z())
           + q_OldDW[cidx+IntVector(1,1,0)]*   w.x() *   w.y() *(1-w.z())
           + q_OldDW[cidx+IntVector(0,0,1)]*(1-w.x())*(1-w.y())*   w.z()
           + q_OldDW[cidx+IntVector(1,0,1)]*   w.x( )*(1-w.y())*   w.z()
           + q_OldDW[cidx+IntVector(0,1,1)]*(1-w.x())*    w.y()*   w.z()
           + q_OldDW[cidx+IntVector(1,1,1)]*   w.x() *    w.y()*   w.z();
          
          // coarse_new_dw data 
         double x1 = q_NewDW[cidx+IntVector(0,0,0)]*(1-w.x())*(1-w.y())*(1-w.z())
           + q_NewDW[cidx+IntVector(1,0,0)]*   w.x() *(1-w.y())*(1-w.z())
           + q_NewDW[cidx+IntVector(0,1,0)]*(1-w.x())*   w.y() *(1-w.z())
           + q_NewDW[cidx+IntVector(1,1,0)]*   w.x() *   w.y() *(1-w.z())
           + q_NewDW[cidx+IntVector(0,0,1)]*(1-w.x())*(1-w.y())*   w.z()
           + q_NewDW[cidx+IntVector(1,0,1)]*   w.x() *(1-w.y())*   w.z()
           + q_NewDW[cidx+IntVector(0,1,1)]*(1-w.x())*   w.y() *   w.z()
           + q_NewDW[cidx+IntVector(1,1,1)]*   w.x() *   w.y() *   w.z();
           
         // Interpolate temporally  
         double x = (1-subCycleProgress_var)*x0 + subCycleProgress_var*x1;
         Q[idx] = x;
       }
      }
    }
  }
}
/*___________________________________________________________________
 Function~  AMRICE::refineFaces--   vector version
_____________________________________________________________________*/
void refineFaces(const Patch* patch, 
                const Level* level,
                const Level* coarseLevel, 
                const IntVector& dir,
                Patch::FaceType lowFace, 
                Patch::FaceType highFace,
                CCVariable<Vector>& Q, 
                const VarLabel* label,
                double subCycleProgress_var, 
                int matl, 
                DataWarehouse* coarse_old_dw,
                DataWarehouse* coarse_new_dw, 
                Patch::VariableBasis basis)
{
  for(Patch::FaceType face = Patch::startFace;
      face <= Patch::endFace; face=Patch::nextFace(face)){
    if(patch->getBCType(face) != Patch::Coarse)
      continue;

    {
     //__________________________________
     //  determine low and high cell iter limits
     // and coarselevel hi and low index
      CellIterator iter_tmp = patch->getFaceCellIterator(face, "plusEdgeCells");
      IntVector l = iter_tmp.begin();
      IntVector h = iter_tmp.end(); 
      IntVector refineRatio = level->getRefinementRatio();
      IntVector coarseLow  = level->mapCellToCoarser(l);
      IntVector coarseHigh = level->mapCellToCoarser(h + refineRatio - IntVector(1,1,1));

      //__________________________________
      // enlarge the coarse foot print by oneCell
      // x-           x+        y-       y+       z-        z+
      // (-1,0,0)  (1,0,0)  (0,-1,0)  (0,1,0)  (0,0,-1)  (0,0,1)
      IntVector oneCell = patch->faceDirection(face);
      if( face == Patch::xminus || face == Patch::yminus || face == Patch::zminus) {
        coarseHigh -= oneCell;
      }
      if( face == Patch::xplus || face == Patch::yplus || face == Patch::zplus) {
        coarseLow -= oneCell;
      }

      cout_dbg << "face " << face << " FineLevel " << iter_tmp;
      cout_dbg << "  coarseLow " << coarseLow << " coarse high " << coarseHigh;
      cout_dbg << "  refine Patch " << *patch << endl;
      //__________________________________
      //   subCycleProgress_var  = 0
      if(subCycleProgress_var < 1.e-10){
       constCCVariable<Vector> q_OldDW;
       coarse_old_dw->getRegion(q_OldDW, label, matl, coarseLevel,
                             coarseLow, coarseHigh);
                             
       for(CellIterator iter(l,h); !iter.done(); iter++){
         IntVector idx = *iter;
         IntVector cidx;
         Vector w;
        
         linearInterpolationWeights( idx,w, cidx, coarseHigh, level, face);
         
         //_________________
         //  interpolation, using the coarse old_DW
         Vector x0 = q_OldDW[cidx+IntVector(0,0,0)]*(1-w.x())*(1-w.y())*(1-w.z())
           + q_OldDW[cidx+IntVector(1,0,0)]*   w.x() *(1-w.y())*(1-w.z())
           + q_OldDW[cidx+IntVector(0,1,0)]*(1-w.x())*   w.y() *(1-w.z())
           + q_OldDW[cidx+IntVector(1,1,0)]*   w.x() *   w.y() *(1-w.z())
           + q_OldDW[cidx+IntVector(0,0,1)]*(1-w.x())*(1-w.y())*   w.z()
           + q_OldDW[cidx+IntVector(1,0,1)]*   w.x() *(1-w.y())*   w.z()
           + q_OldDW[cidx+IntVector(0,1,1)]*(1-w.x())*   w.y() *   w.z()
           + q_OldDW[cidx+IntVector(1,1,1)]*   w.x() *   w.y() *   w.z();
         Q[idx] = x0;
       }  // cell iterator
      } else if(subCycleProgress_var > 1-1.e-10){        /// subCycleProgress_var near 1.0
       constCCVariable<Vector> q_NewDW;
       coarse_new_dw->getRegion(q_NewDW, label, matl, coarseLevel,
                             coarseLow, coarseHigh);
                             
       for(CellIterator iter(l,h); !iter.done(); iter++){
         IntVector idx = *iter;
         IntVector cidx;
         Vector w;
         linearInterpolationWeights( idx, w, cidx, coarseHigh, level, face);
         //_________________
         //  interpolation using the coarse_new_dw data
         Vector x1 = q_NewDW[cidx+IntVector(0,0,0)]*(1-w.x())*(1-w.y())*(1-w.z())
           + q_NewDW[cidx+IntVector(1,0,0)]*   w.x() *(1-w.y())*(1-w.z())
           + q_NewDW[cidx+IntVector(0,1,0)]*(1-w.x())*   w.y() *(1-w.z())
           + q_NewDW[cidx+IntVector(1,1,0)]*   w.x() *   w.y() *(1-w.z())
           + q_NewDW[cidx+IntVector(0,0,1)]*(1-w.x())*(1-w.y())*   w.z()
           + q_NewDW[cidx+IntVector(1,0,1)]*   w.x() *(1-w.y())*   w.z()
           + q_NewDW[cidx+IntVector(0,1,1)]*(1-w.x())*   w.y() *   w.z()
           + q_NewDW[cidx+IntVector(1,1,1)]*   w.x() *   w.y() *   w.z();
         Q[idx] = x1;
       }  // cell iterator
      } else {                    // subCycleProgress_var neither 0 or 1 
       constCCVariable<Vector> q_OldDW, q_NewDW;
       coarse_old_dw->getRegion(q_OldDW, label, matl, coarseLevel,
                             coarseLow, coarseHigh);
       coarse_new_dw->getRegion(q_NewDW, label, matl, coarseLevel,
                             coarseLow, coarseHigh);

       for(CellIterator iter(l,h); !iter.done(); iter++){
         IntVector idx = *iter;
         IntVector cidx;
         Vector w;
         linearInterpolationWeights( idx, w, cidx, coarseHigh, level, face); 
         //_________________
         //  interpolation from both coarse new and old dw
         // coarse_old_dw data
         Vector x0 = q_OldDW[cidx+IntVector(0,0,0)]*(1-w.x())*(1-w.y())*(1-w.z())           
           + q_OldDW[cidx+IntVector(1,0,0)]*   w.x() *(1-w.y())*(1-w.z())
           + q_OldDW[cidx+IntVector(0,1,0)]*(1-w.x())*   w.y() *(1-w.z())
           + q_OldDW[cidx+IntVector(1,1,0)]*   w.x() *   w.y() *(1-w.z())
           + q_OldDW[cidx+IntVector(0,0,1)]*(1-w.x())*(1-w.y())*   w.z()
           + q_OldDW[cidx+IntVector(1,0,1)]*   w.x( )*(1-w.y())*   w.z()
           + q_OldDW[cidx+IntVector(0,1,1)]*(1-w.x())*    w.y()*   w.z()
           + q_OldDW[cidx+IntVector(1,1,1)]*   w.x() *    w.y()*   w.z();
          
          // coarse_new_dw data 
         Vector x1 = q_NewDW[cidx+IntVector(0,0,0)]*(1-w.x())*(1-w.y())*(1-w.z())
           + q_NewDW[cidx+IntVector(1,0,0)]*   w.x() *(1-w.y())*(1-w.z())
           + q_NewDW[cidx+IntVector(0,1,0)]*(1-w.x())*   w.y() *(1-w.z())
           + q_NewDW[cidx+IntVector(1,1,0)]*   w.x() *   w.y() *(1-w.z())
           + q_NewDW[cidx+IntVector(0,0,1)]*(1-w.x())*(1-w.y())*   w.z()
           + q_NewDW[cidx+IntVector(1,0,1)]*   w.x() *(1-w.y())*   w.z()
           + q_NewDW[cidx+IntVector(0,1,1)]*(1-w.x())*   w.y() *   w.z()
           + q_NewDW[cidx+IntVector(1,1,1)]*   w.x() *   w.y() *   w.z();
           
         // Interpolate temporally  
         Vector x = (1-subCycleProgress_var)*x0 + subCycleProgress_var*x1;
         Q[idx] = x;
       }
      }
    }
  }
}
/*___________________________________________________________________
 Function~  AMRICE::addRefineDependencies--
_____________________________________________________________________*/
void AMRICE::addRefineDependencies(Task* task, 
                                   const VarLabel* var,
                                   int step, 
                                   int nsteps)
{
  cout_dbg << "\t addRefineDependencies (" << var->getName()
           << ") \t step " << step << " nsteps " << nsteps;
  ASSERTRANGE(step, 0, nsteps+1);
  Ghost::GhostType  gac = Ghost::AroundCells;
  
  if(step != nsteps) {
    cout_dbg << " requires from CoarseOldDW ";
    task->requires(Task::CoarseOldDW, var,
                 0, Task::CoarseLevel, 0, Task::NormalDomain, gac, 1);
  }
  if(step != 0) {
    cout_dbg << " requires from CoarseNewDW ";
    task->requires(Task::CoarseNewDW, var,
                 0, Task::CoarseLevel, 0, Task::NormalDomain, gac, 1);
  }
  cout_dbg <<""<<endl;
}
/*___________________________________________________________________
 Function~  AMRICE::refineBoundaries--   double 
_____________________________________________________________________*/
void AMRICE::refineBoundaries(const Patch* patch,
                           CCVariable<double>& val,
                           DataWarehouse* new_dw,
                           const VarLabel* label,
                           int matl,
                           double subCycleProgress_var)
{
  cout_dbg << "\t refineBoundaries ("<<label->getName() << ") \t" 
           << " subCycleProgress_var " << subCycleProgress_var<< '\n';
  DataWarehouse* coarse_old_dw = new_dw->getOtherDataWarehouse(Task::CoarseOldDW);
  DataWarehouse* coarse_new_dw = new_dw->getOtherDataWarehouse(Task::CoarseNewDW);
  
  const Level* level = patch->getLevel();
  const Level* coarseLevel = level->getCoarserLevel().get_rep();
  
  refineFaces<CCVariable<double>, constCCVariable<double> >
    (patch, level, coarseLevel, IntVector(0,0,0), Patch::invalidFace,
     Patch::invalidFace, val, label, subCycleProgress_var, matl,
     coarse_old_dw, coarse_new_dw, Patch::CellBased);
}
/*___________________________________________________________________
 Function~  AMRICE::refineBoundaries--   Vector 
_____________________________________________________________________*/
void AMRICE::refineBoundaries(const Patch* patch,
                           CCVariable<Vector>& val,
                           DataWarehouse* new_dw,
                           const VarLabel* label,
                           int matl,
                           double subCycleProgress_var)
{
  cout_dbg << "\t refineBoundaries ("<<label->getName() << ") \t" 
           << " subCycleProgress_var " << subCycleProgress_var<< '\n';
  DataWarehouse* coarse_old_dw = new_dw->getOtherDataWarehouse(Task::CoarseOldDW);
  DataWarehouse* coarse_new_dw = new_dw->getOtherDataWarehouse(Task::CoarseNewDW);
  
  const Level* level = patch->getLevel();
  const Level* coarseLevel = level->getCoarserLevel().get_rep();

  refineFaces(patch, level, coarseLevel, IntVector(0,0,0), Patch::invalidFace,
              Patch::invalidFace, val, label, subCycleProgress_var, matl,
              coarse_old_dw, coarse_new_dw, Patch::CellBased);
}


//______________________________________________________________________
//  SFCXVariable version        N O T   U S E D
void AMRICE::refineBoundaries(const Patch* patch,
                           SFCXVariable<double>& val,
                           DataWarehouse* new_dw,
                           const VarLabel* label,
                           int matl,
                           double subCycleProgress_var)
{
  cout_dbg << "\t refineBoundaries ("<<label->getName() << ") \t" 
           << " subCycleProgress_var " << subCycleProgress_var<< '\n';
  DataWarehouse* coarse_old_dw = new_dw->getOtherDataWarehouse(Task::CoarseOldDW);
  DataWarehouse* coarse_new_dw = new_dw->getOtherDataWarehouse(Task::CoarseNewDW);
  
  const Level* level = patch->getLevel();
  const Level* coarseLevel = level->getCoarserLevel().get_rep();
  
  refineFaces<SFCXVariable<double>, constSFCXVariable<double> >
    (patch, level, coarseLevel, IntVector(1,0,0), Patch::xminus,
     Patch::xplus, val, label, subCycleProgress_var, matl,
     coarse_old_dw, coarse_new_dw, Patch::XFaceBased);
}
//______________________________________________________________________
//  SFCYVariable version        N O T   U S E D
void AMRICE::refineBoundaries(const Patch* patch,
                           SFCYVariable<double>& val,
                           DataWarehouse* new_dw,
                           const VarLabel* label,
                           int matl,
                           double subCycleProgress_var)
{
  cout_dbg << "\t refineBoundaries ("<<label->getName() << ") \t" 
           << " subCycleProgress_var " << subCycleProgress_var<< '\n';
  DataWarehouse* coarse_old_dw = new_dw->getOtherDataWarehouse(Task::CoarseOldDW);
  DataWarehouse* coarse_new_dw = new_dw->getOtherDataWarehouse(Task::CoarseNewDW);
  
  const Level* level = patch->getLevel();
  const Level* coarseLevel = level->getCoarserLevel().get_rep();
  
  refineFaces<SFCYVariable<double>, constSFCYVariable<double> >
    (patch, level, coarseLevel, IntVector(0,1,0), Patch::yminus,
     Patch::yplus, val, label, subCycleProgress_var, matl,
     coarse_old_dw, coarse_new_dw, Patch::YFaceBased);
}
//______________________________________________________________________
//  SFCZVariable version        N O T   U S E D
void AMRICE::refineBoundaries(const Patch* patch,
                            SFCZVariable<double>& val,
                            DataWarehouse* new_dw,
                            const VarLabel* label,
                            int matl,
                            double subCycleProgress_var)
{
  cout_dbg << "\t refineBoundaries ("<<label->getName() << ") \t" 
           << " subCycleProgress_var " << subCycleProgress_var<< '\n';
  DataWarehouse* coarse_old_dw = new_dw->getOtherDataWarehouse(Task::CoarseOldDW);
  DataWarehouse* coarse_new_dw = new_dw->getOtherDataWarehouse(Task::CoarseNewDW);
  
  const Level* level = patch->getLevel();
  const Level* coarseLevel = level->getCoarserLevel().get_rep();
  
  refineFaces<SFCZVariable<double>, constSFCZVariable<double> >
    (patch, level, coarseLevel, IntVector(0,0,1), Patch::zminus,
     Patch::zplus, val, label, subCycleProgress_var, matl,
     coarse_old_dw, coarse_new_dw, Patch::ZFaceBased);
}
/*___________________________________________________________________
 Function~  AMRICE::scheduleCoarsen--  
_____________________________________________________________________*/
void AMRICE::scheduleCoarsen(const LevelP& coarseLevel,
                               SchedulerP& sched)
{
  Ghost::GhostType  gn = Ghost::None; 
  cout_dbg << "AMRICE::scheduleCoarsen\t\t\t\tL-" << coarseLevel->getIndex() << '\n';
  Task* task = scinew Task("coarsen",this, &AMRICE::coarsen);

  task->requires(Task::NewDW, lb->press_CCLabel,
               0, Task::FineLevel, 0, Task::NormalDomain, gn, 0);
                 
  task->requires(Task::NewDW, lb->rho_CCLabel,
               0, Task::FineLevel, 0, Task::NormalDomain, gn, 0);
               
  task->requires(Task::NewDW, lb->sp_vol_CCLabel,
               0, Task::FineLevel, 0, Task::NormalDomain, gn, 0);
  
  task->requires(Task::NewDW, lb->temp_CCLabel,
               0, Task::FineLevel, 0, Task::NormalDomain, gn, 0);
  
  task->requires(Task::NewDW, lb->vel_CCLabel,
               0, Task::FineLevel, 0, Task::NormalDomain, gn, 0);
  
  task->modifies(lb->press_CCLabel);
  task->modifies(lb->rho_CCLabel);
  task->modifies(lb->sp_vol_CCLabel);
  task->modifies(lb->temp_CCLabel);
  task->modifies(lb->vel_CCLabel);

  sched->addTask(task, coarseLevel->eachPatch(), d_sharedState->allMaterials()); 
}

/*___________________________________________________________________
 Function~  AMRICE::Coarsen--  
_____________________________________________________________________*/
void AMRICE::coarsen(const ProcessorGroup*,
                        const PatchSubset* patches,
                        const MaterialSubset* matls,
                        DataWarehouse* old_dw,
                        DataWarehouse* new_dw)
{
  cout_doing << "Doing coarsen \t\t\t\t\t\t AMRICE";
  const Level* coarseLevel = getLevel(patches);
  const Level* fineLevel = coarseLevel->getFinerLevel().get_rep();
  
  IntVector rr(fineLevel->getRefinementRatio());
  double invRefineRatio = 1./(rr.x()*rr.y()*rr.z());
  
  for(int p=0;p<patches->size();p++){  
    const Patch* coarsePatch = patches->get(p);
    cout_doing << " patch " << coarsePatch->getID()<< endl;
    
    Level::selectType finePatches;
    coarsePatch->getFineLevelPatches(finePatches);

    for(int m = 0;m<matls->size();m++){
      int indx = matls->get(m);

      CCVariable<double> rho_CC, press_CC, temp, sp_vol_CC;
      CCVariable<Vector> vel_CC;
      new_dw->getModifiable(press_CC, lb->press_CCLabel,  indx, coarsePatch);
      new_dw->getModifiable(rho_CC,   lb->rho_CCLabel,    indx, coarsePatch);
      new_dw->getModifiable(sp_vol_CC,lb->sp_vol_CCLabel, indx, coarsePatch);
      new_dw->getModifiable(temp,     lb->temp_CCLabel,   indx, coarsePatch);
      new_dw->getModifiable(vel_CC,   lb->vel_CCLabel,    indx, coarsePatch);  
      
      // coarsen
      fineToCoarseOperator<double>(press_CC,  lb->press_CCLabel,indx, new_dw, 
                         invRefineRatio, coarsePatch, coarseLevel, fineLevel);
                         
      fineToCoarseOperator<double>(rho_CC,    lb->rho_CCLabel,  indx, new_dw, 
                         invRefineRatio, coarsePatch, coarseLevel, fineLevel);      

      fineToCoarseOperator<double>(sp_vol_CC, lb->sp_vol_CCLabel,indx, new_dw, 
                         invRefineRatio, coarsePatch, coarseLevel, fineLevel);

      fineToCoarseOperator<double>(temp,      lb->temp_CCLabel, indx, new_dw, 
                         invRefineRatio, coarsePatch, coarseLevel, fineLevel);
       
      fineToCoarseOperator<Vector>( vel_CC,   lb->vel_CCLabel,  indx, new_dw, 
                         invRefineRatio, coarsePatch, coarseLevel, fineLevel);    
    }
  }  // course patch loop 
}
/*_____________________________________________________________________
 Function~  AMRICE::fineToCoarseOperator--
 Purpose~   averages the fine grid data onto the coarse grid data
_____________________________________________________________________*/
template<class T>
void AMRICE::fineToCoarseOperator(CCVariable<T>& q_CC,
                                  const VarLabel* varLabel,
                                  const int indx,
                                  DataWarehouse* new_dw,
                                  const double ratio,
                                  const Patch* coarsePatch,
                                  const Level* coarseLevel,
                                  const Level* fineLevel)
{
  Level::selectType finePatches;
  coarsePatch->getFineLevelPatches(finePatches);
  IntVector extraCells(1,1,1);    // ICE always has 1 layer of extra cells
                            
  for(int i=0;i<finePatches.size();i++){
    const Patch* finePatch = finePatches[i];
    
    constCCVariable<T> fine_q_CC;
    new_dw->get(fine_q_CC, varLabel, indx, finePatch,Ghost::None, 0);

    IntVector fl(finePatch->getCellLowIndex());
    IntVector fh(finePatch->getCellHighIndex());
    IntVector cl(fineLevel->mapCellToCoarser(fl) + extraCells);
    IntVector ch(fineLevel->mapCellToCoarser(fh) - extraCells);
    
    cl = Max(cl, coarsePatch->getCellLowIndex());
    ch = Min(ch, coarsePatch->getCellHighIndex());
    
    cout_dbg << " fineToCoarseOperator: coarselevel"<< cl << ch 
             << " fineLevel " << fl << fh 
             << " patches " << *finePatch << endl;
    IntVector refinementRatio = fineLevel->getRefinementRatio();
    
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
        q_CC_tmp += fine_q_CC[fc];
      }
      q_CC[c] =q_CC_tmp*ratio;
    }
  }
}
/*_____________________________________________________________________
 Function~  AMRICE::scheduleErrorEstimate--
______________________________________________________________________*/
void AMRICE::scheduleErrorEstimate(const LevelP& coarseLevel,
                                   SchedulerP& sched)
{
  cout_doing << "AMRICE::scheduleErrorEstimate \t\t\tL-" << coarseLevel->getIndex() << '\n';
  // when we know what to estimate we'll fill it in
}

void AMRICE::errorEstimate(const ProcessorGroup*,
                           const PatchSubset* patches,
                           const MaterialSubset* matls,
                           DataWarehouse*,
                           DataWarehouse* new_dw)
{
}
