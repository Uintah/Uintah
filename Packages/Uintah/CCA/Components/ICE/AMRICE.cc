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

/*`==========TESTING==========*/
#define test 0 
/*===========TESTING==========`*/

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
//______________________________________________________________________
//
void AMRICE::problemSetup(const ProblemSpecP& params, GridP& grid,
                            SimulationStateP& sharedState)
{
  cout_doing << "Doing problemSetup  \t\t\t AMRICE" << '\n';
  ICE::problemSetup(params, grid, sharedState);
}
//______________________________________________________________________
// 
void AMRICE::scheduleInitialize(const LevelP& level,
                                  SchedulerP& sched)
{
  cout_doing << "AMRICE::scheduleInitialize \t\tL-" << level->getIndex() << '\n';
  ICE::scheduleInitialize(level, sched);
}
//______________________________________________________________________
// 
void AMRICE::initialize(const ProcessorGroup*,
                           const PatchSubset* patches, const MaterialSubset* matls,
                           DataWarehouse* old_dw, DataWarehouse* new_dw)
{
}

//______________________________________________________________________
// 
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
    const MaterialSet* ice_matls = d_sharedState->allICEMaterials();
    sched->addTask(task, fineLevel->eachPatch(), ice_matls);
  }
}
/* ---------------------------------------------------------------------
 Function~  AMRICE::refineInterface
 Purpose~   
 ---------------------------------------------------------------------  */
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
      
/*`==========TESTING==========*/
      // debugging 
      IntVector l = patch->getCellLowIndex(); 
      IntVector h = patch->getCellHighIndex();
      IntVector cLo = level->mapCellToCoarser(l);
      IntVector cHi = level->mapCellToCoarser(h+level->getRefinementRatio()-IntVector(1,1,1));
      cout << "level " << level->getIndex() << " patch index" << l << " " << h
           << " coarse Level index" << cLo << " " << cHi << endl; 
/*===========TESTING==========`*/
      
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

      }
    }
  }             
}
//______________________________________________________________________
// S C A L A R    V E R S I O N   R E F I N E F A C E S
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
     //  determine low and high cell iter limits
#if 0        
      IntVector l,h;
      patch->getFace(face, IntVector(0,0,0), IntVector(1,1,1), l, h);
      cout_dbg << " face " << face << " base " << l << ","<< h << endl;
      
      // first adjustement to itertor     --steve help
      if(face == highFace){
       l += dir;
       h += dir;
      } else if(face != lowFace){
       h += dir;
      }
      switch(face){
      case Patch::xminus:
      case Patch::xplus:
       l-=IntVector(0,1,1);
       h+=IntVector(0,1,1);
       break;
      case Patch::yminus:
      case Patch::yplus:
       l-=IntVector(1,0,1);
       h+=IntVector(1,0,1);
       break;
      case Patch::zminus:
      case Patch::zplus:
       l-=IntVector(1,1,0);
       h+=IntVector(1,1,0);
       break;
      default:
       break;
      }
      cout_dbg <<"\t firstAdj " << l << ","<< h << endl;
      
      // second adjustment to iterator    --steve help
      if(face != Patch::xminus &&   // Y & Z FACES 
         face != Patch::xplus && patch->getBCType(Patch::xminus) == Patch::None){
       l+=IntVector(1,0,0);
      }
      if(face != Patch::xminus &&   
         face != Patch::xplus && patch->getBCType(Patch::xplus) == Patch::None){
       h-=IntVector(1,0,0);
      }
                                   // X & Z FACES
      if(face != Patch::yminus &&   
         face != Patch::yplus && patch->getBCType(Patch::yminus) == Patch::None){
       l+=IntVector(0,1,0);
      }
      if(face != Patch::yminus &&   
         face != Patch::yplus && patch->getBCType(Patch::yplus) == Patch::None){
       h-=IntVector(0,1,0);
      }
      if(face != Patch::zminus &&   // X & Y FACES
         face != Patch::zplus && patch->getBCType(Patch::zminus) == Patch::None){
       l+=IntVector(0,0,1);
      }
      if(face != Patch::zminus &&   
         face != Patch::zplus && patch->getBCType(Patch::zplus) == Patch::None){
       h-=IntVector(0,0,1);
      }
      cout_dbg <<"\t secondAdj " << l << ","<< h << endl;
      
#endif
/*`==========TESTING==========*/
      CellIterator iter_tmp = patch->getFaceCellIterator(face, "plusEdgeCells");
      IntVector l = iter_tmp.begin();
      IntVector h = iter_tmp.end(); 
     // cout_dbg << "face " << face << " base iterator " << iter_tmp << endl;
/*===========TESTING==========`*/
      //__________________________________
      //  determine low and high coarse level
      //  iteration limits.
/*`==========TESTING==========*/
 //     IntVector coarseLow = level->mapCellToCoarser(l);
      IntVector coarseLow = level->mapCellToCoarser(l - IntVector(1,1,1)); 

      IntVector coarseHigh = level->mapCellToCoarser(h+level->getRefinementRatio()-IntVector(1,1,1));
     // cout_dbg << " base coarseLow " << coarseLow << " coarse high " << coarseHigh << endl;
/*===========TESTING==========`*/      
      switch(face){
      case Patch::xminus:
       coarseHigh+=IntVector(1,0,0);
       break;
      case Patch::xplus: 
       if(basis == Patch::XFaceBased)    // SFCXVariables
         coarseHigh += IntVector(1,0,0);
       else
         coarseLow -= IntVector(1,0,0);
       break;
      case Patch::yminus:
       coarseHigh+=IntVector(0,1,0);
       break;
      case Patch::yplus:
       if(basis == Patch::YFaceBased)   // SFCYVariables
         coarseHigh += IntVector(0,1,0);
       else
         coarseLow -= IntVector(0,1,0);
       break;
      case Patch::zminus:
       coarseHigh+=IntVector(0,0,1);
       break;
      case Patch::zplus:
       if(basis == Patch::ZFaceBased)   // SFCZVariables
         coarseHigh += IntVector(0,0,1);
       else
         coarseLow -= IntVector(0,0,1);
       break;
      default:
       break;
      }  // switch face
    
      
      l = Max(l, Q.getLowIndex());
      h = Min(h, Q.getHighIndex());
/*`==========TESTING==========*/
#if 0
      cout_dbg << "\t variable lo " << Q.getLowIndex() << " hi " << Q.getHighIndex()
               << "\t Celliterator " << l << ","<< h
               << " coarseLow " << coarseLow << " coarseHigh " << coarseHigh << endl; 
#endif
/*===========TESTING==========`*/
      //__________________________________
      //   subCycleProgress_var  = 0
      if(subCycleProgress_var < 1.e-10){
      
       constArrayType q_OldDW;
       coarse_old_dw->getRegion(q_OldDW, label, matl, coarseLevel,
                             coarseLow, coarseHigh);
                             
       for(CellIterator iter(l,h); !iter.done(); iter++){
         IntVector idx = *iter;
         //_________________
         //  deterimine the interpolation weights
         Vector w;
         IntVector cidx = level->mapToCoarser(idx, dir, w);
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
         
/*`==========TESTING==========*/  
         if (idx == IntVector(-1,-1,-1) ) {
          cout << label->getName() << " OLDdw level " << level->getIndex() << " x0 " << x0 << endl;
         } 
/*===========TESTING==========`*/
         
       }  // cell iterator
      } else if(subCycleProgress_var > 1-1.e-10){        /// subCycleProgress_var near 1.0
       constArrayType q_NewDW;
       coarse_new_dw->getRegion(q_NewDW, label, matl, coarseLevel,
                             coarseLow, coarseHigh);
                             
       for(CellIterator iter(l,h); !iter.done(); iter++){
         IntVector idx = *iter;
         //_________________
         //  deterimine the interpolation weights
         Vector w;      
         IntVector cidx = level->mapToCoarser(idx, dir, w);
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
 /*`==========TESTING==========*/
         if (idx == IntVector(-1,-1,-1) ) {
          cout << label->getName() << "NEWDW level " << level->getIndex() << " x1 " << x1 << endl;
         } 
/*===========TESTING==========`*/
       }  // cell iterator
      } else {                    // subCycleProgress_var neither 0 or 1 
       constArrayType q_OldDW, q_NewDW;
       coarse_old_dw->getRegion(q_OldDW, label, matl, coarseLevel,
                             coarseLow, coarseHigh);
       coarse_new_dw->getRegion(q_NewDW, label, matl, coarseLevel,
                             coarseLow, coarseHigh);
                             
      for(CellIterator iter(l,h); !iter.done(); iter++){
         IntVector idx = *iter;
         //_________________
         //  deterimine the interpolation weights
         Vector w;
         IntVector cidx = level->mapToCoarser(idx, dir, w);
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
 /*`==========TESTING==========*/
         if (idx == IntVector(-1,-1,-1) ) {
          cout << label->getName() << " OLDdw + NewDW level " 
               << level->getIndex() << " x0 " << x0 << " x1 " << x1 << endl;
         } 
/*===========TESTING==========`*/
       }
      }
    }
  }
}
//_________________________________________________________________
//   V E C T O R     V E R S I O N     R E F I N E F A C E S
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
#if 0
     //__________________________________
     //  determine low and high cell iter limits
      IntVector l,h;
      patch->getFace(face, IntVector(0,0,0), IntVector(1,1,1), l, h);
      if(face == highFace){
       l += dir;
       h += dir;
      } else if(face != lowFace){
       h += dir;
      }
      switch(face){
      case Patch::xminus:
      case Patch::xplus:
       l-=IntVector(0,1,1);
       h+=IntVector(0,1,1);
       break;
      case Patch::yminus:
      case Patch::yplus:                       
       l-=IntVector(1,0,1);
       h+=IntVector(1,0,1);
       break;
      case Patch::zminus:
      case Patch::zplus:
       l-=IntVector(1,1,0);
       h+=IntVector(1,1,0);
       break;
      default:
       break;
      }
      
      if(face != Patch::xminus && 
         face != Patch::xplus && patch->getBCType(Patch::xminus) == Patch::None){
       l+=IntVector(1,0,0);
      }
      if(face != Patch::xminus && 
         face != Patch::xplus && patch->getBCType(Patch::xplus) == Patch::None){
       h-=IntVector(1,0,0);
      }
      if(face != Patch::yminus && 
         face != Patch::yplus && patch->getBCType(Patch::yminus) == Patch::None){
       l+=IntVector(0,1,0);
      }
      if(face != Patch::yminus && 
         face != Patch::yplus && patch->getBCType(Patch::yplus) == Patch::None){
       h-=IntVector(0,1,0);
      }
      if(face != Patch::zminus && 
         face != Patch::zplus && patch->getBCType(Patch::zminus) == Patch::None){
       l+=IntVector(0,0,1);
      }
      if(face != Patch::zminus && 
         face != Patch::zplus && patch->getBCType(Patch::zplus) == Patch::None){
       h-=IntVector(0,0,1);
      }
#endif

/*`==========TESTING==========*/
      CellIterator iter_tmp = patch->getFaceCellIterator(face, "plusEdgeCells");
      IntVector l = iter_tmp.begin();
      IntVector h = iter_tmp.end(); 
     // cout_dbg << "face " << face << " base iterator " << iter_tmp << endl;
/*===========TESTING==========`*/
      //__________________________________
      //  determine low and high coarse level
      //  iteration limits.
/*`==========TESTING==========*/
 //     IntVector coarseLow = level->mapCellToCoarser(l);
      IntVector coarseLow = level->mapCellToCoarser(l - IntVector(1,1,1)); 

      IntVector coarseHigh = level->mapCellToCoarser(h+level->getRefinementRatio()-IntVector(1,1,1));
      //cout_dbg << " base coarseLow " << coarseLow << " coarse high " << coarseHigh << endl;
/*===========TESTING==========`*/       

      switch(face){
      case Patch::xminus:
       coarseHigh+=IntVector(1,0,0);
       break;
      case Patch::xplus:
       if(basis == Patch::XFaceBased)
         coarseHigh += IntVector(1,0,0);
       else
         coarseLow -= IntVector(1,0,0);
       break;
      case Patch::yminus:
       coarseHigh+=IntVector(0,1,0);
       break;
      case Patch::yplus:
       if(basis == Patch::YFaceBased)
         coarseHigh += IntVector(0,1,0);
       else
         coarseLow -= IntVector(0,1,0);
       break;
      case Patch::zminus:
       coarseHigh+=IntVector(0,0,1);
       break;
      case Patch::zplus:
       if(basis == Patch::ZFaceBased)
         coarseHigh += IntVector(0,0,1);
       else
         coarseLow -= IntVector(0,0,1);
       break;
      default:
       break;
      }  // switch face
      
      l = Max(l, Q.getLowIndex());
      h = Min(h, Q.getHighIndex());
      //__________________________________
      //   subCycleProgress_var  = 0
      if(subCycleProgress_var < 1.e-10){
       constCCVariable<Vector> q_OldDW;
       coarse_old_dw->getRegion(q_OldDW, label, matl, coarseLevel,
                             coarseLow, coarseHigh);
                             
       for(CellIterator iter(l,h); !iter.done(); iter++){
         IntVector idx = *iter;
         //_________________
         //  deterimine the interpolation weights
         Vector w;
         IntVector cidx = level->mapToCoarser(idx, dir, w);
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
         //_________________
         //  deterimine the interpolation weights
         Vector w;      
         IntVector cidx = level->mapToCoarser(idx, dir, w);
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
         //_________________
         //  deterimine the interpolation weights
         Vector w;
         IntVector cidx = level->mapToCoarser(idx, dir, w);
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
//______________________________________________________________________
//
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

//______________________________________________________________________
//  CCVariable<double> version
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

//______________________________________________________________________
//  CCVariable<Vector> version
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
//  SFCXVariable version
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
//  SFCYVariable version
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
//  SFCZVariable version
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
//______________________________________________________________________
//
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

//______________________________________________________________________
//
void AMRICE::coarsen(const ProcessorGroup*,
                        const PatchSubset* patches,
                        const MaterialSubset* matls,
                        DataWarehouse* old_dw,
                        DataWarehouse* new_dw)
{
  cout_doing << "Doing coarsen \t\t\t AMRICE" << '\n';
  const Level* coarseLevel = getLevel(patches);
  const Level* fineLevel = coarseLevel->getFinerLevel().get_rep();
  IntVector rr(fineLevel->getRefinementRatio());
  double ratio = 1./(rr.x()*rr.y()*rr.z());
  
  for(int p=0;p<patches->size();p++){  
    const Patch* coarsePatch = patches->get(p);
    cout_doing << "\t\t on patch " << coarsePatch->getID();
    // Find the overlapping regions...
    Level::selectType finePatches;
    coarsePatch->getFineLevelPatches(finePatches);

    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);

      //__________________________________
      //   D E N S I T Y                     clean out when you've done the test
      CCVariable<double> rho_CC;
      new_dw->getModifiable(rho_CC, lb->rho_CCLabel, matl, coarsePatch);
/*`==========TESTING==========*/
#ifdef test      
      fineToCoarseOperator<double>(rho_CC, lb->rho_CCLabel, matl, new_dw, 
                                   ratio, coarsePatch, coarseLevel, fineLevel);
#else 
/*===========TESTING==========`*/
      //print(density, "before coarsen density");
      
      for(int i=0;i<finePatches.size();i++){
       const Patch* finePatch = finePatches[i];
       constCCVariable<double> fine_den;
       new_dw->get(fine_den, lb->rho_CCLabel, matl, finePatch,
                  Ghost::None, 0);
                  
       IntVector fl(finePatch->getCellLowIndex());
       IntVector fh(finePatch->getCellHighIndex());
       IntVector l(fineLevel->mapCellToCoarser(fl));
       IntVector h(fineLevel->mapCellToCoarser(fh));
       l = Max(l, coarsePatch->getCellLowIndex());
       h = Min(h, coarsePatch->getCellHighIndex());
       
       for(CellIterator iter(l, h); !iter.done(); iter++){
         double rho_tmp=0;
         IntVector fineStart(coarseLevel->mapCellToFiner(*iter));
         
         for(CellIterator inside(IntVector(0,0,0), fineLevel->getRefinementRatio());
             !inside.done(); inside++){
           rho_tmp+=fine_den[fineStart+*inside];
         }
         rho_CC[*iter]=rho_tmp*ratio;
       }
      } // fine patch loop
      //print(density, "coarsened density");
#endif      
      //__________________________________
      //  P R E S S U R E 
      CCVariable<double> press_CC;
      new_dw->getModifiable(press_CC, lb->press_CCLabel, matl, coarsePatch);
#ifdef test      
      fineToCoarseOperator<double>(press_CC, lb->press_CCLabel, matl, new_dw, 
                                   ratio, coarsePatch, coarseLevel, fineLevel);
#else 
      
      for(int i=0;i<finePatches.size();i++){
       const Patch* finePatch = finePatches[i];
       constCCVariable<double> fine_press;
       new_dw->get(fine_press, lb->press_CCLabel, matl, finePatch,
                  Ghost::None, 0);
                  
       IntVector fl(finePatch->getCellLowIndex());
       IntVector fh(finePatch->getCellHighIndex());
       IntVector l(fineLevel->mapCellToCoarser(fl));
       IntVector h(fineLevel->mapCellToCoarser(fh));
       l = Max(l, coarsePatch->getCellLowIndex());
       h = Min(h, coarsePatch->getCellHighIndex());
       
       for(CellIterator iter(l, h); !iter.done(); iter++){
         double press_tmp=0;
         IntVector fineStart(coarseLevel->mapCellToFiner(*iter));
         
         for(CellIterator inside(IntVector(0,0,0), fineLevel->getRefinementRatio());
             !inside.done(); inside++){
           press_tmp+=fine_press[fineStart+*inside];
         }
         press_CC[*iter]=press_tmp*ratio;
       }
      }  // fine patch loop
#endif
      //print(pressure, "coarsened pressure");
      //__________________________________
      //      T E M P E R A T U R E
      CCVariable<double> temp;
      new_dw->getModifiable(temp, lb->temp_CCLabel, matl, coarsePatch);
/*`==========TESTING==========*/
#ifdef test      
      fineToCoarseOperator<double>(temp, lb->temp_CCLabel, matl, new_dw, 
                                   ratio, coarsePatch, coarseLevel, fineLevel);
#else 
/*===========TESTING==========`*/                                   
       for(int i=0;i<finePatches.size();i++){
         const Patch* finePatch = finePatches[i];
         constCCVariable<double> fine_temp;
         new_dw->get(fine_temp, lb->temp_CCLabel, matl, finePatch,
                    Ghost::None, 0);
                    
         IntVector fl(finePatch->getCellLowIndex());
         IntVector fh(finePatch->getCellHighIndex());
         IntVector l(fineLevel->mapCellToCoarser(fl));
         IntVector h(fineLevel->mapCellToCoarser(fh));
         l = Max(l, coarsePatch->getCellLowIndex());
         h = Min(h, coarsePatch->getCellHighIndex());
         
         for(CellIterator iter(l, h); !iter.done(); iter++){
           double temp_tmp=0;
           IntVector fineStart(coarseLevel->mapCellToFiner(*iter));
           
           for(CellIterator inside(IntVector(0,0,0), fineLevel->getRefinementRatio());
              !inside.done(); inside++){
             temp_tmp+=fine_temp[fineStart+*inside];
           }
           temp[*iter]=temp_tmp*ratio;
         }
       }
#endif
       //print(temp, "coarsened temperature");
       
      //___________________________________________
      //     C E L L-C E N T E R-V E L O C I T Y   
       CCVariable<Vector> vel_CC;
       new_dw->getModifiable(vel_CC, lb->vel_CCLabel, matl, coarsePatch);
/*`==========TESTING==========*/
#ifdef test        
        fineToCoarseOperator<Vector>(vel_CC, lb->vel_CCLabel, matl, new_dw, 
                                     ratio, coarsePatch, coarseLevel, fineLevel);
#else  
/*===========TESTING==========`*/                                        
       for(int i=0;i<finePatches.size();i++){
         const Patch* finePatch = finePatches[i];
         constCCVariable<Vector> fine_vel;
         new_dw->get(fine_vel, lb->vel_CCLabel, matl, finePatch,
                    Ghost::None, 0);
                    
         IntVector fl(finePatch->getCellLowIndex());
         IntVector fh(finePatch->getCellHighIndex());
         IntVector l(fineLevel->mapCellToCoarser(fl));
         IntVector h(fineLevel->mapCellToCoarser(fh));
         l = Max(l, coarsePatch->getCellLowIndex());
         h = Min(h, coarsePatch->getCellHighIndex());
         
         for(CellIterator iter(l, h); !iter.done(); iter++){
           Vector vel_tmp= Vector(0,0,0);
           IntVector fineStart(coarseLevel->mapCellToFiner(*iter));
           
           for(CellIterator inside(IntVector(0,0,0), fineLevel->getRefinementRatio());
              !inside.done(); inside++){
             vel_tmp+=fine_vel[fineStart+*inside];
           }
           vel_CC[*iter] =vel_tmp*ratio;
         }
       }
       //print(temp, "coarsened temperature"); 
#endif    
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
                                  const int matl,
                                  DataWarehouse* new_dw,
                                  const double ratio,
                                  const Patch* coarsePatch,
                                  const Level* coarseLevel,
                                  const Level* fineLevel)
{      
  // Find the overlapping regions...
  Level::selectType finePatches;
  coarsePatch->getFineLevelPatches(finePatches);  
                            
  for(int i=0;i<finePatches.size();i++){
    const Patch* finePatch = finePatches[i];
    constCCVariable<T> fine_q_CC;
    new_dw->get(fine_q_CC, varLabel, matl, finePatch,Ghost::None, 0);

    IntVector fl(finePatch->getCellLowIndex());
    IntVector fh(finePatch->getCellHighIndex());
    IntVector l(fineLevel->mapCellToCoarser(fl));
    IntVector h(fineLevel->mapCellToCoarser(fh));
    l = Max(l, coarsePatch->getCellLowIndex());
    h = Min(h, coarsePatch->getCellHighIndex());
    
    T zero(0.0);
    for(CellIterator iter(l, h); !iter.done(); iter++){
      T q_CC_tmp(zero);
      IntVector fineStart(coarseLevel->mapCellToFiner(*iter));

      for(CellIterator inside(IntVector(0,0,0), fineLevel->getRefinementRatio());
           !inside.done(); inside++){
         q_CC_tmp+=fine_q_CC[fineStart+*inside];
      }
      q_CC[*iter] =q_CC_tmp*ratio;
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
