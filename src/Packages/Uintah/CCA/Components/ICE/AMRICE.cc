
#include <Packages/Uintah/CCA/Components/ICE/AMRICE.h>
#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <Packages/Uintah/CCA/Components/Regridder/PerPatchVars.h>
#include <Packages/Uintah/CCA/Ports/SolverInterface.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/PerPatch.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Math/FastMatrix.h>
#include <Core/Math/MiscMath.h>
#include <Core/Util/DebugStream.h>
#include <iomanip>

using namespace Uintah;
using namespace std;
static DebugStream cout_doing("AMRICE_DOING_COUT", false);
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
  ProblemSpecP cfd_ps = params->findBlock("CFD");
  ProblemSpecP ice_ps = cfd_ps->findBlock("ICE");
  ProblemSpecP amr_ps = ice_ps->findBlock("AMR_Refinement_Criteria_Thresholds");
  if(!amr_ps){
    string warn;
    warn ="\n INPUT FILE ERROR:\n <AMR_Refinement_Criteria_Thresholds> "
         " block not found inside of <ICE> block \n";
    throw ProblemSetupException(warn);
  }
  amr_ps->getWithDefault("Density",     d_rho_threshold,     1e100);
  amr_ps->getWithDefault("Temperature", d_temp_threshold,    1e100);
  amr_ps->getWithDefault("Pressure",    d_press_threshold,   1e100);
  amr_ps->getWithDefault("VolumeFrac",  d_vol_frac_threshold,1e100);
  amr_ps->getWithDefault("Velocity",    d_vel_threshold,     1e100);
}
//___________________________________________________________________              
void AMRICE::scheduleInitialize(const LevelP& level,
                                  SchedulerP& sched)
{
  cout_doing << "AMRICE::scheduleInitialize \t\tL-"<<level->getIndex()<< '\n';
  ICE::scheduleInitialize(level, sched);
}
//___________________________________________________________________
void AMRICE::initialize(const ProcessorGroup*,
                           const PatchSubset*, const MaterialSubset*,
                           DataWarehouse*, DataWarehouse*)
{
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
  if(fineLevel->getIndex() > 0){
    cout_doing << "AMRICE::scheduleRefineInterface \t\tL-" 
               << fineLevel->getIndex() << " step "<< step <<'\n';
               
    double subCycleProgress = double(step)/double(nsteps);
    
    Task* task = scinew Task("AMRICE::refineCoarseFineInterface", 
                       this, &AMRICE::refineCoarseFineInterface, 
                       subCycleProgress);
    
    addRefineDependencies(task, lb->press_CCLabel,    step, nsteps);
    addRefineDependencies(task, lb->rho_CCLabel,      step, nsteps);
    addRefineDependencies(task, lb->sp_vol_CCLabel,   step, nsteps);
    addRefineDependencies(task, lb->temp_CCLabel,     step, nsteps);
    addRefineDependencies(task, lb->vel_CCLabel,      step, nsteps);
    
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
 Function~  AMRICE::refineCoarseFineInterface
 Purpose~  
______________________________________________________________________*/
void AMRICE::refineCoarseFineInterface(const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset*,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw,
                                       double subCycleProgress)
{
  const Level* level = getLevel(patches);
  if(level->getIndex() > 0){     
    cout_doing << "Doing refineCoarseFineInterface"<< "\t\t\t\t AMRICE L-" 
               << level->getIndex() << " step " << subCycleProgress<<endl;
    int  numMatls = d_sharedState->getNumICEMatls();
    Ghost::GhostType  gac = Ghost::AroundCells;

    bool dbg_onOff = cout_dbg.active();      // is cout_dbg switch on or off
      
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

        refineCoarseFineBoundaries(patch, press_CC.castOffConst(), new_dw, 
                                   lb->press_CCLabel,  indx,subCycleProgress);

        refineCoarseFineBoundaries(patch, rho_CC.castOffConst(),   new_dw, 
                                   lb->rho_CCLabel,    indx,subCycleProgress);

        refineCoarseFineBoundaries(patch, sp_vol_CC.castOffConst(),new_dw,
                                   lb->sp_vol_CCLabel, indx,subCycleProgress);

        refineCoarseFineBoundaries(patch, temp_CC.castOffConst(),  new_dw,
                                   lb->temp_CCLabel,   indx,subCycleProgress);

        refineCoarseFineBoundaries(patch, vel_CC.castOffConst(),new_dw,
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
              refineCoarseFineBoundaries(patch, q_CC.castOffConst(),new_dw,
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
        desc << "refineInterface_Mat_" << indx << "_patch_"<< patch->getID()
             << " step " << subCycleProgress;
        printData(indx, patch,   1, desc.str(), "press_CC",    press_CC);
        printData(indx, patch,   1, desc.str(), "rho_CC",      rho_CC);
        printData(indx, patch,   1, desc.str(), "sp_vol_CC",   sp_vol_CC);
        printData(indx, patch,   1, desc.str(), "Temp_CC",     temp_CC);
        printVector(indx, patch, 1, desc.str(), "vel_CC", 0,   vel_CC);
#endif
      }
    }
    cout_dbg.setActive(dbg_onOff);  // reset on/off switch for cout_dbg
  }             
}

/*___________________________________________________________________
 Function~  AMRICE::interpolationWrapper--
 Purpose:    depending on which 
_____________________________________________________________________*/
template<class T>
  void interpolationWrapper(CCVariable<T>& q_CL,// course level
                             const Level* coarseLevel,
                             const Level* fineLevel,
                             const IntVector& refineRatio,
                             const IntVector& fl,
                             const IntVector& fh,
                             CCVariable<T>& q_FineLevel,
                             T& error)
{
  linearInterpolation<varType>(test, coarseLevel, fineLevel,
                                     refineRatio, fl,fh, Q, error); 
                                     
  quadraticInterpolation<varType>(test, coarseLevel, fineLevel,
                                     refineRatio, fl,fh, Q, error); 

}

/*___________________________________________________________________
 Function~  AMRICE::linearInterpolation--
_____________________________________________________________________*/
template<class T>
  void linearInterpolation(CCVariable<T>& q_CL,// course level
                             const Level* coarseLevel,
                             const Level* fineLevel,
                             const IntVector& /*refineRatio*/,
                             const IntVector& fl,
                             const IntVector& fh,
                             CCVariable<T>& q_FineLevel,
                             T& error)
{
/*`==========TESTING==========*/
  int ncell = 0;
/*===========TESTING==========`*/  
  Vector c_dx = coarseLevel->dCell();
  Vector inv_c_dx = Vector(1.0)/c_dx;
  
  for(CellIterator iter(fl,fh); !iter.done(); iter++){
    IntVector f_cell = *iter;
    IntVector c_cell = fineLevel->mapCellToCoarser(f_cell);
    
    //__________________________________
    // Offset for coarse level surrounding cells:
    //  -find the normalized distance between the coarse and fine level cell centers  
    Point coarse_cell_pos = coarseLevel->getCellPosition(c_cell);
    Point fine_cell_pos   = fineLevel->getCellPosition(f_cell);
    Vector dist = (fine_cell_pos.asVector() - coarse_cell_pos.asVector()) * inv_c_dx;
  
    // take the sign to determing which cells are the surrounding cells 
    int i = Sign(dist.x());
    int j = Sign(dist.y());
    int k = Sign(dist.z());
    
    dist = Abs(dist);  
/*`==========TESTING==========*/
//cout << "dist " << dist << " i " << i << " j " << j << " k " << k << endl;   
/*===========TESTING==========`*/
    //__________________________________
    //  Find the weights      
    double w0 = (1. - dist.x()) * (1. - dist.y());
    double w1 = dist.x() * (1. - dist.y());
    double w2 = dist.y() * (1. - dist.x());
    double w3 = dist.x() * dist.y(); 
     
            
    T q_XY_Plane_1   // X-Y plane closest to the fine level cell 
        = w0 * q_CL[c_cell] 
        + w1 * q_CL[c_cell + IntVector( i, 0, 0)] 
        + w2 * q_CL[c_cell + IntVector( 0, j, 0)]
        + w3 * q_CL[c_cell + IntVector( i, j, 0)];
                   
    T q_XY_Plane_2   // X-Y plane furthest from the fine level cell
        = w0 * q_CL[c_cell + IntVector( 0, 0, k)] 
        + w1 * q_CL[c_cell + IntVector( i, 0, k)]  
        + w2 * q_CL[c_cell + IntVector( 0, j, k)]  
        + w3 * q_CL[c_cell + IntVector( i, j, k)]; 

    // interpolate the two X-Y planes in the k direction
    q_FineLevel[f_cell] = (1.0 - dist.z()) * q_XY_Plane_1 
                       + dist.z() * q_XY_Plane_2;
    
/*`==========TESTING==========*/
    Vector f_dx = fineLevel->dCell();
    double X = ((f_cell.x() * f_dx.x()) + f_dx.x()/2.0);
    double Y = ((f_cell.y() * f_dx.y()) + f_dx.y()/2.0);
    double Z = ((f_cell.z() * f_dx.z()) + f_dx.z()/2.0);
    
   T exact( X * X * X * Y* Y * Y  * Z * Z *Z );
  // T exact(5.0);
           
    error = error + (q_FineLevel[f_cell] - exact) * (q_FineLevel[f_cell] - exact);
    ncell += 1; 
    
#if 0
    cout  << "f_cell \t" << f_cell << " c_cell "<< c_cell
          << " plane 1 " << q_XY_Plane_1 << " plane2 " << q_XY_Plane_2
          << " q_FineLevel[f_cell] " << q_FineLevel[f_cell]
          <<  " error " << error << " exact " << exact << endl;
#endif
/*===========TESTING==========`*/
  }
  error = error/(double)ncell;
  cout << " error/ncell " << error << endl;
}

/*___________________________________________________________________
 Function~  AMRICE::QuadraticInterpolation--
_____________________________________________________________________*/
template<class T>
  void quadraticInterpolation(CCVariable<T>& q_CL,// course level
                             const Level* coarseLevel,
                             const Level* fineLevel,
                             const IntVector& refineRatio,
                             const IntVector& fl,
                             const IntVector& fh,
                             CCVariable<T>& q_FineLevel,
                             T& error)
{
/*`==========TESTING==========*/
  int ncell = 0;
/*===========TESTING==========`*/
  Vector c_dx = coarseLevel->dCell();
  Vector inv_c_dx = Vector(1.0)/c_dx;
  
  for(CellIterator iter(fl,fh); !iter.done(); iter++){
    IntVector f_cell = *iter;
    IntVector c_cell = fineLevel->mapCellToCoarser(f_cell);
    
    //__________________________________
    // Offset for coarse level surrounding cells:
    //  -find the normalized distance between the coarse and fine level cell centers  
    Point coarse_cell_pos = coarseLevel->getCellPosition(c_cell);
    Point fine_cell_pos   = fineLevel->getCellPosition(f_cell);
    Vector dist = (fine_cell_pos.asVector() - coarse_cell_pos.asVector()) * inv_c_dx;
    
    //dist = Abs(dist); 
    //__________________________________
    //  Find the weights 
    double x = dist.x();
    double y = dist.y();
    double z = dist.z();
    
    double w0_x =  0.5 * x  * (x - 1.0);
    double w1_x = -(x + 1.0)* (x - 1.0);
    double w2_x =  0.5 * x  * (x + 1.0);
    
    double w0_y =  0.5 * y  * (y - 1.0);
    double w1_y = -(y + 1.0)* (y - 1.0);
    double w2_y =  0.5 * y  * (y + 1.0);
    
    double w0_z =  0.5 * z  * (z - 1.0);
    double w1_z = -(z + 1.0)* (z - 1.0);
    double w2_z =  0.5 * z  * (z + 1.0);
    
    FastMatrix w(3, 3);
    //  Q_CL(-1,-1,k)      Q_CL(0,-1,k)          Q_CL(1,-1,k)
    w(0,0) = w0_x * w0_y; w(1,0) = w1_x * w0_y; w(2,0) = w2_x * w0_y;
    w(0,1) = w0_x * w1_y; w(1,1) = w1_x * w1_y; w(2,1) = w2_x * w1_y;
    w(0,2) = w0_x * w2_y; w(1,2) = w1_x * w2_y; w(2,2) = w2_x * w2_y;  
    //  Q_CL(-1, 1,k)      Q_CL(0, 1,k)          Q_CL(1, 1,k)      
    
    vector<T> q_XY_Plane(3);
     
    int k = -2; 
    // loop over the three X-Y planes
    for(int p = 0; p < 3; p++){
      k += 1;

      q_XY_Plane[p]   // X-Y plane
        = w(0,0) * q_CL[c_cell + IntVector( -1, -1, k)]   
        + w(1,0) * q_CL[c_cell + IntVector(  0, -1, k)]           
        + w(2,0) * q_CL[c_cell + IntVector(  1, -1, k)]           
        + w(0,1) * q_CL[c_cell + IntVector( -1,  0, k)]            
        + w(1,1) * q_CL[c_cell + IntVector(  0,  0, k)]    
        + w(2,1) * q_CL[c_cell + IntVector(  1,  0, k)]     
        + w(0,2) * q_CL[c_cell + IntVector( -1,  1, k)]   
        + w(1,2) * q_CL[c_cell + IntVector(  0,  1, k)]     
        + w(2,2) * q_CL[c_cell + IntVector(  1,  1, k)]; 
    }
    
    // interpolate the 3 X-Y planes 
    q_FineLevel[f_cell] = w0_z * q_XY_Plane[0] 
                        + w1_z * q_XY_Plane[1] 
                        + w2_z * q_XY_Plane[2];
/*`==========TESTING==========*/
    Vector f_dx = fineLevel->dCell();
    
    double X = ((f_cell.x() * f_dx.x()) + f_dx.x()/2.0);
    double Y = ((f_cell.y() * f_dx.y()) + f_dx.y()/2.0);
    double Z = ((f_cell.z() * f_dx.z()) + f_dx.z()/2.0);
    
    T exact( X * X * X * Y* Y * Y  * Z * Z *Z );
    //T exact(5.0);
           
    error = error +  (q_FineLevel[f_cell] - exact) * (q_FineLevel[f_cell] - exact);
    ncell += 1; 
#if 0
    cout  << "f_cell \t" << f_cell << " c_cell "<< c_cell
          << " plane 1 " << q_XY_Plane[0] << " plane2 " << q_XY_Plane[1] << " plane3 "<< q_XY_Plane[3]
          << " q_FineLevel[f_cell] " << q_FineLevel[f_cell]
          <<  " error " << error << " exact " << exact << endl;    
#endif
    
//    cout  << "f_cell \t" << f_cell << " c_cell "<< c_cell<< " q_FineLevel[f_cell] " << q_FineLevel[f_cell]
//          <<  " error " << error << " exact " << exact << endl;
/*===========TESTING==========`*/                        
  } 
  error = error/(double)ncell;
  cout << " error/ncell " << error << endl;
}

/*___________________________________________________________________
 Function~  AMRICE::refine_CF_interfaceOperator-- 
_____________________________________________________________________*/
template<class varType>
void refine_CF_interfaceOperator(const Patch* patch, 
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
  cout_dbg << *patch << endl;
  
/*`==========TESTING==========*/
    varType error1(0.0);
    varType error2(0.0); 
    varType error(0.0);
/*===========TESTING==========`*/
  
  
  for(Patch::FaceType face = Patch::startFace;
      face <= Patch::endFace; face=Patch::nextFace(face)){

   if(patch->getBCType(face) == Patch::Coarse) {
     //__________________________________
     // fine level hi & lo cell iter limits
     // coarselevel hi and low index
      CellIterator iter_tmp = patch->getFaceCellIterator(face, "plusEdgeCells");
      IntVector fl = iter_tmp.begin();
      IntVector fh = iter_tmp.end(); 
      IntVector refineRatio = fineLevel->getRefinementRatio();
      IntVector coarseLow  = fineLevel->mapCellToCoarser(fl);
      //IntVector coarseHigh = fineLevel->mapCellToCoarser(fh+refineRatio-IntVector(1,1,1));
      IntVector coarseHigh = fineLevel->mapCellToCoarser(fh);

/*`==========TESTING==========*/
 refineRatio = IntVector(4,4,4);
       cout << " refineRatio "<< refineRatio << " fl " << fl << " fh " << fh
          << " coarseHigh " << coarseHigh << " coarseLow " << coarseLow << endl;
/*===========TESTING==========`*/
      

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
      cout_dbg << " face " << face 
               << " FineLevel iterator" << iter_tmp.begin() << " " << iter_tmp.end() 
               << " \t coarseLevel iterator " << coarseLow << " " << coarseHigh<<endl;
      
/*`==========TESTING==========*/
        cout.setf(ios::scientific,ios::floatfield);
        cout.precision(16);

    CCVariable<varType> test;

    Vector c_dx = coarseLevel->dCell();
    for(Level::const_patchIterator iter = coarseLevel->patchesBegin();
                                  iter != coarseLevel->patchesEnd(); iter++){
      const Patch* patch2 = *iter;
      coarse_new_dw->allocateTemporary(test,patch2);
      for(CellIterator iter = patch2->getCellIterator(); !iter.done();iter++){
        IntVector c = *iter; 
        double X = ((c.x() * c_dx.x()) + c_dx.x()/2.0);
        double Y = ((c.y() * c_dx.y()) + c_dx.y()/2.0);
        double Z = ((c.z() * c_dx.z()) + c_dx.z()/2.0);
        
        //test[c] = varType( X * X * X * Y* Y * Y  * Z * Z *Z );
        test[c] = varType(5.0);
      }
      
      cout << " linearInterpolation:  face " << face;
      linearInterpolation<varType>(test, coarseLevel, fineLevel,
                                     refineRatio, fl,fh, Q, error); 
      error1 = error1 + error;
      error = varType(0.0);

      cout << " quadraticInterpolation: face " << face;                             
      quadraticInterpolation(test, coarseLevel, fineLevel,
                             refineRatio, fl,fh, Q, error);
      error2 = error2 + error;                            
    }
/*===========TESTING==========`*/
      
      //__________________________________
      //   subCycleProgress_var  = 0
      //  interpolation using the coarse_old_dw data
      if(subCycleProgress_var < 1.e-10){
       constCCVariable<varType> q_OldDW;
       coarse_old_dw->getRegion(q_OldDW, label, matl, coarseLevel,
                                coarseLow, coarseHigh);

#if 0
       linearInterpolation<varType>(q_OldDW, coarseLevel, fineLevel,
                                      refineRatio, fl,fh, Q);
#endif
      } 
       
       //__________________________________
       // subCycleProgress_var near 1.0
       //  interpolation using the coarse_new_dw data
      else if(subCycleProgress_var > 1-1.e-10){ 
       constCCVariable<varType> q_NewDW;
       coarse_new_dw->getRegion(q_NewDW, label, matl, coarseLevel,
                             coarseLow, coarseHigh);
#if 0                               
       linearInterpolation<varType>(q_NewDW, coarseLevel, fineLevel,
                                      refineRatio, fl,fh, Q); 
 
#endif
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
        
        Q_old.initialize(varType(0));
        Q_new.initialize(varType(0));
                             
#if 0                              
        linearInterpolation<varType>(q_OldDW, coarseLevel, fineLevel,
                                       refineRatio, fl,fh, Q_old);
                                      
        linearInterpolation<varType>(q_NewDW, coarseLevel, fineLevel,
                                       refineRatio, fl,fh, Q_New);
#endif
        // Linear interpolation in time
        for(CellIterator iter(fl,fh); !iter.done(); iter++){
          IntVector f_cell = *iter;
          Q[f_cell] = (1. - subCycleProgress_var)*Q_old[f_cell] 
                          + subCycleProgress_var *Q_new[f_cell];
        }
      }
    }
  }  // face 
  cout << "Sum over all faces " << endl;
  cout << "linearInterpolator  error: " << error1 << endl;
  cout << "QuadraticInterpolator error: " << error2 << endl;
  
  cout_dbg.setActive(false);// turn off the switch for cout_dbg
}

/*___________________________________________________________________
 Function~  AMRICE::addRefineDependencies--
 Purpose:  
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
 Function~  AMRICE::refineCoarseFineBoundaries--    D O U B L E  
_____________________________________________________________________*/
void AMRICE::refineCoarseFineBoundaries(const Patch* patch,
                           CCVariable<double>& val,
                           DataWarehouse* fine_new_dw,
                           const VarLabel* label,
                           int matl,
                           double subCycleProgress_var)
{
  cout_dbg << "\t refineCoarseFineBoundaries ("<<label->getName() << ") \t" 
           << " subCycleProgress_var " << subCycleProgress_var<< '\n';
  DataWarehouse* coarse_old_dw = 
                 fine_new_dw->getOtherDataWarehouse(Task::CoarseOldDW);
  DataWarehouse* coarse_new_dw = 
                 fine_new_dw->getOtherDataWarehouse(Task::CoarseNewDW);
  
  const Level* level = patch->getLevel();
  const Level* coarseLevel = level->getCoarserLevel().get_rep();
  
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
  cout_dbg << "\t refineCoarseFineBoundaries ("<<label->getName() << ") \t" 
           << " subCycleProgress_var " << subCycleProgress_var<< '\n';
  DataWarehouse* coarse_old_dw = 
                 fine_new_dw->getOtherDataWarehouse(Task::CoarseOldDW);
  DataWarehouse* coarse_new_dw = 
                 fine_new_dw->getOtherDataWarehouse(Task::CoarseNewDW);
  
  const Level* level = patch->getLevel();
  const Level* coarseLevel = level->getCoarserLevel().get_rep();

  refine_CF_interfaceOperator<Vector>
    (patch, level, coarseLevel, val, label, subCycleProgress_var, matl,
     fine_new_dw, coarse_old_dw, coarse_new_dw);
}

/*___________________________________________________________________
 Function~  AMRICE::scheduleRefine--  
_____________________________________________________________________*/
void AMRICE::scheduleRefine(const LevelP& fineLevel,
                               SchedulerP& sched)
{
  Ghost::GhostType  gn = Ghost::None; 
  cout_dbg << "AMRICE::scheduleRefine\t\t\t\tL-" << fineLevel->getIndex() << '\n';
  Task* task = scinew Task("refine",this, &AMRICE::refine);

  MaterialSubset* subset = scinew MaterialSubset;
  subset->add(0);

  task->requires(Task::NewDW, lb->press_CCLabel,
               0, Task::CoarseLevel, subset, Task::OutOfDomain, gn, 0);
                 
  task->requires(Task::NewDW, lb->rho_CCLabel,
               0, Task::CoarseLevel, 0, Task::NormalDomain, gn, 0);
               
  task->requires(Task::NewDW, lb->sp_vol_CCLabel,
               0, Task::CoarseLevel, 0, Task::NormalDomain, gn, 0);
  
  task->requires(Task::NewDW, lb->temp_CCLabel,
               0, Task::CoarseLevel, 0, Task::NormalDomain, gn, 0);
  
  task->requires(Task::NewDW, lb->vel_CCLabel,
               0, Task::CoarseLevel, 0, Task::NormalDomain, gn, 0);

  //__________________________________
  // Model Variables.
  if(d_modelSetup && d_modelSetup->tvars.size() > 0){
    vector<TransportedVariable*>::iterator iter;

    for(iter = d_modelSetup->tvars.begin();
       iter != d_modelSetup->tvars.end(); iter++){
      TransportedVariable* tvar = *iter;
      task->requires(Task::NewDW, tvar->var,
                  0, Task::CoarseLevel, 0, Task::NormalDomain, gn, 0);
      task->computes(tvar->var);
    }
  }
  
  task->computes(lb->press_CCLabel);
  task->computes(lb->rho_CCLabel);
  task->computes(lb->sp_vol_CCLabel);
  task->computes(lb->temp_CCLabel);
  task->computes(lb->vel_CCLabel);

  sched->addTask(task, fineLevel->eachPatch(), d_sharedState->allMaterials()); 
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
  cout_doing << "Doing refine \t\t\t\t\t\t AMRICE";
  const Level* fineLevel = getLevel(patches);
  const Level* coarseLevel = fineLevel->getCoarserLevel().get_rep();
  
  IntVector rr(fineLevel->getRefinementRatio());
  double invRefineRatio = 1./(rr.x()*rr.y()*rr.z());
  
  for(int p=0;p<patches->size();p++){  
    const Patch* finePatch = patches->get(p);
    cout_doing << " patch " << finePatch->getID()<< endl;
    
    Level::selectType coarsePatches;
    finePatch->getCoarseLevelPatches(coarsePatches);

    for(int m = 0;m<matls->size();m++){
      int indx = matls->get(m);

      CCVariable<double> rho_CC, press_CC, temp, sp_vol_CC;
      CCVariable<Vector> vel_CC;
      new_dw->allocateAndPut(press_CC, lb->press_CCLabel,  indx, finePatch);
      new_dw->allocateAndPut(rho_CC,   lb->rho_CCLabel,    indx, finePatch);
      new_dw->allocateAndPut(sp_vol_CC,lb->sp_vol_CCLabel, indx, finePatch);
      new_dw->allocateAndPut(temp,     lb->temp_CCLabel,   indx, finePatch);
      new_dw->allocateAndPut(vel_CC,   lb->vel_CCLabel,    indx, finePatch);  
      
      // refine data
      CoarseToFineOperator<double>(press_CC,  lb->press_CCLabel,indx, new_dw, 
                         invRefineRatio, finePatch, fineLevel, coarseLevel);
                         
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
            CoarseToFineOperator<double>(q_CC, tvar->var, indx, new_dw, 
                       invRefineRatio, finePatch, fineLevel, coarseLevel);
          #if 0  
            string name = tvar->var->getName();
            printData(indx, finePatch, 1, "refine_models", name, q_CC);
          #endif                 
          }
        }
      }    
 #if 0
      //__________________________________
      //  Print Data 
      ostringstream desc;     
      desc << "refine_Mat_" << indx << "_patch_"<< finePatch->getID();
      printData(indx, finePatch,   1, desc.str(), "press_CC",    press_CC);
      printData(indx, finePatch,   1, desc.str(), "rho_CC",      rho_CC);
      printData(indx, finePatch,   1, desc.str(), "sp_vol_CC",   sp_vol_CC);
      printData(indx, finePatch,   1, desc.str(), "Temp_CC",     temp);
      printVector(indx, finePatch, 1, desc.str(), "vel_CC", 0,   vel_CC);
#endif
    }
  }  // course patch loop 
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
  Level::selectType coarsePatches;
  finePatch->getCoarseLevelPatches(coarsePatches);
  IntVector extraCells(1,1,1);    // ICE always has 1 layer of extra cells
                            
  for(int i=0;i<coarsePatches.size();i++){
    const Patch* coarsePatch = coarsePatches[i];
    
    constCCVariable<T> coarse_q_CC;
    new_dw->get(coarse_q_CC, varLabel, indx, coarsePatch,Ghost::None, 0);

    IntVector cl(coarsePatch->getCellLowIndex());
    IntVector ch(coarsePatch->getCellHighIndex());
    IntVector fl(coarseLevel->mapCellToFiner(cl) + extraCells);
    IntVector fh(coarseLevel->mapCellToFiner(ch) - extraCells);
    
    fl = Max(fl, finePatch->getCellLowIndex());
    fh = Min(fh, finePatch->getCellHighIndex());
    
    cout_dbg << " coarseToFineOperator: coarsePatch " << *coarsePatch
             << " finePatch " << *finePatch << endl;
    //IntVector refinementRatio = coarseLevel->getRefinementRatio();
    
    // iterate over fine level cells
    for(CellIterator iter(fl, fh); !iter.done(); iter++){
      IntVector c = *iter;
      IntVector coarseCell = fineLevel->mapCellToCoarser(c);
      q_CC[c] = coarse_q_CC[coarseCell];
    }
  }
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

  //__________________________________
  // Model Variables.
  if(d_modelSetup && d_modelSetup->tvars.size() > 0){
    vector<TransportedVariable*>::iterator iter;

    for(iter = d_modelSetup->tvars.begin();
       iter != d_modelSetup->tvars.end(); iter++){
      TransportedVariable* tvar = *iter;
      task->requires(Task::NewDW, tvar->var,
                  0, Task::FineLevel, 0, Task::NormalDomain, gn, 0);
      task->modifies(tvar->var);
    }
  }
  
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
                        DataWarehouse*,
                        DataWarehouse* new_dw)
{
  cout_doing << "Doing coarsen \t\t\t\t\t\t AMRICE";
  const Level* coarseLevel = getLevel(patches);
  const Level* fineLevel = coarseLevel->getFinerLevel().get_rep();
  
  IntVector rr(fineLevel->getRefinementRatio());
  double invRefineRatio = 1./(rr.x()*rr.y()*rr.z());
  
  bool dbg_onOff = cout_dbg.active();      // is cout_dbg switch on or off
  
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
            fineToCoarseOperator<double>(q_CC, tvar->var, indx, new_dw, 
                       invRefineRatio, coarsePatch, coarseLevel, fineLevel);
          #if 0  
            string name = tvar->var->getName();
            printData(indx, coarsePatch, 1, "coarsen_models", name, q_CC);
          #endif                 
          }
        }
      }    
 #if 0
      //__________________________________
      //  Print Data 
      ostringstream desc;     
      desc << "coarsen_Mat_" << indx << "_patch_"<< coarsePatch->getID();
      printData(indx, coarsePatch,   1, desc.str(), "press_CC",    press_CC);
      printData(indx, coarsePatch,   1, desc.str(), "rho_CC",      rho_CC);
      printData(indx, coarsePatch,   1, desc.str(), "sp_vol_CC",   sp_vol_CC);
      printData(indx, coarsePatch,   1, desc.str(), "Temp_CC",     temp);
      printVector(indx, coarsePatch, 1, desc.str(), "vel_CC", 0,   vel_CC);
#endif
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
                            
  for(int i=0;i<finePatches.size();i++){
    const Patch* finePatch = finePatches[i];
    
    constCCVariable<T> fine_q_CC;
    new_dw->get(fine_q_CC, varLabel, indx, finePatch,Ghost::None, 0);

    IntVector fl(finePatch->getInteriorCellLowIndex());
    IntVector fh(finePatch->getInteriorCellHighIndex());
    IntVector cl(fineLevel->mapCellToCoarser(fl));
    IntVector ch(fineLevel->mapCellToCoarser(fh));
    
    cl = Max(cl, coarsePatch->getCellLowIndex());
    ch = Min(ch, coarsePatch->getCellHighIndex());
    
    cout_dbg << " fineToCoarseOperator: finePatch "<< fl << " " << fh 
             << " coarsePatch "<< cl << " " << ch << endl;
             
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
  cout_dbg.setActive(false);// turn off the switch for cout_dbg
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
  cout_doing << "AMRICE::scheduleErrorEstimate \t\t\t\tL-" 
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
  
  t->computes(lb->rho_CC_gradLabel);
  t->computes(lb->temp_CC_gradLabel);
  t->computes(lb->vel_CC_mag_gradLabel);
  t->computes(lb->vol_frac_CC_gradLabel);
  t->computes(lb->press_CC_gradLabel);
  
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
Function~  AMRICE::compute_q_CC_gradient--
Purpose~   computes the gradient of q_CC in each direction.
           First order central difference.
______________________________________________________________________*/
void AMRICE::compute_q_CC_gradient( constCCVariable<double>& q_CC,
                                    CCVariable<Vector>& q_CC_grad,
                                    const Patch* patch) 
{                  
  Vector dx = patch->dCell(); 
      
  for(int dir = 0; dir <3; dir ++ ) { 
    double inv_dx = 0.5 /dx[dir];
    for(CellIterator iter = patch->getCellIterator();!iter.done();iter++){
        IntVector c = *iter;
        IntVector r = c;
        IntVector l = c;
        r[dir] += 1;
        l[dir] -= 1;
        q_CC_grad[c][dir] = (q_CC[r] - q_CC[l])*inv_dx;
    }
  }
}
//______________________________________________________________________
//          vector version
void AMRICE::compute_q_CC_gradient( constCCVariable<Vector>& q_CC,
                                    CCVariable<Vector>& q_CC_grad,
                                    const Patch* patch) 
{                  
  Vector dx = patch->dCell(); 
  
  //__________________________________
  // Vectors:  take the gradient of the magnitude
  for(int dir = 0; dir <3; dir ++ ) { 
    double inv_dx = 0.5 /dx[dir];
    for(CellIterator iter = patch->getCellIterator();!iter.done();iter++){
        IntVector c = *iter;
        IntVector r = c;
        IntVector l = c;
        r[dir] += 1;
        l[dir] -= 1;
        q_CC_grad[c][dir] = (q_CC[r].length() - q_CC[l].length())*inv_dx;
    }
  }
}
/*_____________________________________________________________________
 Function~  AMRICE::set_refinementFlags
______________________________________________________________________*/         
void AMRICE::set_refineFlags( CCVariable<Vector>& q_CC_grad,
                              double threshold,
                              CCVariable<int>& refineFlag,
                              PerPatch<PatchFlagP>& refinePatchFlag,
                              const Patch* patch) 
{                  
  PatchFlag* refinePatch = refinePatchFlag.get().get_rep();
  for(CellIterator iter = patch->getCellIterator();!iter.done();iter++){
    IntVector c = *iter;
    if( q_CC_grad[c].length() > threshold){
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
  cout_doing << "Doing errorEstimate \t\t\t\t\t AMRICE"<< endl;
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    
    Ghost::GhostType  gac  = Ghost::AroundCells;
    const VarLabel* refineFlagLabel = d_sharedState->get_refineFlag_label();
    const VarLabel* refinePatchLabel= d_sharedState->get_refinePatchFlag_label();
    
    CCVariable<int> refineFlag;
    new_dw->getModifiable(refineFlag, refineFlagLabel, 0, patch);      

    PerPatch<PatchFlagP> refinePatchFlag;
    new_dw->get(refinePatchFlag, refinePatchLabel, 0, patch);


    //__________________________________
    //  PRESSURE       --- just computes the gradient
    //  I still need to figure out how 
    //  set the refinement flags for this index 0
    //  and then do it again in the matl loop below
    constCCVariable<double> press_CC;
    CCVariable<Vector> press_CC_grad;
    
    new_dw->get(press_CC, lb->press_CCLabel,    0,patch,gac,1);
    new_dw->allocateAndPut(press_CC_grad,
                       lb->press_CC_gradLabel,  0,patch);
    
    compute_q_CC_gradient(press_CC, press_CC_grad, patch);
   // set_refineFlags( press_CC_grad, d_press_threshold,refineFlag, 
   //                         refinePatchFlag, patch);
    //__________________________________
    //  RHO, TEMP, VEL_CC, VOL_FRAC
    int numICEMatls = d_sharedState->getNumICEMatls();
    for(int m=0;m < numICEMatls;m++){
      Material* matl = d_sharedState->getICEMaterial( m );
      int indx = matl->getDWIndex();
              
      constCCVariable<double> rho_CC, temp_CC, vol_frac_CC;
      constCCVariable<Vector> vel_CC;
      CCVariable<Vector> rho_CC_grad, temp_CC_grad; 
      CCVariable<Vector> vel_CC_mag_grad, vol_frac_CC_grad;
      
      new_dw->get(rho_CC,      lb->rho_CCLabel,      indx,patch,gac,1);
      new_dw->get(temp_CC,     lb->temp_CCLabel,     indx,patch,gac,1);
      new_dw->get(vel_CC,      lb->vel_CCLabel,      indx,patch,gac,1);
      new_dw->get(vol_frac_CC, lb->vol_frac_CCLabel, indx,patch,gac,1);

      new_dw->allocateAndPut(rho_CC_grad,     
                         lb->rho_CC_gradLabel,     indx,patch);
      new_dw->allocateAndPut(temp_CC_grad,    
                         lb->temp_CC_gradLabel,    indx,patch);
      new_dw->allocateAndPut(vel_CC_mag_grad, 
                         lb->vel_CC_mag_gradLabel, indx,patch);
      new_dw->allocateAndPut(vol_frac_CC_grad,
                         lb->vol_frac_CC_gradLabel,indx,patch);
      
      //__________________________________
      // compute the gradients and set the refinement flags
                                        // Density
      compute_q_CC_gradient(rho_CC,      rho_CC_grad,      patch); 
      set_refineFlags( rho_CC_grad,     d_rho_threshold,refineFlag, 
                            refinePatchFlag, patch);
      
                                        // Temperature
      compute_q_CC_gradient(temp_CC,     temp_CC_grad,     patch); 
      set_refineFlags( temp_CC_grad,    d_temp_threshold,refineFlag, 
                            refinePatchFlag, patch);
      
                                        // Vol Fraction
      compute_q_CC_gradient(vol_frac_CC, vol_frac_CC_grad, patch); 
      set_refineFlags( vol_frac_CC_grad, d_vol_frac_threshold,refineFlag, 
                            refinePatchFlag, patch);
      
                                        // Velocity
      compute_q_CC_gradient(vel_CC,      vel_CC_mag_grad,  patch); 
      set_refineFlags( vel_CC_mag_grad, d_vel_threshold,refineFlag, 
                            refinePatchFlag, patch);
    }  // matls
  }  // patches
}
