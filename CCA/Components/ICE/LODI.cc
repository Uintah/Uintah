#include <Packages/Uintah/CCA/Components/ICE/LODI.h>
#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <Packages/Uintah/CCA/Components/ICE/EOS/EquationOfState.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Labels/ICELabel.h>
#include <Packages/Uintah/Core/Math/MiscMath.h>
#include <Core/Util/DebugStream.h>
#include <Core/Math/MiscMath.h>
#include <typeinfo>

using namespace Uintah;
namespace Uintah {
//__________________________________
//  To turn on couts
//  setenv SCI_DEBUG "LODI_DOING_COUT:+, LODI_DBG_COUT:+"
static DebugStream cout_doing("LODI_DOING_COUT", false);
static DebugStream cout_dbg("LODI_DBG_COUT", false);

/* ---------------------------------------------------------------------
 Function~  read_LODI_BC_inputs--   
 Purpose~   returns if we are using LODI BC on any face and reads in
            any lodi parameters 
 ---------------------------------------------------------------------  */
bool read_LODI_BC_inputs(const ProblemSpecP& prob_spec,
                         Lodi_user_inputs* userInputs)
{
  //__________________________________
  // search the BoundaryConditions problem spec
  // determine if LODI bcs are specified
  ProblemSpecP grid_ps= prob_spec->findBlock("Grid");
  ProblemSpecP bc_ps  = grid_ps->findBlock("BoundaryConditions");
 
  bool usingLODI = false;
  
  for (ProblemSpecP face_ps = bc_ps->findBlock("Face");face_ps != 0; 
                    face_ps=face_ps->findNextBlock("Face")) {
    
    for(ProblemSpecP bc_iter = face_ps->findBlock("BCType"); bc_iter != 0;
                     bc_iter = bc_iter->findNextBlock("BCType")){
      map<string,string> bc_type;
      bc_iter->getAttributes(bc_type);
      if (bc_type["var"] == "LODI") {
       usingLODI = true;
      }
    }
  }
  //__________________________________
  //  read in variables
  if(usingLODI ){
    ProblemSpecP lodi = bc_ps->findBlock("LODI");
    if (!lodi) {
      string warn="ERROR:\n Inputs:Boundary Conditions: Cannot find LODI block";
      throw ProblemSetupException(warn);
    }

    lodi->require("press_infinity",userInputs->press_infinity);
    lodi->getWithDefault("sigma",  userInputs->sigma, 0.27);
  }
  
  if (usingLODI) {
    cout << "\n WARNING:  LODI boundary conditions are "
         << " NOT set during the problem initialization \n " << endl;
  }
  return usingLODI;
}
/* --------------------------------------------------------------------- 
 Function~  is_LODI_face--   
 Purpose~   returns true if this face on this patch is using LODI bcs
 ---------------------------------------------------------------------  */
bool is_LODI_face(const Patch* patch,
                  Patch::FaceType face,
                  SimulationStateP& sharedState)
{
  bool is_lodi_face = false;
  int numMatls = sharedState->getNumICEMatls();

  for (int m = 0; m < numMatls; m++ ) {
    ICEMaterial* ice_matl = sharedState->getICEMaterial(m);
    int indx= ice_matl->getDWIndex();
    bool lodi_pressure =    patch->haveBC(face,indx,"LODI","Pressure");
    bool lodi_density  =    patch->haveBC(face,indx,"LODI","Density");
    bool lodi_temperature = patch->haveBC(face,indx,"LODI","Temperature");
    bool lodi_velocity =    patch->haveBC(face,indx,"LODI","Velocity");

    if (lodi_pressure || lodi_density || lodi_temperature || lodi_velocity) {
      is_lodi_face = true; 
    }
  }
  return is_lodi_face;
}

/* --------------------------------------------------------------------- 
 Function~  lodi_getVars_pressBC-- 
 Purpose~   Get all the data required for setting the pressure BC
 ---------------------------------------------------------------------  */
void lodi_getVars_pressBC( const Patch* patch,
                           Lodi_vars_pressBC* lodi_vars,
                           ICELabel* lb,
                           SimulationStateP sharedState,
                           DataWarehouse* old_dw,
                           DataWarehouse* new_dw)
{
  cout_doing << "lodi_getVars_pressBC on patch "<<patch->getID()<< endl;
  int numMatls = sharedState->getNumMatls();
  StaticArray<constCCVariable<double> > Temp_CC(numMatls);
  StaticArray<constCCVariable<double> > f_theta_CC(numMatls);
  StaticArray<constCCVariable<double> > gamma(numMatls);
  StaticArray<constCCVariable<double> > cv(numMatls);
  Ghost::GhostType  gn = Ghost::None;
  
  for(int m = 0; m < numMatls; m++) {
    Material* matl = sharedState->getMaterial( m );
    int indx = matl->getDWIndex();
    ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);

    if(ice_matl){                // I C E
      old_dw->get(Temp_CC[m],     lb->temp_CCLabel,       indx,patch,gn,0);
      new_dw->get(f_theta_CC[m],  lb->f_theta_CCLabel,    indx,patch,gn,0);
      new_dw->get(gamma[m],       lb->gammaLabel,         indx,patch,gn,0);
      new_dw->get(cv[m],          lb->specific_heatLabel, indx,patch,gn,0);
    }
    if(mpm_matl){                // M P M
      new_dw->get(Temp_CC[m],     lb->temp_CCLabel,    indx,patch,gn,0);
      new_dw->get(f_theta_CC[m],  lb->f_theta_CCLabel, indx,patch,gn,0);
    }
    lodi_vars->f_theta[m] = f_theta_CC[m];
    lodi_vars->Temp_CC[m] = Temp_CC[m];
    lodi_vars->gamma[m]   = gamma[m];
    lodi_vars->cv[m]      = cv[m];
  }
}
/*__________________________________________________________________
 Function~ Di-- do the actual work of computing di
____________________________________________________________________*/
inline void Di(StaticArray<CCVariable<Vector> >& d,
               const int dir,
               const IntVector& c,
               const Patch::FaceType face,
               const Vector domainLength,
               const Lodi_user_inputs* user_inputs,
               const double maxMach,
               const double press,
               const double speedSound,
               const double rho,
               const double normalVel,
               const double drho_dx,
               const double dp_dx,
               const Vector dVel_dx)
{
  IntVector dd;  // directions debugging 

  //__________________________________
  // compute Li terms
  //  see table 7 of Sutherland and Kennedy 
  double L3 = -9e1000, L4 = -9e1000;
  double speedSoundsqr = speedSound * speedSound;
  
  double A = rho * speedSound * dVel_dx[dir];
  double L1 = (normalVel - speedSound) * (dp_dx - A);
  double L2 = normalVel * (speedSoundsqr * drho_dx - dp_dx);
  double L5 = (normalVel + speedSound) * (dp_dx + A);
  
  if (dir == 0) {   // X-normal direction
    L3 = normalVel * dVel_dx[1];  // u dv/dx
    L4 = normalVel * dVel_dx[2];  // u dw/dx
    dd=IntVector(0,1,2);
  }
  if (dir == 1) {   // Y-normal direction
    L3 = normalVel * dVel_dx[0];   // v du/dy
    L4 = normalVel * dVel_dx[2];   // v dw/dy
    dd=IntVector(1,0,2);
  }
  if (dir == 2) {   // Z-normal direction
    L3 = normalVel * dVel_dx[0];   // w du/dz
    L4 = normalVel * dVel_dx[1];   // w dv/dz
    dd=IntVector(2,0,1);
  }
   
  //__________________________________
  //  user_inputs
  double p_infinity = user_inputs->press_infinity;
  double sigma      = user_inputs->sigma;
  double sp         = 0;
  
  //__________________________________
  //  Modify the Li terms
  //  equation 8 & 9 of Sutherland
  double K =  sigma * (1.0 - maxMach*maxMach)/domainLength[dir];
  
  if (face == Patch::xminus ||     // LEFT  BOTTOM  BACK FACES
      face == Patch::yminus ||
      face == Patch::zminus) {
    if(normalVel < 0 ){   // flowing out of the domain
      L5 = K * speedSound * (press - p_infinity) + 0.5 * sp;
    }
    if(normalVel >=0) {   // flowing into the domain
      L1 = L1;            // unchanged
      L2 = -sp/speedSoundsqr;
      L3 = 0.0;
      L4 = 0.0;
      L5 = 0.5 * sp;
    }
  }
  
  if (face == Patch::xplus ||     //  RIGHT  TOP  FRONT FACES
      face == Patch::yplus ||
      face == Patch::zplus) {
    double K = 0.0;
    if(normalVel > 0 ){   // flowing out of the domain
      L1 = K * speedSound * (press - p_infinity) + 0.5 * sp;
    }
    if(normalVel <=0) {   // flowing into the domain
      L1 = 0.5 * sp;
      L2 = -sp/speedSoundsqr;
      L3 = 0.0;
      L4 = 0.0;
      L5 = L5;            //unchanged
    }
  }  
  
  //__________________________________
  //  compute Di terms based on the 
  // modified Ls
  d[1][c][dir] = (L2 + 0.5 * (L1 + L5))/speedSoundsqr;
  d[2][c][dir] = 0.5 * (L5 + L1);
 
  if (dir == 0) {   // X-normal direction
    d[3][c][dir] = 0.5 * (L5 - L1)/(rho * speedSound);
    d[4][c][dir] = L3;
    d[5][c][dir] = L4;
    dd=IntVector(0,1,2);
  }
  if (dir == 1) {   // Y-normal direction
    d[3][c][dir] = L3;
    d[4][c][dir] = 0.5 * (L5 - L1)/(rho * speedSound);
    d[5][c][dir] = L4;
    dd=IntVector(1,0,2);
  }
  if (dir == 2) {   // Z-normal direction
    d[3][c][dir] = L3;
    d[4][c][dir] = L4;
    d[5][c][dir] = 0.5 * (L5 - L1)/(rho * speedSound);
    dd=IntVector(2,0,1);
  }
  
#if 0
   //__________________________________
   //  debugging
   // hardCode the cells you want to intergate;

   vector<IntVector> dbgCells;
   dbgCells.push_back(IntVector(50,-1,2));
   dbgCells.push_back(IntVector(50,0,2));        
   dbgCells.push_back(IntVector(50,-1,-1));        
   dbgCells.push_back(IntVector(50,-1,5)); 
     
   for (int i = 0; i<(int) dbgCells.size(); i++) {
     if (c == dbgCells[i]) {
    
       cout << c << " dd[0] " << dd[0] << " dd[1] " << dd[1] << " dd[2] " << dd[2]<< endl;
       cout << " normalVel " << normalVel << " drho_dx " 
            << drho_dx << " dp_dx " << dp_dx << endl;
       cout << " dvel_dx " << dVel_dx<< endl;
       cout << " A:  \t" << "rho * C * du"<<dd[0]<<"/dx"<<dd[0]<<endl;
       cout << " L1: \t" << L1 << " (u"<< dd[0] <<" - c^2)*(dp/dx"<<dd[0]<< " - A)"<<endl;
       cout << " L2: \t" << L2 << "  u"<< dd[0] <<"*(c^2*drho/dx"<<dd[0]<< " - dp/dx"<<dd[0]<<")"<<endl;
       cout << " L3: \t" << L3 << "  u"<< dd[0] <<"*du"<<dd[1]<<"/dx"<<dd[0]<<endl;
       cout << " L4: \t" << L4 << "  u"<< dd[0] <<"*du"<<dd[2]<<"/dx"<<dd[0]<<endl;
       cout << " L5: \t" << L5 << " (u"<< dd[0] <<" + c^2)*(dp/dx"<<dd[0]<< " + A)"<<endl;
       for (int i = 1; i<= 5; i++ ) {
         cout << " d[" << i << "]:\t"<< d[i][c]<< endl;
       }
       cout << " ----------------------------\n"<<endl;
       
     }  // if(dbgCells)
   }  // dbgCells loop
 #endif
} 


/*__________________________________________________________________
 Function~ computeDi--
 Purpose~  compute Di's at the boundary cells using upwind first-order 
           differenceing scheme
____________________________________________________________________*/
void computeDi(StaticArray<CCVariable<Vector> >& d,
               constCCVariable<double>& rho,              
               const CCVariable<double>& press,                   
               constCCVariable<Vector>& vel,                  
               constCCVariable<double>& speedSound,                    
               const Patch* patch,
               SimulationStateP& sharedState,
               const Lodi_user_inputs* user_inputs)                              
{
  cout_doing << "LODI computeLODIFirstOrder "<< endl;
  Vector dx = patch->dCell();
  vector<IntVector> R_Offset(6);
  R_Offset[Patch::xminus] = IntVector(1,0,0);  // right cell offset
  R_Offset[Patch::xplus]  = IntVector(0,0,0);
  R_Offset[Patch::yminus] = IntVector(0,1,0);
  R_Offset[Patch::yplus]  = IntVector(0,0,0);
  R_Offset[Patch::zminus] = IntVector(0,0,1);
  R_Offset[Patch::zplus]  = IntVector(0,0,0);

  vector<IntVector> L_Offset(6);
  L_Offset[Patch::xminus] = IntVector(0, 0, 0);   // left cell offset
  L_Offset[Patch::xplus]  = IntVector(-1,0, 0);
  L_Offset[Patch::yminus] = IntVector(0, 0, 0);
  L_Offset[Patch::yplus]  = IntVector(0,-1, 0);
  L_Offset[Patch::zminus] = IntVector(0, 0, 0);
  L_Offset[Patch::zplus]  = IntVector(0, 0, -1);
  
  // Characteristic Length of the overall domain
  Vector domainLength;
  const Level* level = patch->getLevel();
  GridP grid = level->getGrid();
  grid->getLength(domainLength, "minusExtraCells");

  for (int i = 1; i<= 5; i++ ) {           // don't initialize inside main loop
    d[i].initialize(Vector(0.0,0.0,0.0));  // you'll overwrite previously compute di
  }
  
  //__________________________________
  // Iterate over the faces encompassing the domain
  vector<Patch::FaceType>::const_iterator iter;
  
  for (iter  = patch->getBoundaryFaces()->begin(); 
       iter != patch->getBoundaryFaces()->end(); ++iter){
    Patch::FaceType face = *iter;
 
    cout_dbg << " computing DI on face " << face 
             << " patch " << patch->getID()<<endl;
    //_____________________________________
    // S I D E S
    IntVector axes = patch->faceAxes(face);
    int P_dir = axes[0]; // find the principal dir
    double delta = dx[P_dir];
    
/*`==========TESTING==========*/
    // find max Mach Number
    double maxMach = 0.0;
    for(CellIterator iter=patch->getFaceCellIterator(face, "plusEdgeCells"); 
        !iter.done();iter++) {
      IntVector c = *iter;
      maxMach = Max(maxMach, fabs(vel[c][P_dir]/speedSound[c]));
    } 
/*===========TESTING==========`*/

    for(CellIterator iter=patch->getFaceCellIterator(face, "plusEdgeCells"); 
        !iter.done();iter++) {
      IntVector c = *iter;
      IntVector r = c + R_Offset[face];
      IntVector l = c + L_Offset[face];

      double drho_dx = (rho[r]   - rho[l])/delta; 
      double dp_dx   = (press[r] - press[l])/delta;
      Vector dVel_dx = (vel[r]   - vel[l])/delta;

      Di(d, P_dir, c, face, domainLength, user_inputs, maxMach, press[c],
         speedSound[c], rho[c], vel[c][P_dir], drho_dx, dp_dx, dVel_dx); 

    }  // faceCelliterator 
  }  // loop over faces
}// end of function

/*__________________________________________________________________
 Function~ computeNu--                    L   O   D   I
 Purpose~  compute dissipation coefficients 
__________________________________________________________________*/ 
void computeNu(CCVariable<Vector>& nu,
               const CCVariable<double>& p, 
               const Patch* patch,
               SimulationStateP& sharedState)
{
  cout_doing << "LODI computeNu "<< endl;
  double d_SMALL_NUM = 1.0e-100;
    
  // Iterate over the faces encompassing the domain
  vector<Patch::FaceType>::const_iterator iter;
  
  for (iter  = patch->getBoundaryFaces()->begin(); 
       iter != patch->getBoundaryFaces()->end(); ++iter){
    Patch::FaceType face = *iter;
    
    if (is_LODI_face(patch, face, sharedState) ) {
      cout_dbg << " computing Nu on face " << face 
               << " patch " << patch->getID()<<endl;   
              
      vector<int> otherDir(2);
      IntVector axes = patch->faceAxes(face);
      int P_dir   = axes[0]; // principal direction
      otherDir[0] = axes[1]; // other vector directions
      otherDir[1] = axes[2];  
      
      //__________________________________
      //  At patch boundaries you need to extend
      // the computational footprint by one cell in ghostCells
      CellIterator hiLo = patch->getFaceCellIterator(face, "minusEdgeCells");
      CellIterator iterPlusGhost = patch->addGhostCell_Iter(hiLo,1);
                        
      for(CellIterator iter=iterPlusGhost; !iter.done();iter++) {
        IntVector c = *iter;

        for ( int i = 0; i < 2 ; i++ ) {  // set both orthogonal components
          int dir = otherDir[i];
          IntVector r = c;
          IntVector l = c;
          r[dir] += 1;  // tweak the r and l cell indices
          l[dir] -= 1; 
                        // 2nd order cell centered difference 
          nu[c][dir] = fabs(p[r] - 2.0 * p[c] + p[l])/
                        (fabs(p[r] - p[c]) + fabs(p[c] - p[l])  + d_SMALL_NUM);
        }
      }
      //__________________________________
      //    E D G E S  -- on boundaryFaces only
      // use cell centered AND one sided differencing
      vector<Patch::FaceType> b_faces;
      getBoundaryEdges(patch,face,b_faces);

      vector<Patch::FaceType>::const_iterator iter;  
      for(iter = b_faces.begin(); iter != b_faces.end(); ++ iter ) {
        Patch::FaceType face0 = *iter;
        //__________________________________
        //  Find the Vector components Edir1 and Edir2
        //  for this particular edge
        IntVector faceDir = patch->faceDirection(face0);
        IntVector axes = patch->faceAxes(face0);
        int Edir1 = axes[0];
        int Edir2 = otherDirection(P_dir, Edir1);
        
        //-----------  THIS IS GROSS-------
        // Find an edge iterator that 
        // a) doesn't hit the corner cells and
        // b) extends one cell into the next patch over
        CellIterator edgeIter =  
                patch->getEdgeCellIterator(face, face0, "minusCornerCells");

        IntVector patchNeighborLow  = patch->neighborsLow();
        IntVector patchNeighborHigh = patch->neighborsHigh();
        
        IntVector lo = edgeIter.begin();
        IntVector hi = edgeIter.end();
        lo[Edir2] -= abs(1 - patchNeighborLow[Edir2]);  // increase footprint
        hi[Edir2] += abs(1 - patchNeighborHigh[Edir2]); // by 1
        CellIterator iterLimits(lo,hi);
        //__________________________________
        
        for(CellIterator iter = iterLimits;!iter.done();iter++){ 

          IntVector c = *iter;
          IntVector r  = c;
          IntVector rr = c;
          r[Edir1]  -= faceDir[Edir1];      // tweak the r and l cell indices
          rr[Edir1] -= 2 * faceDir[Edir1];  // One sided differencing
          nu[c][Edir1] = fabs(p[c] - 2.0 * p[r] + p[rr])/
                        (fabs(p[c] - p[r]) + fabs(p[r] - p[rr])  + d_SMALL_NUM);

          IntVector r2 = c;
          IntVector l2 = c;
          r2[Edir2] += 1;  // tweak the r and l cell indices
          l2[Edir2] -= 1;  // cell centered differencing
          nu[c][Edir2] = fabs(p[r2] - 2.0 * p[c] + p[l2])/
                        (fabs(p[r2] - p[c]) + fabs(p[c] - p[l2])  + d_SMALL_NUM);
        }
      }
      //________________________________________________________
      // C O R N E R S   
  /*`==========TESTING==========*/
  // Need a clever way to figure out the r and rr indicies
  //  for the two different directions

      const vector<IntVector> corner = patch->getCornerCells(face);
      vector<IntVector>::const_iterator itr;
      for(itr = corner.begin(); itr != corner.end(); ++ itr ) {
        IntVector c = *itr;
        nu[c] = Vector(0,0,0);
        //cout << " I'm working on cell " << c << endl;
      } 
  /*==========TESTING==========`*/
    }  // on the LODI bc face
  }
}
/* --------------------------------------------------------------------- 
 Function~  lodi_bc_preprocess-- 
 Purpose~   Take care: getting data from dw, allocate temporary vars,
            initialize vars, compute some vars, compute di and nu for LODI bcs
 ---------------------------------------------------------------------  */
void  lodi_bc_preprocess( const Patch* patch,
                          Lodi_vars* lv,
                          ICELabel* lb,
                          const int indx,
                          DataWarehouse* old_dw,
                          DataWarehouse* new_dw,
                          SimulationStateP& sharedState)
{
  cout_doing << "lodi_bc_preprocess on patch "<<patch->getID()<< endl;
  
  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gn  = Ghost::None;
  constCCVariable<double> press_old;
  constCCVariable<double> vol_frac_old;
  
  CCVariable<double>& E   = lv->E;  // shortcuts to Lodi_vars struct
  CCVariable<Vector>& vel_CC = lv->vel_CC;
  CCVariable<double>& rho_CC = lv->rho_CC;
  CCVariable<double>& press_tmp = lv->press_tmp; 
  CCVariable<Vector>& nu = lv->nu;
  constCCVariable<double>& cv         = lv->cv;
  constCCVariable<double>& gamma      = lv->gamma;
  constCCVariable<double>& rho_old    = lv->rho_old;
  constCCVariable<double>& temp_old   = lv->temp_old;
  constCCVariable<double>& speedSound = lv->speedSound;
  constCCVariable<Vector>& vel_old    = lv->vel_old;
  StaticArray<CCVariable<Vector> >& di = lv->di;
  
  //__________________________________
  //   get the data LODI needs from old dw
  new_dw->get(gamma,        lb->gammaLabel,       indx,patch,gn ,0);
  old_dw->get(temp_old,     lb->temp_CCLabel,     indx,patch,gac,1);
  old_dw->get(rho_old,      lb->rho_CCLabel,      indx,patch,gac,1);
  old_dw->get(vel_old,      lb->vel_CCLabel,      indx,patch,gac,1);
  old_dw->get(press_old,    lb->press_CCLabel,    0,   patch,gac,2);
  old_dw->get(vol_frac_old, lb->vol_frac_CCLabel, indx,patch,gac,2);

  new_dw->allocateTemporary(press_tmp, patch, gac, 2);
  new_dw->allocateTemporary(nu,        patch, gac, 1);
  new_dw->allocateTemporary(E,         patch, gac, 1);
  new_dw->allocateTemporary(rho_CC,    patch, gac, 1);  
  new_dw->allocateTemporary(vel_CC,    patch, gac, 1);

  for (int i = 0; i <= 5; i++){ 
    new_dw->allocateTemporary(di[i], patch, gac, 1);
  }                             
   
  //__________________________________
  // only work on those faces that have lodi bcs
  // and are on the edge of the computational domain
  vector<Patch::FaceType>::const_iterator iter;
  for (iter  = patch->getBoundaryFaces()->begin(); 
       iter != patch->getBoundaryFaces()->end(); ++iter){
    Patch::FaceType face = *iter;
    
    if (is_LODI_face(patch,face, sharedState) ) {
      //__________________________________
      // Create an iterator that iterates over the face
      // + 2 cells inward.  We don't need to hit every
      // cell on the patch.  At patch boundaries you need to extend
      // the footprint by one/two cells into the next patch
      CellIterator iter=patch->getFaceCellIterator(face, "plusEdgeCells");
      IntVector lo = iter.begin();
      IntVector hi = iter.end();
    
      int P_dir = patch->faceAxes(face)[0];  //principal dir.
      if(face==Patch::xminus || face==Patch::yminus || face==Patch::zminus){
        hi[P_dir] += 2;
      }
      if(face==Patch::xplus || face==Patch::yplus || face==Patch::zplus){
        lo[P_dir] -= 2;
      }
      CellIterator iterLimits(lo,hi);
      CellIterator iterPlusGhost1 = patch->addGhostCell_Iter(iterLimits,1);
      CellIterator iterPlusGhost2 = patch->addGhostCell_Iter(iterLimits,2);

      //  plus one layer of ghostcells
      Vector nanV(-9e1000, -9e1000, -9e1000);
      for(CellIterator iter = iterPlusGhost1; !iter.done(); iter++) {  
        IntVector c = *iter;
        nu[c] = nanV;
        E[c] = rho_old[c] * (cv[c] * temp_old[c]  + 0.5 * vel_old[c].length2() ); 
      }
      //  plut two layers of ghostcells
      for(CellIterator iter = iterPlusGhost2; !iter.done(); iter++) {  
        IntVector c = *iter;
        press_tmp[c] = vol_frac_old[c] * press_old[c];
      }
    }  //lodi face
  }  // boundary face

  //compute dissipation coefficients
  computeNu(nu, press_tmp, patch, sharedState);

  //compute Di at boundary cells
  computeDi(di, rho_old,  press_tmp, vel_old, 
            speedSound, patch, sharedState, lv->user_inputs);
}  
 
/*________________________________________________________
 Function~ computeConvection--
 Purpose~  Compute the convection term in conservation law
_________________________________________________________*/
inline double computeConvection(
              const Vector& nuFrt, const Vector& nuMid, const Vector& nuLast,    
              const double& qFrt,  const double& qMid,  const double& qLast,
              const double& q_R,   const double& q_C,   const double& q_L,
              const double& deltaT,    const Vector& dx,
              const int& dir) 
{
   //__________________________________
   // Artifical dissipation term
/*`==========TESTING==========*/
#if 0
    double k_const = 0.4;
    doubl eeplus  = 0.5 * k_const * dx[dir] * (nuFrt[dir]   + nuMid[dir])/deltaT;
    double eminus = 0.5 * k_const * dx[dir] * (nuLast[dir]  + nuMid[dir])/deltaT;
    double dissipation = (eplus * qFrt - (eplus + eminus) * qMid 
              +  eminus * qLast)/dx[dir]; 
#endif
    double dissipation  = 0;
/*==========TESTING==========`*/
 double conv;
 
  // central differenceing
  conv = 0.5 * (q_R - q_L)/dx[dir] - dissipation;
   
#if 0      
   // upwind differenceing  (we'll make this fast after we know that it works)
   if (q_C > 0) {
    conv = (q_R - q_C)/dx[dir] - dissipation;
   }else{
    conv = (q_L - q_C)/dx[dir] - dissipation;
   }         
#endif
   return conv; 
} 
/*_________________________________________________________________
 Function~ getBoundaryEdges--
 Purpose~  returns a list of edges where boundary conditions
           should be applied.
___________________________________________________________________*/  
void getBoundaryEdges(const Patch* patch,
                      const Patch::FaceType face,
                      vector<Patch::FaceType>& face0)
{
  IntVector patchNeighborLow  = patch->neighborsLow();
  IntVector patchNeighborHigh = patch->neighborsHigh();
  
  //__________________________________
  // Looking down on the face, examine 
  // each edge (clockwise).  If there
  // are no neighboring patches then it's a valid edge
  IntVector axes = patch->faceAxes(face);
  int dir1  = axes[1];  // other vector directions
  int dir2  = axes[2];   
  IntVector minus(Patch::xminus, Patch::yminus, Patch::zminus);
  IntVector plus(Patch::xplus,   Patch::yplus,  Patch::zplus);
  
  if ( patchNeighborLow[dir1] == 1 ){
   face0.push_back(Patch::FaceType(minus[dir1]));
  }  
  if ( patchNeighborHigh[dir1] == 1 ) {
   face0.push_back(Patch::FaceType(plus[dir1]));
  }
  if ( patchNeighborLow[dir2] == 1 ){
   face0.push_back(Patch::FaceType(minus[dir2]));
  }  
  if ( patchNeighborHigh[dir2] == 1 ) {
   face0.push_back(Patch::FaceType(plus[dir2]));
  }
}

/*_________________________________________________________________
 Function~ otherDirection--
 Purpose~  returns the remaining vector component.
___________________________________________________________________*/  
int otherDirection(int dir1, int dir2)
{ 
  int dir3 = -9;
  if ((dir1 == 0 && dir2 == 1) || (dir1 == 1 && dir2 == 0) ){  // x, y
    dir3 = 2; // z
  }
  if ((dir1 == 0 && dir2 == 2) || (dir1 == 2 && dir2 == 0) ){  // x, z
    dir3 = 1; // y
  }
  if ((dir1 == 1 && dir2 == 2) || (dir1 == 2 && dir2 == 1) ){   // y, z
    dir3 = 0; //x
  }
  return dir3;
}
/*_________________________________________________________________
 Function~ FaceDensity_LODI--
 Purpose~  Compute density in boundary cells on any face
___________________________________________________________________*/
void FaceDensity_LODI(const Patch* patch,
                const Patch::FaceType face,
                CCVariable<double>& rho_CC,
                Lodi_vars* lv,
                const Vector& dx)
{
  cout_doing << "Setting FaceDensity_LODI on face " << face<<endl;
  // bulletproofing
  if (!lv){
    throw InternalError("FaceDensityLODI: Lodi_vars = null");
  }  
  
  // shortcuts
  StaticArray<CCVariable<Vector> >& d = lv->di;
  const CCVariable<Vector>& nu      = lv->nu;
  const CCVariable<double>& rho_old = lv->rho_old;
  const CCVariable<Vector>& vel_old = lv->vel_old;
  const double delT = lv->delT;
  
  IntVector axes = patch->faceAxes(face);
  int P_dir = axes[0];  // principal direction
  int dir1  = axes[1];  // other vector directions
  int dir2  = axes[2];
  
  IntVector offset = IntVector(1,1,1) - Abs(patch->faceDirection(face));
  //__________________________________
  //    S I D E   M I N U S   E D G E S
  for(CellIterator iter=patch->getFaceCellIterator(face, "minusEdgeCells"); 
                                                      !iter.done();iter++) {
    IntVector c = *iter;
    
    IntVector r1 = c;
    IntVector l1 = c;
    r1[dir1] += offset[dir1];  // tweak the r and l cell indices
    l1[dir1] -= offset[dir1];
    double q_R = rho_old[r1] * vel_old[r1][dir1];
    double q_C = rho_old[c]  * vel_old[c][dir1];
    double q_L = rho_old[l1] * vel_old[l1][dir1];
    
    double conv_dir1 = computeConvection(nu[r1], nu[c], nu[l1], 
                                         rho_old[r1], rho_old[c], rho_old[l1], 
                                         q_R, q_C, q_L, 
                                         delT, dx, dir1);
    IntVector r2 = c;
    IntVector l2 = c;
    r2[dir2] += offset[dir2];  // tweak the r and l cell indices
    l2[dir2] -= offset[dir2];
    
    q_R = rho_old[r2] * vel_old[r2][dir2];
    q_C = rho_old[c]  * vel_old[c][dir2];
    q_L = rho_old[l2] * vel_old[l2][dir2];
    double conv_dir2 = computeConvection(nu[r2], nu[c], nu[l2],
                                         rho_old[r2], rho_old[c], rho_old[l2], 
                                         q_R, q_C, q_L, 
                                         delT, dx, dir2);

    rho_CC[c] = rho_old[c] - delT * (d[1][c][P_dir] + conv_dir1 + conv_dir2);

  }
 
  //__________________________________
  //    E D G E S  -- on boundaryFaces only
  vector<Patch::FaceType> b_faces;
  getBoundaryEdges(patch,face,b_faces);
  
  vector<Patch::FaceType>::const_iterator iter;  
  for(iter = b_faces.begin(); iter != b_faces.end(); ++ iter ) {
    Patch::FaceType face0 = *iter;
    //__________________________________
    //  Find the offset for the r and l cells
    //  and the Vector components Edir1 and Edir2
    //  for this particular edge
    IntVector offset = IntVector(1,1,1)  - Abs(patch->faceDirection(face)) 
                                         - Abs(patch->faceDirection(face0));
           
    IntVector axes = patch->faceAxes(face0);
    int Edir1 = axes[0];
    int Edir2 = otherDirection(P_dir, Edir1);
  
    CellIterator iterLimits =  
                patch->getEdgeCellIterator(face, face0, "minusCornerCells");
                
    for(CellIterator iter = iterLimits;!iter.done();iter++){ 
      IntVector c = *iter;  
      IntVector r = c + offset;  
      IntVector l = c - offset;
      double q_R = rho_old[r] * vel_old[r][Edir2];
      double q_C = rho_old[c] * vel_old[c][Edir2];
      double q_L = rho_old[l] * vel_old[l][Edir2];
      double conv = computeConvection(nu[r], nu[c], nu[l],
                                      rho_old[r], rho_old[c], rho_old[l], 
                                      q_R, q_C, q_L, 
                                      delT, dx, Edir2); 
                                                                
      rho_CC[c] = rho_old[c] - delT * (d[1][c][P_dir] + d[1][c][Edir1] + conv);
    }
  }

  //__________________________________
  // C O R N E R S    
  const vector<IntVector> corner = patch->getCornerCells(face);
  vector<IntVector>::const_iterator itr;
  
  for(itr = corner.begin(); itr != corner.end(); ++ itr ) {
    IntVector c = *itr;
    rho_CC[c] = rho_old[c] 
              - delT * (d[1][c].x() + d[1][c].y() + d[1][c].z());
  }           
}

/*_________________________________________________________________
 Function~ FaceVel_LODI--
 Purpose~  Compute velocity in boundary cells on face
___________________________________________________________________*/
void FaceVel_LODI(const Patch* patch,
                 Patch::FaceType face,
                 CCVariable<Vector>& vel_CC,                 
                 Lodi_vars* lv,
                 const Vector& dx,
                 SimulationStateP& sharedState)                     

{
  cout_doing << "Setting FaceVel_LODI on face " << face << endl;
  // bulletproofing
  if (!lv){
    throw InternalError("FaceVelLODI: Lodi_vars = null");
  }
 
/*`==========TESTING==========*/
// hardCode the cells you want to intergate;
vector<IntVector> dbgCells;
#if 0
  dbgCells.push_back(IntVector(-1,-1,2));
  dbgCells.push_back(IntVector(0,-1,2));      
  dbgCells.push_back(IntVector(1,-1,2));        
  dbgCells.push_back(IntVector(2,-1,2));
  dbgCells.push_back(IntVector(3,-1,2));
#endif
/*===========TESTING==========`*/
     
  // shortcuts       
  StaticArray<CCVariable<Vector> >& d = lv->di;      
  constCCVariable<double>& rho_old = lv->rho_old;
  constCCVariable<Vector>& vel_old = lv->vel_old; 
  CCVariable<double>& rho_new = lv->rho_CC;
  
  CCVariable<double>& p  = lv->press_tmp;          
  double delT = lv->delT;
                 
  IntVector axes = patch->faceAxes(face);
  int P_dir = axes[0];  // principal direction
  int dir1  = axes[1];  // other vector directions
  int dir2  = axes[2];
  
  Vector gravity = sharedState->getGravity();
  IntVector offset = IntVector(1,1,1) - Abs(patch->faceDirection(face));
  //__________________________________
  //    S I D E   M I N U S   E D G E S 
  for(CellIterator iter=patch->getFaceCellIterator(face, "minusEdgeCells"); 
                                                      !iter.done();iter++) {
    IntVector c = *iter;
    //__________________________________
    // Transverse convection terms
    IntVector r1 = c;
    IntVector l1 = c;
    r1[dir1] += offset[dir1];  // tweak the r and l cell indices
    l1[dir1] -= offset[dir1]; 
       
    Vector convect1(0,0,0);
    for(int dir = 0; dir <3; dir ++ ) {              
      double q_R = rho_old[r1] * vel_old[r1][dir] * vel_old[r1][dir1];
      double q_C = rho_old[c]  * vel_old[c][dir]  * vel_old[c][dir1];
      double q_L = rho_old[l1] * vel_old[l1][dir] * vel_old[l1][dir1];
      Vector placeHolder;

      convect1[dir] = computeConvection(placeHolder, placeHolder, placeHolder, 
                                        rho_old[r1], rho_old[c], rho_old[l1], 
                                        q_R, q_C, q_L, 
                                        delT, dx, dir1);
    }
    
    IntVector r2 = c;
    IntVector l2 = c;
    r2[dir2] += offset[dir2];  // tweak the r and l cell indices
    l2[dir2] -= offset[dir2];  
    
    Vector convect2(0,0,0);
    for(int dir = 0; dir <3; dir ++ ) {            
      double q_R = rho_old[r2] * vel_old[r2][dir] * vel_old[r2][dir2];
      double q_C = rho_old[c]  * vel_old[c][dir]  * vel_old[c][dir2];
      double q_L = rho_old[l2] * vel_old[l2][dir] * vel_old[l2][dir2];
      Vector placeHolder;
      
      convect2[dir] = computeConvection(placeHolder, placeHolder, placeHolder, 
                                        rho_old[r2], rho_old[c], rho_old[l2], 
                                        q_R, q_C, q_L, 
                                        delT, dx, dir2); 
     }       
    //__________________________________
    // Pressure gradient terms
    Vector pressGradient;  
    pressGradient[P_dir] = 0.0;   // This is accounted for in Li terms
    pressGradient[dir1]  = 0.5 * (p[r1] - p[l1])/dx[dir1];  // centered difference  
    pressGradient[dir2]  = 0.5 * (p[r2] - p[l2])/dx[dir2];  //  ---//---- 
    
    //__________________________________
    // Boundary normal convection Terms
    // See Sutherland Table 8
    Vector BN_convect;
    BN_convect.x( vel_old[c].x() * d[1][c][P_dir] + rho_old[c] *d[3][c][P_dir] );
    BN_convect.y( vel_old[c].y() * d[1][c][P_dir] + rho_old[c] *d[4][c][P_dir] );
    BN_convect.z( vel_old[c].z() * d[1][c][P_dir] + rho_old[c] *d[5][c][P_dir] );
     
    //__________________________________
    //  Equations 9.9-9.10
    Vector momOld, momChange;
    momOld    = rho_old[c] * vel_old[c];
                         
    momChange =  - delT * ( BN_convect + convect1 + convect2 + pressGradient )  
                 + delT *rho_old[c] * gravity;
         
    vel_CC[c] = (momOld + momChange)/rho_new[c];

#if 0
    //__________________________________
    //  debugging
    for (int i = 0; i<(int) dbgCells.size(); i++) {
      if (c == dbgCells[i]) {
        cout.setf(ios::scientific,ios::floatfield);
        cout.precision(10);
        cout << " \n c " << c << "--------------------------  F A C E " << face << " P_dir " << P_dir << endl;
        cout << c <<" P_dir " << P_dir << " dir1 " << dir1 << "dir2 " << dir2 << endl;
        cout << " rho_old[c] * vel_old[c] " << momOld << endl;
        cout << " convect1                " << convect1 << endl;
        cout << " convect2                " << convect2 << endl;
        cout << " BN_convect              " << BN_convect<< " term1 " << term1 << " term2 " << term2 <<endl;
        cout << " pressGradient           " << pressGradient << endl;
        cout << " rho_old * gravity       " << rho_old[c] * gravity << endl;
        cout << " vel                     " << vel_CC[c] << "\n"<<endl;
        cout << " convect1: rho_old, vel_old["<<dir1<<"], vel_old["<<dir1<<"], dx["<<dir1<<"]" << endl;
        cout << " convect2: rho_old, vel_old["<<dir1<<"], vel_old["<<dir2<<"], dx["<<dir2<<"]" << endl;
      }
    }  //  dbgCells loop
#endif
  }

  //__________________________________
  //    E D G E S  -- on boundaryFaces only
  vector<Patch::FaceType> b_faces;
  getBoundaryEdges(patch,face,b_faces);
  
  vector<Patch::FaceType>::const_iterator iter;  
  for(iter = b_faces.begin(); iter != b_faces.end(); ++ iter ) {
    Patch::FaceType face0 = *iter;
    
    //__________________________________
    //  Find the offset for the r and l cells
    //  and the Vector components Edir1 and Edir2
    //  for this particular edge
    IntVector offset = IntVector(1,1,1)  - Abs(patch->faceDirection(face)) 
                                         - Abs(patch->faceDirection(face0));
           
    IntVector axes = patch->faceAxes(face0);
    int Edir1 = axes[0];
    int Edir2 = otherDirection(P_dir, Edir1);
     
    CellIterator iterLimits =  
                patch->getEdgeCellIterator(face, face0, "minusCornerCells");
                      
    for(CellIterator iter = iterLimits;!iter.done();iter++){ 
      IntVector c = *iter;

      //__________________________________
      // convective terms
      IntVector r2 = c;
      IntVector l2 = c;
      r2[Edir2] += offset[Edir2];  // tweak the r and l cell indices
      l2[Edir2] -= offset[Edir2]; 

      Vector convect1(0,0,0);
      for(int dir = 0; dir <3; dir ++ ) {
        double q_R = rho_old[r2] * vel_old[r2][dir] * vel_old[r2][Edir2];
        double q_C = rho_old[c]  * vel_old[c][dir]  * vel_old[c][Edir2];
        double q_L = rho_old[l2] * vel_old[l2][dir] * vel_old[l2][Edir2];
        Vector placeHolder;
        
        convect1[dir] = computeConvection(placeHolder, placeHolder, placeHolder, 
                                          rho_old[r2], rho_old[c], rho_old[l2], 
                                          q_R, q_C, q_L, 
                                          delT, dx, Edir2);
      }
      //__________________________________
      // Pressure gradient terms
      Vector pressGradient; 
      pressGradient[P_dir] = 0.0;   // accounted for in Li terms
      pressGradient[Edir1] = 0.0;
      pressGradient[Edir2] = 0.5 * (p[r2] - p[l2])/dx[Edir2];

      //__________________________________
      // Equation 9.9 - 9.10
      Vector mom; // momentum
      Vector term1, term2;
      Vector weight(1,1,1);
      weight[Edir1] = 0.0;  // so you don't double count di values
      
      term1.x(vel_old[c].x() * (d[1][c][P_dir] + weight.x() * d[1][c][Edir1]) );
      term1.y(vel_old[c].y() * (d[1][c][P_dir] + weight.y() * d[1][c][Edir1]) );
      term1.z(vel_old[c].z() * (d[1][c][P_dir] + weight.z() * d[1][c][Edir1]) );   
      
      term2.x( (d[3][c][P_dir] + weight.x() * d[3][c][Edir1]) );        
      term2.y( (d[4][c][P_dir] + weight.y() * d[4][c][Edir1]) );       
      term2.z( (d[5][c][P_dir] + weight.z() * d[5][c][Edir1]) );
      term2 *= rho_old[c];
      
      mom  = rho_old[c] * vel_old[c]                                
           - delT * ( term1 +  term2 + convect1 + pressGradient )  
           + delT *rho_old[c] * gravity;                     
              
      vel_CC[c] = mom/rho_new[c];

#if 0      
      //__________________________________
      //  debugging
      for (int i = 0; i<(int) dbgCells.size(); i++) {
        if (c == dbgCells[i]) {
          cout.setf(ios::scientific,ios::floatfield);
          cout.precision(10);
          int Pd = P_dir;
          int E1 = Edir1;
          int E2 = Edir2;

          cout << " -------------------------- E D G E " << endl;
          cout << c <<" P_dir " << Pd << " Edir1 " << Edir1 << " Edir2 " << Edir2 << endl;
          cout << " rho_old[c] * vel_old[c] " << rho_old[c] * vel_old[c] << endl;
          cout << " term1                   " << term1 << endl;
          cout << " term2                   " << term2 << endl;
          cout << " convect1                " << convect1 << endl;
          cout << " pressGradient           " << pressGradient << endl;
          cout << " rho_old gravity         " << rho_old[c] * gravity << endl;
          cout << " mom                     " << mom << endl;
          cout << " vel                     " << vel_CC[c] << "\n"<<endl;

          for (int i = 1; i<= 5; i++ ) {
            cout << " d[" << i << "]:\t"<< d[i][c]<< endl;
          }
          cout << " -------------------------- " << endl;
        }
      }
#endif
    }
  }  
  //________________________________________________________
  // C O R N E R S
  const vector<IntVector> corner = patch->getCornerCells(face);
  vector<IntVector>::const_iterator itr;
  
  for(itr = corner.begin(); itr != corner.end(); ++ itr ) {
    IntVector c = *itr;
    Vector term1, term2;
   
    term1.x((d[1][c].x() + d[1][c].y() + d[1][c].z()) * vel_old[c].x());
    term1.y((d[1][c].x() + d[1][c].y() + d[1][c].z()) * vel_old[c].y());
    term1.z((d[1][c].x() + d[1][c].y() + d[1][c].z()) * vel_old[c].z());
    
    term2.x((d[3][c].x() + d[3][c].y() + d[3][c].z()) * rho_old[c]);
    term2.y((d[4][c].x() + d[4][c].y() + d[4][c].z()) * rho_old[c]);
    term2.z((d[5][c].x() + d[5][c].y() + d[5][c].z()) * rho_old[c]);
                                        
    Vector mom = rho_old[c] * vel_old[c] - delT * (term1 + term2)
                                         + delT * rho_old[c] * gravity;

    vel_CC[c] = mom/ rho_new[c];

      //__________________________________
      //  debugging
#if 0
      for (int i = 0; i<(int) dbgCells.size(); i++) {
        if (c == dbgCells[i]) {
         cout << "-------------------------- C O R N E R " << c << endl;
         cout << " rho_old[c] * vel_old[c] " << rho_old[c] * vel_old[c] << << endl;
         cout << " term1 = " << term1 << endl;
         cout << " term2 = " << term2 << endl;
         cout << " d1 " << d[1][c] << endl; 
         cout << " d3 " << d[3][c] << endl; 
         cout << " d4 " << d[4][c] << endl;
         cout << " d5 " << d[5][c] << endl; 
         cout << " mom/delT " <<  mom/delT<< endl;
         cout << " vel_CC[c] " << vel_CC[c] <<" \n\n"<< endl;
        }
      }
#endif 
    
  }
} //end of the function FaceVelLODI() 

/*_________________________________________________________________
 Function~ FaceTemp_LODI--
 Purpose~  Compute temperature in boundary cells on faces
           Solves equation 9.8 of Reference.
___________________________________________________________________*/
void FaceTemp_LODI(const Patch* patch,
             const Patch::FaceType face,
             CCVariable<double>& temp_CC,
             Lodi_vars* lv, 
             const Vector& dx,
             SimulationStateP& sharedState)
{
  cout_doing << "Setting FaceTemp_LODI on face " <<face<< endl; 
  
  // bulletproofing
  if (!lv){
    throw InternalError("FaceTempLODI: Lodi_vars = null");
  } 
  // shortcuts  
  StaticArray<CCVariable<Vector> >& d = lv->di;
  const CCVariable<double>& E         = lv->E;
  const CCVariable<double>& cv        = lv->cv;
  const CCVariable<double>& gamma     = lv->gamma;
  const CCVariable<double>& rho_new   = lv->rho_CC;
  const CCVariable<double>& rho_old   = lv->rho_old;
  const CCVariable<double>& press_tmp = lv->press_tmp;
  const CCVariable<Vector>& vel_old   = lv->vel_old;
  const CCVariable<Vector>& vel_new   = lv->vel_CC;
  const CCVariable<Vector>& nu  = lv->nu;
  const double delT  = lv->delT;
              
  IntVector axes = patch->faceAxes(face);
  int P_dir = axes[0];  // principal direction
  int dir1  = axes[1];  // other vector directions
  int dir2  = axes[2];
  
  Vector gravity = sharedState->getGravity();
  IntVector offset = IntVector(1,1,1) - Abs(patch->faceDirection(face));
  //__________________________________
  //    S I D E   M I N U S   E D G E S  
  for(CellIterator iter=patch->getFaceCellIterator(face, "minusEdgeCells"); 
                                                      !iter.done();iter++) {
    IntVector c = *iter;
    
    IntVector r1 = c;
    IntVector l1 = c;
    r1[dir1] += offset[dir1];  // tweak the r and l cell indices
    l1[dir1] -= offset[dir1];
    double q_R = vel_old[r1][dir1] * (E[r1] + press_tmp[r1]);
    double q_C = vel_old[c][dir1]  * (E[c]  + press_tmp[c]);
    double q_L = vel_old[l1][dir1] * (E[l1] + press_tmp[l1]);
    
    double conv_dir1 = computeConvection(nu[r1], nu[c], nu[l1], 
                                          E[r1], E[c], E[l1], 
                                          q_R, q_C, q_L, 
                                          delT, dx, dir1);
    IntVector r2 = c;
    IntVector l2 = c;
    r2[dir2] += offset[dir2];  // tweak the r and l cell indices
    l2[dir2] -= offset[dir2];
    
    q_R = vel_old[r2][dir2] * (E[r2] + press_tmp[r2]);
    q_C = vel_old[c][dir2]  * (E[c]  + press_tmp[c]);
    q_L = vel_old[l2][dir2] * (E[l2] + press_tmp[l2]);
    double conv_dir2 = computeConvection(nu[r2], nu[c], nu[l2],
                                         E[r2], E[c], E[l2], 
                                         q_R, q_C, q_L, 
                                         delT, dx, dir2);

    double vel_old_sqr = vel_old[c].length2();
    double vel_new_sqr = vel_new[c].length2();
    double term1 = 0.5 * d[1][c][P_dir] * vel_old_sqr;
    
    double term2 = d[2][c][P_dir]/(gamma[c] - 1.0);
    double term3 = rho_old[c] * ( vel_old[c].x() * d[3][c][P_dir] + 
                                  vel_old[c].y() * d[4][c][P_dir] +
                                  vel_old[c].z() * d[5][c][P_dir] );
    
    //__________________________________
    //  See Thompson II, pg 451, eq 56
    double gravityTerm = 0.0;
    for(int dir = 0; dir <3; dir ++ ) { 
      gravityTerm += rho_old[c] * vel_old[c][dir] * gravity[dir]; 
    }
                                                                                 
    double E_new = E[c] - delT * (term1 + term2 + term3 + conv_dir1 + conv_dir2
                                  - gravityTerm);               

    temp_CC[c] = E_new/(rho_new[c]*cv[c]) - 0.5 * vel_new_sqr/cv[c];
  }
  
  //__________________________________
  //    E D G E S  -- on boundaryFaces only
  vector<Patch::FaceType> b_faces;
  getBoundaryEdges(patch,face,b_faces);
  
  vector<Patch::FaceType>::const_iterator iter;  
  for(iter = b_faces.begin(); iter != b_faces.end(); ++ iter ) {
    Patch::FaceType face0 = *iter;
    //__________________________________
    //  Find the offset for the r and l cells
    //  and the Vector components Edir1 and Edir2
    //  for this particular edge
    
    IntVector edge = Abs(patch->faceDirection(face)) 
                   + Abs(patch->faceDirection(face0));
    IntVector offset = IntVector(1,1,1) - edge;
           
    IntVector axes = patch->faceAxes(face0);
    int Edir1 = axes[0];
    int Edir2 = otherDirection(P_dir, Edir1);
     
    CellIterator iterLimits =  
                patch->getEdgeCellIterator(face, face0, "minusCornerCells");
                  
    for(CellIterator iter = iterLimits;!iter.done();iter++){ 
      IntVector c = *iter;  
      IntVector r = c + offset;  
      IntVector l = c - offset;

      double q_R = vel_old[r][Edir2] * (E[r] + press_tmp[r]);
      double q_C = vel_old[c][Edir2] * (E[c] + press_tmp[c]);
      double q_L = vel_old[l][Edir2] * (E[l] + press_tmp[l]);
    
      double conv = computeConvection(nu[r], nu[c], nu[l],
                                      E[r], E[c], E[l],               
                                      q_R, q_C, q_L, 
                                      delT, dx, Edir2); 
                                      
      double vel_old_sqr = vel_old[c].length2();

      double term1 = 0.5 * (d[1][c][P_dir] + d[1][c][Edir1]) * vel_old_sqr;

      double term2 = (d[2][c][P_dir] + d[2][c][Edir1])/(gamma[c] - 1.0);
      
      double term3 =
           rho_old[c] * (vel_old[c].x() * (d[3][c][P_dir] + d[3][c][Edir1])  
                      +  vel_old[c].y() * (d[4][c][P_dir] + d[4][c][Edir1])  
                      +  vel_old[c].z() * (d[5][c][P_dir] + d[5][c][Edir1]) );
     
      //__________________________________
      //  See Thompson II, pg 451, eq 56
      double gravityTerm = 0.0;
      for(int dir = 0; dir <3; dir ++ ) { 
        gravityTerm += rho_old[c] * vel_old[c][dir] * gravity[dir]; 
      }
      
      double E_new = E[c] - delT * ( term1 + term2 + term3 + conv - gravityTerm);
      double vel_new_sqr = vel_new[c].length2();

      temp_CC[c] = E_new/(rho_new[c] *cv[c]) - 0.5 * vel_new_sqr/cv[c];

    }
  }  
 
  //________________________________________________________
  // C O R N E R S    
  const vector<IntVector> corner = patch->getCornerCells(face);
  vector<IntVector>::const_iterator itr;
  
  for(itr = corner.begin(); itr != corner.end(); ++ itr ) {
    IntVector c = *itr;
    double vel_old_sqr = vel_old[c].length2();
    double vel_new_sqr = vel_new[c].length2();
    double term1, term2, term3;
    
    term1 = 0.5 * (d[1][c].x() + d[1][c].y() + d[1][c].z()) * vel_old_sqr;
    term2 =       (d[2][c].x() + d[2][c].y() + d[2][c].z())/(gamma[c] - 1.0);
    
    term3 =
        rho_old[c] * vel_old[c].x() * (d[3][c].x() + d[3][c].y() + d[3][c].z())  
      + rho_old[c] * vel_old[c].y() * (d[4][c].x() + d[4][c].y() + d[4][c].z())  
      + rho_old[c] * vel_old[c].z() * (d[5][c].x() + d[5][c].y() + d[5][c].z()); 

    //__________________________________
    //  See Thompson II, pg 451, eq 56
    double gravityTerm = 0.0;
    for(int dir = 0; dir <3; dir ++ ) { 
      gravityTerm += rho_old[c] * vel_old[c][dir] * gravity[dir]; 
    }

    double E_new = E[c] - delT * ( term1 + term2 + term3 - gravityTerm);

    temp_CC[c] = E_new/(rho_new[c] * cv[c]) - 0.5 * vel_new_sqr/cv[c]; 
  }
} //end of function FaceTempLODI()  


/* --------------------------------------------------------------------- 
 Function~  FacePress_LODI--
 Purpose~   Back out the pressure from f_theta and P_EOS
---------------------------------------------------------------------  */
void FacePress_LODI(const Patch* patch,
                    CCVariable<double>& press_CC,
                    StaticArray<CCVariable<double> >& rho_micro,
                    SimulationStateP& sharedState, 
                    Patch::FaceType face,
                    Lodi_vars_pressBC* lv)
{
  cout_doing << " I am in FacePress_LODI on face " <<face<< endl;
  // bulletproofing
  if (!lv){
    throw InternalError("FacePress_LODI: Lodi_vars_pressBC = null");
  }

  int numMatls = sharedState->getNumMatls();
  StaticArray<double> press_eos(numMatls);
  StaticArray<constCCVariable<double> >& gamma   = lv->gamma;
  StaticArray<constCCVariable<double> >& cv      = lv->cv;
  StaticArray<constCCVariable<double> >& Temp_CC = lv->Temp_CC;
  StaticArray<constCCVariable<double> >& f_theta = lv->f_theta;

  //__________________________________  
  double press_ref= sharedState->getRefPress();    
     
  //__________________________________ 
  for(CellIterator iter=patch->getFaceCellIterator(face, "plusEdgeCells"); 
    !iter.done();iter++) {
    IntVector c = *iter;

    press_CC[c] = 0.0;
    for (int m = 0; m < numMatls; m++) {
      Material* matl = sharedState->getMaterial( m );
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
      double tmp;

      if(ice_matl){                // I C E
        ice_matl->getEOS()->computePressEOS(rho_micro[m][c],gamma[m][c],
                                       cv[m][c],Temp_CC[m][c],
                                       press_eos[m],tmp,tmp);        
      } 
#ifndef __APPLE__
      if(mpm_matl){                //  M P M
        mpm_matl->getConstitutiveModel()->
          computePressEOSCM(rho_micro[m][c],press_eos[m], press_ref,
                            tmp, tmp,mpm_matl);
      }
#else
      // This needs to be rethought, due to circular dependencies...
      cerr << "Temporarily commented out by Steve\n";
#endif
      press_CC[c] += f_theta[m][c]*press_eos[m];
//     cout << "press_CC" << c << press_CC[c] << endl;           
    }  // for ALLMatls...
  }
} 
  
/*______________________________________________________________________
          S H E M A T I C   D I A G R A M S

                          Looking down x(plus/minus) faces
                             yplus face  Edir0 = 0 (x)
                                         Edir1 = 1 (y)
                                         Edir2 = 2 (z)
                        _____________________
                        |   |   |   |   |   |
                        | + | + | + | + | + |
                        |---+-----------+---|
                        |   |           |   |
  z(plus/minus)  face   | + |           | + |   z(plus/minus)  face
                        |---+           +---|
                        |   |           |   |    
                        | + |           | + |  
                        |---+           +---|
                        |   |           |   |        
                        | + |           | + |        
                        |---+-----------+---|
                        |   |   |   |   |   |
                        | + | + | + | + | + |     
                        -------------------- 
                              yminus face  Edir0 = 0 (x)
                                           Edir1 = 1 (y)
                                           Edir2 = 2 (z)
______________________________________________________________________*/   
}  // using namespace Uintah
