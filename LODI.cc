#include <Packages/Uintah/CCA/Components/ICE/LODI.h>
#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
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
  Ghost::GhostType  gn = Ghost::None;
  
  for(int m = 0; m < numMatls; m++) {
    Material* matl = sharedState->getMaterial( m );
    int indx = matl->getDWIndex();
    ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);

    if(ice_matl){                // I C E
      old_dw->get(Temp_CC[m],     lb->temp_CCLabel,    indx,patch,gn,0);
      new_dw->get(f_theta_CC[m],  lb->f_theta_CCLabel, indx,patch,gn,0); 
    }
    if(mpm_matl){                // M P M
      new_dw->get(Temp_CC[m],     lb->temp_CCLabel,    indx,patch,gn,0);
      new_dw->get(f_theta_CC[m],  lb->f_theta_CCLabel, indx,patch,gn,0);
    }
    lodi_vars->f_theta[m] = f_theta_CC[m];
    lodi_vars->Temp_CC[m] = Temp_CC[m];
  }
   
}  

/*__________________________________________________________________
 Function~ computeDi--
 Purpose~  compute Di's at the boundary cells using upwind first-order 
           differenceing scheme
____________________________________________________________________*/
void computeDi(StaticArray<CCVariable<Vector> >& d,
               const vector<bool>& d_is_LODI_face,
               constCCVariable<double>& rho,              
               const CCVariable<double>& press,                   
               constCCVariable<Vector>& vel,                  
               constCCVariable<double>& speedSound_,                    
               const Patch* patch)                              
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

  // Iterate over the faces encompassing the domain
  // only set DI on Boundariesfaces that are LODI
  vector<Patch::FaceType>::const_iterator iter;
  
  for (iter  = patch->getBoundaryFaces()->begin(); 
       iter != patch->getBoundaryFaces()->end(); ++iter){
    Patch::FaceType face = *iter;
 
    if (d_is_LODI_face[face] ) {
      cout_dbg << " computing DI on face " << face 
               << " patch " << patch->getID()<<endl;
      //_____________________________________
      //Compute Di at
      IntVector axes = patch->faceAxes(face);
      int dir0 = axes[0]; // find the principal dir and other 2 directions
      int dir1 = axes[1]; 
      int dir2 = axes[2];    

      double delta = dx[dir0];

      IntVector normal = patch->faceDirection(face);
      double norm = (double)normal[dir0];

      for(CellIterator iter=patch->getFaceCellIterator(face, "plusEdgeCells"); 
          !iter.done();iter++) {
        IntVector c = *iter;
        IntVector r = c + R_Offset[face];
        IntVector l = c + L_Offset[face];

        double speedSound = speedSound_[c];
        double speedSoundsqr = speedSound * speedSound;
        double vel_bndry = vel[c][dir0];

        double drho_dx = (rho[r] - rho[l])/delta;
        double dp_dx   = (press[r] - press[l])/delta;
        Vector dVel_dx = (vel[r] - vel[l])/(delta);
        
        //__________________________________
        // L1 Wave Amplitude
        int L1_sign;
        double L1 = 0;
        L1_sign = Sign(norm * (vel_bndry - speedSound));
        if(L1_sign > 0) {       // outgoing waves
          L1 = (vel_bndry - speedSound) 
             * (dp_dx - rho[c] * speedSound * dVel_dx[dir0]);
        } 
        //__________________________________
        // L2, 3, 4 Wave Amplitude
        int L234_sign;
        double L2=0, L3=0, L4=0;
        L234_sign = Sign(norm * vel_bndry);
        if(L234_sign > 0) {     // outgoing waves
          L2 = vel_bndry * (speedSoundsqr * drho_dx - dp_dx);
          L3 = vel_bndry * dVel_dx[dir1];
          L4 = vel_bndry * dVel_dx[dir2];
        } 
        //__________________________________
        // L5 Wave Amplitude
        int L5_sign;
        double L5=0;
        L5_sign =  Sign(norm * (vel_bndry + speedSound));
        if(L5_sign > 0) {      // outgoing wave
          L5 = (vel_bndry + speedSound) 
             * (dp_dx + rho[c] * speedSound * dVel_dx[dir0]);
        } 
        //__________________________________
        // Compute d1-5
        d[1][c][dir0] = (L2 + 0.5 * (L1 + L5))/(speedSoundsqr);
        d[2][c][dir0] = 0.5 * (L5 + L1);
        d[3][c][dir0] = 0.5 * (L5 - L1)/(rho[c] * speedSound);
        d[4][c][dir0] = L3;
        d[5][c][dir0] = L4;
      }
    } // if(onEdgeOfDomain) 
  } //end of for loop over faces
}//end of function

/*__________________________________________________________________
 Function~ computeNu--                    L   O   D   I
 Purpose~  compute dissipation coefficients 
__________________________________________________________________*/ 
void computeNu(CCVariable<Vector>& nu,
               const vector<bool>& is_LODI_face,
               const CCVariable<double>& p, 
               const Patch* patch)
{
  cout_doing << "LODI computeNu "<< endl;
  double d_SMALL_NUM = 1.0e-100;
    
  // Iterate over the faces encompassing the domain
  // only set DI on Boundariesfaces that are LODI
  vector<Patch::FaceType>::const_iterator iter;
  
  for (iter  = patch->getBoundaryFaces()->begin(); 
       iter != patch->getBoundaryFaces()->end(); ++iter){
    Patch::FaceType face = *iter;
    
    if (is_LODI_face[face] ) {
      cout_dbg << " computing Nu on face " << face 
               << " patch " << patch->getID()<<endl;   
              
      vector<int> otherDir(2);
      IntVector axes = patch->faceAxes(face);
      int P_dir   = axes[0]; // principal direction
      otherDir[0] = axes[1]; // other vector directions
      otherDir[1] = axes[2];  

      for(CellIterator iter=patch->getFaceCellIterator(face, "minusEdgeCells"); 
                                                          !iter.done();iter++) {
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
      // use cell centered and one sided differencing
      
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

        CellIterator iterLimits =  
                      patch->getEdgeCellIterator(face,face0,"minusCornerCells");

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
  #if 0 
      vector<IntVector> crn(4);
      computeCornerCellIndices(patch, face, crn);

      for( int corner = 0; corner < 4; corner ++ ) {
        IntVector c = crn[corner];



        IntVector r  = c;
        IntVector rr = c;
        for ( dir.begin();
          r[Edir2]  -= 1;  // tweak the r and l cell indices
          rr[Edir2] -= 2;  // One sided differencing

          IntVector adj = c - offset;
          nu[c][Edir1] = fabs(p[c] - 2.0 * p[r] + p[rr])/
                        (fabs(p[c] - p[r]) + fabs(p[r] - p[rr])  + d_SMALL_NUM);
      } 
  #endif     
  /*==========TESTING==========`*/
    }  // on the right face with LODI BCs
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
                          const vector<bool>& is_LODI_face,
                          DataWarehouse* old_dw,
                          DataWarehouse* new_dw)
{
  cout_doing << "lodi_bc_preprocess on patch "<<patch->getID()<< endl;
  
  Ghost::GhostType  gac = Ghost::AroundCells;
  constCCVariable<double> press_old;
  constCCVariable<double> vol_frac_old;
  
  const double cv = lv->cv;
  CCVariable<double>& e = lv->e;  // shortcuts to Lodi_vars struct
  CCVariable<double>& rho_CC = lv->rho_CC;
  CCVariable<double>& press_tmp = lv->press_tmp; 
  CCVariable<Vector>& nu = lv->nu;  
  constCCVariable<double>& rho_old    = lv->rho_old;
  constCCVariable<double>& temp_old   = lv->temp_old;
  constCCVariable<double>& speedSound = lv->speedSound;
  constCCVariable<Vector>& vel_old    = lv->vel_old;
  StaticArray<CCVariable<Vector> >& di = lv->di;
  
  //__________________________________
  //   get the data LODI needs from old dw
  old_dw->get(temp_old,     lb->temp_CCLabel,     indx,patch,gac,1);
  old_dw->get(rho_old,      lb->rho_CCLabel,      indx,patch,gac,1);
  old_dw->get(vel_old,      lb->vel_CCLabel,      indx,patch,gac,1);
  old_dw->get(press_old,    lb->press_CCLabel,    0,   patch,gac,2);
  old_dw->get(vol_frac_old, lb->vol_frac_CCLabel, indx,patch,gac,1);

  new_dw->allocateTemporary(press_tmp,  patch, gac, 2);
  new_dw->allocateTemporary(nu, patch, gac, 1);
  new_dw->allocateTemporary(e,  patch, gac, 1);
  new_dw->allocateTemporary(rho_CC,  patch, gac, 1);
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
    
    if (is_LODI_face[face] ) {
      //__________________________________
      // Create an iterator that iterates over the face
      // + 2 cells inward.  We don't need to hit every
      // cell on the patch.
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

      //__________________________________
      // Initialize
      for(CellIterator iter = iterLimits; !iter.done(); iter++) {
        IntVector c = *iter;
        nu[c] = Vector(-9,-9,-9);
        press_tmp[c] = -9;
        e[c] = -9;
        
        for (int i = 0; i <= 5; i++){
          di[i][c] =Vector(0.,0.,0.);
        } 
      }
      //__________________________________
      //  At patch boundaries you need to extend
      // the computational footprint by one cell in ghostCells
      CellIterator iterPlusGhost = patch->addGhostCell_Iter(iterLimits,1);

      for(CellIterator iter = iterPlusGhost; !iter.done(); iter++) {  
        IntVector c = *iter;
        e[c] = rho_old[c] * (cv * temp_old[c] +  0.5 * vel_old[c].length2() );                                                 
        press_tmp[c] = vol_frac_old[c] * press_old[c];        
      }        
    }  //lodi face
  }  // boundary face

  //compute dissipation coefficients
  computeNu(nu, is_LODI_face, press_tmp, patch);

  //compute Di at boundary cells
  computeDi(di, is_LODI_face, rho_old,  press_tmp, vel_old, 
            speedSound, patch);
}  
 
/*________________________________________________________
 Function~ computeConvection--
 Purpose~  Compute the convection term in conservation law
_________________________________________________________*/
double computeConvection(const double& nuFrt,     const double& nuMid, 
                         const double& nuLast,    const double& qFrt, 
                         const double& qMid,      const double& qLast,
                         const double& qConFrt,   const double& qConLast,
                         const double& deltaT,    const double& deltaX) 
{
   //__________________________________
   // Artifical dissipation term
   double eplus, eminus, dissipation;
   double k_const = 0.4;

   eplus  = 0.5 * k_const * deltaX * (nuFrt   + nuMid)/deltaT;
   eminus = 0.5 * k_const * deltaX * (nuLast  + nuMid)/deltaT;
   dissipation = (eplus * qFrt - (eplus + eminus) * qMid 
              +  eminus * qLast)/deltaX; 
 
/*`==========TESTING==========*/
 dissipation  = 0; 
/*==========TESTING==========`*/
             
   return  0.5 * (qConFrt - qConLast)/deltaX - dissipation;
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
 Function~ computeCornerCellIndices--
 Purpose~ return a list of indicies of the corner cells that are on the
           boundaries of the domain
___________________________________________________________________*/  
void computeCornerCellIndices(const Patch* patch,
                              const Patch::FaceType face,
                              vector<IntVector>& crn)
{     
  IntVector low,hi;
  low = patch->getLowIndex();
  hi  = patch->getHighIndex() - IntVector(1,1,1);  
  
  IntVector patchNeighborLow  = patch->neighborsLow();
  IntVector patchNeighborHigh = patch->neighborsHigh();
 
  IntVector axes = patch->faceAxes(face);
  int P_dir = axes[0];  // principal direction
  int dir1  = axes[1];  // other vector directions
  int dir2  = axes[2]; 

  //__________________________________
  // main index for that face plane
  int plusMinus = patch->faceDirection(face)[P_dir];
  int main_index = 0;
  if( plusMinus == 1 ) { // plus face
    main_index = hi[P_dir];
  } else {               // minus faces
    main_index = low[P_dir];
  }

  //__________________________________
  // Looking down on the face examine 
  // each corner (clockwise) and if there
  // are no neighboring patches then set the
  // index
  // 
  // Top-right corner
  IntVector corner(-9,-9,-9);
  if ( patchNeighborHigh[dir1] == 1 && patchNeighborHigh[dir2] == 1) {
   corner[P_dir] = main_index;
   corner[dir1]  = hi[dir1];
   corner[dir2]  = hi[dir2];
   crn.push_back(corner);
  }
  // bottom-right corner
  if ( patchNeighborLow[dir1] == 1 && patchNeighborHigh[dir2] == 1) {
   corner[P_dir] = main_index;
   corner[dir1]  = low[dir1];
   corner[dir2]  = hi[dir2];
   crn.push_back(corner);
  } 
  // bottom-left corner
  if ( patchNeighborLow[dir1] == 1 && patchNeighborLow[dir2] == 1) {
   corner[P_dir] = main_index;
   corner[dir1]  = low[dir1];
   corner[dir2]  = low[dir2];
   crn.push_back(corner);
  } 
  // Top-left corner
  if ( patchNeighborHigh[dir1] == 1 && patchNeighborLow[dir2] == 1) {
   corner[P_dir] = main_index;
   corner[dir1]  = hi[dir1];
   corner[dir2]  = low[dir2];
   crn.push_back(corner);
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
  const CCVariable<double>& rho_tmp = lv->rho_old;
  const CCVariable<Vector>& vel     = lv->vel_old;
  const double delT = lv->delT;

  double conv_dir1, conv_dir2;
  double qConFrt,qConLast;
  
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
    qConFrt  = rho_tmp[r1] * vel[r1][dir1];
    qConLast = rho_tmp[l1] * vel[l1][dir1];
    
    conv_dir1 = computeConvection(nu[r1][dir1], nu[c][dir1], nu[l1][dir1], 
                                  rho_tmp[r1], rho_tmp[c], rho_tmp[l1], 
                                  qConFrt, qConLast, delT, dx[dir1]);
    IntVector r2 = c;
    IntVector l2 = c;
    r2[dir2] += offset[dir2];  // tweak the r and l cell indices
    l2[dir2] -= offset[dir2];
    
    qConFrt  = rho_tmp[r2] * vel[r2][dir2];
    qConLast = rho_tmp[l2] * vel[l2][dir2];
    conv_dir2 = computeConvection(nu[r2][dir2], nu[c][dir2], nu[l2][dir2],
                                  rho_tmp[r2], rho_tmp[c], rho_tmp[l2], 
                                  qConFrt, qConLast, delT, dx[dir2]);

    rho_CC[c] = rho_tmp[c] - delT * (d[1][c][P_dir] + conv_dir1 + conv_dir2);
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

      qConFrt  = rho_tmp[r] * vel[r][Edir2];
      qConLast = rho_tmp[l] * vel[l][Edir2];
      double conv = computeConvection(nu[r][Edir2], nu[c][Edir2], nu[l][Edir2],
                                rho_tmp[r], rho_tmp[c], rho_tmp[l], 
                                qConFrt, qConLast, delT, dx[Edir2]);
                                
      rho_CC[c] = rho_tmp[c] - delT * (d[1][c][P_dir] + d[1][c][Edir1] + conv);
    }
  }

  //__________________________________
  // C O R N E R S    
  vector<IntVector> crn;
  computeCornerCellIndices(patch, face, crn);
 
  vector<IntVector>::iterator itr;
  for(itr = crn.begin(); itr != crn.end(); ++ itr ) {
    IntVector c = *itr;
    rho_CC[c] = rho_tmp[c] 
              - delT * (d[1][c][P_dir] + d[1][c][dir1] + d[1][c][dir2]);
  }           
}

/*_________________________________________________________________
 Function~ FaceVel_LODI--
 Purpose~  Compute velocity in boundary cells on x_plus face
___________________________________________________________________*/
void FaceVel_LODI(const Patch* patch,
                 Patch::FaceType face,
                 CCVariable<Vector>& vel_CC,                 
                 Lodi_vars* lv,
                 const Vector& dx)                     

{
  cout_doing << "Setting FaceVel_LODI on face " << face << endl;
  // bulletproofing
  if (!lv){
    throw InternalError("FaceVelLODI: Lodi_vars = null");
  }
     
  // shortcuts       
  StaticArray<CCVariable<Vector> >& d = lv->di;      
  constCCVariable<double>& rho_tmp = lv->rho_old;    
  constCCVariable<Vector>& vel     = lv->vel_old; 
  CCVariable<double>& p  = lv->press_tmp;          
  double delT = lv->delT;
                 
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
    //__________________________________
    // convective terms
    IntVector r1 = c;
    IntVector l1 = c;
    r1[dir1] += offset[dir1];  // tweak the r and l cell indices
    l1[dir1] -= offset[dir1]; 
       
    Vector convect1(0,0,0);
    for(int dir = 0; dir <3; dir ++ ) {
      convect1[dir] = 
        0.5 * ( (rho_tmp[r1] * vel[r1][dir] * vel[r1][dir1]
              -  rho_tmp[l1] * vel[l1][dir] * vel[l1][dir1] )/dx[dir1] );
    }
    
    IntVector r2 = c;
    IntVector l2 = c;
    r2[dir2] += offset[dir2];  // tweak the r and l cell indices
    l2[dir2] -= offset[dir2];  
    
    Vector convect2(0,0,0);
     for(int dir = 0; dir <3; dir ++ ) {
       convect2[dir] = 
         0.5 * ( (rho_tmp[r2] * vel[r2][dir] * vel[r2][dir2]
               -  rho_tmp[l2] * vel[l2][dir] * vel[l2][dir2] )/dx[dir2] );
     }       
    //__________________________________
    // Pressure gradient terms
    Vector pressGradient(0,0,0); 
    pressGradient[dir1] = 0.5 * (p[r1] - p[l1])/dx[dir1];  
    pressGradient[dir2] = 0.5 * (p[r2] - p[l2])/dx[dir2];   
    
    //__________________________________
    // Equation 9.9 - 9.10
    vel_CC[c][P_dir] =
           rho_tmp[c] * vel[c][P_dir] - delT * ( vel[c][P_dir] * d[1][c][P_dir] 
                                             + rho_tmp[c]      * d[3][c][P_dir]
                                             + convect1[P_dir] + convect2[P_dir]
                                             +   pressGradient[P_dir] );     
    vel_CC[c][dir1]  =
           rho_tmp[c] * vel[c][dir1] - delT * ( vel[c][dir1] * d[1][c][P_dir]
                                             +  rho_tmp[c]   * d[4][c][P_dir] 
                                             +  convect1[dir1] + convect2[dir1]
                                             +  pressGradient[dir1] );
    vel_CC[c][dir2] =
           rho_tmp[c] * vel[c][dir2] - delT * ( vel[c][dir2] * d[1][c][P_dir]
                                             +  rho_tmp[c]   * d[5][c][P_dir]
                                             +  convect1[dir2] + convect2[dir2]
                                             +  pressGradient[dir2] );
    vel_CC[c] /= rho_tmp[c];
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
    IntVector r1 = c;
    IntVector l1 = c;
    r1[Edir2] += offset[Edir2];  // tweak the r and l cell indices
    l1[Edir2] -= offset[Edir2]; 
       
    Vector convect1(0,0,0);
    for(int dir = 0; dir <3; dir ++ ) {
      convect1[dir] = 
        0.5 * ( (rho_tmp[r1] * vel[r1][dir] * vel[r1][Edir2]
              -  rho_tmp[l1] * vel[l1][dir] * vel[l1][Edir2] )/dx[Edir2] );
    }
    //__________________________________
    // Pressure gradient terms
    Vector pressGradient(0,0,0); 
    pressGradient[Edir2] = 0.5 * (p[r1] - p[l1])/dx[Edir2];  
    
    //__________________________________
    // Equation 9.9 - 9.10
    vel_CC[c][P_dir]= rho_tmp[c] * vel[c][P_dir] 
                    - delT * ( vel[c][P_dir] * (d[1][c][P_dir] + d[1][c][Edir1])
                           +   rho_tmp[c]    * (d[3][c][P_dir] + d[4][c][Edir1])
                           +   convect1[P_dir]
                           +   pressGradient[P_dir] );
    vel_CC[c][Edir1]= rho_tmp[c] * vel[c][Edir1]
                    - delT * ( vel[c][Edir1] * (d[1][c][P_dir] + d[1][c][Edir1])
                           +  rho_tmp[c]     * (d[4][c][P_dir] + d[3][c][Edir1])
                           +  convect1[Edir1]
                           +  pressGradient[Edir1] );
    vel_CC[c][Edir2]= rho_tmp[c] * vel[c][Edir2]
                    - delT * ( vel[c][Edir2] * (d[1][c][P_dir] + d[1][c][Edir1])
                           +  rho_tmp[c]     * (d[5][c][P_dir] + d[5][c][Edir1])
                           +  convect1[Edir2]
                           +  pressGradient[Edir2] );
    vel_CC[c] /= rho_tmp[c];
    }
  }  
  //________________________________________________________
  // C O R N E R S    
  double uVel, vVel, wVel;
  vector<IntVector> crn;
  computeCornerCellIndices(patch, face, crn);
 
  vector<IntVector>::iterator itr;
  for(itr = crn.begin(); itr != crn.end(); ++ itr ) {
    IntVector c = *itr;
    uVel = rho_tmp[c] * vel[c].x() - delT 
         * ((d[1][c].x() + d[1][c].y() + d[1][c].z()) * vel[c].x()  
         +  (d[3][c].x() + d[4][c].y() + d[5][c].z()) * rho_tmp[c]);

    vVel = rho_tmp[c] * vel[c].y() - delT 
         * ((d[1][c].x() + d[1][c].y() + d[1][c].z()) * vel[c].y() 
         +  (d[4][c].x() + d[3][c].y() + d[4][c].z()) * rho_tmp[c]);

    wVel = rho_tmp[c] * vel[c].z() - delT 
         * ((d[1][c].x() + d[1][c].y() + d[1][c].z()) * vel[c].z() 
         +  (d[5][c].x() + d[5][c].y() + d[3][c].z()) * rho_tmp[c]);

    vel_CC[c] = Vector(uVel, vVel, wVel)/ rho_tmp[c];
  }
} //end of the function FaceVelLODI() 

/*_________________________________________________________________
 Function~ FaceTemp_LODI--
 Purpose~  Compute temperature in boundary cells on faces
___________________________________________________________________*/
void FaceTemp_LODI(const Patch* patch,
             const Patch::FaceType face,
             CCVariable<double>& temp_CC,
             Lodi_vars* lv, 
             const Vector& dx)
{
  cout_doing << "Setting FaceTemp_LODI on face " <<face<< endl; 
  
  // bulletproofing
  if (!lv){
    throw InternalError("FaceTempLODI: Lodi_vars = null");
  } 
  // shortcuts  
  StaticArray<CCVariable<Vector> >& d = lv->di;
  const CCVariable<double>& e         = lv->e;
  const CCVariable<double>& rho_CC    = lv->rho_CC;
  const CCVariable<double>& rho_old   = lv->rho_old;
  const CCVariable<double>& press_tmp = lv->press_tmp;
  const CCVariable<Vector>& vel = lv->vel_old;
  const CCVariable<Vector>& nu  = lv->nu;
  const double delT  = lv->delT;
  const double gamma = lv->gamma;
  const double cv    = lv->cv;
             
  double qConFrt,qConLast, conv_dir1, conv_dir2;
  double term1, term2, term3;
  
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
    qConFrt  = vel[r1][dir1] * (e[r1] + press_tmp[r1]);
    qConLast = vel[l1][dir1] * (e[l1] + press_tmp[l1]);
    
    conv_dir1 = computeConvection(nu[r1][dir1], nu[c][dir1], nu[l1][dir1], 
                                  e[r1], e[c], e[l1], 
                                  qConFrt, qConLast, delT, dx[dir1]);
    IntVector r2 = c;
    IntVector l2 = c;
    r2[dir2] += offset[dir2];  // tweak the r and l cell indices
    l2[dir2] -= offset[dir2];
    
    qConFrt  = vel[r2][dir2] * (e[r2] + press_tmp[r2]);
    qConLast = vel[l2][dir2] * (e[l2] + press_tmp[l2]);
    conv_dir2 = computeConvection(nu[r2][dir2], nu[c][dir2], nu[l2][dir2],
                                  e[r2], e[c], e[l2], 
                                  qConFrt, qConLast, delT, dx[dir2]);

    double vel_sqr = vel[c].length2();
    term1 = 0.5 * d[1][c][P_dir] * vel_sqr;
    
    term2 = d[2][c][P_dir]/(gamma - 1.0) 
          + rho_old[c] * ( vel[c][P_dir] * d[3][c][P_dir] + 
                           vel[c][dir1]  * d[4][c][P_dir] +
                           vel[c][dir2]  * d[5][c][P_dir] );
                                 
    term3 = conv_dir1 + conv_dir2;
                                                      
    double e_tmp = e[c] - delT * (term1 + term2 + term3);               

    temp_CC[c] = e_tmp/(rho_CC[c]*cv) - 0.5 * vel_sqr/cv;
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

      qConFrt  = vel[r][Edir2] * (e[r] + press_tmp[r]);
      qConLast = vel[l][Edir2] * (e[l] + press_tmp[l]);
    
      double conv = computeConvection(nu[r][Edir2], nu[c][Edir2], nu[l][Edir2],
                                      e[r], e[c], e[l],               
                                      qConFrt, qConLast, delT, dx[Edir2]); 
                                      
      double vel_sqr = vel[c].length2();

      term1 = 0.5 * (d[1][c][P_dir] + d[1][c][Edir1]) * vel_sqr;

      term2 =  (d[2][c][P_dir] + d[2][c][Edir1])/(gamma - 1.0);
      
      if( edge == IntVector(1,1,0) ) { // Left/Right faces top/bottom edges
        term3 = rho_old[c] * vel[c][P_dir] * (d[3][c][P_dir] + d[4][c][Edir1])  
              + rho_old[c] * vel[c][Edir1] * (d[4][c][P_dir] + d[3][c][Edir1])  
              + rho_old[c] * vel[c][Edir2] * (d[5][c][P_dir] + d[5][c][Edir1]); 
      }
      
      if( edge == IntVector(1,0,1) ) { // Left/Right faces  Front/Back edges
        term3 = rho_old[c] * vel[c][P_dir] * (d[3][c][P_dir] + d[5][c][Edir1])  
              + rho_old[c] * vel[c][Edir2] * (d[4][c][P_dir] + d[4][c][Edir1])  
              + rho_old[c] * vel[c][Edir1] * (d[5][c][P_dir] + d[3][c][Edir1]); 
      }
      
      if( edge == IntVector(0,1,1) ) { // Top/Bottom faces  Front/Back edges
        term3 = rho_old[c] * vel[c][Edir2] * (d[4][c][P_dir] + d[5][c][Edir1])  
              + rho_old[c] * vel[c][P_dir] * (d[3][c][P_dir] + d[4][c][Edir1])  
              + rho_old[c] * vel[c][Edir1] * (d[5][c][P_dir] + d[3][c][Edir1]); 
      }      
      
      double e_tmp = e[c] - delT * ( term1 + term2 + term3 + conv);

      temp_CC[c] = e_tmp/(rho_CC[c] *cv) - 0.5 * vel_sqr/cv;
    }
  }  
 
  //________________________________________________________
  // C O R N E R S    
  vector<IntVector> crn;
  computeCornerCellIndices(patch, face, crn);
 
  vector<IntVector>::iterator itr;
  for(itr = crn.begin(); itr != crn.end(); ++ itr ) {
    IntVector c = *itr;
    double vel_sqr = vel[c].length2();

    term1 = 0.5 * (d[1][c].x() + d[1][c].y() + d[1][c].z()) * vel_sqr;
    term2 =       (d[2][c].x() + d[2][c].y() + d[2][c].z())/(gamma - 1.0);
    
    term3 = rho_old[c] * vel[c].x() * (d[3][c].x() + d[4][c].y() + d[5][c].z())  
          + rho_old[c] * vel[c].y() * (d[4][c].x() + d[3][c].y() + d[4][c].z())  
          + rho_old[c] * vel[c].z() * (d[5][c].x() + d[5][c].y() + d[3][c].z()); 

    double e_tmp = e[c] - delT * ( term1 + term2 + term3);

    temp_CC[c] = e_tmp/rho_CC[c]/cv - 0.5 * vel_sqr/cv;
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
  StaticArray<double> cv(numMatls);
  StaticArray<double> gamma(numMatls);

  StaticArray<constCCVariable<double> >& Temp_CC = lv->Temp_CC;
  StaticArray<constCCVariable<double> >& f_theta = lv->f_theta;

  //__________________________________  
  double press_ref= sharedState->getRefPress();    

  for (int m = 0; m < numMatls; m++) {
    Material* matl = sharedState->getMaterial( m );
    ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
    if(ice_matl){       
      cv[m]     = ice_matl->getSpecificHeat();
      gamma[m]  = ice_matl->getGamma();;        
    }
  }     
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
        ice_matl->getEOS()->computePressEOS(rho_micro[m][c],gamma[m],
                                       cv[m],Temp_CC[m][c],
                                       press_eos[m],tmp,tmp);        
      } 
      if(mpm_matl){                //  M P M
        mpm_matl->getConstitutiveModel()->
          computePressEOSCM(rho_micro[m][c],press_eos[m], press_ref,
                            tmp, tmp,mpm_matl);
      }              
      press_CC[c] += f_theta[m][c]*press_eos[m];
//     cout << "press_CC" << c << press_CC[c] << endl;           
    }  // for ALLMatls...
  }
} 
  
  
}  // using namespace Uintah
