#include <Packages/Uintah/CCA/Components/ICE/BoundaryCond.h>
#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>

#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/PerPatch.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/BoundCond.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/fillFace.h>
#include <Core/Util/DebugStream.h>
#include <typeinfo>

using namespace Uintah;
namespace Uintah {

#ifdef LODI_BCS     // defined in BoundaryCond.h
//__________________________________
//  To turn on couts
//  setenv SCI_DEBUG "LODI_DOING_COUT:+, LODI_DBG_COUT:+"
static DebugStream cout_doing("LODI_DOING_COUT", false);
static DebugStream cout_dbg("LODI_DBG_COUT", false);
       
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
 Function~ computeCornerCellIndices--
 Purpose~  compute the corner cells for any face
___________________________________________________________________*/  
void computeCornerCellIndices(const Patch* patch,
                              const Patch::FaceType face,
                              vector<IntVector>& crn)
{     
  IntVector low,hi;
  low = patch->getLowIndex();
  hi  = patch->getHighIndex() - IntVector(1,1,1);  
  
  switch(face){
  case Patch::xminus:
    crn[0] = IntVector(low.x(), low.y(), hi.z());   
    crn[1] = IntVector(low.x(), hi.y(),  hi.z());   
    crn[2] = IntVector(low.x(), hi.y(),  low.z());
    crn[3] = IntVector(low.x(), low.y(), low.z());
    break;
  case Patch::xplus:
    crn[0] = IntVector(hi.x(), low.y(), hi.z());   
    crn[1] = IntVector(hi.x(), hi.y(),  hi.z());   
    crn[2] = IntVector(hi.x(), hi.y(),  low.z());
    crn[3] = IntVector(hi.x(), low.y(), low.z());
    break;
  case Patch::yminus:
    crn[0] = IntVector(hi.x(),  low.y(), hi.z());    
    crn[1] = IntVector(low.x(), low.y(), hi.z());    
    crn[2] = IntVector(low.x(), low.y(), low.z()); 
    crn[3] = IntVector(hi.x(),  low.y(), low.z()); 
    break;
  case Patch::yplus:
    crn[0] = IntVector(hi.x(),  hi.y(), hi.z());    
    crn[1] = IntVector(low.x(), hi.y(), hi.z());    
    crn[2] = IntVector(low.x(), hi.y(), low.z()); 
    crn[3] = IntVector(hi.x(),  hi.y(), low.z()); 
    break;
  case Patch::zminus:
    crn[0] = IntVector(hi.x(),  low.y(), low.z()); 
    crn[1] = IntVector(hi.x(),  hi.y(),  low.z()); 
    crn[2] = IntVector(low.x(), hi.y(),  low.z()); 
    crn[3] = IntVector(low.x(), low.y(), low.z()); 
    break;
  case Patch::zplus:
    crn[0] = IntVector(hi.x(),  low.y(), hi.z()); 
    crn[1] = IntVector(hi.x(),  hi.y(),  hi.z()); 
    crn[2] = IntVector(low.x(), hi.y(),  hi.z()); 
    crn[3] = IntVector(low.x(), low.y(), hi.z()); 
    break;
  default:
    throw InternalError("Illegal FaceType in LODIBaseFunc.h"
                        " computeCornerCellIndices");
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
 Function~ FaceDensityLODI--
 Purpose~  Compute density in boundary cells on any face
___________________________________________________________________*/
void FaceDensityLODI(const Patch* patch,
                const Patch::FaceType face,
                CCVariable<double>& rho_CC,
                StaticArray<CCVariable<Vector> >& d,
                const CCVariable<Vector>& nu,
                const CCVariable<double>& rho_tmp,
                const CCVariable<Vector>& vel,
                const double delT,
                const Vector& dx)
{
  cout_doing << "I am in FaceDensityLODI on face " << face<<endl;
  double conv_dir1, conv_dir2;
  double qConFrt,qConLast;
  
  IntVector axes = patch->faceAxes(face);
  int P_dir = axes[0];  // principal direction
  int dir1  = axes[1];  // other vector directions
  int dir2  = axes[2];
  
  IntVector offset = IntVector(1,1,1) - Abs(patch->faceDirection(face));
  
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
  //    E D G E S
  for(Patch::FaceType face0 = Patch::startFace; face0 <= Patch::endFace; 
                                                  face0=Patch::nextFace(face0)){
/*`==========TESTING==========*/
 /*__________________________________
 *CHEEZY BULLET PROOFING UNTIL CELL ITERATOR is FIXED
 *__________________________________*/
IntVector Tdir0 = patch->faceDirection(face);
IntVector Tdir1 = patch->faceDirection(face0);
IntVector test = Abs(Tdir0) - Abs(Tdir1); 
if (test != IntVector(0,0,0)) {  //  no edge here 
/*==========TESTING==========`*/
    
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
}  // cheezy bulletproofing
  }

  //__________________________________
  // C O R N E R S    
  vector<IntVector> crn(4);
  computeCornerCellIndices(patch, face, crn);
 
  for(int corner = 0; corner <4; corner ++ ) {
    IntVector c = crn[corner];
    rho_CC[c] = rho_tmp[c] - delT * (d[1][c][P_dir] + d[1][c][dir1] + d[1][c][dir2]);
  }           
}
/*_________________________________________________________________
 Function~ fillFaceDensityLODI--
 Purpose~  
___________________________________________________________________*/
void fillFaceDensityLODI(const Patch* patch,
                   CCVariable<double>& rho_CC,
                   StaticArray<CCVariable<Vector> >& di,
                   const CCVariable<Vector>& nu,
                   const CCVariable<double>& rho_tmp,
                   const CCVariable<Vector>& vel,
                   const Patch::FaceType face,
                   const double delT,
                   const Vector& dx)
{   
  if (face == Patch::xplus || face == Patch::xminus ) {
    FaceDensityLODI(patch, face, rho_CC, di, nu, rho_tmp, vel, delT, dx);
  } 

  if (face == Patch::yplus || face == Patch::yminus ) {
    FaceDensityLODI(patch, face, rho_CC, di, nu, rho_tmp, vel, delT, dx);
  } 

  if (face == Patch::zplus || face == Patch::zminus ) { 
    FaceDensityLODI(patch, face, rho_CC, di, nu, rho_tmp, vel, delT, dx);
  } 
}

/*_________________________________________________________________
 Function~ FaceVelLODI--
 Purpose~  Compute velocity in boundary cells on x_plus face
___________________________________________________________________*/
void FaceVelLODI(const Patch* patch,
                 Patch::FaceType face,                 
                 CCVariable<Vector>& vel_CC,           
                 StaticArray<CCVariable<Vector> >& d,  
                 const CCVariable<Vector>& nu,         
                 const CCVariable<double>& rho_tmp,    
                 const CCVariable<double>& p,          
                 const CCVariable<Vector>& vel,        
                 const double delT,                    
                 const Vector& dx)                     

{
  cout_doing << " I am in FaceVelLODI on face " << face << endl;
  IntVector axes = patch->faceAxes(face);
  int P_dir = axes[0];  // principal direction
  int dir1  = axes[1];  // other vector directions
  int dir2  = axes[2];
  
  IntVector offset = IntVector(1,1,1) - Abs(patch->faceDirection(face));
 
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
      convect1[dir] = 0.5 * ( (rho_tmp[r1] * vel[r1][dir] * vel[r1][dir1]
                            -  rho_tmp[l1] * vel[l1][dir] * vel[l1][dir1] )/dx[dir1] );
    }
    
    IntVector r2 = c;
    IntVector l2 = c;
    r2[dir2] += offset[dir2];  // tweak the r and l cell indices
    l2[dir2] -= offset[dir2];  
    
    Vector convect2(0,0,0);
     for(int dir = 0; dir <3; dir ++ ) {
       convect2[dir] = 0.5 * ( (rho_tmp[r2] * vel[r2][dir] * vel[r2][dir2]
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
                                             +   rho_tmp[c]    * d[3][c][P_dir]          
                                             +   convect1[P_dir] + convect2[P_dir] 
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
  //    E D G E S
  for(Patch::FaceType face0 = Patch::startFace; face0 <= Patch::endFace; 
                                                  face0=Patch::nextFace(face0)){
/*`==========TESTING==========*/
 /*__________________________________
 *CHEEZY BULLET PROOFING UNTIL CELL ITERATOR is FIXED
 *__________________________________*/
IntVector Tdir0 = patch->faceDirection(face);
IntVector Tdir1 = patch->faceDirection(face0);
IntVector test = Abs(Tdir0) - Abs(Tdir1); 
if (test != IntVector(0,0,0)) {  //  no edge here 
/*==========TESTING==========`*/
    
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
      convect1[dir] = 0.5 * ( (rho_tmp[r1] * vel[r1][dir] * vel[r1][Edir2]
                            -  rho_tmp[l1] * vel[l1][dir] * vel[l1][Edir2] )/dx[Edir2] );
    }
    //__________________________________
    // Pressure gradient terms
    Vector pressGradient(0,0,0); 
    pressGradient[Edir2] = 0.5 * (p[r1] - p[l1])/dx[Edir2];  
    
    //__________________________________
    // Equation 9.9 - 9.10
    vel_CC[c][P_dir] =
           rho_tmp[c] * vel[c][P_dir] - delT * ( vel[c][P_dir] * (d[1][c][P_dir] + d[1][c][Edir1])
                                             +   rho_tmp[c]    * (d[3][c][P_dir] + d[4][c][Edir1])          
                                             +   convect1[P_dir]
                                             +   pressGradient[P_dir] );     
    vel_CC[c][Edir1]  =
           rho_tmp[c] * vel[c][Edir1] - delT * ( vel[c][Edir1] * (d[1][c][P_dir] + d[1][c][Edir1])
                                             +  rho_tmp[c]     * (d[4][c][P_dir] + d[3][c][Edir1])
                                             +  convect1[Edir1]
                                             +  pressGradient[Edir1] );
    vel_CC[c][Edir2] =
           rho_tmp[c] * vel[c][Edir2] - delT * ( vel[c][Edir2] * (d[1][c][P_dir] + d[1][c][Edir1])
                                             +  rho_tmp[c]     * (d[5][c][P_dir] + d[5][c][Edir1])
                                             +  convect1[Edir2]
                                             +  pressGradient[Edir2] );
    vel_CC[c] /= rho_tmp[c];
    }
}  // cheezy bulletproofing
  }  
   //________________________________________________________
   // C O R N E R S    
   vector<IntVector> crn(4);
   double uVel, vVel, wVel;
   computeCornerCellIndices(patch, face, crn);

   for( int corner = 0; corner < 4; corner ++ ) {
     IntVector c = crn[corner];
     uVel = vel[c].x() - delT 
          * ((d[1][c].x() + d[1][c].y() + d[1][c].z()) * vel[c].x()  
          +  (d[3][c].x() + d[4][c].y() + d[5][c].z()) * rho_tmp[c]);

     vVel = vel[c].y() - delT 
          * ((d[1][c].x() + d[1][c].y() + d[1][c].z()) * vel[c].y() 
          +  (d[4][c].x() + d[3][c].y() + d[4][c].z()) * rho_tmp[c]);

     wVel = vel[c].z() - delT 
          * ((d[1][c].x() + d[1][c].y() + d[1][c].z()) * vel[c].z() 
          +  (d[5][c].x() + d[5][c].y() + d[3][c].z()) * rho_tmp[c]);
          
     vel_CC[c] = Vector(uVel, vVel, wVel)/ rho_tmp[c];
   }
} //end of the function FaceVelLODI() 


/*_________________________________________________________________
 Function~ fillFaceVelLODI--
 Purpose~  Compute velocity at boundary cells 
___________________________________________________________________*/

void   fillFaceVelLODI(const Patch* patch,
                 CCVariable<Vector>& vel_CC,
                 StaticArray<CCVariable<Vector> >& di,
                 const CCVariable<Vector>& nu,
                 const CCVariable<double>& rho_tmp,
                 const CCVariable<double>& p,
                 const CCVariable<Vector>& vel,
                 const Patch::FaceType face,
                 const double delT,
                 const Vector& dx)

{ 
  if(face ==  Patch::xplus || face ==  Patch::xminus) {
    FaceVelLODI( patch, face, vel_CC, di, nu,
                  rho_tmp, p, vel, delT, dx);
  }
  if(face ==  Patch::yplus || face ==  Patch::yminus) {
    FaceVelLODI( patch, face, vel_CC, di, nu,
                  rho_tmp, p, vel, delT, dx);        
  } 
  if(face ==  Patch::zplus || face ==  Patch::zminus) {
    FaceVelLODI( patch, face, vel_CC, di, nu,
                  rho_tmp, p, vel, delT, dx);
  }
}


/*`==========TESTING==========*/
/*_________________________________________________________________
 Function~ FaceTempLODI--
 Purpose~  Compute temperature in boundary cells on faces
___________________________________________________________________*/
void FaceTempLODI(const Patch* patch,
             const Patch::FaceType face,
             CCVariable<double>& temp_CC, 
             StaticArray<CCVariable<Vector> >& d,
             const CCVariable<double>& e,
             const CCVariable<double>& rho_CC,
             const CCVariable<Vector>& nu,
             const CCVariable<double>& rho_tmp,
             const CCVariable<double>& p,
             const CCVariable<Vector>& vel,
             const double delT,
             const double cv,
             const double gamma, 
             const Vector& dx)
{
  cout_doing << " I am in FaceTempLODI on face " <<face<< endl;    

  double qConFrt,qConLast, conv_dir1, conv_dir2;
  double term1, term2, term3;
  
  IntVector axes = patch->faceAxes(face);
  int P_dir = axes[0];  // principal direction
  int dir1  = axes[1];  // other vector directions
  int dir2  = axes[2];
  
  IntVector offset = IntVector(1,1,1) - Abs(patch->faceDirection(face));
  
  for(CellIterator iter=patch->getFaceCellIterator(face, "minusEdgeCells"); 
                                                      !iter.done();iter++) {
    IntVector c = *iter;
    
    IntVector r1 = c;
    IntVector l1 = c;
    r1[dir1] += offset[dir1];  // tweak the r and l cell indices
    l1[dir1] -= offset[dir1];
    qConFrt  = vel[r1][dir1] * (e[r1] + p[r1]);
    qConLast = vel[l1][dir1] * (e[l1] + p[l1]);
    
    conv_dir1 = computeConvection(nu[r1][dir1], nu[c][dir1], nu[l1][dir1], 
                                  e[r1], e[c], e[l1], 
                                  qConFrt, qConLast, delT, dx[dir1]);
    IntVector r2 = c;
    IntVector l2 = c;
    r2[dir2] += offset[dir2];  // tweak the r and l cell indices
    l2[dir2] -= offset[dir2];
    
    qConFrt  = vel[r2][dir2] * (e[r2] + p[r2]);
    qConLast = vel[l2][dir2] * (e[l2] + p[l2]);
    conv_dir2 = computeConvection(nu[r2][dir2], nu[c][dir2], nu[l2][dir2],
                                  e[r2], e[c], e[l2], 
                                  qConFrt, qConLast, delT, dx[dir2]);


    double vel_sqr = vel[c].length2();
    term1 = 0.5 * d[1][c][P_dir] * vel_sqr;
    
    term2 = d[2][c][P_dir]/(gamma - 1.0) 
          + rho_tmp[c] * ( vel[c][P_dir] * d[3][c][P_dir] + 
                           vel[c][dir1]  * d[4][c][P_dir] +
                           vel[c][dir2]  * d[5][c][P_dir] );
                                 
    term3 = conv_dir1 + conv_dir2;
                                                      
    double e_tmp = e[c] - delT * (term1 + term2 + term3);               

    temp_CC[c] = e_tmp/rho_CC[c]/cv - 0.5 * vel_sqr/cv;
  }
  
  
  //__________________________________
  //    E D G E S
  for(Patch::FaceType face0 = Patch::startFace; face0 <= Patch::endFace; 
                                                  face0=Patch::nextFace(face0)){

 /*__________________________________
 *CHEEZY BULLET PROOFING UNTIL CELL ITERATOR is FIXED
 *__________________________________*/
IntVector Tdir0 = patch->faceDirection(face);
IntVector Tdir1 = patch->faceDirection(face0);
IntVector test = Abs(Tdir0) - Abs(Tdir1); 
if (test != IntVector(0,0,0)) {  //  no edge here 
    
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

      qConFrt  = vel[r][Edir2] * (e[r] + p[r]);
      qConLast = vel[l][Edir2] * (e[l] + p[l]);
    
      double conv = computeConvection(nu[r][Edir2], nu[c][Edir2], nu[l][Edir2],
                                      e[r], e[c], e[l],               
                                      qConFrt, qConLast, delT, dx[Edir2]); 
                                      
      double vel_sqr = vel[c].length2();

      term1 = 0.5 * (d[1][c][P_dir] + d[1][c][Edir1]) * vel_sqr;

      term2 =  (d[2][c][P_dir] + d[2][c][Edir1])/(gamma - 1.0);

      term3 =  rho_tmp[c] * vel[c][P_dir] * (d[3][c][P_dir] + d[4][c][Edir1])
            +  rho_tmp[c] * vel[c][Edir1] * (d[4][c][P_dir] + d[3][c][Edir1]) 
            +  rho_tmp[c] * vel[c][Edir2] * (d[5][c][P_dir] + d[5][c][Edir1]);

      double e_tmp = e[c] - delT * ( term1 + term2 + term3 + conv);

      temp_CC[c] = e_tmp/rho_CC[c]/cv - 0.5 * vel_sqr/cv;
    }
}  // cheezy bulletproofing
  }  
#if 0       
  //__________________________________________________________
  //  E D G E     right-bottom
  for(int k = low.z()+1; k < hi_z; k++) {
    IntVector c  (xFaceCell, low.y(), k);
    IntVector f  (xFaceCell, low.y(), k+1);
    IntVector bk (xFaceCell, low.y(), k-1);     
    qConFrt  = vel[f].z()  * (e[f]  + p[f]);
    qConLast = vel[bk].z() * (e[bk] + p[bk]);
    qFrt     = e[f];
    qMid     = e[c];
    qLast    = e[bk];

     z_conv = computeConvection(nu[f].z(), nu[c].z(), nu[bk].z(),
                                   qFrt, qMid, qLast, 
                                   qConFrt, qConLast, delT, dx.z());

     double vel_sqr = vel[c].length2();

     term1 = 0.5 * (d[1][c].x() + d[1][c].y()) * vel_sqr;
     term2 =  (d[2][c].x() + d[2][c].y())/(gamma - 1.0) 
           +  rho_tmp[c] * vel[c].x() * (d[3][c].x() + d[4][c].y());
     term3 =  rho_tmp[c] * vel[c].y() * (d[4][c].x() + d[3][c].y()) 
           +  rho_tmp[c] * vel[c].z() * (d[5][c].x() + d[5][c].y());

     double e_tmp = e[c] - delT * ( term1 + term2 + term3 + z_conv);

     temp_CC[c] = e_tmp/rho_CC[c]/cv - 0.5 * vel_sqr/cv;
  } //end of k loop

  //_______________________________________________________
  //    E D G E   right/left-top
  for(int k = low.z()+1; k < hi_z; k++) {
    IntVector c  (xFaceCell,   hi_y,   k);
    IntVector f  (xFaceCell,   hi_y,   k+1);
    IntVector bk (xFaceCell,   hi_y,   k-1);

    qConFrt  = vel[f].z() * (e[f]  + p[f]);
    qConLast = vel[bk].z() *(e[bk] + p[bk]);
    qFrt     = e[f];
    qMid     = e[c];
    qLast    = e[bk];

    z_conv = computeConvection(nu[f].z(), nu[c].z(), nu[bk].z(),
                     qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.z());

    double vel_sqr = vel[c].length2();

    term1 = 0.5 * (d[1][c].x() + d[1][c].y()) * vel_sqr;
    term2 =  (d[2][c].x() + d[2][c].y())/(gamma - 1.0) 
          +  rho_tmp[c] * vel[c].x() * (d[3][c].x() + d[4][c].y());
    term3 =  rho_tmp[c] * vel[c].y() * (d[4][c].x() + d[3][c].y()) 
          +  rho_tmp[c] * vel[c].z() * (d[5][c].x() + d[5][c].y());

    double e_tmp = e[c] - delT * ( term1 + term2 + term3 + z_conv);

    temp_CC[c] = e_tmp/rho_CC[c]/cv - 0.5 * vel_sqr/cv;
  } 
  //_____________________________________________________
  //    E D G E    right/left-back
  for(int j = low.y()+1; j < hi_y; j++) {
    IntVector c (xFaceCell,   j,   low.z());
    IntVector t (xFaceCell,   j+1, low.z());
    IntVector b (xFaceCell,   j-1, low.z());

    qConFrt  = vel[t].y() * (e[t] + p[t]);
    qConLast = vel[b].y() * (e[b] + p[b]);
    qFrt     = e[t];
    qMid     = e[c];
    qLast    = e[b];

    y_conv = computeConvection(nu[t].y(), nu[c].y(), nu[b].y(),
                    qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.y());

    double vel_sqr = vel[c].length2();

    term1 = 0.5 * (d[1][c].x() + d[1][c].z()) * vel_sqr;
    term2 =  (d[2][c].x() + d[2][c].z())/(gamma - 1.0) 
          +  rho_tmp[c] * vel[c].x() * (d[3][c].x() + d[4][c].z());
    term3 =  rho_tmp[c] * vel[c].y() * (d[4][c].x() + d[5][c].z()) 
          +  rho_tmp[c] * vel[c].z() * (d[5][c].x() + d[3][c].z());

    double  e_tmp = e[c] - delT * ( term1 + term2 + term3 + y_conv);

    temp_CC[c] = e_tmp/rho_CC[c]/cv - 0.5 * vel_sqr/cv;
  }
  //_________________________________________________
  // E D G E     right/left-front
  for(int j = low.y()+1; j < hi_y; j++) {
    IntVector c (xFaceCell,   j,   hi_z);
    IntVector t (xFaceCell,   j+1, hi_z);
    IntVector b (xFaceCell,   j-1, hi_z);

    qConFrt  = vel[t].y() * (e[t] + p[t]);
    qConLast = vel[b].y() * (e[b] + p[b]);
    qFrt     = e[t];
    qMid     = e[c];
    qLast    = e[b];

    y_conv = computeConvection(nu[t].y(), nu[c].y(), nu[b].y(),
                      qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.y());

    double vel_sqr = vel[c].length2();

    term1 = 0.5 * (d[1][c].x() + d[1][c].z()) * vel_sqr;
    term2 =  (d[2][c].x() + d[2][c].z())/(gamma - 1.0) 
          +  rho_tmp[c] * vel[c].x() * (d[3][c].x() + d[4][c].z());
    term3 =  rho_tmp[c] * vel[c].y() * (d[4][c].x() + d[5][c].z()) 
          +  rho_tmp[c] * vel[c].z() * (d[5][c].x() + d[3][c].z());

    double e_tmp = e[c] - delT * ( term1 + term2 + term3 + y_conv);

    temp_CC[c] = e_tmp/rho_CC[c]/cv - 0.5 * vel_sqr/cv;
  }
  //________________________________________________________
  // C O R N E R S    
  vector<IntVector> crn(4);
  crn[0] = IntVector(xFaceCell, low.y(), hi_z);     // right/left-bottom-front
  crn[1] = IntVector(xFaceCell, hi_y,    hi_z);     // right/left-top-front   
  crn[2] = IntVector(xFaceCell, hi_y,    low.z());  // right/left-top-back
  crn[3] = IntVector(xFaceCell, low.y(), low.z());  // right/left-bottom-back 

  for( int corner = 0; corner < 4; corner ++ ) {
     IntVector c = crn[corner];
     double vel_sqr = vel[c].length2();

     term1 = 0.5 * (d[1][c].x() + d[1][c].y() + d[1][c].z()) * vel_sqr;
     term2 =       (d[2][c].x() + d[2][c].y() + d[2][c].z())/(gamma - 1.0) 
           + 
     term3 =  rho_tmp[c] * vel[c].x() * (d[3][c].x() + d[4][c].y() + d[4][c].z()); 
           +  rho_tmp[c] * vel[c].y() * (d[4][c].x() + d[3][c].y() + d[5][c].z()) 
           +  rho_tmp[c] * vel[c].z() * (d[5][c].x() + d[5][c].y() + d[3][c].z());

     double e_tmp = e[c] - delT * ( term1 + term2 + term3);

     temp_CC[c] = e_tmp/rho_CC[c]/cv - 0.5 * vel_sqr/cv;
  }
#endif
} //end of function FaceTempLODI()  
/*==========TESTING==========`*/


/*_________________________________________________________________
 Function~ xFaceTempLODI--
 Purpose~  Compute temperature in boundary cells on x_plus/minus faces
___________________________________________________________________*/
void xFaceTempLODI(const Patch::FaceType face,
             CCVariable<double>& temp_CC, 
             StaticArray<CCVariable<Vector> >& d,
             const CCVariable<double>& e,
             const CCVariable<double>& rho_CC,
             const CCVariable<Vector>& nu,
             const CCVariable<double>& rho_tmp,
             const CCVariable<double>& p,
             const CCVariable<Vector>& vel,
             const double delT,
             const double cv,
             const double gamma, 
             const Vector& dx)
{
  cout_doing << " I am in xFaceTempLODI on face " <<face<< endl;    
  IntVector low,hi;
  int xFaceCell;
  low = vel.getLowIndex();
  hi  = vel.getHighIndex();
  int hi_x = hi.x() - 1;
  int hi_y = hi.y() - 1;
  int hi_z = hi.z() - 1;
  double term1,term2,term3,term4;
  double y_conv, z_conv;
  double qConFrt,qConLast,qFrt,qMid,qLast;
  
  if(face == Patch::xplus ){
    xFaceCell = hi_x;  
  } else {
    xFaceCell = low.x();
  } 
  for(int j = low.y()+1; j < hi_y; j++) {
    for(int k = low.z()+1; k < hi_z; k++) {
      IntVector c  (xFaceCell,  j,   k);
      IntVector t  (xFaceCell,  j+1, k);
      IntVector b  (xFaceCell,  j-1, k);
      IntVector f  (xFaceCell,  j,   k+1);
      IntVector bk (xFaceCell,  j,   k-1);

      //_______________________________________________________________
      // energy conservation law, computing pressure, or temperature
      // Please remember e is the total energy per unit volume, 
      // not per unit mass, must be given or compute. The ICE code compute 
      // just internal energy, not the total energy. Here the code is written 
      // just for ICE materials

      qConFrt  = vel[t].y() * (e[t] + p[t]);
      qConLast = vel[b].y() * (e[b] + p[b]);
      qFrt     = e[t];
      qMid     = e[c];
      qLast    = e[b];
      y_conv   = computeConvection(nu[t].y(), nu[c].y(), nu[b].y(), 
                                qFrt, qMid, qLast, qConFrt, 
                                qConLast, delT, dx.y());

      qConFrt  = vel[f].z()  * (e[f] + p[f]);
      qConLast = vel[bk].z() * (e[bk] + p[bk]);
      qFrt     = e[f];
      qLast    = e[bk];

      z_conv = computeConvection(nu[f].z(), nu[c].z(), nu[bk].z(),
                                 qFrt, qMid, qLast, 
                                 qConFrt, qConLast, delT, dx.z()); 

      double vel_sqr = vel[c].length2();

      term1 = 0.5 * d[1][c].x() * vel_sqr;
      term2 = d[2][c].x()/(gamma - 1.0) + rho_tmp[c] * vel[c].x() * d[3][c].x();
      term3 = rho_tmp[c] * vel[c].y() * d[4][c].x() 
            + rho_tmp[c] * vel[c].z() * d[5][c].x();
      term4 = y_conv + z_conv;
      double e_tmp = e[c] - delT * (term1 + term2 + term3 + term4);
/*`==========TESTING==========*/
     double junk = e_tmp/rho_CC[c]/cv - 0.5 * vel_sqr/cv;
     if ( fabs(junk - temp_CC[c]) > 1e-9) {
        cout << " XFace " << face << " " << c << " oldStyleTemp " << junk << " New Style Temp " << temp_CC[c] << endl;
     } 
/*==========TESTING==========`*/
      temp_CC[c] = e_tmp/rho_CC[c]/cv - 0.5 * vel_sqr/cv;  
    } // end of j loop
  } //end of k loop
       
  //__________________________________________________________
  //  E D G E     right-bottom
  for(int k = low.z()+1; k < hi_z; k++) {
    IntVector c  (xFaceCell, low.y(), k);
    IntVector f  (xFaceCell, low.y(), k+1);
    IntVector bk (xFaceCell, low.y(), k-1);     
    qConFrt  = vel[f].z()  * (e[f]  + p[f]);
    qConLast = vel[bk].z() * (e[bk] + p[bk]);
    qFrt     = e[f];
    qMid     = e[c];
    qLast    = e[bk];

     z_conv = computeConvection(nu[f].z(), nu[c].z(), nu[bk].z(),
                      qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.z());

     double vel_sqr = vel[c].length2();

     term1 = 0.5 * (d[1][c].x() + d[1][c].y()) * vel_sqr;
     term2 =  (d[2][c].x() + d[2][c].y())/(gamma - 1.0) 
           +  rho_tmp[c] * vel[c].x() * (d[3][c].x() + d[4][c].y());
     term3 =  rho_tmp[c] * vel[c].y() * (d[4][c].x() + d[3][c].y()) 
           +  rho_tmp[c] * vel[c].z() * (d[5][c].x() + d[5][c].y());

     double e_tmp = e[c] - delT * ( term1 + term2 + term3 + z_conv);
/*`==========TESTING==========*/
     double junk = e_tmp/rho_CC[c]/cv - 0.5 * vel_sqr/cv;
     if ( fabs(junk - temp_CC[c]) > 1e-9) {
        cout << " XFace Bottom Edge " << face << " " << c << " oldStyleTemp " << junk << " New Style Temp " << temp_CC[c] << endl;
     } 
/*==========TESTING==========`*/
     temp_CC[c] = e_tmp/rho_CC[c]/cv - 0.5 * vel_sqr/cv;


  } //end of k loop

  //_______________________________________________________
  //    E D G E   right/left-top
  for(int k = low.z()+1; k < hi_z; k++) {
    IntVector c  (xFaceCell,   hi_y,   k);
    IntVector f  (xFaceCell,   hi_y,   k+1);
    IntVector bk (xFaceCell,   hi_y,   k-1);

    qConFrt  = vel[f].z() * (e[f]  + p[f]);
    qConLast = vel[bk].z() *(e[bk] + p[bk]);
    qFrt     = e[f];
    qMid     = e[c];
    qLast    = e[bk];

    z_conv = computeConvection(nu[f].z(), nu[c].z(), nu[bk].z(),
                     qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.z());

    double vel_sqr = vel[c].length2();

    term1 = 0.5 * (d[1][c].x() + d[1][c].y()) * vel_sqr;
    term2 =  (d[2][c].x() + d[2][c].y())/(gamma - 1.0) 
          +  rho_tmp[c] * vel[c].x() * (d[3][c].x() + d[4][c].y());
    term3 =  rho_tmp[c] * vel[c].y() * (d[4][c].x() + d[3][c].y()) 
          +  rho_tmp[c] * vel[c].z() * (d[5][c].x() + d[5][c].y());

    double e_tmp = e[c] - delT * ( term1 + term2 + term3 + z_conv);
/*`==========TESTING==========*/
     double junk = e_tmp/rho_CC[c]/cv - 0.5 * vel_sqr/cv;
     if ( fabs(junk - temp_CC[c]) > 1e-9) {
        cout << " XFace Top Edge " << face << " " << c << " oldStyleTemp " << junk << " New Style Temp " << temp_CC[c] << endl;
     } 
/*==========TESTING==========`*/
    temp_CC[c] = e_tmp/rho_CC[c]/cv - 0.5 * vel_sqr/cv;
  } 
  //_____________________________________________________
  //    E D G E    right/left-back
  for(int j = low.y()+1; j < hi_y; j++) {
    IntVector c (xFaceCell,   j,   low.z());
    IntVector t (xFaceCell,   j+1, low.z());
    IntVector b (xFaceCell,   j-1, low.z());

    qConFrt  = vel[t].y() * (e[t] + p[t]);
    qConLast = vel[b].y() * (e[b] + p[b]);
    qFrt     = e[t];
    qMid     = e[c];
    qLast    = e[b];

    y_conv = computeConvection(nu[t].y(), nu[c].y(), nu[b].y(),
                    qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.y());

    double vel_sqr = vel[c].length2();

    term1 = 0.5 * (d[1][c].x() + d[1][c].z()) * vel_sqr;
    term2 =  (d[2][c].x() + d[2][c].z())/(gamma - 1.0) 
          +  rho_tmp[c] * vel[c].x() * (d[3][c].x() + d[4][c].z());
    term3 =  rho_tmp[c] * vel[c].y() * (d[4][c].x() + d[5][c].z()) 
          +  rho_tmp[c] * vel[c].z() * (d[5][c].x() + d[3][c].z());

    double  e_tmp = e[c] - delT * ( term1 + term2 + term3 + y_conv);

    temp_CC[c] = e_tmp/rho_CC[c]/cv - 0.5 * vel_sqr/cv;
  }
  //_________________________________________________
  // E D G E     right/left-front
  for(int j = low.y()+1; j < hi_y; j++) {
    IntVector c (xFaceCell,   j,   hi_z);
    IntVector t (xFaceCell,   j+1, hi_z);
    IntVector b (xFaceCell,   j-1, hi_z);

    qConFrt  = vel[t].y() * (e[t] + p[t]);
    qConLast = vel[b].y() * (e[b] + p[b]);
    qFrt     = e[t];
    qMid     = e[c];
    qLast    = e[b];

    y_conv = computeConvection(nu[t].y(), nu[c].y(), nu[b].y(),
                      qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.y());

    double vel_sqr = vel[c].length2();

    term1 = 0.5 * (d[1][c].x() + d[1][c].z()) * vel_sqr;
    term2 =  (d[2][c].x() + d[2][c].z())/(gamma - 1.0) 
          +  rho_tmp[c] * vel[c].x() * (d[3][c].x() + d[4][c].z());
    term3 =  rho_tmp[c] * vel[c].y() * (d[4][c].x() + d[5][c].z()) 
          +  rho_tmp[c] * vel[c].z() * (d[5][c].x() + d[3][c].z());

    double e_tmp = e[c] - delT * ( term1 + term2 + term3 + y_conv);

    temp_CC[c] = e_tmp/rho_CC[c]/cv - 0.5 * vel_sqr/cv;
  }
  //________________________________________________________
  // C O R N E R S    
  vector<IntVector> crn(4);
  crn[0] = IntVector(xFaceCell, low.y(), hi_z);     // right/left-bottom-front
  crn[1] = IntVector(xFaceCell, hi_y,    hi_z);     // right/left-top-front   
  crn[2] = IntVector(xFaceCell, hi_y,    low.z());  // right/left-top-back
  crn[3] = IntVector(xFaceCell, low.y(), low.z());  // right/left-bottom-back 

  for( int corner = 0; corner < 4; corner ++ ) {
     IntVector c = crn[corner];
     double vel_sqr = vel[c].length2();

     term1 = 0.5 * (d[1][c].x() + d[1][c].y() + d[1][c].z()) * vel_sqr;
     term2 =       (d[2][c].x() + d[2][c].y() + d[2][c].z())/(gamma - 1.0) 
           +  rho_tmp[c] * vel[c].x() * (d[3][c].x() + d[4][c].y() + d[4][c].z());
     term3 =  rho_tmp[c] * vel[c].y() * (d[4][c].x() + d[3][c].y() + d[5][c].z()) 
           +  rho_tmp[c] * vel[c].z() * (d[5][c].x() + d[5][c].y() + d[3][c].z());

     double e_tmp = e[c] - delT * ( term1 + term2 + term3);

     temp_CC[c] = e_tmp/rho_CC[c]/cv - 0.5 * vel_sqr/cv;
  }
} //end of function xFaceTempLODI() 


/*_________________________________________________________________
 Function~ yFaceTempLODI--
 Purpose~  Compute temperature in boundary cells on y_plus/minus face
___________________________________________________________________*/
void yFaceTempLODI(const Patch::FaceType face,
             CCVariable<double>& temp_CC, 
             StaticArray<CCVariable<Vector> >& d,
             const CCVariable<double>& e,
             const CCVariable<double>& rho_CC,
             const CCVariable<Vector>& nu,
             const CCVariable<double>& rho_tmp,
             const CCVariable<double>& p,
             const CCVariable<Vector>& vel,
             const double delT,
             const double cv,
             const double gamma, 
             const Vector& dx)
{
  cout_doing << " I am in yFaceTempLODI on face " <<face<< endl;    
  IntVector low,hi;
  int yFaceCell;
  low = vel.getLowIndex();
  hi  = vel.getHighIndex();
  int hi_x = hi.x() - 1;
  int hi_y = hi.y() - 1;
  int hi_z = hi.z() - 1;
  double term1,term2,term3,term4;
  double x_conv, z_conv;
  double qConFrt,qConLast,qFrt,qMid,qLast;
  
  if(face == Patch::yplus ){
    yFaceCell = hi_y;  
  } else {
    yFaceCell = low.y();
  } 
           
  for(int i = low.x()+1; i < hi_x; i++) {
    for(int k = low.z()+1; k < hi_z; k++) {
      IntVector r  (i+1,  yFaceCell,   k);
      IntVector l  (i-1,  yFaceCell,   k);
      IntVector c  (i,    yFaceCell,   k);
      IntVector f  (i,    yFaceCell,   k+1);
      IntVector bk (i,    yFaceCell,   k-1);       
                     
      //_______________________________________________________________
      // energy conservation law, computing pressure, or temperature
      // Please remember e is the total energy per unit volume, 
      // not per unit mass, must be given or compute. The ICE code compute 
      // just internal energy, not the total energy. Here the code is written 
      // just for ICE materials
          
      qConFrt  = vel[r].x() * (e[r] + p[r]);
      qConLast = vel[l].x() * (e[l] + p[l]);
      qFrt     = e[r];
      qMid     = e[c];
      qLast    = e[l];
      x_conv = computeConvection(nu[r].x(), nu[c].x(), nu[l].x(), 
                                   qFrt, qMid, qLast, qConFrt, 
                                   qConLast, delT, dx.x());

      qConFrt  = vel[f].z()  * (e[f] + p[f]);
      qConLast = vel[bk].z() * (e[bk] + p[bk]);
      qFrt     = e[f];
      qLast    = e[bk];

      z_conv = computeConvection(nu[f].z(), nu[c].z(), nu[bk].z(),
                                 qFrt, qMid, qLast, 
                                 qConFrt, qConLast, delT, dx.z()); 

      double vel_sqr = vel[c].length2();

      term1 = 0.5 * d[1][c].y() * vel_sqr;
      term2 = d[2][c].y()/(gamma - 1.0) + rho_tmp[c] * vel[c].x() * d[4][c].y();
      term3 = rho_tmp[c] * vel[c].y() * d[3][c].y() 
            + rho_tmp[c] * vel[c].z() * d[5][c].y();
      term4 = x_conv + z_conv;
      double e_tmp = e[c] - delT * (term1 + term2 + term3 + term4);

      temp_CC[c] = e_tmp/rho_CC[c]/cv - 0.5 * vel_sqr/cv;
    } // end of i loop
  } //end of k loop
  
  //__________________________________________________________
  //  E D G E    right-top/bottom
  for(int k = low.z()+1; k < hi_z; k++) {
    IntVector c  (hi_x,   yFaceCell,   k);
    IntVector f  (hi_x,   yFaceCell,   k+1);
    IntVector bk (hi_x,   yFaceCell,   k-1);

    qConFrt  = vel[f].z()  * (e[f]  + p[f]);
    qConLast = vel[bk].z() * (e[bk] + p[bk]);
    qFrt     = e[f];
    qMid     = e[c];
    qLast    = e[bk];

    z_conv = computeConvection(nu[f].z(), nu[c].z(), nu[bk].z(),
                               qFrt, qMid, qLast, 
                               qConFrt, qConLast, delT, dx.z());

    double vel_sqr = vel[c].length2();

    term1 = 0.5 * (d[1][c].x() + d[1][c].y()) * vel_sqr;
    term2 =  (d[2][c].x() + d[2][c].y())/(gamma - 1.0) 
          +  rho_tmp[c] * vel[c].x() * (d[3][c].x() + d[4][c].y());
    term3 =  rho_tmp[c] * vel[c].y() * (d[4][c].x() + d[3][c].y()) 
          +  rho_tmp[c] * vel[c].z() * (d[5][c].x() + d[5][c].y());

    double e_tmp = e[c] - delT * ( term1 + term2 + term3 + z_conv);

    temp_CC[c] = e_tmp/rho_CC[c]/cv - 0.5 * vel_sqr/cv;
  }
  //_____________________________________________________
  //    E D G E   left-top/bottom
  for(int k = low.z()+1; k < hi_z; k++) {
    IntVector c  (low.x(),   yFaceCell,   k);
    IntVector f  (low.x(),   yFaceCell,   k+1);
    IntVector bk (low.x(),   yFaceCell,   k-1);

    qConFrt  = vel[f].z() * (e[f]  + p[f]);
    qConLast = vel[bk].z() *(e[bk] + p[bk]);
    qFrt     = e[f];
    qMid     = e[c];
    qLast    = e[bk];

    z_conv = computeConvection(nu[f].z(), nu[c].z(), nu[bk].z(),
                               qFrt, qMid, qLast, 
                               qConFrt, qConLast, delT, dx.z());

    double vel_sqr = vel[c].length2();

    term1 = 0.5 * (d[1][c].x() + d[1][c].y()) * vel_sqr;
    term2 =  (d[2][c].x() + d[2][c].y())/(gamma - 1.0) 
          +  rho_tmp[c] * vel[c].x() * (d[3][c].x() + d[4][c].y());
    term3 =  rho_tmp[c] * vel[c].y() * (d[4][c].x() + d[3][c].y()) 
          +  rho_tmp[c] * vel[c].z() * (d[5][c].x() + d[5][c].y());

    double e_tmp = e[c] - delT * ( term1 + term2 + term3 + z_conv);

    temp_CC[c] = e_tmp/rho_CC[c]/cv - 0.5 * vel_sqr/cv;
  }

  //_______________________________________________________
  //    E D G E   top/bottom-front
  for(int i = low.x()+1; i < hi_x; i++) {
    IntVector r (i+1, yFaceCell,    hi_z);
    IntVector l (i-1, yFaceCell,    hi_z);
    IntVector c (i,   yFaceCell,    hi_z);

    qConFrt  = vel[r].x() * (e[r] + p[r]);
    qConLast = vel[l].x() * (e[l] + p[l]);
    qFrt     = e[r];
    qMid     = e[c];
    qLast    = e[l];

    x_conv = computeConvection(nu[r].x(), nu[c].x(), nu[l].x(),
                               qFrt, qMid, qLast, 
                               qConFrt, qConLast, delT, dx.x());

    double vel_sqr = vel[c].length2();

    term1 = 0.5 * (d[1][c].y() + d[1][c].z()) * vel_sqr;
    term2 =  (d[2][c].y() + d[2][c].z())/(gamma - 1.0) 
          +  rho_tmp[c] * vel[c].x() * (d[4][c].y() + d[4][c].z());
    term3 =  rho_tmp[c] * vel[c].y() * (d[3][c].y() + d[5][c].z()) 
          +  rho_tmp[c] * vel[c].z() * (d[5][c].y() + d[3][c].z());

    double e_tmp = e[c] - delT * ( term1 + term2 + term3 + x_conv);

    temp_CC[c] = e_tmp/rho_CC[c]/cv - 0.5 * vel_sqr/cv;
  } 
  //_________________________________________________
  // E D G E      top/bottom-back
  for(int i = low.x()+1; i < hi_x; i++) {
    IntVector r (i+1, yFaceCell,   low.z());
    IntVector l (i-1, yFaceCell,   low.z());
    IntVector c (i,   yFaceCell,   low.z());
    
    qConFrt  = vel[r].x() * (e[r] + p[r]);
    qConLast = vel[l].x() * (e[l] + p[l]);
    qFrt     = e[r];
    qMid     = e[c];
    qLast    = e[l];

    x_conv = computeConvection(nu[r].x(), nu[c].x(), nu[l].x(),
                               qFrt, qMid, qLast, 
                               qConFrt, qConLast, delT, dx.x());

    double vel_sqr = vel[c].length2();

    term1 = 0.5 * (d[1][c].y() + d[1][c].z()) * vel_sqr;
    term2 =  (d[2][c].y() + d[2][c].z())/(gamma - 1.0) 
          +  rho_tmp[c] * vel[c].x() * (d[4][c].y() + d[4][c].z());
    term3 =  rho_tmp[c] * vel[c].y() * (d[3][c].y() + d[5][c].z()) 
          +  rho_tmp[c] * vel[c].z() * (d[5][c].y() + d[3][c].z());

    double e_tmp = e[c] - delT * ( term1 + term2 + term3 + x_conv);

    temp_CC[c] = e_tmp/rho_CC[c]/cv - 0.5 * vel_sqr/cv;        
  }
 
  //________________________________________________________
  // C O R N E R S    
  vector<IntVector> crn(4);
  crn[0] = IntVector(hi_x,    yFaceCell, hi_z);     // right-top/bot-front
  crn[1] = IntVector(low.x(), yFaceCell, hi_z);     // left-top/bot-front   
  crn[2] = IntVector(low.x(), yFaceCell, low.z());  // left-top/bot-back
  crn[3] = IntVector(hi_x,    yFaceCell, low.z());  // right-top/bot-back 

  for( int corner = 0; corner < 4; corner ++ ) {
     IntVector c = crn[corner];
     double vel_sqr = vel[c].length2();

     term1 = 0.5 * (d[1][c].x() + d[1][c].y() + d[1][c].z()) * vel_sqr;
     term2 =       (d[2][c].x() + d[2][c].y() + d[2][c].z())/(gamma - 1.0) 
           +  rho_tmp[c] * vel[c].x() * (d[3][c].x() + d[4][c].y() + d[4][c].z());
     term3 =  rho_tmp[c] * vel[c].y() * (d[4][c].x() + d[3][c].y() + d[5][c].z()) 
           +  rho_tmp[c] * vel[c].z() * (d[5][c].x() + d[5][c].y() + d[3][c].z());

     double e_tmp = e[c] - delT * ( term1 + term2 + term3);

     temp_CC[c] = e_tmp/rho_CC[c]/cv - 0.5 * vel_sqr/cv;
  }             
}

/*_________________________________________________________________
 Function~ zFaceTempLODI--
 Purpose~  Compute temperature boundary condition on the z_plus/minus faces
___________________________________________________________________*/
void zFaceTempLODI(const Patch::FaceType face,
             CCVariable<double>& temp_CC, 
             StaticArray<CCVariable<Vector> >& d,
             const CCVariable<double>& e,
             const CCVariable<double>& rho_CC,
             const CCVariable<Vector>& nu,
             const CCVariable<double>& rho_tmp,
             const CCVariable<double>& p,
             const CCVariable<Vector>& vel,
             const double delT,
             const double cv,
             const double gamma, 
             const Vector& dx)

{
  cout_doing << " I am in zFaceTempLODI on face " <<face<< endl;    
  IntVector low,hi;
  int zFaceCell;
  low = vel.getLowIndex();
  hi  = vel.getHighIndex();
  int hi_x = hi.x() - 1;
  int hi_y = hi.y() - 1;
  int hi_z = hi.z() - 1;
  double term1,term2,term3,term4;
  double x_conv, y_conv;
  double qConFrt,qConLast,qFrt,qMid,qLast;
  if(face == Patch::zplus ){
    zFaceCell = hi_z;  
  } else {
    zFaceCell = low.z();
  } 
  
  for(int i = low.x()+1; i < hi_x; i++) {
    for(int j = low.y()+1; j < hi_y; j++) {
      IntVector r (i+1, j,   zFaceCell);
      IntVector l (i-1, j,   zFaceCell);
      IntVector t (i,   j+1, zFaceCell);
      IntVector b (i,   j-1, zFaceCell);
      IntVector c (i,   j,   zFaceCell);

      //_______________________________________________________________
      // energy conservation law, computing pressure, or temperature
      // Please remember e is the total energy per unit volume, 
      // not per unit mass, must be given or compute. The ICE code compute 
      // just internal energy, not the total energy. Here the code is written 
      // just for ICE materials

      qConFrt  = vel[r].x() * (e[r] + p[r]);
      qConLast = vel[l].x() * (e[l] + p[l]);
      qFrt     = e[r];
      qMid     = e[c];
      qLast    = e[l];
      x_conv = computeConvection(nu[r].x(), nu[c].x(), nu[l].x(), 
                                 qFrt, qMid, qLast, qConFrt, 
                                 qConLast, delT, dx.x());

      qConFrt  = vel[t].y() * (e[t] + p[t]);
      qConLast = vel[b].y() * (e[b] + p[b]);
      qFrt     = e[t];
      qLast    = e[b];

      y_conv = computeConvection(nu[t].y(), nu[c].y(), nu[b].y(),
                                 qFrt, qMid, qLast, 
                                 qConFrt, qConLast, delT, dx.y()); 

      double vel_sqr = vel[c].length2();

      term1 = 0.5 * d[1][c].z() * vel_sqr;
      term2 = d[2][c].z() / (gamma - 1.0) 
            + rho_tmp[c] * vel[c].x() * d[4][c].z();
      term3 = rho_tmp[c] * vel[c].y() * d[5][c].z()  
            + rho_tmp[c] * vel[c].z() * d[3][c].z();
      term4 = x_conv + y_conv;
      double e_tmp = e[c] - delT * (term1 + term2 + term3 + term4);

      temp_CC[c] = e_tmp/rho_CC[c]/cv - 0.5 * vel_sqr/cv;
    } // end of j loop
  } //end of i loop
     
  //__________________________________________________________
  //  E D G E      right-front/back
  for(int j = low.y()+1; j < hi_y; j++) {
    IntVector c (hi_x, j,   zFaceCell);
    IntVector t (hi_x, j+1, zFaceCell);
    IntVector b (hi_x, j-1, zFaceCell);

    qConFrt  = vel[t].y() * (e[t] + p[t]);
    qConLast = vel[b].y() * (e[b] + p[b]);
    qFrt     = e[t];
    qMid     = e[c];
    qLast    = e[b];

    y_conv = computeConvection(nu[t].y(), nu[c].y(), nu[b].y(),
                     qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.y());

    double vel_sqr = vel[c].length2();

    term1 = 0.5 * (d[1][c].x() + d[1][c].z()) * vel_sqr;
    term2 =       (d[2][c].x() + d[2][c].z())/(gamma - 1.0) 
          +  rho_tmp[c] * vel[c].x() * (d[3][c].x() + d[4][c].z());
    term3 =  rho_tmp[c] * vel[c].y() * (d[4][c].x() + d[5][c].z()) 
          +  rho_tmp[c] * vel[c].z() * (d[5][c].x() + d[3][c].z());

    double e_tmp = e[c] - delT * ( term1 + term2 + term3 + y_conv);

    temp_CC[c] = e_tmp/rho_CC[c]/cv - 0.5 * vel_sqr/cv;
  } 
  //__________________________________________________________
  //  E D G E    left-front/back
  for(int j = low.y()+1; j < hi_y; j++) {
    IntVector c (low.x(), j,   zFaceCell);
    IntVector t (low.x(), j+1, zFaceCell);
    IntVector b (low.x(), j-1, zFaceCell);

    qConFrt  = vel[t].y() * (e[t] + p[t]);
    qConLast = vel[b].y() * (e[b] + p[b]);
    qFrt     = e[t];
    qMid     = e[c];
    qLast    = e[b];

    y_conv = computeConvection(nu[t].y(), nu[c].y(), nu[b].y(),
                     qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.y());

    double vel_sqr = vel[c].length2();

    term1 = 0.5 * (d[1][c].x() + d[1][c].z()) * vel_sqr;
    term2 =       (d[2][c].x() + d[2][c].z())/(gamma - 1.0) 
          +  rho_tmp[c] * vel[c].x() * (d[3][c].x() + d[4][c].z());
    term3 =  rho_tmp[c] * vel[c].y() * (d[4][c].x() + d[5][c].z()) 
          +  rho_tmp[c] * vel[c].z() * (d[5][c].x() + d[3][c].z());

    double e_tmp = e[c] - delT * ( term1 + term2 + term3 + y_conv);

    temp_CC[c] = e_tmp/rho_CC[c]/cv - 0.5 * vel_sqr/cv;
  }
  //_______________________________________________________
  //    E D G E     top-front/back
  for(int i = low.x()+1; i < hi_x; i++) {
    IntVector r (i+1, hi_y,   zFaceCell);
    IntVector l (i-1, hi_y,   zFaceCell);
    IntVector c (i,   hi_y,   zFaceCell);

    qConFrt  = vel[r].x() * (e[r] + p[r]);
    qConLast = vel[l].x() * (e[l] + p[l]);
    qFrt     = e[r];
    qMid     = e[c];
    qLast    = e[l];

    x_conv = computeConvection(nu[r].x(), nu[c].x(), nu[l].x(),
                     qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.x());

    double vel_sqr = vel[c].length2();

    term1 = 0.5 * (d[1][c].y() + d[1][c].z()) * vel_sqr;
    term2 =       (d[2][c].y() + d[2][c].z())/(gamma - 1.0) 
          +  rho_tmp[c] * vel[c].x() * (d[4][c].y() + d[4][c].z());
    term3 =  rho_tmp[c] * vel[c].y() * (d[3][c].y() + d[5][c].z()) 
          +  rho_tmp[c] * vel[c].z() * (d[5][c].y() + d[3][c].z());

    double e_tmp = e[c] - delT * ( term1 + term2 + term3 + x_conv);

    temp_CC[c] = e_tmp/rho_CC[c]/cv - 0.5 * vel_sqr/cv;
  }  
  //_______________________________________________________
  //    E D G E     bottom-front/back
  for(int i = low.x()+1; i < hi_x; i++) {
    IntVector r (i+1, low.y(), zFaceCell);
    IntVector l (i-1, low.y(), zFaceCell);
    IntVector c (i,   low.y(), zFaceCell);

    qConFrt  = vel[r].x() * (e[r] + p[r]);       
    qConLast = vel[l].x() * (e[l] + p[l]);
    qFrt     = e[r];
    qMid     = e[c];
    qLast    = e[l];

    x_conv = computeConvection(nu[r].x(), nu[c].x(), nu[l].x(),
                     qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.x());

    double vel_sqr = vel[c].length2();

    term1 = 0.5 * (d[1][c].y() + d[1][c].z()) * vel_sqr;
    term2 =       (d[2][c].y() + d[2][c].z())/(gamma - 1.0) 
          +  rho_tmp[c] * vel[c].x() * (d[4][c].y() + d[4][c].z());
    term3 =  rho_tmp[c] * vel[c].y() * (d[3][c].y() + d[5][c].z()) 
          +  rho_tmp[c] * vel[c].z() * (d[5][c].y() + d[3][c].z());

    double e_tmp = e[c] - delT * ( term1 + term2 + term3 + x_conv);

    temp_CC[c] = e_tmp/rho_CC[c]/cv - 0.5 * vel_sqr/cv;
  }

  //________________________________________________________
  // C O R N E R S    
  vector<IntVector> crn(4);
  crn[0] = IntVector(hi_x,    low.y(), zFaceCell);  // right-bottom-front/back
  crn[1] = IntVector(hi_x,    hi_y,    zFaceCell);  // right-top-front/back   
  crn[2] = IntVector(low.x(), hi_y,    zFaceCell);  // left-top-front/back
  crn[3] = IntVector(low.x(), low.y(), zFaceCell);  // left-bottom-front/back 

  for( int corner = 0; corner < 4; corner ++ ) {
    IntVector c = crn[corner];
    double vel_sqr = vel[c].length2();

    term1 = 0.5 * (d[1][c].x() + d[1][c].y() + d[1][c].z()) * vel_sqr;
    term2 =       (d[2][c].x() + d[2][c].y() + d[2][c].z())/(gamma - 1.0) 
          +  rho_tmp[c] * vel[c].x() * (d[3][c].x() + d[4][c].y() + d[4][c].z());
    term3 =  rho_tmp[c] * vel[c].y() * (d[4][c].x() + d[3][c].y() + d[5][c].z()) 
          +  rho_tmp[c] * vel[c].z() * (d[5][c].x() + d[5][c].y() + d[3][c].z());

    double e_tmp = e[c] - delT * ( term1 + term2 + term3);

    temp_CC[c] = e_tmp/rho_CC[c]/cv - 0.5 * vel_sqr/cv;
  }              
} //end of the function zFaceTempLODI() 

/*_________________________________________________________________
 Function~ fillFaceTempLODI--
___________________________________________________________________*/
void fillFaceTempLODI(const Patch* patch,
              CCVariable<double>& temp_CC, 
              StaticArray<CCVariable<Vector> >& di,
              const CCVariable<double>& e,
              const CCVariable<double>& rho_CC,
              const CCVariable<Vector>& nu,
              const CCVariable<double>& rho_tmp,
              const CCVariable<double>& p,
              const CCVariable<Vector>& vel,
              const Patch::FaceType face,
              const double delT,
              const double cv,
              const double gamma, 
              const Vector& dx)

{
  if (face == Patch::xplus || face == Patch::xminus){
    xFaceTempLODI(face, temp_CC, di,
                  e, rho_CC,nu,
                  rho_tmp, p, vel, 
                  delT, cv, gamma, dx);
  }
  if (face == Patch::yplus || face == Patch::yminus){
    yFaceTempLODI(face, temp_CC, di,
                  e, rho_CC, nu, 
                  rho_tmp, p, vel, 
                  delT, cv, gamma, dx);
  }
  if (face == Patch::zplus || face == Patch::zminus){
    zFaceTempLODI(face, temp_CC, di,
                  e, rho_CC,nu, 
                  rho_tmp, p, vel, 
                  delT, cv, gamma, dx);
  }
}
/* --------------------------------------------------------------------- 
 Function~  fillFacePressLODI--
 Purpose~   Back out the pressure from f_theta and P_EOS
---------------------------------------------------------------------  */
void fillFacePress_LODI(const Patch* patch,
                        CCVariable<double>& press_CC,
                        const StaticArray<CCVariable<double> >& rho_micro,
                        const StaticArray<constCCVariable<double> >& Temp_CC,
                        const StaticArray<CCVariable<double> >& f_theta,
                        const int numALLMatls,
                        SimulationStateP& sharedState, 
                        Patch::FaceType face)
{
    cout_doing << " I am in fillFacePress_LODI on face " <<face<< endl;         
    IntVector low,hi;
    low = press_CC.getLowIndex();
    hi = press_CC.getHighIndex();
    StaticArray<double> press_eos(numALLMatls);
    StaticArray<double> cv(numALLMatls);
    StaticArray<double> gamma(numALLMatls);
    double press_ref= sharedState->getRefPress();    
  
    for (int m = 0; m < numALLMatls; m++) {
      Material* matl = sharedState->getMaterial( m );
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      if(ice_matl){       
        cv[m]     = ice_matl->getSpecificHeat();
        gamma[m]  = ice_matl->getGamma();;        
      }
    } 
    
   for(CellIterator iter=patch->getFaceCellIterator(face, "plusEdgeCells"); 
    !iter.done();iter++) {
      IntVector c = *iter;
          
      press_CC[c] = 0.0;
      for (int m = 0; m < numALLMatls; m++) {
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

#endif    // LODI_BCS
}  // using namespace Uintah
