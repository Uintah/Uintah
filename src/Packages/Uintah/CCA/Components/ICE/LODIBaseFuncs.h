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
   return  0.5 * (qConFrt - qConLast)/deltaX - dissipation;
}

/*_________________________________________________________________
 Function~ xFaceDensityLODI--
 Purpose~  Compute density in boundary cells on x_plus face
___________________________________________________________________*/
void xFaceDensityLODI(const Patch::FaceType face,
                CCVariable<double>& rho_CC,
                const CCVariable<double>& d1_x, 
                const CCVariable<double>& d1_y, 
                const CCVariable<double>& d1_z, 
                const CCVariable<double>& nux,
                const CCVariable<double>& nuy,
                const CCVariable<double>& nuz,
                const CCVariable<double>& rho_tmp,
                const CCVariable<Vector>& vel,
                const double delT,
                const Vector& dx)
{
  cout_doing << "I am in xFaceDensityLODI on face " << face<<endl;
  IntVector low,hi;
  int xFaceCell;
  double y_conv, z_conv;
  double qConFrt,qConLast;
  low = vel.getLowIndex();
  hi  = vel.getHighIndex();
  int hi_x = hi.x() - 1;
  int hi_y = hi.y() - 1;
  int hi_z = hi.z() - 1;
  
  if(face == Patch::xplus ){
    xFaceCell = hi_x;  
  } else {
    xFaceCell = low.x();
  }
  for(int j = low.y()+1; j < hi_y; j++) {
    for(int k = low.z()+1; k < hi_z; k++) {
      IntVector c (xFaceCell,  j,   k);
      IntVector t (xFaceCell,  j+1, k);
      IntVector b (xFaceCell,  j-1, k);
      IntVector f (xFaceCell,  j,   k+1);
      IntVector bk(xFaceCell,  j,   k-1);       

      qConFrt  = rho_tmp[t] * vel[t].y();
      qConLast = rho_tmp[b] * vel[b].y();
      y_conv = computeConvection(nuy[t], nuy[c], nuy[b], 
                                 rho_tmp[t], rho_tmp[c], rho_tmp[b], qConFrt, 
                                 qConLast, delT, dx.y());

      qConFrt  = rho_tmp[f] * vel[f].z();
      qConLast = rho_tmp[bk] * vel[bk].z();
      z_conv = computeConvection(nuz[f], nuz[c], nuz[bk],
                                 rho_tmp[f], rho_tmp[c], rho_tmp[bk], 
                                 qConFrt, qConLast, delT, dx.z());

      rho_CC[c] = rho_tmp[c] - delT * (d1_x[c] + y_conv + z_conv);

     // rho_CC[c] = rho_tmp[c] - delT * d1_x[c] ;
    } // end of j loop
  } //end of k loop

  //__________________________________
  //  E D G E    right/left-bottom
  for(int k = low.z()+1; k < hi_z; k++) {
    IntVector c (xFaceCell,  low.y(), k);     
    IntVector f (xFaceCell,  low.y(), k+1);   
    IntVector bk(xFaceCell, low.y(), k-1);    
    
    qConFrt  = rho_tmp[f]  * vel[f].z();
    qConLast = rho_tmp[bk] * vel[bk].z();
    z_conv = computeConvection(nuz[f], nuz[c], nuz[bk],
                               rho_tmp[f], rho_tmp[c], rho_tmp[bk], 
                               qConFrt, qConLast, delT, dx.z());

    rho_CC[c] = rho_tmp[c] - delT * (d1_x[c] + d1_y[c] + z_conv);
  }

  //__________________________________
  //    E D G E   right/left-top
  for(int k = low.z()+1; k < hi_z; k++) {
    IntVector c (xFaceCell,  hi_y, k);     
    IntVector f (xFaceCell,  hi_y, k+1);   
    IntVector bk(xFaceCell, hi_y, k-1);    

    qConFrt  = rho_tmp[f]  * vel[f].z();
    qConLast = rho_tmp[bk] * vel[bk].z();
    z_conv   = computeConvection(nuz[f], nuz[c], nuz[bk],
                              rho_tmp[f], rho_tmp[c], rho_tmp[bk], 
                              qConFrt, qConLast, delT, dx.z());
    rho_CC[c] = rho_tmp[c] - delT * (d1_x[c] + d1_y[c] + z_conv);
  }

  //_____________________________________________________
  //    E D G E   right/left--back
  for(int j = low.y()+1; j < hi_y; j++) {
    IntVector c (xFaceCell,   j,   low.z());  
    IntVector t (xFaceCell,   j+1, low.z());  
    IntVector b (xFaceCell,   j-1, low.z());  
    qConFrt  = rho_tmp[t] * vel[t].y();
    qConLast = rho_tmp[b] * vel[b].y();
    y_conv = computeConvection(nuy[t], nuy[c], nuy[b],
                           rho_tmp[t], rho_tmp[c], rho_tmp[b], 
                           qConFrt, qConLast, delT, dx.y());
    rho_CC[c] = rho_tmp[c] - delT * (d1_x[c] + d1_z[c] + y_conv);
  } 

  //_________________________________________________
  // E D G E    right/left--front
  for(int j = low.y()+1; j < hi_y; j++) {
    IntVector c (xFaceCell,   j,   hi_z);
    IntVector t (xFaceCell,   j+1, hi_z);
    IntVector b (xFaceCell,   j-1, hi_z);

    qConFrt  = rho_tmp[t] * vel[t].y();
    qConLast = rho_tmp[b] * vel[b].y();
    y_conv   = computeConvection(nuy[t], nuy[c], nuy[b],
                              rho_tmp[t], rho_tmp[c], rho_tmp[b], 
                              qConFrt, qConLast, delT, dx.y());
    rho_CC[c] = rho_tmp[c] - delT * (d1_x[c] + d1_z[c] + y_conv);
  } 

  //________________________________________________________
  // C O R N E R S    
  vector<IntVector> crn(4);
  crn[0] = IntVector(xFaceCell, low.y(), hi_z);   // right/left-bottom-front
  crn[1] = IntVector(xFaceCell, hi_y,    hi_z);   // right/left-top-front
  crn[2] = IntVector(xFaceCell, hi_y,    low.z());// right/left-top-back
  crn[3] = IntVector(xFaceCell, low.y(), low.z());// right/left-bottom-back

  for(int corner = 0; corner <4; corner ++ ) {
    IntVector c = crn[corner];
    rho_CC[c] = rho_tmp[c] - delT * (d1_x[c] + d1_y[c] + d1_z[c]);
  }                     
} 

/*_________________________________________________________________
 Function~ yFaceDensityLODI--
 Purpose~  Compute density in boundary cells on y_plus face 
___________________________________________________________________*/
void yFaceDensityLODI(const Patch::FaceType face,
                CCVariable<double>& rho_CC,
                const CCVariable<double>& d1_x, 
                const CCVariable<double>& d1_y, 
                const CCVariable<double>& d1_z, 
                const CCVariable<double>& nux,
                const CCVariable<double>& nuy,
                const CCVariable<double>& nuz,
                const CCVariable<double>& rho_tmp,
                const CCVariable<Vector>& vel,
                const double delT,
                const Vector& dx)

{
  cout_doing << "I am in yFaceDensityLODI on face " <<face << endl;    
  IntVector low,hi;
  int yFaceCell;
  low = vel.getLowIndex();
  hi  = vel.getHighIndex();
  int hi_x = hi.x() - 1;
  int hi_y = hi.y() - 1;
  int hi_z = hi.z() - 1;
  double x_conv, z_conv;
  double qConFrt,qConLast;

  if(face == Patch::yplus ){
    yFaceCell = hi_y;  
  } else {
    yFaceCell = low.y();
  }

  for(int i = low.x()+1; i < hi_x; i++) {
    for(int k = low.z()+1; k < hi_z; k++) {
      IntVector r (i+1,  yFaceCell,   k);
      IntVector l (i-1,  yFaceCell,   k);
      IntVector c (i,    yFaceCell,   k);
      IntVector f (i,    yFaceCell,   k+1);
      IntVector bk(i,    yFaceCell,   k-1);       

      qConFrt  = rho_tmp[r] * vel[r].x();
      qConLast = rho_tmp[l] * vel[l].x();
      x_conv = computeConvection(nux[r], nux[c], nux[l], 
                                 rho_tmp[r], rho_tmp[c], rho_tmp[l], qConFrt, 
                                 qConLast, delT, dx.x());

      qConFrt  = rho_tmp[f] * vel[f].z();
      qConLast = rho_tmp[bk] * vel[bk].z();
      z_conv = computeConvection(nuz[f], nuz[c], nuz[bk],
                                 rho_tmp[f], rho_tmp[c], rho_tmp[bk], 
                                 qConFrt, qConLast, delT, dx.z());

      rho_CC[c] = rho_tmp[c] - delT * (d1_y[c] + x_conv + z_conv);
     // rho_CC[c] = rho_tmp[c] - delT * d1_y[c] ;   
    } 
  } 

  //__________________________________________________________
  //  E D G E    right-top/bottom
  for(int k = low.z()+1; k < hi_z; k++) {
    IntVector c (hi_x,   yFaceCell,   k);
    IntVector f (hi_x,   yFaceCell,   k+1);
    IntVector bk(hi_x,   yFaceCell,   k-1);
    qConFrt  = rho_tmp[f]  * vel[f].z();
    qConLast = rho_tmp[bk] * vel[bk].z();
    z_conv = computeConvection(nuz[f], nuz[c], nuz[bk],
                               rho_tmp[f], rho_tmp[c], rho_tmp[bk], 
                               qConFrt, qConLast, delT, dx.z());

    rho_CC[c] = rho_tmp[c] - delT * (d1_x[c] + d1_y[c] + z_conv);
  }
  //_______________________________________________________
  //    E D G E    top/bottom-front
  for(int i = low.x()+1; i < hi_x; i++) {
    IntVector r (i+1, yFaceCell,  hi_z);
    IntVector l (i-1, yFaceCell,  hi_z);
    IntVector c (i,   yFaceCell,  hi_z);
    qConFrt  = rho_tmp[r] * vel[r].x();
    qConLast = rho_tmp[l] * vel[l].x();
    x_conv   = computeConvection(nux[r], nux[c], nux[l],
                              rho_tmp[r], rho_tmp[c], rho_tmp[l], 
                              qConFrt, qConLast, delT, dx.x());
    rho_CC[c] = rho_tmp[c] - delT * (d1_y[c] + d1_z[c] + x_conv);
  }
         
  //_____________________________________________________
  //    E D G E    left-top/bottom
  for(int k = low.z()+1; k < hi_z; k++) {
     IntVector c  (low.x(),   yFaceCell,   k);
     IntVector f  (low.x(),   yFaceCell,   k+1);
     IntVector bk (low.x(),   yFaceCell,   k-1);
     qConFrt  = rho_tmp[f] * vel[f].z();
     qConLast = rho_tmp[bk] * vel[bk].z();
     z_conv   = computeConvection(nuz[f], nuz[c], nuz[bk],
                               rho_tmp[f], rho_tmp[c], rho_tmp[bk], 
                               qConFrt, qConLast, delT, dx.z());
     rho_CC[c] = rho_tmp[c] - delT * (d1_x[c] + d1_y[c] + z_conv);
   }

  //_________________________________________________
  // E D G E    top/bottom-back
  for(int i = low.x()+1; i < hi_x; i++) {
    IntVector r (i+1, yFaceCell,   low.z());
    IntVector l (i-1, yFaceCell,   low.z());
    IntVector c (i,   yFaceCell,   low.z());

    qConFrt  = rho_tmp[r] * vel[r].x();
    qConLast = rho_tmp[l] * vel[l].x();
    x_conv   = computeConvection(nux[r], nux[c], nux[l],
                              rho_tmp[r], rho_tmp[c], rho_tmp[l], 
                              qConFrt, qConLast, delT, dx.x());
    rho_CC[c] = rho_tmp[c] - delT * (d1_y[c] + d1_z[c] + x_conv);
  }
  //________________________________________________________
  // C O R N E R S    
  vector<IntVector> crn(4);
  crn[0] = IntVector(hi_x,    yFaceCell, hi_z);     // right-top/bottom-front
  crn[1] = IntVector(low.x(), yFaceCell, hi_z);     // left-top/bottom-front
  crn[2] = IntVector(low.x(), yFaceCell, low.z());  // left-top/bottom-back
  crn[3] = IntVector(hi_x,    yFaceCell, low.z());  // right-top/bottom-back

  for( int corner = 0; corner <4; corner ++ ) {
    IntVector c = crn[corner];
    rho_CC[c] = rho_tmp[c] - delT * (d1_x[c] + d1_y[c] + d1_z[c]);
  }
}
/*_________________________________________________________________
 Function~ zFaceDensityLODI--
 Purpose~  Compute density in boundary cells on the z_plus face
___________________________________________________________________*/
void zFaceDensityLODI( const Patch::FaceType face,
                CCVariable<double>& rho_CC,
                const CCVariable<double>& d1_x, 
                const CCVariable<double>& d1_y, 
                const CCVariable<double>& d1_z, 
                const CCVariable<double>& nux,
                const CCVariable<double>& nuy,
                const CCVariable<double>& nuz,
                const CCVariable<double>& rho_tmp,
                const CCVariable<Vector>& vel,
                const double delT,
                const Vector& dx)

{
  cout << "I am in zFaceDensityLODI on face " <<face<< endl;    
  IntVector low,hi;
  int zFaceCell;
  low = vel.getLowIndex();
  hi  = vel.getHighIndex();
  int hi_x = hi.x() - 1;
  int hi_y = hi.y() - 1;
  int hi_z = hi.z() - 1;

  double x_conv, y_conv;
  double qConFrt,qConLast;
  
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

      qConFrt  = rho_tmp[r] * vel[r].x();
      qConLast = rho_tmp[l] * vel[l].x();
      x_conv = computeConvection(nux[r], nux[c], nux[l], 
                                 rho_tmp[r], rho_tmp[c], rho_tmp[l], qConFrt, 
                                 qConLast, delT, dx.x());

      qConFrt  = rho_tmp[t] * vel[t].y();
      qConLast = rho_tmp[b] * vel[b].y();
      y_conv = computeConvection(nuy[t], nuy[c], nuy[b],
                                 rho_tmp[t], rho_tmp[c], rho_tmp[b], 
                                 qConFrt, qConLast, delT, dx.y());

      rho_CC[c] = rho_tmp[c] - delT * (d1_z[c] + x_conv + y_conv);
     // rho_CC[c] = rho_tmp[c] - delT * d1_x[c] ; 
     } // end of j loop
  } //end of i loop
     
  //__________________________________________________________
  //  E D G E      right-front/back
  for(int j = low.y()+1; j < hi_y; j++) {
    IntVector c (hi_x,   j,   zFaceCell);
    IntVector t (hi_x,   j+1, zFaceCell);
    IntVector b (hi_x,   j-1, zFaceCell);
    qConFrt  = rho_tmp[t] * vel[t].y();
    qConLast = rho_tmp[b] * vel[b].y();
    y_conv = computeConvection(nuy[t], nuy[c], nuy[b],
                               rho_tmp[t], rho_tmp[c], rho_tmp[b], 
                               qConFrt, qConLast, delT, dx.y());

    rho_CC[c] = rho_tmp[c] - delT * (d1_x[c] + d1_z[c] + y_conv);
  } 
  //__________________________________________________________
  //  E D G E      left-front/back
  for(int j = low.y()+1; j < hi_y; j++) {
    IntVector c (low.x(),   j,   zFaceCell);
    IntVector t (low.x(),   j+1, zFaceCell);
    IntVector b (low.x(),   j-1, zFaceCell);
    qConFrt  = rho_tmp[t] * vel[t].y();
    qConLast = rho_tmp[b] * vel[b].y();
    y_conv = computeConvection(nuy[t], nuy[c], nuy[b],
                               rho_tmp[t], rho_tmp[c], rho_tmp[b], 
                               qConFrt, qConLast, delT, dx.y());

    rho_CC[c] = rho_tmp[c] - delT * (d1_x[c] + d1_z[c] + y_conv);

  } //end of j loop

  //_______________________________________________________
  //    E D G E   top-front/back
  for(int i = low.x()+1; i < hi_x; i++) {
    IntVector r (i+1, hi_y,   zFaceCell);
    IntVector l (i-1, hi_y,   zFaceCell);
    IntVector c (i,   hi_y,   zFaceCell);
    qConFrt  = rho_tmp[r] * vel[r].x();
    qConLast = rho_tmp[l] * vel[l].x();
    x_conv = computeConvection(nux[r], nux[c], nux[l],
                           rho_tmp[r], rho_tmp[c], rho_tmp[l], 
                            qConFrt, qConLast, delT, dx.x());
    rho_CC[c] = rho_tmp[c] - delT * (d1_y[c] + d1_z[c] + x_conv);
  }   
  //_______________________________________________________
  //    E D G E    bottom-front/back
  for(int i = low.x()+1; i < hi_x; i++) {
    IntVector r (i+1, low.y(), zFaceCell);
    IntVector l (i-1, low.y(), zFaceCell);
    IntVector b (i,   low.y(), zFaceCell);

    qConFrt  = rho_tmp[r] * vel[r].x();
    qConLast = rho_tmp[l] * vel[l].x();
    x_conv = computeConvection(nux[r], nux[b], nux[l],
                            rho_tmp[r], rho_tmp[b], rho_tmp[l], 
                            qConFrt, qConLast, delT, dx.x());
     rho_CC[b] = rho_tmp[b] - delT * (d1_y[b] + d1_z[b] + x_conv);
   } 

   //________________________________________________________
   // C O R N E R S    
   vector<IntVector> crn(4);
   crn[0] = IntVector(hi_x,    low.y(), zFaceCell); // right-bottom-front/back
   crn[1] = IntVector(hi_x,    hi_y,    zFaceCell); // right-top-front/back
   crn[2] = IntVector(low.x(), hi_y,    zFaceCell); // left-top-front/back 
   crn[3] = IntVector(low.x(), low.y(), zFaceCell); // left-bottom-front/back

   for( int corner = 0; corner < 4; corner ++ ) {
     IntVector c = crn[corner];
     rho_CC[c] = rho_tmp[c] - delT * (d1_x[c] + d1_y[c] + d1_z[c]);
   }                
}
/*_________________________________________________________________
 Function~ fillFaceDensityLODI--
 Purpose~  
___________________________________________________________________*/
void fillFaceDensityLODI(CCVariable<double>& rho_CC,
                   const CCVariable<double>& d1_x, 
                   const CCVariable<double>& d1_y, 
                   const CCVariable<double>& d1_z, 
                   const CCVariable<double>& nux,
                   const CCVariable<double>& nuy,
                   const CCVariable<double>& nuz,
                   const CCVariable<double>& rho_tmp,
                   const CCVariable<Vector>& vel,
                   const Patch::FaceType face,
                   const double delT,
                   const Vector& dx)

{   
  if (face == Patch::xplus || face == Patch::xminus ) {
    xFaceDensityLODI(face,rho_CC,d1_x,d1_y,d1_z,nux,nuy,nuz,rho_tmp,vel,delT,dx);
  } 

  if (face == Patch::yplus || face == Patch::yminus ) {
    yFaceDensityLODI(face,rho_CC,d1_x,d1_y,d1_z,nux,nuy,nuz,rho_tmp,vel,delT,dx);
  } 

  if (face == Patch::zplus || face == Patch::zminus ) { 
    zFaceDensityLODI(face,rho_CC,d1_x,d1_y,d1_z,nux,nuy,nuz,rho_tmp,vel,delT,dx);
  } 
}

/*_________________________________________________________________
 Function~ xFaceVelLODI--
 Purpose~  Compute velocity in boundary cells on x_plus face
___________________________________________________________________*/
void xFaceVelLODI(Patch::FaceType face,
            CCVariable<Vector>& vel_CC,
            const CCVariable<double>& d1_x,  
            const CCVariable<double>& d3_x, 
            const CCVariable<double>& d4_x, 
            const CCVariable<double>& d5_x, 
            const CCVariable<double>& d1_y,  
            const CCVariable<double>& d3_y, 
            const CCVariable<double>& d4_y, 
            const CCVariable<double>& d5_y, 
            const CCVariable<double>& d1_z,  
            const CCVariable<double>& d3_z, 
            const CCVariable<double>& d4_z,
            const CCVariable<double>& d5_z,
            const CCVariable<double>& nux,
            const CCVariable<double>& nuy,
            const CCVariable<double>& nuz,
            const CCVariable<double>& rho_tmp,
            const CCVariable<double>& p,
            const CCVariable<Vector>& vel,
            const double delT,
            const Vector& dx)

{
  cout_doing << " I am in xFaceVelLODI on face " << face << endl;
  IntVector low,hi;
  int xFaceCell;
  low = vel.getLowIndex();
  hi  = vel.getHighIndex();
  int hi_x = hi.x() - 1;
  int hi_y = hi.y() - 1;
  int hi_z = hi.z() - 1;

  double y_conv, z_conv;
  double uVel, vVel, wVel;
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

       //__________________________________
       //         X   V E L O C I T Y 
       qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].x();
       qConLast = rho_tmp[b] * vel[b].y() * vel[b].x();
       qFrt     = rho_tmp[t] * vel[t].x();
       qMid     = rho_tmp[c] * vel[c].x();
       qLast    = rho_tmp[b] * vel[b].x();
       y_conv   = computeConvection(nuy[t], nuy[c], nuy[b], 
                                  qFrt, qMid, qLast, qConFrt, 
                                  qConLast, delT, dx.y());

       qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].x();
       qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].x();
       qFrt     = rho_tmp[f]  * vel[f].x();
       qLast    = rho_tmp[bk] * vel[bk].x();

       z_conv   = computeConvection(nuz[f], nuz[c], nuz[bk],
                                    qFrt, qMid, qLast, qConFrt, 
                                    qConLast, delT, dx.z());       
       uVel     = (qMid - delT * (d1_x[c] * vel[c].x() 
                +  rho_tmp[c] * d3_x[c] + y_conv + z_conv ))/rho_tmp[c];
      //__________________________________
      //         Y   V E L O C I T Y     
      qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].y() + p[t];
      qConLast = rho_tmp[b] * vel[b].y() * vel[b].y() + p[b];
      qFrt     = rho_tmp[t] * vel[t].y();
      qMid     = rho_tmp[c] * vel[c].y();
      qLast    = rho_tmp[b] * vel[b].y();
      y_conv   = computeConvection(nuy[t], nuy[c], nuy[b], 
                                   qFrt, qMid, qLast, qConFrt, 
                                   qConLast, delT, dx.y());

      qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].y();
      qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].y();
      qFrt     = rho_tmp[f]  * vel[f].y();
      qLast    = rho_tmp[bk] * vel[bk].y();

      z_conv   = computeConvection(nuz[f], nuz[c], nuz[bk],
                                 qFrt, qMid, qLast, qConFrt, 
                             qConLast, delT, dx.z());        

      vVel = (qMid- delT * (d1_x[c] * vel[c].y()
           +  rho_tmp[c] * d4_x[c] + y_conv + z_conv))/rho_tmp[c];
      //__________________________________
      //         Z   V E L O C I T Y               
      qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].z();
      qConLast = rho_tmp[b] * vel[b].y() * vel[b].z();
      qFrt     = rho_tmp[t] * vel[t].z();
      qMid     = rho_tmp[c] * vel[c].z();
      qLast    = rho_tmp[b] * vel[b].z();
      y_conv   = computeConvection(nuy[t], nuy[c], nuy[b], 
                                 qFrt, qMid, qLast, qConFrt, 
                                 qConLast, delT, dx.y());

      qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].z()  + p[f];
      qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].z() + p[bk];
      qFrt     = rho_tmp[f]  * vel[f].z();
      qLast    = rho_tmp[bk] * vel[bk].z();

      z_conv = computeConvection(nuz[f], nuz[c], nuz[bk],
                                 qFrt, qMid, qLast, qConFrt, 
                             qConLast, delT, dx.z());        


      wVel = (qMid - delT * (d1_x[c] * vel[c].z() + rho_tmp[c] * d5_x[c]
                   + y_conv + z_conv))/rho_tmp[c];

      vel_CC[c] = Vector(uVel, vVel, wVel);
       
    } // end of j loop
  } //end of k loop

  //__________________________________________________________
  //  E D G E        right/left-bottom
  for(int k = low.z()+1; k < hi_z; k++) {
    IntVector c  (xFaceCell, low.y(), k);
    IntVector f  (xFaceCell, low.y(), k+1);
    IntVector bk (xFaceCell, low.y(), k-1);

    //__________________________________
    //         X   V E L O C I T Y                          
    qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].x();
    qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].x();
    qFrt     = rho_tmp[f]  * vel[f].x();
    qMid     = rho_tmp[c]  * vel[c].x();
    qLast    = rho_tmp[bk] * vel[bk].x();

    z_conv = computeConvection(nuz[f], nuz[c], nuz[bk],
                               qFrt, qMid, qLast, 
                               qConFrt,qConLast, delT, dx.z());       

    uVel = (qMid - delT * ((d1_x[c] + d1_y[c]) * vel[c].x() + 
                        +  (d3_x[c] + d4_y[c]) * rho_tmp[c] 
                        +   z_conv))/rho_tmp[c];
    //__________________________________
    //         Y   V E L O C I T Y
    qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].y();
    qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].y();
    qFrt     = rho_tmp[f]  * vel[f].y();
    qMid     = rho_tmp[c]  * vel[c].y();
    qLast    = rho_tmp[bk] * vel[bk].y();

    z_conv = computeConvection(nuz[f], nuz[c], nuz[bk],
                 qFrt, qMid, qLast, qConFrt,qConLast, delT, dx.z()); 

    vVel = (qMid - delT * ((d1_x[c] + d1_y[c]) * vel[c].y() 
                        +  (d4_x[c] + d3_y[c]) * rho_tmp[c] + z_conv))
                            /rho_tmp[c];

    //__________________________________
    //         Z   V E L O C I T Y
    qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].z()  + p[f];
    qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].z() + p[bk];
    qFrt     = rho_tmp[f]  * vel[f].z();
    qMid     = rho_tmp[c]  * vel[c].z();
    qLast    = rho_tmp[bk] * vel[bk].z();

    z_conv = computeConvection(nuz[f], nuz[c], nuz[bk],
                    qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.z());

    wVel = (qMid - delT * ((d1_x[c] + d1_y[c]) * vel[c].z() 
                        +  (d5_x[c] + d5_y[c]) * rho_tmp[c]  
                        + z_conv))/rho_tmp[c];
    vel_CC[c] = Vector(uVel, vVel, wVel);

  } //end of k loop

  //_______________________________________________________
  //    E D G E     right/left-top
  for(int k = low.z()+1; k < hi_z; k++) {
    IntVector c  (xFaceCell,   hi_y,   k);
    IntVector f  (xFaceCell,   hi_y,   k+1);
    IntVector bk (xFaceCell,   hi_y,   k-1);

    //__________________________________
    //         X   V E L O C I T Y                         
    qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].x();
    qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].x();
    qFrt     = rho_tmp[f]  * vel[f].x();
    qMid     = rho_tmp[c]  * vel[c].x();
    qLast    = rho_tmp[bk] * vel[bk].x();

    z_conv = computeConvection(nuz[f], nuz[c], nuz[bk],
                               qFrt, qMid, qLast, 
                           qConFrt, qConLast, delT, dx.z());       

    uVel = (qMid - delT * ((d1_x[c] + d1_y[c]) * vel[c].x() + 
                        +  (d3_x[c] + d4_y[c]) * rho_tmp[c] 
                        +   z_conv))/rho_tmp[c];
      if (c == IntVector(-1,9,9)){
        cout << " 3 " << uVel << endl;
      }
    //__________________________________
    //         Y   V E L O C I T Y            
    qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].y();
    qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].y();
    qFrt     = rho_tmp[f]  * vel[f].y();
    qMid     = rho_tmp[c]  * vel[c].y();
    qLast    = rho_tmp[bk] * vel[bk].y();

    z_conv = computeConvection(nuz[f], nuz[c], nuz[bk],
                   qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.z()); 

    vVel = (qMid - delT * ((d1_x[c] + d1_y[c]) * vel[c].y() 
                        +  (d4_x[c] + d3_y[c]) * rho_tmp[c] + z_conv))
                             /rho_tmp[c];

    //__________________________________
    //         Z   V E L O C I T Y
    qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].z()  + p[f];
    qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].z() + p[bk];
    qFrt     = rho_tmp[f]  * vel[f].z();
    qMid     = rho_tmp[c]  * vel[c].z();
    qLast    = rho_tmp[bk] * vel[bk].z();

    z_conv = computeConvection(nuz[f], nuz[c], nuz[bk],
                    qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.z());

    wVel = (qMid - delT * ((d1_x[c] + d1_y[c]) * vel[c].z() 
                        +  (d5_x[c] + d5_y[c]) * rho_tmp[c]  
                          + z_conv))/rho_tmp[c];
    vel_CC[c] = Vector(uVel, vVel, wVel);
  } //end of k loop
       
  //_____________________________________________________
  //    E D G E     right/left-back
  for(int j = low.y()+1; j < hi_y; j++) {
    IntVector c (xFaceCell,   j,   low.z());
    IntVector t (xFaceCell,   j+1, low.z());
    IntVector b (xFaceCell,   j-1, low.z());

    //__________________________________
    //         X   V E L O C I T Y                         
    qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].x();
    qConLast = rho_tmp[b] * vel[b].y() * vel[b].x();
    qFrt     = rho_tmp[t] * vel[t].x();
    qMid     = rho_tmp[c] * vel[c].x();
    qLast    = rho_tmp[b] * vel[b].x();

    y_conv = computeConvection(nuy[t], nuy[c], nuy[b],
                               qFrt, qMid, qLast, 
                           qConFrt, qConLast, delT, dx.y());       

    uVel = (qMid - delT * ((d1_x[c] + d1_z[c]) * vel[c].x() + 
                        +  (d3_x[c] + d4_z[c]) * rho_tmp[c] 
                        +   y_conv))/rho_tmp[c];
    //__________________________________
    //         Y   V E L O C I T Y ()
    qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].y() + p[t];
    qConLast = rho_tmp[b] * vel[b].y() * vel[b].y() + p[b];
    qFrt     = rho_tmp[t] * vel[t].y();
    qMid     = rho_tmp[c] * vel[c].y();
    qLast    = rho_tmp[b] * vel[b].y();

    y_conv = computeConvection(nuy[t], nuy[c], nuy[b],
                 qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.y()); 

    vVel = (qMid - delT * ((d1_x[c] + d1_z[c]) * vel[c].y() 
                        +  (d4_x[c] + d5_z[c]) * rho_tmp[c] + y_conv))
                             /rho_tmp[c];
    //__________________________________
    //         Z   V E L O C I T Y ()
    qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].z();
    qConLast = rho_tmp[b] * vel[b].y() * vel[b].z();
    qFrt     = rho_tmp[t] * vel[t].z();
    qMid     = rho_tmp[c] * vel[c].z();
    qLast    = rho_tmp[b] * vel[b].z();

    y_conv = computeConvection(nuy[t], nuy[c], nuy[b],
                    qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.y());

    wVel = (qMid - delT * ((d1_x[c] + d1_z[c]) * vel[c].z() 
                        +  (d5_x[c] + d3_z[c]) * rho_tmp[c]  
                        + y_conv))/rho_tmp[c];
    vel_CC[c] = Vector(uVel, vVel, wVel);
  }

  //_________________________________________________
  // E D G E    right/left-front
  for(int j = low.y()+1; j < hi_y; j++) {
    IntVector c (xFaceCell,   j,   hi_z);
    IntVector t (xFaceCell,   j+1, hi_z);
    IntVector b (xFaceCell,   j-1, hi_z);

    //__________________________________
    //         X   V E L O C I T Y ()                          
    qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].x();
    qConLast = rho_tmp[b] * vel[b].y() * vel[b].x();
    qFrt     = rho_tmp[t] * vel[t].x();
    qMid     = rho_tmp[c] * vel[c].x();
    qLast    = rho_tmp[b] * vel[b].x();

    y_conv = computeConvection(nuy[t], nuy[c], nuy[b],
                               qFrt, qMid, qLast, 
                               qConFrt, qConLast, delT, dx.y());       

    uVel = (qMid - delT * ((d1_x[c] + d1_z[c]) * vel[c].x() + 
                        +  (d3_x[c] + d4_z[c]) * rho_tmp[c] 
                        +   y_conv))/rho_tmp[c];
    //__________________________________
    //         Y   V E L O C I T Y ()
    qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].y() + p[t];
    qConLast = rho_tmp[b] * vel[b].y() * vel[b].y() + p[b];
    qFrt     = rho_tmp[t] * vel[t].y();
    qMid     = rho_tmp[c] * vel[c].y();
    qLast    = rho_tmp[b] * vel[b].y();

    y_conv = computeConvection(nuy[t], nuy[c], nuy[b],
                   qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.y()); 

    vVel = (qMid - delT * ((d1_x[c] + d1_z[c]) * vel[c].y() 
                        +  (d4_x[c] + d5_z[c]) * rho_tmp[c] + y_conv))
                             /rho_tmp[c];
    //__________________________________
    //         Z   V E L O C I T Y ()
    qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].z();
    qConLast = rho_tmp[b] * vel[b].y() * vel[b].z();
    qFrt     = rho_tmp[t] * vel[t].z();
    qMid     = rho_tmp[c] * vel[c].z();
    qLast    = rho_tmp[b] * vel[b].z();

    y_conv = computeConvection(nuy[t], nuy[c], nuy[b],
                    qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.y());

    wVel = (qMid - delT * ((d1_x[c] + d1_z[c]) * vel[c].z() 
                        +  (d5_x[c] + d3_z[c]) * rho_tmp[c]  
                        + y_conv))/rho_tmp[c];
    vel_CC[c] = Vector(uVel, vVel, wVel);        
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
     uVel = vel[c].x() - delT * ((d1_x[c] + d1_y[c] + d1_z[c]) * vel[c].x()  
                              +  (d3_x[c] + d4_y[c] + d4_z[c]) * rho_tmp[c]) 
                         / rho_tmp[c];

     vVel = vel[c].y() - delT * ((d1_x[c] + d1_y[c] + d1_z[c]) * vel[c].y() 
                              +  (d4_x[c] + d3_y[c] + d5_z[c]) * rho_tmp[c])
                         / rho_tmp[c];

     wVel = vel[c].z() - delT * ((d1_x[c] + d1_y[c] + d1_z[c]) * vel[c].z() 
                              +  (d5_x[c] + d5_y[c] + d3_z[c]) * rho_tmp[c])
                         / rho_tmp[c];
     vel_CC[c] = Vector(uVel, vVel, wVel);
   }
} //end of the function xFaceVelLODI() 

/*_________________________________________________________________
 Function~ yFaceVelLODI--
 Purpose~  Compute velocity in boundary cells on y_plus face 
___________________________________________________________________*/
void yFaceVelLODI(Patch::FaceType face,
             CCVariable<Vector>& vel_CC,
             const CCVariable<double>& d1_x,  
             const CCVariable<double>& d3_x, 
             const CCVariable<double>& d4_x, 
             const CCVariable<double>& d5_x, 
             const CCVariable<double>& d1_y,  
             const CCVariable<double>& d3_y, 
             const CCVariable<double>& d4_y, 
             const CCVariable<double>& d5_y, 
             const CCVariable<double>& d1_z,  
             const CCVariable<double>& d3_z, 
             const CCVariable<double>& d4_z,
             const CCVariable<double>& d5_z,
             const CCVariable<double>& nux,
             const CCVariable<double>& nuy,
             const CCVariable<double>& nuz,
             const CCVariable<double>& rho_tmp,
             const CCVariable<double>& p,
             const CCVariable<Vector>& vel,
             const double delT,
             const Vector& dx)
{
  cout_doing << " I am in yFaceVelLODI on face " <<face << endl;    
  IntVector low,hi;
  int yFaceCell;
  low = vel.getLowIndex();
  hi  = vel.getHighIndex();
  int hi_x = hi.x() - 1;
  int hi_y = hi.y() - 1;
  int hi_z = hi.z() - 1;

  double uVel, vVel, wVel;
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

      //__________________________________
      //         X   V E L O C I T Y 
      qConFrt  = rho_tmp[r] * vel[r].x() * vel[r].x() + p[r];
      qConLast = rho_tmp[l] * vel[l].x() * vel[l].x() + p[l];
      qFrt   = rho_tmp[r] * vel[r].x();
      qMid   = rho_tmp[c] * vel[c].x();
      qLast  = rho_tmp[l] * vel[l].x();
      
      x_conv = computeConvection(nux[r], nux[c], nux[l], 
                                 qFrt, qMid, qLast, qConFrt, 
                                 qConLast, delT, dx.x());

      qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].x();
      qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].x();
      qFrt     = rho_tmp[f]  * vel[f].x();
      qLast    = rho_tmp[bk] * vel[bk].x();

      z_conv = computeConvection(nuz[f], nuz[c], nuz[bk],
                                 qFrt, qMid, qLast, qConFrt, 
                                 qConLast, delT, dx.z());       
      uVel = (qMid - delT * (d1_y[c] * vel[c].x() 
           +  rho_tmp[c] * d4_y[c] + x_conv + z_conv ))/rho_tmp[c];
      //__________________________________
      //         Y   V E L O C I T Y        
      qConFrt  = rho_tmp[r] * vel[r].x() * vel[r].y();
      qConLast = rho_tmp[l] * vel[l].x() * vel[l].y();
      qFrt   = rho_tmp[r] * vel[r].y();
      qMid   = rho_tmp[c] * vel[c].y();
      qLast  = rho_tmp[l] * vel[l].y();
      x_conv = computeConvection(nux[r], nux[c], nux[l], 
                                 qFrt, qMid, qLast, qConFrt, 
                                 qConLast, delT, dx.x());

      qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].y();
      qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].y();
      qFrt     = rho_tmp[f]  * vel[f].y();
      qLast    = rho_tmp[bk] * vel[bk].y();

      z_conv = computeConvection(nuz[f], nuz[c], nuz[bk],
                                 qFrt, qMid, qLast, qConFrt, 
                                 qConLast, delT, dx.z());        

      vVel = (qMid- delT * (d1_y[c] * vel[c].y()
           +  rho_tmp[c] * d3_y[c] + x_conv + z_conv))/rho_tmp[c];
      //__________________________________
      //         Z   V E L O C I T Y
      qConFrt  = rho_tmp[r] * vel[r].x() * vel[r].z();
      qConLast = rho_tmp[l] * vel[l].x() * vel[l].z();
      qFrt     = rho_tmp[r] * vel[r].z();
      qMid     = rho_tmp[c] * vel[c].z();
      qLast    = rho_tmp[l] * vel[l].z();
      x_conv = computeConvection(nux[r], nux[c], nux[l], 
                                 qFrt, qMid, qLast, qConFrt, 
                                 qConLast, delT, dx.x());

      qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].z()  + p[f];
      qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].z() + p[bk];
      qFrt     = rho_tmp[f]  * vel[f].z();
      qLast    = rho_tmp[bk] * vel[bk].z();

      z_conv  = computeConvection(nuz[f], nuz[c], nuz[bk],
                                  qFrt, qMid, qLast, qConFrt, 
                                  qConLast, delT, dx.z());        

      wVel = (qMid - delT * (d1_y[c] * vel[c].z() 
           + rho_tmp[c] * d5_y[c] + x_conv + z_conv))/rho_tmp[c];

      vel_CC[c] = Vector(uVel, vVel, wVel);
    } // end of i loop
  } //end of k loop

  //__________________________________________________________
  //  E D G E    right-top/bottom
  for(int k = low.z()+1; k < hi_z; k++) {
    IntVector c  (hi_x,   yFaceCell,   k);
    IntVector f  (hi_x,   yFaceCell,   k+1);
    IntVector bk (hi_x,   yFaceCell,   k-1);

    //__________________________________
    //         X   V E L O C I T Y                         
    qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].x();
    qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].x();
    qFrt     = rho_tmp[f]  * vel[f].x();
    qMid     = rho_tmp[c]  * vel[c].x();
    qLast    = rho_tmp[bk] * vel[bk].x();

    z_conv = computeConvection(nuz[f], nuz[c], nuz[bk],
                               qFrt, qMid, qLast, 
                               qConFrt,qConLast, delT, dx.z());       

    uVel = (qMid - delT * ((d1_x[c] + d1_y[c]) * vel[c].x() + 
                        +  (d3_x[c] + d4_y[c]) * rho_tmp[c] 
                        +   z_conv))/rho_tmp[c];

    //__________________________________
    //         Y   V E L O C I T Y 
    qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].y();
    qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].y();
    qFrt     = rho_tmp[f]  * vel[f].y();
    qMid     = rho_tmp[c]  * vel[c].y();
    qLast    = rho_tmp[bk] * vel[bk].y();

    z_conv = computeConvection(nuz[f], nuz[c], nuz[bk],
                               qFrt, qMid, qLast, 
                               qConFrt,qConLast, delT, dx.z()); 

    vVel = (qMid - delT * ((d1_x[c] + d1_y[c]) * vel[c].y() 
                        +  (d4_x[c] + d3_y[c]) * rho_tmp[c] + z_conv))
                            /rho_tmp[c];
    //__________________________________
    //         Z   V E L O C I T Y 
    qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].z()  + p[f];
    qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].z() + p[bk];
    qFrt     = rho_tmp[f]  * vel[f].z();
    qMid     = rho_tmp[c]  * vel[c].z();
    qLast    = rho_tmp[bk] * vel[bk].z();

    z_conv = computeConvection(nuz[f], nuz[c], nuz[bk],
                               qFrt, qMid, qLast, 
                               qConFrt, qConLast, delT, dx.z());

    wVel = (qMid - delT * ((d1_x[c] + d1_y[c]) * vel[c].z() 
                        +  (d5_x[c] + d5_y[c]) * rho_tmp[c]  
                        + z_conv))/rho_tmp[c];
    vel_CC[c] = Vector(uVel, vVel, wVel);
  } //end of k loop
     
         
  //_____________________________________________________
  //    E D G E   left-top/bottom
  for(int k = low.z()+1; k < hi_z; k++) {
    IntVector c  (low.x(),   yFaceCell,   k);
    IntVector f  (low.x(),   yFaceCell,   k+1);
    IntVector bk (low.x(),   yFaceCell,   k-1);

    //__________________________________
    //         X   V E L O C I T Y                          
    qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].x();
    qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].x();
    qFrt     = rho_tmp[f]  * vel[f].x();
    qMid     = rho_tmp[c]  * vel[c].x();
    qLast    = rho_tmp[bk] * vel[bk].x();

    z_conv = computeConvection(nuz[f], nuz[c], nuz[bk],
                               qFrt, qMid, qLast, 
                               qConFrt, qConLast, delT, dx.z());       

    uVel = (qMid - delT * ((d1_x[c] + d1_y[c]) * vel[c].x() + 
                         + (d3_x[c] + d4_y[c]) * rho_tmp[c] 
                         +  z_conv))/rho_tmp[c];
    //__________________________________
    //         Y   V E L O C I T Y 
    qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].y();
    qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].y();
    qFrt     = rho_tmp[f]  * vel[f].y();
    qMid     = rho_tmp[c]  * vel[c].y();
    qLast    = rho_tmp[bk] * vel[bk].y();

    z_conv = computeConvection(nuz[f], nuz[c], nuz[bk],
                                  qFrt, qMid, qLast, 
                                  qConFrt, qConLast, delT, dx.z()); 

    vVel = (qMid - delT * ((d1_x[c] + d1_y[c]) * vel[c].y() 
                        +  (d4_x[c] + d3_y[c]) * rho_tmp[c] + z_conv))
                           /rho_tmp[c];
    //__________________________________
    //         Z   V E L O C I T Y
    qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].z()  + p[f];
    qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].z() + p[bk];
    qFrt     = rho_tmp[f]  * vel[f].z();
    qMid     = rho_tmp[c]  * vel[c].z();
    qLast    = rho_tmp[bk] * vel[bk].z();

    z_conv = computeConvection(nuz[f], nuz[c], nuz[bk],
                               qFrt, qMid, qLast, 
                               qConFrt, qConLast, delT, dx.z());

    wVel = (qMid - delT * ((d1_x[c] + d1_y[c]) * vel[c].z() 
                        +  (d5_x[c] + d5_y[c]) * rho_tmp[c]  
                        + z_conv))/rho_tmp[c];
    vel_CC[c] = Vector(uVel, vVel, wVel);

  } 
  //_______________________________________________________
  //    E D G E     top/bottom-front
  for(int i = low.x()+1; i < hi_x; i++) {
    IntVector r (i+1, yFaceCell,    hi_z);
    IntVector l (i-1, yFaceCell,    hi_z);
    IntVector c (i,   yFaceCell,    hi_z);

    //__________________________________
    //         X   V E L O C I T Y                          
    qConFrt  = rho_tmp[r] * vel[r].x() * vel[r].x() + p[r];
    qConLast = rho_tmp[l] * vel[l].x() * vel[l].x() + p[l];
    qFrt     = rho_tmp[r] * vel[r].x();
    qMid     = rho_tmp[c] * vel[c].x();
    qLast    = rho_tmp[l] * vel[l].x();

    x_conv = computeConvection(nux[r], nux[c], nux[l],
                               qFrt, qMid, qLast, 
                               qConFrt, qConLast, delT, dx.x());       

    uVel = (qMid - delT * ((d1_y[c] + d1_z[c]) * vel[c].x() + 
                        +  (d4_y[c] + d4_z[c]) * rho_tmp[c] 
                        +   x_conv))/rho_tmp[c];
    //__________________________________
    //         Y   V E L O C I T Y 
    qConFrt  = rho_tmp[r] * vel[r].x() * vel[r].y();
    qConLast = rho_tmp[l] * vel[l].x() * vel[l].y();
    qFrt     = rho_tmp[r] * vel[r].y();
    qMid     = rho_tmp[c] * vel[c].y();
    qLast    = rho_tmp[l] * vel[l].y();

    x_conv = computeConvection(nux[r], nux[c], nux[l],
                              qFrt, qMid, qLast, 
                              qConFrt, qConLast, delT, dx.x()); 

    vVel = (qMid - delT * ((d1_y[c] + d1_z[c]) * vel[c].y() 
                        +  (d3_y[c] + d5_z[c]) * rho_tmp[c] + x_conv))
                           /rho_tmp[c];
    //__________________________________
    //         Z   V E L O C I T Y
    qConFrt  = rho_tmp[r] * vel[r].x() * vel[r].z();
    qConLast = rho_tmp[l] * vel[l].x() * vel[l].z();
    qFrt     = rho_tmp[r] * vel[r].z();
    qMid     = rho_tmp[c] * vel[c].z();
    qLast    = rho_tmp[l] * vel[l].z();

    x_conv = computeConvection(nux[r], nux[c], nux[l],
                               qFrt, qMid, qLast, 
                               qConFrt, qConLast, delT, dx.x());

    wVel = (qMid - delT * ((d1_y[c] + d1_z[c]) * vel[c].z() 
                        +  (d5_y[c] + d3_z[c]) * rho_tmp[c]  
                        + x_conv))/rho_tmp[c];
    vel_CC[c] = Vector(uVel, vVel, wVel);
  } //end of i loop
       
  //_________________________________________________
  // E D G E    top/bottom-back
  for(int i = low.x()+1; i < hi_x; i++) {
    IntVector r (i+1, yFaceCell,   low.z());
    IntVector l (i-1, yFaceCell,   low.z());
    IntVector c (i,   yFaceCell,   low.z());

    //__________________________________
    //         X   V E L O C I T Y                          
    qConFrt  = rho_tmp[r] * vel[r].x() * vel[r].x() + p[r];
    qConLast = rho_tmp[l] * vel[l].x() * vel[l].x() + p[l];
    qFrt   = rho_tmp[r] * vel[r].x();
    qMid   = rho_tmp[c] * vel[c].x();
    qLast  = rho_tmp[l] * vel[l].x();

    x_conv = computeConvection(nux[r], nux[c], nux[l],
                               qFrt, qMid, qLast, 
                               qConFrt, qConLast, delT, dx.x());       

    uVel = (qMid - delT * ((d1_y[c] + d1_z[c]) * vel[c].x() + 
                        +  (d4_y[c] + d4_z[c]) * rho_tmp[c] 
                        +   x_conv))/rho_tmp[c];

    //__________________________________
    //         Y   V E L O C I T Y
    qConFrt  = rho_tmp[r] * vel[r].x() * vel[r].y();
    qConLast = rho_tmp[l] * vel[l].x() * vel[l].y();
    qFrt     = rho_tmp[r] * vel[r].y();
    qMid     = rho_tmp[c] * vel[c].y();
    qLast    = rho_tmp[l] * vel[l].y();

    x_conv = computeConvection(nux[r], nux[c], nux[l],
                 qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.x()); 

    vVel = (qMid - delT * ((d1_y[c] + d1_z[c]) * vel[c].y() 
                        +  (d3_y[c] + d5_z[c]) * rho_tmp[c] + x_conv))
                          /rho_tmp[c];

    //__________________________________
    //         Z   V E L O C I T Y
    qConFrt  = rho_tmp[r] * vel[r].x() * vel[r].z();
    qConLast = rho_tmp[l] * vel[l].x() * vel[l].z();
    qFrt     = rho_tmp[r] * vel[r].z();
    qMid     = rho_tmp[c] * vel[c].z();
    qLast    = rho_tmp[l] * vel[l].z();

    x_conv = computeConvection(nux[r], nux[c], nux[l],
                    qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.x());

    wVel = (qMid - delT * ((d1_y[c] + d1_z[c]) * vel[c].z() 
                        +  (d5_y[c] + d3_z[c]) * rho_tmp[c]  
                        + x_conv))/rho_tmp[c];
    vel_CC[c] = Vector(uVel, vVel, wVel);
  } 
  //________________________________________________________
  // C O R N E R S    
  vector<IntVector> crn(4);
  crn[0] = IntVector(hi_x,    yFaceCell, hi_z);     // right-top/bottom-front
  crn[1] = IntVector(low.x(), yFaceCell, hi_z);     // left-top/bottom-front   
  crn[2] = IntVector(low.x(), yFaceCell, low.z());  // left-top/bottom-back  
  crn[3] = IntVector(hi_x,    yFaceCell, low.z());  // right-top/bottom-back 

  for( int corner = 0; corner < 4; corner ++ ) {
    IntVector c = crn[corner];
    uVel = vel[c].x() - delT * ((d1_x[c] + d1_y[c] + d1_z[c]) * vel[c].x()  
                             +  (d3_x[c] + d4_y[c] + d4_z[c]) * rho_tmp[c]) 
                             / rho_tmp[c];

    vVel = vel[c].y() - delT * ((d1_x[c] + d1_y[c] + d1_z[c]) * vel[c].y() 
                             +  (d4_x[c] + d3_y[c] + d5_z[c]) * rho_tmp[c])
                             / rho_tmp[c];

    wVel = vel[c].z() - delT * ((d1_x[c] + d1_y[c] + d1_z[c]) * vel[c].z() 
                             +  (d5_x[c] + d5_y[c] + d3_z[c]) * rho_tmp[c])
                             / rho_tmp[c];
    vel_CC[c] = Vector(uVel, vVel, wVel);
  }                     
} //end of the function yFaceVelLODI() 

/*_________________________________________________________________
 Function~ zFaceVelLODI--
 Purpose~  Compute velocity in boundary cells on the z_plus face
___________________________________________________________________*/
void zFaceVelLODI(const Patch::FaceType face,
             CCVariable<Vector>& vel_CC,
             const CCVariable<double>& d1_x,  
             const CCVariable<double>& d3_x, 
             const CCVariable<double>& d4_x, 
             const CCVariable<double>& d5_x, 
             const CCVariable<double>& d1_y,  
             const CCVariable<double>& d3_y, 
             const CCVariable<double>& d4_y, 
             const CCVariable<double>& d5_y, 
             const CCVariable<double>& d1_z,  
             const CCVariable<double>& d3_z, 
             const CCVariable<double>& d4_z,
             const CCVariable<double>& d5_z,
             const CCVariable<double>& nux,
             const CCVariable<double>& nuy,
             const CCVariable<double>& nuz,
             const CCVariable<double>& rho_tmp,
             const CCVariable<double>& p,
             const CCVariable<Vector>& vel,
             const double delT,
             const Vector& dx)

{
  cout_doing << " I am in zFaceVelLODI on face " <<face << endl;    
  IntVector low,hi;
  int zFaceCell;
  low = vel.getLowIndex();
  hi  = vel.getHighIndex();
  int hi_x = hi.x() - 1;
  int hi_y = hi.y() - 1;
  int hi_z = hi.z() - 1;
  
  double x_conv, y_conv;
  double uVel, vVel, wVel;
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

      //__________________________________
      //         X   V E L O C I T Y
      qConFrt  = rho_tmp[r] * vel[r].x() * vel[r].x() + p[r];
      qConLast = rho_tmp[l] * vel[l].x() * vel[l].x() + p[l];
      qFrt     = rho_tmp[r] * vel[r].x();
      qMid     = rho_tmp[c] * vel[c].x();
      qLast    = rho_tmp[l] * vel[l].x();
      x_conv   = computeConvection(nux[r], nux[c], nux[l], 
                                 qFrt, qMid, qLast, qConFrt, 
                                 qConLast, delT, dx.x());

      qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].x();
      qConLast = rho_tmp[b] * vel[b].y() * vel[b].x();
      qFrt     = rho_tmp[t] * vel[t].x();
      qLast    = rho_tmp[b] * vel[b].x();

      y_conv = computeConvection(nuy[t], nuy[c], nuy[b],
                                 qFrt, qMid, qLast, qConFrt, 
                                 qConLast, delT, dx.y());       
      uVel = (qMid - delT * (d1_z[c] * vel[c].x() 
           +  rho_tmp[c] * d4_z[c] + x_conv + y_conv))/rho_tmp[c];
      //__________________________________
      //         Y   V E L O C I T Y        
      qConFrt  = rho_tmp[r] * vel[r].x() * vel[r].y();
      qConLast = rho_tmp[l] * vel[l].x() * vel[l].y();
      qFrt     = rho_tmp[r] * vel[r].y();
      qMid     = rho_tmp[c] * vel[c].y();
      qLast    = rho_tmp[l] * vel[l].y();

      x_conv = computeConvection(nux[r], nux[c], nux[l],
                                 qFrt, qMid, qLast, qConFrt, 
                                 qConLast, delT, dx.x()); 

      qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].y() + p[t];
      qConLast = rho_tmp[b] * vel[b].y() * vel[b].y() + p[b];
      qFrt     = rho_tmp[t] * vel[t].y();
      qLast    = rho_tmp[b] * vel[b].y();
      y_conv   = computeConvection(nuy[t], nuy[c], nuy[b], 
                                   qFrt, qMid, qLast, qConFrt, 
                                   qConLast, delT, dx.y());

      vVel = (qMid- delT * (d1_z[c] * vel[c].y()
           +  rho_tmp[c] * d5_z[c] + x_conv + y_conv))/rho_tmp[c];
                        
      //__________________________________
      //         Z   V E L O C I T Y
      qConFrt  = rho_tmp[r] * vel[r].x() * vel[r].z();
      qConLast = rho_tmp[l] * vel[l].x() * vel[l].z() ;
      qFrt     = rho_tmp[r] * vel[r].z();
      qMid     = rho_tmp[c] * vel[c].z();
      qLast    = rho_tmp[l] * vel[l].z();
      x_conv   = computeConvection(nux[r], nux[c], nux[l],
                                   qFrt, qMid, qLast, qConFrt, 
                                   qConLast, delT, dx.x());  

      qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].z();
      qConLast = rho_tmp[b] * vel[b].y() * vel[b].z();
      qFrt     = rho_tmp[t] * vel[t].z();
      qLast    = rho_tmp[b] * vel[b].z();
      y_conv   = computeConvection(nuy[t], nuy[c], nuy[b], 
                                   qFrt, qMid, qLast, qConFrt, 
                                   qConLast, delT, dx.y());

      wVel = (qMid - delT * (d1_z[c] * vel[c].z() 
           + rho_tmp[c] * d3_z[c] + x_conv + y_conv))/rho_tmp[c];
           
      vel_CC[c] = Vector(uVel, vVel, wVel);
    } // end of j loop
  } //end of i loop
    
       
  //__________________________________________________________
  //  E D G E    right-front/back
  for(int j = low.y()+1; j < hi_y; j++) {
    IntVector c (hi_x,   j,   zFaceCell);
    IntVector t (hi_x,   j+1, zFaceCell);
    IntVector b (hi_x,   j-1, zFaceCell);

    //__________________________________
    //         X   V E L O C I T Y                         
    qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].x();
    qConLast = rho_tmp[b] * vel[b].y() * vel[b].x();
    qFrt     = rho_tmp[t] * vel[t].x();
    qMid     = rho_tmp[c] * vel[c].x();
    qLast    = rho_tmp[b] * vel[b].x();

    y_conv = computeConvection(nuy[t], nuy[c], nuy[b],
                               qFrt, qMid, qLast, 
                               qConFrt,qConLast, delT, dx.y());       

    uVel = (qMid - delT * ((d1_x[c] + d1_z[c]) * vel[c].x() + 
                        +  (d3_x[c] + d4_z[c]) * rho_tmp[c] 
                        +   y_conv))/rho_tmp[c];
    //__________________________________
    //         Y   V E L O C I T Y 
    qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].y() + p[t];
    qConLast = rho_tmp[b] * vel[b].y() * vel[b].y() + p[b];
    qFrt     = rho_tmp[t] * vel[t].y();
    qMid     = rho_tmp[c] * vel[c].y();
    qLast    = rho_tmp[b] * vel[b].y();

    y_conv = computeConvection(nuy[t], nuy[c], nuy[b],
                               qFrt, qMid, qLast, 
                               qConFrt,qConLast, delT, dx.y()); 

    vVel = (qMid - delT * ((d1_x[c] + d1_z[c]) * vel[c].y() 
                        +  (d4_x[c] + d5_z[c]) * rho_tmp[c] + y_conv))
                        /rho_tmp[c];
    //__________________________________
    //         Z   V E L O C I T Y
    qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].z();
    qConLast = rho_tmp[b] * vel[b].y() * vel[b].z();
    qFrt     = rho_tmp[t] * vel[t].z();
    qMid     = rho_tmp[c] * vel[c].z();
    qLast    = rho_tmp[b] * vel[b].z();

    y_conv = computeConvection(nuy[t], nuy[c], nuy[b],
                               qFrt, qMid, qLast, 
                               qConFrt, qConLast, delT, dx.y());

    wVel = (qMid - delT * ((d1_x[c] + d1_z[c]) * vel[c].z() 
                        +  (d5_x[c] + d3_z[c]) * rho_tmp[c]  
                        + y_conv))/rho_tmp[c];

    vel_CC[c] = Vector(uVel, vVel, wVel);

  } //end of j loop

  //__________________________________________________________
  //  E D G E    left-front/back
  for(int j = low.y()+1; j < hi_y; j++) {
    IntVector c (low.x(), j,   zFaceCell);
    IntVector t (low.x(), j+1, zFaceCell);
    IntVector b (low.x(), j-1, zFaceCell);

    //__________________________________
    //         X   V E L O C I T Y                         
    qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].x();
    qConLast = rho_tmp[b] * vel[b].y() * vel[b].x();
    qFrt     = rho_tmp[t] * vel[t].x();
    qMid     = rho_tmp[c] * vel[c].x();
    qLast    = rho_tmp[b] * vel[b].x();

    y_conv = computeConvection(nuy[t], nuy[c], nuy[b],
                               qFrt, qMid, qLast, 
                               qConFrt,qConLast, delT, dx.y());       

    uVel = (qMid - delT * ((d1_x[c] + d1_z[c]) * vel[c].x() + 
                        +  (d3_x[c] + d4_z[c]) * rho_tmp[c] 
                        +   y_conv))/rho_tmp[c];
    //__________________________________
    //         Y   V E L O C I T Y 
    qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].y() + p[t];
    qConLast = rho_tmp[b] * vel[b].y() * vel[b].y() + p[b];
    qFrt     = rho_tmp[t] * vel[t].y();
    qMid     = rho_tmp[c] * vel[c].y();
    qLast    = rho_tmp[b] * vel[b].y();

    y_conv = computeConvection(nuy[t], nuy[c], nuy[b],
                               qFrt, qMid, qLast, 
                               qConFrt,qConLast, delT, dx.y()); 

    vVel = (qMid - delT * ((d1_x[c] + d1_z[c]) * vel[c].y() 
                        +  (d4_x[c] + d5_z[c]) * rho_tmp[c] + y_conv))
                              /rho_tmp[c];
    //__________________________________
    //         Z   V E L O C I T Y
    qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].z();
    qConLast = rho_tmp[b] * vel[b].y() * vel[b].z();
    qFrt     = rho_tmp[t] * vel[t].z();
    qMid     = rho_tmp[c] * vel[c].z();
    qLast    = rho_tmp[b] * vel[b].z();

    y_conv = computeConvection(nuy[t], nuy[c], nuy[b],
                               qFrt, qMid, qLast, 
                               qConFrt, qConLast, delT, dx.y());

    wVel = (qMid - delT * ((d1_x[c] + d1_z[c]) * vel[c].z() 
                        +  (d5_x[c] + d3_z[c]) * rho_tmp[c]  
                        + y_conv))/rho_tmp[c];

    vel_CC[c] = Vector(uVel, vVel, wVel);
  } 
  //_______________________________________________________
  //    E D G E   top-front/back
  for(int i = low.x()+1; i < hi_x; i++) {
    IntVector r (i+1, hi_y,   zFaceCell);
    IntVector l (i-1, hi_y,   zFaceCell);
    IntVector c (i,   hi_y,   zFaceCell);

    //__________________________________
    //         X   V E L O C I T Y                          
    qConFrt  = rho_tmp[r] * vel[r].x() * vel[r].x() + p[r];
    qConLast = rho_tmp[l] * vel[l].x() * vel[l].x() + p[l];
    qFrt     = rho_tmp[r] * vel[r].x();
    qMid     = rho_tmp[c] * vel[c].x();
    qLast    = rho_tmp[l] * vel[l].x();

    x_conv = computeConvection(nux[r], nux[c], nux[l],
                                 qFrt, qMid, qLast, 
                                 qConFrt, qConLast, delT, dx.x());       

    uVel = (qMid - delT * ((d1_y[c] + d1_z[c]) * vel[c].x() + 
                        +  (d4_y[c] + d4_z[c]) * rho_tmp[c] 
                        +   x_conv))/rho_tmp[c];
    //__________________________________
    //         Y   V E L O C I T Y
    qConFrt  = rho_tmp[r] * vel[r].x() * vel[r].y();
    qConLast = rho_tmp[l] * vel[l].x() * vel[l].y();
    qFrt     = rho_tmp[r] * vel[r].y();
    qMid     = rho_tmp[c] * vel[c].y();
    qLast    = rho_tmp[l] * vel[l].y();

    x_conv = computeConvection(nux[r], nux[c], nux[l],
                               qFrt, qMid, qLast, 
                               qConFrt, qConLast, delT, dx.x()); 

    vVel = (qMid - delT * ((d1_y[c] + d1_z[c]) * vel[c].y() 
                        +  (d3_y[c] + d5_z[c]) * rho_tmp[c] + x_conv))
                               /rho_tmp[c];
    //__________________________________
    //         Z   V E L O C I T Y
    qConFrt  = rho_tmp[r] * vel[r].x() * vel[r].z();
    qConLast = rho_tmp[l] * vel[l].x() * vel[l].z();
    qFrt     = rho_tmp[r] * vel[r].z();
    qMid     = rho_tmp[c] * vel[c].z();
    qLast    = rho_tmp[l] * vel[l].z();

    x_conv = computeConvection(nux[r], nux[c], nux[l],
                               qFrt, qMid, qLast, 
                               qConFrt, qConLast, delT, dx.x());

    wVel = (qMid - delT * ((d1_y[c] + d1_z[c]) * vel[c].z() 
                        +  (d5_y[c] + d3_z[c]) * rho_tmp[c]  
                        + x_conv))/rho_tmp[c];
                        
    vel_CC[c] = Vector(uVel, vVel, wVel);
  } 
         
  //_______________________________________________________
  //    E D G E   bottom-front/back
  for(int i = low.x()+1; i < hi_x; i++) {
    IntVector r (i+1, low.y(), zFaceCell);
    IntVector l (i-1, low.y(), zFaceCell);
    IntVector c (i,   low.y(), zFaceCell);

    //__________________________________
    //         X   V E L O C I T Y                          
    qConFrt  = rho_tmp[r] * vel[r].x() * vel[r].x() + p[r];
    qConLast = rho_tmp[l] * vel[l].x() * vel[l].x() + p[l];
    qFrt     = rho_tmp[r] * vel[r].x();
    qMid     = rho_tmp[c] * vel[c].x();
    qLast    = rho_tmp[l] * vel[l].x();

    x_conv = computeConvection(nux[r], nux[c], nux[l],
                                 qFrt, qMid, qLast, 
                                 qConFrt, qConLast, delT, dx.x());       

    uVel = (qMid - delT * ((d1_y[c] + d1_z[c]) * vel[c].x() + 
                        +  (d4_y[c] + d4_z[c]) * rho_tmp[c] 
                        +   x_conv))/rho_tmp[c];
        
    //__________________________________
    //         Y   V E L O C I T Y
    qConFrt  = rho_tmp[r] * vel[r].x() * vel[r].y();
    qConLast = rho_tmp[l] * vel[l].x() * vel[l].y();
    qFrt     = rho_tmp[r] * vel[r].y();
    qMid     = rho_tmp[c] * vel[c].y();
    qLast    = rho_tmp[l] * vel[l].y();

    x_conv = computeConvection(nux[r], nux[c], nux[l],
                               qFrt, qMid, qLast, 
                               qConFrt, qConLast, delT, dx.x()); 

    vVel = (qMid - delT * ((d1_y[c] + d1_z[c]) * vel[c].y() 
                        +  (d3_y[c] + d5_z[c]) * rho_tmp[c] + x_conv))
                               /rho_tmp[c];

    //__________________________________
    //         Z   V E L O C I T Y
    qConFrt  = rho_tmp[r] * vel[r].x() * vel[r].z();
    qConLast = rho_tmp[l] * vel[l].x() * vel[l].z();
    qFrt     = rho_tmp[r] * vel[r].z();
    qMid     = rho_tmp[c] * vel[c].z();
    qLast    = rho_tmp[l] * vel[l].z();

    x_conv = computeConvection(nux[r], nux[c], nux[l],
                               qFrt, qMid, qLast,
                               qConFrt, qConLast, delT, dx.x());

    wVel = (qMid - delT * ((d1_y[c] + d1_z[c]) * vel[c].z() 
                        +  (d5_y[c] + d3_z[c]) * rho_tmp[c]  
                        + x_conv))/rho_tmp[c];
                        
    vel_CC[c] = Vector(uVel, vVel, wVel);
  }  
  //________________________________________________________
  // C O R N E R S    
  vector<IntVector> crn(4);
  crn[0] = IntVector(hi_x,    low.y(), zFaceCell);   // right-bottom-front/back
  crn[1] = IntVector(hi_x,    hi_y,    zFaceCell);   // right-top-front/back   
  crn[2] = IntVector(low.x(), hi_y,    zFaceCell);   // left-top-front/back 
  crn[3] = IntVector(low.x(), low.y(), zFaceCell);;  // left-bottom-front/back

  for( int corner = 0; corner < 4; corner ++ ) {
    IntVector c = crn[corner];
    uVel = vel[c].x() - delT * ((d1_x[c] + d1_y[c] + d1_z[c]) * vel[c].x()  
                             +  (d3_x[c] + d4_y[c] + d4_z[c]) * rho_tmp[c]) 
                        / rho_tmp[c];

    vVel = vel[c].y() - delT * ((d1_x[c] + d1_y[c] + d1_z[c]) * vel[c].y() 
                             +  (d4_x[c] + d3_y[c] + d5_z[c]) * rho_tmp[c])
                        / rho_tmp[c];

    wVel = vel[c].z() - delT * ((d1_x[c] + d1_y[c] + d1_z[c]) * vel[c].z() 
                             +  (d5_x[c] + d5_y[c] + d3_z[c]) * rho_tmp[c])
                        / rho_tmp[c];
    vel_CC[c] = Vector(uVel, vVel, wVel);
  }                      
} //end of the function zFaceVelLODI()

/*_________________________________________________________________
 Function~ fillFaceVelLODI--
 Purpose~  Compute velocity at boundary cells 
___________________________________________________________________*/

void   fillFaceVelLODI(CCVariable<Vector>& vel_CC,
                 const CCVariable<double>& d1_x,  
                 const CCVariable<double>& d3_x, 
                 const CCVariable<double>& d4_x,
                 const CCVariable<double>& d5_x,
                 const CCVariable<double>& d1_y,  
                 const CCVariable<double>& d3_y, 
                 const CCVariable<double>& d4_y, 
                 const CCVariable<double>& d5_y, 
                 const CCVariable<double>& d1_z,  
                 const CCVariable<double>& d3_z, 
                 const CCVariable<double>& d4_z,
                 const CCVariable<double>& d5_z,
                 const CCVariable<double>& nux,
                 const CCVariable<double>& nuy,
                 const CCVariable<double>& nuz,
                 const CCVariable<double>& rho_tmp,
                 const CCVariable<double>& p,
                 const CCVariable<Vector>& vel,
                 const Patch::FaceType face,
                 const double delT,
                 const Vector& dx)

{ 
  if(face ==  Patch::xplus || face ==  Patch::xminus) {
    xFaceVelLODI( face, vel_CC, 
                  d1_x, d3_x, d4_x, d5_x,
                  d1_y, d3_y, d4_y, d5_y, 
                  d1_z, d3_z, d4_z, d5_z, 
                  nux,nuy,nuz,
                  rho_tmp, p, vel, delT, dx);
  }
  if(face ==  Patch::yplus || face ==  Patch::yminus) {
       yFaceVelLODI(face, vel_CC, 
                    d1_x, d3_x, d4_x, d5_x,
                    d1_y, d3_y, d4_y, d5_y,             
                    d1_z, d3_z, d4_z, d5_z,             
                    nux,nuy,nuz,                        
                    rho_tmp, p, vel, delT, dx);         
  } 
  if(face ==  Patch::zplus || face ==  Patch::zminus) {  
    zFaceVelLODI(face, vel_CC, 
                 d1_x, d3_x, d4_x, d5_x,
                 d1_y, d3_y, d4_y, d5_y,
                 d1_z, d3_z, d4_z, d5_z,
                 nux,nuy,nuz,
                 rho_tmp, p, vel, delT, dx);
  }
}

/*_________________________________________________________________
 Function~ xFaceTempLODI--
 Purpose~  Compute temperature in boundary cells on x_plus/minus faces
___________________________________________________________________*/
void xFaceTempLODI(const Patch::FaceType face,
             CCVariable<double>& temp_CC, 
             const CCVariable<double>& d1_x, 
             const CCVariable<double>& d2_x, 
             const CCVariable<double>& d3_x, 
             const CCVariable<double>& d4_x, 
             const CCVariable<double>& d5_x,
             const CCVariable<double>& d1_y, 
             const CCVariable<double>& d2_y, 
             const CCVariable<double>& d3_y, 
             const CCVariable<double>& d4_y, 
             const CCVariable<double>& d5_y,
             const CCVariable<double>& d1_z, 
             const CCVariable<double>& d2_z, 
             const CCVariable<double>& d3_z, 
             const CCVariable<double>& d4_z, 
             const CCVariable<double>& d5_z,
             const CCVariable<double>& e,
             const CCVariable<double>& rho_CC,
             const CCVariable<double>& nux,
             const CCVariable<double>& nuy,
             const CCVariable<double>& nuz,
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
      y_conv   = computeConvection(nuy[t], nuy[c], nuy[b], 
                                qFrt, qMid, qLast, qConFrt, 
                                qConLast, delT, dx.y());

      qConFrt  = vel[f].z()  * (e[f] + p[f]);
      qConLast = vel[bk].z() * (e[bk] + p[bk]);
      qFrt     = e[f];
      qLast    = e[bk];

      z_conv = computeConvection(nuz[f], nuz[c], nuz[bk],
                                 qFrt, qMid, qLast, 
                                 qConFrt, qConLast, delT, dx.z()); 

      double vel_sqr = vel[c].length2();

      term1 = 0.5 * d1_x[c] * vel_sqr;
      term2 = d2_x[c]/(gamma - 1.0) + rho_tmp[c] * vel[c].x() * d3_x[c];
      term3 = rho_tmp[c] * vel[c].y() * d4_x[c] 
            + rho_tmp[c] * vel[c].z() * d5_x[c];
      term4 = y_conv + z_conv;
      double e_tmp = e[c] - delT * (term1 + term2 + term3 + term4);

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

     z_conv = computeConvection(nuz[f], nuz[c], nuz[bk],
                      qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.z());

     double vel_sqr = vel[c].length2();

     term1 = 0.5 * (d1_x[c] + d1_y[c]) * vel_sqr;
     term2 =  (d2_x[c] + d2_y[c])/(gamma - 1.0) 
           +  rho_tmp[c] * vel[c].x() * (d3_x[c] + d4_y[c]);
     term3 =  rho_tmp[c] * vel[c].y() * (d4_x[c] + d3_y[c]) 
           +  rho_tmp[c] * vel[c].z() * (d5_x[c] + d5_y[c]);

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

    z_conv = computeConvection(nuz[f], nuz[c], nuz[bk],
                     qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.z());

    double vel_sqr = vel[c].length2();

    term1 = 0.5 * (d1_x[c] + d1_y[c]) * vel_sqr;
    term2 =  (d2_x[c] + d2_y[c])/(gamma - 1.0) 
          +  rho_tmp[c] * vel[c].x() * (d3_x[c] + d4_y[c]);
    term3 =  rho_tmp[c] * vel[c].y() * (d4_x[c] + d3_y[c]) 
          +  rho_tmp[c] * vel[c].z() * (d5_x[c] + d5_y[c]);

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

    y_conv = computeConvection(nuy[t], nuy[c], nuy[b],
                    qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.y());

    double vel_sqr = vel[c].length2();

    term1 = 0.5 * (d1_x[c] + d1_z[c]) * vel_sqr;
    term2 =  (d2_x[c] + d2_z[c])/(gamma - 1.0) 
          +  rho_tmp[c] * vel[c].x() * (d3_x[c] + d4_z[c]);
    term3 =  rho_tmp[c] * vel[c].y() * (d4_x[c] + d5_z[c]) 
          +  rho_tmp[c] * vel[c].z() * (d5_x[c] + d3_z[c]);

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

    y_conv = computeConvection(nuy[t], nuy[c], nuy[b],
                      qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.y());

    double vel_sqr = vel[c].length2();

    term1 = 0.5 * (d1_x[c] + d1_z[c]) * vel_sqr;
    term2 =  (d2_x[c] + d2_z[c])/(gamma - 1.0) 
          +  rho_tmp[c] * vel[c].x() * (d3_x[c] + d4_z[c]);
    term3 =  rho_tmp[c] * vel[c].y() * (d4_x[c] + d5_z[c]) 
          +  rho_tmp[c] * vel[c].z() * (d5_x[c] + d3_z[c]);

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

     term1 = 0.5 * (d1_x[c] + d1_y[c] + d1_z[c]) * vel_sqr;
     term2 =       (d2_x[c] + d2_y[c] + d2_z[c])/(gamma - 1.0) 
           +  rho_tmp[c] * vel[c].x() * (d3_x[c] + d4_y[c] + d4_z[c]);
     term3 =  rho_tmp[c] * vel[c].y() * (d4_x[c] + d3_y[c] + d5_z[c]) 
           +  rho_tmp[c] * vel[c].z() * (d5_x[c] + d5_y[c] + d3_z[c]);

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
             const CCVariable<double>& d1_x, 
             const CCVariable<double>& d2_x, 
             const CCVariable<double>& d3_x, 
             const CCVariable<double>& d4_x, 
             const CCVariable<double>& d5_x,
             const CCVariable<double>& d1_y, 
             const CCVariable<double>& d2_y, 
             const CCVariable<double>& d3_y, 
             const CCVariable<double>& d4_y, 
             const CCVariable<double>& d5_y,
             const CCVariable<double>& d1_z, 
             const CCVariable<double>& d2_z, 
             const CCVariable<double>& d3_z, 
             const CCVariable<double>& d4_z, 
             const CCVariable<double>& d5_z,
             const CCVariable<double>& e,
             const CCVariable<double>& rho_CC,
             const CCVariable<double>& nux,
             const CCVariable<double>& nuy,
             const CCVariable<double>& nuz,
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
      x_conv = computeConvection(nux[r], nux[c], nux[l], 
                                   qFrt, qMid, qLast, qConFrt, 
                                   qConLast, delT, dx.x());

      qConFrt  = vel[f].z()  * (e[f] + p[f]);
      qConLast = vel[bk].z() * (e[bk] + p[bk]);
      qFrt     = e[f];
      qLast    = e[bk];

      z_conv = computeConvection(nuz[f], nuz[c], nuz[bk],
                                 qFrt, qMid, qLast, 
                                 qConFrt, qConLast, delT, dx.z()); 

      double vel_sqr = vel[c].length2();

      term1 = 0.5 * d1_y[c] * vel_sqr;
      term2 = d2_y[c]/(gamma - 1.0) + rho_tmp[c] * vel[c].x() * d4_y[c];
      term3 = rho_tmp[c] * vel[c].y() * d3_y[c] 
            + rho_tmp[c] * vel[c].z() * d5_y[c];
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

    z_conv = computeConvection(nuz[f], nuz[c], nuz[bk],
                               qFrt, qMid, qLast, 
                               qConFrt, qConLast, delT, dx.z());

    double vel_sqr = vel[c].length2();

    term1 = 0.5 * (d1_x[c] + d1_y[c]) * vel_sqr;
    term2 =  (d2_x[c] + d2_y[c])/(gamma - 1.0) 
          +  rho_tmp[c] * vel[c].x() * (d3_x[c] + d4_y[c]);
    term3 =  rho_tmp[c] * vel[c].y() * (d4_x[c] + d3_y[c]) 
          +  rho_tmp[c] * vel[c].z() * (d5_x[c] + d5_y[c]);

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

    z_conv = computeConvection(nuz[f], nuz[c], nuz[bk],
                               qFrt, qMid, qLast, 
                               qConFrt, qConLast, delT, dx.z());

    double vel_sqr = vel[c].length2();

    term1 = 0.5 * (d1_x[c] + d1_y[c]) * vel_sqr;
    term2 =  (d2_x[c] + d2_y[c])/(gamma - 1.0) 
          +  rho_tmp[c] * vel[c].x() * (d3_x[c] + d4_y[c]);
    term3 =  rho_tmp[c] * vel[c].y() * (d4_x[c] + d3_y[c]) 
          +  rho_tmp[c] * vel[c].z() * (d5_x[c] + d5_y[c]);

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

    x_conv = computeConvection(nux[r], nux[c], nux[l],
                               qFrt, qMid, qLast, 
                               qConFrt, qConLast, delT, dx.x());

    double vel_sqr = vel[c].length2();

    term1 = 0.5 * (d1_y[c] + d1_z[c]) * vel_sqr;
    term2 =  (d2_y[c] + d2_z[c])/(gamma - 1.0) 
          +  rho_tmp[c] * vel[c].x() * (d4_y[c] + d4_z[c]);
    term3 =  rho_tmp[c] * vel[c].y() * (d3_y[c] + d5_z[c]) 
          +  rho_tmp[c] * vel[c].z() * (d5_y[c] + d3_z[c]);

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

    x_conv = computeConvection(nux[r], nux[c], nux[l],
                               qFrt, qMid, qLast, 
                               qConFrt, qConLast, delT, dx.x());

    double vel_sqr = vel[c].length2();

    term1 = 0.5 * (d1_y[c] + d1_z[c]) * vel_sqr;
    term2 =  (d2_y[c] + d2_z[c])/(gamma - 1.0) 
          +  rho_tmp[c] * vel[c].x() * (d4_y[c] + d4_z[c]);
    term3 =  rho_tmp[c] * vel[c].y() * (d3_y[c] + d5_z[c]) 
          +  rho_tmp[c] * vel[c].z() * (d5_y[c] + d3_z[c]);

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

     term1 = 0.5 * (d1_x[c] + d1_y[c] + d1_z[c]) * vel_sqr;
     term2 =       (d2_x[c] + d2_y[c] + d2_z[c])/(gamma - 1.0) 
           +  rho_tmp[c] * vel[c].x() * (d3_x[c] + d4_y[c] + d4_z[c]);
     term3 =  rho_tmp[c] * vel[c].y() * (d4_x[c] + d3_y[c] + d5_z[c]) 
           +  rho_tmp[c] * vel[c].z() * (d5_x[c] + d5_y[c] + d3_z[c]);

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
             const CCVariable<double>& d1_x, 
             const CCVariable<double>& d2_x, 
             const CCVariable<double>& d3_x, 
             const CCVariable<double>& d4_x, 
             const CCVariable<double>& d5_x,
             const CCVariable<double>& d1_y, 
             const CCVariable<double>& d2_y, 
             const CCVariable<double>& d3_y, 
             const CCVariable<double>& d4_y, 
             const CCVariable<double>& d5_y,
             const CCVariable<double>& d1_z, 
             const CCVariable<double>& d2_z, 
             const CCVariable<double>& d3_z, 
             const CCVariable<double>& d4_z, 
             const CCVariable<double>& d5_z,
             const CCVariable<double>& e,
             const CCVariable<double>& rho_CC,
             const CCVariable<double>& nux,
             const CCVariable<double>& nuy,
             const CCVariable<double>& nuz,
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
      x_conv = computeConvection(nux[r], nux[c], nux[l], 
                                 qFrt, qMid, qLast, qConFrt, 
                                 qConLast, delT, dx.x());

      qConFrt  = vel[t].y() * (e[t] + p[t]);
      qConLast = vel[b].y() * (e[b] + p[b]);
      qFrt     = e[t];
      qLast    = e[b];

      y_conv = computeConvection(nuy[t], nuy[c], nuy[b],
                                 qFrt, qMid, qLast, 
                                 qConFrt, qConLast, delT, dx.y()); 

      double vel_sqr = vel[c].length2();

      term1 = 0.5 * d1_z[c] * vel_sqr;
      term2 = d2_z[c] / (gamma - 1.0) 
            + rho_tmp[c] * vel[c].x() * d4_z[c];
      term3 = rho_tmp[c] * vel[c].y() * d5_z[c]  
            + rho_tmp[c] * vel[c].z() * d3_z[c];
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

    y_conv = computeConvection(nuy[t], nuy[c], nuy[b],
                     qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.y());

    double vel_sqr = vel[c].length2();

    term1 = 0.5 * (d1_x[c] + d1_z[c]) * vel_sqr;
    term2 =       (d2_x[c] + d2_z[c])/(gamma - 1.0) 
          +  rho_tmp[c] * vel[c].x() * (d3_x[c] + d4_z[c]);
    term3 =  rho_tmp[c] * vel[c].y() * (d4_x[c] + d5_z[c]) 
          +  rho_tmp[c] * vel[c].z() * (d5_x[c] + d3_z[c]);

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

    y_conv = computeConvection(nuy[t], nuy[c], nuy[b],
                     qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.y());

    double vel_sqr = vel[c].length2();

    term1 = 0.5 * (d1_x[c] + d1_z[c]) * vel_sqr;
    term2 =       (d2_x[c] + d2_z[c])/(gamma - 1.0) 
          +  rho_tmp[c] * vel[c].x() * (d3_x[c] + d4_z[c]);
    term3 =  rho_tmp[c] * vel[c].y() * (d4_x[c] + d5_z[c]) 
          +  rho_tmp[c] * vel[c].z() * (d5_x[c] + d3_z[c]);

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

    x_conv = computeConvection(nux[r], nux[c], nux[l],
                     qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.x());

    double vel_sqr = vel[c].length2();

    term1 = 0.5 * (d1_y[c] + d1_z[c]) * vel_sqr;
    term2 =       (d2_y[c] + d2_z[c])/(gamma - 1.0) 
          +  rho_tmp[c] * vel[c].x() * (d4_y[c] + d4_z[c]);
    term3 =  rho_tmp[c] * vel[c].y() * (d3_y[c] + d5_z[c]) 
          +  rho_tmp[c] * vel[c].z() * (d5_y[c] + d3_z[c]);

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

    x_conv = computeConvection(nux[r], nux[c], nux[l],
                     qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.x());

    double vel_sqr = vel[c].length2();

    term1 = 0.5 * (d1_y[c] + d1_z[c]) * vel_sqr;
    term2 =       (d2_y[c] + d2_z[c])/(gamma - 1.0) 
          +  rho_tmp[c] * vel[c].x() * (d4_y[c] + d4_z[c]);
    term3 =  rho_tmp[c] * vel[c].y() * (d3_y[c] + d5_z[c]) 
          +  rho_tmp[c] * vel[c].z() * (d5_y[c] + d3_z[c]);

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

    term1 = 0.5 * (d1_x[c] + d1_y[c] + d1_z[c]) * vel_sqr;
    term2 =       (d2_x[c] + d2_y[c] + d2_z[c])/(gamma - 1.0) 
          +  rho_tmp[c] * vel[c].x() * (d3_x[c] + d4_y[c] + d4_z[c]);
    term3 =  rho_tmp[c] * vel[c].y() * (d4_x[c] + d3_y[c] + d5_z[c]) 
          +  rho_tmp[c] * vel[c].z() * (d5_x[c] + d5_y[c] + d3_z[c]);

    double e_tmp = e[c] - delT * ( term1 + term2 + term3);

    temp_CC[c] = e_tmp/rho_CC[c]/cv - 0.5 * vel_sqr/cv;
  }              
} //end of the function zFaceTempLODI() 

/*_________________________________________________________________
 Function~ fillFaceTempLODI--
___________________________________________________________________*/
void fillFaceTempLODI(CCVariable<double>& temp_CC, 
              const CCVariable<double>& d1_x, 
              const CCVariable<double>& d2_x, 
              const CCVariable<double>& d3_x, 
              const CCVariable<double>& d4_x, 
              const CCVariable<double>& d5_x,
              const CCVariable<double>& d1_y, 
              const CCVariable<double>& d2_y, 
              const CCVariable<double>& d3_y, 
              const CCVariable<double>& d4_y, 
              const CCVariable<double>& d5_y,
              const CCVariable<double>& d1_z, 
              const CCVariable<double>& d2_z, 
              const CCVariable<double>& d3_z, 
              const CCVariable<double>& d4_z, 
              const CCVariable<double>& d5_z,
              const CCVariable<double>& e,
              const CCVariable<double>& rho_CC,
              const CCVariable<double>& nux,
              const CCVariable<double>& nuy,
              const CCVariable<double>& nuz,
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
    xFaceTempLODI(face, temp_CC,
                  d1_x, d2_x, d3_x, d4_x, d5_x,
                  d1_y, d2_y, d3_y, d4_y, d5_y,
                  d1_z, d2_z, d3_z, d4_z, d5_z,
                  e, rho_CC,
                  nux,nuy,nuz,
                  rho_tmp, p, vel, 
                  delT, cv, gamma, dx);
  }
  if (face == Patch::yplus || face == Patch::yminus){
    yFaceTempLODI(face, temp_CC,
                  d1_x, d2_x, d3_x, d4_x, d5_x,
                  d1_y, d2_y, d3_y, d4_y, d5_y,
                  d1_z, d2_z, d3_z, d4_z, d5_z,
                  e, rho_CC,
                  nux,nuy,nuz,
                  rho_tmp, p, vel, 
                  delT, cv, gamma, dx);
  }
  if (face == Patch::zplus || face == Patch::zminus){
    zFaceTempLODI(face, temp_CC,
                  d1_x, d2_x, d3_x, d4_x, d5_x,
                  d1_y, d2_y, d3_y, d4_y, d5_y,
                  d1_z, d2_z, d3_z, d4_z, d5_z,
                  e, rho_CC,
                  nux,nuy,nuz,
                  rho_tmp, p, vel, 
                  delT, cv, gamma, dx);
  }
}
/* --------------------------------------------------------------------- 
 Function~  fillFacePressLODI--
 Purpose~   Back out the pressure from f_theta and P_EOS
---------------------------------------------------------------------  */
void fillFacePress_LODI(CCVariable<double>& press_CC,
                        const StaticArray<CCVariable<double> >& rho_micro,
                        const StaticArray<constCCVariable<double> >& Temp_CC,
                        const StaticArray<CCVariable<double> >& f_theta,
                        const int numALLMatls,
                        SimulationStateP& sharedState, 
                        Patch::FaceType face)
{ 
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
  
    switch (face) {
    case Patch::xplus:
      for (int j = low.y(); j<hi.y(); j++) {
        for (int k = low.z(); k<hi.z(); k++) {
          IntVector c = IntVector(hi.x()-1,j,k);
          
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
     //         cout << "press_CC" << c << press_CC[c] << endl;           
          }  // for ALLMatls...
        } // k loop
      }  //j loop
      break;
      
    case Patch::xminus:
      for (int j = low.y(); j<hi.y(); j++) {
        for (int k = low.z(); k<hi.z(); k++) {
          IntVector c = IntVector(low.x(),j,k);
          
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
        } // k loop
      }  //j loop
      
      break;
    case Patch::yplus:
      for (int i = low.x(); i<hi.x(); i++) {
        for (int k = low.z(); k<hi.z(); k++) {

          IntVector c = IntVector(i,hi.y()-1,k);
          
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
        } // k loop
      } // i loop
      
      break;
    case Patch::yminus:
      for (int i = low.x(); i<hi.x(); i++) {
        for (int k = low.z(); k<hi.z(); k++) {

          IntVector c = IntVector(i,low.y(),k);
          
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
   //       cout << "press_CC" << c << press_CC[c] << endl;           
          } 
        } // k loop
      } // i loop
      
      break;
    case Patch::zplus:
      for (int i = low.x(); i<hi.x(); i++) {
        for (int j = low.y(); j<hi.y(); j++) {

          IntVector c = IntVector(i,j,hi.z()-1);
          
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
   //       cout << "press_CC" << c << press_CC[c] << endl;           
          }  // for ALLMatls...
        } // j loop
      } // i loop
      
      break;
    case Patch::zminus:       //   Z M I N U S
      for (int i = low.x(); i<hi.x(); i++) {
        for (int j = low.y(); j<hi.y(); j++) {

          IntVector c = IntVector(i,j,low.z());
          
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
 //         cout << "press_CC" << c << press_CC[c] << endl;           
          } 
        } // j loop
      } // i loop
      break;
      
    case Patch::numFaces:
      break;
    case Patch::invalidFace:
      break;
    }
  }
#endif    // LODI_BCS
}  // using namespace Uintah
