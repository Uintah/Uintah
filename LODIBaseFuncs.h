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
 Function~ xPlusDensityLODI--
 Purpose~  Compute density in boundary cells on x_plus face
___________________________________________________________________*/
void xPlusDensityLODI(CCVariable<double>& rho_CC,
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
  cout_doing << " I am in xPlusDensityLODI" << endl;
  IntVector low,hi,r,l,t,b,f,bk;

  low = vel.getLowIndex();
  hi  = vel.getHighIndex();
  int hi_x = hi.x() - 1;
  int hi_y = hi.y() - 1;
  int hi_z = hi.z() - 1;
  double y_conv, z_conv;
  double qConFrt,qConLast;
  //___________________________________________________________________
  // Compute the density on the area of i = hi_x, low.y() < j < hi_y
  // and low.z() < k < hi_z 
  for(int j = low.y()+1; j < hi_y; j++) {
    for(int k = low.z()+1; k < hi_z; k++) {
      r  =  IntVector(hi_x,  j,   k);
      t  =  IntVector(hi_x,  j+1, k);
      b  =  IntVector(hi_x,  j-1, k);
      f  =  IntVector(hi_x,  j,   k+1);
      bk =  IntVector(hi_x,  j,   k-1);       

      qConFrt  = rho_tmp[t] * vel[t].y();
      qConLast = rho_tmp[b] * vel[b].y();
      y_conv = computeConvection(nuy[t], nuy[r], nuy[b], 
                                 rho_tmp[t], rho_tmp[r], rho_tmp[b], qConFrt, 
                                 qConLast, delT, dx.y());

      qConFrt  = rho_tmp[f] * vel[f].z();
      qConLast = rho_tmp[bk] * vel[bk].z();
      z_conv = computeConvection(nuz[f], nuz[r], nuz[bk],
                                 rho_tmp[f], rho_tmp[r], rho_tmp[bk], 
                                 qConFrt, qConLast, delT, dx.z());

      rho_CC[r] = rho_tmp[r] - delT * (d1_x[r] + y_conv + z_conv);

     // rho_CC[r] = rho_tmp[r] - delT * d1_x[r] ;
 #if ON
       cout.setf(ios::scientific,ios::floatfield);
       cout.precision(16);
       cout << "rho_CC,rho_tmp, d1_x, dt, y_conv" << r << "= " << rho_CC[r] << ",";
       cout << rho_tmp[r] << "," << d1_x[r] << "," << delT << y_conv << endl;  
 #endif    
    } // end of j loop
  } //end of k loop

  //__________________________________
  //  E D G E    right-bottom
  for(int k = low.z()+1; k < hi_z; k++) {
    r  =  IntVector(hi_x,   low.y(),   k);
    f  =  IntVector(hi_x,   low.y(),   k+1);
    bk =  IntVector(hi_x,   low.y(),   k-1);
    qConFrt  = rho_tmp[f]  * vel[f].z();
    qConLast = rho_tmp[bk] * vel[bk].z();
    z_conv = computeConvection(nuz[f], nuz[r], nuz[bk],
                               rho_tmp[f], rho_tmp[r], rho_tmp[bk], 
                               qConFrt, qConLast, delT, dx.z());

    rho_CC[r] = rho_tmp[r] - delT * (d1_x[r] + d1_y[r] + z_conv);
  }

  //__________________________________
  //    E D G E   right-top
  for(int k = low.z()+1; k < hi_z; k++) {
    r  =  IntVector(hi_x,   hi_y,   k);
    f  =  IntVector(hi_x,   hi_y,   k+1);
    bk =  IntVector(hi_x,   hi_y,   k-1);

    qConFrt  = rho_tmp[f]  * vel[f].z();
    qConLast = rho_tmp[bk] * vel[bk].z();
    z_conv   = computeConvection(nuz[f], nuz[r], nuz[bk],
                              rho_tmp[f], rho_tmp[r], rho_tmp[bk], 
                              qConFrt, qConLast, delT, dx.z());
    rho_CC[r] = rho_tmp[r] - delT * (d1_x[r] + d1_y[r] + z_conv);
  }

  //_____________________________________________________
  //    E D G E   right--back
  for(int j = low.y()+1; j < hi_y; j++) {
    r  =  IntVector(hi_x,   j,   low.z());
    t  =  IntVector(hi_x,   j+1, low.z());
    b  =  IntVector(hi_x,   j-1, low.z());
    qConFrt  = rho_tmp[t] * vel[t].y();
    qConLast = rho_tmp[b] * vel[b].y();
    y_conv = computeConvection(nuy[t], nuy[r], nuy[b],
                           rho_tmp[t], rho_tmp[r], rho_tmp[b], 
                           qConFrt, qConLast, delT, dx.y());
    rho_CC[r] = rho_tmp[r] - delT * (d1_x[r] + d1_z[r] + y_conv);
  } 

  //_________________________________________________
  // E D G E    right--front
  for(int j = low.y()+1; j < hi_y; j++) {
    r  =  IntVector(hi_x,   j,   hi_z);
    t  =  IntVector(hi_x,   j+1, hi_z);
    b  =  IntVector(hi_x,   j-1, hi_z);

    qConFrt  = rho_tmp[t] * vel[t].y();
    qConLast = rho_tmp[b] * vel[b].y();
    y_conv   = computeConvection(nuy[t], nuy[r], nuy[b],
                              rho_tmp[t], rho_tmp[r], rho_tmp[b], 
                              qConFrt, qConLast, delT, dx.y());
    rho_CC[r] = rho_tmp[r] - delT * (d1_x[r] + d1_z[r] + y_conv);
  } 

  //________________________________________________________
  // C O R N E R S    
  vector<IntVector> crn(3);
  crn[0] = IntVector(hi_x, low.y(), hi_z);         // right-bottom-front
  crn[1] = IntVector(hi_x, hi_y,    hi_z);         // right-top-front    
  crn[2] = IntVector(hi_x, hi_y,    low.z());      // right-top-back    
  crn[3] = IntVector(hi_x, low.y(), low.z());      // right-bottom-back 

  for(int corner = 0; corner <4; corner ++ ) {
    IntVector c = crn[corner];
    rho_CC[c] = rho_tmp[c] - delT * (d1_x[c] + d1_y[c] + d1_z[c]);
  }                     
}

/*_________________________________________________________________
 Function~ xMinusDensityLODI--
 Purpose~  Compute density in boundary cells on x_minus face
           using Characteristic Boundary Condition
___________________________________________________________________*/
void xMinusDensityLODI(CCVariable<double>& rho_CC,
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
  cout_doing << " I am in xMinusDensityLODI" << endl;    
  IntVector low,hi,r,l,t,b,f,bk;

  low = vel.getLowIndex();
  hi  = vel.getHighIndex();

  int hi_y = hi.y() - 1;
  int hi_z = hi.z() - 1;
  double y_conv, z_conv;
  double qConFrt,qConLast;
  
  for(int j = low.y()+1; j < hi_y; j++) {
    for(int k = low.z()+1; k < hi_z; k++) {
      l  =  IntVector(low.x(),  j,   k);
      t  =  IntVector(low.x(),  j+1, k);
      b  =  IntVector(low.x(),  j-1, k);
      f  =  IntVector(low.x(),  j,   k+1);
      bk =  IntVector(low.x(),  j,   k-1);       

      qConFrt  = rho_tmp[t] * vel[t].y();
      qConLast = rho_tmp[b] * vel[b].y();
      y_conv = computeConvection(nuy[t], nuy[l], nuy[b], 
                                 rho_tmp[t], rho_tmp[l], rho_tmp[b], qConFrt, 
                                 qConLast, delT, dx.y());

      qConFrt  = rho_tmp[f] * vel[f].z();
      qConLast = rho_tmp[bk] * vel[bk].z();
      z_conv = computeConvection(nuz[f], nuz[l], nuz[bk],
                                 rho_tmp[f], rho_tmp[l], rho_tmp[bk], 
                                 qConFrt, qConLast, delT, dx.z());

      rho_CC[l] = rho_tmp[l] - delT * (d1_x[l] + y_conv + z_conv);

       // rho_CC[l] = rho_tmp[l] - delT * d1_x[l] ;
#if ON
       cout.setf(ios::scientific,ios::floatfield);
       cout.precision(16);
       cout << "rho_CC,rho_tmp, d1_x, dt, y_conv" << l << "= " << rho_CC[l] << ",";
       cout << rho_tmp[l] << "," << d1_x[l] << "," << delT << y_conv << endl;  
#endif    
    } // end of j loop
  } //end of k loop
  
  //__________________________________________________________
  //  E D G E   left -bottom
  for(int k = low.z()+1; k < hi_z; k++) {
     l  =  IntVector(low.x(),   low.y(),   k);
     f  =  IntVector(low.x(),   low.y(),   k+1);
     bk =  IntVector(low.x(),   low.y(),   k-1);

     qConFrt  = rho_tmp[f]  * vel[f].z();
     qConLast = rho_tmp[bk] * vel[bk].z();
     z_conv = computeConvection(nuz[f], nuz[l], nuz[bk],
                                rho_tmp[f], rho_tmp[l], rho_tmp[bk], 
                                qConFrt, qConLast, delT, dx.z());

     rho_CC[l] = rho_tmp[l] - delT * (d1_x[l] + d1_y[l] + z_conv);
  }

  //_______________________________________________________
  //    E D G E    left- top
  for(int k = low.z()+1; k < hi_z; k++) {
    l  =  IntVector(low.x(),   hi_y,   k);
    f  =  IntVector(low.x(),   hi_y,   k+1);
    bk =  IntVector(low.x(),   hi_y,   k-1);

    qConFrt  = rho_tmp[f]  * vel[f].z();
    qConLast = rho_tmp[bk] * vel[bk].z();
    z_conv = computeConvection(nuz[f], nuz[l], nuz[bk],
                              rho_tmp[f], rho_tmp[l], rho_tmp[bk], 
                              qConFrt, qConLast, delT, dx.z());
    rho_CC[l] = rho_tmp[l] - delT * (d1_x[l] + d1_y[l] + z_conv);
  } 
  //_____________________________________________________
  //    E D G E     left-back
  for(int j = low.y()+1; j < hi_y; j++) {
    l  =  IntVector(low.x(),   j,   low.z());
    t  =  IntVector(low.x(),   j+1, low.z());
    b  =  IntVector(low.x(),   j-1, low.z());

    qConFrt  = rho_tmp[t] * vel[t].y();
    qConLast = rho_tmp[b] * vel[b].y();
    y_conv = computeConvection(nuy[t], nuy[l], nuy[b],
                              rho_tmp[t], rho_tmp[l], rho_tmp[b], 
                              qConFrt, qConLast, delT, dx.y());
    rho_CC[l] = rho_tmp[l] - delT * (d1_x[l] + d1_z[l] + y_conv);
  }

  //_________________________________________________
  // E D G E        left-front
  for(int j = low.y()+1; j < hi_y; j++) {
    l  =  IntVector(low.x(),   j,   hi_z);
    t  =  IntVector(low.x(),   j+1, hi_z);
    b  =  IntVector(low.x(),   j-1, hi_z);

    qConFrt  = rho_tmp[t] * vel[t].y();
    qConLast = rho_tmp[b] * vel[b].y();
    y_conv   = computeConvection(nuy[t], nuy[l], nuy[b],
                              rho_tmp[t], rho_tmp[l], rho_tmp[b], 
                              qConFrt, qConLast, delT, dx.y());
    rho_CC[l] = rho_tmp[l] - delT * (d1_x[l] + d1_z[l] + y_conv);
  } 
  //________________________________________________________
  // C O R N E R S    
  vector<IntVector> crn(3);
  crn[0] = IntVector(low.x(), low.y(), hi_z);         // left-bottom-front
  crn[1] = IntVector(low.x(), hi_y,    hi_z);         // left-top-front    
  crn[2] = IntVector(low.x(), hi_y,    low.z());      // left-top-back    
  crn[3] = IntVector(low.x(), low.y(), low.z());      // left-bottom-back 

  for( int corner = 0; corner < 4; corner ++ ) {
    IntVector c = crn[corner];
    rho_CC[c] = rho_tmp[c] - delT * (d1_x[c] + d1_y[c] + d1_z[c]);
  }
}

/*_________________________________________________________________
 Function~ yPlusDensityLODI--
 Purpose~  Compute density in boundary cells on y_plus face 
___________________________________________________________________*/
void yPlusDensityLODI(CCVariable<double>& rho_CC,
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
  cout_doing << " I am in yPlusDensityLODI" << endl;    
  IntVector low,hi,r,l,t,b,f,bk;

  low = vel.getLowIndex();
  hi  = vel.getHighIndex();
  int hi_x = hi.x() - 1;
  int hi_y = hi.y() - 1;
  int hi_z = hi.z() - 1;
  double x_conv, z_conv;
  double qConFrt,qConLast;
  //___________________________________________________________________
  // Compute the density on the area of i = hi_x, low.y() < j < hi_y
  // and low.z() < k < hi_z 
  for(int i = low.x()+1; i < hi_x; i++) {
    for(int k = low.z()+1; k < hi_z; k++) {
      r  =  IntVector(i+1,  hi_y,   k);
      l  =  IntVector(i-1,  hi_y,   k);
      t  =  IntVector(i,    hi_y,   k);
      f  =  IntVector(i,    hi_y,   k+1);
      bk =  IntVector(i,    hi_y,   k-1);       

      //__________________________________________
      // mass conservation law, computing density
      qConFrt  = rho_tmp[r] * vel[r].x();
      qConLast = rho_tmp[l] * vel[l].x();
      x_conv = computeConvection(nux[r], nux[t], nux[l], 
                                 rho_tmp[r], rho_tmp[t], rho_tmp[l], qConFrt, 
                                 qConLast, delT, dx.x());

      qConFrt  = rho_tmp[f] * vel[f].z();
      qConLast = rho_tmp[bk] * vel[bk].z();
      z_conv = computeConvection(nuz[f], nuz[t], nuz[bk],
                                 rho_tmp[f], rho_tmp[t], rho_tmp[bk], 
                                 qConFrt, qConLast, delT, dx.z());

      rho_CC[t] = rho_tmp[t] - delT * (d1_y[t] + x_conv + z_conv);
     // rho_CC[t] = rho_tmp[t] - delT * d1_y[t] ;
  #if ON
      cout.setf(ios::scientific,ios::floatfield);
      cout.precision(16);
      cout << "rho_CC,rho_tmp, d1_y, dt, x,z_conv" << t << "= " << rho_CC[t] << ",";
      cout << rho_tmp[t] << "," << d1_y[t] << "," << delT << x_conv << "," << z_conv << endl;  
  #endif    
    } 
  } 

  //__________________________________________________________
  //  E D G E    right- top
  for(int k = low.z()+1; k < hi_z; k++) {
    t  =  IntVector(hi_x,   hi_y,   k);
    f  =  IntVector(hi_x,   hi_y,   k+1);
    bk =  IntVector(hi_x,   hi_y,   k-1);
    qConFrt  = rho_tmp[f]  * vel[f].z();
    qConLast = rho_tmp[bk] * vel[bk].z();
    z_conv = computeConvection(nuz[f], nuz[t], nuz[bk],
                               rho_tmp[f], rho_tmp[t], rho_tmp[bk], 
                               qConFrt, qConLast, delT, dx.z());

    rho_CC[t] = rho_tmp[t] - delT * (d1_x[t] + d1_y[t] + z_conv);
  }

  //_______________________________________________________
  //    E D G E    top-front
  for(int i = low.x()+1; i < hi_x; i++) {
    r  =  IntVector(i+1, hi_y,  hi_z);
    l  =  IntVector(i-1, hi_y,  hi_z);
    t  =  IntVector(i,   hi_y,  hi_z);
    qConFrt  = rho_tmp[r] * vel[r].x();
    qConLast = rho_tmp[l] * vel[l].x();
    x_conv   = computeConvection(nux[r], nux[t], nux[l],
                              rho_tmp[r], rho_tmp[t], rho_tmp[l], 
                              qConFrt, qConLast, delT, dx.x());
    rho_CC[t] = rho_tmp[t] - delT * (d1_y[t] + d1_z[t] + x_conv);
  }
         
  //_____________________________________________________
  //    E D G E    left-top
  for(int k = low.z()+1; k < hi_z; k++) {
     t  = IntVector(low.x(),   hi_y,   k);
     f  = IntVector(low.x(),   hi_y,   k+1);
     bk = IntVector(low.x(),   hi_y,   k-1);
     qConFrt  = rho_tmp[f] * vel[f].z();
     qConLast = rho_tmp[bk] * vel[bk].z();
     z_conv   = computeConvection(nuz[f], nuz[t], nuz[bk],
                               rho_tmp[f], rho_tmp[t], rho_tmp[bk], 
                               qConFrt, qConLast, delT, dx.z());
     rho_CC[t] = rho_tmp[t] - delT * (d1_x[t] + d1_y[t] + z_conv);
   }

  //_________________________________________________
  // E D G E    top-back
  for(int i = low.x()+1; i < hi_x; i++) {
    r  =  IntVector(i+1, hi_y,   low.z());
    l  =  IntVector(i-1, hi_y,   low.z());
    t  =  IntVector(i,   hi_y,   low.z());

    qConFrt  = rho_tmp[r] * vel[r].x();
    qConLast = rho_tmp[l] * vel[l].x();
    x_conv   = computeConvection(nux[r], nux[t], nux[l],
                              rho_tmp[r], rho_tmp[t], rho_tmp[l], 
                              qConFrt, qConLast, delT, dx.x());
    rho_CC[t] = rho_tmp[t] - delT * (d1_y[t] + d1_z[t] + x_conv);
  }
  //________________________________________________________
  // C O R N E R S    
  vector<IntVector> crn(3);
  crn[0] = IntVector(hi_x,    hi_y, hi_z);         // right-top-front
  crn[1] = IntVector(low.x(), hi_y, hi_z);         // left-top-front    
  crn[2] = IntVector(low.x(), hi_y, low.z());      // left-top-back    
  crn[3] = IntVector(hi_x,    hi_y, low.z());      // right-top-back 

  for( int corner = 0; corner <4; corner ++ ) {
    IntVector c = crn[corner];
    rho_CC[c] = rho_tmp[c] - delT * (d1_x[c] + d1_y[c] + d1_z[c]);
  }
} //end of the function yPlusDensityLODI()

/*_________________________________________________________________
 Function~ yMinusDensityLODI--
 Purpose~  Compute density in boundary cells on yMinus face
___________________________________________________________________*/
void yMinusDensityLODI(CCVariable<double>& rho_CC,
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
     cout << " I am in yMinusDensityLODI" << endl;    
     IntVector low,hi,r,l,t,b,f,bk;

     low = vel.getLowIndex();
     hi  = vel.getHighIndex();
     int hi_x = hi.x() - 1;
     int hi_z = hi.z() - 1;
     double x_conv,  z_conv;
     double qConFrt,qConLast;
     //___________________________________________________________________
     // Compute the density on the area of low.x() < i < hi_x, j = hi_y
     // and low.z() < k < hi_z 
     for(int i = low.x()+1; i < hi_x; i++) {
       for(int k = low.z()+1; k < hi_z; k++) {
         r  =  IntVector(i+1,  low.y(),   k);
         l  =  IntVector(i-1,  low.y(),   k);
         b  =  IntVector(i,    low.y(),   k);
         f  =  IntVector(i,    low.y(),   k+1);
         bk =  IntVector(i,    low.y(),   k-1);     
         qConFrt  = rho_tmp[r] * vel[r].x();
         qConLast = rho_tmp[l] * vel[l].x();
         x_conv = computeConvection(nux[r], nux[b], nux[l], 
                                    rho_tmp[r], rho_tmp[b], rho_tmp[l], qConFrt, 
                                    qConLast, delT, dx.x());

         qConFrt  = rho_tmp[f] * vel[f].z();
         qConLast = rho_tmp[bk] * vel[bk].z();
         z_conv = computeConvection(nuz[f], nuz[b], nuz[bk],
                                    rho_tmp[f], rho_tmp[b], rho_tmp[bk], 
                                    qConFrt, qConLast, delT, dx.z());

         rho_CC[b] = rho_tmp[b] - delT * (d1_y[b] + x_conv + z_conv);
         // rho_CC[b] = rho_tmp[b] - delT * d1_y[b] ;
 #if ON
         cout.setf(ios::scientific,ios::floatfield);
         cout.precision(16);
         cout << "rho_CC,rho_tmp, d1_y, dt, x,z_conv" << b << "= " << rho_CC[b] << ",";
         cout << rho_tmp[b] << "," << d1_y[b] << "," << delT << "," <<  x_conv << "," << z_conv << endl;  
 #endif    
       } // end of i loop
     } //end of k loop 
     //__________________________________________________________
     //  E D G E      right-bottom
     for(int k = low.z()+1; k < hi_z; k++) {

       b  =  IntVector(hi_x,   low.y(),   k);
       f  =  IntVector(hi_x,   low.y(),   k+1);
       bk =  IntVector(hi_x,   low.y(),   k-1);

      //__________________________________________
      // mass conservation law, computing density
        qConFrt  = rho_tmp[f]  * vel[f].z();
        qConLast = rho_tmp[bk] * vel[bk].z();
        z_conv = computeConvection(nuz[f], nuz[b], nuz[bk],
                                   rho_tmp[f], rho_tmp[b], rho_tmp[bk], 
                                   qConFrt, qConLast, delT, dx.z());

        rho_CC[b] = rho_tmp[b] - delT * (d1_x[b] + d1_y[b] + z_conv);
     } //end of k loop 

     //_______________________________________________________
     //    E D G E     bottom-front
       for(int i = low.x()+1; i < hi_x; i++) {
         r  =  IntVector(i+1, low.y(),    hi_z);
         l  =  IntVector(i-1, low.y(),    hi_z);
         b  =  IntVector(i,   low.y(),    hi_z);

          qConFrt  = rho_tmp[r] * vel[r].x();
          qConLast = rho_tmp[l] * vel[l].x();
          x_conv = computeConvection(nux[r], nux[b], nux[l],
                                  rho_tmp[r], rho_tmp[b], rho_tmp[l], 
                                  qConFrt, qConLast, delT, dx.x());
         rho_CC[b] = rho_tmp[b] - delT * (d1_y[b] + d1_z[b] + x_conv);  
      } 
      //_____________________________________________________
      //    E D G E   left-bottom
      for(int k = low.z()+1; k < hi_z; k++) {
        b  = IntVector(low.x(),   low.y(),   k);
        f  = IntVector(low.x(),   low.y(),   k+1);
        bk = IntVector(low.x(),   low.y(),   k-1);
        qConFrt  = rho_tmp[f] * vel[f].z();
        qConLast = rho_tmp[bk] * vel[bk].z();
        z_conv = computeConvection(nuz[f], nuz[b], nuz[bk],
                                rho_tmp[f], rho_tmp[b], rho_tmp[bk], 
                                qConFrt, qConLast, delT, dx.z());
        rho_CC[b] = rho_tmp[b] - delT * (d1_x[b] + d1_y[b] + z_conv);
      }

      //_________________________________________________
      // E D G E    bottom-back
      for(int i = low.x()+1; i < hi_x; i++) {
        r  =  IntVector(i+1, low.y(),   low.z());
        l  =  IntVector(i-1, low.y(),   low.z());
        b  =  IntVector(i,   low.y(),   low.z());

        qConFrt  = rho_tmp[r] * vel[r].x();
        qConLast = rho_tmp[l] * vel[l].x();
        x_conv   = computeConvection(nux[r], nux[b], nux[l],
                                  rho_tmp[r], rho_tmp[b], rho_tmp[l], 
                                  qConFrt, qConLast, delT, dx.x());
        rho_CC[b] = rho_tmp[b] - delT * (d1_y[b] + d1_z[b] + x_conv);
      }
        
      //________________________________________________________
      // C O R N E R S    
      vector<IntVector> crn(3);
      crn[0] = IntVector(hi_x,    low.y(), hi_z);         // right-bottom-front
      crn[1] = IntVector(low.x(), low.y(), hi_z);         // left-bottom-front    
      crn[2] = IntVector(low.x(), low.y(), low.z());      // left-bottom-back    
      crn[3] = IntVector(hi_x,    low.y(), low.z());      // right-bottom-back 

      for( int corner = 0; corner <4; corner ++ ) {
        IntVector c = crn[corner];
        rho_CC[c] = rho_tmp[c] - delT * (d1_x[c] + d1_y[c] + d1_z[c]);
      }         
} //end of the function yMinusDensityLODI()


/*_________________________________________________________________
 Function~ zPlusDensityLODI--
 Purpose~  Compute density in boundary cells on the z_plus face
___________________________________________________________________*/
void zPlusDensityLODI(CCVariable<double>& rho_CC,
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
  cout << " I am in zPlusDensityLODI" << endl;    
  IntVector low,hi,r,l,t,b,f,bk;

  low = vel.getLowIndex();
  hi  = vel.getHighIndex();
  int hi_x = hi.x() - 1;
  int hi_y = hi.y() - 1;
  int hi_z = hi.z() - 1;

  double x_conv, y_conv;
  double qConFrt,qConLast;
  //___________________________________________________________________
  // Compute the density on the area of low.x() < i < hi_x, low.y() < j < hi_y
  // and  k = hi_z 
  for(int i = low.x()+1; i < hi_x; i++) {
    for(int j = low.y()+1; j < hi_y; j++) {
      r  =  IntVector(i+1, j,   hi_z);
      l  =  IntVector(i-1, j,   hi_z);
      t  =  IntVector(i,   j+1, hi_z);
      b  =  IntVector(i,   j-1, hi_z);
      f  =  IntVector(i,   j,   hi_z);

      qConFrt  = rho_tmp[r] * vel[r].x();
      qConLast = rho_tmp[l] * vel[l].x();
      x_conv = computeConvection(nux[r], nux[f], nux[l], 
                                 rho_tmp[r], rho_tmp[f], rho_tmp[l], qConFrt, 
                                 qConLast, delT, dx.x());

      qConFrt  = rho_tmp[t] * vel[t].y();
      qConLast = rho_tmp[b] * vel[b].y();
      y_conv = computeConvection(nuy[t], nuy[f], nuy[b],
                                 rho_tmp[t], rho_tmp[f], rho_tmp[b], 
                                 qConFrt, qConLast, delT, dx.y());

      rho_CC[f] = rho_tmp[f] - delT * (d1_z[f] + x_conv + y_conv);
     // rho_CC[f] = rho_tmp[f] - delT * d1_x[f] ;
  #if ON
        cout.setf(ios::scientific,ios::floatfield);
        cout.precision(16);
        cout << "rho_CC,rho_tmp, d1_z, dt, y_conv" << f << "= " << rho_CC[f] << ",";
        cout << rho_tmp[f] << "," << d1_z[f] << "," << delT << y_conv << endl;  
  #endif    
     } // end of j loop
  } //end of i loop
     
  //__________________________________________________________
  //  E D G E      right-front
  for(int j = low.y()+1; j < hi_y; j++) {
    r  =  IntVector(hi_x,   j,   hi_z);
    t  =  IntVector(hi_x,   j+1, hi_z);
    b  =  IntVector(hi_x,   j-1, hi_z);
    qConFrt  = rho_tmp[t] * vel[t].y();
    qConLast = rho_tmp[b] * vel[b].y();
    y_conv = computeConvection(nuy[t], nuy[r], nuy[b],
                               rho_tmp[t], rho_tmp[r], rho_tmp[b], 
                               qConFrt, qConLast, delT, dx.y());

    rho_CC[r] = rho_tmp[r] - delT * (d1_x[r] + d1_z[r] + y_conv);
  } 
  //__________________________________________________________
  //  E D G E      left-front
  for(int j = low.y()+1; j < hi_y; j++) {
    l  =  IntVector(low.x(),   j,   hi_z);
    t  =  IntVector(low.x(),   j+1, hi_z);
    b  =  IntVector(low.x(),   j-1, hi_z);
    qConFrt  = rho_tmp[t] * vel[t].y();
    qConLast = rho_tmp[b] * vel[b].y();
    y_conv = computeConvection(nuy[t], nuy[l], nuy[b],
                               rho_tmp[t], rho_tmp[l], rho_tmp[b], 
                               qConFrt, qConLast, delT, dx.y());

    rho_CC[l] = rho_tmp[l] - delT * (d1_x[l] + d1_z[l] + y_conv);

  } //end of j loop

  //_______________________________________________________
  //    E D G E   top-front
  for(int i = low.x()+1; i < hi_x; i++) {
    r  =  IntVector(i+1, hi_y,   hi_z);
    l  =  IntVector(i-1, hi_y,   hi_z);
    t  =  IntVector(i,   hi_y,   hi_z);
    qConFrt  = rho_tmp[r] * vel[r].x();
    qConLast = rho_tmp[l] * vel[l].x();
    x_conv = computeConvection(nux[r], nux[t], nux[l],
                           rho_tmp[r], rho_tmp[t], rho_tmp[l], 
                            qConFrt, qConLast, delT, dx.x());
    rho_CC[t] = rho_tmp[t] - delT * (d1_y[t] + d1_z[t] + x_conv);
  }   
  //_______________________________________________________
  //    E D G E    bottom-front
  for(int i = low.x()+1; i < hi_x; i++) {
    r  =  IntVector(i+1, low.y(), hi_z);
    l  =  IntVector(i-1, low.y(), hi_z);
    b  =  IntVector(i,   low.y(), hi_z);

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
   crn[0] = IntVector(hi_x,    low.y(), hi_z);      // right-bottom-front
   crn[1] = IntVector(hi_x,    hi_y,    hi_z);      // right-top-front    
   crn[2] = IntVector(low.x(), hi_y,    hi_z);      // left-top-front   
   crn[3] = IntVector(low.x(), low.y(), hi_z);      // left-bottom-front 

   for( int corner = 0; corner < 4; corner ++ ) {
     IntVector c = crn[corner];
     rho_CC[c] = rho_tmp[c] - delT * (d1_x[c] + d1_y[c] + d1_z[c]);
   }                
} 

/*_________________________________________________________________
 Function~ zMinusDensityLODI--
 Purpose~  Compute density in boundary cells on the z_minus face
___________________________________________________________________*/
void zMinusDensityLODI(CCVariable<double>& rho_CC,
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
  //cout << " I am in zMinusDensityLODI" << endl;    
  IntVector low,hi,r,l,t,b,f,bk;

  low = vel.getLowIndex();
  hi  = vel.getHighIndex();
  int hi_x = hi.x() - 1;
  int hi_y = hi.y() - 1;

  double x_conv, y_conv;
  double qConFrt,qConLast;
  //___________________________________________________________________
  // Compute the density on the area of low.x() < i < hi_x, low.y() < j < hi_y
  // and  k = low.z() 
  for(int i = low.x()+1; i < hi_x; i++) {
    for(int j = low.y()+1; j < hi_y; j++) {
      r  =  IntVector(i+1, j,   low.z());
      l  =  IntVector(i-1, j,   low.z());
      t  =  IntVector(i,   j+1, low.z());
      b  =  IntVector(i,   j-1, low.z());
      f  =  IntVector(i,   j,   low.z());

      qConFrt  = rho_tmp[r] * vel[r].x();
      qConLast = rho_tmp[l] * vel[l].x();
      x_conv = computeConvection(nux[r], nux[f], nux[l], 
                                 rho_tmp[r], rho_tmp[f], rho_tmp[l], qConFrt, 
                                 qConLast, delT, dx.x());

      qConFrt  = rho_tmp[t] * vel[t].y();
      qConLast = rho_tmp[b] * vel[b].y();
      y_conv = computeConvection(nuy[t], nuy[f], nuy[b],
                                 rho_tmp[t], rho_tmp[f], rho_tmp[b], 
                                 qConFrt, qConLast, delT, dx.y());

      rho_CC[f] = rho_tmp[f] - delT * (d1_z[f] + x_conv + y_conv);
     // rho_CC[f] = rho_tmp[f] - delT * d1_x[f] ;
#if ON
      cout.setf(ios::scientific,ios::floatfield);
      cout.precision(16);
      cout << "rho_CC,rho_tmp, d1_z, dt, x,y_conv" << f << "= " << rho_CC[f] << ",";
      cout << rho_tmp[f] << "," << d1_z[f] << "," << delT << x_conv << "," << y_conv << endl;
#endif    
    } 
  } 
  //__________________________________________________________
  //  E D G E     right- back
  for(int j = low.y()+1; j < hi_y; j++) {
    r  =  IntVector(hi_x,   j,   low.z());
    t  =  IntVector(hi_x,   j+1, low.z());
    b  =  IntVector(hi_x,   j-1, low.z());

    qConFrt  = rho_tmp[t] * vel[t].y();
    qConLast = rho_tmp[b] * vel[b].y();
    y_conv = computeConvection(nuy[t], nuy[r], nuy[b],
                               rho_tmp[t], rho_tmp[r], rho_tmp[b], 
                               qConFrt, qConLast, delT, dx.y());

    rho_CC[r] = rho_tmp[r] - delT * (d1_x[r] + d1_z[r] + y_conv);
  }
  //__________________________________________________________
  //  E D G E       left- back
  for(int j = low.y()+1; j < hi_y; j++) {
     l  =  IntVector(low.x(), j,   low.z());  
     t  =  IntVector(low.x(), j+1, low.z());  
     b  =  IntVector(low.x(), j-1, low.z());  

     qConFrt  = rho_tmp[t] * vel[t].y();
     qConLast = rho_tmp[b] * vel[b].y();
     y_conv = computeConvection(nuy[t], nuy[l], nuy[b],
                                rho_tmp[t], rho_tmp[l], rho_tmp[b], 
                                qConFrt, qConLast, delT, dx.y());

     rho_CC[l] = rho_tmp[l] - delT * (d1_x[l] + d1_z[l] + y_conv);
  } 
  //_______________________________________________________
  //    E D G E     top-back
  for(int i = low.x()+1; i < hi_x; i++) {
    r  =  IntVector(i+1, hi_y,   low.z());
    l  =  IntVector(i-1, hi_y,   low.z());
    t  =  IntVector(i,   hi_y,   low.z());

    qConFrt  = rho_tmp[r] * vel[r].x();
    qConLast = rho_tmp[l] * vel[l].x();
    x_conv = computeConvection(nux[r], nux[t], nux[l],
                             rho_tmp[r], rho_tmp[t], rho_tmp[l], 
                              qConFrt, qConLast, delT, dx.x());
    rho_CC[t] = rho_tmp[t] - delT * (d1_y[t] + d1_z[t] + x_conv);
  } 

  //_______________________________________________________
  //    E D G E     bottom-back
  for(int i = low.x()+1; i < hi_x; i++) {
    r  =  IntVector(i+1, low.y(), low.z());
    l  =  IntVector(i-1, low.y(), low.z());
    b  =  IntVector(i,   low.y(), low.z());
    
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
  crn[0] = IntVector(hi_x,    low.y(), low.z());     // right-bottom-back
  crn[1] = IntVector(hi_x,    hi_y,    low.z());     // right-top-back   
  crn[2] = IntVector(low.x(), hi_y,    low.z());     // left-top-back   
  crn[3] = IntVector(low.x(), low.y(), low.z());     // left-bottom-back 

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
  cout_doing << " I am in fillFaceDensityLODI" << endl;    
  switch(face) { 
    case Patch::xplus:
    { 
     xPlusDensityLODI(rho_CC,d1_x,d1_y,d1_z,nux,nuy,nuz,rho_tmp,vel,delT,dx);
    } 
    break;

    case Patch::xminus:
    { 
     xMinusDensityLODI(rho_CC,d1_x,d1_y,d1_z,nux,nuy,nuz,rho_tmp,vel,delT,dx);
    } 
    break;

    case Patch::yplus:
    { 
     yPlusDensityLODI(rho_CC,d1_x,d1_y,d1_z,nux,nuy,nuz,rho_tmp,vel,delT,dx);
    } 
    break;

    case Patch::yminus:
    { 
     yMinusDensityLODI(rho_CC,d1_x,d1_y,d1_z,nux,nuy,nuz,rho_tmp,vel,delT,dx);
    } 
    break;

    case Patch::zplus:
    { 
     zPlusDensityLODI(rho_CC,d1_x,d1_y,d1_z,nux,nuy,nuz,rho_tmp,vel,delT,dx);
    } 
    break;

    case Patch::zminus:
    {
     zMinusDensityLODI(rho_CC,d1_x,d1_y,d1_z,nux,nuy,nuz,rho_tmp,vel,delT,dx);
    } 
    break;

    default:
    break;
   }
   cout_doing << " I finished fillFaceDensityLODI" << endl; 
}

/*_________________________________________________________________
 Function~ xPlusVelLODI--
 Purpose~  Compute velocity in boundary cells on x_plus face
___________________________________________________________________*/
void xPlusVelLODI(CCVariable<Vector>& vel_CC,
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
   cout_doing << " I am in xPlusVelLODI" << endl;
   IntVector low,hi,r,l,t,b,f,bk;

   low = vel.getLowIndex();
   hi  = vel.getHighIndex();
   int hi_x = hi.x() - 1;
   int hi_y = hi.y() - 1;
   int hi_z = hi.z() - 1;

   double y_conv, z_conv;
   double uVel, vVel, wVel;
   double qConFrt,qConLast,qFrt,qMid,qLast;

   for(int j = low.y()+1; j < hi_y; j++) {
     for(int k = low.z()+1; k < hi_z; k++) {
       r  =  IntVector(hi_x,  j,   k);
       t  =  IntVector(hi_x,  j+1, k);
       b  =  IntVector(hi_x,  j-1, k);
       f  =  IntVector(hi_x,  j,   k+1);
       bk =  IntVector(hi_x,  j,   k-1);       

       //__________________________________
       //         X   V E L O C I T Y              
       qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].x();
       qConLast = rho_tmp[b] * vel[b].y() * vel[b].x();
       qFrt     = rho_tmp[t] * vel[t].x();
       qMid     = rho_tmp[r] * vel[r].x();
       qLast    = rho_tmp[b] * vel[b].x();
       y_conv   = computeConvection(nuy[t], nuy[r], nuy[b], 
                                  qFrt, qMid, qLast, qConFrt, 
                                  qConLast, delT, dx.y());

       qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].x();
       qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].x();
       qFrt     = rho_tmp[f]  * vel[f].x();
       qLast    = rho_tmp[bk] * vel[bk].x();

       z_conv   = computeConvection(nuz[f], nuz[r], nuz[bk],
                                    qFrt, qMid, qLast, qConFrt, 
				    qConLast, delT, dx.z());       
       uVel     = (qMid - delT * (d1_x[r] * vel[r].x() 
                +  rho_tmp[r] * d3_x[r] + y_conv + z_conv ))/rho_tmp[r];   
      //__________________________________
      //         Y   V E L O C I T Y     
      qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].y() + p[t];
      qConLast = rho_tmp[b] * vel[b].y() * vel[b].y() + p[b];
      qFrt     = rho_tmp[t] * vel[t].y();
      qMid     = rho_tmp[r] * vel[r].y();
      qLast    = rho_tmp[b] * vel[b].y();
      y_conv   = computeConvection(nuy[t], nuy[r], nuy[b], 
                                   qFrt, qMid, qLast, qConFrt, 
                                   qConLast, delT, dx.y());

      qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].y();
      qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].y();
      qFrt     = rho_tmp[f]  * vel[f].y();
      qLast    = rho_tmp[bk] * vel[bk].y();

      z_conv   = computeConvection(nuz[f], nuz[r], nuz[bk],
                                 qFrt, qMid, qLast, qConFrt, 
				 qConLast, delT, dx.z());        

      vVel = (qMid- delT * (d1_x[r] * vel[r].y()
           +  rho_tmp[r] * d4_x[r] + y_conv + z_conv))/rho_tmp[r];
      //__________________________________
      //         Z   V E L O C I T Y               
      qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].z();
      qConLast = rho_tmp[b] * vel[b].y() * vel[b].z();
      qFrt     = rho_tmp[t] * vel[t].z();
      qMid     = rho_tmp[r] * vel[r].z();
      qLast    = rho_tmp[b] * vel[b].z();
      y_conv   = computeConvection(nuy[t], nuy[r], nuy[b], 
                                 qFrt, qMid, qLast, qConFrt, 
                                 qConLast, delT, dx.y());

      qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].z()  + p[f];
      qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].z() + p[bk];
      qFrt     = rho_tmp[f]  * vel[f].z();
      qLast    = rho_tmp[bk] * vel[bk].z();

      z_conv = computeConvection(nuz[f], nuz[r], nuz[bk],
                                 qFrt, qMid, qLast, qConFrt, 
				 qConLast, delT, dx.z());        


      wVel = (qMid - delT * (d1_x[r] * vel[r].z() + rho_tmp[r] * d5_x[r]
                   + y_conv + z_conv))/rho_tmp[r];

      vel_CC[r] = Vector(uVel, vVel, wVel);
    } // end of j loop
  } //end of k loop
  //__________________________________________________________
  //  E D G E        right-bottom
  for(int k = low.z()+1; k < hi_z; k++) {
    r  =  IntVector(hi_x, low.y(), k);
    f  =  IntVector(hi_x, low.y(), k+1);
    bk =  IntVector(hi_x, low.y(), k-1);

    //__________________________________
    //         X   V E L O C I T Y                          
    qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].x();
    qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].x();
    qFrt     = rho_tmp[f]  * vel[f].x();
    qMid     = rho_tmp[r]  * vel[r].x();
    qLast    = rho_tmp[bk] * vel[bk].x();

    z_conv = computeConvection(nuz[f], nuz[r], nuz[bk],
                               qFrt, qMid, qLast, 
                               qConFrt,qConLast, delT, dx.z());       

    uVel = (qMid - delT * ((d1_x[r] + d1_y[r]) * vel[r].x() + 
                        +  (d3_x[r] + d4_y[r]) * rho_tmp[r] 
                        +   z_conv))/rho_tmp[r];

    //__________________________________
    //         Y   V E L O C I T Y
    qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].y();
    qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].y();
    qFrt     = rho_tmp[f]  * vel[f].y();
    qMid     = rho_tmp[r]  * vel[r].y();
    qLast    = rho_tmp[bk] * vel[bk].y();

    z_conv = computeConvection(nuz[f], nuz[r], nuz[bk],
                 qFrt, qMid, qLast, qConFrt,qConLast, delT, dx.z()); 

    vVel = (qMid - delT * ((d1_x[r] + d1_y[r]) * vel[r].y() 
                        +  (d4_x[r] + d3_y[r]) * rho_tmp[r] + z_conv))
                            /rho_tmp[r];

    //__________________________________
    //         Z   V E L O C I T Y
    qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].z()  + p[f];
    qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].z() + p[bk];
    qFrt     = rho_tmp[f]  * vel[f].z();
    qMid     = rho_tmp[r]  * vel[r].z();
    qLast    = rho_tmp[bk] * vel[bk].z();

    z_conv = computeConvection(nuz[f], nuz[r], nuz[bk],
                    qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.z());

    wVel = (qMid - delT * ((d1_x[r] + d1_y[r]) * vel[r].z() 
                        +  (d5_x[r] + d5_y[r]) * rho_tmp[r]  
                        + z_conv))/rho_tmp[r];
    vel_CC[r] = Vector(uVel, vVel, wVel);

  } //end of k loop
  //_______________________________________________________
  //    E D G E     right-top
  for(int k = low.z()+1; k < hi_z; k++) {

    r  =  IntVector(hi_x,   hi_y,   k);
    f  =  IntVector(hi_x,   hi_y,   k+1);
    bk =  IntVector(hi_x,   hi_y,   k-1);

    //__________________________________
    //         X   V E L O C I T Y                         
    qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].x();
    qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].x();
    qFrt     = rho_tmp[f]  * vel[f].x();
    qMid     = rho_tmp[r]  * vel[r].x();
    qLast    = rho_tmp[bk] * vel[bk].x();

    z_conv = computeConvection(nuz[f], nuz[r], nuz[bk],
                               qFrt, qMid, qLast, 
                           qConFrt, qConLast, delT, dx.z());       

    uVel = (qMid - delT * ((d1_x[r] + d1_y[r]) * vel[r].x() + 
                        +  (d3_x[r] + d4_y[r]) * rho_tmp[r] 
                         +   z_conv))/rho_tmp[r];
    //__________________________________
    //         Y   V E L O C I T Y            
    qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].y();
    qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].y();
    qFrt     = rho_tmp[f]  * vel[f].y();
    qMid     = rho_tmp[r]  * vel[r].y();
    qLast    = rho_tmp[bk] * vel[bk].y();

    z_conv = computeConvection(nuz[f], nuz[r], nuz[bk],
                   qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.z()); 

    vVel = (qMid - delT * ((d1_x[r] + d1_y[r]) * vel[r].y() 
                        +  (d4_x[r] + d3_y[r]) * rho_tmp[r] + z_conv))
                             /rho_tmp[r];

    //__________________________________
    //         Z   V E L O C I T Y
    qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].z()  + p[f];
    qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].z() + p[bk];
    qFrt     = rho_tmp[f]  * vel[f].z();
    qMid     = rho_tmp[r]  * vel[r].z();
    qLast    = rho_tmp[bk] * vel[bk].z();

    z_conv = computeConvection(nuz[f], nuz[r], nuz[bk],
                    qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.z());

    wVel = (qMid - delT * ((d1_x[r] + d1_y[r]) * vel[r].z() 
                        +  (d5_x[r] + d5_y[r]) * rho_tmp[r]  
                          + z_conv))/rho_tmp[r];
    vel_CC[r] = Vector(uVel, vVel, wVel);
  } //end of k loop
         
  //_____________________________________________________
  //    E D G E     right-back
  for(int j = low.y()+1; j < hi_y; j++) {
    r  =  IntVector(hi_x,   j,   low.z());
    t  =  IntVector(hi_x,   j+1, low.z());
    b  =  IntVector(hi_x,   j-1, low.z());

    //__________________________________
    //         X   V E L O C I T Y                         
    qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].x();
    qConLast = rho_tmp[b] * vel[b].y() * vel[b].x();
    qFrt     = rho_tmp[t] * vel[t].x();
    qMid     = rho_tmp[r] * vel[r].x();
    qLast    = rho_tmp[b] * vel[b].x();

    y_conv = computeConvection(nuy[t], nuy[r], nuy[b],
                               qFrt, qMid, qLast, 
                           qConFrt, qConLast, delT, dx.y());       

    uVel = (qMid - delT * ((d1_x[r] + d1_z[r]) * vel[r].x() + 
                        +  (d3_x[r] + d4_z[r]) * rho_tmp[r] 
                        +   y_conv))/rho_tmp[r];

    //__________________________________
    //         Y   V E L O C I T Y ()
    qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].y() + p[t];
    qConLast = rho_tmp[b] * vel[b].y() * vel[b].y() + p[b];
    qFrt     = rho_tmp[t] * vel[t].y();
    qMid     = rho_tmp[r] * vel[r].y();
    qLast    = rho_tmp[b] * vel[b].y();

    y_conv = computeConvection(nuy[t], nuy[r], nuy[b],
                 qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.y()); 

    vVel = (qMid - delT * ((d1_x[r] + d1_z[r]) * vel[r].y() 
                        +  (d4_x[r] + d5_z[r]) * rho_tmp[r] + y_conv))
                             /rho_tmp[r];
    //__________________________________
    //         Z   V E L O C I T Y ()
    qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].z();
    qConLast = rho_tmp[b] * vel[b].y() * vel[b].z();
    qFrt     = rho_tmp[t] * vel[t].z();
    qMid     = rho_tmp[r] * vel[r].z();
    qLast    = rho_tmp[b] * vel[b].z();

    y_conv = computeConvection(nuy[t], nuy[r], nuy[b],
                    qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.y());

    wVel = (qMid - delT * ((d1_x[r] + d1_z[r]) * vel[r].z() 
                        +  (d5_x[r] + d3_z[r]) * rho_tmp[r]  
                        + y_conv))/rho_tmp[r];
    vel_CC[r] = Vector(uVel, vVel, wVel);
  }

  //_________________________________________________
  // E D G E    right-front
  for(int j = low.y()+1; j < hi_y; j++) {
    r  =  IntVector(hi_x,   j,   hi_z);
    t  =  IntVector(hi_x,   j+1, hi_z);
    b  =  IntVector(hi_x,   j-1, hi_z);

    //__________________________________
    //         X   V E L O C I T Y ()                          
    qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].x();
    qConLast = rho_tmp[b] * vel[b].y() * vel[b].x();
    qFrt     = rho_tmp[t] * vel[t].x();
    qMid     = rho_tmp[r] * vel[r].x();
    qLast    = rho_tmp[b] * vel[b].x();

    y_conv = computeConvection(nuy[t], nuy[r], nuy[b],
                               qFrt, qMid, qLast, 
                               qConFrt, qConLast, delT, dx.y());       

    uVel = (qMid - delT * ((d1_x[r] + d1_z[r]) * vel[r].x() + 
                        +  (d3_x[r] + d4_z[r]) * rho_tmp[r] 
                        +   y_conv))/rho_tmp[r];

    //__________________________________
    //         Y   V E L O C I T Y ()
    qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].y() + p[t];
    qConLast = rho_tmp[b] * vel[b].y() * vel[b].y() + p[b];
    qFrt     = rho_tmp[t] * vel[t].y();
    qMid     = rho_tmp[r] * vel[r].y();
    qLast    = rho_tmp[b] * vel[b].y();

    y_conv = computeConvection(nuy[t], nuy[r], nuy[b],
                   qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.y()); 

    vVel = (qMid - delT * ((d1_x[r] + d1_z[r]) * vel[r].y() 
                        +  (d4_x[r] + d5_z[r]) * rho_tmp[r] + y_conv))
                             /rho_tmp[r];
#ifdef ON
    cout << "vVel, y_conv" << r << "= " << vVel
          << "," << y_conv  << endl;
    cout << "p" << t << "= " << p[t] << "," << "p" << b << "= " << p[b] << endl; 
#endif

    //__________________________________
    //         Z   V E L O C I T Y ()
    qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].z();
    qConLast = rho_tmp[b] * vel[b].y() * vel[b].z();
    qFrt     = rho_tmp[t] * vel[t].z();
    qMid     = rho_tmp[r] * vel[r].z();
    qLast    = rho_tmp[b] * vel[b].z();

    y_conv = computeConvection(nuy[t], nuy[r], nuy[b],
                    qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.y());

    wVel = (qMid - delT * ((d1_x[r] + d1_z[r]) * vel[r].z() 
                        +  (d5_x[r] + d3_z[r]) * rho_tmp[r]  
                        + y_conv))/rho_tmp[r];
    vel_CC[r] = Vector(uVel, vVel, wVel);        
  } 
        
   //________________________________________________________
   // C O R N E R S    
   vector<IntVector> crn(4);
   crn[0] = IntVector(hi_x, low.y(), hi_z);     // right-bottom-front
   crn[1] = IntVector(hi_x, hi_y,    hi_z);     // right-top-front   
   crn[2] = IntVector(hi_x, hi_y,    low.z());  // right-top-back  
   crn[3] = IntVector(hi_x, low.y(), low.z());  // right-bottom-back 

   for( int corner = 0; corner < 4; corner ++ ) {
     IntVector c = crn[corner];
     uVel = vel[r].x() - delT * ((d1_x[c] + d1_y[c] + d1_z[c]) * vel[c].x()  
                              +  (d3_x[c] + d4_y[c] + d4_z[c]) * rho_tmp[c]) 
                         / rho_tmp[c];

     vVel = vel[c].y() - delT * ((d1_x[c] + d1_y[c] + d1_z[c]) * vel[c].y() 
                              +  (d4_x[c] + d3_y[c] + d5_z[c]) * rho_tmp[c])
                         / rho_tmp[c];

     wVel = vel[c].z() - delT * ((d1_x[c] + d1_y[c] + d1_z[c]) * vel[c].z() 
                              +  (d5_x[c] + d5_y[c] + d3_z[c]) * rho_tmp[c])
                         / rho_tmp[c];
     vel_CC[r] = Vector(uVel, vVel, wVel);
   }
} //end of the function xPlusVelLODI()

/*_________________________________________________________________
 Function~ xMinusVelLODI--
 Purpose~  Compute velocity in boundary cells on x_minus face
___________________________________________________________________*/
void xMinusVelLODI(CCVariable<Vector>& vel_CC,
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
  cout_doing << " I am in xMinusVelLODI" << endl; 
  IntVector low,hi,r,l,t,b,f,bk;

  low = vel.getLowIndex();
  hi  = vel.getHighIndex();
  int hi_y = hi.y() - 1;
  int hi_z = hi.z() - 1;

  double uVel, vVel, wVel;
  double y_conv, z_conv;
  double qConFrt,qConLast,qFrt,qMid,qLast;
  
  for(int j = low.y()+1; j < hi_y; j++) {
    for(int k = low.z()+1; k < hi_z; k++) {
      r  =  IntVector(low.x(),  j,   k);
      t  =  IntVector(low.x(),  j+1, k);
      b  =  IntVector(low.x(),  j-1, k);
      f  =  IntVector(low.x(),  j,   k+1);
      bk =  IntVector(low.x(),  j,   k-1);  

      //__________________________________
      //         X   V E L O C I T Y ()
      qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].x();
      qConLast = rho_tmp[b] * vel[b].y() * vel[b].x();
      qFrt     = rho_tmp[t] * vel[t].x();
      qMid     = rho_tmp[r] * vel[r].x();
      qLast    = rho_tmp[b] * vel[b].x();
      y_conv   = computeConvection(nuy[t], nuy[r], nuy[b], 
                                 qFrt, qMid, qLast, qConFrt, 
                                 qConLast, delT, dx.y());

      qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].x();
      qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].x();
      qFrt     = rho_tmp[f]  * vel[f].x();
      qLast    = rho_tmp[bk] * vel[bk].x();

      z_conv = computeConvection(nuz[f], nuz[r], nuz[bk],
                               qFrt, qMid, qLast, qConFrt, 
			      qConLast, delT, dx.z());       
      uVel = (qMid - delT * (d1_x[r] * vel[r].x() 
           +  rho_tmp[r] * d3_x[r] + y_conv + z_conv ))/rho_tmp[r];	
                
      //__________________________________
      //         Y   V E L O C I T Y ()       
      qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].y() + p[t];
      qConLast = rho_tmp[b] * vel[b].y() * vel[b].y() + p[b];
      qFrt     = rho_tmp[t] * vel[t].y();
      qMid     = rho_tmp[r] * vel[r].y();
      qLast    = rho_tmp[b] * vel[b].y();
      y_conv   = computeConvection(nuy[t], nuy[r], nuy[b], 
                                 qFrt, qMid, qLast, qConFrt, 
                                 qConLast, delT, dx.y());

      qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].y();
      qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].y();
      qFrt     = rho_tmp[f]  * vel[f].y();
      qLast    = rho_tmp[bk] * vel[bk].y();

      z_conv = computeConvection(nuz[f], nuz[r], nuz[bk],
                                 qFrt, qMid, qLast, qConFrt, 
				 qConLast, delT, dx.z());        

      vVel = (qMid- delT * (d1_x[r] * vel[r].y()
                +  rho_tmp[r] * d4_x[r] + y_conv + z_conv))/rho_tmp[r];

      //__________________________________
      //         Z  V E L O C I T Y () 
      qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].z();
      qConLast = rho_tmp[b] * vel[b].y() * vel[b].z();
      qFrt     = rho_tmp[t] * vel[t].z();
      qMid     = rho_tmp[r] * vel[r].z();
      qLast    = rho_tmp[b] * vel[b].z();
      y_conv   = computeConvection(nuy[t], nuy[r], nuy[b], 
                                 qFrt, qMid, qLast, qConFrt, 
                                 qConLast, delT, dx.y());

      qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].z()  + p[f];
      qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].z() + p[bk];
      qFrt     = rho_tmp[f]  * vel[f].z();
      qLast    = rho_tmp[bk] * vel[bk].z();

      z_conv = computeConvection(nuz[f], nuz[r], nuz[bk],
                               qFrt, qMid, qLast, qConFrt, 
			      qConLast, delT, dx.z());        


      wVel = (qMid - delT * (d1_x[r] * vel[r].z() + rho_tmp[r] * d5_x[r]
                  + y_conv + z_conv))/rho_tmp[r];
      vel_CC[r] = Vector(uVel, vVel, wVel); 
    } // end of j loop
  } //end of k loop
  //__________________________________________________________
  //  E D G E    left-bottom
  for(int k = low.z()+1; k < hi_z; k++) {

    l  =  IntVector(low.x(),   low.y(),   k);
    f  =  IntVector(low.x(),   low.y(),   k+1);
    bk =  IntVector(low.x(),   low.y(),   k-1);
    //__________________________________
    //         X   V E L O C I T Y                         
    qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].x();
    qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].x();
    qFrt     = rho_tmp[f]  * vel[f].x();
    qMid     = rho_tmp[r]  * vel[r].x();
    qLast    = rho_tmp[bk] * vel[bk].x();

    z_conv = computeConvection(nuz[f], nuz[r], nuz[bk],
                               qFrt, qMid, qLast, 
                           qConFrt,qConLast, delT, dx.z());       

    uVel = (qMid - delT * ((d1_x[r] + d1_y[r]) * vel[r].x() + 
                        +  (d3_x[r] + d4_y[r]) * rho_tmp[r] 
                        +   z_conv))/rho_tmp[r];

    //__________________________________
    //         Y   V E L O C I T Y
    qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].y();
    qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].y();
    qFrt     = rho_tmp[f]  * vel[f].y();
    qMid     = rho_tmp[r]  * vel[r].y();
    qLast    = rho_tmp[bk] * vel[bk].y();

    z_conv = computeConvection(nuz[f], nuz[r], nuz[bk],
                   qFrt, qMid, qLast, qConFrt,qConLast, delT, dx.z()); 

    vVel = (qMid - delT * ((d1_x[r] + d1_y[r]) * vel[r].y() 
                        +  (d4_x[r] + d3_y[r]) * rho_tmp[r] + z_conv))
                            /rho_tmp[r];
                            
    //__________________________________
    //         Z   V E L O C I T Y
    qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].z()  + p[f];
    qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].z() + p[bk];
    qFrt     = rho_tmp[f]  * vel[f].z();
    qMid     = rho_tmp[r]  * vel[r].z();
    qLast    = rho_tmp[bk] * vel[bk].z();

    z_conv = computeConvection(nuz[f], nuz[r], nuz[bk],
                    qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.z());

    wVel = (qMid - delT * ((d1_x[r] + d1_y[r]) * vel[r].z() 
                        +  (d5_x[r] + d5_y[r]) * rho_tmp[r]  
                        + z_conv))/rho_tmp[r];
    vel_CC[r] = Vector(uVel, vVel, wVel);
  } //end of k loop

  //_______________________________________________________
  //    E D G E   left- top
  for(int k = low.z()+1; k < hi_z; k++) {
    r  =  IntVector(low.x(),   hi_y,   k);
    f  =  IntVector(low.x(),   hi_y,   k+1);
    bk =  IntVector(low.x(),   hi_y,   k-1);

    //__________________________________
    //         X   V E L O C I T Y                          
    qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].x();
    qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].x();
    qFrt     = rho_tmp[f]  * vel[f].x();
    qMid     = rho_tmp[r]  * vel[r].x();
    qLast    = rho_tmp[bk] * vel[bk].x();

    z_conv = computeConvection(nuz[f], nuz[r], nuz[bk],
                               qFrt, qMid, qLast, 
                           qConFrt,qConLast, delT, dx.z());       

    uVel = (qMid - delT * ((d1_x[r] + d1_y[r]) * vel[r].x() + 
                        +  (d3_x[r] + d4_y[r]) * rho_tmp[r] 
                        +   z_conv))/rho_tmp[r];

    //__________________________________
    //         Y   V E L O C I T Y 
    qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].y();
    qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].y();
    qFrt     = rho_tmp[f]  * vel[f].y();
    qMid     = rho_tmp[r]  * vel[r].y();
    qLast    = rho_tmp[bk] * vel[bk].y();

    z_conv = computeConvection(nuz[f], nuz[r], nuz[bk],
                   qFrt, qMid, qLast, qConFrt,qConLast, delT, dx.z()); 

    vVel = (qMid - delT * ((d1_x[r] + d1_y[r]) * vel[r].y() 
                        +  (d4_x[r] + d3_y[r]) * rho_tmp[r] + z_conv))
                           /rho_tmp[r];

    //__________________________________
    //         Z   V E L O C I T Y
    qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].z()  + p[f];
    qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].z() + p[bk];
    qFrt     = rho_tmp[f]  * vel[f].z();
    qMid     = rho_tmp[r]  * vel[r].z();
    qLast    = rho_tmp[bk] * vel[bk].z();

    z_conv = computeConvection(nuz[f], nuz[r], nuz[bk],
                    qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.z());

    wVel = (qMid - delT * ((d1_x[r] + d1_y[r]) * vel[r].z() 
                        +  (d5_x[r] + d5_y[r]) * rho_tmp[r]  
                        + z_conv))/rho_tmp[r];
    vel_CC[r] = Vector(uVel, vVel, wVel);
  } //end of k loop
  
         
  //_____________________________________________________
  //    E D G E   left-bottom
  for(int j = low.y()+1; j < hi_y; j++) {
    r  =  IntVector(low.x(),   j,   low.z());
    t  =  IntVector(low.x(),   j+1, low.z());
    b  =  IntVector(low.x(),   j-1, low.z());

    //__________________________________
    //         X   V E L O C I T Y                          
    qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].x();
    qConLast = rho_tmp[b] * vel[b].y() * vel[b].x();
    qFrt     = rho_tmp[t] * vel[t].x();
    qMid     = rho_tmp[r] * vel[r].x();
    qLast    = rho_tmp[b] * vel[b].x();

    y_conv = computeConvection(nuy[t], nuy[r], nuy[b],
                               qFrt, qMid, qLast, 
                           qConFrt, qConLast, delT, dx.y());       

    uVel = (qMid - delT * ((d1_x[r] + d1_z[r]) * vel[r].x() + 
                        +  (d3_x[r] + d4_z[r]) * rho_tmp[r] 
                        +   y_conv))/rho_tmp[r];
        
    //__________________________________
    //         Y   V E L O C I T Y
    qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].y() + p[t];
    qConLast = rho_tmp[b] * vel[b].y() * vel[b].y() + p[b];
    qFrt     = rho_tmp[t] * vel[t].y();
    qMid     = rho_tmp[r] * vel[r].y();
    qLast    = rho_tmp[b] * vel[b].y();

    y_conv = computeConvection(nuy[t], nuy[r], nuy[b],
                 qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.y()); 

    vVel = (qMid - delT * ((d1_x[r] + d1_z[r]) * vel[r].y() 
                        +  (d4_x[r] + d5_z[r]) * rho_tmp[r] + y_conv))
                             /rho_tmp[r];

    //__________________________________
    //         Z   V E L O C I T Y
    qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].z();
    qConLast = rho_tmp[b] * vel[b].y() * vel[b].z();
    qFrt     = rho_tmp[t] * vel[t].z();
    qMid     = rho_tmp[r] * vel[r].z();
    qLast    = rho_tmp[b] * vel[b].z();

    y_conv = computeConvection(nuy[t], nuy[r], nuy[b],
                    qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.y());

    wVel = (qMid - delT * ((d1_x[r] + d1_z[r]) * vel[r].z() 
                        +  (d5_x[r] + d3_z[r]) * rho_tmp[r]  
                          + y_conv))/rho_tmp[r];
    vel_CC[r] = Vector(uVel, vVel, wVel);
  }

  //_________________________________________________
  // E D G E    left-front
  for(int j = low.y()+1; j < hi_y; j++) {
    r  =  IntVector(low.x(),   j,   hi_z);
    t  =  IntVector(low.x(),   j+1, hi_z);
    b  =  IntVector(low.x(),   j-1, hi_z);

    //__________________________________
    //         X   V E L O C I T Y                          
    qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].x();
    qConLast = rho_tmp[b] * vel[b].y() * vel[b].x();
    qFrt     = rho_tmp[t] * vel[t].x();
    qMid     = rho_tmp[r] * vel[r].x();
    qLast    = rho_tmp[b] * vel[b].x();

    y_conv = computeConvection(nuy[t], nuy[r], nuy[b],
                               qFrt, qMid, qLast, 
                           qConFrt, qConLast, delT, dx.y());       

    uVel = (qMid - delT * ((d1_x[r] + d1_z[r]) * vel[r].x() + 
                        +  (d3_x[r] + d4_z[r]) * rho_tmp[r] 
                        +   y_conv))/rho_tmp[r];
        
    //__________________________________
    //         Y   V E L O C I T Y 
    qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].y() + p[t];
    qConLast = rho_tmp[b] * vel[b].y() * vel[b].y() + p[b];
    qFrt     = rho_tmp[t] * vel[t].y();
    qMid     = rho_tmp[r] * vel[r].y();
    qLast    = rho_tmp[b] * vel[b].y();

    y_conv = computeConvection(nuy[t], nuy[r], nuy[b],
                 qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.y()); 

    vVel = (qMid - delT * ((d1_x[r] + d1_z[r]) * vel[r].y() 
                        +  (d4_x[r] + d5_z[r]) * rho_tmp[r] + y_conv))
                             /rho_tmp[r];

    //__________________________________
    //         Z   V E L O C I T Y 
    qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].z();
    qConLast = rho_tmp[b] * vel[b].y() * vel[b].z();
    qFrt     = rho_tmp[t] * vel[t].z();
    qMid     = rho_tmp[r] * vel[r].z();
    qLast    = rho_tmp[b] * vel[b].z();

    y_conv = computeConvection(nuy[t], nuy[r], nuy[b],
                    qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.y());

    wVel = (qMid - delT * ((d1_x[r] + d1_z[r]) * vel[r].z() 
                        +  (d5_x[r] + d3_z[r]) * rho_tmp[r]  
                        + y_conv))/rho_tmp[r];
    vel_CC[r] = Vector(uVel, vVel, wVel);
  } 
  //________________________________________________________
  // C O R N E R S    
  vector<IntVector> crn(4);
  crn[0] = IntVector(low.x(), low.y(), hi_z);     // left-bottom-front
  crn[1] = IntVector(low.x(), hi_y,    hi_z);     // left-top-front   
  crn[2] = IntVector(low.x(), hi_y,    low.z());  // left-top-back  
  crn[3] = IntVector(low.x(), low.y(), low.z());  // left-bottom-back 

  for( int corner = 0; corner < 4; corner ++ ) {
    IntVector c = crn[corner];
     uVel = vel[r].x() - delT * ((d1_x[c] + d1_y[c] + d1_z[c]) * vel[c].x()  
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
} //end of the function xMinusVelLODI()

/*_________________________________________________________________
 Function~ yPlusVelLODI--
 Purpose~  Compute velocity in boundary cells on y_plus face 
___________________________________________________________________*/
void yPlusVelLODI(CCVariable<Vector>& vel_CC,
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
  cout << " I am in yPlusVelLODI" << endl;    
  IntVector low,hi,r,l,t,b,f,bk;

  low = vel.getLowIndex();
  hi  = vel.getHighIndex();
  int hi_x = hi.x() - 1;
  int hi_y = hi.y() - 1;
  int hi_z = hi.z() - 1;

  double uVel, vVel, wVel;
  double x_conv, z_conv;
  double qConFrt,qConLast,qFrt,qMid,qLast;
  
  for(int i = low.x()+1; i < hi_x; i++) {
    for(int k = low.z()+1; k < hi_z; k++) {
      r  =  IntVector(i+1,  hi_y,   k);
      l  =  IntVector(i-1,  hi_y,   k);
      t  =  IntVector(i,    hi_y,   k);
      f  =  IntVector(i,    hi_y,   k+1);
      bk =  IntVector(i,    hi_y,   k-1);       

      //__________________________________
      //         X   V E L O C I T Y 
      qConFrt  = rho_tmp[r] * vel[r].x() * vel[r].x() + p[r];
      qConLast = rho_tmp[l] * vel[l].x() * vel[l].x() + p[l];
      qFrt   = rho_tmp[r] * vel[r].x();
      qMid   = rho_tmp[t] * vel[t].x();
      qLast  = rho_tmp[l] * vel[l].x();
      x_conv = computeConvection(nux[r], nux[t], nux[l], 
                                 qFrt, qMid, qLast, qConFrt, 
                                 qConLast, delT, dx.x());

      qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].x();
      qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].x();
      qFrt     = rho_tmp[f]  * vel[f].x();
      qLast    = rho_tmp[bk] * vel[bk].x();

      z_conv = computeConvection(nuz[f], nuz[t], nuz[bk],
                              qFrt, qMid, qLast, qConFrt, 
			     qConLast, delT, dx.z());       
      uVel = (qMid - delT * (d1_y[t] * vel[t].x() 
           +  rho_tmp[t] * d4_y[t] + x_conv + z_conv ))/rho_tmp[t];
      //__________________________________
      //         Y   V E L O C I T Y        
      qConFrt  = rho_tmp[r] * vel[r].x() * vel[r].y();
      qConLast = rho_tmp[l] * vel[l].x() * vel[l].y();
      qFrt   = rho_tmp[r] * vel[r].y();
      qMid   = rho_tmp[t] * vel[t].y();
      qLast  = rho_tmp[l] * vel[l].y();
      x_conv = computeConvection(nux[r], nux[t], nux[l], 
                                 qFrt, qMid, qLast, qConFrt, 
                                 qConLast, delT, dx.x());

      qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].y();
      qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].y();
      qFrt     = rho_tmp[f]  * vel[f].y();
      qLast    = rho_tmp[bk] * vel[bk].y();

      z_conv = computeConvection(nuz[f], nuz[t], nuz[bk],
                                 qFrt, qMid, qLast, qConFrt, 
				 qConLast, delT, dx.z());        

      vVel = (qMid- delT * (d1_y[t] * vel[t].y()
               +  rho_tmp[t] * d3_y[t] + x_conv + z_conv))/rho_tmp[t];
      //__________________________________
      //         Z   V E L O C I T Y
      qConFrt  = rho_tmp[r] * vel[r].x() * vel[r].z();
      qConLast = rho_tmp[l] * vel[l].x() * vel[l].z();
      qFrt     = rho_tmp[r] * vel[r].z();
      qMid     = rho_tmp[t] * vel[t].z();
      qLast    = rho_tmp[l] * vel[l].z();
      x_conv   = computeConvection(nux[r], nux[t], nux[l], 
                                 qFrt, qMid, qLast, qConFrt, 
                                 qConLast, delT, dx.x());

      qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].z()  + p[f];
      qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].z() + p[bk];
      qFrt     = rho_tmp[f]  * vel[f].z();
      qLast    = rho_tmp[bk] * vel[bk].z();

      z_conv = computeConvection(nuz[f], nuz[t], nuz[bk],
                             qFrt, qMid, qLast, qConFrt, 
			    qConLast, delT, dx.z());        

      wVel = (qMid - delT * (d1_y[t] * vel[t].z() + rho_tmp[t] * d5_y[t]
                + x_conv + z_conv))/rho_tmp[t];

      vel_CC[t] = Vector(uVel, vVel, wVel);
    } // end of i loop
  } //end of k loop

  //__________________________________________________________
  //  E D G E    right-top
  for(int k = low.z()+1; k < hi_z; k++) {
    t  =  IntVector(hi_x,   hi_y,   k);
    f  =  IntVector(hi_x,   hi_y,   k+1);
    bk =  IntVector(hi_x,   hi_y,   k-1);

    //__________________________________
    //         X   V E L O C I T Y                         
    qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].x();
    qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].x();
    qFrt     = rho_tmp[f]  * vel[f].x();
    qMid     = rho_tmp[t]  * vel[t].x();
    qLast    = rho_tmp[bk] * vel[bk].x();

    z_conv = computeConvection(nuz[f], nuz[t], nuz[bk],
                               qFrt, qMid, qLast, 
                               qConFrt,qConLast, delT, dx.z());       

    uVel = (qMid - delT * ((d1_x[t] + d1_y[t]) * vel[t].x() + 
                        +  (d3_x[t] + d4_y[t]) * rho_tmp[t] 
                        +   z_conv))/rho_tmp[t];

    //__________________________________
    //         Y   V E L O C I T Y 
    qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].y();
    qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].y();
    qFrt     = rho_tmp[f]  * vel[f].y();
    qMid     = rho_tmp[t]  * vel[t].y();
    qLast    = rho_tmp[bk] * vel[bk].y();

    z_conv = computeConvection(nuz[f], nuz[t], nuz[bk],
                 qFrt, qMid, qLast, qConFrt,qConLast, delT, dx.z()); 

    vVel = (qMid - delT * ((d1_x[t] + d1_y[t]) * vel[t].y() 
                        +  (d4_x[t] + d3_y[t]) * rho_tmp[t] + z_conv))
                            /rho_tmp[t];
    //__________________________________
    //         Z   V E L O C I T Y 
    qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].z()  + p[f];
    qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].z() + p[bk];
    qFrt     = rho_tmp[f]  * vel[f].z();
    qMid     = rho_tmp[t]  * vel[t].z();
    qLast    = rho_tmp[bk] * vel[bk].z();

    z_conv = computeConvection(nuz[f], nuz[t], nuz[bk],
                    qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.z());

    wVel = (qMid - delT * ((d1_x[t] + d1_y[t]) * vel[t].z() 
                        +  (d5_x[t] + d5_y[t]) * rho_tmp[t]  
                        + z_conv))/rho_tmp[t];
    vel_CC[t] = Vector(uVel, vVel, wVel);
  } //end of k loop
     
         
  //_____________________________________________________
  //    E D G E   left-top
  for(int k = low.z()+1; k < hi_z; k++) {
    t  = IntVector(low.x(),   hi_y,   k);
    f  = IntVector(low.x(),   hi_y,   k+1);
    bk = IntVector(low.x(),   hi_y,   k-1);

    //__________________________________
    //         X   V E L O C I T Y                          
    qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].x();
    qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].x();
    qFrt     = rho_tmp[f]  * vel[f].x();
    qMid     = rho_tmp[t]  * vel[t].x();
    qLast    = rho_tmp[bk] * vel[bk].x();

    z_conv = computeConvection(nuz[f], nuz[t], nuz[bk],
                                qFrt, qMid, qLast, 
                            qConFrt, qConLast, delT, dx.z());       

    uVel = (qMid - delT * ((d1_x[t] + d1_y[t]) * vel[t].x() + 
                         +  (d3_x[t] + d4_y[t]) * rho_tmp[t] 
                         +   z_conv))/rho_tmp[t];
    //__________________________________
    //         Y   V E L O C I T Y 
    qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].y();
    qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].y();
    qFrt     = rho_tmp[f]  * vel[f].y();
    qMid     = rho_tmp[t]  * vel[t].y();
    qLast    = rho_tmp[bk] * vel[bk].y();

    z_conv = computeConvection(nuz[f], nuz[t], nuz[bk],
                qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.z()); 

    vVel = (qMid - delT * ((d1_x[t] + d1_y[t]) * vel[t].y() 
                       +  (d4_x[t] + d3_y[t]) * rho_tmp[t] + z_conv))
                            /rho_tmp[t];
    //__________________________________
    //         Z   V E L O C I T Y
    qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].z()  + p[f];
    qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].z() + p[bk];
    qFrt     = rho_tmp[f]  * vel[f].z();
    qMid     = rho_tmp[t]  * vel[t].z();
    qLast    = rho_tmp[bk] * vel[bk].z();

    z_conv = computeConvection(nuz[f], nuz[t], nuz[bk],
                    qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.z());

    wVel = (qMid - delT * ((d1_x[t] + d1_y[t]) * vel[t].z() 
                        +  (d5_x[t] + d5_y[t]) * rho_tmp[t]  
                        + z_conv))/rho_tmp[t];
    vel_CC[t] = Vector(uVel, vVel, wVel);

  } 
  //_______________________________________________________
  //    E D G E     top-front
  for(int i = low.x()+1; i < hi_x; i++) {
    r  =  IntVector(i+1, hi_y,    hi_z);
    l  =  IntVector(i-1, hi_y,    hi_z);
    t  =  IntVector(i,   hi_y,    hi_z);

    //__________________________________
    //         X   V E L O C I T Y                          
    qConFrt  = rho_tmp[r] * vel[r].x() * vel[r].x() + p[r];
    qConLast = rho_tmp[l] * vel[l].x() * vel[l].x() + p[l];
    qFrt     = rho_tmp[r] * vel[r].x();
    qMid     = rho_tmp[t] * vel[t].x();
    qLast    = rho_tmp[l] * vel[l].x();

    x_conv = computeConvection(nux[r], nux[t], nux[l],
                               qFrt, qMid, qLast, 
                           qConFrt, qConLast, delT, dx.x());       

    uVel = (qMid - delT * ((d1_y[t] + d1_z[t]) * vel[t].x() + 
                        +  (d4_y[t] + d4_z[t]) * rho_tmp[t] 
                        +   x_conv))/rho_tmp[t];
    //__________________________________
    //         Y   V E L O C I T Y 
    qConFrt  = rho_tmp[r] * vel[r].x() * vel[r].y();
    qConLast = rho_tmp[l] * vel[l].x() * vel[l].y();
    qFrt     = rho_tmp[r] * vel[r].y();
    qMid     = rho_tmp[t] * vel[t].y();
    qLast    = rho_tmp[l] * vel[l].y();

    x_conv = computeConvection(nux[r], nux[t], nux[l],
                 qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.x()); 

    vVel = (qMid - delT * ((d1_y[t] + d1_z[t]) * vel[t].y() 
                        +  (d3_y[t] + d5_z[t]) * rho_tmp[t] + x_conv))
                           /rho_tmp[t];
    //__________________________________
    //         Z   V E L O C I T Y
    qConFrt  = rho_tmp[r] * vel[r].x() * vel[r].z();
    qConLast = rho_tmp[l] * vel[l].x() * vel[l].z();
    qFrt     = rho_tmp[r] * vel[r].z();
    qMid     = rho_tmp[t] * vel[t].z();
    qLast    = rho_tmp[l] * vel[l].z();

    x_conv = computeConvection(nux[r], nux[t], nux[l],
                    qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.x());

    wVel = (qMid - delT * ((d1_y[t] + d1_z[t]) * vel[t].z() 
                        +  (d5_y[t] + d3_z[t]) * rho_tmp[t]  
                        + x_conv))/rho_tmp[t];
    vel_CC[t] = Vector(uVel, vVel, wVel);
  } //end of i loop
       
  //_________________________________________________
  // E D G E    top-back
  for(int i = low.x()+1; i < hi_x; i++) {
    r  =  IntVector(i+1, hi_y,   low.z());
    l  =  IntVector(i-1, hi_y,   low.z());
    t  =  IntVector(i,   hi_y,   low.z());

    //__________________________________
    //         X   V E L O C I T Y                          
    qConFrt  = rho_tmp[r] * vel[r].x() * vel[r].x() + p[r];
    qConLast = rho_tmp[l] * vel[l].x() * vel[l].x() + p[l];
    qFrt   = rho_tmp[r] * vel[r].x();
    qMid   = rho_tmp[t] * vel[t].x();
    qLast  = rho_tmp[l] * vel[l].x();

    x_conv = computeConvection(nux[r], nux[t], nux[l],
                              qFrt, qMid, qLast, 
                          qConFrt, qConLast, delT, dx.x());       

    uVel = (qMid - delT * ((d1_y[t] + d1_z[t]) * vel[t].x() + 
                        +  (d4_y[t] + d4_z[t]) * rho_tmp[t] 
                         +   x_conv))/rho_tmp[t];

    //__________________________________
    //         Y   V E L O C I T Y
    qConFrt  = rho_tmp[r] * vel[r].x() * vel[r].y();
    qConLast = rho_tmp[l] * vel[l].x() * vel[l].y();
    qFrt     = rho_tmp[r] * vel[r].y();
    qMid     = rho_tmp[t] * vel[t].y();
    qLast    = rho_tmp[l] * vel[l].y();

    x_conv = computeConvection(nux[r], nux[t], nux[l],
                 qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.x()); 

    vVel = (qMid - delT * ((d1_y[t] + d1_z[t]) * vel[t].y() 
                       +  (d3_y[t] + d5_z[t]) * rho_tmp[t] + x_conv))
                            /rho_tmp[t];

    //__________________________________
    //         Z   V E L O C I T Y
    qConFrt  = rho_tmp[r] * vel[r].x() * vel[r].z();
    qConLast = rho_tmp[l] * vel[l].x() * vel[l].z();
    qFrt     = rho_tmp[r] * vel[r].z();
    qMid     = rho_tmp[t] * vel[t].z();
    qLast    = rho_tmp[l] * vel[l].z();

    x_conv = computeConvection(nux[r], nux[t], nux[l],
                    qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.x());

    wVel = (qMid - delT * ((d1_y[t] + d1_z[t]) * vel[t].z() 
                        +  (d5_y[t] + d3_z[t]) * rho_tmp[t]  
                        + x_conv))/rho_tmp[t];
    vel_CC[t] = Vector(uVel, vVel, wVel);
  } 
  //________________________________________________________
  // C O R N E R S    
  vector<IntVector> crn(4);
  crn[0] = IntVector(hi_x,    hi_y, hi_z);     // right-top-front
  crn[1] = IntVector(low.x(), hi_y, hi_z);     // left-top-front   
  crn[2] = IntVector(low.x(), hi_y, low.z());  // left-top-back  
  crn[3] = IntVector(hi_x,    hi_y, low.z());  // right-top-back 

  for( int corner = 0; corner < 4; corner ++ ) {
    IntVector c = crn[corner];
    uVel = vel[r].x() - delT * ((d1_x[c] + d1_y[c] + d1_z[c]) * vel[c].x()  
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
} //end of the function yPlusVelLODI()

/*_________________________________________________________________
 Function~ yMinusVelLODI--
 Purpose~  Compute velocity in boundary cells on yMinus 
___________________________________________________________________*/
void yMinusVelLODI(CCVariable<Vector>& vel_CC,
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
  cout_doing << " I am in yMinusVelLODI" << endl;    
  IntVector low,hi,r,l,t,b,f,bk;

  low = vel.getLowIndex();
  hi  = vel.getHighIndex();
  int hi_x = hi.x() - 1;
  int hi_z = hi.z() - 1;

  double uVel, vVel, wVel;
  double x_conv, z_conv;
  double qConFrt,qConLast,qFrt,qMid,qLast;

   
  for(int i = low.x()+1; i < hi_x; i++) {
    for(int k = low.z()+1; k < hi_z; k++) {
      r  =  IntVector(i+1,  low.y(),   k);
      l  =  IntVector(i-1,  low.y(),   k);
      b  =  IntVector(i,    low.y(),   k);
      f  =  IntVector(i,    low.y(),   k+1);
      bk =  IntVector(i,    low.y(),   k-1);       

      //__________________________________
      //         X   V E L O C I T Y
      qConFrt  = rho_tmp[r] * vel[r].x() * vel[r].x() + p[r];
      qConLast = rho_tmp[l] * vel[l].x() * vel[l].x() + p[l];
      qFrt   = rho_tmp[r] * vel[r].x();
      qMid   = rho_tmp[b] * vel[b].x();
      qLast  = rho_tmp[l] * vel[l].x();
      x_conv = computeConvection(nux[r], nux[b], nux[l], 
                                 qFrt, qMid, qLast, qConFrt, 
                                 qConLast, delT, dx.x());

      qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].x();
      qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].x();
      qFrt     = rho_tmp[f]  * vel[f].x();
      qLast    = rho_tmp[bk] * vel[bk].x();

      z_conv = computeConvection(nuz[f], nuz[b], nuz[bk],
                                 qFrt, qMid, qLast, qConFrt, 
				 qConLast, delT, dx.z());       
      uVel = (qMid - delT * (d1_y[b] * vel[b].x() 
           +  rho_tmp[b] * d4_y[b] + x_conv + z_conv ))/rho_tmp[b];     
      //__________________________________
      //         Y   V E L O C I T Y        
      qConFrt  = rho_tmp[r] * vel[r].x() * vel[r].y();
      qConLast = rho_tmp[l] * vel[l].x() * vel[l].y();
      qFrt     = rho_tmp[r] * vel[r].y();
      qMid     = rho_tmp[b] * vel[b].y();
      qLast    = rho_tmp[l] * vel[l].y();
      x_conv   = computeConvection(nux[r], nux[b], nux[l], 
                                 qFrt, qMid, qLast, qConFrt, 
                                 qConLast, delT, dx.x());

      qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].y();
      qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].y();
      qFrt     = rho_tmp[f]  * vel[f].y();
      qLast    = rho_tmp[bk] * vel[bk].y();

      z_conv = computeConvection(nuz[f], nuz[b], nuz[bk],
                              qFrt, qMid, qLast, qConFrt, 
			     qConLast, delT, dx.z());        

      vVel = (qMid- delT * (d1_y[b] * vel[b].y()
                 +  rho_tmp[b] * d3_y[b] + x_conv + z_conv))/rho_tmp[b];
      //__________________________________
      //         Z   V E L O C I T Y
      qConFrt  = rho_tmp[r] * vel[r].x() * vel[r].z();
      qConLast = rho_tmp[l] * vel[l].x() * vel[l].z();
      qFrt     = rho_tmp[r] * vel[r].z();
      qMid     = rho_tmp[b] * vel[b].z();
      qLast    = rho_tmp[l] * vel[l].z();
      x_conv   = computeConvection(nux[r], nux[b], nux[l], 
                                 qFrt, qMid, qLast, qConFrt, 
                                 qConLast, delT, dx.x());

      qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].z()  + p[f];
      qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].z() + p[bk];
      qFrt     = rho_tmp[f]  * vel[f].z();
      qLast    = rho_tmp[bk] * vel[bk].z();

      z_conv = computeConvection(nuz[f], nuz[b], nuz[bk],
                                 qFrt, qMid, qLast, qConFrt, 
				 qConLast, delT, dx.z());        


      wVel = (qMid - delT * (d1_y[b] * vel[b].z() + rho_tmp[b] * d5_y[b]
                  + x_conv + z_conv))/rho_tmp[b];
      vel_CC[b] = Vector(uVel, vVel, wVel); 
    } // end of i loop
  } //end of k loop
  //__________________________________________________________
  //  E D G E      right-bottom
  for(int k = low.z()+1; k < hi_z; k++) {
    b  =  IntVector(hi_x,   low.y(),   k);
    f  =  IntVector(hi_x,   low.y(),   k+1);
    bk =  IntVector(hi_x,   low.y(),   k-1);

    //__________________________________
    //         X   V E L O C I T Y                          
    qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].x();
    qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].x();
    qFrt     = rho_tmp[f]  * vel[f].x();
    qMid     = rho_tmp[b]  * vel[b].x();
    qLast    = rho_tmp[bk] * vel[bk].x();

    z_conv = computeConvection(nuz[f], nuz[b], nuz[bk],
                               qFrt, qMid, qLast, 
                               qConFrt,qConLast, delT, dx.z());       

    uVel = (qMid - delT * ((d1_x[b] + d1_y[b]) * vel[b].x() + 
                        +  (d3_x[b] + d4_y[b]) * rho_tmp[b] 
                        +   z_conv))/rho_tmp[b];
        
    //__________________________________
    //         Y   V E L O C I T Y 
    qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].y();
    qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].y();
    qFrt     = rho_tmp[f]  * vel[f].y();
    qMid     = rho_tmp[b]  * vel[b].y();
    qLast    = rho_tmp[bk] * vel[bk].y();

    z_conv = computeConvection(nuz[f], nuz[b], nuz[bk],
                 qFrt, qMid, qLast, qConFrt,qConLast, delT, dx.z()); 

    vVel = (qMid - delT * ((d1_x[b] + d1_y[b]) * vel[b].y() 
                        +  (d4_x[b] + d3_y[b]) * rho_tmp[b] + z_conv))
                            /rho_tmp[b];
    //__________________________________
    //         Z   V E L O C I T Y 
    qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].z()  + p[f];
    qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].z() + p[bk];
    qFrt     = rho_tmp[f]  * vel[f].z();
    qMid     = rho_tmp[b]  * vel[b].z();
    qLast    = rho_tmp[bk] * vel[bk].z();

    z_conv = computeConvection(nuz[f], nuz[b], nuz[bk],
                    qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.z());

    wVel = (qMid - delT * ((d1_x[b] + d1_y[b]) * vel[b].z() 
                        +  (d5_x[b] + d5_y[b]) * rho_tmp[b]  
                        + z_conv))/rho_tmp[b];
    vel_CC[b] = Vector(uVel, vVel, wVel);
  } 
  //_____________________________________________________
  //    E D G E   left-bottom
  for(int k = low.z()+1; k < hi_z; k++) {
    b  = IntVector(low.x(),   low.y(),   k);
    f  = IntVector(low.x(),   low.y(),   k+1);
    bk = IntVector(low.x(),   low.y(),   k-1);

    //__________________________________
    //         X   V E L O C I T Y                           
    qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].x();
    qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].x();
    qFrt     = rho_tmp[f]  * vel[f].x();
    qMid     = rho_tmp[b]  * vel[b].x();
    qLast    = rho_tmp[bk] * vel[bk].x();

    z_conv = computeConvection(nuz[f], nuz[b], nuz[bk],
                               qFrt, qMid, qLast, 
                           qConFrt, qConLast, delT, dx.z());       

    uVel = (qMid - delT * ((d1_x[b] + d1_y[b]) * vel[b].x() + 
                        +  (d3_x[b] + d4_y[b]) * rho_tmp[b] 
                        +   z_conv))/rho_tmp[b];
    //__________________________________
    //         Y   V E L O C I T Y
    qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].y();
    qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].y();
    qFrt     = rho_tmp[f]  * vel[f].y();
    qMid     = rho_tmp[b]  * vel[b].y();
    qLast    = rho_tmp[bk] * vel[bk].y();

    z_conv = computeConvection(nuz[f], nuz[b], nuz[bk],
                 qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.z()); 

    vVel = (qMid - delT * ((d1_x[b] + d1_y[b]) * vel[b].y() 
                        +  (d4_x[b] + d3_y[b]) * rho_tmp[b] + z_conv))
                             /rho_tmp[b];

    //__________________________________
    //         Z   V E L O C I T Y
    qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].z()  + p[f];
    qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].z() + p[bk];
    qFrt     = rho_tmp[f]  * vel[f].z();
    qMid     = rho_tmp[b]  * vel[b].z();
    qLast    = rho_tmp[bk] * vel[bk].z();

    z_conv = computeConvection(nuz[f], nuz[b], nuz[bk],
                    qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.z());

    wVel = (qMid - delT * ((d1_x[b] + d1_y[b]) * vel[b].z() 
                        +  (d5_x[b] + d5_y[b]) * rho_tmp[b]  
                        + z_conv))/rho_tmp[b];
    vel_CC[b] = Vector(uVel, vVel, wVel);
  } 
  //_______________________________________________________
  //    E D G E   bottom-front
  for(int i = low.x()+1; i < hi_x; i++) {
    r  =  IntVector(i+1, low.y(), hi_z);
    l  =  IntVector(i-1, low.y(), hi_z);
    b  =  IntVector(i,   low.y(), hi_z);

    //__________________________________
    //         X   V E L O C I T Y                         
    qConFrt  = rho_tmp[r] * vel[r].x() * vel[r].x() + p[r];
    qConLast = rho_tmp[l] * vel[l].x() * vel[l].x() + p[l];
    qFrt   = rho_tmp[r] * vel[r].x();
    qMid   = rho_tmp[b] * vel[b].x();
    qLast  = rho_tmp[l] * vel[l].x();

    x_conv = computeConvection(nux[r], nux[b], nux[l],
                               qFrt, qMid, qLast, 
                           qConFrt, qConLast, delT, dx.x());       

    uVel = (qMid - delT * ((d1_y[b] + d1_z[b]) * vel[b].x() + 
                        +  (d4_y[b] + d4_z[b]) * rho_tmp[b] 
                         +   x_conv))/rho_tmp[b];
    //__________________________________
    //         Y   V E L O C I T Y
    qConFrt  = rho_tmp[r] * vel[r].x() * vel[r].y();
    qConLast = rho_tmp[l] * vel[l].x() * vel[l].y();
    qFrt     = rho_tmp[r] * vel[r].y();
    qMid     = rho_tmp[b] * vel[b].y();
    qLast    = rho_tmp[l] * vel[l].y();

    x_conv = computeConvection(nux[r], nux[b], nux[l],
                   qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.x()); 

    vVel = (qMid - delT * ((d1_y[b] + d1_z[b]) * vel[b].y() 
                        +  (d3_y[b] + d5_z[b]) * rho_tmp[b] + x_conv))
                             /rho_tmp[b];
    //__________________________________
    //         Z   V E L O C I T Y
    qConFrt  = rho_tmp[r] * vel[r].x() * vel[r].z();
    qConLast = rho_tmp[l] * vel[l].x() * vel[l].z();
    qFrt     = rho_tmp[r] * vel[r].z();
    qMid     = rho_tmp[b] * vel[b].z();
    qLast    = rho_tmp[l] * vel[l].z();

    x_conv = computeConvection(nux[r], nux[b], nux[l],
                    qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.x());

    wVel = (qMid - delT * ((d1_y[b] + d1_z[b]) * vel[b].z() 
                        +  (d5_y[b] + d3_z[b]) * rho_tmp[b]  
                          + x_conv))/rho_tmp[b];
    vel_CC[b] = Vector(uVel, vVel, wVel);
  } 
  
  //_________________________________________________
  // E D G E    bottom-back
  for(int i = low.x()+1; i < hi_x; i++) {
    r  =  IntVector(i+1, low.y(),   low.z());
    l  =  IntVector(i-1, low.y(),   low.z());
    b  =  IntVector(i,   low.y(),   low.z());

    //__________________________________
    //         X   V E L O C I T Y                          
    qConFrt  = rho_tmp[r] * vel[r].x() * vel[r].x() + p[r];
    qConLast = rho_tmp[l] * vel[l].x() * vel[l].x() + p[l];
    qFrt     = rho_tmp[r] * vel[r].x();
    qMid     = rho_tmp[b] * vel[b].x();
    qLast    = rho_tmp[l] * vel[l].x();

    x_conv = computeConvection(nux[r], nux[b], nux[l],
                               qFrt, qMid, qLast, 
                           qConFrt, qConLast, delT, dx.x());       

    uVel = (qMid - delT * ((d1_y[b] + d1_z[b]) * vel[b].x() + 
                        +  (d4_y[b] + d4_z[b]) * rho_tmp[b] 
                        +   x_conv))/rho_tmp[b];
    //__________________________________
    //         Y   V E L O C I T Y
    qConFrt  = rho_tmp[r] * vel[r].x() * vel[r].y();
    qConLast = rho_tmp[l] * vel[l].x() * vel[l].y();
    qFrt     = rho_tmp[r] * vel[r].y();
    qMid     = rho_tmp[b] * vel[b].y();
    qLast    = rho_tmp[l] * vel[l].y();

    x_conv = computeConvection(nux[r], nux[b], nux[l],
                qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.x()); 

    vVel = (qMid - delT * ((d1_y[b] + d1_z[b]) * vel[b].y() 
                        +  (d3_y[b] + d5_z[b]) * rho_tmp[b] + x_conv))
                            /rho_tmp[b];
    //__________________________________
    //         Z   V E L O C I T Y
    qConFrt  = rho_tmp[r] * vel[r].x() * vel[r].z();
    qConLast = rho_tmp[l] * vel[l].x() * vel[l].z();
    qFrt     = rho_tmp[r] * vel[r].z();
    qMid     = rho_tmp[b] * vel[b].z();
    qLast    = rho_tmp[l] * vel[l].z();

    x_conv = computeConvection(nux[r], nux[b], nux[l],
                   qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.x());

    wVel = (qMid - delT * ((d1_y[b] + d1_z[b]) * vel[b].z() 
                        +  (d5_y[b] + d3_z[b]) * rho_tmp[b]  
                        + x_conv))/rho_tmp[b];
    vel_CC[b] = Vector(uVel, vVel, wVel);
  }

  //________________________________________________________
  // C O R N E R S    
  vector<IntVector> crn(4);
  crn[0] = IntVector(hi_x,    low.y(), hi_z);     // right-bottom-front
  crn[1] = IntVector(low.x(), low.y(), hi_z);     // left-bottom-front   
  crn[2] = IntVector(low.x(), low.y(), low.z());  // left-bottom-back  
  crn[3] = IntVector(hi_x,    low.y(), low.z());  // right-bottom-back 

  for( int corner = 0; corner < 4; corner ++ ) {
    IntVector c = crn[corner];
     uVel = vel[r].x() - delT * ((d1_x[c] + d1_y[c] + d1_z[c]) * vel[c].x()  
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
} //end of the function yMinusVelLODI()


/*_________________________________________________________________
 Function~ zPlusVelLODI--
 Purpose~  Compute velocity in boundary cells on the z_plus face
___________________________________________________________________*/
void zPlusVelLODI(CCVariable<Vector>& vel_CC,
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
  cout_doing << " I am in zPlusvelocityLODI" << endl;    
  IntVector low,hi,r,l,t,b,f,bk;

  low = vel.getLowIndex();
  hi  = vel.getHighIndex();
  int hi_x = hi.x() - 1;
  int hi_y = hi.y() - 1;
  int hi_z = hi.z() - 1;

  double x_conv, y_conv;
  double uVel, vVel, wVel;
  double qConFrt,qConLast,qFrt,qMid,qLast;
         
  for(int i = low.x()+1; i < hi_x; i++) {
    for(int j = low.y()+1; j < hi_y; j++) {

      r  =  IntVector(i+1, j,   hi_z);
      l  =  IntVector(i-1, j,   hi_z);
      t  =  IntVector(i,   j+1, hi_z);
      b  =  IntVector(i,   j-1, hi_z);
      f  =  IntVector(i,   j,   hi_z);

      //__________________________________
      //         X   V E L O C I T Y
      qConFrt  = rho_tmp[r] * vel[r].x() * vel[r].x() + p[r];
      qConLast = rho_tmp[l] * vel[l].x() * vel[l].x() + p[l];
      qFrt     = rho_tmp[r] * vel[r].x();
      qMid     = rho_tmp[f] * vel[f].x();
      qLast    = rho_tmp[l] * vel[l].x();
      x_conv   = computeConvection(nux[r], nux[f], nux[l], 
                                 qFrt, qMid, qLast, qConFrt, 
                                 qConLast, delT, dx.x());

      qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].x();
      qConLast = rho_tmp[b] * vel[b].y() * vel[b].x();
      qFrt     = rho_tmp[t] * vel[t].x();
      qLast    = rho_tmp[b] * vel[b].x();

      y_conv = computeConvection(nuy[t], nuy[f], nuy[b],
                                 qFrt, qMid, qLast, qConFrt, 
				     qConLast, delT, dx.y());       
      uVel = (qMid - delT * (d1_z[f] * vel[f].x() 
           +  rho_tmp[f] * d4_z[f] + x_conv + y_conv))/rho_tmp[f];
      //__________________________________
      //         Y   V E L O C I T Y        
      qConFrt  = rho_tmp[r] * vel[r].x() * vel[r].y();
      qConLast = rho_tmp[l] * vel[l].x() * vel[l].y();
      qFrt     = rho_tmp[r] * vel[r].y();
      qMid     = rho_tmp[f] * vel[f].y();
      qLast    = rho_tmp[l] * vel[l].y();

      x_conv = computeConvection(nux[r], nux[f], nux[l],
                                 qFrt, qMid, qLast, qConFrt, 
				 qConLast, delT, dx.x()); 

      qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].y() + p[t];
      qConLast = rho_tmp[b] * vel[b].y() * vel[b].y() + p[b];
      qFrt     = rho_tmp[t] * vel[t].y();
      qLast    = rho_tmp[b] * vel[b].y();
      y_conv   = computeConvection(nuy[t], nuy[f], nuy[b], 
                                 qFrt, qMid, qLast, qConFrt, 
                                 qConLast, delT, dx.y());

      vVel = (qMid- delT * (d1_z[f] * vel[f].y()
                  +  rho_tmp[f] * d5_z[f] + x_conv + y_conv))/rho_tmp[f];
                        
      //__________________________________
      //         Z   V E L O C I T Y
      qConFrt  = rho_tmp[r] * vel[r].x() * vel[r].z();
      qConLast = rho_tmp[l] * vel[l].x() * vel[l].z() ;
      qFrt     = rho_tmp[r] * vel[r].z();
      qMid     = rho_tmp[f] * vel[f].z();
      qLast    = rho_tmp[l] * vel[l].z();
      x_conv   = computeConvection(nux[r], nux[f], nux[l],
                                 qFrt, qMid, qLast, qConFrt, 
				 qConLast, delT, dx.x());  

      qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].z();
      qConLast = rho_tmp[b] * vel[b].y() * vel[b].z();
      qFrt     = rho_tmp[t] * vel[t].z();
      qLast    = rho_tmp[b] * vel[b].z();
      y_conv   = computeConvection(nuy[t], nuy[f], nuy[b], 
                                   qFrt, qMid, qLast, qConFrt, 
                                   qConLast, delT, dx.y());

      wVel = (qMid - delT * (d1_z[f] * vel[f].z() + rho_tmp[f] * d3_z[f]
                  + x_conv + y_conv))/rho_tmp[f];
      vel_CC[f] = Vector(uVel, vVel, wVel);
    } // end of j loop
  } //end of i loop
    
	
  //__________________________________________________________
  //  E D G E    right-front
  for(int j = low.y()+1; j < hi_y; j++) {
    r  =  IntVector(hi_x,   j,   hi_z);
    t  =  IntVector(hi_x,   j+1, hi_z);
    b  =  IntVector(hi_x,   j-1, hi_z);

    //__________________________________
    //         X   V E L O C I T Y                         
    qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].x();
    qConLast = rho_tmp[b] * vel[b].y() * vel[b].x();
    qFrt     = rho_tmp[t] * vel[t].x();
    qMid     = rho_tmp[r] * vel[r].x();
    qLast    = rho_tmp[b] * vel[b].x();

    y_conv = computeConvection(nuy[t], nuy[r], nuy[b],
                               qFrt, qMid, qLast, 
                           qConFrt,qConLast, delT, dx.y());       

    uVel = (qMid - delT * ((d1_x[r] + d1_z[r]) * vel[r].x() + 
                        +  (d3_x[r] + d4_z[r]) * rho_tmp[r] 
                        +   y_conv))/rho_tmp[r];
    //__________________________________
    //         Y   V E L O C I T Y 
    qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].y() + p[t];
    qConLast = rho_tmp[b] * vel[b].y() * vel[b].y() + p[b];
    qFrt     = rho_tmp[t] * vel[t].y();
    qMid     = rho_tmp[r] * vel[r].y();
    qLast    = rho_tmp[b] * vel[b].y();

    y_conv = computeConvection(nuy[t], nuy[r], nuy[b],
                   qFrt, qMid, qLast, qConFrt,qConLast, delT, dx.y()); 

    vVel = (qMid - delT * ((d1_x[r] + d1_z[r]) * vel[r].y() 
                        +  (d4_x[r] + d5_z[r]) * rho_tmp[r] + y_conv))
                              /rho_tmp[r];
    //__________________________________
    //         Z   V E L O C I T Y
    qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].z();
    qConLast = rho_tmp[b] * vel[b].y() * vel[b].z();
    qFrt     = rho_tmp[t] * vel[t].z();
    qMid     = rho_tmp[r] * vel[r].z();
    qLast    = rho_tmp[b] * vel[b].z();

    y_conv = computeConvection(nuy[t], nuy[r], nuy[b],
                    qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.y());

    wVel = (qMid - delT * ((d1_x[r] + d1_z[r]) * vel[r].z() 
                        +  (d5_x[r] + d3_z[r]) * rho_tmp[r]  
                        + y_conv))/rho_tmp[r];

    vel_CC[r] = Vector(uVel, vVel, wVel);

  } //end of j loop

  //__________________________________________________________
  //  E D G E    left-front
  for(int j = low.y()+1; j < hi_y; j++) {
    l  =  IntVector(low.x(), j,   hi_z);
    t  =  IntVector(low.x(), j+1, hi_z);
    b  =  IntVector(low.x(), j-1, hi_z);

    //__________________________________
    //         X   V E L O C I T Y                         
    qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].x();
    qConLast = rho_tmp[b] * vel[b].y() * vel[b].x();
    qFrt     = rho_tmp[t] * vel[t].x();
    qMid     = rho_tmp[l] * vel[l].x();
    qLast    = rho_tmp[b] * vel[b].x();

    y_conv = computeConvection(nuy[t], nuy[l], nuy[b],
                               qFrt, qMid, qLast, 
                           qConFrt,qConLast, delT, dx.y());       

    uVel = (qMid - delT * ((d1_x[l] + d1_z[l]) * vel[l].x() + 
                        +  (d3_x[l] + d4_z[l]) * rho_tmp[l] 
                        +   y_conv))/rho_tmp[l];
    //__________________________________
    //         Y   V E L O C I T Y 
    qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].y() + p[t];
    qConLast = rho_tmp[b] * vel[b].y() * vel[b].y() + p[b];
    qFrt     = rho_tmp[t] * vel[t].y();
    qMid     = rho_tmp[l] * vel[l].y();
    qLast    = rho_tmp[b] * vel[b].y();

    y_conv = computeConvection(nuy[t], nuy[l], nuy[b],
                   qFrt, qMid, qLast, qConFrt,qConLast, delT, dx.y()); 

    vVel = (qMid - delT * ((d1_x[l] + d1_z[l]) * vel[l].y() 
                        +  (d4_x[l] + d5_z[l]) * rho_tmp[l] + y_conv))
                              /rho_tmp[l];
    //__________________________________
    //         Z   V E L O C I T Y
    qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].z();
    qConLast = rho_tmp[b] * vel[b].y() * vel[b].z();
    qFrt     = rho_tmp[t] * vel[t].z();
    qMid     = rho_tmp[l] * vel[l].z();
    qLast    = rho_tmp[b] * vel[b].z();

    y_conv = computeConvection(nuy[t], nuy[l], nuy[b],
                      qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.y());

    wVel = (qMid - delT * ((d1_x[l] + d1_z[l]) * vel[l].z() 
                        +  (d5_x[l] + d3_z[l]) * rho_tmp[l]  
                        + y_conv))/rho_tmp[l];

    vel_CC[l] = Vector(uVel, vVel, wVel);
  } 
  //_______________________________________________________
  //    E D G E   top-front
  for(int i = low.x()+1; i < hi_x; i++) {
    r  =  IntVector(i+1, hi_y,   hi_z);
    l  =  IntVector(i-1, hi_y,   hi_z);
    t  =  IntVector(i,   hi_y,   hi_z);

    //__________________________________
    //         X   V E L O C I T Y                          
    qConFrt  = rho_tmp[r] * vel[r].x() * vel[r].x() + p[r];
    qConLast = rho_tmp[l] * vel[l].x() * vel[l].x() + p[l];
    qFrt     = rho_tmp[r] * vel[r].x();
    qMid     = rho_tmp[t] * vel[t].x();
    qLast    = rho_tmp[l] * vel[l].x();

    x_conv = computeConvection(nux[r], nux[t], nux[l],
                                 qFrt, qMid, qLast, 
                                 qConFrt, qConLast, delT, dx.x());       

    uVel = (qMid - delT * ((d1_y[t] + d1_z[t]) * vel[t].x() + 
                        +  (d4_y[t] + d4_z[t]) * rho_tmp[t] 
                        +   x_conv))/rho_tmp[t];
    //__________________________________
    //         Y   V E L O C I T Y
    qConFrt  = rho_tmp[r] * vel[r].x() * vel[r].y();
    qConLast = rho_tmp[l] * vel[l].x() * vel[l].y();
    qFrt     = rho_tmp[r] * vel[r].y();
    qMid     = rho_tmp[t] * vel[t].y();
    qLast    = rho_tmp[l] * vel[l].y();

    x_conv = computeConvection(nux[r], nux[t], nux[l],
                   qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.x()); 

    vVel = (qMid - delT * ((d1_y[t] + d1_z[t]) * vel[t].y() 
                        +  (d3_y[t] + d5_z[t]) * rho_tmp[t] + x_conv))
                               /rho_tmp[t];
    //__________________________________
    //         Z   V E L O C I T Y
    qConFrt  = rho_tmp[r] * vel[r].x() * vel[r].z();
    qConLast = rho_tmp[l] * vel[l].x() * vel[l].z();
    qFrt     = rho_tmp[r] * vel[r].z();
    qMid     = rho_tmp[t] * vel[t].z();
    qLast    = rho_tmp[l] * vel[l].z();

    x_conv = computeConvection(nux[r], nux[t], nux[l],
                    qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.x());

    wVel = (qMid - delT * ((d1_y[t] + d1_z[t]) * vel[t].z() 
                        +  (d5_y[t] + d3_z[t]) * rho_tmp[t]  
                        + x_conv))/rho_tmp[t];
    vel_CC[t] = Vector(uVel, vVel, wVel);
  } 
         
  //_______________________________________________________
  //    E D G E   bottom-front
  for(int i = low.x()+1; i < hi_x; i++) {
    r  =  IntVector(i+1, low.y(), hi_z);
    l  =  IntVector(i-1, low.y(), hi_z);
    b  =  IntVector(i,   low.y(), hi_z);

    //__________________________________
    //         X   V E L O C I T Y                          
    qConFrt  = rho_tmp[r] * vel[r].x() * vel[r].x() + p[r];
    qConLast = rho_tmp[l] * vel[l].x() * vel[l].x() + p[l];
    qFrt     = rho_tmp[r] * vel[r].x();
    qMid     = rho_tmp[b] * vel[b].x();
    qLast    = rho_tmp[l] * vel[l].x();

    x_conv = computeConvection(nux[r], nux[b], nux[l],
                                 qFrt, qMid, qLast, 
                                 qConFrt, qConLast, delT, dx.x());       

    uVel = (qMid - delT * ((d1_y[b] + d1_z[b]) * vel[b].x() + 
                        +  (d4_y[b] + d4_z[b]) * rho_tmp[b] 
                        +   x_conv))/rho_tmp[b];
        
    //__________________________________
    //         Y   V E L O C I T Y
    qConFrt  = rho_tmp[r] * vel[r].x() * vel[r].y();
    qConLast = rho_tmp[l] * vel[l].x() * vel[l].y();
    qFrt     = rho_tmp[r] * vel[r].y();
    qMid     = rho_tmp[b] * vel[b].y();
    qLast    = rho_tmp[l] * vel[l].y();

    x_conv = computeConvection(nux[r], nux[b], nux[l],
                   qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.x()); 

    vVel = (qMid - delT * ((d1_y[b] + d1_z[b]) * vel[b].y() 
                        +  (d3_y[b] + d5_z[b]) * rho_tmp[b] + x_conv))
                               /rho_tmp[b];

    //__________________________________
    //         Z   V E L O C I T Y
    qConFrt  = rho_tmp[r] * vel[r].x() * vel[r].z();
    qConLast = rho_tmp[l] * vel[l].x() * vel[l].z();
    qFrt     = rho_tmp[r] * vel[r].z();
    qMid     = rho_tmp[b] * vel[b].z();
    qLast    = rho_tmp[l] * vel[l].z();

    x_conv = computeConvection(nux[r], nux[b], nux[l],
                    qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.x());

    wVel = (qMid - delT * ((d1_y[b] + d1_z[b]) * vel[b].z() 
                        +  (d5_y[b] + d3_z[b]) * rho_tmp[b]  
                        + x_conv))/rho_tmp[b];
    vel_CC[b] = Vector(uVel, vVel, wVel);
  }  
  //________________________________________________________
  // C O R N E R S    
  vector<IntVector> crn(4);
  crn[0] = IntVector(hi_x,    low.y(), hi_z);   // right-bottom-front
  crn[1] = IntVector(hi_x,    hi_y,    hi_z);   // right-top-front   
  crn[2] = IntVector(low.x(), hi_y,    hi_z);   // left-top-front 
  crn[3] = IntVector(low.x(), low.y(), hi_z);;  // left-bottom-back 

  for( int corner = 0; corner < 4; corner ++ ) {
    IntVector c = crn[corner];
    uVel = vel[r].x() - delT * ((d1_x[c] + d1_y[c] + d1_z[c]) * vel[c].x()  
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
} //end of the function zPlusVelLODI()

/*_________________________________________________________________
 Function~ zMinusVelLODI--
 Purpose~  Compute velocity in boundary cells on the z_minus face
___________________________________________________________________*/
void zMinusVelLODI(CCVariable<Vector>& vel_CC,
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

  cout << " I am in zMinusVelLODI" << endl;    
  IntVector low,hi,r,l,t,b,f,bk;

  low = vel.getLowIndex();
  hi  = vel.getHighIndex();
  int hi_x = hi.x() - 1;
  int hi_y = hi.y() - 1;
  double uVel, vVel, wVel;
  double x_conv, y_conv;
  double qConFrt,qConLast,qFrt,qMid,qLast;


  for(int i = low.x()+1; i < hi_x; i++) {
    for(int j = low.y()+1; j < hi_y; j++) {
      r  =  IntVector(i+1, j,   low.z());
      l  =  IntVector(i-1, j,   low.z());
      t  =  IntVector(i,   j+1, low.z());
      b  =  IntVector(i,   j-1, low.z());
      f  =  IntVector(i,   j,   low.z());

      //__________________________________
      //         X   V E L O C I T Y
      qConFrt  = rho_tmp[r] * vel[r].x() * vel[r].x() + p[r];
      qConLast = rho_tmp[l] * vel[l].x() * vel[l].x() + p[l];
      qFrt     = rho_tmp[r] * vel[r].x();
      qMid     = rho_tmp[f] * vel[f].x();
      qLast    = rho_tmp[l] * vel[l].x();
      x_conv   = computeConvection(nux[r], nux[f], nux[l], 
                                 qFrt, qMid, qLast, qConFrt, 
                                 qConLast, delT, dx.x());

      qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].x();
      qConLast = rho_tmp[b] * vel[b].y() * vel[b].x();
      qFrt     = rho_tmp[t] * vel[t].x();
      qLast    = rho_tmp[b] * vel[b].x();

      y_conv = computeConvection(nuy[t], nuy[f], nuy[b],
                                qFrt, qMid, qLast, qConFrt, 
			      qConLast, delT, dx.y());       
      uVel = (qMid - delT * (d1_z[f] * vel[f].x() 
           +  rho_tmp[f] * d4_z[f] + x_conv + y_conv))/rho_tmp[f];
     
      //__________________________________
      //         Y   V E L O C I T Y        
      qConFrt  = rho_tmp[r] * vel[r].x() * vel[r].y();
      qConLast = rho_tmp[l] * vel[l].x() * vel[l].y();
      qFrt     = rho_tmp[r] * vel[r].y();
      qMid     = rho_tmp[f] * vel[f].y();
      qLast    = rho_tmp[l] * vel[l].y();

      x_conv = computeConvection(nux[r], nux[f], nux[l],
                               qFrt, qMid, qLast, qConFrt, 
			      qConLast, delT, dx.x()); 

      qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].y() + p[t];
      qConLast = rho_tmp[b] * vel[b].y() * vel[b].y() + p[b];
      qFrt     = rho_tmp[t] * vel[t].y();
      qLast    = rho_tmp[b] * vel[b].y();
      y_conv   = computeConvection(nuy[t], nuy[f], nuy[b], 
                                 qFrt, qMid, qLast, qConFrt, 
                                 qConLast, delT, dx.y());

      vVel = (qMid- delT * (d1_z[f] * vel[f].y()
                  +  rho_tmp[f] * d5_z[f] + x_conv + y_conv))/rho_tmp[f];
      //__________________________________
      //         Z   V E L O C I T Y
      qConFrt  = rho_tmp[r] * vel[r].x() * vel[r].z();
      qConLast = rho_tmp[l] * vel[l].x() * vel[l].z() ;
      qFrt     = rho_tmp[r] * vel[r].z();
      qMid     = rho_tmp[f] * vel[f].z();
      qLast    = rho_tmp[l] * vel[l].z();
      x_conv   = computeConvection(nux[r], nux[f], nux[l],
                                 qFrt, qMid, qLast, qConFrt, 
				 qConLast, delT, dx.x());  

      qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].z();
      qConLast = rho_tmp[b] * vel[b].y() * vel[b].z();
      qFrt     = rho_tmp[t] * vel[t].z();
      qLast    = rho_tmp[b] * vel[b].z();
      y_conv   = computeConvection(nuy[t], nuy[f], nuy[b], 
                                   qFrt, qMid, qLast, qConFrt, 
                                   qConLast, delT, dx.y());


      wVel = (qMid - delT * (d1_z[f] * vel[f].z() + rho_tmp[f] * d3_z[f]
                   + x_conv + y_conv))/rho_tmp[f];
      vel_CC[f] = Vector(uVel, vVel, wVel);
 
    } // end of j loop
  } //end of i loop
	
  //__________________________________________________________
  //  E D G E    right-back
  for(int j = low.y()+1; j < hi_y; j++) {
    r  =  IntVector(hi_x,   j,   low.z());
    t  =  IntVector(hi_x,   j+1, low.z());
    b  =  IntVector(hi_x,   j-1, low.z());

    //__________________________________
    //         X   V E L O C I T Y                        
    qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].x();
    qConLast = rho_tmp[b] * vel[b].y() * vel[b].x();
    qFrt     = rho_tmp[t] * vel[t].x();
    qMid     = rho_tmp[r] * vel[r].x();
    qLast    = rho_tmp[b] * vel[b].x();

    y_conv = computeConvection(nuy[t], nuy[r], nuy[b],
                               qFrt, qMid, qLast, 
                           qConFrt,qConLast, delT, dx.y());       

    uVel = (qMid - delT * ((d1_x[r] + d1_z[r]) * vel[r].x() + 
                        +  (d3_x[r] + d4_z[r]) * rho_tmp[r] 
                        +   y_conv))/rho_tmp[r]; 
    //__________________________________
    //         Y   V E L O C I T Y
    qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].y() + p[t];
    qConLast = rho_tmp[b] * vel[b].y() * vel[b].y() + p[b];
    qFrt     = rho_tmp[t] * vel[t].y();
    qMid     = rho_tmp[r] * vel[r].y();
    qLast    = rho_tmp[b] * vel[b].y();

    y_conv = computeConvection(nuy[t], nuy[r], nuy[b],
                   qFrt, qMid, qLast, qConFrt,qConLast, delT, dx.y()); 

    vVel = (qMid - delT * ((d1_x[r] + d1_z[r]) * vel[r].y() 
                        +  (d4_x[r] + d5_z[r]) * rho_tmp[r] + y_conv))
                              /rho_tmp[r];

    //__________________________________
    //         Z   V E L O C I T Y
    qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].z();
    qConLast = rho_tmp[b] * vel[b].y() * vel[b].z();
    qFrt     = rho_tmp[t] * vel[t].z();
    qMid     = rho_tmp[r] * vel[r].z();
    qLast    = rho_tmp[b] * vel[b].z();

    y_conv = computeConvection(nuy[t], nuy[r], nuy[b],
                    qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.y());

    wVel = (qMid - delT * ((d1_x[r] + d1_z[r]) * vel[r].z() 
                        +  (d5_x[r] + d3_z[r]) * rho_tmp[r]  
                        + y_conv))/rho_tmp[r];

    vel_CC[r] = Vector(uVel, vVel, wVel);
  } //end of j loop

  //__________________________________________________________
  //  E D G E    left-back
  for(int j = low.y()+1; j < hi_y; j++) {
    l  =  IntVector(low.x(), j,   low.z());
    t  =  IntVector(low.x(), j+1, low.z());
    b  =  IntVector(low.x(), j-1, low.z());

    //__________________________________
    //         X   V E L O C I T Y                         
    qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].x();
    qConLast = rho_tmp[b] * vel[b].y() * vel[b].x();
    qFrt     = rho_tmp[t] * vel[t].x();
    qMid     = rho_tmp[l] * vel[l].x();
    qLast    = rho_tmp[b] * vel[b].x();

    y_conv = computeConvection(nuy[t], nuy[l], nuy[b],
                               qFrt, qMid, qLast, 
                           qConFrt,qConLast, delT, dx.y());       

    uVel = (qMid - delT * ((d1_x[l] + d1_z[l]) * vel[l].x() + 
                        +  (d3_x[l] + d4_z[l]) * rho_tmp[l] 
                        +   y_conv))/rho_tmp[l];
    //__________________________________
    //         Y   V E L O C I T Y
    qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].y() + p[t];
    qConLast = rho_tmp[b] * vel[b].y() * vel[b].y() + p[b];
    qFrt     = rho_tmp[t] * vel[t].y();
    qMid     = rho_tmp[l] * vel[l].y();
    qLast    = rho_tmp[b] * vel[b].y();

    y_conv = computeConvection(nuy[t], nuy[l], nuy[b],
                   qFrt, qMid, qLast, qConFrt,qConLast, delT, dx.y()); 

    vVel = (qMid - delT * ((d1_x[l] + d1_z[l]) * vel[l].y() 
                        +  (d4_x[l] + d5_z[l]) * rho_tmp[l] + y_conv))
                              /rho_tmp[l];
    //__________________________________
    //         Z   V E L O C I T Y
    qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].z();
    qConLast = rho_tmp[b] * vel[b].y() * vel[b].z();
    qFrt     = rho_tmp[t] * vel[t].z();
    qMid     = rho_tmp[l] * vel[l].z();
    qLast    = rho_tmp[b] * vel[b].z();

    y_conv = computeConvection(nuy[t], nuy[l], nuy[b],
                      qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.y());

    wVel = (qMid - delT * ((d1_x[l] + d1_z[l]) * vel[l].z() 
                        +  (d5_x[l] + d3_z[l]) * rho_tmp[l]  
                        + y_conv))/rho_tmp[l];

    vel_CC[l] = Vector(uVel, vVel, wVel);
  } 

  //_______________________________________________________
  //    E D G E     top-back
  for(int i = low.x()+1; i < hi_x; i++) {
    r  =  IntVector(i+1, hi_y,   low.z());
    l  =  IntVector(i-1, hi_y,   low.z());
    t  =  IntVector(i,   hi_y,   low.z());
    //__________________________________
    //         X   V E L O C I T Y                          
    qConFrt  = rho_tmp[r] * vel[r].x() * vel[r].x() + p[r];
    qConLast = rho_tmp[l] * vel[l].x() * vel[l].x() + p[l];
    qFrt     = rho_tmp[r] * vel[r].x();
    qMid     = rho_tmp[t] * vel[t].x();
    qLast    = rho_tmp[l] * vel[l].x();

    x_conv = computeConvection(nux[r], nux[t], nux[l],
                                 qFrt, qMid, qLast, 
                                 qConFrt, qConLast, delT, dx.x());       

    uVel = (qMid - delT * ((d1_y[t] + d1_z[t]) * vel[t].x() + 
                        +  (d4_y[t] + d4_z[t]) * rho_tmp[t] 
                        +   x_conv))/rho_tmp[t];          
    //__________________________________
    //         Y   V E L O C I T Y 
    qConFrt  = rho_tmp[r] * vel[r].x() * vel[r].y();
    qConLast = rho_tmp[l] * vel[l].x() * vel[l].y();
    qFrt     = rho_tmp[r] * vel[r].y();
    qMid     = rho_tmp[t] * vel[t].y();
    qLast    = rho_tmp[l] * vel[l].y();

    x_conv = computeConvection(nux[r], nux[t], nux[l],
                   qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.x()); 

    vVel = (qMid - delT * ((d1_y[t] + d1_z[t]) * vel[t].y() 
                        +  (d3_y[t] + d5_z[t]) * rho_tmp[t] + x_conv))
                               /rho_tmp[t];

    //__________________________________
    //         Z   V E L O C I T Y
    qConFrt  = rho_tmp[r] * vel[r].x() * vel[r].z();
    qConLast = rho_tmp[l] * vel[l].x() * vel[l].z();
    qFrt     = rho_tmp[r] * vel[r].z();
    qMid     = rho_tmp[t] * vel[t].z();
    qLast    = rho_tmp[l] * vel[l].z();

    x_conv = computeConvection(nux[r], nux[t], nux[l],
                    qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.x());

    wVel = (qMid - delT * ((d1_y[t] + d1_z[t]) * vel[t].z() 
                        +  (d5_y[t] + d3_z[t]) * rho_tmp[t]  
                        + x_conv))/rho_tmp[t];
    vel_CC[t] = Vector(uVel, vVel, wVel);
  }
         
  //_______________________________________________________
  //    E D G E   bottom-back
  for(int i = low.x()+1; i < hi_x; i++) {
    r  =  IntVector(i+1, low.y(), low.z());
    l  =  IntVector(i-1, low.y(), low.z());
    b  =  IntVector(i,   low.y(), low.z());

    //__________________________________
    //         X   V E L O C I T Y                         
    qConFrt  = rho_tmp[r] * vel[r].x() * vel[r].x() + p[r];
    qConLast = rho_tmp[l] * vel[l].x() * vel[l].x() + p[l];
    qFrt     = rho_tmp[r] * vel[r].x();
    qMid     = rho_tmp[b] * vel[b].x();
    qLast    = rho_tmp[l] * vel[l].x();

    x_conv = computeConvection(nux[r], nux[b], nux[l],
                                 qFrt, qMid, qLast, 
                               qConFrt, qConLast, delT, dx.x());       

    uVel = (qMid - delT * ((d1_y[b] + d1_z[b]) * vel[b].x() + 
                      +  (d4_y[b] + d4_z[b]) * rho_tmp[b] 
                      +   x_conv))/rho_tmp[b];
    //__________________________________
    //         Y   V E L O C I T Y
    qConFrt  = rho_tmp[r] * vel[r].x() * vel[r].y();
    qConLast = rho_tmp[l] * vel[l].x() * vel[l].y();
    qFrt     = rho_tmp[r] * vel[r].y();
    qMid     = rho_tmp[b] * vel[b].y();
    qLast    = rho_tmp[l] * vel[l].y();

    x_conv = computeConvection(nux[r], nux[b], nux[l],
                   qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.x()); 

    vVel = (qMid - delT * ((d1_y[b] + d1_z[b]) * vel[b].y() 
                        +  (d3_y[b] + d5_z[b]) * rho_tmp[b] + x_conv))
                               /rho_tmp[b];
    //__________________________________
    //         Z  V E L O C I T Y
    qConFrt  = rho_tmp[r] * vel[r].x() * vel[r].z();
    qConLast = rho_tmp[l] * vel[l].x() * vel[l].z();
    qFrt     = rho_tmp[r] * vel[r].z();
    qMid     = rho_tmp[b] * vel[b].z();
    qLast    = rho_tmp[l] * vel[l].z();

    x_conv = computeConvection(nux[r], nux[b], nux[l],
                    qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.x());

    wVel = (qMid - delT * ((d1_y[b] + d1_z[b]) * vel[b].z() 
                        +  (d5_y[b] + d3_z[b]) * rho_tmp[b]  
                        + x_conv))/rho_tmp[b];
    vel_CC[b] = Vector(uVel, vVel, wVel);       
  }   
  //________________________________________________________
  // C O R N E R S    
  vector<IntVector> crn(4);
  crn[0] = IntVector(hi_x,    low.y(), low.z());   // right-bottom-front
  crn[1] = IntVector(hi_x,    hi_y,    low.z());   // right-top-front   
  crn[2] = IntVector(low.x(), hi_y,    low.z());   // left-top-front 
  crn[3] = IntVector(low.x(), low.y(), low.z());;  // left-bottom-back 

  for( int corner = 0; corner < 4; corner ++ ) {
     IntVector c = crn[corner];
     uVel = vel[r].x() - delT * ((d1_x[c] + d1_y[c] + d1_z[c]) * vel[c].x()  
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
} //end of the function zMinusVelLODI()


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
   cout_doing << "I am in fillFaceVelLODI()" << endl;
   switch(face) {
     case Patch::xplus:{ 
     xPlusVelLODI(vel_CC, d1_x, d3_x, d4_x, d5_x,
		            d1_y, d3_y, d4_y, d5_y, 
		            d1_z, d3_z, d4_z, d5_z, 
			     nux,nuy,nuz,
			     rho_tmp, p, vel, delT, dx);
     }
     break;
     case Patch::xminus:{ 
       xMinusVelLODI(vel_CC,d1_x, d3_x, d4_x, d5_x,
		              d1_y, d3_y, d4_y, d5_y,
			       d1_z, d3_z, d4_z, d5_z, 
		              nux,nuy,nuz,
		              rho_tmp, p, vel, delT, dx);
     }
     break;
     case Patch::yplus:{ 
       yPlusVelLODI(vel_CC, d1_x, d3_x, d4_x, d5_x,
		              d1_y, d3_y, d4_y, d5_y,
			       d1_z, d3_z, d4_z, d5_z,
		               nux,nuy,nuz,
		               rho_tmp, p, vel, delT, dx);
     } 
     break;
     case Patch::yminus:{ 
       yMinusVelLODI(vel_CC, d1_x, d3_x, d4_x, d5_x,
		               d1_y, d3_y, d4_y, d5_y,
			        d1_z, d3_z, d4_z, d5_z,
		               nux,nuy,nuz,
		               rho_tmp, p, vel, delT, dx);
     }
     break;
     case Patch::zplus:{ 
       zPlusVelLODI(vel_CC, d1_x, d3_x, d4_x, d5_x,
	                     d1_y, d3_y, d4_y, d5_y,
		              d1_z, d3_z, d4_z, d5_z,
		              nux,nuy,nuz,
		              rho_tmp, p, vel, delT, dx);
     }
     break;
     case Patch::zminus:{
     zMinusVelLODI(vel_CC, d1_x, d3_x, d4_x, d5_x,
	                    d1_y, d3_y, d4_y, d5_y,
	 	             d1_z, d3_z, d4_z, d5_z,
		             nux,nuy,nuz,
		             rho_tmp, p, vel, delT, dx);
     }
     break;
     default:
     break;
    }
}
  
/*_________________________________________________________________
 Function~ xPlusTempLODI--
 Purpose~  Compute temperature in boundary cells on x_plus face
           using Characteristic Boundary Condition 
___________________________________________________________________*/
void xPlusTempLODI(CCVariable<double>& temp_CC, 
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
     cout_doing << " I am in xPlusTempLODI" << endl;    
     IntVector low,hi,r,l,t,b,f,bk;

     low = vel.getLowIndex();
     hi  = vel.getHighIndex();
     int hi_x = hi.x() - 1;
     int hi_y = hi.y() - 1;
     int hi_z = hi.z() - 1;
     double term1,term2,term3,term4;
     double y_conv, z_conv;
     double qConFrt,qConLast,qFrt,qMid,qLast;
 
     for(int j = low.y()+1; j < hi_y; j++) {
       for(int k = low.z()+1; k < hi_z; k++) {
         r  =  IntVector(hi_x,  j,   k);
         t  =  IntVector(hi_x,  j+1, k);
         b  =  IntVector(hi_x,  j-1, k);
         f  =  IntVector(hi_x,  j,   k+1);
         bk =  IntVector(hi_x,  j,   k-1);

         //_______________________________________________________________
         // energy conservation law, computing pressure, or temperature
         // Please remember e is the total energy per unit volume, 
         // not per unit mass, must be given or compute. The ICE code compute 
         // just internal energy, not the total energy. Here the code is written 
         // just for ICE materials

         qConFrt  = vel[t].y() * (e[t] + p[t]);
         qConLast = vel[b].y() * (e[b] + p[b]);
         qFrt     = e[t];
         qMid     = e[r];
         qLast    = e[b];
         y_conv   = computeConvection(nuy[t], nuy[r], nuy[b], 
                                   qFrt, qMid, qLast, qConFrt, 
                                   qConLast, delT, dx.y());

         qConFrt  = vel[f].z()  * (e[f] + p[f]);
         qConLast = vel[bk].z() * (e[bk] + p[bk]);
         qFrt     = e[f];
         qLast    = e[bk];

         z_conv = computeConvection(nuz[f], nuz[r], nuz[bk],
                        qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.z()); 

        double vel_sqr = vel[r].length2();

        term1 = 0.5 * d1_x[r] * vel_sqr;
        term2 = d2_x[r]/(gamma - 1.0) + rho_tmp[r] * vel[r].x() * d3_x[r];
        term3 = rho_tmp[r] * vel[r].y() * d4_x[r] + rho_tmp[r] * vel[r].z() * d5_x[r];
        term4 = y_conv + z_conv;
        double e_tmp = e[r] - delT * (term1 + term2 + term3 + term4);

        temp_CC[r] = e_tmp/rho_CC[r]/cv - 0.5 * vel_sqr/cv;  
      } // end of j loop
    } //end of k loop
       
   //__________________________________________________________
   //  E D G E     right-bottom
   for(int k = low.z()+1; k < hi_z; k++) {
     r  =  IntVector(hi_x, low.y(), k);
     f  =  IntVector(hi_x, low.y(), k+1);
     bk =  IntVector(hi_x, low.y(), k-1);     
     qConFrt  = vel[f].z()  * (e[f]  + p[f]);
     qConLast = vel[bk].z() * (e[bk] + p[bk]);
     qFrt     = e[f];
     qMid     = e[r];
     qLast    = e[bk];

      z_conv = computeConvection(nuz[f], nuz[r], nuz[bk],
                       qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.z());

      double vel_sqr = vel[r].length2();

      term1 = 0.5 * (d1_x[r] + d1_y[r]) * vel_sqr;
      term2 =  (d2_x[r] + d2_y[r])/(gamma - 1.0) 
            +  rho_tmp[r] * vel[r].x() * (d3_x[r] + d4_y[r]);
      term3 =  rho_tmp[r] * vel[r].y() * (d4_x[r] + d3_y[r]) 
            +  rho_tmp[r] * vel[r].z() * (d5_x[r] + d5_y[r]);

      double e_tmp = e[r] - delT * ( term1 + term2 + term3 + z_conv);

      temp_CC[r] = e_tmp/rho_CC[r]/cv - 0.5 * vel_sqr/cv;
   } //end of k loop

   //_______________________________________________________
   //    E D G E   left-top
   for(int k = low.z()+1; k < hi_z; k++) {
     r  =  IntVector(hi_x,   hi_y,   k);
     f  =  IntVector(hi_x,   hi_y,   k+1);
     bk =  IntVector(hi_x,   hi_y,   k-1);

     qConFrt  = vel[f].z() * (e[f]  + p[f]);
     qConLast = vel[bk].z() *(e[bk] + p[bk]);
     qFrt     = e[f];
     qMid     = e[r];
     qLast    = e[bk];

     z_conv = computeConvection(nuz[f], nuz[r], nuz[bk],
                      qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.z());

     double vel_sqr = vel[r].length2();

     term1 = 0.5 * (d1_x[r] + d1_y[r]) * vel_sqr;
     term2 =  (d2_x[r] + d2_y[r])/(gamma - 1.0) 
           +  rho_tmp[r] * vel[r].x() * (d3_x[r] + d4_y[r]);
     term3 =  rho_tmp[r] * vel[r].y() * (d4_x[r] + d3_y[r]) 
           +  rho_tmp[r] * vel[r].z() * (d5_x[r] + d5_y[r]);

     double e_tmp = e[r] - delT * ( term1 + term2 + term3 + z_conv);

     temp_CC[r] = e_tmp/rho_CC[r]/cv - 0.5 * vel_sqr/cv;
   } 
   //_____________________________________________________
   //    E D G E    right-back
   for(int j = low.y()+1; j < hi_y; j++) {
     r  =  IntVector(hi_x,   j,   low.z());
     t  =  IntVector(hi_x,   j+1, low.z());
     b  =  IntVector(hi_x,   j-1, low.z());

     qConFrt  = vel[t].y() * (e[t] + p[t]);
     qConLast = vel[b].y() * (e[b] + p[b]);
     qFrt     = e[t];
     qMid     = e[r];
     qLast    = e[b];

     y_conv = computeConvection(nuy[t], nuy[r], nuy[b],
                     qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.y());

     double vel_sqr = vel[r].length2();

     term1 = 0.5 * (d1_x[r] + d1_z[r]) * vel_sqr;
     term2 =  (d2_x[r] + d2_z[r])/(gamma - 1.0) 
           +  rho_tmp[r] * vel[r].x() * (d3_x[r] + d4_z[r]);
     term3 =  rho_tmp[r] * vel[r].y() * (d4_x[r] + d5_z[r]) 
           +  rho_tmp[r] * vel[r].z() * (d5_x[r] + d3_z[r]);

     double  e_tmp = e[r] - delT * ( term1 + term2 + term3 + y_conv);

     temp_CC[r] = e_tmp/rho_CC[r]/cv - 0.5 * vel_sqr/cv;
   }

   //_________________________________________________
   // E D G E     right-front
   for(int j = low.y()+1; j < hi_y; j++) {
     r  =  IntVector(hi_x,   j,   hi_z);
     t  =  IntVector(hi_x,   j+1, hi_z);
     b  =  IntVector(hi_x,   j-1, hi_z);

     qConFrt  = vel[t].y() * (e[t] + p[t]);
     qConLast = vel[b].y() * (e[b] + p[b]);
     qFrt     = e[t];
     qMid     = e[r];
     qLast    = e[b];

     y_conv = computeConvection(nuy[t], nuy[r], nuy[b],
                       qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.y());

     double vel_sqr = vel[r].length2();

     term1 = 0.5 * (d1_x[r] + d1_z[r]) * vel_sqr;
     term2 =  (d2_x[r] + d2_z[r])/(gamma - 1.0) 
           +  rho_tmp[r] * vel[r].x() * (d3_x[r] + d4_z[r]);
     term3 =  rho_tmp[r] * vel[r].y() * (d4_x[r] + d5_z[r]) 
           +  rho_tmp[r] * vel[r].z() * (d5_x[r] + d3_z[r]);

     double e_tmp = e[r] - delT * ( term1 + term2 + term3 + y_conv);

     temp_CC[r] = e_tmp/rho_CC[r]/cv - 0.5 * vel_sqr/cv;
   }
   //________________________________________________________
   // C O R N E R S    
   vector<IntVector> crn(4);
   crn[0] = IntVector(hi_x, low.y(), hi_z);     // right-bottom-front
   crn[1] = IntVector(hi_x, hi_y,    hi_z);     // right-top-front   
   crn[2] = IntVector(hi_x, hi_y,    low.z());  // right-top-back
   crn[3] = IntVector(hi_x, low.y(), low.z());  // right-bottom-back 

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
 } //end of function xPlusTempLODI()

/*_________________________________________________________________
 Function~ xMinusTempLODI--
 Purpose~  Compute temperature in boundary cells on x_minus face
___________________________________________________________________*/
void xMinusTempLODI(CCVariable<double>& temp_CC, 
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
  cout_doing << " I am in fillFacetemperatureLODI" << endl;    
  IntVector low,hi,r,l,t,b,f,bk;

  low = vel.getLowIndex();
  hi  = vel.getHighIndex();

  int hi_y = hi.y() - 1;
  int hi_z = hi.z() - 1;
  double term1,term2,term3,term4;
  double y_conv, z_conv;
  double qConFrt,qConLast,qFrt,qMid,qLast;

  for(int j = low.y()+1; j < hi_y; j++) {
    for(int k = low.z()+1; k < hi_z; k++) {

      l  =  IntVector(low.x(),  j,   k);
      t  =  IntVector(low.x(),  j+1, k);
      b  =  IntVector(low.x(),  j-1, k);
      f  =  IntVector(low.x(),  j,   k+1);
      bk =  IntVector(low.x(),  j,   k-1);       

      //_______________________________________________________________
      // energy conservation law, computing pressure, or temperature
      // Please remember e is the total energy per unit volume, 
      // not per unit mass, must be given or compute. The ICE code compute 
      // just internal energy, not the total energy. Here the code is written 
      // just for ICE materials
      qConFrt  = vel[t].y() * (e[t] + p[t]);
      qConLast = vel[b].y() * (e[b] + p[b]);
      qFrt     = e[t];
      qMid     = e[l];
      qLast    = e[b];
      y_conv   = computeConvection(nuy[t], nuy[l], nuy[b], 
                                  qFrt, qMid, qLast, qConFrt, 
                                  qConLast, delT, dx.y());

      qConFrt  = vel[f].z()  * (e[f]  + p[f]);
      qConLast = vel[bk].z() * (e[bk] + p[bk]);
      qFrt     = e[f];
      qLast    = e[bk];

      z_conv = computeConvection(nuz[f], nuz[l], nuz[bk],
                      qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.z()); 

      double vel_sqr = (vel[l].x() * vel[l].x()
                     +  vel[l].y() * vel[l].y()
                     +  vel[l].z() * vel[l].z());

      term1 = 0.5 * d1_x[l] * vel_sqr;
      term2 = d2_x[l]/(gamma - 1.0) + rho_tmp[l] * vel[l].x() * d3_x[l];
      term3 = rho_tmp[l] * vel[l].y() * d4_x[l] + rho_tmp[l] * vel[l].z() * d5_x[l];
      term4 = y_conv + z_conv;

      double e_tmp = e[l] - delT * (term1 + term2 + term3 + term4);

      temp_CC[l] = e_tmp/rho_CC[l]/cv - 0.5 * vel_sqr/cv;      
    } // end of j loop
  } //end of k loop
      
  //__________________________________________________________
  //  E D G E    left-bottom
  for(int k = low.z()+1; k < hi_z; k++) {
    r  =  IntVector(low.x(), low.y(), k);
    f  =  IntVector(low.x(), low.y(), k+1);
    bk =  IntVector(low.x(), low.y(), k-1);     
    qConFrt  = vel[f].z()  * (e[f]  + p[f]);
    qConLast = vel[bk].z() * (e[bk] + p[bk]);
    qFrt     = e[f];
    qMid     = e[r];
    qLast    = e[bk];

    z_conv = computeConvection(nuz[f], nuz[r], nuz[bk],
                     qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.z());

    double vel_sqr = vel[r].length2();

    term1 = 0.5 * (d1_x[r] + d1_y[r]) * vel_sqr;
    term2 =  (d2_x[r] + d2_y[r])/(gamma - 1.0) 
          +  rho_tmp[r] * vel[r].x() * (d3_x[r] + d4_y[r]);
    term3 =  rho_tmp[r] * vel[r].y() * (d4_x[r] + d3_y[r]) 
          +  rho_tmp[r] * vel[r].z() * (d5_x[r] + d5_y[r]);

    double e_tmp = e[r] - delT * ( term1 + term2 + term3 + z_conv);

    temp_CC[r] = e_tmp/rho_CC[r]/cv - 0.5 * vel_sqr/cv;
  }

  //_______________________________________________________
  //    E D G E   left-top
  for(int k = low.z()+1; k < hi_z; k++) {
    r  =  IntVector(low.x(), hi_y, k);
    f  =  IntVector(low.x(), hi_y, k+1);
    bk =  IntVector(low.x(), hi_y, k-1);

    qConFrt  = vel[f].z() * (e[f]  + p[f]);
    qConLast = vel[bk].z() *(e[bk] + p[bk]);
    qFrt     = e[f];
    qMid     = e[r];
    qLast    = e[bk];

    z_conv = computeConvection(nuz[f], nuz[r], nuz[bk],
                     qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.z());

    double vel_sqr = vel[r].length2();

    term1 = 0.5 * (d1_x[r] + d1_y[r]) * vel_sqr;
    term2 =  (d2_x[r] + d2_y[r])/(gamma - 1.0) 
          +  rho_tmp[r] * vel[r].x() * (d3_x[r] + d4_y[r]);
    term3 =  rho_tmp[r] * vel[r].y() * (d4_x[r] + d3_y[r]) 
          +  rho_tmp[r] * vel[r].z() * (d5_x[r] + d5_y[r]);

    double e_tmp = e[r] - delT * ( term1 + term2 + term3 + z_conv);

    temp_CC[r] = e_tmp/rho_CC[r]/cv - 0.5 * vel_sqr/cv;
  } //end of k loop
         
  //_____________________________________________________
  //    E D G E   left-back
  for(int j = low.y()+1; j < hi_y; j++) {
    r  =  IntVector(low.x(), j,   low.z());
    t  =  IntVector(low.x(), j+1, low.z());
    b  =  IntVector(low.x(), j-1, low.z());
    qConFrt  = vel[t].y() * (e[t] + p[t]);
    qConLast = vel[b].y() * (e[b] + p[b]);
    qFrt     = e[t];
    qMid     = e[r];
    qLast    = e[b];

    y_conv = computeConvection(nuy[t], nuy[r], nuy[b],
                    qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.y());

    double vel_sqr = vel[r].length2();

    term1 = 0.5 * (d1_x[r] + d1_z[r]) * vel_sqr;
    term2 =  (d2_x[r] + d2_z[r])/(gamma - 1.0) 
          +  rho_tmp[r] * vel[r].x() * (d3_x[r] + d4_z[r]);
    term3 =  rho_tmp[r] * vel[r].y() * (d4_x[r] + d5_z[r]) 
          +  rho_tmp[r] * vel[r].z() * (d5_x[r] + d3_z[r]);

    double  e_tmp = e[r] - delT * ( term1 + term2 + term3 + y_conv);

    temp_CC[r] = e_tmp/rho_CC[r]/cv - 0.5 * vel_sqr/cv;
  } //end of j loop

  //_________________________________________________
  // E D G E    left-top
  for(int j = low.y()+1; j < hi_y; j++) {

    r  =  IntVector(low.x(),   j,   hi_z);
    t  =  IntVector(low.x(),   j+1, hi_z);
    b  =  IntVector(low.x(),   j-1, hi_z);

    qConFrt  = vel[t].y() * (e[t] + p[t]);
    qConLast = vel[b].y() * (e[b] + p[b]);
    qFrt     = e[t];
    qMid     = e[r];
    qLast    = e[b];

    y_conv = computeConvection(nuy[t], nuy[r], nuy[b],
                    qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.y());

    double vel_sqr = vel[r].length2();

    term1 = 0.5 * (d1_x[r] + d1_z[r]) * vel_sqr;
    term2 =       (d2_x[r] + d2_z[r])/(gamma - 1.0) 
          +  rho_tmp[r] * vel[r].x() * (d3_x[r] + d4_z[r]);
    term3 =  rho_tmp[r] * vel[r].y() * (d4_x[r] + d5_z[r]) 
          +  rho_tmp[r] * vel[r].z() * (d5_x[r] + d3_z[r]);

    double  e_tmp = e[r] - delT * ( term1 + term2 + term3 + y_conv);

    temp_CC[r] = e_tmp/rho_CC[r]/cv - 0.5 * vel_sqr/cv;
  } 
  //________________________________________________________
  // C O R N E R S    
  vector<IntVector> crn(4);
  crn[0] = IntVector(low.x(), low.y(), hi_z);     // left-bottom-front
  crn[1] = IntVector(low.x(), hi_y,    hi_z);     // left-top-front   
  crn[2] = IntVector(low.x(), hi_y,    low.z());  // left-top-back
  crn[3] = IntVector(low.x(), low.y(), low.z());  // left-bottom-back 

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
} //end of the function xMinusTempLODI()

/*_________________________________________________________________
 Function~ yPlusTempLODI--
 Purpose~  Compute temperature in boundary cells on y_plus face
___________________________________________________________________*/
void yPlusTempLODI(CCVariable<double>& temp_CC, 
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
  cout_doing << " I am in yPlusTempLODI" << endl;    
  IntVector low,hi,r,l,t,b,f,bk;

  low = vel.getLowIndex();
  hi  = vel.getHighIndex();
  int hi_x = hi.x() - 1;
  int hi_y = hi.y() - 1;
  int hi_z = hi.z() - 1;
  double term1,term2,term3,term4;
  double x_conv, z_conv;
  double qConFrt,qConLast,qFrt,qMid,qLast;
         
  for(int i = low.x()+1; i < hi_x; i++) {
    for(int k = low.z()+1; k < hi_z; k++) {
      r  =  IntVector(i+1,  hi_y,   k);
      l  =  IntVector(i-1,  hi_y,   k);
      t  =  IntVector(i,    hi_y,   k);
      f  =  IntVector(i,    hi_y,   k+1);
      bk =  IntVector(i,    hi_y,   k-1);       
                     
      //_______________________________________________________________
      // energy conservation law, computing pressure, or temperature
      // Please remember e is the total energy per unit volume, 
      // not per unit mass, must be given or compute. The ICE code compute 
      // just internal energy, not the total energy. Here the code is written 
      // just for ICE materials
	   
      qConFrt  = vel[r].x() * (e[r] + p[r]);
      qConLast = vel[l].x() * (e[l] + p[l]);
      qFrt     = e[r];
      qMid     = e[t];
      qLast    = e[l];
      x_conv   = computeConvection(nux[r], nux[t], nux[l], 
                                qFrt, qMid, qLast, qConFrt, 
                                qConLast, delT, dx.x());

      qConFrt  = vel[f].z()  * (e[f] + p[f]);
      qConLast = vel[bk].z() * (e[bk] + p[bk]);
      qFrt   = e[f];
      qLast  = e[bk];

      z_conv = computeConvection(nuz[f], nuz[t], nuz[bk],
                      qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.z()); 

      double vel_sqr = (vel[t].x() * vel[t].x()
                     +  vel[t].y() * vel[t].y()
                     +  vel[t].z() * vel[t].z());

      term1 = 0.5 * d1_y[t] * vel_sqr;
      term2 = d2_y[t]/(gamma - 1.0) + rho_tmp[t] * vel[t].x() * d4_y[t];
      term3 = rho_tmp[t] * vel[t].y() * d3_y[t] + rho_tmp[t] * vel[t].z() * d5_y[t];
      term4 = x_conv + z_conv;
      double e_tmp = e[t] - delT * (term1 + term2 + term3 + term4);

      temp_CC[t] = e_tmp/rho_CC[t]/cv - 0.5 * vel_sqr/cv;
    } // end of i loop
  } //end of k loop
  
  //__________________________________________________________
  //  E D G E    right top
  for(int k = low.z()+1; k < hi_z; k++) {
    t  =  IntVector(hi_x,   hi_y,   k);
    f  =  IntVector(hi_x,   hi_y,   k+1);
    bk =  IntVector(hi_x,   hi_y,   k-1);

    qConFrt  = vel[f].z()  * (e[f]  + p[f]);
    qConLast = vel[bk].z() * (e[bk] + p[bk]);
    qFrt     = e[f];
    qMid     = e[t];
    qLast    = e[bk];

    z_conv = computeConvection(nuz[f], nuz[t], nuz[bk],
                     qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.z());

    double vel_sqr = (vel[t].x() * vel[t].x()
                   +  vel[t].y() * vel[t].y()
                   +  vel[t].z() * vel[t].z());

    term1 = 0.5 * (d1_x[t] + d1_y[t]) * vel_sqr;
    term2 =  (d2_x[t] + d2_y[t])/(gamma - 1.0) 
          +  rho_tmp[t] * vel[t].x() * (d3_x[t] + d4_y[t]);
    term3 =  rho_tmp[t] * vel[t].y() * (d4_x[t] + d3_y[t]) 
          +  rho_tmp[t] * vel[t].z() * (d5_x[t] + d5_y[t]);

    double e_tmp = e[t] - delT * ( term1 + term2 + term3 + z_conv);

    temp_CC[t] = e_tmp/rho_CC[t]/cv - 0.5 * vel_sqr/cv;

  }

  //_____________________________________________________
  //    E D G E   left- top
  for(int k = low.z()+1; k < hi_z; k++) {
    t  = IntVector(low.x(),   hi_y,   k);
    f  = IntVector(low.x(),   hi_y,   k+1);
    bk = IntVector(low.x(),   hi_y,   k-1);

    qConFrt  = vel[f].z() * (e[f]  + p[f]);
    qConLast = vel[bk].z() *(e[bk] + p[bk]);
    qFrt     = e[f];
    qMid     = e[t];
    qLast    = e[bk];

    z_conv = computeConvection(nuz[f], nuz[t], nuz[bk],
                     qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.z());

    double vel_sqr = vel[t].length2();

    term1 = 0.5 * (d1_x[t] + d1_y[t]) * vel_sqr;
    term2 =  (d2_x[t] + d2_y[t])/(gamma - 1.0) 
          +  rho_tmp[t] * vel[t].x() * (d3_x[t] + d4_y[t]);
    term3 =  rho_tmp[t] * vel[t].y() * (d4_x[t] + d3_y[t]) 
          +  rho_tmp[t] * vel[t].z() * (d5_x[t] + d5_y[t]);

    double e_tmp = e[t] - delT * ( term1 + term2 + term3 + z_conv);

    temp_CC[t] = e_tmp/rho_CC[t]/cv - 0.5 * vel_sqr/cv;
  }

  //_______________________________________________________
  //    E D G E   top-front
  for(int i = low.x()+1; i < hi_x; i++) {
    r  =  IntVector(i+1, hi_y,    hi_z);
    l  =  IntVector(i-1, hi_y,    hi_z);
    t  =  IntVector(i,   hi_y,    hi_z);

    qConFrt  = vel[r].x() * (e[r] + p[r]);
    qConLast = vel[l].x() * (e[l] + p[l]);
    qFrt     = e[r];
    qMid     = e[t];
    qLast    = e[l];

    x_conv = computeConvection(nux[r], nux[t], nux[l],
                     qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.x());

    double vel_sqr = vel[t].length2();

    term1 = 0.5 * (d1_y[t] + d1_z[t]) * vel_sqr;
    term2 =  (d2_y[t] + d2_z[t])/(gamma - 1.0) 
          +  rho_tmp[t] * vel[t].x() * (d4_y[t] + d4_z[t]);
    term3 =  rho_tmp[t] * vel[t].y() * (d3_y[t] + d5_z[t]) 
          +  rho_tmp[t] * vel[t].z() * (d5_y[t] + d3_z[t]);

    double e_tmp = e[t] - delT * ( term1 + term2 + term3 + x_conv);

    temp_CC[t] = e_tmp/rho_CC[t]/cv - 0.5 * vel_sqr/cv;
  } 
  //_________________________________________________
  // E D G E      top-back
  for(int i = low.x()+1; i < hi_x; i++) {
    r  =  IntVector(i+1, hi_y,   low.z());
    l  =  IntVector(i-1, hi_y,   low.z());
    t  =  IntVector(i,   hi_y,   low.z());
    
    qConFrt  = vel[r].x() * (e[r] + p[r]);
    qConLast = vel[l].x() * (e[l] + p[l]);
    qFrt     = e[r];
    qMid     = e[t];
    qLast    = e[l];

    x_conv = computeConvection(nux[r], nux[t], nux[l],
                     qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.x());

    double vel_sqr = vel[t].length2();

    term1 = 0.5 * (d1_y[t] + d1_z[t]) * vel_sqr;
    term2 =  (d2_y[t] + d2_z[t])/(gamma - 1.0) 
          +  rho_tmp[t] * vel[t].x() * (d4_y[t] + d4_z[t]);
    term3 =  rho_tmp[t] * vel[t].y() * (d3_y[t] + d5_z[t]) 
          +  rho_tmp[t] * vel[t].z() * (d5_y[t] + d3_z[t]);

    double e_tmp = e[t] - delT * ( term1 + term2 + term3 + x_conv);

    temp_CC[t] = e_tmp/rho_CC[t]/cv - 0.5 * vel_sqr/cv;        
  }
 
  //________________________________________________________
  // C O R N E R S    
  vector<IntVector> crn(4);
  crn[0] = IntVector(hi_x,    hi_y, hi_z);     // right-top-front
  crn[1] = IntVector(low.x(), hi_y, hi_z);     // left-top-front   
  crn[2] = IntVector(low.x(), hi_y, low.z());  // left-top-back
  crn[3] = IntVector(hi_x,    hi_y, low.z());  // right-top-back 

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
 Function~ yMinusTempLODI--
 Purpose~  Compute temperature in boundary cells on yMinus face
___________________________________________________________________*/
void yMinusTempLODI(CCVariable<double>& temp_CC, 
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
  cout_doing << " I am in yMinusTempLODI" << endl;    
  IntVector low,hi,r,l,t,b,f,bk;

  low = vel.getLowIndex();
  hi  = vel.getHighIndex();
  int hi_x = hi.x() - 1;
  int hi_z = hi.z() - 1;
  double term1,term2,term3,term4;
  double x_conv, z_conv;
  double qConFrt,qConLast,qFrt,qMid,qLast;

  for(int i = low.x()+1; i < hi_x; i++) {
    for(int k = low.z()+1; k < hi_z; k++) {
      r  =  IntVector(i+1,  low.y(),   k);
      l  =  IntVector(i-1,  low.y(),   k);
      b  =  IntVector(i,    low.y(),   k);
      f  =  IntVector(i,    low.y(),   k+1);
      bk =  IntVector(i,    low.y(),   k-1);       

      //_______________________________________________________________
      // energy conservation law, computing pressure, or temperature
      // Please remember e is the total energy per unit volume, 
      // not per unit mass, must be given or compute. The ICE code compute 
      // just internal energy, not the total energy. Here the code is written 
      // just for ICE materials

      qConFrt  = vel[r].x() * (e[r] + p[r]);
      qConLast = vel[l].x() * (e[l] + p[l]);
      qFrt     = e[r];
      qMid     = e[b];
      qLast    = e[l];
      x_conv   = computeConvection(nux[r], nux[b], nux[l], 
                                qFrt, qMid, qLast, qConFrt, 
                                qConLast, delT, dx.x());

      qConFrt  = vel[f].z()  * (e[f] + p[f]);
      qConLast = vel[bk].z() * (e[bk] + p[bk]);
      qFrt   = e[f];
      qLast  = e[bk];

      z_conv = computeConvection(nuz[f], nuz[b], nuz[bk],
                      qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.z()); 

      double vel_sqr = (vel[b].x() * vel[b].x()
                     +  vel[b].y() * vel[b].y()
                     +  vel[b].z() * vel[b].z());

      term1 = 0.5 * d1_y[b] * vel_sqr;
      term2 = d2_y[b]/(gamma - 1.0) + rho_tmp[b] * vel[b].x() * d4_y[b];
      term3 = rho_tmp[b] * vel[b].y() * d3_y[b] + rho_tmp[b] * vel[b].z() * d5_y[b];
      term4 = x_conv + z_conv;
      double e_tmp = e[b] - delT * (term1 + term2 + term3 + term4);

      temp_CC[b] = e_tmp/rho_CC[b]/cv - 0.5 * vel_sqr/cv;
    } // end of i loop
  } //end of k loop
  //__________________________________________________________
  //  E D G E    right-bottom 
  for(int k = low.z()+1; k < hi_z; k++) {   
    b  =  IntVector(hi_x,   low.y(),   k);
    f  =  IntVector(hi_x,   low.y(),   k+1);
    bk =  IntVector(hi_x,   low.y(),   k-1);     
    qConFrt  = vel[f].z()  * (e[f]  + p[f]);
    qConLast = vel[bk].z() * (e[bk] + p[bk]);
    qFrt     = e[f];
    qMid     = e[b];
    qLast    = e[bk];

    z_conv = computeConvection(nuz[f], nuz[b], nuz[bk],
                     qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.z());

    double vel_sqr = vel[b].length2();

    term1 = 0.5 * (d1_x[b] + d1_y[b]) * vel_sqr;
    term2 =       (d2_x[b] + d2_y[b])/(gamma - 1.0) 
          +  rho_tmp[b] * vel[b].x() * (d3_x[b] + d4_y[b]);
    term3 =  rho_tmp[b] * vel[b].y() * (d4_x[b] + d3_y[b]) 
          +  rho_tmp[b] * vel[b].z() * (d5_x[b] + d5_y[b]);

    double e_tmp = e[b] - delT * ( term1 + term2 + term3 + z_conv);

    temp_CC[b] = e_tmp/rho_CC[b]/cv - 0.5 * vel_sqr/cv;
  } 
  //_____________________________________________________
  //    E D G E   left-bottom
  for(int k = low.z()+1; k < hi_z; k++) {
    b  = IntVector(low.x(),   low.y(),   k);
    f  = IntVector(low.x(),   low.y(),   k+1);
    bk = IntVector(low.x(),   low.y(),   k-1);

    qConFrt  = vel[f].z() * (e[f]  + p[f]);
    qConLast = vel[bk].z() *(e[bk] + p[bk]);
    qFrt     = e[f];
    qMid     = e[b];
    qLast    = e[bk];

    z_conv = computeConvection(nuz[f], nuz[b], nuz[bk],
                     qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.z());

    double vel_sqr = vel[b].length2();

    term1 = 0.5 * (d1_x[b] + d1_y[b]) * vel_sqr;
    term2 =       (d2_x[b] + d2_y[b])/(gamma - 1.0) 
          +  rho_tmp[b] * vel[b].x() * (d3_x[b] + d4_y[b]);
    term3 =  rho_tmp[b] * vel[b].y() * (d4_x[b] + d3_y[b]) 
          +  rho_tmp[b] * vel[b].z() * (d5_x[b] + d5_y[b]);

    double e_tmp = e[b] - delT * ( term1 + term2 + term3 + z_conv);

    temp_CC[b] = e_tmp/rho_CC[b]/cv - 0.5 * vel_sqr/cv;
  }

  //_______________________________________________________
  //    E D G E   bottom-back
  for(int i = low.x()+1; i < hi_x; i++) {

    r  =  IntVector(i+1, low.y(),    hi_z);
    l  =  IntVector(i-1, low.y(),    hi_z);
    b  =  IntVector(i,   low.y(),    hi_z);

    qConFrt  = vel[r].x() * (e[r] + p[r]);
    qConLast = vel[l].x() * (e[l] + p[l]);
    qFrt     = e[r];
    qMid     = e[b];
    qLast    = e[l];

    x_conv = computeConvection(nux[r], nux[b], nux[l],
                     qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.x());

    double vel_sqr = vel[b].length2();

    term1 = 0.5 * (d1_y[b] + d1_z[b]) * vel_sqr;
    term2 =       (d2_y[b] + d2_z[b])/(gamma - 1.0) 
          +  rho_tmp[b] * vel[b].x() * (d4_y[b] + d4_z[b]);
    term3 =  rho_tmp[b] * vel[b].y() * (d3_y[b] + d5_z[b]) 
          +  rho_tmp[b] * vel[b].z() * (d5_y[b] + d3_z[b]);

    double e_tmp = e[b] - delT * ( term1 + term2 + term3 + x_conv);

    temp_CC[b] = e_tmp/rho_CC[b]/cv - 0.5 * vel_sqr/cv;
  } 
        
  //_________________________________________________
  // E D G E    bottom-back
  for(int i = low.x()+1; i < hi_x; i++) {

    r  =  IntVector(i+1, low.y(),   low.z());
    l  =  IntVector(i-1, low.y(),   low.z());
    b  =  IntVector(i,   low.y(),   low.z());

    qConFrt  = vel[r].x() * (e[r] + p[r]);
    qConLast = vel[l].x() * (e[l] + p[l]);
    qFrt     = e[r];
    qMid     = e[b];
    qLast    = e[l];

    x_conv = computeConvection(nux[r], nux[b], nux[l],
                     qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.x());

    double vel_sqr = vel[b].length2();

    term1 = 0.5 * (d1_y[b] + d1_z[b]) * vel_sqr;
    term2 =       (d2_y[b] + d2_z[b])/(gamma - 1.0) 
          +  rho_tmp[b] * vel[b].x() * (d4_y[b] + d4_z[b]);
    term3 =  rho_tmp[b] * vel[b].y() * (d3_y[b] + d5_z[b]) 
          +  rho_tmp[b] * vel[b].z() * (d5_y[b] + d3_z[b]);

    double e_tmp = e[b] - delT * ( term1 + term2 + term3 + x_conv);

    temp_CC[b] = e_tmp/rho_CC[b]/cv - 0.5 * vel_sqr/cv;
  } 
  //________________________________________________________
  // C O R N E R S    
  vector<IntVector> crn(4);
  crn[0] = IntVector(hi_x,    low.y(), hi_z);     // right-bottom-front
  crn[1] = IntVector(low.x(), low.y(), hi_z);     // left-bottom-front   
  crn[2] = IntVector(low.x(), low.y(), low.z());  // left-bottom-back
  crn[3] = IntVector(hi_x,    low.y(), low.z());  // right-bottom-back 

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
} //end of the function yMinusTempLODI()


/*_________________________________________________________________
 Function~ zPlusTempLODI--
 Purpose~  Compute temperature in boundary cells on the z_plus face
           using Characteristic Boundary Condition 
___________________________________________________________________*/
void zPlusTempLODI(CCVariable<double>& temp_CC, 
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
  cout_doing << " I am in zPlus_temperatureLODI" << endl;    
  IntVector low,hi,r,l,t,b,f,bk;

  low = vel.getLowIndex();
  hi  = vel.getHighIndex();
  int hi_x = hi.x() - 1;
  int hi_y = hi.y() - 1;
  int hi_z = hi.z() - 1;
  double term1,term2,term3,term4;
  double x_conv, y_conv;
  double qConFrt,qConLast,qFrt,qMid,qLast;
 
  for(int i = low.x()+1; i < hi_x; i++) {
    for(int j = low.y()+1; j < hi_y; j++) {
      r  =  IntVector(i+1, j,   hi_z);
      l  =  IntVector(i-1, j,   hi_z);
      t  =  IntVector(i,   j+1, hi_z);
      b  =  IntVector(i,   j-1, hi_z);
      f  =  IntVector(i,   j,   hi_z);

      //_______________________________________________________________
      // energy conservation law, computing pressure, or temperature
      // Please remember e is the total energy per unit volume, 
      // not per unit mass, must be given or compute. The ICE code compute 
      // just internal energy, not the total energy. Here the code is written 
      // just for ICE materials

      qConFrt  = vel[r].x() * (e[r] + p[r]);
      qConLast = vel[l].x() * (e[l] + p[l]);
      qFrt     = e[r];
      qMid     = e[f];
      qLast    = e[l];
      x_conv   = computeConvection(nux[r], nux[f], nux[l], 
                                qFrt, qMid, qLast, qConFrt, 
                                qConLast, delT, dx.x());

      qConFrt  = vel[t].y() * (e[t] + p[t]);
      qConLast = vel[b].y() * (e[b] + p[b]);
      qFrt     = e[t];
      qLast    = e[b];

      y_conv = computeConvection(nuy[t], nuy[f], nuy[b],
                     qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.y()); 

      double vel_sqr = vel[f].length();

      term1 = 0.5 * d1_z[f] * vel_sqr;
      term2 = d2_z[f] / (gamma - 1.0) 
            + rho_tmp[f] * vel[f].x() * d4_z[f];
      term3 = rho_tmp[f] * vel[f].y() * d5_z[f]  
            + rho_tmp[f] * vel[f].z() * d3_z[f];
      term4 = x_conv + y_conv;
      double e_tmp = e[f] - delT * (term1 + term2 + term3 + term4);

      temp_CC[f] = e_tmp/rho_CC[f]/cv - 0.5 * vel_sqr/cv;
   } // end of j loop
 } //end of i loop
     
  //__________________________________________________________
  //  E D G E      right-front
  for(int j = low.y()+1; j < hi_y; j++) {
    r  =  IntVector(hi_x, j,   hi_z);
    t  =  IntVector(hi_x, j+1, hi_z);
    b  =  IntVector(hi_x, j-1, hi_z);

    qConFrt  = vel[t].y() * (e[t] + p[t]);
    qConLast = vel[b].y() * (e[b] + p[b]);
    qFrt     = e[t];
    qMid     = e[r];
    qLast    = e[b];

    y_conv = computeConvection(nuy[t], nuy[r], nuy[b],
                     qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.y());

    double vel_sqr = vel[r].length2();

    term1 = 0.5 * (d1_x[r] + d1_z[r]) * vel_sqr;
    term2 =       (d2_x[r] + d2_z[r])/(gamma - 1.0) 
          +  rho_tmp[r] * vel[r].x() * (d3_x[r] + d4_z[r]);
    term3 =  rho_tmp[r] * vel[r].y() * (d4_x[r] + d5_z[r]) 
          +  rho_tmp[r] * vel[r].z() * (d5_x[r] + d3_z[r]);

    double e_tmp = e[r] - delT * ( term1 + term2 + term3 + y_conv);

    temp_CC[r] = e_tmp/rho_CC[r]/cv - 0.5 * vel_sqr/cv;
  } 
  //__________________________________________________________
  //  E D G E    left- back
  for(int j = low.y()+1; j < hi_y; j++) {
    l  =  IntVector(low.x(), j,   hi_z);
    t  =  IntVector(low.x(), j+1, hi_z);
    b  =  IntVector(low.x(), j-1, hi_z);

    qConFrt  = vel[t].y() * (e[t] + p[t]);
    qConLast = vel[b].y() * (e[b] + p[b]);
    qFrt     = e[t];
    qMid     = e[l];
    qLast    = e[b];

    y_conv = computeConvection(nuy[t], nuy[l], nuy[b],
                     qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.y());

    double vel_sqr = vel[l].length2();

    term1 = 0.5 * (d1_x[l] + d1_z[l]) * vel_sqr;
    term2 =       (d2_x[l] + d2_z[l])/(gamma - 1.0) 
          +  rho_tmp[l] * vel[l].x() * (d3_x[l] + d4_z[l]);
    term3 =  rho_tmp[l] * vel[l].y() * (d4_x[l] + d5_z[l]) 
          +  rho_tmp[l] * vel[l].z() * (d5_x[l] + d3_z[l]);

    double e_tmp = e[l] - delT * ( term1 + term2 + term3 + y_conv);

    temp_CC[l] = e_tmp/rho_CC[l]/cv - 0.5 * vel_sqr/cv;
  }
  //_______________________________________________________
  //    E D G E     top-front
  for(int i = low.x()+1; i < hi_x; i++) {
    r  =  IntVector(i+1, hi_y,   hi_z);
    l  =  IntVector(i-1, hi_y,   hi_z);
    t  =  IntVector(i,   hi_y,   hi_z);

    qConFrt  = vel[r].x() * (e[r] + p[r]);
    qConLast = vel[l].x() * (e[l] + p[l]);
    qFrt     = e[r];
    qMid     = e[t];
    qLast    = e[l];

    x_conv = computeConvection(nux[r], nux[t], nux[l],
                     qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.x());

    double vel_sqr = vel[t].length2();

    term1 = 0.5 * (d1_y[t] + d1_z[t]) * vel_sqr;
    term2 =       (d2_y[t] + d2_z[t])/(gamma - 1.0) 
          +  rho_tmp[t] * vel[t].x() * (d4_y[t] + d4_z[t]);
    term3 =  rho_tmp[t] * vel[t].y() * (d3_y[t] + d5_z[t]) 
          +  rho_tmp[t] * vel[t].z() * (d5_y[t] + d3_z[t]);

    double e_tmp = e[t] - delT * ( term1 + term2 + term3 + x_conv);

    temp_CC[t] = e_tmp/rho_CC[t]/cv - 0.5 * vel_sqr/cv;
  }  
  //_______________________________________________________
  //    E D G E     bottom-front
  for(int i = low.x()+1; i < hi_x; i++) {
    r  =  IntVector(i+1, low.y(), hi_z);
    l  =  IntVector(i-1, low.y(), hi_z);
    b  =  IntVector(i,   low.y(), hi_z);

    qConFrt  = vel[r].x() * (e[r] + p[r]);       
    qConLast = vel[l].x() * (e[l] + p[l]);
    qFrt     = e[r];
    qMid     = e[b];
    qLast    = e[l];

    x_conv = computeConvection(nux[r], nux[b], nux[l],
                     qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.x());

    double vel_sqr = vel[b].length2();

    term1 = 0.5 * (d1_y[b] + d1_z[b]) * vel_sqr;
    term2 =       (d2_y[b] + d2_z[b])/(gamma - 1.0) 
          +  rho_tmp[b] * vel[b].x() * (d4_y[b] + d4_z[b]);
    term3 =  rho_tmp[b] * vel[b].y() * (d3_y[b] + d5_z[b]) 
          +  rho_tmp[b] * vel[b].z() * (d5_y[b] + d3_z[b]);

    double e_tmp = e[b] - delT * ( term1 + term2 + term3 + x_conv);

    temp_CC[b] = e_tmp/rho_CC[b]/cv - 0.5 * vel_sqr/cv;
  }

  //________________________________________________________
  // C O R N E R S    
  vector<IntVector> crn(4);
  crn[0] = IntVector(hi_x,    low.y(), hi_z);  // right-bottom-front
  crn[1] = IntVector(hi_x,    hi_y,    hi_z);  // right-top-front   
  crn[2] = IntVector(low.x(), hi_y,    hi_z);  // left-top-right
  crn[3] = IntVector(low.x(), low.y(), hi_z);  // left-bottom-right 

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
} //end of the function zPlusTempLODI()


/*_________________________________________________________________
 Function~ zMinusTempLODI--
 Purpose~  Compute temperature in boundary cells on the z_Minus face 
___________________________________________________________________*/
void zMinusTempLODI(CCVariable<double>& temp_CC, 
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
  cout_doing << " I am in zMinusTempLODI" << endl;        
  IntVector low,hi,r,l,t,b,f,bk;

  low = vel.getLowIndex();
  hi  = vel.getHighIndex();
  int hi_x = hi.x() - 1;
  int hi_y = hi.y() - 1;
  double term1,term2,term3,term4;
  double x_conv, y_conv;
  double qConFrt,qConLast,qFrt,qMid,qLast;

  for(int i = low.x()+1; i < hi_x; i++) {
    for(int j = low.y()+1; j < hi_y; j++) {
      r   =  IntVector(i+1, j,   low.z());
      l   =  IntVector(i-1, j,   low.z());
      t   =  IntVector(i,   j+1, low.z());
      b   =  IntVector(i,   j-1, low.z());
      bk  =  IntVector(i,   j,   low.z());

      //_______________________________________________________________
      // energy conservation law, computing pressure, or temperature
      // Please remember e is the total energy per unit volume, 
      // not per unit mass, must be given or compute. The ICE code compute 
      // just internal energy, not the total energy. Here the code is written 
      // just for ICE materials

      qConFrt  = vel[r].x() * (e[r] + p[r]);
      qConLast = vel[l].x() * (e[l] + p[l]);
      qFrt     = e[r];
      qMid     = e[bk];
      qLast    = e[l];
      x_conv   = computeConvection(nux[r], nux[bk], nux[l], 
                                qFrt, qMid, qLast, qConFrt, 
                                qConLast, delT, dx.x());

      qConFrt  = vel[t].y() * (e[t] + p[t]);
      qConLast = vel[b].y() * (e[b] + p[b]);
      qFrt     = e[t];
      qLast    = e[b];

      y_conv = computeConvection(nuy[t], nuy[bk], nuy[b],
                      qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.y()); 

      double vel_sqr = vel[bk].length2();

      term1 = 0.5 * d1_z[bk] * vel_sqr;
      term2 = d2_z[bk] /(gamma - 1.0) 
            + rho_tmp[bk] * vel[bk].x() * d4_z[bk];
      term3 = rho_tmp[bk] * vel[bk].y() * d5_z[bk] 
            + rho_tmp[bk] * vel[bk].z() * d3_z[bk];
      term4 = x_conv + y_conv;
      double e_tmp = e[bk] - delT * (term1 + term2 + term3 + term4);

      temp_CC[bk] = e_tmp/rho_CC[bk]/cv - 0.5 * vel_sqr/cv;   
    } // end of j loop
  } //end of i loop
	
  //__________________________________________________________
  //  E D G E      right-back
  for(int j = low.y()+1; j < hi_y; j++) {

    r  =  IntVector(hi_x, j,   low.z());
    t  =  IntVector(hi_x, j+1, low.z());
    b  =  IntVector(hi_x, j-1, low.z());

    qConFrt  = vel[t].y() * (e[t] + p[t]);
    qConLast = vel[b].y() * (e[b] + p[b]);
    qFrt     = e[t];
    qMid     = e[r];
    qLast    = e[b];

    y_conv = computeConvection(nuy[t], nuy[r], nuy[b],
                     qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.y());

    double vel_sqr = vel[r].length2();

    term1 = 0.5 * (d1_x[r] + d1_z[r]) * vel_sqr;
    term2 =       (d2_x[r] + d2_z[r])/(gamma - 1.0) 
          +  rho_tmp[r] * vel[r].x() * (d3_x[r] + d4_z[r]);
    term3 =  rho_tmp[r] * vel[r].y() * (d4_x[r] + d5_z[r]) 
          +  rho_tmp[r] * vel[r].z() * (d5_x[r] + d3_z[r]);

    double e_tmp = e[r] - delT * ( term1 + term2 + term3 + y_conv);

    temp_CC[r] = e_tmp/rho_CC[r]/cv - 0.5 * vel_sqr/cv;
  } 
  //__________________________________________________________
  //  E D G E        left- back
  for(int j = low.y()+1; j < hi_y; j++) {
    l  =  IntVector(low.x(), j,   low.z());
    t  =  IntVector(low.x(), j+1, low.z());
    b  =  IntVector(low.x(), j-1, low.z());

    qConFrt  = vel[t].y() * (e[t] + p[t]);
    qConLast = vel[b].y() * (e[b] + p[b]);
    qFrt     = e[t];
    qMid     = e[l];
    qLast    = e[b];

    y_conv = computeConvection(nuy[t], nuy[l], nuy[b],
                     qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.y());

    double vel_sqr = vel[l].length2();

    term1 = 0.5 * (d1_x[l] + d1_z[l]) * vel_sqr;
    term2 =       (d2_x[l] + d2_z[l])/(gamma - 1.0) 
          +  rho_tmp[l] * vel[l].x() * (d3_x[l] + d4_z[l]);
    term3 =  rho_tmp[l] * vel[l].y() * (d4_x[l] + d5_z[l]) 
          +  rho_tmp[l] * vel[l].z() * (d5_x[l] + d3_z[l]);

    double e_tmp = e[l] - delT * ( term1 + term2 + term3 + y_conv);

    temp_CC[l] = e_tmp/rho_CC[l]/cv - 0.5 * vel_sqr/cv;
  }  
  //_______________________________________________________
  //    E D G E   top --back
  for(int i = low.x()+1; i < hi_x; i++) {
    r  =  IntVector(i+1, hi_y,   low.z());
    l  =  IntVector(i-1, hi_y,   low.z());
    t  =  IntVector(i,   hi_y,   low.z());
    qConFrt  = vel[r].x() * (e[r] + p[r]);
    qConLast = vel[l].x() * (e[l] + p[l]);
    qFrt     = e[r];
    qMid     = e[t];
    qLast    = e[l];

    x_conv = computeConvection(nux[r], nux[t], nux[l],
                     qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.x());

    double vel_sqr = vel[t].length2();

    term1 = 0.5 * (d1_y[t] + d1_z[t]) * vel_sqr;
    term2 =       (d2_y[t] + d2_z[t])/(gamma - 1.0) 
          +  rho_tmp[t] * vel[t].x() * (d4_y[t] + d4_z[t]);
    term3 =  rho_tmp[t] * vel[t].y() * (d3_y[t] + d5_z[t]) 
          +  rho_tmp[t] * vel[t].z() * (d5_y[t] + d3_z[t]);

    double e_tmp = e[t] - delT * ( term1 + term2 + term3 + x_conv);

    temp_CC[t] = e_tmp/rho_CC[t]/cv - 0.5 * vel_sqr/cv;
  } 

  //_______________________________________________________
  //    E D G E   bottom-back
  for(int i = low.x()+1; i < hi_x; i++) {
    r  =  IntVector(i+1, low.y(), low.z());
    l  =  IntVector(i-1, low.y(), low.z());
    b  =  IntVector(i,   low.y(), low.z());
    qConFrt  = vel[r].x() * (e[r] + p[r]);       
    qConLast = vel[l].x() * (e[l] + p[l]);
    qFrt     = e[r];
    qMid     = e[b];
    qLast    = e[l];

    x_conv = computeConvection(nux[r], nux[b], nux[l],
                     qFrt, qMid, qLast, qConFrt, qConLast, delT, dx.x());

    double vel_sqr = vel[b].length2();

    term1 = 0.5 * (d1_y[b] + d1_z[b]) * vel_sqr;
    term2 =       (d2_y[b] + d2_z[b])/(gamma - 1.0) 
          +  rho_tmp[b] * vel[b].x() * (d4_y[b] + d4_z[b]);
    term3 =  rho_tmp[b] * vel[b].y() * (d3_y[b] + d5_z[b]) 
          +  rho_tmp[b] * vel[b].z() * (d5_y[b] + d3_z[b]);

    double e_tmp = e[b] - delT * ( term1 + term2 + term3 + x_conv);

    temp_CC[b] = e_tmp/rho_CC[b]/cv - 0.5 * vel_sqr/cv;
  } 
  //________________________________________________________
  // C O R N E R S    
  vector<IntVector> crn(4);
  crn[0] = IntVector(hi_x,    low.y(), low.z());  // right-bottom-back
  crn[1] = IntVector(hi_x,    hi_y,    low.z());  // right-top-back   
  crn[2] = IntVector(low.x(), hi_y,    low.z());  // left-top-back
  crn[3] = IntVector(low.x(), low.y(), low.z());  // left-bottom-back 

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
} //end of the function zMinusTempLODI()


/*_________________________________________________________________
 Function~ fillFaceTempLODI--n
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
  cout_doing << " I am in fillFaceTempLODI" << endl;
  switch(face) {
   case Patch::xplus:{ 
     xPlusTempLODI(temp_CC,
                   d1_x, d2_x, d3_x, d4_x, d5_x,
                   d1_y, d2_y, d3_y, d4_y, d5_y,
                   d1_z, d2_z, d3_z, d4_z, d5_z,
                   e, rho_CC,
	            nux,nuy,nuz,
	            rho_tmp, p, vel, 
                   delT, cv, gamma, dx);
   } 
   break;
   case Patch::xminus:{ 
     xMinusTempLODI(temp_CC,
                    d1_x, d2_x, d3_x, d4_x, d5_x,
                    d1_y, d2_y, d3_y, d4_y, d5_y,
                    d1_z, d2_z, d3_z, d4_z, d5_z,
                    e, rho_CC,
		      nux,nuy,nuz,
		      rho_tmp, p, vel, 
                    delT, cv, gamma, dx);
   } 
   break;
   case Patch::yplus:{ 
     yPlusTempLODI(temp_CC,
                   d1_x, d2_x, d3_x, d4_x, d5_x,
                   d1_y, d2_y, d3_y, d4_y, d5_y,
                   d1_z, d2_z, d3_z, d4_z, d5_z,
                   e, rho_CC,
	            nux,nuy,nuz,
	            rho_tmp, p, vel, 
                   delT, cv, gamma, dx);
   }
   break;

   case Patch::yminus:{ 
     yMinusTempLODI(temp_CC,
                    d1_x, d2_x, d3_x, d4_x, d5_x,
                    d1_y, d2_y, d3_y, d4_y, d5_y,
                    d1_z, d2_z, d3_z, d4_z, d5_z,
                    e, rho_CC,
		      nux,nuy,nuz,
		      rho_tmp, p, vel, 
                    delT, cv, gamma, dx);
   } 
   break;
   case Patch::zplus:{ 
     zPlusTempLODI(temp_CC,
                   d1_x, d2_x, d3_x, d4_x, d5_x,
                   d1_y, d2_y, d3_y, d4_y, d5_y,
                   d1_z, d2_z, d3_z, d4_z, d5_z,
                   e, rho_CC,
	            nux,nuy,nuz,
	            rho_tmp, p, vel, 
                   delT, cv, gamma, dx);
   }
   break;
   case Patch::zminus:{
     zMinusTempLODI(temp_CC,
                    d1_x, d2_x, d3_x, d4_x, d5_x,
                    d1_y, d2_y, d3_y, d4_y, d5_y,
                    d1_z, d2_z, d3_z, d4_z, d5_z,
                    e, rho_CC,
		      nux,nuy,nuz,
		      rho_tmp, p, vel, 
                    delT, cv, gamma, dx);
   }
   break;

   default:
   break;
  }
  cout_doing << "end of computing temperature" << endl;              
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
