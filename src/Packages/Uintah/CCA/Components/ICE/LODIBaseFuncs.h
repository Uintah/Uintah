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
#include <typeinfo>
#include <Core/Util/DebugStream.h>

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

double computeConvection(const double& k_const,   const double& nuFrt, 
                         const double& nuMid,     const double& nuLast, 
                         const double& qFrt,      const double& qMid, 
                         const double& qLast,     const double& qConFrt, 
                         const double& qConLast,  const double& delta)
{
   //__________________________________
   // Artifical dissipation term
   double eplus, eminus, dissipation;

   eplus  = 0.5 * k_const * delta * (nuFrt   + nuMid)/delta;
   eminus = 0.5 * k_const * delta * (nuLast  + nuMid)/delta;
   dissipation = (eplus * qFrt - (eplus + eminus) * qMid +  eminus * qLast)/delta; 
 
   cout_dbg << "qConFrt,qConLast,nuFrt,nuMid,nuLast,eplus,eminus,dissipation" << 
           qConFrt << "," << qConLast << "," << nuFrt << "," << nuMid << "," << nuLast << 
           "," << eplus << "," << eminus << "," << dissipation << endl;  
   return 0.5 * (qConFrt - qConLast)/delta - dissipation;
                        
}

/*_________________________________________________________________
 Function~ fillFaceDensityLODI--
 Purpose~  Compute density on boundary cells using Characteristic 
           Boundary Condition
___________________________________________________________________*/

void fillFaceDensityLODI(CCVariable<double>& rho_CC,
                   const CCVariable<double>& d1_x, 
                   const CCVariable<double>& d1_y, 
                   const CCVariable<double>& d1_z, 
                   const CCVariable<double>& nux,
                   const CCVariable<double>& nuy,
                   const CCVariable<double>& nuz,
                   const CCVariable<double>& rho_tmp,
                   const CCVariable<double>& p,
                   const CCVariable<Vector>& vel,
                   const CCVariable<double>& c,
                   const Patch::FaceType face,
                   const double delT,
                   const double gamma,
                   const double R_gas,
                   const int mat_id, 
                   const Vector& dx)

{
     cout_doing << " I am in fillFaceDensityLODI" << endl;    
     IntVector low,hi,r,l,t,b,f,bk;

     low = p.getLowIndex();
     hi  = p.getHighIndex();
     int hi_x = hi.x() - 1;
     int hi_y = hi.y() - 1;
     int hi_z = hi.z() - 1; 
     double k_const = 0.3;
     double y_conv = 0.0, z_conv = 0.0;
    
     switch(face) { // switch:4
       case Patch::xplus:
       { //case: 5
        
         //___________________________________________________________________
         // Compute the density on the area of i = hi_x, low.y() < j < hi_y
         // and low.z() < k < hi_z 
        //for(int j = low.y()+1; j < hi_y; j++) {
        //   for(int k = low.z()+1; k < hi_z; k++) {
        for(int j = low.y(); j <= hi_y; j++) {
           for(int k = low.z(); k <= hi_z; k++) {
          
          r  =  IntVector(hi_x,  j,   k);
          l  =  IntVector(hi_x-1,j,   k);
#if 0
          t  =  IntVector(hi_x,  j+1, k);
          b  =  IntVector(hi_x,  j-1, k);
          f  =  IntVector(hi_x,  j,   k+1);
          bk =  IntVector(hi_x,  j,   k-1);       
                     
          double qConFrt, qConLast;
          //__________________________________________
          // mass conservation law, computing density
          qConFrt  = rho_tmp[t] * vel[t].y();
          qConLast = rho_tmp[b] * vel[b].y();
          y_conv = computeConvection(k_const, nuy[t], nuy[r], nuy[b], 
                                     rho_tmp[t], rho_tmp[r], rho_tmp[b], 
                                     qConFrt, qConLast, dx.y());
                                  
          qConFrt  = rho_tmp[f] * vel[f].z();
          qConLast = rho_tmp[bk] * vel[bk].z();
          z_conv = computeConvection(k_const, nuz[f], nuz[r], nuz[bk],
                                     rho_tmp[f], rho_tmp[r], rho_tmp[bk], 
                                     qConFrt, qConLast, dx.z());

#endif

       // rho_CC[r] = rho_tmp[r] - delT * (d1_x[r] + y_conv + z_conv);
          rho_CC[r] = rho_tmp[r] - delT * d1_x[r] ;

          cout_dbg.setf(ios::scientific,ios::floatfield);
          cout_dbg.precision(16);
          cout_dbg << "rho_CC,rho_tmp, d1_x, dt, y_conv" << r << "= " << rho_CC[r] << ",";
          cout_dbg << rho_tmp[r] << "," << d1_x[r] << "," << delT << y_conv << endl;  

       } // end of j loop
      } //end of k loop
     // end of computing density on the rea of low.y() < j < hi_y 
     // and low.z() < k < hi_z 
 /*
     //__________________________________________________________
     //  E D G E
     // Compute density on the edge of i = hi_x, j = low.y() 
     // and low.z() < k < hi_z, this needs to compute the 
     // chracteristic waves in both x and y direction.
        for(int k = low.z()+1; k < hi_z; k++) {
          
          r  =  IntVector(hi_x,   low.y(),   k);
          l  =  IntVector(hi_x-1, low.y(),   k);
          t  =  IntVector(hi_x,   low.y()+1, k);
          f  =  IntVector(hi_x,   low.y(),   k+1);
          bk =  IntVector(hi_x,   low.y(),   k-1);
          
        //__________________________________________
         // mass conservation law, computing density
          qConFrt  = rho_tmp[f]  * vel[f].z();
          qConLast = rho_tmp[bk] * vel[bk].z();
          z_conv = computeConvection(k_const, nuz[f], nuz[r], nuz[bk],
                                     rho_tmp[f], rho_tmp[r], rho_tmp[bk], 
                                     qConFrt, qConLast, dx.z());

          rho_CC[r] = rho_tmp[r] - delT * (d1_x[r] + d1_y[r] + z_conv);

        } //end of k loop
        // end of computing density on the edge of j = low.y() 
        // and low.z() < k < hi_z

        //_______________________________________________________
        //    E D G E
        // Compute the edge of i = hi_x, j = hi_y 
        // and low.z() < k < hi_z, this needs to compute the 
        // chracteristic waves in y dircetion too.
          for(int k = low.z()+1; k < hi_z; k++) {
          
          r  =  IntVector(hi_x,   hi_y,   k);
          l  =  IntVector(hi_x-1, hi_y,   k);
          b  =  IntVector(hi_x,   hi_y-1, k);
          f  =  IntVector(hi_x,   hi_y,   k+1);
          bk =  IntVector(hi_x,   hi_y,   k-1);
       
        //__________________________________________
        // mass conservation law, computing density
           qConFrt  = rho_tmp[f]  * vel[f].z();
           qConLast = rho_tmp[bk] * vel[bk].z();
             z_conv = computeConvection(k_const, nuz[f], nuz[r], nuz[bk],
                                     rho_tmp[f], rho_tmp[r], rho_tmp[bk], 
                                     qConFrt, qConLast, dx.z());
          rho_CC[r] = rho_tmp[r] - delT * (d1_x[r] + d1_y[r] + z_conv);
        } //end of k loop
        // end of computing on the edge of i = hi_x, 
       // j = hi_y and low.z() < k < hi_z
         
        //_____________________________________________________
       //    E D G E
        // Compute the edge of i = hi_x, k = low.z() 
        // and low.y() < j < hi_y, this needs to compute the 
        // chracteristic waves in both x and z dircetion.
           for(int j = low.y()+1; j < hi_y; j++) {
          
          r  =  IntVector(hi_x,   j,   low.z());
          l  =  IntVector(hi_x-1, j,   low.z());
          t  =  IntVector(hi_x,   j+1, low.z());
          b  =  IntVector(hi_x,   j-1, low.z());
          f  =  IntVector(hi_x,   j,   low.z()+1);
       

        //__________________________________________
        // mass conservation law, computing density
           qConFrt  = rho_tmp[t] * vel[t].y();
           qConLast = rho_tmp[b] * vel[b].y();
             y_conv = computeConvection(k_const, nuy[t], nuy[r], nuy[b],
                                     rho_tmp[t], rho_tmp[r], rho_tmp[b], 
                                     qConFrt, qConLast, dx.y());
          rho_CC[r] = rho_tmp[r] - delT * (d1_x[r] + d1_z[r] + y_conv);
        } //end of j loop
        // end of computing on the edge of i = hi_x, k = low.z() 
        // and low.y() < j < hi_y

        //_________________________________________________
        // E D G E
        // Compute the edge of i = hi_x, k = hi_z 
        // and low.y() < j < hi_y, this needs to compute the 
        // chracteristic waves in both  x and z dircetion.
        for(int j = low.y()+1; j < hi_y; j++) {
          r  =  IntVector(hi_x,   j,   hi_z);
          l  =  IntVector(hi_x-1, j,   hi_z);
          t  =  IntVector(hi_x,   j+1, hi_z);
          b  =  IntVector(hi_x,   j-1, hi_z);
          bk =  IntVector(hi_x,   j,   hi_z-1);
       
          //__________________________________________
          // mass conservation law, computing density
          qConFrt  = rho_tmp[t] * vel[t].y();
          qConLast = rho_tmp[b] * vel[b].y();
          y_conv   = computeConvection(k_const, nuy[t], nuy[r], nuy[b],
                                    rho_tmp[t], rho_tmp[r], rho_tmp[b], 
                                    qConFrt, qConLast, dx.y());
          rho_CC[r] = rho_tmp[r] - delT * (d1_x[r] + d1_z[r] + y_conv);
        
        } //end of j loop
        // end of computing on the edge of k = hi_z 
        // and low.y() < j < hi_y
        
        //________________________________________________________
        // C O R N E R
        // Compute density on the corner of i = hi_x, j = low.y() 
        // and k = hi_z, this needs to compute the chracteristic 
        // waves in x, y and z dircetion.
        r   =  IntVector(hi_x,   low.y(),   hi_z);
        l   =  IntVector(hi_x-1, low.y(),   hi_z);
        t   =  IntVector(hi_x,   low.y()+1, hi_z);
        bk  =  IntVector(hi_x,   low.y(),   hi_z-1);
          
        //__________________________________________
        // mass conservation law, computing density
        rho_CC[r] = rho_tmp[r] - delT * (d1_x[r] + d1_y[r] + d1_z[r]);
        
        // end of computing on the corner of i = hi_x,
        // j = low.y() and k = hi_z

        //_________________________________________________________
       // C O R N E R
        // Compute the corner of i = hi_x, j = hi_y 
        // and k = hi_z,this needs to compute the chracteristic 
        // waves in x, y and z dircetion.
          
          r   =  IntVector(hi_x,   hi_y,    hi_z);
          l   =  IntVector(hi_x-1, hi_y,    hi_z);
          b   =  IntVector(hi_x,   hi_y-1,  hi_z);
          bk  =  IntVector(hi_x,   hi_y,    hi_z-1);
         

        //__________________________________________
        // mass conservation law, computing density
          rho_CC[r] = rho_tmp[r] - delT * (d1_x[r] + d1_y[r] + d1_z[r]);
         
        // end of computing on the corner of i = hi_x,
        // j = hi_y and k = hi_z
       
        //__________________________________________________________
        //  C O R N E R
        // Compute density on the corner of i = hi_x, j = hi_y 
        // and k = low.z(),this needs to compute the chracteristic 
        // waves in x, y and z dircetion.
          
          r   =  IntVector(hi_x,   hi_y,   low.z());
          l   =  IntVector(hi_x-1, hi_y,   low.z());
          b   =  IntVector(hi_x,   hi_y-1, low.z());
          f   =  IntVector(hi_x,   hi_y,   low.z()+1);
          

        //__________________________________________
        //mass conservation law, computing density
          rho_CC[r] = rho_tmp[r] - delT * (d1_x[r] + d1_y[r] + d1_z[r]);
       
        // end of computing density on the corner of i = hi_x,
        // j = hi_y and k = low.z()
        
        //__________________________________________________________
        //  C O R N E R
        // Compute density on the corner of i = hi_x, j = low.y() 
        // and k = low.z(),this needs to compute the chracteristic 
        // waves in x, y and z dircetion.
          
          r   =  IntVector(hi_x,   low.y(),   low.z());
          l   =  IntVector(hi_x-1, low.y(),   low.z());
          t   =  IntVector(hi_x,   low.y()+1, low.z());
          f   =  IntVector(hi_x,   low.y(),   low.z()+1);
          
        //__________________________________________
        // mass conservation law, computing density
          rho_CC[r] = rho_tmp[r] - delT * (d1_x[r] + d1_y[r] + d1_z[r]);
       
        // end of computing density on the corner of i = hi_x,
        // j = low.y() and k = low.z()
         
 */
                
       } // end of case x_plus: 5
       break;
        //here will insert the other 5 faces: xminus, yplus, yminus, zplus, zminus
       default:
       break;
      }//end of switch: 4   
                        
}


/*_________________________________________________________________
 Function~ fillFaceVelLODI--
 Purpose~  Compute velocity at boundary cells using Characteristic 
           Boundary Condition
___________________________________________________________________*/
void fillFaceVelLODI(CCVariable<Vector>& vel_CC,
                 const CCVariable<double>& d1_x,  
                 const CCVariable<double>& d3_x, 
                 const CCVariable<double>& d4_x, 
                 const CCVariable<double>& d1_y,  
                 const CCVariable<double>& d3_y, 
                 const CCVariable<double>& d4_y, 
                 const CCVariable<double>& d1_z,  
                 const CCVariable<double>& d3_z, 
                 const CCVariable<double>& d4_z,
                 const CCVariable<double>& nux,
                 const CCVariable<double>& nuy,
                 const CCVariable<double>& nuz,
                 const CCVariable<double>& rho_tmp,
                 const CCVariable<double>& p,
                 const CCVariable<Vector>& vel,
                 const CCVariable<double>& c,
                 const Patch::FaceType face,
                 const double delT,
                 const double gamma,
                 const double R_gas,
                 const int mat_id,
                 const Vector& dx)

{       
         cout_doing << "I am in fillFaceVelLODI()" << endl;
         double qConFrt,qConLast,qFrt,qMid,qLast;
         double uVel, vVel, wVel;
         IntVector low,hi,r,l,t,b,f,bk;
         low = p.getLowIndex();
         hi  = p.getHighIndex();
         int hi_x = hi.x() - 1;
         int hi_y = hi.y() - 1;
         int hi_z = hi.z() - 1;
         double y_conv,z_conv;
         double k_const = 0.3;

         switch (face) { // switch:4
         case Patch::xplus:
          { //case: 5
        
         //___________________________________________________________________
         // Compute the velocity on the area of i = hi_x, low.y() < j < hi_y
         // and low.z() < k < hi_z 
         for(int j = low.y()+1; j < hi_y; j++) {
          for(int k = low.z()+1; k < hi_z; k++) {
          
           r  =  IntVector(hi_x,  j,   k);
           l  =  IntVector(hi_x-1,j,   k);
           t  =  IntVector(hi_x,  j+1, k);
           b  =  IntVector(hi_x,  j-1, k);
           f  =  IntVector(hi_x,  j,   k+1);
           bk =  IntVector(hi_x,  j,   k-1);

        //__________________________________________________________________
        //Solve momentum conservation law in x-direction, computing the vel[r].x()
          qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].x();
          qConLast = rho_tmp[b] * vel[b].y() * vel[b].x();
          qFrt   = rho_tmp[t] * vel[t].x();
          qMid   = rho_tmp[r] * vel[r].x();
          qLast  = rho_tmp[b] * vel[b].x();
          y_conv = computeConvection(k_const, nuy[t], nuy[r], nuy[b], 
                                     qFrt, qMid, qLast, qConFrt, 
                                     qConLast, dx.y());

          qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].x();
          qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].x();
            qFrt   = rho_tmp[f]  * vel[f].x();
            qLast  = rho_tmp[bk] * vel[bk].x();

            z_conv = computeConvection(k_const, nuz[f], nuz[r], nuz[bk],
                                       qFrt, qMid, qLast, qConFrt,qConLast, dx.z());       
              uVel = (qMid - delT * (d1_x[r] * vel[r].x() 
                   +  rho_tmp[r] * d3_x[r] + y_conv + z_conv ))/rho_tmp[r];

             cout_dbg << "uVel, y_conv, z_conv" << r << "= " << uVel
                  << "," << y_conv  << "," << z_conv << endl;
            
         //__________________________________________________________________
         // Solve momentum conservation law in y-direction, computing vel[r].y()        
          qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].y() + p[t];
          qConLast = rho_tmp[b] * vel[b].y() * vel[b].y() + p[b];
            qFrt   = rho_tmp[t] * vel[t].y();
            qMid   = rho_tmp[r] * vel[r].y();
            qLast  = rho_tmp[b] * vel[b].y();
            y_conv = computeConvection(k_const, nuy[t], nuy[r], nuy[b], 
                                       qFrt, qMid, qLast, qConFrt, 
                                       qConLast, dx.y());
                                  
          qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].y();
          qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].y();
            qFrt   = rho_tmp[f]  * vel[f].y();
            qLast  = rho_tmp[bk] * vel[bk].y();

          z_conv = computeConvection(k_const, nuz[f], nuz[r], nuz[bk],
                                     qFrt, qMid, qLast, qConFrt,qConLast, dx.z());        
        
            vVel = (qMid- delT * (d1_x[r] * vel[r].y()
                        +  rho_tmp[r] * d4_x[r] + y_conv + z_conv))/rho_tmp[r];
                        
            cout_dbg << "vVel,y_conv, z_conv" << r << "= " << vVel
                  << "," << y_conv  << "," << z_conv << endl;
             cout_dbg << "p" << t << "= " << p[t] << "," << "p" << b << "= " << p[b] << endl; 

        //________________________________________________________________
        //Solve momentum conservation law in z-direction, computing vel[r].z()
          qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].z();
          qConLast = rho_tmp[b] * vel[b].y() * vel[b].z();
            qFrt   = rho_tmp[t] * vel[t].z();
            qMid   = rho_tmp[r] * vel[r].z();
            qLast  = rho_tmp[b] * vel[b].z();
            y_conv = computeConvection(k_const, nuy[t], nuy[r], nuy[b], 
                                       qFrt, qMid, qLast, qConFrt, 
                                       qConLast, dx.y());
                                  
          qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].z()  + p[f];
          qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].z() + p[bk];
            qFrt   = rho_tmp[f]  * vel[f].z();
            qLast  = rho_tmp[bk] * vel[bk].z();

          z_conv = computeConvection(k_const, nuz[f], nuz[r], nuz[bk],
                                     qFrt, qMid, qLast, qConFrt,qConLast, dx.z());        
                 
        
            wVel = (qMid - delT * (d1_x[r] * vel[r].z() + rho_tmp[r] * d4_x[r]
                        + y_conv + z_conv))/rho_tmp[r];
           vel_CC[r] = Vector(uVel, vVel, wVel);

             cout_dbg << "wVel, y_conv, z_conv" << r << "= " << wVel
                  << "," << y_conv  << "," << z_conv << endl;
             cout_dbg << "p" << f << "= " << p[f] << "," << "p" << bk << "= " << p[bk] << endl;


        } // end of j loop
        } //end of k loop
        // end of computing velocity on the area of i = hi_x, low.y() < j < hi_y 
        // and low.z() < k < hi_z 
 
        //__________________________________________________________
        //  E D G E
        // Compute velocity on the edge of i = hi_x, j = low.y() 
        // and low.z() < k < hi_z, this needs to compute the 
        // chracteristic waves in both x and y dircetion.
         for(int k = low.z()+1; k < hi_z; k++) {
          
           r  =  IntVector(hi_x,   low.y(),   k);
           l  =  IntVector(hi_x-1, low.y(),   k);
           t  =  IntVector(hi_x,   low.y()+1, k);
           f  =  IntVector(hi_x,   low.y(),   k+1);
           bk =  IntVector(hi_x,   low.y(),   k-1);
          

         double qConFrt,qConLast;
         double qFrt,qMid,qLast;

        //__________________________________________________________________
        // Solve momentum conservation law in x-direction, computing the vel[r].x()                          
          qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].x();
          qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].x();
            qFrt   = rho_tmp[f]  * vel[f].x();
            qMid   = rho_tmp[r]  * vel[r].x();
            qLast  = rho_tmp[bk] * vel[bk].x();

            z_conv = computeConvection(k_const, nuz[f], nuz[r], nuz[bk],
                                       qFrt, qMid, qLast, 
                                   qConFrt,qConLast, dx.z());       
              
              uVel = (qMid - delT * ((d1_x[r] + d1_y[r]) * vel[r].x() + 
                                  +  (d3_x[r] + d4_y[r]) * rho_tmp[r] 
                                  +   z_conv))/rho_tmp[r];
             cout_dbg << "uVel, z_conv" << r << "= " << uVel
                  << "," << z_conv  << endl;
        
        //__________________________________________________________________
        // Solve momentum conservation law in y-direction, computing vel[r].y()
          qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].y();
          qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].y();
            qFrt   = rho_tmp[f]  * vel[f].y();
            qMid   = rho_tmp[r]  * vel[r].y();
            qLast  = rho_tmp[bk] * vel[bk].y();

          z_conv = computeConvection(k_const, nuz[f], nuz[r], nuz[bk],
                         qFrt, qMid, qLast, qConFrt,qConLast, dx.z()); 
        
            vVel = (qMid - delT * ((d1_x[r] + d1_y[r]) * vel[r].y() 
                                +  (d4_x[r] + d3_y[r]) * rho_tmp[r] + z_conv))
                                    /rho_tmp[r];

             cout_dbg << "vVel, z_conv" << r << "= " << vVel
                  << "," << z_conv  << endl;

        //________________________________________________________________
        // Solve momentum conservation law in z-direction, computing vel[r].z()
          qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].z()  + p[f];
          qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].z() + p[bk];
            qFrt   = rho_tmp[f]  * vel[f].z();
            qMid   = rho_tmp[r]  * vel[r].z();
            qLast  = rho_tmp[bk] * vel[bk].z();

            z_conv = computeConvection(k_const, nuz[f], nuz[r], nuz[bk],
                            qFrt, qMid, qLast, qConFrt, qConLast, dx.z());
        
              wVel = (qMid - delT * ((d1_x[r] + d1_y[r]) * vel[r].z() 
                                  +  (d4_x[r] + d4_y[r]) * rho_tmp[r]  
                                  + z_conv))/rho_tmp[r];
         vel_CC[r] = Vector(uVel, vVel, wVel);

             cout_dbg << "wVel, z_conv" << r << "= " << wVel
                  << "," << z_conv  << endl;
             cout_dbg << "p" << f << "= " << p[f] << "," << "p" << bk << "= " << p[bk] << endl; 

        } //end of k loop
        // end of computing density on the edge of j = low.y() 
        // and low.z() < k < hi_z

        //_______________________________________________________
        //    E D G E
        // Compute the edge of i = hi_x, j = hi_y 
        // and low.z() < k < hi_z, this needs to compute the 
        // chracteristic waves in y dircetion too.
        for(int k = low.z()+1; k < hi_z; k++) {
          
            r  =  IntVector(hi_x,   hi_y,   k);
            l  =  IntVector(hi_x-1, hi_y,   k);
            b  =  IntVector(hi_x,   hi_y-1, k);
            f  =  IntVector(hi_x,   hi_y,   k+1);
            bk =  IntVector(hi_x,   hi_y,   k-1);
       
       
        //__________________________________________________________________
        // Solve momentum conservation law in x-direction, computing the vel[r].x()                          
          qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].x();
          qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].x();
            qFrt   = rho_tmp[f]  * vel[f].x();
            qMid   = rho_tmp[r]  * vel[r].x();
            qLast  = rho_tmp[bk] * vel[bk].x();

            z_conv = computeConvection(k_const, nuz[f], nuz[r], nuz[bk],
                                       qFrt, qMid, qLast, 
                                   qConFrt,qConLast, dx.z());       
              
              uVel = (qMid - delT * ((d1_x[r] + d1_y[r]) * vel[r].x() + 
                                  +  (d3_x[r] + d4_y[r]) * rho_tmp[r] 
                                   +   z_conv))/rho_tmp[r];

             cout_dbg << "uVel, z_conv" << r << "= " << uVel
                  << "," << z_conv  << endl;
        //__________________________________________________________________
        // Solve momentum conservation law in y-direction, computing vel[r].y()
          qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].y();
          qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].y();
            qFrt   = rho_tmp[f]  * vel[f].y();
            qMid   = rho_tmp[r]  * vel[r].y();
            qLast  = rho_tmp[bk] * vel[bk].y();

          z_conv = computeConvection(k_const, nuz[f], nuz[r], nuz[bk],
                         qFrt, qMid, qLast, qConFrt,qConLast, dx.z()); 
        
            vVel = (qMid - delT * ((d1_x[r] + d1_y[r]) * vel[r].y() 
                                +  (d4_x[r] + d3_y[r]) * rho_tmp[r] + z_conv))
                                     /rho_tmp[r];

             cout_dbg << "vVel, z_conv" << r << "= " << vVel
                  << "," << z_conv  << endl;

        //________________________________________________________________
        // Solve momentum conservation law in z-direction, computing vel[r].z()
          qConFrt  = rho_tmp[f]  * vel[f].z()  * vel[f].z()  + p[f];
          qConLast = rho_tmp[bk] * vel[bk].z() * vel[bk].z() + p[bk];
            qFrt   = rho_tmp[f]  * vel[f].z();
            qMid   = rho_tmp[r]  * vel[r].z();
            qLast  = rho_tmp[bk] * vel[bk].z();

            z_conv = computeConvection(k_const, nuz[f], nuz[r], nuz[bk],
                            qFrt, qMid, qLast, qConFrt, qConLast, dx.z());
        
              wVel = (qMid - delT * ((d1_x[r] + d1_y[r]) * vel[r].z() 
                                  +  (d4_x[r] + d4_y[r]) * rho_tmp[r]  
                                    + z_conv))/rho_tmp[r];
         vel_CC[r] = Vector(uVel, vVel, wVel);

           cout_dbg << "wVel, z_conv" << r << "= " << wVel
                << "," << z_conv  << endl;
           cout_dbg << "p" << f << "= " << p[f] << "," << "p" << bk << "= " << p[bk] << endl; 

        } //end of k loop
        // end of computing on the edge of i = hi_x, j = low.y() 
        // and low.z() < k < hi_z
         
        //_____________________________________________________
        //    E D G E
        // Compute the edge of i = hi_x, k = low.z() 
        // and low.y() < j < hi_y, this needs to compute the 
        // chracteristic waves in both x and z dircetion.
        for(int j = low.y()+1; j < hi_y; j++) {
          
          r  =  IntVector(hi_x,   j,   low.z());
          l  =  IntVector(hi_x-1, j,   low.z());
          t  =  IntVector(hi_x,   j+1, low.z());
          b  =  IntVector(hi_x,   j-1, low.z());
          f  =  IntVector(hi_x,   j,   low.z()+1);
       

        //__________________________________________________________________
        // Solve momentum conservation law in x-direction, computing the vel[r].x()                          
          qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].x();
          qConLast = rho_tmp[b] * vel[b].y() * vel[b].x();
            qFrt   = rho_tmp[t] * vel[t].x();
            qMid   = rho_tmp[r] * vel[r].x();
            qLast  = rho_tmp[b] * vel[b].x();

            y_conv = computeConvection(k_const, nuy[t], nuy[r], nuy[b],
                                       qFrt, qMid, qLast, 
                                   qConFrt,qConLast, dx.y());       
              
              uVel = (qMid - delT * ((d1_x[r] + d1_z[r]) * vel[r].x() + 
                                  +  (d3_x[r] + d4_z[r]) * rho_tmp[r] 
                                   +   y_conv))/rho_tmp[r];
             cout_dbg << "uVel, y_conv" << r << "= " << uVel
                  << "," << y_conv  << endl;
        
        //__________________________________________________________________
        // Solve momentum conservation law in y-direction, computing vel[r].y()
          qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].y() + p[t];
          qConLast = rho_tmp[b] * vel[b].y() * vel[b].y() + p[b];
            qFrt   = rho_tmp[t] * vel[t].y();
            qMid   = rho_tmp[r] * vel[r].y();
            qLast  = rho_tmp[b] * vel[b].y();

          y_conv = computeConvection(k_const, nuy[t], nuy[r], nuy[b],
                         qFrt, qMid, qLast, qConFrt,qConLast, dx.y()); 
        
            vVel = (qMid - delT * ((d1_x[r] + d1_z[r]) * vel[r].y() 
                                +  (d4_x[r] + d4_z[r]) * rho_tmp[r] + y_conv))
                                     /rho_tmp[r];

             cout_dbg << "vVel, y_conv" << r << "= " << vVel
                  << "," << y_conv  << endl;
             cout_dbg << "p" << t << "= " << p[t] << "," << "p" << b << "= " << p[b] << endl; 

        //________________________________________________________________
        // Solve momentum conservation law in z-direction, computing vel[r].z()
          qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].z();
          qConLast = rho_tmp[b] * vel[b].y() * vel[b].z();
            qFrt   = rho_tmp[t] * vel[t].z();
            qMid   = rho_tmp[r] * vel[r].z();
            qLast  = rho_tmp[b] * vel[b].z();

            y_conv = computeConvection(k_const, nuy[t], nuy[r], nuy[b],
                            qFrt, qMid, qLast, qConFrt, qConLast, dx.y());
        
              wVel = (qMid - delT * ((d1_x[r] + d1_z[r]) * vel[r].z() 
                                  +  (d4_x[r] + d3_z[r]) * rho_tmp[r]  
                                    + y_conv))/rho_tmp[r];
             vel_CC[r] = Vector(uVel, vVel, wVel);

             cout_dbg << "wVel, y_conv" << r << "= " << wVel
                  << "," << y_conv  << endl;

        } //end of j loop
        // end of computing on the edge of i = hi_x, k = low.z() 
        // and low.y() < j < hi_y

        //_________________________________________________
       // E D G E
        // Compute the edge of i = hi_x, k = hi_z 
        // and low.y() < j < hi_y, this needs to compute the 
        // chracteristic waves in both  x and z dircetion.
           for(int j = low.y()+1; j < hi_y; j++) {

           r  =  IntVector(hi_x,   j,   hi_z);
           l  =  IntVector(hi_x-1, j,   hi_z);
           t  =  IntVector(hi_x,   j+1, hi_z);
           b  =  IntVector(hi_x,   j-1, hi_z);
           bk =  IntVector(hi_x,   j,   hi_z-1);
       
        //__________________________________________________________________
        // Solve momentum conservation law in x-direction, computing the vel[r].x()                          
          qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].x();
          qConLast = rho_tmp[b] * vel[b].y() * vel[b].x();
            qFrt   = rho_tmp[t] * vel[t].x();
            qMid   = rho_tmp[r] * vel[r].x();
            qLast  = rho_tmp[b] * vel[b].x();

            y_conv = computeConvection(k_const, nuy[t], nuy[r], nuy[b],
                                       qFrt, qMid, qLast, 
                                   qConFrt, qConLast, dx.y());       
              
              uVel = (qMid - delT * ((d1_x[r] + d1_z[r]) * vel[r].x() + 
                                  +  (d3_x[r] + d4_z[r]) * rho_tmp[r] 
                                   +   y_conv))/rho_tmp[r];

             cout_dbg << "uVel, y_conv" << r << "= " << uVel
                  << "," << y_conv  << endl;

        
        //__________________________________________________________________
        // Solve momentum conservation law in y-direction, computing vel[r].y()
          qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].y() + p[t];
          qConLast = rho_tmp[b] * vel[b].y() * vel[b].y() + p[b];
            qFrt   = rho_tmp[t] * vel[t].y();
            qMid   = rho_tmp[r] * vel[r].y();
            qLast  = rho_tmp[b] * vel[b].y();

          y_conv = computeConvection(k_const, nuy[t], nuy[r], nuy[b],
                         qFrt, qMid, qLast, qConFrt, qConLast, dx.y()); 
        
            vVel = (qMid - delT * ((d1_x[r] + d1_z[r]) * vel[r].y() 
                                +  (d4_x[r] + d4_z[r]) * rho_tmp[r] + y_conv))
                                     /rho_tmp[r];

             cout_dbg << "vVel, y_conv" << r << "= " << vVel
                  << "," << y_conv  << endl;
             cout_dbg << "p" << t << "= " << p[t] << "," << "p" << b << "= " << p[b] << endl; 


          //________________________________________________________________
          // Solve momentum conservation law in z-direction, computing vel[r].z()
          qConFrt  = rho_tmp[t] * vel[t].y() * vel[t].z();
          qConLast = rho_tmp[b] * vel[b].y() * vel[b].z();
          qFrt   = rho_tmp[t] * vel[t].z();
          qMid   = rho_tmp[r] * vel[r].z();
          qLast  = rho_tmp[b] * vel[b].z();

          y_conv = computeConvection(k_const, nuy[t], nuy[r], nuy[b],
                          qFrt, qMid, qLast, qConFrt, qConLast, dx.y());

          wVel = (qMid - delT * ((d1_x[r] + d1_z[r]) * vel[r].z() 
                              +  (d4_x[r] + d3_z[r]) * rho_tmp[r]  
                                + y_conv))/rho_tmp[r];
          vel_CC[r] = Vector(uVel, vVel, wVel);

          cout_dbg<< "wVel, y_conv" << r << "= " << wVel<< "," << y_conv  << endl;
        } //end of j loop
        // end of computing on the edge of i = hi_x, k = hi_z 
        // and low.y() < j < hi_y
        
        //________________________________________________________
        // C O R N E R
        // Compute density on the corner of i = hi_x, j = low.y() 
        // and k = hi_z, this needs to compute the chracteristic 
        // waves in x, y and z dircetion.
           r   =  IntVector(hi_x,   low.y(),   hi_z);
           l   =  IntVector(hi_x-1, low.y(),   hi_z);
           t   =  IntVector(hi_x,   low.y()+1, hi_z);
           bk  =  IntVector(hi_x,   low.y(),   hi_z-1);
          

        //__________________________________________________________________
        // Solve momentum conservation law in x-direction, computing the vel[r].x()
        uVel = (qMid - delT * ((d1_x[r] + d1_y[r] + d1_z[r]) * vel[r].x()  
                            +  (d3_x[r] + d4_y[r] + d4_z[r]) * rho_tmp[r]))
                               /   rho_tmp[r];
        
        //__________________________________________________________________
        // Solve momentum conservation law in y-direction, computing vel[r].y()
        vVel = (qMid - delT * ((d1_x[r] + d1_y[r] + d1_z[r]) * vel[r].y() 
                            +  (d4_x[r] + d3_y[r] + d4_z[r]) * rho_tmp[r]))
                               /   rho_tmp[r];

        //________________________________________________________________
        // Solve momentum conservation law in z-direction, computing vel[r].z()
        wVel = (qMid - delT * ((d1_x[r] + d1_y[r] + d1_z[r]) * vel[r].z() 
                            +  (d4_x[r] + d4_y[r] + d3_z[r]) * rho_tmp[r]))
                               /   rho_tmp[r];
                               
              vel_CC[r] = Vector(uVel, vVel, wVel);
       // end of computing on the corner of i = hi_x,
        // j = low.y() and k = hi_z

        //_________________________________________________________
        // C O R N E R
        // Compute the corner of i = hi_x, j = hi_y 
        // and k = hi_z,this needs to compute the chracteristic 
        // waves in x, y and z dircetion.
          
          r   =  IntVector(hi_x,   hi_y,    hi_z);
          l   =  IntVector(hi_x-1, hi_y,    hi_z);
          b   =  IntVector(hi_x,   hi_y-1,  hi_z);
          bk  =  IntVector(hi_x,   hi_y,    hi_z-1);

        //__________________________________________________________________
        // Solve momentum conservation law in x-direction, computing the vel[r].x()                          
              
          uVel = (qMid - delT * ((d1_x[r] + d1_y[r] + d1_z[r]) * vel[r].x()  
                              +  (d3_x[r] + d4_y[r] + d4_z[r]) * rho_tmp[r]))
                                     /   rho_tmp[r];
        
        //__________________________________________________________________
        // Solve momentum conservation law in y-direction, computing vel[r].y()
        
          vVel = (qMid - delT * ((d1_x[r] + d1_y[r] + d1_z[r]) * vel[r].y() 
                              +  (d4_x[r] + d3_y[r] + d4_z[r]) * rho_tmp[r]))
                                     /   rho_tmp[r];

        //________________________________________________________________
        // Solve momentum conservation law in z-direction, computing vel[r].z()
        
          wVel = (qMid - delT * ((d1_x[r] + d1_y[r] + d1_z[r]) * vel[r].z() 
                              +  (d4_x[r] + d4_y[r] + d3_z[r]) * rho_tmp[r]))
                                     /   rho_tmp[r];
          vel_CC[r] = Vector(uVel, vVel, wVel);
        // end of computing on the corner of i = hi_x,
        // j = hi_y and k = low.z()
       
        //__________________________________________________________
        //  C O R N E R
        // Compute density on the corner of i = hi_x, j = hi_y 
        // and k = low.z(),this needs to compute the chracteristic 
        // waves in x, y and z dircetion.
          
          r   =  IntVector(hi_x,   hi_y,   low.z());
          l   =  IntVector(hi_x-1, hi_y,   low.z());
          b   =  IntVector(hi_x,   hi_y-1, low.z());
          f   =  IntVector(hi_x,   hi_y,   low.z()+1);

        //__________________________________________________________________
        // Solve momentum conservation law in x-direction, computing the vel[r].x()                          
              
          uVel = (qMid - delT * ((d1_x[r] + d1_y[r] + d1_z[r]) * vel[r].x()  
                              +  (d3_x[r] + d4_y[r] + d4_z[r]) * rho_tmp[r]))
                                     /   rho_tmp[r];
        
        //__________________________________________________________________
        // Solve momentum conservation law in y-direction, computing vel[r].y()
        
          vVel = (qMid - delT * ((d1_x[r] + d1_y[r] + d1_z[r]) * vel[r].y() 
                              +  (d4_x[r] + d3_y[r] + d4_z[r]) * rho_tmp[r]))
                                     /   rho_tmp[r];

        //________________________________________________________________
        // Solve momentum conservation law in z-direction, computing vel[r].z()
        
          wVel = (qMid - delT * ((d1_x[r] + d1_y[r] + d1_z[r]) * vel[r].z() 
                              +  (d4_x[r] + d4_y[r] + d3_z[r]) * rho_tmp[r]))
                                     /   rho_tmp[r];
         vel_CC[r] = Vector(uVel, vVel, wVel);
       
       // end of computing density on the corner of i = hi_x,
        // j = hi_y and k = low.z()
        
        //__________________________________________________________
        //  C O R N E R
        // Compute density on the corner of i = hi_x, j = low.y() 
        // and k = low.z(),this needs to compute the chracteristic 
        // waves in x, y and z dircetion.
          
          r   =  IntVector(hi_x,   low.y(),   low.z());
          l   =  IntVector(hi_x-1, low.y(),   low.z());
          t   =  IntVector(hi_x,   low.y()+1, low.z());
          f   =  IntVector(hi_x,   low.y(),   low.z()+1);

        //__________________________________________________________________
        //Solve momentum conservation law in x-direction, computing the vel[r].x()
        uVel = (qMid - delT * ((d1_x[r] + d1_y[r] + d1_z[r]) * vel[r].x()  
                            +  (d3_x[r] + d4_y[r] + d4_z[r]) * rho_tmp[r]))
                                   /   rho_tmp[r];
        
        //__________________________________________________________________
        //Solve momentum conservation law in y-direction, computing vel[r].y()
        vVel = (qMid - delT * ((d1_x[r] + d1_y[r] + d1_z[r]) * vel[r].y() 
                            +  (d4_x[r] + d3_y[r] + d4_z[r]) * rho_tmp[r]))
                                   /   rho_tmp[r];

        //________________________________________________________________
        //Solve momentum conservation law in z-direction, computing vel[r].z()
        wVel = (qMid - delT * ((d1_x[r] + d1_y[r] + d1_z[r]) * vel[r].z() 
                            +  (d4_x[r] + d4_y[r] + d3_z[r]) * rho_tmp[r]))
                                   /   rho_tmp[r];
         vel_CC[r] = Vector(uVel, vVel, wVel);

       }
       break;
        // do nothing
       default:
       break;
    }
} 
  
/* --------------------------------------------------------------------- 
 Function~  fillFaceTempLODI--
 Purpose~   Calculate  temperature using  Characteristic Boundary Conditions
---------------------------------------------------------------------  */
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
                const CCVariable<double>& c,
                const Patch::FaceType face,
                const double delT,
                const double cv,
                const double gamma, 
                const int mat_id,
                const Vector& dx)

{
         IntVector low,hi,r,l,t,b,f,bk;

         low = p.getLowIndex();
         hi  = p.getHighIndex();
         int hi_x = hi.x() - 1;
         int hi_y = hi.y() - 1;
         int hi_z = hi.z() - 1;
         double x_conv = 0.0;
         double y_conv = 0.0;
         double z_conv = 0.0;

         double term1,term2,term3,term4;
         double qMid;
         double k_const = 0.3;
         switch (face) { // switch:4
         case Patch::xplus:
          { //case: 5
         //___________________________________________________________________
         // Compute the density on the area of i = hi_x, low.y() < j < hi_y
         // and low.z() < k < hi_z 
         //for(int j = low.y()+1; j < hi_y; j++) {
         //  for(int k = low.z()+1; k < hi_z; k++) {
          
         for(int j = low.y(); j <= hi_y; j++) {
           for(int k = low.z(); k <= hi_z; k++) {

           r  =  IntVector(hi_x,  j,   k);
           l  =  IntVector(hi_x-1,j,   k);
/*
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
     
           double qConFrt  = vel[t].y() * (e[t] + p[t]);
           double qConLast = vel[b].y() * (e[b] + p[b]);
           doubleqFrt   = e[t];
           
             double qMid   = e[r];
             qLast  = e[b];
             y_conv = computeConvection(k_const, nuy[t], nuy[r], nuy[b], 
                                       qFrt, qMid, qLast, qConFrt, 
                                       qConLast, dx.y());
                                  
          qConFrt  = vel[f].z()  * (e[f] + p[f]);
          qConLast = vel[bk].z() * (e[bk] + p[bk]);
            qFrt   = e[f];
            qLast  = e[bk];

          z_conv = computeConvection(k_const, nuz[f], nuz[r], nuz[bk],
                          qFrt, qMid, qLast, qConFrt,qConLast, dx.z()); 
*/
          double vel_sqr = (vel[r].x() * vel[r].x()
                         +  vel[r].y() * vel[r].y()
                         +  vel[r].z() * vel[r].z());
       
          term1 = 0.5 * d1_x[r] * vel_sqr;
          term2 = d2_x[r]/(gamma - 1.0) + rho_tmp[r] * vel[r].x() * d3_x[r];
          term3 = rho_tmp[r] * vel[r].y() * d4_x[r] + rho_tmp[r] * vel[r].z() * d5_x[r];
          term4 = y_conv + z_conv;
   double e_tmp = e[r] - delT * (term1 + term2 + term3 + term4);
        
     temp_CC[r] = e_tmp/rho_CC[r]/cv - 0.5 * vel_sqr/cv;
   //       cout << "e_tmp, e, term1-4" << r << "=" 
   //            << e_tmp  <<  " " << e[r] << " " <<  term1 << " " << term2 << " " << term3 
   //              << " " <<  term4 << endl;
        
          } // end of j loop
        } //end of k loop
         // end of computing energy on the area of i = hi_x, 
        // low.y() < j < hi_y and low.z() < k < hi_z 
 /*
        //__________________________________________________________
        //  E D G E
        // Compute the edge of i = hi_x, j = low.y() 
        // and low.z() < k < hi_z, this needs to compute the 
        // chracteristic waves in both x and y dircetion.
         for(int k = low.z()+1; k < hi_z; k++) {
          
           r  =  IntVector(hi_x,   low.y(),   k);
           l  =  IntVector(hi_x-1, low.y(),   k);
           t  =  IntVector(hi_x,   low.y()+1, k);
           f  =  IntVector(hi_x,   low.y(),   k+1);
           bk =  IntVector(hi_x,   low.y(),   k-1);
         

        //_______________________________________________________________
        // energy conservation law, computing pressure, or temperature
        // Please remember e is the total energy per unit volume, 
        // not per unit mass, must be given or compute. The ICE code compute 
        // just internal energy, not the total energy. Here the code is written 
        // just for ICE materials
          qConFrt  = vel[f].z()  * (e[f]  + p[f]);
          qConLast = vel[bk].z() * (e[bk] + p[bk]);
            qFrt   = e[f];
            qMid   = e[r];
            qLast  = e[bk];

            z_conv = computeConvection(k_const, nuz[f], nuz[r], nuz[bk],
                            qFrt, qMid, qLast, qConFrt, qConLast, dx.z());
         
         double vel_sqr = (vel[r].x() * vel[r].x()
                        +  vel[r].y() * vel[r].y()
                        +  vel[r].z() * vel[r].z());

          term1 = 0.5 * (d1_x[r] + d1_y[r]) * vel_sqr;
          term2 =  (d2_x[r] + d2_y[r])/(gamma - 1.0) 
              +  rho_tmp[r] * vel[r].x() * (d3_x[r] + d3_y[r]);
          term3 =  rho_tmp[r] * vel[r].y() * (d4_x[r] + d4_y[r]) 
              +  rho_tmp[r] * vel[r].z() * (d5_x[r] + d5_y[r]);

   double e_tmp = e[r] - delT * ( term1 + term2 + term3 + z_conv);
     
     temp_CC[r] = e_tmp/rho_CC[r]/cv - 0.5 * vel_sqr/cv;
        

        } //end of k loop
        // end of computing energy on the edge of i = hi_x, 
       // j = low.y() and low.z() < k < hi_z

        //_______________________________________________________
        //    E D G E
        // Compute the edge of i = hi_x, j = hi_y 
        // and low.z() < k < hi_z, this needs to compute the 
        // chracteristic waves in y dircetion too.
          for(int k = low.z()+1; k < hi_z; k++) {
          
          r  =  IntVector(hi_x,   hi_y,   k);
          l  =  IntVector(hi_x-1, hi_y,   k);
          b  =  IntVector(hi_x,   hi_y-1, k);
          f  =  IntVector(hi_x,   hi_y,   k+1);
          bk =  IntVector(hi_x,   hi_y,   k-1);
       
        //_______________________________________________________________
        // energy conservation law, computing pressure, or temperature
        // Please remember e is the total energy per unit volume, 
        // not per unit mass, must be given or compute. The ICE code compute 
        // just internal energy, not the total energy. Here the code is written 
        // just for ICE materials
          qConFrt  = vel[f].z() * (e[f]  + p[f]);
          qConLast = vel[bk].z() *(e[bk] + p[bk]);
            qFrt   = e[f];
            qMid   = e[r];
            qLast  = e[bk];

            z_conv = computeConvection(k_const, nuz[f], nuz[r], nuz[bk],
                            qFrt, qMid, qLast, qConFrt, qConLast, dx.z());
         
           cout << " qConFrt, qConLast = " << qConFrt << " " << qConLast << endl;
           cout << " qFrt, qMid, qLast = " << qFrt << " " << qMid << qLast << endl;
 
         double vel_sqr = (vel[r].x() * vel[r].x()
                      +  vel[r].y() * vel[r].y()
                      +  vel[r].z() * vel[r].z());

          term1 = 0.5 * (d1_x[r] + d1_y[r]) * vel_sqr;
          term2 =  (d2_x[r] + d2_y[r])/(gamma - 1.0) 
              +  rho_tmp[r] * vel[r].x() * (d3_x[r] + d3_y[r]);
          term3 =  rho_tmp[r] * vel[r].y() * (d4_x[r] + d4_y[r]) 
              +  rho_tmp[r] * vel[r].z() * (d5_x[r] + d5_y[r]);

   double e_tmp = e[r] - delT * ( term1 + term2 + term3 + z_conv);
        
     temp_CC[r] = e_tmp/rho_CC[r]/cv - 0.5 * vel_sqr/cv;

               cout << " term1, term2, term3 = " << term1 << "," << term2 << term3;
                 cout << " e " << r << " = "  << e[r] << endl; 

        } //end of k loop
        // end of computing on the edge of i = hi_x, j = hi_y 
       // and low.z() < k < hi_z
         
         cout << "fillFaceTemp 33333333333333333" << endl;
        //_____________________________________________________
       //    E D G E
        // Compute the edge of i = hi_x, k = low.z() 
        // and low.y() < j < hi_y, this needs to compute the 
        // chracteristic waves in both x and z dircetion.
          for(int j = low.y()+1; j < hi_y; j++) {
          
          r  =  IntVector(hi_x,   j,   low.z());
          l  =  IntVector(hi_x-1, j,   low.z());
          t  =  IntVector(hi_x,   j+1, low.z());
          b  =  IntVector(hi_x,   j-1, low.z());
          f  =  IntVector(hi_x,   j,   low.z()+1);
       
        //_______________________________________________________________
        // energy conservation law, computing pressure, or temperature
        // Please remember e is the total energy per unit volume, 
        // not per unit mass, must be given or compute. The ICE code compute 
        // just internal energy, not the total energy. Here the code is written 
        // just for ICE materials
          qConFrt  = vel[t].z() * (e[t] + p[t]);
          qConLast = vel[b].z() * (e[b] + p[b]);
            qFrt   = e[t];
            qMid   = e[r];
            qLast  = e[b];

            y_conv = computeConvection(k_const, nuy[t], nuy[r], nuy[b],
                            qFrt, qMid, qLast, qConFrt, qConLast, dx.y());
         
         double vel_sqr = (vel[r].x() * vel[r].x()
                      +  vel[r].y() * vel[r].y()
                      +  vel[r].z() * vel[r].z());

          term1 = 0.5 * (d1_x[r] + d1_z[r]) * vel_sqr;
          term2 =  (d2_x[r] + d2_z[r])/(gamma - 1.0) 
              +  rho_tmp[r] * vel[r].x() * (d3_x[r] + d3_z[r]);
          term3 =  rho_tmp[r] * vel[r].y() * (d4_x[r] + d4_z[r]) 
              +  rho_tmp[r] * vel[r].z() * (d5_x[r] + d5_z[r]);

  double  e_tmp = e[r] - delT * ( term1 + term2 + term3 + y_conv);
        
     temp_CC[r] = e_tmp/rho_CC[r]/cv - 0.5 * vel_sqr/cv;

        } //end of j loop
        // end of computing on the edge of i = hi_x, k = low.z() 
        // and low.y() < j < hi_y
         cout << "fillFaceTemp 44444444444444444" << endl;

        //_________________________________________________
       // E D G E
        // Compute the edge of i = hi_x, k = hi_z 
        // and low.y() < j < hi_y, this needs to compute the 
        // chracteristic waves in both  x and z dircetion.
           for(int j = low.y()+1; j < hi_y; j++) {

          r  =  IntVector(hi_x,   j,   hi_z);
          l  =  IntVector(hi_x-1, j,   hi_z);
          t  =  IntVector(hi_x,   j+1, hi_z);
          b  =  IntVector(hi_x,   j-1, hi_z);
          bk =  IntVector(hi_x,   j,   hi_z-1);
       

        //_______________________________________________________________
        // energy conservation law, computing pressure, or temperature
        // Please remember e is the total energy per unit volume, 
        // not per unit mass, must be given or compute. The ICE code compute 
        // just internal energy, not the total energy. Here the code is written 
        // just for ICE materials
          qConFrt  = vel[t].z() * (e[t] + p[t]);
          qConLast = vel[b].z() * (e[b] + p[b]);
            qFrt   = e[t];
            qMid   = e[r];
            qLast  = e[b];

            y_conv = computeConvection(k_const, nuy[t], nuy[r], nuy[b],
                            qFrt, qMid, qLast, qConFrt, qConLast, dx.y());
         
         double vel_sqr = (vel[r].x() * vel[r].x()
                      +  vel[r].y() * vel[r].y()
                      +  vel[r].z() * vel[r].z());

          term1 = 0.5 * (d1_x[r] + d1_z[r]) * vel_sqr;
          term2 =  (d2_x[r] + d2_z[r])/(gamma - 1.0) 
              +  rho_tmp[r] * vel[r].x() * (d3_x[r] + d3_z[r]);
          term3 =  rho_tmp[r] * vel[r].y() * (d4_x[r] + d4_z[r]) 
              +  rho_tmp[r] * vel[r].z() * (d5_x[r] + d5_z[r]);

   double e_tmp = e[r] - delT * ( term1 + term2 + term3 + y_conv);

     temp_CC[r] = e_tmp/rho_CC[r]/cv - 0.5 * vel_sqr/cv;
        
        } //end of j loop
        // end of computing on the edge of i = hi_x, k = hi_z 
        // and low.y() < j < hi_y
         cout << "fillFaceTemp 555555555555555" << endl;
        
        //________________________________________________________
       // C O R N E R
        // Compute the corner of i = hi_x, j = low.y() 
        // and k = hi_z, this needs to compute the chracteristic 
        // waves in x, y and z dircetion.
          r   =  IntVector(hi_x,   low.y(),   hi_z);
          l   =  IntVector(hi_x-1, low.y(),   hi_z);
          t   =  IntVector(hi_x,   low.y()+1, hi_z);
          bk  =  IntVector(hi_x,   low.y(),   hi_z-1);

        //_______________________________________________________________
        // energy conservation law, computing pressure, or temperature
        // Please remember e is the total energy per unit volume, 
        // not per unit mass, must be given or compute. The ICE code compute 
        // just internal energy, not the total energy. Here the code is written 
        // just for ICE materials
         
         vel_sqr = (vel[r].x() * vel[r].x()
                      +  vel[r].y() * vel[r].y()
                      +  vel[r].z() * vel[r].z());

          term1 = 0.5 * (d1_x[r] + d1_y[r] + d1_z[r]) * vel_sqr;
          term2 =       (d2_x[r] + d2_y[r] + d2_z[r])/(gamma - 1.0) 
              +  rho_tmp[r] * vel[r].x() * (d3_x[r] + d3_y[r] + d3_z[r]);
          term3 =  rho_tmp[r] * vel[r].y() * (d4_x[r] + d4_y[r] + d4_z[r]) 
              +  rho_tmp[r] * vel[r].z() * (d5_x[r] + d5_y[r] + d5_z[r]);

   double e_tmp = e[r] - delT * ( term1 + term2 + term3);
        
     temp_CC[r] = e_tmp/rho_CC[r]/cv - 0.5 * vel_sqr/cv;
        
       // end of computing on the corner of i = hi_x,
        // j = low.y() and k = hi_z
         cout << "fillFaceTemp 666666666666666" << endl;

        //_________________________________________________________
       // C O R N E R
        // Compute the corner of i = hi_x, j = hi_y 
        // and k = hi_z,this needs to compute the chracteristic 
        // waves in x, y and z dircetion.
          
          r   =  IntVector(hi_x,   hi_y,    hi_z);
          l   =  IntVector(hi_x-1, hi_y,    hi_z);
          b   =  IntVector(hi_x,   hi_y-1,  hi_z);
          bk  =  IntVector(hi_x,   hi_y,    hi_z-1);

        //_______________________________________________________________
        // energy conservation law, computing pressure, or temperature
        // Please remember e is the total energy per unit volume, 
        // not per unit mass, must be given or compute. The ICE code compute 
        // just internal energy, not the total energy. Here the code is written 
        // just for ICE materials
         
         vel_sqr = (vel[r].x() * vel[r].x()
                      +  vel[r].y() * vel[r].y()
                      +  vel[r].z() * vel[r].z());

          term1 = 0.5 * (d1_x[r] + d1_y[r] + d1_z[r]) * vel_sqr;
          term2 = (d2_x[r] + d2_y[r] + d2_z[r])/(gamma - 1.0) 
              +  rho_tmp[r] * vel[r].x() * (d3_x[r] + d3_y[r] + d3_z[r]);
          term3 =  rho_tmp[r] * vel[r].y() * (d4_x[r] + d4_y[r] + d4_z[r]) 
              +  rho_tmp[r] * vel[r].z() * (d5_x[r] + d5_y[r] + d5_z[r]);

          e_tmp = e[r] - delT * ( term1 + term2 + term3);
        
     temp_CC[r] = e_tmp/rho_CC[r]/cv - 0.5 * vel_sqr/cv;
          
        // end of computing on the corner of i = hi_x,
        // j = hi_y and k = low.z()
       
         cout << "fillFaceTemp 777777777777777" << endl;
        //__________________________________________________________
        //  C O R N E R
        // Compute on the corner of i = hi_x, j = hi_y 
        // and k = low.z(),this needs to compute the chracteristic 
        // waves in x, y and z dircetion.
          
          r   =  IntVector(hi_x,   hi_y,   low.z());
          l   =  IntVector(hi_x-1, hi_y,   low.z());
          b   =  IntVector(hi_x,   hi_y-1, low.z());
          f   =  IntVector(hi_x,   hi_y,   low.z()+1);

        //_______________________________________________________________
        // energy conservation law, computing pressure, or temperature
        // Please remember e is the total energy per unit volume, 
        // not per unit mass, must be given or compute. The ICE code compute 
        // just internal energy, not the total energy. Here the code is written 
        // just for ICE materials
         
         vel_sqr = (vel[r].x() * vel[r].x()
                +  vel[r].y() * vel[r].y()
                +  vel[r].z() * vel[r].z());

          term1 = 0.5 * (d1_x[r] + d1_y[r] + d1_z[r]) * vel_sqr;
          term2 = (d2_x[r] + d2_y[r] + d2_z[r])/(gamma - 1.0) 
              +  rho_tmp[r] * vel[r].x() * (d3_x[r] + d3_y[r] + d3_z[r]);
          term3 =  rho_tmp[r] * vel[r].y() * (d4_x[r] + d4_y[r] + d4_z[r]) 
              +  rho_tmp[r] * vel[r].z() * (d5_x[r] + d5_y[r] + d5_z[r]);

          e_tmp = e[r] - delT * ( term1 + term2 + term3);
        
          temp_CC[r] = e_tmp/rho_CC[r]/cv - 0.5 * vel_sqr/cv;

       // end of computing on the corner of i = hi_x,
        // j = hi_y and k = low.z()
        
         cout << "fillFaceTemp 88888888888888888" << endl;
        //__________________________________________________________
        //  C O R N E R
        // Compute on the corner of i = hi_x, j = low.y() 
        // and k = low.z(),this needs to compute the chracteristic 
        // waves in x, y and z dircetion.
          
          r   =  IntVector(hi_x,   low.y(),   low.z());
          l   =  IntVector(hi_x-1, low.y(),   low.z());
          t   =  IntVector(hi_x,   low.y()+1, low.z());
          f   =  IntVector(hi_x,   low.y(),   low.z()+1);
          
        //_______________________________________________________________
        // energy conservation law, computing pressure, or temperature
        // Please remember e is the total energy per unit volume, 
        // not per unit mass, must be given or compute. The ICE code compute 
        // just internal energy, not the total energy. Here the code is written 
        // just for ICE materials
         
         vel_sqr = (vel[r].x() * vel[r].x()
                      +  vel[r].y() * vel[r].y()
                      +  vel[r].z() * vel[r].z());

          term1 = 0.5 * (d1_x[r] + d1_y[r] + d1_z[r]) * vel_sqr;
          term2 =       (d2_x[r] + d2_y[r] + d2_z[r])/(gamma - 1.0) 
              +  rho_tmp[r] * vel[r].x() * (d3_x[r] + d3_y[r] + d3_z[r]);
          term3 =  rho_tmp[r] * vel[r].y() * (d4_x[r] + d4_y[r] + d4_z[r]) 
              +  rho_tmp[r] * vel[r].z() * (d5_x[r] + d5_y[r] + d5_z[r]);

         e_tmp = e[r] - delT * ( term1 + term2 + term3);
        
     temp_CC[r] = e_tmp/rho_CC[r]/cv - 0.5 * vel_sqr/cv;

       // end of computing density on the corner of i = hi_x,
       // j = low.y() and k = low.z()
         cout << "fillFaceTemp 999999999999999" << endl;
   */     
       } // end of case x_plus: 5
       break;
        //here will insert the other 5 faces: xminus, yplus, yminus, zplus, zminus
       default:
       break;
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
              
          }  // for ALLMatls...
        } // k loop
      }  //j loop
      break;
    case Patch::xminus:
        // do nothing
      break;
    case Patch::yplus:
        // do nothing
      break;
    case Patch::yminus:
        // do nothing
      break;
    case Patch::zplus:
        // do nothing
      break;
    case Patch::zminus:
        // do nothing
      break;
    case Patch::numFaces:
      break;
    case Patch::invalidFace:
      break;
    }
    
  }
  
 #endif   // LODI_BCS 
}  // using namespace Uintah


