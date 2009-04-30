#include <CCA/Components/Arches/TransportEqns/Discretization_new.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Exceptions/InvalidValue.h>

using namespace std;
using namespace Uintah;

Discretization_new::Discretization_new()
{
}

Discretization_new::~Discretization_new()
{
}

//---------------------------------------------------------------------------
// Method: Compute the convection term
//---------------------------------------------------------------------------
template <class fT, class oldPhiT> void 
Discretization_new::computeConv(const Patch* p, fT& Fconv, oldPhiT& oldPhi, 
            constSFCXVariable<double>& uVel, constSFCYVariable<double>& vVel, 
            constSFCZVariable<double>& wVel, constCCVariable<double>& den,
            std::string convScheme ) 
{
  // This class computes convection without assumptions about the boundary conditions 
  // (ie, it doens't adjust values of fluxes on the boundaries but assume you have 
  // done so previously)
  Vector Dx = p->dCell(); 
  FaceData<double> F; // FACE value of phi
  IntVector cLow  = p->getCellLowIndex__New(); 
  IntVector cHigh = p->getCellHighIndex__New();  

  if (convScheme == "upwind") {

    // ------ UPWIND ------
    for (CellIterator iter=p->getCellIterator__New(); !iter.done(); iter++){

      IntVector c = *iter;
      IntVector cxp = *iter + IntVector(1,0,0);
      IntVector cxm = *iter - IntVector(1,0,0);
      IntVector cyp = *iter + IntVector(0,1,0);
      IntVector cym = *iter - IntVector(0,1,0);
      IntVector czp = *iter + IntVector(0,0,1);
      IntVector czm = *iter - IntVector(0,0,1);

      double xmVel = uVel[c];
      double xpVel = uVel[cxp];

      double xmDen = ( den[c] + den[cxm] ) / 2.0;
      double xpDen = ( den[c] + den[cxp] ) / 2.0;

      if ( xmVel > 0.0 )
        F.w = oldPhi[cxm];
      else if (xmVel < 0.0 )
        F.w = oldPhi[c]; 
      else 
        F.w = 0.0;

      if ( xpVel > 0.0 )
        F.e = oldPhi[c];
      else if ( xpVel < 0.0 )
        F.e = oldPhi[cxp];
      else 
        F.e = 0.0;  

      Fconv[c] = Dx.y()*Dx.z()*( F.e * xpDen * xpVel - F.w * xmDen * xmVel );

#ifdef YDIM
      double ymVel = vVel[c];
      double ypVel = vVel[cyp];

      double ymDen = ( den[c] + den[cym] ) / 2.0;
      double ypDen = ( den[c] + den[cyp] ) / 2.0;

      if ( ymVel > 0.0 )
        F.s = oldPhi[cym];
      else if ( ymVel < 0.0 )
        F.s = oldPhi[c]; 
      else
        F.s = 0.0;  


      if ( ypVel > 0.0 )
        F.n = oldPhi[c];
      else if ( ypVel < 0.0 )
        F.n = oldPhi[cyp];
      else  
        F.n = 0.0; 

      Fconv[c] += Dx.x()*Dx.z()*( F.n * ypDen * ypVel - F.s * ymDen * ymVel ); 
#endif
#ifdef ZDIM
      double zmVel = wVel[c];
      double zpVel = wVel[czp];

      double zmDen = ( den[c] + den[czm] ) / 2.0;
      double zpDen = ( den[c] + den[czp] ) / 2.0;

      if ( zmVel > 0.0 )
        F.b = oldPhi[czm];
      else if ( zmVel < 0.0 )
        F.b = oldPhi[c]; 
      else 
        F.b = 0.0;   

      if ( zpVel > 0.0 )
        F.t = oldPhi[c];
      else if ( zpVel < 0.0 )
        F.t = oldPhi[czp];
      else 
        F.t = 0.0;  

      Fconv[c] += Dx.x()*Dx.y()*( F.t * zpDen * zpVel - F.b * zmDen * zmVel ); 
#endif 
    }
  } else if (convScheme == "super_bee") { 

    // ------ SUPERBEE ------
    bool xminus = p->getBCType(Patch::xminus) != Patch::Neighbor;
    bool xplus =  p->getBCType(Patch::xplus) != Patch::Neighbor;
    bool yminus = p->getBCType(Patch::yminus) != Patch::Neighbor;
    bool yplus =  p->getBCType(Patch::yplus) != Patch::Neighbor;
    bool zminus = p->getBCType(Patch::zminus) != Patch::Neighbor;
    bool zplus =  p->getBCType(Patch::zplus) != Patch::Neighbor;

    IntVector clow  = p->getCellLowIndex__New();
    IntVector chigh = p->getCellHighIndex__New(); 
    IntVector clow_mod;
    IntVector chigh_mod; 

    if (xminus)
      clow_mod = clow + IntVector(1,0,0);
    if (xplus)
      chigh_mod = chigh - IntVector(1,0,0); 
    if (yminus)
      clow_mod = clow + IntVector(0,1,0);
    if (yplus)
      chigh_mod = chigh - IntVector(0,1,0);
    if (zminus)
      clow_mod = clow + IntVector(0,0,1);
    if (zplus)
      chigh_mod = chigh - IntVector(0,0,1);

    for (int i = clow_mod.x(); i < chigh_mod.x(); i++){
      for (int j = clow_mod.y(); j < chigh_mod.y(); j++){
        for (int k = clow_mod.z(); k < chigh_mod.z(); k++){

          IntVector c(i,j,k);
          IntVector cxp = c + IntVector(1,0,0);
          IntVector cxpp= c + IntVector(2,0,0);
          IntVector cxm = c - IntVector(1,0,0);
          IntVector cxmm= c - IntVector(2,0,0);
          IntVector cyp = c + IntVector(0,1,0);
          IntVector cypp= c + IntVector(0,2,0);
          IntVector cym = c - IntVector(0,1,0);
          IntVector cymm= c - IntVector(0,2,0);
          IntVector czp = c + IntVector(0,0,1);
          IntVector czpp= c + IntVector(0,0,2);
          IntVector czm = c - IntVector(0,0,1);
          IntVector czmm= c - IntVector(0,0,2);

          double r; 
          double psi; 
          double Sup;
          double Sdn;

          double xmVel = uVel[c];
          double xpVel = uVel[cxp];

          double xmDen = ( den[c] + den[cxm] ) / 2.0;
          double xpDen = ( den[c] + den[cxp] ) / 2.0;

          // EAST
          if ( xpVel > 0.0 ) {
            r = ( oldPhi[c] - oldPhi[cxm] ) / ( oldPhi[cxp] - oldPhi[c] );
            Sup = oldPhi[c];
            Sdn = oldPhi[cxp];
          } else if ( xpVel < 0.0 ) {
            r = ( oldPhi[cxpp] - oldPhi[cxp] ) / ( oldPhi[cxp] - oldPhi[c] );
            Sup = oldPhi[cxp];
            Sdn = oldPhi[c]; 
          } else {
            Sup = 0.0;
            Sdn = 0.0; 
            psi = 0.0;
          }
          psi = max( min(2.0*r, 1.0), min(r, 2.0) );
          psi = max( 0.0, psi );

          F.e = Sup + 0.5*psi*( Sdn - Sup ); 

          // WEST
          if ( xmVel > 0.0 ) {
            Sup = oldPhi[cxm];
            Sdn = oldPhi[c];
            r = ( oldPhi[cxm] - oldPhi[cxmm] ) / ( oldPhi[c] - oldPhi[cxm] ); 
          } else if ( xmVel < 0.0 ) {
            Sup = oldPhi[c];
            Sdn = oldPhi[cxm];
            r = ( oldPhi[cxp] - oldPhi[c] ) / ( oldPhi[c] - oldPhi[cxm] );
          } else { 
            Sup = 0.0;
            Sdn = 0.0; 
            psi = 0.0;
          }
          psi = max( min(2.0*r, 1.0), min(r, 2.0) );
          psi = max( 0.0, psi );
      
          F.w = Sup + 0.5*psi*( Sdn - Sup ); 

          Fconv[c] = Dx.y()*Dx.z()*( F.e * xpDen * xpVel - F.w * xmDen * xmVel );
#ifdef YDIM
          double ymVel = vVel[c];
          double ypVel = vVel[cyp];

          double ymDen = ( den[c] + den[cym] ) / 2.0;
          double ypDen = ( den[c] + den[cyp] ) / 2.0;

          // NORTH
          if ( ypVel > 0.0 ) {
            r = ( oldPhi[c] - oldPhi[cym] ) / ( oldPhi[cyp] - oldPhi[c] );
            Sup = oldPhi[c];
            Sdn = oldPhi[cyp];
          } else if ( ypVel < 0.0 ) {
            r = ( oldPhi[cypp] - oldPhi[cyp] ) / ( oldPhi[cyp] - oldPhi[c] );
            Sup = oldPhi[cyp];
            Sdn = oldPhi[c]; 
          } else {
            Sup = 0.0;
            Sdn = 0.0; 
            psi = 0.0;
          } 
          psi = max( min(2.0*r, 1.0), min(r, 2.0) );
          psi = max( 0.0, psi );

          F.n = Sup + 0.5*psi*( Sdn - Sup ); 
          // SOUTH
              
          if ( ymVel > 0.0 ) {
            Sup = oldPhi[cym];
            Sdn = oldPhi[c];
            r = ( oldPhi[cym] - oldPhi[cymm] ) / ( oldPhi[c] - oldPhi[cym] ); 
          } else if ( ymVel > 0.0 ) {
            Sup = oldPhi[c];
            Sdn = oldPhi[cym];
            r = ( oldPhi[cyp] - oldPhi[c] ) / ( oldPhi[c] - oldPhi[cym] ); 
          } else {
            Sup = 0.0;
            Sdn = 0.0; 
            psi = 0.0;
          } 
          psi = max( min(2.0*r, 1.0), min(r, 2.0) );
          psi = max( 0.0, psi );

          F.s = Sup + 0.5*psi*( Sdn - Sup ); 

          Fconv[c] += Dx.x()*Dx.z()*( F.n * ypDen * ypVel - F.s * ymDen * ymVel ); 
#endif          
#ifdef ZDIM
          double zmVel = wVel[c];
          double zpVel = wVel[czp];

          double zmDen = ( den[c] + den[czm] ) / 2.0;
          double zpDen = ( den[c] + den[czp] ) / 2.0;

          // TOP
          if ( zpVel > 0.0 ) {
            r = ( oldPhi[c] - oldPhi[czm] ) / ( oldPhi[czp] - oldPhi[c] );
            Sup = oldPhi[c];
            Sdn = oldPhi[czp];
          } else if ( zpVel < 0.0 ) {
            r = ( oldPhi[czpp] - oldPhi[czp] ) / ( oldPhi[czp] - oldPhi[c] );
            Sup = oldPhi[czp];
            Sdn = oldPhi[c]; 
          } else { 
            Sup = 0.0;
            Sdn = 0.0; 
            psi = 0.0;
          }

          psi = max( min(2.0*r, 1.0), min(r, 2.0) );
          psi = max( 0.0, psi );

          F.t = Sup + 0.5*psi*( Sdn - Sup ); 

          // BOTTOM 
          if ( zmVel > 0.0 ) {
            Sup = oldPhi[czm];
            Sdn = oldPhi[c];
            r = ( oldPhi[czm] - oldPhi[czmm] ) / ( oldPhi[c] - oldPhi[czm] ); 
          } else if ( zmVel > 0.0 ) {
            Sup = oldPhi[c];
            Sdn = oldPhi[czm];
            r = ( oldPhi[czp] - oldPhi[c] ) / ( oldPhi[c] - oldPhi[czm] );
          } else {  
            Sup = 0.0;
            Sdn = 0.0; 
            psi = 0.0;
          }
          psi = max( min(2.0*r, 1.0), min(r, 2.0) );
          psi = max( 0.0, psi );

          F.b = Sup + 0.5*psi*( Sdn - Sup ); 

          Fconv[c] += Dx.x()*Dx.y()*( F.t * zpDen * zpVel - F.b * zmDen * zmVel ); 
#endif          
        }
      }
    }

    // ----- BOUNDARIES -----
    // For patches with boundaries, do upwind along cells that abut the boundary 
    if (xplus) {
      double r; 
      double psi; 
      double Sup;
      double Sdn;
      // EAST BOUNDARY
      int i = chigh.x();
      for ( int j = clow.y(); j <= chigh.y(); j++ ){
        for ( int k = clow.z(); k <= chigh.z(); k++ ){

          IntVector c(i,j,k);
          IntVector cxp = c + IntVector(1,0,0);
          IntVector cxm = c - IntVector(1,0,0);
          IntVector cxmm= c - IntVector(2,0,0);

          double xmVel = uVel[c];
          double xpVel = uVel[cxp];

          double xmDen = ( den[c] + den[cxm] ) / 2.0;
          double xpDen = ( den[c] + den[cxp] ) / 2.0;

          if ( xmVel > 0.0 ) {
            Sup = oldPhi[cxm];
            Sdn = oldPhi[c];
            r = ( oldPhi[cxm] - oldPhi[cxmm] ) / ( oldPhi[c] - oldPhi[cxm] ); 
          } else if ( xmVel < 0.0 ) {
            Sup = oldPhi[c];
            Sdn = oldPhi[cxm];
            r = ( oldPhi[cxp] - oldPhi[c] ) / ( oldPhi[c] - oldPhi[cxm] );
          } else { 
            Sup = 0.0;
            Sdn = 0.0; 
            psi = 0.0;
          }
          psi = max( min(2.0*r, 1.0), min(r, 2.0) );
          psi = max( 0.0, psi );
      
          F.w = Sup + 0.5*psi*( Sdn - Sup ); 

          if ( xpVel > 0.0 )
            F.e = oldPhi[c];
          else if ( xpVel < 0.0 )
            F.e = oldPhi[cxp];
          else 
            F.e = 0.0;  

          Fconv[c] = Dx.y()*Dx.z()*( F.e * xpDen * xpVel - F.w * xmDen * xmVel );

        }
      }
    }
    if (xminus) {
      double r; 
      double psi; 
      double Sup;
      double Sdn;

      // WEST BOUNDARY
      int i = clow.x();
      for ( int j = clow.y(); j <= chigh.y(); j++ ){
        for ( int k = clow.z(); k <= chigh.z(); k++ ){

          IntVector c(i,j,k);
          IntVector cxp = c + IntVector(1,0,0);
          IntVector cxpp= c + IntVector(2,0,0);
          IntVector cxm = c - IntVector(1,0,0);

          double xmVel = uVel[c];
          double xpVel = uVel[cxp];

          double xmDen = ( den[c] + den[cxm] ) / 2.0;
          double xpDen = ( den[c] + den[cxp] ) / 2.0;

          if ( xmVel > 0.0 )
            F.w = oldPhi[cxm];
          else if (xmVel < 0.0 )
            F.w = oldPhi[c]; 
          else 
            F.w = 0.0;

          if ( xpVel > 0.0 ) {
            r = ( oldPhi[c] - oldPhi[cxm] ) / ( oldPhi[cxp] - oldPhi[c] );
            Sup = oldPhi[c];
            Sdn = oldPhi[cxp];
          } else if ( xpVel < 0.0 ) {
            r = ( oldPhi[cxpp] - oldPhi[cxp] ) / ( oldPhi[cxp] - oldPhi[c] );
            Sup = oldPhi[cxp];
            Sdn = oldPhi[c]; 
          } else {
            Sup = 0.0;
            Sdn = 0.0; 
            psi = 0.0;
          }
          psi = max( min(2.0*r, 1.0), min(r, 2.0) );
          psi = max( 0.0, psi );

          F.e = Sup + 0.5*psi*( Sdn - Sup ); 

          Fconv[c] = Dx.y()*Dx.z()*( F.e * xpDen * xpVel - F.w * xmDen * xmVel );

        }
      }
    }
#ifdef YDIM    
    if (yplus) {
      double r; 
      double psi; 
      double Sup;
      double Sdn;

      // NORTH BOUNDARY
      int j = chigh.y();
      for ( int i = clow.x(); i < chigh.x(); i++ ) {
        for ( int k = chigh.z(); k < chigh.z(); k++ ) {

          IntVector c(i,j,k);
          IntVector cyp = c + IntVector(0,1,0);
          IntVector cym = c - IntVector(0,1,0);
          IntVector cymm= c - IntVector(0,2,0);

          double ymVel = vVel[c];
          double ypVel = vVel[cyp];

          double ymDen = ( den[c] + den[cym] ) / 2.0;
          double ypDen = ( den[c] + den[cyp] ) / 2.0;

          if ( ymVel > 0.0 ) {
            Sup = oldPhi[cym];
            Sdn = oldPhi[c];
            r = ( oldPhi[cym] - oldPhi[cymm] ) / ( oldPhi[c] - oldPhi[cym] ); 
          } else if ( ymVel > 0.0 ) {
            Sup = oldPhi[c];
            Sdn = oldPhi[cym];
            r = ( oldPhi[cyp] - oldPhi[c] ) / ( oldPhi[c] - oldPhi[cym] ); 
          } else {
            Sup = 0.0;
            Sdn = 0.0; 
            psi = 0.0;
          } 
          psi = max( min(2.0*r, 1.0), min(r, 2.0) );
          psi = max( 0.0, psi );

          F.s = Sup + 0.5*psi*( Sdn - Sup ); 

          if ( ypVel > 0.0 )
            F.n = oldPhi[c];
          else if ( ypVel < 0.0 )
            F.n = oldPhi[cyp];
          else  
            F.n = 0.0; 

          Fconv[c] += Dx.x()*Dx.z()*( F.n * ypDen * ypVel - F.s * ymDen * ymVel ); 

        }
      }
    }
    if (yminus) {
      double r; 
      double psi; 
      double Sup;
      double Sdn;

      // SOUTH BOUNDARY
      int j = clow.y();
      for ( int i = clow.x(); i < chigh.x(); i++ ) {
        for ( int k = chigh.z(); k < chigh.z(); k++ ) {

          IntVector c(i,j,k);
          IntVector cyp = c + IntVector(0,1,0);
          IntVector cypp= c + IntVector(0,2,0);
          IntVector cym = c - IntVector(0,1,0);

          double ymVel = vVel[c];
          double ypVel = vVel[cyp];

          double ymDen = ( den[c] + den[cym] ) / 2.0;
          double ypDen = ( den[c] + den[cyp] ) / 2.0;

          if ( ymVel > 0.0 )
            F.s = oldPhi[cym];
          else if ( ymVel < 0.0 )
            F.s = oldPhi[c]; 
          else
            F.s = 0.0;  

          if ( ypVel > 0.0 ) {
            r = ( oldPhi[c] - oldPhi[cym] ) / ( oldPhi[cyp] - oldPhi[c] );
            Sup = oldPhi[c];
            Sdn = oldPhi[cyp];
          } else if ( ypVel < 0.0 ) {
            r = ( oldPhi[cypp] - oldPhi[cyp] ) / ( oldPhi[cyp] - oldPhi[c] );
            Sup = oldPhi[cyp];
            Sdn = oldPhi[c]; 
          } else {
            Sup = 0.0;
            Sdn = 0.0; 
            psi = 0.0;
          } 
          psi = max( min(2.0*r, 1.0), min(r, 2.0) );
          psi = max( 0.0, psi );

          F.n = Sup + 0.5*psi*( Sdn - Sup ); 
          Fconv[c] += Dx.x()*Dx.z()*( F.n * ypDen * ypVel - F.s * ymDen * ymVel ); 

        }
      }
    }
#endif
#ifdef ZDIM
    if (zplus) {
      double r; 
      double psi; 
      double Sup;
      double Sdn;

      // TOP BOUNDARY
      int k = chigh.z(); 
      for ( int i = clow.x(); i < chigh.x(); i++ ) {
        for ( int j = clow.y(); j < chigh.y(); j++ ) {

          IntVector c(i,j,k);
          IntVector czp = c + IntVector(0,0,1);
          IntVector czm = c - IntVector(0,0,1);
          IntVector czmm= c - IntVector(0,0,2);

          double zmVel = wVel[c];
          double zpVel = wVel[czp];

          double zmDen = ( den[c] + den[czm] ) / 2.0;
          double zpDen = ( den[c] + den[czp] ) / 2.0;

          if ( zpVel > 0.0 )
            F.t = oldPhi[c];
          else if ( zpVel < 0.0 )
            F.t = oldPhi[czp];
          else 
            F.t = 0.0;  

          if ( zmVel > 0.0 ) {
            Sup = oldPhi[czm];
            Sdn = oldPhi[c];
            r = ( oldPhi[czm] - oldPhi[czmm] ) / ( oldPhi[c] - oldPhi[czm] ); 
          } else if ( zmVel > 0.0 ) {
            Sup = oldPhi[c];
            Sdn = oldPhi[czm];
            r = ( oldPhi[czp] - oldPhi[c] ) / ( oldPhi[c] - oldPhi[czm] );
          } else {  
            Sup = 0.0;
            Sdn = 0.0; 
            psi = 0.0;
          }
          psi = max( min(2.0*r, 1.0), min(r, 2.0) );
          psi = max( 0.0, psi );

          F.b = Sup + 0.5*psi*( Sdn - Sup ); 

          Fconv[c] += Dx.x()*Dx.y()*( F.t * zpVel * zpVel - F.b * zmVel * zmVel ); 


        }
      }
    }
    if (zminus) {
      double r; 
      double psi; 
      double Sup;
      double Sdn;

      // BOTTOM BOUNDARY
      int k = clow.z();
      for ( int i = clow.x(); i < chigh.x(); i++ ) {
        for ( int j = clow.y(); j < chigh.y(); j++ ) {

          IntVector c(i,j,k);
          IntVector czp = c + IntVector(0,0,1);
          IntVector czpp= c + IntVector(0,0,2);
          IntVector czm = c - IntVector(0,0,1);

          double zmVel = wVel[c];
          double zpVel = wVel[czp];

          double zmDen = ( den[c] + den[czm] ) / 2.0;
          double zpDen = ( den[c] + den[czp] ) / 2.0;

          if ( zmVel > 0.0 )
            F.b = oldPhi[czm];
          else if ( zmVel < 0.0 )
            F.b = oldPhi[c]; 
          else 
            F.b = 0.0;   

          if ( zpVel > 0.0 ) {
            r = ( oldPhi[c] - oldPhi[czm] ) / ( oldPhi[czp] - oldPhi[c] );
            Sup = oldPhi[c];
            Sdn = oldPhi[czp];
          } else if ( zpVel < 0.0 ) {
            r = ( oldPhi[czpp] - oldPhi[czp] ) / ( oldPhi[czp] - oldPhi[c] );
            Sup = oldPhi[czp];
            Sdn = oldPhi[c]; 
          } else { 
            Sup = 0.0;
            Sdn = 0.0; 
            psi = 0.0;
          }

          psi = max( min(2.0*r, 1.0), min(r, 2.0) );
          psi = max( 0.0, psi );

          F.t = Sup + 0.5*psi*( Sdn - Sup ); 

          Fconv[c] += Dx.x()*Dx.y()*( F.t * zpDen * zpVel - F.b * zpDen * zmVel ); 
        }
      }
    }

#endif    
  } else {

    cout << "Convection scheme not supported! " << endl;

  }
}

