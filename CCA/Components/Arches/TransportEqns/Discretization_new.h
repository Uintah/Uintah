#ifndef Uintah_Component_Arches_Discretization_new_h
#define Uintah_Component_Arches_Discretization_new_h

#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Exceptions/InvalidValue.h>
#include <CCA/Components/Arches/Directives.h>

//==========================================================================

/**
 * @class Discretization_new
 * @author Jeremy Thornock
 * @date Oct 16, 2008
 *
 * @brief A discretization toolbox.
 *       
 *
 *
 */

namespace Uintah{
  class Discretization_new {

    public:

      Discretization_new();
      ~Discretization_new();

      /** @brief Computes the convection term.  This method is overloaded.  */
      template <class fT, class oldPhiT, class uT, class vT, class wT> void 
        computeConv(const Patch* p, fT& Fconv, oldPhiT& oldPhi, 
            uT& uVel, vT& vVel, 
            wT& wVel, constCCVariable<double>& den, constCCVariable<Vector>& areaFraction, 
            std::string convScheme);

      /** @brief Computes the convection term (no density term) */
      template <class fT, class oldPhiT> void 
        computeConv(const Patch* p, fT& Fconv, oldPhiT& oldPhi, 
            constSFCXVariable<double>& uVel, constSFCYVariable<double>& vVel, 
            constSFCZVariable<double>& wVel, constCCVariable<Vector>& areaFraction, 
            std::string convScheme);

      /** @brief Computes the convection term without density and using a special 
       *         particle velocity vector (or any type of CC velocity vector) instead 
       *         of the standard, face centered gas velocity.  */
      template <class fT, class oldPhiT> void 
        computeConv(const Patch* p, fT& Fconv, oldPhiT& oldPhi, 
            constSFCXVariable<double>& uVel, constSFCYVariable<double>& vVel, 
            constSFCZVariable<double>& wVel, constCCVariable<Vector>& partVel, 
            constCCVariable<Vector>& areaFraction, 
            std::string convScheme);

      inline CellIterator getInteriorCellIterator( const Patch* p ) const 
      {

        bool xminus = p->getBCType(Patch::xminus) == Patch::None;
        bool xplus =  p->getBCType(Patch::xplus)  == Patch::None;
        bool yminus = p->getBCType(Patch::yminus) == Patch::None;
        bool yplus =  p->getBCType(Patch::yplus)  == Patch::None;
        bool zminus = p->getBCType(Patch::zminus) == Patch::None;
        bool zplus =  p->getBCType(Patch::zplus)  == Patch::None;

        IntVector clow  = p->getCellLowIndex();
        IntVector chigh = p->getCellHighIndex(); 
        IntVector clow_mod = clow;
        IntVector chigh_mod = chigh; 

        if (xminus)
          clow_mod = clow_mod + IntVector(1,0,0);
        if (xplus)
          chigh_mod = chigh_mod - IntVector(1,0,0); 
        if (yminus)
          clow_mod = clow_mod + IntVector(0,1,0);
        if (yplus)
          chigh_mod = chigh_mod - IntVector(0,1,0);
        if (zminus)
          clow_mod = clow_mod + IntVector(0,0,1);
        if (zplus)
          chigh_mod = chigh_mod - IntVector(0,0,1);

        CellIterator the_iterator = CellIterator(clow_mod, chigh_mod); 

        the_iterator.reset(); 
        return the_iterator; 
      }

      inline CellIterator getInteriorBoundaryCellIterator( const Patch* p, const vector<Patch::FaceType>::const_iterator bf_iter ) const 
      {

        Patch::FaceType face = *bf_iter; 
        IntVector l,h; 
        p->getFaceCells( face, 0, l, h ); 

        //dont want edge cells:
        if ( face == Patch::xminus || face == Patch::xplus ){

          bool yminus = p->getBCType(Patch::yminus) == Patch::None;
          bool yplus =  p->getBCType(Patch::yplus)  == Patch::None;
          bool zminus = p->getBCType(Patch::zminus) == Patch::None;
          bool zplus =  p->getBCType(Patch::zplus)  == Patch::None;

          if (yminus)
            l[1] += 2;
          if (yplus)
            h[1] -= 2;
          if (zminus)
            l[2] += 2;
          if (zplus)
            h[2] -= 2;

        } else if ( face == Patch::yminus || face == Patch::yplus ){

          bool xminus = p->getBCType(Patch::xminus) == Patch::None;
          bool xplus =  p->getBCType(Patch::xplus)  == Patch::None;
          bool zminus = p->getBCType(Patch::zminus) == Patch::None;
          bool zplus =  p->getBCType(Patch::zplus)  == Patch::None;

          if (xminus)
            l[0] += 2;
          if (xplus)
            h[0] -= 2;
          if (zminus)
            l[2] += 2;
          if (zplus)
            h[2] -= 2;
        } else if ( face == Patch::zminus || face == Patch::zplus ){

          bool yminus = p->getBCType(Patch::yminus) == Patch::None;
          bool yplus =  p->getBCType(Patch::yplus)  == Patch::None;
          bool xminus = p->getBCType(Patch::xminus) == Patch::None;
          bool xplus =  p->getBCType(Patch::xplus)  == Patch::None;

          if (yminus)
            l[1] += 2;
          if (yplus)
            h[1] -= 2;
          if (xminus)
            l[0] += 2;
          if (xplus)
            h[0] -= 2;
        }

        CellIterator the_iterator = CellIterator( l, h ); 
        return the_iterator; 

      }

      struct FaceBoundaryBool
      { 
        bool minus;
        bool plus; 
      };

      inline FaceBoundaryBool checkFacesForBoundaries( const Patch* p, const IntVector c, const IntVector coord )
      {

        FaceBoundaryBool b;
        b.minus = false; 
        b.plus  = false; 

        IntVector l = p->getCellLowIndex();
        IntVector h = p->getCellHighIndex(); 

        if ( coord[0] == 1 ) {

          if ( l[0] == c[0] ) b.minus = true;
          if ( h[0] == c[0] ) b.plus  = true; 

        } else if ( coord[1] == 1 ) {

          if ( l[1] == c[1] ) b.minus = true;
          if ( h[1] == c[1] ) b.plus = true; 

        } else if ( coord[2] == 1 ) {

          if ( l[2] == c[2] ) b.minus = true; 
          if ( h[2] == c[2] ) b.plus = true; 

        }

        return b; 
      }

      struct FaceData1D {
        double minus; // minus face
        double plus;  // plus face
      };

      inline double getFlux( const double area, FaceData1D den, FaceData1D vel, FaceData1D phi, constCCVariable<Vector> areaFraction, IntVector coord, IntVector c )
      {
        double F; 
        FaceData1D areaFrac;
        IntVector cp = c + coord; 
        Vector curr_areaFrac = areaFraction[c]; 
        Vector plus_areaFrac = areaFraction[cp]; 
        // may want to just pass dim in for efficiency sake
        int dim = 0; 
        if (coord[0] == 1)
          dim =0; 
        else if (coord[1] == 1)
          dim = 1; 
        else 
          dim = 2; 
        areaFrac.plus  = plus_areaFrac[dim];
        areaFrac.minus = curr_areaFrac[dim]; 

        return F = area * (  areaFrac.plus * den.plus * vel.plus * phi.plus 
                           - areaFrac.minus * den.minus * vel.minus * phi.minus ); 
      }
      inline double getFlux( const double area, FaceData1D vel, FaceData1D phi, constCCVariable<Vector> areaFraction, IntVector coord, IntVector c )
      {
        double F; 
        FaceData1D areaFrac;
        IntVector cp = c + coord; 
        Vector curr_areaFrac = areaFraction[c]; 
        Vector plus_areaFrac = areaFraction[cp]; 
        // may want to just pass dim in for efficiency sake
        int dim = 0; 
        if (coord[0] == 1)
          dim =0; 
        else if (coord[1] == 1)
          dim = 1; 
        else 
          dim = 2; 
        areaFrac.plus  = plus_areaFrac[dim];
        areaFrac.minus = curr_areaFrac[dim]; 

        return F = area * (  areaFrac.plus * vel.plus * phi.plus 
                           - areaFrac.minus * vel.minus * phi.minus ); 
      }

      inline FaceData1D getFaceVelocity( const IntVector c, const CCVariable<double>& F, constCCVariable<Vector> vel, const IntVector coord ){
      
        FaceData1D the_vel;
        the_vel.minus = 0.0;
        the_vel.plus  = 0.0; 

        int coord_sum = coord[0] + coord[1] + coord[2]; 

        if (coord[0] == 1) {

          IntVector cxm = c - IntVector(1,0,0);
          IntVector cxp = c + IntVector(1,0,0); 

          the_vel.minus = 0.5 * ( vel[c].x() + vel[cxm].x() ); 
          the_vel.plus  = 0.5 * ( vel[c].x() + vel[cxp].x() ); 

        } else if (coord[1] == 1) {

          IntVector cym = c - IntVector(0,1,0);
          IntVector cyp = c + IntVector(0,1,0); 

          the_vel.minus = 0.5 * ( vel[c].y() + vel[cym].y() ); 
          the_vel.plus  = 0.5 * ( vel[c].y() + vel[cyp].y() ); 

        } else if (coord[2] == 1) {

          IntVector czm = c - IntVector(0,0,1);
          IntVector czp = c + IntVector(0,0,1); 

          the_vel.minus = 0.5 * ( vel[c].z() + vel[czm].z() ); 
          the_vel.plus  = 0.5 * ( vel[c].z() + vel[czp].z() ); 

        } else if (coord[0] == 0 && coord[1] == 0 && coord[2] == 0) {

          // no coordinate specified
          throw InternalError("ERROR! No coordinate specified for getFaceVelocity", __FILE__, __LINE__);

        } else if (coord_sum > 1) {

          // too many coordinates specified
          throw InternalError("ERROR! Too many coordinates specified for getFaceVelocity", __FILE__, __LINE__);

        }

        return the_vel; 
      }

      inline FaceData1D getFaceVelocity( const IntVector c, const CCVariable<double>& F, constSFCXVariable<double> vel ){
        // cell-centered, x-direction
        FaceData1D the_vel; 
        the_vel.minus = vel[c];
        the_vel.plus  = vel[c + IntVector(1,0,0)];

        return the_vel; 
      }
      inline FaceData1D getFaceVelocity( const IntVector c, const CCVariable<double>& F, constSFCYVariable<double> vel ){
        // cell-centered, y-direction
        FaceData1D the_vel; 
        the_vel.minus = vel[c];
        the_vel.plus  = vel[c + IntVector(0,1,0)];

        return the_vel; 
      }
      inline FaceData1D getFaceVelocity( const IntVector c, const CCVariable<double>& F, constSFCZVariable<double> vel ){
        // cell-centered, z-direction
        FaceData1D the_vel; 
        the_vel.minus = vel[c];
        the_vel.plus  = vel[c + IntVector(0,0,1)];

        return the_vel; 
      }

      template< class phiT > 
        inline FaceData1D getDensityOnFace( const IntVector c, const IntVector coord, phiT& phi, constCCVariable<double>& den ){

          FaceData1D face_values; 
          face_values.minus = 0.0;
          face_values.plus  = 0.0;

          TypeDescription::Type type = phi.getTypeDescription()->getType(); 

          if ( type == TypeDescription::CCVariable ) {
            IntVector cxm = c - coord; 
            IntVector cxp = c + coord; 

            face_values.minus = 0.5 * (den[c] + den[cxm]); 
            face_values.plus  = 0.5 * (den[c] + den[cxp]); 
          } else {
            // assume the only other type is a face type...
            IntVector cxm = c - coord; 

            face_values.minus = den[cxm];
            face_values.plus  = den[c]; 
          }

          return face_values; 
        }

      template< class phiT >
        inline FaceData1D centralInterp( const IntVector c, const IntVector coord, phiT& phi )
        {
          IntVector cxp = c + coord; 
          IntVector cxm = c - coord; 

          FaceData1D face_values; 

          face_values.minus = 0.5 * ( phi[c] + phi[cxm] ); 
          face_values.plus  = 0.5 * ( phi[c] + phi[cxp] ); 

          return face_values; 

        }

      template< class phiT >
        inline FaceData1D superBeeInterp( const IntVector c, const IntVector coord, phiT& phi, 
            FaceData1D vel ) {

          FaceData1D face_values;
          face_values.plus  = 0.0;
          face_values.minus = 0.0;

          double r=0.; 
          double psi; 
          double Sup;
          double Sdn;

          IntVector cxp  = c + coord; 
          IntVector cxpp = c + coord + coord; 
          IntVector cxm  = c - coord; 
          IntVector cxmm = c - coord - coord; 

          // - FACE
          if ( vel.minus > 0.0 ) {
            Sup = phi[cxm];
            Sdn = phi[c];
            r = ( phi[cxm] - phi[cxmm] ) / ( phi[c] - phi[cxm] ); 
          } else if ( vel.minus < 0.0 ) {
            Sup = phi[c];
            Sdn = phi[cxm];
            r = ( phi[cxp] - phi[c] ) / ( phi[c] - phi[cxm] );
          } else { 
            Sup = 0.0;
            Sdn = 0.0; 
            psi = 0.0;
          }
          psi = max( min(2.0*r, 1.0), min(r, 2.0) );
          psi = max( 0.0, psi );

          face_values.minus = Sup + 0.5*psi*( Sdn - Sup );

          // + FACE
          if ( vel.plus > 0.0 ) {
            r = ( phi[c] - phi[cxm] ) / ( phi[cxp] - phi[c] );
            Sup = phi[c];
            Sdn = phi[cxp];
          } else if ( vel.plus < 0.0 ) {
            r = ( phi[cxpp] - phi[cxp] ) / ( phi[cxp] - phi[c] );
            Sup = phi[cxp];
            Sdn = phi[c]; 
          } else {
            Sup = 0.0;
            Sdn = 0.0; 
            psi = 0.0;
          }
          psi = max( min(2.0*r, 1.0), min(r, 2.0) );
          psi = max( 0.0, psi );

          face_values.plus = Sup + 0.5*psi*( Sdn - Sup );

          return face_values; 
        }
      template< class phiT >
        inline FaceData1D superBeeInterp( const IntVector c, const IntVector coord, phiT& phi, 
            FaceData1D vel, FaceBoundaryBool isBoundary) {

          FaceData1D face_values;
          face_values.plus  = 0.0;
          face_values.minus = 0.0;

          double r; 
          double psi; 
          double Sup;
          double Sdn;

          IntVector cxp  = c + coord; 
          IntVector cxpp = c + coord + coord; 
          IntVector cxm  = c - coord; 
          IntVector cxmm = c - coord - coord; 

          // - FACE
          if (isBoundary.minus) 
            face_values.minus = 0.5*(phi[c]+phi[cxm]);
          else { 
            if ( vel.minus > 0.0 ) {
              Sup = phi[cxm];
              Sdn = phi[c];
              r = ( phi[cxm] - phi[cxmm] ) / ( phi[c] - phi[cxm] ); 
            } else if ( vel.minus < 0.0 ) {
              Sup = phi[c];
              Sdn = phi[cxm];
              r = ( phi[cxp] - phi[c] ) / ( phi[c] - phi[cxm] );
            } else { 
              Sup = 0.0;
              Sdn = 0.0; 
              psi = 0.0;
            }
            psi = max( min(2.0*r, 1.0), min(r, 2.0) );
            psi = max( 0.0, psi );

            face_values.minus = Sup + 0.5*psi*( Sdn - Sup );
          }

          // + FACE
          if (isBoundary.plus)
            face_values.plus = 0.5*(phi[c] + phi[cxp]);
          else { 
            if ( vel.plus > 0.0 ) {
              r = ( phi[c] - phi[cxm] ) / ( phi[cxp] - phi[c] );
              Sup = phi[c];
              Sdn = phi[cxp];
            } else if ( vel.plus < 0.0 ) {
              r = ( phi[cxpp] - phi[cxp] ) / ( phi[cxp] - phi[c] );
              Sup = phi[cxp];
              Sdn = phi[c]; 
            } else {
              Sup = 0.0;
              Sdn = 0.0; 
              psi = 0.0;
            }
            psi = max( min(2.0*r, 1.0), min(r, 2.0) );
            psi = max( 0.0, psi );

            face_values.plus = Sup + 0.5*psi*( Sdn - Sup );
          }

          return face_values; 
        }

      template< class phiT >
        inline FaceData1D upwindInterp( const IntVector c, const IntVector coord, phiT& phi, 
            FaceData1D vel) {

          Discretization_new::FaceData1D face_values; 
          face_values.minus = 0.0;
          face_values.plus = 0.0;

          IntVector cxp = c + coord; 
          IntVector cxm = c - coord; 

          // - FACE 
          if ( vel.minus > 0.0 )
            face_values.minus = phi[cxm];
          else if ( vel.minus <= 0.0 )
            face_values.minus = phi[c]; 

          // + FACE 
          if ( vel.plus >= 0.0 )
            face_values.plus = phi[c]; 
          else if ( vel.plus < 0.0 )
            face_values.plus = phi[cxp]; 

          return face_values; 

        }
      template< class phiT >
        inline FaceData1D upwindInterp( const IntVector c, const IntVector coord, phiT& phi, 
            FaceData1D vel, FaceBoundaryBool isBoundary ) {

          Discretization_new::FaceData1D face_values; 
          face_values.minus = 0.0;
          face_values.plus = 0.0;

          IntVector cxp = c + coord; 
          IntVector cxm = c - coord; 

          // - FACE 
          if (isBoundary.minus)
            face_values.minus = 0.5*(phi[c] + phi[cxm]);
          else {
            if ( vel.minus > 0.0 )
              face_values.minus = phi[cxm];
            else if ( vel.minus <= 0.0 )
              face_values.minus = phi[c]; 
          }

          // + FACE 
          if (isBoundary.plus)
            face_values.plus = 0.5*(phi[c] + phi[cxp]);
          else {
            if ( vel.plus >= 0.0 )
              face_values.plus = phi[c]; 
            else if ( vel.plus < 0.0 )
              face_values.plus = phi[cxp]; 
          }

          return face_values; 

        }


  }; // class Discretization_new

  template<class T> 
    struct FaceData {
      // 0 = e, 1=w, 2=n, 3=s, 4=t, 5=b
      //vector<T> values_[6];
      T p; 
      T e; 
      T w; 
      T n; 
      T s;
      T t;
      T b;
    };

  struct FaceData1D {
    double minus; // minus face
    double plus;  // plus face
  };

  //---------------------------------------------------------------------------
  // Method: Compute the convection term (with Density) 
  //---------------------------------------------------------------------------
  template <class fT, class oldPhiT, class uT, class vT, class wT> void 
    Discretization_new::computeConv(const Patch* p, fT& Fconv, oldPhiT& oldPhi, 
        uT& uVel, vT& vVel, 
        wT& wVel, constCCVariable<double>& den, constCCVariable<Vector>& areaFraction, 
        std::string convScheme ) 
    {
      // This class computes convection without assumptions about the boundary conditions 
      // (ie, it doesn't adjust values of fluxes on the boundaries but assume you have 
      // done so previously or that you will go back and repair them)
      Vector Dx = p->dCell(); 

      // get the cell interior iterator
      CellIterator iIter  = Discretization_new::getInteriorCellIterator( p ); 

      if (convScheme == "upwind") {

        for (iIter.begin(); !iIter.done(); iIter++){

          IntVector c   = *iIter;
          IntVector cxp = c + IntVector(1,0,0);
          IntVector cxm = c - IntVector(1,0,0);
          IntVector cyp = c + IntVector(0,1,0);
          IntVector cym = c - IntVector(0,1,0);
          IntVector czp = c + IntVector(0,0,1);
          IntVector czm = c - IntVector(0,0,1);

          FaceData1D face_den;
          FaceData1D face_phi; 
          FaceData1D face_vel;
          double area; 
          IntVector coord; 

          //X-dimension
          coord[0] = 1; coord[1] = 0; coord[2] = 0; 
          area = Dx.y()*Dx.z();

          face_den       = getDensityOnFace( c, coord, Fconv, den ); 
          face_vel       = getFaceVelocity( c, Fconv, uVel ); 
          face_phi       = upwindInterp( c, coord, oldPhi, face_vel ); 
          Fconv[c]       = getFlux( area, face_den, face_vel, face_phi, areaFraction, coord, c ); 

#ifdef YDIM
          //Y-dimension
          coord[0] = 0; coord[1] = 1; coord[2] = 0; 
          area = Dx.x()*Dx.z(); 

          face_den       = getDensityOnFace( c, coord, Fconv, den ); 
          face_vel       = getFaceVelocity( c, Fconv, vVel ); 
          face_phi       = upwindInterp( c, coord, oldPhi, face_vel ); 
          Fconv[c]      += getFlux( area, face_den, face_vel, face_phi, areaFraction, coord, c ); 

#endif 
#ifdef ZDIM
          //Z-dimension
          coord[0] = 0; coord[1] = 0; coord[2] = 1; 
          area = Dx.x()*Dx.y(); 

          face_den       = getDensityOnFace( c, coord, Fconv, den ); 
          face_vel       = getFaceVelocity( c, Fconv, wVel ); 
          face_phi       = upwindInterp( c, coord, oldPhi, face_vel ); 
          Fconv[c]      += getFlux( area, face_den, face_vel, face_phi, areaFraction, coord, c ); 

#endif
        }
        // Boundaries
        vector<Patch::FaceType> bf;
        vector<Patch::FaceType>::const_iterator bf_iter;
        p->getBoundaryFaces(bf);

        // Loop over all boundary faces on this patch
        for (bf_iter = bf.begin(); bf_iter != bf.end(); bf_iter++){

          Patch::FaceType face = *bf_iter; 
          IntVector inside = p->faceDirection(face); 
          CellIterator c_iter = getInteriorBoundaryCellIterator( p, bf_iter ); 
          FaceBoundaryBool faceIsBoundary; 

          for (c_iter.begin(); !c_iter.done(); c_iter++){

            IntVector c = *c_iter - inside; 
            IntVector cxp = c + IntVector(1,0,0);
            IntVector cxm = c - IntVector(1,0,0); 
            IntVector cyp = c + IntVector(0,1,0); 
            IntVector cym = c - IntVector(0,1,0);
            IntVector czp = c + IntVector(0,0,1);
            IntVector czm = c - IntVector(0,0,1);

            FaceData1D face_den;
            FaceData1D face_phi; 
            FaceData1D face_vel;
            double area; 
            IntVector coord; 

            //X-dimension
            coord[0] = 1; coord[1] = 0; coord[2] = 0; 
            area = Dx.y()*Dx.z();            

            faceIsBoundary = checkFacesForBoundaries( p, c, coord ); 
            face_den       = getDensityOnFace( c, coord, Fconv, den ); 
            face_vel       = getFaceVelocity( c, Fconv, uVel ); 
            face_phi       = upwindInterp( c, coord, oldPhi, face_vel, faceIsBoundary ); 
            Fconv[c]       = getFlux( area, face_den, face_vel, face_phi, areaFraction, coord, c ); 

#ifdef YDIM
            //Y-dimension
            coord[0] = 0; coord[1] = 1; coord[2] = 0; 
            area = Dx.x()*Dx.z(); 

            faceIsBoundary = checkFacesForBoundaries( p, c, coord ); 
            face_den       = getDensityOnFace( c, coord, Fconv, den ); 
            face_vel       = getFaceVelocity( c, Fconv, vVel ); 
            face_phi       = upwindInterp( c, coord, oldPhi, face_vel, faceIsBoundary ); 
            Fconv[c]      += getFlux( area, face_den, face_vel, face_phi, areaFraction, coord, c ); 

#endif 
#ifdef ZDIM
            //Z-dimension
            coord[0] = 0; coord[1] = 0; coord[2] = 1; 
            area = Dx.x()*Dx.y(); 

            faceIsBoundary = checkFacesForBoundaries( p, c, coord ); 
            face_den       = getDensityOnFace( c, coord, Fconv, den ); 
            face_vel       = getFaceVelocity( c, Fconv, wVel ); 
            face_phi       = upwindInterp( c, coord, oldPhi, face_vel, faceIsBoundary ); 
            Fconv[c]      += getFlux( area, face_den, face_vel, face_phi, areaFraction, coord, c ); 

#endif
          }
        }

      } else if (convScheme == "super_bee") { 

        for (iIter.begin(); !iIter.done(); iIter++){

          IntVector c   = *iIter;
          IntVector cxp = c + IntVector(1,0,0);
          IntVector cxm = c - IntVector(1,0,0);
          IntVector cyp = c + IntVector(0,1,0);
          IntVector cym = c - IntVector(0,1,0);
          IntVector czp = c + IntVector(0,0,1);
          IntVector czm = c - IntVector(0,0,1);

          FaceData1D face_den;
          FaceData1D face_phi; 
          FaceData1D face_vel;
          double area; 
          IntVector coord; 

          //X-dimension
          coord[0] = 1; coord[1] = 0; coord[2] = 0; 
          area = Dx.y()*Dx.z(); 

          face_den       = getDensityOnFace( c, coord, Fconv, den ); 
          face_vel       = getFaceVelocity( c, Fconv, uVel ); 
          face_phi       = superBeeInterp( c, coord, oldPhi, face_vel ); 
          Fconv[c]       = getFlux( area, face_den, face_vel, face_phi, areaFraction, coord, c ); 

#ifdef YDIM
          //Y-dimension
          coord[0] = 0; coord[1] = 1; coord[2] = 0; 
          area = Dx.x()*Dx.z(); 

          face_den       = getDensityOnFace( c, coord, Fconv, den ); 
          face_vel       = getFaceVelocity( c, Fconv, vVel ); 
          face_phi       = superBeeInterp( c, coord, oldPhi, face_vel ); 
          Fconv[c]      += getFlux( area, face_den, face_vel, face_phi, areaFraction, coord, c ); 
#endif 
#ifdef ZDIM
          //Z-dimension
          coord[0] = 0; coord[1] = 0; coord[2] = 1; 
          area = Dx.x()*Dx.y(); 

          face_den       = getDensityOnFace( c, coord, Fconv, den ); 
          face_vel       = getFaceVelocity( c, Fconv, wVel ); 
          face_phi       = superBeeInterp( c, coord, oldPhi, face_vel ); 
          Fconv[c]      += getFlux( area, face_den, face_vel, face_phi, areaFraction, coord, c ); 
#endif
        }
        // Boundaries
        vector<Patch::FaceType> bf;
        vector<Patch::FaceType>::const_iterator bf_iter;
        p->getBoundaryFaces(bf);

        // Loop over all boundary faces on this patch
        for (bf_iter = bf.begin(); bf_iter != bf.end(); bf_iter++){

          Patch::FaceType face = *bf_iter; 
          IntVector inside = p->faceDirection(face); 
          CellIterator c_iter = getInteriorBoundaryCellIterator( p, bf_iter ); 
          FaceBoundaryBool faceIsBoundary; 

          for (c_iter.begin(); !c_iter.done(); c_iter++){

            IntVector c = *c_iter - inside; 
            IntVector cxp = c + IntVector(1,0,0);
            IntVector cxm = c - IntVector(1,0,0); 
            IntVector cyp = c + IntVector(0,1,0); 
            IntVector cym = c - IntVector(0,1,0);
            IntVector czp = c + IntVector(0,0,1);
            IntVector czm = c - IntVector(0,0,1);

            FaceData1D face_den;
            FaceData1D face_phi; 
            FaceData1D face_vel;
            double area; 
            IntVector coord; 

            //X-dimension
            coord[0] = 1; coord[1] = 0; coord[2] = 0; 
            area = Dx.y()*Dx.z(); 

            faceIsBoundary = checkFacesForBoundaries( p, c, coord ); 
            face_den       = getDensityOnFace( c, coord, Fconv, den ); 
            face_vel       = getFaceVelocity( c, Fconv, uVel ); 
            face_phi       = superBeeInterp( c, coord, oldPhi, face_vel, faceIsBoundary ); 
            Fconv[c]       = getFlux( area, face_den, face_vel, face_phi, areaFraction, coord, c ); 

#ifdef YDIM
            //Y-dimension
            coord[0] = 0; coord[1] = 1; coord[2] = 0; 
            area = Dx.x()*Dx.z(); 

            faceIsBoundary = checkFacesForBoundaries( p, c, coord ); 
            face_den       = getDensityOnFace( c, coord, Fconv, den ); 
            face_vel       = getFaceVelocity( c, Fconv, vVel ); 
            face_phi       = superBeeInterp( c, coord, oldPhi, face_vel, faceIsBoundary ); 
            Fconv[c]      += getFlux( area, face_den, face_vel, face_phi, areaFraction, coord, c ); 

#endif 
#ifdef ZDIM
            //Z-dimension
            coord[0] = 0; coord[1] = 0; coord[2] = 1; 
            area = Dx.x()*Dx.y(); 

            faceIsBoundary = checkFacesForBoundaries( p, c, coord ); 
            face_den       = getDensityOnFace( c, coord, Fconv, den ); 
            face_vel       = getFaceVelocity( c, Fconv, wVel ); 
            face_phi       = superBeeInterp( c, coord, oldPhi, face_vel, faceIsBoundary ); 
            Fconv[c]      += getFlux( area, face_den, face_vel, face_phi, areaFraction, coord, c ); 

#endif
          }
        }

      } else if (convScheme == "central") {

        for (CellIterator iter=p->getCellIterator(); !iter.done(); iter++){

          IntVector c = *iter; 
          IntVector cxp = c + IntVector(1,0,0);
          IntVector cxm = c - IntVector(1,0,0);
          IntVector cyp = c + IntVector(0,1,0);
          IntVector cym = c - IntVector(0,1,0);
          IntVector czp = c + IntVector(0,0,1);
          IntVector czm = c - IntVector(0,0,1);

          IntVector coord; 

          FaceData1D face_den;
          FaceData1D face_phi; 
          FaceData1D face_vel; 
          double area; 

          //X-FACES
          coord[0] = 1; coord[1] = 0; coord[2] = 0; 
          area = Dx.y()*Dx.z();

          face_den = getDensityOnFace( c, coord, Fconv, den ); 
          face_vel = getFaceVelocity( c, Fconv, uVel ); 
          face_phi = centralInterp( c, coord, oldPhi ); 

          Fconv[c] = getFlux( area, face_den, face_vel, face_phi, areaFraction, coord, c );
#ifdef YDIM
          //Y-FACES
          coord[0] = 0; coord[1] = 1; coord[2] = 0; 
          area = Dx.x()*Dx.z();

          face_den  = getDensityOnFace( c, coord, Fconv, den ); 
          face_vel  = getFaceVelocity( c, Fconv, vVel ); 
          face_phi  = centralInterp( c, coord, oldPhi ); 

          Fconv[c] += getFlux( area, face_den, face_vel, face_phi, areaFraction, coord, c );
#endif

#ifdef ZDIM
          //Z-FACES
          coord[0] = 0; coord[1] = 0; coord[2] = 1; 
          area = Dx.x()*Dx.y();

          face_den = getDensityOnFace( c, coord, Fconv, den ); 
          face_vel = getFaceVelocity( c, Fconv, wVel ); 
          face_phi = centralInterp( c, coord, oldPhi ); 

          Fconv[c] += getFlux( area, face_den, face_vel, face_phi, areaFraction, coord, c );
#endif
        }
      }
    }

  //---------------------------------------------------------------------------
  // Method: Compute the convection term (no density)
  //---------------------------------------------------------------------------
  template <class fT, class oldPhiT> void 
    Discretization_new::computeConv( const Patch* p, fT& Fconv, oldPhiT& oldPhi, 
        constSFCXVariable<double>& uVel, constSFCYVariable<double>& vVel, 
        constSFCZVariable<double>& wVel, constCCVariable<Vector>& areaFraction, 
        std::string convScheme ) 
    {
      // This class computes convection without assumptions about the boundary conditions 
      // (ie, it doens't adjust values of fluxes on the boundaries but assume you have 
      // done so previously)
      Vector Dx = p->dCell(); 

      // get the cell interior iterator
      CellIterator iIter  = Discretization_new::getInteriorCellIterator( p ); 

      if (convScheme == "upwind") {

        for (iIter.begin(); !iIter.done(); iIter++){

          IntVector c   = *iIter;
          IntVector cxp = c + IntVector(1,0,0);
          IntVector cxm = c - IntVector(1,0,0);
          IntVector cyp = c + IntVector(0,1,0);
          IntVector cym = c - IntVector(0,1,0);
          IntVector czp = c + IntVector(0,0,1);
          IntVector czm = c - IntVector(0,0,1);

          FaceData1D face_phi; 
          FaceData1D face_vel;
          double area; 
          IntVector coord; 

          //X-dimension
          coord[0] = 1; coord[1] = 0; coord[2] = 0; 
          area = Dx.y()*Dx.z(); 

          face_vel       = getFaceVelocity( c, Fconv, uVel ); 
          face_phi       = upwindInterp( c, coord, oldPhi, face_vel ); 
          Fconv[c]       = getFlux( area, face_vel, face_phi, areaFraction, coord, c ); 

#ifdef YDIM
          //Y-dimension
          coord[0] = 0; coord[1] = 1; coord[2] = 0; 
          area = Dx.x()*Dx.z(); 

          face_vel       = getFaceVelocity( c, Fconv, vVel ); 
          face_phi       = upwindInterp( c, coord, oldPhi, face_vel ); 
          Fconv[c]      += getFlux( area, face_vel, face_phi, areaFraction, coord, c ); 

#endif 
#ifdef ZDIM
          //Z-dimension
          coord[0] = 0; coord[1] = 0; coord[2] = 1; 
          area = Dx.x()*Dx.y(); 

          face_vel       = getFaceVelocity( c, Fconv, wVel ); 
          face_phi       = upwindInterp( c, coord, oldPhi, face_vel ); 
          Fconv[c]      += getFlux( area, face_vel, face_phi, areaFraction, coord, c ); 

#endif
        }
        // Boundaries
        vector<Patch::FaceType> bf;
        vector<Patch::FaceType>::const_iterator bf_iter;
        p->getBoundaryFaces(bf);

        // Loop over all boundary faces on this patch
        for (bf_iter = bf.begin(); bf_iter != bf.end(); bf_iter++){

          Patch::FaceType face = *bf_iter; 
          IntVector inside = p->faceDirection(face); 
          CellIterator c_iter = getInteriorBoundaryCellIterator( p, bf_iter ); 
          FaceBoundaryBool faceIsBoundary; 

          for (c_iter.begin(); !c_iter.done(); c_iter++){

            IntVector c = *c_iter - inside; 
            IntVector cxp = c + IntVector(1,0,0);
            IntVector cxm = c - IntVector(1,0,0); 
            IntVector cyp = c + IntVector(0,1,0); 
            IntVector cym = c - IntVector(0,1,0);
            IntVector czp = c + IntVector(0,0,1);
            IntVector czm = c - IntVector(0,0,1);

            FaceData1D face_phi; 
            FaceData1D face_vel;
            double area; 
            IntVector coord; 

            //X-dimension
            coord[0] = 1; coord[1] = 0; coord[2] = 0; 
            area = Dx.y()*Dx.z(); 

            faceIsBoundary = checkFacesForBoundaries( p, c, coord ); 
            face_vel       = getFaceVelocity( c, Fconv, uVel ); 
            face_phi       = upwindInterp( c, coord, oldPhi, face_vel, faceIsBoundary ); 
            Fconv[c]       = getFlux( area, face_vel, face_phi, areaFraction, coord, c ); 

#ifdef YDIM
            //Y-dimension
            coord[0] = 0; coord[1] = 1; coord[2] = 0; 
            area = Dx.x()*Dx.z(); 

            faceIsBoundary = checkFacesForBoundaries( p, c, coord ); 
            face_vel       = getFaceVelocity( c, Fconv, vVel ); 
            face_phi       = upwindInterp( c, coord, oldPhi, face_vel, faceIsBoundary ); 
            Fconv[c]      += getFlux( area, face_vel, face_phi, areaFraction, coord, c ); 

#endif 
#ifdef ZDIM
            //Z-dimension
            coord[0] = 0; coord[1] = 0; coord[2] = 1; 
            area = Dx.x()*Dx.y(); 

            faceIsBoundary = checkFacesForBoundaries( p, c, coord ); 
            face_vel       = getFaceVelocity( c, Fconv, wVel ); 
            face_phi       = upwindInterp( c, coord, oldPhi, face_vel, faceIsBoundary ); 
            Fconv[c]      += getFlux( area, face_vel, face_phi, areaFraction, coord, c ); 

#endif
          }
        }

      } else if (convScheme == "super_bee") { 

        for (iIter.begin(); !iIter.done(); iIter++){

          IntVector c   = *iIter;
          IntVector cxp = c + IntVector(1,0,0);
          IntVector cxm = c - IntVector(1,0,0);
          IntVector cyp = c + IntVector(0,1,0);
          IntVector cym = c - IntVector(0,1,0);
          IntVector czp = c + IntVector(0,0,1);
          IntVector czm = c - IntVector(0,0,1);

          FaceData1D face_phi; 
          FaceData1D face_vel;
          double area; 
          IntVector coord; 

          //X-dimension
          coord[0] = 1; coord[1] = 0; coord[2] = 0; 
          area = Dx.y()*Dx.z(); 

          face_vel       = getFaceVelocity( c, Fconv, uVel ); 
          face_phi       = superBeeInterp( c, coord, oldPhi, face_vel ); 
          Fconv[c]       = getFlux( area, face_vel, face_phi, areaFraction, coord, c ); 

#ifdef YDIM
          //Y-dimension
          coord[0] = 0; coord[1] = 1; coord[2] = 0; 
          area = Dx.x()*Dx.z(); 

          face_vel       = getFaceVelocity( c, Fconv, vVel ); 
          face_phi       = superBeeInterp( c, coord, oldPhi, face_vel ); 
          Fconv[c]      += getFlux( area, face_vel, face_phi, areaFraction, coord, c ); 
#endif 
#ifdef ZDIM
          //Z-dimension
          coord[0] = 0; coord[1] = 0; coord[2] = 1; 
          area = Dx.x()*Dx.y(); 

          face_vel       = getFaceVelocity( c, Fconv, wVel ); 
          face_phi       = superBeeInterp( c, coord, oldPhi, face_vel ); 
          Fconv[c]      += getFlux( area, face_vel, face_phi, areaFraction, coord, c  ); 
#endif
        }
        // Boundaries
        vector<Patch::FaceType> bf;
        vector<Patch::FaceType>::const_iterator bf_iter;
        p->getBoundaryFaces(bf);

        // Loop over all boundary faces on this patch
        for (bf_iter = bf.begin(); bf_iter != bf.end(); bf_iter++){

          Patch::FaceType face = *bf_iter; 
          IntVector inside = p->faceDirection(face); 
          CellIterator c_iter = getInteriorBoundaryCellIterator( p, bf_iter ); 
          FaceBoundaryBool faceIsBoundary; 

          for (c_iter.begin(); !c_iter.done(); c_iter++){

            IntVector c = *c_iter - inside; 
            IntVector cxp = c + IntVector(1,0,0);
            IntVector cxm = c - IntVector(1,0,0); 
            IntVector cyp = c + IntVector(0,1,0); 
            IntVector cym = c - IntVector(0,1,0);
            IntVector czp = c + IntVector(0,0,1);
            IntVector czm = c - IntVector(0,0,1);

            FaceData1D face_phi; 
            FaceData1D face_vel;
            double area; 
            IntVector coord; 

            //X-dimension
            coord[0] = 1; coord[1] = 0; coord[2] = 0; 
            area = Dx.y()*Dx.z(); 

            faceIsBoundary = checkFacesForBoundaries( p, c, coord ); 
            face_vel       = getFaceVelocity( c, Fconv, uVel ); 
            face_phi       = superBeeInterp( c, coord, oldPhi, face_vel, faceIsBoundary ); 
            Fconv[c]       = getFlux( area, face_vel, face_phi, areaFraction, coord, c ); 

#ifdef YDIM
            //Y-dimension
            coord[0] = 0; coord[1] = 1; coord[2] = 0; 
            area = Dx.x()*Dx.z(); 

            faceIsBoundary = checkFacesForBoundaries( p, c, coord ); 
            face_vel       = getFaceVelocity( c, Fconv, vVel ); 
            face_phi       = superBeeInterp( c, coord, oldPhi, face_vel, faceIsBoundary ); 
            Fconv[c]      += getFlux( area, face_vel, face_phi, areaFraction, coord, c ); 

#endif 
#ifdef ZDIM
            //Z-dimension
            coord[0] = 0; coord[1] = 0; coord[2] = 1; 
            area = Dx.x()*Dx.y(); 

            faceIsBoundary = checkFacesForBoundaries( p, c, coord ); 
            face_vel       = getFaceVelocity( c, Fconv, wVel ); 
            face_phi       = superBeeInterp( c, coord, oldPhi, face_vel, faceIsBoundary ); 
            Fconv[c]      += getFlux( area, face_vel, face_phi, areaFraction, coord, c  ); 

#endif
          }
        }

      } else if (convScheme == "central") {

        for (CellIterator iter=p->getCellIterator(); !iter.done(); iter++){

          IntVector c = *iter; 
          IntVector cxp = c + IntVector(1,0,0);
          IntVector cxm = c - IntVector(1,0,0);
          IntVector cyp = c + IntVector(0,1,0);
          IntVector cym = c - IntVector(0,1,0);
          IntVector czp = c + IntVector(0,0,1);
          IntVector czm = c - IntVector(0,0,1);

          IntVector coord; 

          FaceData1D face_phi; 
          FaceData1D face_vel; 
          FaceData1D face_af; 
          double area; 

          //X-FACES
          coord[0] = 1; coord[1] = 0; coord[2] = 0; 
          area = Dx.y()*Dx.z();

          face_vel = getFaceVelocity( c, Fconv, uVel ); 
          face_phi = centralInterp( c, coord, oldPhi ); 

          Fconv[c] = getFlux( area, face_vel, face_phi, areaFraction, coord, c );
#ifdef YDIM
          //Y-FACES
          coord[0] = 0; coord[1] = 1; coord[2] = 0; 
          area = Dx.x()*Dx.z();

          face_vel  = getFaceVelocity( c, Fconv, vVel ); 
          face_phi  = centralInterp( c, coord, oldPhi ); 

          Fconv[c] += getFlux( area, face_vel, face_phi, areaFraction, coord, c );
#endif

#ifdef ZDIM
          //Z-FACES
          coord[0] = 0; coord[1] = 0; coord[2] = 1; 
          area = Dx.x()*Dx.y();

          face_vel = getFaceVelocity( c, Fconv, wVel ); 
          face_phi = centralInterp( c, coord, oldPhi ); 

          Fconv[c] += getFlux( area, face_vel, face_phi, areaFraction, coord, c );
#endif
        }
      }
    }

  //---------------------------------------------------------------------------
  // Method: Compute the convection term (no density)
  // Specialized for DQMOM
  //---------------------------------------------------------------------------
  template <class fT, class oldPhiT> void 
    Discretization_new::computeConv( const Patch* p, fT& Fconv, oldPhiT& oldPhi, 
        constSFCXVariable<double>& uVel, constSFCYVariable<double>& vVel, 
        constSFCZVariable<double>& wVel, constCCVariable<Vector>& partVel, 
        constCCVariable<Vector>& areaFraction, 
        std::string convScheme ) 
    {
      // This class computes convection without assumptions about the boundary conditions 
      // (ie, it doens't adjust values of fluxes on the boundaries but assume you have 
      // done so previously)
      Vector Dx = p->dCell(); 

      // get the cell interior iterator
      CellIterator iIter  = Discretization_new::getInteriorCellIterator( p ); 

      if (convScheme == "upwind") {

        for (iIter.begin(); !iIter.done(); iIter++){

          IntVector c   = *iIter;
          IntVector cxp = c + IntVector(1,0,0);
          IntVector cxm = c - IntVector(1,0,0);
          IntVector cyp = c + IntVector(0,1,0);
          IntVector cym = c - IntVector(0,1,0);
          IntVector czp = c + IntVector(0,0,1);
          IntVector czm = c - IntVector(0,0,1);

          FaceData1D face_phi; 
          FaceData1D face_vel;
          double area; 
          IntVector coord; 

          //X-dimension
          coord[0] = 1; coord[1] = 0; coord[2] = 0; 
          area = Dx.y()*Dx.z(); 

          face_vel       = getFaceVelocity( c, Fconv, partVel, coord ); 
          face_phi       = upwindInterp( c, coord, oldPhi, face_vel ); 
          Fconv[c]       = getFlux( area, face_vel, face_phi, areaFraction, coord, c ); 

#ifdef YDIM
          //Y-dimension
          coord[0] = 0; coord[1] = 1; coord[2] = 0; 
          area = Dx.x()*Dx.z(); 

          face_vel       = getFaceVelocity( c, Fconv, partVel, coord ); 
          face_phi       = upwindInterp( c, coord, oldPhi, face_vel ); 
          Fconv[c]      += getFlux( area, face_vel, face_phi, areaFraction, coord, c ); 

#endif 
#ifdef ZDIM
          //Z-dimension
          coord[0] = 0; coord[1] = 0; coord[2] = 1; 
          area = Dx.x()*Dx.y(); 

          face_vel       = getFaceVelocity( c, Fconv, partVel, coord ); 
          face_phi       = upwindInterp( c, coord, oldPhi, face_vel ); 
          Fconv[c]      += getFlux( area, face_vel, face_phi, areaFraction, coord, c ); 

#endif
        }
        // Boundaries
        vector<Patch::FaceType> bf;
        vector<Patch::FaceType>::const_iterator bf_iter;
        p->getBoundaryFaces(bf);

        // Loop over all boundary faces on this patch
        for (bf_iter = bf.begin(); bf_iter != bf.end(); bf_iter++){

          Patch::FaceType face = *bf_iter; 
          IntVector inside = p->faceDirection(face); 
          CellIterator c_iter = getInteriorBoundaryCellIterator( p, bf_iter ); 
          FaceBoundaryBool faceIsBoundary; 

          for (c_iter.begin(); !c_iter.done(); c_iter++){

            IntVector c = *c_iter - inside; 
            IntVector cxp = c + IntVector(1,0,0);
            IntVector cxm = c - IntVector(1,0,0); 
            IntVector cyp = c + IntVector(0,1,0); 
            IntVector cym = c - IntVector(0,1,0);
            IntVector czp = c + IntVector(0,0,1);
            IntVector czm = c - IntVector(0,0,1);

            FaceData1D face_phi; 
            FaceData1D face_vel;
            double area; 
            IntVector coord; 

            //X-dimension
            coord[0] = 1; coord[1] = 0; coord[2] = 0; 
            area = Dx.y()*Dx.z(); 

            faceIsBoundary = checkFacesForBoundaries( p, c, coord ); 
            face_vel       = getFaceVelocity( c, Fconv, partVel, coord ); 
            face_phi       = upwindInterp( c, coord, oldPhi, face_vel, faceIsBoundary ); 
            Fconv[c]       = getFlux( area, face_vel, face_phi, areaFraction, coord, c ); 

#ifdef YDIM
            //Y-dimension
            coord[0] = 0; coord[1] = 1; coord[2] = 0; 
            area = Dx.x()*Dx.z(); 

            faceIsBoundary = checkFacesForBoundaries( p, c, coord ); 
            face_vel       = getFaceVelocity( c, Fconv, partVel, coord ); 
            face_phi       = upwindInterp( c, coord, oldPhi, face_vel, faceIsBoundary ); 
            Fconv[c]      += getFlux( area, face_vel, face_phi, areaFraction, coord, c ); 

#endif 
#ifdef ZDIM
            //Z-dimension
            coord[0] = 0; coord[1] = 0; coord[2] = 1; 
            area = Dx.x()*Dx.y(); 

            faceIsBoundary = checkFacesForBoundaries( p, c, coord ); 
            face_vel       = getFaceVelocity( c, Fconv, partVel, coord ); 
            face_phi       = upwindInterp( c, coord, oldPhi, face_vel, faceIsBoundary ); 
            Fconv[c]      += getFlux( area, face_vel, face_phi, areaFraction, coord, c ); 

#endif
          }
        }

      } else if (convScheme == "super_bee") { 

        for (iIter.begin(); !iIter.done(); iIter++){

          IntVector c   = *iIter;
          IntVector cxp = c + IntVector(1,0,0);
          IntVector cxm = c - IntVector(1,0,0);
          IntVector cyp = c + IntVector(0,1,0);
          IntVector cym = c - IntVector(0,1,0);
          IntVector czp = c + IntVector(0,0,1);
          IntVector czm = c - IntVector(0,0,1);

          FaceData1D face_phi; 
          FaceData1D face_vel;
          double area; 
          IntVector coord; 

          //X-dimension
          coord[0] = 1; coord[1] = 0; coord[2] = 0; 
          area = Dx.y()*Dx.z(); 

          face_vel       = getFaceVelocity( c, Fconv, partVel, coord ); 
          face_phi       = superBeeInterp( c, coord, oldPhi, face_vel ); 
          Fconv[c]       = getFlux( area, face_vel, face_phi, areaFraction, coord, c ); 

#ifdef YDIM
          //Y-dimension
          coord[0] = 0; coord[1] = 1; coord[2] = 0; 
          area = Dx.x()*Dx.z(); 

          face_vel       = getFaceVelocity( c, Fconv, partVel, coord ); 
          face_phi       = superBeeInterp( c, coord, oldPhi, face_vel ); 
          Fconv[c]      += getFlux( area, face_vel, face_phi, areaFraction, coord, c ); 
#endif 
#ifdef ZDIM
          //Z-dimension
          coord[0] = 0; coord[1] = 0; coord[2] = 1; 
          area = Dx.x()*Dx.y(); 

          face_vel       = getFaceVelocity( c, Fconv, partVel, coord ); 
          face_phi       = superBeeInterp( c, coord, oldPhi, face_vel ); 
          Fconv[c]      += getFlux( area, face_vel, face_phi, areaFraction, coord, c ); 
#endif
        }
        // Boundaries
        vector<Patch::FaceType> bf;
        vector<Patch::FaceType>::const_iterator bf_iter;
        p->getBoundaryFaces(bf);

        // Loop over all boundary faces on this patch
        for (bf_iter = bf.begin(); bf_iter != bf.end(); bf_iter++){

          Patch::FaceType face = *bf_iter; 
          IntVector inside = p->faceDirection(face); 
          CellIterator c_iter = getInteriorBoundaryCellIterator( p, bf_iter ); 
          FaceBoundaryBool faceIsBoundary; 

          for (c_iter.begin(); !c_iter.done(); c_iter++){

            IntVector c = *c_iter - inside; 
            IntVector cxp = c + IntVector(1,0,0);
            IntVector cxm = c - IntVector(1,0,0); 
            IntVector cyp = c + IntVector(0,1,0); 
            IntVector cym = c - IntVector(0,1,0);
            IntVector czp = c + IntVector(0,0,1);
            IntVector czm = c - IntVector(0,0,1);

            FaceData1D face_phi; 
            FaceData1D face_vel;
            double area; 
            IntVector coord; 

            //X-dimension
            coord[0] = 1; coord[1] = 0; coord[2] = 0; 
            area = Dx.y()*Dx.z(); 

            faceIsBoundary = checkFacesForBoundaries( p, c, coord ); 
            face_vel       = getFaceVelocity( c, Fconv, partVel, coord ); 
            face_phi       = superBeeInterp( c, coord, oldPhi, face_vel, faceIsBoundary ); 
            Fconv[c]       = getFlux( area, face_vel, face_phi, areaFraction, coord, c ); 

#ifdef YDIM
            //Y-dimension
            coord[0] = 0; coord[1] = 1; coord[2] = 0; 
            area = Dx.x()*Dx.z(); 

            faceIsBoundary = checkFacesForBoundaries( p, c, coord ); 
            face_vel       = getFaceVelocity( c, Fconv, partVel, coord ); 
            face_phi       = superBeeInterp( c, coord, oldPhi, face_vel, faceIsBoundary ); 
            Fconv[c]      += getFlux( area, face_vel, face_phi, areaFraction, coord, c ); 

#endif 
#ifdef ZDIM
            //Z-dimension
            coord[0] = 0; coord[1] = 0; coord[2] = 1; 
            area = Dx.x()*Dx.y(); 

            faceIsBoundary = checkFacesForBoundaries( p, c, coord ); 
            face_vel       = getFaceVelocity( c, Fconv, partVel, coord ); 
            face_phi       = superBeeInterp( c, coord, oldPhi, face_vel, faceIsBoundary ); 
            Fconv[c]      += getFlux( area, face_vel, face_phi, areaFraction, coord, c ); 

#endif
          }
        }

      } else if (convScheme == "central") {

        for (CellIterator iter=p->getCellIterator(); !iter.done(); iter++){

          IntVector c = *iter; 
          IntVector cxp = c + IntVector(1,0,0);
          IntVector cxm = c - IntVector(1,0,0);
          IntVector cyp = c + IntVector(0,1,0);
          IntVector cym = c - IntVector(0,1,0);
          IntVector czp = c + IntVector(0,0,1);
          IntVector czm = c - IntVector(0,0,1);

          IntVector coord; 

          FaceData1D face_phi; 
          FaceData1D face_vel; 
          double area; 

          //X-FACES
          coord[0] = 1; coord[1] = 0; coord[2] = 0; 
          area = Dx.y()*Dx.z();

          face_vel = getFaceVelocity( c, Fconv, uVel ); 
          face_phi = centralInterp( c, coord, oldPhi ); 

          Fconv[c] = getFlux( area, face_vel, face_phi, areaFraction, coord, c );
#ifdef YDIM
          //Y-FACES
          coord[0] = 0; coord[1] = 1; coord[2] = 0; 
          area = Dx.x()*Dx.z();

          face_vel  = getFaceVelocity( c, Fconv, vVel ); 
          face_phi  = centralInterp( c, coord, oldPhi ); 

          Fconv[c] += getFlux( area, face_vel, face_phi, areaFraction, coord, c );
#endif

#ifdef ZDIM
          //Z-FACES
          coord[0] = 0; coord[1] = 0; coord[2] = 1; 
          area = Dx.x()*Dx.y();

          face_vel = getFaceVelocity( c, Fconv, wVel ); 
          face_phi = centralInterp( c, coord, oldPhi ); 

          Fconv[c] += getFlux( area, face_vel, face_phi, areaFraction, coord, c );
#endif
        }
      }
    }


} // namespace Uintah
#endif
