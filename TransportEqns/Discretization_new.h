#ifndef Uintah_Component_Arches_Discretization_new_h
#define Uintah_Component_Arches_Discretization_new_h

#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/BoundaryConditions/BoundCond.h>
#include <Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Core/Exceptions/InvalidValue.h>
#include <CCA/Components/Arches/Directives.h>
#include <Core/Parallel/Parallel.h>

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

    // Functionally includes:
    // 1) Discretization functions for the outside world (transport equations)**. 
    // 2) Custom Data Managers
    // 3) Helper/Utility type functions
    // 4) Interpolation
    // 5) Discreization 
    //
    // ** These functions are intended to be the only ones accessed by the 
    //    transport equations. 

    public:

      Discretization_new();
      ~Discretization_new();

      //---------------------------------------------------------------------------
      // Discretization functions for transport equations. 
      // --------------------------------------------------------------------------

      /** @brief Computes the convection term.  This method is overloaded.  */
      template <class fT, class oldPhiT, class uT, class vT, class wT> void 
        computeConv(const Patch* p, fT& Fconv, oldPhiT& oldPhi, 
            uT& uVel, vT& vVel, 
            wT& wVel, constCCVariable<double>& den, constCCVariable<Vector>& areaFraction, 
            std::string convScheme);

      /** @brief Computes the convection term (no density term). This method is overloaded */
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

      /** @brief Computes the diffusion term for a scalar:
       * \int_S \grad \phi \cdot \dS */
      template <class fT, class oldPhiT, class gammaT> void 
        computeDiff( const Patch* p, fT& Fdiff, oldPhiT& oldPhi, gammaT& gamma,
        constCCVariable<Vector>& areaFraction, double turbPr, int mat_id, string varName );

      //---------------------------------------------------------------------------
      // Custom Data Managers
      // --------------------------------------------------------------------------
      
      //  A boolean for marking faces of the cell as touching a boundary 
      struct FaceBoundaryBool
      { 
        bool minus;
        bool plus; 
      };

      // Stores values in a one-D line on each face for a given cell 
      struct FaceData1D {
        double minus; // minus face
        double plus;  // plus face
      };

      //---------------------------------------------------------------------------
      // Helper/Utility type functions
      // --------------------------------------------------------------------------
      
      /** @brief Returns an iterator for all cells not touching the domain boundaries */
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

      /** @brief Returns an iterator for all cell touching a domain boundary */ 
      inline CellIterator getInteriorBoundaryCellIterator( const Patch* p, 
          const vector<Patch::FaceType>::const_iterator bf_iter ) const 
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
            l[1] += 1;
          if (yplus)
            h[1] -= 1;
          if (zminus)
            l[2] += 1;
          if (zplus)
            h[2] -= 1;

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

      /** @brief Checks if cell has a face on a boundary in a given absolute normal direction (coord) */
      inline FaceBoundaryBool checkFacesForBoundaries( const Patch* p, const IntVector c, const IntVector coord )
      {

        FaceBoundaryBool b;
        b.minus = false; 
        b.plus  = false; 

        IntVector l = p->getCellLowIndex();
        IntVector h = p->getCellHighIndex(); 

        if ( coord[0] == 1 ) {

          if ( c[0] == l[0] ) b.minus = true;
          if ( c[0] == h[0] - 1 ) b.plus  = true; 

        } else if ( coord[1] == 1 ) {

          if ( c[1] == l[1] ) b.minus = true;
          if ( c[1] == h[1] - 1 ) b.plus  = true; 

        } else if ( coord[2] == 1 ) {

          if ( c[2] == l[2] ) b.minus = true;
          if ( c[2] == h[2] - 1 ) b.plus  = true; 

        }

        return b; 
      }

      //---------------------------------------------------------------------------
      // Assemblers
      // These functions assemble other terms
      // --------------------------------------------------------------------------

      /** @brief Computes the flux term, int_A div(\rho u \phi) \cdot dA, where u is the velocity
       *          in the normal (coord) direction.  Note version has density. */
      inline double getFlux( const double area, FaceData1D den, FaceData1D vel, 
          FaceData1D phi, constCCVariable<Vector> areaFraction, IntVector coord, IntVector c )
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
      /** @brief Computes the flux term, int_A div(u \phi) \cdot dA, where u is the velocity
       *          in the normal (coord) direction.  Note version does not have density. */
      inline double getFlux( const double area, FaceData1D vel, FaceData1D phi, 
          constCCVariable<Vector> areaFraction, IntVector coord, IntVector c )
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

      //---------------------------------------------------------------------------
      // Interpolation
      // These functions interpolate
      // --------------------------------------------------------------------------
      
      /** @brief Return the face velocity for a CC cell given a CC velocity VECTOR */
      inline FaceData1D getFaceVelocity( const IntVector c, const CCVariable<double>& F, 
          constCCVariable<Vector> vel, const IntVector coord ){

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

      /** @brief Return the face velocity for a CC cell given an FCX vel */
      inline FaceData1D getFaceVelocity( const IntVector c, const CCVariable<double>& F, 
          constSFCXVariable<double> vel ){
        // cell-centered, x-direction
        FaceData1D the_vel; 
        the_vel.minus = vel[c];
        the_vel.plus  = vel[c + IntVector(1,0,0)];

        return the_vel; 
      }
      /** @brief Return the face velocity for a CC cell given an FCY vel */
      inline FaceData1D getFaceVelocity( const IntVector c, const CCVariable<double>& F, 
          constSFCYVariable<double> vel ){
        // cell-centered, y-direction
        FaceData1D the_vel; 
        the_vel.minus = vel[c];
        the_vel.plus  = vel[c + IntVector(0,1,0)];

        return the_vel; 
      }
      /** @brief Return the face velocity for a CC cell given an FCZ vel */
      inline FaceData1D getFaceVelocity( const IntVector c, const CCVariable<double>& F, 
          constSFCZVariable<double> vel ){
        // cell-centered, z-direction
        FaceData1D the_vel; 
        the_vel.minus = vel[c];
        the_vel.plus  = vel[c + IntVector(0,0,1)];

        return the_vel; 
      }

      /** @brief Return the face density for all cell types */
      template< class phiT > 
        inline FaceData1D getDensityOnFace( const IntVector c, const IntVector coord, 
            phiT& phi, constCCVariable<double>& den ){

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
      /** @brief Return the face density if there is a boundary for all cell types */
      template< class phiT > 
        inline FaceData1D getDensityOnFace( const IntVector c, const IntVector coord, 
            phiT& phi, constCCVariable<double>& den , FaceBoundaryBool isBoundary){

          FaceData1D face_values; 
          face_values.minus = 0.0;
          face_values.plus  = 0.0;

          TypeDescription::Type type = phi.getTypeDescription()->getType(); 

          if ( type == TypeDescription::CCVariable ) {
            IntVector cxm = c - coord; 
            IntVector cxp = c + coord; 

            if (isBoundary.minus)
              face_values.minus = den[cxm];
            else
              face_values.minus = 0.5 * (den[c] + den[cxm]); 
            if (isBoundary.plus)
              face_values.plus = den[cxp];
            else
              face_values.plus  = 0.5 * (den[c] + den[cxp]); 
          } else {
            // assume the only other type is a face type...
            IntVector cxm = c - coord; 

            face_values.minus = den[cxm];
            face_values.plus  = den[c]; 
          }

          return face_values; 
        }

      /** @brief Cell-centered interolation -- should work for all cell types */ 
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

      /** @brief Super Bee Interpolation -- should work for all cell types.
       *      This function does not have boundary checking (for speed). */
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

      /** @brief Super Bee Interpolation -- should work for all cell types. 
       *       This function includes boundary checking (slower).  */ 
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

      /* @brief Upwind interpolation -- should work for all data types. 
       *      This function does not have boundary checking (for speed). */ 
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

      /* @brief Upwind interpolation -- should work for all data types. 
       *      This function includes boundary checking (slower). */ 
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

      //---------------------------------------------------------------------------
      // Derivatives
      // These functions take derivatives
      // --------------------------------------------------------------------------

      /* @brief Computes the gradient of a scalar. */
      template <class phiT> 
        inline FaceData1D gradPtoF( const IntVector c, phiT& phi, const double dx, const IntVector coord )
        {
          Discretization_new::FaceData1D face_dx; 
          face_dx.plus = 0.0; 
          face_dx.minus = 0.0; 

          IntVector cxm = c - coord; 
          IntVector cxp = c + coord; 

          face_dx.plus  =  ( phi[cxp] - phi[c] ) / dx;
          face_dx.minus =  ( phi[c] - phi[cxm] ) / dx;

          return face_dx; 
        } 
        /* @brief Computes the gradient of a scalar, returning zero gradient if the face
         *         is a boundary cell that is set by a scalar dirichet condition. */
      template <class phiT> 
        inline FaceData1D gradPtoF( const IntVector c, phiT& phi, const double dx, const IntVector coord, FaceBoundaryBool isBoundary )
        {
          Discretization_new::FaceData1D face_dx; 
          face_dx.plus = 0.0; 
          face_dx.minus = 0.0; 

          IntVector cxm = c - coord; 
          IntVector cxp = c + coord; 

          if (isBoundary.plus) {

          }
          else 
            face_dx.plus  =  ( phi[cxp] - phi[c] ) / dx;

          if (isBoundary.minus) {

          }
          else
            face_dx.minus =  ( phi[c] - phi[cxm] ) / dx;

          return face_dx; 
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

  //========================= Convection ======================================

  //---------------------------------------------------------------------------
  // Method: Compute the convection term (with explicit Density interpolation) 
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
            face_den       = getDensityOnFace( c, coord, Fconv, den, faceIsBoundary ); 
            face_vel       = getFaceVelocity( c, Fconv, uVel ); 
            face_phi       = upwindInterp( c, coord, oldPhi, face_vel, faceIsBoundary ); 
            Fconv[c]       = getFlux( area, face_den, face_vel, face_phi, areaFraction, coord, c ); 

#ifdef YDIM
            //Y-dimension
            coord[0] = 0; coord[1] = 1; coord[2] = 0; 
            area = Dx.x()*Dx.z(); 

            faceIsBoundary = checkFacesForBoundaries( p, c, coord ); 
            face_den       = getDensityOnFace( c, coord, Fconv, den, faceIsBoundary ); 
            face_vel       = getFaceVelocity( c, Fconv, vVel ); 
            face_phi       = upwindInterp( c, coord, oldPhi, face_vel, faceIsBoundary ); 
            Fconv[c]      += getFlux( area, face_den, face_vel, face_phi, areaFraction, coord, c ); 

#endif 
#ifdef ZDIM
            //Z-dimension
            coord[0] = 0; coord[1] = 0; coord[2] = 1; 
            area = Dx.x()*Dx.y(); 

            faceIsBoundary = checkFacesForBoundaries( p, c, coord ); 
            face_den       = getDensityOnFace( c, coord, Fconv, den, faceIsBoundary ); 
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
            face_den       = getDensityOnFace( c, coord, Fconv, den, faceIsBoundary ); 
            face_vel       = getFaceVelocity( c, Fconv, uVel ); 
            face_phi       = superBeeInterp( c, coord, oldPhi, face_vel, faceIsBoundary ); 
            Fconv[c]       = getFlux( area, face_den, face_vel, face_phi, areaFraction, coord, c ); 

#ifdef YDIM
            //Y-dimension
            coord[0] = 0; coord[1] = 1; coord[2] = 0; 
            area = Dx.x()*Dx.z(); 

            faceIsBoundary = checkFacesForBoundaries( p, c, coord ); 
            face_den       = getDensityOnFace( c, coord, Fconv, den, faceIsBoundary ); 
            face_vel       = getFaceVelocity( c, Fconv, vVel ); 
            face_phi       = superBeeInterp( c, coord, oldPhi, face_vel, faceIsBoundary ); 
            Fconv[c]      += getFlux( area, face_den, face_vel, face_phi, areaFraction, coord, c ); 

#endif 
#ifdef ZDIM
            //Z-dimension
            coord[0] = 0; coord[1] = 0; coord[2] = 1; 
            area = Dx.x()*Dx.y(); 

            faceIsBoundary = checkFacesForBoundaries( p, c, coord ); 
            face_den       = getDensityOnFace( c, coord, Fconv, den, faceIsBoundary ); 
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
  // Method: Compute the convection term (no explicit density)
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

  //========================= Diffusion ======================================

  //---------------------------------------------------------------------------
  // Method: Compute the diffusion term
  // Simple diffusion term: \int_S \grad \phi \cdot dS 
  //---------------------------------------------------------------------------
  template <class fT, class oldPhiT, class gammaT> void 
    Discretization_new::computeDiff( const Patch* p, fT& Fdiff, oldPhiT& oldPhi, gammaT& gamma,
        constCCVariable<Vector>& areaFraction, double turbPr, int mat_id, string varName )
    {

      Vector Dx = p->dCell(); //assuming uniform grid
      
      for (CellIterator iter = p->getCellIterator(); !iter.done(); iter++){

        IntVector c = *iter; 
        IntVector coord; 

        FaceData1D face_gamma; 
        FaceData1D grad_phi; 

        coord[0] = 1; coord[1] = 0; coord[2] = 0; 
        double dx = Dx.x(); 

        face_gamma = centralInterp( c, coord, gamma ); 
        grad_phi   = gradPtoF( c, oldPhi, dx, coord ); 

        Vector c_af = areaFraction[c]; 
        Vector cp_af = areaFraction[c + coord]; 

        Fdiff[c] = 1.0/turbPr * Dx.y()*Dx.z() * 
                   ( face_gamma.plus * grad_phi.plus * cp_af.x() - 
                     face_gamma.minus * grad_phi.minus * c_af.x() ); 

#ifdef YDIM
        coord[0] = 0; coord[1] = 1; coord[2] = 0; 
        double dy = Dx.y(); 

        face_gamma = centralInterp( c, coord, gamma ); 
        grad_phi   = gradPtoF( c, oldPhi, dy, coord ); 

        cp_af = areaFraction[c + coord]; 

        Fdiff[c] += 1.0/turbPr * Dx.x()*Dx.z() *  
                   ( face_gamma.plus * grad_phi.plus * cp_af.y() - 
                     face_gamma.minus * grad_phi.minus * c_af.y() ); 
#endif
#ifdef ZDIM
        coord[0] = 0; coord[1] = 0; coord[2] = 1; 
        double dz = Dx.z(); 

        face_gamma = centralInterp( c, coord, gamma ); 
        grad_phi   = gradPtoF( c, oldPhi, dz, coord ); 

        cp_af = areaFraction[c + coord]; 

        Fdiff[c] += 1.0/turbPr * Dx.x()*Dx.y() * 
                   ( face_gamma.plus * grad_phi.plus * cp_af.z() - 
                      face_gamma.minus * grad_phi.minus * c_af.z() ); 

#endif

      }
      // boundaries ::: diffusion
      // need to go back and remove the contribution on the cell 
      // face if the face touched a dirichlet boundary condition 
      vector<Patch::FaceType> bf;
      vector<Patch::FaceType>::const_iterator bf_iter;
      p->getBoundaryFaces(bf);
      // Loop over all boundary faces on this patch
      for (bf_iter = bf.begin(); bf_iter != bf.end(); bf_iter++){
        Patch::FaceType face = *bf_iter; 

        IntVector insideCellDir = p->faceDirection(face); 
        int numChildren = p->getBCDataArray(face)->getNumberChildren(mat_id);
        for (int child = 0; child < numChildren; child++){

          string bc_kind = "NotSet"; 
          Iterator bound_ptr; 
          Iterator nu; //not used...who knows why?
          const BoundCondBase* bc = p->getArrayBCValues( face, mat_id, 
                                                         varName, bound_ptr, 
                                                         nu, child ); 
          const BoundCond<double> *new_bcs = dynamic_cast<const BoundCond<double> *>(bc); 
          if (new_bcs != 0) 
            bc_kind = new_bcs->getBCType__NEW(); 
          else {
            cout << "Warning!  Boundary condition not set for: " << endl;
            cout << "variable = " << varName << endl;
            cout << "face = " << face << endl;
          }

          delete bc; 
          if (bc_kind == "Dirichlet"){
            
            IntVector coord; 
            FaceData1D face_gamma; 
            FaceData1D grad_phi; 
            double dx; 
            switch (face) {
              case Patch::xminus:

                coord[0] = 1; coord[1] = 0; coord[2] = 0;
                dx = Dx.x();

                for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++){

                  IntVector bp1(*bound_ptr - insideCellDir); 
                  face_gamma = centralInterp( bp1, coord, gamma ); 
                  grad_phi   = gradPtoF( bp1, oldPhi, dx, coord ); 

                  Vector c_af = areaFraction[bp1]; 
                  Vector cp_af = areaFraction[bp1 + coord]; 

                  Fdiff[bp1] += 1.0/turbPr * Dx.y()*Dx.z() * 
                             ( face_gamma.minus * grad_phi.minus * c_af.x() ); 
                }
                break; 
              case Patch::xplus:

                coord[0] = 1; coord[1] = 0; coord[2] = 0;
                dx = Dx.x(); 

                for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++){

                  IntVector bp1(*bound_ptr - insideCellDir); 
                  face_gamma = centralInterp( bp1, coord, gamma ); 
                  grad_phi   = gradPtoF( bp1, oldPhi, dx, coord ); 

                  Vector c_af = areaFraction[bp1]; 
                  Vector cp_af = areaFraction[bp1 + coord]; 

                  Fdiff[bp1] -= 1.0/turbPr * Dx.y()*Dx.z() * 
                             ( face_gamma.plus * grad_phi.plus * cp_af.x() ); 
                }
                break; 
#ifdef YDIM
              case Patch::yminus:

                coord[0] = 0; coord[1] = 1; coord[2] = 0;
                dx = Dx.y(); 

                for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++){
                  IntVector bp1(*bound_ptr - insideCellDir); 
                  face_gamma = centralInterp( bp1, coord, gamma ); 
                  grad_phi   = gradPtoF( bp1, oldPhi, dx, coord ); 

                  Vector c_af = areaFraction[bp1]; 
                  Vector cp_af = areaFraction[bp1 + coord]; 

                  Fdiff[bp1] += 1.0/turbPr * Dx.x()*Dx.z() * 
                             ( face_gamma.minus * grad_phi.minus * c_af.y() ); 
                }
                break; 
              case Patch::yplus:

                coord[0] = 0; coord[1] = 1; coord[2] = 0;
                dx = Dx.y();

                for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++){
                  IntVector bp1(*bound_ptr - insideCellDir); 
                  face_gamma = centralInterp( bp1, coord, gamma ); 
                  grad_phi   = gradPtoF( bp1, oldPhi, dx, coord ); 

                  Vector c_af = areaFraction[bp1]; 
                  Vector cp_af = areaFraction[bp1 + coord]; 

                  Fdiff[bp1] -= 1.0/turbPr * Dx.x()*Dx.z() * 
                             ( face_gamma.plus * grad_phi.plus * cp_af.y() ); 
                }
                break;
#endif 
#ifdef ZDIM
              case Patch::zminus:

                coord[0] = 0; coord[1] = 0; coord[2] = 1;
                dx = Dx.z(); 

                for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++){

                  IntVector bp1(*bound_ptr - insideCellDir); 
                  face_gamma = centralInterp( bp1, coord, gamma ); 
                  grad_phi   = gradPtoF( bp1, oldPhi, dx, coord ); 

                  Vector c_af = areaFraction[bp1]; 
                  Vector cp_af = areaFraction[bp1 + coord]; 

                  Fdiff[bp1] += 1.0/turbPr * Dx.x()*Dx.y() * 
                             ( face_gamma.minus * grad_phi.minus * c_af.z() ); 
                }
                break; 
              case Patch::zplus:

                coord[0] = 0; coord[1] = 0; coord[2] = 1;
                dx = Dx.z(); 

                for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++){

                  IntVector bp1(*bound_ptr - insideCellDir); 
                  face_gamma = centralInterp( bp1, coord, gamma ); 
                  grad_phi   = gradPtoF( bp1, oldPhi, dx, coord ); 

                  Vector c_af = areaFraction[bp1]; 
                  Vector cp_af = areaFraction[bp1 + coord]; 

                  Fdiff[bp1] -= 1.0/turbPr * Dx.x()*Dx.y() * 
                             ( face_gamma.plus * grad_phi.plus * cp_af.z() ); 
                }
                break; 
#endif
            case Patch::numFaces:
              break;
            case Patch::invalidFace:
              break; 
            }
          }
        }
      }
    }

} // namespace Uintah
#endif
