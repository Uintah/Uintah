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
       * \f$ \int_{S} \nabla \phi \cdot dS \f$ 
       * for a non-constant pr number */
      template <class fT, class oldPhiT, class gammaT> void 
        computeDiff( const Patch* p, fT& Fdiff, oldPhiT& oldPhi, gammaT& gamma,
        constCCVariable<Vector>& areaFraction, constCCVariable<double>& prNo, int mat_id, string varName );

      /** @brief Computes the diffusion term for a scalar:
       * \f$ \int_{S} \nabla \phi \cdot dS \f$
       * assuming a constant pr number */
      template <class fT, class oldPhiT, class gammaT> void 
        computeDiff( const Patch* p, fT& Fdiff, oldPhiT& oldPhi, gammaT& gamma,
        constCCVariable<Vector>& areaFraction, double const_prNo, int mat_id, string varName );

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
            l[2] += 1;
          if (zplus)
            h[2] -= 1;
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

        bool fplus  = false; 
        bool fminus = false; 

        int I = -1; 
        if ( coord[0] == 1 ) { 
          fminus = p->getBCType(Patch::xminus) != Patch::Neighbor; 
          fplus  = p->getBCType(Patch::xplus ) != Patch::Neighbor; 
          I = 0; 
        } else if ( coord[1] == 1 ){ 
          fminus = p->getBCType(Patch::yminus) != Patch::Neighbor; 
          fplus  = p->getBCType(Patch::yplus ) != Patch::Neighbor; 
          I = 1; 
        } else if ( coord[2] == 1 ){ 
          fminus = p->getBCType(Patch::zminus) != Patch::Neighbor; 
          fplus  = p->getBCType(Patch::zplus ) != Patch::Neighbor; 
          I = 2; 
        } 

        if ( fminus && c[I] == l[I] ) { 
          b.minus = true; 
        } 
        if ( fplus  && c[I] == h[I]-1 ) { 
          b.plus = true; 
        } 

        return b; 
      }

      //---------------------------------------------------------------------------
      // Assemblers
      // These functions assemble other terms
      // --------------------------------------------------------------------------

      /** @brief Computes the flux term, \f$ int_A div{\rho u \phi} \cdot dA \f$, where u is the velocity
       *          in the normal (coord) direction.  Note version has density. */
      inline double getFlux( const double area, FaceData1D den, FaceData1D vel, 
          FaceData1D phi, constCCVariable<Vector>& areaFraction, IntVector coord, IntVector c )
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
      /** @brief Computes the flux term, \f$ int_A div{u \phi} \cdot dA \f$, where u is the velocity
       *          in the normal (coord) direction.  Note version does not have density. */
      inline double getFlux( const double area, FaceData1D vel, FaceData1D phi, 
          constCCVariable<Vector>& areaFraction, IntVector coord, IntVector c )
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
      
      /** @brief Return the face velocity for a CC cell given a CC velocity VECTOR */
      inline FaceData1D getFaceVelocity( const IntVector c, const CCVariable<double>& F, 
          constCCVariable<Vector>& vel, const IntVector coord ){

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
          constSFCXVariable<double>& vel ){
        // cell-centered, x-direction
        FaceData1D the_vel; 
        the_vel.minus = vel[c];
        the_vel.plus  = vel[c + IntVector(1,0,0)];

        return the_vel; 
      }
      /** @brief Return the face velocity for a CC cell given an FCY vel */
      inline FaceData1D getFaceVelocity( const IntVector c, const CCVariable<double>& F, 
          constSFCYVariable<double>& vel ){
        // cell-centered, y-direction
        FaceData1D the_vel; 
        the_vel.minus = vel[c];
        the_vel.plus  = vel[c + IntVector(0,1,0)];

        return the_vel; 
      }
      /** @brief Return the face velocity for a CC cell given an FCZ vel */
      inline FaceData1D getFaceVelocity( const IntVector c, const CCVariable<double>& F, 
          constSFCZVariable<double>& vel ){
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

      /** @brief Return the face density for all cell types when boundaries are near */
      template< class phiT > 
        inline FaceData1D getDensityOnFace( const IntVector c, const IntVector coord, 
            phiT& phi, constCCVariable<double>& den, Discretization_new::FaceBoundaryBool isBoundary ){

          FaceData1D face_values; 
          face_values.minus = 0.0;
          face_values.plus  = 0.0;

          TypeDescription::Type type = phi.getTypeDescription()->getType(); 

          if ( type == TypeDescription::CCVariable ) {
            IntVector cxm = c - coord; 
            IntVector cxp = c + coord; 
            
            if ( isBoundary.minus ){ 
              face_values.minus = den[cxm]; 
            } else { 
              face_values.minus = 0.5 * (den[c] + den[cxm]); 
            }

            if ( isBoundary.plus ){ 
              face_values.plus = den[cxp]; 
            } else { 
              face_values.plus  = 0.5 * (den[c] + den[cxp]); 
            }
          } else {
            // assume the only other type is a face type...
            IntVector cxm = c - coord; 

            face_values.minus = den[cxm];
            face_values.plus  = den[c]; 
          }

          return face_values; 
        }

      //---------------------------------------------------------------------------
      // Interpolation Class
      //
      /** @brief Calls the specific interpolant */ 
      template <typename operT, typename phiT>
      class ConvHelper1 { 
        public: 
          ConvHelper1<operT, phiT>( operT* opr, phiT& phi ) : _opr(opr), _phi(phi){};
          ~ConvHelper1(){}; 

          // FaceData1D no_bc( const IntVector c, const IntVector coord, FaceData1D vel, constCCVariable<Vector>& area_fraction ){ 
          //   FaceData1D result = _opr->do_interpolation_nobc( c, coord, _phi, vel, area_fraction ); 
          //   return result; 
          // }; 

          // FaceData1D with_bc( const IntVector c, const IntVector coord, FaceData1D vel, constCCVariable<Vector>& area_fraction, FaceBoundaryBool isBoundary ){ 
          //   FaceData1D result = _opr->do_interpolation_withbc( c, coord, _phi, vel, area_fraction, isBoundary ); 
          //   return result; 
          // }; 

          //----------------------------------------------------------------------------------
          // With explicit density treatment
          //
          /** @brief actual computes the convection term with the specified operator */ 
          template <class uT, class vT, class wT, class fT> 
          void do_convection( const Patch* p, fT& Fconv, uT& uVel, vT& vVel, wT& wVel, 
              constCCVariable<double>& den, constCCVariable<Vector>& area_fraction, Discretization_new* D){

            Vector Dx = p->dCell(); 
            CellIterator iIter  = D->getInteriorCellIterator( p ); 
           //-------------------- Interior 
           for (iIter.begin(); !iIter.done(); iIter++){

             IntVector c   = *iIter;

             Discretization_new::FaceData1D face_den;
             Discretization_new::FaceData1D face_phi; 
             Discretization_new::FaceData1D face_vel;
             double area; 
             IntVector coord; 

             //X-dimension
             coord[0] = 1; coord[1] = 0; coord[2] = 0; 
             area = Dx.y()*Dx.z();

             face_den       = D->getDensityOnFace( c, coord, Fconv, den ); 
             face_vel       = D->getFaceVelocity( c, Fconv, uVel ); 
             face_phi       = _opr->no_bc( c, coord, _phi, face_vel, area_fraction ); 
             Fconv[c]       = D->getFlux( area, face_den, face_vel, face_phi, area_fraction, coord, c ); 
#ifdef YDIM
             //Y-dimension
             coord[0] = 0; coord[1] = 1; coord[2] = 0; 
             area = Dx.x()*Dx.z(); 

             face_den       = D->getDensityOnFace( c, coord, Fconv, den ); 
             face_vel       = D->getFaceVelocity( c, Fconv, vVel ); 
             face_phi       = _opr->no_bc( c, coord, _phi, face_vel, area_fraction ); 
             Fconv[c]      += D->getFlux( area, face_den, face_vel, face_phi, area_fraction, coord, c ); 

#endif 
#ifdef ZDIM
             //Z-dimension
             coord[0] = 0; coord[1] = 0; coord[2] = 1; 
             area = Dx.x()*Dx.y(); 

             face_den       = D->getDensityOnFace( c, coord, Fconv, den ); 
             face_vel       = D->getFaceVelocity( c, Fconv, wVel ); 
             face_phi       = _opr->no_bc( c, coord, _phi, face_vel, area_fraction ); 
             Fconv[c]      += D->getFlux( area, face_den, face_vel, face_phi, area_fraction, coord, c ); 

#endif
            }

            //--------------- Boundaries
            vector<Patch::FaceType> bf;
            vector<Patch::FaceType>::const_iterator bf_iter;
            p->getBoundaryFaces(bf);

            // Loop over all boundary faces on this patch
            for (bf_iter = bf.begin(); bf_iter != bf.end(); bf_iter++){

             Patch::FaceType face = *bf_iter; 
             IntVector inside = p->faceDirection(face); 
             CellIterator c_iter = D->getInteriorBoundaryCellIterator( p, bf_iter ); 
             Discretization_new::FaceBoundaryBool faceIsBoundary; 

             for (c_iter.begin(); !c_iter.done(); c_iter++){

               IntVector c = *c_iter - inside; 

               Discretization_new::FaceData1D face_den;
               Discretization_new::FaceData1D face_phi; 
               Discretization_new::FaceData1D face_vel;
               double area; 
               IntVector coord; 

               //X-dimension
               coord[0] = 1; coord[1] = 0; coord[2] = 0; 
               area = Dx.y()*Dx.z();            

               faceIsBoundary = D->checkFacesForBoundaries( p, c, coord ); 
               face_den       = D->getDensityOnFace( c, coord, Fconv, den ); 
               face_vel       = D->getFaceVelocity( c, Fconv, uVel ); 
               face_phi       = _opr->with_bc( c, coord, _phi, face_vel, area_fraction, faceIsBoundary ); 
               Fconv[c]       = D->getFlux( area, face_den, face_vel, face_phi, area_fraction, coord, c ); 

#ifdef YDIM
               //Y-dimension
               coord[0] = 0; coord[1] = 1; coord[2] = 0; 
               area = Dx.x()*Dx.z(); 

               faceIsBoundary = D->checkFacesForBoundaries( p, c, coord ); 
               face_den       = D->getDensityOnFace( c, coord, Fconv, den ); 
               face_vel       = D->getFaceVelocity( c, Fconv, vVel ); 
               face_phi       = _opr->with_bc( c, coord, _phi, face_vel, area_fraction, faceIsBoundary ); 
               Fconv[c]      += D->getFlux( area, face_den, face_vel, face_phi, area_fraction, coord, c ); 

#endif 
#ifdef ZDIM
               //Z-dimension
               coord[0] = 0; coord[1] = 0; coord[2] = 1; 
               area = Dx.x()*Dx.y(); 

               faceIsBoundary = D->checkFacesForBoundaries( p, c, coord ); 
               face_den       = D->getDensityOnFace( c, coord, Fconv, den ); 
               face_vel       = D->getFaceVelocity( c, Fconv, wVel ); 
               face_phi       = _opr->with_bc( c, coord, _phi, face_vel, area_fraction, faceIsBoundary ); 
               Fconv[c]      += D->getFlux( area, face_den, face_vel, face_phi, area_fraction, coord, c ); 

#endif
              }
            }
          }

          //--------------------------------------------------------------------------
          // no explicit density representation 
          //
          /** @brief Actually computers the convection term without density rep. */ 
          template <class uT, class vT, class wT, class fT> 
          void do_convection( const Patch* p, fT& Fconv, uT& uVel, vT& vVel, wT& wVel, 
              constCCVariable<Vector>& area_fraction, Discretization_new* D){

            Vector Dx = p->dCell(); 
            CellIterator iIter  = D->getInteriorCellIterator( p ); 
           //-------------------- Interior 
           for (iIter.begin(); !iIter.done(); iIter++){

             IntVector c   = *iIter;

             Discretization_new::FaceData1D face_phi; 
             Discretization_new::FaceData1D face_vel;
             double area; 
             IntVector coord; 

             //X-dimension
             coord[0] = 1; coord[1] = 0; coord[2] = 0; 
             area = Dx.y()*Dx.z();

             face_vel       = D->getFaceVelocity( c, Fconv, uVel ); 
             face_phi       = _opr->no_bc( c, coord, _phi, face_vel, area_fraction ); 
             Fconv[c]       = D->getFlux( area, face_vel, face_phi, area_fraction, coord, c ); 
#ifdef YDIM
             //Y-dimension
             coord[0] = 0; coord[1] = 1; coord[2] = 0; 
             area = Dx.x()*Dx.z(); 

             face_vel       = D->getFaceVelocity( c, Fconv, vVel ); 
             face_phi       = _opr->no_bc( c, coord, _phi, face_vel, area_fraction ); 
             Fconv[c]      += D->getFlux( area, face_vel, face_phi, area_fraction, coord, c ); 

#endif 
#ifdef ZDIM
             //Z-dimension
             coord[0] = 0; coord[1] = 0; coord[2] = 1; 
             area = Dx.x()*Dx.y(); 

             face_vel       = D->getFaceVelocity( c, Fconv, wVel ); 
             face_phi       = _opr->no_bc( c, coord, _phi, face_vel, area_fraction ); 
             Fconv[c]      += D->getFlux( area, face_vel, face_phi, area_fraction, coord, c ); 

#endif
            }

            //--------------- Boundaries
            vector<Patch::FaceType> bf;
            vector<Patch::FaceType>::const_iterator bf_iter;
            p->getBoundaryFaces(bf);

            // Loop over all boundary faces on this patch
            for (bf_iter = bf.begin(); bf_iter != bf.end(); bf_iter++){

             Patch::FaceType face = *bf_iter; 
             IntVector inside = p->faceDirection(face); 
             CellIterator c_iter = D->getInteriorBoundaryCellIterator( p, bf_iter ); 
             Discretization_new::FaceBoundaryBool faceIsBoundary; 

             for (c_iter.begin(); !c_iter.done(); c_iter++){

               IntVector c = *c_iter - inside; 

               Discretization_new::FaceData1D face_phi; 
               Discretization_new::FaceData1D face_vel;
               double area; 
               IntVector coord; 

               //X-dimension
               coord[0] = 1; coord[1] = 0; coord[2] = 0; 
               area = Dx.y()*Dx.z();            

               faceIsBoundary = D->checkFacesForBoundaries( p, c, coord ); 
               face_vel       = D->getFaceVelocity( c, Fconv, uVel ); 
               face_phi       = _opr->with_bc( c, coord, _phi, face_vel, area_fraction, faceIsBoundary ); 
               Fconv[c]       = D->getFlux( area, face_vel, face_phi, area_fraction, coord, c ); 

#ifdef YDIM
               //Y-dimension
               coord[0] = 0; coord[1] = 1; coord[2] = 0; 
               area = Dx.x()*Dx.z(); 

               faceIsBoundary = D->checkFacesForBoundaries( p, c, coord ); 
               face_vel       = D->getFaceVelocity( c, Fconv, vVel ); 
               face_phi       = _opr->with_bc( c, coord, _phi, face_vel, area_fraction, faceIsBoundary ); 
               Fconv[c]      += D->getFlux( area, face_vel, face_phi, area_fraction, coord, c ); 

#endif 
#ifdef ZDIM
               //Z-dimension
               coord[0] = 0; coord[1] = 0; coord[2] = 1; 
               area = Dx.x()*Dx.y(); 

               faceIsBoundary = D->checkFacesForBoundaries( p, c, coord ); 
               face_vel       = D->getFaceVelocity( c, Fconv, wVel ); 
               face_phi       = _opr->with_bc( c, coord, _phi, face_vel, area_fraction, faceIsBoundary ); 
               Fconv[c]      += D->getFlux( area, face_vel, face_phi, area_fraction, coord, c ); 

#endif
              }
            }
          }

          //--------------------------------------------------------------------------
          // Speicalized for dqmom 
          //
          /** @brief Actually computers the convection term without density rep. */ 
          template <class uT, class vT, class wT, class fT> 
          void do_convection( const Patch* p, fT& Fconv, uT& uVel, vT& vVel, wT& wVel, 
              constCCVariable<Vector>& partVel, 
              constCCVariable<Vector>& area_fraction, Discretization_new* D){

            Vector Dx = p->dCell(); 
            CellIterator iIter  = D->getInteriorCellIterator( p ); 
           //-------------------- Interior 
           for (iIter.begin(); !iIter.done(); iIter++){

             IntVector c   = *iIter;

             Discretization_new::FaceData1D face_phi; 
             Discretization_new::FaceData1D face_vel;
             double area; 
             IntVector coord; 

             //X-dimension
             coord[0] = 1; coord[1] = 0; coord[2] = 0; 
             area = Dx.y()*Dx.z();

             face_vel       = D->getFaceVelocity( c, Fconv, partVel, coord ); 
             face_phi       = _opr->no_bc( c, coord, _phi, face_vel, area_fraction ); 
             Fconv[c]       = D->getFlux( area, face_vel, face_phi, area_fraction, coord, c ); 
#ifdef YDIM
             //Y-dimension
             coord[0] = 0; coord[1] = 1; coord[2] = 0; 
             area = Dx.x()*Dx.z(); 

             face_vel       = D->getFaceVelocity( c, Fconv, partVel, coord ); 
             face_phi       = _opr->no_bc( c, coord, _phi, face_vel, area_fraction ); 
             Fconv[c]      += D->getFlux( area, face_vel, face_phi, area_fraction, coord, c ); 

#endif 
#ifdef ZDIM
             //Z-dimension
             coord[0] = 0; coord[1] = 0; coord[2] = 1; 
             area = Dx.x()*Dx.y(); 

             face_vel       = D->getFaceVelocity( c, Fconv, partVel, coord ); 
             face_phi       = _opr->no_bc( c, coord, _phi, face_vel, area_fraction ); 
             Fconv[c]      += D->getFlux( area, face_vel, face_phi, area_fraction, coord, c ); 

#endif
            }

            //--------------- Boundaries
            vector<Patch::FaceType> bf;
            vector<Patch::FaceType>::const_iterator bf_iter;
            p->getBoundaryFaces(bf);

            // Loop over all boundary faces on this patch
            for (bf_iter = bf.begin(); bf_iter != bf.end(); bf_iter++){

             Patch::FaceType face = *bf_iter; 
             IntVector inside = p->faceDirection(face); 
             CellIterator c_iter = D->getInteriorBoundaryCellIterator( p, bf_iter ); 
             Discretization_new::FaceBoundaryBool faceIsBoundary; 

             for (c_iter.begin(); !c_iter.done(); c_iter++){

               IntVector c = *c_iter - inside; 

               Discretization_new::FaceData1D face_phi; 
               Discretization_new::FaceData1D face_vel;
               double area; 
               IntVector coord; 

               //X-dimension
               coord[0] = 1; coord[1] = 0; coord[2] = 0; 
               area = Dx.y()*Dx.z();            

               faceIsBoundary = D->checkFacesForBoundaries( p, c, coord ); 
               face_vel       = D->getFaceVelocity( c, Fconv, partVel, coord ); 
               face_phi       = _opr->with_bc( c, coord, _phi, face_vel, area_fraction, faceIsBoundary ); 
               Fconv[c]       = D->getFlux( area, face_vel, face_phi, area_fraction, coord, c ); 

#ifdef YDIM
               //Y-dimension
               coord[0] = 0; coord[1] = 1; coord[2] = 0; 
               area = Dx.x()*Dx.z(); 

               faceIsBoundary = D->checkFacesForBoundaries( p, c, coord ); 
               face_vel       = D->getFaceVelocity( c, Fconv, partVel, coord ); 
               face_phi       = _opr->with_bc( c, coord, _phi, face_vel, area_fraction, faceIsBoundary ); 
               Fconv[c]      += D->getFlux( area, face_vel, face_phi, area_fraction, coord, c ); 

#endif 
#ifdef ZDIM
               //Z-dimension
               coord[0] = 0; coord[1] = 0; coord[2] = 1; 
               area = Dx.x()*Dx.y(); 

               faceIsBoundary = D->checkFacesForBoundaries( p, c, coord ); 
               face_vel       = D->getFaceVelocity( c, Fconv, partVel, coord ); 
               face_phi       = _opr->with_bc( c, coord, _phi, face_vel, area_fraction, faceIsBoundary ); 
               Fconv[c]      += D->getFlux( area, face_vel, face_phi, area_fraction, coord, c ); 

#endif
              }
            }
          }

        private: 
          operT* _opr; 
          phiT& _phi; 

      };

      // ---------------------------------------------------------------------------
      // Upwind Interpolation 
      //
      /** @brief Upwind interolation -- should work for all cell types */ 
      template <typename phiT>
      class UpwindInterpolation { 

        public: 

        UpwindInterpolation(){};
        ~UpwindInterpolation(){}; 

        FaceData1D no_bc( const IntVector c, const IntVector coord, phiT& phi, 
            FaceData1D vel, constCCVariable<Vector>& areaFraction ) 
        { 
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
        }; 

        FaceData1D inline with_bc( const IntVector c, const IntVector coord, phiT& phi, 
            FaceData1D vel, constCCVariable<Vector>& areaFraction, FaceBoundaryBool isBoundary ) 
        { 
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
        }; 
      }; 

      // ---------------------------------------------------------------------------
      // Central Interpolation 
      //
      /** @brief Cell-centered interolation -- should work for all cell types */ 
      template <typename phiT>
      class CentralInterpolation { 

        public: 

        CentralInterpolation(){};
        ~CentralInterpolation(){}; 

        FaceData1D inline no_bc( const IntVector c, const IntVector coord, phiT& phi, 
            FaceData1D vel, constCCVariable<Vector>& areaFraction ) 
        { 
          IntVector cxp = c + coord; 
          IntVector cxm = c - coord; 

          FaceData1D face_values; 

          face_values.minus = 0.5 * ( phi[c] + phi[cxm] ); 
          face_values.plus  = 0.5 * ( phi[c] + phi[cxp] ); 

          return face_values; 
        }; 

        FaceData1D inline with_bc( const IntVector c, const IntVector coord, phiT& phi, 
            FaceData1D vel, constCCVariable<Vector>& areaFraction, FaceBoundaryBool isBoundary ) 
        { 
          IntVector cxp = c + coord; 
          IntVector cxm = c - coord; 

          FaceData1D face_values; 

          face_values.minus = 0.5 * ( phi[c] + phi[cxm] ); 
          face_values.plus  = 0.5 * ( phi[c] + phi[cxp] ); 

          return face_values; 
        }; 
      }; 

      // ---------------------------------------------------------------------------
      // Flux Limiters
      //
      // Limiter Functions:
      // This is the base class: 
      // Given r, should return psi
      //
      // To add a new limiter: 
      // 1) Add a derived LimiterFunctionBase class to return psi
      // 2) Add an instance of the function in the FluxLimiterInterpolation() constructor
      // 3) Add the option in the spec file 
      
      /** @brief Limiter function base class */
      class LimiterFunctionBase { 

        public: 

          LimiterFunctionBase() : _huge(1e10) {}; 
          virtual ~LimiterFunctionBase(){}; 

          virtual double get_psi(double const r) = 0;

        protected: 

          const double _huge; 

      };

      /** @brief Super Bee function */ 
      class SuperBeeFunction : public LimiterFunctionBase { 

        public: 

        SuperBeeFunction(){};
        ~SuperBeeFunction(){}; 

        double get_psi( double const r ){ 

          double psi = 2.0; // when r = infinity

          if ( r < _huge ){ 
            psi = std::max( std::min( 2.0*r, 1.0 ), std::min( r, 2.0 ) ); 
            psi = std::max( 0.0, psi ); 
          } 

          return psi; 

        };

      };

      /** @brief Roe MinMod function */ 
      class RoeMindModFunction : public LimiterFunctionBase { 

        public: 

          RoeMindModFunction(){}; 
          ~RoeMindModFunction(){};

          double get_psi( double const r ){ 

            double psi = 1.0; // when r = infinity

            if ( r < _huge ) { 
              psi = std::min(r, 1.0);
              psi = std::max( 0.0, psi );
            }

            return psi; 

          };
      }; 

      /** @brief Van Leer function */ 
      class VanLeerFunction : public LimiterFunctionBase { 

        public: 

          VanLeerFunction(){}; 
          ~VanLeerFunction(){};

          double get_psi( double const r ){ 

            double psi = 2.0; // when r = infinity

            if ( r < _huge ) { 
              psi = ( r + std::abs(r) ) / ( 1.0 + std::abs(r) ); 
            }

            return psi; 

          };
      }; 
      //--- end functions --- below is the actual interpolation for all limiters

      /** @brief Generalized Flux Limiter */ 
      template <typename phiT>
      class FluxLimiterInterpolation { 

        public: 

        FluxLimiterInterpolation( std::string type ){
       
          if ( type == "super_bee" ) { 

            _limiter_function = scinew SuperBeeFunction(); 

          } else if ( type == "roe_minmod" ) { 

            _limiter_function = scinew RoeMindModFunction(); 

          } else if ( type == "vanleer" ) { 

            _limiter_function = scinew VanLeerFunction(); 

          } else { 

            throw InternalError("ERROR: Limiter function not recognized.", __FILE__, __LINE__);

          } 
        
        };
        ~FluxLimiterInterpolation(){
        
          delete _limiter_function; 

        }; 

        FaceData1D inline no_bc( const IntVector c, const IntVector coord, phiT& phi, 
            FaceData1D vel, constCCVariable<Vector>& areaFraction ) 
        { 
          FaceData1D face_values;
          face_values.plus  = 0.0;
          face_values.minus = 0.0;

          double r   = 0.0; 
          double psi = 0.0; 
          double Sup = 0.0;
          double Sdn = 0.0;
          const double tiny = 1.0e-16; 

          IntVector cxp  = c + coord; 
          IntVector cxpp = c + coord + coord; 
          IntVector cxm  = c - coord; 
          IntVector cxmm = c - coord - coord; 

          int dim = 0; 
          if (coord[0] == 1){
            dim =0; 
          } else if (coord[1] == 1) { 
            dim = 1; 
          } else {  
            dim = 2;
          }

          // - FACE
          if ( vel.minus > 0.0 ) {
            Sup = phi[cxm];
            Sdn = phi[c];
            r = ( phi[cxm] - phi[cxmm] ) / ( phi[c] - phi[cxm] );

            if ( areaFraction[cxm][dim] < tiny || areaFraction[c][dim] < tiny ){
              r = 0.0;
            }

          } else if ( vel.minus < 0.0 ) {
            Sup = phi[c];
            Sdn = phi[cxm];
            r = ( phi[cxp] - phi[c] ) / ( phi[c] - phi[cxm] );

            if ( areaFraction[cxp][dim] < tiny || areaFraction[c][dim] < tiny ){
              r = 0.0;
            }
          }

          psi = _limiter_function->get_psi( r ); 

          face_values.minus = Sup + 0.5*psi*( Sdn - Sup );

          Sup = 0.0;
          Sdn = 0.0; 

          // + FACE
          if ( vel.plus > 0.0 ) {
            r = ( phi[c] - phi[cxm] ) / ( phi[cxp] - phi[c] );
            Sup = phi[c];
            Sdn = phi[cxp];

            if ( areaFraction[c][dim] < tiny || areaFraction[cxp][dim] < tiny ) {
              r = 0.0;
            }

          } else if ( vel.plus < 0.0 ) {
            r = ( phi[cxpp] - phi[cxp] ) / ( phi[cxp] - phi[c] );
            Sup = phi[cxp];
            Sdn = phi[c]; 

            if ( areaFraction[cxpp][dim] < tiny || areaFraction[cxp][dim] < tiny ){
              r = 0.0; 
            }
          }

          psi = _limiter_function->get_psi( r ); 

          face_values.plus = Sup + 0.5*psi*( Sdn - Sup );

          return face_values; 
        }; 

        FaceData1D inline with_bc( const IntVector c, const IntVector coord, phiT& phi, 
            FaceData1D vel, constCCVariable<Vector>& areaFraction, FaceBoundaryBool isBoundary ) 
        { 
          FaceData1D face_values;
          face_values.plus  = 0.0;
          face_values.minus = 0.0;

          double r   = 0.0; 
          double psi = 0.0; 
          double Sup = 0.0;
          double Sdn = 0.0;
          const double tiny = 1.0e-16; 

          IntVector cxp  = c + coord; 
          IntVector cxpp = c + coord + coord; 
          IntVector cxm  = c - coord; 
          IntVector cxmm = c - coord - coord; 

          int dim = 0; 
          if (coord[0] == 1){
            dim =0; 
          } else if (coord[1] == 1) {
            dim = 1; 
          } else {  
            dim = 2;
          }

          // - FACE
          if (isBoundary.minus) {
            face_values.minus = 0.5*(phi[c]+phi[cxm]);
          } else { 
            if ( vel.minus > 0.0 ) {
              Sup = phi[cxm];
              Sdn = phi[c];
              r = ( phi[cxm] - phi[cxmm] ) / ( phi[c] - phi[cxm] ); 

              if ( areaFraction[cxm][dim] < tiny || areaFraction[c][dim] < tiny ){
                r = 0.0;
              }

            } else if ( vel.minus < 0.0 ) {
              Sup = phi[c];
              Sdn = phi[cxm];
              r = ( phi[cxp] - phi[c] ) / ( phi[c] - phi[cxm] );

              if ( areaFraction[cxp][dim] < tiny || areaFraction[c][dim] < tiny ){
                r = 0.0;
              }
            }

            psi = _limiter_function->get_psi( r ); 

            face_values.minus = Sup + 0.5*psi*( Sdn - Sup );
          }

          Sup = 0.0; 
          Sdn = 0.0; 

          // + FACE
          if (isBoundary.plus) {
            face_values.plus = 0.5*(phi[c] + phi[cxp]);
          } else { 
            if ( vel.plus > 0.0 ) {
              r = ( phi[c] - phi[cxm] ) / ( phi[cxp] - phi[c] );
              Sup = phi[c];
              Sdn = phi[cxp];

            if ( areaFraction[c][dim] < tiny || areaFraction[cxp][dim] < tiny ){
              r = 0.0;
            }

            } else if ( vel.plus < 0.0 ) {
              r = ( phi[cxpp] - phi[cxp] ) / ( phi[cxp] - phi[c] );
              Sup = phi[cxp];
              Sdn = phi[c]; 

              if ( areaFraction[cxpp][dim] < tiny || areaFraction[cxp][dim] < tiny ){
                r = 0.0; 
              }
            }

            psi = _limiter_function->get_psi( r ); 

            face_values.plus = Sup + 0.5*psi*( Sdn - Sup );
          }

          return face_values; 
        };


        private: 

          LimiterFunctionBase* _limiter_function; 

      }; 

      

      // ---------------------------------------------------------------------------
      // Old Super Bee interpolator
      //
      /** @brief Old Super Bee Interpolation with upwinding at boundaries */ 
      template <typename phiT>
      class OldSuperBeeInterpolation { 

        public: 

        OldSuperBeeInterpolation(){};
        ~OldSuperBeeInterpolation(){}; 

        FaceData1D inline no_bc( const IntVector c, const IntVector coord, phiT& phi, 
            FaceData1D vel, constCCVariable<Vector>& areaFraction ) 
        { 
          FaceData1D face_values;
          face_values.plus  = 0.0;
          face_values.minus = 0.0;

          double r=0.0; 
          double psi; 
          double Sup;
          double Sdn;
          const double tiny = 1.0e-16; 

          IntVector cxp  = c + coord; 
          IntVector cxpp = c + coord + coord; 
          IntVector cxm  = c - coord; 
          IntVector cxmm = c - coord - coord; 

          int dim = 0; 
          if (coord[0] == 1)
            dim =0; 
          else if (coord[1] == 1)
            dim = 1; 
          else 
            dim = 2; 

          // - FACE
          if ( vel.minus > 0.0 ) {
            Sup = phi[cxm];
            Sdn = phi[c];
            r = ( phi[cxm] - phi[cxmm] ) / ( phi[c] - phi[cxm] );

            if ( areaFraction[cxm][dim] < tiny || areaFraction[c][dim] < tiny )
              r = 0.0;

          } else if ( vel.minus < 0.0 ) {
            Sup = phi[c];
            Sdn = phi[cxm];
            r = ( phi[cxp] - phi[c] ) / ( phi[c] - phi[cxm] );

            if ( areaFraction[cxp][dim] < tiny || areaFraction[c][dim] < tiny )
              r = 0.0;

          } else { 
            Sup = 0.0;
            Sdn = 0.0; 
            psi = 0.0;
          }
          psi = std::max( std::min(2.0*r, 1.0), std::min(r, 2.0) );
          psi = std::max( 0.0, psi );

          face_values.minus = Sup + 0.5*psi*( Sdn - Sup );

          // + FACE
          if ( vel.plus > 0.0 ) {
            r = ( phi[c] - phi[cxm] ) / ( phi[cxp] - phi[c] );
            Sup = phi[c];
            Sdn = phi[cxp];

            if ( areaFraction[c][dim] < tiny || areaFraction[cxp][dim] < tiny )
              r = 0.0;

          } else if ( vel.plus < 0.0 ) {
            r = ( phi[cxpp] - phi[cxp] ) / ( phi[cxp] - phi[c] );
            Sup = phi[cxp];
            Sdn = phi[c]; 

            if ( areaFraction[cxpp][dim] < tiny || areaFraction[cxp][dim] < tiny )
              r = 0.0; 

          } else {
            Sup = 0.0;
            Sdn = 0.0; 
            psi = 0.0;
          }
          psi = std::max( std::min(2.0*r, 1.0), std::min(r, 2.0) );
          psi = std::max( 0.0, psi );

          face_values.plus = Sup + 0.5*psi*( Sdn - Sup );

          return face_values; 
        }; 

        FaceData1D inline with_bc( const IntVector c, const IntVector coord, phiT& phi, 
            FaceData1D vel, constCCVariable<Vector>& areaFraction, FaceBoundaryBool isBoundary ) 
        { 
          FaceData1D face_values;
          face_values.plus  = 0.0;
          face_values.minus = 0.0;

          double r = 0; 
          double psi; 
          double Sup;
          double Sdn;
          const double tiny = 1.0e-16; 

          IntVector cxp  = c + coord; 
          IntVector cxpp = c + coord + coord; 
          IntVector cxm  = c - coord; 
          IntVector cxmm = c - coord - coord; 

          int dim = 0; 
          if (coord[0] == 1)
            dim =0; 
          else if (coord[1] == 1)
            dim = 1; 
          else 
            dim = 2; 

          // - FACE
          if (isBoundary.minus) {
            if ( vel.minus > 0.0 ) { 
              Sup = ( phi[cxm] + phi[c] ) / 2.0; 
            } else { 
              Sup = phi[c]; 
            }
            face_values.minus = Sup; 
          } else { 
            if ( vel.minus > 0.0 ) {
              Sup = phi[cxm];
              Sdn = phi[c];
              r = ( phi[cxm] - phi[cxmm] ) / ( phi[c] - phi[cxm] ); 

            if ( areaFraction[cxm][dim] < tiny || areaFraction[c][dim] < tiny )
              r = 0.0;

            } else if ( vel.minus < 0.0 ) {
              Sup = phi[c];
              Sdn = phi[cxm];
              r = ( phi[cxp] - phi[c] ) / ( phi[c] - phi[cxm] );

            if ( areaFraction[cxp][dim] < tiny || areaFraction[c][dim] < tiny )
              r = 0.0;

            } else { 
              Sup = 0.0;
              Sdn = 0.0; 
              psi = 0.0;
            }
            psi = std::max( std::min(2.0*r, 1.0), std::min(r, 2.0) );
            psi = std::max( 0.0, psi );

            face_values.minus = Sup + 0.5*psi*( Sdn - Sup );
          }

          // + FACE
          if (isBoundary.plus) {
            if ( vel.plus > 0.0 ) { 
              Sup = phi[c]; 
            } else { 
              Sup = ( phi[cxp] + phi[c] ) / 2.0;
            }
            face_values.plus = Sup; 
          } else { 
            if ( vel.plus > 0.0 ) {
              r = ( phi[c] - phi[cxm] ) / ( phi[cxp] - phi[c] );
              Sup = phi[c];
              Sdn = phi[cxp];

            if ( areaFraction[c][dim] < tiny || areaFraction[cxp][dim] < tiny )
              r = 0.0;

            } else if ( vel.plus < 0.0 ) {
              r = ( phi[cxpp] - phi[cxp] ) / ( phi[cxp] - phi[c] );
              Sup = phi[cxp];
              Sdn = phi[c]; 

            if ( areaFraction[cxpp][dim] < tiny || areaFraction[cxp][dim] < tiny )
              r = 0.0; 

            } else {
              Sup = 0.0;
              Sdn = 0.0; 
              psi = 0.0;
            }
            psi = std::max( std::min(2.0*r, 1.0), std::min(r, 2.0) );
            psi = std::max( 0.0, psi );

            face_values.plus = Sup + 0.5*psi*( Sdn - Sup );
          }

          return face_values; 
        };
      }; 


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
        uT& uVel, vT& vVel, wT& wVel, 
        constCCVariable<double>& den, constCCVariable<Vector>& areaFraction, 
        std::string convScheme ) 
    {

      if (convScheme == "upwind") { 

       UpwindInterpolation<oldPhiT>* the_interpolant = scinew UpwindInterpolation<oldPhiT>(); 
       ConvHelper1<UpwindInterpolation<oldPhiT>, oldPhiT>* convection_helper = 
         scinew ConvHelper1<UpwindInterpolation<oldPhiT>, oldPhiT>(the_interpolant, oldPhi);

       convection_helper->do_convection( p, Fconv, uVel, vVel, wVel, den, areaFraction, this ); 

       delete convection_helper; 
       delete the_interpolant; 

      } else if ( convScheme == "super_bee" || convScheme == "roe_minmod" || convScheme == "vanleer" ) { 

       FluxLimiterInterpolation<oldPhiT>* the_interpolant = scinew FluxLimiterInterpolation<oldPhiT>( convScheme ); 
       ConvHelper1<FluxLimiterInterpolation<oldPhiT>, oldPhiT>* convection_helper = 
         scinew ConvHelper1<FluxLimiterInterpolation<oldPhiT>, oldPhiT>(the_interpolant, oldPhi);

       convection_helper->do_convection( p, Fconv, uVel, vVel, wVel, den, areaFraction, this ); 

       delete convection_helper; 
       delete the_interpolant; 

      } else if (convScheme == "old_super_bee") { 

       OldSuperBeeInterpolation<oldPhiT>* the_interpolant = scinew OldSuperBeeInterpolation<oldPhiT>(); 
       ConvHelper1<OldSuperBeeInterpolation<oldPhiT>, oldPhiT>* convection_helper = 
         scinew ConvHelper1<OldSuperBeeInterpolation<oldPhiT>, oldPhiT>(the_interpolant, oldPhi);

       convection_helper->do_convection( p, Fconv, uVel, vVel, wVel, den, areaFraction, this ); 

       delete convection_helper; 
       delete the_interpolant; 

      } else if (convScheme == "central") {

       CentralInterpolation<oldPhiT>* the_interpolant = scinew CentralInterpolation<oldPhiT>(); 
       ConvHelper1<CentralInterpolation<oldPhiT>, oldPhiT>* convection_helper = 
         scinew ConvHelper1<CentralInterpolation<oldPhiT>, oldPhiT>(the_interpolant, oldPhi);

       convection_helper->do_convection( p, Fconv, uVel, vVel, wVel, den, areaFraction, this ); 

       delete convection_helper; 
       delete the_interpolant; 

      } else { 

        throw InvalidValue("Error: Convection scheme not recognized. Check UPS file and try again.", __FILE__, __LINE__);

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

      if (convScheme == "upwind") { 

       UpwindInterpolation<oldPhiT>* the_interpolant = scinew UpwindInterpolation<oldPhiT>(); 
       ConvHelper1<UpwindInterpolation<oldPhiT>, oldPhiT>* convection_helper = 
         scinew ConvHelper1<UpwindInterpolation<oldPhiT>, oldPhiT>(the_interpolant, oldPhi);

       convection_helper->do_convection( p, Fconv, uVel, vVel, wVel, areaFraction, this ); 

       delete convection_helper; 
       delete the_interpolant; 

      } else if ( convScheme == "super_bee" || convScheme == "roe_minmod" || convScheme == "vanleer" ) { 

       FluxLimiterInterpolation<oldPhiT>* the_interpolant = scinew FluxLimiterInterpolation<oldPhiT>( convScheme ); 
       ConvHelper1<FluxLimiterInterpolation<oldPhiT>, oldPhiT>* convection_helper = 
         scinew ConvHelper1<FluxLimiterInterpolation<oldPhiT>, oldPhiT>(the_interpolant, oldPhi);

       convection_helper->do_convection( p, Fconv, uVel, vVel, wVel, areaFraction, this ); 

       delete convection_helper; 
       delete the_interpolant; 

      } else if (convScheme == "central") {

       CentralInterpolation<oldPhiT>* the_interpolant = scinew CentralInterpolation<oldPhiT>(); 
       ConvHelper1<CentralInterpolation<oldPhiT>, oldPhiT>* convection_helper = 
         scinew ConvHelper1<CentralInterpolation<oldPhiT>, oldPhiT>(the_interpolant, oldPhi);

       convection_helper->do_convection( p, Fconv, uVel, vVel, wVel, areaFraction, this ); 

       delete convection_helper; 
       delete the_interpolant; 

      } else { 

        throw InvalidValue("Error: Convection scheme not recognized. Check UPS file and try again.", __FILE__, __LINE__);

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
      if (convScheme == "upwind") { 

       UpwindInterpolation<oldPhiT>* the_interpolant = scinew UpwindInterpolation<oldPhiT>(); 
       ConvHelper1<UpwindInterpolation<oldPhiT>, oldPhiT>* convection_helper = 
         scinew ConvHelper1<UpwindInterpolation<oldPhiT>, oldPhiT>(the_interpolant, oldPhi);

       convection_helper->do_convection( p, Fconv, uVel, vVel, wVel, partVel, areaFraction, this ); 

       delete convection_helper; 
       delete the_interpolant; 

      } else if ( convScheme == "super_bee" || convScheme == "roe_minmod" || convScheme == "vanleer" ) { 

       FluxLimiterInterpolation<oldPhiT>* the_interpolant = scinew FluxLimiterInterpolation<oldPhiT>( convScheme ); 
       ConvHelper1<FluxLimiterInterpolation<oldPhiT>, oldPhiT>* convection_helper = 
         scinew ConvHelper1<FluxLimiterInterpolation<oldPhiT>, oldPhiT>(the_interpolant, oldPhi);
 
       convection_helper->do_convection( p, Fconv, uVel, vVel, wVel, partVel,  areaFraction, this );

       delete convection_helper; 
       delete the_interpolant; 

      } else if (convScheme == "central") {

       CentralInterpolation<oldPhiT>* the_interpolant = scinew CentralInterpolation<oldPhiT>(); 
       ConvHelper1<CentralInterpolation<oldPhiT>, oldPhiT>* convection_helper = 
         scinew ConvHelper1<CentralInterpolation<oldPhiT>, oldPhiT>(the_interpolant, oldPhi);

       convection_helper->do_convection( p, Fconv, uVel, vVel, wVel, partVel, areaFraction, this ); 

       delete convection_helper; 
       delete the_interpolant; 

      } else { 

        throw InvalidValue("Error: Convection scheme not recognized. Check UPS file and try again.", __FILE__, __LINE__);

      }
    }

  //========================= Diffusion ======================================

  //---------------------------------------------------------------------------
  // Method: Compute the diffusion term
  // Simple diffusion term: \f$ \int_{S} \nabla \phi \cdot dS \f$ 
  //---------------------------------------------------------------------------
  template <class fT, class oldPhiT, class gammaT> void 
    Discretization_new::computeDiff( const Patch* p, fT& Fdiff, oldPhiT& oldPhi, gammaT& gamma,
        constCCVariable<Vector>& areaFraction, constCCVariable<double>& prNo, int mat_id, string varName )
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

        double pr_no = prNo[c]; 

        Fdiff[c] = 1.0/pr_no * Dx.y()*Dx.z() * 
                   ( face_gamma.plus * grad_phi.plus * cp_af.x() - 
                     face_gamma.minus * grad_phi.minus * c_af.x() ); 

#ifdef YDIM
        coord[0] = 0; coord[1] = 1; coord[2] = 0; 
        double dy = Dx.y(); 

        face_gamma = centralInterp( c, coord, gamma ); 
        grad_phi   = gradPtoF( c, oldPhi, dy, coord ); 

        cp_af = areaFraction[c + coord]; 

        Fdiff[c] += 1.0/pr_no * Dx.x()*Dx.z() *  
                   ( face_gamma.plus * grad_phi.plus * cp_af.y() - 
                     face_gamma.minus * grad_phi.minus * c_af.y() ); 
#endif
#ifdef ZDIM
        coord[0] = 0; coord[1] = 0; coord[2] = 1; 
        double dz = Dx.z(); 

        face_gamma = centralInterp( c, coord, gamma ); 
        grad_phi   = gradPtoF( c, oldPhi, dz, coord ); 

        cp_af = areaFraction[c + coord]; 

        Fdiff[c] += 1.0/pr_no * Dx.x()*Dx.y() * 
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
          double bc_value = 0.0; 
          Iterator bound_ptr; 
          Iterator nu; //not used...who knows why?
          const BoundCondBase* bc = p->getArrayBCValues( face, mat_id, 
                                                         varName, bound_ptr, 
                                                         nu, child ); 
          const BoundCond<double> *new_bcs = dynamic_cast<const BoundCond<double> *>(bc); 
          if (new_bcs != 0) {
            bc_kind = new_bcs->getBCType__NEW(); 
            bc_value = new_bcs->getValue();
          } else {
            std::cout << "Warning!  Boundary condition not set for: " << std::endl
                      << "variable = " << varName << std::endl
                      << "face = " << face << std::endl;
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
                  IntVector c = *bound_ptr; 

                  face_gamma = centralInterp( bp1, coord, gamma ); 
                  grad_phi   = gradPtoF( bp1, oldPhi, dx, coord ); 

                  Vector c_af = areaFraction[bp1]; 
                  Vector cp_af = areaFraction[bp1 + coord]; 

                  Fdiff[bp1] += 1.0/prNo[bp1] * Dx.y()*Dx.z() * 
                             ( face_gamma.minus * grad_phi.minus * c_af.x() ); 

                  // to match with the old code...
                  Fdiff[bp1] -= 1.0/prNo[bp1] * Dx.y()*Dx.z() * 
                            ( face_gamma.minus * ( oldPhi[bp1] - bc_value )/Dx.x() * c_af.x() ); 

                }
                break; 
              case Patch::xplus:

                coord[0] = 1; coord[1] = 0; coord[2] = 0;
                dx = Dx.x(); 

                for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++){

                  IntVector bp1(*bound_ptr - insideCellDir); 
                  IntVector c = *bound_ptr; 
                  face_gamma = centralInterp( bp1, coord, gamma ); 
                  grad_phi   = gradPtoF( bp1, oldPhi, dx, coord ); 

                  Vector c_af = areaFraction[bp1]; 
                  Vector cp_af = areaFraction[bp1 + coord]; 

                  Fdiff[bp1] -= 1.0/prNo[bp1] * Dx.y()*Dx.z() * 
                             ( face_gamma.plus * grad_phi.plus * cp_af.x() ); 

                  // to match with the old code...
                  Fdiff[bp1] += 1.0/prNo[bp1] * Dx.y()*Dx.z() * 
                             ( face_gamma.plus * (bc_value - oldPhi[bp1])/Dx.x() * cp_af.x() ); 
                }
                break; 
#ifdef YDIM
              case Patch::yminus:

                coord[0] = 0; coord[1] = 1; coord[2] = 0;
                dx = Dx.y(); 

                for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++){

                  IntVector bp1(*bound_ptr - insideCellDir); 
                  IntVector c = *bound_ptr; 
                  face_gamma = centralInterp( bp1, coord, gamma ); 
                  grad_phi   = gradPtoF( bp1, oldPhi, dx, coord ); 

                  Vector c_af = areaFraction[bp1]; 
                  Vector cp_af = areaFraction[bp1 + coord]; 

                  Fdiff[bp1] += 1.0/prNo[bp1] * Dx.x()*Dx.z() * 
                             ( face_gamma.minus * grad_phi.minus * c_af.y() ); 

                  // to match with the old code...
                  Fdiff[bp1] -= 1.0/prNo[bp1] * Dx.x()*Dx.z() * 
                            ( face_gamma.minus * ( oldPhi[bp1] - bc_value )/Dx.y() * c_af.y() ); 
                }
                break; 
              case Patch::yplus:

                coord[0] = 0; coord[1] = 1; coord[2] = 0;
                dx = Dx.y();

                for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++){

                  IntVector bp1(*bound_ptr - insideCellDir); 
                  IntVector c = *bound_ptr; 
                  face_gamma = centralInterp( bp1, coord, gamma ); 
                  grad_phi   = gradPtoF( bp1, oldPhi, dx, coord ); 

                  Vector c_af = areaFraction[bp1]; 
                  Vector cp_af = areaFraction[bp1 + coord]; 

                  Fdiff[bp1] -= 1.0/prNo[bp1] * Dx.x()*Dx.z() * 
                             ( face_gamma.plus * grad_phi.plus * cp_af.y() ); 

                  // to match with the old code...
                  Fdiff[bp1] += 1.0/prNo[bp1] * Dx.x()*Dx.z() * 
                             ( face_gamma.plus * (bc_value - oldPhi[bp1])/Dx.y() * cp_af.y() ); 
                }
                break;
#endif 
#ifdef ZDIM
              case Patch::zminus:

                coord[0] = 0; coord[1] = 0; coord[2] = 1;
                dx = Dx.z(); 

                for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++){

                  IntVector bp1(*bound_ptr - insideCellDir); 
                  IntVector c = *bound_ptr; 
                  face_gamma = centralInterp( bp1, coord, gamma ); 
                  grad_phi   = gradPtoF( bp1, oldPhi, dx, coord ); 

                  Vector c_af = areaFraction[bp1]; 
                  Vector cp_af = areaFraction[bp1 + coord]; 

                  Fdiff[bp1] += 1.0/prNo[bp1] * Dx.x()*Dx.y() * 
                             ( face_gamma.minus * grad_phi.minus * c_af.z() ); 

                  // to match with the old code...
                  Fdiff[bp1] -= 1.0/prNo[bp1] * Dx.x()*Dx.y() * 
                            ( face_gamma.minus * ( oldPhi[bp1] - bc_value )/Dx.z() * c_af.z() ); 

                }
                break; 
              case Patch::zplus:

                coord[0] = 0; coord[1] = 0; coord[2] = 1;
                dx = Dx.z(); 

                for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++){

                  IntVector bp1(*bound_ptr - insideCellDir); 
                  IntVector c = *bound_ptr; 
                  face_gamma = centralInterp( bp1, coord, gamma ); 
                  grad_phi   = gradPtoF( bp1, oldPhi, dx, coord ); 

                  Vector c_af = areaFraction[bp1]; 
                  Vector cp_af = areaFraction[bp1 + coord]; 

                  Fdiff[bp1] -= 1.0/prNo[bp1] * Dx.x()*Dx.y() * 
                             ( face_gamma.plus * grad_phi.plus * cp_af.z() ); 

                  // to match with the old code...
                  Fdiff[bp1] += 1.0/prNo[bp1] * Dx.x()*Dx.y() * 
                             ( face_gamma.plus * (bc_value - oldPhi[bp1])/Dx.z() * cp_af.z() ); 
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


  //---------------------------------------------------------------------------
  // Method: Compute the diffusion term
  // Simple diffusion term: \f$ \int_{S} \nabla \phi \cdot dS \f$
  //---------------------------------------------------------------------------
  template <class fT, class oldPhiT, class gammaT> void 
    Discretization_new::computeDiff( const Patch* p, fT& Fdiff, oldPhiT& oldPhi, gammaT& gamma,
        constCCVariable<Vector>& areaFraction, double const_prNo , int mat_id, string varName )
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

        Fdiff[c] = 1.0/const_prNo * Dx.y()*Dx.z() * 
                   ( face_gamma.plus * grad_phi.plus * cp_af.x() - 
                     face_gamma.minus * grad_phi.minus * c_af.x() ); 

#ifdef YDIM
        coord[0] = 0; coord[1] = 1; coord[2] = 0; 
        double dy = Dx.y(); 

        face_gamma = centralInterp( c, coord, gamma ); 
        grad_phi   = gradPtoF( c, oldPhi, dy, coord ); 

        cp_af = areaFraction[c + coord]; 

        Fdiff[c] += 1.0/const_prNo * Dx.x()*Dx.z() *  
                   ( face_gamma.plus * grad_phi.plus * cp_af.y() - 
                     face_gamma.minus * grad_phi.minus * c_af.y() ); 
#endif
#ifdef ZDIM
        coord[0] = 0; coord[1] = 0; coord[2] = 1; 
        double dz = Dx.z(); 

        face_gamma = centralInterp( c, coord, gamma ); 
        grad_phi   = gradPtoF( c, oldPhi, dz, coord ); 

        cp_af = areaFraction[c + coord]; 

        Fdiff[c] += 1.0/const_prNo * Dx.x()*Dx.y() * 
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
            std::cout << "Warning!  Boundary condition not set for: " << std::endl
                      << "variable = " << varName << std::endl
                      << "face = " << face << std::endl;
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

                  Fdiff[bp1] += 1.0/const_prNo * Dx.y()*Dx.z() * 
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

                  Fdiff[bp1] -= 1.0/const_prNo * Dx.y()*Dx.z() * 
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

                  Fdiff[bp1] += 1.0/const_prNo * Dx.x()*Dx.z() * 
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

                  Fdiff[bp1] -= 1.0/const_prNo * Dx.x()*Dx.z() * 
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

                  Fdiff[bp1] += 1.0/const_prNo * Dx.x()*Dx.y() * 
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

                  Fdiff[bp1] -= 1.0/const_prNo * Dx.x()*Dx.y() * 
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

// Code attic -- clean out after regression tests pass: 
//
//
//      /** @brief Minmod Interpolation -- should work for all cell types.
//       *      This function does not have boundary checking (for speed). */
//      template< class phiT >
//        inline FaceData1D minmodInterp( const IntVector c, const IntVector coord, phiT& phi,
//            FaceData1D vel, constCCVariable<Vector>& areaFraction ) {
//
//          FaceData1D face_values;
//          face_values.plus  = 0.0;
//          face_values.minus = 0.0;
//
//          double r=0.;
//          double psi;
//          double Sup;
//          double Sdn;
//          const double tiny = 1.0e-16;
//
//          IntVector cxp  = c + coord;
//          IntVector cxpp = c + coord + coord;
//          IntVector cxm  = c - coord;
//          IntVector cxmm = c - coord - coord;
//
//          int dim = 0;
//          if (coord[0] == 1)
//            dim =0;
//          else if (coord[1] == 1)
//            dim = 1;
//          else
//            dim = 2;
//
//          // - FACE
//          if ( vel.minus > 0.0 ) {
//            Sup = phi[cxm];
//            Sdn = phi[c];
//            r = ( phi[cxm] - phi[cxmm] ) / ( phi[c] - phi[cxm] );
//
//            if ( areaFraction[cxm][dim] < tiny || areaFraction[c][dim] < tiny )
//              r = 0.0;
//
//          } else if ( vel.minus < 0.0 ) {
//            Sup = phi[c];
//            Sdn = phi[cxm];
//            r = ( phi[cxp] - phi[c] ) / ( phi[c] - phi[cxm] );
//
//            if ( areaFraction[cxp][dim] < tiny || areaFraction[c][dim] < tiny )
//              r = 0.0;
//
//          } else {
//            Sup = 0.0;
//            Sdn = 0.0;
//            psi = 0.0;
//          }
//          psi = std::min(r, 1.0);
//          psi = std::max( 0.0, psi );
//
//          face_values.minus = Sup + 0.5*psi*( Sdn - Sup );
//
//          // + FACE
//          if ( vel.plus > 0.0 ) {
//            r = ( phi[c] - phi[cxm] ) / ( phi[cxp] - phi[c] );
//            Sup = phi[c];
//            Sdn = phi[cxp];
//
//            if ( areaFraction[c][dim] < tiny || areaFraction[cxp][dim] < tiny )
//              r = 0.0;
//
//          } else if ( vel.plus < 0.0 ) {
//            r = ( phi[cxpp] - phi[cxp] ) / ( phi[cxp] - phi[c] );
//            Sup = phi[cxp];
//            Sdn = phi[c];
//
//            if ( areaFraction[cxpp][dim] < tiny || areaFraction[cxp][dim] < tiny )
//              r = 0.0;
//
//          } else {
//            Sup = 0.0;
//            Sdn = 0.0;
//            psi = 0.0;
//          }
//          psi = std::min(r, 1.0);
//          psi = std::max( 0.0, psi );
//
//          face_values.plus = Sup + 0.5*psi*( Sdn - Sup );
//
//          return face_values;
//        }
//
//      /** @brief Minmod Interpolation -- should work for all cell types. 
//       *       This function includes boundary checking (slower).  */
//      template< class phiT >
//        inline FaceData1D minmodInterp( const IntVector c, const IntVector coord, phiT& phi,
//            FaceData1D vel, FaceBoundaryBool isBoundary, constCCVariable<Vector>& areaFraction ) {
//
//          FaceData1D face_values;
//          face_values.plus  = 0.0;
//          face_values.minus = 0.0;
//
//          double r = 0;
//          double psi;
//          double Sup;
//          double Sdn;
//          const double tiny = 1.0e-16;
//
//          IntVector cxp  = c + coord;
//          IntVector cxpp = c + coord + coord;
//          IntVector cxm  = c - coord;
//          IntVector cxmm = c - coord - coord;
//
//          int dim = 0;
//          if (coord[0] == 1)
//            dim =0;
//          else if (coord[1] == 1)
//            dim = 1;
//          else
//            dim = 2;
//
//          // - FACE
//          if (isBoundary.minus)
//            face_values.minus = 0.5*(phi[c]+phi[cxm]);
//          else {
//            if ( vel.minus > 0.0 ) {
//              Sup = phi[cxm];
//              Sdn = phi[c];
//              r = ( phi[cxm] - phi[cxmm] ) / ( phi[c] - phi[cxm] );
//
//            if ( areaFraction[cxm][dim] < tiny || areaFraction[c][dim] < tiny )
//              r = 0.0;
//
//            } else if ( vel.minus < 0.0 ) {
//              Sup = phi[c];
//              Sdn = phi[cxm];
//              r = ( phi[cxp] - phi[c] ) / ( phi[c] - phi[cxm] );
//
//            if ( areaFraction[cxp][dim] < tiny || areaFraction[c][dim] < tiny )
//              r = 0.0;
//
//            } else {
//              Sup = 0.0;
//              Sdn = 0.0;
//              psi = 0.0;
//            }
//            psi = std::min(r, 1.0);
//            psi = std::max( 0.0, psi );
//
//            face_values.minus = Sup + 0.5*psi*( Sdn - Sup );
//          }
//
//          // + FACE
//          if (isBoundary.plus)
//            face_values.plus = 0.5*(phi[c] + phi[cxp]);
//          else {
//            if ( vel.plus > 0.0 ) {
//              r = ( phi[c] - phi[cxm] ) / ( phi[cxp] - phi[c] );
//              Sup = phi[c];
//              Sdn = phi[cxp];
//
//            if ( areaFraction[c][dim] < tiny || areaFraction[cxp][dim] < tiny )
//              r = 0.0;
//
//            } else if ( vel.plus < 0.0 ) {
//              r = ( phi[cxpp] - phi[cxp] ) / ( phi[cxp] - phi[c] );
//              Sup = phi[cxp];
//              Sdn = phi[c];
//
//            if ( areaFraction[cxpp][dim] < tiny || areaFraction[cxp][dim] < tiny )
//              r = 0.0;
//
//            } else {
//              Sup = 0.0;
//              Sdn = 0.0;
//              psi = 0.0;
//            }
//            psi = std::min(r, 1.0);
//            psi = std::max( 0.0, psi );
//
//            face_values.plus = Sup + 0.5*psi*( Sdn - Sup );
//          }
//
//          return face_values;
//        }
//
//
////      /** @brief Super Bee Interpolation -- should work for all cell types.
////       *      This function does not have boundary checking (for speed). */
////      template< class phiT >
////        inline FaceData1D superBeeInterp( const IntVector c, const IntVector coord, phiT& phi, 
////            FaceData1D vel, constCCVariable<Vector>& areaFraction ) {
////
////          FaceData1D face_values;
////          face_values.plus  = 0.0;
////          face_values.minus = 0.0;
////
////          double r=0.; 
////          double psi; 
////          double Sup;
////          double Sdn;
////          const double tiny = 1.0e-16; 
////
////          IntVector cxp  = c + coord; 
////          IntVector cxpp = c + coord + coord; 
////          IntVector cxm  = c - coord; 
////          IntVector cxmm = c - coord - coord; 
////
////          int dim = 0; 
////          if (coord[0] == 1)
////            dim =0; 
////          else if (coord[1] == 1)
////            dim = 1; 
////          else 
////            dim = 2; 
////
////          // - FACE
////          if ( vel.minus > 0.0 ) {
////            Sup = phi[cxm];
////            Sdn = phi[c];
////            r = ( phi[cxm] - phi[cxmm] ) / ( phi[c] - phi[cxm] );
////
////            if ( areaFraction[cxm][dim] < tiny || areaFraction[c][dim] < tiny )
////              r = 0.0;
////
////          } else if ( vel.minus < 0.0 ) {
////            Sup = phi[c];
////            Sdn = phi[cxm];
////            r = ( phi[cxp] - phi[c] ) / ( phi[c] - phi[cxm] );
////
////            if ( areaFraction[cxp][dim] < tiny || areaFraction[c][dim] < tiny )
////              r = 0.0;
////
////          } else { 
////            Sup = 0.0;
////            Sdn = 0.0; 
////            psi = 0.0;
////          }
////          psi = std::max( std::min(2.0*r, 1.0), std::min(r, 2.0) );
////          psi = std::max( 0.0, psi );
////
////          face_values.minus = Sup + 0.5*psi*( Sdn - Sup );
////
////          // + FACE
////          if ( vel.plus > 0.0 ) {
////            r = ( phi[c] - phi[cxm] ) / ( phi[cxp] - phi[c] );
////            Sup = phi[c];
////            Sdn = phi[cxp];
////
////            if ( areaFraction[c][dim] < tiny || areaFraction[cxp][dim] < tiny )
////              r = 0.0;
////
////          } else if ( vel.plus < 0.0 ) {
////            r = ( phi[cxpp] - phi[cxp] ) / ( phi[cxp] - phi[c] );
////            Sup = phi[cxp];
////            Sdn = phi[c]; 
////
////            if ( areaFraction[cxpp][dim] < tiny || areaFraction[cxp][dim] < tiny )
////              r = 0.0; 
////
////          } else {
////            Sup = 0.0;
////            Sdn = 0.0; 
////            psi = 0.0;
////          }
////          psi = std::max( std::min(2.0*r, 1.0), std::min(r, 2.0) );
////          psi = std::max( 0.0, psi );
////
////          face_values.plus = Sup + 0.5*psi*( Sdn - Sup );
////
////          return face_values; 
////        }
////
////      /** @brief Super Bee Interpolation -- should work for all cell types. 
////       *       This function includes boundary checking (slower).  */ 
////      template< class phiT >
////        inline FaceData1D superBeeInterp( const IntVector c, const IntVector coord, phiT& phi, 
////            FaceData1D vel, FaceBoundaryBool isBoundary, constCCVariable<Vector>& areaFraction ) {
////
////          FaceData1D face_values;
////          face_values.plus  = 0.0;
////          face_values.minus = 0.0;
////
////          double r = 0; 
////          double psi; 
////          double Sup;
////          double Sdn;
////          const double tiny = 1.0e-16; 
////
////          IntVector cxp  = c + coord; 
////          IntVector cxpp = c + coord + coord; 
////          IntVector cxm  = c - coord; 
////          IntVector cxmm = c - coord - coord; 
////
////          int dim = 0; 
////          if (coord[0] == 1)
////            dim =0; 
////          else if (coord[1] == 1)
////            dim = 1; 
////          else 
////            dim = 2; 
////
////          // - FACE
////          if (isBoundary.minus) 
////            face_values.minus = 0.5*(phi[c]+phi[cxm]);
////          else { 
////            if ( vel.minus > 0.0 ) {
////              Sup = phi[cxm];
////              Sdn = phi[c];
////              r = ( phi[cxm] - phi[cxmm] ) / ( phi[c] - phi[cxm] ); 
////
////            if ( areaFraction[cxm][dim] < tiny || areaFraction[c][dim] < tiny )
////              r = 0.0;
////
////            } else if ( vel.minus < 0.0 ) {
////              Sup = phi[c];
////              Sdn = phi[cxm];
////              r = ( phi[cxp] - phi[c] ) / ( phi[c] - phi[cxm] );
////
////            if ( areaFraction[cxp][dim] < tiny || areaFraction[c][dim] < tiny )
////              r = 0.0;
////
////            } else { 
////              Sup = 0.0;
////              Sdn = 0.0; 
////              psi = 0.0;
////            }
////            psi = std::max( std::min(2.0*r, 1.0), std::min(r, 2.0) );
////            psi = std::max( 0.0, psi );
////
////            face_values.minus = Sup + 0.5*psi*( Sdn - Sup );
////          }
////
////          // + FACE
////          if (isBoundary.plus)
////            face_values.plus = 0.5*(phi[c] + phi[cxp]);
////          else { 
////            if ( vel.plus > 0.0 ) {
////              r = ( phi[c] - phi[cxm] ) / ( phi[cxp] - phi[c] );
////              Sup = phi[c];
////              Sdn = phi[cxp];
////
////            if ( areaFraction[c][dim] < tiny || areaFraction[cxp][dim] < tiny )
////              r = 0.0;
////
////            } else if ( vel.plus < 0.0 ) {
////              r = ( phi[cxpp] - phi[cxp] ) / ( phi[cxp] - phi[c] );
////              Sup = phi[cxp];
////              Sdn = phi[c]; 
////
////            if ( areaFraction[cxpp][dim] < tiny || areaFraction[cxp][dim] < tiny )
////              r = 0.0; 
////
////            } else {
////              Sup = 0.0;
////              Sdn = 0.0; 
////              psi = 0.0;
////            }
////            psi = std::max( std::min(2.0*r, 1.0), std::min(r, 2.0) );
////            psi = std::max( 0.0, psi );
////
////            face_values.plus = Sup + 0.5*psi*( Sdn - Sup );
////          }
////
////          return face_values; 
////        }
//
//      /* @brief Upwind interpolation -- should work for all data types. 
//       *      This function does not have boundary checking (for speed). */ 
//      template< class phiT >
//        inline FaceData1D upwindInterp( const IntVector c, const IntVector coord, phiT& phi, 
//            FaceData1D vel) {
//
//          Discretization_new::FaceData1D face_values; 
//          face_values.minus = 0.0;
//          face_values.plus = 0.0;
//
//          IntVector cxp = c + coord; 
//          IntVector cxm = c - coord; 
//
//          // - FACE 
//          if ( vel.minus > 0.0 )
//            face_values.minus = phi[cxm];
//          else if ( vel.minus <= 0.0 )
//            face_values.minus = phi[c]; 
//
//          // + FACE 
//          if ( vel.plus >= 0.0 )
//            face_values.plus = phi[c]; 
//          else if ( vel.plus < 0.0 )
//            face_values.plus = phi[cxp]; 
//
//          return face_values; 
//
//        }
//
//      /* @brief Upwind interpolation -- should work for all data types. 
//       *      This function includes boundary checking (slower). */ 
//      template< class phiT >
//        inline FaceData1D upwindInterp( const IntVector c, const IntVector coord, phiT& phi, 
//            FaceData1D vel, FaceBoundaryBool isBoundary ) {
//
//          Discretization_new::FaceData1D face_values; 
//          face_values.minus = 0.0;
//          face_values.plus = 0.0;
//
//          IntVector cxp = c + coord; 
//          IntVector cxm = c - coord; 
//
//          // - FACE 
//          if (isBoundary.minus)
//            face_values.minus = 0.5*(phi[c] + phi[cxm]);
//          else {
//            if ( vel.minus > 0.0 )
//              face_values.minus = phi[cxm];
//            else if ( vel.minus <= 0.0 )
//              face_values.minus = phi[c]; 
//          }
//
//          // + FACE 
//          if (isBoundary.plus)
//            face_values.plus = 0.5*(phi[c] + phi[cxp]);
//          else {
//            if ( vel.plus >= 0.0 )
//              face_values.plus = phi[c]; 
//            else if ( vel.plus < 0.0 )
//              face_values.plus = phi[cxp]; 
//          }
//
//          return face_values; 
//
//        }
//      // ---------------------------------------------------------------------------
//      // Minmod interpolator
//      //
//      /** @brief Minmod Interpolation */ 
//      template <typename phiT>
//      class MinmodInterpolation { 
//
//        public: 
//
//        MinmodInterpolation(){};
//        ~MinmodInterpolation(){}; 
//
//        FaceData1D inline no_bc( const IntVector c, const IntVector coord, phiT& phi, 
//            FaceData1D vel, constCCVariable<Vector>& areaFraction ) 
//        { 
//          FaceData1D face_values;
//          face_values.plus  = 0.0;
//          face_values.minus = 0.0;
//
//          double r=0.0; 
//          double psi; 
//          double Sup;
//          double Sdn;
//          const double tiny = 1.0e-16; 
//
//          IntVector cxp  = c + coord; 
//          IntVector cxpp = c + coord + coord; 
//          IntVector cxm  = c - coord; 
//          IntVector cxmm = c - coord - coord; 
//
//          int dim = 0; 
//          if (coord[0] == 1)
//            dim =0; 
//          else if (coord[1] == 1)
//            dim = 1; 
//          else 
//            dim = 2; 
//
//          // - FACE
//          if ( vel.minus > 0.0 ) {
//            Sup = phi[cxm];
//            Sdn = phi[c];
//            if(fabs(phi[c] - phi[cxm]) >  tiny){
//              r = ( phi[cxm] - phi[cxmm] ) / ( phi[c] - phi[cxm] );
//            } else {
//              r = 0.0;
//            }
//
//            if ( areaFraction[cxm][dim] < tiny || areaFraction[c][dim] < tiny )
//              r = 0.0;
//
//          } else if ( vel.minus < 0.0 ) {
//            Sup = phi[c];
//            Sdn = phi[cxm];
//            if(fabs(phi[c] - phi[cxm]) > tiny){
//              r = ( phi[cxp] - phi[c] ) / ( phi[c] - phi[cxm] );
//            } else {
//              r = 0.0;
//            }
//
//            if ( areaFraction[cxp][dim] < tiny || areaFraction[c][dim] < tiny )
//              r = 0.0;
//
//          } else { 
//            Sup = 0.0;
//            Sdn = 0.0; 
//            psi = 0.0;
//          }
//
//          psi = std::min(r, 1.0);
//          psi = std::max( 0.0, psi );
//
//          face_values.minus = Sup + 0.5*psi*( Sdn - Sup );
//
//          // + FACE
//          if ( vel.plus > 0.0 ) {
//            if(fabs(phi[cxp] - phi[c]) > tiny){
//              r = ( phi[c] - phi[cxm] ) / ( phi[cxp] - phi[c] );
//            } else {
//              r = 0.0;
//            }
//            Sup = phi[c];
//            Sdn = phi[cxp];
//
//            if ( areaFraction[c][dim] < tiny || areaFraction[cxp][dim] < tiny )
//              r = 0.0;
//
//          } else if ( vel.plus < 0.0 ) {
//            if(fabs(phi[cxp] - phi[c]) > tiny){
//              r = ( phi[cxpp] - phi[cxp] ) / ( phi[cxp] - phi[c] );
//            } else {
//              r = 0.0;
//            }
//            Sup = phi[cxp];
//            Sdn = phi[c]; 
//
//            if ( areaFraction[cxpp][dim] < tiny || areaFraction[cxp][dim] < tiny )
//              r = 0.0; 
//
//          } else {
//            Sup = 0.0;
//            Sdn = 0.0; 
//            psi = 0.0;
//          }
//
//          psi = std::min(r, 1.0);
//          psi = std::max( 0.0, psi );
//
//          face_values.plus = Sup + 0.5*psi*( Sdn - Sup );
//
//          return face_values; 
//        }; 
//
//        FaceData1D inline with_bc( const IntVector c, const IntVector coord, phiT& phi, 
//            FaceData1D vel, constCCVariable<Vector>& areaFraction, FaceBoundaryBool isBoundary ) 
//        { 
//          FaceData1D face_values;
//          face_values.plus  = 0.0;
//          face_values.minus = 0.0;
//
//          double r = 0; 
//          double psi; 
//          double Sup;
//          double Sdn;
//          const double tiny = 1.0e-16; 
//
//          IntVector cxp  = c + coord; 
//          IntVector cxpp = c + coord + coord; 
//          IntVector cxm  = c - coord; 
//          IntVector cxmm = c - coord - coord; 
//
//          int dim = 0; 
//          if (coord[0] == 1)
//            dim =0; 
//          else if (coord[1] == 1)
//            dim = 1; 
//          else 
//            dim = 2; 
//
//          // - FACE
//          if (isBoundary.minus) 
//            face_values.minus = 0.5*(phi[c]+phi[cxm]);
//          else { 
//            if ( vel.minus > 0.0 ) {
//              Sup = phi[cxm];
//              Sdn = phi[c];
//              if(fabs(phi[c] - phi[cxm]) > tiny){
//                r = ( phi[cxm] - phi[cxmm] ) / ( phi[c] - phi[cxm] );
//              } else {
//                r = 0.0;
//              } 
//
//            if ( areaFraction[cxm][dim] < tiny || areaFraction[c][dim] < tiny )
//              r = 0.0;
//
//            } else if ( vel.minus < 0.0 ) {
//              Sup = phi[c];
//              Sdn = phi[cxm];
//              if(fabs(phi[c] - phi[cxm]) > tiny){
//                r = ( phi[cxp] - phi[c] ) / ( phi[c] - phi[cxm] );
//              } else {
//                r = 0.0;
//              }
//
//            if ( areaFraction[cxp][dim] < tiny || areaFraction[c][dim] < tiny )
//              r = 0.0;
//
//            } else { 
//              Sup = 0.0;
//              Sdn = 0.0; 
//              psi = 0.0;
//            }
//
//            psi = std::min(r, 1.0);
//            psi = std::max( 0.0, psi );
//
//            face_values.minus = Sup + 0.5*psi*( Sdn - Sup );
//          }
//
//          // + FACE
//          if (isBoundary.plus)
//            face_values.plus = 0.5*(phi[c] + phi[cxp]);
//          else { 
//            if ( vel.plus > 0.0 ) {
//              if(fabs(phi[cxp] - phi[c]) > tiny){
//                r = ( phi[c] - phi[cxm] ) / ( phi[cxp] - phi[c] );
//              } else {
//                r = 0.0;
//              }
//              Sup = phi[c];
//              Sdn = phi[cxp];
//
//            if ( areaFraction[c][dim] < tiny || areaFraction[cxp][dim] < tiny )
//              r = 0.0;
//
//            } else if ( vel.plus < 0.0 ) {
//              if(fabs(phi[cxp] - phi[c]) > tiny){
//                r = ( phi[cxpp] - phi[cxp] ) / ( phi[cxp] - phi[c] );
//              } else {
//                r = 0.0;
//              }
//              Sup = phi[cxp];
//              Sdn = phi[c]; 
//
//            if ( areaFraction[cxpp][dim] < tiny || areaFraction[cxp][dim] < tiny )
//              r = 0.0; 
//
//            } else {
//              Sup = 0.0;
//              Sdn = 0.0; 
//              psi = 0.0;
//            }
//            psi = std::min(r, 1.0);
//            psi = std::max( 0.0, psi );
//
//            face_values.plus = Sup + 0.5*psi*( Sdn - Sup );
//          }
//
//          return face_values; 
//        };
//      }; 
//      
//      // ---------------------------------------------------------------------------
//      // Super Bee interpolator
//      //
//      /** @brief Super Bee Interpolation */ 
//      template <typename phiT>
//      class SuperBeeInterpolation { 
//
//        public: 
//
//        SuperBeeInterpolation(){};
//        ~SuperBeeInterpolation(){}; 
//
//        FaceData1D inline no_bc( const IntVector c, const IntVector coord, phiT& phi, 
//            FaceData1D vel, constCCVariable<Vector>& areaFraction ) 
//        { 
//          FaceData1D face_values;
//          face_values.plus  = 0.0;
//          face_values.minus = 0.0;
//
//          double r=0.0; 
//          double psi; 
//          double Sup;
//          double Sdn;
//          const double tiny = 1.0e-16; 
//
//          IntVector cxp  = c + coord; 
//          IntVector cxpp = c + coord + coord; 
//          IntVector cxm  = c - coord; 
//          IntVector cxmm = c - coord - coord; 
//
//          int dim = 0; 
//          if (coord[0] == 1)
//            dim =0; 
//          else if (coord[1] == 1)
//            dim = 1; 
//          else 
//            dim = 2; 
//
//          // - FACE
//          if ( vel.minus > 0.0 ) {
//            Sup = phi[cxm];
//            Sdn = phi[c];
//            r = ( phi[cxm] - phi[cxmm] ) / ( phi[c] - phi[cxm] );
//
//            if ( areaFraction[cxm][dim] < tiny || areaFraction[c][dim] < tiny )
//              r = 0.0;
//
//          } else if ( vel.minus < 0.0 ) {
//            Sup = phi[c];
//            Sdn = phi[cxm];
//            r = ( phi[cxp] - phi[c] ) / ( phi[c] - phi[cxm] );
//
//            if ( areaFraction[cxp][dim] < tiny || areaFraction[c][dim] < tiny )
//              r = 0.0;
//
//          } else { 
//            Sup = 0.0;
//            Sdn = 0.0; 
//            psi = 0.0;
//          }
//          psi = std::max( std::min(2.0*r, 1.0), std::min(r, 2.0) );
//          psi = std::max( 0.0, psi );
//
//          face_values.minus = Sup + 0.5*psi*( Sdn - Sup );
//
//          // + FACE
//          if ( vel.plus > 0.0 ) {
//            r = ( phi[c] - phi[cxm] ) / ( phi[cxp] - phi[c] );
//            Sup = phi[c];
//            Sdn = phi[cxp];
//
//            if ( areaFraction[c][dim] < tiny || areaFraction[cxp][dim] < tiny )
//              r = 0.0;
//
//          } else if ( vel.plus < 0.0 ) {
//            r = ( phi[cxpp] - phi[cxp] ) / ( phi[cxp] - phi[c] );
//            Sup = phi[cxp];
//            Sdn = phi[c]; 
//
//            if ( areaFraction[cxpp][dim] < tiny || areaFraction[cxp][dim] < tiny )
//              r = 0.0; 
//
//          } else {
//            Sup = 0.0;
//            Sdn = 0.0; 
//            psi = 0.0;
//          }
//          psi = std::max( std::min(2.0*r, 1.0), std::min(r, 2.0) );
//          psi = std::max( 0.0, psi );
//
//          face_values.plus = Sup + 0.5*psi*( Sdn - Sup );
//
//          return face_values; 
//        }; 
//
//        FaceData1D inline with_bc( const IntVector c, const IntVector coord, phiT& phi, 
//            FaceData1D vel, constCCVariable<Vector>& areaFraction, FaceBoundaryBool isBoundary ) 
//        { 
//          FaceData1D face_values;
//          face_values.plus  = 0.0;
//          face_values.minus = 0.0;
//
//          double r = 0; 
//          double psi; 
//          double Sup;
//          double Sdn;
//          const double tiny = 1.0e-16; 
//
//          IntVector cxp  = c + coord; 
//          IntVector cxpp = c + coord + coord; 
//          IntVector cxm  = c - coord; 
//          IntVector cxmm = c - coord - coord; 
//
//          int dim = 0; 
//          if (coord[0] == 1)
//            dim =0; 
//          else if (coord[1] == 1)
//            dim = 1; 
//          else 
//            dim = 2; 
//
//          // - FACE
//          if (isBoundary.minus) 
//            face_values.minus = 0.5*(phi[c]+phi[cxm]);
//          else { 
//            if ( vel.minus > 0.0 ) {
//              Sup = phi[cxm];
//              Sdn = phi[c];
//              r = ( phi[cxm] - phi[cxmm] ) / ( phi[c] - phi[cxm] ); 
//
//            if ( areaFraction[cxm][dim] < tiny || areaFraction[c][dim] < tiny )
//              r = 0.0;
//
//            } else if ( vel.minus < 0.0 ) {
//              Sup = phi[c];
//              Sdn = phi[cxm];
//              r = ( phi[cxp] - phi[c] ) / ( phi[c] - phi[cxm] );
//
//            if ( areaFraction[cxp][dim] < tiny || areaFraction[c][dim] < tiny )
//              r = 0.0;
//
//            } else { 
//              Sup = 0.0;
//              Sdn = 0.0; 
//              psi = 0.0;
//            }
//            psi = std::max( std::min(2.0*r, 1.0), std::min(r, 2.0) );
//            psi = std::max( 0.0, psi );
//
//            face_values.minus = Sup + 0.5*psi*( Sdn - Sup );
//          }
//
//          // + FACE
//          if (isBoundary.plus)
//            face_values.plus = 0.5*(phi[c] + phi[cxp]);
//          else { 
//            if ( vel.plus > 0.0 ) {
//              r = ( phi[c] - phi[cxm] ) / ( phi[cxp] - phi[c] );
//              Sup = phi[c];
//              Sdn = phi[cxp];
//
//            if ( areaFraction[c][dim] < tiny || areaFraction[cxp][dim] < tiny )
//              r = 0.0;
//
//            } else if ( vel.plus < 0.0 ) {
//              r = ( phi[cxpp] - phi[cxp] ) / ( phi[cxp] - phi[c] );
//              Sup = phi[cxp];
//              Sdn = phi[c]; 
//
//            if ( areaFraction[cxpp][dim] < tiny || areaFraction[cxp][dim] < tiny )
//              r = 0.0; 
//
//            } else {
//              Sup = 0.0;
//              Sdn = 0.0; 
//              psi = 0.0;
//            }
//            psi = std::max( std::min(2.0*r, 1.0), std::min(r, 2.0) );
//            psi = std::max( 0.0, psi );
//
//            face_values.plus = Sup + 0.5*psi*( Sdn - Sup );
//          }
//
//          return face_values; 
//        };
//      }; 
