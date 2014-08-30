/*
 * Copyright (c) 2014 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */
#ifndef ParticleOperators_h
#define ParticleOperators_h

#include <spatialops/particles/ParticleFieldTypes.h>
#include <spatialops/Nebo.h>

#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace SpatialOps{
namespace Particle{

   /**
    *  @class  ParticleToCell
    *  @author James C. Sutherland
    *  @brief Interpolates an extensive particle field onto an underlying mesh field.
    *  @ingroup optypes
    *
    *  Note that this should only be used to interpolate extensive quantities and
    *  not intensive quantities.
    */
   template< typename CellField >
   class ParticleToCell
   {
   public:
     typedef ParticleField SrcFieldType;
     typedef CellField     DestFieldType;

     /**
      * @brief construct a ParticleToCell operator
      * @param dx  x-direction mesh spacing
      * @param xlo x-direction low coordinate of CellField (interior - not including ghost cells)
      * @param dy  y-direction mesh spacing
      * @param ylo y-direction low coordinate of CellField (interior - not including ghost cells)
      * @param dz  z-direction mesh spacing
      * @param zlo z-direction low coordinate of CellField (interior - not including ghost cells)
      */
     ParticleToCell( const double dx,    const double xlo,
                     const double dy=-1, const double ylo =0,
                     const double dz=-1, const double zlo =0 );

     /**
      * @param pxcoord x-coordinate of each particle
      * @param pycoord y-coordinate of each particle
      * @param pzcoord z-coordinate of each particle
      * @param psize   diameter of each particle
      */
     void set_coordinate_information( const ParticleField * const pxcoord,
                                      const ParticleField * const pycoord,
                                      const ParticleField * const pzcoord,
                                      const ParticleField * const psize );

     /**
      *  @param src source field from which values are interpolated to particles (ParticleField)
      *  @param dest destination field to which values are interpolated (CellField)
      */
     void apply_to_field( const SrcFieldType& src,
                          DestFieldType& dest ) const;
   private:
     const double dx_, dy_, dz_;
     const double xlo_, ylo_, zlo_;
     const ParticleField *px_, *py_, *pz_, *psize_;
   };


  //==================================================================


  /**
   *  @class CellToParticle
   *  @brief Operator to interpolate a mesh field onto a particle.
   *  @author James C. Sutherland
   *  @ingroup optypes
   *
   *  Note that this can be used for either intensive or extensive quantities.
   */
  template< typename CellField >
  class CellToParticle
  {
  public:
    typedef CellField     SrcFieldType;
    typedef ParticleField DestFieldType;

    /**
     * @brief Construct a CellToParticle operator
     * @param dx  x-direction mesh spacing
     * @param xlo x-direction low coordinate of CellField (interior - not including ghost cells)
     * @param dy  y-direction mesh spacing
     * @param ylo y-direction low coordinate of CellField (interior - not including ghost cells)
     * @param dz  z-direction mesh spacing
     * @param zlo z-direction low coordinate of CellField (interior - not including ghost cells)
     */
    CellToParticle( const double dx,    const double xlo,
                    const double dy=-1, const double ylo =0,
                    const double dz=-1, const double zlo =0 );

    /**
     * @param pxcoord x-coordinate of each particle
     * @param pycoord y-coordinate of each particle
     * @param pzcoord z-coordinate of each particle
     * @param psize   size of each particle
     */
    void set_coordinate_information( const ParticleField* pxcoord,
                                     const ParticleField* pycoord,
                                     const ParticleField* pzcoord,
                                     const ParticleField* psize );

    /**
     * @param src source field from which values are interpolated to particles (VolField)
     * @param dest destination field to which values are interpolated (ParticleField)
     */
    void apply_to_field( const SrcFieldType& src,
                         DestFieldType& dest ) const;
  private:
    const double dx_, dy_, dz_;
    const double xlo_, ylo_, zlo_;
    const ParticleField *px_, *py_, *pz_, *psize_;
  };



  // =================================================================
  //
  //                           Implementation
  //
  // =================================================================



   template< typename CellField >
   ParticleToCell<CellField>::
   ParticleToCell( const double dx, const double xlo,
                   const double dy, const double ylo,
                   const double dz, const double zlo )
     : dx_ ( dx<=0 ? 1.0 : dx ),
       dy_ ( dy<=0 ? 1.0 : dy ),
       dz_ ( dz<=0 ? 1.0 : dz ),
       xlo_( dx<=0 ? 0.0 : xlo ),
       ylo_( dy<=0 ? 0.0 : ylo ),
       zlo_( dz<=0 ? 0.0 : zlo )
   {
     px_    = NULL;
     py_    = NULL;
     pz_    = NULL;
     psize_ = NULL;
   }

   //------------------------------------------------------------------

   template< typename CellField >
   void
   ParticleToCell<CellField>::
   set_coordinate_information( const ParticleField *pxcoord,
                               const ParticleField *pycoord,
                               const ParticleField *pzcoord,
                               const ParticleField *psize )
   {
     px_    = pxcoord;
     py_    = pycoord;
     pz_    = pzcoord;
     psize_ = psize  ;
   }

   template< typename CellField >
   void
   ParticleToCell<CellField>::
   apply_to_field( const SrcFieldType& src,
                   DestFieldType& dest ) const
   {
     assert( psize_ != NULL );

     const IntVec& ngm = dest.get_ghost_data().get_minus();
     const IntVec& nmax = dest.window_with_ghost().extent();

     const double xloface = xlo_ - dx_*ngm[0] - dx_/2;
     const double yloface = ylo_ - dy_*ngm[1] - dy_/2;
     const double zloface = zlo_ - dz_*ngm[2] - dz_/2;

     dest <<= 0.0;  // set to zero and accumulate contributions from each particle below.

     // Note: here the outer loop is over particles and the inner loop is over
     // cells to sum in the particle contributions.  This should be optimal for
     // situations where there are a relatively small number of particles. In
     // cases where many particles per cell present, it may be more effective
     // to invert the loop structure.
     ParticleField::const_iterator ipx    = px_ ? px_->begin() : src.begin();
     ParticleField::const_iterator ipy    = py_ ? py_->begin() : src.begin();
     ParticleField::const_iterator ipz    = pz_ ? pz_->begin() : src.begin();
     ParticleField::const_iterator ipsize = psize_->begin();
     ParticleField::const_iterator isrc   = src.begin();
     const ParticleField::const_iterator ise = src.end();
     for( ; isrc != ise; ++ipx, ++ipy, ++ipz, ++ipsize, ++isrc ){

       const double rp = *ipsize * 0.5;

       // Identify the location of the particle boundary (assuming that it is a cube)
       const double pxlo = px_ ? *ipx - rp : 0;
       const double pylo = py_ ? *ipy - rp : 0;
       const double pzlo = pz_ ? *ipz - rp : 0;

       const double pxhi = px_ ? pxlo + *ipsize : 0;
       const double pyhi = py_ ? pylo + *ipsize : 0;
       const double pzhi = pz_ ? pzlo + *ipsize : 0;

       const size_t ixlo = ( pxlo - xloface ) / dx_;
       const size_t iylo = ( pylo - yloface ) / dy_;
       const size_t izlo = ( pzlo - zloface ) / dz_;

       // hi indices are 1 past the end
       const size_t ixhi = px_ ? std::min( nmax[0], int(( pxhi - xloface ) / dx_) + 1 ) : ixlo+1;
       const size_t iyhi = py_ ? std::min( nmax[1], int(( pyhi - yloface ) / dy_) + 1 ) : iylo+1;
       const size_t izhi = pz_ ? std::min( nmax[2], int(( pzhi - zloface ) / dz_) + 1 ) : izlo+1;

       // Distribute particle through the volume(s) it touches. Here we are
       // doing a highly approximate job at approximating how much fractional
       // particle volume is in each cell.
       const double pvol = (px_ ? 2*rp : 1)
                         * (py_ ? 2*rp : 1)
                         * (pz_ ? 2*rp : 1);
#      ifndef NDEBUG
       double sumterm=0.0;
#      endif
       for( size_t k=izlo; k<izhi; ++k ){
         const double zcm = zloface + k*dz_;
         const double zcp = zcm + dz_;
         // determine the z bounding box for the particle in this cell
         const double zcont = pz_ ? std::min(zcp,pzhi) - std::max(zcm,pzlo) : 1;
         for( size_t j=iylo; j<iyhi; ++j ){
           const double ycm = yloface + j*dy_;
           const double ycp = ycm + dy_;
           // determine the y bounding box for the particle in this cell
           const double ycont = py_ ? std::min(ycp,pyhi) - std::max(ycm,pylo) : 1;
           for( size_t i=ixlo; i<ixhi; ++i ){
             const double xcm = xloface + i*dx_;
             const double xcp = xcm + dx_;
             const double xcont = px_ ? std::min(xcp,pxhi) - std::max(xcm,pxlo) : 1;
             // contribution is the fraction of the particle volume in this cell.
             const double contribution = xcont*ycont*zcont / pvol;
             dest(i,j,k) += *isrc * contribution;
#            ifndef NDEBUG
             sumterm += contribution;
#            endif
           }
         }
       }
#      ifndef NDEBUG
       if( std::abs(1.0-sumterm) >= 1e-10 ) std::cout << "sum: " << sumterm << std::endl;
       assert( std::abs(1.0-sumterm) < 1e-10 );
#      endif
     } // particle loop
   }

 //==================================================================

  template< typename CellField >
  CellToParticle<CellField>::
  CellToParticle( const double dx, const double xlo,
                  const double dy, const double ylo,
                  const double dz, const double zlo )
  : dx_ ( dx<=0 ? 1.0 : dx ),
    dy_ ( dy<=0 ? 1.0 : dy ),
    dz_ ( dz<=0 ? 1.0 : dz ),
    xlo_( dx<=0 ? 0.0 : xlo ),
    ylo_( dy<=0 ? 0.0 : ylo ),
    zlo_( dz<=0 ? 0.0 : zlo )
  {
    px_    = NULL;
    py_    = NULL;
    pz_    = NULL;
    psize_ = NULL;
  }

  //------------------------------------------------------------------

  template<typename CellField>
  void
  CellToParticle<CellField>::
  set_coordinate_information( const ParticleField *pxcoord,
                              const ParticleField *pycoord,
                              const ParticleField *pzcoord,
                              const ParticleField *psize )
  {
    px_    = pxcoord;
    py_    = pycoord;
    pz_    = pzcoord;
    psize_ = psize  ;
  }

  //------------------------------------------------------------------

  template<typename CellField>
  void
  CellToParticle<CellField>::
  apply_to_field( const SrcFieldType& src,
                  DestFieldType& dest ) const
  {
    const IntVec& ngm = src.get_ghost_data().get_minus();
    const IntVec& nmax = src.window_with_ghost().extent();

    // Get the location of the (-) face associated with this cell field. Here we
    // shift the xlo_ value (the value associated with this field type's origin)
    // by its ghost offset and by the distance from the cell center to the face.
    const double xloface = xlo_ - dx_*ngm[0] - dx_/2;
    const double yloface = ylo_ - dy_*ngm[1] - dy_/2;
    const double zloface = zlo_ - dz_*ngm[2] - dz_/2;

    ParticleField::const_iterator ipx    = px_ ? px_->begin() : dest.begin();
    ParticleField::const_iterator ipy    = py_ ? py_->begin() : dest.begin();
    ParticleField::const_iterator ipz    = pz_ ? pz_->begin() : dest.begin();
    ParticleField::const_iterator ipsize = psize_->begin();
    ParticleField::iterator        idst  = dest.begin();
    const ParticleField::iterator  idste = dest.end();
    for( ; idst != idste; ++ipx, ++ipy, ++ipz, ++ipsize, ++idst ){

      *idst = 0.0;

      const double rp = *ipsize * 0.5;

      // Identify the location of the particle boundary
      const double pxlo = px_ ? *ipx - rp : 0;
      const double pylo = py_ ? *ipy - rp : 0;
      const double pzlo = pz_ ? *ipz - rp : 0;

      const double pxhi = px_ ? pxlo + *ipsize : 0;
      const double pyhi = py_ ? pylo + *ipsize : 0;
      const double pzhi = pz_ ? pzlo + *ipsize : 0;

      const size_t ixlo = ( pxlo - xloface ) / dx_;
      const size_t iylo = ( pylo - yloface ) / dy_;
      const size_t izlo = ( pzlo - zloface ) / dz_;

      // hi indices are 1 past the end
      const size_t ixhi = px_ ? std::min( nmax[0], int(( pxhi - xloface ) / dx_) + 1 ) : ixlo+1;
      const size_t iyhi = py_ ? std::min( nmax[1], int(( pyhi - yloface ) / dy_) + 1 ) : iylo+1;
      const size_t izhi = pz_ ? std::min( nmax[2], int(( pzhi - zloface ) / dz_) + 1 ) : izlo+1;

      // Distribute particle through the volume(s) it touches. Here we are
      // doing a highly approximate job at approximating how much fractional
      // particle volume is in each cell.
      const double pvol = (px_ ? 2*rp : 1)
                        * (py_ ? 2*rp : 1)
                        * (pz_ ? 2*rp : 1);

#     ifndef NDEBUG
      double sumterm=0.0;
#     endif
      for( size_t k=izlo; k<izhi; ++k ){
        const double zcm = zloface + k*dz_; // (-) face of the cell
        const double zcp = zcm + dz_;       // (+) face of the cell
        // determine the z bounding box for the particle in this cell
        const double zcont = pz_ ? std::min(zcp,pzhi) - std::max(zcm,pzlo) : 1;
        for( size_t j=iylo; j<iyhi; ++j ){
          const double ycm = yloface + j*dy_;
          const double ycp = ycm + dy_;
          // determine the y bounding box for the particle in this cell
          const double ycont = py_ ? std::min(ycp,pyhi) - std::max(ycm,pylo) : 1;
          for( size_t i=ixlo; i<ixhi; ++i ){
            const double xcm = xloface + i*dx_;
            const double xcp = xcm + dx_;
            const double xcont = px_ ? std::min(xcp,pxhi) - std::max(xcm,pxlo) : 1;
            // contribution is the fraction of the particle volume in this cell.
            const double contribution = xcont*ycont*zcont / pvol;
            *idst += src(i,j,k) * contribution;
#           ifndef NDEBUG
            assert( contribution > 0 );
            sumterm += contribution;
#           endif
          }
        }
      }
#     ifndef NDEBUG
      if( std::abs(1.0-sumterm) >= 1e-10 ) std::cout << "C2P sum: " << sumterm << std::endl;
      assert( std::abs(1.0-sumterm) < 1e-10 );
#     endif
    } // particle loop
  }

} // namespace Particle
} // namespace SpatialOps

#endif // ParticleOperators_h
