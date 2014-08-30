/**
 *  \file   BoundaryCellInfo.h
 *  \date   Jul 10, 2013
 *  \author "James C. Sutherland"
 *
 *
 * The MIT License
 *
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
 *
 */

#ifndef SpatialOps_BOUNDARYCELLINFO_H_
#define SpatialOps_BOUNDARYCELLINFO_H_

#include <spatialops/structured/IntVec.h>

namespace SpatialOps {

  /**
   *  \class  BoundaryCellInfo
   *  \date   Jul 10, 2013
   *  \author "James C. Sutherland"
   *
   *  \brief Provides information about boundary cells for various fields.
   */
  class BoundaryCellInfo
  {
    IntVec isbc_;
    IntVec nExtra_;

    /**
     * Construct a BoundaryCellInfo.  This is private to disallow direct construction.
     * @param bcx true if a physical boundary is present on the (+x) face
     * @param bcy true if a physical boundary is present on the (+y) face
     * @param bcz true if a physical boundary is present on the (+z) face
     * @param nExtra the number of cells to augment this field by if a (+) boundary is present
     */
    inline BoundaryCellInfo( const bool bcx,
                             const bool bcy,
                             const bool bcz,
                             const IntVec nExtra )
    : isbc_(bcx, bcy, bcz ),
      nExtra_( isbc_ * nExtra ) // only extras if there is a BC present
    {}

  public:

    /**
     * \brief obtain a BoundaryCellInfo object for the requested field type
     * @param bcx true if a physical boundary is present on the (+x) face
     * @param bcy true if a physical boundary is present on the (+y) face
     * @param bcz true if a physical boundary is present on the (+z) face
     * @return the constructed BoundaryCellInfo object
     */
    template<typename FieldT>
    BoundaryCellInfo
    static inline build( const bool bcx, const bool bcy, const bool bcz ){
      return BoundaryCellInfo( bcx,bcy,bcz, FieldT::Location::BCExtra::int_vec() );
    }

    /**
     * \brief obtain a BoundaryCellInfo object for the requested field type
     * @param bc indicates if a physical boundary is present on each of the (+) faces
     * @return the constructed BoundaryCellInfo object
     */
    template<typename FieldT>
    BoundaryCellInfo
    static inline build( const IntVec& bc ){
      return BoundaryCellInfo( bc[0],bc[1],bc[2], FieldT::Location::BCExtra::int_vec() );
    }

    /**
     * \brief obtain a BoundaryCellInfo object for the requested field type,
     *        assuming that there is no physical boundary present on the (+) faces.
     * @return the constructed BoundaryCellInfo object
     */
    template<typename FieldT>
    BoundaryCellInfo
    static inline build(){
      return BoundaryCellInfo( false, false, false, FieldT::Location::BCExtra::int_vec() );
    }

    /**
     * \brief assignment operator
     */
    inline BoundaryCellInfo& operator=( const BoundaryCellInfo& bc ){
      isbc_   = bc.isbc_;
      nExtra_ = bc.nExtra_;
      return *this;
    }

    /**
     * \brief copy constructor
     */
    inline BoundaryCellInfo( const BoundaryCellInfo& bc ) : isbc_(bc.isbc_), nExtra_(bc.nExtra_) {}

    inline ~BoundaryCellInfo(){}

    /**
     * \brief query to see if a physical boundary is present in the given direction (0=x, 1=y, 2=z)
     */
    inline bool has_bc( const int dir ) const{ return isbc_[dir]; }

    /**
     * \brief obtain an IntVec indicating the presence of physical boundaries on the (+) faces
     */
    inline IntVec has_bc() const{ return isbc_; }

    /**
     * \brief obtain the number of extra cells *potentially* present on this field due to presence of physical boundaries
     * @param dir the direction of interest (0=x, 1=y, 2=z)
     */
    inline int num_extra( const int dir ) const{ assert(dir<3 && dir>=0); return nExtra_[dir]; }

    /**
     * \brief obtain the number of extra cells present on this field due to presence of physical boundaries.  If no physical boundary is present, this returns zero.
     */
    inline IntVec num_extra() const{ return nExtra_; }

    /**
     * \brief obtain the number of extra cells *actually* present on this field due to presence of physical boundaries
     * @param dir the direction of interest (0=x, 1=y, 2=z)
     */
    inline int has_extra( const int dir ) const{ assert(dir<3 && dir>=0); return has_bc(dir) ? num_extra(dir) : 0; }

    /**
     * \brief obtain the number of extra cells *actually* present on this field due to presence of physical boundaries
     */
    inline IntVec has_extra() const{ return IntVec(has_extra(0), has_extra(1), has_extra(2)); }

    /**
     * \brief limit extra cells to dimensions with extents > 1
     */
    inline BoundaryCellInfo limit_by_extent( const IntVec& extent ) const{
      return BoundaryCellInfo( has_bc(0), has_bc(1), has_bc(2),
                               IntVec( extent[0] == 1 ? 0 : has_extra(0),
                                       extent[1] == 1 ? 0 : has_extra(1),
                                       extent[2] == 1 ? 0 : has_extra(2) ) );
    }
  };

  inline std::ostream& operator<<( std::ostream& out, const BoundaryCellInfo& bc ){
    out << "BC flags: " << bc.has_bc() << "  #extra: " << bc.num_extra() << " has_extra: " << bc.has_extra();
    return out;
  }

} /* namespace SpatialOps */

#endif /* SpatialOps_BOUNDARYCELLINFO_H_ */
