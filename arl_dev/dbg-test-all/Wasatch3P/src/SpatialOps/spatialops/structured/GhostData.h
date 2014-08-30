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

#ifndef SpatialOps_GhostData_h
#define SpatialOps_GhostData_h

/**
 *  \file   GhostData.h
 *
 *  \date   August, 2012
 *  \author Christopher Earl
 */

#include <spatialops/SpatialOpsConfigure.h>
#include <spatialops/SpatialOpsTools.h>
#include <spatialops/SpatialOpsDefs.h>

#include <spatialops/structured/IntVec.h>

#include <string>
#include <sstream>
#include <stdexcept>

#define GHOST_MAX 9001

namespace SpatialOps{

  /**
   * \class GhostData
   * \ingroup fields
   * \date July, 2013
   * \author James C. Sutherland
   * \brief Holds information about the number of ghost cells on each side of the domain
   */
  class GhostData
  {
    // note that this is a header-only class to inline its methods to improve performance.

    IntVec minus_, plus_;
    bool isInf_;

    // If any value is infinite (>= GHOST_MAX), then return true
    //  Infinite ghost data is used within Nebo to account for scalars
    inline bool is_IntVec_infinite(const IntVec & values) {
      return (values[0] >= GHOST_MAX &&
              values[1] >= GHOST_MAX &&
              values[2] >= GHOST_MAX);
    }

    // If any value is infinite (>= GHOST_MAX), then return true
    //  Infinite ghost data is used within Nebo to account for scalars
    inline bool is_ghost_infinite(const IntVec & minus, const IntVec & plus) {
      return (is_IntVec_infinite(minus) &&
              is_IntVec_infinite(plus));
    }

  public:

    /**
     * @brief Construct a GhostData
     * @param nx Number of ghost cells on the -x face
     * @param px Number of ghost cells on the +x face
     * @param ny Number of ghost cells on the -y face
     * @param py Number of ghost cells on the +y face
     * @param nz Number of ghost cells on the -z face
     * @param pz Number of ghost cells on the +z face
     */
    GhostData( const int nx, const int px,
               const int ny, const int py,
               const int nz, const int pz )
    : minus_( nx, ny, nz ),
      plus_ ( px, py, pz ),
      isInf_(is_ghost_infinite(minus_,plus_))
    {}


    /**
     * @brief Construct a GhostData
     * @param minus Number of ghost cells on the (-) x, y, and z faces
     * @param plus  Number of ghost cells on the (+) x, y, and z faces
     */
    GhostData( const IntVec& minus,
               const IntVec& plus )
    : minus_( minus ),
      plus_ ( plus  ),
      isInf_(is_ghost_infinite(minus_,plus_))
    {}

    /**
     * \brief construct a GhostData with the same number of ghost cells on each face
     * @param n the number of ghost cells on each face (defaults to zero)
     */
    GhostData( const int n=0 )
    : minus_( n, n, n ),
      plus_ ( n, n, n ),
      isInf_(is_ghost_infinite(minus_,plus_))
    {}

    GhostData( const GhostData& rhs )
    {
       minus_ = rhs.minus_;
       plus_  = rhs.plus_;
       isInf_ = rhs.isInf_;
     }

    inline GhostData& operator=( const GhostData& rhs )
    {
      minus_ = rhs.minus_;
      plus_  = rhs.plus_;
      isInf_ = rhs.isInf_;
      return *this;
    }

    /**
     * @brief obtain the IntVec containing the number of ghost cells on the (-) faces
     */
    inline IntVec get_minus() const{ return minus_; }

    /**
     * @brief obtain the number of ghost cells on the requested (-) face (0=x, 1=y, 2=z)
     */
    inline int get_minus( const int i ) const{ return minus_[i]; }

    /**
     * @brief obtain the IntVec containing the number of ghost cells on the (+) faces
     */
    inline IntVec get_plus() const{ return plus_; }

    /**
     * @brief obtain the number of ghost cells on the requested (+) face (0=x, 1=y, 2=z)
     */
    inline int get_plus( const int i ) const{ return plus_[i]; }

    /**
     * @brief set the number of ghost cells on the requested (-) face (0=x, 1=y, 2=z)
     */
    inline void set_minus( const IntVec& minus){
      minus_ = minus;
      isInf_ = is_ghost_infinite(minus_,plus_);
    }


    /**
     * @brief set the number of ghost cells on the requested (+) face (0=x, 1=y, 2=z)
     */
    inline void set_plus( const IntVec& plus ){
      plus_ = plus;
      isInf_ = is_ghost_infinite(minus_,plus_);
    }

    inline GhostData  operator+ ( const GhostData& rhs ) const{
      GhostData g(*this);
      g += rhs;
      return g;
    }

    inline GhostData& operator+=( const GhostData& rhs ){
      if(!isInf_) {
        if(rhs.isInf_) {
          *this = rhs;
        }
        else {
          minus_ += rhs.minus_;
          plus_  += rhs.plus_;
        }
      }
      return *this;
    }

    inline GhostData  operator- ( const GhostData& rhs ) const{
      GhostData g(*this);
      g -= rhs;
      return g;
    }

    inline GhostData& operator-=( const GhostData& rhs ){
      if(rhs.isInf_) {
        throw(std::runtime_error("Cannot use infinite ghost data on the right-hand side of subtraction."));
      }
      minus_ -= rhs.minus_;
      plus_  -= rhs.plus_;
      return *this;
    }

    inline bool operator==( const GhostData& rhs ) const{
      return (minus_ == rhs.minus_) && (plus_ == rhs.plus_);
    }

    inline GhostData limit_by_extent( IntVec const & extent) const {
      return GhostData((extent[0] == 1 ? 0 : minus_[0]),
                       (extent[0] == 1 ? 0 : plus_[0]),
                       (extent[1] == 1 ? 0 : minus_[1]),
                       (extent[1] == 1 ? 0 : plus_[1]),
                       (extent[2] == 1 ? 0 : minus_[2]),
                       (extent[2] == 1 ? 0 : plus_[2]));
    }
  };

  inline GhostData min( const GhostData& first, const GhostData& second )
  {
    return GhostData( min( first.get_minus(), second.get_minus() ),
                      min( first.get_plus(),  second.get_plus()  ));
  }

  inline GhostData point_to_ghost( const IntVec&  given )
  {
      return GhostData((given[0] < 0 ? - given[0] : 0),
                       (given[0] > 0 ?   given[0] : 0),
                       (given[1] < 0 ? - given[1] : 0),
                       (given[1] > 0 ?   given[1] : 0),
                       (given[2] < 0 ? - given[2] : 0),
                       (given[2] > 0 ?   given[2] : 0));
  }

  inline GhostData reductive_point_to_ghost( const IntVec&  given )
  {
      return GhostData((given[0] < 0 ?   given[0] : 0),
                       (given[0] > 0 ? - given[0] : 0),
                       (given[1] < 0 ?   given[1] : 0),
                       (given[1] > 0 ? - given[1] : 0),
                       (given[2] < 0 ?   given[2] : 0),
                       (given[2] > 0 ? - given[2] : 0));
  }

  inline GhostData additive_point_to_ghost( const IntVec&  given )
  {
      return GhostData((given[0] > 0 ?   given[0] : 0),
                       (given[0] < 0 ? - given[0] : 0),
                       (given[1] > 0 ?   given[1] : 0),
                       (given[1] < 0 ? - given[1] : 0),
                       (given[2] > 0 ?   given[2] : 0),
                       (given[2] < 0 ? - given[2] : 0));
  }

  inline GhostData additive_reductive_point_to_ghost( const IntVec&  given )
  {
    return additive_point_to_ghost(given) + reductive_point_to_ghost(given);
  }

  inline std::ostream& operator<<( std::ostream& out, const GhostData& gd )
  {
    out << "{ " << gd.get_minus() << " " << gd.get_plus() << " }";
    return out;
  }

} // namespace SpatialOps

#endif /* SpatialOps_GhostData_h */
