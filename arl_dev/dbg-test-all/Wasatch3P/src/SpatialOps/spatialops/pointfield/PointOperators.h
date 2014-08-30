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

#ifndef PointOperators_h
#define PointOperators_h

#include "PointFieldTypes.h"

#include <vector>

namespace SpatialOps{
namespace Point{

  /**
   *  \class PointToField
   *  \brief Puts point field data into a parent field
   *
   *  \tparam FieldT the type of the parent field
   */
  template< typename FieldT >
  class PointToField
  {
    typedef std::vector<size_t> Indices;
    Indices indices_;

  public:
    typedef PointField SrcFieldType;
    typedef FieldT     DestFieldType;

    PointToField( const FieldT& fieldLocations );
    PointToField( const std::vector<size_t>& flatIndices );

    void apply_to_field( const SrcFieldType&, DestFieldType& ) const;
  };

  /**
   *  \class PointToField
   *  \brief Extracts data from a subset of a parent field into a point field
   *
   *  \tparam FieldT the type of the parent field
   */
  template< typename FieldT >
  class FieldToPoint
  {
    typedef std::vector<size_t> Indices;
    Indices indices_;
  public:
    typedef FieldT     SrcFieldType;
    typedef PointField DestFieldType;

    FieldToPoint( const FieldT& fieldLocations );
    FieldToPoint( const std::vector<size_t>& flatIndices );

    void apply_to_field( const SrcFieldType&, DestFieldType& ) const;
  };



  // =================================================================
  //
  //                           Implementation
  //
  // =================================================================



  //------------------------------------------------------------------

  template< typename FieldT >
  PointToField<FieldT>::PointToField( const Indices& locs )
  {
    indices_ = locs;
  }

  //------------------------------------------------------------------

  template< typename FieldT >
  PointToField<FieldT>::PointToField( const FieldT& locs )
  {
    size_t i=0;
    for( typename FieldT::const_iterator ii=locs.begin(); ii!=locs.end(); ++ii, ++i ){
      if( *ii != 0.0 ) indices_.push_back(i);
    }
  }

  //------------------------------------------------------------------

  template< typename FieldT >
  void
  PointToField<FieldT>::apply_to_field( const SrcFieldType& src, DestFieldType& dest ) const
  {
    typename SrcFieldType::const_iterator isrc = src.begin();
    const typename SrcFieldType::const_iterator isrce = src.end();

    size_t i=0;
    for( ; isrc!=isrce; ++isrc, ++i ){
      dest[ indices_[i] ] = *isrc;
    }

    assert( i == indices_.size() );
  }

  //==================================================================

  template< typename FieldT >
  FieldToPoint<FieldT>::FieldToPoint( const Indices& locs )
  {
    indices_ = locs;
  }

  //------------------------------------------------------------------

  template< typename FieldT >
  FieldToPoint<FieldT>::FieldToPoint( const FieldT& locs )
  {
    size_t i=0;
    for( typename FieldT::const_iterator ii=locs.begin(); ii!=locs.end(); ++ii, ++i ){
      if( *ii != 0.0 ) indices_.push_back(i);
    }
  }

  //------------------------------------------------------------------

  template< typename FieldT >
  void
  FieldToPoint<FieldT>::apply_to_field( const SrcFieldType& src, DestFieldType& dest ) const
  {
    typename DestFieldType::iterator idest = dest.begin();
    const typename DestFieldType::iterator ideste = dest.end();

    size_t i=0;
    for( ; idest!=ideste; ++idest, ++i ){
      *idest = src[ indices_[i] ];
    }

    assert( i == indices_.size() );
  }

  //------------------------------------------------------------------


} // namespace Point
} // namespace SpatialOps


#endif // PointOperators_h
