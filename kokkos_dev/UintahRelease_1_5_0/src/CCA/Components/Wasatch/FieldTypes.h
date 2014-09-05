/*
 * The MIT License
 *
 * Copyright (c) 2012 The University of Utah
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

#ifndef Wasatch_FieldTypes_h
#define Wasatch_FieldTypes_h

/**
 *  \file FieldTypes.h
 *  \brief Defines field types for use in Wasatch.
 */

#include <spatialops/SpatialOpsDefs.h>
#include <spatialops/structured/FVStaggeredFieldTypes.h>


using SpatialOps::structured::SVolField;   ///< the scalar volume field type
using SpatialOps::structured::XVolField;   ///< the x-staggered volume field type
using SpatialOps::structured::YVolField;   ///< the y-staggered volume field type
using SpatialOps::structured::ZVolField;   ///< the z-staggered volume field type

using SpatialOps::structured::FaceTypes;   ///< allows deducing face types from volume types

namespace Wasatch{

  /** \addtogroup WasatchFields
   *  @{
   */


  /**
   *  \enum Direction
   *  \brief enumerates directions
   */
  enum Direction{
    XDIR  = SpatialOps::XDIR::value,
    YDIR  = SpatialOps::YDIR::value,
    ZDIR  = SpatialOps::ZDIR::value,
    NODIR = SpatialOps::NODIR::value
  };

  /**
   *  \brief returns the staggered location of type FieldT.
   */
  template< typename FieldT >
  Direction get_staggered_location()
  {
    typedef typename SpatialOps::structured::GetNonzeroDir<typename FieldT::Location::Offset>::DirT  DirT;
    if     ( SpatialOps::IsSameType<DirT,SpatialOps::XDIR>::result ) return XDIR;
    else if( SpatialOps::IsSameType<DirT,SpatialOps::YDIR>::result ) return YDIR;
    else if( SpatialOps::IsSameType<DirT,SpatialOps::ZDIR>::result ) return ZDIR;
    return NODIR;
  }
  /** @} */

} // namespace Wasatch

#endif // Wasatch_FieldTypes_h
