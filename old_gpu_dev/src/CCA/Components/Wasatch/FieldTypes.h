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
