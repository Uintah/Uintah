#ifndef Wasatch_FieldTypes_h
#define Wasatch_FieldTypes_h

/**
 *  \file FieldTypes.h
 *  \brief Defines field types for use in Wasatch.
 */

#include <spatialops/SpatialOpsDefs.h>
#include <spatialops/structured/FVStaggeredFieldTypes.h>


using SpatialOps::structured::SVolField;
using SpatialOps::structured::XVolField;
using SpatialOps::structured::YVolField;
using SpatialOps::structured::ZVolField;

using SpatialOps::structured::FaceTypes;

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
    if     ( SpatialOps::CompareDirection< typename FieldT::Location::StagLoc, typename SpatialOps::XDIR >::same() ) return XDIR;
    else if( SpatialOps::CompareDirection< typename FieldT::Location::StagLoc, typename SpatialOps::YDIR >::same() ) return YDIR;
    else if( SpatialOps::CompareDirection< typename FieldT::Location::StagLoc, typename SpatialOps::ZDIR >::same() ) return ZDIR;
    return NODIR;
  }  
  /** @} */

} // namespace Wasatch

#endif // Wasatch_FieldTypes_h
