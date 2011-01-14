#ifndef Wasatch_FieldTypes_h
#define Wasatch_FieldTypes_h

/**
 *  \file FieldTypes.h
 *  \brief Defines field types for use in Wasatch.
 */

#include <spatialops/structured/FVStaggeredTypes.h>

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

  typedef SpatialOps::structured::SVolField SVolField;  ///< field type for scalar volume
  typedef SpatialOps::structured::XVolField XVolField;  ///< field type for x-staggered volume
  typedef SpatialOps::structured::YVolField YVolField;  ///< field type for y-staggered volume
  typedef SpatialOps::structured::ZVolField ZVolField;  ///< field type for z-staggered volume

  /** @} */

  /**
   *  \ingroup WasatchFields
   *  \struct FaceTypes
   *  \brief Define Face field types in terms of a cell field type.
   *
   *  Specializations of this struct define te following typedefs:
   *   - \c XFace - the type of the field on the x-face
   *   - \c YFace - the type of the field on the yface
   *   - \c ZFace - the type of the field on the z-face
   *
   *  Example usage:
   *  \code
   *  typedef FaceTypes< CellT >::XFace XFaceT;
   *  typedef FaceTypes< CellT >::YFace YFaceT;
   *  typedef FaceTypes< CellT >::ZFace ZFaceT;
   *  \endcode
   *
   *  Class template specializations exist for the following field types:
   *   - ScalarVolField
   *   - XVolField
   *   - YVolField
   *   - ZVolField
   */
  template< typename CellFieldT > struct FaceTypes;  ///< Given the volume field type, defines the flux field types

  template<> struct FaceTypes<SVolField>
  {
    typedef SpatialOps::structured::SSurfXField XFace;
    typedef SpatialOps::structured::SSurfYField YFace;
    typedef SpatialOps::structured::SSurfZField ZFace;
  };

  template<> struct FaceTypes<XVolField>
  {
    typedef SpatialOps::structured::XSurfXField XFace;
    typedef SpatialOps::structured::XSurfYField YFace;
    typedef SpatialOps::structured::XSurfZField ZFace;
  };

  template<> struct FaceTypes<YVolField>
  {
    typedef SpatialOps::structured::YSurfXField XFace;
    typedef SpatialOps::structured::YSurfYField YFace;
    typedef SpatialOps::structured::YSurfZField ZFace;
  };

  template<> struct FaceTypes<ZVolField>
  {
    typedef SpatialOps::structured::ZSurfXField XFace;
    typedef SpatialOps::structured::ZSurfYField YFace;
    typedef SpatialOps::structured::ZSurfZField ZFace;
  };

} // namespace Wasatch

#endif // Wasatch_FieldTypes_h
