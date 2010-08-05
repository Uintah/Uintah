#ifndef Wasatch_FieldTypes_h
#define Wasatch_FieldTypes_h

#include <spatialops/structured/FVStaggeredTypes.h>

namespace Wasatch{


  typedef SpatialOps::structured::SVolField ScalarVolField; ///< field type for scalar volume

  typedef SpatialOps::structured::XVolField XVolField;      ///< field type for x-staggered volume
  typedef SpatialOps::structured::YVolField YVolField;      ///< field type for y-staggered volume
  typedef SpatialOps::structured::ZVolField ZVolField;      ///< field type for z-staggered volume


  template< typename CellFieldT > struct FaceTypes;  ///< Given the volume field type, defines the flux field types

  template<> struct FaceTypes<ScalarVolField>
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
