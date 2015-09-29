#ifndef Uintah_Component_Arches_MomentumHelper
#define Uintah_Component_Arches_MomentumHelper

namespace Uintah{

  template< typename FT>
  struct MomentumHelper{
  };

  template <>
  struct MomentumHelper<SpatialOps::XVolField>{

    typedef FaceTypes<SpatialOps::XVolField>::XFace XFaceT;
    typedef FaceTypes<SpatialOps::XVolField>::YFace YFaceT;
    typedef FaceTypes<SpatialOps::XVolField>::ZFace ZFaceT;

    typedef SpatialOps::YVolField VT;
    typedef SpatialOps::ZVolField WT;

  };

  template <>
  struct MomentumHelper<SpatialOps::YVolField>{

    typedef FaceTypes<SpatialOps::YVolField>::YFace XFaceT;
    typedef FaceTypes<SpatialOps::YVolField>::ZFace YFaceT;
    typedef FaceTypes<SpatialOps::YVolField>::XFace ZFaceT;

    typedef SpatialOps::ZVolField VT;
    typedef SpatialOps::XVolField WT;

  };

  template <>
  struct MomentumHelper<SpatialOps::ZVolField>{

    typedef FaceTypes<SpatialOps::ZVolField>::ZFace XFaceT;
    typedef FaceTypes<SpatialOps::ZVolField>::XFace YFaceT;
    typedef FaceTypes<SpatialOps::ZVolField>::YFace ZFaceT;

    typedef SpatialOps::XVolField VT;
    typedef SpatialOps::YVolField WT;

  };
}
#endif
