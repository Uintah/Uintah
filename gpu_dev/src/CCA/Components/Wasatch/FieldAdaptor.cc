#include <CCA/Components/Wasatch/FieldAdaptor.h>
#include <Core/Grid/Patch.h>

#include <spatialops/structured/FVTools.h>

#include <map>
#include <string>
#include <algorithm>

namespace Wasatch{

  typedef std::map<std::string,FieldTypes> StringMap;
  static StringMap validStrings;

  void set_string_map()
  {
    if( !validStrings.empty() ) return;

    validStrings["SVOL"  ] = SVOL;
    validStrings["SSURFX"] = SSURFX;
    validStrings["SSURFY"] = SSURFY;
    validStrings["SSURFZ"] = SSURFZ;

    validStrings["XVOL"  ] = XVOL;
    validStrings["XSURFX"] = XSURFX;
    validStrings["XSURFY"] = XSURFY;
    validStrings["XSURFZ"] = XSURFZ;

    validStrings["YVOL"  ] = YVOL;
    validStrings["YSURFX"] = YSURFX;
    validStrings["YSURFY"] = YSURFY;
    validStrings["YSURFZ"] = YSURFZ;

    validStrings["ZVOL"  ] = ZVOL;
    validStrings["ZSURFX"] = ZSURFX;
    validStrings["ZSURFY"] = ZSURFY;
    validStrings["ZSURFZ"] = ZSURFZ;
  }

  //------------------------------------------------------------------

  FieldTypes get_field_type( std::string key )
  {
    set_string_map();
    std::transform( key.begin(), key.end(), key.begin(), ::toupper );
    return validStrings[key];
  }

  //------------------------------------------------------------------

  void get_bc_logicals( const Uintah::Patch* const patch,
                        SCIRun::IntVector& bcMinus,
                        SCIRun::IntVector& bcPlus )
  {
    for( int i=0; i<3; ++i ){
      bcMinus[i] = 1;
      bcPlus [i] = 1;
    }
    std::vector<Uintah::Patch::FaceType> faces;
    patch->getNeighborFaces(faces);
    for( std::vector<Uintah::Patch::FaceType>::const_iterator i=faces.begin(); i!=faces.end(); ++i ){
      SCIRun::IntVector dir = patch->getFaceDirection(*i);
      for( int j=0; j<3; ++j ){
        if( dir[j] == -1 ) bcMinus[j]=0;
        if( dir[j] ==  1 ) bcPlus [j]=0;
      }
    }
  }

  //------------------------------------------------------------------

  template< typename FieldT >
  SpatialOps::structured::MemoryWindow
  get_memory_window_for_uintah_field( const Uintah::Patch* const patch )
  {
    SCIRun::IntVector bcMinus, bcPlus;
    get_bc_logicals( patch, bcMinus, bcPlus );
    const SCIRun::IntVector& extraCells = patch->getExtraCells();

    const SCIRun::IntVector gs = patch->getExtraCellHighIndex(0) - patch->getExtraCellLowIndex(0);
    SCIRun::IntVector globSize;
    globSize[0] = SpatialOps::structured::get_nx_with_ghost<FieldT>( gs[0], bcPlus[0] );
    globSize[1] = SpatialOps::structured::get_ny_with_ghost<FieldT>( gs[1], bcPlus[1] );
    globSize[2] = SpatialOps::structured::get_nz_with_ghost<FieldT>( gs[2], bcPlus[2] );

    using SpatialOps::structured::IntVec;
    IntVec glob(0,0,0), extent(0,0,0), offset(0,0,0);
    for( size_t i=0; i<3; ++i ){
      glob  [i] = globSize[i];
      extent[i] = globSize[i];
    }
    return SpatialOps::structured::MemoryWindow( glob, offset, extent, bcPlus[0], bcPlus[1], bcPlus[2] );
  }

  //------------------------------------------------------------------

  template<> Uintah::Ghost::GhostType get_uintah_ghost_type<SpatialOps::structured::SVolField  >(){ return Uintah::Ghost::AroundCells; }
  template<> Uintah::Ghost::GhostType get_uintah_ghost_type<SpatialOps::structured::SSurfXField>(){ return Uintah::Ghost::AroundFaces; }
  template<> Uintah::Ghost::GhostType get_uintah_ghost_type<SpatialOps::structured::SSurfYField>(){ return Uintah::Ghost::AroundFaces; }
  template<> Uintah::Ghost::GhostType get_uintah_ghost_type<SpatialOps::structured::SSurfZField>(){ return Uintah::Ghost::AroundFaces; }

  template<> Uintah::Ghost::GhostType get_uintah_ghost_type<SpatialOps::structured::XVolField  >(){ return Uintah::Ghost::AroundFaces; }
  template<> Uintah::Ghost::GhostType get_uintah_ghost_type<SpatialOps::structured::XSurfXField>(){ return Uintah::Ghost::AroundCells; }
  template<> Uintah::Ghost::GhostType get_uintah_ghost_type<SpatialOps::structured::XSurfYField>(){ return Uintah::Ghost::AroundFaces; }
  template<> Uintah::Ghost::GhostType get_uintah_ghost_type<SpatialOps::structured::XSurfZField>(){ return Uintah::Ghost::AroundFaces; }

  template<> Uintah::Ghost::GhostType get_uintah_ghost_type<SpatialOps::structured::YVolField  >(){ return Uintah::Ghost::AroundFaces; }
  template<> Uintah::Ghost::GhostType get_uintah_ghost_type<SpatialOps::structured::YSurfXField>(){ return Uintah::Ghost::AroundFaces; }
  template<> Uintah::Ghost::GhostType get_uintah_ghost_type<SpatialOps::structured::YSurfYField>(){ return Uintah::Ghost::AroundCells; }
  template<> Uintah::Ghost::GhostType get_uintah_ghost_type<SpatialOps::structured::YSurfZField>(){ return Uintah::Ghost::AroundFaces; }

  template<> Uintah::Ghost::GhostType get_uintah_ghost_type<SpatialOps::structured::ZVolField  >(){ return Uintah::Ghost::AroundFaces; }
  template<> Uintah::Ghost::GhostType get_uintah_ghost_type<SpatialOps::structured::ZSurfXField>(){ return Uintah::Ghost::AroundFaces; }
  template<> Uintah::Ghost::GhostType get_uintah_ghost_type<SpatialOps::structured::ZSurfYField>(){ return Uintah::Ghost::AroundFaces; }
  template<> Uintah::Ghost::GhostType get_uintah_ghost_type<SpatialOps::structured::ZSurfZField>(){ return Uintah::Ghost::AroundCells; }

  //------------------------------------------------------------------

  // macro shortcuts for explicit template instantiation
#define declare_method( FIELDT )                                        \
  template SpatialOps::structured::MemoryWindow                         \
  get_memory_window_for_uintah_field<FIELDT>( const Uintah::Patch* const ); \
  template Uintah::Ghost::GhostType get_uintah_ghost_type<FIELDT>();
  
#define declare_variants( VOLT )                \
  declare_method( VOLT );                       \
  declare_method( FaceTypes<VOLT>::XFace );     \
  declare_method( FaceTypes<VOLT>::YFace );     \
  declare_method( FaceTypes<VOLT>::ZFace );

  declare_variants( SpatialOps::structured::SVolField );
  declare_variants( SpatialOps::structured::XVolField );
  declare_variants( SpatialOps::structured::YVolField );
  declare_variants( SpatialOps::structured::ZVolField );

  //------------------------------------------------------------------

} // namespace Wasatch
