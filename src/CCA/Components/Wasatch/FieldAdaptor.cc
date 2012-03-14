/*
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

#include <CCA/Components/Wasatch/FieldAdaptor.h>
#include <Core/Grid/Patch.h>

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

//    const SCIRun::IntVector gs = patch->getExtraCellHighIndex(0) - patch->getExtraCellLowIndex(0);
    const SCIRun::IntVector gs = patch->getCellHighIndex(0) - patch->getCellLowIndex(0);

    using SpatialOps::structured::IntVec;

    const IntVec glob( gs[0] + FieldT::Ghost::NGHOST*2 + (bcPlus[0] ? FieldT::Location::BCExtra::X : 0),
                       gs[1] + FieldT::Ghost::NGHOST*2 + (bcPlus[1] ? FieldT::Location::BCExtra::Y : 0),
                       gs[2] + FieldT::Ghost::NGHOST*2 + (bcPlus[2] ? FieldT::Location::BCExtra::Z : 0) );
    const IntVec extent = glob;
    const IntVec offset(0,0,0);

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

  template<> Uintah::Ghost::GhostType get_uintah_ghost_type<double>(){ return Uintah::Ghost::None; }
  //------------------------------------------------------------------

  // macro shortcuts for explicit template instantiation
#define declare_method( FIELDT )                                                \
  template SpatialOps::structured::MemoryWindow                                 \
  get_memory_window_for_uintah_field<FIELDT>( const Uintah::Patch* const );     \
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
