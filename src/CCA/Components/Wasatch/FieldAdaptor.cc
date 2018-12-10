/*
 * The MIT License
 *
 * Copyright (c) 2012-2018 The University of Utah
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
#include <Core/Exceptions/ProblemSetupException.h>

#include <map>
#include <string>
#include <algorithm>

namespace so = SpatialOps;

namespace WasatchCore{

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
    
    validStrings["PERPATCH"] = PERPATCH;
    
    validStrings["PARTICLE"] = PARTICLE;
  }

  //------------------------------------------------------------------

  FieldTypes get_field_type( std::string key )
  {
    set_string_map();
    std::transform( key.begin(), key.end(), key.begin(), ::toupper );

    if (validStrings.find(key) == validStrings.end()) {
      std::ostringstream msg;
      msg << "ERROR: unsupported field type '" << key << "'" << std::endl;
      throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
    }
    
    return validStrings[key];
  }

  //------------------------------------------------------------------

  void get_bc_logicals( const Uintah::Patch* const patch,
                        so::IntVec& bcMinus,
                        so::IntVec& bcPlus )
  {
    for( int i=0; i<3; ++i ){
      bcMinus[i] = 1;
      bcPlus [i] = 1;
    }
    std::vector<Uintah::Patch::FaceType> faces;
    patch->getNeighborFaces(faces);
    for( std::vector<Uintah::Patch::FaceType>::const_iterator i=faces.begin(); i!=faces.end(); ++i ){
      Uintah::IntVector dir = patch->getFaceDirection(*i);
      for( int j=0; j<3; ++j ){
        if( dir[j] == -1 ) bcMinus[j]=0;
        if( dir[j] ==  1 ) bcPlus [j]=0;
      }
    }
  }

  //------------------------------------------------------------------

  template< typename FieldT >
  SpatialOps::MemoryWindow
  get_memory_window_for_uintah_field( const Uintah::Patch* const patch )
  {
    so::IntVec bcMinus, bcPlus;
    get_bc_logicals( patch, bcMinus, bcPlus );

    const Uintah::IntVector gs = patch->getCellHighIndex(0) - patch->getCellLowIndex(0);

    const int nGhost = get_n_ghost<FieldT>();
    const so::IntVec glob( gs[0] + nGhost*2 + (bcPlus[0] ? FieldT::Location::BCExtra::X : 0),
                           gs[1] + nGhost*2 + (bcPlus[1] ? FieldT::Location::BCExtra::Y : 0),
                           gs[2] + nGhost*2 + (bcPlus[2] ? FieldT::Location::BCExtra::Z : 0) );
    
    const so::IntVec extent = glob;
    const so::IntVec offset(nGhost,nGhost,nGhost);

    return so::MemoryWindow( glob, offset, extent );
  }

  template<>
  SpatialOps::MemoryWindow
  get_memory_window_for_uintah_field<so::SingleValueField>( const Uintah::Patch* const patch )
  {
    const int nGhost = get_n_ghost<so::SingleValueField>();
    return so::MemoryWindow( so::IntVec(1,1,1), so::IntVec(0,0,0), so::IntVec(nGhost,nGhost,nGhost) );
  }

  //------------------------------------------------------------------
  
  template< typename FieldT >
  SpatialOps::MemoryWindow
  get_memory_window_for_masks( const Uintah::Patch* const patch )
  {
    so::IntVec bcMinus, bcPlus;
    get_bc_logicals( patch, bcMinus, bcPlus );
    
    const Uintah::IntVector gs = patch->getCellHighIndex(0) - patch->getCellLowIndex(0);
    
    const int nGhost = get_n_ghost<FieldT>();
    const so::IntVec glob( gs[0] + nGhost*2 + (bcPlus[0] ? FieldT::Location::BCExtra::X : 0),
                           gs[1] + nGhost*2 + (bcPlus[1] ? FieldT::Location::BCExtra::Y : 0),
                           gs[2] + nGhost*2 + (bcPlus[2] ? FieldT::Location::BCExtra::Z : 0) );
    
    const so::IntVec extent = glob;
    const so::IntVec offset(0,0,0);
    
    return so::MemoryWindow( glob, offset, extent );
  }


  //------------------------------------------------------------------
  template<> Uintah::Ghost::GhostType get_uintah_ghost_type<int            >(){ return Uintah::Ghost::AroundCells; }
  template<> Uintah::Ghost::GhostType get_uintah_ghost_type<so::SVolField  >(){ return Uintah::Ghost::AroundCells; }
  template<> Uintah::Ghost::GhostType get_uintah_ghost_type<so::SSurfXField>(){ return Uintah::Ghost::AroundFaces; }
  template<> Uintah::Ghost::GhostType get_uintah_ghost_type<so::SSurfYField>(){ return Uintah::Ghost::AroundFaces; }
  template<> Uintah::Ghost::GhostType get_uintah_ghost_type<so::SSurfZField>(){ return Uintah::Ghost::AroundFaces; }

  template<> Uintah::Ghost::GhostType get_uintah_ghost_type<so::XVolField  >(){ return Uintah::Ghost::AroundFaces; }
  template<> Uintah::Ghost::GhostType get_uintah_ghost_type<so::XSurfXField>(){ return Uintah::Ghost::AroundCells; }
  template<> Uintah::Ghost::GhostType get_uintah_ghost_type<so::XSurfYField>(){ return Uintah::Ghost::AroundFaces; }
  template<> Uintah::Ghost::GhostType get_uintah_ghost_type<so::XSurfZField>(){ return Uintah::Ghost::AroundFaces; }

  template<> Uintah::Ghost::GhostType get_uintah_ghost_type<so::YVolField  >(){ return Uintah::Ghost::AroundFaces; }
  template<> Uintah::Ghost::GhostType get_uintah_ghost_type<so::YSurfXField>(){ return Uintah::Ghost::AroundFaces; }
  template<> Uintah::Ghost::GhostType get_uintah_ghost_type<so::YSurfYField>(){ return Uintah::Ghost::AroundCells; }
  template<> Uintah::Ghost::GhostType get_uintah_ghost_type<so::YSurfZField>(){ return Uintah::Ghost::AroundFaces; }

  template<> Uintah::Ghost::GhostType get_uintah_ghost_type<so::ZVolField  >(){ return Uintah::Ghost::AroundFaces; }
  template<> Uintah::Ghost::GhostType get_uintah_ghost_type<so::ZSurfXField>(){ return Uintah::Ghost::AroundFaces; }
  template<> Uintah::Ghost::GhostType get_uintah_ghost_type<so::ZSurfYField>(){ return Uintah::Ghost::AroundFaces; }
  template<> Uintah::Ghost::GhostType get_uintah_ghost_type<so::ZSurfZField>(){ return Uintah::Ghost::AroundCells; }

  template<> Uintah::Ghost::GhostType get_uintah_ghost_type<so::SingleValueField>(){ return Uintah::Ghost::None; }

  template<> Uintah::Ghost::GhostType get_uintah_ghost_type<SpatialOps::Particle::ParticleField>(){ return Uintah::Ghost::None; }
  //------------------------------------------------------------------

  // macro shortcuts for explicit template instantiation
#define declare_method( FIELDT )                                              \
  template so::MemoryWindow                                                   \
  get_memory_window_for_uintah_field<FIELDT>( const Uintah::Patch* const  );  \
  template so::MemoryWindow get_memory_window_for_masks<FIELDT>( const Uintah::Patch* const  );

#define declare_variants( VOLT )                \
  declare_method( VOLT );                       \
  declare_method( FaceTypes<VOLT>::XFace );     \
  declare_method( FaceTypes<VOLT>::YFace );     \
  declare_method( FaceTypes<VOLT>::ZFace );

  declare_variants( so::SVolField );
  declare_variants( so::XVolField );
  declare_variants( so::YVolField );
  declare_variants( so::ZVolField );

  declare_method( SpatialOps::Particle::ParticleField );

  //------------------------------------------------------------------

} // namespace WasatchCore
