/*
 * The MIT License
 *
 * Copyright (c) 2012-2014 The University of Utah
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

namespace SS = SpatialOps::structured;

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
    
    validStrings["PERPATCH"] = PERPATCH;
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
                        SS::IntVec& bcMinus,
                        SS::IntVec& bcPlus )
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
  get_memory_window_for_uintah_field( const AllocInfo& ainfo )
  {
    SS::IntVec bcMinus, bcPlus;
    get_bc_logicals( ainfo.patch, bcMinus, bcPlus );

    const SCIRun::IntVector gs = ainfo.patch->getCellHighIndex(0)
                               - ainfo.patch->getCellLowIndex(0);

    const int nGhost = get_n_ghost<FieldT>();
    const SS::IntVec glob( gs[0] + nGhost*2 + (bcPlus[0] ? FieldT::Location::BCExtra::X : 0),
                           gs[1] + nGhost*2 + (bcPlus[1] ? FieldT::Location::BCExtra::Y : 0),
                           gs[2] + nGhost*2 + (bcPlus[2] ? FieldT::Location::BCExtra::Z : 0) );
    
    const SS::IntVec extent = glob;
    const SS::IntVec offset(nGhost,nGhost,nGhost);

    return SS::MemoryWindow( glob, offset, extent );
  }

  template<>
  SpatialOps::structured::MemoryWindow
  get_memory_window_for_uintah_field<SS::SingleValueField>( const AllocInfo& ainfo  )
  {
    const int nGhost = get_n_ghost<SS::SingleValueField>();
    return SS::MemoryWindow( SS::IntVec(1,1,1), SS::IntVec(0,0,0), SS::IntVec(nGhost,nGhost,nGhost) );
  }

  template<>
  SpatialOps::structured::MemoryWindow
  get_memory_window_for_uintah_field<ParticleField>( const AllocInfo& ainfo )
  {
    const int npar = ainfo.oldDW->getParticleSubset( ainfo.materialIndex, ainfo.patch )->numParticles();
    const int nGhost = get_n_ghost<ParticleField>();
    return SS::MemoryWindow( SS::IntVec(npar,1,1) );
  }


  //------------------------------------------------------------------

  template< typename FieldT, typename UFT >
  FieldT*
  wrap_uintah_field_as_spatialops( UFT& uintahVar,
                                   const AllocInfo& ainfo,
                                   const SpatialOps::MemoryType mtype,
                                   const unsigned short int deviceIndex,
                                   double* uintahDeviceVar )
  {
    /*
     * NOTE: before changing things here, look at the line:
     *    Uintah::OnDemandDataWarehouse::d_combineMemory = false;
     * in Wasatch.cc.  This is currently preventing Uintah from
     * combining patch memory.
     */
    namespace SS = SpatialOps::structured;

    using SCIRun::IntVector;

    const SCIRun::IntVector lowIx       = uintahVar.getLowIndex();
    const SCIRun::IntVector highIx      = uintahVar.getHighIndex();
    const SCIRun::IntVector fieldSize   = uintahVar.getWindow()->getData()->size();
    const SCIRun::IntVector fieldOffset = uintahVar.getWindow()->getOffset();
    const SCIRun::IntVector fieldExtent = highIx - lowIx;

    const SS::IntVec   size(   fieldSize[0],   fieldSize[1],   fieldSize[2] );
    const SS::IntVec extent( fieldExtent[0], fieldExtent[1], fieldExtent[2] );
    const SS::IntVec offset( lowIx[0]-fieldOffset[0],
                             lowIx[1]-fieldOffset[1],
                             lowIx[2]-fieldOffset[2] );

    SS::IntVec bcMinus, bcPlus;
    get_bc_logicals( ainfo.patch, bcMinus, bcPlus );

    double* fieldValues_ = NULL;
    if( mtype == SpatialOps::EXTERNAL_CUDA_GPU ){
#     ifdef HAVE_CUDA
      fieldValues_ = const_cast<double*>( uintahDeviceVar );
#     endif
    }
    else{
      fieldValues_ = const_cast<typename FieldT::value_type*>( uintahVar.getPointer() );
    }

    // jcs why aren't we using get_memory_window_for_uintah_field here???
    return new FieldT( SS::MemoryWindow( size, offset, extent ),
                       SS::BoundaryCellInfo::build<FieldT>(bcPlus),
                       SS::GhostData( get_n_ghost<FieldT>() ),
                       fieldValues_,
                       SS::ExternalStorage,
                       mtype,
                       deviceIndex );
  }


  template< >
  ParticleField*
  wrap_uintah_field_as_spatialops<ParticleField,SelectUintahFieldType<ParticleField>::type>(
      SelectUintahFieldType<ParticleField>::type& uintahVar,
      const AllocInfo& ainfo,
      const SpatialOps::MemoryType mtype,
      const unsigned short int deviceIndex,
      double* uintahDeviceVar )
  {
    namespace SS = SpatialOps::structured;
    typedef ParticleField::value_type ValT;
    ValT* fieldValues = NULL;
    if( mtype == SpatialOps::EXTERNAL_CUDA_GPU ){
#     ifdef HAVE_CUDA
      fieldValues = const_cast<ValT*>( uintahDeviceVar );
#     endif
    }
    else{
      fieldValues = const_cast<ParticleField::value_type*>( (ValT*)uintahVar.getBasePointer() );
    }

    // jcs need to get GPU support ready...
    return new ParticleField( get_memory_window_for_uintah_field<ParticleField>(ainfo),
                              SS::BoundaryCellInfo::build<ParticleField>(),
                              SS::GhostData( get_n_ghost<ParticleField>() ),
                              fieldValues,
                              SS::ExternalStorage,
                              mtype,
                              deviceIndex );
  }

  template< >
  ParticleField*
  wrap_uintah_field_as_spatialops<ParticleField,SelectUintahFieldType<ParticleField>::const_type>(
      SelectUintahFieldType<ParticleField>::const_type& uintahVar,
      const AllocInfo& ainfo,
      const SpatialOps::MemoryType mtype,
      const unsigned short int deviceIndex,
      double* uintahDeviceVar )
  {
    namespace SS = SpatialOps::structured;
    typedef ParticleField::value_type ValT;
    ValT* fieldValues = NULL;
    if( mtype == SpatialOps::EXTERNAL_CUDA_GPU ){
#     ifdef HAVE_CUDA
      fieldValues = const_cast<ValT*>( uintahDeviceVar );
#     endif
    }
    else{
      fieldValues = const_cast<ValT*>( (ValT*)uintahVar.getBaseRep().getBasePointer() );
    }

    // jcs need to get GPU support ready...
    return new ParticleField( get_memory_window_for_uintah_field<ParticleField>(ainfo),
                              SS::BoundaryCellInfo::build<ParticleField>(),
                              SS::GhostData( get_n_ghost<ParticleField>() ),
                              fieldValues,
                              SS::ExternalStorage,
                              mtype,
                              deviceIndex );
  }

  //-----------------------------------------------------------------
  // NOTE: this wraps a raw uintah field type, whereas the default
  //       implementations work with an Expr::UintahFieldContainer
  template<>
  SpatialOps::structured::SingleValueField*
  wrap_uintah_field_as_spatialops<SpatialOps::structured::SingleValueField,Uintah::PerPatch<double*> >(
      Uintah::PerPatch<double*>& uintahVar,
      const AllocInfo& ainfo,
      const SpatialOps::MemoryType mtype,
      const unsigned short int deviceIndex,
      double* uintahDeviceVar )
  {
    namespace SS = SpatialOps::structured;
    typedef SS::SingleValueField FieldT;
    return new FieldT( get_memory_window_for_uintah_field<FieldT>(ainfo),
                       SS::BoundaryCellInfo::build<FieldT>(false,false,false),    // bc doesn't matter for single value fields
                       SS::GhostData( get_n_ghost<FieldT>() ),
                       uintahVar.get(),
                       SS::ExternalStorage,
                       mtype,
                       deviceIndex );
  }

  // NOTE: this wraps a raw uintah field type, whereas the default
  //       implementations work with an Expr::UintahFieldContainer
  template<>
  inline SpatialOps::structured::SingleValueField*
  wrap_uintah_field_as_spatialops<SpatialOps::structured::SingleValueField,Uintah::ReductionVariableBase>(
      Uintah::ReductionVariableBase& uintahVar,
      const AllocInfo& ainfo,
      const SpatialOps::MemoryType mtype,
      const unsigned short int deviceIndex,
      double* uintahDeviceVar )
  {
    namespace SS = SpatialOps::structured;
    typedef SS::SingleValueField FieldT;
    return new FieldT( SS::MemoryWindow( SS::IntVec(1,1,1), SS::IntVec(0,0,0), SS::IntVec(1,1,1) ),
                       SS::BoundaryCellInfo::build<FieldT>(false,false,false),    // bc doesn't matter for single value fields
                       SS::GhostData( get_n_ghost<FieldT>() ),
                       (double*)( uintahVar.getBasePointer() ),  // jcs this is a bit sketchy because of the type casting.  It will only work for reductions on doubles
                       SS::ExternalStorage,
                       mtype,
                       deviceIndex );
  }


  //------------------------------------------------------------------
  template<> Uintah::Ghost::GhostType get_uintah_ghost_type<int            >(){ return Uintah::Ghost::AroundCells; }
  template<> Uintah::Ghost::GhostType get_uintah_ghost_type<SS::SVolField  >(){ return Uintah::Ghost::AroundCells; }
  template<> Uintah::Ghost::GhostType get_uintah_ghost_type<SS::SSurfXField>(){ return Uintah::Ghost::AroundFaces; }
  template<> Uintah::Ghost::GhostType get_uintah_ghost_type<SS::SSurfYField>(){ return Uintah::Ghost::AroundFaces; }
  template<> Uintah::Ghost::GhostType get_uintah_ghost_type<SS::SSurfZField>(){ return Uintah::Ghost::AroundFaces; }

  template<> Uintah::Ghost::GhostType get_uintah_ghost_type<SS::XVolField  >(){ return Uintah::Ghost::AroundFaces; }
  template<> Uintah::Ghost::GhostType get_uintah_ghost_type<SS::XSurfXField>(){ return Uintah::Ghost::AroundCells; }
  template<> Uintah::Ghost::GhostType get_uintah_ghost_type<SS::XSurfYField>(){ return Uintah::Ghost::AroundFaces; }
  template<> Uintah::Ghost::GhostType get_uintah_ghost_type<SS::XSurfZField>(){ return Uintah::Ghost::AroundFaces; }

  template<> Uintah::Ghost::GhostType get_uintah_ghost_type<SS::YVolField  >(){ return Uintah::Ghost::AroundFaces; }
  template<> Uintah::Ghost::GhostType get_uintah_ghost_type<SS::YSurfXField>(){ return Uintah::Ghost::AroundFaces; }
  template<> Uintah::Ghost::GhostType get_uintah_ghost_type<SS::YSurfYField>(){ return Uintah::Ghost::AroundCells; }
  template<> Uintah::Ghost::GhostType get_uintah_ghost_type<SS::YSurfZField>(){ return Uintah::Ghost::AroundFaces; }

  template<> Uintah::Ghost::GhostType get_uintah_ghost_type<SS::ZVolField  >(){ return Uintah::Ghost::AroundFaces; }
  template<> Uintah::Ghost::GhostType get_uintah_ghost_type<SS::ZSurfXField>(){ return Uintah::Ghost::AroundFaces; }
  template<> Uintah::Ghost::GhostType get_uintah_ghost_type<SS::ZSurfYField>(){ return Uintah::Ghost::AroundFaces; }
  template<> Uintah::Ghost::GhostType get_uintah_ghost_type<SS::ZSurfZField>(){ return Uintah::Ghost::AroundCells; }

  template<> Uintah::Ghost::GhostType get_uintah_ghost_type<SS::SingleValueField>(){ return Uintah::Ghost::None; }

  template<> Uintah::Ghost::GhostType get_uintah_ghost_type<SpatialOps::Particle::ParticleField>(){ return Uintah::Ghost::None; }
  //------------------------------------------------------------------

  // macro shortcuts for explicit template instantiation
#define declare_wrap( FIELDT )                                                                        \
  template FIELDT* wrap_uintah_field_as_spatialops<FIELDT,SelectUintahFieldType<FIELDT>::type>(       \
      SelectUintahFieldType<FIELDT>::type&,                                                           \
      const AllocInfo&,                                                                               \
      const SpatialOps::MemoryType,                                                                   \
      const unsigned short int,double* );
#define declare_const_wrap( FIELDT )                                                                  \
  template FIELDT* wrap_uintah_field_as_spatialops<FIELDT,SelectUintahFieldType<FIELDT>::const_type>( \
      SelectUintahFieldType<FIELDT>::const_type&,                                                     \
      const AllocInfo&,                                                                               \
      const SpatialOps::MemoryType,                                                                   \
      const unsigned short int,double* );
#define declare_methods( FIELDT )                                                                     \
  template SS::MemoryWindow get_memory_window_for_uintah_field<FIELDT>( const AllocInfo& );           \
  template Uintah::Ghost::GhostType get_uintah_ghost_type<FIELDT>();                                  \

#define declare_variants( VOLT )                \
  declare_methods   ( VOLT                   ); \
  declare_methods   ( FaceTypes<VOLT>::XFace ); \
  declare_methods   ( FaceTypes<VOLT>::YFace ); \
  declare_methods   ( FaceTypes<VOLT>::ZFace ); \
  declare_wrap      ( VOLT                   ); \
  declare_wrap      ( FaceTypes<VOLT>::XFace ); \
  declare_wrap      ( FaceTypes<VOLT>::YFace ); \
  declare_wrap      ( FaceTypes<VOLT>::ZFace ); \
  declare_const_wrap( VOLT                   ); \
  declare_const_wrap( FaceTypes<VOLT>::XFace ); \
  declare_const_wrap( FaceTypes<VOLT>::YFace ); \
  declare_const_wrap( FaceTypes<VOLT>::ZFace );

  declare_variants( SS::SVolField );
  declare_variants( SS::XVolField );
  declare_variants( SS::YVolField );
  declare_variants( SS::ZVolField );

  template SS::SingleValueField*
  wrap_uintah_field_as_spatialops<SS::SingleValueField,Uintah::PerPatch<double*> >(
      Uintah::PerPatch<double*>&,
      const AllocInfo&,
      const SpatialOps::MemoryType,
      const unsigned short int,double* );
  template SS::SingleValueField*
  wrap_uintah_field_as_spatialops<SS::SingleValueField,Uintah::ReductionVariableBase>(
      Uintah::ReductionVariableBase&,
      const AllocInfo&,
      const SpatialOps::MemoryType,
      const unsigned short int,double* );
  declare_methods( SS::SingleValueField );

  declare_methods   ( ParticleField );
  declare_wrap      ( ParticleField );
  declare_const_wrap( ParticleField );

  //------------------------------------------------------------------

} // namespace Wasatch
