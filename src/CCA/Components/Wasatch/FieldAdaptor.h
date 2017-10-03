/*
 * The MIT License
 *
 * Copyright (c) 2012-2017 The University of Utah
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

#ifndef Wasatch_FieldAdaptor_h
#define Wasatch_FieldAdaptor_h

#include <spatialops/structured/FVStaggeredFieldTypes.h>
#include <spatialops/structured/SpatialFieldStore.h>

#include <Core/Grid/Variables/SFCXVariable.h>  /* x-face variable */
#include <Core/Grid/Variables/SFCYVariable.h>  /* y-face variable */
#include <Core/Grid/Variables/SFCZVariable.h>  /* z-face variable */
#include <Core/Grid/Variables/CCVariable.h>    /* cell variable   */
#include <Core/Grid/Variables/PerPatch.h>      /* single double per patch */
#include <Core/Grid/Variables/ReductionVariable.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Disclosure/TypeDescription.h>

#include <CCA/Ports/DataWarehouse.h>

#include <sci_defs/uintah_defs.h>
#include <sci_defs/cuda_defs.h>


#include <CCA/Components/Wasatch/FieldTypes.h>


/**
 *  \file FieldAdaptor.h
 *
 *  \brief provides tools to translate between Uintah field types and
 *  SpatialOps-compatible field types.
 *
 *  This information is provided for the interface to the Expression
 *  library.  There should be no reason to use it otherwise.
 */

namespace Uintah{ class Patch; class ProcessorGroup; }

namespace WasatchCore{

  /**
   *  \ingroup WasatchParser
   *  \enum FieldTypes
   *  \brief Enumerate the field types in Wasatch.
   */
  enum FieldTypes{
    SVOL, SSURFX, SSURFY, SSURFZ,
    XVOL, XSURFX, XSURFY, XSURFZ,
    YVOL, YSURFX, YSURFY, YSURFZ,
    ZVOL, ZSURFX, ZSURFY, ZSURFZ,
    PERPATCH, PARTICLE
  };

  /**
   * \fn void get_bc_logicals( const Uintah::Patch* const, SpatialOps::IntVec&, SpatialOps::IntVec& );
   * \brief Given the patch, populate information about whether a physical
   *        boundary exists on each side of the patch.
   * \param patch   - the patch of interest
   * \param bcMinus - assigned to 0 if no BC present on (-) faces, 1 if present
   * \param bcPlus  - assigned to 0 if no BC present on (+) faces, 1 if present
   */
  void get_bc_logicals( const Uintah::Patch* const patch,
                        SpatialOps::IntVec& bcMinus,
                        SpatialOps::IntVec& bcPlus );

  /**
   *  \ingroup WasatchFields
   *  \brief obtain the memory window for a uintah field that is to be wrapped as a SpatialOps field
   *  \param patch - the patch that the field is associated with.
   *
   *  Note that this method is only intended for use by the UintahFieldManager
   *  when obtaining scratch fields from patch information. When an actual field
   *  from Uintah is available, you should use wrap_uintah_field_as_spatialops
   */
  template<typename FieldT>
  SpatialOps::MemoryWindow
  get_memory_window_for_uintah_field( const Uintah::Patch* const patch );

  /**
   *  \ingroup WasatchFields
   *  \brief Obtain a memory window that can be used to construct a SpatialMask for FieldT
   *  \param patch - The patch that the field is associated with.
   */
  template<typename FieldT>
  SpatialOps::MemoryWindow
  get_memory_window_for_masks( const Uintah::Patch* const patch );

  /**
   *  \ingroup WasatchParser
   *  \brief translate a string describing a field type to the FieldTypes enum.
   */
  FieldTypes get_field_type( std::string );

  /**
   *  \ingroup WasatchFields
   *  \struct SelectUintahFieldType
   *  \brief Convert SpatialOps field types to Uintah field types
   *
   *  This struct template provides two typedefs that define Uintah
   *  field types from SpatialOps field types:
   *   - \c type : the Uintah type
   *   - \c const_type : the Uintah const field type.
   */
  template<typename FieldT> struct SelectUintahFieldType;

  template<> struct SelectUintahFieldType<SpatialOps::SingleValueField>{
    typedef Uintah::PerPatch<double> type;
    typedef Uintah::PerPatch<double> const_type;
  };

  template<> struct SelectUintahFieldType<int>{
    typedef Uintah::     CCVariable<int>  type;
    typedef Uintah::constCCVariable<int>  const_type;
  };

  template<> struct SelectUintahFieldType<SpatialOps::SVolField>{
    typedef Uintah::     CCVariable<double>  type;
    typedef Uintah::constCCVariable<double>  const_type;
  };
  template<> struct SelectUintahFieldType<SpatialOps::SSurfXField>{
    typedef Uintah::     SFCXVariable<double>  type;
    typedef Uintah::constSFCXVariable<double>  const_type;
  };
  template<> struct SelectUintahFieldType<SpatialOps::SSurfYField>{
    typedef Uintah::     SFCYVariable<double>  type;
    typedef Uintah::constSFCYVariable<double>  const_type;
  };
  template<> struct SelectUintahFieldType<SpatialOps::SSurfZField>{
    typedef Uintah::     SFCZVariable<double>  type;
    typedef Uintah::constSFCZVariable<double>  const_type;
  };

  template<> struct SelectUintahFieldType<SpatialOps::XVolField>{
    typedef Uintah::     SFCXVariable<double>  type;
    typedef Uintah::constSFCXVariable<double>  const_type;
  };
  template<> struct SelectUintahFieldType<SpatialOps::XSurfXField>{
    typedef Uintah::     CCVariable<double>  type;
    typedef Uintah::constCCVariable<double>  const_type;
  };
  template<> struct SelectUintahFieldType<SpatialOps::XSurfYField>{
    typedef Uintah::     SFCYVariable<double>  type;
    typedef Uintah::constSFCYVariable<double>  const_type;
  };
  template<> struct SelectUintahFieldType<SpatialOps::XSurfZField>{
    typedef Uintah::     SFCZVariable<double>  type;
    typedef Uintah::constSFCZVariable<double>  const_type;
  };

  template<> struct SelectUintahFieldType<SpatialOps::YVolField>{
     typedef Uintah::     SFCYVariable<double>  type;
     typedef Uintah::constSFCYVariable<double>  const_type;
   };
   template<> struct SelectUintahFieldType<SpatialOps::YSurfXField>{
     typedef Uintah::     SFCXVariable<double>  type;
     typedef Uintah::constSFCXVariable<double>  const_type;
   };
   template<> struct SelectUintahFieldType<SpatialOps::YSurfYField>{
     typedef Uintah::     CCVariable<double>  type;
     typedef Uintah::constCCVariable<double>  const_type;
   };
   template<> struct SelectUintahFieldType<SpatialOps::YSurfZField>{
     typedef Uintah::     SFCZVariable<double>  type;
     typedef Uintah::constSFCZVariable<double>  const_type;
   };

   template<> struct SelectUintahFieldType<SpatialOps::ZVolField>{
     typedef Uintah::     SFCZVariable<double>  type;
     typedef Uintah::constSFCZVariable<double>  const_type;
   };
   template<> struct SelectUintahFieldType<SpatialOps::ZSurfXField>{
     typedef Uintah::     SFCXVariable<double>  type;
     typedef Uintah::constSFCXVariable<double>  const_type;
   };
   template<> struct SelectUintahFieldType<SpatialOps::ZSurfYField>{
     typedef Uintah::     SFCYVariable<double>  type;
     typedef Uintah::constSFCYVariable<double>  const_type;
   };
   template<> struct SelectUintahFieldType<SpatialOps::ZSurfZField>{
     typedef Uintah::     CCVariable<double>  type;
     typedef Uintah::constCCVariable<double>  const_type;
   };

   // currently, particle fields are only supported for double, not int or Point types.
   template<> struct SelectUintahFieldType<SpatialOps::Particle::ParticleField>{
     typedef Uintah::     ParticleVariable<double>  type;
     typedef Uintah::constParticleVariable<double>  const_type;
   };

  /**
   *  \ingroup WasatchFields
   *  \brief Given the SpatialOps field type, this returns the
   *         Uintah::TypeDescription for the corresponding Uintah
   *         field type.
   */
  template<typename FieldT>
  inline const Uintah::TypeDescription* get_uintah_field_type_descriptor()
  {
    return SelectUintahFieldType<FieldT>::type::getTypeDescription();
  }

  /**
   *  \ingroup WasatchFields
   *  \brief Obtain the number of ghost cells for a given SpatialOps
   *         field type, assuming that there are the same number of
   *         ghost cells in each direction and on the (+) side as the
   *         (-) side.
   */
  template<typename FieldT> inline int get_n_ghost(){
    return 1;
  }

  template<> inline int get_n_ghost<SpatialOps::SingleValueField>(){
    return 0;
  };

  template<> inline int get_n_ghost<SpatialOps::Particle::ParticleField>(){
    return 0;
  };

  //====================================================================

  /**
   *  \ingroup WasatchFields
   *  \brief Given the SpatialOps field type as a template parameter,
   *         determine the Uintah GhostType information.
   *
   *  \return The Uintah::Ghost::GhostType for this field type.
   *
   *  Note that this is specialized for each of the supported types of fields in Wasatch.
   */
  template<typename FieldT> Uintah::Ghost::GhostType get_uintah_ghost_type();


  /**
   *  This is used to pass required information through to the FieldManager::allocate_fields() method.
   */
  struct AllocInfo
  {
    Uintah::DataWarehouse* const oldDW;
    Uintah::DataWarehouse* const newDW;
    const int materialIndex;
    const Uintah::Patch* const patch;
    Uintah::ParticleSubset* const pset;
    const Uintah::ProcessorGroup* const procgroup;
    const bool isGPUTask;

    AllocInfo( Uintah::DataWarehouse* const olddw,
               Uintah::DataWarehouse* const newdw,
               const int matlIndex,
               const Uintah::Patch* p,
               Uintah::ParticleSubset* particlesubset,
               const Uintah::ProcessorGroup* const pg,
               const bool isgpu=false )
    : oldDW( olddw ),
      newDW( newdw ),
      materialIndex( matlIndex ),
      patch( p ),
      pset (particlesubset),
      procgroup( pg ),
      isGPUTask( isgpu )
    {}
    
    AllocInfo( Uintah::DataWarehouse* const olddw,
              Uintah::DataWarehouse* const newdw,
              const int matlIndex,
              const Uintah::Patch* p,
              const Uintah::ProcessorGroup* const pg,
              const bool isgpu=false )
    : oldDW( olddw ),
    newDW( newdw ),
    materialIndex( matlIndex ),
    patch( p ),
    pset (nullptr),
    procgroup( pg ),
    isGPUTask( isgpu )
    {}
    
  };


  /**
   *  \ingroup WasatchFields
   *
   *  \brief wrap a uintah field to obtain a SpatialOps field,
   *         returning a new pointer.  The caller is responsible for
   *         freeing the memory.
   *  \param uintahVar the uintah variable to wrap
   *  \param ainfo information required about the size of the field
   *  \param ghostData information about the number of ghosts required for this field
   *  \param deviceIndex in the case of a GPU field, this specifies which GPU it is on
   *  \param uintahDeviceVar for GPU fields, this is the pointer to the field on the device
   *
   *  \tparam FieldT the SpatialOps field type to produce
   *  \tparam UFT the Uintah field type that we began with
   *
   *  \todo use type inference to go between FieldT and UFT.  Note that this is tied into ExprLib.
   */
  template< typename FieldT, typename UFT >
  inline
  SpatialOps::SpatFldPtr<FieldT>
  wrap_uintah_field_as_spatialops( UFT& uintahVar,
                                   const AllocInfo& ainfo,
                                   const SpatialOps::GhostData ghostData,
                                   short int deviceIndex=CPU_INDEX,
                                   double* uintahDeviceVar = nullptr )
  {
    /*
     * NOTE: before changing things here, look at the line:
     *    Uintah::OnDemandDataWarehouse::d_combineMemory = false;
     * in Wasatch.cc.  This is currently preventing Uintah from
     * combining patch memory.
     */
    namespace so = SpatialOps;

    using Uintah::IntVector;

    const Uintah::IntVector lowIx       = uintahVar.getLowIndex();
    const Uintah::IntVector highIx      = uintahVar.getHighIndex();
    const Uintah::IntVector fieldSize   = uintahVar.getWindow()->getData()->size();
    const Uintah::IntVector fieldOffset = uintahVar.getWindow()->getOffset();
    const Uintah::IntVector fieldExtent = highIx - lowIx;
    
    const so::IntVec   size(   fieldSize[0],   fieldSize[1],   fieldSize[2] );
    const so::IntVec extent( fieldExtent[0], fieldExtent[1], fieldExtent[2] );
    const so::IntVec offset( lowIx[0]-fieldOffset[0], lowIx[1]-fieldOffset[1], lowIx[2]-fieldOffset[2] );

    so::IntVec bcMinus, bcPlus;
    get_bc_logicals( ainfo.patch, bcMinus, bcPlus );
    
    // In certain cases, when the number of ghosts is different from the number of extra cells,
    // one must change the ghostData to reflect this discrepancy. The general rule is that, at physical boundaries,
    // the number of extra cells supersedes the number of ghost cells. At processor boundaries, then number
    // of ghost cells takes over.
    const Uintah::IntVector extraCells = ainfo.patch->getExtraCells(); // get the extra cells associated with this patch
    SpatialOps::GhostData newGData = ghostData; // copy ghost data
    so::IntVec gMinus = ghostData.get_minus();
    so::IntVec gPlus = ghostData.get_plus();
    for (int i = 0; i < 3; ++i) {
      gMinus[i] = bcMinus[i] == 0 ? gMinus[i] : extraCells[i];
      gPlus[i]  = bcPlus[i]  == 0 ? gPlus[i]  : extraCells[i];
    }
    newGData.set_minus(gMinus);
    newGData.set_plus(gPlus);
    
    double* fieldValues_ = nullptr;
    FieldT* field;
    if( ainfo.isGPUTask && IS_GPU_INDEX(deviceIndex) ){ // homogeneous GPU task
      assert( uintahDeviceVar != nullptr );
      fieldValues_ = uintahDeviceVar;
      field = new FieldT( so::MemoryWindow( size, offset, extent ),
                          so::BoundaryCellInfo::build<FieldT>(bcMinus,bcPlus),
                          newGData,
                          fieldValues_,
                          so::ExternalStorage,
                          deviceIndex );
    }
    else{ // heterogeneous task
      fieldValues_ = const_cast<typename FieldT::value_type*>( uintahVar.getPointer() );
      field = new FieldT( so::MemoryWindow( size, offset, extent ),
                          so::BoundaryCellInfo::build<FieldT>(bcMinus,bcPlus),
                          newGData,
                          fieldValues_,
                          so::ExternalStorage,
                          CPU_INDEX );
#     ifdef HAVE_CUDA
      if(IS_GPU_INDEX(deviceIndex)) field->add_device(GPU_INDEX);
#     endif
    }

    return so::SpatFldPtr<FieldT>(field);
  }

  //-----------------------------------------------------------------

  template< > inline
  SpatialOps::SpatFldPtr<ParticleField>
  wrap_uintah_field_as_spatialops<ParticleField,SelectUintahFieldType<ParticleField>::type>(
      SelectUintahFieldType<ParticleField>::type& uintahVar,
      const AllocInfo& ainfo,
      const SpatialOps::GhostData ghostData,
      const short int deviceIndex,
      double* uintahDeviceVar ) // abhi : not being used yet)
  {
    namespace so = SpatialOps;
    typedef ParticleField::value_type ValT;
    ValT* fieldValues = nullptr;
    if( IS_GPU_INDEX(deviceIndex) ){
#     ifdef HAVE_CUDA
      fieldValues = const_cast<ValT*>( uintahDeviceVar );
#     endif
    }
    else{
      fieldValues = const_cast<ParticleField::value_type*>( (ValT*)uintahVar.getBasePointer() );
    }

    const int npar = ainfo.oldDW
        ? ainfo.oldDW->getParticleSubset( ainfo.materialIndex, ainfo.patch )->numParticles()
        : ainfo.newDW->getParticleSubset( ainfo.materialIndex, ainfo.patch )->numParticles();

    return so::SpatFldPtr<ParticleField>(
        new ParticleField( so::MemoryWindow( so::IntVec(npar,1,1) ),
                           so::BoundaryCellInfo::build<ParticleField>(),
                           ghostData,
                           fieldValues,
                           so::ExternalStorage,
                           deviceIndex ) );
  }

  template< > inline
  SpatialOps::SpatFldPtr<ParticleField>
  wrap_uintah_field_as_spatialops<ParticleField,SelectUintahFieldType<ParticleField>::const_type>(
      SelectUintahFieldType<ParticleField>::const_type& uintahVar,
      const AllocInfo& ainfo,
      const SpatialOps::GhostData ghostData,
      const short int deviceIndex,
      double* uintahDeviceVar ) // abhi : not being used yet)
  {
    namespace so = SpatialOps;
    typedef ParticleField::value_type ValT;
    ValT* fieldValues = nullptr;
    if( IS_GPU_INDEX(deviceIndex) ){
#     ifdef HAVE_CUDA
      fieldValues = const_cast<ValT*>( uintahDeviceVar );
#     endif
    }
    else{
      fieldValues = const_cast<ValT*>( (ValT*)uintahVar.getBaseRep().getBasePointer() );
    }

    const int npar = ainfo.oldDW
        ? ainfo.oldDW->getParticleSubset( ainfo.materialIndex, ainfo.patch )->numParticles()
        : ainfo.newDW->getParticleSubset( ainfo.materialIndex, ainfo.patch )->numParticles();

    return so::SpatFldPtr<ParticleField>(
        new ParticleField( so::MemoryWindow( so::IntVec(npar,1,1) ),
                           so::BoundaryCellInfo::build<ParticleField>(),
                           ghostData,
                           fieldValues,
                           so::ExternalStorage,
                           deviceIndex ) );
  }

  //-----------------------------------------------------------------
  // NOTE: this wraps a raw uintah field type, whereas the default
  //       implementations work with an Expr::UintahFieldContainer
  // Default arguments cannot be passed to Explicit Template Specialization
  template<>
  inline
  SpatialOps::SpatFldPtr<SpatialOps::SingleValueField>
  wrap_uintah_field_as_spatialops<SpatialOps::SingleValueField,Uintah::PerPatch<double> >(
      Uintah::PerPatch<double>& uintahVar,
      const AllocInfo& ainfo,
      const SpatialOps::GhostData ghostData,
      const short int deviceIndex,
      double* uintahDeviceVar ) // abhi : not being used yet
  {
    namespace so = SpatialOps;
    typedef so::SingleValueField FieldT;
    const so::IntVec noBC(false,false,false);    // bc doesn't matter for single value fields
    return so::SpatFldPtr<FieldT>(
        new FieldT( so::MemoryWindow( so::IntVec(1,1,1), so::IntVec(0,0,0), so::IntVec(1,1,1) ),
                    so::BoundaryCellInfo::build<FieldT>(noBC,noBC),
                    ghostData,
                    &uintahVar.get(),
                    so::ExternalStorage,
                    deviceIndex ) );
  }

  // NOTE: this wraps a raw uintah field type, whereas the default
  //       implementations work with an Expr::UintahFieldContainer
  template<>
  inline
  SpatialOps::SpatFldPtr<SpatialOps::SingleValueField>
  wrap_uintah_field_as_spatialops<SpatialOps::SingleValueField,Uintah::ReductionVariableBase>(
      Uintah::ReductionVariableBase& uintahVar,
      const AllocInfo& ainfo,
      const SpatialOps::GhostData ghostData,
      const short int deviceIndex,
      double* uintahDeviceVar ) // abhi : not being used yet
  {
    namespace so = SpatialOps;
    typedef so::SingleValueField FieldT;
    const so::IntVec noBC(false,false,false);    // bc doesn't matter for single value fields
    return so::SpatFldPtr<FieldT>(
        new FieldT( so::MemoryWindow( so::IntVec(1,1,1), so::IntVec(0,0,0), so::IntVec(1,1,1) ),
                    so::BoundaryCellInfo::build<FieldT>(noBC,noBC),
                    ghostData,
                    (double*)( uintahVar.getBasePointer() ),  // jcs this is a bit sketchy because of the type casting.  It will only work for reductions on doubles
                    so::ExternalStorage,
                    deviceIndex ) );
  }

}

#endif // Wasatch_FieldAdaptor_h
