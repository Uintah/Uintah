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

#ifndef Wasatch_FieldAdaptor_h
#define Wasatch_FieldAdaptor_h

#include <spatialops/structured/FVStaggeredFieldTypes.h>
#include <spatialops/structured/MemoryWindow.h>

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

namespace Wasatch{

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
    PERPATCH
  };

  /**
   * \fn void get_bc_logicals( const Uintah::Patch* const, SpatialOps::structured::IntVec&, SpatialOps::structured::IntVec& );
   * \brief Given the patch, populate information about whether a physical
   *        boundary exists on each side of the patch.
   * \param patch   - the patch of interest
   * \param bcMinus - assigned to 0 if no BC present on (-) faces, 1 if present
   * \param bcPlus  - assigned to 0 if no BC present on (+) faces, 1 if present
   */
  void get_bc_logicals( const Uintah::Patch* const patch,
                        SpatialOps::structured::IntVec& bcMinus,
                        SpatialOps::structured::IntVec& bcPlus );

  /**
   * \ingroup WasatchFields
   * \brief This is used to pass required information through to the FieldManager::allocate_fields() method.
   */
  struct AllocInfo
  {
    Uintah::DataWarehouse* const oldDW;
    Uintah::DataWarehouse* const newDW;
    const int materialIndex;
    const Uintah::Patch* const patch;
    const Uintah::ProcessorGroup* const procgroup;
    AllocInfo( Uintah::DataWarehouse* const olddw,
               Uintah::DataWarehouse* const newdw,
               const int mi,
               const Uintah::Patch* p,
               const Uintah::ProcessorGroup* const pg )
    : oldDW( olddw ),
      newDW( newdw ),
      materialIndex( mi ),
      patch( p ),
      procgroup( pg )
    {}
  };

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
  SpatialOps::structured::MemoryWindow
  get_memory_window_for_uintah_field( const AllocInfo& ainfo );

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

  template<> struct SelectUintahFieldType<SpatialOps::structured::SingleValueField>{
    typedef Uintah::PerPatch<double> type;
    typedef Uintah::PerPatch<double> const_type;
  };

  template<> struct SelectUintahFieldType<int>{
    typedef Uintah::     CCVariable<int>  type;
    typedef Uintah::constCCVariable<int>  const_type;
  };

  template<> struct SelectUintahFieldType<SpatialOps::structured::SVolField>{
    typedef Uintah::     CCVariable<double>  type;
    typedef Uintah::constCCVariable<double>  const_type;
  };
  template<> struct SelectUintahFieldType<SpatialOps::structured::SSurfXField>{
    typedef Uintah::     SFCXVariable<double>  type;
    typedef Uintah::constSFCXVariable<double>  const_type;
  };
  template<> struct SelectUintahFieldType<SpatialOps::structured::SSurfYField>{
    typedef Uintah::     SFCYVariable<double>  type;
    typedef Uintah::constSFCYVariable<double>  const_type;
  };
  template<> struct SelectUintahFieldType<SpatialOps::structured::SSurfZField>{
    typedef Uintah::     SFCZVariable<double>  type;
    typedef Uintah::constSFCZVariable<double>  const_type;
  };

  template<> struct SelectUintahFieldType<SpatialOps::structured::XVolField>{
    typedef Uintah::     SFCXVariable<double>  type;
    typedef Uintah::constSFCXVariable<double>  const_type;
  };
  template<> struct SelectUintahFieldType<SpatialOps::structured::XSurfXField>{
    typedef Uintah::     CCVariable<double>  type;
    typedef Uintah::constCCVariable<double>  const_type;
  };
  template<> struct SelectUintahFieldType<SpatialOps::structured::XSurfYField>{
    typedef Uintah::     SFCYVariable<double>  type;
    typedef Uintah::constSFCYVariable<double>  const_type;
  };
  template<> struct SelectUintahFieldType<SpatialOps::structured::XSurfZField>{
    typedef Uintah::     SFCZVariable<double>  type;
    typedef Uintah::constSFCZVariable<double>  const_type;
  };

  template<> struct SelectUintahFieldType<SpatialOps::structured::YVolField>{
     typedef Uintah::     SFCYVariable<double>  type;
     typedef Uintah::constSFCYVariable<double>  const_type;
   };
   template<> struct SelectUintahFieldType<SpatialOps::structured::YSurfXField>{
     typedef Uintah::     SFCXVariable<double>  type;
     typedef Uintah::constSFCXVariable<double>  const_type;
   };
   template<> struct SelectUintahFieldType<SpatialOps::structured::YSurfYField>{
     typedef Uintah::     CCVariable<double>  type;
     typedef Uintah::constCCVariable<double>  const_type;
   };
   template<> struct SelectUintahFieldType<SpatialOps::structured::YSurfZField>{
     typedef Uintah::     SFCZVariable<double>  type;
     typedef Uintah::constSFCZVariable<double>  const_type;
   };

   template<> struct SelectUintahFieldType<SpatialOps::structured::ZVolField>{
     typedef Uintah::     SFCZVariable<double>  type;
     typedef Uintah::constSFCZVariable<double>  const_type;
   };
   template<> struct SelectUintahFieldType<SpatialOps::structured::ZSurfXField>{
     typedef Uintah::     SFCXVariable<double>  type;
     typedef Uintah::constSFCXVariable<double>  const_type;
   };
   template<> struct SelectUintahFieldType<SpatialOps::structured::ZSurfYField>{
     typedef Uintah::     SFCYVariable<double>  type;
     typedef Uintah::constSFCYVariable<double>  const_type;
   };
   template<> struct SelectUintahFieldType<SpatialOps::structured::ZSurfZField>{
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

  template<> inline int get_n_ghost<SpatialOps::structured::SingleValueField>(){
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
   *  \ingroup WasatchFields
   *
   *  \brief wrap a uintah field to obtain a SpatialOps field,
   *         returning a new pointer.  The caller is responsible for
   *         freeing the memory.
   *  \param uintahVar the uintah variable to wrap
   *  \param patch the patch that the field is associated with.
   *  \param mtype specifies the location for the field (GPU,CPU)
   *  \param deviceIndex in the case of a GPU field, this specifies which GPU it is on
   *  \param uintahDeviceVar for GPU fields, this is the pointer to the field on the device
   *
   *  \tparam FieldT the SpatialOps field type to produce
   *  \tparam UFT the Uintah field type that we began with
   *
   *  \todo use type inference to go between FieldT and UFT.  Note that this is tied into ExprLib.
   */
  template< typename FieldT, typename UFT >
  FieldT* wrap_uintah_field_as_spatialops( UFT& uintahVar,
                                           const AllocInfo& ainfo,
                                           const SpatialOps::MemoryType mtype=SpatialOps::LOCAL_RAM,
                                           const unsigned short int deviceIndex=0,
                                           double* uintahDeviceVar = NULL );

  //-----------------------------------------------------------------

}

#endif // Wasatch_FieldAdaptor_h
