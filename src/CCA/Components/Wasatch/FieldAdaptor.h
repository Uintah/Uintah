#ifndef Wasatch_FieldAdaptor_h
#define Wasatch_FieldAdaptor_h

#include <spatialops/structured/FVStaggeredFieldTypes.h>
#include <spatialops/structured/MemoryWindow.h>

#include <Core/Grid/Variables/SFCXVariable.h>  /* x-face variable */
#include <Core/Grid/Variables/SFCYVariable.h>  /* y-face variable */
#include <Core/Grid/Variables/SFCZVariable.h>  /* z-face variable */
#include <Core/Grid/Variables/CCVariable.h>    /* cell variable   */
#include <Core/Disclosure/TypeDescription.h>

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

namespace Uintah{ class Patch; }

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
    ZVOL, ZSURFX, ZSURFY, ZSURFZ
  };

  /**
   *  \ingroup WasatchFields
   *  \brief obtain the memory window for a uintah field that is to be wrapped as a SpatialOps field
   *  \param globalSize - the full size of the parent field.
   *  \param patch - the patch that the field is associated with.
   */
  template< typename FieldT >
  SpatialOps::structured::MemoryWindow
  get_memory_window_for_uintah_field( const SCIRun::IntVector& globSize,
                                      const Uintah::Patch* const patch );

  /**
   *  \ingroup WasatchFields
   *
   *  \brief wrap a uintah field to obtain a SpatialOps field,
   *         returning a new pointer.  The caller is responsible for
   *         freeing the memory.
   *  \param uintahVar - the uintah variable to wrap
   *  \param patch - the patch that the field is associated with.
   */
  template< typename FieldT, typename UFT >
  inline FieldT* wrap_uintah_field_as_spatialops( UFT& uintahVar,
                                                  const Uintah::Patch* const patch )
  {
    using SCIRun::IntVector;

    return new FieldT( get_memory_window_for_uintah_field<FieldT>( uintahVar.getWindow()->getData()->size(), patch ),
                       const_cast<double*>( uintahVar.getPointer() ),
                       SpatialOps::structured::ExternalStorage );
  }

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

  typedef double AtomicT;

#define DECLARE_SELECT_FIELD_TYPE_STRUCT( VOLT )                        \
  template<> struct SelectUintahFieldType< VOLT >{                      \
    typedef Uintah::CCVariable     <AtomicT>  type;                     \
    typedef Uintah::constCCVariable<AtomicT>  const_type;               \
  };                                                                    \
  template<> struct SelectUintahFieldType< SpatialOps::structured::FaceTypes<VOLT>::XFace >{ \
    typedef Uintah::SFCXVariable     <AtomicT>  type;                   \
    typedef Uintah::constSFCXVariable<AtomicT>  const_type;             \
  };                                                                    \
  template<> struct SelectUintahFieldType< SpatialOps::structured::FaceTypes<VOLT>::YFace >{ \
    typedef Uintah::SFCYVariable     <AtomicT>  type;                   \
    typedef Uintah::constSFCYVariable<AtomicT>  const_type;             \
  };                                                                    \
  template<> struct SelectUintahFieldType< SpatialOps::structured::FaceTypes<VOLT>::ZFace >{ \
    typedef Uintah::SFCZVariable     <AtomicT>  type;                   \
    typedef Uintah::constSFCZVariable<AtomicT>  const_type;             \
  };
  
  DECLARE_SELECT_FIELD_TYPE_STRUCT( SpatialOps::structured::SVolField );
  DECLARE_SELECT_FIELD_TYPE_STRUCT( SpatialOps::structured::XVolField );
  DECLARE_SELECT_FIELD_TYPE_STRUCT( SpatialOps::structured::YVolField );
  DECLARE_SELECT_FIELD_TYPE_STRUCT( SpatialOps::structured::ZVolField );

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
   *         field type.
   */
  template<typename FieldT> inline int get_n_ghost(){ return FieldT::Ghost::NGHOST; }
  template<> inline int get_n_ghost<double>(){ return 0; };

  /**
   *  \ingroup WasatchFields
   *  \brief Obtain the number of ghost cells in each direction for
   *         the given SpatialOps field type as a template parameter.
   *
   *  \return the Uintah::IntVector describing the number of ghost
   *          cells in each direction.
   */  
  template<typename FieldT>
  inline Uintah::IntVector get_uintah_ghost_descriptor()
  {
    const int ng = get_n_ghost<FieldT>();
    return Uintah::IntVector(ng,ng,ng);
  }

  //====================================================================

  /**
   *  \ingroup WasatchFields
   *  \brief Given the SpatialOps field type as a template parameter,
   *         determine the Uintah GhostType information.
   *
   *  \return The Uintah::Ghost::GhostType for this field type.
   */
  template<typename FieldT> Uintah::Ghost::GhostType get_uintah_ghost_type();
}

#endif // Wasatch_FieldAdaptor_h
