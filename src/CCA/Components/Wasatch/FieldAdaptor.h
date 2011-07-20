#ifndef Wasatch_FieldAdaptor_h
#define Wasatch_FieldAdaptor_h

#include <spatialops/structured/FVStaggeredFieldTypes.h>
#include <spatialops/structured/MemoryWindow.h>

#include <Core/Grid/Variables/SFCXVariable.h>  /* x-face variable */
#include <Core/Grid/Variables/SFCYVariable.h>  /* y-face variable */
#include <Core/Grid/Variables/SFCZVariable.h>  /* z-face variable */
#include <Core/Grid/Variables/CCVariable.h>    /* cell variable   */
#include <Core/Disclosure/TypeDescription.h>


/**
 *  \file FieldAdaptor.h
 *
 *  \brief provides tools to translate between Uintah field types and
 *  SpatialOps-compatible field types.
 *
 *  This information is provided for the interface to the Expression
 *  library.  There should be no reason to use it otherwise.
 */

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
   *
   *  \brief wrap a uintah field to obtain a SpatialOps field,
   *         returning a new pointer.  The caller is responsible for
   *         freeing the memory.
   */
  template< typename FieldT, typename UFT >
  inline FieldT* wrap_uintah_field_as_spatialops( UFT& uintahVar )
  {
    using SCIRun::IntVector;
    const IntVector uvarGlobSize = uintahVar.getWindow()->getData()->size();
    const IntVector uvarHi       = uintahVar.getWindow()->getHighIndex();
    const IntVector uvarLo       = uintahVar.getWindow()->getLowIndex();
    const IntVector uvarOffset   = uintahVar.getWindow()->getOffset();
    SpatialOps::structured::IntVec varGlobSize(1,1,1), varExtent(1,1,1), varOffset(0,0,0);
    for( size_t i=0; i<3; ++i ){
      varGlobSize[i] = uvarGlobSize[i];
      if( uvarGlobSize[i]>1 ){
        varExtent[i] = uvarHi[i] - uvarLo[i];
        varOffset[i] = uvarLo[i] - uvarOffset[i];
      }
    }

    return new FieldT( SpatialOps::structured::MemoryWindow( varGlobSize, varOffset, varExtent ),
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

#define DECLARE_SELECT_FIELD_TYPE_STRUCT( VOLT )                \
  template<> struct SelectUintahFieldType< VOLT >{              \
    typedef Uintah::CCVariable     <AtomicT>  type;             \
    typedef Uintah::constCCVariable<AtomicT>  const_type;       \
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
  inline const Uintah::TypeDescription* getUintahFieldTypeDescriptor()
  {
    return SelectUintahFieldType<FieldT>::type::getTypeDescription();
  }

  /**
   *  \ingroup WasatchFields
   *  \brief Obtain the number of ghost cells for a given SpatialOps
   *         field type.
   */
  template<typename FieldT> inline int getNGhost(){ return FieldT::Ghost::NGHOST; }
  template<> inline int getNGhost<double>(){ return 0; };

  /**
   *  \ingroup WasatchFields
   *  \brief Obtain the number of ghost cells in each direction for
   *         the given SpatialOps field type as a template parameter.
   *
   *  \return the Uintah::IntVector describing the number of ghost
   *          cells in each direction.
   */  
  template<typename FieldT>
  inline Uintah::IntVector getUintahGhostDescriptor()
  {
    const int ng = getNGhost<FieldT>();
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
  template<typename FieldT> Uintah::Ghost::GhostType getUintahGhostType();

  template<> inline Uintah::Ghost::GhostType getUintahGhostType<SpatialOps::structured::SVolField  >(){ return Uintah::Ghost::AroundCells; }
  template<> inline Uintah::Ghost::GhostType getUintahGhostType<SpatialOps::structured::SSurfXField>(){ return Uintah::Ghost::AroundFaces; }
  template<> inline Uintah::Ghost::GhostType getUintahGhostType<SpatialOps::structured::SSurfYField>(){ return Uintah::Ghost::AroundFaces; }
  template<> inline Uintah::Ghost::GhostType getUintahGhostType<SpatialOps::structured::SSurfZField>(){ return Uintah::Ghost::AroundFaces; }

  template<> inline Uintah::Ghost::GhostType getUintahGhostType<SpatialOps::structured::XVolField  >(){ return Uintah::Ghost::AroundCells; }
  template<> inline Uintah::Ghost::GhostType getUintahGhostType<SpatialOps::structured::XSurfXField>(){ return Uintah::Ghost::AroundFaces; }
  template<> inline Uintah::Ghost::GhostType getUintahGhostType<SpatialOps::structured::XSurfYField>(){ return Uintah::Ghost::AroundFaces; }
  template<> inline Uintah::Ghost::GhostType getUintahGhostType<SpatialOps::structured::XSurfZField>(){ return Uintah::Ghost::AroundFaces; }

  template<> inline Uintah::Ghost::GhostType getUintahGhostType<SpatialOps::structured::YVolField  >(){ return Uintah::Ghost::AroundCells; }
  template<> inline Uintah::Ghost::GhostType getUintahGhostType<SpatialOps::structured::YSurfXField>(){ return Uintah::Ghost::AroundFaces; }
  template<> inline Uintah::Ghost::GhostType getUintahGhostType<SpatialOps::structured::YSurfYField>(){ return Uintah::Ghost::AroundFaces; }
  template<> inline Uintah::Ghost::GhostType getUintahGhostType<SpatialOps::structured::YSurfZField>(){ return Uintah::Ghost::AroundFaces; }

  template<> inline Uintah::Ghost::GhostType getUintahGhostType<SpatialOps::structured::ZVolField  >(){ return Uintah::Ghost::AroundCells; }
  template<> inline Uintah::Ghost::GhostType getUintahGhostType<SpatialOps::structured::ZSurfXField>(){ return Uintah::Ghost::AroundFaces; }
  template<> inline Uintah::Ghost::GhostType getUintahGhostType<SpatialOps::structured::ZSurfYField>(){ return Uintah::Ghost::AroundFaces; }
  template<> inline Uintah::Ghost::GhostType getUintahGhostType<SpatialOps::structured::ZSurfZField>(){ return Uintah::Ghost::AroundFaces; }

}

#endif // Wasatch_FieldAdaptor_h
