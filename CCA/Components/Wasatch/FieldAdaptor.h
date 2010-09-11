#ifndef Wasatch_FieldAdaptor_h
#define Wasatch_FieldAdaptor_h

#include <spatialops/structured/FVStaggeredTypes.h>

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

  template<> struct SelectUintahFieldType<SpatialOps::structured::SVolField  >{ typedef Uintah::CCVariable  <double> type;  typedef Uintah::constCCVariable  <double> const_type; };
  template<> struct SelectUintahFieldType<SpatialOps::structured::SSurfXField>{ typedef Uintah::SFCXVariable<double> type;  typedef Uintah::constSFCXVariable<double> const_type; };
  template<> struct SelectUintahFieldType<SpatialOps::structured::SSurfYField>{ typedef Uintah::SFCYVariable<double> type;  typedef Uintah::constSFCYVariable<double> const_type; };
  template<> struct SelectUintahFieldType<SpatialOps::structured::SSurfZField>{ typedef Uintah::SFCZVariable<double> type;  typedef Uintah::constSFCZVariable<double> const_type; };

  template<> struct SelectUintahFieldType<SpatialOps::structured::XVolField  >{ typedef Uintah::SFCXVariable<double> type;  typedef Uintah::constSFCXVariable<double> const_type; };
  template<> struct SelectUintahFieldType<SpatialOps::structured::XSurfXField>{ typedef Uintah::CCVariable  <double> type;  typedef Uintah::constCCVariable  <double> const_type; };
  template<> struct SelectUintahFieldType<SpatialOps::structured::XSurfYField>{ typedef Uintah::SFCYVariable<double> type;  typedef Uintah::constSFCYVariable<double> const_type; };
  template<> struct SelectUintahFieldType<SpatialOps::structured::XSurfZField>{ typedef Uintah::SFCZVariable<double> type;  typedef Uintah::constSFCZVariable<double> const_type; };

  template<> struct SelectUintahFieldType<SpatialOps::structured::YVolField  >{ typedef Uintah::SFCYVariable<double> type;  typedef Uintah::constSFCYVariable<double> const_type; };
  template<> struct SelectUintahFieldType<SpatialOps::structured::YSurfXField>{ typedef Uintah::SFCXVariable<double> type;  typedef Uintah::constSFCXVariable<double> const_type; };
  template<> struct SelectUintahFieldType<SpatialOps::structured::YSurfYField>{ typedef Uintah::CCVariable  <double> type;  typedef Uintah::constCCVariable  <double> const_type; };
  template<> struct SelectUintahFieldType<SpatialOps::structured::YSurfZField>{ typedef Uintah::SFCZVariable<double> type;  typedef Uintah::constSFCZVariable<double> const_type; };

  template<> struct SelectUintahFieldType<SpatialOps::structured::ZVolField  >{ typedef Uintah::SFCZVariable<double> type;  typedef Uintah::constSFCZVariable<double> const_type; };
  template<> struct SelectUintahFieldType<SpatialOps::structured::ZSurfXField>{ typedef Uintah::SFCXVariable<double> type;  typedef Uintah::constSFCXVariable<double> const_type; };
  template<> struct SelectUintahFieldType<SpatialOps::structured::ZSurfYField>{ typedef Uintah::SFCYVariable<double> type;  typedef Uintah::constSFCYVariable<double> const_type; };
  template<> struct SelectUintahFieldType<SpatialOps::structured::ZSurfZField>{ typedef Uintah::CCVariable  <double> type;  typedef Uintah::constCCVariable  <double> const_type; };


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
                                                                                                                            
  template<> inline Uintah::Ghost::GhostType getUintahGhostType<SpatialOps::structured::XVolField  >(){ return Uintah::Ghost::AroundFaces; }
  template<> inline Uintah::Ghost::GhostType getUintahGhostType<SpatialOps::structured::XSurfXField>(){ return Uintah::Ghost::AroundCells; }
  template<> inline Uintah::Ghost::GhostType getUintahGhostType<SpatialOps::structured::XSurfYField>(){ return Uintah::Ghost::AroundFaces; }
  template<> inline Uintah::Ghost::GhostType getUintahGhostType<SpatialOps::structured::XSurfZField>(){ return Uintah::Ghost::AroundFaces; }
                                                                                                                            
  template<> inline Uintah::Ghost::GhostType getUintahGhostType<SpatialOps::structured::YVolField  >(){ return Uintah::Ghost::AroundFaces; }
  template<> inline Uintah::Ghost::GhostType getUintahGhostType<SpatialOps::structured::YSurfXField>(){ return Uintah::Ghost::AroundFaces; }
  template<> inline Uintah::Ghost::GhostType getUintahGhostType<SpatialOps::structured::YSurfYField>(){ return Uintah::Ghost::AroundCells; }
  template<> inline Uintah::Ghost::GhostType getUintahGhostType<SpatialOps::structured::YSurfZField>(){ return Uintah::Ghost::AroundFaces; }
                                                                                                                            
  template<> inline Uintah::Ghost::GhostType getUintahGhostType<SpatialOps::structured::ZVolField  >(){ return Uintah::Ghost::AroundFaces; }
  template<> inline Uintah::Ghost::GhostType getUintahGhostType<SpatialOps::structured::ZSurfXField>(){ return Uintah::Ghost::AroundFaces; }
  template<> inline Uintah::Ghost::GhostType getUintahGhostType<SpatialOps::structured::ZSurfYField>(){ return Uintah::Ghost::AroundFaces; }
  template<> inline Uintah::Ghost::GhostType getUintahGhostType<SpatialOps::structured::ZSurfZField>(){ return Uintah::Ghost::AroundCells; }

}

#endif // Wasatch_FieldAdaptor_h
