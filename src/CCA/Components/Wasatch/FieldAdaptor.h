#ifndef Wasatch_FieldAdaptor_h
#define Wasatch_FieldAdaptor_h

#include <spatialops/structured/FVStaggeredTypes.h>

#include <Core/Grid/Variables/SFCXVariable.h>  /* x-face variable */
#include <Core/Grid/Variables/SFCYVariable.h>  /* y-face variable */
#include <Core/Grid/Variables/SFCZVariable.h>  /* z-face variable */
#include <Core/Grid/Variables/CCVariable.h>    /* cell variable   */
#include <Core/Disclosure/TypeDescription.h>

namespace Wasatch{

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


  template<typename FieldT>
  inline const Uintah::TypeDescription* getUintahFieldTypeDescriptor()
  {
    return SelectUintahFieldType<FieldT>::type::getTypeDescription();
  }

  template<typename FieldT> inline int getNGhost(){ return FieldT::Ghost::NM; }
  template<> inline int getNGhost<double>(){ return 0; };
  
  template<typename FieldT>
  inline Uintah::IntVector getUintahGhostDescriptor()
  {
    const int ng = getNGhost<FieldT>();
    return Uintah::IntVector(ng,ng,ng);
  }

  //====================================================================

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
