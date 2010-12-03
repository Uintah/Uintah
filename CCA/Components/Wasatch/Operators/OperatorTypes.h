#ifndef Wasatch_OperatorTypes_h
#define Wasatch_OperatorTypes_h

#include "../FieldTypes.h"
#include "UpwindInterpolant.h"

#include <spatialops/structured/FVStaggered.h>

/**
 *  \file OperatorTypes.h
 */

namespace Wasatch{

  typedef SpatialOps::Divergence  Divergence;
  typedef SpatialOps::Gradient    Gradient;
  typedef SpatialOps::Interpolant Interpolant;

  /**
   *  \ingroup WasatchOperators
   *  \ingroup WasatchCore
   *  \struct OperatorTypeBuilder
   *  \brief Convenience definition for a SpatialOperator
   *
   *  Supplies a typedef defining \c type, which defines the operator.
   *
   *  This should not generally be used by programmers.  Rather, it is
   *  used internally here to define various operators.
   */
  template<typename Op, typename SrcT, typename DestT>
  struct OperatorTypeBuilder{
    typedef SpatialOps::SpatialOperator< LinAlg, Op, SrcT, DestT >  type;
  };

  template< typename CellT > struct OpTypes
  {
    typedef typename OperatorTypeBuilder< Gradient,
                                          CellT,
                                          typename FaceTypes<CellT>::XFace >::type	GradX;
    typedef typename OperatorTypeBuilder< Gradient,
                                          CellT,
                                          typename FaceTypes<CellT>::YFace >::type	GradY;
    typedef typename OperatorTypeBuilder< Gradient,
                                          CellT,
                                          typename FaceTypes<CellT>::ZFace >::type	GradZ;

    typedef typename OperatorTypeBuilder< Divergence,
                                          typename FaceTypes<CellT>::XFace,
                                          CellT >::type					DivX;
    typedef typename OperatorTypeBuilder< Divergence,
                                          typename FaceTypes<CellT>::YFace,
                                          CellT >::type 				DivY;
    typedef typename OperatorTypeBuilder< Divergence,
                                          typename FaceTypes<CellT>::ZFace,
                                          CellT >::type 				DivZ;

    typedef typename OperatorTypeBuilder< Interpolant,
                                          CellT,
                                          typename FaceTypes<CellT>::XFace >::type 	InterpC2FX;
    typedef typename OperatorTypeBuilder< Interpolant,
                                          CellT,
                                          typename FaceTypes<CellT>::YFace >::type 	InterpC2FY;
    typedef typename OperatorTypeBuilder< Interpolant,
                                          CellT,
                                          typename FaceTypes<CellT>::ZFace >::type 	InterpC2FZ;

    typedef UpwindInterpolant< CellT, typename FaceTypes<CellT>::XFace > 		InterpC2FXUpwind;
    typedef UpwindInterpolant< CellT, typename FaceTypes<CellT>::YFace > 		InterpC2FYUpwind;
    typedef UpwindInterpolant< CellT, typename FaceTypes<CellT>::ZFace > 		InterpC2FZUpwind;

    typedef typename OperatorTypeBuilder< Interpolant,
                                          typename FaceTypes<CellT>::XFace,
                                          CellT >::type 				InterpF2CX;
    typedef typename OperatorTypeBuilder< Interpolant,
                                          typename FaceTypes<CellT>::YFace,
                                          CellT >::type 				InterpF2CY;
    typedef typename OperatorTypeBuilder< Interpolant,
                                          typename FaceTypes<CellT>::ZFace,
                                          CellT >::type 				InterpF2CZ;
  };
}

#endif // Wasatch_OperatorTypes_h
