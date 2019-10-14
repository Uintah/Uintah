#ifndef DynamicSmagorinskyCoefficient_h
#define DynamicSmagorinskyCoefficient_h

#include "StrainTensorBase.h"
#include "StrainTensorMagnitude.h"
#include <expression/Expression.h>
#include <spatialops/structured/FVStaggered.h>
/**
 *  \class  DynamicSmagorinskyCoefficient
 *  \author Tony Saad
 *  \date   April, 2013
 *  \brief Given the filtered components of a velocity field \f$\tilde{u}_i\f$,
 this expression calculates the dynamic smagorinsky coefficient. For convenience,
 it also calculates and stores the straintensormagnitude. This is useful because in many
 cases we wish to visualize the dynamic coefficient. Once these two quantities are
 calculated, the TurbulentViscosity expression will take their product.
 */

template<typename ResultT, typename Vel1T, typename Vel2T, typename Vel3T>
class DynamicSmagorinskyCoefficient
: public StrainTensorBase<ResultT, Vel1T, Vel2T, Vel3T>
{

private:
  const bool isConstDensity_;
  const bool doExtraFiltering_; // experimental
  
  DECLARE_FIELD(SVolField, rho_)
  
  // filtering operators
  typedef  typename SpatialOps::OperatorTypeBuilder< SpatialOps::Filter, SVolField, SVolField >::type BoxFilterT;
  typedef  typename SpatialOps::OperatorTypeBuilder< SpatialOps::Filter, Vel1T, Vel1T >::type XBoxFilterT;
  typedef  typename SpatialOps::OperatorTypeBuilder< SpatialOps::Filter, Vel2T, Vel2T >::type YBoxFilterT;
  typedef  typename SpatialOps::OperatorTypeBuilder< SpatialOps::Filter, Vel3T, Vel3T >::type ZBoxFilterT;

  const BoxFilterT*  boxFilterOp_;
  const XBoxFilterT* xBoxFilterOp_;
  const YBoxFilterT* yBoxFilterOp_;
  const ZBoxFilterT* zBoxFilterOp_;

  // velocity interpolants
  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, Vel1T, ResultT >::type Vel1InterpT;
  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, Vel2T, ResultT >::type Vel2InterpT;
  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, Vel3T, ResultT >::type Vel3InterpT;

  const Vel1InterpT* vel1InterpOp_;
  const Vel2InterpT* vel2InterpOp_;
  const Vel3InterpT* vel3InterpOp_;  
  
  
  // extrapolant operators
  typedef typename WasatchCore::OpTypes<SVolField>::BoundaryExtrapolant ExOpT;
  typedef typename WasatchCore::OpTypes<Vel1T>::BoundaryExtrapolant XExOpT;
  typedef typename WasatchCore::OpTypes<Vel2T>::BoundaryExtrapolant YExOpT;
  typedef typename WasatchCore::OpTypes<Vel3T>::BoundaryExtrapolant ZExOpT;
  
  ExOpT*   exOp_;
  XExOpT*  xexOp_;
  YExOpT*  yexOp_;
  ZExOpT*  zexOp_;
  
  DynamicSmagorinskyCoefficient( const Expr::TagList& velTags,
                                 const Expr::Tag& rhoTag,
                                 const bool isConstDensity);
protected:
  //A_SURF_B_Field = A vol, B surface
  typedef typename SpatialOps::SpatFldPtr<ResultT> SVolPtr;
  typedef typename std::vector< SVolPtr  > SVolVecT;
  typedef typename std::vector< SVolVecT > SVolTensorT;


public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::TagList& results,
             const Expr::TagList& velTags,
             const Expr::Tag& rhoTag,
             const bool isConstDensity )
    : ExpressionBuilder( results ),
      velTags_( velTags ),
      rhot_ ( rhoTag  ),
      isConstDensity_(isConstDensity)
    {}
    
    Expr::ExpressionBase* build() const
    {
      return new DynamicSmagorinskyCoefficient( velTags_, rhot_, isConstDensity_ );
    }
  private:
    const Expr::TagList velTags_;
    const Expr::Tag rhot_;
    const bool isConstDensity_;
  };
  
  ~DynamicSmagorinskyCoefficient();
  
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();  
};

#endif // DynamicSmagorinskyCoefficient_h
