#ifndef DynamicSmagorinskyCoefficient_h
#define DynamicSmagorinskyCoefficient_h

#include "StrainTensorBase.h"
#include <CCA/Components/Wasatch/Operators/Operators.h>
#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>
#include "StrainTensorMagnitude.h"
#include <expression/Expression.h>

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

class DynamicSmagorinskyCoefficient
: public StrainTensorBase
{
  const Expr::Tag rhot_;
  const bool isConstDensity_;
  const bool doExtraFiltering_; // experimental
  
  const SVolField* rho_;

  // filtering operators
  typedef  SpatialOps::OperatorTypeBuilder< SpatialOps::Filter, SVolField, SVolField >::type BoxFilterT;
  typedef  SpatialOps::OperatorTypeBuilder< SpatialOps::Filter, XVolField, XVolField >::type XBoxFilterT;
  typedef  SpatialOps::OperatorTypeBuilder< SpatialOps::Filter, YVolField, YVolField >::type YBoxFilterT;
  typedef  SpatialOps::OperatorTypeBuilder< SpatialOps::Filter, ZVolField, ZVolField >::type ZBoxFilterT;

  const BoxFilterT*  boxFilterOp_;
  const XBoxFilterT* xBoxFilterOp_;
  const YBoxFilterT* yBoxFilterOp_;
  const ZBoxFilterT* zBoxFilterOp_;

  // velocity interpolants
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, XVolField, SVolField >::type Vel1InterpT;
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, YVolField, SVolField >::type Vel2InterpT;
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, ZVolField, SVolField >::type Vel3InterpT;

  const Vel1InterpT* vel1InterpOp_;
  const Vel2InterpT* vel2InterpOp_;
  const Vel3InterpT* vel3InterpOp_;  
  
  
  // extrapolant operators
  typedef Wasatch::OpTypes<SVolField>::BoundaryExtrapolant ExOpT;
  typedef Wasatch::OpTypes<XVolField>::BoundaryExtrapolant XExOpT;
  typedef Wasatch::OpTypes<YVolField>::BoundaryExtrapolant YExOpT;
  typedef Wasatch::OpTypes<ZVolField>::BoundaryExtrapolant ZExOpT;
  
  ExOpT*   exOp_;
  XExOpT*  xexOp_;
  YExOpT*  yexOp_;
  ZExOpT*  zexOp_;
  
  DynamicSmagorinskyCoefficient( const Expr::TagList& velTags,
                                 const Expr::Tag& rhoTag,
                                 const bool isConstDensity);
  

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
  
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();  
};

#endif // DynamicSmagorinskyCoefficient_h
