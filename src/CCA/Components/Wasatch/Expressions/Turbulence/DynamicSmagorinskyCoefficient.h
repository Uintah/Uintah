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
  
  typedef  SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Filter, SVolField, SVolField >::type BoxFilterT;
  typedef  SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Filter, XVolField, XVolField >::type XBoxFilterT;
  typedef  SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Filter, YVolField, YVolField >::type YBoxFilterT;
  typedef  SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Filter, ZVolField, ZVolField >::type ZBoxFilterT;

  
  typedef SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Interpolant, XVolField, SVolField >::type Vel1InterpT;
  typedef SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Interpolant, YVolField, SVolField >::type Vel2InterpT;
  typedef SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Interpolant, ZVolField, SVolField >::type Vel3InterpT;
  
  const SVolField* rho_;
  
  const BoxFilterT*  boxFilterOp_;
  const XBoxFilterT* xBoxFilterOp_;
  const YBoxFilterT* yBoxFilterOp_;
  const ZBoxFilterT* zBoxFilterOp_;
  
  const Vel1InterpT* vel1InterpOp_;
  const Vel2InterpT* vel2InterpOp_;
  const Vel3InterpT* vel3InterpOp_;  
  
  
  typedef Wasatch::OpTypes<SVolField>::BoundaryExtrapolant ExOpT;
  typedef Wasatch::OpTypes<XVolField>::BoundaryExtrapolant XExOpT;
  typedef Wasatch::OpTypes<YVolField>::BoundaryExtrapolant YExOpT;
  typedef Wasatch::OpTypes<ZVolField>::BoundaryExtrapolant ZExOpT;
  
  ExOpT*   exOp_;
  XExOpT*  xexOp_;
  YExOpT*  yexOp_;
  ZExOpT*  zexOp_;
  
  DynamicSmagorinskyCoefficient( const Expr::Tag& vel1Tag,
                                 const Expr::Tag& vel2Tag,
                                 const Expr::Tag& vel3Tag,
                                 const Expr::Tag& rhoTag,
                                 const bool isConstDensity);
  

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::TagList& results,
             const Expr::Tag& vel1Tag,
             const Expr::Tag& vel2Tag,
             const Expr::Tag& vel3Tag,
             const Expr::Tag& rhoTag,
             const bool isConstDensity )
    : ExpressionBuilder( results ),
      vel1t_( vel1Tag ),
      vel2t_( vel2Tag ),
      vel3t_( vel3Tag ),
      rhot_ ( rhoTag  ),
      isConstDensity_(isConstDensity)
    {}
    
    Expr::ExpressionBase* build() const
    {
      return new DynamicSmagorinskyCoefficient( vel1t_, vel2t_, vel3t_, rhot_, isConstDensity_ );
    }
  private:
    const Expr::Tag vel1t_, vel2t_, vel3t_, rhot_;
    const bool isConstDensity_;
  };
  
  ~DynamicSmagorinskyCoefficient();
  
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();  
};

#endif // DynamicSmagorinskyCoefficient_h
