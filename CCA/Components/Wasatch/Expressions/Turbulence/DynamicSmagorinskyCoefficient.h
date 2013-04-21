#ifndef DynamicSmagorinskyCoefficient_h
#define DynamicSmagorinskyCoefficient_h

#include "StrainTensorBase.h"
#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>
#include "StrainTensorMagnitude.h"
#include <expression/Expression.h>

/**
 *  \brief obtain the tag for the dynamic smagorinsky coefficient
 */
Expr::Tag dynamic_smagorinsky_coefficient_tag();

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
  
  typedef  SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Filter, SVolField, SVolField >::type BoxFilterT;
  typedef  SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Filter, XVolField, XVolField >::type XBoxFilterT;
  typedef  SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Filter, YVolField, YVolField >::type YBoxFilterT;
  typedef  SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Filter, ZVolField, ZVolField >::type ZBoxFilterT;

  
  typedef SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Interpolant, XVolField, SVolField >::type Vel1InterpT;
  typedef SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Interpolant, YVolField, SVolField >::type Vel2InterpT;
  typedef SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Interpolant, ZVolField, SVolField >::type Vel3InterpT;  
  
  const SVolField* rho_;
  
  const BoxFilterT*  BoxFilterOp_;
  const XBoxFilterT* xBoxFilterOp_;
  const YBoxFilterT* yBoxFilterOp_;
  const ZBoxFilterT* zBoxFilterOp_;
  
  const Vel1InterpT* vel1InterpOp_;
  const Vel2InterpT* vel2InterpOp_;
  const Vel3InterpT* vel3InterpOp_;  
  
  DynamicSmagorinskyCoefficient( const Expr::Tag vel1Tag,
                                 const Expr::Tag vel2Tag,
                                 const Expr::Tag vel3Tag,
                                 const Expr::Tag rhoTag );
  

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::TagList& results,
            const Expr::Tag vel1Tag,
            const Expr::Tag vel2Tag,
            const Expr::Tag vel3Tag,
            const Expr::Tag rhoTag )
    : ExpressionBuilder(results),
    vel1t_     ( vel1Tag      ),
    vel2t_     ( vel2Tag      ),
    vel3t_     ( vel3Tag      ),
    rhot_      ( rhoTag       )
    {}
    
    Expr::ExpressionBase* build() const
    {
      return new DynamicSmagorinskyCoefficient( vel1t_, vel2t_, vel3t_, rhot_ );
    }
  private:
    const Expr::Tag vel1t_, vel2t_, vel3t_, rhot_;
  };
  
  ~DynamicSmagorinskyCoefficient();
  
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();  
};

#endif // DynamicSmagorinskyCoefficient_h
