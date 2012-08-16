#ifndef Wall_Distance_Expr_h
#define Wall_Distance_Expr_h

#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>
#include <CCA/Components/Wasatch/Expressions/PoissonExpression.h>

#include <expression/Expression.h>

/**
 *  \class WallDistance
 *  \author Tony Saad
 *  \date   June, 2012
 *  \ingroup Expressions
 *  \brief Calculates the distance to the nearest wall based on Spalding's 
           differential equation for wall distance. NOTE: you must solve a Poisson
           system of equations \f$\nabla^2\phi = -1\f$ with Dirichlet conditions
           on walls (\f$\phi = 0\f$) and Neumann conditions on all other boundary
           types (\f$\frac{\partial \phi}{\partial n} = 0\f$.
 *
 */

class WallDistance
: public Expr::Expression<SVolField>
{
  const Expr::Tag phit_;
  const SVolField* phi_;
   
  // gradient operators are only here to extract spacing information out of them
  typedef SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Gradient, SVolField, XVolField >::type GradXT;
  typedef SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Gradient, SVolField, YVolField >::type GradYT;
  typedef SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Gradient, SVolField, ZVolField >::type GradZT;
  
  typedef SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Interpolant, XVolField, SVolField >::type XtoSInterpT;  
  typedef SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Interpolant, YVolField, SVolField >::type YtoSInterpT;  
  typedef SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Interpolant, ZVolField, SVolField >::type ZtoSInterpT;  
  
  const GradXT*  gradXOp_;            ///< x-component of the gradient operator
  const GradYT*  gradYOp_;            ///< y-component of the gradient operator  
  const GradZT*  gradZOp_;            ///< z-component of the gradient operator

  const XtoSInterpT*  xToSInterpOp_;            ///< x-component of the gradient operator
  const YtoSInterpT*  yToSInterpOp_;            ///< y-component of the gradient operator  
  const ZtoSInterpT*  zToSInterpOp_;            ///< z-component of the gradient operator    
  
  WallDistance( const Expr::Tag& phitag );
  
public:
  class Builder : public Expr::ExpressionBuilder
  {
  private:
    const Expr::Tag phit_;
    
  public:
    Builder( const Expr::Tag& result,
            const Expr::Tag& phit )
    : ExpressionBuilder(result),
      phit_         ( phit )
    {}    
    
    Expr::ExpressionBase* build() const 
    {
      return new WallDistance(phit_);
    }
    
    ~Builder(){}
  };
  
  ~WallDistance();
  
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
  
};


// ###################################################################
//
//                          Implementation
//
// ###################################################################

WallDistance::
WallDistance( const Expr::Tag& phitag )
: Expr::Expression<SVolField>(),
  phit_(phitag)
{}

//--------------------------------------------------------------------

WallDistance::
~WallDistance()
{}

//--------------------------------------------------------------------

void
WallDistance::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( phit_ );
}

//--------------------------------------------------------------------

void
WallDistance::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  gradXOp_ = opDB.retrieve_operator<GradXT>();
  gradYOp_ = opDB.retrieve_operator<GradYT>();
  gradZOp_ = opDB.retrieve_operator<GradZT>();
  
  xToSInterpOp_ = opDB.retrieve_operator<XtoSInterpT>();
  yToSInterpOp_ = opDB.retrieve_operator<YtoSInterpT>();
  zToSInterpOp_ = opDB.retrieve_operator<ZtoSInterpT>();  
}

//--------------------------------------------------------------------

void
WallDistance::
bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldMgrSelector<SVolField>::type& scalarfm = fml.field_manager<SVolField>();
  phi_ = &scalarfm.field_ref( phit_ );
}

//--------------------------------------------------------------------

void
WallDistance::
evaluate()
{
  std::cout <<"evaluating wall distance\n";
  using namespace SpatialOps;
  SVolField& walld = this->value();
  walld <<= 0.0;
  
  SpatialOps::SpatFldPtr<SVolField> tmp1 = SpatialOps::SpatialFieldStore::get<SVolField>( walld );
  *tmp1 <<= 0.0;
  SpatialOps::SpatFldPtr<SVolField> tmp2 = SpatialOps::SpatialFieldStore::get<SVolField>( walld );
  *tmp2 <<= 0.0;

  SpatialOps::SpatFldPtr<XVolField> xfield = SpatialOps::SpatialFieldStore::get<XVolField>( walld );
  SpatialOps::SpatFldPtr<YVolField> yfield = SpatialOps::SpatialFieldStore::get<YVolField>( walld );
  SpatialOps::SpatFldPtr<ZVolField> zfield = SpatialOps::SpatialFieldStore::get<ZVolField>( walld );
  
  gradXOp_->apply_to_field( *phi_, *xfield);
  *xfield <<= *xfield * *xfield;
  gradYOp_->apply_to_field( *phi_, *yfield);
  *yfield <<= *yfield * *yfield;
  gradZOp_->apply_to_field( *phi_, *zfield);
  *zfield <<= *zfield * *zfield;
  
  xToSInterpOp_->apply_to_field(*xfield,*tmp1);
  *tmp2 <<= *tmp2 + *tmp1;
  yToSInterpOp_->apply_to_field(*yfield,*tmp1);
  *tmp2 <<= *tmp2 + *tmp1;  
  zToSInterpOp_->apply_to_field(*zfield,*tmp1);
  *tmp2 <<= *tmp2 + *tmp1;  
  
  walld <<= sqrt(*tmp2 + 2.0 * *phi_)  - sqrt(*tmp2); 
}

//--------------------------------------------------------------------

#endif // Wall_Distance_Expr_h
