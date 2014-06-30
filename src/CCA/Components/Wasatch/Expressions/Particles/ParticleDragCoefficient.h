#ifndef ParticleDragCoefficient_Expr_h
#define ParticleDragCoefficient_Expr_h

#include <expression/Expression.h>
#include <cmath>

//==================================================================
/**
 *  \class  ParticleDensity
 *  \author Tony Saad, ODT
 *  \date   June, 2014
 *  \brief  Calculates the drag coefficient.
 */
class ParticleDragCoefficient
 : public Expr::Expression<ParticleField>
{


  ParticleDragCoefficient( const Expr::Tag& pReTag );

  const Expr::Tag pReTag_;
  const ParticleField  *pRe_;

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& dragTag,
             const Expr::Tag& reTag );
    ~Builder(){}
    Expr::ExpressionBase* build() const{ return new ParticleDragCoefficient(pRet_); }
  private:
    const Expr::Tag pRet_;

  };

  ~ParticleDragCoefficient();

  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
};

//###########################################################
//
//                  Implementation
//
//##########################################################

ParticleDragCoefficient::ParticleDragCoefficient( const Expr::Tag& pReTag )
  : Expr::Expression<ParticleField>(),
    pReTag_(pReTag)
{
  this->set_gpu_runnable(true);
}

//--------------------------------------------------------------------

ParticleDragCoefficient::~ParticleDragCoefficient()
{}

//--------------------------------------------------------------------

void
ParticleDragCoefficient::advertise_dependents( Expr::ExprDeps& exprDeps)
{
  exprDeps.requires_expression( pReTag_  );
}

//--------------------------------------------------------------------

void
ParticleDragCoefficient::bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldMgrSelector<ParticleField>::type& fm = fml.field_manager<ParticleField>();
  pRe_ = &fm.field_ref( pReTag_  );
}

//--------------------------------------------------------------------

void
ParticleDragCoefficient::bind_operators( const SpatialOps::OperatorDatabase& opDB )
{}

//--------------------------------------------------------------------

void
ParticleDragCoefficient::evaluate()
{
  using namespace SpatialOps;
  ParticleField& result = this->value();

  result <<= cond( *pRe_ <= 1.0,  1.0 )
                 ( *pRe_ <= 1000, 1.0 + 0.15 * pow( *pRe_ , 0.687 ) )
                 ( 0.0183 * *pRe_ );
}

//--------------------------------------------------------------------

ParticleDragCoefficient::Builder::Builder( const Expr::Tag& dragTag,
                                   const Expr::Tag& pReTag )
: ExpressionBuilder(dragTag),
  pRet_(pReTag)
{}

//--------------------------------------------------------------------

#endif // ParticleDragCoefficient_Expr_h
