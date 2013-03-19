#ifndef TimeDerivativeExpr_h
#define TimeDerivativeExpr_h

#include <expression/Expression.h>
#include <spatialops/structured/SpatialFieldStore.h>

template< typename ValT >
class TimeDerivative : public Expr::Expression<ValT>
{
public:
  
  /**
   *  \brief Builds a dphi/dt expression.
   */
  struct Builder : public Expr::ExpressionBuilder
  {
    Builder( const Expr::Tag& result,
            const Expr::Tag& newVarTag,
            const Expr::Tag& oldVarTag,
            const Expr::Tag& timestepTag );
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  private:
    const Expr::Tag newvartag_;
    const Expr::Tag oldvartag_;
    const Expr::Tag timesteptag_;
  };
  
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void evaluate();
  
private:
  
  TimeDerivative( const Expr::Tag& newVarTag,
                 const Expr::Tag& oldVarTag,
                 const Expr::Tag& timestepTag );
  const Expr::Tag newvartag_;
  const Expr::Tag oldvartag_;
  const Expr::Tag timesteptag_;
  const ValT* newvar_;
  const ValT* oldvar_;
  const double* dt_;
};

//====================================================================
//--------------------------------------------------------------------

template<typename ValT>
TimeDerivative<ValT>::
TimeDerivative( const Expr::Tag& newVarTag,
               const Expr::Tag& oldVarTag,
               const Expr::Tag& timestepTag )
: Expr::Expression<ValT>(),
newvartag_  ( newVarTag ),
oldvartag_  ( oldVarTag ),
timesteptag_( timestepTag )
{}

//--------------------------------------------------------------------

template< typename ValT >
void
TimeDerivative<ValT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( newvartag_ );
  exprDeps.requires_expression( oldvartag_ );  
  exprDeps.requires_expression( timesteptag_ );
}

//--------------------------------------------------------------------

template< typename ValT >
void
TimeDerivative<ValT>::
bind_fields( const Expr::FieldManagerList& fml )
{  
  const typename Expr::FieldMgrSelector<ValT>::type& valtfm = fml.template field_manager<ValT>();
  newvar_ = &valtfm.field_ref( newvartag_ );
  oldvar_ = &valtfm.field_ref( oldvartag_ );
  dt_     = &fml.template field_manager<double>().field_ref( timesteptag_ );
}

//--------------------------------------------------------------------

template< typename ValT >
void
TimeDerivative<ValT>::
evaluate()
{
  using namespace SpatialOps;
  ValT& phi = this->value();
  phi <<= (*newvar_ - *oldvar_)/ *dt_;
}

//--------------------------------------------------------------------

template< typename ValT >
TimeDerivative<ValT>::Builder::
Builder( const Expr::Tag& result,
        const Expr::Tag& newVarTag,
        const Expr::Tag& oldVarTag,
        const Expr::Tag& timestepTag )
: ExpressionBuilder(result),
newvartag_  ( newVarTag ),
oldvartag_  ( oldVarTag ),
timesteptag_( timestepTag )
{}

//--------------------------------------------------------------------

template< typename ValT >
Expr::ExpressionBase*
TimeDerivative<ValT>::Builder::build() const
{
  return new TimeDerivative<ValT>( newvartag_, oldvartag_, timesteptag_ );
}

//--------------------------------------------------------------------


#endif // TimeDerivativeExpr_h
