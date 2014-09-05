#ifndef Interpolate_Expr_h
#define Interpolate_Expr_h

#include <expression/Expression.h>
#include <spatialops/structured/FVStaggeredOperatorTypes.h>
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>

/**
 *  \class 	Interpolate Expression
 *  \author Tony Saad
 *  \date 	 February, 2012
 *  \ingroup	Expressions
 *
 *  \brief An expression that interpolates between different field types.
           For example, this can be usedto calculate cell centered velocities.
           This expression is currently specialized for staggered-to-cell centered
           interpolation.
 *  \param SrcT: Source field type.
    \param DestT: Destination field type.
 *
 */
template< typename SrcT, typename DestT >
class InterpolateExpression
: public Expr::Expression<DestT>
{
  const Expr::Tag srct_;
  
  typedef typename SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Interpolant, SrcT, DestT >::type InpterpSrcT2DestT;
  
  const SrcT* src_;
  
  const InpterpSrcT2DestT* InpterpSrcT2DestTOp_;
  
  InterpolateExpression( const Expr::Tag& srctag );
  
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    
    /**
     *  \param srctag  Tag of the source field
     *  \param desttag Tag of the destination field
     */
    Builder( const Expr::Tag& result,
             const Expr::Tag& srctag );
    ~Builder(){}
    Expr::ExpressionBase* build() const;
    
  private:
    const Expr::Tag srct_;
  };
  
  ~InterpolateExpression();
  
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
};

#endif // Interpolate_Expr_h
