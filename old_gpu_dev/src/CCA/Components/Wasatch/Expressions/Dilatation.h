#ifndef Dilatation_Expr_h
#define Dilatation_Expr_h

#include <expression/Expression.h>

#include <spatialops/structured/FVStaggeredOperatorTypes.h>

/**
 *  \class Dilatation
 *  \ingroup Expressions
 *
 *  \brief calculates \f$\nabla\cdot\mathbf{u}$\f
 *
 *  \tparam FieldT the field type for the dilatation (nominally the scalar volume field)
 *  \tparam Vel1T  the field type for the first velocity component
 *  \tparam Vel2T  the field type for the second velocity component
 *  \tparam Vel3T  the field type for the third velocity component
 */
template< typename FieldT,
          typename Vel1T,
          typename Vel2T,
          typename Vel3T >
class Dilatation
 : public Expr::Expression<FieldT>
{
  const Expr::Tag vel1t_, vel2t_, vel3t_;

  typedef typename SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Gradient, Vel1T, FieldT >::type Vel1GradT;
  typedef typename SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Gradient, Vel2T, FieldT >::type Vel2GradT;
  typedef typename SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Gradient, Vel3T, FieldT >::type Vel3GradT;

  const Vel1T* vel1_;
  const Vel2T* vel2_;
  const Vel3T* vel3_;

  const Vel1GradT* vel1GradOp_;
  const Vel2GradT* vel2GradOp_;
  const Vel3GradT* vel3GradOp_;

  Dilatation( const Expr::Tag& vel1tag,
              const Expr::Tag& vel2tag,
              const Expr::Tag& vel3tag );

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:

    /**
     *  \param vel1tag the velocity corresponding to the Vel1T template parameter
     *  \param vel2tag the velocity corresponding to the Vel2T template parameter
     *  \param vel3tag the velocity corresponding to the Vel3T template parameter
     */
    Builder( const Expr::Tag& result,
             const Expr::Tag& vel1tag,
             const Expr::Tag& vel2tag,
             const Expr::Tag& vel3tag );
    ~Builder(){}
    Expr::ExpressionBase* build() const;

  private:
    const Expr::Tag v1t_, v2t_, v3t_;
  };

  ~Dilatation();

  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
};

#endif // Dilatation_Expr_h
