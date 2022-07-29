#ifndef CourantNumber_Expr_h
#define CourantNumber_Expr_h

#include <expression/Expression.h>
#include <spatialops/structured/FVStaggered.h>
#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>
/**
 *  \class 	CourantNumber
 *  \author 	Tony Saad
 *  \date 	 June 2013
 *  \ingroup	Expressions
 *
 *  \brief Calculates cell-based Courant number in a given direction.
 *
 */
template< typename VelT >
class CourantNumber
 : public Expr::Expression<SpatialOps::SVolField>
{
  typedef SVolField FieldT;
  DECLARE_FIELD(FieldT, rho_)
  DECLARE_FIELD(VelT, rhovel_)

  typedef typename SpatialOps::SingleValueField TimeField;
  DECLARE_FIELD (TimeField, dt_)
  
  double h_;
  std::string direction_;
  
  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, VelT, SVolField >::type interpVelT2SVolT;
  const interpVelT2SVolT* interpVelT2SVolOp_;

  typedef typename SpatialOps::BasicOpTypes<SVolField> OpTypes;
  typedef typename OpTypes::GradX GradXT;
  typedef typename OpTypes::GradY GradYT;
  typedef typename OpTypes::GradZ GradZT;
  const GradXT* gradXOp_;
  const GradYT* gradYOp_;
  const GradZT* gradZOp_;

  CourantNumber(const Expr::Tag& rhoTag,
                const Expr::Tag& rhovelTag,
                const Expr::Tag& dtTag,
                const std::string& direction);
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  @brief Build a StableTimestep expression
     *  @param resultTag the tag for the value that this expression computes
     */
    Builder( const Expr::Tag& resultTag,
             const Expr::Tag& rhoTag,
             const Expr::Tag& rhovelTag,
             const Expr::Tag& timeTag,
             const std::string direction);

    Expr::ExpressionBase* build() const;

  private:
    const Expr::Tag rhoTag_, rhovelTag_, dtTag_;
    const std::string direction_;
  };

  ~CourantNumber();
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
};

#endif // CourantNumber_Expr_h
