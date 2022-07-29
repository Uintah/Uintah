#ifndef CellReynoldsNumber_Expr_h
#define CellReynoldsNumber_Expr_h

#include <expression/Expression.h>
#include <spatialops/structured/FVStaggered.h>
#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>
/**
 *  \class 	CellReynoldsNumber
 *  \author 	Tony Saad
 *  \date 	 March 2022
 *  \ingroup	Expressions
 *
 *  \brief Calculates a cell Reynolds number in a given direction.
 *
 */
template< typename VelT >
class CellReynoldsNumber
 : public Expr::Expression<SpatialOps::SVolField>
{
  DECLARE_FIELD(VelT, rhovel_)
  DECLARE_FIELD (SVolField, visc_)
  
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

  CellReynoldsNumber( const Expr::Tag& rhovelTag,
                      const Expr::Tag& viscosityTag,
                      const std::string& direction);
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  @brief Build a StableTimestep expression
     *  @param resultTag the tag for the value that this expression computes
     */
    Builder( const Expr::TagList& resultTags,
             const Expr::Tag& rhovelTag,
             const Expr::Tag& viscosityTag,
             const std::string direction);

    Expr::ExpressionBase* build() const;

  private:
    const Expr::Tag rhovelTag_, viscosityTag_;
    const std::string direction_;
  };

  ~CellReynoldsNumber();
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
};

#endif // CellReynoldsNumber_Expr_h
