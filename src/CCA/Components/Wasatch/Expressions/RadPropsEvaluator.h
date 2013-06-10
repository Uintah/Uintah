#ifndef RadPropsEvaluator_Expr_h
#define RadPropsEvaluator_Expr_h

#include <expression/Expression.h>

#include <radprops/AbsCoeffGas.h>

typedef std::map<RadiativeSpecies,Expr::Tag> RadSpecMap;

/**
 *  \class RadPropsEvaluator
 *  \author James C. Sutherland
 *  \brief Provides evaluation of radiative properties
 */
template< typename FieldT >
class RadPropsEvaluator
 : public Expr::Expression<FieldT>
{
  typedef std::vector<const FieldT*>  IndepVarVec;
  typedef std::vector<Expr::Tag>      VarNames;

  const Expr::Tag tempTag_;
  GreyGas* greyGas_;

  VarNames indepVarNames_;

  IndepVarVec indepVars_;

  const FieldT* temp_;

  RadPropsEvaluator( const Expr::Tag& tempTag,
                     const RadSpecMap& species,
                     const std::string& fileName );

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  @brief Build a RadPropsEvaluator expression
     *  @param resultTag the tag for the value that this expression computes
     *  @param tempTag the tag for the temperature
     *  @param species
     *  @param fileName
     */
    Builder( const Expr::Tag& resultTag,
             const Expr::Tag& tempTag,
             const RadSpecMap& species,
             const std::string& fileName  );

    Expr::ExpressionBase* build() const;

  private:
    const RadSpecMap rsm_;
    const Expr::Tag tempTag_;
    const std::string fileName_;
  };

  ~RadPropsEvaluator();

  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void evaluate();
};



#endif // RadPropsEvaluator_Expr_h
