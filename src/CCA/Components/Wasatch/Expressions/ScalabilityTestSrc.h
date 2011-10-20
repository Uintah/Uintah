#ifndef Scalability_Test_Src
#define Scalability_Test_Src

#include <expression/Expr_Expression.h>

//-- ExprLib Includes --//
#include <expression/ExprLib.h>

#include <spatialops/structured/FVStaggeredFieldTypes.h>
#include <spatialops/structured/FVStaggeredOperatorTypes.h>

/**
 *  \ingroup 	Expressions
 *  \class 	ScalabilityTestSrc
 *  \date 	April, 2011
 *  \author 	Tony Saad
 *
 *  \brief Creates an all-to-all strongly coupled source term for use in
 *         scalability tests.
 *
 */
template< typename FieldT >
class ScalabilityTestSrc : public Expr::Expression<FieldT>
{
  const Expr::Tag phiTag_;
  const int nvar_;
  
  typedef std::vector<const FieldT*> FieldVecT;
  FieldVecT phi_;
  
  typedef std::vector<typename FieldT::const_iterator> IterVec;
  IterVec iterVec_;
  std::vector<double> tmpVec_;
  
  ScalabilityTestSrc( const Expr::Tag var,
                      const int nvar,
                      const Expr::ExpressionID& id,
                      const Expr::ExpressionRegistry& reg );
  
  ~ScalabilityTestSrc();
  
public:
  
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void evaluate();
  
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Expr::ExpressionBase* build( const Expr::ExpressionID& id,
                                 const Expr::ExpressionRegistry& reg ) const;
    Builder( const Expr::Tag var,
             const int nvar );
  private:
    const Expr::Tag tag_;
    const int nvar_;
  };
  
};

//====================================================================

#endif // Scalability_Test_Src
