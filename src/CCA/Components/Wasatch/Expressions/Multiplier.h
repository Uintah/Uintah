#ifndef Multiplier_Expr_h
#define Multiplier_Expr_h

#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>

#include <expression/Expression.h>

/**
 *  \class Multiplier
 *  \author Amir Biglari
 *  \brief given \f$Var1$\f and \f$Var2$\f, this calculates \f$Var1*Var2$\f.
 *
 *   Note: It is currently assumed that \f$Var2$\f is basically density which is
 *         "SVolField"s type. Therefore, no interpolation of the variables occurs
 *         in that case. In other cases, the \f$Var2$\f is interpolated to the
 *         location of \f$Var1$\f.
 *         Note that, in order to use as something else you may need to add new
 *         instantiaions to the class
 */
template< typename Field1T,   // jcs why do you have these templated if you assume they are SVolField as you state above?
          typename Field2T >  // if Field2T is always SVolField, then you don't need to template it.
class Multiplier
 : public Expr::Expression<Field1T>
{
  const Expr::Tag var1t_, var2t_;

  typedef typename OperatorTypeBuilder< Interpolant, Field2T, Field1T >::type  InterpT;

  const Field2T* var2_;
  const Field1T* var1_;
  const InterpT* interpOp_;

  Multiplier( const Expr::Tag& var1Tag,
              const Expr::Tag& var2Tag );

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& result,
             const Expr::Tag& var1Tag,
             const Expr::Tag& var2Tag );
    ~Builder(){}
    Expr::ExpressionBase* build() const;

  private:
  const Expr::Tag var1t_, var2t_;
  };

  ~Multiplier();

  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();

};



template< typename FieldT >
class Multiplier<FieldT,FieldT>
 : public Expr::Expression<FieldT>
{
  const Expr::Tag var1t_, var2t_;

  const FieldT* var1_;
  const FieldT* var2_;

  Multiplier( const Expr::Tag& var1Tag,
              const Expr::Tag& var2Tag );

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& result,
             const Expr::Tag& var1Tag,
             const Expr::Tag& var2Tag );
    ~Builder(){}
    Expr::ExpressionBase* build() const;

  private:
  const Expr::Tag var1t_, var2t_;
  };

  ~Multiplier();

  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void evaluate();
};

#endif // Multiplier_Expr_h
