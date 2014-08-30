#ifndef RHS_Expr_h
#define RHS_Expr_h

#include <expression/Expression.h>

typedef SpatialOps::SingleValueField FieldT;

/*
  % Matlab code:
  D=dsolve( ...
   'DcA=-k1*cA',...
   'DcB=k1*cA-k2*cB',...
   'DcC=k2*cB',...
   'cA(0)=cA0',...
   'cB(0)=0',...
   'cC(0)=0'...
  );

  cA = simple( D.cA );
  cB = simple( D.cB );
  cC = simple( D.cC );

  pretty(cA);
  pretty(cB);
  pretty(cC);
 */



/**
 *  \class RHS
 *  \brief test class for simple kinetics problem
 *
 *  This class calculates the RHS term for the following reaction scheme:
 *     A -> B   (k1)
 *     B -> C   (k2)
 *  This implies the following ODEs
 *   \f{eqnarray*}{
 *    \frac{ d c_A }{ d t } &=& -r_1 \\
 *    \frac{ d c_B }{ d t } &=& r_1 -r_2 \\
 *    \frac{ d c_C }{ d t } &=& r_2
 *   \f}
 *  with
 *   \f{eqnarray*}{
 *     r_1 &=& k_1 c_A \\
 *     r_2 &=& k_2 c_B
 *   \f}
 *
 *  The analytic solution is given as
 *   \f{eqnarray*}{
 *      c_A &=& c_{A0} \exp(-k_1 t ) \\
 *      c_B &=& -\frac{c_{A0} k_1}{k_1 - k_2} \left( \exp(- k_1 t) - \exp(- k_2 t) \right)
 *      c_C &=& \frac{c_{A0}}{k_1 - k_2} \left(k_1 - k_2 - k_1 \exp(-k_2 t) + k_2 \exp(-k_1 t) \right)
 *   \f}
 */
class RHS
 : public Expr::Expression<FieldT>
{
  const Expr::TagList& ctags_;
  const double k1_, k2_;

  typedef std::vector<      FieldT*>      FieldVec;
  typedef std::vector<const FieldT*> ConstFieldVec;
  ConstFieldVec c_;

  RHS( const Expr::TagList& ctags,
       const double k1,
       const double k2 );

public:
  class Builder : public Expr::ExpressionBuilder
  {
    const Expr::TagList& ctags_;
    const double k1_, k2_;
  public:
    Builder( const Expr::TagList& exprValues,
             const Expr::TagList& ctags,
             const double k1, const double k2 )
      : Expr::ExpressionBuilder(exprValues),
        ctags_( ctags ), k1_(k1), k2_(k2)
    {}

    Expr::ExpressionBase*
    build() const
    {
      return new RHS( ctags_, k1_, k2_);
    }
  };

  ~RHS(){}

  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void evaluate();
};


#endif // RHS_Expr_h
