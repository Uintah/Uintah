#ifndef GasificationH2O_CHAR_h
#define GasificationH2O_CHAR_h

#include <expression/Expression.h>


/**
 *  \class GasificationH2O
 *
 *  Reaction :
 *    \f[ C_{(char)}+H_{2}O\rightarrow CO+H_{2} \f]
 */
namespace CHAR{

template< typename FieldT >
class GasificationH2O
 : public Expr::Expression<FieldT>
{
  DECLARE_FIELDS( FieldT, mchar_, prtdim_, tempp_, msfrch2o_, totalmw_, gaspress_ )

  GasificationH2O( const Expr::Tag& mchart,
                   const Expr::Tag& prtdimt,
                   const Expr::Tag& temppt,
                   const Expr::Tag& msfrch2ot,
                   const Expr::Tag& totalmwt,
                   const Expr::Tag& gaspresst );

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& gasifH2OTag,
             const Expr::Tag& mchart,
             const Expr::Tag& prtdimt,
             const Expr::Tag& temppt,
             const Expr::Tag& msfrch2ot,
             const Expr::Tag& totalmwt,
             const Expr::Tag& gaspresst );
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  private:
    const Expr::Tag mchart_, prtdimt_, temppt_, msfrch2ot_, totalmwt_, gaspresst_;
  };

  void evaluate();
};



// ###################################################################
//
//                          Implementation
//
// ###################################################################


template< typename FieldT >
GasificationH2O<FieldT>::
GasificationH2O( const Expr::Tag& mchart,
                 const Expr::Tag& prtdimt,
                 const Expr::Tag& temppt,
                 const Expr::Tag& msfrch2ot,
                 const Expr::Tag& totalmwt,
                 const Expr::Tag& gaspresst )
  : Expr::Expression<FieldT>()
{
  this->set_gpu_runnable(true);

  mchar_    = this->template create_field_request<FieldT>( mchart    );
  prtdim_   = this->template create_field_request<FieldT>( prtdimt   );
  tempp_    = this->template create_field_request<FieldT>( temppt    );
  msfrch2o_ = this->template create_field_request<FieldT>( msfrch2ot );
  totalmw_  = this->template create_field_request<FieldT>( totalmwt  );
  gaspress_ = this->template create_field_request<FieldT>( gaspresst );
}

//--------------------------------------------------------------------
template< typename FieldT >
void
GasificationH2O<FieldT>::evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  const FieldT& mchar    = mchar_   ->field_ref();
  const FieldT& prtdim   = prtdim_  ->field_ref();
  const FieldT& tempp    = tempp_   ->field_ref();
  const FieldT& msfrch2o = msfrch2o_->field_ref();
  const FieldT& totalmw  = totalmw_ ->field_ref();
  const FieldT& gaspress = gaspress_->field_ref();

  const double R = 8.314;
  const double A1 = 2.89E8, A2 = 8.55E4; // see ref [2] below
  const double E1 = 2.52E5, E2 = 1.40E5; // j/mol
  const double n1 = 0.64,   n2 = 0.84;

  /*
   * Rate of char gasification by species i calculated as follows:
   *
   * r_i = -k * charMass,
   * k   = A * (P_i)^n * exp(-E/RT),
   * P_i = (y_i * MW_tot / MW_i) * P,
   *
   * where P_i is the partial pressure of species i, y_i is the
   * mass fraction of species i, MW_i is the molecular weight of
   * species i, and MW_tot is the molecular weight of the mixture.
   */

  result <<= cond( mchar <= 0.0, 0.0 )
                 ( tempp < 1533.0, - mchar * A1*exp(-E1/(R * tempp))* pow(msfrch2o * totalmw / 18.02 * gaspress*1e-6, n1) )
                 (                 - mchar * A2*exp(-E2/(R * tempp))* pow(msfrch2o * totalmw / 18.02 * gaspress*1e-6, n2) );
}

//--------------------------------------------------------------------
template< typename FieldT >
GasificationH2O<FieldT>::Builder::
Builder( const Expr::Tag& gasifH2OTag,
         const Expr::Tag& mchart,
         const Expr::Tag& prtdimt,
         const Expr::Tag& temppt,
         const Expr::Tag& msfrch2ot,
         const Expr::Tag& totalmwt,
         const Expr::Tag& gaspresst)
: ExpressionBuilder(gasifH2OTag),
  mchart_   ( mchart   ),
  prtdimt_  ( prtdimt  ),
  temppt_   ( temppt   ),
  msfrch2ot_( msfrch2ot),
  totalmwt_ ( totalmwt ),
  gaspresst_( gaspresst)
{}

//--------------------------------------------------------------------
template< typename FieldT >
Expr::ExpressionBase*
GasificationH2O<FieldT>::Builder::build() const
{
  return new GasificationH2O( mchart_, prtdimt_, temppt_, msfrch2ot_, totalmwt_, gaspresst_ );
}

}  // namesspace GASIFs

/* [1] Xiaofang Wang, X, B Jin, and W Zhong. Three-dimensional simulation of fluidized bed coal gasification.
      Chemical Engineering and Processing: Process Intensification 48, no. 2 (February 2009): 695-705.
         http://linkinghub.elsevier.com/retrieve/pii/S0255270108001803.

   [2] Watanabe, H, and M Otaka. Numerical simulation of coal gasification in entrained flow coal gasifier.Fuel 85,
       no. 12-13 (September 2006): 1935-1943.
      http://linkinghub.elsevier.com/retrieve/pii/S0016236106000548.
      http://www.sciencedirect.com/science/article/pii/S0016236106000548
*/

#endif // GasificationH2O_GASIF_h
