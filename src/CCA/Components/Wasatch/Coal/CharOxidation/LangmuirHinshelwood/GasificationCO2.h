#ifndef GasificationCO2_CHAR_h
#define GasificationCO2_CHAR_h

#include <expression/Expression.h>

/**
 *  \class GasificationCO2
 */
namespace CHAR{

template< typename FieldT >
class GasificationCO2
 : public Expr::Expression<FieldT>
{
  DECLARE_FIELDS( FieldT, mchar_, tempp_, msfrcco2_, totalmw_, gaspress_ )

  GasificationCO2( const Expr::Tag& mchart,
                   const Expr::Tag& temppt,
                   const Expr::Tag& msfrcco2t,
                   const Expr::Tag& totalmwt,
                   const Expr::Tag& gaspresst )
    : Expr::Expression<FieldT>()
  {
    this->set_gpu_runnable(true);

    mchar_    = this->template create_field_request<FieldT>( mchart    );
    tempp_    = this->template create_field_request<FieldT>( temppt    );
    msfrcco2_ = this->template create_field_request<FieldT>( msfrcco2t );
    totalmw_  = this->template create_field_request<FieldT>( totalmwt  );
    gaspress_ = this->template create_field_request<FieldT>( gaspresst );
  }


public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder(const Expr::Tag& gasifCO2Tag,
            const Expr::Tag& mchart,
            const Expr::Tag& temppt,
            const Expr::Tag& msfrcco2t,
            const Expr::Tag& totalmwt,
            const Expr::Tag& gaspresst)
    : ExpressionBuilder(gasifCO2Tag),
      mchart_   ( mchart   ),
      temppt_   ( temppt   ),
      msfrcco2t_( msfrcco2t),
      totalmwt_ ( totalmwt ),
      gaspresst_( gaspresst)
    {}

    ~Builder(){}
    Expr::ExpressionBase* build() const{
      return new GasificationCO2( mchart_, temppt_, msfrcco2t_, totalmwt_, gaspresst_ );
    }

  private:
    const Expr::Tag mchart_, temppt_, msfrcco2t_, totalmwt_, gaspresst_;
  };

  void evaluate()
  {
    using namespace SpatialOps;

    FieldT& result = this->value();

    const FieldT& mchar    = mchar_   ->field_ref();
    const FieldT& tempp    = tempp_   ->field_ref();
    const FieldT& msfrcco2 = msfrcco2_->field_ref();
    const FieldT& totalmw  = totalmw_ ->field_ref();
    const FieldT& gaspress = gaspress_->field_ref();

    const double R = 8.314;

    const double A1 = 3.34E8, A2 = 6.78E4;
    const double E1 = 2.71E5, E2 = 1.63E5; // j/mol
    const double n1 = 0.54,   n2 = 0.73;

    /*
     * Rate of char gasification by species i calculated as follows:
     *
     * r_i = -k * charMass,
     * k   = A * (P_i)^n * exp(-E/RT),
     * P_i = (y_i * MW_tot / MW_i) * P,
     *
     * where P_i is the partial pressure of species i, y_i is the
     * mass fraction of species i, MW_i is the molecular weight of
     * species i, and MW_tot is the molecular weight of the mixture,
     *
     * The rate equation form is taken from [1], k is taken from [2].
     *
     * [1] Xiaofang Wang, X, B Jin, and W Zhong. Three-dimensional simulation of fluidized bed coal gasification.
     *     Chemical Engineering and Processing: Process Intensification 48, no. 2 (February 2009): 695-705.
     *     http://linkinghub.elsevier.com/retrieve/pii/S0255270108001803.
     *
     * [2] Watanabe, H, and M Otaka. Numerical simulation of coal gasification in entrained flow coal gasifier.
     *     Fuel 85, no. 12-13 (September 2006): 1935-1943.
     *     http://linkinghub.elsevier.com/retrieve/pii/S0016236106000548.
     */

    result <<= cond( mchar <= 0.0, 0.0 )
                   ( tempp < 1473.0, - mchar * A1*exp(-E1/(R * tempp))* pow(msfrcco2 * totalmw / 44.01 * gaspress*1e-6, n1) )
                   (                 - mchar * A2*exp(-E2/(R * tempp))* pow(msfrcco2 * totalmw / 44.01 * gaspress*1e-6, n2) );
  }

};

}  // namesspace CHAR

#endif // GasificationCO2_CHAR_h
