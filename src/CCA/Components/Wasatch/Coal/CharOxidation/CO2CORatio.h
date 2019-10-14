#ifndef CO2CORatio_Expr_h
#define CO2CORatio_Expr_h

#include <expression/Expr_Expression.h>

namespace CHAR {


/**
 *  \ingroup CharOxidation
 *  \class CO2CORatio
 *
 *  \brief calculates the ratio CO2 production and CO production.
 *
 *
 *  The ratio is calculated as:
 *  \f[
 *    \frac{moles CO}{moles CO_{2}} = A \exp \left( \frac{-E}{RT}\right)
 *  \f]
 *    Also in [2]
 *  \f[
 *    \frac{CO_{2}}{CO}=A_{0}P_{O2}^{\eta_{0}}\exp\left(\frac{B}{T_{p}}\right)
 *  \f]
 *  where \f$A_{0}=0.02\f$, \f$B=3070\f$ K and \f$\eta_{0}=0.21\f$
 *
 * [1] "On the Products of the Heteroheneous Oxidation Reaction At The Surface of Burning Coal Char Particle,"
 *     Reginald E Mitchel, 21 Symposium on Combustion - 1988, pp 69-78
 *
 * [2] A. F. Sarofim L. Tognotti, J. P. Longwell. "The products of the high tem- perature oxidation of a single
 *     char particle in an electrodynamic balance", Symposium (International) on Combustion, Twenty-Third:1207-1213, 1990.
 *
 */
template< typename FieldT >
class CO2CORatio : public Expr::Expression<FieldT>
{
  DECLARE_FIELDS( FieldT, tempP_, o2massf_, totalmw_, gaspress_ )

  /**
   *  \param tempPtag  : particle temperature
   *  \param o2massft  : O2 mass fraction in gas pahse at the surfece of particle
   *  \param totalmwt  : total molecular wight of gas phase around particle
   *  \param gaspresst : gas pressure at the particle surface
   */
  CO2CORatio( const Expr::Tag tempPtag,
              const Expr::Tag o2massft,
              const Expr::Tag totalmwt,
              const Expr::Tag gaspresst,
              const Expr::ExpressionID& id,
              const Expr::ExpressionRegistry& reg )
    : Expr::Expression<FieldT>(id,reg)
  {
    this->set_gpu_runnable(true);
    tempP_ = this->template create_field_request<FieldT>( tempPtag );
  }

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag tempPtag,
             const Expr::Tag o2massft,
             const Expr::Tag totalmwt,
             const Expr::Tag gaspresst )
      : tempPtag_  ( tempPtag  ),
        o2massft_  ( o2massft  ),
        totalmwt_  ( totalmwt  ),
        gaspresst_ ( gaspresst )
    {}

    Expr::ExpressionBase*
    build( const Expr::ExpressionID& id, const Expr::ExpressionRegistry& reg ) const{
      return new CO2CORatio<FieldT>( tempPtag_, o2massft_, totalmwt_, gaspresst_, id, reg );
    }


  private:
    const Expr::Tag tempPtag_, o2massft_, totalmwt_, gaspresst_;
  };

  void evaluate()
  {
    using namespace SpatialOps;
    FieldT& result = this->value();

    const FieldT& tempP = tempP_->field_ref();

    // Data by [1]
    const double e = 14300;       // cal/mole
    const double a = 1.9953e+03;  // 10^3.3
    const double r = 1.9858775;   //  gas constant - cal/K/Mole
    result <<= a * exp( -e/( r * *tempP ) );

    // [2]
    //  const FieldT& o2massf  = o2massf_ ->field_ref();
    //  const FieldT& totalmw  = totalmw_ ->field_ref();
    //  const FieldT& gaspress = gaspress_->field_ref();
    //result <<= 0.02* pow((*gaspress_ * *o2massf_ * *totalmw_ / 32), 0.21)*exp(3070/ *tempP_);
  }
};

} // namespace CHAR

#endif // CO2CORatio_Expr_h
