#ifndef HeatFromCharReactions_h
#define HeatFromCharReactions_h

#include <expression/Expression.h>

/**
 *   \class HeatFromCharReactions
 *   \Author Josh McConnell
 *   \Date   April 2018
 *
 *   \brief Calculates the species mass fractions and total consumption rate (kg/s)
 *          of gas produced at the particle. The rates inputed to this expression
 *          are "consumption" rates so all rate values should be negative. The
 *          the total consumption rate should always have a non-negative value.
 *
 *   \param charOxidTag  : char consumption due to oxidation
 *   \param co2GasifTag  : char consumption due to co2 gasification reaction
 *   \param h2oGasifTag  : char consumption due to h2o gasification reaction
 *   \param co2CORatioTag: molar CO2/CO ratio of char oxidation
 *
 */
namespace Coal {

//--------------------------------------------------------------------

  template< typename FieldT >
class HeatFromCharReactions
 : public Expr::Expression<FieldT>
{
  const double hOxidCO2_, hOxidCO_, hGasifCO2_, hGasifH2O_;

  DECLARE_FIELDS( FieldT, charOxid_, co2GasifRHS_, h2oGasifRHS_, co2coRatio_ )

  HeatFromCharReactions( const Expr::Tag& o2RHSTag,
                         const Expr::Tag& co2GasifRHSTag,
                         const Expr::Tag& h2oGasifRHSTag,
                         const Expr::Tag& co2CoRatioTag );
public:

  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *   \param resultTag    : total heat released by char reactions
     *   \param charOxidTag  : char consumption due to oxidation
     *   \param co2GasifTag  : char consumption due to co2 gasification reaction
     *   \param h2oGasifTag  : char consumption due to h2o gasification reaction
     *   \param co2CORatioTag: molar CO2/CO ratio of char oxidation
     */
    Builder( const Expr::Tag& resultTag,
             const Expr::Tag& charOxidTag,
             const Expr::Tag& co2GasifRHSTag,
             const Expr::Tag& h2oGasifRHSTag,
             const Expr::Tag& co2CoRatioTag );

    Expr::ExpressionBase* build() const;

  private:
    const Expr::Tag charOxidTag_, co2GasifRHSTag_,
                    h2oGasifRHSTag_, co2CoRatioTag_;
  };

  void evaluate();
};



// ###################################################################
//
//                          Implementation
//
// ###################################################################



  template< typename FieldT >
  HeatFromCharReactions<FieldT>::
  HeatFromCharReactions( const Expr::Tag& charOxidTag,
                         const Expr::Tag& co2GasifRHSTag,
                         const Expr::Tag& h2oGasifRHSTag,
                         const Expr::Tag& co2coRatioTag )
    : Expr::Expression<FieldT>(),
      hOxidCO2_    ( -33075.72E3 ),
      hOxidCO_     ( -9629.64E3  ),
      hGasifCO2_   ( 14.37E6     ),
      hGasifH2O_   ( 10.94E6     )

  {
    this->set_gpu_runnable(true);

    charOxid_    = this->template create_field_request<FieldT>( charOxidTag    );
    co2GasifRHS_ = this->template create_field_request<FieldT>( co2GasifRHSTag );
    h2oGasifRHS_ = this->template create_field_request<FieldT>( h2oGasifRHSTag );
    co2coRatio_  = this->template create_field_request<FieldT>( co2coRatioTag  );


  }

//--------------------------------------------------------------------

template< typename FieldT >
void
HeatFromCharReactions<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();

  const FieldT& charOxid    = charOxid_   ->field_ref();
  const FieldT& co2GasifRHS = co2GasifRHS_->field_ref();
  const FieldT& h2oGasifRHS = h2oGasifRHS_->field_ref();
  const FieldT& co2coRatio  = co2coRatio_ ->field_ref();

  SpatFldPtr<FieldT> coFrac  = SpatialFieldStore::get<FieldT,FieldT>( result );

  // fraction of oxidized char that ends up as CO. The balance ends up as CO2.
  *coFrac <<= 1. / ( 1. + co2coRatio );

  result <<= -charOxid *
             (
                *coFrac *        hOxidCO_   // for char + O2 --> 2CO
              + (1. - *coFrac) * hOxidCO2_  // for char + O2 --> CO2
             )
           -  co2GasifRHS * hGasifCO2_
           -  h2oGasifRHS * hGasifH2O_;

}

//--------------------------------------------------------------------

template< typename FieldT >
HeatFromCharReactions<FieldT>::
Builder::Builder( const Expr::Tag& resultTag,
                  const Expr::Tag& charOxidTag,
                  const Expr::Tag& co2GasifRHSTag,
                  const Expr::Tag& h2oGasifRHSTag,
                  const Expr::Tag& co2CoRatioTag )
  : ExpressionBuilder( resultTag ),
    charOxidTag_   ( charOxidTag    ),
    co2GasifRHSTag_( co2GasifRHSTag ),
    h2oGasifRHSTag_( h2oGasifRHSTag ),
    co2CoRatioTag_ ( co2CoRatioTag  )
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
HeatFromCharReactions<FieldT>::
Builder::build() const
{
  return new HeatFromCharReactions<FieldT>( charOxidTag_, co2GasifRHSTag_,
                                                   h2oGasifRHSTag_, co2CoRatioTag_ );
}

} // namespace Coal
#endif // HeatFromCharReactions_h

