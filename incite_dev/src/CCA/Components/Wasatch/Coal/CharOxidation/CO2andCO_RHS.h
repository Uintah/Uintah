#ifndef CO2andCO_RHS_Expr_h
#define CO2andCO_RHS_Expr_h

#include <expression/Expression.h>
namespace CHAR{


/**
 *  \ingroup CharOxidation
 *  \class CO2andCO_RHS
 *
 *  This Expression calcuate the CO2 and CO production rate.
 *
    Output is an TagList which have two tag :
    1- CO2
    2- CO

    Inputs are :
    charoxidationRHStag : which is the consumption rate of char.
    coCO2RatioTag       : is the ration between the moles of CO2/ moles of CO
    hetrocoreactag      : Char reaction rate for CO2 hetrogenouse reaction
                          C_{char}+CO_{2}->2CO

 */
template< typename FieldT >
class CO2andCO_RHS
 : public Expr::Expression<FieldT>
{
  DECLARE_FIELDS( FieldT, charoxidationRHS_, coCO2Ratio_, heterocoreac_, heteroh2oreac_ )

  CO2andCO_RHS( const Expr::Tag& charoxidationRHStag,
                const Expr::Tag& coCO2RatioTag,
                const Expr::Tag& hetrocoreactag,
                const Expr::Tag& hetroh2oreactag )
    : Expr::Expression<FieldT>()
  {
    this->set_gpu_runnable(true);
    charoxidationRHS_ = this->template create_field_request<FieldT>( charoxidationRHStag );
    coCO2Ratio_       = this->template create_field_request<FieldT>( coCO2RatioTag       );
    heterocoreac_     = this->template create_field_request<FieldT>( hetrocoreactag      );
    heteroh2oreac_    = this->template create_field_request<FieldT>( hetroh2oreactag     );
  }

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::TagList& coco2rhsTag,
             const Expr::Tag& charoxidationRHStag,
             const Expr::Tag& coCO2RatioTag,
             const Expr::Tag& hetrocoreactag,
             const Expr::Tag& hetroh2oreactag)
      : ExpressionBuilder(coco2rhsTag),
        charoxidationRHStag_( charoxidationRHStag ),
        coCO2RatioTag_      ( coCO2RatioTag       ),
        hetrocoreactag_     ( hetrocoreactag      ),
        hetroh2oreactag_    ( hetroh2oreactag     )
    {}

    ~Builder(){}
    Expr::ExpressionBase* build() const{
      return new CO2andCO_RHS<FieldT>( charoxidationRHStag_, coCO2RatioTag_, hetrocoreactag_, hetroh2oreactag_ );
    }


  private:
    const Expr::Tag charoxidationRHStag_, coCO2RatioTag_, hetrocoreactag_, hetroh2oreactag_;
  };

  void evaluate()
  {
    using namespace SpatialOps;

    typename Expr::Expression<FieldT>::ValVec& dyi = this->get_value_vec();
    assert( dyi.size() == 2 );

    const FieldT& charOxidRHS   = charoxidationRHS_->field_ref();
    const FieldT& coCO2Ratio    = coCO2Ratio_      ->field_ref();
    const FieldT& heteroCOReac  = heterocoreac_    ->field_ref();
    const FieldT& heteroH2OReac = heteroh2oreac_   ->field_ref();

    FieldT& co2 = *dyi[0];
    co2 <<= charOxidRHS * coCO2Ratio / (1.0 + coCO2Ratio) /12.0 * 44.0
            - heteroCOReac/12.0 * 44.0;
    FieldT& co = *dyi[1];
    co  <<= charOxidRHS / (1.0 + coCO2Ratio) / 12.0 * 28.0
           + heteroCOReac/12.0 * 28.0*2.0 + heteroH2OReac/12.0*28.0  ;
  }

};

//--------------------------------------------------------------------

} // namesapace CHAR


#endif // CO2andCO_RHS_Expr_h
