#ifndef Vaporization_RHS_Expr_h
#define Vaporization_RHS_Expr_h

#include <expression/Expression.h>

#include <CCA/Components/Wasatch/Coal/CharOxidation/CharData.h>

namespace EVAP{

/**
 *  \class Vaporization_RHS
 */
template< typename FieldT >
class Vaporization_RHS
 : public Expr::Expression<FieldT>
{
   DECLARE_FIELDS( FieldT, tempG_, tempP_, diamP_, reP_, scG_, waterMasFrac_, totalMW_, gasPressure_, moisture_ )

   Vaporization_RHS( const Expr::Tag& tempGTag,
                     const Expr::Tag& tempPTag,
                     const Expr::Tag& diamPTag,
                     const Expr::Tag& rePTag,
                     const Expr::Tag& scGTag,
                     const Expr::Tag& waterMasFracTag,
                     const Expr::Tag& totalMWTag,
                     const Expr::Tag& gasPressureTag,
                     const Expr::Tag& moistureTag );

public:
   class Builder : public Expr::ExpressionBuilder
   {
   public:
     /*
      *  vapRHSTag : rate of vaporization
      *  tempGTag  : Temperature of Gas phase
      *  tempPTag  : Temperature of particle
      *  rePTag    : Re of particle
      *  scGTag    : Sc of gas phase
      *  dgTag     : Diffusivity constant of H2O into the air
      *  waterMassFracTag : mass fraction of water in the gas phase
      *  totalMWTag: total molecular weight of gas phase
      *  gasPressuseTag : Total pressure of gas phase
      *  moistureTag : moisture content of particle
      *  prtmasstag : mass of the particle
      */
     Builder( const Expr::Tag& vapRHSTag,
              const Expr::Tag& tempGTag,
              const Expr::Tag& tempPTag,
              const Expr::Tag& diamPTag,
              const Expr::Tag& rePTag,
              const Expr::Tag& scGTag,
              const Expr::Tag& waterMasFracTag,
              const Expr::Tag& totalMWTag,
              const Expr::Tag& gasPressureTag,
              const Expr::Tag& moistureTag );

     Expr::ExpressionBase* build() const;

   private:
     const Expr::Tag tempGTag_, tempPTag_, diamPTag_, rePTag_, scGTag_, waterMasFracTag_, totalMWTag_, gasPressureTag_, moistureTag_;
   };

   ~Vaporization_RHS(){}
   void evaluate();
};



// ###################################################################
//
//                          Implementation
//
// ###################################################################



template< typename FieldT >
Vaporization_RHS<FieldT>::
Vaporization_RHS( const Expr::Tag& tempGTag,
                  const Expr::Tag& tempPTag,
                  const Expr::Tag& diamPTag,
                  const Expr::Tag& rePTag,
                  const Expr::Tag& scGTag,
                  const Expr::Tag& waterMasFracTag,
                  const Expr::Tag& totalMWTag,
                  const Expr::Tag& gasPressureTag,
                  const Expr::Tag& moistureTag )
  : Expr::Expression<FieldT>()
{
  this->set_gpu_runnable(true);

  tempG_        = this->template create_field_request<FieldT>( tempGTag        );
  tempP_        = this->template create_field_request<FieldT>( tempPTag        );
  diamP_        = this->template create_field_request<FieldT>( diamPTag        );
  reP_          = this->template create_field_request<FieldT>( rePTag          );
  scG_          = this->template create_field_request<FieldT>( scGTag          );
  waterMasFrac_ = this->template create_field_request<FieldT>( waterMasFracTag );
  totalMW_      = this->template create_field_request<FieldT>( totalMWTag      );
  gasPressure_  = this->template create_field_request<FieldT>( gasPressureTag  );
  moisture_     = this->template create_field_request<FieldT>( moistureTag     );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
Vaporization_RHS<FieldT>::
evaluate()
{
  using namespace SpatialOps;

  FieldT& result = this->value();

  const FieldT& tempG        = tempG_       ->field_ref();
  const FieldT& tempP        = tempP_       ->field_ref();
  const FieldT& diamP        = diamP_       ->field_ref();
  const FieldT& reP          = reP_         ->field_ref();
  const FieldT& scG          = scG_         ->field_ref();
  const FieldT& waterMasFrac = waterMasFrac_->field_ref();
  const FieldT& totalMW      = totalMW_     ->field_ref();
  const FieldT& gasPressure  = gasPressure_ ->field_ref();
  const FieldT& moisture     = moisture_    ->field_ref();

  const double R = 8.314472;
  const double critT = 647.3;  // critical temperature of water, K

  // pSat = 1.0E6 * exp(-0.000073375 * pow(*tempP_,2.0) + 0.09048861*(*tempP_) - 25.8925454); // Saturation Pressure of Water -- Martin equation

  SpatFldPtr<FieldT> ptemp = SpatialFieldStore::get<FieldT,FieldT>( result );
  *ptemp <<= min( critT, tempP );

  //  Arden Buck equation
  result <<= cond( moisture <= 0.0, 0.0 )
                 ( min( 0.0,
                       -1.0 * (-2.775E-6 + 4.479E-8 * tempG + 1.656E-10 * tempG * tempG)  // water in air diffusivity
                         * (2.0+0.6*pow(reP,0.5)*pow(scG,1.0/3.0)) / diamP
                         * ( 6.1121e2 * exp((18.678-(*ptemp-273.15)/234.5)* (*ptemp-273.15)/(257.14+(*ptemp-273.15))) / R / *ptemp
                             - waterMasFrac*totalMW/18.0  // mole fraction water
                             * gasPressure / tempG/R
                           )
                         * diamP * diamP * 3.14159  // particle area (neglects internal surface area)
                         * 18.0 /1000.0
                     )
                 );
}

//--------------------------------------------------------------------

template< typename FieldT >
Vaporization_RHS<FieldT>::
Builder::Builder( const Expr::Tag& vapRHSTag,
                  const Expr::Tag& tempGTag,
                  const Expr::Tag& tempPTag,
                  const Expr::Tag& diamPTag,
                  const Expr::Tag& rePTag,
                  const Expr::Tag& scGTag,
                  const Expr::Tag& waterMasFracTag,
                  const Expr::Tag& totalMWTag,
                  const Expr::Tag& gasPressureTag,
                  const Expr::Tag& moistureTag )
  : ExpressionBuilder(vapRHSTag),
    tempGTag_       ( tempGTag       ),
    tempPTag_       ( tempPTag       ),
    diamPTag_       ( diamPTag       ),
    rePTag_         ( rePTag         ),
    scGTag_         ( scGTag         ),
    waterMasFracTag_( waterMasFracTag),
    totalMWTag_     ( totalMWTag     ),
    gasPressureTag_ ( gasPressureTag ),
    moistureTag_    ( moistureTag    )
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
Vaporization_RHS<FieldT>::Builder::build() const
{
  return new Vaporization_RHS<FieldT>( tempGTag_, tempPTag_, diamPTag_, rePTag_, scGTag_, waterMasFracTag_, totalMWTag_, gasPressureTag_, moistureTag_ );
}

} // namespace EVAP

#endif // Vaporization_RHS_Expr_h
