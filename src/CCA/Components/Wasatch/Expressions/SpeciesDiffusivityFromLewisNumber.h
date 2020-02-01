#ifndef SpeciesDiffusivityFromLewisNumber_Expr_h
#define SpeciesDiffusivityFromLewisNumber_Expr_h

#include <expression/Expression.h>

/**
 *  @class SpeciesDiffusivityFromLewisNumber
 *  @author Josh McConnell
 *  @date   February, 2018
 *  @brief Calculates a diffusivity given a value for Lewis number and fields 
 *         for thermal conductivity, heat capacity, and density.
 *
 *
 *  @param density_            : gas phase density
 *  @param thermalConductivity_: gas phase thermal conductivity
 *  @param heatCapacity_       : gas phase heat capacity
 *  @param lewisNumber_:       : Lewis number
 */
template< typename ScalarT >
class SpeciesDiffusivityFromLewisNumber
        : public Expr::Expression<ScalarT>
{
    DECLARE_FIELDS( ScalarT, density_, thermalConductivity_, heatCapacity_ )
    const double lewisNumber_;

    SpeciesDiffusivityFromLewisNumber( const Expr::Tag& densityTag,
               const Expr::Tag& thermalConductivityTag,
               const Expr::Tag& heatCapacityTag,
               const double     lewisNumber );
public:
    class Builder : public Expr::ExpressionBuilder
    {
    public:
        /**
         *  @brief Build a SpeciesDiffusivityFromLewisNumber expression
         *  @param resultTag the tag for the value that this expression computes
         */
        Builder( const Expr::Tag& resultTag,
                 const Expr::Tag& densityTag,
                 const Expr::Tag& thermalConductivityTag,
                 const Expr::Tag& heatCapacityTag,
                 const double     lewisNumber );

        Expr::ExpressionBase* build() const;

    private:
        const Expr::Tag densityTag_, thermalConductivityTag_, heatCapacityTag_;
        const double lewisNumber_;
    };

    void evaluate();
};



// ###################################################################
//
//                          Implementation
//
// ###################################################################



template< typename ScalarT >
SpeciesDiffusivityFromLewisNumber<ScalarT>::
SpeciesDiffusivityFromLewisNumber( const Expr::Tag& densityTag,
                                   const Expr::Tag& thermalConductivityTag,
                                   const Expr::Tag& heatCapacityTag,
                                   const double     lewisNumber )
  : Expr::Expression<ScalarT>(),
  lewisNumber_( lewisNumber )
{
    density_             = this->template create_field_request<ScalarT>( densityTag   );
    thermalConductivity_ = this->template create_field_request<ScalarT>( thermalConductivityTag );
    heatCapacity_        = this->template create_field_request<ScalarT>( heatCapacityTag    );
}

//--------------------------------------------------------------------

template< typename ScalarT >
void
SpeciesDiffusivityFromLewisNumber<ScalarT>::
evaluate()
{
    using namespace SpatialOps;
    ScalarT& result = this->value();

    const ScalarT& density             = density_            ->field_ref();
    const ScalarT& thermalConductivity = thermalConductivity_->field_ref();
    const ScalarT& heatCapacity        = heatCapacity_       ->field_ref();

    result <<= 1./lewisNumber_*(thermalConductivity)/(density * heatCapacity);
}

//--------------------------------------------------------------------

template< typename ScalarT >
SpeciesDiffusivityFromLewisNumber<ScalarT>::
Builder::Builder( const Expr::Tag& resultTag,
                  const Expr::Tag& densityTag,
                  const Expr::Tag& thermalConductivityTag,
                  const Expr::Tag& heatCapacityTag,
                  const double     lewisNumber )
  : ExpressionBuilder( resultTag ),
    densityTag_            ( densityTag             ),
    thermalConductivityTag_( thermalConductivityTag ),
    heatCapacityTag_       ( heatCapacityTag        ),
    lewisNumber_           ( lewisNumber            )
{}

//--------------------------------------------------------------------

template< typename ScalarT >
Expr::ExpressionBase*
SpeciesDiffusivityFromLewisNumber<ScalarT>::
Builder::build() const
{
    return new SpeciesDiffusivityFromLewisNumber<ScalarT>( densityTag_, thermalConductivityTag_, heatCapacityTag_, lewisNumber_ );
}


#endif // SpeciesDiffusivityFromLewisNumber
