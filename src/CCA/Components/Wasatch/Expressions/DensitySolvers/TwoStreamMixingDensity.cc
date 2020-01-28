#include <CCA/Components/Wasatch/Expressions/DensitySolvers/TwoStreamMixingDensity.h>

namespace WasatchCore{

    template< typename FieldT >
    TwoStreamMixingDensity<FieldT>::
    TwoStreamMixingDensity( const Expr::Tag& rhofTag,
                            const double rho0,
                            const double rho1 )
    : Expr::Expression<FieldT>(),
        rho0_(rho0), rho1_(rho1),
        rhoMin_( rho0_ < rho1_ ? rho0_ : rho1),
        rhoMax_( rho0_ > rho1_ ? rho0_ : rho1)
    {
    this->set_gpu_runnable(true);
    rhof_ = this->template create_field_request<FieldT>(rhofTag);
    }

    //--------------------------------------------------------------------

    template< typename FieldT >
    void
    TwoStreamMixingDensity<FieldT>::
    evaluate()
    {
    using namespace SpatialOps;
    typename Expr::Expression<FieldT>::ValVec& results  = this->get_value_vec();
    FieldT& rho    = *results[0];
    FieldT& drhodf = *results[1];
    
    const FieldT& rf = rhof_->field_ref();
    
    // compute the density in one shot from rhof
    rho <<= rho0_ + (1 - rho0_/rho1_)*rf;

    // repair bounds
    rho <<= max( min(rho, rhoMax_), rhoMin_ );
    
    drhodf <<= (1/rho0_ - 1/rho1_)*rho*rho;
    }

    //--------------------------------------------------------------------

    template< typename FieldT >
    TwoStreamMixingDensity<FieldT>::
    Builder::Builder( const Expr::TagList& resultsTagList,
                    const Expr::Tag& rhofTag,
                    const double rho0,
                    const double rho1 )
    : ExpressionBuilder( resultsTagList ),
        rho0_(rho0), rho1_(rho1),
        rhofTag_( rhofTag )
    {}

    //====================================================================


    template< typename FieldT >
    TwoStreamDensFromMixfr<FieldT>::
    TwoStreamDensFromMixfr( const Expr::Tag& mixfrTag,
                            const double rho0,
                            const double rho1 )
    : Expr::Expression<FieldT>(),
        rho0_(rho0), rho1_(rho1),
        rhoMin_( rho0_ < rho1_ ? rho0_ : rho1),
        rhoMax_( rho0_ > rho1_ ? rho0_ : rho1)
    {
    this->set_gpu_runnable(true);
    mixfr_ = this->template create_field_request<FieldT>(mixfrTag);
    }

    //--------------------------------------------------------------------

    template< typename FieldT >
    void
    TwoStreamDensFromMixfr<FieldT>::
    evaluate()
    {
    using namespace SpatialOps;
    
    typename Expr::Expression<FieldT>::ValVec& results  = this->get_value_vec();
    FieldT& rho    = *results[0];
    FieldT& drhodf = *results[1];
    
    const FieldT& f = mixfr_->field_ref();
    
    rho <<= 1.0 / ( f/rho1_ + (1.0-f)/rho0_ );
    // repair bounds
    rho <<= max ( min(rho, rhoMax_), rhoMin_);

    drhodf <<= -rho0_*rho1_*(rho0_-rho1_)/( (rho0_ - rho1_)*f - rho1_ )/( (rho0_ - rho1_)*f - rho1_ );
    }

    //--------------------------------------------------------------------

    template< typename FieldT >
    TwoStreamDensFromMixfr<FieldT>::
    Builder::Builder( const Expr::TagList& resultsTagList,
                    const Expr::Tag& mixfrTag,
                    const double rho0,
                    const double rho1 )
    : ExpressionBuilder( resultsTagList ),
        rho0_(rho0), rho1_(rho1),
        mixfrTag_( mixfrTag )
    {}

    //====================================================================
    // explicit template instantiation
    #include <spatialops/structured/FVStaggeredFieldTypes.h>
    template class TwoStreamDensFromMixfr<SpatialOps::SVolField>;
    template class TwoStreamMixingDensity<SpatialOps::SVolField>;
}