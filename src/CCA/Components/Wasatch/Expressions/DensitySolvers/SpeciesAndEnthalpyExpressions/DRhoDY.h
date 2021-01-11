#ifndef DRhoDy_h
#define DRhoDy_h

#include <expression/Expression.h>
#include <spatialops/structured/SpatialFieldStore.h>
#include <pokitt/CanteraObjects.h>


/**
 *  \class DRhoDY
 *  \author Josh McConnell
 *  \date   March 2017
 *
 *  \brief Computes the vector given by \f$\frac{\partial \rho}{\partial Y}\f$
 *  where \f$Y\f$ is the vector of the first \f$n-1\f$ species mass fractions
 *  in the system. Each element of \f$\frac{\partial \rho}{\partial Y}\f$ is
 *  given by
 *  \f[
 *   \frac{\partial \rho}{\partial Y_i} = \rho \M (1/M_i - 1/M_n),
 *  \f]
 *  where \f$\M\f$ is the mixture molecular weight and \f$\M_i\f$ is the
 *  molecular weight of species \f$\i\f$.
 */
namespace WasatchCore{

template<typename FieldT>
class DRhoDY : public Expr::Expression<FieldT>
{
  DECLARE_FIELDS(FieldT, rho_, mmw_ )


  const double nSpec_;
  std::vector<double> mwInvTerm_;

  DRhoDY( const Expr::Tag&     rhoTag,
          const Expr::Tag&     mmwTag );

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  @brief Build a PartialJacobian_Species expression
     *  @param resultTag the tag for the value that this expression computes
     */
    Builder( const Expr::TagList& resultTags,
             const Expr::Tag&     rhoTag,
             const Expr::Tag&     mmwTag );

    Expr::ExpressionBase* build() const{
      return new DRhoDY( rhoTag_, mmwTag_ );
    }

  private:
    const Expr::Tag rhoTag_, mmwTag_;
  };

  void evaluate();

};


// ###################################################################
//
//                          Implementation
//
// ###################################################################

template<typename FieldT>
DRhoDY<FieldT>::
DRhoDY( const Expr::Tag&     rhoTag,
        const Expr::Tag&     mmwTag )
  : Expr::Expression<FieldT>(),
    nSpec_( CanteraObjects::number_species()    )
{
  this->set_gpu_runnable(true);

  const std::vector<double>& mw =  CanteraObjects::molecular_weights();
  mwInvTerm_.clear();
  for(int j=0; j<nSpec_-1; ++j)
  mwInvTerm_.push_back( 1.0/mw[j] - 1.0/mw[nSpec_-1] );

  rho_ = this->template create_field_request<FieldT>( rhoTag );
  mmw_ = this->template create_field_request<FieldT>( mmwTag );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
DRhoDY<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  typename Expr::Expression<FieldT>::ValVec&  resultVec = this->get_value_vec();

  const FieldT& rho = rho_->field_ref();
  const FieldT& mmw = mmw_->field_ref();

  SpatFldPtr<FieldT> x = SpatialFieldStore::get<FieldT>( rho );
  *x <<= -rho*mmw;

  for(int j=0; j<nSpec_-1; ++j){
    *resultVec[j] <<= (*x)*mwInvTerm_[j];
  }
}

//--------------------------------------------------------------------

template<typename FieldT>
DRhoDY<FieldT>::
Builder::Builder( const Expr::TagList& resultTags,
                  const Expr::Tag&     rhoTag,
                  const Expr::Tag&     mmwTag )
  : ExpressionBuilder( resultTags ),
    rhoTag_ ( rhoTag ),
    mmwTag_ ( mmwTag )
{
  const int m     = resultTags.size();
  const int nSpec = CanteraObjects::number_species();
  if( resultTags.size() != nSpec - 1 )
  {
    std::ostringstream msg;
    msg << __FILE__ << " : " << __LINE__
        << std::endl
        <<"the number of result tags, must equal n-1, where n is the number of species"
        <<"number of species     : "<< nSpec << std::endl
        <<"number of resultTags  : "<< m     << std::endl;
    throw std::runtime_error( msg.str() );
  }
}

//====================================================================
}

#endif
