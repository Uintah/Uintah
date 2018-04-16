#ifndef kg_i_Expr_h
#define kg_i_Expr_h

#include <expression/Expression.h>

#include "CPDData.h"
#include "c0_fun.h"

namespace CPD{

/**
 *  \ingroup CPD
 *  \class kg_i
 *  Calculating the reaction constant for each g_i,
 *    \f[ kg_{i}=A_{i}exp\left(\frac{-E_{i}}{RT}\right) \f]
 *  where
 *
 *  \f$A_{i}\f$ is constant and obtain from CPDInformation class
 *  \f$E_{i}\f$ is change with time and being calculated by \codeEb_fun()\endcode.
 */
template <typename FieldT>
class kg_i
  : public Expr::Expression<FieldT>
{
  const CPDInformation& cpd_;
  const std::vector<double>& A0_, E0_, sigma_;
  std::vector<double> gimax_;

  DECLARE_FIELDS( FieldT, temp_, initprtmas_ )
  DECLARE_VECTOR_OF_FIELDS( FieldT, gi_ )

  kg_i( const Expr::TagList& giTag,
        const Expr::Tag& tempTag,
        const Expr::Tag& initprtmastag,
        const CPDInformation& cpd );

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  \param kgiTag
     *  \param giTag     gi amount kg (internal CPD expression )
     *  \param tempTag   Particle Temperature K
     *  \param initprtmastag   Initial particle mass kg
     */
    Builder( const Expr::TagList& kgiTag,
             const Expr::TagList& giTag,
             const Expr::Tag& tempTag,
             const Expr::Tag& initprtmastag,
             const CPDInformation& cpd );
    ~Builder(){}
    Expr::ExpressionBase* build() const;

  private:
    const Expr::Tag tempt_;
    const Expr::TagList git_;
    const Expr::Tag initprtmast_;
    const CPDInformation& cpd_;
  };

  void evaluate();
};



// ###################################################################
//
//                          Implementation
//
// ###################################################################


template<typename FieldT>
kg_i<FieldT>::
kg_i( const Expr::TagList& giTag,
      const Expr::Tag& tempTag,
      const Expr::Tag& initprtmastag,
      const CPDInformation& cpd )
  : Expr::Expression<FieldT>(),
    cpd_ ( cpd ),
    A0_( cpd.get_A0() ),
    E0_( cpd.get_E0() ),
    sigma_( cpd.get_sigma())
{
  this->set_gpu_runnable(true);

  const std::vector<double>& fg = cpd_.get_fgi();
  const double sumfg = cpd_.get_sumfg();
  const double c0 = c0_fun( cpd_.get_coal_composition().get_C(), cpd_.get_coal_composition().get_O());
  const size_t nspec = cpd_.get_nspec();
  gimax_.resize(nspec,0.0);
  for( size_t i=0; i<nspec; ++i ){
	gimax_[i] = 2.0 * (1.0-c0) * fg[i] / sumfg;
  }

  temp_       = this->template create_field_request<FieldT>( tempTag       );
  initprtmas_ = this->template create_field_request<FieldT>( initprtmastag );

  this->template create_field_vector_request<FieldT>( giTag, gi_ );
}

//--------------------------------------------------------------------

template<typename FieldT>
void
kg_i<FieldT>::
evaluate()
{
  using namespace SpatialOps;

  typename Expr::Expression<FieldT>::ValVec& kgi = this->get_value_vec();

  const double vm = cpd_.get_coal_composition().get_vm();

  const FieldT& temp       = temp_      ->field_ref();
  const FieldT& initprtmas = initprtmas_->field_ref();

  SpatFldPtr<FieldT>  tmp = SpatialFieldStore::get<FieldT,FieldT>( temp );
  SpatFldPtr<FieldT> prob = SpatialFieldStore::get<FieldT,FieldT>( temp );

  for( size_t i=0; i<gi_.size(); ++i ){

    const FieldT& gF  = gi_[i]->field_ref();
    FieldT& kgF = *kgi[i];
    if( gimax_[i] == 0 ){
      kgF <<= 0.0;
      continue;
    }

    //*prob <<= gF / ( gimax_[i] * initprtmas * vm / cpd_.get_hypothetical_volatile_mw() ); //this assumes mole basis
    *prob <<= gF / ( gimax_[i] * initprtmas * vm ); //this assumes mass basis
    Eb_fun( *tmp, *prob, E0_[i], sigma_[i] );
    kgF <<= A0_[i] * exp( -*tmp / temp ); // E0 was divided by R already !
  }
}
//--------------------------------------------------------------------

template<typename FieldT>
kg_i<FieldT>::
Builder::Builder( const Expr::TagList& kgiTag,
                  const Expr::TagList& giTag,
                  const Expr::Tag& tempTag,
                  const Expr::Tag& initprtmastag,
                  const CPDInformation& cpd )
 : ExpressionBuilder(kgiTag),
   tempt_( tempTag ),
   git_  ( giTag   ),
   initprtmast_( initprtmastag),
   cpd_  ( cpd     )
{}

//--------------------------------------------------------------------

template<typename FieldT>
Expr::ExpressionBase*
kg_i<FieldT>::
Builder::build() const
{
  return new kg_i<FieldT>( git_, tempt_, initprtmast_, cpd_ );
}

} // namespace CPD

#endif // Gi_RHS_Expr_h
