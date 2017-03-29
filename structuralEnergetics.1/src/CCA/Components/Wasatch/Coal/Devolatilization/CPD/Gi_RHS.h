#ifndef Gi_RHS_Expr_h
#define Gi_RHS_Expr_h

#include <expression/Expression.h>
#include "CPDData.h"

namespace CPD{

/**
 *  \ingroup CPD
 *  \class Gi_RHS
 *   This is the expression of right hand side of the 16 diffretial equation
 *	 of delta. So this expresion will return a vector.
 *
 *  In this Expression the RHS of the differential equation in below being evaluated :
 *
 *  \f[ \frac{dg_{i}}{dt}=\left[\frac{2\rho k_{b}\ell}{\rho+1}\right]\frac{fg_{i}}{\sum_{j}^{17}fg_{j}}+k_{gi}\delta_{i} \f]
 *
 *  where
 *  - \f$ k_{b} \f$ : reaction constant of labile bridge
 *  - \f$\ell\f$ : labile bridge
 *  - \f$\rho = 0.9\f$
 *  - \f$fg_{i}\f$ = functional group for each bond.
 *  - \f$\delta_{i}\f$ : amout of side chain i
 *  - \f$g_{i}\f$ : amout of gas produced from side chains and labile bridge.
 *
 * In this expression, \f$\delta_{i}\f$ and \f$g_{i}\f$ is a vector of FieldT.
 */
 template <typename FieldT>
class Gi_RHS
  : public Expr::Expression<FieldT>
{
  DECLARE_FIELDS( FieldT, kb_, l_ )
  DECLARE_VECTOR_OF_FIELDS( FieldT, kgi_ )
  DECLARE_VECTOR_OF_FIELDS( FieldT, deltai_ )

  const CPDInformation& cpd_;

  Gi_RHS( const Expr::Tag      &kbTag,
          const Expr::TagList  &kgiTag,
          const Expr::TagList  &deltaiTag,
          const Expr::Tag      &lTag,
          const CPDInformation &cpd );

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  \param kbTag : reaction constant of labile bridge (FieldT)
     *  \param kgiTag : reaction constat of gas (vecotr of FieldT)
     *  \particle deltaiTag : moles of delta_i
     *  \param lTag  : moles of laible bridges
     *  \param cpd the CPDInformation object.  This must have a
     *         lifetime at least as long as this expression.
     *
     */
    Builder( const Expr::TagList  &rhsTags,
             const Expr::Tag      &kbTag,
             const Expr::TagList  &kgiTag,
             const Expr::TagList  &deltaiTag,
             const Expr::Tag      &lTag,
             const CPDInformation & cpd );
    ~Builder(){}
    Expr::ExpressionBase* build() const;

  private:
    const Expr::Tag kbt_, lt_;
    const Expr::TagList kgit_, deltait_;
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
Gi_RHS<FieldT>::
Gi_RHS( const Expr::Tag      &kbTag,
        const Expr::TagList  &kgiTag,
        const Expr::TagList  &deltaiTag,
        const Expr::Tag      &lTag,
        const CPDInformation & cpd )
  : Expr::Expression<FieldT>(),
    cpd_( cpd )
{
  this->set_gpu_runnable(true);

  kb_ = this->template create_field_request<FieldT>( kbTag );
  l_  = this->template create_field_request<FieldT>( lTag  );

  this->template create_field_vector_request<FieldT>( kgiTag,    kgi_    );
  this->template create_field_vector_request<FieldT>( deltaiTag, deltai_ );
}

//--------------------------------------------------------------------

template<typename FieldT>
void
Gi_RHS<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  typename Expr::Expression<FieldT>::ValVec& rhs = this->get_value_vec();
  const double rho = 0.9;
  const double sumfg = cpd_.get_sumfg();
  const double mwl0  = cpd_.l0_molecular_weight();

  const std::vector<double>& fgi  = cpd_.get_fgi();
  const std::vector<double>& mwfg = cpd_.get_mwVec();

  const FieldT& kb  = kb_->field_ref();
  const FieldT& ell = l_ ->field_ref();
  for( size_t i=0; i<rhs.size(); ++i ){

    const FieldT& kgF    =    kgi_[i]->field_ref();;
    const FieldT& deltaF = deltai_[i]->field_ref();

    FieldT& rhsF = *rhs[i];
    if( fgi[i] == 0 ){
      rhsF <<= 0.0;
    }
    else{
      rhsF <<= 2.0 * kb * ell / (rho + 1.0 ) * fgi[i] / sumfg
    		  *mwfg[i]/mwl0
      + cond( deltaF >= 0.0, kgF * deltaF )
            ( 0.0 );
    }
  }
}

//--------------------------------------------------------------------

template<typename FieldT>
Gi_RHS<FieldT>::
Builder::Builder( const Expr::TagList  &rhsTags,
                  const Expr::Tag      &kbTag,
                  const Expr::TagList  &kgiTag,
                  const Expr::TagList  &deltaiTag,
                  const Expr::Tag      &lTag,
                  const CPDInformation & cpd )
 : ExpressionBuilder(rhsTags),
   kbt_	   ( kbTag     ),
   lt_     ( lTag      ),
   kgit_   ( kgiTag    ),
   deltait_( deltaiTag ),
   cpd_( cpd )
{}

//--------------------------------------------------------------------

template<typename FieldT>
Expr::ExpressionBase*
Gi_RHS<FieldT>::
Builder::build() const
{
  return new Gi_RHS<FieldT>( kbt_ ,kgit_, deltait_, lt_, cpd_ );
}

} // namespace CPD

#endif // Gi_RHS_Expr_h
