#ifndef Deltai_RHS_Expr_h
#define Deltai_RHS_Expr_h

#include <expression/Expression.h>
#include "CPDData.h"
#include <vector>

namespace CPD{

  /**
   *  \ingroup CPD
   *  \class Deltai_RHS
   *
   *  \brief This is the expression of right hand side of the 16
   *	  diffretial equation of delta. So this expresion will return
   *	  a vector.
   *
   *  In this Expression the RHS of the differential equation in below being evaluated :
   *  \f[
   *   \frac{d\delta_{i}}{dt}=\left[\frac{2\rho k_{b}\ell}{\rho+1}\right]\frac{fg_{i}}{\sum_{j=1}^{17}fg_{j}}-k_{gi}\delta_{i}
   *  \f]
   *  where
   *  - \f$ k_{b} \f$ : reaction constant of labile bridge
   *  - \f$ \ell \f$ : labile bridge
   *  - \f$ \rho = 0.9 \f$
   *  - \f$ fg_{i} \f$ = functional group for each bond.
   *  - \f$ \delta_{i} \f$ : amout of side chain i
   * In this expression, \f$ \delta_{i} \f$ is a vector of FieldT.
   */
  template<typename FieldT>
  class Deltai_RHS
    : public Expr::Expression<FieldT>
  {
    const CPDInformation& cpd_;

    DECLARE_FIELDS( FieldT, kb_, l_ )
    DECLARE_VECTOR_OF_FIELDS( FieldT, kgi_ )
    DECLARE_VECTOR_OF_FIELDS( FieldT, deltai_ )

    Deltai_RHS( const Expr::Tag&     kbTag,
                const Expr::TagList& kgiTag,
                const Expr::TagList& deltaiTag,
                const Expr::Tag&     lTag,
                const CPDInformation& cpd );

  public:
    class Builder : public Expr::ExpressionBuilder
    {
    public:
      Builder( const Expr::TagList &rhsTags,
               const Expr::Tag     &kbTag,
               const Expr::TagList &kgiTag,
               const Expr::TagList &deltaiTag,
               const Expr::Tag     &lTag,
               const CPDInformation& cpd );
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
  Deltai_RHS<FieldT>::
  Deltai_RHS( const Expr::Tag     &kbTag,
              const Expr::TagList &kgiTag,
              const Expr::TagList &deltaiTag,
              const Expr::Tag     &lTag,
              const CPDInformation& cpd )
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
  Deltai_RHS<FieldT>::
  evaluate()
  {
    using namespace SpatialOps;

    const double rho                = 0.9;
    const double sumfg              = cpd_.get_sumfg();
    const double mwl0               = cpd_.l0_molecular_weight();
    const std::vector<double>& fgi  = cpd_.get_fgi();
    const std::vector<double>& mwfg = cpd_.get_mwVec();

    typename Expr::Expression<FieldT>::ValVec& rhs = this->get_value_vec();
    const FieldT& kb   = kb_  ->field_ref();
    const FieldT& l    = l_   ->field_ref();

    for( size_t i=0; i<rhs.size(); ++i ){

      FieldT& rhsi = *rhs[i];
      const FieldT& deltaF = deltai_[i]->field_ref();
      const FieldT& kgF = kgi_[i]->field_ref();

      if( fgi[i] == 0.0 ){
        rhsi <<= 0.0;
      }
      else{
        rhsi <<= 2.0 * kb * l * rho / (rho + 1.0 ) * fgi[i] / sumfg
                * mwfg[i]/mwl0
                - cond( deltaF >= 0, kgF * deltaF )
                      ( 0.0 );
      }
    }
  }

  //--------------------------------------------------------------------

  template<typename FieldT>
  Deltai_RHS<FieldT>::
  Builder::Builder( const Expr::TagList &rhsTags,
                    const Expr::Tag     &kbTag,
                    const Expr::TagList &kgiTag,
                    const Expr::TagList &deltaiTag,
                    const Expr::Tag     &lTag,
                    const CPDInformation& cpd )
    : ExpressionBuilder(rhsTags),
      kbt_    ( kbTag     ),
      lt_     ( lTag      ),
      kgit_   ( kgiTag    ),
      deltait_( deltaiTag ),
      cpd_( cpd )
  {}

  //--------------------------------------------------------------------

  template<typename FieldT>
  Expr::ExpressionBase*
  Deltai_RHS<FieldT>::
  Builder::build() const
  {
    return new Deltai_RHS<FieldT>( kbt_ ,kgit_, deltait_, lt_, cpd_ );
  }

  //--------------------------------------------------------------------

} // namespace CPD

#endif // Deltai_RHS_Expr_h
