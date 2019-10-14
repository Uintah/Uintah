#ifndef TarProductionRate_h
#define TarProductionRate_h

#include <expression/Expression.h>

//#include "CPDData.h"


/**
 *  @class TarProductionRate
 *  @author Josh McConnell
 *  @date   December, 2014
 *  @brief Calculates the rate of production of tar (in kg). It is assumed that
 *         Volitilizable aromatic ring sites are initially in an infinite
 *         Bethe psuedolattice, and that all tar produced is composed of monomers
 *
 *
 *  The tar production rate is given as
 * \f[
 *     r_{\mathrm{tar}} = mass_{p,0} * x_{v,0} * (1 - x_{l.0} - sum{x_{g_{i}}} * \frac{dQ_{1}}{dt},
 * \f]
 *
 * with
 * \f[
 *     Q_{1} = (1-p)^(s + 3)
 * \f]
 * 
 *  <ul>
 *  <li> \f$ m_{p} \f$ is the mass of the coal particles,
 *  <li> \f$ s + 1 \f$ is the coordination number for the coal
 *       lattice structure,
 *  <li> \f$ x_{{v}} \f$ is the volatiliziable mass fraction in the coal,
 *  <li> \f$ x_{l,0} \f$ is the mass fraction of labile bridge in th volatiliziable mass,
 *  <li> \f$ x_{g_{i}} \f$ are the mass fraction of functional groups in th volatiliziable mass, and
 *  <li> \f$ p = l/l_{0} \f$ is the normalized population of intact labile
 *        and char bridges.
 *  </ul>
 *
 *  @param lbPop:     (kg lb/kg lb0) labile bridge population
 *  @param lbPopRHS:  (1/s) rate of change of labile bridge population
 *  @param prtMass0_: (kg) initial particle mass
 *  @param vmFrac0_:  (kg vm/kg) intital weight fraction of volatile mass in coal
 *  @param tar0_:     (kg tar/kg) intital weight fraction of volatile mass in coal
 */

template< typename FieldT >
class TarProductionRate
 : public Expr::Expression<FieldT>
{
  typedef CPD::CPDInformation CPDInfo;

  DECLARE_FIELDS( FieldT, lbPop_,lbPopRHS_, prtMass0_ )
  const double  tar0_;
  const double  vmFrac0_;
  const CPDInfo& cpdInfo_;

  /* declare operators associated with this expression here */

  TarProductionRate( const Expr::Tag& lbPopTag,
                     const Expr::Tag& lbPopRHSTag,
                     const Expr::Tag& prtMass0Tag,
                     const double vmFrac0,
                     const double tar0,
                     const CPDInfo& cpdInfo );
public:
  class Builder : public Expr::ExpressionBuilder
  {
    const Expr::Tag lbPopTag_, lbPopRHSTag_, prtMass0Tag_;
    const double vmFrac0_, tar0_;
    const CPDInfo& cpdInfo_;
  public:
    /**
     *  @brief Build a TarProductionRate expression
     *  @param resultTag the tag for the value that this expression computes
     */
    Builder( const Expr::Tag& resultTag,
             const Expr::Tag& lbPopTag,
    	        const Expr::Tag& lbPopRHSTag,
             const Expr::Tag& prtMass0Tag,
             const double vmFrac0,
             const double tar0,
             const CPDInfo& cpdInfo )
      : ExpressionBuilder ( resultTag ),
        lbPopTag_   ( lbPopTag    ),
        lbPopRHSTag_( lbPopRHSTag ),
        prtMass0Tag_( prtMass0Tag ),
        vmFrac0_    ( vmFrac0     ),
        tar0_       ( tar0        ),
        cpdInfo_    ( cpdInfo     )
      {}

    Expr::ExpressionBase* build() const{
      return new TarProductionRate<FieldT>(lbPopTag_,lbPopRHSTag_,prtMass0Tag_,vmFrac0_,tar0_,cpdInfo_ );
    }
  };

  ~TarProductionRate(){}
  void evaluate();
};



// ###################################################################
//
//                          Implementation
//
// ###################################################################



template< typename FieldT >
TarProductionRate<FieldT>::
TarProductionRate( const Expr::Tag& lbPopTag,
		   const Expr::Tag& lbPopRHSTag,
                   const Expr::Tag& prtMass0Tag,
                   const double    vmFrac0,
                   const double    tar0,
                   const CPDInfo&   cpdInfo )
  : Expr::Expression<FieldT>(),
    tar0_   ( tar0    ),
    vmFrac0_( vmFrac0 ),
    cpdInfo_( cpdInfo )
{
  this->set_gpu_runnable(true);

  lbPop_ =     this->template create_field_request<FieldT>( lbPopTag    );
  lbPopRHS_  = this->template create_field_request<FieldT>( lbPopRHSTag );
  prtMass0_  = this->template create_field_request<FieldT>( prtMass0Tag );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
TarProductionRate<FieldT>::
evaluate()
{
  using namespace SpatialOps;

  const double s   = cpdInfo_.get_coordNo() - 1;  // (coordination number) -1 of coal lattice
  const double tau = s + 1;

  FieldT& result = this->value();

  // Result should always be negative. Result is defined to be negative because
  // of the convention set for all other species (negative ==> production).
  const FieldT& lbPop =    lbPop_   ->field_ref();
  const FieldT& lbPopRHS = lbPopRHS_->field_ref();
  const FieldT& prtMass0 = prtMass0_->field_ref();

  result <<= tau * pow(1.0 - lbPop, tau - 1)
                 * vmFrac0_ * tar0_ * (prtMass0) * (lbPopRHS);
}

//--------------------------------------------------------------------

#endif // TarProductionRate_h
