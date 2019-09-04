#ifndef CoalHeatCapacity_coal_h
#define CoalHeatCapacity_coal_h

#include <expression/Expression.h>

/**
 *   Author : Babak Goshayeshi
 *   Date   : Feb 11 2011
 *   University of Utah - Institute for Clean and Secure Energy
 *
 *   Specific Heat Capacity of Coal
 *   mvolt   : mass of Volatile
 *   mchart  : mass of Fixed Carbon
 *   mmoistt : mass of moisture
 *   prtmast   : Mass of Partile
 *   tempPt  : temperature of particle
 *
 *
 *   Refrences :
 *
 *   [1] MacDonald, Rosemary a., Jane E. Callanan, and Kathleen M. McDermott.
 *       Heat capacity of a medium-volatile bituminous premium coal from 300 to 520 K.
 *       Comparison with a high-volatile bituminous nonpremium coal,
 *       Energy & Fuels 1, no. 6 (November 1987): 535-540.
 *       http://pubs.acs.org/doi/abs/10.1021/ef00006a014.
 *
 *
 *   [2] W. Eisermann, P. Johnson, W.L. Conger, Estimating thermodynamic properties of
 *       coal, char, tar and ash,Fuel Processing Technology, Volume 3,
 *       Issue 1, January 1980, Pages 39-53
 *       http://www.sciencedirect.com/science/article/pii/0378382080900223
 *
 *
 */
namespace Coal {

template< typename FieldT >
class CoalHeatCapacity
 : public Expr::Expression<FieldT>
{
  DECLARE_FIELDS( FieldT, mvol_, mchar_, mmois_, prtmas_, tempP_ )

  CoalHeatCapacity( const Expr::Tag& mvolt,
                    const Expr::Tag& mchart,
                    const Expr::Tag& mmoistt,
                    const Expr::Tag& prtmast,
                    const Expr::Tag& tempPt );

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& coalCpTag,
             const Expr::Tag& mvolt,
             const Expr::Tag& mchart,
             const Expr::Tag& mmoistt,
             const Expr::Tag& prtmast,
             const Expr::Tag& tempPt );
    ~Builder(){}
    Expr::ExpressionBase* build() const;

  private:
    const Expr::Tag mvolt_, mchart_, mmoistt_, prtmast_, tempPt_;
  };

  void evaluate();
};



// ###################################################################
//
//                          Implementation
//
// ###################################################################



template< typename FieldT >
CoalHeatCapacity<FieldT>::
CoalHeatCapacity(const Expr::Tag& mvolt,
                 const Expr::Tag& mchart,
                 const Expr::Tag& mmoistt,
                 const Expr::Tag& prtmast,
                 const Expr::Tag& tempPt )
  : Expr::Expression<FieldT>()
{
  this->set_gpu_runnable(true);

  mvol_   = this->template create_field_request<FieldT>( mvolt   );
  mchar_  = this->template create_field_request<FieldT>( mchart  );
  mmois_  = this->template create_field_request<FieldT>( mmoistt );
  prtmas_ = this->template create_field_request<FieldT>( prtmast );
  tempP_  = this->template create_field_request<FieldT>( tempPt  );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
CoalHeatCapacity<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& cp = this->value();

  const FieldT& mvol   = mvol_  ->field_ref();
  const FieldT& mchar  = mchar_ ->field_ref();
  const FieldT& mmois  = mmois_ ->field_ref();
  const FieldT& prtmas = prtmas_->field_ref();
  const FieldT& tempP  = tempP_ ->field_ref();

  SpatFldPtr<FieldT> g1 = SpatialFieldStore::get<FieldT,FieldT>( cp );
  *g1 <<= ( exp( 380.0 / tempP ) - 1.0 ) / ( 380.0 / tempP );
  *g1 <<= exp( 380.0 / tempP ) / ( *g1 * *g1 );

  SpatFldPtr<FieldT> g2 = SpatialFieldStore::get<FieldT,FieldT>( cp );
  *g2 <<= ( exp( 1800.0 / tempP ) - 1.0 ) / ( 1800.0 / tempP );
  *g2 <<= exp( 1800.0 / tempP ) / ( *g2 * *g2 );

  SpatFldPtr<FieldT> ashMass = SpatialFieldStore::get<FieldT,FieldT>( cp );
  *ashMass <<= prtmas - ( mchar + mmois + mvol );

  cp <<= (
           cond( mchar > 0.0, (8.314/12.0*( *g1 + 2.0 * *g2 )) * mchar )
               ( 0.0 )
         + cond( mvol > 0.0, ( 1.5005 + 2.9725E-3 * tempP) * mvol )
               ( 0.0 )
         + cond( mmois > 0.0,  4.2 * mmois )
               ( 0.0 )
         + ( 0.594 + 5.86E-4 * tempP) * *ashMass
         ) * 1000.0 / prtmas;
}

//--------------------------------------------------------------------

template< typename FieldT >
CoalHeatCapacity<FieldT>::
Builder::Builder(const Expr::Tag& coalCpTag,
                 const Expr::Tag& mvolt,
                 const Expr::Tag& mchart,
                 const Expr::Tag& mmoistt,
                 const Expr::Tag& prtmast,
                 const Expr::Tag& tempPt)
  : ExpressionBuilder(coalCpTag),
    mvolt_  ( mvolt   ),
    mchart_ ( mchart  ),
    mmoistt_( mmoistt ),
    prtmast_( prtmast ),
    tempPt_ ( tempPt  )
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
CoalHeatCapacity<FieldT>::Builder::build() const
{
  return new CoalHeatCapacity<FieldT>( mvolt_, mchart_, mmoistt_, prtmast_, tempPt_ );
}

} // namespace coal

#endif // CoalHeatCapacity_coal_h
