#ifndef Dev_SingleRateModel_Expr_h
#define Dev_SingleRateModel_Expr_h

#include <expression/Expression.h>
#include "SingleRateData.h"
#include <CCA/Components/Wasatch/Coal/Devolatilization/CPD/Eb_fun.h>

/**
 *  \class   SingleRateModel
 *  \ingroup Devolatilization
 *  \brief   Evaluates the devolatilization rate and product by 
 *           Single Rate  model
 *  \f[
 *      CH_{h}O_{o}\rightarrow oCO+\frac{h-(1-o)}{2}H_{2}+\frac{(1-o)}{2}C_{2}H_{2}
 *  \f]
 *  \author  Babak Goshayeshi
 *
 *  \param   tempPtag : Particle Temperature tag in K
 *  \param   mvtag    : Volatile matter tag ( kg )
 *
 */
namespace SNGRATE {


template <typename FieldT>
class SingleRateModel
 : public Expr::Expression<FieldT>
{
  typedef typename Expr::Expression<FieldT>::ValVec SpecT;

  DECLARE_FIELDS( FieldT, tempP_, mv_, initprtmas_ )
  const double mw_, h_, o_, tarMW_, volatilefrac_;
  const bool isDAE_;
  
  SingleRateModel( const Expr::Tag& tempPtag,
                   const Expr::Tag& mvtag,
                   const Expr::Tag& initprtmast,
                   const double volatilefrac,
                   const SingleRateInformation& data,
                   const bool isDAE);
public:
  class Builder : public Expr::ExpressionBuilder
  {
    const Expr::Tag tempPt_, mvt_, initprtmast_;
    const SingleRateInformation data_;
    const bool isDAE_;
    const double volatilefrac_;
  public:
    /**
     *  brief Build a SingleRateModel expression
     *  param tempPtag : Particle Temperature tag in K
     *  param mvtag    : Volatile matter tag ( kg )
     */
    Builder( const Expr::TagList& resultTag, 
             const Expr::Tag& tempPtag,
             const Expr::Tag& mvtag,
             const Expr::Tag& initprtmast,
             const double volatilefrac,
             const SingleRateInformation& data,
             const bool isDAE )
    : Expr::ExpressionBuilder( resultTag ),
      tempPt_      (tempPtag    ),
      mvt_         (mvtag       ),
      initprtmast_ (initprtmast ),
      data_        (data        ),
      isDAE_       (isDAE       ),
      volatilefrac_(volatilefrac)
    {}

    Expr::ExpressionBase* build() const{
      return new SingleRateModel<FieldT>( tempPt_, mvt_, initprtmast_, volatilefrac_, data_, isDAE_);
    }

  };

  ~SingleRateModel(){}
  void evaluate();
};



// ###################################################################
//
//                          Implementation
//
// ###################################################################


template <typename FieldT>
SingleRateModel<FieldT>::
SingleRateModel( const Expr::Tag& tempPtag,
                 const Expr::Tag& mvtag,
                 const Expr::Tag& initprtmast,
                 const double volatilefrac,
                 const SingleRateInformation& data,
                 const bool isDAE)
: Expr::Expression<FieldT>(),
  mw_   ( data.get_molecularweight()      ),
  h_    ( data.get_hydrogen_coefficient() ),
  o_    ( data.get_oxygen_coefficient()   ),
  tarMW_( data.get_tarMonoMW()            ),
  volatilefrac_( volatilefrac ),
  isDAE_(isDAE)
{
  this->set_gpu_runnable(true);

  tempP_ = this->template create_field_request<FieldT>( tempPtag );
  mv_    = this->template create_field_request<FieldT>( mvtag    );

  if( isDAE ) initprtmas_ = this->template create_field_request<FieldT>( initprtmast );
}

//--------------------------------------------------------------------

template <typename FieldT>
void
SingleRateModel<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  SpecT& mvcharrhs = this->get_value_vec();

  FieldT& mvrhs   = *mvcharrhs[0]; // Volitle RHS

  // Produced Species according to reaction [1]
  FieldT& co   = *mvcharrhs[1]; // CO   Consumption rate ( for gas Phase )
  FieldT& h2   = *mvcharrhs[2]; // H2   Consumption rate ( for gas Phase )
  FieldT& tar  = *mvcharrhs[3]; // tar  Consumption rate ( for gas Phase )

  const FieldT& tempP = tempP_->field_ref();
  const FieldT& mv    = mv_   ->field_ref();

  if( isDAE_ ) {
    // Distributed Acticvation Energy Model
    // Parameters for DAE
    // Pedel, J., Thornock, J., & Smith, P. (n.d.).
    // Validation and Uncertainty Quantification of Coal Devolatization Models.
    const double E0 = 215.0E3; // J/ mol; 
    const double A0 = 3.5318e+13; // (1/s) 10^(5.72e-5*E0+1.25)
    const double sigma = 19.95E3; // J

    const FieldT& initprtmas = initprtmas_->field_ref();
    SpatFldPtr<FieldT> tmp = SpatialFieldStore::get<FieldT,FieldT>(mvrhs);
    SpatFldPtr<FieldT>   p = SpatialFieldStore::get<FieldT,FieldT>(mvrhs);

    *p <<= mv / initprtmas / volatilefrac_;
    CPD::Eb_fun( *tmp, *p, E0, sigma );

    mvrhs <<= -A0 * exp( - *tmp / ( 8.31446 * tempP ) ) * mv;
  }
  else{   // Single Rate Model
    // Arheniuse parameters for Single Rate Model from Jovanovic (2012):
    const double A0  =  4.5E5;      // 1/s; k1 pre-exponential factor
    const double E0  =  8.1E4;      // J/ mol;  k1 activation energy
    mvrhs <<= -A0 * exp(-E0/ 8.31446 / tempP) * mv;
  }
  /*
   * a,b, and c are derived from stoichiometric coefficients for the reaction
   * CHaOb --> b*CO + c1*H2 + c2*CjHk,
   * c2 = (1-b)/j, and c1 = (a-c2*k)/2
   *
   * Currently, tar is C10H8.
   */
  co    <<= mvrhs / mw_ * o_ * 28.0;
  h2    <<= mvrhs / mw_ * (h_ - (1-o_)*8.0/10.0); // / 2 / 2 -> * 1
  tar   <<= mvrhs / mw_ * (1-o_)/10.0 * tarMW_;
}

//--------------------------------------------------------------------

} // end of namespace SNGRATE

#endif // Dev_SingleRateModel_h
