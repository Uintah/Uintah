#ifndef Dev_KobayashiSarofim_Expr_h
#define Dev_KobayashiSarofim_Expr_h

#include <expression/Expression.h>

#include <CCA/Components/Wasatch/Coal/Devolatilization/KobayashiSarofim/KobSarofimData.h>
#include <spatialops/structured/SpatialFieldStore.h>

/**
 *  \class   KobayashiSarofim
 *  \ingroup Devolatilization
 *  \brief   Evaluates the devolatilization rate and product by 
 *           Kobayashi & Sarofim model
 *
 *  \author  Babak Goshayeshi
 *
 *  \param   tempPtag : Particle Temperature tag in K
 *  \param   mvtag    : Volatile matter tag ( kg )
 *
 */
namespace SAROFIM {


template <typename FieldT>
class KobayashiSarofim
 : public Expr::Expression<FieldT>
{
  DECLARE_FIELDS( FieldT, tempP_, mv_, u_, o_ )
  const double tarMW_;

  KobayashiSarofim( const Expr::Tag& tempPtag,
                    const Expr::Tag& mvtag,
                    const Expr::TagList& elementtags,
                    const KobSarofimInformation& data);
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  brief Build a KobayashiSarrofim expression
     *  param tempPtag : Particle Temperature tag in K
     *  param mvtag    : Volatile matter tag ( kg )
     */
    Builder( const Expr::TagList& resultTag, 
             const Expr::Tag& tempPtag,
             const Expr::Tag& mvtag,
             const Expr::TagList& elementtags,
             const KobSarofimInformation& data);

    Expr::ExpressionBase* build() const;

  private:
    const Expr::Tag tempPt_, mvt_;
    const Expr::TagList elementst_;
    const KobSarofimInformation data_;
  };

  ~KobayashiSarofim(){}
  void evaluate();
};



// ###################################################################
//
//                          Implementation
//
// ###################################################################


template <typename FieldT>
KobayashiSarofim<FieldT>::
KobayashiSarofim( const Expr::Tag& tempPtag,
                  const Expr::Tag& mvtag,
                  const Expr::TagList& elementtags,
                  const KobSarofimInformation& data )
: Expr::Expression<FieldT>(),
  tarMW_( data.get_tarMonoMW() )
{
  tempP_ = this->template create_field_request<FieldT>( tempPtag );
  mv_    = this->template create_field_request<FieldT>( mvtag );
  u_     = this->template create_field_request<FieldT>( elementtags[0] );
  o_     = this->template create_field_request<FieldT>( elementtags[1] );
}

//--------------------------------------------------------------------

template <typename FieldT>
void
KobayashiSarofim<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  typename Expr::Expression<FieldT>::ValVec& mvcharrhs = this->get_value_vec();

  FieldT& mvrhs   = *mvcharrhs[0]; // Volitle RHS
  FieldT& charrhs = *mvcharrhs[1]; // Char    RHS

  // Produced Species according to reaction [1]
  FieldT& co      = *mvcharrhs[2]; // CO   Consumption rate ( for gas Phase )
  FieldT& h2      = *mvcharrhs[3]; // H2   Consumption rate ( for gas Phase )
  FieldT& tar    =  *mvcharrhs[4];  // tar  Consumption rate ( for gas Phase )

  // FieldT& dk      = *mvcharrhs[5]; // Carbon Consumption rate
  FieldT& du      = *mvcharrhs[5]; // Hydrogen Consumption rate
  FieldT& dox     = *mvcharrhs[6]; // Oxygen Consumption rate

  const double R =  1.987;        // kcal/kmol; ideal gas constant

  // Values from Ubhayakar (1976):
  const double A1  =  3.7e5;      // 1/s; k1 pre-exponential factor
  const double A2  =  1.46e13;    // 1/s; k2 pre-exponential factor
  const double E1  =  17600;      // kcal/kmol;  k1 activation energy
  const double E2  =  60000;      // kcal/kmol;  k2 activation energy
  /*
  const double Y1 = 0.39; // volatile fraction from proximate analysis
  const double Y2 = 0.80; // fraction devolatilized at higher temperatures
   */
  /*
   // Values from Kobayashi (1976):
  const double A1  =  2.0e5;       // 1/s; pre-exponential factor for k1
  const double A2  =  1.3e7;       // 1/s; pre-exponential factor for k2
  const double E1  =  -25000;      // kcal/kmol;  k1 activation energy
  const double E2  =  -40000;      // kcal/kmol;  k2 activation energy
   */

  /* Method of calculation 
   *
   *  B1 = A1 * exp(-E1/RT) * mv
   *  B2 = A2 * exp(-E2/RT) * mv
   *
   *  mvrhs   = B1 + B2
   *  charrhs = (1-Y1)B1 + (1-Y2)B2 = (1-Y1)B1
   *  Species = Y1B1 + Y2B2 = Y1B1 + B2
   */
  // Y values from white book:
  
  const double alpha1 = 0.3; // volatile fraction from proximate analysis
  const double alpha2 = 1.0; // fraction devolatilized at higher temperatures

  SpatFldPtr<FieldT> r1 = SpatialFieldStore::get<FieldT,FieldT>( mvrhs );
  SpatFldPtr<FieldT> r2 = SpatialFieldStore::get<FieldT,FieldT>( mvrhs );

  SpatFldPtr<FieldT> beta = SpatialFieldStore::get<FieldT,FieldT>( mvrhs );
  
  SpatFldPtr<FieldT> x = SpatialFieldStore::get<FieldT,FieldT>( mvrhs );
  SpatFldPtr<FieldT> y = SpatialFieldStore::get<FieldT,FieldT>( mvrhs );
  
  const FieldT& tempP = tempP_->field_ref();
  const FieldT& mv    = mv_   ->field_ref();

  *r1 <<= A1 * exp(-E1/(R * tempP));
  *r2 <<= A2 * exp(-E2/(R * tempP));

  mvrhs   <<= -(*r1 + *r2) * mv; // volatile mass RHS;

  *beta   <<= *r1 / (*r1 + *r2);
  
  *x <<= *beta * alpha1 + (1.0 - *beta) * alpha2;
  *y <<= *beta * (1.0 - alpha1) + (1.0 - *beta) * (1.0 - alpha2);

  typename FieldT::const_iterator ix = x->begin();
  typename FieldT::const_iterator iy = y->begin();

  typename FieldT::const_iterator io = o_->field_ref().begin();
  typename FieldT::const_iterator iu = u_->field_ref().begin();

  typename FieldT::const_iterator imv= mv.begin();

  typename FieldT::iterator imvrhs   = mvrhs.begin();
  typename FieldT::iterator icharrhs = charrhs.begin();
  typename FieldT::iterator ico      = co.begin();
  typename FieldT::iterator ih2      = h2.begin();
  typename FieldT::iterator itar    = tar.begin();

  typename FieldT::iterator idu = du.begin();
  typename FieldT::iterator idox= dox.begin();

  for( ; ix!=x->end() ; ++ix, ++iy, ++io, ++iu, ++imv, ++imvrhs, ++icharrhs,
                        ++ico, ++ih2, ++itar, ++idu, ++idox)
  {
    // In some of the included equations, x is appeared in the denominator 
    if (*ix > 0.0){ // if alpha1 > alpha2 x value cannot be zero!
      /*
       * a,b, and c are derived from stoichiometric coefficients for the reaction
       * CHwOz --> z*CO + c1*H2 + c2*CjHk,
       * c2 = (1-z)/j, and c1 = (w-c2*k)/2
       */
      const double a = *io / *ix;

      const double c = (1.0 - *iy - *io) > 0.0 ? (1.0 - *iy - *io) / 10.0 / *ix : 0.0;
      const double b = *iu / 2.0 / *ix - c*8.0;

      const double mwv = 12.0 + *iu + *io * 16.0;

      *ico    = *ix * a * *imvrhs / mwv * 28.0;
      *ih2    = *ix * b * *imvrhs / mwv * 2.0;
      *itar   = *ix * c * *imvrhs / mwv * tarMW_;

      *icharrhs = - *iy * *imvrhs / mwv * 12.0;

      *idu      = (*ih2 + 2 * *itar / tarMW_ ) / (*imv / mwv);
      *idox     = (*ico / 28.0) / (*imv / mwv);

      // The carbon number is always one therefore hydrogen and oxygen number should
      // change accordingly. 
      const double dk = (*ico / 28.0 + 2 * *itar / tarMW_ + *icharrhs / 12.0) / (*imv / mwv);

      *idu = (*idu  * 1.0  - *iu * dk) / (1.0 - dk);
      *idox= (*idox * 1.0  - *io * dk) / (1.0 - dk);

    }
    else{
      *ico    = 0.0;
      *ih2    = 0.0;
      *itar   = 0.0;

      const double mwv = 12.0 + *iu + *io * 16.0;
      
      *icharrhs = - *imvrhs / mwv * 12.0;
      *idu      = 0.0; //  x is zero when u and h are zero! 
      *idox     = 0.0;
    } // else
  } // for
}

//--------------------------------------------------------------------

template <typename FieldT>
KobayashiSarofim<FieldT>::
Builder::Builder( const Expr::TagList& resultTag,
                  const Expr::Tag& tempPtag,
                  const Expr::Tag& mvtag,
                  const Expr::TagList& elementtags,
                  const KobSarofimInformation& data )
: ExpressionBuilder( resultTag ),
  tempPt_(tempPtag),
  mvt_   (mvtag   ),
  elementst_ (elementtags ),
  data_  (data    )
{}

//--------------------------------------------------------------------

template <typename FieldT>
Expr::ExpressionBase*
KobayashiSarofim<FieldT>::
Builder::build() const
{
  return new KobayashiSarofim<FieldT>( tempPt_, mvt_, elementst_, data_);
}
//--------------------------------------------------------------------

} // end of namespace SAROFIM

#endif // Dev_KobayashiSarrofim_Expr_h
