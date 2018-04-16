#ifndef kb_Expr_h
#define kb_Expr_h

#include <expression/Expression.h>

#include "CPDData.h"
#include "Eb_fun.h"

namespace CPD{

/**
 *  \ingroup CPD
 *  \class kb to calculate reaction constant for l ( liable bridige )
 *
 *  \param lTag           :  Liable bridge amount kg
 *  \param temptag        :  Temoperature of Particle
 *  \param initprtmastag  :  Initial Particle Mass Tag
 *  \param cpdinfo        :  A class which contains input data for CPD model
 *
 *  Reaction Constant for labile bridge reaction
 *  \f[ k_{b}=A_{0} \exp{\left(-\frac{E}{RT}\right)} \f]
 *  where E is the activation energy of reaction and calculated by the following equation :
 *  \f[
 *     1-\frac{l}{\ell_{0}}=\frac{1}{\sqrt{2\pi}\sigma}\intop_{-\infty}^{E}exp\left\{ -\frac{1}{2}\left(\frac{E-E_{0}}{\sigma}\right)\right\} dE
 * \f]
 */
template<typename FieldT>
class kb
  : public Expr::Expression<FieldT>
{
  DECLARE_FIELD( FieldT, temp_ )

  const CPDInformation& cpdinfo_;

  kb( const Expr::Tag temptag,
      const CPDInformation& cpdinfo );

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  \param temp the gas phase temperature at the particle surface
     *  \param laible bridge (mole)
     *  \param cpd the CPDInformation object.
     */
    Builder( const Expr::Tag kbTag,
             const Expr::Tag temptag,
             const CPDInformation& cpdinfo);
    ~Builder(){}
    Expr::ExpressionBase* build() const;

  private:
    const Expr::Tag tempt_;
    const CPDInformation& cpdinfo_;
  };

  void evaluate();
};



// ###################################################################
//
//                          Implementation
//
// ###################################################################


template<typename FieldT>
kb<FieldT>::kb( const Expr::Tag temptag,
                const CPDInformation& cpdinfo )
  : Expr::Expression<FieldT>(),
    cpdinfo_ ( cpdinfo )
{
  this->set_gpu_runnable(true);

  temp_ = this->template create_field_request<FieldT>( temptag );
}

//--------------------------------------------------------------------

template<typename FieldT>
void
kb<FieldT>::evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();

  const FieldT& temp = temp_->field_ref();

  // Arhenius parameters from:
  // Grant, D. M., Pugmire, R. J., Fletcher, T. H., & Kerstein, A. R. (1989). 
  // Chemical Model of Coal Devolatilization Using Percolation Lattice Statistics. Energy & Fuels, 3, 175–186.
  const double gascon = 1.985;
  const double Ab = 2.6e15;
  result <<= Ab * exp( -55.4E3 / (gascon * temp) );
}

//--------------------------------------------------------------------

template<typename FieldT>
kb<FieldT>::
Builder::Builder( const Expr::Tag kbTag,
                  const Expr::Tag temptag,
                  const CPDInformation& cpdinfo )
  : ExpressionBuilder(kbTag),
    tempt_      ( temptag ),
    cpdinfo_( cpdinfo )
{}

//--------------------------------------------------------------------

template<typename FieldT>
Expr::ExpressionBase*
kb<FieldT>::Builder::build() const
{
  return new kb<FieldT>( tempt_, cpdinfo_ );
}

} // namespace CPD

#endif // kb_Expr_h
