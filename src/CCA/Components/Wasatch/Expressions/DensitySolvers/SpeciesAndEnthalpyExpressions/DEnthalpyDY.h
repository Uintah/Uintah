#ifndef DEnthalpyDy_h
#define DEnthalpyDy_h

#include <expression/Expression.h>
#include <pokitt/CanteraObjects.h>

/**
 *  \class DEnthalpyDY
 *  \author Josh McConnell
 *  \date   March 2017
 *
 *  \brief Computes the vector given by \f$\frac{\partial \h}{\partial Y}\f$
 *  where \f$Y\f$ is the vector of the first \f$n-1\f$ species mass fractions
 *  in the system. Each element of \f$\frac{\partial \h}{\partial Y}\f$ is
 *  given by
 *  \f[
 *   \frac{\partial \h}{\partial Y_i} = M_i - M_n,
 *  \f]
 *  where \f$\M\f$ is the mixture molecular weight and \f$\M_i\f$ is the
 *  molecular weight of species \f$\i\f$.
 */

namespace WasatchCore{

template<typename FieldT>
class DEnthalpyDY : public Expr::Expression<FieldT>
{
  DECLARE_VECTOR_OF_FIELDS(FieldT, hi_ )


  const double nSpec_;

  DEnthalpyDY( const Expr::TagList& hiTags_ );

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  @brief Build a DEnthalpyDY expression
     *  @param resultTags the tag for the values that this expression computes
     *  @param hiTags the tag for pure species enthalpies
     */
    Builder( const Expr::TagList& resultTags,
             const Expr::TagList& hiTags_ );

    Expr::ExpressionBase* build() const{
      return new DEnthalpyDY( hiTags_ );
    }

  private:
    const Expr::TagList hiTags_;
  };

  void evaluate();

};


// ###################################################################
//
//                          Implementation
//
// ###################################################################

template<typename FieldT>
DEnthalpyDY<FieldT>::
DEnthalpyDY( const Expr::TagList& hiTags )
  : Expr::Expression<FieldT>(),
    nSpec_( CanteraObjects::number_species()    )
{
  this->set_gpu_runnable(true);

  this->template create_field_vector_request<FieldT>( hiTags, hi_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
DEnthalpyDY<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  typename Expr::Expression<FieldT>::ValVec&  resultVec = this->get_value_vec();

  const FieldT& hn = hi_[nSpec_-1]->field_ref();

  for(int i=0; i<nSpec_-1; ++i){
    const FieldT& hi = hi_[i]->field_ref();
    *resultVec[i] <<= hi - hn;
  }
}

//--------------------------------------------------------------------

template<typename FieldT>
DEnthalpyDY<FieldT>::
Builder::Builder( const Expr::TagList& resultTags,
                  const Expr::TagList& hiTags )
  : ExpressionBuilder( resultTags ),
    hiTags_ ( hiTags )
{
  assert(resultTags.size() == CanteraObjects::number_species()-1);
  assert(hiTags    .size() == CanteraObjects::number_species()  );
}

//====================================================================
}

#endif
