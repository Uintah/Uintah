/*
 * The MIT License
 *
 * Copyright (c) 2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef HomogeneousNucleationCoefficient_Expr_h
#define HomogeneousNucleationCoefficient_Expr_h
#include <spatialops/structured/FVStaggeredFieldTypes.h>
#include <spatialops/structured/FVStaggeredOperatorTypes.h>

#include <expression/Expression.h>

#ifndef PI
#define PI 3.1415926535897932384626433832795
#endif

#ifndef NA
#define NA 6.02214129e23
#endif

#ifndef KB
#define KB 1.3806488e-23
#endif

/**
 *  \ingroup WasatchExpressions
 *  \class HomogeneousNucleationCoefficient
 *  \author Alex Abboud
 *  \date February 2013
 *
 *  \tparam FieldT the type of field.
 *
 *  \brief Nucleation Coeffcient Source term for use in QMOM
 *  the modified homogeneous nucleation refers to this value as
 *  \f$ B_0 = z * k_f * N_1^2 * \exp ( -\Delta G / K_B T )  \f$ with
 *  \f$ \Delta G = \frac{16 \pi}{3} \frac{nu^2 \sigma^3}{K_B^2 T^2 \ln( S)^2 )} \f$
 *  \f$ \k_f = D*(48 \pi^2 \nu i_c)^{1/3} \f$
 *  \f$ z = \sqrt{ \frac{\Delta G}{3 \pi K_B T i_c^2} \f$
 *  \f$ N_1 = N_A c_{eq} S \f$
 *  \f$ i_c = \frac{32 \pi}{3} \frac{\nu^2 \sigma^3}{K_B^3 T^3 \ln(S)^3} \f$
 */
template< typename FieldT >
class HomogeneousNucleationCoefficient
: public Expr::Expression<FieldT>
{
  const Expr::Tag superSatTag_, eqConcTag_; //Tag for supersturation and equilibrium concentration
  const Expr::Tag surfaceEngTag_; //Tag for variable surface energy
  const FieldT* superSat_; 
  const FieldT* eqConc_;
  const FieldT* surfaceEng_;      //field for variable surface energy
  const double molecularVolume_;
  const double surfaceEnergy_;    //value if constant surface energy
  const double temperature_; 
  const double diffusionCoef_; //diffusion coefficient
  
  HomogeneousNucleationCoefficient( const Expr::Tag& superSatTag,
                                    const Expr::Tag& eqConcTag,
                                    const Expr::Tag& surfaceEngTag,
                                    const double molecularVolume,
                                    const double surfaceEnergy,
                                    const double temperature,
                                    const double diffusionCoef);
  
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& result,
             const Expr::Tag& superSatTag,
             const Expr::Tag& eqConcTag,
             const Expr::Tag& surfaceEngTag,
             const double molecularVolume,
             const double surfaceEnergy,
             const double temperature,
             const double diffusionCoef)
    : ExpressionBuilder(result),
    supersatt_(superSatTag),
    eqconct_(eqConcTag),
    surfaceengt_(surfaceEngTag),
    molecularvolume_(molecularVolume),
    surfaceenergy_(surfaceEnergy),
    temperature_(temperature),
    diffusioncoef_(diffusionCoef)
    {}
    
    ~Builder(){}
    
    Expr::ExpressionBase* build() const
    {
      return new HomogeneousNucleationCoefficient<FieldT>( supersatt_, eqconct_, surfaceengt_, molecularvolume_, surfaceenergy_, temperature_, diffusioncoef_);
    }
    
  private:
    const Expr::Tag supersatt_, eqconct_, surfaceengt_;
    const double molecularvolume_;
    const double surfaceenergy_;
    const double temperature_;
    const double diffusioncoef_;
  };
  
  ~HomogeneousNucleationCoefficient();
  
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
  
};

// ###################################################################
//
//                          Implementation
//
// ###################################################################



template< typename FieldT >
HomogeneousNucleationCoefficient<FieldT>::
HomogeneousNucleationCoefficient( const Expr::Tag& superSatTag,
                                  const Expr::Tag& eqConcTag,
                                  const Expr::Tag& surfaceEngTag,
                                  const double molecularVolume,
                                  const double surfaceEnergy,
                                  const double temperature,
                                  const double diffusionCoef)
: Expr::Expression<FieldT>(),
superSatTag_(superSatTag),
eqConcTag_(eqConcTag),
surfaceEngTag_(surfaceEngTag),
molecularVolume_(molecularVolume),
surfaceEnergy_(surfaceEnergy),
temperature_(temperature),
diffusionCoef_(diffusionCoef)
{}

//--------------------------------------------------------------------

template< typename FieldT >
HomogeneousNucleationCoefficient<FieldT>::
~HomogeneousNucleationCoefficient()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
HomogeneousNucleationCoefficient<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( superSatTag_ );
  exprDeps.requires_expression( eqConcTag_);
  if ( surfaceEngTag_ != Expr::Tag() )
    exprDeps.requires_expression( surfaceEngTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
HomogeneousNucleationCoefficient<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  superSat_ = &fml.template field_manager<FieldT>().field_ref( superSatTag_ );
  eqConc_ = &fml.template field_manager<FieldT>().field_ref( eqConcTag_ );
  if ( surfaceEngTag_ != Expr::Tag() )
    surfaceEng_ = &fml.template field_manager<FieldT>().field_ref( surfaceEngTag_ );
}

//--------------------------------------------------------------------
template< typename FieldT >
void
HomogeneousNucleationCoefficient<FieldT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
HomogeneousNucleationCoefficient<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  result <<= 0.0;
  
  //temporary fields to set before calculating coefficient
  SpatFldPtr<FieldT> delG = SpatialFieldStore::get<FieldT>( result );
  SpatFldPtr<FieldT> iC = SpatialFieldStore::get<FieldT>( result );
  SpatFldPtr<FieldT> z = SpatialFieldStore::get<FieldT>( result );
  SpatFldPtr<FieldT> kF = SpatialFieldStore::get<FieldT>( result );
  SpatFldPtr<FieldT> N1 = SpatialFieldStore::get<FieldT>( result );
  
  if ( surfaceEngTag_ != Expr::Tag() ) {
    *delG <<= 16.0*PI/3.0* molecularVolume_ * molecularVolume_ * *surfaceEng_ * *surfaceEng_ * *surfaceEng_ / (KB * KB * temperature_ * temperature_ * log(*superSat_) * log(*superSat_) )+
              KB*temperature_*log(*superSat_) - *surfaceEng_ * pow(36.0*PI*molecularVolume_*molecularVolume_, 1.0/3.0);
    
    *iC <<= 32.0*PI/3.0* molecularVolume_ * molecularVolume_ * *surfaceEng_ * *surfaceEng_ * *surfaceEng_ / 
            (KB*KB*KB *temperature_*temperature_*temperature_ *log(*superSat_) *log(*superSat_)* log(*superSat_) );
  } else {
    *delG <<= 16.0*PI/3.0* molecularVolume_ * molecularVolume_ * surfaceEnergy_ * surfaceEnergy_ * surfaceEnergy_ / (KB * KB * temperature_ * temperature_ * log(*superSat_) * log(*superSat_) )+
              KB*temperature_*log(*superSat_) - surfaceEnergy_ * pow(36.0*PI*molecularVolume_*molecularVolume_, 1.0/3.0);
  
    *iC <<= 32.0*PI/3.0* molecularVolume_ * molecularVolume_ * surfaceEnergy_ * surfaceEnergy_ * surfaceEnergy_ / 
            (KB*KB*KB *temperature_*temperature_*temperature_ *log(*superSat_) *log(*superSat_)* log(*superSat_) );
  }
  *z <<= sqrt( *delG/ (3.0*PI*KB*temperature_* *iC * *iC) );
  
  *kF <<= diffusionCoef_ * pow(48.0*PI*PI*molecularVolume_ * *iC, 1.0/3.0);
  *N1 <<= NA * *eqConc_ * *superSat_;

  result <<= cond( *superSat_ > 1.0, *z * *kF * *N1 * *N1 * exp( - *delG/ KB /temperature_ ) )
                 ( 0.0 );
}

//--------------------------------------------------------------------

#endif // HomogeneousNucleationCoefficient_Expr_h
