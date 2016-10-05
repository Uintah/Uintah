#ifndef RadPropsEvaluator_Expr_h
#define RadPropsEvaluator_Expr_h

#include <expression/Expression.h>

#include <radprops/AbsCoeffGas.h>
#include <complex>

typedef std::map<RadProps::RadiativeSpecies,Expr::Tag> RadSpecMap;

/**
 *  \class RadPropsEvaluator
 *  \author James C. Sutherland
 *  \brief Provides evaluation of radiative properties
 */
template< typename FieldT >
class RadPropsEvaluator
 : public Expr::Expression<FieldT>
{

  RadProps::GreyGas* greyGas_;
  
  DECLARE_FIELD(FieldT, temp_)
  DECLARE_VECTOR_OF_FIELDS(FieldT, indepVars_)

  RadPropsEvaluator( const Expr::Tag& tempTag,
                     const RadSpecMap& species,
                     const std::string& fileName );

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  @brief Build a RadPropsEvaluator expression
     *  @param resultTag the tag for the value that this expression computes
     *  @param tempTag the tag for the temperature
     *  @param species
     *  @param fileName
     */
    Builder( const Expr::Tag& resultTag,
             const Expr::Tag& tempTag,
             const RadSpecMap& species,
             const std::string& fileName  );

    Expr::ExpressionBase* build() const;

  private:
    const RadSpecMap rsm_;
    const Expr::Tag tempTag_;
    const std::string fileName_;
  };

  ~RadPropsEvaluator();

  void evaluate();
};


enum ParticleRadProp{
  PLANCK_SCATTERING_COEFF,
  PLANCK_ABSORPTION_COEFF,
  ROSSELAND_SCATTERING_COEFF,
  ROSSELAND_ABSORPTION_COEFF,
};

namespace RadProps{ class ParticleRadCoeffs; } // forward declaration

/**
 *  \class ParticleRadProps
 */
template< typename FieldT >
class ParticleRadProps
 : public Expr::Expression<FieldT>
{
  const RadProps::ParticleRadCoeffs* const props_;
  const ParticleRadProp prop_;

  DECLARE_FIELDS(FieldT, temp_, pRadius_)
  
  ParticleRadProps( const ParticleRadProp prop,
                    const Expr::Tag& tempTag,
                    const Expr::Tag& pRadiusTag,
                    const std::complex<double>& refIndex );
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  @brief Build a ParticleRadProps expression
     *  @param resultTag the tag for the value that this expression computes
     *  @param tempTag the tag for the temperature
     *  @param pRadiusTag the tag for the particle radius
     *  @param refIndex the refractive index for the particles
     */
    Builder( const ParticleRadProp prop,
             const Expr::Tag& resultTag,
             const Expr::Tag& tempTag,
             const Expr::Tag& pRadiusTag,
             const std::complex<double> refIndex );

    Expr::ExpressionBase* build() const;

  private:
    const ParticleRadProp prop_;
    const Expr::Tag tempTag_, pRadiusTag_;
    const std::complex<double> refIndex_;
  };

  ~ParticleRadProps();
  void evaluate();
};


#endif // RadPropsEvaluator_Expr_h
