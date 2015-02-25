#ifndef VarDen_MMS_Expressions
#define VarDen_MMS_Expressions

#include <expression/Expression.h>

#include <spatialops/structured/FVStaggered.h>

#include <Core/Exceptions/ProblemSetupException.h>
#include <CCA/Components/Wasatch/VardenParameters.h>
#include <limits>
#include <iostream>
#include <fstream>
#include <algorithm>

#include <ctime>            // std::time

//--------------------------------------------------------------------

/**
 *  \class VarDen1DMMSMixFracSrc
 *  \author Amir Biglari
 *  \date November, 2012
 *  \brief A source term for a method of manufactured solution applied on pressure projection method
 *
 *  In this study the manufactured solution for velocity is defined as:
 *
 *  \f$ u = -\frac{5t}{t^2+1} \sin{\frac{2\pi x}{3t+30}} \f$
 *
 *  The manufactured solution for mixture fraction is defined as:
 *
 *  \f$ f = \frac{5}{2t+5} \exp{-\frac{5x^2}{10+t}} \f$
 *
 *  And the density functionality with respect to mixture fraction is based on non-reacting mixing model:
 *
 *  \f$ \frac{1}{\rho}=\frac{f}{\rho_1}+\frac{1-f}{\rho_0} \f$
 *
 */
template< typename FieldT >
class VarDen1DMMSMixFracSrc : public Expr::Expression<FieldT>
{
  
public:
  struct Builder : public Expr::ExpressionBuilder
  {
    /**
     * @param result Tag of the resulting expression.
     * @param xTag   Tag of the first coordinate.
     * @param tTag   Tag of time.
     * @param D      a double containing the diffusivity constant.
     * @param rho0   a double holding the density value at f=0.
     * @param rho1   a double holding the density value at f=1.
     */
    Builder( const Expr::Tag& result,
            const Expr::Tag& xTag,
            const Expr::Tag& tTag,
            const double D,
            const double rho0,
            const double rho1 );
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  private:
    const double d_, rho0_, rho1_;
    const Expr::Tag xTag_, tTag_;
  };
  
  void evaluate();
  
private:
  typedef SpatialOps::SingleValueField TimeField;
  
  VarDen1DMMSMixFracSrc( const Expr::Tag& xTag,
                       const Expr::Tag& tTag,
                       const double D,
                       const double rho0,
                       const double rho1 );
  const double d_, rho0_, rho1_;
  DECLARE_FIELD(FieldT, x_);
  DECLARE_FIELD(TimeField, t_);
};

/**
 *  \class VarDen1DMMSContinuitySrc
 *  \author Amir Biglari
 *  \date March, 2013
 *  \brief The source term for continuity equation in the MMS applied on pressure projection method.
 *         This is forced on the continuity equation by the manufactured solutions.
 *
 ** Note that rho0 and rho1 are being passed to this class from the input file. They need to be consistant with their values in the table being used by the MMS test case. Right now the defualt one is usinga specific NonReacting mixing model table of H2 (fuel) and O2 (oxidizer) at 300K
 *
 */
template< typename FieldT >
class VarDen1DMMSContinuitySrc : public Expr::Expression<FieldT>
{
  
public:
  struct Builder : public Expr::ExpressionBuilder
  {
    /**
     * @param result      Tag of the resulting expression.
     * @param xTag        Tag of the first coordinate.
     * @param tTag        Tag of time.
     * @param timestepTag Tag of time step.
     */
    Builder( const Expr::Tag& result,
             const double rho0,
             const double rho1,
             const Expr::Tag densTag,
             const Expr::Tag densStarTag,
             const Expr::Tag dens2StarTag,
             const Expr::TagList& velTags,
             const Expr::Tag& xTag,
             const Expr::Tag& tTag,
             const Expr::Tag& timestepTag,
             const Wasatch::VarDenParameters varDenParams );
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  private:
    const double rho0_, rho1_;
    const Expr::TagList velTs_;
    const Expr::Tag densTag_, densStarTag_, dens2StarTag_, xTag_, tTag_, timestepTag_;
    const Wasatch::VarDenParameters varDenParams_;
  };
  
  void bind_operators( const SpatialOps::OperatorDatabase& opDB);
  void evaluate();
  
private:
  typedef typename SpatialOps::SingleValueField TimeField;
  
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::GradientX, SVolField, SVolField >::type GradXT;
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::GradientY, SVolField, SVolField >::type GradYT;
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::GradientZ, SVolField, SVolField >::type GradZT;
  
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, XVolField, SVolField >::type X2SInterpOpT;
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, YVolField, SVolField >::type Y2SInterpOpT;
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, ZVolField, SVolField >::type Z2SInterpOpT;
  
  VarDen1DMMSContinuitySrc( const double rho0,
                            const double rho1,
                            const Expr::Tag densTag,
                            const Expr::Tag densStarTag,
                            const Expr::Tag dens2StarTag,
                            const Expr::TagList& velTags,
                            const Expr::Tag& xTag,
                            const Expr::Tag& tTag,
                            const Expr::Tag& timestepTag,
                            const Wasatch::VarDenParameters varDenParams);
  const double rho0_, rho1_;
  const bool doX_, doY_, doZ_, is3d_;
  const double a0_;
  const Wasatch::VarDenParameters::VariableDensityModels model_;
  const bool useOnePredictor_;
  
  DECLARE_FIELD(FieldT, x_);
  DECLARE_FIELD(XVolField, u_);
  DECLARE_FIELD(YVolField, v_);
  DECLARE_FIELD(ZVolField, w_);
  DECLARE_FIELDS(SVolField, dens_, densStar_, dens2Star_);
  DECLARE_FIELDS(TimeField, t_, dt_);
  
  const GradXT* gradXOp_;
  const GradYT* gradYOp_;
  const GradZT* gradZOp_;
  const X2SInterpOpT* x2SInterpOp_;
  const Y2SInterpOpT* y2SInterpOp_;
  const Z2SInterpOpT* z2SInterpOp_;
};



/**
 *  \class VarDen1DMMSPressureContSrc
 *  \author Amir Biglari
 *  \date October, 2013
 *  \brief The source term for pressure source equation in the MMS applied on pressure projection method.
 *         This is forced on the pressure source equation by the continuity source term caused by the manufactured solutions.
 *
 ** Note
 *
 */
template< typename FieldT >
class VarDen1DMMSPressureContSrc : public Expr::Expression<FieldT>
{
  
public:
  struct Builder : public Expr::ExpressionBuilder
  {
    /**
     * @param result          Tag of the resulting expression.
     * @param continutySrcTag Tag of the continuty source term.
     * @param timestepTag     Tag of time step.
     */
    Builder( const Expr::Tag& result,
            const Expr::Tag continutySrcTag,
            const Expr::Tag& timestepTag );
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  private:
    const Expr::Tag continutySrcTag_, timestepTag_;
  };
  
  void evaluate();
  
private:
  
  VarDen1DMMSPressureContSrc( const Expr::Tag continutySrcTag,
                            const Expr::Tag& timestepTag);

  typedef typename SpatialOps::SingleValueField TimeField;
  DECLARE_FIELD(FieldT, continutySrc_);
  DECLARE_FIELD(TimeField, dt_);
};

#endif /* defined(__uintah_xcode_local__VardenMMS__) */
