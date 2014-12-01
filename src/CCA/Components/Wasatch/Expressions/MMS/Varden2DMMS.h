#ifndef VarDen_2D_MMS_Expressions
#define VarDen_2D_MMS_Expressions

#include <expression/Expression.h>
#include <Core/Exceptions/ProblemSetupException.h>

#include <spatialops/structured/FVStaggeredFieldTypes.h>
#include <spatialops/structured/stencil/FVStaggeredOperatorTypes.h>
#include <CCA/Components/Wasatch/VardenParameters.h>
#include <limits>
#include <iostream>
#include <fstream>
#include <algorithm>

#include <ctime>

//**********************************************************************
// OSCILLATING MMS: MIXTURE FRACTION SOURCE TERM
//**********************************************************************

/**
 *  \class VarDenMMSOscillatingMixFracSrc
 *  \author Amir Biglari, Tony Saad
 *  \date December, 2013
 *  \brief A source term for a method of manufactured solution applied on pressure projection method
 *
 *  In this study the manufactured solution for x-velocity is defined as:
 *
 *  \f$ u = -\frac{\omega}{4k} \frac{\rho_1 - \rho_0}{\rho} \cos{\pi k \hat{x}}\sin{\pi k \hat{y}}\sin{\pi \omega t} \f$
 *
 *  The manufactured solution for y-velocity is defined as:
 *
 *  \f$ y = -\frac{\omega}{4k} \frac{\rho_1 - \rho_0}{\rho} \sin{\pi k \hat{x}} \cos{\pi k \hat{y}} \sin{\pi \omega t} \f$
 *
 *  The manufactured solution for mixture fraction is defined as:
 *
 *  \f$ f = \frac{1 + \sin{\pi k \hat{x}} \sin{\pi k \hat{y}} \cos{\pi \omega t}} 
 *               { (1+\frac{\rho_0}{\rho_1}) + (1-\frac{\rho_0}{\rho_1}) \sin{\pi k \hat{x}}\sin{\pi k \hat{y}} \cos{\pi \omega t} } \f$
 *
 *  And the density functionality with respect to mixture fraction is based on non-reacting mixing model:
 *
 *  \f$ \frac{1}{\rho}=\frac{f}{\rho_1}+\frac{1-f}{\rho_0} \f$
 *
 *  Where \f$ \hat{x}= u_F t - x \f$ and  \f$ \hat{y}= v_F t - y \f$
 *
 */
template< typename FieldT >
class VarDenMMSOscillatingMixFracSrc : public Expr::Expression<FieldT>
{
  
public:
  struct Builder : public Expr::ExpressionBuilder
  {
    /**
     * @param result Tag of the resulting expression.
     * @param xTag   Tag of the first coordinate.
     * @param yTag   Tag of the second coordinate.
     * @param tTag   Tag of time.
     * @param r0   a double holding the density value at f=0.
     * @param r1   a double holding the density value at f=1.
     * @param d
     * @param w
     * @param k
     * @param uf
     * @param vf
     */
    Builder( const Expr::Tag& result,
             const Expr::Tag& xTag,
             const Expr::Tag& yTag,
             const Expr::Tag& tTag,
             const double r0,
             const double r1,
             const double d,
             const double w,
             const double k,
             const double uf,
             const double vf);
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  private:
    const double r0_, r1_, d_, w_, k_, uf_, vf_;
    const Expr::Tag xTag_, yTag_, tTag_;
  };
  
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void evaluate();
  
private:
  typedef SpatialOps::SingleValueField TimeField;
  
  VarDenMMSOscillatingMixFracSrc( const Expr::Tag& xTag,
                                  const Expr::Tag& yTag,
                                  const Expr::Tag& tTag,
                                 const double r0,
                                 const double r1,
                                 const double d,
                                 const double w,
                                 const double k,
                                 const double uf,
                                 const double vf);
  const double r0_, r1_, d_, w_, k_, uf_, vf_;
  const Expr::Tag xTag_, yTag_, tTag_;
  const FieldT *x_, *y_;
  const TimeField* t_;
};

//**********************************************************************
// OSCILLATING MMS: CONTINUITY SOURCE TERM
//**********************************************************************

/**
 *  \class VarDenMMSOscillatingContinuitySrc
 *  \author Tony Saad, Amir Biglari
 *  \date December, 2013
 *  \brief The source term for continuity equation in the 2D-MMS applied on pressure projection method.
 *         This is forced on the continuity equation by the manufactured solutions.
 *
 ** Note that r0 and r1 are being passed to this class from the input file. They need to be consistant with their values in the table being used by the MMS test case. Right now the defualt one is using a specific NonReacting mixing model table of H2 (fuel) and O2 (oxidizer) at 300K
 *
 */
template< typename FieldT >
class VarDenMMSOscillatingContinuitySrc : public Expr::Expression<FieldT>
{
  
public:
  struct Builder : public Expr::ExpressionBuilder
  {
    Builder( const Expr::Tag& result,
             const Expr::Tag densTag,
             const Expr::Tag densStarTag,
             const Expr::Tag dens2StarTag,
             const Expr::TagList& velTags,
             const Expr::TagList& velStarTags,
             const double r0,
             const double r1,
             const double w,
             const double k,
             const double uf,
             const double vf,
             const Expr::Tag& xTag,
             const Expr::Tag& yTag,
             const Expr::Tag& tTag,
             const Expr::Tag& timestepTag,
             const Wasatch::VarDenParameters varDenParams);
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  private:
    const double r0_, r1_, w_, k_, uf_, vf_;
    const Expr::Tag xTag_, yTag_, tTag_, timestepTag_;
    const Expr::Tag denst_, densStart_, dens2Start_;
    const Expr::TagList velTs_,velStarTs_;
    const Wasatch::VarDenParameters varDenParams_;
  };
  
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB);
  void evaluate();
  
private:
  typedef typename SpatialOps::SingleValueField TimeField;
  
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, SVolField, XVolField >::type S2XInterpOpT;
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, SVolField, YVolField >::type S2YInterpOpT;
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, SVolField, ZVolField >::type S2ZInterpOpT;
  
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, XVolField, SVolField >::type X2SInterpOpT;
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, YVolField, SVolField >::type Y2SInterpOpT;
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, ZVolField, SVolField >::type Z2SInterpOpT;
  
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Gradient, XVolField, SVolField >::type GradXT;
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Gradient, YVolField, SVolField >::type GradYT;
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Gradient, ZVolField, SVolField >::type GradZT;

  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::GradientX, SVolField, SVolField >::type GradXST;
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::GradientY, SVolField, SVolField >::type GradYST;
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::GradientZ, SVolField, SVolField >::type GradZST;
  
  VarDenMMSOscillatingContinuitySrc( const Expr::Tag densTag,
                                     const Expr::Tag densStarTag,
                                     const Expr::Tag dens2StarTag,
                                     const Expr::TagList& velTags,
                                     const Expr::TagList& velStarTags,
                                     const double r0,
                                     const double r1,
                                     const double w,
                                     const double k,
                                     const double uf,
                                     const double vf,
                                     const Expr::Tag& xTag,
                                     const Expr::Tag& yTag,
                                     const Expr::Tag& tTag,
                                     const Expr::Tag& timestepTag,
                                     const Wasatch::VarDenParameters varDenParams );
  const Expr::Tag xVelt_, yVelt_, zVelt_, xVelStart_, yVelStart_, zVelStart_, denst_, densStart_, dens2Start_;
  const bool doX_, doY_, doZ_, is3d_;
  const double r0_, r1_, w_, k_, uf_, vf_;
  const Expr::Tag xTag_, yTag_, tTag_, timestepTag_;
  const double a0_;
  const Wasatch::VarDenParameters::VariableDensityModels model_;
  const bool useOnePredictor_;
  const XVolField *uStar_, *xVel_;
  const YVolField *vStar_, *yVel_;
  const ZVolField *wStar_, *zVel_;
  const SVolField *dens_, *densStar_, *dens2Star_;
  const FieldT *x_, *y_;
  const TimeField *t_, *timestep_;
  const GradXT* gradXOp_;
  const GradYT* gradYOp_;
  const GradZT* gradZOp_;
  const GradXST* gradXSOp_;
  const GradYST* gradYSOp_;
  const GradZST* gradZSOp_;
  const S2XInterpOpT* s2XInterpOp_;
  const S2YInterpOpT* s2YInterpOp_;
  const S2ZInterpOpT* s2ZInterpOp_;
  const X2SInterpOpT* x2SInterpOp_;
  const Y2SInterpOpT* y2SInterpOp_;
  const Z2SInterpOpT* z2SInterpOp_;
};

//**********************************************************************
// OSCILLATING MMS: X VELOCITY
//**********************************************************************

/**
 *  \class VarDenOscillatingMMSxVel
 *  \author Amir Biglari
 *  \date December, 2013
 *  \brief Implements Shunn's 2D mms momentum field in x direction
 *
 *  Shunn's 2D mms velocity field in x direction is given as
 *  \f[
 *    \rho u(x,y,t)= -\frac{\omega}{4k} (\rho_1 - \rho_0) \cos{\pi k \hat{x}}\sin{\pi k \hat{y}}\sin{\pi \omega t}
 *  \f]
 *  where
 *  \f$ \hat{x}= u_F t - x \f$ and  \f$ \hat{y}= v_F t - y \f$
 */
template< typename FieldT >
class VarDenOscillatingMMSxVel : public Expr::Expression<FieldT>
{
  typedef typename SpatialOps::SingleValueField TimeField;

  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, SVolField, FieldT >::type S2FInterpOpT;
  const S2FInterpOpT* s2FInterpOp_;
public:
  
  /**
   *  \brief Builds a Shunn's 2D mms velocity function in x direction Expression.
   */
  struct Builder : public Expr::ExpressionBuilder
  {
    Builder( const Expr::Tag& result,
             const Expr::Tag& rhoTag,
             const Expr::Tag& xTag,
             const Expr::Tag& yTag,
             const Expr::Tag& tTag,
             const double r0,
             const double r1,
             const double w,
             const double k,
             const double uf,
             const double vf );
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  private:
    const double r0_, r1_, w_, k_, uf_, vf_;
    const Expr::Tag xTag_, yTag_, tTag_, rhoTag_;
  };
  
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
  
private:
  
  VarDenOscillatingMMSxVel( const Expr::Tag& rhoTag,
                            const Expr::Tag& xTag,
                            const Expr::Tag& yTag,
                            const Expr::Tag& tTag,
                            const double r0,
                            const double r1,
                            const double w,
                            const double k,
                            const double uf,
                            const double vf );
  const double r0_, r1_, w_, k_, uf_, vf_;
  const Expr::Tag xTag_, yTag_, tTag_, rhoTag_;
  const FieldT* x_;
  const FieldT* y_;
  const SVolField* rho_;
  const TimeField* t_;
};

//**********************************************************************
// OSCILLATING MMS: Y VELOCITY
//**********************************************************************

/**
 *  \class VarDenOscillatingMMSyVel
 *  \author Amir Biglari
 *  \date December, 2013
 *  \brief Implements Shunn's 2D mms momentum field in y direction
 *
 *  Shunn's 2D mms velocity field in y direction is given as
 *  \f[
 *    \rho v(x,y,t)= -\frac{\omega}{4k} (\rho_1 - \rho_0) \sin{\pi k \hat{x}} \cos{\pi k \hat{y}} \sin{\pi \omega t}
 *  \f]
 *  where
 *  \f$ \hat{x}= u_F t - x \f$ and  \f$ \hat{y}= v_F t - y \f$
 */
template< typename FieldT >
class VarDenOscillatingMMSyVel : public Expr::Expression<FieldT>
{
  typedef typename SpatialOps::SingleValueField TimeField;

  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, SVolField, FieldT >::type S2FInterpOpT;
  const S2FInterpOpT* s2FInterpOp_;
public:
  
  /**
   *  \brief Builds a Shunn's 2D mms Velocity Function in y direction Expression.
   */
  struct Builder : public Expr::ExpressionBuilder
  {
    Builder( const Expr::Tag& result,
             const Expr::Tag& rhoTag,
             const Expr::Tag& xTag,
             const Expr::Tag& yTag,
             const Expr::Tag& tTag,
              const double r0,
              const double r1,
              const double w,
              const double k,
              const double uf,
              const double vf );
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  private:
  const double r0_, r1_, w_, k_, uf_, vf_;
    const Expr::Tag xTag_, yTag_, tTag_, rhoTag_;
  };
  
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
  
private:
  
  VarDenOscillatingMMSyVel( const Expr::Tag& rhoTag,
        const Expr::Tag& xTag,
        const Expr::Tag& yTag,
        const Expr::Tag& tTag,
        const double r0,
        const double r1,
        const double w,
        const double k,
        const double uf,
        const double vf );
  const double r0_, r1_, w_, k_, uf_, vf_;
  const Expr::Tag xTag_, yTag_, tTag_, rhoTag_;
  const FieldT* x_;
  const FieldT* y_;
  const SVolField* rho_;
  const TimeField* t_;
};

//**********************************************************************
// OSCILLATING MMS: MIXTURE FRACTION (INITIALIZATION, ETC...)
//**********************************************************************

/**
 *  \class VarDenOscillatingMMSMixFrac
 *  \author Amir Biglari
 *  \date December, 2013
 *  \brief Implements Shunn's 2D mms scalar field (mixture fraction)
 *
 *  Shunn's 2D mms scalar field (mixture fraction) is given as
 *  \f[
 *    f(x,y,t)= \frac{1 + \sin{\pi k \hat{x}} \sin{\pi k \hat{y}} \cos{\pi \omega t}}
 *               { (1+\frac{\rho_0}{\rho_1}) + (1-\frac{\rho_0}{\rho_1}) \sin{\pi k \hat{x}}\sin{\pi k \hat{y}} \cos{\pi \omega t} }
 *  \f]
 *  where
 *  \f$ \hat{x}= u_F t - x \f$ and  \f$ \hat{y}= v_F t - y \f$
 */
template< typename FieldT >
class VarDenOscillatingMMSMixFrac : public Expr::Expression<FieldT>
{
  typedef typename SpatialOps::SingleValueField TimeField;
public:
  
  /**
   *  \brief Builds an Expression for Shunn's 2D mms scalar field (mixture fraction).
   */
  struct Builder : public Expr::ExpressionBuilder
  {
    Builder( const Expr::Tag& result,
             const Expr::Tag& xTag,
             const Expr::Tag& yTag,
             const Expr::Tag& tTag,
             const double r0,
             const double r1,
             const double w,
             const double k,
             const double uf,
             const double vf);
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  private:
    const double r0_, r1_, w_, k_, uf_, vf_;
    const Expr::Tag xTag_, yTag_, tTag_;
  };
  
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void evaluate();
  
private:
  
  VarDenOscillatingMMSMixFrac( const Expr::Tag& xTag,
                   const Expr::Tag& yTag,
                   const Expr::Tag& tTag,
                   const double r0,
                   const double r1,
                   const double w,
                   const double k,
                   const double uf,
                   const double vf );
  const double r0_, r1_, w_, k_, uf_, vf_;
  const Expr::Tag xTag_, yTag_, tTag_;
  const FieldT* x_;
  const FieldT* y_;
  const TimeField* t_;
};

//**********************************************************************
// ALL MMS: DIFFUSION COEFFICENT: rho * D = constant -> D = constant/rho
//**********************************************************************

/**
 *  \class DiffusiveConstant
 *  \author Amir Biglari
 *  \date December, 2013
 *  \brief Implements Shunn's 2D mms diffusion coefficient field
 *
 *  Shunn's 2D mms diffusion coefficient field is given as
 *  \f[
 *    D(x,y,t)= 0.001 / \rho  \f]
 *  
 */
template< typename FieldT >
class DiffusiveConstant : public Expr::Expression<FieldT>
{
public:
  
  /**
   *  \brief Builds an Expression for Shunn's 2D mms diffusion coefficient field.
   */
  struct Builder : public Expr::ExpressionBuilder
  {
    Builder( const Expr::Tag& result,
             const Expr::Tag& rhoTag,
             const double d);
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  private:
    const Expr::Tag rhoTag_;
    const double d_;
  };
  
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void evaluate();
  
private:
  
  DiffusiveConstant( const Expr::Tag& rhoTag,
                     const double d);
  const Expr::Tag rhoTag_;
  const double d_;
  const FieldT* rho_;
};

#endif //VarDen_2D_MMS_Expressions
