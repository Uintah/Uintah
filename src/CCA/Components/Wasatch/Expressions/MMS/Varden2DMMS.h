#ifndef VarDen_2D_MMS_Expressions
#define VarDen_2D_MMS_Expressions

#include <expression/Expression.h>
#include <Core/Exceptions/ProblemSetupException.h>

#include <spatialops/structured/FVStaggeredFieldTypes.h>
#include <spatialops/structured/stencil/FVStaggeredOperatorTypes.h>
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
             const double vf,
             const bool atNP1);
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  private:
    const double r0_, r1_, d_, w_, k_, uf_, vf_;
    const Expr::Tag xTag_, yTag_, tTag_;
    const bool atNP1_;
  };
  
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
                                 const double vf,
                                 const bool atNP1);
  const double r0_, r1_, d_, w_, k_, uf_, vf_;
  const bool atNP1_;
  DECLARE_FIELDS(FieldT, x_, y_)
  DECLARE_FIELDS (TimeField, t_, dt_)
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
  DECLARE_FIELDS(FieldT, x_, y_)
  DECLARE_FIELD (TimeField, t_)
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
  void evaluate();
  
private:
  
  DiffusiveConstant( const Expr::Tag& rhoTag,
                     const double d);
  const double d_;
  DECLARE_FIELD(FieldT, rho_)
};

#endif //VarDen_2D_MMS_Expressions
