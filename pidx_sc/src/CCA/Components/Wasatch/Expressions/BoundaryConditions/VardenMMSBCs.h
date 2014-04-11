#ifndef VardenMMSBCs_h
#define VardenMMSBCs_h

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

#include <expression/Expression.h>
#include <spatialops/structured/FVStaggered.h>

#include <CCA/Components/Wasatch/Expressions/BoundaryConditions/BoundaryConditionBase.h>

/***********************************************************************************************/
/**
 *  \class    VarDen1DMMSDensity
 *  \ingroup  Expressions
 *  \author   Amir Biglari
 *  \date     December, 2012
 *
 *  \brief Provides an expression for density at the boundaries in the MMS 
 *         that is written for the pressure projection method verification
 *
 *  \tparam FieldT - the type of field for the density.
 */

template< typename FieldT >
class VarDen1DMMSDensity : public BoundaryConditionBase<FieldT>
{
  typedef typename SpatialOps::structured::SingleValueField TimeField;
  VarDen1DMMSDensity( const Expr::Tag& indepVarTag,
                     const double rho0,
                     const double rho1 )
  : indepVarTag_ (indepVarTag),
    rho0_ (rho0),
    rho1_ (rho1)
  {
    this->set_gpu_runnable(false);
  }

public:

  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  \brief Construct an expression for the density at the boundaries, given the tag for time
     *         at x = 15 and x = -15
     *
     *  \param indepVarTag the Expr::Tag for holding the time variable.
     */
    Builder( const Expr::Tag& resultTag,
             const Expr::Tag& indepVarTag,
             const double rho0,
             const double rho1) :
    ExpressionBuilder(resultTag),
    indepVarTag_ (indepVarTag),
    rho0_ (rho0),
    rho1_ (rho1)
    {}
    Expr::ExpressionBase* build() const{ return new VarDen1DMMSDensity(indepVarTag_, rho0_, rho1_); }
  private:
    const Expr::Tag indepVarTag_;
    const double rho0_, rho1_;
  };
  
  ~VarDen1DMMSDensity(){}
  void advertise_dependents( Expr::ExprDeps& exprDeps ){  exprDeps.requires_expression( indepVarTag_ );}
  void bind_fields( const Expr::FieldManagerList& fml ){
    t_    = &fml.template field_manager<TimeField>().field_ref( indepVarTag_ );
  }
  void evaluate();
private:
  const TimeField* t_;
  const Expr::Tag indepVarTag_;
  const double rho0_, rho1_;
};

/***********************************************************************************************/
/**
 *  \class    VarDen1DMMSMixtureFraction
 *  \ingroup  Expressions
 *  \author   Amir Biglari
 *  \date     December, 2012
 *
 *  \brief Provides an expression for mixture fraction at the boundaries in the MMS
 *         that is written for the pressure projection method verification
 *
 *  \tparam FieldT - the type of field for the mixture fraction.
 */
template< typename FieldT >
class VarDen1DMMSMixtureFraction
: public BoundaryConditionBase<FieldT>
{
  typedef typename SpatialOps::structured::SingleValueField TimeField;
  VarDen1DMMSMixtureFraction( const Expr::Tag& indepVarTag ) :
  indepVarTag_ (indepVarTag)
  {
    this->set_gpu_runnable(false);
  }
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  \brief Construct an expression for the mixture fraction at the boundaries, given the tag for time
     *         at x = 15 and x = -15
     *
     *  \param indepVarTag the Expr::Tag for holding the time variable.
     */
    Builder( const Expr::Tag& resultTag,
            const Expr::Tag& indepVarTag) :
    ExpressionBuilder(resultTag),
    indepVarTag_ (indepVarTag)
    {}
    Expr::ExpressionBase* build() const{ return new VarDen1DMMSMixtureFraction(indepVarTag_); }
  private:
    const Expr::Tag indepVarTag_;
  };
  
  ~VarDen1DMMSMixtureFraction(){}
  void advertise_dependents( Expr::ExprDeps& exprDeps ){  exprDeps.requires_expression( indepVarTag_ );}
  void bind_fields( const Expr::FieldManagerList& fml ){
    t_    = &fml.template field_manager<TimeField>().field_ref( indepVarTag_ );
  }
  void evaluate();
private:
  const TimeField* t_;
  const Expr::Tag indepVarTag_;
};

/***********************************************************************************************/
/**
 *  \class    VarDen1DMMSMomentum
 *  \ingroup  Expressions
 *  \author   Amir Biglari
 *  \date     December, 2012
 *
 *  \brief Provides an expression for momentum at the boundaries in the MMS
 *         that is written for the pressure projection method verification
 *
 *  \tparam FieldT - the type of field for the momentum.
 */
template< typename FieldT >
class VarDen1DMMSMomentum
: public BoundaryConditionBase<FieldT>
{
  typedef typename SpatialOps::structured::SingleValueField TimeField;
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  \brief Construct an expression for the momentum at the boundaries, given the tag for time
     *         at x = 15 and x = -15
     *
     *  \param indepVarTag the Expr::Tag for holding the time variable.
     *
     *  \param side an enum for specifying the side of the boundary, whether it is on the left or the right side
     */
    Builder( const Expr::Tag& resultTag,
            const Expr::Tag& indepVarTag,
            const double rho0,
            const double rho1,
            const SpatialOps::structured::BCSide side )
    : ExpressionBuilder(resultTag),
    indepVarTag_ (indepVarTag),
    rho0_( rho0 ),
    rho1_( rho1 ),
    side_( side )
    {}
    Expr::ExpressionBase* build() const{ return new VarDen1DMMSMomentum(indepVarTag_, rho0_, rho1_, side_); }
  private:
    const Expr::Tag indepVarTag_;
    const double rho0_, rho1_;
    const SpatialOps::structured::BCSide side_;
  };
  
  ~VarDen1DMMSMomentum(){}
  void advertise_dependents( Expr::ExprDeps& exprDeps ){ exprDeps.requires_expression( indepVarTag_ ); }
  void bind_fields( const Expr::FieldManagerList& fml ){ t_ = &fml.template field_ref<TimeField>( indepVarTag_ ); }
  void evaluate();
private:
  VarDen1DMMSMomentum( const Expr::Tag& indepVarTag,
                     const double rho0,
                     const double rho1,
                     const SpatialOps::structured::BCSide side )
  : indepVarTag_( indepVarTag ),
  rho0_( rho0 ),
  rho1_( rho1 ),
  side_( side )
  {
    this->set_gpu_runnable(false);
  }
  const TimeField* t_;
  const Expr::Tag indepVarTag_;
  const double rho0_, rho1_;
  const SpatialOps::structured::BCSide side_;
};

/***********************************************************************************************/
/**
 *  \class    VarDen1DMMSSolnVar
 *  \ingroup  Expressions
 *  \author   Amir Biglari
 *  \date     December, 2012
 *
 *  \brief Provides an expression for solution variable at the boundaries in the MMS
 *         that is written for the pressure projection method verification
 *
 *  \tparam FieldT - the type of field for the solution variable.
 */
template< typename FieldT >
class VarDen1DMMSSolnVar
: public BoundaryConditionBase<FieldT>
{
  typedef typename SpatialOps::structured::SingleValueField TimeField;
  VarDen1DMMSSolnVar( const Expr::Tag& indepVarTag,
                    const double rho0,
                    const double rho1  )
  : indepVarTag_ (indepVarTag),
  rho0_ (rho0),
  rho1_ (rho1)
  {
    this->set_gpu_runnable(false);
  }
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  \brief Construct an expression for the solution variable at the boundaries, given the tag for time
     *         at x = 15 and x = -15
     *
     *  \param indepVarTag the Expr::Tag for holding the time variable.
     */
    Builder( const Expr::Tag& resultTag,
            const Expr::Tag& indepVarTag,
            const double rho0,
            const double rho1 ) :
    ExpressionBuilder(resultTag),
    indepVarTag_ (indepVarTag),
    rho0_ (rho0),
    rho1_ (rho1)
    {}
    Expr::ExpressionBase* build() const{ return new VarDen1DMMSSolnVar(indepVarTag_, rho0_, rho1_); }
  private:
    const Expr::Tag indepVarTag_;
    const double rho0_, rho1_;
  };
  
  ~VarDen1DMMSSolnVar(){}
  void advertise_dependents( Expr::ExprDeps& exprDeps ){ exprDeps.requires_expression( indepVarTag_ );}
  void bind_fields( const Expr::FieldManagerList& fml ){ t_ = &fml.template field_ref<TimeField>( indepVarTag_ );
  }
  void evaluate();
private:
  const TimeField* t_;
  const Expr::Tag indepVarTag_;
  const double rho0_, rho1_;
};

/***********************************************************************************************/
/**
 *  \class    VarDen1DMMSVelocity
 *  \ingroup  Expressions
 *  \author   Amir Biglari
 *  \date     December, 2012
 *
 *  \brief Provides an expression for velocity at the boundaries in the MMS
 *         that is written for the pressure projection method verification
 *
 *  \tparam FieldT - the type of field for the velocity.
 */
template< typename FieldT >
class VarDen1DMMSVelocity
: public BoundaryConditionBase<FieldT>
{
  typedef typename SpatialOps::structured::SingleValueField TimeField;
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  \brief Construct an expression for the velocity at the boundaries, given the tag for time
     *         at x = 15 and x = -15
     *
     *  \param indepVarTag the Expr::Tag for holding the time variable.
     *
     *  \param side an enum for specifying the side of the boundary, wether it is on the left or the right side
     */
    Builder( const Expr::Tag& resultTag,
            const Expr::Tag& indepVarTag,
            const SpatialOps::structured::BCSide side ) :
    ExpressionBuilder(resultTag),
    indepVarTag_ (indepVarTag),
    side_ (side)
    {}
    Expr::ExpressionBase* build() const{ return new VarDen1DMMSVelocity(indepVarTag_, side_); }
  private:
    const Expr::Tag indepVarTag_;
    const SpatialOps::structured::BCSide side_;
  };
  
  ~VarDen1DMMSVelocity(){}
  void advertise_dependents( Expr::ExprDeps& exprDeps ){ exprDeps.requires_expression( indepVarTag_ );}
  void bind_fields( const Expr::FieldManagerList& fml ){ t_ = &fml.template field_ref<TimeField>( indepVarTag_ );}
  void evaluate();
private:
  VarDen1DMMSVelocity( const Expr::Tag& indepVarTag,
                     const SpatialOps::structured::BCSide side )
  : indepVarTag_ (indepVarTag),
  side_ (side)
  {
    this->set_gpu_runnable(false);
  }
  const TimeField* t_;
  const Expr::Tag indepVarTag_;
  const SpatialOps::structured::BCSide side_;
};

/***********************************************************************************************/
/**
 *  \class    VarDenCorrugatedMMSMixFrac
 *  \ingroup  Expressions
 *  \author   Tony Saad
 *  \date     January, 2014
 *
 *  \brief Provides an expression for the corrugated-MMS mixture fraction.
 *
 *  \tparam FieldT - the type of field for the mixture fraction.
 */
template< typename FieldT >
class VarDenCorrugatedMMSBCBase
: public BoundaryConditionBase<FieldT>
{
public:
  ~VarDenCorrugatedMMSBCBase(){}
  
  void advertise_dependents( Expr::ExprDeps& exprDeps ){
    exprDeps.requires_expression( xTag_ );
    exprDeps.requires_expression( yTag_ );
    exprDeps.requires_expression( tTag_ );
  }
  
  void bind_fields( const Expr::FieldManagerList& fml ){
    x_ = &fml.template field_manager<FieldT   >().field_ref( xTag_ );
    y_ = &fml.template field_manager<FieldT   >().field_ref( yTag_ );
    t_ = &fml.template field_manager<TimeField>().field_ref( tTag_ );
  }
  void evaluate(){}
  
protected:
  typedef typename SpatialOps::structured::SingleValueField TimeField;
  VarDenCorrugatedMMSBCBase( const Expr::Tag& xTag,
                            const Expr::Tag& yTag,
                            const Expr::Tag& tTag,
                            const double r0,
                            const double r1,
                            const double d,
                            const double w,
                            const double k,
                            const double a,
                            const double b,
                            const double uf,
                            const double vf ) :
  r0_( r0 ),
  r1_( r1 ),
  d_ ( d ),
  w_ ( w ),
  k_ ( k ),
  a_ ( a ),
  b_ ( b ),
  uf_ ( uf ),
  vf_ ( vf ),
  xTag_( xTag ),
  yTag_( yTag ),
  tTag_( tTag )
  {
    this->set_gpu_runnable(false);
  }
  const FieldT *x_, *y_;
  const TimeField* t_;
  const double r0_, r1_, d_, w_, k_, a_, b_, uf_, vf_;
  const Expr::Tag xTag_, yTag_, tTag_;
};


/***********************************************************************************************/
/**
 *  \class    VarDenCorrugatedMMSMixFrac
 *  \ingroup  Expressions
 *  \author   Tony Saad
 *  \date     January, 2014
 *
 *  \brief Provides an expression for the corrugated-MMS mixture fraction.
 *
 *  \tparam FieldT - the type of field for the mixture fraction.
 */
template< typename FieldT >
class VarDenCorrugatedMMSMixFracBC
: public VarDenCorrugatedMMSBCBase<FieldT>
{
  typedef typename SpatialOps::structured::SingleValueField TimeField;
  VarDenCorrugatedMMSMixFracBC( const Expr::Tag& xTag,
                            const Expr::Tag& yTag,
                            const Expr::Tag& tTag,
                            const double r0,
                            const double r1,
                            const double d,
                            const double w,
                            const double k,
                            const double a,
                            const double b,
                            const double uf,
                            const double vf ) :
  VarDenCorrugatedMMSBCBase<FieldT>(xTag, yTag, tTag, r0, r1, d, w, k, a, b, uf, vf)
  {
    this->set_gpu_runnable(false);
  }
  
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  \brief Construct an expression for the mixture fraction at the boundaries, given the tag for time
     *         at x = 15 and x = -15
     *
     *  \param indepVarTag the Expr::Tag for holding the time variable.
     */
    Builder( const Expr::Tag& resultTag,
            const Expr::Tag& xTag,
            const Expr::Tag& yTag,
            const Expr::Tag& tTag,
            const double r0,
            const double r1,
            const double d,
            const double w,
            const double k,
            const double a,
            const double b,
            const double uf,
            const double vf) :
    ExpressionBuilder(resultTag),
    r0_( r0 ),
    r1_( r1 ),
    d_ ( d ),
    w_ ( w ),
    k_ ( k ),
    a_ ( a ),
    b_ ( b ),
    uf_ ( uf ),
    vf_ ( vf ),
    xTag_( xTag ),
    yTag_( yTag ),
    tTag_( tTag )
    {}
    Expr::ExpressionBase* build() const{
      return new VarDenCorrugatedMMSMixFracBC(xTag_, yTag_, tTag_, r0_, r1_, d_, w_, k_, a_, b_, uf_, vf_);
    }
  private:
    const double r0_, r1_, d_, w_, k_, a_, b_, uf_, vf_;
    const Expr::Tag xTag_, yTag_, tTag_;
  };
  
  ~VarDenCorrugatedMMSMixFracBC(){}
  void evaluate();
};

/***********************************************************************************************/
/**
 *  \class    VarDenCorrugatedMMSRhof
 *  \ingroup  Expressions
 *  \author   Tony Saad
 *  \date     January, 2014
 *
 *  \brief Provides an expression for the corrugated-MMS mixture fraction.
 *
 *  \tparam FieldT - the type of field for the mixture fraction.
 */
template< typename FieldT >
class VarDenCorrugatedMMSRhofBC
: public VarDenCorrugatedMMSBCBase<FieldT>
{
  typedef typename SpatialOps::structured::SingleValueField TimeField;
  VarDenCorrugatedMMSRhofBC( const Expr::Tag& xTag,
                               const Expr::Tag& yTag,
                               const Expr::Tag& tTag,
                               const double r0,
                               const double r1,
                               const double d,
                               const double w,
                               const double k,
                               const double a,
                               const double b,
                               const double uf,
                               const double vf ) :
  VarDenCorrugatedMMSBCBase<FieldT>(xTag, yTag, tTag, r0, r1, d, w, k, a, b, uf, vf)
  {
    this->set_gpu_runnable(false);
  }
  
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  \brief Construct an expression for the mixture fraction at the boundaries, given the tag for time
     *         at x = 15 and x = -15
     *
     *  \param indepVarTag the Expr::Tag for holding the time variable.
     */
    Builder( const Expr::Tag& resultTag,
            const Expr::Tag& xTag,
            const Expr::Tag& yTag,
            const Expr::Tag& tTag,
            const double r0,
            const double r1,
            const double d,
            const double w,
            const double k,
            const double a,
            const double b,
            const double uf,
            const double vf) :
    ExpressionBuilder(resultTag),
    r0_( r0 ),
    r1_( r1 ),
    d_ ( d ),
    w_ ( w ),
    k_ ( k ),
    a_ ( a ),
    b_ ( b ),
    uf_ ( uf ),
    vf_ ( vf ),
    xTag_( xTag ),
    yTag_( yTag ),
    tTag_( tTag )
    {}
    Expr::ExpressionBase* build() const{
      return new VarDenCorrugatedMMSRhofBC(xTag_, yTag_, tTag_, r0_, r1_, d_, w_, k_, a_, b_, uf_, vf_);
    }
  private:
    const double r0_, r1_, d_, w_, k_, a_, b_, uf_, vf_;
    const Expr::Tag xTag_, yTag_, tTag_;
  };
  
  ~VarDenCorrugatedMMSRhofBC(){}
  void evaluate();
};


/***********************************************************************************************/
/**
 *  \class    VarDenCorrugatedMMSVelocityBC
 *  \ingroup  Expressions
 *  \author   Tony Saad
 *  \date     January, 2014
 *
 *  \brief Provides an expression for the corrugated-MMS mixture fraction.
 *
 *  \tparam FieldT - the type of field for the mixture fraction.
 */
template< typename FieldT >
class VarDenCorrugatedMMSVelocityBC
: public VarDenCorrugatedMMSBCBase<FieldT>
{
  typedef typename SpatialOps::structured::SingleValueField TimeField;
  VarDenCorrugatedMMSVelocityBC( const Expr::Tag& xTag,
                               const Expr::Tag& yTag,
                               const Expr::Tag& tTag,
                               const double r0,
                               const double r1,
                               const double d,
                               const double w,
                               const double k,
                               const double a,
                               const double b,
                               const double uf,
                               const double vf ) :
  VarDenCorrugatedMMSBCBase<FieldT>(xTag, yTag, tTag, r0, r1, d, w, k, a, b, uf, vf)
  {
    this->set_gpu_runnable(false);
  }
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  \brief Construct an expression for the mixture fraction at the boundaries, given the tag for time
     *         at x = 15 and x = -15
     *
     *  \param indepVarTag the Expr::Tag for holding the time variable.
     */
    Builder( const Expr::Tag& resultTag,
            const Expr::Tag& xTag,
            const Expr::Tag& yTag,
            const Expr::Tag& tTag,
            const double r0,
            const double r1,
            const double d,
            const double w,
            const double k,
            const double a,
            const double b,
            const double uf,
            const double vf) :
    ExpressionBuilder(resultTag),
    r0_( r0 ),
    r1_( r1 ),
    d_ ( d ),
    w_ ( w ),
    k_ ( k ),
    a_ ( a ),
    b_ ( b ),
    uf_ ( uf ),
    vf_ ( vf ),
    xTag_( xTag ),
    yTag_( yTag ),
    tTag_( tTag )
    {}
    Expr::ExpressionBase* build() const{
      return new VarDenCorrugatedMMSVelocityBC(xTag_, yTag_, tTag_, r0_, r1_, d_, w_, k_, a_, b_, uf_, vf_);
    }
  private:
    const double r0_, r1_, d_, w_, k_, a_, b_, uf_, vf_;
    const Expr::Tag xTag_, yTag_, tTag_;
  };
  
  ~VarDenCorrugatedMMSVelocityBC(){}
  void evaluate();
};

/***********************************************************************************************/
/**
 *  \class    VarDenCorrugatedMMSMomBC
 *  \ingroup  Expressions
 *  \author   Tony Saad
 *  \date     January, 2014
 *
 *  \brief Provides an expression for the corrugated-MMS mixture fraction.
 *
 *  \tparam FieldT - the type of field for the mixture fraction.
 */
template< typename FieldT >
class VarDenCorrugatedMMSMomBC
: public VarDenCorrugatedMMSBCBase<FieldT>
{
  typedef typename SpatialOps::structured::SingleValueField TimeField;
  VarDenCorrugatedMMSMomBC( const Expr::Tag& xTag,
                                const Expr::Tag& yTag,
                                const Expr::Tag& tTag,
                                const double r0,
                                const double r1,
                                const double d,
                                const double w,
                                const double k,
                                const double a,
                                const double b,
                                const double uf,
                                const double vf ) :
  VarDenCorrugatedMMSBCBase<FieldT>(xTag, yTag, tTag, r0, r1, d, w, k, a, b, uf, vf)
  {
    this->set_gpu_runnable(false);
  }
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  \brief Construct an expression for the mixture fraction at the boundaries, given the tag for time
     *         at x = 15 and x = -15
     *
     *  \param indepVarTag the Expr::Tag for holding the time variable.
     */
    Builder( const Expr::Tag& resultTag,
            const Expr::Tag& xTag,
            const Expr::Tag& yTag,
            const Expr::Tag& tTag,
            const double r0,
            const double r1,
            const double d,
            const double w,
            const double k,
            const double a,
            const double b,
            const double uf,
            const double vf) :
    ExpressionBuilder(resultTag),
    r0_( r0 ),
    r1_( r1 ),
    d_ ( d ),
    w_ ( w ),
    k_ ( k ),
    a_ ( a ),
    b_ ( b ),
    uf_ ( uf ),
    vf_ ( vf ),
    xTag_( xTag ),
    yTag_( yTag ),
    tTag_( tTag )
    {}
    Expr::ExpressionBase* build() const{
      return new VarDenCorrugatedMMSMomBC(xTag_, yTag_, tTag_, r0_, r1_, d_, w_, k_, a_, b_, uf_, vf_);
    }
  private:
    const double r0_, r1_, d_, w_, k_, a_, b_, uf_, vf_;
    const Expr::Tag xTag_, yTag_, tTag_;
  };
  
  ~VarDenCorrugatedMMSMomBC(){}
  void evaluate();
};

/***********************************************************************************************/
/**
 *  \class    VarDenCorrugatedMMSMomBC
 *  \ingroup  Expressions
 *  \author   Tony Saad
 *  \date     January, 2014
 *
 *  \brief Provides an expression for the corrugated-MMS mixture fraction.
 *
 *  \tparam FieldT - the type of field for the mixture fraction.
 */
template< typename FieldT >
class VarDenCorrugatedMMSyMomBC
: public VarDenCorrugatedMMSBCBase<FieldT>
{
  typedef typename SpatialOps::structured::SingleValueField TimeField;
  VarDenCorrugatedMMSyMomBC( const Expr::Tag& xTag,
                           const Expr::Tag& yTag,
                           const Expr::Tag& tTag,
                           const double r0,
                           const double r1,
                           const double d,
                           const double w,
                           const double k,
                           const double a,
                           const double b,
                           const double uf,
                           const double vf ) :
  VarDenCorrugatedMMSBCBase<FieldT>(xTag, yTag, tTag, r0, r1, d, w, k, a, b, uf, vf)
  {
    this->set_gpu_runnable(false);
  }
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  \brief Construct an expression for the mixture fraction at the boundaries, given the tag for time
     *         at x = 15 and x = -15
     *
     *  \param indepVarTag the Expr::Tag for holding the time variable.
     */
    Builder( const Expr::Tag& resultTag,
            const Expr::Tag& xTag,
            const Expr::Tag& yTag,
            const Expr::Tag& tTag,
            const double r0,
            const double r1,
            const double d,
            const double w,
            const double k,
            const double a,
            const double b,
            const double uf,
            const double vf) :
    ExpressionBuilder(resultTag),
    r0_( r0 ),
    r1_( r1 ),
    d_ ( d ),
    w_ ( w ),
    k_ ( k ),
    a_ ( a ),
    b_ ( b ),
    uf_ ( uf ),
    vf_ ( vf ),
    xTag_( xTag ),
    yTag_( yTag ),
    tTag_( tTag )
    {}
    Expr::ExpressionBase* build() const{
      return new VarDenCorrugatedMMSyMomBC(xTag_, yTag_, tTag_, r0_, r1_, d_, w_, k_, a_, b_, uf_, vf_);
    }
  private:
    const double r0_, r1_, d_, w_, k_, a_, b_, uf_, vf_;
    const Expr::Tag xTag_, yTag_, tTag_;
  };
  
  ~VarDenCorrugatedMMSyMomBC(){}
  void evaluate();
};

/***********************************************************************************************/
/**
 *  \class    VarDenCorrugatedMMSMomBC
 *  \ingroup  Expressions
 *  \author   Tony Saad
 *  \date     January, 2014
 *
 *  \brief Provides an expression for the corrugated-MMS mixture fraction.
 *
 *  \tparam FieldT - the type of field for the mixture fraction.
 */
template< typename FieldT >
class VarDenCorrugatedMMSRho
: public VarDenCorrugatedMMSBCBase<FieldT>
{
  typedef typename SpatialOps::structured::SingleValueField TimeField;
  VarDenCorrugatedMMSRho( const Expr::Tag& xTag,
                            const Expr::Tag& yTag,
                            const Expr::Tag& tTag,
                            const double r0,
                            const double r1,
                            const double d,
                            const double w,
                            const double k,
                            const double a,
                            const double b,
                            const double uf,
                            const double vf ) :
  VarDenCorrugatedMMSBCBase<FieldT>(xTag, yTag, tTag, r0, r1, d, w, k, a, b, uf, vf)
  {
    this->set_gpu_runnable(false);
  }
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  \brief Construct an expression for the mixture fraction at the boundaries, given the tag for time
     *         at x = 15 and x = -15
     *
     *  \param indepVarTag the Expr::Tag for holding the time variable.
     */
    Builder( const Expr::Tag& resultTag,
            const Expr::Tag& xTag,
            const Expr::Tag& yTag,
            const Expr::Tag& tTag,
            const double r0,
            const double r1,
            const double d,
            const double w,
            const double k,
            const double a,
            const double b,
            const double uf,
            const double vf) :
    ExpressionBuilder(resultTag),
    r0_( r0 ),
    r1_( r1 ),
    d_ ( d ),
    w_ ( w ),
    k_ ( k ),
    a_ ( a ),
    b_ ( b ),
    uf_ ( uf ),
    vf_ ( vf ),
    xTag_( xTag ),
    yTag_( yTag ),
    tTag_( tTag )
    {}
    Expr::ExpressionBase* build() const{
      return new VarDenCorrugatedMMSRho(xTag_, yTag_, tTag_, r0_, r1_, d_, w_, k_, a_, b_, uf_, vf_);
    }
  private:
    const double r0_, r1_, d_, w_, k_, a_, b_, uf_, vf_;
    const Expr::Tag xTag_, yTag_, tTag_;
  };
  
  ~VarDenCorrugatedMMSRho(){}
  void evaluate();
};

/***********************************************************************************************/

#endif // Var_Dens_MMS_Density_Expr_h
