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

/**
 *  \class    VarDensMMSDensity
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
class VarDensMMSDensity : public BoundaryConditionBase<FieldT>
{
  typedef typename SpatialOps::structured::SingleValueField TimeField;
  VarDensMMSDensity( const Expr::Tag& indepVarTag,
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
    Expr::ExpressionBase* build() const{ return new VarDensMMSDensity(indepVarTag_, rho0_, rho1_); }
  private:
    const Expr::Tag indepVarTag_;
    const double rho0_, rho1_;
  };
  
  ~VarDensMMSDensity(){}
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

/**
 *  \class    VarDensMMSMixtureFraction
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
class VarDensMMSMixtureFraction
: public BoundaryConditionBase<FieldT>
{
  typedef typename SpatialOps::structured::SingleValueField TimeField;
  VarDensMMSMixtureFraction( const Expr::Tag& indepVarTag ) :
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
    Expr::ExpressionBase* build() const{ return new VarDensMMSMixtureFraction(indepVarTag_); }
  private:
    const Expr::Tag indepVarTag_;
  };
  
  ~VarDensMMSMixtureFraction(){}
  void advertise_dependents( Expr::ExprDeps& exprDeps ){  exprDeps.requires_expression( indepVarTag_ );}
  void bind_fields( const Expr::FieldManagerList& fml ){
    t_    = &fml.template field_manager<TimeField>().field_ref( indepVarTag_ );
  }
  void evaluate();
private:
  const TimeField* t_;
  const Expr::Tag indepVarTag_;
};

/**
 *  \class    VarDensMMSMomentum
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
class VarDensMMSMomentum
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
    Expr::ExpressionBase* build() const{ return new VarDensMMSMomentum(indepVarTag_, rho0_, rho1_, side_); }
  private:
    const Expr::Tag indepVarTag_;
    const double rho0_, rho1_;
    const SpatialOps::structured::BCSide side_;
  };
  
  ~VarDensMMSMomentum(){}
  void advertise_dependents( Expr::ExprDeps& exprDeps ){ exprDeps.requires_expression( indepVarTag_ ); }
  void bind_fields( const Expr::FieldManagerList& fml ){ t_ = &fml.template field_ref<TimeField>( indepVarTag_ ); }
  void evaluate();
private:
  VarDensMMSMomentum( const Expr::Tag& indepVarTag,
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

/**
 *  \class    VarDensMMSSolnVar
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
class VarDensMMSSolnVar
: public BoundaryConditionBase<FieldT>
{
  typedef typename SpatialOps::structured::SingleValueField TimeField;
  VarDensMMSSolnVar( const Expr::Tag& indepVarTag,
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
    Expr::ExpressionBase* build() const{ return new VarDensMMSSolnVar(indepVarTag_, rho0_, rho1_); }
  private:
    const Expr::Tag indepVarTag_;
    const double rho0_, rho1_;
  };
  
  ~VarDensMMSSolnVar(){}
  void advertise_dependents( Expr::ExprDeps& exprDeps ){ exprDeps.requires_expression( indepVarTag_ );}
  void bind_fields( const Expr::FieldManagerList& fml ){ t_ = &fml.template field_ref<TimeField>( indepVarTag_ );
  }
  void evaluate();
private:
  const TimeField* t_;
  const Expr::Tag indepVarTag_;
  const double rho0_, rho1_;
};

/**
 *  \class    VarDensMMSVelocity
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
class VarDensMMSVelocity
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
    Expr::ExpressionBase* build() const{ return new VarDensMMSVelocity(indepVarTag_, side_); }
  private:
    const Expr::Tag indepVarTag_;
    const SpatialOps::structured::BCSide side_;
  };
  
  ~VarDensMMSVelocity(){}
  void advertise_dependents( Expr::ExprDeps& exprDeps ){ exprDeps.requires_expression( indepVarTag_ );}
  void bind_fields( const Expr::FieldManagerList& fml ){ t_ = &fml.template field_ref<TimeField>( indepVarTag_ );}
  void evaluate();
private:
  VarDensMMSVelocity( const Expr::Tag& indepVarTag,
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

#endif // Var_Dens_MMS_Density_Expr_h
