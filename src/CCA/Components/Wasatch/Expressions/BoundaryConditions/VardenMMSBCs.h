/*
 * The MIT License
 *
 * Copyright (c) 2012-2017 The University of Utah
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

#ifndef VardenMMSBCs_h
#define VardenMMSBCs_h

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
class VarDen1DMMSDensity : public WasatchCore::BoundaryConditionBase<FieldT>
{
  typedef typename SpatialOps::SingleValueField TimeField;
  VarDen1DMMSDensity( const Expr::Tag& indepVarTag,
                      const double rho0,
                      const double rho1 )
  : rho0_(rho0),
    rho1_(rho1)
  {
    this->set_gpu_runnable(false);
     t_ = this->template create_field_request<TimeField>(indepVarTag);
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
             const double rho1 )
    : ExpressionBuilder(resultTag),
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
  void evaluate();
private:
  const double rho0_, rho1_;
  DECLARE_FIELD(TimeField, t_)
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
: public WasatchCore::BoundaryConditionBase<FieldT>
{
  typedef typename SpatialOps::SingleValueField TimeField;
  VarDen1DMMSMixtureFraction( const Expr::Tag& indepVarTag )
  {
    this->set_gpu_runnable(false);
     t_ = this->template create_field_request<TimeField>(indepVarTag);
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
    Builder( const Expr::Tag& resultTag, const Expr::Tag& indepVarTag )
    : ExpressionBuilder(resultTag),
      indepVarTag_(indepVarTag)
    {}
    Expr::ExpressionBase* build() const{ return new VarDen1DMMSMixtureFraction(indepVarTag_); }
  private:
    const Expr::Tag indepVarTag_;
  };
  
  ~VarDen1DMMSMixtureFraction(){}
  void evaluate();
private:
  DECLARE_FIELD(TimeField, t_)
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
: public WasatchCore::BoundaryConditionBase<FieldT>
{
  typedef typename SpatialOps::SingleValueField TimeField;
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
             const SpatialOps::BCSide side )
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
    const SpatialOps::BCSide side_;
  };
  
  ~VarDen1DMMSMomentum(){}
  void evaluate();
private:
  VarDen1DMMSMomentum( const Expr::Tag& indepVarTag,
                       const double rho0,
                       const double rho1,
                       const SpatialOps::BCSide side )
  : rho0_( rho0 ),
    rho1_( rho1 ),
    side_( side )
  {
    this->set_gpu_runnable(false);
     t_ = this->template create_field_request<TimeField>(indepVarTag);
  }
  const double rho0_, rho1_;
  const SpatialOps::BCSide side_;
  DECLARE_FIELD(TimeField, t_)
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
: public WasatchCore::BoundaryConditionBase<FieldT>
{
  typedef typename SpatialOps::SingleValueField TimeField;
  VarDen1DMMSSolnVar( const Expr::Tag& indepVarTag,
                     const double rho0,
                     const double rho1  )
  : rho0_( rho0 ),
    rho1_( rho1 )
  {
    this->set_gpu_runnable(false);
     t_ = this->template create_field_request<TimeField>(indepVarTag);
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
             const double rho1 )
   : ExpressionBuilder(resultTag),
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
  void evaluate();
private:
  const double rho0_, rho1_;
  DECLARE_FIELD(TimeField, t_)
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
: public WasatchCore::BoundaryConditionBase<FieldT>
{
  typedef typename SpatialOps::SingleValueField TimeField;
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
             const SpatialOps::BCSide side )
    : ExpressionBuilder(resultTag),
      indepVarTag_ (indepVarTag),
      side_ (side)
    {}
    Expr::ExpressionBase* build() const{ return new VarDen1DMMSVelocity(indepVarTag_, side_); }
  private:
    const Expr::Tag indepVarTag_;
    const SpatialOps::BCSide side_;
  };
  
  ~VarDen1DMMSVelocity(){}
  void evaluate();
private:
  VarDen1DMMSVelocity( const Expr::Tag& indepVarTag,
                       const SpatialOps::BCSide side )
  : side_(side)
  {
    this->set_gpu_runnable(false);
     t_ = this->template create_field_request<TimeField>(indepVarTag);
  }
  DECLARE_FIELD(TimeField, t_)
  const SpatialOps::BCSide side_;
};

/***********************************************************************************************/

#endif // Var_Dens_MMS_Density_Expr_h
