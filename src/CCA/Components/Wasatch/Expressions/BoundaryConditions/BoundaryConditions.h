/*
 * The MIT License
 *
 * Copyright (c) 2012-2018 The University of Utah
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

#ifndef BoundaryConditions_h
#define BoundaryConditions_h

#include "BoundaryConditionBase.h"
#include <expression/Expression.h>

namespace WasatchCore{

  //-------------------------------------------------------------------------------------------------
  /**
   *  \class 	ConstantBC
   *  \ingroup 	Expressions
   *  \author 	Tony Saad
   *  \date    September, 2012
   *
   *  \brief Provides an expression to set basic Dirichlet and Neumann boundary
   *  conditions. Given a BCValue, we set the ghost value such that
   *  \f$ f[ghost] = \alpha f[interior] + \beta BCValue \f$
   *
   *  \tparam FieldT - The type of field for the expression on which this bc applies.
   */
  template< typename FieldT >
  class ConstantBC : public BoundaryConditionBase<FieldT>
  {
    const double bcValue_;
  public:
    ConstantBC( const double bcValue)
      : bcValue_(bcValue)
    {
      this->set_gpu_runnable(true);
    }

    class Builder : public Expr::ExpressionBuilder
    {
      const double bcValue_;
    public:
      /**
       * @param result Tag of the resulting expression.
       * @param bcValue   constant boundary condition value.
       */
      Builder( const Expr::Tag& resultTag, const double bcValue )
    : ExpressionBuilder(resultTag),
      bcValue_(bcValue)
    {}
      inline Expr::ExpressionBase* build() const{ return new ConstantBC(bcValue_); }
    };

    ~ConstantBC(){}
    void evaluate();
  };

  //-------------------------------------------------------------------------------------------------
  /**
   *  \class 	LinearBC
   *  \ingroup 	Expressions
   *  \author 	Tony Saad
   *  \date    September, 2012
   *
   *  \brief Implements a linear profile at the boundary.
   *
   *  \tparam FieldT - The type of field for the expression on which this bc applies.
   */
  template< typename FieldT >
  class LinearBC : public BoundaryConditionBase<FieldT>
  {
    DECLARE_FIELD(FieldT, x_)
    const double a_, b_;

    LinearBC( const Expr::Tag& indepVarTag,
              const double a,
              const double b )
    : a_(a), b_(b)
    {
      this->set_gpu_runnable(true);
      x_ = this->template create_field_request<FieldT>(indepVarTag);
    }
  public:
    class Builder : public Expr::ExpressionBuilder
    {
      const Expr::Tag indepVarTag_;
      const double a_, b_;
    public:
      Builder( const Expr::Tag& resultTag,
               const Expr::Tag& indepVarTag,
               const double a,
               const double b )
    : ExpressionBuilder(resultTag),
      indepVarTag_ (indepVarTag),
      a_(a), b_(b)
    {}
      inline Expr::ExpressionBase* build() const{ return new LinearBC(indepVarTag_, a_, b_); }
    };

    ~LinearBC(){}
    void evaluate();
  };

  //-------------------------------------------------------------------------------------------------
  /**
   *  \class 	ParabolicBC
   *  \ingroup 	Expressions
   *  \author 	Tony Saad
   *  \date    September, 2012
   *
   *  \brief Implements a parabolic profile at the boundary of the form: a*x^2 + b*x + c.
   *
   *  \tparam FieldT - The type of field for the expression on which this bc applies.
   */
  template< typename FieldT >
  class ParabolicBC : public BoundaryConditionBase<FieldT>
  {
    DECLARE_FIELD(FieldT, x_)
    const double a_, b_, c_, x0_;

    ParabolicBC( const Expr::Tag& indepVarTag,
                 const double a, const double b,
                 const double c, const double x0 )
    : a_(a), b_(b), c_(c), x0_(x0)
    {
      this->set_gpu_runnable(true);
      x_ = this->template create_field_request<FieldT>(indepVarTag);
    }
  public:
    class Builder : public Expr::ExpressionBuilder
    {
      const Expr::Tag indepVarTag_;
      const double a_, b_, c_, x0_;
    public:
      /**
       *  \param resultTag The tag of the resulting expression computed here.
       *  \param indepVarTag The tag of the independent variable
       *  \param a  The coefficient of x^2 in the parabolic formula
       *  \param b  The coefficient of x in the parabolic formula
       *  \param c  The constant in the parabolic formula
       *  \param x0 The value of the point (independent variable) where the parabola assumes its maximum/minimum
       */
      Builder( const Expr::Tag& resultTag,
               const Expr::Tag& indepVarTag,
               const double a, const double b,
               const double c, const double x0 )
    : ExpressionBuilder(resultTag),
      indepVarTag_ (indepVarTag),
      a_(a), b_(b), c_(c), x0_(x0)
    {}
      inline Expr::ExpressionBase* build() const{ return new ParabolicBC(indepVarTag_, a_, b_, c_, x0_); }
    };

    ~ParabolicBC(){}
    void evaluate();
  };

  //-------------------------------------------------------------------------------------------------
  /**
   *  \class 	PowerLawBC
   *  \ingroup 	Expressions
   *  \author 	Tony Saad
   *  \date    September, 2012
   *
   *  \brief Implements a powerlaw profile at the boundary.
   *
   *  \tparam FieldT - The type of field for the expression on which this bc applies.
   */
  template< typename FieldT >
  class PowerLawBC : public BoundaryConditionBase<FieldT>
  {
    DECLARE_FIELD(FieldT, x_)
    const double x0_, phic_, R_, n_;
    PowerLawBC( const Expr::Tag& indepVarTag,
                const double x0, const double phiCenter,
                const double halfHeight, const double n )
    : x0_(x0), phic_(phiCenter), R_(halfHeight), n_(n)
    {
      this->set_gpu_runnable(true);
      x_ = this->template create_field_request<FieldT>(indepVarTag);
    }
  public:
    class Builder : public Expr::ExpressionBuilder
    {
      const Expr::Tag indepVarTag_;
      const double x0_, phic_, R_, n_;
    public:
      Builder( const Expr::Tag& resultTag,
               const Expr::Tag& indepVarTag,
               const double x0, const double phiCenter,
               const double halfHeight, const double n)
    : ExpressionBuilder(resultTag),
      indepVarTag_ (indepVarTag),
      x0_(x0), phic_(phiCenter), R_(halfHeight), n_(n)
    {}
      inline Expr::ExpressionBase* build() const{ return new PowerLawBC(indepVarTag_, x0_, phic_, R_, n_); }
    };

    ~PowerLawBC(){}
    void evaluate();
  };

  //-------------------------------------------------------------------------------------------------

  /**
   *  @class GaussianBC
   *  @author James C. Sutherland
   *  @date April, 2009
   *  @brief Implements a gaussian function of a single independent variable.
   *
   * The gaussian function is written as
   *  \f[
   *    f(x) = y_0 + a \exp\left( \frac{\left(x-x_0\right)^2 }{2\sigma^2} \right)
   *  \f]
   * where
   *  - \f$x_0\f$ is the mean (center of the gaussian)
   *  - \f$\sigma\f$ is the standard deviation (width of the gaussian)
   *  - \f$a\f$ is the amplitude of the gaussian
   *  - \f$y_0\f$ is the baseline value.
   */
  template< typename FieldT >
  class GaussianBC : public BoundaryConditionBase<FieldT>
  {
    DECLARE_FIELD(FieldT, x_)
    const double a_, sigma_, mean_, yo_;

    GaussianBC( const Expr::Tag& indepVarTag,
                const double a,
                const double stddev,
                const double mean,
                const double yo )
      : a_( a ), sigma_( stddev ), mean_( mean ), yo_( yo )
    {
      this->set_gpu_runnable(true);
      x_ = this->template create_field_request<FieldT>(indepVarTag);
    }
  public:
    class Builder : public Expr::ExpressionBuilder
    {
      const double a_, sigma_, mean_, yo_;
      const Expr::Tag ivarTag_;
    public:
      Builder( const Expr::Tag& depVarTag,   ///<   dependent variable tag
               const Expr::Tag& indepVarTag, ///< independent variable tag
               const double a,         ///< Amplitude of the Gaussian spike
               const double stddev,    ///< Standard deviation
               const double mean,      ///< Mean of the function
               const double yo=0.0    ///< baseline value
      )
        : Expr::ExpressionBuilder(depVarTag),
          a_(a),
          sigma_(stddev),
          mean_(mean),
          yo_(yo),
          ivarTag_( indepVarTag )
      {}
      inline Expr::ExpressionBase* build() const{ return new GaussianBC( ivarTag_, a_, sigma_, mean_, yo_ ); }
    };

    ~GaussianBC(){}
    void evaluate();
  };

  //-------------------------------------------------------------------------------------------------
  /**
   *  \class 	BCCopier
   *  \ingroup 	Expressions
   *  \author 	Amir Biglari
   *  \date    September, 2012
   *
   *  \brief Provides a mechanism to copy boundary values from one field to another.
   *
   *  \tparam FieldT - The type of field for the expression on which this bc applies.
   */
  template< typename FieldT >
  class BCCopier : public BoundaryConditionBase<FieldT>
  {
    DECLARE_FIELD(FieldT, src_)
    BCCopier( const Expr::Tag& srcTag )
    {
      this->set_gpu_runnable(true);
      src_ = this->template create_field_request<FieldT>(srcTag);
    }
  public:
    class Builder : public Expr::ExpressionBuilder
    {
      const Expr::Tag srcTag_;
    public:
      Builder( const Expr::Tag& resultTag,
               const Expr::Tag& srcTag )
    : ExpressionBuilder(resultTag),
      srcTag_ (srcTag)
    {}
      inline Expr::ExpressionBase* build() const{ return new BCCopier(srcTag_); }
    };

    ~BCCopier(){}
    void evaluate();
  };

  //-------------------------------------------------------------------------------------------------
  /**
   *  \class 	BCPrimVar
   *  \ingroup 	Expressions
   *  \author 	Tony Saad
   *  \date    March, 2014
   *
   *  \brief Provides a mechanism to copy boundary values from one field to another.
   *
   *  \tparam FieldT - the type of field for the RHS.
   */
  template< typename FieldT >
  class BCPrimVar
      : public BoundaryConditionBase<FieldT>
  {
    const bool hasDensity_;
    typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, SVolField, FieldT >::type DenInterpT;
    const DenInterpT* rhoInterpOp_;

    typedef typename SpatialOps::BasicOpTypes<FieldT>::GradX      GradX;
    typedef typename SpatialOps::BasicOpTypes<FieldT>::GradY      GradY;
    typedef typename SpatialOps::BasicOpTypes<FieldT>::GradZ      GradZ;

    typedef typename SpatialOps::NeboBoundaryConditionBuilder<GradX> Neum2XOpT;
    typedef typename SpatialOps::NeboBoundaryConditionBuilder<GradY> Neum2YOpT;
    typedef typename SpatialOps::NeboBoundaryConditionBuilder<GradZ> Neum2ZOpT;
    const Neum2XOpT* neux_;
    const Neum2YOpT* neuy_;
    const Neum2ZOpT* neuz_;

    DECLARE_FIELD(FieldT, src_)
    DECLARE_FIELD(SVolField, rho_)

    BCPrimVar( const Expr::Tag& srcTag,
               const Expr::Tag& densityTag) :
                 hasDensity_(densityTag != Expr::Tag() )
    {
      this->set_gpu_runnable(true);
      src_ = this->template create_field_request<FieldT>(srcTag);
      if (hasDensity_)  rho_ = this->template create_field_request<SVolField>(densityTag);
    }
  public:
    class Builder : public Expr::ExpressionBuilder
    {
      const Expr::Tag srcTag_, densityTag_;
    public:
      Builder( const Expr::Tag& resultTag,
               const Expr::Tag& srcTag,
               const Expr::Tag densityTag = Expr::Tag() )
    : ExpressionBuilder(resultTag),
      srcTag_ (srcTag),
      densityTag_(densityTag)
    {}
      inline Expr::ExpressionBase* build() const{ return new BCPrimVar(srcTag_, densityTag_); }
    };

    ~BCPrimVar(){}
    void evaluate();
    void bind_operators( const SpatialOps::OperatorDatabase& opdb );
  };

  //-------------------------------------------------------------------------------------------------
  /**
   *  \class 	ConstantBCNew
   *  \ingroup 	Expressions
   *  \author 	Tony Saad
   *  \date    Dec, 2016
   *
   *  \brief Prototype for new style of Boundary Conditions.
   *
   *  \tparam FieldT - the type of field for the RHS.
   */
  template< typename FieldT, typename OpT >
  class ConstantBCNew
      : public BoundaryConditionBase<FieldT>
  {
    const OpT* op_;
    double bcVal_;
    ConstantBCNew(double bcVal):
      bcVal_(bcVal)
    {
      this->set_gpu_runnable(true);
    }
  public:
    class Builder : public Expr::ExpressionBuilder
    {
      double bcVal_;
    public:
      Builder( const Expr::Tag& resultTag, double bcVal)
        : ExpressionBuilder(resultTag), bcVal_(bcVal)
      {}
      inline Expr::ExpressionBase* build() const{ return new ConstantBCNew(bcVal_); }
    };

    ~ConstantBCNew(){}
    void evaluate()
    {
      FieldT& lhs = this->value();
      (*op_)(*this->interiorSvolSpatialMask_, lhs, bcVal_, this->isMinusFace_);
    }
    void bind_operators( const SpatialOps::OperatorDatabase& opdb )
    {
      op_ = opdb.retrieve_operator<OpT>();
    }
  };

  //-------------------------------------------------------------------------------------------------
  /**
   *  \class    VelocityDependentConstantBC
   *  \ingroup  Expressions
   *  \author   Mike Hansen
   *  \date     August, 2017
   *
   *  \brief Applies a boundary condition on outflows (or inflows) only, as determined by a velocity field, used for nonreflecting compressible flow BCs
   */
  template< typename FieldT, typename OpT >
  class VelocityDependentConstantBC
      : public BoundaryConditionBase<FieldT>
  {
  public:
    class Builder : public Expr::ExpressionBuilder
    {
    public:
      enum FlowType
      {
        APPLY_ON_INFLOW,
        APPLY_ON_OUTFLOW
      };

    private:
      const double bcVal_;
      double plusMinusSide_;
      const Expr::Tag velocityTag_;
      FlowType flowType_;

    public:
      Builder( const Expr::Tag& resultTag,
               const double bcVal,
               const Uintah::Patch::FaceType& face,
               const Expr::Tag& velocityTag,
               FlowType flowType )
    : ExpressionBuilder( resultTag ),
      bcVal_( bcVal ),
      velocityTag_( velocityTag ),
      flowType_( flowType )
    {
        switch ( face ) {
          case Uintah::Patch::xplus:
          case Uintah::Patch::yplus:
          case Uintah::Patch::zplus:
          {
            plusMinusSide_ = +1.;
          }
          break;
          case Uintah::Patch::xminus:
          case Uintah::Patch::yminus:
          case Uintah::Patch::zminus:
          {
            plusMinusSide_ = -1.;
          }
          break;
          default:
            break;
        }
    }
      inline Expr::ExpressionBase* build() const{ return new VelocityDependentConstantBC( bcVal_,
                                                                                          plusMinusSide_,
                                                                                          velocityTag_,
                                                                                          flowType_ ); }

    };

    private:
    const OpT* op_;

    typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Interpolant,SpatialOps::SVolField,FieldT>::type VelocityInterpOpT;
    const VelocityInterpOpT* velocityInterpOp_;

    const double bcVal_;
    const double plusMinusSide_;
    typename Builder::FlowType flowType_;
    DECLARE_FIELD(SpatialOps::SVolField, normalVelocity_)

    VelocityDependentConstantBC( const double bcVal,
                                 const double plusMinusSide,
                                 const Expr::Tag& velocityTag,
                                 typename Builder::FlowType flowType ) :
                                 bcVal_( bcVal ),
                                 plusMinusSide_( plusMinusSide ),
                                 flowType_( flowType )
    {
      this->set_gpu_runnable( true );
      normalVelocity_ = this->template create_field_request<SpatialOps::SVolField>( velocityTag );
    }

    public:
    ~VelocityDependentConstantBC(){}
    void evaluate()
    {
      using namespace SpatialOps;
      FieldT& lhs = this->value();
      const SpatialOps::SVolField& velocity = normalVelocity_->field_ref();
      SpatFldPtr<FieldT> lhsWithBcApplied = SpatialFieldStore::get<FieldT>( lhs );

      *lhsWithBcApplied <<= lhs;
      (*op_)(*this->interiorSvolSpatialMask_, *lhsWithBcApplied, bcVal_, this->isMinusFace_);


      switch( flowType_ ){
        case Builder::APPLY_ON_OUTFLOW:
          lhs <<= cond( ( plusMinusSide_ * (*velocityInterpOp_)(velocity) ) > 0.0, *lhsWithBcApplied ) ( lhs );
          break;
        case Builder::APPLY_ON_INFLOW:
          lhs <<= cond( ( plusMinusSide_ * (*velocityInterpOp_)(velocity) ) < 0.0, *lhsWithBcApplied ) ( lhs );
          break;
        default:
          break;
      }
    }
    void bind_operators( const SpatialOps::OperatorDatabase& opdb )
    {
      op_ = opdb.retrieve_operator<OpT>();
      velocityInterpOp_ = opdb.retrieve_operator<VelocityInterpOpT>();
    }
  };
  //-------------------------------------------------------------------------------------------------

  /*
   \brief The STAGGERED_MASK macro returns a mask that consists of the two points on each side of the svol extra cell mask.
   This mask is usually used for direct assignment on staggered fields on normal boundaries (XVol on x boundaries, YVol on y boundaries, and ZVol on z boundaries).
   The reason that we create this mask is to set values in the staggered extra cells for appropriate visualization
   */
#define STAGGERED_MASK \
    convert<FieldT>( *(this->svolSpatialMask_), SpatialOps::MINUS_SIDE, SpatialOps::PLUS_SIDE)

  /*
   \brief The APPLY_COMPLEX_BC macro allows the application of complex boundary conditions. The general formulation is as follows:

   APPLY_COMPLEX_BC(f, my_complex_bc)

   where f is the computed field on which the BC is being applied and my_complex_bc is ANY nebo expression of the same type as the field that is
   computed by the expression. So, if you want to apply a BC on a field of type FieldT, then my_complex_bc is of type FieldT.

   Examples:
   APPLY_COMPLEX_BC(f, a*x + b);
   APPLY_COMPLEX_BC(f, a_ * (x - x0_)*(x - x0_) + b_ * (x - x0_) + c_);
   APPLY_COMPLEX_BC(f, phic_ * pow( 1.0 - abs(x - x0_) / R_ , 1.0/n_ ));

   If your independent variable is NOT of the same type as the computed field, then simply use an interpolant
   to move things to the correct location.

   e.g.

   APPLY_COMPLEX_BC(f, a * interOp(y) + b); where interpOp interpolates from yType to fType
   */
#define APPLY_COMPLEX_BC(f, BCVALUE)                                                               \
    {                                                                                               \
  if( this->isStaggeredNormal_ ){                                                                     \
    masked_assign(STAGGERED_MASK, f, BCVALUE);                                                  \
  } else {                                                                                      \
    if (this->setInExtraCellsOnly_)                                                             \
    {                                                                                           \
      masked_assign( *(this->spatialMask_), f, BCVALUE);                                        \
    } else {                                                                                    \
      typedef WasatchCore::BCOpTypeSelector<FieldT> OpT;                                            \
      switch (this->bcTypeEnum_) {                                                              \
        case WasatchCore::DIRICHLET:                                                                \
        {                                                                                       \
          switch (this->faceTypeEnum_) {                                                        \
            case Uintah::Patch::xminus:                                                         \
            case Uintah::Patch::xplus:                                                          \
            {                                                                                   \
              typedef typename OpT::DirichletX::DestFieldType DesT;                             \
              (*this->diriXOp_)(convert<DesT>( *(this->spatialMask_),this->shiftSide_), f, (*this->interpXOp_)(BCVALUE), this->isMinusFace_);  \
              break;                                                                            \
            }                                                                                   \
            case Uintah::Patch::yminus:                                                         \
            case Uintah::Patch::yplus:                                                          \
            {                                                                                   \
              typedef typename OpT::DirichletY::DestFieldType DesT;                             \
              (*this->diriYOp_)(convert<DesT>( *(this->spatialMask_),this->shiftSide_), f, (*this->interpYOp_)(BCVALUE), this->isMinusFace_);  \
              break;                                                                            \
            }                                                                                   \
            case Uintah::Patch::zminus:                                                         \
            case Uintah::Patch::zplus:                                                          \
            {                                                                                   \
              typedef typename OpT::DirichletZ::DestFieldType DesT;                             \
              (*this->diriZOp_)(convert<DesT>( *(this->spatialMask_),this->shiftSide_), f, (*this->interpZOp_)(BCVALUE), this->isMinusFace_);  \
              break;                                                                            \
            }                                                                                   \
            default:                                                                            \
            {                                                                                   \
              break;                                                                            \
            }                                                                                   \
          }                                                                                     \
          break;                                                                                \
        }                                                                                       \
        case WasatchCore::NEUMANN:                                                                  \
        {                                                                                       \
          switch (this->faceTypeEnum_) {                                                        \
            case Uintah::Patch::xminus:                                                         \
            case Uintah::Patch::xplus:                                                          \
            {                                                                                   \
              typedef typename OpT::NeumannX::DestFieldType DesT;                               \
              if (this->isStaggeredNormal_)                                                           \
              {                                                                                 \
                (*this->neumXOp_)(convert<DesT>( *(this->svolSpatialMask_),this->shiftSide_), f, (*this->interpNeuXOp_)(BCVALUE), this->isMinusFace_);  \
              } else {                                                                                                          \
                (*this->neumXOp_)(convert<DesT>( *(this->spatialMask_),this->shiftSide_), f, (*this->interpNeuXOp_)(BCVALUE), this->isMinusFace_);      \
              }                                                                                 \
              break;                                                                            \
            }                                                                                   \
            case Uintah::Patch::yminus:                                                         \
            case Uintah::Patch::yplus:                                                          \
            {                                                                                   \
              typedef typename OpT::NeumannY::DestFieldType DesT;                               \
              if (this->isStaggeredNormal_)                                                           \
              {                                                                                 \
                (*this->neumYOp_)(convert<DesT>( *(this->svolSpatialMask_),this->shiftSide_), f, (*this->interpNeuYOp_)(BCVALUE), this->isMinusFace_);  \
              } else {                                                                                                          \
                (*this->neumYOp_)(convert<DesT>( *(this->spatialMask_),this->shiftSide_), f, (*this->interpNeuYOp_)(BCVALUE), this->isMinusFace_);      \
              }                                                                                 \
              break;                                                                            \
            }                                                                                   \
            case Uintah::Patch::zminus:                                                         \
            case Uintah::Patch::zplus:                                                          \
            {                                                                                   \
              typedef typename OpT::NeumannZ::DestFieldType DesT;                               \
              if (this->isStaggeredNormal_)                                                           \
              {                                                                                 \
                (*this->neumZOp_)(convert<DesT>( *(this->svolSpatialMask_),this->shiftSide_), f, (*this->interpNeuZOp_)(BCVALUE), this->isMinusFace_);  \
              } else {                                                                          \
                (*this->neumZOp_)(convert<DesT>( *(this->spatialMask_),this->shiftSide_), f, (*this->interpNeuZOp_)(BCVALUE), this->isMinusFace_);      \
              }                                                                                 \
              break;                                                                            \
            }                                                                                   \
            default:                                                                            \
            {                                                                                   \
              break;                                                                            \
            }                                                                                   \
          }                                                                                     \
          break;                                                                                \
        }                                                                                       \
        default:                                                                                \
        {                                                                                       \
          std::ostringstream msg;                                                               \
          msg << "ERROR: It looks like you have specified an UNSUPPORTED boundary condition type!"  \
          << "Basic boundary types can only be either DIRICHLET or NEUMANN. Please revise your input file." << std::endl; \
          break;                                                                                \
        }                                                                                     \
      }                                                                                       \
    }                                                                                         \
  }                                                                                             \
    }

  /*
   \brief The APPLY_CONSTANT_BC macro applies a constant boundary condition.
   Here are the rules:

   Staggered field on a boundary perpendicular to the staggered direction (we call this staggered normal):
   ======================================================================================================
   This occurs when setting an inlet x-velocity for example at an x boundary. In general, XVol and X-boundaries,
   YVol and y-boundaries, ZVol and z-boundaries fall into this category. For this case, the domain's boundary
   coincides with the first interior cell for the staggered field. Also, for visualization purposes, we choose
   to set the same bc value in the staggered extra cell. That's why you see the use of STAGGERED_MASK
   in the masked_assign below.

   setInExtraCellsOnly - or Direct Assignment to the extra cell:
   =============================================================
   This will be visited by Scalar BCs only. As the name designates, this will set the bcvalue in the
   extra cell (or spatialMask_) only

   Dirichlet Boundary Conditions:
   ==============================
   Applies standard operator inversion

   Neumann Boundary Condition:
   ===========================
   Applies standard operator inversion. When the field is staggered-normal, use the svolmask instead
   of the staggered mask.
   */
#define APPLY_CONSTANT_BC(f, BCVALUE)                                                              \
  {                                                                                               \
  if( this->isStaggeredNormal_ && this->bcTypeEnum_ != WasatchCore::NEUMANN ){                            \
    masked_assign ( STAGGERED_MASK, f, bcValue_ );                                              \
  } else {                                                                                      \
    if (this->setInExtraCellsOnly_)                                                             \
    {                                                                                           \
      masked_assign( *(this->spatialMask_), f, BCVALUE);                                        \
    } else {                                                                                    \
      typedef WasatchCore::BCOpTypeSelector<FieldT> OpT;                                            \
      switch (this->bcTypeEnum_) {                                                              \
        case WasatchCore::DIRICHLET:                                                                \
        {                                                                                       \
          switch (this->faceTypeEnum_) {                                                        \
            case Uintah::Patch::xminus:                                                         \
            case Uintah::Patch::xplus:                                                          \
            {                                                                                   \
              typedef typename OpT::DirichletX::DestFieldType DesT;                             \
              (*this->diriXOp_)(convert<DesT>( *(this->spatialMask_), this->shiftSide_ ), f, BCVALUE, this->isMinusFace_);  \
              break;                                                                            \
            }                                                                                   \
            case Uintah::Patch::yminus:                                                         \
            case Uintah::Patch::yplus:                                                          \
            {                                                                                   \
              typedef typename OpT::DirichletY::DestFieldType DesT;                             \
              (*this->diriYOp_)( convert<DesT>( *(this->spatialMask_), this->shiftSide_ ), f, BCVALUE, this->isMinusFace_);  \
              break;                                                                            \
            }                                                                                   \
            case Uintah::Patch::zminus:                                                         \
            case Uintah::Patch::zplus:                                                          \
            {                                                                                   \
              typedef typename OpT::DirichletZ::DestFieldType DesT;                             \
              (*this->diriZOp_)(convert<DesT>( *(this->spatialMask_), this->shiftSide_ ), f, BCVALUE, this->isMinusFace_);  \
              break;                                                                            \
            }                                                                                   \
            default:                                                                            \
            {                                                                                   \
              break;                                                                            \
            }                                                                                   \
          }                                                                                     \
          break;                                                                                \
        }                                                                                       \
        case WasatchCore::NEUMANN:                                                                  \
        {                                                                                       \
          switch (this->faceTypeEnum_) {                                                        \
            case Uintah::Patch::xminus:                                                         \
            case Uintah::Patch::xplus:                                                          \
            {                                                                                   \
              typedef typename OpT::NeumannX::DestFieldType DesT;                               \
              if (this->isStaggeredNormal_)                                                           \
              {                                                                                 \
                (*this->neumXOp_)(convert<DesT>( *(this->svolSpatialMask_),this->shiftSide_), f, BCVALUE, this->isMinusFace_);  \
              } else {                                                                                                          \
                (*this->neumXOp_)(convert<DesT>( *(this->spatialMask_),this->shiftSide_), f, BCVALUE, this->isMinusFace_);      \
              }                                                                                 \
              break;                                                                            \
            }                                                                                   \
            case Uintah::Patch::yminus:                                                         \
            case Uintah::Patch::yplus:                                                          \
            {                                                                                   \
              typedef typename OpT::NeumannY::DestFieldType DesT;                               \
              if (this->isStaggeredNormal_)                                                           \
              {                                                                                 \
                (*this->neumYOp_)(convert<DesT>( *(this->svolSpatialMask_),this->shiftSide_), f, BCVALUE, this->isMinusFace_);  \
              } else {                                                                                                          \
                (*this->neumYOp_)(convert<DesT>( *(this->spatialMask_),this->shiftSide_), f, BCVALUE, this->isMinusFace_);      \
              }                                                                                 \
              break;                                                                            \
            }                                                                                   \
            case Uintah::Patch::zminus:                                                         \
            case Uintah::Patch::zplus:                                                          \
            {                                                                                   \
              typedef typename OpT::NeumannZ::DestFieldType DesT;                               \
              if (this->isStaggeredNormal_)                                                           \
              {                                                                                 \
                (*this->neumZOp_)(convert<DesT>( *(this->svolSpatialMask_),this->shiftSide_), f, BCVALUE, this->isMinusFace_);  \
              } else {                                                                          \
                (*this->neumZOp_)(convert<DesT>( *(this->spatialMask_),this->shiftSide_), f, BCVALUE, this->isMinusFace_);      \
              }                                                                                 \
              break;                                                                            \
            }                                                                                   \
            default:                                                                            \
            {                                                                                   \
              break;                                                                            \
            }                                                                                   \
          }                                                                                     \
          break;                                                                                \
        }                                                                                       \
        default:                                                                                \
        {                                                                                       \
          std::ostringstream msg;                                                               \
          msg << "ERROR: It looks like you have specified an UNSUPPORTED boundary condition type!"  \
          << "Basic boundary types can only be either DIRICHLET or NEUMANN. Please revise your input file." << std::endl; \
          break;                                                                                \
        }                                                                                     \
      }                                                                                       \
    }                                                                                         \
  }                                                                                           \
  }
} // namespace WasatchCore

#endif // BoundaryConditions_h
