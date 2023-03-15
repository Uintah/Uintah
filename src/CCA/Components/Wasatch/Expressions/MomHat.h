/*
 * The MIT License
 *
 * Copyright (c) 2012-2023 The University of Utah
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

#ifndef MOM_HAT
#define MOM_HAT

//-- ExprLib Includes --//
#include <expression/Expression.h>

//-- SpatialOps Includes --//
#include <spatialops/structured/FVStaggered.h>

//-- Wasatch Includes --//
#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>
#include <CCA/Components/Wasatch/TimeIntegratorTools.h>

/**
 *  \class MomHat
 *  \ingroup Expressions
 *  \author Mokbel Karam
 *  \brief Calculates the momentum hat value at the intermediate stage i
 *
 *  \tparam FieldT the type of field for the momentum RHS (nominally
 *          XVolField, YVolField, ZVolField).
 *
 *  The MomHat is the intermediate value of momentum where the velocity doesn't satisfy continuity:
 */
template< typename FieldT, typename DirT>
class MomHat
: public Expr::Expression<FieldT>
{
    typedef typename SpatialOps::SingleValueField SingleValue;
    typedef typename SpatialOps::SVolField PFieldT;
    typedef typename SpatialOps::OperatorTypeBuilder< typename WasatchCore::GradOpSelector<FieldT, DirT>::Gradient, PFieldT, FieldT >::type Grad;

    const bool hasIntrusion_;

    DECLARE_FIELDS(FieldT, rhsPart_,momN_, volfrac_)
    DECLARE_FIELDS(SingleValue, dt_, rkstage_, timestep_)
    
    const WasatchCore::TimeIntegrator* timeIntInfo_;
    const Grad* gradOp_;

    MomHat(const Expr::Tag& partRHS,const Expr::Tag& momN, const Expr::Tag& volFracTag);
    
public:
    class Builder : public Expr::ExpressionBuilder
    {
        const Expr::Tag pressuret_, rhspartt_, momNt_, volFract_;
    public:
        /**
         *  \param result the result of this expression
         *  \param partRHS the expression to compute the other terms in
         *         the momentum RHS (body force, divergence of convection
         *         and strain)
         *  \param momN the expression to compute the momentum as a X,Y,or Z volume fields at time N
         */
        Builder( const Expr::Tag& result,
                const Expr::Tag& partRHS,
                const Expr::Tag& momN,
                const Expr::Tag& volFracTag);

        ~Builder(){}
        Expr::ExpressionBase* build() const;
    };
    
    ~MomHat();

    void bind_operators( const SpatialOps::OperatorDatabase& opDB );
    void evaluate();
    
};

#endif // MOM_HAT
