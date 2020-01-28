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

#ifndef Wasatch_TwoStreamMixingDensity_h
#define Wasatch_TwoStreamMixingDensity_h

#include <tabprops/TabProps.h>

#include <expression/Expression.h>

namespace WasatchCore{

    /**
     *  \class TwoStreamMixingDensity
     *  \author James C. Sutherland, Tony Saad
     *  \date   November, 2013
     *
     *  \brief Computes the density from the density-weighted mixture fraction.
     *
     *  Given the density of two streams 0 and 1, where the mixture fraction
     *  indicates the relative amount of stream 1 present (f=1 being  pure stream 1),
     *  the density is given as
     *  \f[
     *  \frac{1}{\rho} = \frac{f}{\rho_1} + \frac{1-f}{\rho_0}
     *  \f]
     *  This expression calculates \f$\rho\f$ from \f$\rho f\f$ and the above equation.
     *  Assuming that we know the product \f$(\rho f)\f$, we want to solve for the roots of
     *  \f[
     *   (\rho f) = \frac{f}{\frac{f}{\rho_1} + \frac{1-f}{\rho_0}}
     *  \f]
     *  Letting \f$\alpha\equiv\rho f\f$, we find that
     *  \f[
     *  f = \frac{\alpha}{\rho_0}\left[1+\frac{\alpha}{\rho_0}-\frac{\alpha}{\rho_1}\right]^{-1}
     *  \f]
     *  This expression first calculates \f$f\f$ and then uses that to calculate \f$\rho\f$
     */
    template< typename FieldT >
    class TwoStreamMixingDensity : public Expr::Expression<FieldT>
    {
    const double rho0_, rho1_, rhoMin_, rhoMax_;
    DECLARE_FIELD(FieldT, rhof_)
    
    TwoStreamMixingDensity( const Expr::Tag& rhofTag,
                            const double rho0,
                            const double rho1  );
    public:
    class Builder : public Expr::ExpressionBuilder
    {
    public:
        /**
         *  @brief Build a TwoStreamMixingDensity expression
         *  @param resultTag the tag for the value that this expression computes
         */
        Builder( const Expr::TagList& resultsTagList,
                const Expr::Tag& rhofTag,
                const double rho0,
                const double rho1 );

        Expr::ExpressionBase* build() const{ return new TwoStreamMixingDensity<FieldT>(rhofTag_,rho0_,rho1_); }

    private:
        const double rho0_, rho1_;
        const Expr::Tag rhofTag_;
    };

    ~TwoStreamMixingDensity(){}
    void evaluate();
    };

    /**
     *  \class TwoStreamDensFromMixfr
     *  \author James C. Sutherland
     *  \date November, 2013
     *  \brief Given the mixture fraction, calculate the density as \f$\rho=\left(\frac{f}{\rho_1}+\frac{1-f}{\rho_0}\right)^{-1}\f$,
     *  where \f$\rho_0\f$ corresponds to the density when \f$f=0\f$.
     */
    template< typename FieldT >
    class TwoStreamDensFromMixfr : public Expr::Expression<FieldT>
    {
    const double rho0_, rho1_, rhoMin_, rhoMax_;
    DECLARE_FIELD(FieldT, mixfr_)
    
    TwoStreamDensFromMixfr( const Expr::Tag& mixfrTag,
                            const double rho0,
                            const double rho1 );
    public:
    class Builder : public Expr::ExpressionBuilder
    {
    public:
        /**
         *  @brief Build a TwoStreamDensFromMixfr expression
         *  @param resultTag the tag for the value that this expression computes
         */
        Builder( const Expr::TagList& resultsTagList,
                const Expr::Tag& mixfrTag,
                const double rho0,
                const double rho1 );

        Expr::ExpressionBase* build() const{ return new TwoStreamDensFromMixfr<FieldT>(mixfrTag_,rho0_,rho1_); }

    private:
        const double rho0_, rho1_;
        const Expr::Tag mixfrTag_;
    };

    ~TwoStreamDensFromMixfr(){}
    void evaluate();
    };

}

#endif // Wasatch_TwoStreamMixingDensity_h
