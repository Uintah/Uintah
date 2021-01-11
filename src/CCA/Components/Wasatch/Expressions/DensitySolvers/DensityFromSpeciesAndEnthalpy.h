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

#ifndef Wasatch_DensityFromSpeciesAndEnthalpy_h
#define Wasatch_DensityFromSpeciesAndEnthalpy_h

#include <sci_defs/wasatch_defs.h>

#ifndef HAVE_POKITT
// kill compilation if we don't have pokitt.
#error density solver for low-Mach species transport requires PoKiTT.
#endif

#include <CCA/Components/Wasatch/Expressions/DensitySolvers/DensityCalculatorBase.h>

namespace WasatchCore{


/**
 *  \class DensityFromSpeciesAndEnthalpy
 *  \author Josh McConnell
 *  \date September 2018
 *
 *  \brief When transporting \f$\rho Y_i\f$ and \f$\rho h\f$, this expression will
 *  calculate \f$\rho\f$,  \f$\Y_i\f$, \f$\h\f$, and \f$\T\f$.
 *
 *  Given the density-weighted species mass fractions, \f$\rho Y_i\f$ and density-weighted
 *  enthalpy, \f$\rho h\f$, this expression finds \f$\rho\f$, \f$\h\f$, and f$\f$\Y_i\f$
 *  for \f$\0 \leq i \leq n-1\f$, where \f$\n\f$ is the number of species.
 *  A guess for density is given in the following form:
 *  \f[
 *    \rho = \mathcal{G}_\rho (Y_j, T) = \frac{pM}{RT}
 *  \f]
 *  In residual form, the equations to be solved are
 *  \f[
 *   r_{\phi_i} = \mathcal{G}_\rho phi_i - (\rho phi_i)
 *  \f]
 *  Where \f$\phi = \left\{ Y_i,h \}\right\f$
 *  and elements of the Jacobian matrix are
 *  \f[
 *  J_{ij}=\frac{\partial r_i}{\partial \beta_j}
 *  \f]
 *  Where \f$\beta = \left\{ Y_i,T \}\right\f$
 */
  template< typename FieldT >
  class DensityFromSpeciesAndEnthalpy : protected DensityCalculatorBase<FieldT>
  {
    const int nSpec_;
    const Expr::TagList yOldTags_, yNewTags_, rhoYTags_,  dRhodYTags_;
    const Expr::Tag     &hOldTag_, &hNewTag_, &rhoHTag_, &dRhodHTag_, &temperatureOldTag_, 
                        &temperatureNewTag_, mmwOldTag_, mmwNewTag_, &dRhodTempertureTag_, 
                        pressureTag_, cpTag_;

    Expr::TagList hiTags_;

    DECLARE_FIELDS(FieldT, rhoOld_, rhoH_, hOld_, temperatureOld_, mmwOld_, pressure_)
    DECLARE_VECTOR_OF_FIELDS(FieldT, rhoY_ )
    DECLARE_VECTOR_OF_FIELDS(FieldT, yOld_ )
        
    DensityFromSpeciesAndEnthalpy( const Expr::Tag&     rhoOldTag,
                                   const Expr::TagList& rhoYTags,
                                   const Expr::Tag&     rhoHTag,
                                   const Expr::TagList& yOldTags,
                                   const Expr::Tag&     hOldTag,
                                   const Expr::Tag&     temperatureOldTag,
                                   const Expr::Tag&     mmwOldTag,
                                   const Expr::Tag&     pressureTag,
                                   const double         rTol,
                                   const unsigned       maxIter );
    
    //todo: decide if this should be different 
    inline double get_normalization_factor( const unsigned i ) const{
      if(i < nSpec_-1) return 0.5; // nominal value for each species
      else return 500; // nominal value for temperature
    }


  public:
    /**
     *  @class Builder
     *  @brief Build a DensityFromSpeciesAndEnthalpy expression
     */
    class Builder : public Expr::ExpressionBuilder
    {
    public:
      /**
       *  @param rhoNewTag density computed by this expression
       *  @param temperatureNewTag temperature computed by this expression
       *  @param dRhodYTags derivative of density w.r.t. each species computed by this expression
       *  @param dRhodHTag derivative of density w.r.t.e nthalpy computed by this expression
       *  @param rhoOldTag the density from the previous timestep (used as a guess)
       *  @param rhoYTags density weighted species mass fractions
       *  @param rhoHTag density weighted enthalpy
       *  @param yOldTags the species mass fractions from the previous timestep (used as a guess)
       *  @param hOldTag the enthalpy from the previous timestep (used as a guess)
       *  @param mmwOldTag the mixture molecular weight from the previous timestep
       *  @param temperatureOldTag the heat loss from the previous timestep (used as a guess)
       *  @param pressureTag the heat loss from the previous timestep (used as a guess)
       *  @param rTol  relative solver tolerance
       *  @param maxIter maximum number of solver iterations allowed
       */
      Builder( const Expr::Tag&     rhoNewTag,
               const Expr::Tag&     temperatureNewTag,
               const Expr::TagList& dRhodYTags,
               const Expr::Tag&     dRhodHTag,
               const Expr::Tag&     rhoOldTag,
               const Expr::TagList& rhoYTags,
               const Expr::Tag&     rhoHTag,
               const Expr::TagList& yOldTags,
               const Expr::Tag&     hOldTag,
               const Expr::Tag&     temperatureOldTag,
               const Expr::Tag&     mmwOldTag,
               const Expr::Tag&     pressureTag,
               const double         rtol,
               const unsigned       maxIter );
      
      ~Builder(){}
      Expr::ExpressionBase* build() const{
        return new DensityFromSpeciesAndEnthalpy<FieldT>( rhoOldTag_, rhoYTags_, rhoHTag_, yOldTags_, hOldTag_, 
                                                          temperatureOldTag_, mmwOldTag_, pressureTag_, rtol_, maxIter_ );
      }

    private:
      const Expr::TagList rhoYTags_, yOldTags_;
      const Expr::Tag rhoOldTag_, rhoHTag_, hOldTag_, temperatureOldTag_, mmwOldTag_, pressureTag_; 
      const double rtol_;    ///< relative error tolerance
      const unsigned maxIter_; ///< maximum number of iterations  
    };

    ~DensityFromSpeciesAndEnthalpy();
    void update_other_fields(Expr::UintahFieldManager<FieldT>& fieldTManager) override;
    void set_initial_guesses();
    Expr::IDSet register_local_expressions();
    void evaluate();
  };

}


#endif // Wasatch_DensityFromSpeciesAndEnthalpy_h
