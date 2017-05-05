/**
 *  \file   EnthalpyTransportEquation.h
 *  \date   Nov 12, 2013
 *  \author "James C. Sutherland"
 *
 *
 * The MIT License
 *
 * Copyright (c) 2013-2017 The University of Utah
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
 *
 */
#ifndef ENTHALPYTRANSPORTEQUATION_H_
#define ENTHALPYTRANSPORTEQUATION_H_

#include <CCA/Components/Wasatch/Transport/ScalarTransportEquation.h>

namespace WasatchCore {

  /**
   *  \class  WasatchCore::EnthalpyTransportEquation
   *  \date   November, 2013
   *  \author "James C. Sutherland"
   *
   *  The enthalpy transport equation has the form:
   *  \f[
   *  \frac{\partial \rho h}{\partial t} = \frac{Dp}{Dt} - \nabla\cdot(\rho h\vec{v})
   *      - \tau:\nabla\vec{v} - \nabla\cdot\vec{q}
   *  \f]
   *  where, neglecting the Dufour effect, the heat flux is:
   *  \f[
   *    \vec{q} = -\lambda\nabla T + \sum_{i=1}^{n}h_i \vec{j}_i + \vec{q}_\mathrm{rad},
   *  \f]
   *  \f$\vec{j}_i\f$ is the mass diffusive flux of species \f$i\f$ relative
   *  to a mass-averaged velocity, and \f$\vec{q}_\mathrm{rad}\f$ is the
   *  radiative heat flux.
   *
   *  If we assume:
   *   - negligible viscous heating \f$(\tau:\nabla\vec{v} \approx 0)\f$
   *   - constant pressure (to leading order): \f$\frac{Dp}{Dt}\approx0\f$
   *   - all species have unity Lewis numbers, \f$Le_i=\frac{\lambda}{\rho c_p D_i}\f$
   *
   *  then the enthalpy transport equation can be written as
   *  \f[
   *  \frac{\partial \rho h}{\partial t} = - \nabla\cdot(\rho h\vec{v})
   *   - \nabla\cdot\left(-\frac{\lambda}{c_p}\nabla h\right)
   *   - \nabla\cdot\vec{q}_\mathrm{rad}
   *  \f]
   *  where \f$f\f$ is the mixture fraction and \f$D_f\f$ is its diffusivity.
   *  In obtaining this equation, we have used the thermodynamic relationship
   *  \f[
   *  dh = c_p dT +\sum h_i dY_i
   *  \f]
   *  in addition to the unity Lewis number assumption to simplify the diffusive terms.
   *
   *  In the case of turbulent flow, the molecular diffusivity is augmented:
   *  \f[
   *    \left( \frac{\lambda}{c_p} + \frac{\mu_t}{Pr_t} \right)
   *  \f]
   */
  class EnthalpyTransportEquation : public ScalarTransportEquation<SVolField>
  {
  public:
    /**
     *  \brief Construct an EnthalpyTransportEquation
     *  \param enthalpyName the name for enthalpy
     *  \param params the tag from the input file specifying the transport equation.
     *  \param gc
     *  \param densityTag a tag containing density for necessary cases. it will be empty where
     *         it is not needed.
     *  \param isConstDensity true for constant density
     *  \param turbulenceParams
     */
    EnthalpyTransportEquation( const std::string enthalpyName,
                               Uintah::ProblemSpecP params,
                               GraphCategories& gc,
                               const Expr::Tag densityTag,
                               const bool isConstDensity,
                               const TurbulenceParameters& turbulenceParams );

    ~EnthalpyTransportEquation();

  protected:
    void setup_diffusive_flux( FieldTagInfo& );
    void setup_source_terms( FieldTagInfo&, Expr::TagList& );
    const Expr::Tag diffCoeffTag_;

  };

} /* namespace WasatchCore */

#endif /* ENTHALPYTRANSPORTEQUATION_H_ */
