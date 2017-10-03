/*
 * The MIT License
 *
 * Copyright (c) 2015-2017 Parresia Research Limited, New Zealand
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

#include <CCA/Components/MPM/ConstitutiveModel/ArenaSoilBanerjeeBrannon/Models/ElasticModuli_Arena.h>
#include <CCA/Components/MPM/ConstitutiveModel/ArenaSoilBanerjeeBrannon/Models/ModelState_Arena.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Exceptions/ProblemSetupException.h>

using namespace Uintah;
using namespace Vaango;

//#define DEBUG_BULK_MODULUS
         
// Construct a default elasticity model.  
/* From Arenisca3:
 * If the user has specified a nonzero nu1 and nu2, these are used to define a pressure
 * dependent Poisson ratio, which is used to adjust the shear modulus along with the
 * bulk modulus.  The high pressure limit has nu=nu1+nu2;
 * if ((d_cm.nu1!=0.0)&&(d_cm.nu2!=0.0)){
 *     // High pressure bulk modulus:
 *     double nu = d_cm.nu1+d_cm.nu2;
 *     shear = 1.5*bulk*(1.0-2.0*nu)/(1.0+nu);
 * } 
 */

ElasticModuli_Arena::ElasticModuli_Arena(Uintah::ProblemSpecP& ps)
{
  ps->require("b0", d_bulk.b0);      // Tangent Elastic Bulk Modulus Parameter
  ps->require("b1", d_bulk.b1);      // Tangent Elastic Bulk Modulus Parameter
  ps->require("b2", d_bulk.b2);      // Tangent Elastic Bulk Modulus Parameter
  ps->require("b3", d_bulk.b3);      // Tangent Elastic Bulk Modulus Parameter
  ps->require("b4", d_bulk.b4);      // Tangent Elastic Bulk Modulus Parameter

  ps->require("G0", d_shear.G0);     // Tangent Elastic Shear Modulus Parameter
  ps->require("nu1", d_shear.nu1);     // Tangent Elastic Shear Modulus Parameter
  ps->require("nu2", d_shear.nu2);     // Tangent Elastic Shear Modulus Parameter

  checkInputParameters();
}

//--------------------------------------------------------------
// Check that the input parameters are reasonable
//--------------------------------------------------------------
void
ElasticModuli_Arena::checkInputParameters()
{
  std::ostringstream warn;

  if (d_bulk.b0 <= 0.0) {
    warn << "b0 must be positive. b0 = " << d_bulk.b0 << std::endl;
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
  if (d_bulk.b1 < 0.0) {
    warn << "b1 must be nonnegative. b1 = " << d_bulk.b1 << std::endl;
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
  if (d_bulk.b2 < 0.0) {
    warn << "b2 must be nonnegative. b2 = " << d_bulk.b2 << std::endl;
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
  if (d_bulk.b3 < 0.0) {
    warn << "b3 must be nonnegative. b3 = " << d_bulk.b3 << std::endl;
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
  if (d_bulk.b4 < 1.0) {
    warn << "b4 must be >= 1. b4 = " << d_bulk.b4 << std::endl;
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
  if (d_shear.G0 <= 0.0) {
    warn << "G0 must be positive. G0 = " << d_shear.G0 << std::endl;
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
  if (d_bulk.b1 < 1.0e-16) {
    std::cout << "**WARNING** b1 < 1.0e-16. Setting b3 = 0." << __FILE__
              << ":" << __LINE__ << std::endl;
    d_bulk.b3 = 0.0;
  }
}

// Construct a copy of a elasticity model.  
ElasticModuli_Arena::ElasticModuli_Arena(const ElasticModuli_Arena* model)
{
  d_bulk = model->d_bulk;
  d_shear = model->d_shear;
}

// Destructor of elasticity model.  
ElasticModuli_Arena::~ElasticModuli_Arena()
{
}

void 
ElasticModuli_Arena::outputProblemSpec(Uintah::ProblemSpecP& ps)
{
  ProblemSpecP elasticModuli_ps = ps->appendChild("elastic_moduli_model");
  elasticModuli_ps->setAttribute("type", "arena");

  elasticModuli_ps->appendElement("b0",d_bulk.b0);
  elasticModuli_ps->appendElement("b1",d_bulk.b1);
  elasticModuli_ps->appendElement("b2",d_bulk.b2);
  elasticModuli_ps->appendElement("b3",d_bulk.b3);
  elasticModuli_ps->appendElement("b4",d_bulk.b4);

  elasticModuli_ps->appendElement("G0",d_shear.G0);
  elasticModuli_ps->appendElement("nu1",d_shear.nu1);  // Low pressure Poisson ratio
  elasticModuli_ps->appendElement("nu2",d_shear.nu2);  // Pressure-dependent Poisson ratio term
}
         
// Compute the elastic moduli
ElasticModuli 
ElasticModuli_Arena::getInitialElasticModuli() const
{
  double Ks = d_granite.getBulkModulus();
  double KK = Ks*d_bulk.b0;
  double KRatio = KK/Ks;
  double nu = d_shear.nu1 + d_shear.nu2*exp(-KRatio);
  double GG = (nu > 0.0) ? 1.5*KK*(1.0-2.0*nu)/(1.0+nu) : d_shear.G0;
  return ElasticModuli(KK, GG);
}

ElasticModuli 
ElasticModuli_Arena::getCurrentElasticModuli(const ModelStateBase* state_input) 
{
  const ModelState_Arena* state = 
    dynamic_cast<const ModelState_Arena*>(state_input);
  if (!state) {
    std::ostringstream out;
    out << "**ERROR** The correct ModelState object has not been passed."
        << " Need ModelState_Arena.";
    throw Uintah::InternalError(out.str(), __FILE__, __LINE__);
  }

  // Make sure the quantities are positive in compression
  double I1_eff_bar = -state->I1_eff;
  double ev_p_bar = -(state->plasticStrainTensor).Trace();  
  double pw_bar = state->pbar_w;

  // Compute the elastic moduli
  double KK = 0.0;
  double GG = 0.0;

  // *Note*
  // 1) If the mean stress is tensile (I1_eff_bar <= 0.0), the soil has the properties of 
  //    drained soil
  // 2) If the saturation is > 0.0 and the mean stress is compressive, the soil has
  //    properties of the partially saturated soil
  if (state->saturation > 0.0) {
    // Partially saturated material
    double phi = state->porosity;
    double Sw = state->saturation;
    computePartialSaturatedModuli(I1_eff_bar, pw_bar, ev_p_bar, phi, Sw, KK, GG);
#ifdef DEBUG_BULK_MODULUS
    std::cout << "Computing bulk modulus for saturated material:" << std::endl;
    std::cout << "  phi = " << phi << " Sw = " << Sw << " I1bar = " << I1_eff_bar << " K = " << KK << std::endl;
#endif
  } else {
    // Drained material
    computeDrainedModuli(I1_eff_bar, ev_p_bar, KK, GG);

#ifdef DEBUG_BULK_MODULUS
    std::cout << " I1bar = " << I1_eff_bar 
              << " K = " << KK << " G = " << GG << std::endl;
#endif
  }

  /*
  // If tensile then scale by porosity
  if (I1_eff_bar < 0.0) {
    double fac = std::max((1.0 - state->porosity)/(1.0 + state->porosity), 0.00001);
    KK *= fac;
    GG *= fac;
  }
  */

  return ElasticModuli(KK, GG);
}

void
ElasticModuli_Arena::computeDrainedModuli(const double& I1_eff_bar, 
                                              const double& ev_p_bar,
                                              double& KK,
                                              double& GG) 
{
  if (I1_eff_bar > 0.0) {   // Compressive mean stress

    double pressure = I1_eff_bar/3.0;

    // Compute solid matrix bulk modulus
    double K_s = d_granite.computeBulkModulus(pressure); 
    double ns = d_granite.computeDerivBulkModulusPressure(pressure);
    double KsRatio = K_s/(1.0 - ns*pressure/K_s);

    if (d_bulk.b1 > 1.0e-30) {
      // Compute Ev_e
      double ev_e = std::pow((d_bulk.b3*pressure)/(d_bulk.b1*K_s - d_bulk.b2*pressure),
                             (1.0/d_bulk.b4));

      // Compute y, z
      double y = std::pow(ev_e, d_bulk.b4);
      double z = d_bulk.b2*y + d_bulk.b3;

      // Compute compressive bulk modulus
      KK = KsRatio*(d_bulk.b0 + (1/ev_e)*d_bulk.b1*d_bulk.b3*d_bulk.b4*y/(z*z));
#ifdef DEBUG_BULK_MODULUS
      if (std::isnan(KK)) {
       std::cout << "b1 = " << d_bulk.b1 << "K_s = " << K_s << " ns = " << ns << " KsRatio = " << KsRatio
                 << " eve = " << ev_e << " y = " << y << " z = " << z  << std::endl;
      }
#endif
    } else {
      KK = KsRatio*d_bulk.b0;
#ifdef DEBUG_BULK_MODULUS
      if (std::isnan(KK)) {
       std::cout << "K_s = " << K_s << " ns = " << ns << " KsRatio = " << KsRatio << std::endl;
      }
#endif
    }


    // Update the shear modulus (if needed, i.e., nu1 & nu2 > 0)
    double KRatio = KK/K_s;
    double nu = d_shear.nu1 + d_shear.nu2*exp(-KRatio);
    GG = (nu > 0.0) ? 1.5*KK*(1.0-2.0*nu)/(1.0+nu) : d_shear.G0;

  } else {

    // Tensile bulk modulus = Bulk modulus at p = 0
    double K_s0 = d_granite.computeBulkModulus(0.0);
    KK = d_bulk.b0*K_s0;
    double KRatio = KK/K_s0;

    // Tensile shear modulus
    double nu = d_shear.nu1 + d_shear.nu2*exp(-KRatio);
    GG = (nu > 0.0) ? 1.5*KK*(1.0-2.0*nu)/(1.0+nu) : d_shear.G0;
  }

  return;
}

void
ElasticModuli_Arena::computePartialSaturatedModuli(const double& I1_eff_bar, 
                                                       const double& pw_bar,
                                                       const double& ev_p_bar,
                                                       const double& phi,
                                                       const double& S_w,
                                                       double& KK,
                                                       double& GG)
{
  if (I1_eff_bar > 0.0) { // Compressive mean stress

    // Bulk modulus of grains
    double pressure = I1_eff_bar/3.0;
    double K_s = d_granite.computeBulkModulus(pressure);

    // Bulk modulus of air
    double K_a = d_air.computeBulkModulus(pw_bar);

    // Bulk modulus of water
    double K_w = d_water.computeBulkModulus(pw_bar);

    // Bulk modulus of drained material 
    double K_d = 0.0;
    GG = 0.0;
    computeDrainedModuli(I1_eff_bar, ev_p_bar, K_d, GG);

    // Bulk modulus of air + water mixture
    double K_f = 1.0/(S_w/K_w + (1.0-S_w)/K_a);

    // Bulk modulus of partially saturated material (Biot-Grassman model)
    double numer = (1.0 - K_d/K_s)*(1.0 - K_d/K_s);
    double denom = 1.0/K_s*(1.0 - K_d/K_s) + phi*(1.0/K_f - 1.0/K_s);
    KK = K_d + numer/denom;

  } else { // Tensile mean stress

    computeDrainedModuli(I1_eff_bar, ev_p_bar, KK, GG);

  }

  return;
}
