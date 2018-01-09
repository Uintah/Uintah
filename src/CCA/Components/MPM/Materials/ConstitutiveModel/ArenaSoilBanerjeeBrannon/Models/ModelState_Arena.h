/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
 * Copyright (c) 2015-2016 Parresia Research Limited, New Zealand
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

#ifndef __MODEL_STATE_ARENISCA3_PARTIALLY_SATURATED_H__
#define __MODEL_STATE_ARENISCA3_PARTIALLY_SATURATED_H__

#include <CCA/Components/MPM/Materials/ConstitutiveModel/ArenaSoilBanerjeeBrannon/Models/ModelStateBase.h>
#include <Core/Math/Matrix3.h>

namespace Vaango {

  /////////////////////////////////////////////////////////////////////////////
  /*!
    \class ModelState_Arena
    \brief A structure that stores the state data that is specialized for
    the PartallySaturated model.
    ** Derived from PlasticityState:ModelState
    \author Biswajit Banerjee \n
  */
  /////////////////////////////////////////////////////////////////////////////

  class ModelState_Arena: public ModelStateBase {

  public:

    static const Uintah::Matrix3 Identity;
    static const double sqrtTwo;
    static const double sqrtThree;

    Uintah::long64 particleID;
 
    double capX;      // The cap hydrostatic compressive strength X 
    double kappa;     // The cap kappa parameter (branch point)
    double pbar_w;    // The back stress parameter (-1/3*trace of isotropic backstress)

    Uintah::Matrix3 stressTensor;           // The tensor form of the total stress
    Uintah::Matrix3 deviatoricStressTensor; // The deviatoric part of the total stress
    double I1_eff;    // I1_eff = Tr(sigma_eff) = Tr(sigma) + 3*pbar_w
    double J2;
    double sqrt_J2;   // sqrt(J2) 
    double rr;        // Lode coordinate 'r'
    double zz_eff;    // Lode coordinate 'z'

    Uintah::Matrix3 plasticStrainTensor;  // The tensor form of plastic strain
    double ep_v;      // ep_v = Tr(ep) : Volumetric part of the plastic strain
    double dep_v;     // Increment of the volumetric plastic strain
    double ep_cum_eq; // The cumulative equivalent plastic strain
                      // (This quantity always increases)
    double ep_eq;     // The equivalent plastic strain computed from the current plastic strain
                      // (This quantity can decrease)

    double phi0;        // Initial porosity
    double Sw0;         // Initial saturation
    double saturation;  // Water saturation

    //std::vector<double> yieldParams;  
    std::map<std::string, double> yieldParams;  // The yield parameters for a single particle
                                                // (variability)

    double p3;          // P3 used by disaggregation algorithm
    double t_grow;      // The damage growth time in the Kayenta model
    double coherence;   // The coherence parameter in the Kayenta model

    // Was defined in base class
    double bulkModulus;   // Bulk and shear moduli
    double shearModulus;
    double porosity;    // Porosity
    double density;
    // Matrix3 backStress; // Back stress

    ModelState_Arena();

    ModelState_Arena(const ModelState_Arena& state);
    ModelState_Arena(const ModelState_Arena* state);

    ~ModelState_Arena();

    ModelState_Arena& operator=(const ModelState_Arena& state);
    ModelState_Arena* operator=(const ModelState_Arena* state);

    void updateStressInvariants();
    void updatePlasticStrainInvariants();

    friend std::ostream& operator<<(std::ostream& os, 
                                    const ModelState_Arena& state) {
      os << "\t ParticleID = " << state.particleID
         << " I1_eff = " << state.I1_eff << ", sqrt_J2 = " << state.sqrt_J2
         << ", r = " << state.rr << ", z_eff = " << state.zz_eff
         << ", evp = " << state.ep_v << ", p3 = " << state.p3 << "\n"
         << "\t K = " << state.bulkModulus << ", G = " << state.shearModulus << "\n"
         << "\t X = " << state.capX << ", kappa = " << state.kappa
         << ", pbar_w = " << state.pbar_w  << "\n"
         << "\t phi = " << state.porosity << ", Sw = " << state.saturation 
         << " ep_eq = " << state.ep_eq
         << " t_grow = " << state.t_grow << " coherence = " << state.coherence << std::endl;
      os << "\t Yield parameters: ";
      for (auto val : state.yieldParams) {
        os << "[ " << val.first << ", " << val.second << "], ";
      }
      os << std::endl;
      return os;
    }
    
  };

} // End namespace Uintah

#endif  // __MODEL_STATE_ARENISCA3_PARTIALLY_SATURATED_H__ 
