/*
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
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

#ifndef __THRESHOLD_DAMAGE_MODEL_H__
#define __THRESHOLD_DAMAGE_MODEL_H__


#include "DamageModel.h"
#include <Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {


  class ThresholdDamage : public DamageModel {

  public:

  //______________________________________________________________________
  //
  private:

    struct FailureStressOrStrainData {
      double mean;         /* Mean failure stress, strain or cohesion */
      double std;          /* Standard deviation of failure strain */
                           /* or Weibull modulus */
      double exponent;     /* Exponent used in volume scaling of failure crit */
      double refVol;       /* Reference volume for scaling failure criteria */
      std::string scaling; /* Volume scaling method: "none" or "kayenta" */
      std::string dist;    /* Failure distro: "constant", "gauss" or "weibull"*/
      int seed;            /* seed for random number distribution generator */
      double t_char;       /* characteristic time for damage to occur */
    };

    bool d_useDamage  = false;
    FailureStressOrStrainData d_epsf;
    std::string d_failure_criteria; /* Options are:  "MaximumPrincipalStrain" */
                                    /* "MaximumPrincipalStress", "MohrColoumb"*/

    // MohrColoumb options
    double d_friction_angle;  // Assumed to come in degrees
    double d_tensile_cutoff;  // Fraction of the cohesion at which
                              // tensile failure occurs

    enum erosionAlgo { ZeroStress, AllowNoTension, AllowNoShear, none};
    erosionAlgo d_erosionAlgo = none;

    // Prevent copying of this class copy constructor
    ThresholdDamage& operator=(const ThresholdDamage &cm);


  //______________________________________________________________________
  //
  public:
    // constructors
    ThresholdDamage( ProblemSpecP& ps,
                     MPMFlags* Mflags  );

    ThresholdDamage(const ThresholdDamage* cm);

    // destructor
    virtual ~ThresholdDamage();

    virtual void outputProblemSpec(ProblemSpecP& ps);

    double initialize();

    bool hasFailed(double damage);

    //////////
    // Calculate the scalar damage parameter
    virtual
    double computeScalarDamage(const double& plasticStrainRate,
                               const Matrix3& stress,
                               const double& temperature,
                               const double& delT,
                               const MPMMaterial* matl,
                               const double& tolerance,
                               const double& damage_old);
    virtual
    void updateFailedParticlesAndModifyStress2(const Matrix3& FF,
                                               const double& pFailureStrain,
                                               const int& pLocalized,
                                               int& pLocalized_new,
                                               const double& pTimeOfLoc,
                                               double& pTimeOfLoc_new,
                                               Matrix3& pStress_new,
                                               const long64 particleID,
                                               double time);

  };

} // End namespace Uintah

#endif  // __Threshold_DAMAGE_MODEL_H__
