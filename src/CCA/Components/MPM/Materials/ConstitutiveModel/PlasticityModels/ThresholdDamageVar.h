/*
 * The MIT License
 *
 * Copyright (c) 1997-2023 The University of Utah
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

#ifndef __THRESHOLD_DAMAGE_VAR_MODEL_H__
#define __THRESHOLD_DAMAGE_VAR_MODEL_H__


#include <CCA/Components/MPM/Materials/ConstitutiveModel/PlasticityModels/DamageModel.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/MaterialManagerP.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  class ThresholdDamageVar : public DamageModel {

  public:

  //______________________________________________________________________
  //
  private:

    struct MeanAndStd{
      double mean;         /* Mean failure stress, strain or cohesion */
      double std;          /* Standard deviation of failure strain */
    };

    struct FailureStressOrStrainData {
      double mean;         /* Mean failure stress, strain or cohesion */
      double std;          /* Standard deviation of failure strain */
                           /* or Weibull modulus */
      double exponent;     /* Exponent used in volume scaling of failure crit */
      double refVol;       /* Reference volume for scaling failure criteria */
      std::string scaling; /* Volume scaling method: "none" or "kayenta" */
      std::string dist;    /* Failure distro: "constant", "gauss" or "weibull"*/
      int seed;            /* seed for random number distribution generator */
      bool localizeOrNot;
      
      void print(){
        std::cout << " mean:" << mean << " std: " << std << " exponent: " << exponent
                  << " refVol: " << refVol << " scaling: " << scaling << " dist: " << dist
                  << " seed: " << seed << "/n";
      }
    };

    FailureStressOrStrainData d_epsf;
    std::string d_failure_criteria; /* Options are:  "MaximumPrincipalStrain" */
                                    /* "MaximumPrincipalStress", "MohrColoumb"*/

    // MohrColoumb options
    double d_friction_angle;           // Assumed to come in degrees
    double d_tensile_cutoff;           // Fraction of the cohesion at which
                                       // tensile failure occurs
    //__________________________________
    //  Labels
    const VarLabel* pFailureStressOrStrainLabel;
    const VarLabel* pFailureStressOrStrainLabel_preReloc;

    // Prevent copying of this class copy constructor
    ThresholdDamageVar& operator=(const ThresholdDamageVar &cm);

    // Private Data
    // Load curve information 
    std::vector<double> d_Color;
    std::vector<double> d_mean;
    std::vector<double> d_std;

  //______________________________________________________________________
  //
  public:
    // constructors
    ThresholdDamageVar( ProblemSpecP    & ps,
                     MPMFlags        * Mflags,
                     MaterialManager * materialManager );

    ThresholdDamageVar(const ThresholdDamageVar* cm);

    // destructor
    virtual ~ThresholdDamageVar();

    virtual void outputProblemSpec(ProblemSpecP& ps);

    virtual
    void addComputesAndRequires(Task* task,
                                const MPMMaterial* matl);

    virtual
    void  computeSomething( ParticleSubset    * pset,
                            const MPMMaterial * matl,            
                            const Patch       * patch,          
                            DataWarehouse     * old_dw,         
                            DataWarehouse     * new_dw );       

    virtual
    void carryForward( const PatchSubset* patches,
                       const MPMMaterial* matl,
                       DataWarehouse*     old_dw,
                       DataWarehouse*     new_dw);

    virtual
    void addParticleState(std::vector<const VarLabel*>& from,
                          std::vector<const VarLabel*>& to);

    virtual 
    void addInitialComputesAndRequires(Task* task,
                                       const MPMMaterial* matl);

    virtual
    void initializeLabels(const Patch*       patch,
                          const MPMMaterial* matl,
                          DataWarehouse*     new_dw);

     inline MeanAndStd findPropertiesFromColor(double color){

       MeanAndStd props;
       int n_entries = static_cast<int>(d_Color.size());
       if (color >= d_Color[n_entries-1]){
          props.mean    = d_mean[n_entries-1];
          props.std     = d_std[n_entries-1];
          return props;
       }

       for (int ii = 1; ii < n_entries; ++ii) {
        if (color <= d_Color[ii]) {
          double s = (d_Color[ii]-color)/(d_Color[ii]-d_Color[ii-1]);
          props.mean    = d_mean[ii-1]*s + d_mean[ii]*(1.0-s);
          props.std     = d_std[ii-1]*s + d_std[ii]*(1.0-s);
          return props;
         }
       }

       props.mean       = d_mean[0];
       props.std     = d_std[0];
       return props;
     }
  };

} // End namespace Uintah

#endif  // __THRESHOLD_DAMAGE_VAR_MODEL_H__
