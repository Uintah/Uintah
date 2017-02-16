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

#ifndef __BRITTLE_DAMAGE_MODEL_H__
#define __BRITTLE_DAMAGE_MODEL_H__


#include "DamageModel.h"        
#include <Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {


  class BrittleDamage : public DamageModel {

  public:

  private:
    //Create datatype for brittle damage
    struct BrittleDamageData {
      double r0b;            /* Initial energy threshold (\sqrt{Pa}) */
      double Gf;             /* Fracture energy (J/m^3) */
      double constant_D;     /* Shape factor in softening function */
      double maxDamageInc;   /* Maximum damage increment in a time step */
      bool   allowRecovery;  /* Recovery of stiffness allowed */
      double recoveryCoeff;  /* Fraction of stiffness to be recovered */
      bool   printDamage;    /* Flag to print damage */
      double Bulk;
      double tauDev;
    };
    
    BrittleDamageData d_brittle_damage;
    
    
    // Prevent copying of this class copy constructor
    BrittleDamage& operator=(const BrittleDamage &cm);

  public:
    // constructors
    BrittleDamage( ProblemSpecP& ps );
                   
    BrittleDamage(const BrittleDamage* cm);
         
    // destructor 
    virtual ~BrittleDamage();

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
    void updateDamageAndModifyStress2(const Matrix3& FF,
                                      const double&  pFailureStrain,
                                      double&        pFailureStrain_new,
                                      const double&  pVolume,
                                      const double&  pDamage,
                                      double&        pDamage_new,
                                      Matrix3&       pStress_new,
                                      const long64   particleID);
  
  };

} // End namespace Uintah

#endif  // __Brittle_DAMAGE_MODEL_H__ 
