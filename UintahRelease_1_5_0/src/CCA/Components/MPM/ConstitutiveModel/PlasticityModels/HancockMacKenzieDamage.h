/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

#ifndef __HANCOCKMACKENZIE_DAMAGE_MODEL_H__
#define __HANCOCKMACKENZIE_DAMAGE_MODEL_H__


#include "DamageModel.h"        
#include <Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  /////////////////////////////////////////////////////////////////////////////
  /*!
    \class HancockMacKenzieDamage
    \brief HancockMacKenzie Damage Model
    \author Biswajit Banerjee \n
    C-SAFE and Department of Mechanical Engineering \n
    University of Utah \n

    References:
    1) Hancock and MacKenzie, 1976, Int. J. Mech. Phys. Solids, 24, 147-169.

    The damage evolution rule is given by \n
    \f$
    \dot{D} = (1/1.65)\dot{\epsilon_p}\exp(3\sigma_h/2\sigma_{eq})
    \f$ \n
    where \n
    \f$ D \f$ = damage variable \n
    where \f$ D = 0 \f$ for virgin material, \n
    \f$ \epsilon_p \f$ is the plastic strain, \n
    \f$ \sigma_h = (1/3) Tr(\sigma) \f$ \n
    \f$ \sigma_{eq} = \sqrt{(3/2) \sigma_{dev}:\sigma_{dev}} \f$
  */
  /////////////////////////////////////////////////////////////////////////////

  class HancockMacKenzieDamage : public DamageModel {

  public:
    // Create datatype for storing model parameters
    struct CMData {
      double D0; /*< Initial damage */
      double Dc; /*< Critical damage */
    };   

  private:

    CMData d_initialData;
         
    // Prevent copying of this class
    // copy constructor
    //HancockMacKenzieDamage(const HancockMacKenzieDamage &cm);
    HancockMacKenzieDamage& operator=(const HancockMacKenzieDamage &cm);

  public:
    // constructors
    HancockMacKenzieDamage(ProblemSpecP& ps); 
    HancockMacKenzieDamage(const HancockMacKenzieDamage* cm);
         
    // destructor 
    virtual ~HancockMacKenzieDamage();

    virtual void outputProblemSpec(ProblemSpecP& ps);
         
    //////////////////////////////////////////////////////////////////////////
    /*! 
      Initialize the damage parameter in the calling function
    */
    //////////////////////////////////////////////////////////////////////////
    double initialize();

    //////////////////////////////////////////////////////////////////////////
    /*! 
      Determine if damage has crossed cut off
    */
    //////////////////////////////////////////////////////////////////////////
    bool hasFailed(double damage);
    
    //////////
    // Calculate the scalar damage parameter 
    virtual double computeScalarDamage(const double& plasticStrainRate,
                                       const Matrix3& stress,
                                       const double& temperature,
                                       const double& delT,
                                       const MPMMaterial* matl,
                                       const double& tolerance,
                                       const double& damage_old);
  };

} // End namespace Uintah

#endif  // __HANCOCKMACKENZIE_DAMAGE_MODEL_H__ 
