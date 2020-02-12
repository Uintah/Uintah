/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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


#include <CCA/Components/MPM/Materials/ConstitutiveModel/PlasticityModels/DamageModel.h>
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
     // Create datatype for storing damage parameters
    struct damageData {
      double D0;                    /*< Initial mean scalar damage */
      double D0_std;                /*< Initial standard deviation of scalar damage */
      double Dc;                    /*< Critical scalar damage */
      std::string dist;             /*< Initial damage distrinution */
    };

  private:
    
    damageData d_initialData;

    // Prevent copying of this class
    // copy constructor
    //HancockMacKenzieDamage(const HancockMacKenzieDamage &cm);
    HancockMacKenzieDamage& operator=(const HancockMacKenzieDamage &cm);
    
    const VarLabel* pDamageLabel;
    const VarLabel* pDamageLabel_preReloc;
    const VarLabel* pPlasticStrainRateLabel_preReloc;

  public:
    // constructors
    HancockMacKenzieDamage(ProblemSpecP& ps); 
    HancockMacKenzieDamage(const HancockMacKenzieDamage* cm);
         
    // destructor 
    virtual ~HancockMacKenzieDamage();

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
    void addParticleState(std::vector<const VarLabel*>& from,
                          std::vector<const VarLabel*>& to);

    virtual 
    void addInitialComputesAndRequires(Task* task,
                                       const MPMMaterial* matl );
                                               
    virtual
    void initializeLabels(const Patch*       patch,
                          const MPMMaterial* matl,
                          DataWarehouse*     new_dw ); 
  };

} // End namespace Uintah

#endif  // __HANCOCKMACKENZIE_DAMAGE_MODEL_H__ 
