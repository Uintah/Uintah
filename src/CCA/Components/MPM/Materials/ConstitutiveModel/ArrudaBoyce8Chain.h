/*
 * ArrudaBoyce8Chain.h
 *
 *  Created on: May 25, 2018
 *      Author: jbhooper
 */

#ifndef SRC_CCA_COMPONENTS_MPM_MATERIALS_CONSTITUTIVEMODEL_ARRUDABOYCE8CHAIN_H_
#define SRC_CCA_COMPONENTS_MPM_MATERIALS_CONSTITUTIVEMODEL_ARRUDABOYCE8CHAIN_H_

#include <CCA/Components/MPM/Materials/ConstitutiveModel/ConstitutiveModel.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/ImplicitCM.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>
#include <CCA/Components/MPM/Core/MPMLabel.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Variables/VarTypes.h>

#include <cmath>

namespace Uintah {

  class ArrudaBoyce8Chain : public ConstitutiveModel {
      // Variables

    public:
      ArrudaBoyce8Chain(ProblemSpecP  & ps
                       ,MPMFlags      * mFlags);

      virtual void outputProblemSpec(ProblemSpecP & ps
                                    ,bool           output_cm_tag = true);

      ArrudaBoyce8Chain* clone();

      virtual ~ArrudaBoyce8Chain();

      // Initialization (and carryForward for RigidMPM where there's nothing to
      //   do here.
      virtual void carryForward(const PatchSubset   * patches
                               ,const MPMMaterial   * matl
                               ,      DataWarehouse * old_dw
                               ,      DataWarehouse * new_dw  );

      virtual void initializeCMData(const Patch         * patch
                                   ,const MPMMaterial   * matl
                                   ,      DataWarehouse * new_dw);

      // Scheduling functions
      virtual void addComputesAndRequires(      Task        * task
                                         ,const MPMMaterial * matl
                                         ,const PatchSet    * patches ) const;

      virtual void addComputesAndRequires(      Task        * task
                                         ,const MPMMaterial * matl
                                         ,const PatchSet    * patches
                                         ,const bool          recursion
                                         ,const bool          schedPar  = true) const;

      virtual void addInitialComputesAndRequires(       Task        * task
                                                ,const  MPMMaterial * matl
                                                ,const  PatchSet    * patches );

      //Computation functions
      // Compute pressure from constitutive model's EOS
      virtual void computePressEOSCM(       double        rho_m
                                    ,       double      & press_eos
                                    ,       double        p_ref
                                    ,       double      & dp_rho
                                    ,       double      & ss_new
                                    ,const  MPMMaterial * matl
                                    ,       double        temperature);

      // Compte density from constitutive model's EOS
      virtual double computeRhoMicroCM(       double        pressure
                                      ,const  double        p_ref
                                      ,const  MPMMaterial * matl
                                      ,       double        temperature
                                      ,       double        rho_guess
                                      );

      virtual void computeStableTimeStep(const  Patch         * patch
                                        ,const  MPMMaterial   * matl
                                        ,       DataWarehouse * new_dw);

      virtual void computeStressTensor(const  PatchSubset   * patches
                                      ,const  MPMMaterial   * matl
                                      ,       DataWarehouse * old_dw
                                      ,       DataWarehouse * new_dw);

      virtual void computeStressTensorImplicit(const  PatchSubset   * patches
                                              ,const  MPMMaterial   * matl
                                              ,       DataWarehouse * old_dw
                                              ,       DataWarehouse * new_dw  );

      // Helper Functions
      //   Add particle variables to tracker
      virtual void addParticleState(std::vector<const VarLabel*>  & from
                                   ,std::vector<const VarLabel*>  & to    );

      // Returns compressibility of the material
      virtual double getCompressibility();

      virtual void addSplitParticlesComputesAndRequires(      Task        * task
                                                       ,const MPMMaterial * matl
                                                       ,const PatchSet    * patches );

      virtual void splitCMSpecificParticleData(const  Patch                 * patch
                                              ,const  int                     dwi
                                              ,const  int                     fourOrEight
                                              ,       ParticleVariable<int> & pRefOld
                                              ,       ParticleVariable<int> & pRefNew
                                              ,const  unsigned int            oldNumPar
                                              ,const  unsigned int            numNewPartNeeded
                                              ,       DataWarehouse         * old_dw
                                              ,       DataWarehouse         * new_dw            );

    private:
      ArrudaBoyce8Chain& operator=(const ArrudaBoyce8Chain & ab8_cm);

      double m_bulkIn;  // Input bulk modulus
      double m_shearIn; // Input shear modulus
      double m_betaIn;  // Input beta parameter (sqrt(N))
      bool   m_useModifiedEOS;
      int    m_8or27;

  };
}




#endif /* SRC_CCA_COMPONENTS_MPM_MATERIALS_CONSTITUTIVEMODEL_ARRUDABOYCE8CHAIN_H_ */
