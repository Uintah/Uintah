/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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

//  ViscoScramImplicit.h 
//  class ConstitutiveModel ConstitutiveModel data type -- 3D - 
//  holds ConstitutiveModel
//  information for the MPM technique:
//    This is for ViscoScram
//    Features:
//      This model is, in actuality, mostly just a holding place
//      for the ViscoScram variables needed in the explosion phase
//      of the calculation
//      Usage:

#ifndef __VISCOSCRAM_IMPLICIT_CONSTITUTIVE_MODEL_H__
#define __VISCOSCRAM_IMPLICIT_CONSTITUTIVE_MODEL_H__



#include "ViscoScram.h"
#include "ConstitutiveModel.h"        
#include "ImplicitCM.h"
#include <Core/Disclosure/TypeDescription.h>
#include <CCA/Components/MPM/Solver.h>
#include <vector>



namespace Uintah {
      class ViscoScramImplicit : public ConstitutiveModel, public ImplicitCM {
      public:

         struct CMData {
           double PR;
           double CoefThermExp;
           double CrackParameterA;
           double CrackPowerValue;
           double CrackMaxGrowthRate;
           double StressIntensityF;
           double CrackFriction;
           double InitialCrackRadius;
           double CrackGrowthRate;
           double G[5];
           double RTau[5];
           double Beta, Gamma;
           double DCp_DTemperature;
           int LoadCurveNumber, NumberOfPoints;
         };

         struct TimeTemperatureData {
           double T0_WLF;
           double C1_WLF;
           double C2_WLF;
         };

         typedef ViscoScramStateData StateData;

         const VarLabel* pVolChangeHeatRateLabel;
         const VarLabel* pViscousHeatRateLabel;
         const VarLabel* pCrackHeatRateLabel;
         const VarLabel* pCrackRadiusLabel;
         const VarLabel* pStatedataLabel;
         const VarLabel* pRandLabel;
         const VarLabel* pStrainRateLabel;
         const VarLabel* pVolChangeHeatRateLabel_preReloc;
         const VarLabel* pViscousHeatRateLabel_preReloc;
         const VarLabel* pCrackHeatRateLabel_preReloc;
         const VarLabel* pCrackRadiusLabel_preReloc;
         const VarLabel* pStatedataLabel_preReloc;
         const VarLabel* pRandLabel_preReloc;
         const VarLabel* pStrainRateLabel_preReloc;

         protected:
                                                                                
           // Create datatype for storing model parameters
           bool d_useModifiedEOS;
           bool d_random;
           bool d_doTimeTemperature;
           bool d_useObjectiveRate;
           double d_bulk;
           double d_G;

      private:
         CMData d_initialData;
         TimeTemperatureData d_tt;

         // Prevent copying of this class
         // copy constructor
         //ViscoScramImplicit(const ViscoScramImplicit &cm);
         ViscoScramImplicit& operator=(const ViscoScramImplicit &cm);

      public:
         // constructors
         ViscoScramImplicit(ProblemSpecP& ps, MPMFlags* flag);
         ViscoScramImplicit(const ViscoScramImplicit* cm);
       
         // destructor
         virtual ~ViscoScramImplicit();

         virtual void outputProblemSpec(ProblemSpecP& ps,
                                        bool output_cm_tag = true);

         // clone
         ViscoScramImplicit* clone();

         // compute stable timestep for this patch
         virtual void computeStableTimestep(const Patch* patch,
                                            const MPMMaterial* matl,
                                            DataWarehouse* new_dw);

         virtual void computeStressTensorImplicit(const PatchSubset* patches,
                                                  const MPMMaterial* matl,
                                                  DataWarehouse* old_dw,
                                                  DataWarehouse* new_dw,
                                                  Solver* solver,
                                                  const bool recursion);

         virtual void computeStressTensorImplicit(const PatchSubset* patches,
                                                  const MPMMaterial* matl,
                                                  DataWarehouse* old_dw,
                                                  DataWarehouse* new_dw);

         // initialize  each particle's constitutive model data
         virtual void initializeCMData(const Patch* patch,
                                       const MPMMaterial* matl,
                                       DataWarehouse* new_dw);


         virtual void allocateCMDataAddRequires(Task* task, 
                                                const MPMMaterial* matl,
                                                const PatchSet* patch, 
                                                MPMLabel* lb) const;


         virtual void allocateCMDataAdd(DataWarehouse* new_dw,
                                        ParticleSubset* subset,
                                        map<const VarLabel*,
                                        ParticleVariableBase*>* newState,
                                        ParticleSubset* delset,
                                        DataWarehouse* old_dw);


         virtual void addInitialComputesAndRequires(Task* task,
                                             const MPMMaterial* matl,
                                             const PatchSet* patches) const;

         virtual void addComputesAndRequires(Task* task,
                                             const MPMMaterial* matl,
                                             const PatchSet* patches,
                                             const bool recursion) const;

         virtual void addComputesAndRequires(Task* task,
                                             const MPMMaterial* matl,
                                             const PatchSet* patches) const;


         virtual double computeRhoMicroCM(double pressure,
                                          const double p_ref,
                                          const MPMMaterial* matl,
                                          double temperature,
                                          double rho_guess);

         virtual void computePressEOSCM(double rho_m, double& press_eos,
                                        double p_ref,
                                        double& dp_drho, double& ss_new,
                                        const MPMMaterial* matl,
                                        double temperature);

         virtual double getCompressibility();


         virtual void addParticleState(std::vector<const VarLabel*>& from,
                                       std::vector<const VarLabel*>& to);

      };
} // End namespace Uintah
      


#endif  // __VISCOSCRAM_IMPLICIT_CONSTITUTIVE_MODEL_H__ 

