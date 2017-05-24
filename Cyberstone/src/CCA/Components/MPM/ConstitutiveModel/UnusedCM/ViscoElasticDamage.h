/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

//  ViscoElasticDamage.h 
//  class ConstitutiveModel ConstitutiveModel data type -- 3D - 
//  holds ConstitutiveModel
//  information for the FLIP technique:
//    This constitutive model is based on theorem of Continuum Damage 
//    Mechanics (CDM) with following assumptions:
//      i.   Standard solid type of viscoelastic behavior with linear rate
//           equation.
//      ii.  Free energy equation is uncoupled as volumetric and deviatoric
//           parts and the equation is the same as Neo-Hookean model (refer:
//           CompNeoHook.h and CompNeoHook.cc).
//      iii. The damage mechanism is associated with maximum distortional
//           energy and is independent of hydrostatic pressure.
//      iv.  The damage criterion is only function of equivalent strain.
//      v.   The damage process character function has exponential form.
//    Material property constants:
//      Elasticity: Young's modulus, Poisson's ratio;
//      Damage parameters: alpha[0, infinity), beta[0, 1].
//      Visco properties: tau[0,infinity) -- relaxation time,
//                        gamma[0,1) -- stiffness ratio.
//      Maximum equivalent strain: strainmax -- there will be no damage
//                                 when strain is less than this value.
//  Reference: "ON A FULLY THREE-DIMENSIONAL FINITE-STRAIN VISCOELASTIC
//              DAMAGE MODEL: FORMULATION AND COMPUTATIONAL ASPECTS", by
//              J.C.Simo, Computer Methods in Applied Mechanics and
//              Engineering 60 (1987) 153-173.
//              "COMPUTATIONAL INELASTICITY" by J.C.Simo & T.J.R.Hughes,
//              Springer, 1997.
//    Features:
//      Usage:



#ifndef __VISCOELASTIC_DAMAGE_CONSTITUTIVE_MODEL_H__
#define __VISCOELASTIC_DAMAGE_CONSTITUTIVE_MODEL_H__


#include <cmath>
#include "ConstitutiveModel.h"  
#include <vector>
#include <CCA/Components/MPM/Util/Matrix3.h>


namespace Uintah {
      class ViscoElasticDamage : public ConstitutiveModel {
      private:
         // data areas
         // deformation gradient tensor (3 x 3 Matrix)
         Matrix3 deformationGradient;
         // Deviatoric-Elastic Part of the left Cauchy-Green Tensor (3 x 3 Matrix)
         // or CeBar -- a defferent notation
         Matrix3 bElBar;
         // Deviatoric-Elastic part of the Lagrangian Strain Tensor 
         Matrix3 E_bar, current_E_bar;
         // symmetric stress tensor (3 x 3 Matrix)  
         Matrix3 stressTensor;
         // Internal functions
         Matrix3 func_PI_n, func_PI_nn; // function PI at n and n+1
         Matrix3 func_Hbar_n, func_Hbar_nn;     // function Hbar at n and n+1
         
         // ConstitutiveModel's properties
         // ViscoElasticDamage Constants
         double d_Bulk,d_Shear;
         // Damage parameters
         double d_Alpha, d_Beta;
         // Viscoelastic constants
         double d_Tau, d_Gamma;
         // Damage Character 
         double damageG;
         // Maximum equivalent strain
         double maxEquivStrain;
         
      public:
         // constructors
         ViscoElasticDamage(ProblemSpecP& ps);
         ViscoElasticDamage(double bulk,double shear,double alpha,double beta,
                            double tau, double gamma, double strainmax);
         
         // copy constructor
         ViscoElasticDamage(const ViscoElasticDamage &cm);
         
         // destructor 
         virtual ~ViscoElasticDamage();
         
         // assign the ViscoElasticDamage components 
         
         // set Bulk Modulus
         void setBulk(double bulk);
         // set Shear Modulus
         void setShear(double shear);
         // set Damage Parameters
         void setDamageParameters(double alpha, double beta);
         // set maximum equivalent strain
         void setMaxEquivStrain(double strainmax);
         // set Viscoelastic Parameters
         void setViscoelasticParameters(double tau, double gamma);
         // assign the deformation gradient tensor
         virtual void setDeformationMeasure(Matrix3 dg);
         // assign the symmetric stress tensor
         virtual void setStressTensor(Matrix3 st);
         
         
         // access components of the ViscoElasticDamage model
         // access the symmetric stress tensor
         virtual Matrix3 getStressTensor() const;
         
         virtual Matrix3 getDeformationMeasure() const;
         // access the mechanical properties
         virtual std::vector<double> getMechProps() const;
         
         
         
         
         // Compute the various quantities of interest
         
         void computeDeformationGradient();
         //////////
         // Basic constitutive model calculations
         virtual void computeStressTensor(const PatchSubset* patches,
                                          const MPMMaterial* matl,
                                          DataWarehouse* old_dw,
                                          DataWarehouse* new_dw);
         
         //////////
         // Computation of strain energy.  Useful for tracking energy balance.
         virtual double computeStrainEnergy(const Patch* patch,
                                            const MPMMaterial* matl,
                                            DataWarehouse* new_dw);
         
         // initialize  each particle's constitutive model data
         virtual void initializeCMData(const Patch* patch,
                                       const MPMMaterial* matl,
                                       DataWarehouse* new_dw);      
         
         virtual void addComputesAndRequires(Task* task,
                                             const MPMMaterial* matl,
                                             const PatchSet* patches) const;

         virtual void addParticleState(std::vector<const VarLabel*>& from,
                                       std::vector<const VarLabel*>& to);

         // Return the Lame constants
         virtual double getMu() const;
         virtual double getLambda() const;
         
         // class function to read correct number of parameters
         // from the input file
         static void readParameters(ProblemSpecP ps, double *p_array);
         
         // class function to write correct number of parameters
         // to the output file
         static void writeParameters(std::ofstream& out, double *p_array);
         
         // class function to read correct number of parameters
         // from the input file, and create a new object
         static ConstitutiveModel* readParametersAndCreate(ProblemSpecP ps);
         
         // member function to write correct number of parameters
         // to output file, and to write any other particle information
         // needed to restart the model for this particle
         virtual void writeRestartParameters(std::ofstream& out) const;
         
         // member function to read correct number of parameters
         // from the input file, and any other particle information
         // need to restart the model for this particle 
         // and create a new object
         static ConstitutiveModel* readRestartParametersAndCreate(ProblemSpecP ps);
         
         // class function to create a new object from parameters
         static ConstitutiveModel* create(double *p_array);
         
         // member function to determine the model type.
         virtual int getType() const;
         // member function to get model's name
         virtual std::string getName() const;
         // member function to get number of parameters for model
         virtual int getNumParameters() const;
         // member function to print parameter names for model
         virtual void printParameterNames(std::ofstream& out) const;
         
         // member function to make a duplicate
         virtual ConstitutiveModel* copy() const;
         
         virtual int getSize() const;
      };
      
} // End namespace Uintah
      


#endif // __VISCOELASTIC_DAMAGE_CONSTITUTIVE_MODEL_H__ 

