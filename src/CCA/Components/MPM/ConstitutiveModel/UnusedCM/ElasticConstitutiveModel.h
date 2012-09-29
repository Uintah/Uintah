/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

//  ElasticConstitutiveModel.h 
//  class ConstitutiveModel ConstitutiveModel data type -- 3D - 
//  holds ConstitutiveModel
//  information for the FLIP technique:
//    This is for elastic materials
//    Features:
//      Usage:



#ifndef __ELASTIC_CONSTITUTIVE_MODEL_H__
#define __ELASTIC_CONSTITUTIVE_MODEL_H__

#include "ConstitutiveModel.h"  
#include <CCA/Components/MPM/Util/Matrix3.h>
#include <vector>


namespace Uintah {
      class ElasticConstitutiveModel : public ConstitutiveModel {
      private:
         // data areas
         // Symmetric stress tensor (3 x 3 matrix)  
         Matrix3 stressTensor;
         // Symmetric strain tensor (3 x 3 matrix)
         Matrix3 strainTensor;
         // Symmetric strain increment tensor (3 x 3 matrix)
         Matrix3 strainIncrement;
         // Symmetric stress increment tensor (3 x 3 matrix)
         Matrix3 stressIncrement;
         // rotation increment (3 x 3 matrix)
         Matrix3 rotationIncrement;

      public:
         struct CMData {
            // ConstitutiveModel's properties
            // Young's Modulus
            double YngMod;
            // Poisson's Ratio
            double PoiRat;
         };
      private:
         friend const TypeDescription* fun_getTypeDescription(CMData*);
         
      public:
         // constructors
         ElasticConstitutiveModel(ProblemSpecP& ps);
         ElasticConstitutiveModel(double YM,double PR);
         
         // copy constructor
         ElasticConstitutiveModel(const ElasticConstitutiveModel &cm);
         
         // destructor 
         virtual ~ElasticConstitutiveModel();
         
         // assign the ElasticConstitutiveModel components 
         
         // assign the Symmetric stress tensor
         virtual void setStressTensor(Matrix3 st);
         // assign the strain increment tensor
         void  setDeformationMeasure(Matrix3 strain);
         // assign the strain increment tensor
         void  setStrainIncrement(Matrix3 si);
         // assign the stress increment tensor
         void  setStressIncrement(Matrix3 si);
         // assign the rotation increment tensor
         void  setRotationIncrement(Matrix3 ri);
         
         // access the Symmetric stress tensor
         virtual Matrix3 getStressTensor() const;
         // access the Symmetric strain tensor
         virtual Matrix3 getDeformationMeasure() const;
         // access the mechanical properties
         std::vector<double> getMechProps() const;
         
         // access the strain increment tensor
         Matrix3 getStrainIncrement() const;
         // access the stress increment tensor
         Matrix3 getStressIncrement() const;
         // access the rotation increment tensor
         Matrix3 getRotationIncrement() const;
         
         // Return the Lame constants - used for computing sound speeds
         double getLambda() const;
         double getMu() const;
         
         
         // Compute the various quantities of interest
         
         Matrix3 deformationIncrement(double time_step);
         void computeStressIncrement();
         void computeRotationIncrement(Matrix3 defInc);
         //////////
         // Basic constitutive model calculations
         virtual void computeStressTensor(const PatchSubset* patches,
                                          const MPMMaterial* matl,
                                          DataWarehouse* new_dw,
                                          DataWarehouse* old_dw);
         
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
         
         ConstitutiveModel & operator=(const ElasticConstitutiveModel &cm);
         
         virtual int getSize() const;

         CMData d_initialData;
         const VarLabel* p_cmdata_label;
         const VarLabel* p_cmdata_label_preReloc;
      };
      
} // End namespace Uintah
      


#endif  // __ELASTIC_CONSTITUTIVE_MODEL_H__ 

