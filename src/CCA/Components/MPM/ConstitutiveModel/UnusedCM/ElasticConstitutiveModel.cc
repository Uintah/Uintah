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

//  ElasticConstitutiveModel.cc 
//  class ConstitutiveModel ConstitutiveModel data type -- 3D - 
//  holds ConstitutiveModel
//  information for the FLIP technique:
//    This is for elastic materials
//    Features:
//      Usage:


#include "ConstitutiveModelFactory.h"
#include "ElasticConstitutiveModel.h"
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Components/MPM/MPMLabel.h>

#include <CCA/Components/MPM/Util/Matrix.cc> // for bounded array multiplier    
#include <fstream>
#include <iostream>
using namespace std;
using namespace Uintah;

ElasticConstitutiveModel::ElasticConstitutiveModel(ProblemSpecP &ps)
{
  ps->require("youngs_modulus",d_initialData.YngMod);
  ps->require("poissons_ratio",d_initialData.PoiRat); 
  p_cmdata_label = VarLabel::create("p.cmdata",
                                ParticleVariable<CMData>::getTypeDescription());
  p_cmdata_label_preReloc = VarLabel::create("p.cmdata+",
                                ParticleVariable<CMData>::getTypeDescription());

}

void ElasticConstitutiveModel::addParticleState(std::vector<const VarLabel*>& from,
                                                std::vector<const VarLabel*>& to)
{
   from.push_back(p_cmdata_label);
   to.push_back(p_cmdata_label_preReloc);
}

ElasticConstitutiveModel::~ElasticConstitutiveModel()
{
  // Destructor

  //cout << "Calling ElasticConstitutiveModel destructor . . . " << endl;
  VarLabel::destroy(p_cmdata_label);
  VarLabel::destroy(p_cmdata_label_preReloc);
 
}



void ElasticConstitutiveModel::setStressTensor(Matrix3 st) 
{
  // Assign the stress tensor (3 x 3 matrix)

  stressTensor = st;

}

void ElasticConstitutiveModel::setDeformationMeasure(Matrix3 st) 
{
  // Assign the strain tensor (3 x 3 matrix)

  strainTensor = st;

}

void ElasticConstitutiveModel::setStrainIncrement(Matrix3 si) 
{
  // Assign the strain increment tensor (3 x 3 matrix)

  strainIncrement = si;

}

void ElasticConstitutiveModel::setStressIncrement(Matrix3 si) 
{
  // Assign the stress increment tensor (3 x 3 matrix)

  stressIncrement = si;

}

void ElasticConstitutiveModel::setRotationIncrement(Matrix3 ri) 
{
  // Assign the rotation increment tensor (3 x 3 matrix)

  rotationIncrement = ri;

}

Matrix3 ElasticConstitutiveModel::getStressTensor() const
{
  // Return the stress tensor (3 x 3 matrix)

  return stressTensor;

}

Matrix3 ElasticConstitutiveModel::getDeformationMeasure() const
{
  // Return the strain tensor (3 x 3 matrix)

  return strainTensor;

}


#if 0
std::vector<double> ElasticConstitutiveModel::getMechProps() const
{
  // Return Young's Mod and Poisson's ratio

  std::vector<double> props(2);

  props[0] = YngMod;
  props[1] = PoiRat;

  return props;

}
#endif

Matrix3 ElasticConstitutiveModel::getStrainIncrement() const
{
  // Return the strain increment tensor (3 x 3 matrix)

  return strainIncrement;

}

Matrix3 ElasticConstitutiveModel::getStressIncrement() const
{
  // Return the stress increment tensor (3 x 3 matrix)

  return stressIncrement;

}

Matrix3 ElasticConstitutiveModel::getRotationIncrement() const
{
  // Return the rotation increment tensor (3 x 3 matrix)

  return rotationIncrement;

}

#if 0
double ElasticConstitutiveModel::getLambda() const
{
  // Return the Lame constant lambda

  double lambda = (YngMod * PoiRat)/((1. + PoiRat) *(1. - 2.* PoiRat));

  return lambda;
}

double ElasticConstitutiveModel::getMu() const
{
  // Return the Lame constant Mu

  double mu = YngMod/(2.*(1. + PoiRat));

  return mu;
}
#endif

void ElasticConstitutiveModel::computeRotationIncrement(Matrix3 defInc)
{
  // Compute the rotation increment following Sulsky in CFDLIB

  // This is hocus pocus

 
  BoundedArray<double> alpha(1,3,0.);
  
  alpha[1] = defInc(2,3) - defInc(3,2);
  alpha[2] = defInc(3,1) - defInc(1,3);
  alpha[3] = defInc(1,2) - defInc(2,1);

  double q = alpha * alpha;
  q = q/4.;
  double tracea = 2.0 - defInc.Trace();
  double p = (tracea * tracea)/4.;
  double fn = 1. - p - q;
  double fd = p + q;
  double tempc = p * (1. + fn*p/(fd*fd)*(3.-2.*p/fd));
  tempc = sqrt(tempc);
  if(tracea>=0.0){
                tempc = fabs(tempc);
  }
  else {
                tempc = -fabs(tempc);
  }

  //  Get sin(thetaa)/2/sqrt(q) 
  double temps=sqrt((p*q*(3.0-q)+pow(p,3.0)+pow(q,2.0))/pow((p+q),3.0))/2.0;

  //  Get (1-cos(thetaa))/4/q  

  double tempc2;

  if(q > 0.01){
     tempc2=(1.0-tempc)/4.0/q;
  }
  else{
      // I am sure that this can be simplified if it ever shows up
      // on a profile.
      // - Steve
     tempc2 =1.0/8.0+q/32.0/pow(p,2.0)*(pow(p,2.0)-12.0*p+12.0) 
      +pow(q,2.0)/64.0/pow(p,3.0)*(p-2.0)*(pow(p,2.0)-10.0*p+32.0)
      +pow(q,3.0)/512.0/pow(p,4.0)*(1104.0-992.0*p+376.0*pow(p,2.0)
                                    -72.0*pow(p,3.0)+5.0*pow(p,4.0));
  }

 
  rotationIncrement(1,1) = tempc+pow(alpha[1],2.0)*tempc2;
  rotationIncrement(1,2) = tempc2*alpha[1]*alpha[2]+temps*alpha[3];
  rotationIncrement(1,3) = tempc2*alpha[1]*alpha[3]-temps*alpha[2];

  rotationIncrement(2,1) = tempc2*alpha[1]*alpha[2]-temps*alpha[3];
  rotationIncrement(2,2) = tempc+pow(alpha[2],2.0)*tempc2;
  rotationIncrement(2,3) = tempc2*alpha[2]*alpha[3]+temps*alpha[1];

  rotationIncrement(3,1) = tempc2*alpha[1]*alpha[3]+temps*alpha[2];
  rotationIncrement(3,2) = tempc2*alpha[2]*alpha[3]-temps*alpha[1];
  rotationIncrement(3,3) = tempc+pow(alpha[3],2.0)*tempc2;

}

#if 0
void ElasticConstitutiveModel::computeStressIncrement()
{
  // Computes the stress increment given the strain increment
  // Youngs Modulus and Poisson's Ratio

  Matrix3 tempStrainIncrement(0.0);

  double shear = YngMod/2./(1. + PoiRat);
  double bulk = YngMod/3./(1. - 2.*PoiRat);

  // Volumetric strain

  double dekk = strainIncrement.Trace()/3.;

  tempStrainIncrement = strainIncrement;

  for (int i = 1; i<=3;i++) {
    tempStrainIncrement(i,i) -= dekk;
  }

  // Volumetric stress

  double volstress = bulk * 3. * dekk;
  
  // Total stress increment is the sum of the deviatoric and 
  // volumetric stresses.
 
  stressIncrement = tempStrainIncrement * (2. * shear);

  for (int i = 1; i<=3; i++) {
    stressIncrement(i,i) += volstress;
  }

}
#endif

void ElasticConstitutiveModel::computeStressTensor(const PatchSubset* /*patches*/,
                                                   const MPMMaterial* /*matl*/,
                                                   DataWarehouse* /*new_dw*/,
                                                   DataWarehouse* /*old_dw*/)
{
  cerr << "computeStressTensor not finished\n";
}

double ElasticConstitutiveModel::computeStrainEnergy(const Patch* /*patch*/,
                                                     const MPMMaterial* /*matl*/,
                                                     DataWarehouse* /*new_dw*/)
{
  cerr << "computeStrainEnergy not finished\n";
  return -1;
}

void ElasticConstitutiveModel::initializeCMData(const Patch* patch,
                                                const MPMMaterial* matl,
                                                DataWarehouse* new_dw)
{
  //   const MPMLabel* lb = MPMLabel::getLabels();
   ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);
   ParticleVariable<CMData> cmdata;
   new_dw->allocateAndPut(cmdata, p_cmdata_label, pset);
   for(ParticleSubset::iterator iter = pset->begin();
       iter != pset->end(); iter++)
      cmdata[*iter] = d_initialData;
   // allocateAndPut instead:
   /* new_dw->put(cmdata, p_cmdata_label); */;
}

#ifdef WONT_COMPILE_YET
double ElasticConstitutiveModel::computeStrainEnergy()
{
  double se;

  se = ((1.0/(2.0*YngMod))*
       (pow(stressTensor(1,1),2.0)+pow(stressTensor(2,2),2.0)+
                                         pow(stressTensor(3,3),2.0) -
       2.0*PoiRat*(stressTensor(1,1)*stressTensor(2,2)+
       stressTensor(2,2)*stressTensor(3,3)+stressTensor(3,3)*stressTensor(1,1)))+
       (1.0/(2.0*YngMod/(2.*(1. + PoiRat))))*
       (pow(stressTensor(1,2),2.0)+pow(stressTensor(2,3),2.0)+
                                         pow(stressTensor(1,3),2.0)));


  return se;
}

void ElasticConstitutiveModel::computeStressTensor
                        (Matrix3 velocityGradient, double time_step)
{

  // Calculate the stress Tensor (Symmetric 3 x 3 matrix) given the
  // time step and the velocity gradient and the material constants
  // Poisson's Ratio and Young's Modulus

  
  // Compute the deformation gradient using the time_step and velocity 
  // gradient
  Matrix3 defInc(0.0);
  defInc = velocityGradient * time_step;

  // Compute the strain Increment given the deformation Increment
  strainIncrement = (defInc + defInc.Transpose())*.5;

  // Update the strain tensor with the strain increment
  strainTensor = strainTensor +  strainIncrement;

  // Compute the rotation increment given the deformation increment
  computeRotationIncrement(defInc);

  // Compute the stress increment
  computeStressIncrement();

  // Rotate the stress increment
  Matrix3 tempStressIncrement(0.0);
  tempStressIncrement = stressIncrement;
  stressIncrement = 
    rotationIncrement * tempStressIncrement * rotationIncrement.Transpose();

  // Update the stress tensor with the rotated stress increment
  stressTensor = stressTensor +  stressIncrement;

}
#endif

void ElasticConstitutiveModel::addComputesAndRequires(Task* task,
                                                      const MPMMaterial* /*matl*/,
                                                      const PatchSet* patches) const
{
   cerr << "ElasticConsitutive::addComputesAndRequires needs to be filled in\n";
}

void ElasticConstitutiveModel::readParameters(ProblemSpecP ps, double *p_array)
{

  ps->require("youngs_modulus",p_array[0]);
  ps->require("poissons_ratio",p_array[1]);

}

void ElasticConstitutiveModel::writeParameters(ofstream& out, double *p_array)
{
  out << p_array[0] << " " << p_array[1] << " ";
}

ConstitutiveModel*
ElasticConstitutiveModel::readParametersAndCreate(ProblemSpecP ps)
{
  double p_array[2];
  readParameters(ps, p_array);
  return(create(p_array));
}

void ElasticConstitutiveModel::writeRestartParameters(ofstream& out) const
{
#if 0
  out << getType() << " ";
  out << YngMod << " " << PoiRat << " ";
  out << (getStressTensor())(1,1) << " "
      << (getStressTensor())(1,2) << " "
      << (getStressTensor())(1,3) << " "
      << (getStressTensor())(2,2) << " "
      << (getStressTensor())(2,3) << " "
      << (getStressTensor())(3,3) << endl;
#endif
}

ConstitutiveModel*
ElasticConstitutiveModel::readRestartParametersAndCreate(ProblemSpecP ps)
{
#if 0
  Matrix3 st(0.0);
  ConstitutiveModel *cm = readParametersAndCreate(ps);
  
  in >> st(1,1) >> st(1,2) >> st(1,3)
     >> st(2,2) >> st(2,3) >> st(3,3);
  st(2,1)=st(1,2);
  st(3,1)=st(1,3);
  st(3,2)=st(2,3);
  cm->setStressTensor(st);
  
  return(cm);
#else
  return 0;
#endif
}

ConstitutiveModel*
ElasticConstitutiveModel::create(double *p_array)
{
#ifdef WONT_COMPILE_YET
  return(scinew ElasticConstitutiveModel(p_array[0], p_array[1]));
#else
  return 0;
#endif
}

int ElasticConstitutiveModel::getType() const
{
  //  return(ConstitutiveModelFactory::CM_ELASTIC);
}

string ElasticConstitutiveModel::getName() const
{
  return("Elastic");
}

int ElasticConstitutiveModel::getNumParameters() const
{
  return(2);
}

void ElasticConstitutiveModel::printParameterNames(ofstream& out) const
{
  out << "Yng's Mod" << endl
      << "Pois. Rat" << endl;
}


namespace Uintah {

static MPI_Datatype makeMPI_CMData()
{
   ASSERTEQ(sizeof(ElasticConstitutiveModel::CMData), sizeof(double)*2);
   MPI_Datatype mpitype;
   MPI_Type_vector(1, 2, 2, MPI_DOUBLE, &mpitype);
   MPI_Type_commit(&mpitype);
   return mpitype;
}

const TypeDescription* fun_getTypeDescription(ElasticConstitutiveModel::CMData*)
{
   static TypeDescription* td = 0;
   if(!td){
      td = scinew TypeDescription(TypeDescription::Other, "ElasticConstitutiveModel::CMData", true, &makeMPI_CMData);
   }
   return td;   
}

ConstitutiveModel*
ElasticConstitutiveModel::copy() const
{
#ifdef WONT_COMPILE_YET
  return( scinew ElasticConstitutiveModel(*this) );
#else
  return 0;
#endif
}

#if 0
ConstitutiveModel&
ElasticConstitutiveModel::operator=(const ElasticConstitutiveModel &cm)
{
 
  stressTensor=cm.stressTensor;
  strainTensor=cm.strainTensor;
  strainIncrement=cm.strainIncrement;
  stressIncrement=cm.stressIncrement;
  rotationIncrement=cm.rotationIncrement;
  YngMod=cm.YngMod;
  PoiRat=cm.PoiRat;

} // End namespace Uintah
  return (*this);
#endif

int ElasticConstitutiveModel::getSize() const
{
  int s = 0;
  s += sizeof(double) * 6;  // stressTensor elements
  s += sizeof(double) * 2;  // properties
  s += sizeof(int) * 1;     // type
  return(s);
}

} //namespace Uintah
