/*
 * The MIT License
 *
 * Copyright (c) 2013-2014 The Johns Hopkins University
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

// Adapted from UCNH.cc by Andy Tonge Dec 2011 altonge@gmail.com

#include <CCA/Components/MPM/ConstitutiveModel/TongeRamesh.h>
#include <CCA/Components/MPM/ConstitutiveModel/TongeRamesh_gitInfo.h>
#include <CCA/Components/MPM/ConstitutiveModel/PlasticityModels/MPMEquationOfStateFactory.h>
#include <CCA/Components/MPM/ConstitutiveModel/PlasticityModels/PlasticityState.h>

#include <Core/Grid/Patch.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Math/Matrix3.h>
#include <Core/Math/FastMatrix.h>
// #include <Core/Grid/Variables/NodeIterator.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/Gaussian.h>
#include <Core/Math/Weibull.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MersenneTwister.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <complex>

#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ConvergenceFailure.h>
#include <stdexcept>

using namespace std;
using namespace Uintah;

#define PI 3.1415926535897931
//#define Comer
#undef Comer

// Constructors //
//////////////////
TongeRamesh::TongeRamesh(ProblemSpecP& ps, MPMFlags* Mflag)
  : ConstitutiveModel(Mflag), ImplicitCM()
{
  proc0cout << "TongeRamesh Material model:\n\t Last commit date:"
            << build_date << "\n"
            << "\t Commit sha and message: " << build_git_commit
            << endl;
  ps->require("bulk_modulus",         d_initialData.Bulk);
  ps->require("shear_modulus",        d_initialData.tauDev);
  // ps->get("useModifiedEOS",           d_useModifiedEOS);

  d_8or27=Mflag->d_8or27;

  // This was taken from ElasticPlastic.cc it lets me leverage the
  // equations of state that are in PlasticityModels/
  d_eos = MPMEquationOfStateFactory::create(ps);
  d_eos->setBulkModulus(d_initialData.Bulk);
  
  if(!d_eos){
    ostringstream desc;
    desc << "An error occured in the EquationOfStateFactory that has \n"
         << " slipped through the existing bullet proofing. Please tell \n"
         << " Biswajit.  "<< endl;
    throw ParameterNotFound(desc.str(), __FILE__, __LINE__);
  }
  
  // Plasticity
  ps->getWithDefault("usePlasticity", d_usePlasticity, false);
  if(d_usePlasticity) {
    ps->getWithDefault("alpha",       d_initialData.Alpha,0.0);
    ps->require("yield_stress",       d_initialData.FlowStress);
    ps->require("hardening_modulus",  d_initialData.K);
    ps->getWithDefault("timeConstant", d_initialData.timeConstant, 0.0);

      
    pPlasticStrain_label          = VarLabel::create("p.pPlasticStrain_cnhp",
                                                     ParticleVariable<double>::getTypeDescription());
    pPlasticStrain_label_preReloc = VarLabel::create("p.pPlasticStrain_cnhp+",
                                                     ParticleVariable<double>::getTypeDescription());
    pPlasticEnergy_label = VarLabel::create("p.pPlasticEnergy",
                                            ParticleVariable<double>::getTypeDescription());
    pPlasticEnergy_label_preReloc = VarLabel::create("p.pPlasticEnergy+",
                                                     ParticleVariable<double>::getTypeDescription());
  } // End Plasticity
  
  // Damage
  ps->getWithDefault("useDamage", d_useDamage, false);
  if(d_useDamage) {
    // Initialize local VarLabels
    initializeLocalMPMLabels();
    getBrittleDamageData(ps);
    getFlawDistributionData(ps);    
    
    // I could enclose this in a try block:
    wingLengthLabel_array          = new VarLabel*[d_flawDistData.numCrackFamilies];
    wingLengthLabel_array_preReloc = new VarLabel*[d_flawDistData.numCrackFamilies];
    starterFlawSize_array          = new VarLabel*[d_flawDistData.numCrackFamilies];
    starterFlawSize_array_preReloc = new VarLabel*[d_flawDistData.numCrackFamilies];
    flawNumber_array               = new VarLabel*[d_flawDistData.numCrackFamilies];
    flawNumber_array_preReloc      = new VarLabel*[d_flawDistData.numCrackFamilies];

    for(int i = 0; i<d_flawDistData.numCrackFamilies; i++){
      stringstream ss;
      ss << i;
      wingLengthLabel_array[i] = VarLabel::create("p.wingLength_" + ss.str(),
                                                  ParticleVariable<double>::getTypeDescription());
      wingLengthLabel_array_preReloc[i] = VarLabel::create("p.wingLength_" + ss.str()+"+",
                                                           ParticleVariable<double>::getTypeDescription());
      starterFlawSize_array[i] = VarLabel::create("p.starterFlawSize_" + ss.str(),
                                                  ParticleVariable<double>::getTypeDescription());
      starterFlawSize_array_preReloc[i] = VarLabel::create("p.starterFlawSize_" + ss.str() + "+",
                                                           ParticleVariable<double>::getTypeDescription());
      flawNumber_array[i] = VarLabel::create("p.flawNumber_" + ss.str(),
                                             ParticleVariable<double>::getTypeDescription());
      flawNumber_array_preReloc[i] = VarLabel::create("p.flawNumber_" + ss.str() + "+",
                                                      ParticleVariable<double>::getTypeDescription());
    }

    if(d_brittle_damage.useNonlocalDamage){
      gDamage_Label = VarLabel::create("g.Damage", NCVariable<double>::getTypeDescription());
    }

    // Set the erosion algorithm
    setErosionAlgorithm();

  } // End Damage

  // Granular Plasticity:
  ps->getWithDefault("useGranularPlasticity", d_useGranularPlasticity, false);
  if(d_useGranularPlasticity){
    if(Mflag->d_doPressureStabilization){
      cout << "***WARNING*** the TongeRamesh constitutive model with GranularPlasticity enabled\n";
      cout << "\t is not designed to work with Pressure Stabilization\n";
      cout << "\t it can lead to large non-realistic fluctutations in the stress and instabilities";
      cout << endl;
    }
    
    getGranularPlasticityData(ps);

    // cout << "Creating GP labels:" << endl;
    // Create labels:
    pGPJLabel          = VarLabel::create("p.GPJ",
                                          ParticleVariable<double>::getTypeDescription());
    pGPJLabel_preReloc = VarLabel::create("p.GPJ+",
                                          ParticleVariable<double>::getTypeDescription());
    pGP_plasticStrainLabel          = VarLabel::create("p.GP_plasticStrain",
                                                       ParticleVariable<double>::getTypeDescription());
    pGP_plasticStrainLabel_preReloc = VarLabel::create("p.GP_plasticStrain+",
                                                       ParticleVariable<double>::getTypeDescription());
    pGP_plasticEnergyLabel          = VarLabel::create("p.GP_plasticEnergy",
                                                       ParticleVariable<double>::getTypeDescription());
    pGP_plasticEnergyLabel_preReloc = VarLabel::create("p.GP_plasticEnergy+",
                                                       ParticleVariable<double>::getTypeDescription());
  }
  
  // Universal Labels
  pEnergyLabel               = VarLabel::create("p.energy",
                                                ParticleVariable<double>::getTypeDescription());
  pEnergyLabel_preReloc      = VarLabel::create("p.energy+",
                                                ParticleVariable<double>::getTypeDescription());
  bElBarLabel                = VarLabel::create("p.bElBar",
                                                ParticleVariable<Matrix3>::getTypeDescription());
  bElBarLabel_preReloc       = VarLabel::create("p.bElBar+",
                                                ParticleVariable<Matrix3>::getTypeDescription());
  pDeformRateLabel           = VarLabel::create("p.deformRate",
                                                ParticleVariable<Matrix3>::getTypeDescription());
  pDeformRateLabel_preReloc  = VarLabel::create("p.deformRate+",
                                                ParticleVariable<Matrix3>::getTypeDescription());
}

TongeRamesh::TongeRamesh(const TongeRamesh* cm) : ConstitutiveModel(cm), ImplicitCM(cm)
{
  // d_useModifiedEOS     = cm->d_useModifiedEOS ;
  d_initialData.Bulk   = cm->d_initialData.Bulk;
  d_initialData.tauDev = cm->d_initialData.tauDev;

  d_eos = MPMEquationOfStateFactory::createCopy(cm->d_eos);
  d_eos->setBulkModulus(d_initialData.Bulk);
  
  // Plasticity Setup
  d_usePlasticity      = cm->d_usePlasticity;
  if(d_usePlasticity) {
    d_initialData.FlowStress = cm->d_initialData.FlowStress;
    d_initialData.K          = cm->d_initialData.K;
    d_initialData.Alpha      = cm->d_initialData.Alpha;
    d_initialData.timeConstant = cm->d_initialData.timeConstant;
    
    pPlasticStrain_label          = VarLabel::create("p.pPlasticStrain_cnhp",
                                                     ParticleVariable<double>::getTypeDescription());
    pPlasticStrain_label_preReloc = VarLabel::create("p.pPlasticStrain_cnhp+",
                                                     ParticleVariable<double>::getTypeDescription());
    pPlasticEnergy_label = VarLabel::create("p.pPlasticEnergy",
                                            ParticleVariable<double>::getTypeDescription());
    pPlasticEnergy_label_preReloc = VarLabel::create("p.pPlasticEnergy+",
                                                     ParticleVariable<double>::getTypeDescription());
  } // End Plasticity Setup
  
  // Damage Setup
  d_useDamage = cm->d_useDamage;
 
  if(d_useDamage) {
    // Initialize local VarLabels
    initializeLocalMPMLabels();
    setBrittleDamageData(cm);
    setFlawDistributionData(cm);    
    
    // I could enclose this in a try block:
    wingLengthLabel_array          = new VarLabel*[d_flawDistData.numCrackFamilies];
    wingLengthLabel_array_preReloc = new VarLabel*[d_flawDistData.numCrackFamilies];
    starterFlawSize_array          = new VarLabel*[d_flawDistData.numCrackFamilies];
    starterFlawSize_array_preReloc = new VarLabel*[d_flawDistData.numCrackFamilies];
    flawNumber_array               = new VarLabel*[d_flawDistData.numCrackFamilies];
    flawNumber_array_preReloc      = new VarLabel*[d_flawDistData.numCrackFamilies];

    for(int i = 0; i<d_flawDistData.numCrackFamilies; i++){
      stringstream ss;
      ss << i;
      wingLengthLabel_array[i] =
        VarLabel::create("p.wingLength_" + ss.str(),
                         ParticleVariable<double>::getTypeDescription());
      wingLengthLabel_array_preReloc[i] =
        VarLabel::create("p.wingLength_" + ss.str()+"+",
                         ParticleVariable<double>::getTypeDescription());
      starterFlawSize_array[i] =
        VarLabel::create("p.starterFlawSize_" + ss.str(),
                         ParticleVariable<double>::getTypeDescription());
      starterFlawSize_array_preReloc[i] =
        VarLabel::create("p.starterFlawSize_" + ss.str() + "+",
                         ParticleVariable<double>::getTypeDescription());
      flawNumber_array[i] =
        VarLabel::create("p.flawNumber_" + ss.str(),
                         ParticleVariable<double>::getTypeDescription());
      flawNumber_array_preReloc[i] =
        VarLabel::create("p.flawNumber_" + ss.str() + "+",
                         ParticleVariable<double>::getTypeDescription());
    }

    // Set the erosion algorithm
    setErosionAlgorithm(cm);
    if(d_brittle_damage.useNonlocalDamage){
      gDamage_Label = VarLabel::create("g.Damage", NCVariable<double>::getTypeDescription());
    }
  } // End Damage Setup

  if(d_useGranularPlasticity){
    // Set the model parameters:
    setGranularPlasticityData(cm);      
    // Create labels:
    pGPJLabel          = VarLabel::create("p.GPJ",
                                          ParticleVariable<double>::getTypeDescription());
    pGPJLabel_preReloc = VarLabel::create("p.GPJ+",
                                          ParticleVariable<double>::getTypeDescription());
    pGP_plasticStrainLabel          = VarLabel::create("p.GP_plasticStrain",
                                                       ParticleVariable<double>::getTypeDescription());
    pGP_plasticStrainLabel_preReloc = VarLabel::create("p.GP_plasticStrain+",
                                                       ParticleVariable<double>::getTypeDescription());
    pGP_plasticEnergyLabel          = VarLabel::create("p.GP_plasticEnergy",
                                                       ParticleVariable<double>::getTypeDescription());
    pGP_plasticEnergyLabel_preReloc = VarLabel::create("p.GP_plasticEnergy+",
                                                       ParticleVariable<double>::getTypeDescription());
  }
  
  
  // Universal Labels
  pEnergyLabel               = VarLabel::create("p.energy",
                                                ParticleVariable<double>::getTypeDescription());
  pEnergyLabel_preReloc      = VarLabel::create("p.energy+",
                                                ParticleVariable<double>::getTypeDescription());
  
  bElBarLabel                = VarLabel::create("p.bElBar",
                                                ParticleVariable<Matrix3>::getTypeDescription());
  bElBarLabel_preReloc       = VarLabel::create("p.bElBar+",
                                                ParticleVariable<Matrix3>::getTypeDescription());
  
  pDeformRateLabel           = VarLabel::create("p.deformRate",
                                                ParticleVariable<Matrix3>::getTypeDescription());
  pDeformRateLabel_preReloc  = VarLabel::create("p.deformRate+",
                                                ParticleVariable<Matrix3>::getTypeDescription());
}

void TongeRamesh::initializeLocalMPMLabels()
{
  pLocalizedLabel     = VarLabel::create("p.localized",
                                         ParticleVariable<int>::getTypeDescription());
  pDamageLabel     = VarLabel::create("p.damage",
                                      ParticleVariable<double>::getTypeDescription());
  
  pLocalizedLabel_preReloc     = VarLabel::create("p.localized+",
                                                  ParticleVariable<int>::getTypeDescription());
  pDamageLabel_preReloc     = VarLabel::create("p.damage+",
                                               ParticleVariable<double>::getTypeDescription());

}


void TongeRamesh::setBrittleDamageData(const TongeRamesh* cm)
{

  // For Bhasker's model:
  d_brittle_damage.KIc    = cm->d_brittle_damage.KIc;
  d_brittle_damage.mu     = cm->d_brittle_damage.mu;
  d_brittle_damage.phi    = cm->d_brittle_damage.phi;
  d_brittle_damage.cgamma = cm->d_brittle_damage.cgamma;
  d_brittle_damage.alpha  = cm->d_brittle_damage.alpha;
  d_brittle_damage.criticalDamage    = cm->d_brittle_damage.criticalDamage;
  d_brittle_damage.usePlaneStrain    = cm->d_brittle_damage.usePlaneStrain;
  
  d_brittle_damage.useDamageTimeStep = cm->d_brittle_damage.useDamageTimeStep;
  d_brittle_damage.useOldStress      = cm->d_brittle_damage.useOldStress;
  d_brittle_damage.dt_increaseFactor = cm->d_brittle_damage.dt_increaseFactor;
  d_brittle_damage.incInitialDamage  = cm->d_brittle_damage.incInitialDamage;
  d_brittle_damage.doFlawInteraction = cm->d_brittle_damage.doFlawInteraction;
  d_brittle_damage.useNonlocalDamage = cm->d_brittle_damage.useNonlocalDamage;
}

void TongeRamesh::setFlawDistributionData(const TongeRamesh* cm){
  d_flawDistData.numCrackFamilies = cm->d_flawDistData.numCrackFamilies;
  d_flawDistData.flawDensity      = cm->d_flawDistData.flawDensity;
  d_flawDistData.type             = cm->d_flawDistData.type;

  d_flawDistData.randomSeed       = cm->d_flawDistData.randomSeed;
  d_flawDistData.randomizeDist = cm->d_flawDistData.randomizeDist;
  d_flawDistData.randomMethod  = cm->d_flawDistData.randomMethod;
  d_flawDistData.binBias       = cm->d_flawDistData.binBias;

  
  if( cm->d_flawDistData.type == "normal"){
    d_flawDistData.meanFlawSize = cm->d_flawDistData.meanFlawSize;
    d_flawDistData.stdFlawSize = cm->d_flawDistData.stdFlawSize;
    d_flawDistData.minFlawSize = cm->d_flawDistData.minFlawSize;
    d_flawDistData.maxFlawSize = cm->d_flawDistData.maxFlawSize;
  } else if ( cm->d_flawDistData.type == "logNormal"){
    d_flawDistData.meanFlawSize = cm->d_flawDistData.meanFlawSize;
    d_flawDistData.stdFlawSize = cm->d_flawDistData.stdFlawSize;
    d_flawDistData.minFlawSize = cm->d_flawDistData.minFlawSize;
    d_flawDistData.maxFlawSize = cm->d_flawDistData.maxFlawSize;
  }else if (cm->d_flawDistData.type == "delta") {
    d_flawDistData.meanFlawSize = cm->d_flawDistData.meanFlawSize;
  } else if (d_flawDistData.type == "pareto") {
    d_flawDistData.minFlawSize = cm->d_flawDistData.minFlawSize;
    d_flawDistData.maxFlawSize = cm->d_flawDistData.maxFlawSize;
    d_flawDistData.exponent    = cm-> d_flawDistData.exponent;
  } else {
    throw ProblemSetupException("The flaw distribution was not reconised"
                                , __FILE__, __LINE__);
  }

  // Assignment of the initial flaw field data:
  d_flawDistData.useEtaField = cm->d_flawDistData.useEtaField;
  d_flawDistData.etaFilename = cm->d_flawDistData.etaFilename;
  d_flawDistData.useSizeField = cm->d_flawDistData.useSizeField;
  d_flawDistData.sizeFilename = cm->d_flawDistData.sizeFilename;
}

void TongeRamesh::setGranularPlasticityData(const TongeRamesh* cm){
  d_GPData.timeConstant = cm->d_GPData.timeConstant;
  d_GPData.JGP_loc      = cm->d_GPData.JGP_loc;

  d_GPData.A            = cm->d_GPData.A;
  d_GPData.B            = cm->d_GPData.B;
  d_GPData.yeildSurfaceType = cm->d_GPData.yeildSurfaceType;

  d_GPData.Pc      = cm->d_GPData.Pc;
  d_GPData.alpha_e = cm->d_GPData.alpha_e;
  d_GPData.Pe       = cm->d_GPData.Pe;
}

void TongeRamesh::getBrittleDamageData(ProblemSpecP& ps)
{
  // Print damage details:
  ps->getWithDefault("brittle_damage_printDamage",         d_brittle_damage.printDamage, false);

  // Damage timestepping control:
  ps->getWithDefault("brittle_damage_max_damage_increment",d_brittle_damage.maxDamageInc, 1e-4);
  ps->getWithDefault("bhasker_use_damage_timestep",d_brittle_damage.useDamageTimeStep, false);
  ps->getWithDefault("bhasker_damage_useOldStress", d_brittle_damage.useOldStress, false);
  ps->getWithDefault("bhasker_damage_dt_increaseFactor", d_brittle_damage.dt_increaseFactor, 10.0);

  // Data for Bhasker's model:
  ps->getWithDefault("bhasker_damage_KIc",   d_brittle_damage.KIc, 2e-6);
  ps->getWithDefault("bhasker_damage_mu",   d_brittle_damage.mu, 0.2);
  ps->getWithDefault("bhasker_damage_phi",   d_brittle_damage.phi,
                     2*atan(1.0)-0.5*atan(1/d_brittle_damage.mu));
  ps->getWithDefault("bhasker_damage_cgamma",   d_brittle_damage.cgamma, 1.0);
  ps->getWithDefault("bhasker_damage_alpha",   d_brittle_damage.alpha, 5.0);
  ps->getWithDefault("bhasker_damage_critDamage", d_brittle_damage.criticalDamage, 1.0);
  ps->getWithDefault("bhasker_damage_maxDamage", d_brittle_damage.maxDamage, d_brittle_damage.criticalDamage);
  ps->getWithDefault("bhasker_damage_incInitialDamage", d_brittle_damage.incInitialDamage, false);
  ps->getWithDefault("bhasker_damage_doFlawInteraction", d_brittle_damage.doFlawInteraction, true);
  ps->getWithDefault("bhasker_damage_usePlaneStrain", d_brittle_damage.usePlaneStrain, false);
  if(d_brittle_damage.criticalDamage > d_brittle_damage.maxDamage){
    cerr << "*** WARNING *** TongeRamesh, setting criticalDamage > "
         << "maxDamage may cause damage evolution to stop before "
         << "either localization or granular plasticity is activated"
         << endl;
  }

  ps->getWithDefault("useNonlocalDamage", d_brittle_damage.useNonlocalDamage, false);
}


void TongeRamesh::getFlawDistributionData(ProblemSpecP& ps){
  // Set the defaults:
  d_flawDistData.numCrackFamilies = 10;
  d_flawDistData.meanFlawSize     = 100e-6; // 100 microns
  d_flawDistData.flawDensity  = 1e14;
  d_flawDistData.stdFlawSize  = 100e-6; // Delta distribution

  // Read from the input file:
  ps->get("flaw_dist_numFamilies",   d_flawDistData.numCrackFamilies);
  ps->get("flaw_dist_flawDensity",   d_flawDistData.flawDensity);

  // Read the distribution type:
  // Valid distribution types are: normal, delta, pareto
  ps->getWithDefault("flaw_dist_type", d_flawDistData.type, "normal");
  
  ps->getWithDefault("flaw_dist_randomize", d_flawDistData.randomizeDist, false);
  ps->getWithDefault("flaw_dist_seed", d_flawDistData.randomSeed, 0);
  ps->getWithDefault("flaw_dist_BinMethod", d_flawDistData.randomMethod, 0);
  ps->getWithDefault("flaw_dist_BinBias", d_flawDistData.binBias, 1.0);
  ps->getWithDefault("flaw_dist_Ncutoff", d_flawDistData.Ncutoff, 20);
  if( d_flawDistData.type == "normal"){
    ps->require("flaw_dist_meanFlawSize",   d_flawDistData.meanFlawSize);
    ps->require("flaw_dist_stdFlawSize",   d_flawDistData.stdFlawSize);
    ps->getWithDefault("flaw_dist_minFlaw",  d_flawDistData.minFlawSize,
                       d_flawDistData.meanFlawSize - 5.0*d_flawDistData.stdFlawSize);
    ps->getWithDefault("flaw_dist_maxFlaw",  d_flawDistData.maxFlawSize,
                       d_flawDistData.meanFlawSize + 5.0*d_flawDistData.stdFlawSize);
    // Error checking:
    d_flawDistData.minFlawSize = d_flawDistData.minFlawSize < 0 ? 0 : d_flawDistData.minFlawSize;

  } else if (d_flawDistData.type == "delta") {
    ps->require("flaw_dist_meanFlawSize",   d_flawDistData.meanFlawSize);
  } else if (d_flawDistData.type == "pareto") {
    ps->require("flaw_dist_minFlaw",  d_flawDistData.minFlawSize);
    ps->require("flaw_dist_maxFlaw",  d_flawDistData.maxFlawSize);
    ps->require("flaw_dist_exponent", d_flawDistData.exponent);
    if((d_flawDistData.maxFlawSize - d_flawDistData.minFlawSize) <= 0){
      throw ProblemSetupException("The maximum flaw size must be greater than the minimum flaw size"
                                  , __FILE__, __LINE__);
    }
    if(d_flawDistData.exponent <= 0){
      throw ProblemSetupException("The flaw distribution exponent must be positive"
                                  , __FILE__, __LINE__);
    }
  } else if( d_flawDistData.type == "logNormal"){
    ps->require("flaw_dist_meanFlawSize",   d_flawDistData.meanFlawSize);
    ps->require("flaw_dist_stdFlawSize",   d_flawDistData.stdFlawSize);
    ps->require("flaw_dist_minFlaw",  d_flawDistData.minFlawSize);
    ps->require("flaw_dist_maxFlaw",  d_flawDistData.maxFlawSize);
    // Error checking:
    if ( d_flawDistData.minFlawSize <= 0 || d_flawDistData.maxFlawSize <= d_flawDistData.minFlawSize ){
      throw ProblemSetupException("For the log normal flaw distribution, the maximum flaw size > minimum flaw size  0 is required", __FILE__, __LINE__);
    }
  } else {
    // The distribution type is not reconized throw an error:
    string txt = "The distribution: " + d_flawDistData.type + " is not valid";
    throw ProblemSetupException(txt, __FILE__, __LINE__);
  }

  // For defining the spatial variability using a fourier method:
  ps->getWithDefault("flaw_dist_useEtaField", d_flawDistData.useEtaField, false);
  ps->getWithDefault("flaw_dist_etaFileName", d_flawDistData.etaFilename, "flawDensityData.txt");
  ps->getWithDefault("flaw_dist_useSizeField", d_flawDistData.useSizeField, false);
  ps->getWithDefault("flaw_dist_sizeFileName", d_flawDistData.sizeFilename, "flawSizeData.txt");
}

void TongeRamesh::getGranularPlasticityData(ProblemSpecP& ps){

  // Yeild surface specification:
  ps->getWithDefault("gp_yeildSurfaceType", d_GPData.yeildSurfaceType, 2);
  ps->require("gp_A", d_GPData.A);
  ps->require("gp_cohesiveStrength", d_GPData.B);

  // Flow controls:
  ps->getWithDefault("gp_JGP_localize", d_GPData.JGP_loc, 5.0);
  ps->getWithDefault("gp_timeConstant",d_GPData.timeConstant, 0.0);

  // P-alpha parameters:
  ps->getWithDefault("gp_Pc", d_GPData.Pc, 1e12);
  ps->getWithDefault("gp_Pe", d_GPData.Pe, 1e11);
  ps->getWithDefault("gp_JGPe", d_GPData.alpha_e, 3.0);

  // Error checking
  if(d_GPData.JGP_loc<=1.0){
    throw ProblemSetupException("The value for gp_JGP_loc must be greater than 1.0",
                                __FILE__, __LINE__);
  }
  if(d_GPData.yeildSurfaceType == 1 && d_GPData.A>10){
    cerr << "***WARNING*** for Granular flow yeild surface type 1,"
         << " A is a slope and should be order 1" << endl;
  }
      
  if(d_GPData.Pc <= d_GPData.Pe){
    throw ProblemSetupException("The value for gp_Pc (full consolidation pressure must be greater than gp_Pe (pressure to start compaction)",
                                __FILE__, __LINE__);
  }
  if(d_GPData.alpha_e <= 1.0){
    throw ProblemSetupException("The value for gp_JGPe (distension for start of compaction) must be greater than 1.0",
                                __FILE__, __LINE__);
  }
  if(d_GPData.A <= 0.0){
    throw ProblemSetupException("The value for gp_A must be greater than 0.0",
                                __FILE__, __LINE__);
  }

}


void TongeRamesh::setErosionAlgorithm()
{
  d_setStressToZero = false;
  d_allowNoTension  = false;
  d_allowNoShear    = false;
  if (flag->d_doErosion) {
    if (flag->d_erosionAlgorithm == "AllowNoTension") 
      d_allowNoTension  = true;
    else if (flag->d_erosionAlgorithm == "ZeroStress") 
      d_setStressToZero = true;
    else if (flag->d_erosionAlgorithm == "AllowNoShear") 
      d_allowNoShear    = true;
  }
}

void TongeRamesh::setErosionAlgorithm(const TongeRamesh* cm)
{
  d_setStressToZero = cm->d_setStressToZero;
  d_allowNoTension  = cm->d_allowNoTension;
  d_allowNoShear    = cm->d_allowNoShear;
}

void TongeRamesh::outputProblemSpec(ProblemSpecP& ps, bool output_cm_tag)
{
  ProblemSpecP cm_ps = ps;
  if (output_cm_tag) {
    cm_ps = ps->appendChild("constitutive_model");
    cm_ps->setAttribute("type","TongeRamesh");
  }
  
  cm_ps->appendElement("bulk_modulus",             d_initialData.Bulk);
  cm_ps->appendElement("shear_modulus",            d_initialData.tauDev);
  // cm_ps->appendElement("useModifiedEOS",           d_useModifiedEOS);
  d_eos->outputProblemSpec(cm_ps);
  cm_ps->appendElement("usePlasticity",            d_usePlasticity);
  cm_ps->appendElement("useDamage",                d_useDamage);
  cm_ps->appendElement("useGranularPlasticity", d_useGranularPlasticity);
  // Plasticity
  if(d_usePlasticity) {
    cm_ps->appendElement("yield_stress",           d_initialData.FlowStress);
    cm_ps->appendElement("hardening_modulus",      d_initialData.K);
    cm_ps->appendElement("alpha",                  d_initialData.Alpha);
    cm_ps->appendElement("timeConstant",           d_initialData.timeConstant);
  }
  
  // Damage
  if(d_useDamage) {
    cm_ps->appendElement("brittle_damage_max_damage_increment", d_brittle_damage.maxDamageInc);
    cm_ps->appendElement("brittle_damage_printDamage",          d_brittle_damage.printDamage);

    cm_ps->appendElement("bhasker_damage_KIc",   d_brittle_damage.KIc);
    cm_ps->appendElement("bhasker_damage_mu",   d_brittle_damage.mu);
    cm_ps->appendElement("bhasker_damage_phi",   d_brittle_damage.phi);
    cm_ps->appendElement("bhasker_damage_cgamma",   d_brittle_damage.cgamma);
    cm_ps->appendElement("bhasker_damage_alpha",   d_brittle_damage.alpha);
    cm_ps->appendElement("bhasker_damage_critDamage", d_brittle_damage.criticalDamage);
    cm_ps->appendElement("bhasker_damage_maxDamage", d_brittle_damage.maxDamage);
    cm_ps->appendElement("bhasker_damage_usePlaneStrain", d_brittle_damage.usePlaneStrain);

    cm_ps->appendElement("bhasker_damage_useOldStress", d_brittle_damage.useOldStress);
    cm_ps->appendElement("bhasker_damage_dt_increaseFactor", d_brittle_damage.dt_increaseFactor);
    cm_ps->appendElement("bhasker_damage_incInitialDamage", d_brittle_damage.incInitialDamage);
    cm_ps->appendElement("bhasker_damage_doFlawInteraction", d_brittle_damage.doFlawInteraction);
    cm_ps->appendElement("useNonlocalDamage", d_brittle_damage.useNonlocalDamage);
    
    cm_ps->appendElement("flaw_dist_numFamilies",   d_flawDistData.numCrackFamilies);
    cm_ps->appendElement("flaw_dist_type",   d_flawDistData.type);
    cm_ps->appendElement("flaw_dist_flawDensity",   d_flawDistData.flawDensity);

    cm_ps->appendElement("flaw_dist_randomize", d_flawDistData.randomizeDist);
    cm_ps->appendElement("flaw_dist_seed", d_flawDistData.randomSeed);
    cm_ps->appendElement("flaw_dist_BinMethod", d_flawDistData.randomMethod);
    cm_ps->appendElement("flaw_dist_BinBias", d_flawDistData.binBias);
    cm_ps->appendElement("flaw_dist_Ncutoff", d_flawDistData.Ncutoff);

    if( d_flawDistData.type == "normal" || d_flawDistData.type == "logNormal"){
      cm_ps->appendElement("flaw_dist_meanFlawSize",   d_flawDistData.meanFlawSize);
      cm_ps->appendElement("flaw_dist_stdFlawSize",   d_flawDistData.stdFlawSize);
      cm_ps->appendElement("flaw_dist_minFlaw",   d_flawDistData.minFlawSize);
      cm_ps->appendElement("flaw_dist_maxFlaw",   d_flawDistData.maxFlawSize);
    } else if (d_flawDistData.type == "delta") {
      cm_ps->appendElement("flaw_dist_meanFlawSize",   d_flawDistData.meanFlawSize);
    } else if (d_flawDistData.type == "pareto") {
      cm_ps->appendElement("flaw_dist_minFlaw",   d_flawDistData.minFlawSize);
      cm_ps->appendElement("flaw_dist_maxFlaw",   d_flawDistData.maxFlawSize);
      cm_ps->appendElement("flaw_dist_exponent",   d_flawDistData.exponent);
    }
    // For defining the spatial variability using a fourier method:
    cm_ps->appendElement("flaw_dist_useEtaField", d_flawDistData.useEtaField);
    cm_ps->appendElement("flaw_dist_etaFileName", d_flawDistData.etaFilename);
    cm_ps->appendElement("flaw_dist_useSizeField", d_flawDistData.useSizeField);
    cm_ps->appendElement("flaw_dist_sizeFileName", d_flawDistData.sizeFilename);
  } //end if d_useDamage

  if(d_useGranularPlasticity){
    cm_ps->appendElement("gp_A", d_GPData.A);
    cm_ps->appendElement("gp_cohesiveStrength", d_GPData.B);
    cm_ps->appendElement("gp_yeildSurfaceType", d_GPData.yeildSurfaceType);
    
    cm_ps->appendElement("gp_JGP_localize", d_GPData.JGP_loc);
    cm_ps->appendElement("gp_timeConstant",d_GPData.timeConstant);

    cm_ps->appendElement("gp_Pc", d_GPData.Pc);
    cm_ps->appendElement("gp_Pe", d_GPData.Pe);
    cm_ps->appendElement("gp_JGPe", d_GPData.alpha_e);
  } // end if d_useGranularPlasticity
}

TongeRamesh* TongeRamesh::clone()
{
  return scinew TongeRamesh(*this);
}

TongeRamesh::~TongeRamesh()
{

  delete d_eos;

  // Plasticity Deletes
  if(d_usePlasticity) {
    VarLabel::destroy(pPlasticStrain_label);
    VarLabel::destroy(pPlasticStrain_label_preReloc);
    VarLabel::destroy(pPlasticEnergy_label);
    VarLabel::destroy(pPlasticEnergy_label_preReloc);
  }
  
  if(d_useDamage) {
    VarLabel::destroy(pLocalizedLabel);
    VarLabel::destroy(pLocalizedLabel_preReloc);
    VarLabel::destroy(pDamageLabel);
    VarLabel::destroy(pDamageLabel_preReloc);
    if(d_brittle_damage.useNonlocalDamage){
      VarLabel::destroy(gDamage_Label);
    }

    for(int i = 0; i<d_flawDistData.numCrackFamilies; i++){
      VarLabel::destroy(wingLengthLabel_array[i]);
      VarLabel::destroy(wingLengthLabel_array_preReloc[i]);
      VarLabel::destroy(starterFlawSize_array[i]);
      VarLabel::destroy(starterFlawSize_array_preReloc[i]);
      VarLabel::destroy(flawNumber_array[i]);
      VarLabel::destroy(flawNumber_array_preReloc[i]);
    }

    delete [] wingLengthLabel_array;
    delete [] wingLengthLabel_array_preReloc;
    delete [] starterFlawSize_array;
    delete [] starterFlawSize_array_preReloc;
    delete [] flawNumber_array;
    delete [] flawNumber_array_preReloc;
  }

  if(d_useGranularPlasticity){
    VarLabel::destroy(pGPJLabel);
    VarLabel::destroy(pGPJLabel_preReloc);
    VarLabel::destroy(pGP_plasticStrainLabel);
    VarLabel::destroy(pGP_plasticStrainLabel_preReloc);
    VarLabel::destroy(pGP_plasticEnergyLabel);
    VarLabel::destroy(pGP_plasticEnergyLabel_preReloc);
  }
  
  // Universal Deletes
  VarLabel::destroy(pEnergyLabel);
  VarLabel::destroy(pEnergyLabel_preReloc);
  VarLabel::destroy(bElBarLabel);
  VarLabel::destroy(bElBarLabel_preReloc);
  VarLabel::destroy(pDeformRateLabel);
  VarLabel::destroy(pDeformRateLabel_preReloc);
}

// // Initialization Functions //
// //////////////////////////////
// void TongeRamesh::allocateCMDataAdd(DataWarehouse* new_dw,
//                              ParticleSubset* addset,
//                              map<const VarLabel*, ParticleVariableBase*>* newState,
//                              ParticleSubset* delset,
//                              DataWarehouse* old_dw )
// {
//   // Copy the data common to all constitutive models from the particle to be 
//   // deleted to the particle to be added. 
//   // This method is defined in the ConstitutiveModel base class.

//   if(flag->d_integrator != MPMFlags::Implicit){
//     copyDelToAddSetForConvertExplicit(new_dw, delset, addset, newState);
//   } else {  // Implicit
//     ParticleVariable<Matrix3>     deformationGradient, pstress;
//     new_dw->allocateTemporary(deformationGradient,addset);
//     new_dw->allocateTemporary(pstress,            addset);
    
//     constParticleVariable<Matrix3> o_deformationGradient, o_stress;
//     new_dw->get(o_deformationGradient,lb->pDeformationMeasureLabel_preReloc, delset);
//     new_dw->get(o_stress,             lb->pStressLabel_preReloc,             delset);
    
//     ParticleSubset::iterator o,n = addset->begin();
//     for (o=delset->begin(); o != delset->end(); o++, n++) {
//       deformationGradient[*n] = o_deformationGradient[*o];
//       pstress[*n]             = o_stress[*o];
//     }
    
//     (*newState)[lb->pDeformationMeasureLabel]=deformationGradient.clone();
//     (*newState)[lb->pStressLabel]=pstress.clone();
//   }
  
//   // Copy the data local to this constitutive model from the particles to 
//   // be deleted to the particles to be added
//   ParticleSubset::iterator nPlas = addset->begin();
//   ParticleSubset::iterator nUniv = addset->begin();
  
  
//   // Plasticity
//   if(d_usePlasticity) {
//     ParticleVariable<double>      pPlasticStrain;
//     ParticleVariable<double>      pPlasticEnergy;
//     new_dw->allocateTemporary(pPlasticStrain,addset);
//     new_dw->allocateTemporary(pPlasticEnergy,addset);
    
//     constParticleVariable<double> o_pPlasticStrain;
//     constParticleVariable<double> o_pPlasticEnergy;
//     new_dw->get(o_pPlasticStrain,pPlasticStrain_label_preReloc,delset);
//     new_dw->get(o_pPlasticStrain,pPlasticEnergy_label_preReloc,delset);
    
//     ParticleSubset::iterator o;
//     for (o=delset->begin(); o != delset->end(); o++, nPlas++) {
//       pPlasticStrain[*nPlas]      = o_pPlasticStrain[*o];
//       pPlasticEnergy[*nPlas]      = o_pPlasticEnergy[*o];
//     }
    
//     (*newState)[pPlasticStrain_label] = pPlasticStrain.clone();
//     (*newState)[pPlasticEnergy_label] = pPlasticEnergy.clone();
//   } // End Plasticity
  
//   // Damage
//   if(d_useDamage) {
//     constParticleVariable<int>     o_pLocalized;
//     constParticleVariable<double>  o_pDamage;
//     new_dw->get(o_pLocalized,      pLocalizedLabel_preReloc,     delset);
//     new_dw->get(o_pDamage,         pDamageLabel_preReloc,        delset);

//     constParticleVariable<double> *o_pWingLength_array;
//     constParticleVariable<double> *o_pflawSize_array;
//     constParticleVariable<double> *o_pFlawNumber_array;

//     o_pWingLength_array = new constParticleVariable<double>[d_flawDistData.numCrackFamilies];
//     o_pflawSize_array   = new constParticleVariable<double>[d_flawDistData.numCrackFamilies];
//     o_pFlawNumber_array = new constParticleVariable<double>[d_flawDistData.numCrackFamilies];
    
//     for (int i = 0; i < d_flawDistData.numCrackFamilies; i++){
//       new_dw->get(o_pWingLength_array[i], wingLengthLabel_array_preReloc[i], delset);
//       new_dw->get(o_pFlawNumber_array[i], flawNumber_array_preReloc[i], delset);
//       new_dw->get(o_pflawSize_array[i], starterFlawSize_array_preReloc[i], delset);
//     }
    
//     ParticleVariable<Matrix3>      pBeBar;
//     ParticleVariable<double>       pFailureStrain;
//     ParticleVariable<int>          pLocalized;
//     ParticleVariable<double>       pDamage;

//     ParticleVariable<double> *pWingLength_array_new;
//     ParticleVariable<double> *pflawSize_array_new;
//     ParticleVariable<double> *pFlawNumber_array_new;
    
//     new_dw->allocateTemporary(pBeBar,         addset);
//     new_dw->allocateTemporary(pFailureStrain, addset);
//     new_dw->allocateTemporary(pLocalized,     addset);
//     new_dw->allocateTemporary(pDamage,        addset);

  
//     pWingLength_array_new = new ParticleVariable<double>[d_flawDistData.numCrackFamilies];
//     pflawSize_array_new   = new ParticleVariable<double>[d_flawDistData.numCrackFamilies];
//     pFlawNumber_array_new = new ParticleVariable<double>[d_flawDistData.numCrackFamilies];

//     for (int i = 0; i < d_flawDistData.numCrackFamilies; i++){
//       new_dw->allocateTemporary(pWingLength_array_new[i], addset);
//       new_dw->allocateTemporary(pFlawNumber_array_new[i], addset);
//       new_dw->allocateTemporary(pflawSize_array_new[i]  , addset);
//     }
  
    
//     ParticleSubset::iterator o,n     = addset->begin();
//     for (o=delset->begin(); o != delset->end(); o++, n++) {
//       pLocalized[*n]                 = o_pLocalized[*o];
//       pDamage[*n]                    = o_pDamage[*o];

//       for (int i = 0; i < d_flawDistData.numCrackFamilies; i++){
//         pWingLength_array_new[i][*n] =  o_pWingLength_array[i][*o];
//         pflawSize_array_new[i][*n]   =  o_pflawSize_array[i][*o];
//         pFlawNumber_array_new[i][*n] =  o_pFlawNumber_array[i][*o];
//       }
//     }
//     (*newState)[pLocalizedLabel]     = pLocalized.clone();
//     (*newState)[pDamageLabel]        = pDamage.clone();

//     for (int i = 0; i < d_flawDistData.numCrackFamilies; i++){
//       (*newState)[wingLengthLabel_array[i]] = pWingLength_array_new[i].clone();
//       (*newState)[starterFlawSize_array[i]] = pflawSize_array_new[i].clone();
//       (*newState)[flawNumber_array[i]]      = pFlawNumber_array_new[i].clone();
//     }

//     delete [] o_pWingLength_array;
//     delete [] o_pflawSize_array;
//     delete [] o_pFlawNumber_array;
    
//     delete [] pWingLength_array_new;
//     delete [] pflawSize_array_new;
//     delete [] pFlawNumber_array_new;
  
//   } // End Damage
  
//   // Granular Plasticity
//   if(d_useGranularPlasticity) {
//     ParticleVariable<double>      pGPJ,pGP_strain,pGP_energy;
    
//     new_dw->allocateTemporary(pGPJ,addset);
//     new_dw->allocateTemporary(pGP_strain,addset);
//     new_dw->allocateTemporary(pGP_energy,addset);
    
//     constParticleVariable<double> o_pGPJ, o_pGP_strain, o_pGP_energy;
//     new_dw->get(o_pGPJ,pGPJLabel_preReloc,delset);
//     new_dw->get(o_pGP_strain,pGP_plasticStrainLabel_preReloc,delset);
//     new_dw->get(o_pGP_energy,pGP_plasticEnergyLabel_preReloc,delset);
    
//     ParticleSubset::iterator o;
//     for (o=delset->begin(); o != delset->end(); o++, nPlas++) {
//       pGPJ[*nPlas]      = o_pGPJ[*o];
//       pGP_strain[*nPlas]= o_pGP_strain[*o];
//       pGP_energy[*nPlas]= o_pGP_energy[*o];
//     }
    
//     (*newState)[pGPJLabel] = pGPJ.clone();
//     (*newState)[pGP_plasticStrainLabel] = pGP_strain.clone();
//     (*newState)[pGP_plasticEnergyLabel] = pGP_energy.clone();
//   } // End Granular Plasticity
  
//   // Universal
//   ParticleVariable<Matrix3>        bElBar;
//   ParticleVariable<Matrix3>        pDeformRate;
//   ParticleVariable<double>         pEnergy;
//   new_dw->allocateTemporary(pEnergy, addset);
//   new_dw->allocateTemporary(pDeformRate, addset);
//   new_dw->allocateTemporary(bElBar,      addset);
  
//   constParticleVariable<Matrix3>   o_bElBar;
//   constParticleVariable<Matrix3>   o_pDeformRate;
//   constParticleVariable<double>    o_pEnergy;
//   new_dw->get(o_pEnergy, pEnergyLabel_preReloc, delset);
//   new_dw->get(o_bElBar,      bElBarLabel_preReloc,      delset);
//   new_dw->get(o_pDeformRate, pDeformRateLabel_preReloc, delset);
  
//   ParticleSubset::iterator o;
//   for (o=delset->begin(); o != delset->end(); o++, nUniv++) {
//     bElBar[*nUniv]                   = o_bElBar[*o];
//     pDeformRate[*nUniv]              = o_pDeformRate[*o];
//     pEnergy[*nUniv]                   = o_pEnergy[*o];
//   }
  
//   (*newState)[ bElBarLabel]          = bElBar.clone();
//   (*newState)[pDeformRateLabel]      = pDeformRate.clone();
//   (*newState)[ pEnergyLabel]          = pEnergy.clone();
// }

// void TongeRamesh::allocateCMDataAddRequires(Task* task,
//                                      const MPMMaterial* matl,
//                                      const PatchSet* patches,
//                                      MPMLabel*lb ) const
// {
//   const MaterialSubset* matlset = matl->thisMaterial();
//   Ghost::GhostType  gnone = Ghost::None;
  
//   // Allocate the variables shared by all constitutive models
//   // for the particle convert operation
//   // This method is defined in the ConstitutiveModel base class.

//   // Add requires local to this model
//   // Plasticity
//   if(d_usePlasticity) {
//     task->requires(Task::NewDW, pPlasticStrain_label_preReloc,matlset, gnone);
//     task->requires(Task::NewDW, pPlasticEnergy_label_preReloc,matlset, gnone);
//   }
  
//   // Damage
//   if(d_useDamage) {  
//     task->requires(Task::NewDW, pLocalizedLabel_preReloc,     matlset, gnone);
//     task->requires(Task::NewDW, pDamageLabel_preReloc,        matlset, gnone);

//     for (int i = 0; i < d_flawDistData.numCrackFamilies; i++){
//       task->requires(Task::NewDW, wingLengthLabel_array_preReloc[i], matlset, gnone);
//       task->requires(Task::NewDW, starterFlawSize_array_preReloc[i], matlset, gnone);
//       task->requires(Task::NewDW, flawNumber_array_preReloc[i], matlset, gnone);
//     }
//   }

//   // Granular Plasticity
//   if(d_useGranularPlasticity) {
//     task->requires(Task::NewDW, pGPJLabel_preReloc,matlset, gnone);
//     task->requires(Task::NewDW, pGP_plasticStrainLabel_preReloc,matlset, gnone);
//     task->requires(Task::NewDW, pGP_plasticEnergyLabel_preReloc,matlset, gnone);
//   }
  
//   // Universal
//   task->requires(Task::NewDW,bElBarLabel_preReloc,            matlset, gnone);
//   task->requires(Task::NewDW,pEnergyLabel_preReloc,            matlset, gnone);
//   if (flag->d_integrator != MPMFlags::Implicit) { // non implicit
//     addSharedRForConvertExplicit(task, matlset, patches);
//     task->requires(Task::NewDW, pDeformRateLabel_preReloc,    matlset, gnone);
//   } else { // Implicit only stuff
//     task->requires(Task::NewDW,lb->pStressLabel_preReloc,             matlset, gnone);
//     task->requires(Task::NewDW,lb->pDeformationMeasureLabel_preReloc, matlset, gnone);
//   }
// }

void TongeRamesh::carryForward(const PatchSubset* patches,
                               const MPMMaterial* matl,
                               DataWarehouse* old_dw,
                               DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    int dwi = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
    
    // Carry forward the data common to all constitutive models 
    // when using RigidMPM.
    // This method is defined in the ConstitutiveModel base class.
    carryForwardSharedData(pset, old_dw, new_dw, matl);
    
    // Carry forward the data local to this constitutive model 
    // Universal
    ParticleVariable<Matrix3> bElBar_new;
    constParticleVariable<Matrix3> bElBar;

    ParticleVariable<double> pEnergy_new;
    constParticleVariable<double> pEnergy_old;
    old_dw->get(pEnergy_old, pEnergyLabel, pset);
    new_dw->allocateAndPut(pEnergy_new, pEnergyLabel_preReloc, pset);
    pEnergy_new.copyData(pEnergy_old);
    
    old_dw->get(bElBar,                bElBarLabel,                    pset);
    new_dw->allocateAndPut(bElBar_new, bElBarLabel_preReloc,           pset);
    for(ParticleSubset::iterator iter = pset->begin();
        iter != pset->end(); iter++){
      particleIndex idx = *iter;
      bElBar_new[idx] = bElBar[idx];
    }
    if (flag->d_integrator != MPMFlags::Implicit) {
      ParticleVariable<Matrix3> pDeformRate;
      new_dw->allocateAndPut(pDeformRate,   pDeformRateLabel_preReloc, pset);
      pDeformRate.copyData(bElBar);
    }
    
    // Plasticity
    if(d_usePlasticity) {
      ParticleVariable<double> pPlasticStrain, pPlasticEnergy;
      constParticleVariable<double> pPlasticStrain_old, pPlasticEnergy_old;
      old_dw->get(pPlasticStrain_old,         pPlasticStrain_label,              pset);
      new_dw->allocateAndPut(pPlasticStrain,  pPlasticStrain_label_preReloc,     pset);
      pPlasticStrain.copyData(pPlasticStrain_old);
      
      old_dw->get(pPlasticEnergy_old,         pPlasticEnergy_label,              pset);
      new_dw->allocateAndPut(pPlasticEnergy,  pPlasticEnergy_label_preReloc,     pset);
      pPlasticEnergy.copyData(pPlasticEnergy_old);
    } // End Plasticity
    
    // Damage
    if(d_useDamage) {
      constParticleVariable<int>     pLocalized;
      constParticleVariable<double>  pDamage;
      ParticleVariable<int>          pLocalized_new;
      ParticleVariable<double>       pDamage_new;
      
      constParticleVariable<double> *o_pWingLength_array;
      constParticleVariable<double> *o_pflawSize_array;
      constParticleVariable<double> *o_pFlawNumber_array;

      ParticleVariable<double> *pWingLength_array_new;
      ParticleVariable<double> *pflawSize_array_new;
      ParticleVariable<double> *pFlawNumber_array_new;

      o_pWingLength_array = new constParticleVariable<double>[d_flawDistData.numCrackFamilies];
      o_pflawSize_array   = new constParticleVariable<double>[d_flawDistData.numCrackFamilies];
      o_pFlawNumber_array = new constParticleVariable<double>[d_flawDistData.numCrackFamilies];

      pWingLength_array_new = new ParticleVariable<double>[d_flawDistData.numCrackFamilies];
      pflawSize_array_new   = new ParticleVariable<double>[d_flawDistData.numCrackFamilies];
      pFlawNumber_array_new = new ParticleVariable<double>[d_flawDistData.numCrackFamilies];

      old_dw->get(pLocalized,     pLocalizedLabel,                 pset);
      old_dw->get(pDamage,        pDamageLabel,                    pset);

      for (int i = 0; i < d_flawDistData.numCrackFamilies; i++){
        old_dw->get(o_pWingLength_array[i], wingLengthLabel_array[i], pset);
        old_dw->get(o_pflawSize_array[i],   starterFlawSize_array[i], pset);
        old_dw->get(o_pFlawNumber_array[i], flawNumber_array[i], pset);
      }
      
      new_dw->allocateAndPut(pLocalized_new,      
                             pLocalizedLabel_preReloc,             pset);
      new_dw->allocateAndPut(pDamage_new,      
                             pDamageLabel_preReloc,                pset);      

      for (int i = 0; i < d_flawDistData.numCrackFamilies; i++){
        new_dw->allocateAndPut(pWingLength_array_new[i], wingLengthLabel_array_preReloc[i], pset);
        new_dw->allocateAndPut(pflawSize_array_new[i],   starterFlawSize_array_preReloc[i], pset);
        new_dw->allocateAndPut(pFlawNumber_array_new[i], flawNumber_array_preReloc[i], pset);
      }
            
      pLocalized_new.copyData(pLocalized);
      pDamage_new.copyData(pDamage);

      for (int i = 0; i < d_flawDistData.numCrackFamilies; i++){
        pWingLength_array_new[i].copyData(o_pWingLength_array[i]);
        pflawSize_array_new[i].copyData(o_pflawSize_array[i]);
        pFlawNumber_array_new[i].copyData(o_pFlawNumber_array[i]);
      }

      delete [] o_pWingLength_array;
      delete [] o_pflawSize_array;
      delete [] o_pFlawNumber_array;
    
      delete [] pWingLength_array_new;
      delete [] pflawSize_array_new;
      delete [] pFlawNumber_array_new;

    } // End damage

    // Granular Plasticity
    if(d_useGranularPlasticity) {
      ParticleVariable<double> pGPJ, pGP_strain, pGP_energy;
      constParticleVariable<double> pGPJ_old, pGP_strain_old, pGP_energy_old;
      old_dw->get(pGPJ_old,         pGPJLabel,              pset);
      new_dw->allocateAndPut(pGPJ,  pGPJLabel_preReloc,     pset);
      pGPJ.copyData(pGPJ_old);

      old_dw->get(pGP_strain_old,         pGP_plasticStrainLabel,          pset);
      new_dw->allocateAndPut(pGP_strain,  pGP_plasticStrainLabel_preReloc, pset);
      pGP_strain.copyData(pGP_strain_old);

      old_dw->get(pGP_energy_old,         pGP_plasticEnergyLabel,          pset);
      new_dw->allocateAndPut(pGP_energy,  pGP_plasticEnergyLabel_preReloc, pset);
      pGP_energy.copyData(pGP_energy_old);
    } // End Granular Plasticity
    
    new_dw->put(delt_vartype(1.e10), lb->delTLabel, patch->getLevel());
    if (flag->d_reductionVars->accStrainEnergy ||
        flag->d_reductionVars->strainEnergy) {
      new_dw->put(sum_vartype(0.),     lb->StrainEnergyLabel);
    }
  } // End Particle Loop
}

void TongeRamesh::initializeCMData(const Patch* patch,
                                   const MPMMaterial* matl,
                                   DataWarehouse* new_dw)
{
  // Initialize the variables shared by all constitutive models
  // This method is defined in the ConstitutiveModel base class.
  if (flag->d_integrator == MPMFlags::Implicit) 
    initSharedDataForImplicit(patch, matl, new_dw);
  else {
    initSharedDataForExplicit(patch, matl, new_dw);
  }
  // Put stuff in here to initialize each particle's
  // constitutive model parameters and deformationMeasure
  Matrix3 Identity;
  Identity.Identity();
  Matrix3 zero(0.0);
  
  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);

  ParticleSubset::iterator iterUniv = pset->begin(); 
  ParticleSubset::iterator iterPlas = pset->begin();

  
  // Plasticity
  if(d_usePlasticity) {
    ParticleVariable<double> pPlasticStrain, pPlasticEnergy;
    
    new_dw->allocateAndPut(pPlasticStrain, pPlasticStrain_label,  pset);
    new_dw->allocateAndPut(pPlasticEnergy, pPlasticEnergy_label,  pset);
    
    for(;iterPlas != pset->end(); iterPlas++){
      pPlasticStrain[*iterPlas] = d_initialData.Alpha;
      pPlasticEnergy[*iterPlas] = 0.0;
    }
  }
  
  // Damage
  if(d_useDamage) {
    // cout << "In TongeRamesh::initializeCMData() if(d_useDamage)" << endl;
    // Initalize the random number generator using the patch number and the seed that
    // was provided in the input.
    // Make the seed differ for each patch, otherwise each patch gets the
    // same set of random #s.
    unsigned int patchID = patch->getID();
    // int patch_div_32 = patchID/32;
    // patchID = patchID%32;
    // unsigned int unique_seed = ((d_flawDistData.randomSeed+patch_div_32+1) << patchID);
    // MusilRNG* randGen = scinew MusilRNG(unique_seed);
    unsigned long bigSeed[2];
    bigSeed[0] = patchID;
    bigSeed[1] = d_flawDistData.randomSeed;
    MTRand randGen(bigSeed,2);

    ParticleVariable<int>         pLocalized;
    constParticleVariable<double> pVolume;
    constParticleVariable<Point>   px;
    ParticleVariable<double>      pDamage;

    ParticleVariable<double> *pWingLength_array_new;
    ParticleVariable<double> *pflawSize_array_new;
    ParticleVariable<double> *pFlawNumber_array_new;

    pWingLength_array_new = new ParticleVariable<double>[d_flawDistData.numCrackFamilies];
    pflawSize_array_new   = new ParticleVariable<double>[d_flawDistData.numCrackFamilies];
    pFlawNumber_array_new = new ParticleVariable<double>[d_flawDistData.numCrackFamilies];

    new_dw->get(pVolume,                   lb->pVolumeLabel,            pset);
    new_dw->get(px,                        lb->pXLabel,                 pset); // This might need a new initial requires statement
    new_dw->allocateAndPut(pLocalized,     pLocalizedLabel,             pset);
    new_dw->allocateAndPut(pDamage,        pDamageLabel,                pset);

    for (int i = 0; i < d_flawDistData.numCrackFamilies; i++){
      new_dw->allocateAndPut(pWingLength_array_new[i], wingLengthLabel_array[i], pset);
      new_dw->allocateAndPut(pflawSize_array_new[i],   starterFlawSize_array[i], pset);
      new_dw->allocateAndPut(pFlawNumber_array_new[i], flawNumber_array[i], pset);
    }

    // If defining flaw data using a field, read that information:
    std::vector<Vector> eta_waveVectors;
    std::vector<double> eta_Amplitude;
    std::vector<double> eta_Phase;
    if(d_flawDistData.useEtaField) {
      string filename(d_flawDistData.etaFilename);
      std::ifstream is(filename.c_str());
      if (!is ){
        throw ProblemSetupException("ERROR Opening flaw density specification file '"+filename+"'\n",
                                    __FILE__, __LINE__);
      }
      string line;

      while(std::getline(is,line)) {
        std::istringstream iss(line);
        double e1,e2,e3,A,phi;
        iss >> e1 >> e2 >> e3 >> A >> phi;
        if(!iss){
          throw ProblemSetupException("ERROR reading flaw density input line:\n'"+line+"'\n",
                                      __FILE__, __LINE__);
        }
        eta_waveVectors.push_back(Vector(e1,e2,e3));
        eta_Amplitude.push_back(A);
        eta_Phase.push_back(phi);
      }
    }

    std::vector<Vector> size_waveVectors;
    std::vector<double> size_Amplitude;
    std::vector<double> size_Phase;
    if(d_flawDistData.useSizeField) {
      string filename(d_flawDistData.sizeFilename);
      std::ifstream is(filename.c_str());
      if (!is ){
        throw ProblemSetupException("ERROR Opening flaw density specification file '"+filename+"'\n",
                                    __FILE__, __LINE__);
      }
      string line;

      while(std::getline(is,line)) {
        std::istringstream iss(line);
        double e1,e2,e3,A,phi;
        iss >> e1 >> e2 >> e3 >> A >> phi;
        if(!iss){
          throw ProblemSetupException("ERROR reading flaw density input line:\n'"+line+"'\n",
                                      __FILE__, __LINE__);
        }
        size_waveVectors.push_back(Vector(e1,e2,e3));
        size_Amplitude.push_back(A);
        size_Phase.push_back(phi);
      }
    }
    // End of reading fourier field information
    
    
    ParticleSubset::iterator iter = pset->begin();

    for(; iter != pset->end(); iter++){
      double eta_shift(0);
      double size_shift(0);
      Vector pos(px[*iter]);

      // Compute the shift in the the flaw density:
      for(size_t i = 0; i < eta_Phase.size(); ++i){
        double x(Dot(eta_waveVectors[i],pos));
        eta_shift += eta_Amplitude[i] * sin(x + eta_Phase[i]);
      }
      // Make sure there are no negitive flaw densities:
      eta_shift = (d_flawDistData.flawDensity + eta_shift)<0 ?
        (1e-8-1.0)*d_flawDistData.flawDensity : eta_shift;
      // Compute the shift in the flaw size:
      for(size_t i = 0; i < size_Phase.size(); ++i){
        double x(Dot(size_waveVectors[i],pos));
        size_shift += size_Amplitude[i] * sin(x + size_Phase[i]);
      }
      
      if( d_flawDistData.type == "normal"){
        // Switch this based on the desired internal flaw distribution:
        // For a normal distribution
        double sd = d_flawDistData.stdFlawSize;
        double smin(d_flawDistData.minFlawSize + size_shift);
        smin = smin < 0 ? 0 : smin;
        double smax(d_flawDistData.maxFlawSize + size_shift);

        double ln(smax-smin); // Size of the sampling interval
        double binWidth(ln/d_flawDistData.numCrackFamilies);
        double pdfValue(0.0);
        double mean(d_flawDistData.meanFlawSize + size_shift);
        double s(mean);
        double eta(d_flawDistData.flawDensity + eta_shift);
        // Generate a normal distribution of internal flaws:
        for (int i = 0; i < d_flawDistData.numCrackFamilies; i++){
          // Calculate the flaw size for the bin:
          s = smax - (i+0.5)*binWidth;
          if(s > 0.0){
            // Calculate the flaw denstiy in the bin:
            pdfValue = exp(-(s-mean)*(s-mean)*0.5/(sd*sd))/(sd*sqrt(2*PI));

            // Assign values:
            pWingLength_array_new[i][*iter] = 0.0;
            pFlawNumber_array_new[i][*iter] = eta*pdfValue*binWidth;
            pflawSize_array_new[i][*iter] = s;
          } else {
            pWingLength_array_new[i][*iter] = 0.0;
            pFlawNumber_array_new[i][*iter] = 0.0;
            pflawSize_array_new[i][*iter] = smin;
          }
        } // End loop through flaw families
      } else       if( d_flawDistData.type == "normal"){
        // Switch this based on the desired internal flaw distribution:
        // For a normal distribution
        double sd = d_flawDistData.stdFlawSize;
        double smin(d_flawDistData.minFlawSize + size_shift);
        smin = smin < 0 ? 0 : smin;
        double smax(d_flawDistData.maxFlawSize + size_shift);

        double ln(smax-smin); // Size of the sampling interval
        double binWidth(ln/d_flawDistData.numCrackFamilies);
        double pdfValue(0.0);
        double mean(d_flawDistData.meanFlawSize + size_shift);
        double s(mean);
        double eta(d_flawDistData.flawDensity + eta_shift);
        // Generate a normal distribution of internal flaws:
        for (int i = 0; i < d_flawDistData.numCrackFamilies; i++){
          // Calculate the flaw size for the bin:
          s = smax - (i+0.5)*binWidth;
          pdfValue = logNormal_PDF(s, sd, mean);
          pdfValue = pdfValue < 0 ? 0 : pdfValue;

          // Assign values:
          pWingLength_array_new[i][*iter] = 0.0;
          pFlawNumber_array_new[i][*iter] = eta*pdfValue*binWidth;
          pflawSize_array_new[i][*iter] = s;
        } // End loop through flaw families
      } else if(d_flawDistData.type == "pareto"){
        if(d_flawDistData.randomizeDist){

          switch (d_flawDistData.randomMethod)
            {
            case 0:
              {
                // All bins have the same probability (weight). The bin centers
                // are chosen by dividing up the Cumulinitive density function
                // and selecting one value from reach segment (for 5 bins it would be
                // 5 values from 0-.2, .2-.4, .4-.6, .6-.8, .8-1.0) and then using
                // the inverse maping technique. This will reproduce the parent distribution,
                // but it tends to produce a few pockets where there is a very high density of
                // large cracks and this is probably not physical.
                double smin = d_flawDistData.minFlawSize;
                double smax = d_flawDistData.maxFlawSize;
                double eta = d_flawDistData.flawDensity;
                double a = d_flawDistData.exponent;
                double s;

                double meanEta = eta/d_flawDistData.numCrackFamilies;
                double stdEta = sqrt(meanEta/pVolume[*iter]);
            
                // Generate a Pareto distribution of internal flaws:
                for (int i = 0; i < d_flawDistData.numCrackFamilies; i++){
                  // Calculate the flaw size for the bin:
                  double fmin = (d_flawDistData.numCrackFamilies - i - 1)
                    /d_flawDistData.numCrackFamilies;
                  double fmax = (d_flawDistData.numCrackFamilies - i)
                    /d_flawDistData.numCrackFamilies;
                  double U = fmin+(fmax-fmin) * randGen.rand53();
                  s = pow(-((U-1)*pow(smax,a)-(U*pow(smin,a))) / (pow(smin*smax, a)), -1.0/a);

                  double x = randGen.rand53();
                  double y = randGen.rand53();
                  double Z = sqrt(-2*log(x))*cos(2*M_PI*y);

                  // Transform standard normal Z to the distribution that I need:
                  double flawNum = Z*stdEta+meanEta; // Select from a normal dist centered at meanEta
                  flawNum = flawNum<0 ? 0 : flawNum; // Do not allow negitive flaw number densties
            
                  // Assign values:
                  pWingLength_array_new[i][*iter] = 0.0;
                  pFlawNumber_array_new[i][*iter] = flawNum;
                  pflawSize_array_new[i][*iter] = s;
                } // End loop through flaw families
                break;
              }
            case 1:
              {
                // Use bins that all have the same width in terms of s, but choose
                // s as a uniformly distributed value within the width of the bin.
                // The height of the bin still comes from the value that would be given
                // if s were located at the midpoint of the bin. Flaw densities are not
                // treated as a stoichastic quantity.
                double smin = d_flawDistData.minFlawSize;
                double smax = d_flawDistData.maxFlawSize;
                double ln   = smax-smin;
                double binWidth = ln/d_flawDistData.numCrackFamilies;
                double pdfValue = 0.0;
                double eta = d_flawDistData.flawDensity;
                double a = d_flawDistData.exponent;
                double s_mid;
                double s;
                                        
                // Generate a Pareto distribution of internal flaws:
                for (int i = 0; i < d_flawDistData.numCrackFamilies; i++){
                  // Calculate the flaw size for the bin:
                  s_mid = smax -(i + 0.5)*binWidth;
                  s = smax - (i + randGen.rand53())*binWidth;
                  // Calculate the flaw denstiy in the bin:
                  pdfValue = a * pow(smin, a) * pow(s_mid, -a-1.0) / (1-pow(smin/smax,a));
            
                  // Assign values:
                  pWingLength_array_new[i][*iter] = 0.0;
                  pFlawNumber_array_new[i][*iter] = eta*pdfValue*binWidth;
                  pflawSize_array_new[i][*iter] = s;
                } // End loop through flaw families
              
                break;
              }
            case 2:
              {
                // Use bins that all have the same width in terms of s, but choose
                // s as a uniformly distributed value within the width of the bin.
                // The height of the bin is sampled using a normal approximation
                // to a poisson distribution
                double smin = d_flawDistData.minFlawSize;
                double smax = d_flawDistData.maxFlawSize;
                double ln   = smax-smin;
                double binWidth = ln/d_flawDistData.numCrackFamilies;
                double pdfValue = 0.0;
                double eta = d_flawDistData.flawDensity;
                double a = d_flawDistData.exponent;
                double /*s_mid,*/ meanEta, stdEta;
                double s;

                // Generate a Pareto distribution of internal flaws:
                for (int i = 0; i < d_flawDistData.numCrackFamilies; i++){
                  // Calculate the flaw size for the bin:
                  //s_mid = smax - (i+0.5)*binWidth;
                  s = smax - (i+randGen.rand53())*binWidth;
                  // Calculate the flaw denstiy in the bin:
                  pdfValue = a * pow(smin, a) * pow(s, -a-1.0) / (1-pow(smin/smax,a));
                  meanEta = eta*pdfValue*binWidth;
                  stdEta = sqrt(meanEta/pVolume[*iter]);
                
                  double x = randGen.rand53();
                  double y = randGen.rand53();
                  double Z = sqrt(-2*log(x))*cos(2*M_PI*y);

                  // Transform standard normal Z to the distribution that I need:
                  double flawNum = Z*stdEta+meanEta; // Select from a normal dist centered at meanEta
                  flawNum = flawNum<0 ? 0 : flawNum; // Do not allow negitive flaw number densties

                  // Assign values:
                  pWingLength_array_new[i][*iter] = 0.0;
                  pFlawNumber_array_new[i][*iter] = flawNum;
                  pflawSize_array_new[i][*iter] = s;
                } // End loop through flaw families
              
                break;
              }
            case 3:
              // Assign bin sizes so that each successive bin has 2x as many flaws.
              // The flaw size within the bin is selected randomly using inverse
              // sampeling. The flaw densities within the bin are deterministic.
              {
                double smin = d_flawDistData.minFlawSize;
                double smax = d_flawDistData.maxFlawSize;
                double a = d_flawDistData.exponent;
                
                int numFlaws = 1;
                double u_cur, u_old, delU_cur, delU_old;
                double invVol = 1.0/pVolume[*iter];
                double invVol_Eta = invVol/d_flawDistData.flawDensity;

                delU_cur = invVol_Eta;
                u_cur = 1-0.5*delU_cur;

                pWingLength_array_new[0][*iter] = 0.0;
                pFlawNumber_array_new[0][*iter] = invVol;
                pflawSize_array_new[0][*iter] =
                  pareto_invCDF( u_cur + ( randGen.rand53()- 0.5 ) * delU_cur,
                                 smin, smax, a);
               
                for(int i = 1; i<d_flawDistData.numCrackFamilies; ++i){
                  // Copy current to old:
                  delU_old = delU_cur;
                  u_old = u_cur;

                  numFlaws *= 2;
                  delU_cur = numFlaws*invVol_Eta;
                  u_cur = u_old - 0.5*(delU_cur + delU_old);

                  pWingLength_array_new[i][*iter] = 0.0;
                  pFlawNumber_array_new[i][*iter] = numFlaws*invVol;
                  pflawSize_array_new[i][*iter] =
                    pareto_invCDF( u_cur + ( randGen.rand53()- 0.5 ) * delU_cur,
                                   smin, smax, a);
                }
                break;
              }
            case 4:
              // Assign bin sizes so that each successive bin has 2x as many flaws.
              // The flaw size within the bin is selected randomly using inverse
              // sampeling. The flaw densities within the bin are deterministic.
              {
                double smin = d_flawDistData.minFlawSize;
                double smax = d_flawDistData.maxFlawSize;
                double a = d_flawDistData.exponent;
                                
                int numFlaws = 1;
                double u_cur, delU_cur, s_l, s_h;
                double invVol = 1.0/pVolume[*iter];
                double invVol_Eta = invVol/d_flawDistData.flawDensity;

                delU_cur = invVol_Eta;
                u_cur = 1-delU_cur; // CDF value at the minimum flaw size for the bin.
                s_h = smax;    // Upper bound on flaw size for the bin.
                s_l = pareto_invCDF( u_cur, smin, smax, a); // Lower bound on flaw size for the bin
                
                pWingLength_array_new[0][*iter] = 0.0;
                pFlawNumber_array_new[0][*iter] = invVol;
                // Rescale the distribution to achieve more accurate sampeling from within
                // the bin.
                pflawSize_array_new[0][*iter] = pareto_invCDF( randGen.rand53(), s_l, s_h, a);
                
                for(int i = 1; i<d_flawDistData.numCrackFamilies; ++i){
                  // Copy current to old:
                  s_h = s_l;

                  numFlaws *= 2;
                  delU_cur = numFlaws*invVol_Eta;
                  u_cur -= delU_cur;
                  s_l = pareto_invCDF( u_cur, smin, smax, a);

                  pWingLength_array_new[i][*iter] = 0.0;
                  pFlawNumber_array_new[i][*iter] = numFlaws*invVol;
                  pflawSize_array_new[i][*iter] = pareto_invCDF( randGen.rand53(), s_l, s_h, a);
                }
                break;
              }
            case 5:
              {
                // Use uniform bin sizes. For each bin calculate the mean number of flaws in the
                // bin. If that number is greater than poissonThreshold then use a
                // Gaussian approximation to a Poisson distribution for the number
                // of flaws in the bin and use the first moment of the Pareto
                // distribution within the bin to compute the representitive flaw size,
                // otherwise compute the number of flaws in the bin from a Poisson
                // distribution and then explicitly simulate that number of flaws
                // and compute the mean.
                double s_min = d_flawDistData.minFlawSize;
                double s_max = d_flawDistData.maxFlawSize;
                double ln   = s_max-s_min;
                double binWidth = ln/d_flawDistData.numCrackFamilies;
                double eta = d_flawDistData.flawDensity;
                double a = d_flawDistData.exponent;
                double s, s_l(s_max), s_h(s_max), meanBinFlaws, binOmega;
                double poissonThreshold = d_flawDistData.Ncutoff;
                                        
                // Generate a Pareto distribution of internal flaws:
                for (int i = 0; i < d_flawDistData.numCrackFamilies; i++){
                  s_h = s_l;
                  s_l = s_h-binWidth;
                  meanBinFlaws = eta*pVolume[*iter]*
                    (pareto_CDF(s_h,s_min,s_max,a)-pareto_CDF(s_l,s_min,s_max,a));
                  if (meanBinFlaws >= poissonThreshold){
                    // Make s a deterministic value. This assumption could be relaxed,
                    // and s could be drawn from an approapreate normal distribution,
                    // but this is ok for now. These are the smallest flaws and most
                    // numerious flaws in the system so it is ok if they have a
                    // deterministic size.
                    
                    s = pareto_FirstMoment(s_l,s_h,a);
                    binOmega = floor(randGen.randNorm(meanBinFlaws,sqrt(meanBinFlaws))+0.5)/
                      pVolume[*iter];
                  } else {
                    // Explicitly sample a Poisson distribution then explicitly calculate
                    // the mean of the number of flaws in the bin
                    double binFlaws = poisson_sample(randGen,meanBinFlaws);
                    if(binFlaws>0){
                      double sum_s = 0;
                      for(unsigned int j=0;j< binFlaws; ++j){
                        sum_s += pareto_invCDF(randGen.rand53(),s_l,s_h,a);
                      }
                      s = sum_s/binFlaws;
                      binOmega = binFlaws/pVolume[*iter];
                    } else {
                      s = 0.5*(s_l + s_h);
                      binOmega = 0.0;
                    }
                  }
                    
                  // Assign values:
                  pWingLength_array_new[i][*iter] = 0.0;
                  pFlawNumber_array_new[i][*iter] = binOmega;
                  pflawSize_array_new[i][*iter] = s;
                } // End loop through flaw families

                break;
              }
            case 6:
              {
                // This is identical to case 5 except the representitive
                // flaw size, when the Gaussian approximation to a Poisson
                // distribution is used, is a random variable. The argument
                // for this is the central limit. The sample mean for a collection
                // of n iid random variables is normally distributed with a mean
                // equal to the population mean and variance equal to the population
                // variance divided by the number of samples.
                double s_min = d_flawDistData.minFlawSize;
                double s_max = d_flawDistData.maxFlawSize;
                double ln   = s_max-s_min;
                double binWidth = ln/d_flawDistData.numCrackFamilies;
                double eta = d_flawDistData.flawDensity;
                double a = d_flawDistData.exponent;
                double s, s_l(s_max), s_h(s_max), meanBinFlaws, binOmega, binFlaws;
                double poissonThreshold = 20;
                double del_xi, xi_l;
                bool do_BinBias = fabs(d_flawDistData.binBias-1)>1e-8;
                bool useRationalBias = d_flawDistData.binBias<0;
                double biasExp = d_flawDistData.binBias;
                if (useRationalBias){
                  xi_l = 1.0;
                  biasExp = -d_flawDistData.binBias;
                  del_xi = (pow(s_max/s_min, 1/biasExp)-1.0)
                    /d_flawDistData.numCrackFamilies;
                } else {
                  xi_l = 1.0;
                  del_xi = 1.0/d_flawDistData.numCrackFamilies;
                }
                                        
                // Generate a Pareto distribution of internal flaws:
                for (int i = 0; i < d_flawDistData.numCrackFamilies; i++){
                  s_h = s_l;
                  if (useRationalBias){
                    xi_l += del_xi;
                    s_l = s_max/pow(xi_l,biasExp);
                  } else if(do_BinBias){
                    xi_l -= del_xi;
                    s_l = pow(xi_l,biasExp)*ln + s_min;
                  } else {
                    s_l = s_h-binWidth;
                  }

                  meanBinFlaws = eta*pVolume[*iter]*
                    (pareto_CDF(s_h,s_min,s_max,a)-pareto_CDF(s_l,s_min,s_max,a));
                  if (meanBinFlaws >= poissonThreshold){
                    binFlaws = floor(randGen.randNorm(meanBinFlaws,sqrt(meanBinFlaws))+0.5);
                  } else {
                    // Explicitly sample a Poisson distribution then explicitly calculate
                    // the mean of the number of flaws in the bin
                    binFlaws = poisson_sample(randGen,meanBinFlaws);
                  }
                  binOmega = binFlaws/pVolume[*iter];

                  if (meanBinFlaws >= poissonThreshold){
                    double s_mean = pareto_FirstMoment(s_l,s_h,a);
                    double s_std = sqrt(pareto_variance(s_l,s_h,a)/binFlaws);
                    // Make sure that the simulated s lies within the bin, if not choose
                    // another sample
                    int j = 0;
                    do{
                      s = randGen.randNorm(s_mean,s_std);
                      ++j;
                    } while ( (s>s_h || s<s_l) && j<10 );
                    if (j>=10){
                      cerr<< "TongeRamesh::InitalizeCMData() bin method 6. "
                          << "Tries for random s exceeded 10 accepting mean for value of s"
                          << endl;
                      s = s_mean;
                    }
                  } else if(binFlaws>0){
                    double sum_s = 0;
                    for(unsigned int j=0;j< binFlaws; ++j){
                      sum_s += pareto_invCDF(randGen.rand53(),s_l,s_h,a);
                    }
                    s = sum_s/binFlaws;

                  } else {
                    s = 0.5*(s_l + s_h);
                    binOmega = 0.0;
                  }

                  // Assign values:
                  pWingLength_array_new[i][*iter] = 0.0;
                  pFlawNumber_array_new[i][*iter] = binOmega;
                  pflawSize_array_new[i][*iter] = s;
                } // End loop through flaw families

                break;
              }
            case 7:
              {
                // This is identical to case 6 except the value of s_k is
                // the cube root of the mean value of the flaw sizes cubed.
                double s_min = d_flawDistData.minFlawSize;
                double s_max = d_flawDistData.maxFlawSize;
                double ln   = s_max-s_min;
                double binWidth = ln/d_flawDistData.numCrackFamilies;
                double eta = d_flawDistData.flawDensity;
                double a = d_flawDistData.exponent;
                double s, s_l(s_max), s_h(s_max), meanBinFlaws, binOmega, binFlaws;
                double poissonThreshold = 20;
                double del_xi, xi_l;
                bool do_BinBias = fabs(d_flawDistData.binBias-1)>1e-8;
                bool useRationalBias = d_flawDistData.binBias<0;
                double biasExp = d_flawDistData.binBias;
                if (useRationalBias){
                  xi_l = 1.0;
                  biasExp = -d_flawDistData.binBias;
                  del_xi = (pow(s_max/s_min, 1/biasExp)-1.0)
                    /d_flawDistData.numCrackFamilies;
                } else {
                  xi_l = 1.0;
                  del_xi = 1.0/d_flawDistData.numCrackFamilies;
                }
                                        
                // Generate a Pareto distribution of internal flaws:
                for (int i = 0; i < d_flawDistData.numCrackFamilies; i++){
                  s_h = s_l;
                  if (useRationalBias){
                    xi_l += del_xi;
                    s_l = s_max/pow(xi_l,biasExp);
                  } else if(do_BinBias){
                    xi_l -= del_xi;
                    s_l = pow(xi_l,biasExp)*ln + s_min;
                  } else {
                    s_l = s_h-binWidth;
                  }

                  meanBinFlaws = eta*pVolume[*iter]*
                    (pareto_CDF(s_h,s_min,s_max,a)-pareto_CDF(s_l,s_min,s_max,a));
                  if (meanBinFlaws >= poissonThreshold){
                    binFlaws = floor(randGen.randNorm(meanBinFlaws,sqrt(meanBinFlaws))+0.5);
                  } else {
                    // Explicitly sample a Poisson distribution then explicitly calculate
                    // the mean of the number of flaws in the bin
                    binFlaws = poisson_sample(randGen,meanBinFlaws);
                  }
                  binOmega = binFlaws/pVolume[*iter];

                  if (binFlaws >= poissonThreshold){
                    double s3_mean = pareto_ThirdMoment(s_l,s_h,a);
                    double s3_std = sqrt(pareto_SixthMoment(s_l,s_h,a)/binFlaws);
                    double s3(s3_mean);
                    // Make sure that the simulated s lies within the bin, if not choose
                    // another sample
                    int j = 0;
                    do{
                      s3 = randGen.randNorm(s3_mean,s3_std);
                      ++j;
                    } while ( ( s3>(s_h*s_h*s_h) || s3<(s_l*s_l*s_l) ) && j<10 );
                    if (j>=10){
                      cerr<< "TongeRamesh::InitalizeCMData() bin method 7. "
                          << "Tries for random s3 exceeded 10 accepting mean for value of s3"
                          << endl;
                      s3 = s3_mean;
                    }
                    s = pow(s3, 1.0/3.0);
                  } else if(binFlaws>0){
                    double sum_s = 0;
                    for(unsigned int j=0;j< binFlaws; ++j){
                      double ai(pareto_invCDF(randGen.rand53(),s_l,s_h,a));
                      sum_s += ai*ai*ai;
                    }
                    s = pow(sum_s/binFlaws, 1.0/3.0);

                  } else {
                    s = 0.5*(s_l + s_h);
                    binOmega = 0.0;
                  }

                  // Assign values:
                  pWingLength_array_new[i][*iter] = 0.0;
                  pFlawNumber_array_new[i][*iter] = binOmega;
                  pflawSize_array_new[i][*iter] = s;
                } // End loop through flaw families

                break;
              }
            default:
              throw ProblemSetupException("Unknown bin selection method", __FILE__, __LINE__);
            }
        } else {
          double smin = d_flawDistData.minFlawSize + size_shift;
          double smax = d_flawDistData.maxFlawSize + size_shift;
          if (smax <=0 ) {
            ostringstream desc;
            desc << "computed an illegal value for smax: " << smax << "\n"
                 << "The initial value was: "<< d_flawDistData.maxFlawSize
                 << " and the shift is: " << size_shift
                 << endl;
            throw ProblemSetupException(desc.str(), __FILE__, __LINE__);
          }
          smin = smin<0 ? 1e-8 * smax : smin;
          double ln   = smax-smin;
          double binWidth = ln/d_flawDistData.numCrackFamilies;
          double pdfValue = 0.0;
          double eta = d_flawDistData.flawDensity + eta_shift;
          double a = d_flawDistData.exponent;
          double s;
            
          // Generate a Pareto distribution of internal flaws:
          for (int i = 0; i < d_flawDistData.numCrackFamilies; i++){
            // Calculate the flaw size for the bin:
            s = smax - (i+0.5)*binWidth;

            // Calculate the flaw denstiy in the bin:
            pdfValue = a * pow(smin, a) * pow(s, -a-1.0) / (1-pow(smin/smax,a));
            
            // Assign values:
            pWingLength_array_new[i][*iter] = 0.0;
            pFlawNumber_array_new[i][*iter] = eta*pdfValue*binWidth;
            pflawSize_array_new[i][*iter] = s;
          } // End loop through flaw families
        }
      } else if(d_flawDistData.type == "delta"){
        double s = d_flawDistData.meanFlawSize + size_shift;
        double N = (d_flawDistData.flawDensity + eta_shift) / d_flawDistData.numCrackFamilies;

        for (int i = 0; i < d_flawDistData.numCrackFamilies; i++){
          pWingLength_array_new[i][*iter] = 0.0;
          pFlawNumber_array_new[i][*iter] = N;
          pflawSize_array_new[i][*iter] = s;
        } // End loop through flaw families
      }

      // Assign values common to all distribution types:
      if(d_brittle_damage.incInitialDamage){
        double damage = 0;
        for (int i = 0; i < d_flawDistData.numCrackFamilies; i++){
          double damageInc = pflawSize_array_new[i][*iter];
          damageInc *= pflawSize_array_new[i][*iter];
          damageInc *= pflawSize_array_new[i][*iter];
          damageInc *= pFlawNumber_array_new[i][*iter];
          damage += damageInc;
        }
        // Make sure the I do not assign an illegal initial damage
        damage = damage > d_brittle_damage.maxDamage ? d_brittle_damage.maxDamage : damage;
        pDamage[*iter] = damage;
      } else {
        pDamage[*iter]    = 0.0;
      }
      pLocalized[*iter] = 0;

    } // End of particle loop
    // cout << "\t End Particle loop" << endl;
    
    delete [] pWingLength_array_new;
    delete [] pflawSize_array_new;
    delete [] pFlawNumber_array_new;

    // delete randGen;
    if(d_brittle_damage.useNonlocalDamage){
      // loop through the nodes and initalize the damage to 0;     
      NCVariable<double> gDamage;
      new_dw->allocateAndPut(gDamage, gDamage_Label, matl->getDWIndex(), patch);
      gDamage.initialize(0);
    }
  } // End if(d_useDamage)

  // Granular Plasticity
  if(d_useGranularPlasticity){
    ParticleVariable<double> pGPJ, pGP_strain, pGP_energy;
    
    new_dw->allocateAndPut(pGPJ, pGPJLabel,  pset);
    new_dw->allocateAndPut(pGP_strain, pGP_plasticStrainLabel,  pset);
    new_dw->allocateAndPut(pGP_energy, pGP_plasticEnergyLabel,  pset);
    
    for(iterPlas = pset->begin() ;iterPlas != pset->end(); iterPlas++){
      pGPJ[*iterPlas] = 1.0;
      pGP_strain[*iterPlas] = 0.0;
      pGP_energy[*iterPlas] = 0.0;
    }
  }
  
  // Universal
  ParticleVariable<Matrix3> deformationGradient, pstress, bElBar, pDeformRate;
  ParticleVariable<double> pEnergy;
  
  new_dw->allocateAndPut(pDeformRate, pDeformRateLabel, pset);
  new_dw->allocateAndPut(bElBar,      bElBarLabel,      pset);
  new_dw->allocateAndPut(pEnergy,      pEnergyLabel,      pset);

  for(;iterUniv != pset->end(); iterUniv++){
    bElBar[*iterUniv]      = Identity;
    pDeformRate[*iterUniv] = zero;
    pEnergy[*iterUniv] = 0;
  }
  
  // If not implicit, compute timestep
  if(!(flag->d_integrator == MPMFlags::Implicit)) {
    // End by computing the stable timestep
    computeStableTimestep(patch, matl, new_dw);
  }

  // cerr << "Done InitalizeCMData() matl " << matl << "Patch: " << patch->getID() << endl;
}

// Scheduling Functions //
//////////////////////////
void TongeRamesh::addComputesAndRequires(Task* task,
                                         const MPMMaterial* matl,
                                         const PatchSet* patches) const
{
  // Add the computes and requires that are common to all explicit 
  // constitutive models.  The method is defined in the ConstitutiveModel
  // base class.
  const MaterialSubset* matlset = matl->thisMaterial();
  if (flag->d_integrator == MPMFlags::Implicit) {
    throw ProblemSetupException("The TongeRamesh damage model should not be used with Implicit analysis"
                                , __FILE__, __LINE__);
    bool reset = flag->d_doGridReset;
    addSharedCRForImplicit(task, matlset, reset);
  }//  else {
  //   addSharedCRForExplicit(task, matlset, patches);
    
  // }
  
  // Other constitutive model and input dependent computes and requires
  Ghost::GhostType  gnone = Ghost::None;
  Ghost::GhostType  gan = Ghost::AroundNodes;
  Ghost::GhostType  gac = Ghost::AroundCells;

  task->requires(Task::OldDW, lb->pParticleIDLabel, matlset, gnone);

  // Add computes and requires from ConstitutiveModel::addSharedCRForExplicit()
  task->requires(Task::OldDW, lb->delTLabel);

  if(d_useDamage && d_brittle_damage.useNonlocalDamage){
    task->requires(Task::OldDW, gDamage_Label, matlset, gac, NGN); // get 2 ghost nodes
    // task->requires(Task::OldDW, lb->pSizeLabel, matlset, gan, NGP);
    // task->requires(Task::OldDW, lb->pXLabel, matlset, gan, NGP);
    // task->requires(Task::OldDW, lb->pDeformationMeasureLabel, matlset, gan, NGP);

    
    // task->requires(Task::OldDW, gDamage_Label, matlset, gnone); // get 2 ghost nodes
    task->requires(Task::OldDW, lb->pSizeLabel, matlset, gnone);
    task->requires(Task::OldDW, lb->pXLabel,                  matlset, gnone);
    task->requires(Task::OldDW, lb->pDeformationMeasureLabel, matlset, gnone);

    task->computes(gDamage_Label, matlset);
  } else {
    task->requires(Task::OldDW, lb->pXLabel,                  matlset, gnone);
    task->requires(Task::OldDW, lb->pDeformationMeasureLabel, matlset, gnone);
  }
  
  task->requires(Task::OldDW, lb->pMassLabel,               matlset, gnone);
  task->requires(Task::OldDW, lb->pVolumeLabel,             matlset, gnone);
  task->requires(Task::OldDW, lb->pTemperatureLabel,        matlset, gnone);
  task->requires(Task::OldDW, lb->pVelocityLabel,           matlset, gnone);
  task->requires(Task::NewDW, lb->pVolumeLabel_preReloc,    matlset, gnone);
  task->requires(Task::NewDW, lb->pDeformationMeasureLabel_preReloc, 
                                                            matlset, gnone);
  task->requires(Task::NewDW, lb->pVelGradLabel_preReloc,   matlset, gnone);

  task->computes(lb->pStressLabel_preReloc,             matlset);
  task->computes(lb->pdTdtLabel_preReloc,               matlset);

  // end of Add computes and requires from ConstitutiveModel::addSharedCRForExplicit()
  
  // Plasticity
  if(d_usePlasticity) {
    task->requires(Task::OldDW, pPlasticStrain_label,   matlset, gnone);
    task->computes(pPlasticStrain_label_preReloc,       matlset);
    task->requires(Task::OldDW, pPlasticEnergy_label,   matlset, gnone);
    task->computes(pPlasticEnergy_label_preReloc,       matlset);
  }
  
  if(d_useDamage) {
    //for pParticleID
    task->requires(Task::OldDW, lb->pParticleIDLabel,   matlset, gnone);
    
    // Other constitutive model and input dependent computes and requires
    task->requires(Task::OldDW, pLocalizedLabel,                matlset, gnone);
    if(d_brittle_damage.useNonlocalDamage){
      task->requires(Task::OldDW, pDamageLabel,                 matlset, gan, NGP);
    } else {
      task->requires(Task::OldDW, pDamageLabel,                 matlset, gnone);
    }

    // Add compute and require for new damage model:
    for (int i = 0; i < d_flawDistData.numCrackFamilies; i++){
      task->requires(Task::OldDW, wingLengthLabel_array[i], matlset, gnone);
      task->requires(Task::OldDW, starterFlawSize_array[i], matlset, gnone);
      task->requires(Task::OldDW, flawNumber_array[i],      matlset, gnone);
      
      task->computes(wingLengthLabel_array_preReloc[i], matlset);
      task->computes(starterFlawSize_array_preReloc[i], matlset);
      task->computes(flawNumber_array_preReloc[i],      matlset);
    }

    task->computes(pLocalizedLabel_preReloc,                    matlset);
    task->computes(pDamageLabel_preReloc,                       matlset);
    task->computes(lb->TotalLocalizedParticleLabel);
  }

  // Granular Plasticity
  if(d_useGranularPlasticity) {
    task->requires(Task::OldDW, pGPJLabel,   matlset, gnone);
    task->requires(Task::OldDW, pGP_plasticStrainLabel,   matlset, gnone);
    task->requires(Task::OldDW, pGP_plasticEnergyLabel,   matlset, gnone);
        
    task->computes(pGPJLabel_preReloc,       matlset);
    task->computes(pGP_plasticStrainLabel_preReloc,       matlset);
    task->computes(pGP_plasticEnergyLabel_preReloc,       matlset);
  }


  if(flag->d_with_color) {
    task->requires(Task::OldDW, lb->pColorLabel,  Ghost::None);
  }
  
  // Universal
  task->requires(Task::OldDW, bElBarLabel,              matlset, gnone);
  task->requires(Task::OldDW, pEnergyLabel,             matlset, gnone);
  task->computes(bElBarLabel_preReloc,                  matlset);
  task->computes(pDeformRateLabel_preReloc,             matlset);
  task->computes(pEnergyLabel_preReloc, matlset);
}

void TongeRamesh::addComputesAndRequires(Task* task,
                                         const MPMMaterial* matl,
                                         const PatchSet* patches,
                                         const bool recurse,
                                         const bool SchedParent) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  throw ProblemSetupException("This addComputesAndRequires() does not add damage requires"
                              , __FILE__, __LINE__);


  if(flag->d_integrator == MPMFlags::Implicit) {
    bool reset = flag->d_doGridReset;
    addSharedCRForImplicit(task, matlset, reset, true,SchedParent);
  } else {
    addSharedCRForExplicit(task, matlset, patches);
  }
  
  Ghost::GhostType  gnone = Ghost::None;
  // Require the particle ID for debugging information
  if(SchedParent){
    task->requires(Task::ParentOldDW, lb->pParticleIDLabel, matlset, gnone);
  } else {
    task->requires(Task::OldDW, lb->pParticleIDLabel, matlset, gnone);
  }

  if(d_usePlasticity){
    if(SchedParent){
      task->requires(Task::ParentOldDW, pPlasticStrain_label, matlset, gnone);
      task->requires(Task::ParentOldDW, pPlasticEnergy_label, matlset, gnone);
    }else{
      task->requires(Task::OldDW,       pPlasticStrain_label, matlset, gnone);
      task->requires(Task::OldDW,       pPlasticEnergy_label, matlset, gnone);
    }
  }

  if(SchedParent){
    task->requires(Task::ParentOldDW,   bElBarLabel,          matlset, gnone);
    task->requires(Task::ParentOldDW, pEnergyLabel,             matlset, gnone);
  }else{
    task->requires(Task::OldDW,         bElBarLabel,          matlset, gnone);
    task->requires(Task::OldDW, pEnergyLabel,             matlset, gnone);
  }
  
  if(d_useDamage){
    // Local stuff
    // task->requires(Task::ParentOldDW, bBeBarLabel, matlset, gnone);
  }

  if(d_useGranularPlasticity){
    if(SchedParent){
      task->requires(Task::ParentOldDW, pGPJLabel, matlset, gnone);
      task->requires(Task::ParentOldDW, pGP_plasticStrainLabel, matlset, gnone);
      task->requires(Task::ParentOldDW, pGP_plasticEnergyLabel, matlset, gnone);
    }else{
      task->requires(Task::OldDW,       pGPJLabel, matlset, gnone);
      task->requires(Task::OldDW,       pGP_plasticStrainLabel, matlset, gnone);
      task->requires(Task::OldDW,       pGP_plasticEnergyLabel, matlset, gnone);
    }
  }

}

void TongeRamesh::addInitialComputesAndRequires(Task* task,
                                                const MPMMaterial* matl,
                                                const PatchSet*) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  // Plasticity
  if(d_usePlasticity){
    task->computes(pPlasticStrain_label,matlset);
    task->computes(pPlasticEnergy_label,matlset);
  }
  // Damage
  if(d_useDamage) {
    
    for (int i = 0; i < d_flawDistData.numCrackFamilies; i++){
      task->computes(wingLengthLabel_array[i], matlset);
      task->computes(starterFlawSize_array[i], matlset);
      task->computes(flawNumber_array[i],      matlset);
    }

    task->computes(pLocalizedLabel,             matlset);
    task->computes(pDamageLabel,                matlset);
    task->computes(lb->TotalLocalizedParticleLabel);

    if(d_brittle_damage.useNonlocalDamage){
      task->computes(gDamage_Label, matlset);
    }

  }

  // Granular Plasticity
  if(d_useGranularPlasticity){
    task->computes(pGPJLabel,matlset);
    task->computes(pGP_plasticStrainLabel,matlset);
    task->computes(pGP_plasticEnergyLabel,matlset);
  }
  
  // Universal
  task->computes(bElBarLabel,           matlset);
  task->computes(pEnergyLabel,           matlset);
  task->computes(pDeformRateLabel,      matlset);
}

void TongeRamesh::addRequiresDamageParameter(Task* task,
                                             const MPMMaterial* matl,
                                             const PatchSet* ) const
{
  if(d_useDamage) {
    const MaterialSubset* matlset = matl->thisMaterial();
    task->requires(Task::NewDW, pLocalizedLabel_preReloc, matlset, Ghost::None);

    task->requires(Task::NewDW, pDamageLabel_preReloc, matlset, Ghost::None);
  }
}


// Compute Functions //
///////////////////////
void TongeRamesh::computePressEOSCM(const double rho_cur,double& pressure, 
                                    const double p_ref,
                                    double& dp_drho, double& cSquared,
                                    const MPMMaterial* matl,
                                    double temperature)
{
  // double bulk = d_initialData.Bulk;
  // double rho_orig = matl->getInitialDensity();
  
  // if(d_useModifiedEOS && rho_cur < rho_orig){
  //   double A = p_ref;           // MODIFIED EOS
  //   double n = bulk/p_ref;
  //   double rho_rat_to_the_n = pow(rho_cur/rho_orig,n);
  //   pressure = A*rho_rat_to_the_n;
  //   dp_drho  = (bulk/rho_cur)*rho_rat_to_the_n;
  //   cSquared = dp_drho;         // speed of sound squared
  // } else {                      // STANDARD EOS            
  //   double p_g = .5*bulk*(rho_cur/rho_orig - rho_orig/rho_cur);
  //   pressure   = p_ref + p_g;
  //   dp_drho    = .5*bulk*(rho_orig/(rho_cur*rho_cur) + 1./rho_orig);
  //   cSquared   = bulk/rho_cur;  // speed of sound squared
  // }
  throw std::runtime_error("TongeRamesh::computePressEOSCM() has not been updated");
}

// The "CM" versions use the pressure-volume relationship of the CNH model
double TongeRamesh::computeRhoMicroCM(double pressure, 
                                      const double p_ref,
                                      const MPMMaterial* matl,
                                      double temperature,
                                      double rho_guess)
{
  // double rho_orig = matl->getInitialDensity();
  // double bulk = d_initialData.Bulk;
  
  // double p_gauge = pressure - p_ref;
  // double rho_cur;
  
  // if(d_useModifiedEOS && p_gauge < 0.0) {
  //   double A = p_ref;           // MODIFIED EOS
  //   double n = p_ref/bulk;
  //   rho_cur = rho_orig*pow(pressure/A,n);
  // } else {                      // STANDARD EOS
  //   double p_g_over_bulk = p_gauge/bulk;
  //   rho_cur=rho_orig*(p_g_over_bulk + sqrt(p_g_over_bulk*p_g_over_bulk +1.));
  // }
  // return rho_cur;
  throw std::runtime_error("TongeRamesh::computeRhoMicroCM() has not been updated");
}

void TongeRamesh::computeStableTimestep(const Patch* patch,
                                        const MPMMaterial* matl,
                                        DataWarehouse* new_dw)
{
  // This is only called for the initial timestep - all other timesteps
  // are computed as a side-effect of computeStressTensor
  Vector dx = patch->dCell();
  int dwi = matl->getDWIndex();
  // Retrieve the array of constitutive parameters
  ParticleSubset* pset = new_dw->getParticleSubset(dwi, patch);
  constParticleVariable<double> pMass, pvolume;
  constParticleVariable<Vector> pVelocity;

  new_dw->get(pMass,     lb->pMassLabel,     pset);
  new_dw->get(pvolume,   lb->pVolumeLabel,   pset);
  new_dw->get(pVelocity, lb->pVelocityLabel, pset);

  double c_dil = 0.0;
  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

  double mu   = d_initialData.tauDev;
  double bulk = d_initialData.Bulk;
  
  for(ParticleSubset::iterator iter = pset->begin();iter != pset->end();iter++){
    particleIndex idx = *iter;

    // Compute wave speed at each particle, store the maximum
    Vector pVelocity_idx = pVelocity[idx];
    if(pMass[idx] > 0){
      c_dil = sqrt((bulk + 4.*mu/3.)*pvolume[idx]/pMass[idx]);
    }
    else{
      c_dil = 0.0;
      pVelocity_idx = Vector(0.0,0.0,0.0);
    }
    WaveSpeed=Vector(Max(c_dil+fabs(pVelocity[idx].x()),WaveSpeed.x()),
                     Max(c_dil+fabs(pVelocity[idx].y()),WaveSpeed.y()),
                     Max(c_dil+fabs(pVelocity[idx].z()),WaveSpeed.z()));
  }

  WaveSpeed = dx/WaveSpeed;
  double delT_new = WaveSpeed.minComponent();
  new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
}

void TongeRamesh::computeStressTensor(const PatchSubset* patches,
                                      const MPMMaterial* matl,
                                      DataWarehouse* old_dw,
                                      DataWarehouse* new_dw)
{
  // Implicit from Damage component
  if (flag->d_integrator == MPMFlags::Implicit) {
    computeStressTensorImplicit(patches, matl, old_dw, new_dw);
    return;
  }
  
  // Constants
  double onethird = (1.0/3.0), sqtwthds = sqrt(2.0/3.0);
  Matrix3 Identity;
  Identity.Identity();
  
  // Grab initial data
  double rho_orig = matl->getInitialDensity();    
  double flow     = 0.0;
  double K        = 0.0;

  // Get delT
  delt_vartype delT;
  old_dw->get(delT, lb->delTLabel, getLevel(patches));

  // for damage calculations keep track of the delta T to maintian the
  // desired resolution of the damage increment:
  double damage_dt;
  if(d_useDamage){
    damage_dt = d_brittle_damage.dt_increaseFactor * (double)delT;
  } else {
    damage_dt = 1e6*delT;       // set to a very large number
  }
 
  // Normal patch loop
  for(int pp=0;pp<patches->size();pp++){
    const Patch* patch = patches->get(pp);

    // cerr << "starting work on patch: " << patch->getID() << endl;

    // Temporary and "get" variables
    double delgamma = 0.0, fTrial = 0.0, IEl = 0.0, J = 0.0, Jinc = 0.0, muBar = 0.0; 
    double p = 0.0, sTnorm = 0.0, JGP = 1.0, JEL= 1.0, U=0.0, W=0.0;
    double se=0.0;     // Strain energy placeholder
    double c_dil=0.0;  // Speed of sound
    long64 totalLocalizedParticle = 0;
    Matrix3 pBBar_new(0.0), bEB_new(0.0), bElBarTrial(0.0), pDefGradInc(0.0);
    Matrix3 displacementGradient(0.0), fBar(0.0), defGrad(0.0), normal(0.0);
    Matrix3 tauDev(0.0), tauDevTrial(0.0);
    Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

    // Get particle info and patch info
    int dwi              = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
    Vector dx            = patch->dCell();

//    Ghost::GhostType  gan = Ghost::AroundNodes;
    Ghost::GhostType  gac = Ghost::AroundCells;
//    Ghost::GhostType  gnone = Ghost::None;
    ParticleSubset* pset2 = pset;
    
    if(d_useDamage){
      if(d_brittle_damage.useNonlocalDamage){
        // cerr << "getting pset2" << endl;
        pset2 = old_dw->getParticleSubset(dwi, patch);
        // pset2 = old_dw->getParticleSubset(dwi, patch,
        //                                   gan, NGP, lb->pXLabel);
        // cerr << "done getting pset2" << endl;
      }
    } 
    // Variables for grid based damage calc:
    
    constParticleVariable<double> nld_pDamage_old; 
    constParticleVariable<Point>   nld_px;
    constParticleVariable<Matrix3> nld_pFold, nld_psize;
    constNCVariable<double> gDamage;

    NCVariable<double>      gDamage_new;

    // Gets for NonLocal damage calc:
    if(d_useDamage && d_brittle_damage.useNonlocalDamage){
      // cerr << "starting get nonlocal damage quantities" << endl;
      old_dw->get(nld_pDamage_old, pDamageLabel,   pset2);
      old_dw->get(gDamage, gDamage_Label, dwi, patch, gac, NGN); // this is NGP in interpolate to particles and update

      old_dw->get(nld_psize,   lb->pSizeLabel, pset2);
      old_dw->get(nld_px,      lb->pXLabel,    pset2);
      old_dw->get(nld_pFold,   lb->pDeformationMeasureLabel, pset2);

      new_dw->allocateAndPut(gDamage_new, gDamage_Label, dwi, patch);
      gDamage_new.initialize(1e-8);
      // cerr << "done get nonlocal damage quantities" << endl;
    }

    
    // Particle and grid data universal to model type
    // Old data containers
    constParticleVariable<int>     pLocalized;
    constParticleVariable<double>  pMass, pDamage, pTemperature, pEnergy;
    constParticleVariable<double>  pPlasticStrain_old,pcolor, pGPJ_old, pGP_strain_old;
    constParticleVariable<double>  pPlasticEnergy_old, pGP_energy_old;
    constParticleVariable<long64>  pParticleID;
    constParticleVariable<Matrix3> pDefGrad, bElBar;
    constParticleVariable<Vector>  pVelocity;

    // For interpolating g.damage to particles
    constParticleVariable<Point>   px;
    constParticleVariable<Matrix3> psize;

    // Old damage data:
    constParticleVariable<double> *pWingLength_array;
    constParticleVariable<double> *pflawSize_array;
    constParticleVariable<double> *pFlawNumber_array;
    

    // New data containers
    ParticleVariable<int>          pLocalized_new;
    ParticleVariable<double>       pPlasticStrain, pPlasticEnergy, pGPJ, pGP_strain, pGP_energy;
    ParticleVariable<double>       pDamage_new;
    ParticleVariable<double>       pdTdt,p_q, pEnergy_new;
    ParticleVariable<Matrix3>      pDeformRate, bElBar_new;
    ParticleVariable<Matrix3>      pStress;

    constParticleVariable<Matrix3> velGrad, pDefGrad_new;
    constParticleVariable<double>  pVolume_new;

    // New Damage data
    ParticleVariable<double> *pWingLength_array_new;
    ParticleVariable<double> *pflawSize_array_new;
    ParticleVariable<double> *pFlawNumber_array_new;

    // cerr << "starting standard data management" << endl;
    if(d_useDamage){
      // Old damge:
      pWingLength_array = new constParticleVariable<double>[d_flawDistData.numCrackFamilies];
      pflawSize_array   = new constParticleVariable<double>[d_flawDistData.numCrackFamilies];
      pFlawNumber_array = new constParticleVariable<double>[d_flawDistData.numCrackFamilies];

      // New Damage:
      pWingLength_array_new = new ParticleVariable<double>[d_flawDistData.numCrackFamilies];
      pflawSize_array_new   = new ParticleVariable<double>[d_flawDistData.numCrackFamilies];
      pFlawNumber_array_new = new ParticleVariable<double>[d_flawDistData.numCrackFamilies];
    } else {
      // Old damge:
      pWingLength_array = NULL;
      pflawSize_array   = NULL;
      pFlawNumber_array = NULL;

      // New Damage:
      pWingLength_array_new = NULL;
      pflawSize_array_new   = NULL;
      pFlawNumber_array_new = NULL;
    }
    
    // Plasticity gets
    if(d_usePlasticity) {
      old_dw->get(pPlasticStrain_old,         
                  pPlasticStrain_label,                pset);
      new_dw->allocateAndPut(pPlasticStrain,  
                             pPlasticStrain_label_preReloc,       pset);
      
      pPlasticStrain.copyData(pPlasticStrain_old);

      old_dw->get(pPlasticEnergy_old,         
                  pPlasticEnergy_label,                pset);
      new_dw->allocateAndPut(pPlasticEnergy,  
                             pPlasticEnergy_label_preReloc,       pset);
      
      pPlasticEnergy.copyData(pPlasticEnergy_old);
      
      // Copy initial data
      flow  = d_initialData.FlowStress;
      K     = d_initialData.K;
    }
    
    // Damage gets
    if(d_useDamage) {
      old_dw->get(pLocalized,               pLocalizedLabel,                pset);
      old_dw->get(pParticleID,              lb->pParticleIDLabel,           pset);
      old_dw->get(pDamage, pDamageLabel,   pset);

      // Get the old data:
      for (int i = 0; i < d_flawDistData.numCrackFamilies; i++){
        old_dw->get(pWingLength_array[i], wingLengthLabel_array[i], pset);
        old_dw->get(pFlawNumber_array[i], flawNumber_array[i], pset);
        old_dw->get(pflawSize_array[i], starterFlawSize_array[i], pset);
      }


      new_dw->allocateAndPut(pLocalized_new, 
                             pLocalizedLabel_preReloc,              pset);
      new_dw->allocateAndPut(pDamage_new, 
                             pDamageLabel_preReloc,                 pset);
      
      for (int i = 0; i < d_flawDistData.numCrackFamilies; i++){
        new_dw->allocateAndPut(pWingLength_array_new[i],
                               wingLengthLabel_array_preReloc[i], pset);
        new_dw->allocateAndPut(pFlawNumber_array_new[i],
                               flawNumber_array_preReloc[i], pset);
        new_dw->allocateAndPut(pflawSize_array_new[i],
                               starterFlawSize_array_preReloc[i], pset);
      }

      // Copy flaw data to the new dw:
      for (int i = 0; i < d_flawDistData.numCrackFamilies; i++){
        pFlawNumber_array_new[i].copyData(pFlawNumber_array[i]);
        pflawSize_array_new[i].copyData(pflawSize_array[i]);
      }

      if(d_useDamage && d_brittle_damage.useNonlocalDamage){
        // used for interpolation from grid to particles
        old_dw->get(psize,   lb->pSizeLabel, pset2); 
        old_dw->get(px,      lb->pXLabel,    pset2);
      }
            
    } //end d_useDamage

    // Granular Plasticity gets
    if(d_useGranularPlasticity) {
      old_dw->get(pGPJ_old, pGPJLabel, pset);
      old_dw->get(pGP_strain_old, pGP_plasticStrainLabel, pset);
      old_dw->get(pGP_energy_old, pGP_plasticEnergyLabel, pset);
      
      new_dw->allocateAndPut(pGPJ, pGPJLabel_preReloc, pset);
      new_dw->allocateAndPut(pGP_strain, pGP_plasticStrainLabel_preReloc, pset);
      new_dw->allocateAndPut(pGP_energy, pGP_plasticEnergyLabel_preReloc, pset);
      
      pGPJ.copyData(pGPJ_old);
      pGP_strain.copyData(pGP_strain_old);
      pGP_energy.copyData(pGP_energy_old);
    }

    // Universal Gets
    old_dw->get(pMass,               lb->pMassLabel,               pset);
    old_dw->get(pVelocity,           lb->pVelocityLabel,           pset);
    old_dw->get(pDefGrad,            lb->pDeformationMeasureLabel, pset2);
    old_dw->get(bElBar,              bElBarLabel,                  pset);
    old_dw->get(pTemperature,        lb->pTemperatureLabel,        pset);
    old_dw->get(pEnergy,             pEnergyLabel,                 pset);
    old_dw->get(pParticleID,         lb->pParticleIDLabel,         pset);


    // Universal Gets for the current timestep:
    new_dw->get(velGrad,             lb->pVelGradLabel_preReloc,   pset);
    new_dw->get(pVolume_new,         lb->pVolumeLabel_preReloc,    pset);
    new_dw->get(pDefGrad_new,lb->pDeformationMeasureLabel_preReloc,pset);
    
    // Universal Allocations
    new_dw->allocateAndPut(bElBar_new,  bElBarLabel_preReloc,      pset);
    new_dw->allocateAndPut(pStress,     lb->pStressLabel_preReloc, pset);
    new_dw->allocateAndPut(pdTdt,       lb->pdTdtLabel_preReloc,   pset);
    new_dw->allocateAndPut(p_q,         lb->p_qLabel_preReloc,     pset);
    new_dw->allocateAndPut(pDeformRate, pDeformRateLabel_preReloc, pset);
    new_dw->allocateAndPut(pEnergy_new, pEnergyLabel_preReloc,     pset);

    if(flag->d_with_color) {
      old_dw->get(pcolor,      lb->pColorLabel,  pset);
    }

    // cerr << "done standard data management" << endl;
    // End data management -------------------------------------------

    // Compute the new grid based damage based on the oldDW p.damage:
    ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
    // cerr << "finished creating the particle interpolator size is:" << interpolator->size() << endl;
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());

    // cerr << "finished creating the vectors ni and S" << endl;
    NCVariable<double> unityPartition;
    new_dw->allocateTemporary(unityPartition, patch);
    unityPartition.initialize(0);
    const double uinityTollerance = 0.1;
    
    if(d_useDamage && d_brittle_damage.useNonlocalDamage){
      // cerr << "Starting calc gDamage_new" << endl;
      // cerr << "pset2->begin() = " << pset2->begin() << endl;
      // cerr << "pset2->end() = " << pset2->end() << endl;
      for (ParticleSubset::iterator iter2 = pset2->begin();
           iter2 != pset2->end(); 
           iter2++){
        particleIndex idx2 = *iter2;
        // interpolator->findCellAndWeights(nld_px[idx2],ni,S,Identity,Identity);
        interpolator->findCellAndWeights(nld_px[idx2],ni,S,nld_psize[idx2],nld_pFold[idx2]);

        // Add each particles contribution to the grid density
        // Must use the node indices
        IntVector node;
        for(int k = 0; k < flag->d_8or27; k++) { // Iterates through the nodes which receive information from the current particle
          node = ni[k];
          if(patch->containsNode(node)) {
            unityPartition[node] += S[k];
            gDamage_new[node]        += nld_pDamage_old[idx2] * S[k];
          }
        }
      } // End of particle loop

      for(NodeIterator iter2=patch->getNodeIterator(); !iter2.done();iter2++){
        IntVector node = *iter2;
        if(unityPartition[node] > uinityTollerance){
          gDamage_new[node] = gDamage_new[node]/unityPartition[node];
        } else {
          gDamage_new[node] = 0.0;
        }
      }
      // cerr << "Done calc gDamage_new" << endl;
    }  else {
      // cerr << "Not doing gDamage_new calc" << endl;
    }  // End non-local damage calculation (interpolate to grid nodes)

    ParticleSubset::iterator iter = pset->begin();
    // cerr << "starting main particle loop pset->begin()=" << pset->begin() << endl;
    // cerr << "starting main particle loop pset->end()=" << pset->end() << endl;
    for(; iter != pset->end(); iter++){
      particleIndex idx = *iter;

      // Assign zero internal heating by default - modify if necessary.
      pdTdt[idx] = 0.0;

      pDeformRate[idx] = (velGrad[idx] + velGrad[idx].Transpose())*0.5;
      pDefGradInc = pDefGrad_new[idx]*pDefGrad[idx].Inverse();
      
      Jinc    = pDefGradInc.Determinant();
      defGrad = pDefGrad_new[idx];
      J = pDefGrad_new[idx].Determinant();
      
      // 1) Get the volumetric part of the deformation
      // Check 1: Look at Jacobian
      if (!(J > 0.0)) {
        cerr << getpid() ;
        cerr << "**ERROR** Negative Jacobian of deformation gradient"
             << " in particle " << pParticleID[idx]  << endl;
        cerr << "F_old = " << pDefGrad[idx]     << endl;
        cerr << "velGrad = " << velGrad[idx] << endl;
        cerr << "F_inc = " << pDefGradInc       << endl;
        cerr << "F_new = " << pDefGrad_new[idx] << endl;
        cerr << "J = "     << J                 << endl;
        if(d_useGranularPlasticity){
          cerr << " JGP = " << pGPJ[idx] << endl;
          cerr << " JEL = " << J/pGPJ[idx] << endl;
        }
        // cerr << "Setting deformation gradient to pDefGrad, values after reset: " << endl;

        // pDefGradInc = Identity;
        // pDefGrad_new[idx] = pDefGrad[idx];
        // Jinc    = pDefGradInc.Determinant();
        // defGrad = pDefGrad_new[idx];
        // J               = defGrad.Determinant();
      
        // // Set the deformation rate and velocity gradient to 0
        // pDeformRate[idx] = Identity*0.0;
        // pDeformRate[idx] = Identity*0.0;

        // cerr << "F_old = " << pDefGrad[idx]     << endl;
        // cerr << "F_inc = " << pDefGradInc       << endl;
        // cerr << "F_new = " << pDefGrad_new[idx] << endl;
        // cerr << "J = "     << J                 << endl;
        // if(d_useGranularPlasticity){
        //   cerr << " JGP = " << pGPJ[idx] << endl;
        //   cerr << " JEL = " << J/pGPJ[idx] << endl;
        // }
        throw std::runtime_error("Negative Jacobian");
      }

      J = defGrad.Determinant();

      if(d_useGranularPlasticity){
        // cout << "In TongeRamesh::computeStressTensor() setting JGP" << endl;
        JGP = pGPJ[idx];
        JEL = J/JGP;
      } else {
        JEL = J;
        JGP = 1.0;
      }
      
      double rho_cur  = rho_orig/JEL;      
      // Set up the PlasticityState (for t_n+1) --------------------------------
      PlasticityState* state = scinew PlasticityState();
      state->pressure = -pStress[idx].Trace()*onethird;
      state->temperature = pTemperature[idx];
      state->initialTemperature = matl->getRoomTemperature();
      state->density = rho_cur;
      state->initialDensity = rho_orig;
      state->volume = pVolume_new[idx];
      state->initialVolume = pMass[idx]/rho_orig;
      
      state->specificHeat = matl->getSpecificHeat();
      state->energy = (state->temperature-state->initialTemperature)*state->specificHeat; // This is the internal energy do
      // to the temperature
      // Set the moduli:
      
      state->initialBulkModulus = JEL * d_eos->computeBulkModulus(state->initialDensity, state->density);
      state->bulkModulus = state->initialBulkModulus;
      
      // The shear modulus is easy b/c it does not depend on the deformation
      state->shearModulus = d_initialData.tauDev; // This is changed later if there is damage
      state->initialShearModulus = d_initialData.tauDev;

      // End PlasticityState setup -----------------------------------------

      // Step 0: Compute input parameters for constitutive update:

      // Calculate nonLocal damage:
      double nonLocalDamage(0);

      if(d_useDamage && d_brittle_damage.useNonlocalDamage){
        double unitySum(0);
        // interpolator->findCellAndWeights(nld_px[idx],ni,S,Identity,Identity);
        interpolator->findCellAndWeights(px[idx],ni,S,psize[idx],pDefGrad[idx]);
        for (int k = 0; k < flag->d_8or27; k++) {
          IntVector node = ni[k];
          nonLocalDamage += gDamage[node] * S[k];
          unitySum += S[k];
        }
        nonLocalDamage = unitySum > uinityTollerance ? nonLocalDamage/unitySum : 0.0;
        // cerr << "nonLocalDamage= " << nonLocalDamage << endl;
      }

      // Step 1: Compute Plastic flow --------------------------------------
      
      // Get the volume preserving part of the deformation gradient increment
      fBar = pDefGradInc/cbrt(Jinc);
      
      // Compute the trial elastic part of the volume preserving 
      // part of the left Cauchy-Green deformation tensor
      bElBarTrial = fBar*bElBar[idx]*fBar.Transpose();
      if(!(d_usePlasticity||d_useGranularPlasticity)){
        double cubeRootJ       = cbrt(JEL);
        double Jtothetwothirds = cubeRootJ*cubeRootJ;
        bElBarTrial            = pDefGrad_new[idx]* pDefGrad_new[idx].Transpose()
          /Jtothetwothirds;
      }
      IEl   = onethird*bElBarTrial.Trace();
      muBar = IEl*state->shearModulus;
      
      // tauDevTrial is equal to the shear modulus times dev(bElBar)
      // Compute ||tauDevTrial||
      tauDevTrial = (bElBarTrial - Identity*IEl)*state->shearModulus;
      sTnorm      = tauDevTrial.Norm();
      
      // Check for plastic loading
      double alpha;
      if(d_usePlasticity)
        { 
          alpha  = pPlasticStrain[idx];
          fTrial = sTnorm - sqtwthds*(K*alpha + flow);
        }
      if (d_usePlasticity && (fTrial > 0.0) ) {
        // plastic
        // Compute increment of slip in the direction of flow (with viscoplastic effects
        // if d_initialData.timeConstant>0.
        delgamma = (fTrial/(2.0*muBar)) / (d_initialData.timeConstant/delT + 1.0 + (K/(3.0*muBar)));
        normal   = tauDevTrial/sTnorm;
        
        // The actual shear stress
        tauDev = tauDevTrial - normal*2.0*muBar*delgamma;
        double IEl_tr = IEl;
        IEl = (IEl_tr - 1) * (tauDev.NormSquared()/tauDevTrial.NormSquared()) + 1;
        
        // Deal with history variables
        pPlasticStrain[idx] = alpha + sqtwthds*delgamma;
        bElBar_new[idx]     = tauDev/state->shearModulus + Identity*IEl;
        pPlasticEnergy[idx] = pPlasticEnergy[idx] + delgamma*tauDev.Norm();
        pdTdt[idx] += delgamma*tauDev.Norm()/(delT * rho_orig * state->specificHeat);
      } else { 
        // The actual shear stress
        tauDev          = tauDevTrial; 
        bElBar_new[idx] = bElBarTrial;
      }

      // Step 2: Compute damage growth -------------------------------------

      if(d_useDamage){
        // if using damage calculate the new bulk and shear modulus
        // based on the particle damage: Update the damage based on the
        // stress from the previous timestep. Then calculate the current
        // stiffness and then update the stress, and plastic terms.

        // copy the flaw size, wing crack length, and number of flaws in the
        // bin to a vector:
      
        double oldDamage = pDamage[idx]; // Damage at the beginning of the step
        double damageTrial = oldDamage;
        double currentDamage;
        vector<double> L_old(d_flawDistData.numCrackFamilies,0);
        vector<double> L_new(d_flawDistData.numCrackFamilies,0);
        vector<double> L_dot_new(d_flawDistData.numCrackFamilies,0);
        vector<double> s(d_flawDistData.numCrackFamilies,0);
        vector<double> N(d_flawDistData.numCrackFamilies,0);

        for (int i = 0; i < d_flawDistData.numCrackFamilies; i++){
          L_old[i] = pWingLength_array[i][idx];
          N[i]     = pFlawNumber_array[i][idx];
          s[i]     = pflawSize_array[i][idx];
        }

        if(d_brittle_damage.useNonlocalDamage){
          state->bulkModulus  = calculateBulkPrefactor(nonLocalDamage, state, JEL)*state->initialBulkModulus;
          state->shearModulus = calculateShearPrefactor(nonLocalDamage, state)*state->initialShearModulus;
        } else {
          state->bulkModulus  = calculateBulkPrefactor(oldDamage, state, JEL)*state->initialBulkModulus;
          state->shearModulus = calculateShearPrefactor(oldDamage, state)*state->initialShearModulus;
        }
        
        // Use the predicted stress instead of the stress from the previous
        // increment:
        double JEL_old = pDefGrad[idx].Determinant()/JGP;
        double p_old = computePressure(matl, pDefGrad[idx]*pow(JGP, -0.3333333), pDefGradInc, state,
                                       delT, oldDamage);
        if(d_brittle_damage.useNonlocalDamage){
          p_old = computePressure(matl, pDefGrad[idx]*pow(JGP, -0.3333333), pDefGradInc, state,
                                       delT, nonLocalDamage);
        }
        double IEl_old = onethird*bElBar[idx].Trace();
        Matrix3 oldStress = (bElBar[idx]-Identity*IEl_old)*state->shearModulus + Identity*p_old;

        if (oldDamage > d_brittle_damage.maxDamage - d_brittle_damage.maxDamage*1e-8){
          currentDamage = d_brittle_damage.maxDamage;
          L_new = L_old;
        } else if(d_brittle_damage.useOldStress){
          // basic forward euler, I only used information from the previous timestep
          // to compute the new values.
          if(d_brittle_damage.useNonlocalDamage){
            currentDamage =  calculateDamageGrowth(oldStress, N, s, L_old, oldDamage,
                                                   L_new, L_dot_new, delT, pLocalized[idx], state);
          } else {
            currentDamage =  calculateDamageGrowth(oldStress, N, s, L_old, oldDamage,
                                                   L_new, L_dot_new, delT, pLocalized[idx], state);
          }
        } else {
          // Try to use a more consistant formulation, so that the current stress
          // is used to calculate the damage growth rate.
          double JEL_new = JEL;
      
          double p_new = computePressure(matl, pDefGrad_new[idx]*pow(JGP, -onethird), pDefGradInc,
                                         state, delT, oldDamage);

          // WARNING THE DAMAGE SUBLOOPING DOES NOT USE THE NONLOCAL DAMAGE
          
          Matrix3 bElBar_test = bElBar_new[idx];

          double IEl_new = onethird*bElBar_test.Trace();

          // Calculate the trial stress using the current strain, and the old damage:
          Matrix3 stress_trial = (bElBar_test-Identity*IEl_new)*state->shearModulus + Identity*p_new;

          // Update the damage state:
          damageTrial =  calculateDamageGrowth(stress_trial, N, s, L_old, oldDamage,
                                               L_new, L_dot_new, delT, pLocalized[idx], state);
          // Compute the increment in damage:
          double damageInc = damageTrial - oldDamage;

          if(damageInc > d_brittle_damage.maxDamageInc && pLocalized[idx] < 1){
            // The timesteps to resolve the damage process need to be smaller
            double numSteps = ceil(damageInc/d_brittle_damage.maxDamageInc);
            // cerr << "Damage subloop number of subloops=\t" << numSteps << endl;
            double invNumSteps = 1.0/numSteps;

            // Compute the strain increment per sub step:
            Matrix3 devStrain_new = bElBar_test-Identity*IEl_new;
            Matrix3 devStrain_old = bElBar[idx]-Identity*IEl_old;
            Matrix3 devStrain_inc = (devStrain_new-devStrain_old)*invNumSteps;

            // Increment in the volumetric strain:
            double volStrain_old  = 0.5*(JEL_old - 1.0/JEL_old);
            double volStrain_new  = 0.5*(JEL_new - 1.0/JEL_new);
            double volStrain_inc  = (volStrain_new-volStrain_old)*invNumSteps;

            // Sub increment in time:
            double dt = delT*invNumSteps;
          
            // Initialze:
            devStrain_new = devStrain_old;
            volStrain_new = volStrain_old;
            currentDamage = oldDamage;

            // subloop:
            for(int i = 0; i<numSteps; i++){
              // Compute the stress:
              state->bulkModulus  = calculateBulkPrefactor(currentDamage, state, JEL)
                *state->initialBulkModulus;
              state->shearModulus = calculateShearPrefactor(currentDamage, state)
                *state->initialShearModulus;

              double J_oneThird = powf(1.0/(1-volStrain_new), onethird);

              double pressure = computePressure(matl, Identity*J_oneThird,
                                                pDefGradInc,
                                                state, delT, currentDamage);
            
              stress_trial = devStrain_new*state->shearModulus+Identity*pressure;

              currentDamage =  calculateDamageGrowth(stress_trial, N, s, L_old, currentDamage,
                                                     L_new, L_dot_new, dt, pLocalized[idx], state);
              // copy L_new to L_old:
              L_old = L_new;

              // Increment strain:
              volStrain_new += volStrain_inc;
              devStrain_new += devStrain_inc;
            }
          } else {
            currentDamage = damageTrial;
          }
        } // End if(useOldStress)

        // Compute the desired timestep based on the damage evolution:
        if (currentDamage-oldDamage > d_brittle_damage.maxDamageInc){
          damage_dt = min(damage_dt, (d_brittle_damage.maxDamageInc /
                                      (currentDamage - oldDamage)) * delT);
        }

        // Set the damage, and wing crack sizes at the end of the step:
        for (int i = 0; i < d_flawDistData.numCrackFamilies; i++){
          pWingLength_array_new[i][idx] = L_new[i];
        }

        pDamage_new[idx] = currentDamage;

        // // Update the moduli:
        if(!d_brittle_damage.useNonlocalDamage){
          state->bulkModulus  = calculateBulkPrefactor(currentDamage, state, JEL)*state->initialBulkModulus;
          state->shearModulus = calculateShearPrefactor(currentDamage, state)*state->initialShearModulus;
        }
      } // end if(d_useDamage)

      // End damage computation -------------------------------------------

      // Step 3: Calculate the flow due to Granular Plasticity:
      if(d_useGranularPlasticity){
        const double gp_TQparam(1.0);
        // if damage is being used only activate graunlar plasticity when the
        // damage level is equal to the critical damage level
        bool doGPcalc(true);
        if(d_useDamage){
          if(d_brittle_damage.useNonlocalDamage){
            doGPcalc = nonLocalDamage >= d_brittle_damage.criticalDamage-1e-8;
          } else {
            doGPcalc = pDamage_new[idx] >= d_brittle_damage.criticalDamage-1e-8;
          }
        }
        if(doGPcalc){
          bElBarTrial = bElBar_new[idx]; // This may have been modified in the Plasticity section
          IEl   = onethird*bElBarTrial.Trace();
          muBar = IEl*state->shearModulus;
      
          // Compute the trial stress:
          tauDevTrial = (bElBarTrial - Identity*IEl)*state->shearModulus;
          sTnorm      = tauDevTrial.Norm();
          // hydrostatic stress:
          double p_trial;
          if( d_useDamage){
            if(d_brittle_damage.useNonlocalDamage){
              p_trial = computePressure(matl, Identity * cbrt(JEL), pDefGradInc,
                                      state, delT, nonLocalDamage);
            } else {
              p_trial = computePressure(matl, Identity * cbrt(JEL), pDefGradInc,
                                      state, delT, pDamage_new[idx]);
            }
          } else {
            p_trial = computePressure(matl, Identity*cbrt(JEL), pDefGradInc,
                                      state, delT, 0);
          }
           
          // Return algorithm discussed by Rebecca Brannon in Appendix 3 of:
          // Geometric Insight into Return Mapping Algorithms 

          double gamma_s=1.0;
          double gamma_p=1.0;
          double sigma_tr_p = gamma_p*sqrt(3)*p_trial;
          double sigma_tr_s = gamma_s*sTnorm;

          double gs,gp;
          double g_tr=calc_yeildFunc_g_gs_gp(sigma_tr_s, sigma_tr_p,
                                             gs, gp);
          // Check for plastic loading:
          if(g_tr>0.0){
            // Compute the rate independent limit tau_bar
            double p_bar=p_trial;
            Matrix3 tauBar = tauDevTrial;

            if(sigma_tr_p > d_GPData.B){
              p_bar=d_GPData.B/sqrt(3.0);
              tauBar = 0.0*tauDevTrial;
            } else {
              Matrix3 S_hat=tauDevTrial/sTnorm;
              double g_test=g_tr;
              double sigma_test_p=sigma_tr_p;
              double sigma_test_s=sigma_tr_s;

              int stepNum;
              for(stepNum=0; stepNum<100; ++stepNum){
                double M_p,M_s;

                if(true){         // Always use associative plastity
                  M_p = gamma_p*gp;
                  M_s = gamma_s*gs;
                } else {
                  calc_returnDir(sigma_test_p, sigma_test_s, JGP, M_p, M_s);
                  M_p *= gamma_p;
                  M_s *= gamma_s;
                }

                double eta = 2*state->shearModulus/(3*state->bulkModulus);
                double A_p=M_p;
                double A_s=eta*M_s;
                double scaleFactor =
                  sqrt( (sigma_test_p*sigma_test_p +sigma_test_s*sigma_test_s)
                        /(A_p*A_p+A_s*A_s));
                A_p *= scaleFactor;
                A_s *= scaleFactor;

                double beta_next = -g_test/(gp*gamma_p*A_p+gs*gamma_s*A_s);
                sigma_test_p += beta_next*gamma_p*A_p;
                sigma_test_s += beta_next*gamma_s*A_s;
                if(abs(beta_next)<1e-8){
                  break;
                }
                g_test=calc_yeildFunc_g_gs_gp(sigma_test_s, sigma_test_p,
                                              gs, gp);
              }
              if(stepNum == 99){
                throw runtime_error("Solver in TongeRamesh GP calc reached max number of iterations");
              }

              // Now we can compute the rate indipendent values:
              p_bar=sigma_test_p/(sqrt(3.0)*gamma_p);
              tauBar = sigma_test_s*S_hat;
            }
            
            double p_target;
            if(d_GPData.timeConstant>0.0){
              double inv_timeConst = 1.0/d_GPData.timeConstant;
              tauDev=(tauDevTrial+ delT*inv_timeConst * tauBar)/(1+delT*inv_timeConst);
          
              p_target = (p_trial + delT*inv_timeConst*p_bar)/(1+delT*inv_timeConst);
            } else {
              tauDev=tauBar;
              p_target = p_bar;
            }

            Matrix3 devPlasticStrainRate = (tauDevTrial-tauDev)/
                                                (2.0*delT*state->shearModulus);
            double tr_d_p = (p_trial-p_target)/(delT*state->bulkModulus);
            // JGP = pGPJ_old[idx]/(1-delT*tr_d_p);
            JGP = pGPJ_old[idx]*exp(delT*tr_d_p);

            pGPJ[idx] = JGP;
            JEL = J/JGP;
            state->density = rho_orig/JEL;
          
            double p_new;
            if( d_useDamage){
              if(d_brittle_damage.useNonlocalDamage){
                p_new = computePressure(matl, Identity * cbrt(JEL), pDefGradInc,
                                        state, delT, nonLocalDamage);
              } else {
                p_new = computePressure(matl, Identity * cbrt(JEL), pDefGradInc,
                                        state, delT, pDamage_new[idx]);
              }
            } else {
              p_new = computePressure(matl, Identity*cbrt(JEL), pDefGradInc,
                                      state, delT, 0);
            }
	    
            
            // pGP_energy[idx] += gp_TQparam*(del_U + del_W); 
            // pdTdt[idx]      += gp_TQparam*(del_U + del_W)/(delT * rho_orig * state->specificHeat);
        
            // Compute changes in energy v2:
            // Update bulk modulus:

            // state->initialBulkModulus = d_eos->computeBulkModulus(state->initialDensity, state->density);
            // state->bulkModulus = state->initialBulkModulus;
            // if(d_useDamage){
            //   if(d_brittle_damage.useNonlocalDamage){
            //     state->bulkModulus  =
            //       calculateBulkPrefactor(nonLocalDamage, state, JEL)*state->initialBulkModulus;
            //     state->shearModulus =
            //       calculateShearPrefactor(nonLocalDamage, state)*state->initialShearModulus;
            //   } else {
            //     state->bulkModulus  =
            //       calculateBulkPrefactor(pDamage_new[idx], state, JEL)*state->initialBulkModulus;
            //     state->shearModulus =
            //       calculateShearPrefactor(pDamage_new[idx], state)*state->initialShearModulus;
            //   }
            // }

            // tr_d_p = (p_trial - p_new)/(delT * state->bulkModulus);
            // double del_U = p_new * tr_d_p *delT;
            // // double del_U = 0.5*(p_trial*p_trial - p_new*p_new)/(state->bulkModulus);
            // // double del_W = 0.5*(tauDevTrial.NormSquared() - tauDev.NormSquared())/state->shearModulus;
            // // double del_W = tauDev.Norm()*devPlasticStrainRate.Norm()*delT;
            // double del_W = tauDev.Contract(devPlasticStrainRate)*delT;

            double IEl_tr = IEl;
            // update IEl based on the ratio of effective strain energies:
            IEl = (IEl_tr - 1) * (tauDev.NormSquared()/tauDevTrial.NormSquared()) + 1;
            // // update the dev strain energy so that W_{n+1} = W_tr - del_W
            // IEl = 1 + onethird * ( (0.5*state->shearModulus*(3*IEl_tr -3)) - del_W )*2.0/state->shearModulus;
            bElBar_new[idx]     = tauDev/state->shearModulus + Identity*IEl;
            pGP_strain[idx]     = pGP_strain[idx] + delT*devPlasticStrainRate.Norm();

	                // Compute the changes in energy:
            double del_vol_plast_str = ((p_trial - p_new)/state->bulkModulus);
            double del_U = (p_new) * del_vol_plast_str;
            double del_Thermal = state->specificHeat * delT * rho_orig *
              d_eos->computeIsentropicTemperatureRate(pTemperature[idx], rho_orig,
                                                  state->density,
                                                  (JGP - pGPJ_old[idx])/(delT*JGP)
                                                  );
            del_U = d_eos->computeStrainEnergy(rho_orig,rho_orig * pGPJ_old[idx]/J) -
              d_eos->computeStrainEnergy(rho_orig,rho_orig * JGP/J)+del_Thermal ;
            if(d_useDamage){
              if(d_brittle_damage.useNonlocalDamage){
                del_U *= calculateBulkPrefactor(nonLocalDamage,state,J);
              } else {
                del_U *= calculateBulkPrefactor(pDamage_new[idx],state,J);
              }
            }
            
            double del_W = 0.5*state->shearModulus * 3.0 *(IEl_tr - IEl); // Note IEl is 1/3* tr(be)
            
            if(del_U + del_W < 0 && del_U<0){
              for (int i = 0; i<110; i++){
                tr_d_p *= 0.9;
                // JGP = pGPJ_old[idx]/(1-delT*tr_d_p);
                JGP = pGPJ_old[idx]*exp(delT*tr_d_p);
                pGPJ[idx] = JGP;
                JEL = J/JGP;
                state->density = rho_orig/JEL;

                if( d_useDamage){
                  if(d_brittle_damage.useNonlocalDamage){
                    p_new = computePressure(matl, Identity * cbrt(JEL), pDefGradInc,
                                            state, delT, nonLocalDamage);
                  } else {
                    p_new = computePressure(matl, Identity * cbrt(JEL), pDefGradInc,
                                            state, delT, pDamage_new[idx]);
                  }
                } else {
                  p_new = computePressure(matl, Identity*cbrt(JEL), pDefGradInc,
                                          state, delT, 0);
                }
                del_Thermal = state->specificHeat * delT * rho_orig *
                  d_eos->computeIsentropicTemperatureRate(pTemperature[idx], rho_orig,
                                                          state->density,
                                                          (JGP - pGPJ_old[idx])/(delT*JGP)
                                                          );
                del_U = d_eos->computeStrainEnergy(rho_orig,rho_orig * pGPJ_old[idx]/J) -
                  d_eos->computeStrainEnergy(rho_orig,rho_orig * JGP/J)+del_Thermal ;
                if(d_useDamage){
                  if(d_brittle_damage.useNonlocalDamage){
                    del_U *= calculateBulkPrefactor(nonLocalDamage,state,J);
                  } else {
                    del_U *= calculateBulkPrefactor(pDamage_new[idx],state,J);
                  }
                }
                if(del_U + del_W >= 0){
                  break;
                }
                if(i>100){
                  // tr_d_p = 26e-6* the original tr_d_p, set it to 0.
                  del_U=0;
                  tr_d_p=0;
                  JGP = pGPJ_old[idx];
                  pGPJ[idx] = JGP;
                  JEL = J/JGP;
                  state->density = rho_orig/JEL;
                  if( d_useDamage){
                    if(d_brittle_damage.useNonlocalDamage){
                      p_new = computePressure(matl, Identity * cbrt(JEL), pDefGradInc,
                                              state, delT, nonLocalDamage);
                    } else {
                      p_new = computePressure(matl, Identity * cbrt(JEL), pDefGradInc,
                                              state, delT, pDamage_new[idx]);
                    }
                  } else {
                    p_new = computePressure(matl, Identity*cbrt(JEL), pDefGradInc,
                                            state, delT, 0);
                  }
                }
              } // End of  for(int i = 0; i<110; i++) 
            }   // End of  if(del_U + del_W < 0 && p_trial < 0)

            if(del_U + del_W < 0){
              if( (del_W < 0.0) & (del_W > -1e-9*state->shearModulus) ) {
                JGP = J;
                pGPJ[idx] = JGP;
                JEL = 1.0;
                state->density = rho_orig;
                bElBar_new[idx] = Identity;
                tauDev = 0*Identity;
              } else {
                cerr << "Negative plastic work detected: del_U+del_W=\t" << del_U+del_W << endl;
                cerr << "Trial State:\n"
                     << "p_trial:\t" << p_trial << "\n"
                     << "J:\t" << J << "\t pGPJ_old:\t" << pGPJ_old[idx] << "\n"
                     << "tauDevTrial.Norm():\t" << tauDevTrial.Norm() <<"\t tauDevTrial:" << endl;
                tauDevTrial.prettyPrint(cerr);

                cerr << "Bulk modulus:\t" << state->bulkModulus << "\t"
                     << "Shear modulus:\t" << state->shearModulus << endl;
                if(d_useDamage){
                  cerr << "Damage:\t" << pDamage_new[idx]<< endl;
                }
              
                cerr << "Return State:\n"
                     << "p_new:\t" << p_new << "\t p_target:\t" << p_target << "\n"
                     << "J:\t" << J << "\t pGPJ:\t" << JGP << "\n"
                     << "tr_d_p:\t" << tr_d_p << "\n"
                     << "tauDev.Norm():\t" << tauDev.Norm() << "\t tauDev:" << endl;
                tauDev.prettyPrint(cerr);
                cerr << "del_U:\t" << del_U << "\t del_W:\t" << del_W << endl;
                
                throw runtime_error("Neg Plastic work");
              }
            }
            pGP_energy[idx] +=  gp_TQparam*(del_U + del_W);
            if(p_new < 0.0){        // only allow frictional heating under compressive states
              pdTdt[idx]      += gp_TQparam*(del_U + del_W)/(delT * rho_orig * state->specificHeat);
            }
            
          } else {
            pGPJ[idx] = JGP;
            JEL = J/JGP;
            state->density = rho_orig/JEL;
          } // End if(sigma_hat > Sigma_c)
        } else { // End damage level test
          // Leave bElBar_new[idx] alone since there is no GP flow it does not
          // need to be updated.
          pGPJ[idx] = JGP;
          JEL = J/JGP;
          state->density = rho_orig/JEL;
        }

        // p-\alpha pore compaction model: -----------------

        // Model parameters: (Note pressure P is + compression for
        // this calculation, this is opposite the standard used in
        // the equation of state calculation).
        double Pc = d_GPData.Pc;
        double JGP_e = d_GPData.alpha_e;
        double Pe = d_GPData.Pe;
        double Ps;
        if( d_useDamage){
          if(d_brittle_damage.useNonlocalDamage){
            Ps = -computePressure(matl, Identity * cbrt(JEL), pDefGradInc,
                                    state, delT, nonLocalDamage);
            state->bulkModulus  = calculateBulkPrefactor(nonLocalDamage, state, JEL)
              *state->initialBulkModulus;

          } else {
            Ps = -computePressure(matl, Identity * cbrt(JEL), pDefGradInc,
                                    state, delT, pDamage_new[idx]);
            state->bulkModulus  = calculateBulkPrefactor(pDamage_new[idx], state, JEL)
              *state->initialBulkModulus;
          }

        } else {
          Ps = -computePressure(matl, Identity*cbrt(JEL), pDefGradInc,
                                state, delT, 0);
        }

        double f_phi_tr;
        double dPs_dJGP = (state->bulkModulus - Ps/JEL)*(JEL/JGP);
        double kappa = (Pc-Pe)/(2*Pe*(JGP_e - 1.0));
        if (Ps < J*Pe){
          f_phi_tr = Ps/(J*Pc-J*Pe) - Pe/(Pc-Pe)*exp(-kappa*(JGP - JGP_e));
        } else if(Ps < J*Pc){
          f_phi_tr = (JGP-1.0) - (JGP_e - 1.0)* pow((J*Pc - Ps)/(J*Pc - J*Pe), 2);
        } else {
          f_phi_tr = JGP - 1.0;
        }
        double abs_toll = 1e-8;
        if (f_phi_tr>abs_toll) {

          // calculate the equlibrium JGP:
          double del_JGP_eq(0), df_dJGP, JGP_k, /*JE_k,*/ Ps_k(Ps);
        
          for (int k = 0; abs(f_phi_tr)>abs_toll && k<200; k++){
            JGP_k = JGP+del_JGP_eq;
            //JE_k = J/JGP_k;
            Ps_k = Ps + dPs_dJGP*del_JGP_eq;

            if(Ps_k < J*Pe){
              f_phi_tr = Ps_k/(J*Pc-J*Pe) - Pe/(Pc-Pe)*exp(-kappa*(JGP_k - JGP_e));
              df_dJGP = dPs_dJGP/(J*Pc-J*Pe) + kappa*Pe/(Pc-Pe)*exp(-kappa*(JGP_k - JGP_e));
            } else if(Ps_k < J*Pc){
              f_phi_tr = (JGP_k-1.0) - (JGP_e - 1.0)* pow((J*Pc - Ps_k)/(J*Pc - J*Pe), 2);
              df_dJGP = 1.0 + 2.0 * (JGP_e - 1.0)*( (J*Pc - Ps_k)/(J*Pc - J*Pe) ) *
                dPs_dJGP/(J*(Pc-Pe));
            } else {
              f_phi_tr = (JGP_k-1.0);
              df_dJGP = 1;
              del_JGP_eq = 1-JGP;
              break;
            }
            del_JGP_eq -= f_phi_tr/df_dJGP;            

            if(abs(del_JGP_eq) < abs_toll || abs(f_phi_tr/df_dJGP) < abs_toll){
              break;
            }

            // Check to make sure I am not using too many iterations:
            if(k == 199){
              throw runtime_error("Solver in TongeRamesh GP porosity calc reached max number of iterations");              
            }
          }
          
          if(JGP + del_JGP_eq<1.0){
            del_JGP_eq = 1.0 - JGP;
          }

          double del_JGP_pore = del_JGP_eq;
          if(d_GPData.timeConstant >0){
            double inv_timeConst = 1.0/d_GPData.timeConstant;
            del_JGP_pore = (JGP + delT * inv_timeConst * (JGP + del_JGP_eq))
              /(1+delT*inv_timeConst) - JGP;
          }
          JGP += del_JGP_pore;
          pGPJ[idx] = JGP;
          JEL = J/JGP;
          state->density = rho_orig/JEL;

          // Compute the energy dissipated in the pore compaction process:
          double sigma_m_old = -Ps;
          double sigma_m_new;
          if( d_useDamage){
            if(d_brittle_damage.useNonlocalDamage){
              sigma_m_new = computePressure(matl, Identity * cbrt(JEL), pDefGradInc,
                                            state, delT, nonLocalDamage);
            } else {
              sigma_m_new = computePressure(matl, Identity * cbrt(JEL), pDefGradInc,
                                            state, delT, pDamage_new[idx]);
            }
          } else {
            sigma_m_new = computePressure(matl, Identity*cbrt(JEL), pDefGradInc,
                                  state, delT, 0);
          }
          // double del_Thermal = state->specificHeat * delT * rho_orig *
          //   d_eos->computeIsentropicTemperatureRate(pTemperature[idx], rho_orig,
          //                                           state->density,
          //                                           - (del_JGP_pore)/(delT*JGP) 
          //                                           );
          // double del_U = d_eos->computeStrainEnergy(rho_orig,rho_orig * (JGP-del_JGP_pore)/J) -
          //   d_eos->computeStrainEnergy(rho_orig,rho_orig * JGP/J)-del_Thermal ;
          // if(d_useDamage){
          //   if(d_brittle_damage.useNonlocalDamage){
          //     del_U *= calculateBulkPrefactor(nonLocalDamage,state,J);
          //   } else {
          //     del_U *= calculateBulkPrefactor(pDamage_new[idx],state,J);
          //   }
          // }

          double pore_energyRate = 0.5*(sigma_m_new+sigma_m_old)*del_JGP_pore/(JGP*delT);
          // double pore_energyRate = del_U;
          pdTdt[idx] += gp_TQparam * pore_energyRate/ (rho_orig *state->specificHeat);
          pGP_energy[idx] += gp_TQparam * pore_energyRate*delT;
          
        } // End of porosity calculation
      }   // End if(d_useGranularPlasticity)

      // End Granular Plasticity Calculation:

      // Compute the pressure ---------------------------------------
      
      // get the hydrostatic part of the stress
      if( d_useDamage){
        if(d_brittle_damage.useNonlocalDamage){
          p = computePressure(matl, pDefGrad_new[idx]/cbrt(JGP), pDefGradInc,
                            state, delT, nonLocalDamage);
        } else {
          p = computePressure(matl, pDefGrad_new[idx]/cbrt(JGP), pDefGradInc,
                            state, delT, pDamage_new[idx]);
        }
      } else {
        p = computePressure(matl, pDefGrad_new[idx]/cbrt(JGP), pDefGradInc,
                            state, delT, 0);
      }

      // Assign the new total stress -------------------------------
      IEl = bElBar_new[idx].Trace()/3.0;
      tauDev = (bElBar_new[idx] - Identity*IEl)*state->shearModulus;
              
      // compute the total stress (volumetric + deviatoric)
      pStress[idx] = (Identity*p + tauDev)/J;

      // Do erosion algorithms -------------------------------------
      if( d_useDamage){
        if(d_useGranularPlasticity){
          // Carry forward pLocalized value:
          pLocalized_new[idx] = pLocalized[idx];
          
          // Check for new localized particles:
          if(pLocalized_new[idx]==0 && JGP>d_GPData.JGP_loc){
            pLocalized_new[idx]=1;
          }
        }
        
        checkStabilityAndDoErosion(defGrad, pDamage_new[idx],
                                   pLocalized[idx], pLocalized_new[idx],
                                   pStress[idx], pParticleID[idx], state);
                  
        if (pLocalized_new[idx]>0){
          totalLocalizedParticle+=1;
          // For localized particles modify the deformation gradient if the
          // Jacobian is sufficiantly tensile. If the particle is localized
          // and the Jacobian large then modify the deformation gradient such
          // that the deformation state is still tensile, but just less tensile.
          // This will work ok if the erosion algorithm is AllowNoTension, or
          // AllowNoShear. This could introduce issues when I add plasitcity, and
          // granular plasticity b/c I need to keep everything consistant. I need
          // to make sure that a particle does not switch from tensile to compressive
          // because of what I do.
        }
      }
      
      // Compute the strain energy for non-localized particles --------

      // Compute the increment in strain energy due to the deviatoric
      if(!d_useDamage || pLocalized_new[idx]==0){
        // stress:
        if(d_useDamage){
          if(d_brittle_damage.useNonlocalDamage){
            U = calculateBulkPrefactor(nonLocalDamage,state,J)
              *d_eos->computeStrainEnergy(rho_orig,state->density);
          } else {
            U = calculateBulkPrefactor(pDamage_new[idx],state,J)
              *d_eos->computeStrainEnergy(rho_orig,state->density);
          }
        } else {
          U = d_eos->computeStrainEnergy(rho_orig, state->density);
        }
        W = .5*state->shearModulus*(bElBar_new[idx].Trace() - 3.0);
        // W = 0.5*tauDev.NormSquared()/state->shearModulus;
        double e = (U + W)*pMass[idx]/(rho_orig);
        se += e;
      }

      // Compute temperature rise ----------------------------------

      double heatRate = 0.0;
      // Isentropic from EOS
      // Compute the elastic rate of volume change.
      
      double tr_d_el(0);
      if(d_useGranularPlasticity){
        // Correct for volume change associated with granular plasticity
        double JEl_old = pDefGrad[idx].Determinant()/pGPJ_old[idx];
        double JEl_new = J/pGPJ[idx];
        tr_d_el = (JEl_new - JEl_old)/(delT * JEl_new);
      } else {
        double JEl_old = pDefGrad[idx].Determinant();
        double JEl_new = J;
        tr_d_el = (JEl_new - JEl_old)/(delT * JEl_new);
      }
      heatRate += (state->bulkModulus/state->initialBulkModulus) *
          d_eos->computeIsentropicTemperatureRate(pTemperature[idx], rho_orig,
                                                  state->density,
                                                  tr_d_el
                                                  );
      pdTdt[idx] += heatRate;
      
      pEnergy_new[idx]= U+W;    // This is the strain energy density stored in the particle
      // set the heating rate based on the work done and the change in strain energy:
      // double StressPower = J*pStress[idx].Contract(pDeformRate[idx]);
      // double StressWork = delT * StressPower + 0.5 * delT * delT * StressPower * StressPower;
      // pdTdt[idx] = (J*pStress[idx].Contract(pDeformRate[idx]) - (pEnergy_new[idx] - pEnergy[idx])/delT)/
      //                   (rho_orig * state->specificHeat);
      // pdTdt[idx] = (StressWork - (pEnergy_new[idx] - pEnergy[idx]))/
      //                   (delT * rho_orig * state->specificHeat);


      // Compute the local sound speed -------------------------------------------------
      
      if(d_useDamage && pLocalized_new[idx]!=0){
        c_dil = 0;             // Localized particles should not contribute to the stable timestep calculation
      } else {
        if(d_useGranularPlasticity){
          // Make sure that I correct for the volumetric expansion, rho_cur gets corrupted and
          // represents the elastic density and does not include the effect of the GPJ.
          c_dil = sqrt((state->bulkModulus + 4.*state->shearModulus/3.)/rho_orig);
          // c_dil = sqrt(((state->bulkModulus + 4.*state->shearModulus/3.)/rho_orig)*J);
        } else {
          c_dil = sqrt((state->bulkModulus + 4.*state->shearModulus/3.)/rho_cur);
        }
      }
      
      // Compute wave speed at each particle, store the maximum
      Vector pvel = pVelocity[idx];
      WaveSpeed=Vector(Max(c_dil+fabs(pvel.x()),WaveSpeed.x()),
                       Max(c_dil+fabs(pvel.y()),WaveSpeed.y()),
                       Max(c_dil+fabs(pvel.z()),WaveSpeed.z()));
      
      // Compute artificial viscosity term --------------------
      
      if (flag->d_artificial_viscosity) {
        double dx_ave = (dx.x() + dx.y() + dx.z())/3.0;
        double c_bulk = sqrt(state->bulkModulus/rho_cur);
        double Dkk = pDeformRate[idx].Trace();
        p_q[idx] = artificialBulkViscosity(Dkk, c_bulk,
                                           rho_cur, dx_ave);
        // Include the heating from artificial viscosity:
        if (flag->d_artificial_viscosity_heating) {
          pdTdt[idx] +=  J*Dkk*(-p_q[idx])/(rho_orig*state->specificHeat);
        }
      } else {
        p_q[idx] = 0.;
      }

      // Delete the Plasticity state
      delete state;
    } // end loop over particles
    
    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    if(d_brittle_damage.useDamageTimeStep){
      delT_new = min(delT_new, damage_dt);
    }
    
    new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
    if (flag->d_reductionVars->accStrainEnergy ||
        flag->d_reductionVars->strainEnergy) {
      new_dw->put(sum_vartype(se),        lb->StrainEnergyLabel);
    }
 
    delete interpolator;
    if (d_useDamage) {
      new_dw->put(sumlong_vartype(totalLocalizedParticle),
                  lb->TotalLocalizedParticleLabel);

      // Clean up the damage related arrays:
      delete [] pWingLength_array;
      delete [] pflawSize_array;
      delete [] pFlawNumber_array;

      delete [] pWingLength_array_new;
      delete [] pflawSize_array_new;
      delete [] pFlawNumber_array_new;
    }
  }
}

void TongeRamesh::computeStressTensor(const PatchSubset* patches,
                                      const MPMMaterial* matl,
                                      DataWarehouse* old_dw,
                                      DataWarehouse* new_dw,
                                      Solver* solver,
                                      const bool )

{
  throw std::runtime_error("The TongeRamash material model is not designed to be used with implicit analysis");

  // Constants
  int dwi         = matl->getDWIndex();
  double onethird = (1.0/3.0);
  double shear    = d_initialData.tauDev;
  double bulk     = d_initialData.Bulk;
  double rho_orig = matl->getInitialDensity();
  
  Ghost::GhostType gac = Ghost::AroundCells;
  Matrix3 Identity; Identity.Identity();
  DataWarehouse* parent_old_dw = new_dw->getOtherDataWarehouse(Task::ParentOldDW);
  
  // Particle and grid variables
  constParticleVariable<double>  pVol,pMass,pvolumeold;
  constParticleVariable<Point>   px;
  constParticleVariable<Matrix3>  pSize;
  constParticleVariable<Matrix3> pDefGrad, pBeBar;
  constNCVariable<Vector>        gDisp;
  ParticleVariable<double>       pVolume_new;
  ParticleVariable<Matrix3>      pDefGrad_new, pBeBar_new, pStress;
  
  // Local variables
  Matrix3 tauDev(0.0), pDefGradInc(0.0), pDispGrad(0.0), pRelDefGradBar(0.0);
  double D[6][6];
  double B[6][24];
  double Bnl[3][24];
  // Unused because not using computeStiffnessMatrix() as in CNHPDamage
  //     double Kmatrix[24][24];
  int dof[24];
  // Unused because each 8 and 27 option have their owndouble v[576];

  IntVector lowIndex=IntVector(0,0,0),highIndex=IntVector(0,0,0); 
  for(int pp=0;pp<patches->size();pp++){
    const Patch* patch = patches->get(pp);
    
    if(d_8or27==8){
      lowIndex  = patch->getNodeLowIndex();
      highIndex = patch->getNodeHighIndex()+IntVector(1,1,1);
    } else if(d_8or27==27){
      lowIndex  = patch->getExtraNodeLowIndex();
      highIndex = patch->getExtraNodeHighIndex()+IntVector(1,1,1);
    }
    
    Array3<int> l2g(lowIndex,highIndex);
    solver->copyL2G(l2g,patch);
    
    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};
    
    ParticleSubset* pset = parent_old_dw->getParticleSubset(dwi, patch);
    parent_old_dw->get(px,       lb->pXLabel,                  pset);
    parent_old_dw->get(pSize,    lb->pSizeLabel,               pset);
    parent_old_dw->get(pMass,    lb->pMassLabel,               pset);
    parent_old_dw->get(pvolumeold, lb->pVolumeLabel,           pset);
    parent_old_dw->get(pDefGrad, lb->pDeformationMeasureLabel, pset);
    parent_old_dw->get(pBeBar,   bElBarLabel,                  pset);

    new_dw->allocateAndPut(pStress,     lb->pStressLabel_preReloc, pset);
    new_dw->allocateAndPut(pVolume_new, lb->pVolumeDeformedLabel,  pset);
    new_dw->allocateTemporary(pDefGrad_new, pset);
    new_dw->allocateTemporary(pBeBar_new,   pset);
    
    ParticleSubset::iterator iter = pset->begin();
    
    double volold, volnew;
    
    if(matl->getIsRigid()){ // Rigid test
      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        particleIndex idx = *iter;
        pStress[idx] = Matrix3(0.0);
        pVolume_new[idx] = pvolumeold[idx];
      }
    }
    else{
      ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
      vector<IntVector> ni(interpolator->size());
      vector<Vector> d_S(interpolator->size());


      if(flag->d_doGridReset){
        constNCVariable<Vector> dispNew;
        old_dw->get(dispNew,lb->dispNewLabel,dwi,patch, gac, 1);
        computeDeformationGradientFromIncrementalDisplacement(
                                                              dispNew, pset, px,
                                                              pDefGrad,
                                                              pDefGrad_new,
                                                              dx, pSize, interpolator);
      }
      else if(!flag->d_doGridReset){
        constNCVariable<Vector> gdisplacement;
        old_dw->get(gdisplacement, lb->gDisplacementLabel,dwi,patch,gac,1);
        computeDeformationGradientFromTotalDisplacement(gdisplacement,
                                                        pset, px,
                                                        pDefGrad_new,
                                                        pDefGrad,
                                                        dx, pSize,interpolator);
      }

      if((d_usePlasticity || d_useDamage) && flag->d_doGridReset){
        old_dw->get(gDisp,           lb->dispNewLabel, dwi, patch, gac, 1);
      }

      for(iter = pset->begin(); iter != pset->end(); iter++){
        particleIndex idx = *iter;
      
        // Compute the displacement gradient and B matrices
        if(d_usePlasticity || d_useDamage){
          interpolator->findCellAndShapeDerivatives(px[idx], ni, d_S, 
                                                    pSize[idx],pDefGrad[idx]);
      
          computeGradAndBmats(pDispGrad,ni,d_S, oodx, gDisp, l2g,B, Bnl, dof);
        }

        // Compute the deformation gradient increment using the pDispGrad
        // Update the deformation gradient tensor to its time n+1 value.
        double J;

        pDefGradInc = pDispGrad + Identity;
        if(d_usePlasticity || d_useDamage) {
          pDefGrad_new[idx] = pDefGradInc*pDefGrad[idx];
          J = pDefGrad_new[idx].Determinant();

          // Compute BeBar
          pRelDefGradBar = pDefGradInc/cbrt(pDefGradInc.Determinant());

          pBeBar_new[idx] = pRelDefGradBar*pBeBar[idx]*pRelDefGradBar.Transpose();
        } else {
          J = pDefGrad_new[idx].Determinant();
          Matrix3 bElBar_new = pDefGrad_new[idx]
            * pDefGrad_new[idx].Transpose()
            * pow(J,-(2./3.));
          pBeBar_new[idx] = bElBar_new;
        }
        
        // Update the particle volume
        volold = (pMass[idx]/rho_orig);
        volnew = volold*J;

        // tauDev is equal to the shear modulus times dev(bElBar)
        double mubar   = onethird*pBeBar_new[idx].Trace()*shear;
        Matrix3 shrTrl = (pBeBar_new[idx]*shear - Identity*mubar);
      
        // get the hydrostatic part of the stress
        double p = bulk*log(J)/J;
      
        // compute the total stress (volumetric + deviatoric)
        pStress[idx] = Identity*p + shrTrl/J;

        // Compute the tangent stiffness matrix
        computeTangentStiffnessMatrix(shrTrl, mubar, J, bulk, D);
      
        
        double sig[3][3];
        for (int i = 0; i < 3; i++) {
          for (int j = 0; j < 3; j++) {
            sig[i][j]=pStress[idx](i,j);
          }
        }
        
        int nDOF=3*d_8or27;      
      
        if(d_8or27==8){

          double B[6][24];
          double Bnl[3][24];
          int dof[24];
          double v[24*24];
          double kmat[24][24];
          double kgeo[24][24];
        
          // Fill in the B and Bnl matrices and the dof vector
          interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S, pSize[idx],pDefGrad[idx]);
          loadBMats(l2g,dof,B,Bnl,d_S,ni,oodx);
          // kmat = B.transpose()*D*B*volold
          BtDB(B,D,kmat);
          // kgeo = Bnl.transpose*sig*Bnl*volnew;
          BnltDBnl(Bnl,sig,kgeo);
        
          for (int I = 0; I < nDOF;I++){
            for (int J = 0; J < nDOF; J++){
              v[nDOF*I+J] = kmat[I][J]*volold + kgeo[I][J]*volnew;
            }
          }
          solver->fillMatrix(nDOF,dof,nDOF,dof,v);
        } else {
          double B[6][81];
          double Bnl[3][81];
          int dof[81];
          double v[81*81];
          double kmat[81][81];
          double kgeo[81][81];
        
          // the code that computes kmat doesn't yet know that D is symmetric
          D[1][0] = D[0][1];
          D[2][0] = D[0][2];
          D[3][0] = D[0][3];
          D[4][0] = D[0][4];
          D[5][0] = D[0][5];
          D[1][1] = D[1][1];
          D[2][1] = D[1][2];
          D[3][1] = D[1][3];
          D[4][1] = D[1][4];
          D[1][2] = D[2][1];
          D[2][2] = D[2][2];
          D[3][2] = D[2][3];
          D[1][3] = D[3][1];
          D[2][3] = D[3][2];
          D[4][3] = D[3][4];
            
          // Fill in the B and Bnl matrices and the dof vector
          interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S, pSize[idx],pDefGrad[idx]);
          loadBMatsGIMP(l2g,dof,B,Bnl,d_S,ni,oodx);
          // kmat = B.transpose()*D*B*volold
          BtDBGIMP(B,D,kmat);
          // kgeo = Bnl.transpose*sig*Bnl*volnew;
          BnltDBnlGIMP(Bnl,sig,kgeo);
          
          for (int I = 0; I < nDOF;I++){
            for (int J = 0; J < nDOF; J++){
              v[nDOF*I+J] = kmat[I][J]*volold + kgeo[I][J]*volnew;
            }
          }
          solver->fillMatrix(nDOF,dof,nDOF,dof,v);
        }
        pVolume_new[idx] = volnew;
      }
      delete interpolator;
    } // end rigid
  }  // end of loop over particles
  
  solver->flushMatrix();
}

// Helper Functions //
//////////////////////
double TongeRamesh::getCompressibility()
{
  return 1.0/d_initialData.Bulk;
}

void TongeRamesh::getDamageParameter(const Patch* patch,
                                     ParticleVariable<int>& damage,
                                     int dwi,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw)
{
  if(d_useDamage){
    ParticleSubset* pset = old_dw->getParticleSubset(dwi,patch);
    constParticleVariable<int> pLocalized;
    new_dw->get(pLocalized, pLocalizedLabel_preReloc, pset);
    
    ParticleSubset::iterator iter;
    for (iter = pset->begin(); iter != pset->end(); iter++) {
      damage[*iter] = pLocalized[*iter];
    }
  }
}

void TongeRamesh::addParticleState(std::vector<const VarLabel*>& from,
                                   std::vector<const VarLabel*>& to)
{
  // Add the local particle state data for this constitutive model.
  // Plasticity
  if(d_usePlasticity) {
    from.push_back(pPlasticStrain_label);
    from.push_back(pPlasticEnergy_label);
    to.push_back(pPlasticStrain_label_preReloc);
    to.push_back(pPlasticEnergy_label_preReloc);
  }
  
  // Damage
  if(d_useDamage) {
    // from.push_back(bBeBarLabel);
    // from.push_back(pFailureStressOrStrainLabel);
    from.push_back(pLocalizedLabel);
    from.push_back(pDamageLabel);

    for(int i = 0; i < d_flawDistData.numCrackFamilies; i++){
      from.push_back(wingLengthLabel_array[i]);
      from.push_back(flawNumber_array[i]);
      from.push_back(starterFlawSize_array[i]);
    }
    
    // to.push_back(bBeBarLabel_preReloc);
    // to.push_back(pFailureStressOrStrainLabel_preReloc);
    to.push_back(pLocalizedLabel_preReloc);
    to.push_back(pDamageLabel_preReloc);

    for(int i = 0; i < d_flawDistData.numCrackFamilies; i++){
      to.push_back(wingLengthLabel_array_preReloc[i]);
      to.push_back(flawNumber_array_preReloc[i]);
      to.push_back(starterFlawSize_array_preReloc[i]);
    }
  }

  // Granular Plasticity
  if(d_useGranularPlasticity) {
    from.push_back(pGPJLabel);
    from.push_back(pGP_plasticStrainLabel);
    from.push_back(pGP_plasticEnergyLabel);

    to.push_back(pGPJLabel_preReloc);
    to.push_back(pGP_plasticStrainLabel_preReloc);
    to.push_back(pGP_plasticEnergyLabel_preReloc);
  }
  
  // Universal
  from.push_back(bElBarLabel);
  to.push_back(bElBarLabel_preReloc);
  from.push_back(pEnergyLabel);
  to.push_back(pEnergyLabel_preReloc);
  if (flag->d_integrator != MPMFlags::Implicit) {
    from.push_back(pDeformRateLabel);
    to.push_back(pDeformRateLabel_preReloc);
  }
}

// Damage requirements //
/////////////////////////
void TongeRamesh::checkStabilityAndDoErosion(const Matrix3& defGrad,
                                             const double& currentDamage,
                                             const int& pLocalized,
                                             int& pLocalized_new,
                                             Matrix3& pStress,
                                             const long64 particleID,
                                             const PlasticityState *state)
{
  Matrix3 Identity, zero(0.0); Identity.Identity();
  
  if(!d_useGranularPlasticity){
    // Compute localization for damage:

    // Find if the particle has failed
    pLocalized_new = pLocalized;
    if (pLocalized == 0 && currentDamage >= d_brittle_damage.criticalDamage){
      pLocalized_new = 1;
      if (d_brittle_damage.printDamage){
        cout << "Particle " << particleID << " failed:"
          " damage=" << currentDamage << endl;
      }
    }
  } // End if(d_useGranularPlasticity)
  
  // If the particle has failed, apply various erosion algorithms
  if (flag->d_doErosion) {
    // Compute pressure
    double pressure = pStress.Trace()/3.0;
    if (pLocalized_new != 0) {
      if (d_allowNoTension) {
        if (pressure > 0.0)
          pStress = zero;
        else
          pStress = Identity*pressure;
      } else if (d_allowNoShear)
        pStress = Identity*pressure;
      else if (d_setStressToZero)
        pStress = zero;
    }
  }
}


void TongeRamesh::computeStressTensorImplicit(const PatchSubset* patches,
                                              const MPMMaterial* matl,
                                              DataWarehouse* old_dw,
                                              DataWarehouse* new_dw)
{

  ostringstream msg;
  msg << "\n ERROR: In TongeRamesh::computeStressTensorImplicit \n"
      << "\t This function has not been updated and should not be used. \n";
  throw ProblemSetupException(msg.str(),__FILE__, __LINE__);

  
  //   // Constants
  //   double onethird = (1.0/3.0);
  //   double sqtwthds = sqrt(2.0/3.0);
  //   Matrix3 Identity; Identity.Identity();
  //   Ghost::GhostType gac = Ghost::AroundCells;
  
  //   double rho_orig    = matl->getInitialDensity();
  //   double shear       = d_initialData.tauDev;
  //   double bulk        = d_initialData.Bulk;
  //   double flowStress  = d_initialData.FlowStress;
  //   double hardModulus = d_initialData.K;
  //   double se          = 0.0;
  
  //   int dwi = matl->getDWIndex();
  // nn  
  //   // Particle and grid data
  //   constParticleVariable<int>     pLocalized;
  //   constParticleVariable<double>  pFailureStrain;
  //   constParticleVariable<double>  pMass, pPlasticStrain, pDamage;
  //   constParticleVariable<long64>  pParticleID;
  //   constParticleVariable<Point>   pX;
  //   constParticleVariable<Vector>  pSize;
  //   constParticleVariable<Matrix3> pDefGrad, pBeBar;
  //   constNCVariable<Vector>        gDisp;
  //   ParticleVariable<int>          pLocalized_new;
  //   ParticleVariable<double>       pFailureStrain_new, pDamage_new;
  //   ParticleVariable<double>       pVolume_new, pdTdt, pPlasticStrain_new;
  //   ParticleVariable<Matrix3>      pDefGrad_new, pBeBar_new, pStress_new;

  //   // Local variables 
  //   Matrix3 dispGrad(0.0), tauDev(0.0), defGradInc(0.0);
  //   Matrix3 beBarTrial(0.0), tauDevTrial(0.0), normal(0.0), relDefGradBar(0.0);
  //   Matrix3 defGrad(0.0);
  
  //   // Loop thru patches
  //   for(int pp=0;pp<patches->size();pp++){
  //     const Patch* patch = patches->get(pp);
    
  //     // Get particle info
  //     ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
  //     // Loop thru particles
  //     ParticleSubset::iterator iter = pset->begin();
    
  //     // Initialize patch variables
  //     se = 0.0;
    
  //     // Get patch info
  //     Vector dx = patch->dCell();
  //     // Unused    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};
    
  //     // Plastic gets and allocates
  //     if(d_usePlasticity){
  //       old_dw->get(pPlasticStrain,           pPlasticStrain_label,       pset);
  //       new_dw->allocateAndPut(pPlasticStrain_new, 
  //                              pPlasticStrain_label_preReloc,             pset);

  //       // Copy failure strains to new dw
  //       pFailureStrain_new.copyData(pFailureStrain);
  //     }
    
  //     // Damage gets and allocates
  //     if(d_useDamage){
  //       old_dw->get(pLocalized,               pLocalizedLabel,                    pset);
  //       old_dw->get(pFailureStrain,           pFailureStressOrStrainLabel,        pset);
  //       old_dw->get(pDamage,                  pDamageLabel,                       pset);
  //       old_dw->get(pParticleID,              lb->pParticleIDLabel,               pset); 

  //       new_dw->allocateAndPut(pLocalized_new,
  //                              pLocalizedLabel_preReloc,                          pset);
  //       new_dw->allocateAndPut(pFailureStrain_new, 
  //                              pFailureStressOrStrainLabel_preReloc,              pset);
  //       new_dw->allocateAndPut(pDamage_new, 
  //                              pDamageLabel_preReloc,                             pset);
  //     }
    
  //     // Universal gets and allocates
  //     old_dw->get(pMass,                    lb->pMassLabel,               pset);
  //     old_dw->get(pX,                       lb->pXLabel,                  pset);
  //     old_dw->get(pSize,                    lb->pSizeLabel,               pset);
  //     old_dw->get(pDefGrad,                 lb->pDeformationMeasureLabel, pset);
  //     old_dw->get(pBeBar,                   bElBarLabel,                  pset);
    
  //     // Allocate space for updated particle variables
  //     new_dw->allocateAndPut(pVolume_new, 
  //                            lb->pVolumeDeformedLabel,              pset);
  //     new_dw->allocateAndPut(pdTdt, 
  //                            lb->pdTdtLabel_preReloc,               pset);
  //     new_dw->allocateAndPut(pDefGrad_new,
  //                            lb->pDeformationMeasureLabel_preReloc, pset);
  //     new_dw->allocateAndPut(pBeBar_new, 
  //                            bElBarLabel_preReloc,                  pset);
  //     new_dw->allocateAndPut(pStress_new,        
  //                            lb->pStressLabel_preReloc,             pset);
 
  //     if(matl->getIsRigid()){
  //       for(iter = pset->begin(); iter != pset->end(); iter++){
  //         particleIndex idx = *iter;
  //         // Assign zero internal heating by default - modify if necessary.
  //         pdTdt[idx]        = 0.0;
  //         pStress_new[idx]  = Matrix3(0.0);
  //         pDefGrad_new[idx] = Identity;
  //         pVolume_new[idx]  = pMass[idx]/rho_orig;
  //       }
  //     } else { /*if(!matl->getIsRigid()) */
  //       // Compute the displacement gradient and the deformation gradient
  //       ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
  //       vector<IntVector> ni(interpolator->size());
  //       vector<Vector> d_S(interpolator->size());
  //       if(flag->d_doGridReset){
  //         constNCVariable<Vector> dispNew;
  //         new_dw->get(dispNew,lb->dispNewLabel,dwi,patch, gac, 1);
  //         computeDeformationGradientFromIncrementalDisplacement(
  //                                                               dispNew, pset, pX,
  //                                                               pDefGrad,
  //                                                               pDefGrad_new,
  //                                                               dx, pSize, interpolator);
  //       }
  //       else /*if(!flag->d_doGridReset)*/{
  //         constNCVariable<Vector> gdisplacement;
  //         new_dw->get(gdisplacement, lb->gDisplacementLabel,dwi,patch,gac,1);
  //         computeDeformationGradientFromTotalDisplacement(gdisplacement,pset, pX, 
  //                                                         pDefGrad_new,
  //                                                         pDefGrad,
  //                                                         dx, pSize,interpolator);
  //       }
      
  //       // Unused because no "active stress carried over from CNHImplicit    
  //       //     double time = d_sharedState->getElapsedTime();
    
  //       for(iter = pset->begin(); iter != pset->end(); iter++){
  //         particleIndex idx = *iter;
      
  //         // Assign zero internal heating by default - modify if necessary.
  //         pdTdt[idx]  = 0.0;
      
  //         defGradInc  = dispGrad + Identity;         
  //         double Jinc = defGradInc.Determinant();
      
  //         // Update the deformation gradient tensor to its time n+1 value.
  //         defGrad  = defGradInc*pDefGrad[idx];
  //         double J = pDefGrad_new[idx].Determinant();

  //         if(d_usePlasticity || d_useDamage) {
  //           J = defGrad.Determinant();
  //           pDefGrad_new[idx] = defGrad;
        
  //           // Compute trial BeBar
  //           relDefGradBar = defGradInc/cbrt(Jinc);
       
  //           // Compute the trial elastic part of the volume preserving 
  //           // part of the left Cauchy-Green deformation tensor
  //           beBarTrial = relDefGradBar*pBeBar[idx]*relDefGradBar.Transpose();
  //         } else {
  //           beBarTrial = pDefGrad_new[idx]
  //                        * pDefGrad_new[idx].Transpose()
  //                        * pow(J,-(2./3.));
  //         }
        
  //         if (!(J > 0.0)) {
  //           cerr << getpid() << " " << idx << " "
  //                << "**ERROR** Negative Jacobian of deformation gradient" << endl;
  //           throw ParameterNotFound("**ERROR**:TongeRamesh", __FILE__, __LINE__);
  //         }
        
  //         // Compute the deformed volume 
  //         double rho_cur   = rho_orig/J;
  //         pVolume_new[idx] = (pMass[idx]/rho_orig)*J;

  //         double IEl   = onethird*beBarTrial.Trace();
  //         double muBar = IEl*shear;
      
  //         // tauDevTrial is equal to the shear modulus times dev(bElBar)
  //         // Compute ||tauDevTrial||
  //         tauDevTrial   = (beBarTrial - Identity*IEl)*shear;
  //         double sTnorm = tauDevTrial.Norm();
        
  //         // get the hydrostatic part of the stress
  //         double p = bulk*log(J)/J;
      
  //         // Check for plastic loading
  //         double alpha = 0.0;
  //         if(d_usePlasticity){
  //           pVolume_new[idx]=pMass[idx]/rho_cur;  // To prevent Gold Standards from Crapping
  //           alpha = pPlasticStrain[idx];
  //           p = 0.5*bulk*(J - 1.0/J);
  //         }
  //         double fTrial = sTnorm - sqtwthds*(hardModulus*alpha + flowStress);
      
  //         if (d_usePlasticity && (fTrial > 0.0)) {
  //           // plastic
  //           // Compute increment of slip in the direction of flow
  //           double delgamma = (fTrial/(2.0*muBar)) /
  //                             (1.0 + (hardModulus/(3.0*muBar)));
  //           normal = tauDevTrial/sTnorm;
        
  //           // The actual shear stress
  //           tauDev = tauDevTrial - normal*2.0*muBar*delgamma;
        
  //           // Deal with history variables
  //           pPlasticStrain_new[idx] = alpha + sqtwthds*delgamma;
  //           pBeBar_new[idx]         = tauDev/shear + Identity*IEl;
  //         }
  //         else {
        
  //           // The actual shear stress
  //           tauDev = tauDevTrial;
  //           pBeBar_new[idx] = beBarTrial;
        
  //           // carry forward in implicit
  //           if(d_usePlasticity){
  //             pPlasticStrain_new[idx] = alpha;
  //           }
  //         }
      
  //         // compute the total stress (volumetric + deviatoric)
  //         pStress_new[idx] = Identity*p + tauDev/J;
      
  //         // Modify the stress if particle has damaged/failed
  //         if(d_useDamage){
  // 	  if (d_brittleDamage) {
  //              updateDamageAndModifyStress(defGrad, pFailureStrain[idx], pFailureStrain_new[idx],
  //                   pVolume_new[idx], pDamage[idx], pDamage_new[idx], pStress_new[idx],
  //                   pParticleID[idx]);
  // 	  } else {
  // 	    updateFailedParticlesAndModifyStress(defGrad, pFailureStrain[idx], 
  //                                                pLocalized[idx], pLocalized_new[idx],
  //                                                pStress_new[idx], pParticleID[idx]);
  // 	  }
  //         }
      
  //         // Compute the strain energy for non-localized particles
  //         double U = .5*bulk*(.5*(J*J - 1.0) - log(J));
  //         double W = .5*shear*(pBeBar_new[idx].Trace() - 3.0);
  //         double e = (U + W)*pVolume_new[idx]/J;
  //         se += e;     
  //         // Don't save strain energy if particle is not localized to point 
  //         if(d_useDamage && pLocalized_new[idx] != 0){
  //           se -= e;
  //         }
  //       }
  //       if (flag->d_reductionVars->accStrainEnergy ||
  //           flag->d_reductionVars->strainEnergy) {
  //         new_dw->put(sum_vartype(se), lb->StrainEnergyLabel);
  //       }
  //       delete interpolator;
  //     } // End rigid else
  //   } // End Patch For Loop
}

/*! Compute tangent stiffness matrix */
void TongeRamesh::computeTangentStiffnessMatrix(const Matrix3& sigdev, 
                                                const double&  mubar,
                                                const double&  J,
                                                const double&  bulk,
                                                double D[6][6])
{
  double twth = 2.0/3.0;
  double frth = 2.0*twth;
  double coef1 = bulk;
  double coef2 = 2.*bulk*log(J);
  
  for (int ii = 0; ii < 6; ++ii) {
    for (int jj = 0; jj < 6; ++jj) {
      D[ii][jj] = 0.0;
    }
  }
  D[0][0] = coef1 - coef2 + mubar*frth - frth*sigdev(0,0);
  D[0][1] = coef1 - mubar*twth - twth*(sigdev(0,0) + sigdev(1,1));
  D[0][2] = coef1 - mubar*twth - twth*(sigdev(0,0) + sigdev(2,2));
  D[0][3] =  - twth*(sigdev(0,1));
  D[0][4] =  - twth*(sigdev(0,2));
  D[0][5] =  - twth*(sigdev(1,2));
  D[1][1] = coef1 - coef2 + mubar*frth - frth*sigdev(1,1);
  D[1][2] = coef1 - mubar*twth - twth*(sigdev(1,1) + sigdev(2,2));
  D[1][3] =  D[0][3];
  D[1][4] =  D[0][4];
  D[1][5] =  D[0][5];
  D[2][2] = coef1 - coef2 + mubar*frth - frth*sigdev(2,2);
  D[2][3] =  D[0][3];
  D[2][4] =  D[0][4];
  D[2][5] =  D[0][5];
  D[3][3] =  -.5*coef2 + mubar;
  D[4][4] =  D[3][3];
  D[5][5] =  D[3][3];
}

/*! Compute K matrix */
void TongeRamesh::computeStiffnessMatrix(const double B[6][24],
                                         const double Bnl[3][24],
                                         const double D[6][6],
                                         const Matrix3& sig,
                                         const double& vol_old,
                                         const double& vol_new,
                                         double Kmatrix[24][24])
{
  
  // Kmat = B.transpose()*D*B*volold
  double Kmat[24][24];
  BtDB(B, D, Kmat);
  
  // Kgeo = Bnl.transpose*sig*Bnl*volnew;
  double Kgeo[24][24];
  BnlTSigBnl(sig, Bnl, Kgeo);
  
  /*
    cout.setf(ios::scientific,ios::floatfield);
    cout.precision(10);
    cout << "Kmat = " << endl;
    for(int kk = 0; kk < 24; kk++) {
    for (int ll = 0; ll < 24; ++ll) {
    cout << Kmat[ll][kk] << " " ;
    }
    cout << endl;
    }
    cout << "Kgeo = " << endl;
    for(int kk = 0; kk < 24; kk++) {
    for (int ll = 0; ll < 24; ++ll) {
    cout << Kgeo[ll][kk] << " " ;
    }
    cout << endl;
    }
  */
  
  for(int ii = 0;ii<24;ii++){
    for(int jj = 0;jj<24;jj++){
      Kmatrix[ii][jj] =  Kmat[ii][jj]*vol_old + Kgeo[ii][jj]*vol_new;
    }
  }
}

void TongeRamesh::BnlTSigBnl(const Matrix3& sig, const double Bnl[3][24],
                             double Kgeo[24][24]) const
{
  double t1, t10, t11, t12, t13, t14, t15, t16, t17;
  double t18, t19, t2, t20, t21, t22, t23, t24, t25;
  double t26, t27, t28, t29, t3, t30, t31, t32, t33;
  double t34, t35, t36, t37, t38, t39, t4, t40, t41;
  double t42, t43, t44, t45, t46, t47, t48, t49, t5;
  double t50, t51, t52, t53, t54, t55, t56, t57, t58;
  double t59, t6, t60, t61, t62, t63, t64, t65, t66;
  double t67, t68, t69, t7, t70, t71, t72, t73, t74;
  double t75, t77, t78, t8, t81, t85, t88, t9, t90;
  double t79, t82, t83, t86, t87, t89;
  
  t1  = Bnl[0][0]*sig(0,0);
  t4  = Bnl[0][0]*sig(0,0);
  t2  = Bnl[0][0]*sig(0,1);
  t3  = Bnl[0][0]*sig(0,2);
  t5  = Bnl[1][1]*sig(1,1);
  t8  = Bnl[1][1]*sig(1,1);
  t6  = Bnl[1][1]*sig(1,2);
  t7  = Bnl[1][1]*sig(0,1);
  t9  = Bnl[2][2]*sig(2,2);
  t12 = Bnl[2][2]*sig(2,2);
  t10 = Bnl[2][2]*sig(0,2);
  t11 = Bnl[2][2]*sig(1,2);
  t13 = Bnl[0][3]*sig(0,0);
  t16 = Bnl[0][3]*sig(0,0);
  t14 = Bnl[0][3]*sig(0,1);
  t15 = Bnl[0][3]*sig(0,2);
  t17 = Bnl[1][4]*sig(1,1);
  t20 = Bnl[1][4]*sig(1,1);
  t18 = Bnl[1][4]*sig(1,2);
  t19 = Bnl[1][4]*sig(0,1);
  t21 = Bnl[2][5]*sig(2,2);
  t22 = Bnl[2][5]*sig(0,2);
  t23 = Bnl[2][5]*sig(1,2);
  t24 = Bnl[2][5]*sig(2,2);
  t25 = Bnl[0][6]*sig(0,0);
  t26 = Bnl[0][6]*sig(0,1);
  t27 = Bnl[0][6]*sig(0,2);
  t28 = Bnl[0][6]*sig(0,0);
  t29 = Bnl[1][7]*sig(1,1);
  t30 = Bnl[1][7]*sig(1,2);
  t31 = Bnl[1][7]*sig(0,1);
  t32 = Bnl[1][7]*sig(1,1);
  t33 = Bnl[2][8]*sig(2,2);
  t34 = Bnl[2][8]*sig(0,2);
  t35 = Bnl[2][8]*sig(1,2);
  t36 = Bnl[2][8]*sig(2,2);
  t37 = Bnl[0][9]*sig(0,0);
  t38 = Bnl[0][9]*sig(0,1);
  t39 = Bnl[0][9]*sig(0,2);
  t40 = Bnl[0][9]*sig(0,0);
  t41 = Bnl[1][10]*sig(1,1);
  t42 = Bnl[1][10]*sig(1,2);
  t43 = Bnl[1][10]*sig(0,1);
  t44 = Bnl[1][10]*sig(1,1);
  t45 = Bnl[2][11]*sig(2,2);
  t46 = Bnl[2][11]*sig(0,2);
  t47 = Bnl[2][11]*sig(1,2);
  t48 = Bnl[2][11]*sig(2,2);
  t49 = Bnl[0][12]*sig(0,0);
  t50 = Bnl[0][12]*sig(0,1);
  t51 = Bnl[0][12]*sig(0,2);
  t52 = Bnl[0][12]*sig(0,0);
  t53 = Bnl[1][13]*sig(1,1);
  t54 = Bnl[1][13]*sig(1,2);
  t55 = Bnl[1][13]*sig(0,1);
  t56 = Bnl[1][13]*sig(1,1);
  t57 = Bnl[2][14]*sig(2,2);
  t58 = Bnl[2][14]*sig(0,2);
  t59 = Bnl[2][14]*sig(1,2);
  t60 = Bnl[2][14]*sig(2,2);
  t61 = Bnl[0][15]*sig(0,0);
  t62 = Bnl[0][15]*sig(0,1);
  t63 = Bnl[0][15]*sig(0,2);
  t64 = Bnl[0][15]*sig(0,0);
  t65 = Bnl[1][16]*sig(1,1);
  t66 = Bnl[1][16]*sig(1,2);
  t67 = Bnl[1][16]*sig(0,1);
  t68 = Bnl[1][16]*sig(1,1);
  t69 = Bnl[2][17]*sig(2,2);
  t70 = Bnl[2][17]*sig(0,2);
  t71 = Bnl[2][17]*sig(1,2);
  t72 = Bnl[2][17]*sig(2,2);
  t73 = Bnl[0][18]*sig(0,0);
  t74 = Bnl[0][18]*sig(0,1);
  t75 = Bnl[0][18]*sig(0,2);
  t77 = Bnl[1][19]*sig(1,1);
  t78 = Bnl[1][19]*sig(1,2);
  t79 = Bnl[1][19]*sig(0,1);
  t81 = Bnl[2][20]*sig(2,2);
  t82 = Bnl[2][20]*sig(0,2);
  t83 = Bnl[2][20]*sig(1,2);
  t85 = Bnl[0][21]*sig(0,0);
  t86 = Bnl[0][21]*sig(0,1);
  t87 = Bnl[0][21]*sig(0,2);
  t88 = Bnl[1][22]*sig(1,1);
  t89 = Bnl[1][22]*sig(1,2);
  t90 = Bnl[2][23]*sig(2,2);
  
  Kgeo[0][0]   = t1*Bnl[0][0];
  Kgeo[0][1]   = t2*Bnl[1][1];
  Kgeo[0][2]   = t3*Bnl[2][2];
  Kgeo[0][3]   = t4*Bnl[0][3];
  Kgeo[0][4]   = t2*Bnl[1][4];
  Kgeo[0][5]   = t3*Bnl[2][5];
  Kgeo[0][6]   = t4*Bnl[0][6];
  Kgeo[0][7]   = t2*Bnl[1][7];
  Kgeo[0][8]   = t3*Bnl[2][8];
  Kgeo[0][9]   = t4*Bnl[0][9];
  Kgeo[0][10]  = t2*Bnl[1][10];
  Kgeo[0][11]  = t3*Bnl[2][11];
  Kgeo[0][12]  = t4*Bnl[0][12];
  Kgeo[0][13]  = t2*Bnl[1][13];
  Kgeo[0][14]  = t3*Bnl[2][14];
  Kgeo[0][15]  = t4*Bnl[0][15];
  Kgeo[0][16]  = t2*Bnl[1][16];
  Kgeo[0][17]  = t3*Bnl[2][17];
  Kgeo[0][18]  = t4*Bnl[0][18];
  Kgeo[0][19]  = t2*Bnl[1][19];
  Kgeo[0][20]  = t3*Bnl[2][20];
  Kgeo[0][21]  = t4*Bnl[0][21];
  Kgeo[0][22]  = t2*Bnl[1][22];
  Kgeo[0][23]  = t3*Bnl[2][23];
  Kgeo[1][0]   = Kgeo[0][1];
  Kgeo[1][1]   = t5*Bnl[1][1];
  Kgeo[1][2]   = t6*Bnl[2][2];
  Kgeo[1][3]   = t7*Bnl[0][3];
  Kgeo[1][4]   = Bnl[1][4]*t8;
  Kgeo[1][5]   = t6*Bnl[2][5];
  Kgeo[1][6]   = t7*Bnl[0][6];
  Kgeo[1][7]   = Bnl[1][7]*t8;
  Kgeo[1][8]   = t6*Bnl[2][8];
  Kgeo[1][9]   = t7*Bnl[0][9];
  Kgeo[1][10]  = Bnl[1][10]*t8;
  Kgeo[1][11]  = t6*Bnl[2][11];
  Kgeo[1][12]  = t7*Bnl[0][12];
  Kgeo[1][13]  = Bnl[1][13]*t8;
  Kgeo[1][14]  = t6*Bnl[2][14];
  Kgeo[1][15]  = t7*Bnl[0][15];
  Kgeo[1][16]  = Bnl[1][16]*t8;
  Kgeo[1][17]  = t6*Bnl[2][17];
  Kgeo[1][18]  = t7*Bnl[0][18];
  Kgeo[1][19]  = Bnl[1][19]*t8;
  Kgeo[1][20]  = t6*Bnl[2][20];
  Kgeo[1][21]  = t7*Bnl[0][21];
  Kgeo[1][22]  = Bnl[1][22]*t8;
  Kgeo[1][23]  = t6*Bnl[2][23];
  Kgeo[2][0]   = Kgeo[0][2];
  Kgeo[2][1]   = Kgeo[1][2];
  Kgeo[2][2]   = t9*Bnl[2][2];
  Kgeo[2][3]   = t10*Bnl[0][3];
  Kgeo[2][4]   = Bnl[1][4]*t11;
  Kgeo[2][5]   = t12*Bnl[2][5];
  Kgeo[2][6]   = t10*Bnl[0][6];
  Kgeo[2][7]   = Bnl[1][7]*t11;
  Kgeo[2][8]   = t12*Bnl[2][8];
  Kgeo[2][9]   = t10*Bnl[0][9];
  Kgeo[2][10]  = Bnl[1][10]*t11;
  Kgeo[2][11]  = t12*Bnl[2][11];
  Kgeo[2][12]  = t10*Bnl[0][12];
  Kgeo[2][13]  = Bnl[1][13]*t11;
  Kgeo[2][14]  = t12*Bnl[2][14];
  Kgeo[2][15]  = t10*Bnl[0][15];
  Kgeo[2][16]  = Bnl[1][16]*t11;
  Kgeo[2][17]  = t12*Bnl[2][17];
  Kgeo[2][18]  = t10*Bnl[0][18];
  Kgeo[2][19]  = t11*Bnl[1][19];
  Kgeo[2][20]  = t12*Bnl[2][20];
  Kgeo[2][21]  = t10*Bnl[0][21];
  Kgeo[2][22]  = t11*Bnl[1][22];
  Kgeo[2][23]  = t12*Bnl[2][23];
  Kgeo[3][0]   = Kgeo[0][3];
  Kgeo[3][1]   = Kgeo[1][3];
  Kgeo[3][2]   = Kgeo[2][3];
  Kgeo[3][3]   = t13*Bnl[0][3];
  Kgeo[3][4]   = t14*Bnl[1][4];
  Kgeo[3][5]   = Bnl[2][5]*t15;
  Kgeo[3][6]   = t16*Bnl[0][6];
  Kgeo[3][7]   = t14*Bnl[1][7];
  Kgeo[3][8]   = Bnl[2][8]*t15;
  Kgeo[3][9]   = t16*Bnl[0][9];
  Kgeo[3][10]  = t14*Bnl[1][10];
  Kgeo[3][11]  = Bnl[2][11]*t15;
  Kgeo[3][12]  = t16*Bnl[0][12];
  Kgeo[3][13]  = t14*Bnl[1][13];
  Kgeo[3][14]  = Bnl[2][14]*t15;
  Kgeo[3][15]  = t16*Bnl[0][15];
  Kgeo[3][16]  = t14*Bnl[1][16];
  Kgeo[3][17]  = Bnl[2][17]*t15;
  Kgeo[3][18]  = t16*Bnl[0][18];
  Kgeo[3][19]  = t14*Bnl[1][19];
  Kgeo[3][20]  = Bnl[2][20]*t15;
  Kgeo[3][21]  = t16*Bnl[0][21];
  Kgeo[3][22]  = t14*Bnl[1][22];
  Kgeo[3][23]  = Bnl[2][23]*t15;
  Kgeo[4][0]   = Kgeo[0][4];
  Kgeo[4][1]   = Kgeo[1][4];
  Kgeo[4][2]   = Kgeo[2][4];
  Kgeo[4][3]   = Kgeo[3][4];
  Kgeo[4][4]   = t17*Bnl[1][4];
  Kgeo[4][5]   = t18*Bnl[2][5];
  Kgeo[4][6]   = t19*Bnl[0][6];
  Kgeo[4][7]   = t20*Bnl[1][7];
  Kgeo[4][8]   = t18*Bnl[2][8];
  Kgeo[4][9]   = t19*Bnl[0][9];
  Kgeo[4][10]  = t20*Bnl[1][10];
  Kgeo[4][11]  = t18*Bnl[2][11];
  Kgeo[4][12]  = t19*Bnl[0][12];
  Kgeo[4][13]  = t20*Bnl[1][13];
  Kgeo[4][14]  = t18*Bnl[2][14];
  Kgeo[4][15]  = t19*Bnl[0][15];
  Kgeo[4][16]  = t20*Bnl[1][16];
  Kgeo[4][17]  = t18*Bnl[2][17];
  Kgeo[4][18]  = t19*Bnl[0][18];
  Kgeo[4][19]  = t20*Bnl[1][19];
  Kgeo[4][20]  = t18*Bnl[2][20];
  Kgeo[4][21]  = t19*Bnl[0][21];
  Kgeo[4][22]  = t20*Bnl[1][22];
  Kgeo[4][23]  = t18*Bnl[2][23];
  Kgeo[5][0]   = Kgeo[0][5];
  Kgeo[5][1]   = Kgeo[1][5];
  Kgeo[5][2]   = Kgeo[2][5];
  Kgeo[5][3]   = Kgeo[3][5];
  Kgeo[5][4]   = Kgeo[4][5];
  Kgeo[5][5]   = t21*Bnl[2][5];
  Kgeo[5][6]   = t22*Bnl[0][6];
  Kgeo[5][7]   = t23*Bnl[1][7];
  Kgeo[5][8]   = t24*Bnl[2][8];
  Kgeo[5][9]   = t22*Bnl[0][9];
  Kgeo[5][10]  = t23*Bnl[1][10];
  Kgeo[5][11]  = t24*Bnl[2][11];
  Kgeo[5][12]  = t22*Bnl[0][12];
  Kgeo[5][13]  = t23*Bnl[1][13];
  Kgeo[5][14]  = t24*Bnl[2][14];
  Kgeo[5][15]  = t22*Bnl[0][15];
  Kgeo[5][16]  = t23*Bnl[1][16];
  Kgeo[5][17]  = t24*Bnl[2][17];
  Kgeo[5][18]  = t22*Bnl[0][18];
  Kgeo[5][19]  = t23*Bnl[1][19];
  Kgeo[5][20]  = t24*Bnl[2][20];
  Kgeo[5][21]  = t22*Bnl[0][21];
  Kgeo[5][22]  = t23*Bnl[1][22];
  Kgeo[5][23]  = t24*Bnl[2][23];
  Kgeo[6][0]   = Kgeo[0][6];
  Kgeo[6][1]   = Kgeo[1][6];
  Kgeo[6][2]   = Kgeo[2][6];
  Kgeo[6][3]   = Kgeo[3][6];
  Kgeo[6][4]   = Kgeo[4][6];
  Kgeo[6][5]   = Kgeo[5][6];
  Kgeo[6][6]   = t25*Bnl[0][6];
  Kgeo[6][7]   = t26*Bnl[1][7];
  Kgeo[6][8]   = t27*Bnl[2][8];
  Kgeo[6][9]   = t28*Bnl[0][9];
  Kgeo[6][10]  = t26*Bnl[1][10];
  Kgeo[6][11]  = t27*Bnl[2][11];
  Kgeo[6][12]  = t28*Bnl[0][12];
  Kgeo[6][13]  = t26*Bnl[1][13];
  Kgeo[6][14]  = t27*Bnl[2][14];
  Kgeo[6][15]  = t28*Bnl[0][15];
  Kgeo[6][16]  = t26*Bnl[1][16];
  Kgeo[6][17]  = t27*Bnl[2][17];
  Kgeo[6][18]  = t28*Bnl[0][18];
  Kgeo[6][19]  = t26*Bnl[1][19];
  Kgeo[6][20]  = t27*Bnl[2][20];
  Kgeo[6][21]  = t28*Bnl[0][21];
  Kgeo[6][22]  = t26*Bnl[1][22];
  Kgeo[6][23]  = t27*Bnl[2][23];
  Kgeo[7][0]   = Kgeo[0][7];
  Kgeo[7][1]   = Kgeo[1][7];
  Kgeo[7][2]   = Kgeo[2][7];
  Kgeo[7][3]   = Kgeo[3][7];
  Kgeo[7][4]   = Kgeo[4][7];
  Kgeo[7][5]   = Kgeo[5][7];
  Kgeo[7][6]   = Kgeo[6][7];
  Kgeo[7][7]   = t29*Bnl[1][7];
  Kgeo[7][8]   = t30*Bnl[2][8];
  Kgeo[7][9]   = t31*Bnl[0][9];
  Kgeo[7][10]  = t32*Bnl[1][10];
  Kgeo[7][11]  = t30*Bnl[2][11];
  Kgeo[7][12]  = t31*Bnl[0][12];
  Kgeo[7][13]  = t32*Bnl[1][13];
  Kgeo[7][14]  = t30*Bnl[2][14];
  Kgeo[7][15]  = t31*Bnl[0][15];
  Kgeo[7][16]  = t32*Bnl[1][16];
  Kgeo[7][17]  = t30*Bnl[2][17];
  Kgeo[7][18]  = t31*Bnl[0][18];
  Kgeo[7][19]  = t32*Bnl[1][19];
  Kgeo[7][20]  = t30*Bnl[2][20];
  Kgeo[7][21]  = t31*Bnl[0][21];
  Kgeo[7][22]  = t32*Bnl[1][22];
  Kgeo[7][23]  = t30*Bnl[2][23];
  Kgeo[8][0]   = Kgeo[0][8];
  Kgeo[8][1]   = Kgeo[1][8];
  Kgeo[8][2]   = Kgeo[2][8];
  Kgeo[8][3]   = Kgeo[3][8];
  Kgeo[8][4]   = Kgeo[4][8];
  Kgeo[8][5]   = Kgeo[5][8];
  Kgeo[8][6]   = Kgeo[6][8];
  Kgeo[8][7]   = Kgeo[7][8];
  Kgeo[8][8]   = t33*Bnl[2][8];
  Kgeo[8][9]   = t34*Bnl[0][9];
  Kgeo[8][10]  = t35*Bnl[1][10];
  Kgeo[8][11]  = t36*Bnl[2][11];
  Kgeo[8][12]  = t34*Bnl[0][12];
  Kgeo[8][13]  = t35*Bnl[1][13];
  Kgeo[8][14]  = t36*Bnl[2][14];
  Kgeo[8][15]  = t34*Bnl[0][15];
  Kgeo[8][16]  = t35*Bnl[1][16];
  Kgeo[8][17]  = t36*Bnl[2][17];
  Kgeo[8][18]  = t34*Bnl[0][18];
  Kgeo[8][19]  = t35*Bnl[1][19];
  Kgeo[8][20]  = t36*Bnl[2][20];
  Kgeo[8][21]  = t34*Bnl[0][21];
  Kgeo[8][22]  = t35*Bnl[1][22];
  Kgeo[8][23]  = t36*Bnl[2][23];
  Kgeo[9][0]   = Kgeo[0][9];
  Kgeo[9][1]   = Kgeo[1][9];
  Kgeo[9][2]   = Kgeo[2][9];
  Kgeo[9][3]   = Kgeo[3][9];
  Kgeo[9][4]   = Kgeo[4][9];
  Kgeo[9][5]   = Kgeo[5][9];
  Kgeo[9][6]   = Kgeo[6][9];
  Kgeo[9][7]   = Kgeo[7][9];
  Kgeo[9][8]   = Kgeo[8][9];
  Kgeo[9][9]   = t37*Bnl[0][9];
  Kgeo[9][10]  = t38*Bnl[1][10];
  Kgeo[9][11]  = t39*Bnl[2][11];
  Kgeo[9][12]  = t40*Bnl[0][12];
  Kgeo[9][13]  = t38*Bnl[1][13];
  Kgeo[9][14]  = t39*Bnl[2][14];
  Kgeo[9][15]  = t40*Bnl[0][15];
  Kgeo[9][16]  = t38*Bnl[1][16];
  Kgeo[9][17]  = t39*Bnl[2][17];
  Kgeo[9][18]  = t40*Bnl[0][18];
  Kgeo[9][19]  = t38*Bnl[1][19];
  Kgeo[9][20]  = t39*Bnl[2][20];
  Kgeo[9][21]  = t40*Bnl[0][21];
  Kgeo[9][22]  = t38*Bnl[1][22];
  Kgeo[9][23]  = t39*Bnl[2][23];
  Kgeo[10][0]  = Kgeo[0][10];
  Kgeo[10][1]  = Kgeo[1][10];
  Kgeo[10][2]  = Kgeo[2][10];
  Kgeo[10][3]  = Kgeo[3][10];
  Kgeo[10][4]  = Kgeo[4][10];
  Kgeo[10][5]  = Kgeo[5][10];
  Kgeo[10][6]  = Kgeo[6][10];
  Kgeo[10][7]  = Kgeo[7][10];
  Kgeo[10][8]  = Kgeo[8][10];
  Kgeo[10][9]  = Kgeo[9][10];
  Kgeo[10][10] = t41*Bnl[1][10];
  Kgeo[10][11] = t42*Bnl[2][11];
  Kgeo[10][12] = t43*Bnl[0][12];
  Kgeo[10][13] = t44*Bnl[1][13];
  Kgeo[10][14] = t42*Bnl[2][14];
  Kgeo[10][15] = t43*Bnl[0][15];
  Kgeo[10][16] = t44*Bnl[1][16];
  Kgeo[10][17] = t42*Bnl[2][17];
  Kgeo[10][18] = t43*Bnl[0][18];
  Kgeo[10][19] = t44*Bnl[1][19];
  Kgeo[10][20] = t42*Bnl[2][20];
  Kgeo[10][21] = t43*Bnl[0][21];
  Kgeo[10][22] = t44*Bnl[1][22];
  Kgeo[10][23] = t42*Bnl[2][23];
  Kgeo[11][0]  = Kgeo[0][11];
  Kgeo[11][1]  = Kgeo[1][11];
  Kgeo[11][2]  = Kgeo[2][11];
  Kgeo[11][3]  = Kgeo[3][11];
  Kgeo[11][4]  = Kgeo[4][11];
  Kgeo[11][5]  = Kgeo[5][11];
  Kgeo[11][6]  = Kgeo[6][11];
  Kgeo[11][7]  = Kgeo[7][11];
  Kgeo[11][8]  = Kgeo[8][11];
  Kgeo[11][9]  = Kgeo[9][11];
  Kgeo[11][10] = Kgeo[10][11];
  Kgeo[11][11] = t45*Bnl[2][11];
  Kgeo[11][12] = t46*Bnl[0][12];
  Kgeo[11][13] = t47*Bnl[1][13];
  Kgeo[11][14] = t48*Bnl[2][14];
  Kgeo[11][15] = t46*Bnl[0][15];
  Kgeo[11][16] = t47*Bnl[1][16];
  Kgeo[11][17] = t48*Bnl[2][17];
  Kgeo[11][18] = t46*Bnl[0][18];
  Kgeo[11][19] = t47*Bnl[1][19];
  Kgeo[11][20] = t48*Bnl[2][20];
  Kgeo[11][21] = t46*Bnl[0][21];
  Kgeo[11][22] = t47*Bnl[1][22];
  Kgeo[11][23] = t48*Bnl[2][23];
  Kgeo[12][0]  = Kgeo[0][12];
  Kgeo[12][1]  = Kgeo[1][12];
  Kgeo[12][2]  = Kgeo[2][12];
  Kgeo[12][3]  = Kgeo[3][12];
  Kgeo[12][4]  = Kgeo[4][12];
  Kgeo[12][5]  = Kgeo[5][12];
  Kgeo[12][6]  = Kgeo[6][12];
  Kgeo[12][7]  = Kgeo[7][12];
  Kgeo[12][8]  = Kgeo[8][12];
  Kgeo[12][9]  = Kgeo[9][12];
  Kgeo[12][10] = Kgeo[10][12];
  Kgeo[12][11] = Kgeo[11][12];
  Kgeo[12][12] = t49*Bnl[0][12];
  Kgeo[12][13] = t50*Bnl[1][13];
  Kgeo[12][14] = t51*Bnl[2][14];
  Kgeo[12][15] = t52*Bnl[0][15];
  Kgeo[12][16] = t50*Bnl[1][16];
  Kgeo[12][17] = t51*Bnl[2][17];
  Kgeo[12][18] = t52*Bnl[0][18];
  Kgeo[12][19] = t50*Bnl[1][19];
  Kgeo[12][20] = t51*Bnl[2][20];
  Kgeo[12][21] = t52*Bnl[0][21];
  Kgeo[12][22] = t50*Bnl[1][22];
  Kgeo[12][23] = t51*Bnl[2][23];
  Kgeo[13][0]  = Kgeo[0][13];
  Kgeo[13][1]  = Kgeo[1][13];
  Kgeo[13][2]  = Kgeo[2][13];
  Kgeo[13][3]  = Kgeo[3][13];
  Kgeo[13][4]  = Kgeo[4][13];
  Kgeo[13][5]  = Kgeo[5][13];
  Kgeo[13][6]  = Kgeo[6][13];
  Kgeo[13][7]  = Kgeo[7][13];
  Kgeo[13][8]  = Kgeo[8][13];
  Kgeo[13][9]  = Kgeo[9][13];
  Kgeo[13][10] = Kgeo[10][13];
  Kgeo[13][11] = Kgeo[11][13];
  Kgeo[13][12] = Kgeo[12][13];
  Kgeo[13][13] = t53*Bnl[1][13];
  Kgeo[13][14] = t54*Bnl[2][14];
  Kgeo[13][15] = t55*Bnl[0][15];
  Kgeo[13][16] = t56*Bnl[1][16];
  Kgeo[13][17] = t54*Bnl[2][17];
  Kgeo[13][18] = t55*Bnl[0][18];
  Kgeo[13][19] = t56*Bnl[1][19];
  Kgeo[13][20] = t54*Bnl[2][20];
  Kgeo[13][21] = t55*Bnl[0][21];
  Kgeo[13][22] = t56*Bnl[1][22];
  Kgeo[13][23] = t54*Bnl[2][23];
  Kgeo[14][0]  = Kgeo[0][14];
  Kgeo[14][1]  = Kgeo[1][14];
  Kgeo[14][2]  = Kgeo[2][14];
  Kgeo[14][3]  = Kgeo[3][14];
  Kgeo[14][4]  = Kgeo[4][14];
  Kgeo[14][5]  = Kgeo[5][14];
  Kgeo[14][6]  = Kgeo[6][14];
  Kgeo[14][7]  = Kgeo[7][14];
  Kgeo[14][8]  = Kgeo[8][14];
  Kgeo[14][9]  = Kgeo[9][14];
  Kgeo[14][10] = Kgeo[10][14];
  Kgeo[14][11] = Kgeo[11][14];
  Kgeo[14][12] = Kgeo[12][14];
  Kgeo[14][13] = Kgeo[13][14];
  Kgeo[14][14] = t57*Bnl[2][14];
  Kgeo[14][15] = t58*Bnl[0][15];
  Kgeo[14][16] = t59*Bnl[1][16];
  Kgeo[14][17] = t60*Bnl[2][17];
  Kgeo[14][18] = t58*Bnl[0][18];
  Kgeo[14][19] = t59*Bnl[1][19];
  Kgeo[14][20] = t60*Bnl[2][20];
  Kgeo[14][21] = t58*Bnl[0][21];
  Kgeo[14][22] = t59*Bnl[1][22];
  Kgeo[14][23] = t60*Bnl[2][23];
  Kgeo[15][0]  = Kgeo[0][15];
  Kgeo[15][1]  = Kgeo[1][15];
  Kgeo[15][2]  = Kgeo[2][15];
  Kgeo[15][3]  = Kgeo[3][15];
  Kgeo[15][4]  = Kgeo[4][15];
  Kgeo[15][5]  = Kgeo[5][15];
  Kgeo[15][6]  = Kgeo[6][15];
  Kgeo[15][7]  = Kgeo[7][15];
  Kgeo[15][8]  = Kgeo[8][15];
  Kgeo[15][9]  = Kgeo[9][15];
  Kgeo[15][10] = Kgeo[10][15];
  Kgeo[15][11] = Kgeo[11][15];
  Kgeo[15][12] = Kgeo[12][15];
  Kgeo[15][13] = Kgeo[13][15];
  Kgeo[15][14] = Kgeo[14][15];
  Kgeo[15][15] = t61*Bnl[0][15];
  Kgeo[15][16] = t62*Bnl[1][16];
  Kgeo[15][17] = t63*Bnl[2][17];
  Kgeo[15][18] = t64*Bnl[0][18];
  Kgeo[15][19] = t62*Bnl[1][19];
  Kgeo[15][20] = t63*Bnl[2][20];
  Kgeo[15][21] = t64*Bnl[0][21];
  Kgeo[15][22] = t62*Bnl[1][22];
  Kgeo[15][23] = t63*Bnl[2][23];
  Kgeo[16][0]  = Kgeo[0][16];
  Kgeo[16][1]  = Kgeo[1][16];
  Kgeo[16][2]  = Kgeo[2][16];
  Kgeo[16][3]  = Kgeo[3][16];
  Kgeo[16][4]  = Kgeo[4][16];
  Kgeo[16][5]  = Kgeo[5][16];
  Kgeo[16][6]  = Kgeo[6][16];
  Kgeo[16][7]  = Kgeo[7][16];
  Kgeo[16][8]  = Kgeo[8][16];
  Kgeo[16][9]  = Kgeo[9][16];
  Kgeo[16][10] = Kgeo[10][16];
  Kgeo[16][11] = Kgeo[11][16];
  Kgeo[16][12] = Kgeo[12][16];
  Kgeo[16][13] = Kgeo[13][16];
  Kgeo[16][14] = Kgeo[14][16];
  Kgeo[16][15] = Kgeo[15][16];
  Kgeo[16][16] = t65*Bnl[1][16];
  Kgeo[16][17] = t66*Bnl[2][17];
  Kgeo[16][18] = t67*Bnl[0][18];
  Kgeo[16][19] = t68*Bnl[1][19];
  Kgeo[16][20] = t66*Bnl[2][20];
  Kgeo[16][21] = t67*Bnl[0][21];
  Kgeo[16][22] = t68*Bnl[1][22];
  Kgeo[16][23] = t66*Bnl[2][23];
  Kgeo[17][0]  = Kgeo[0][17];
  Kgeo[17][1]  = Kgeo[1][17];
  Kgeo[17][2]  = Kgeo[2][17];
  Kgeo[17][3]  = Kgeo[3][17];
  Kgeo[17][4]  = Kgeo[4][17];
  Kgeo[17][5]  = Kgeo[5][17];
  Kgeo[17][6]  = Kgeo[6][17];
  Kgeo[17][7]  = Kgeo[7][17];
  Kgeo[17][8]  = Kgeo[8][17];
  Kgeo[17][9]  = Kgeo[9][17];
  Kgeo[17][10] = Kgeo[10][17];
  Kgeo[17][11] = Kgeo[11][17];
  Kgeo[17][12] = Kgeo[12][17];
  Kgeo[17][13] = Kgeo[13][17];
  Kgeo[17][14] = Kgeo[14][17];
  Kgeo[17][15] = Kgeo[15][17];
  Kgeo[17][16] = Kgeo[16][17];
  Kgeo[17][17] = t69*Bnl[2][17];
  Kgeo[17][18] = t70*Bnl[0][18];
  Kgeo[17][19] = t71*Bnl[1][19];
  Kgeo[17][20] = t72*Bnl[2][20];
  Kgeo[17][21] = t70*Bnl[0][21];
  Kgeo[17][22] = t71*Bnl[1][22];
  Kgeo[17][23] = t72*Bnl[2][23];
  Kgeo[18][0]  = Kgeo[0][18];
  Kgeo[18][1]  = Kgeo[1][18];
  Kgeo[18][2]  = Kgeo[2][18];
  Kgeo[18][3]  = Kgeo[3][18];
  Kgeo[18][4]  = Kgeo[4][18];
  Kgeo[18][5]  = Kgeo[5][18];
  Kgeo[18][6]  = Kgeo[6][18];
  Kgeo[18][7]  = Kgeo[7][18];
  Kgeo[18][8]  = Kgeo[8][18];
  Kgeo[18][9]  = Kgeo[9][18];
  Kgeo[18][10] = Kgeo[10][18];
  Kgeo[18][11] = Kgeo[11][18];
  Kgeo[18][12] = Kgeo[12][18];
  Kgeo[18][13] = Kgeo[13][18];
  Kgeo[18][14] = Kgeo[14][18];
  Kgeo[18][15] = Kgeo[15][18];
  Kgeo[18][16] = Kgeo[16][18];
  Kgeo[18][17] = Kgeo[17][18];
  Kgeo[18][18] = t73*Bnl[0][18];
  Kgeo[18][19] = t74*Bnl[1][19];
  Kgeo[18][20] = t75*Bnl[2][20];
  Kgeo[18][21] = t73*Bnl[0][21];
  Kgeo[18][22] = t74*Bnl[1][22];
  Kgeo[18][23] = t75*Bnl[2][23];
  Kgeo[19][0]  = Kgeo[0][19];
  Kgeo[19][1]  = Kgeo[1][19];
  Kgeo[19][2]  = Kgeo[2][19];
  Kgeo[19][3]  = Kgeo[3][19];
  Kgeo[19][4]  = Kgeo[4][19];
  Kgeo[19][5]  = Kgeo[5][19];
  Kgeo[19][6]  = Kgeo[6][19];
  Kgeo[19][7]  = Kgeo[7][19];
  Kgeo[19][8]  = Kgeo[8][19];
  Kgeo[19][9]  = Kgeo[9][19];
  Kgeo[19][10] = Kgeo[10][19];
  Kgeo[19][11] = Kgeo[11][19];
  Kgeo[19][12] = Kgeo[12][19];
  Kgeo[19][13] = Kgeo[13][19];
  Kgeo[19][14] = Kgeo[14][19];
  Kgeo[19][15] = Kgeo[15][19];
  Kgeo[19][16] = Kgeo[16][19];
  Kgeo[19][17] = Kgeo[17][19];
  Kgeo[19][18] = Kgeo[18][19];
  Kgeo[19][19] = t77*Bnl[1][19];
  Kgeo[19][20] = t78*Bnl[2][20];
  Kgeo[19][21] = t79*Bnl[0][21];
  Kgeo[19][22] = t77*Bnl[1][22];
  Kgeo[19][23] = t78*Bnl[2][23];
  Kgeo[20][0]  = Kgeo[0][20];
  Kgeo[20][1]  = Kgeo[1][20];
  Kgeo[20][2]  = Kgeo[2][20];
  Kgeo[20][3]  = Kgeo[3][20];
  Kgeo[20][4]  = Kgeo[4][20];
  Kgeo[20][5]  = Kgeo[5][20];
  Kgeo[20][6]  = Kgeo[6][20];
  Kgeo[20][7]  = Kgeo[7][20];
  Kgeo[20][8]  = Kgeo[8][20];
  Kgeo[20][9]  = Kgeo[9][20];
  Kgeo[20][10] = Kgeo[10][20];
  Kgeo[20][11] = Kgeo[11][20];
  Kgeo[20][12] = Kgeo[12][20];
  Kgeo[20][13] = Kgeo[13][20];
  Kgeo[20][14] = Kgeo[14][20];
  Kgeo[20][15] = Kgeo[15][20];
  Kgeo[20][16] = Kgeo[16][20];
  Kgeo[20][17] = Kgeo[17][20];
  Kgeo[20][18] = Kgeo[18][20];
  Kgeo[20][19] = Kgeo[19][20];
  Kgeo[20][20] = t81*Bnl[2][20];
  Kgeo[20][21] = t82*Bnl[0][21];
  Kgeo[20][22] = t83*Bnl[1][22];
  Kgeo[20][23] = t81*Bnl[2][23];
  Kgeo[21][0]  = Kgeo[0][21];
  Kgeo[21][1]  = Kgeo[1][21];
  Kgeo[21][2]  = Kgeo[2][21];
  Kgeo[21][3]  = Kgeo[3][21];
  Kgeo[21][4]  = Kgeo[4][21];
  Kgeo[21][5]  = Kgeo[5][21];
  Kgeo[21][6]  = Kgeo[6][21];
  Kgeo[21][7]  = Kgeo[7][21];
  Kgeo[21][8]  = Kgeo[8][21];
  Kgeo[21][9]  = Kgeo[9][21];
  Kgeo[21][10] = Kgeo[10][21];
  Kgeo[21][11] = Kgeo[11][21];
  Kgeo[21][12] = Kgeo[12][21];
  Kgeo[21][13] = Kgeo[13][21];
  Kgeo[21][14] = Kgeo[14][21];
  Kgeo[21][15] = Kgeo[15][21];
  Kgeo[21][16] = Kgeo[16][21];
  Kgeo[21][17] = Kgeo[17][21];
  Kgeo[21][18] = Kgeo[18][21];
  Kgeo[21][19] = Kgeo[19][21];
  Kgeo[21][20] = Kgeo[20][21];
  Kgeo[21][21] = t85*Bnl[0][21];
  Kgeo[21][22] = t86*Bnl[1][22];
  Kgeo[21][23] = t87*Bnl[2][23];
  Kgeo[22][0]  = Kgeo[0][22];
  Kgeo[22][1]  = Kgeo[1][22];
  Kgeo[22][2]  = Kgeo[2][22];
  Kgeo[22][3]  = Kgeo[3][22];
  Kgeo[22][4]  = Kgeo[4][22];
  Kgeo[22][5]  = Kgeo[5][22];
  Kgeo[22][6]  = Kgeo[6][22];
  Kgeo[22][7]  = Kgeo[7][22];
  Kgeo[22][8]  = Kgeo[8][22];
  Kgeo[22][9]  = Kgeo[9][22];
  Kgeo[22][10] = Kgeo[10][22];
  Kgeo[22][11] = Kgeo[11][22];
  Kgeo[22][12] = Kgeo[12][22];
  Kgeo[22][13] = Kgeo[13][22];
  Kgeo[22][14] = Kgeo[14][22];
  Kgeo[22][15] = Kgeo[15][22];
  Kgeo[22][16] = Kgeo[16][22];
  Kgeo[22][17] = Kgeo[17][22];
  Kgeo[22][18] = Kgeo[18][22];
  Kgeo[22][19] = Kgeo[19][22];
  Kgeo[22][20] = Kgeo[20][22];
  Kgeo[22][21] = Kgeo[21][22];
  Kgeo[22][22] = t88*Bnl[1][22];
  Kgeo[22][23] = t89*Bnl[2][23];
  Kgeo[23][0]  = Kgeo[0][23];
  Kgeo[23][1]  = Kgeo[1][23];
  Kgeo[23][2]  = Kgeo[2][23];
  Kgeo[23][3]  = Kgeo[3][23];
  Kgeo[23][4]  = Kgeo[4][23];
  Kgeo[23][5]  = Kgeo[5][23];
  Kgeo[23][6]  = Kgeo[6][23];
  Kgeo[23][7]  = Kgeo[7][23];
  Kgeo[23][8]  = Kgeo[8][23];
  Kgeo[23][9]  = Kgeo[9][23];
  Kgeo[23][10] = Kgeo[10][23];
  Kgeo[23][11] = Kgeo[11][23];
  Kgeo[23][12] = Kgeo[12][23];
  Kgeo[23][13] = Kgeo[13][23];
  Kgeo[23][14] = Kgeo[14][23];
  Kgeo[23][15] = Kgeo[15][23];
  Kgeo[23][16] = Kgeo[16][23];
  Kgeo[23][17] = Kgeo[17][23];
  Kgeo[23][18] = Kgeo[18][23];
  Kgeo[23][19] = Kgeo[19][23];
  Kgeo[23][20] = Kgeo[20][23];
  Kgeo[23][21] = Kgeo[21][23];
  Kgeo[23][22] = Kgeo[22][23];
  Kgeo[23][23] = t90*Bnl[2][23];
}

// Compute the shear modulus from the current damage level:
double TongeRamesh::calculateShearPrefactor(const double currentDamage,
                                            const PlasticityState *state
                                            )
{
  double shear_0 = state->initialShearModulus;
  double bulk_0 =  state->initialBulkModulus;
  double nu_0 = (3*bulk_0-2*shear_0)/(6*bulk_0+2*shear_0);
  double E_0 = 9*bulk_0*shear_0/(3*bulk_0+shear_0);

  double Z_n = 16 * (1-nu_0*nu_0)/(3*E_0);
  double Z_r = Z_n/(1-0.5*nu_0);
  double Z_c = -Z_n/8.0;

  double inv_prefactor = 1 + shear_0 * 2.0/15.0 * (3*Z_r + 2*Z_n - 4*Z_c) * currentDamage;

  return 1/inv_prefactor;
}

double TongeRamesh::calculateBulkPrefactor(const double currentDamage,
                                           const PlasticityState *state,
                                           const double J
                                           )
{
  double shear_0 = state->initialShearModulus;
  double bulk_0 =  state->initialBulkModulus;
  double nu_0 = (3*bulk_0-2*shear_0)/(6*bulk_0+2*shear_0);
  double E_0 = 9*bulk_0*shear_0/(3*bulk_0+shear_0);

  double Z_n = 16 * (1-nu_0*nu_0)/(3*E_0);
  double Z_c = -Z_n/8.0;

  double inv_prefactor = 1 + bulk_0 * (Z_n + 4*Z_c) * currentDamage;

  return 1/inv_prefactor;
}

double TongeRamesh::computePressure(const MPMMaterial *mat, const Matrix3 &F, const Matrix3 &dF,
                                    const PlasticityState *state, const double delT,
                                    const double currentDamage)
{
  double J = F.Determinant();

  double damageFactor = calculateBulkPrefactor(currentDamage, state, J);

  double P_hat = J*d_eos-> computePressure(mat, state, F, dF, delT);

  return P_hat*damageFactor;
}

// Compute the damage growth, returns damage at the end of the
// increment.
double TongeRamesh::calculateDamageGrowth(Matrix3 &stress,
                                          vector<double> &N,
                                          vector<double> &s,
                                          vector<double> &old_L,
                                          const double currentDamage,
                                          vector<double> &new_L,
                                          vector<double> &new_Ldot,
                                          const double dt,
                                          const int Localized,
                                          const PlasticityState *state
                                          )
{

  double new_damage = 0.0;

  double bulk = state->bulkModulus;
  double shear = state->shearModulus;
  double phi = d_brittle_damage.phi;
  double KIc = d_brittle_damage.KIc;
  double cgamma = d_brittle_damage.cgamma;
  double mu = d_brittle_damage.mu;

  // Compute the Rayleigh wave speed and maximum crack velocity vm:
  bulk = state->initialBulkModulus;
  shear = state->initialShearModulus;
  double nu  = (3*bulk-2*shear) / (6*bulk+2*shear);
  double e   = (9*bulk*shear) / (3*bulk+shear);
  double rho = state->initialDensity;

  double Cr = (0.862+1.14*nu) * sqrt(e/(2*(1+nu)*rho)) / (1+nu); 
  double vm = Cr/d_brittle_damage.alpha; // Maximum crack velocity
  if (vm<0.0){
    ostringstream desc;
    desc << "A negative value was computed for the maximum crack velocity \n"
         << "Value for bulk modulus: " << bulk << "\n"
         << "Shear Modulus: " << shear << "\n"
         << " rho: " << rho << "\n"
         << "Youngs Modulus: " << e << "\n"
         << " nu: " << nu << "\n"
         << " vm: " << vm << endl;
    throw InvalidValue(desc.str(), __FILE__, __LINE__);
  }
    
  double matrixStress[2];

  // Use the two extreme principal stresses:
  double sig1, sig2, sig3;
  sig1 = 0;
  sig2 = 0;
  sig3 = 0;
  int numEigenValues;
  numEigenValues = stress.getEigenValues(sig1, sig2, sig3);

  // Assign the maximum principal stress to matrixStress[1]
  // and the minimum principal stress to matrixStress[0]
  if(numEigenValues == 3){
    matrixStress[1]=sig1;
    matrixStress[0]=sig3;
  } else if(numEigenValues == 2){
    matrixStress[1]=sig1;
    matrixStress[0]=sig2;
  } else {
    if(abs(stress.Trace()/3.0 - sig1) < 1e-3*abs(sig1)){
      // The stress state is hydrostatic:
      matrixStress[1]=sig1;
      matrixStress[0]=sig1;
    } else {
      // The other two eigen values are 0
      if(sig1>0){
        matrixStress[1]=sig1;
        matrixStress[0]=0;
      } else {
        matrixStress[1]=0;
        matrixStress[0]=sig1;
      }
    }
  }

  // cout << "Matrix Stress: " << matrixStress[0] << "\t" << matrixStress[1] << endl;
    
  double incStress[3] = {0};

  // if(d_brittle_damage.useBhaskerMatrix){
  //   computeIncStress_Bhasker(matrixStress, incStress, currentDamage, state);
  // } else if(d_brittle_damage.useJunweiMatrix){
  //   computeIncStress_Junwei(matrixStress, incStress, currentDamage, state);
  // }else {
  //   throw std::runtime_error("invalid specification for damaged matrix compliance, valid options are 'Bhasker', or 'Junwei'");
  // }

  // double wingDamage(0);       // Damage associated with wing cracks
  // double parentDamage(0);     // Damage associated with parent cracks
  // // I could add a test to see if all of the bins have reached their maximum damage level
  // for( int i=0; i<d_flawDistData.numCrackFamilies; i++){
  //   wingDamage   += N[i]*(old_L[i] * old_L[i] * old_L[i]);
  //   parentDamage += N[i]*(s[i] * s[i] * s[i]);
  // }
    

  // computeIncStress(matrixStress, incStress, wingDamage, parentDamage, state);
  if(d_brittle_damage.doFlawInteraction){
    computeIncStress(matrixStress, incStress, currentDamage, 0, state); // Place all of the damage in the wing cracks
  } else {
    incStress[0]=matrixStress[0];
    incStress[1]=matrixStress[1];
    incStress[2]=0.0;
  }

    
  // cout << "Inclusion Stress: " << incStress[0] << "\t" << incStress[1] << endl;

  // Now using the stress on the inside of the elipse calculate the crack growth:
  double s11e = incStress[0];
  double s22e = incStress[1];
  double s12e = incStress[2];

  vector<double>::iterator LIt = old_L.begin();
  vector<double>::iterator L_dotIt = new_Ldot.begin();
  vector<double>::iterator L_newIt = new_L.begin();
  vector<double>::iterator sIt = s.begin();
  vector<double>::iterator NIt = N.begin();

  for(;
      LIt < old_L.end();
      ++LIt, ++sIt, ++L_dotIt, ++NIt, ++L_newIt){
    if(*NIt >0 ){
      if( *LIt < (1.0/pow(*NIt, 1.0/3.0) - *sIt) ){
        // Calculate the wedging force:
        double F = -2.0 * (*sIt) * ( -mu*( s11e*cos(phi)*cos(phi) +
                                           s22e*sin(phi)*sin(phi) +
                                           s12e*sin(2*phi)
                                           )
                                     - (-0.5*(s11e-s22e)*sin(2*phi)+s12e*cos(2*phi))
                                     );
        if(F <= 0.0){
          F = 0.0;
        }
        // This assumes that the wing cracking mechanism is active:
        double K1 = F/(sqrt(PI*(*LIt + 0.27* (*sIt)))) + s22e*sqrt(PI*(*LIt+sin(phi)*(*sIt)));
        // double K1 = F/(sqrt(PI*(*LIt + 0.27* (*sIt)))) + s22e*sqrt(PI*(*LIt));
        // Calculate the crack growth:
      
        if(K1 >= KIc){
          *L_dotIt = vm * pow((K1-KIc)/(K1-KIc*0.5), cgamma);
        } else {
          *L_dotIt = 0;
        }

        // calculate the new crack length:
        *L_newIt = min(*LIt + dt*(*L_dotIt), (1.0/pow(*NIt, 1.0/3.0) - *sIt));
      } else {
        // set *LIt_new to the maximum crack length for that family
        *L_newIt = (1.0/pow(*NIt, 1.0/3.0) - *sIt);
      }
        
      // Sum the damage value:
      double damageIncrement = *NIt;
      if(d_brittle_damage.incInitialDamage){
        damageIncrement *=  *L_newIt + *sIt;
        damageIncrement *=  *L_newIt + *sIt;
        damageIncrement *=  *L_newIt + *sIt;
      } else {
        damageIncrement *=  *L_newIt;
        damageIncrement *=  *L_newIt;
        damageIncrement *=  *L_newIt;
      }
      new_damage += damageIncrement;
    } // end if(*NIt>0)
  }   // end flaw family loop
  new_damage = new_damage > d_brittle_damage.maxDamage ? d_brittle_damage.maxDamage : new_damage;

  return new_damage;

#undef PI
}

// Compute the value and derivitive of the gp yeild function in the
// Rendulic plane
inline double TongeRamesh::calc_yeildFunc_g_gs_gp(const double sigma_s, const double sigma_p,
                                                  double &gs, double &gp){
  double A = d_GPData.A;
  double B = d_GPData.B;
  double g;

  switch (d_GPData.yeildSurfaceType){
  case 1:
    {
      // if(sigma_s - sigma_p/A > 0){
        double Y = B*sqrt(1+A*A);
        g = sigma_s - Y + A*sigma_p;
        gs = 1;
        gp = A;
      // } else {
      //   g = sigma_s*sigma_s + sigma_p*sigma_p - B*B;
      //   gs = 2*sigma_s;
      //   gp = 2*sigma_p;
      // }
      break;
    }
  case 2:
    {
      gs = 2.0*sigma_s;
      gp = A;
      g = sigma_s*sigma_s+A*(sigma_p-B);
      break;
    }
  default:
    ostringstream desc;
    desc << "An unknown value for d_GPData.nonAssocMethod was "
         << "given, d_GPData.yeildSurfaceType=" << d_GPData.yeildSurfaceType
         << endl;
    throw InvalidValue(desc.str(), __FILE__, __LINE__);
    break;
  }

  return g;
}

// Compute the components of the return direction in Rendulic plane
inline void TongeRamesh::calc_returnDir(const double sigma_p, const double sigma_s,
                                        const double JGP,
                                        double &M_p, double &M_s
                                        ){
  // double B = d_GPData.B;
  // if(sigma_s == 0){
  //   // Sepcial case (Hydrostatic tension)
  //   M_s=0;
  //   M_p=1.0;
  //   return;
  // }

  // M_s = 1.0;
  // double tanDelta=d_GPData.tanDelta0*(d_GPData.JGPrecMin-JGP)/
  //   (d_GPData.JGPrecMin-1.0);
  // double A = d_GPData.A * (d_GPData.JGPrecMin-JGP)/
  //   (d_GPData.JGPrecMin-1.0);
  // switch(d_GPData.nonAssocMethod){
  // case 0:
  //   M_p = tanDelta>0 ? tanDelta : 0.0;
  //   break;
  // case 1:
  //   M_p = A>0 ? 0.5*A/sigma_s : 0.0;
  //   break;
  // default:
  //   ostringstream desc;
  //   desc << "An unknown value for d_GPData.nonAssocMethod was "
  //        << "given, d_GPData.nonAssocMethod=" << d_GPData.nonAssocMethod
  //        << endl;
  //   throw InvalidValue(desc.str(), __FILE__, __LINE__);
  //   break;
  // }
  
  // if(sigma_p > B){
  //   M_p = max(M_p, (sigma_p-B)/sigma_s);
  // }
}

// void TongeRamesh::computeIncStress_Bhasker
// (const double matrixStress[2], double incStress[3],
//  const double currentDamage, const PlasticityState *state)
// {
//   double km = state->initialBulkModulus;    // Undamaged Bulk Modulus
//   double gm = state->initialShearModulus;   // Undamaged Shear Modulus
//   double e  = (9*km*gm)/(3*km + gm);        // Undamaged Young's Modulus
//   double nu = (3*km - 2*gm)/(6*km + 2*gm);  // Undamaged Poisson Ratio
//   double pfE = 16.0*(1-nu*nu)*(10-3*nu)/(45*(2-nu));
//   double pfG = (32*(1-nu)*(5-nu))/(45*(2-nu));
//   double e1 = e * (1.0 - pfE *currentDamage);
//   double G12 = gm * (1.0 - pfG * currentDamage);
//   double E1 = e1;
//   double E2 = e1;
//   double nu12 = (3*km - 2*gm)/(6*km + 2*gm);
    
//   // Consistant with Bhasker's work:
//   double matCompliance[4];
//   double incCompliance[4];

//   matCompliance[0] = 1.0 / E1;
//   matCompliance[1] = 1.0 / E2;
//   matCompliance[2] = -(nu12) / E1;
//   matCompliance[3] = 1.0 / (2.0 * G12);

//   incCompliance[0] = 1.0/e;
//   incCompliance[1] = 1.0/e;
//   incCompliance[2] = -nu/e;
//   incCompliance[3] = 1.0 / (2.0 * gm);

//   // Define the ellipse: (size does not matter)
//   double ea = 1.0;
//   double eb = 0.306*ea;
//   double c = 0.5*(ea+eb);
//   double d = 0.5*(ea-eb);

//   double s1111[2], s2222[2],s1212[2], s1122[2];
//   // Solve for the complex material properties g1 and g2:

//   s1111[0] = matCompliance[0];
//   s2222[0] = matCompliance[1];
//   s1122[0] = matCompliance[2];
//   s1212[0] = matCompliance[3];

//   s1111[1] = incCompliance[0];
//   s2222[1] = incCompliance[1];
//   s1122[1] = incCompliance[2];  
//   s1212[1] = incCompliance[3]; // This is where Junwei thinks Bahsker made a mistake
//   // The result is that the complex material constants
//   // may be complex sometimes

//   double ap1[2], ap2[2], g1[2], g2[2], b1[2], b2[2], d1[2],
//     d2[2], r1[2], r2[2];
//   complex<double> compI(0,1.0);


//   for(int i=0; i<2; ++i){
//     double aa = s2222[i];
//     double bb = -2.0*(s1122[i] + 2.0*s1212[i]);
//     double cc = s1111[i];

//     double descriminant = bb*bb - 4*aa*cc;
//     if(descriminant >= 0){
//       // There are 2 real roots so g1 and g2 are real:
//       ap1[i] = (-bb-sqrt(descriminant))/(2.0*aa);
//       ap2[i] = (-bb+sqrt(descriminant))/(2.0*aa);
//     } else {
//       throw runtime_error("TongeRamesh.cc descriminant <0, complex math branch is not implimented");
//     }
//     g1[i] = 0;
//     g2[i] = 0;
//     if(fabs((sqrt(ap1[i]) - 1.0) / (sqrt(ap1[i]) + 1.0)) <
//        1.0) {
//       g1[i] = (sqrt(ap1[i]) - 1.0) / (sqrt(ap1[i]) + 1.0);
//     }
//     if(fabs(((-sqrt(ap1[i])) - 1.0) / ((-sqrt(ap1[i])) + 1.0))
//        < 1.0) {
//       g1[i] = ((-sqrt(ap1[i])) - 1.0) / ((-sqrt(ap1[i])) + 1.0);
//     }
//     if(fabs((sqrt(ap2[i]) - 1.0) / (sqrt(ap2[i]) + 1.0)) <
//        1.0) {
//       g2[i] = (sqrt(ap2[i]) - 1.0) / (sqrt(ap2[i]) + 1.0);
//     }
//     if(fabs(((-sqrt(ap2[i])) - 1.0) / ((-sqrt(ap2[i])) + 1.0))
//        < 1.0) {
        
//       g1[i] = ((-sqrt(ap2[i])) - 1.0) / ((-sqrt(ap2[i])) + 1.0); // This should be g2[i]
//       throw runtime_error("TongeRamesh.cc incorrect calculation of g2[[i]");
//     }
//     ap1[i] = pow((1.0 + g1[i]) / (1.0 - g1[i]), 2.0);
//     ap2[i] = pow((1.0 + g2[i]) / (1.0 - g2[i]), 2.0);
//     /* ---delta 1&2, rho 1&2 */
//     b1[i] = s1122[i] - ap1[i] * s2222[i];
//     b2[i] = s1122[i] - ap2[i] * s2222[i];
//     d1[i] = (1.0 + g1[i]) * b2[i] - (1.0 - g1[i]) * b1[i];
//     d2[i] = (1.0 + g2[i]) * b1[i] - (1.0 - g2[i]) * b2[i];
//     r1[i] = (1.0 + g1[i]) * b2[i] + (1.0 - g1[i]) * b1[i];
//     r2[i] = (1.0 + g2[i]) * b1[i] + (1.0 - g2[i]) * b2[i];
//   }
//   // Setup and solve Equation A.18 for the boundary conditions:
    
//   // Bhasker solved the equations and uses the solution in his code:
//   double N1 = matrixStress[0];
//   double N2 = matrixStress[1];
//   double temp = ( ( (sqrt(ap2[0])+1.0)*(sqrt(ap2[0])+1.0) ) *
//                   ( (sqrt(ap1[0])+1.0)*(sqrt(ap1[0])+1.0) ) *
//                   ( (N1+N2)*(1.0+g2[0]*g2[0]) +
//                     (N1-N2)*2.0*g2[0])/
//                   (32.0*(ap1[0]-ap2[0]))
//                   );
//   double B = temp;
//   double C = 0.0;

//   temp = (-( (sqrt(ap2[0])+1.0)*(sqrt(ap2[0])+1.0) ) *
//           ( (sqrt(ap1[0])+1.0)*(sqrt(ap1[0])+1.0) ) *
//           ( (N1+N2)*(1.0+g1[0]*g1[0]) +
//             (N1-N2)*2.0*g1[0])/
//           (32.0*(ap1[0]-ap2[0]))
//           );
//   double C1 = 0.0;
//   double B1 = temp;

//   // Equation A.17
//   complex<double>H1(complex<double>(B,C)*(c+g1[0]*d));
//   complex<double>H2(complex<double>(B1,C1)*(c+g2[0]*d));

//   // Equation A.23 Bhasker's Paper:
//   double m1( 0.5*((1.0+g1[1])*ea + (1.0-g1[1])*eb) );
//   double n1( 0.5*((1.0+g1[1])*ea - (1.0-g1[1])*eb) );
    
//   double m2( 0.5*((1.0+g2[1])*ea + (1.0-g2[1])*eb) );
//   double n2( 0.5*((1.0+g2[1])*ea - (1.0-g2[1])*eb) );

//   // Setup and solve equation A.27 from Bhasker's Paper
//   FastMatrix AmMat(4,4);    // Solve 4 equations the complex portion is identically 0
//   // Real Part of equation 1:
//   AmMat(0,0) = g1[1]*m1 + n1;
//   AmMat(0,1) = g2[1]*m2 + n2;
//   AmMat(0,2) = -1.0;
//   AmMat(0,3) = -1.0;
    
//   // Equation 2:
//   // Real Part:
//   AmMat(1,0) = m1 + g1[1]*n1;
//   AmMat(1,1) = m2 + g2[1]*n2;
//   AmMat(1,2) =-g1[0];
//   AmMat(1,3) =-g2[0];

//   // Equation 3: (Scaled by undamaged shear modulus)
//   double gm_2 = state->initialShearModulus;
//   gm_2 = 1.0;   // Use this to match Bhasker's matlab solution
//   // Real Part:
//   AmMat(2,0) = gm_2*(d1[1]*m1+r1[1]*n1);
//   AmMat(2,1) = gm_2*(d2[1]*m2+r2[1]*n2);
//   AmMat(2,2) =-gm_2*(r1[0]);
//   AmMat(2,3) =-gm_2*(r2[0]);
    
//   // Equation 4: (Scaled by undamaged shear modulus)
//   // Real part:
//   AmMat(3,0) = gm_2*((r1[1])*m1+(d1[1])*(n1));
//   AmMat(3,1) = gm_2*(r2[1])*m2+(d2[1])*(n2);
//   AmMat(3,2) =-gm_2*((d1[0]));
//   AmMat(3,3) =-gm_2*((d2[0]));

//   double BmVec[4];
//   BmVec[0] = real(g1[0]*H1+g2[0]*H2);
//   BmVec[1] = real(H1+H2);
//   // Equation 3 and 4 are scaled by the initial shear modulus
//   BmVec[2] = gm_2*real(d1[0]*H1+d2[0]*H2);
//   BmVec[3] = gm_2*real(conj(r1[0]*H1) +(r2[0])*H2);

//   // Solve the equations:
//   AmMat.destructiveSolve(BmVec);

//   double A1 = BmVec[0];
//   double A2 = BmVec[1];
//   // Correct stress in the inclusion solve equation A.8 using values for A1 and A2
//   // and the complex potential inside the ellipse (Eq A.21):
//   // incStress[0] =-2.0 * ( (g1[1]-1.0)*(g1[1]-1.0)*A1 + (g2[1]-1.0)*(g2[1]-1.0)*A2 );
//   // incStress[1] = 2.0 * ( (g1[1]+1.0)*(g1[1]+1.0)*A1 + (g2[1]+1.0)*(g2[1]+1.0)*A2 );
//   // incStress[2] = 0.0;

//   // Consistant with Bhaskers code (not correct this is the stress in the
//   // inclusion so it should use the material properties from inside the
//   // inclusion. (g1[1] and g2[1] not g1[0] and g2[0])

//   incStress[0] =-2.0 * ( (g1[0]-1.0)*(g1[0]-1.0)*A1 + (g2[0]-1.0)*(g2[0]-1.0)*A2 );
//   incStress[1] = 2.0 * ( (g1[0]+1.0)*(g1[0]+1.0)*A1 + (g2[0]+1.0)*(g2[0]+1.0)*A2 );
//   incStress[2] = 0.0;
// }

// void TongeRamesh::computeIncStress_Junwei
// (const double matrixStress[2], double incStress[3],
//  const double currentDamage, const PlasticityState *state)
// {
//   // Parameters for 3D damage from B&O
//   double km = state->initialBulkModulus;    // Undamaged Bulk Modulus
//   double gm = state->initialShearModulus;   // Undamaged Shear Modulus
//   double damage1 = d_brittle_damage.damageCouple * currentDamage;
//   double damage2 = currentDamage;
//   double e  = (9*km*gm)/(3*km + gm);        // Undamaged Young's Modulus
//   double nu = (3*km - 2*gm)/(6*km + 2*gm);  // Undamaged Poisson Ratio
//   double pfE = (16.0/45.0)*(1-nu*nu)*(10-3*nu)/(2-nu);
//   // double pfG = (32*(1-nu)*(5-nu))/(45*(2-nu));
//   // double pfE = (M_PI*M_PI/30)*(1+nu)*(5-4*nu);
//   // double pfG = (M_PI*M_PI*(10-7*nu))/60.0;
//   double E1 = e * (1.0 - pfE *damage1);;
//   double E2 = e * (1.0 - pfE *damage2);
//   double nu12 = nu*E1/E2;
//   double G12 = E1/(2*(1+nu12));
  
//   double matCompliance[4];
//   double incCompliance[4];

//   matCompliance[0] = 1.0/E1;
//   matCompliance[1] = 1.0/E2;
//   matCompliance[2] = -nu12/E2;
//   matCompliance[3] = 1.0/(2.0 * G12);

//   incCompliance[0] = 1.0/e;
//   incCompliance[1] = 1.0/e;
//   incCompliance[2] = -nu/e;
//   incCompliance[3] = 1.0 / (2.0 * gm);

//   //  ---------- Compute the stress in the Ellipse using Junwei's update ----------

//   // Define the ellipse: These equations need to be corrected for three
//   // dimensional flaw densities
//   double eta2d = pow(d_flawDistData.flawDensity, 2.0/3.0);
//   double ea = 1.4416*sqrt(0.5/eta2d);
//   double eb = 1/(M_PI*ea*eta2d);

//   double c = 0.5*(ea+eb)*sqrt(M_PI*eta2d);
//   double d = 0.5*(ea-eb)*sqrt(M_PI*eta2d);

//   // Solve for the complex material properties g1 and g2:
//   double aa = matCompliance[1];
//   double bb = -2.0*(matCompliance[2] + matCompliance[3]);
//   double cc = matCompliance[0];

//   double descriminant = bb*bb - 4*aa*cc;

//   double nu_inc = (3*state->initialBulkModulus - 2*state->initialShearModulus)/
//     (6*state->initialBulkModulus + 2*state->initialShearModulus);
//   // double kappa = (3-4*nu_inc); // Plane Strain (The compliance matrix also needs to be
//   // recalculated for plane strain.
//   double kappa = (3-nu_inc)/(1+nu_inc); // Plane Stress
    
//   if(descriminant>0){
//     // There are 2 real roots so g1 and g2 are real:
//     double ap1 = (-bb+sqrt(descriminant))/(2.0*aa);
//     double ap2 = (-bb-sqrt(descriminant))/(2.0*aa);

//     double g1((sqrt(ap1)-1.0)/(sqrt(ap1)+1.0));
//     double g2((sqrt(ap2)-1.0)/(sqrt(ap2)+1.0));
//     if(abs(g1)>1.0 || abs(g2)>1.0){
//       throw runtime_error("Both abs(g1) and abs(g2) must be less than 1 (real branch)");
//     }
      
//     if(abs(g1)>1e-6 || abs(g2)>1e-6){
//       // Compute boundary displacements (delta and rho)
//       // Equation A.10
//       double b1(matCompliance[2]-ap1*matCompliance[1]);
//       double b2(matCompliance[2]-ap2*matCompliance[1]);
//       // Equation A.9
//       double d1( ((1.0+g1) * b2 - (1.0-g1)*b1) );
//       double d2( ((1.0+g2) * b1 - (1.0-g2)*b2) );
//       double r1( ((1.0+g1) * b2 + (1.0-g1)*b1) );
//       double r2( ((1.0+g2) * b1 + (1.0-g2)*b2) );

//       // See lab notebook for boundary condition equations:
//       FastMatrix bcMat(2,2);
//       // Solve for B and B', C=C'=0;
//       // Equation 1:
//       bcMat(0,0) = g1;
//       bcMat(0,1) = g2;
//       // Equation 2:
//       bcMat(1,0) = g1*g1+1;
//       bcMat(1,1) = g2*g2+1;

//       double bcVec[2];
//       bcVec[0] = (matrixStress[0]+matrixStress[1])/8.0;
//       bcVec[1] = -(matrixStress[0]-matrixStress[1])/4.0;

//       bcMat.destructiveSolve(bcVec); // The solution is placed in the vector
//       double B = bcVec[0];
//       double B1 = bcVec[1];

//       // Equation A.17 (C=C/=0
//       double H1(B *(c+g1*d));
//       double H2(B1*(c+g2*d));

//       // Setup and solve equation B.13 from Junwei's document:
//       FastMatrix AmMat(4,4);  // 4 real equaitons
    
//       // Real Part of equation 1:
//       AmMat(0,0) = (kappa-1.0)*c;
//       AmMat(0,1) = -d;
//       AmMat(0,2) =-state->initialShearModulus*r1;
//       AmMat(0,3) =-state->initialShearModulus*r2;

//       // Equation 2:
//       // Real Part:
//       AmMat(1,0) = (kappa-1.0)*d;
//       AmMat(1,1) = -c;
//       AmMat(1,2) =-state->initialShearModulus*d1;
//       AmMat(1,3) =-state->initialShearModulus*d2;

//       // Equation 3:
//       // Real Part:
//       AmMat(2,0) = 2*c;
//       AmMat(2,1) = d;
//       AmMat(2,2) =-1.0;
//       AmMat(2,3) =-1.0;

//       // Equation 4:
//       AmMat(3,0) = 2*d;
//       AmMat(3,1) = c;
//       AmMat(3,2) = -g1;
//       AmMat(3,3) = -g2;

//       double BmVec[4];
//       BmVec[0] = state->initialShearModulus*(d1*H1+d2*H2);
//       BmVec[1] = state->initialShearModulus*(r1*H1+r2*H2);
//       BmVec[2] = g1*H1+g2*H2;
//       BmVec[3] = H1+H2;

//       // Solve the equations:
//       AmMat.destructiveSolve(BmVec);
//       double A1_re = BmVec[0];
//       double A2_re = BmVec[1];

//       incStress[0] = 4*A1_re-2*A2_re;
//       incStress[1] = 4*A1_re+2*A2_re;
//       incStress[2] = 0;
//     } else {
//       // Damage is too low to cause any change in the stress in the ellipse accept
//       // the far field stress as the stress in the ellipse:
//       // cout << "g1(" << g1 << ") and g2(" << g2<< ") are small using matrix stress" << endl;
//       incStress[0] = matrixStress[0];
//       incStress[1] = matrixStress[1];
//       incStress[2] = 0.0;
//     }
//   } else {
//     // The roots are complex and should be complex conjegutes of
//     // eachother.
//     complex<double> ap1,ap2;
//     ap1 = complex<double>(-bb/(2.0*aa), sqrt(-descriminant)/(2.0*aa));
//     ap2 = complex<double>(-bb/(2.0*aa),-sqrt(-descriminant)/(2.0*aa));

//     complex<double> g1((sqrt(ap1)-1.0)/(sqrt(ap1)+1.0));
//     complex<double> g2((sqrt(ap2)-1.0)/(sqrt(ap2)+1.0));
//     if(abs(g1)>1.0 || abs(g2)>1.0){
//       throw runtime_error("Both abs(g1) and abs(g2) must be less than 1 (complex branch)");
//     }

//     if(abs(g1)>1e-6){
//       // Compute boundary displacements (delta and rho)
//       // Equation A.10
//       complex<double> b1(matCompliance[2]-ap1*matCompliance[1]);
//       complex<double> b2(matCompliance[2]-ap2*matCompliance[1]);
//       // Equation A.9
//       complex<double> d1( ((1.0+g1) * b2 - (1.0-g1)*b1) );
//       complex<double> d2( ((1.0+g2) * b1 - (1.0-g2)*b2) );
//       complex<double> r1( conj((1.0+g1)*b2 + (1.0-g1)*b1) );
//       complex<double> r2( conj((1.0+g2)*b1 + (1.0-g2)*b2) );
//       complex<double> compI(0,1.0);

//       // Setup and solve Equation A.18 for the boundary conditions:
//       FastMatrix bcMat(2,2);
//       // We know that B=B' and C=-C' for complex g1=conj(g2)
//       // Real part of first equation
//       bcMat(0,0) = 2*real(g1);
//       bcMat(0,1) =-2*imag(g1);
//       // Real part of second equation
//       bcMat(1,0) = real(g1)*real(g1) -imag(g1)*imag(g1) +1;
//       bcMat(1,1) = -2.0*real(g1)*imag(g1);

//       double bcVec[2];
//       bcVec[0] = (matrixStress[0]+matrixStress[1])/8.0;
//       bcVec[1] = -(matrixStress[0]-matrixStress[1])/8.0;

//       bcMat.destructiveSolve(bcVec); // The solution is placed in the vector
//       double B = bcVec[0];
//       double C = bcVec[1];
//       double B1 = B;
//       double C1 = -C;

//       // cout << "B: " << B << " C: " << C << endl;

//       // Equation A.17
//       complex<double>H1(complex<double>(B,C)*(c+g1*d));
//       complex<double>H2(complex<double>(B1,C1)*(c+g2*d));

//       // Setup and solve equation B.13 from Junwei's document:
//       FastMatrix AmMat(8,8);    // Solve 2 complex equations and 2 real equations
//       // The equations are broken into their imagainary and real parts
//       // Real Part of equation 1:
//       AmMat(0,0) = (kappa-1.0)*c;
//       AmMat(0,1) = 0;
//       AmMat(0,2) = -d;
//       AmMat(0,3) = 0;
//       AmMat(0,4) =-state->initialShearModulus*real(conj(r1));
//       AmMat(0,5) = state->initialShearModulus*imag(conj(r1));
//       AmMat(0,6) =-state->initialShearModulus*real(conj(r2));
//       AmMat(0,7) = state->initialShearModulus*imag(conj(r2));
//       // // Imaginary part of equation 1:
//       AmMat(1,0) = 0;
//       AmMat(1,1) = (kappa-1.0)*c;
//       AmMat(1,2) = 0;
//       AmMat(1,3) = -d;
//       AmMat(1,4) =-state->initialShearModulus*imag(conj(r1));
//       AmMat(1,5) =-state->initialShearModulus*real(conj(r1));
//       AmMat(1,6) =-state->initialShearModulus*imag(conj(r2));
//       AmMat(1,7) =-state->initialShearModulus*real(conj(r2));

//       // Equation 2:
//       // Real Part:
//       AmMat(2,0) = (kappa-1.0)*d;
//       AmMat(2,1) = 0;
//       AmMat(2,2) = -c;
//       AmMat(2,3) = 0;
//       AmMat(2,4) =-state->initialShearModulus*real(d1);
//       AmMat(2,5) = state->initialShearModulus*imag(d1);
//       AmMat(2,6) =-state->initialShearModulus*real(d2);
//       AmMat(2,7) = state->initialShearModulus*imag(d2);
//       // Imaginary Part:
//       AmMat(3,0) = 0;
//       AmMat(3,1) = (kappa-1.0)*d;
//       AmMat(3,2) = 0;
//       AmMat(3,3) = -c;
//       AmMat(3,4) =-state->initialShearModulus*imag(conj(d1));
//       AmMat(3,5) =-state->initialShearModulus*real(conj(d1));
//       AmMat(3,6) =-state->initialShearModulus*imag(conj(d2));
//       AmMat(3,7) =-state->initialShearModulus*real(conj(d2));

//       // Equation 3:
//       // Real Part:
//       AmMat(4,0) = 2*c;
//       AmMat(4,1) = 0;
//       AmMat(4,2) = d;
//       AmMat(4,3) = 0;
//       AmMat(4,4) =-1.0;
//       AmMat(4,5) = 0.0;
//       AmMat(4,6) =-1.0;
//       AmMat(4,7) = 0.0;
//       // Imaginary Part:
//       AmMat(5,0) = 0.0;
//       AmMat(5,1) = 2*c;
//       AmMat(5,2) = 0.0;
//       AmMat(5,3) = d;
//       AmMat(5,4) = 0.0;
//       AmMat(5,5) =-1.0;
//       AmMat(5,6) = 0.0;
//       AmMat(5,7) =-1.0;

//       // Equation 4:
//       AmMat(6,0) = 2*d;
//       AmMat(6,1) = 0;
//       AmMat(6,2) = c;
//       AmMat(6,3) = 0;
//       AmMat(6,4) =-real(g1);
//       AmMat(6,5) = imag(g1);
//       AmMat(6,6) =-real(g2);
//       AmMat(6,7) = imag(g2);
//       // Imaginary Part:
//       AmMat(7,0) = 0.0;
//       AmMat(7,1) = 2*d;
//       AmMat(7,2) = 0.0;
//       AmMat(7,3) = c;
//       AmMat(7,4) =-imag(g1);
//       AmMat(7,5) =-real(g1);
//       AmMat(7,6) =-imag(g2);
//       AmMat(7,7) =-real(g2);

//       double BmVec[8];
//       BmVec[0] = state->initialShearModulus*real(conj(d1*H1+d2*H2));
//       BmVec[1] = state->initialShearModulus*imag(conj(d1*H1+d2*H2));
//       BmVec[2] = state->initialShearModulus*real(r1*conj(H1)+r2*conj(H2));
//       BmVec[3] = state->initialShearModulus*imag(r1*conj(H1)+r2*conj(H2));
//       BmVec[4] = real(conj(g1*H1+g2*H2));
//       BmVec[5] = imag(conj(g1*H1+g2*H2));
//       BmVec[6] = real(conj(H1+H2));
//       BmVec[7] = imag(conj(H1+H2));

//       // Solve the equations:
//       AmMat.destructiveSolve(BmVec);
//       double A1_re = BmVec[0];
//       double A2_re = BmVec[2];
//       double A2_im = BmVec[3];

//       incStress[0] = 4*A1_re-2*A2_re;
//       incStress[1] = 4*A1_re+2*A2_re;
//       incStress[2] = -2*A2_im;

//     } else {
//       // Damage is too low to cause any change in the stress in the ellipse accept
//       // the far field stress as the stress in the ellipse:
//       // cout << "g1(" << g1 << ") and g2(" << g2<< ") are small using matrix stress" << endl;
//       incStress[0] = matrixStress[0];
//       incStress[1] = matrixStress[1];
//       incStress[2] = 0.0;
//     }
//   } // End of if(descriminant>0)
// }

void TongeRamesh::computeIncStress
(const double matrixStress[2], double incStress[3],
 const double wingDamage, const double parentDamage,
 const PlasticityState *state)
{
  double Dw = wingDamage;
  // double Dp = parentDamage;
  // Dw = wingDamage > 5e-4 ? 5e-4 : wingDamage;
  // Dp = 2*Dw > parentDamage ? 2*Dw : parentDamage;
  // Dw = min(5.0e-5, Dw);
  // Dp = max(Dp, 2*Dw);
  
  // Grechka and Kachanov 2006 softening for two sets of flaws:
  double shear_0 = state->initialShearModulus;
  double bulk_0 =  state->initialBulkModulus;
  double nu_0 = (3*bulk_0-2*shear_0)/(6*bulk_0+2*shear_0);
  double E_0 = 9*bulk_0*shear_0/(3*bulk_0+shear_0);

  double Z_n = 16 * (1-nu_0*nu_0)/(3*E_0);
  double Z_r = Z_n/(1-0.5*nu_0);
  double Z_c = -Z_n/8.0;

  double mat_s_1111 = 1/E_0;
  double mat_s_2222 = 1/E_0;
  double mat_s_3333 = 1/E_0;
  double mat_s_1212 = (1+nu_0)/(2*E_0);
  double mat_s_1122 = -nu_0/E_0;
  double mat_s_1133 = -nu_0/E_0;
  double mat_s_2233 = -nu_0/E_0;

  // // Add the compliance from the parent flaws:
  // double cos_sq = cos(d_brittle_damage.phi) * cos(d_brittle_damage.phi);
  // double sin_sq = sin(d_brittle_damage.phi) * sin(d_brittle_damage.phi);
  
  // double prefactor = 0.25*Z_r*Dp;
  // mat_s_1111 += prefactor*(4*cos_sq - 2*nu_0*cos_sq*cos_sq);
  // mat_s_2222 += prefactor*(4*sin_sq - 2*nu_0*sin_sq*sin_sq);
  // mat_s_1212 += prefactor*(1 - 2*nu_0*sin_sq*cos_sq);
  // mat_s_1122 += prefactor*(-2.0*nu_0*sin_sq*cos_sq);

  // Add the compliance from the wing cracks:
  double prefactor = 0.25*Z_r*Dw;
  mat_s_2222 += prefactor*(4.0 - 2.0*nu_0);
  mat_s_1212 += prefactor*(1.0);

  // Add the coupling term
  mat_s_1122 += Z_c * Dw;
  mat_s_2233 += Z_c * Dw;

  // Elastic properties of the inclusion
  double kappa = (3-nu_0)/(1+nu_0); // Plane Stress

  // For Plane strain
  if(d_brittle_damage.usePlaneStrain){
    // Invert the stiffness matrix (only the upper 9 elements need to be inverted
    // to calculate the full 3D damaged stiffness matrix
    Matrix3 s_upper(mat_s_1111, mat_s_1122, mat_s_1133,
                    mat_s_1122, mat_s_2222, mat_s_2233,
                    mat_s_1133, mat_s_2233, mat_s_3333);
    Matrix3 c_upper = s_upper.Inverse();

    // The the planar compliance tensor is calculated from the reduced
    // stiffness matrix
    double denom_red = c_upper(0,0)*c_upper(1,1)-c_upper(0,1)*c_upper(0,1);
    mat_s_1111 = c_upper(1,1)/denom_red;
    mat_s_2222 = c_upper(0,0)/denom_red;
    mat_s_1122 = -c_upper(0,1)/denom_red;

    kappa = 3-4*nu_0;
  }


  //  ---------- Compute the stress in the Ellipse using Junwei's update ----------

  // Define the ellipse: These equations need to be corrected for three
  // dimensional flaw densities
  double eta2d = pow(d_flawDistData.flawDensity, 2.0/3.0);
  double ea = 1.4416*sqrt(0.5/eta2d);
  double eb = 1/(M_PI*ea*eta2d);

  double c = 0.5*(ea+eb)*sqrt(M_PI*eta2d);
  double d = 0.5*(ea-eb)*sqrt(M_PI*eta2d);

  // Solve for the complex material properties g1 and g2: (equation 6.9.2 Green and Zerna)
  double aa = mat_s_2222;
  double bb = -2.0*(mat_s_1122 + 2.0*mat_s_1212);
  double cc = mat_s_1111;

  double descriminant = bb*bb - 4*aa*cc;

    
  if(descriminant>0){
    // There are 2 real roots so g1 and g2 are real:
    double ap1 = (-bb+sqrt(descriminant))/(2.0*aa);
    double ap2 = (-bb-sqrt(descriminant))/(2.0*aa);

    double g1((sqrt(ap1)-1.0)/(sqrt(ap1)+1.0));
    double g2((sqrt(ap2)-1.0)/(sqrt(ap2)+1.0));
    if(abs(g1)>1.0 || abs(g2)>1.0){
      throw runtime_error("Both abs(g1) and abs(g2) must be less than 1 (real branch)");
    }
      
    if(abs(g1)>1e-6 || abs(g2)>1e-6){
      // Compute boundary displacements (delta and rho)
      // Equation A.10 (Bhasker) Eqn. 6.9.5 Green and Zerna
      double b1(mat_s_1122-ap1*mat_s_2222);
      double b2(mat_s_1122-ap2*mat_s_2222);
      // Equation A.9
      double d1( ((1.0+g1) * b2 - (1.0-g1)*b1) );
      double d2( ((1.0+g2) * b1 - (1.0-g2)*b2) );
      double r1( ((1.0+g1) * b2 + (1.0-g1)*b1) );
      double r2( ((1.0+g2) * b1 + (1.0-g2)*b2) );

      // See lab notebook for boundary condition equations (Andy Tonge: January 14 2013):
      FastMatrix bcMat(2,2);
      // Solve for B and B', C=C'=0;
      // Equation 1:
      bcMat(0,0) = g1;
      bcMat(0,1) = g2;
      // Equation 2:
      bcMat(1,0) = g1*g1+1;
      bcMat(1,1) = g2*g2+1;

      double bcVec[2];
      bcVec[0] = (matrixStress[0]+matrixStress[1])/8.0;
      bcVec[1] = -(matrixStress[0]-matrixStress[1])/4.0;

      bcMat.destructiveSolve(bcVec); // The solution is placed in the vector
      double B = bcVec[0];
      double B1 = bcVec[1];

      // Equation A.17 (C=C/=0
      double H1(B *(c+g1*d));
      double H2(B1*(c+g2*d));

      // Setup and solve equation B.13 from Junwei's document:
      FastMatrix AmMat(4,4);  // 4 real equaitons
    
      // Real Part of equation 1:
      AmMat(0,0) = (kappa-1.0)*c;
      AmMat(0,1) = -d;
      AmMat(0,2) =-state->initialShearModulus*r1;
      AmMat(0,3) =-state->initialShearModulus*r2;

      // Equation 2:
      // Real Part:
      AmMat(1,0) = (kappa-1.0)*d;
      AmMat(1,1) = -c;
      AmMat(1,2) =-state->initialShearModulus*d1;
      AmMat(1,3) =-state->initialShearModulus*d2;

      // Equation 3:
      // Real Part:
      AmMat(2,0) = 2*c;
      AmMat(2,1) = d;
      AmMat(2,2) =-1.0;
      AmMat(2,3) =-1.0;

      // Equation 4:
      AmMat(3,0) = 2*d;
      AmMat(3,1) = c;
      AmMat(3,2) = -g1;
      AmMat(3,3) = -g2;

      double BmVec[4];
      BmVec[0] = state->initialShearModulus*(d1*H1+d2*H2);
      BmVec[1] = state->initialShearModulus*(r1*H1+r2*H2);
      BmVec[2] = g1*H1+g2*H2;
      BmVec[3] = H1+H2;

      // Solve the equations:
      AmMat.destructiveSolve(BmVec);
      double A1_re = BmVec[0];
      double A2_re = BmVec[1];

      incStress[0] = 4*A1_re-2*A2_re;
      incStress[1] = 4*A1_re+2*A2_re;
      incStress[2] = 0;
    } else {
      // Damage is too low to cause any change in the stress in the ellipse accept
      // the far field stress as the stress in the ellipse:
      // cout << "g1(" << g1 << ") and g2(" << g2<< ") are small using matrix stress" << endl;
      incStress[0] = matrixStress[0];
      incStress[1] = matrixStress[1];
      incStress[2] = 0.0;
    }
  } else {
    // The roots are complex and should be complex conjegutes of
    // eachother.
    complex<double> ap1,ap2;
    ap1 = complex<double>(-bb/(2.0*aa), sqrt(-descriminant)/(2.0*aa));
    ap2 = complex<double>(-bb/(2.0*aa),-sqrt(-descriminant)/(2.0*aa));

    complex<double> g1((sqrt(ap1)-1.0)/(sqrt(ap1)+1.0));
    complex<double> g2((sqrt(ap2)-1.0)/(sqrt(ap2)+1.0));
    if(abs(g1)>1.0 || abs(g2)>1.0){
      throw runtime_error("Both abs(g1) and abs(g2) must be less than 1 (complex branch)");
    }

    if(abs(g1)>1e-6){
      // Compute boundary displacements (delta and rho)
      // Equation A.10
      complex<double> b1(mat_s_1122 - mat_s_2222*ap1);
      complex<double> b2(mat_s_1122 - mat_s_2222*ap2);
      // Equation A.9
      complex<double> d1( ((1.0+g1) * b2 - (1.0-g1)*b1) );
      complex<double> d2( ((1.0+g2) * b1 - (1.0-g2)*b2) );
      complex<double> r1( conj((1.0+g1)*b2 + (1.0-g1)*b1) );
      complex<double> r2( conj((1.0+g2)*b1 + (1.0-g2)*b2) );
      complex<double> compI(0,1.0);

      // Setup and solve Equation A.18 for the boundary conditions:
      FastMatrix bcMat(2,2);
      // We know that B=B' and C=-C' for complex g1=conj(g2)
      // Real part of first equation
      bcMat(0,0) = 2*real(g1);
      bcMat(0,1) =-2*imag(g1);
      // Real part of second equation
      bcMat(1,0) = real(g1)*real(g1) -imag(g1)*imag(g1) +1;
      bcMat(1,1) = -2.0*real(g1)*imag(g1);

      double bcVec[2];
      bcVec[0] = (matrixStress[0]+matrixStress[1])/8.0;
      bcVec[1] = -(matrixStress[0]-matrixStress[1])/8.0;

      bcMat.destructiveSolve(bcVec); // The solution is placed in the vector
      double B = bcVec[0];
      double C = bcVec[1];
      double B1 = B;
      double C1 = -C;

      // cout << "B: " << B << " C: " << C << endl;

      // Equation A.17
      complex<double>H1(complex<double>(B,C)*(c+g1*d));
      complex<double>H2(complex<double>(B1,C1)*(c+g2*d));

      // Setup and solve equation B.13 from Junwei's document:
      FastMatrix AmMat(8,8);    // Solve 2 complex equations and 2 real equations
      // The equations are broken into their imagainary and real parts
      // Real Part of equation 1:
      AmMat(0,0) = (kappa-1.0)*c;
      AmMat(0,1) = 0;
      AmMat(0,2) = -d;
      AmMat(0,3) = 0;
      AmMat(0,4) =-state->initialShearModulus*real(conj(r1));
      AmMat(0,5) = state->initialShearModulus*imag(conj(r1));
      AmMat(0,6) =-state->initialShearModulus*real(conj(r2));
      AmMat(0,7) = state->initialShearModulus*imag(conj(r2));
      // // Imaginary part of equation 1:
      AmMat(1,0) = 0;
      AmMat(1,1) = (kappa-1.0)*c;
      AmMat(1,2) = 0;
      AmMat(1,3) = -d;
      AmMat(1,4) =-state->initialShearModulus*imag(conj(r1));
      AmMat(1,5) =-state->initialShearModulus*real(conj(r1));
      AmMat(1,6) =-state->initialShearModulus*imag(conj(r2));
      AmMat(1,7) =-state->initialShearModulus*real(conj(r2));

      // Equation 2:
      // Real Part:
      AmMat(2,0) = (kappa-1.0)*d;
      AmMat(2,1) = 0;
      AmMat(2,2) = -c;
      AmMat(2,3) = 0;
      AmMat(2,4) =-state->initialShearModulus*real(d1);
      AmMat(2,5) = state->initialShearModulus*imag(d1);
      AmMat(2,6) =-state->initialShearModulus*real(d2);
      AmMat(2,7) = state->initialShearModulus*imag(d2);
      // Imaginary Part:
      AmMat(3,0) = 0;
      AmMat(3,1) = (kappa-1.0)*d;
      AmMat(3,2) = 0;
      AmMat(3,3) = -c;
      AmMat(3,4) =-state->initialShearModulus*imag((d1));
      AmMat(3,5) =-state->initialShearModulus*real((d1));
      AmMat(3,6) =-state->initialShearModulus*imag((d2));
      AmMat(3,7) =-state->initialShearModulus*real((d2));

      // Equation 3:
      // Real Part:
      AmMat(4,0) = 2*c;
      AmMat(4,1) = 0;
      AmMat(4,2) = d;
      AmMat(4,3) = 0;
      AmMat(4,4) =-1.0;
      AmMat(4,5) = 0.0;
      AmMat(4,6) =-1.0;
      AmMat(4,7) = 0.0;
      // Imaginary Part:
      AmMat(5,0) = 0.0;
      AmMat(5,1) = 2*c;
      AmMat(5,2) = 0.0;
      AmMat(5,3) = d;
      AmMat(5,4) = 0.0;
      AmMat(5,5) =-1.0;
      AmMat(5,6) = 0.0;
      AmMat(5,7) =-1.0;

      // Equation 4:
      AmMat(6,0) = 2*d;
      AmMat(6,1) = 0;
      AmMat(6,2) = c;
      AmMat(6,3) = 0;
      AmMat(6,4) =-real(g1);
      AmMat(6,5) = imag(g1);
      AmMat(6,6) =-real(g2);
      AmMat(6,7) = imag(g2);
      // Imaginary Part:
      AmMat(7,0) = 0.0;
      AmMat(7,1) = 2*d;
      AmMat(7,2) = 0.0;
      AmMat(7,3) = c;
      AmMat(7,4) =-imag(g1);
      AmMat(7,5) =-real(g1);
      AmMat(7,6) =-imag(g2);
      AmMat(7,7) =-real(g2);

      double BmVec[8];
      BmVec[0] = state->initialShearModulus*real(conj(d1*H1+d2*H2));
      BmVec[1] = state->initialShearModulus*imag(conj(d1*H1+d2*H2));
      BmVec[2] = state->initialShearModulus*real(r1*conj(H1)+r2*conj(H2));
      BmVec[3] = state->initialShearModulus*imag(r1*conj(H1)+r2*conj(H2));
      BmVec[4] = real(conj(g1*H1+g2*H2));
      BmVec[5] = imag(conj(g1*H1+g2*H2));
      BmVec[6] = real(conj(H1+H2));
      BmVec[7] = imag(conj(H1+H2));

      // Solve the equations:
      AmMat.destructiveSolve(BmVec);
      double A1_re = BmVec[0];
      double A2_re = BmVec[2];
      double A2_im = BmVec[3];

      incStress[0] = 4*A1_re-2*A2_re;
      incStress[1] = 4*A1_re+2*A2_re;
      incStress[2] = -2*A2_im;

    } else {
      // Damage is too low to cause any change in the stress in the ellipse accept
      // the far field stress as the stress in the ellipse:
      // cout << "g1(" << g1 << ") and g2(" << g2<< ") are small using matrix stress" << endl;
      incStress[0] = matrixStress[0];
      incStress[1] = matrixStress[1];
      incStress[2] = 0.0;
    }
  } // End of if(descriminant>0)
}


namespace Uintah {
  /*
    static MPI_Datatype makeMPI_CMData()
    {
    ASSERTEQ(sizeof(TongeRamesh::double), sizeof(double)*0);
    MPI_Datatype mpitype;
    MPI_Type_vector(1, 1, 1, MPI_DOUBLE, &mpitype);
    MPI_Type_commit(&mpitype);
    return mpitype;
    }
  
    const TypeDescription* fun_getTypeDescription(TongeRamesh::double*)
    {
    static TypeDescription* td = 0;
    if(!td){
    td = scinew TypeDescription(TypeDescription::Other,
    "TongeRamesh::double", 
    true, &makeMPI_CMData);
    }
    return td;
    }
  */
} // End namespace Uintah
