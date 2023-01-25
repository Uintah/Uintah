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


  /* Arenisca3 INTRO

  This source code is for a simplified constitutive model, named ``Arenisca3'',
  which has some of the basic features needed for modeling geomaterials.
  To better explain the source code, the comments in this file frequently refer
  to the equations in the following three references:
  1. The Arenisca manual,
  2. R.M. Brannon, "Elements of Phenomenological Plasticity: Geometrical Insight,
     Computational Algorithms, and Topics in Shock Physics", Shock Wave Science
     and Technology Reference Library: Solids I, Springer 2: pp. 189-274, 2007.

  */

  /* Revision Notes
    Pre-December 2012 - Arenisca v1 - Alireza Sadeghirad
    December, 2012 - Arenisca v2 - controlled by James Colovos
    January, 2013 - Arenisca v3 - controlled by Michael Homel

    This is VERSION 3.0 140731
  */

//----------------suggested max line width (72char)-------------------->

//----------DEFINE SECTION----------
#define MHdebug       // Prints errors messages when particles are deleted or subcycling fails
#define MHdeleteBadF  // Prints errors messages when particles are deleted or subcycling fails
//#define MHfastfcns  // Use fast approximate exp(), log() and pow() in deep loops.
                      // This may cause large errors when evaluating exp(x) for large x.  Use with caution!
#define MHdisaggregationStiffness // reduce stiffness with disaggregation

// INCLUDE SECTION: tells the preprocessor to include the necessary files
#include <CCA/Components/MPM/Materials/ConstitutiveModel/Arenisca3.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <CCA/Components/MPM/Core/MPMLabel.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/Matrix3.h>
#include <Core/Math/MiscMath.h>
#include <Core/Malloc/Allocator.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Math/Weibull.h> // For variability
#include <iostream>
#include <cstdio>

#ifdef MHfastfcns
#include <CCA/Components/MPM/Materials/ConstitutiveModel/fastapproximatefunctions.h>
#endif

using std::cerr;
using namespace Uintah;
using namespace std;

// Requires the necessary input parameters CONSTRUCTORS
Arenisca3::Arenisca3(ProblemSpecP& ps, MPMFlags* Mflag)
  : ConstitutiveModel(Mflag)
{
  proc0cout << "Arenisca ver 3.0"<< endl;
  proc0cout << "University of Utah, Mechanical Engineering, Computational Solid Mechanics" << endl;

  // Private Class Variables:
  one_third      = 1.0/3.0;
  two_third      = 2.0/3.0;
  four_third     = 4.0/3.0;
  sqrt_two       = sqrt(2.0);
  one_sqrt_two   = 1.0/sqrt_two;
  sqrt_three     = sqrt(3.0);
  one_sqrt_three = 1.0/sqrt_three;
  one_ninth      = 1.0/9.0;
  one_sixth      = 1.0/6.0;
  pi  = 3.141592653589793238462;
  pi_fourth = 0.25*pi;
  pi_half = 0.5*pi;
  Identity.Identity();

  ps->require("PEAKI1",d_cm.PEAKI1);  // Shear Limit Surface Parameter
  ps->require("FSLOPE",d_cm.FSLOPE);  // Shear Limit Surface Parameter
  ps->require("STREN",d_cm.STREN);    // Shear Limit Surface Parameter
  ps->require("YSLOPE",d_cm.YSLOPE);  // Shear Limit Surface Parameter
  ps->require("BETA_nonassociativity",d_cm.BETA_nonassociativity);   // Nonassociativity Parameter
  ps->require("B0",d_cm.B0);          // Tangent Elastic Bulk Modulus Parameter
  ps->require("B1",d_cm.B1);          // Tangent Elastic Bulk Modulus Parameter
  ps->require("B2",d_cm.B2);          // Tangent Elastic Bulk Modulus Parameter
  ps->require("B3",d_cm.B3);          // Tangent Elastic Bulk Modulus Parameter
  ps->require("B4",d_cm.B4);          // Tangent Elastic Bulk Modulus Parameter
  ps->require("G0",d_cm.G0);          // Tangent Elastic Shear Modulus Parameter
  ps->require("G1",d_cm.G1);          // Tangent Elastic Shear Modulus Parameter
  ps->require("G2",d_cm.G2);          // Tangent Elastic Shear Modulus Parameter
  ps->require("G3",d_cm.G3);          // Tangent Elastic Shear Modulus Parameter
  ps->require("G4",d_cm.G4);          // Tangent Elastic Shear Modulus Parameter
  ps->require("p0_crush_curve",d_cm.p0_crush_curve);  // Crush Curve Parameter
  ps->require("p1_crush_curve",d_cm.p1_crush_curve);  // Crush Curve Parameter
  ps->require("p2_crush_curve",d_cm.p2_crush_curve);  // Crush Curve Parameter (not used)
  ps->require("p3_crush_curve",d_cm.p3_crush_curve);  // Crush Curve Parameter
  ps->require("CR",d_cm.CR);                          // Cap Shape Parameter CR = (peakI1-kappa)/(peakI1-X)
  ps->require("fluid_B0",d_cm.fluid_B0);              // Fluid bulk modulus (K_f)
  ps->require("fluid_pressure_initial",d_cm.fluid_pressure_initial);  // Zero strain Fluid Pressure (Pf0)
  ps->require("T1_rate_dependence",d_cm.T1_rate_dependence);    // Rate dependence parameter
  ps->require("T2_rate_dependence",d_cm.T2_rate_dependence);    // Rate dependence parameter
  ps->getWithDefault("subcycling_characteristic_number",d_cm.subcycling_characteristic_number, 256);    // allowable subcycles
  ps->getWithDefault("Use_Disaggregation_Algorithm",d_cm.Use_Disaggregation_Algorithm, false);
  ps->get("PEAKI1IDIST",wdist.WeibDist);        // For variability
  WeibullParser(wdist);                         // For variability
  proc0cout <<"WeibMed="<<wdist.WeibMed<<endl;  // For variability

  // These class variables are computed from input parameters and are used throughout the code
  // The are evaluates here to avoid repeated computation, or to simplify expressions.

  // This phi_i value is not modified by the disaggregation strain, because
  // it is the same for all particles.  Thus disaggregation strain is
  // not supported when there is a pore fluid.
  phi_i = 1.0 - exp(-d_cm.p3_crush_curve);      // Initial porosity (inferred from crush curve, used for fluid model/
  Km = d_cm.B0 + d_cm.B1;                       // Matrix bulk modulus
  Kf = d_cm.fluid_B0;                           // Fluid bulk modulus
  C1 = Kf*(1.0 - phi_i) + Km*(phi_i);           // Term to simplify the fluid model expressions
  ev0 = C1*d_cm.fluid_pressure_initial/(Kf*Km); // Zero fluid pressure vol. strain.  (will equal zero if pfi=0)

  initializeLocalMPMLabels();
}

// DESTRUCTOR
Arenisca3::~Arenisca3()
{
  VarLabel::destroy(peakI1IDistLabel);          // For variability
  VarLabel::destroy(peakI1IDistLabel_preReloc); // For variability
  VarLabel::destroy(pAreniscaFlagLabel);
  VarLabel::destroy(pAreniscaFlagLabel_preReloc);
  VarLabel::destroy(pScratchDouble1Label);
  VarLabel::destroy(pScratchDouble1Label_preReloc);
  VarLabel::destroy(pScratchDouble2Label);
  VarLabel::destroy(pScratchDouble2Label_preReloc);
  VarLabel::destroy(pPorePressureLabel);
  VarLabel::destroy(pPorePressureLabel_preReloc);
  VarLabel::destroy(pepLabel);               //Plastic Strain Tensor
  VarLabel::destroy(pepLabel_preReloc);
  VarLabel::destroy(pevpLabel);              //Plastic Volumetric Strain
  VarLabel::destroy(pevpLabel_preReloc);
  VarLabel::destroy(peveLabel);              //Elastic Volumetric Strain
  VarLabel::destroy(peveLabel_preReloc);
  VarLabel::destroy(pCapXLabel);
  VarLabel::destroy(pCapXLabel_preReloc);
  VarLabel::destroy(pStressQSLabel);
  VarLabel::destroy(pStressQSLabel_preReloc);
  VarLabel::destroy(pScratchMatrixLabel);
  VarLabel::destroy(pScratchMatrixLabel_preReloc);
  VarLabel::destroy(pZetaLabel);
  VarLabel::destroy(pZetaLabel_preReloc);
  VarLabel::destroy(pP3Label);          // Modified p3 for initial disaggregation strain.
  VarLabel::destroy(pP3Label_preReloc); // Modified p3 for initial disaggregation strain.
}

//adds problem specification values to checkpoint data for restart
void Arenisca3::outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag)
{
  ProblemSpecP cm_ps = ps;
  if (output_cm_tag) {
    cm_ps = ps->appendChild("constitutive_model");
    cm_ps->setAttribute("type","Arenisca3");
  }
  cm_ps->appendElement("FSLOPE",d_cm.FSLOPE);
  cm_ps->appendElement("PEAKI1",d_cm.PEAKI1);
  cm_ps->appendElement("STREN",d_cm.STREN);
  cm_ps->appendElement("YSLOPE",d_cm.YSLOPE);
  cm_ps->appendElement("BETA_nonassociativity",d_cm.BETA_nonassociativity);
  cm_ps->appendElement("B0",d_cm.B0);
  cm_ps->appendElement("B1",d_cm.B1);
  cm_ps->appendElement("B2",d_cm.B2);
  cm_ps->appendElement("B3",d_cm.B3);
  cm_ps->appendElement("B4",d_cm.B4);
  cm_ps->appendElement("G0",d_cm.G0);
  cm_ps->appendElement("G1",d_cm.G1);  // Low pressure Poisson ratio
  cm_ps->appendElement("G2",d_cm.G2);  // Pressure-dependent Poisson ratio term
  cm_ps->appendElement("G3",d_cm.G3);  // Not used
  cm_ps->appendElement("G4",d_cm.G4);  // Not used
  cm_ps->appendElement("p0_crush_curve",d_cm.p0_crush_curve);
  cm_ps->appendElement("p1_crush_curve",d_cm.p1_crush_curve);
  cm_ps->appendElement("p2_crush_curve",d_cm.p2_crush_curve);  // Not used
  cm_ps->appendElement("p3_crush_curve",d_cm.p3_crush_curve);
  cm_ps->appendElement("CR",d_cm.CR);
  cm_ps->appendElement("fluid_B0",d_cm.fluid_B0);
  cm_ps->appendElement("fluid_pressure_initial",d_cm.fluid_pressure_initial);
  cm_ps->appendElement("T1_rate_dependence",d_cm.T1_rate_dependence);
  cm_ps->appendElement("T2_rate_dependence",d_cm.T2_rate_dependence);
  cm_ps->appendElement("subcycling_characteristic_number",d_cm.subcycling_characteristic_number);
  cm_ps->appendElement("Use_Disaggregation_Algorithm",d_cm.Use_Disaggregation_Algorithm);

  //    Uintah Variability Variables
  cm_ps->appendElement("peakI1IPerturb", wdist.Perturb);
  cm_ps->appendElement("peakI1IMed", wdist.WeibMed);
  cm_ps->appendElement("peakI1IMod", wdist.WeibMod);
  cm_ps->appendElement("peakI1IRefVol", wdist.WeibRefVol);
  cm_ps->appendElement("peakI1ISeed", wdist.WeibSeed);
  cm_ps->appendElement("PEAKI1IDIST", wdist.WeibDist);
}

Arenisca3* Arenisca3::clone()
{
  return scinew Arenisca3(*this);
}

void Arenisca3::initializeCMData(const Patch* patch,
                                   const MPMMaterial* matl,
                                   DataWarehouse* new_dw)
{
  // Initialize the variables shared by all constitutive models
  // This method is defined in the ConstitutiveModel base class.
  //
  // Allocates memory for internal state variables at beginning of run.
  //
  // Get the particles in the current patch
  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(),patch);

  ParticleVariable<double>  pdTdt;
  ParticleVariable<Matrix3> pDefGrad,
                            pStress,
                            pStressQS;

  new_dw->allocateAndPut(pdTdt,       lb->pdTdtLabel,               pset);
  new_dw->allocateAndPut(pStress,     lb->pStressLabel,             pset);
  new_dw->allocateAndPut(pStressQS,   pStressQSLabel,               pset);
  new_dw->allocateAndPut(pDefGrad,    lb->pDeformationMeasureLabel, pset);

  // To fix : For a material that is initially stressed we need to
  // modify the stress tensors to comply with the initial stress state
  ParticleSubset::iterator iter = pset->begin();
  for(; iter != pset->end(); iter++){
    pdTdt[*iter] = 0.0;
    pDefGrad[*iter] = Identity;
    pStress[*iter]  = - d_cm.fluid_pressure_initial * Identity;
    pStressQS[*iter] = pStress[*iter];
  }

  // Allocate particle variables
  ParticleVariable<int>     pAreniscaFlag;
  ParticleVariable<double>  peakI1IDist;     // Holder for particles PEAKI1 value for variability
  ParticleVariable<double>  pScratchDouble1, // Developer tool
                            pScratchDouble2, // Developer tool
                            pPorePressure,   // Plottable fluid pressure
                            pevp,            // Plastic Volumetric Strain
                            peve,            // Elastic Volumetric Strain
                            pCapX,           // I1 of cap intercept
                            pZeta,           // Trace of isotropic Backstress
                            pP3;             // Modified p3 for initial disaggregation strain.
  ParticleVariable<Matrix3> pScratchMatrix,  // Developer tool
                            pep;             // Plastic Strain Tensor

  new_dw->allocateAndPut(pAreniscaFlag,   pAreniscaFlagLabel,   pset);
  new_dw->allocateAndPut(pScratchDouble1, pScratchDouble1Label, pset);
  new_dw->allocateAndPut(pScratchDouble2, pScratchDouble2Label, pset);
  new_dw->allocateAndPut(pPorePressure,   pPorePressureLabel,   pset);
  new_dw->allocateAndPut(peakI1IDist,     peakI1IDistLabel,     pset);
  new_dw->allocateAndPut(pep,             pepLabel,             pset);
  new_dw->allocateAndPut(pevp,            pevpLabel,            pset);
  new_dw->allocateAndPut(peve,            peveLabel,            pset);
  new_dw->allocateAndPut(pCapX,           pCapXLabel,           pset);
  new_dw->allocateAndPut(pZeta,           pZetaLabel,           pset);
  new_dw->allocateAndPut(pP3,             pP3Label,             pset);
  new_dw->allocateAndPut(pScratchMatrix,  pScratchMatrixLabel,  pset);

  constParticleVariable<double> pVolume, pMass;
  new_dw->get(pVolume, lb->pVolumeLabel, pset);
  new_dw->get(pMass,   lb->pMassLabel,   pset);

  for(ParticleSubset::iterator iter = pset->begin();
      iter != pset->end();iter++){
    pAreniscaFlag[*iter] = 0;
    pScratchDouble1[*iter] = 0;
    pScratchDouble2[*iter] = 0;
    pPorePressure[*iter] = d_cm.fluid_pressure_initial;
    peakI1IDist[*iter] = d_cm.PEAKI1;
    pevp[*iter] = 0.0;
    peve[*iter] = 0.0;
    pZeta[*iter] = -3.0 * d_cm.fluid_pressure_initial;
    if(d_cm.Use_Disaggregation_Algorithm){pP3[*iter] = log(pVolume[*iter]*(matl->getInitialDensity())/pMass[*iter]);}
    else{pP3[*iter] = d_cm.p3_crush_curve;}
    pCapX[*iter] = computeX(0.0,pP3[*iter]);
    pScratchMatrix[*iter].set(0.0);
    pep[*iter].set(0.0);
  }

  if ( wdist.Perturb){ // For variability
    // Make the seed differ for each patch, otherwise each patch gets the
    // same set of random #s.
    int patchID = patch->getID();
    int patch_div_32 = patchID/32;
    patchID = patchID%32;
    unsigned int unique_seed = ((wdist.WeibSeed+patch_div_32+1) << patchID);
    Weibull weibGen(wdist.WeibMed,wdist.WeibMod,wdist.WeibRefVol,
                            unique_seed,wdist.WeibMod);
    //proc0cout << "Weibull Variables for PEAKI1I: (initialize CMData)\n"
    //          << "Median:            " << wdist.WeibMed
    //          << "\nModulus:         " << wdist.WeibMod
    //          << "\nReference Vol:   " << wdist.WeibRefVol
    //          << "\nSeed:            " << wdist.WeibSeed
    //          << "\nPerturb?:        " << wdist.Perturb << std::endl;
    ParticleSubset::iterator iter = pset->begin();
    for(;iter != pset->end();iter++){
      //set value with variability and scale effects
      peakI1IDist[*iter] = weibGen.rand(pVolume[*iter]);

      //set value with ONLY scale effects
      if(wdist.WeibSeed==0)
        peakI1IDist[*iter]= pow(wdist.WeibRefVol/pVolume[*iter],1./wdist.WeibMod)
                            *wdist.WeibMed;
    }
  }

  computeStableTimeStep(patch, matl, new_dw);
}

// Compute stable timestep based on both the particle velocities
// and wave speed
void Arenisca3::computeStableTimeStep(const Patch* patch,
                                     //ParticleSubset* pset, //T2D: this should be const
                                     const MPMMaterial* matl,
                                     DataWarehouse* new_dw)
{
  // For non-linear elasticity either with or without fluid effects the elastic bulk
  // modulus can be a function of pressure and or strain.  In all cases however, the
  // maximum value the bulk modulus can obtain is the high pressure limit, (B0+B1)
  // and the minimum is the low pressure limit (B0).
  //
  // The high pressure limit is used to conservatively estimate the number of substeps
  // necessary to resolve some loading to a reasonable accuracy as this will produce an
  // upper limit for the magnitude of the trial stress.
  //
  // To compute the stable time step, we compute the wave speed c, and evaluate
  // dt=dx/c.  Thus the upper limit for the bulk modulus, which corresponds to a
  // conservatively high value for the wave speed, will produce a conservatively low
  // value of the time step, ensuring stability.
  int     dwi = matl->getDWIndex();

  double  c_dil = 0.0;
  double  bulk,shear;                   // High pressure limit elastic properties
  computeElasticProperties(bulk,shear);

  Vector  dx = patch->dCell(),
          WaveSpeed(1.e-12,1.e-12,1.e-12);

  // Get the particles in the current patch
  ParticleSubset* pset = new_dw->getParticleSubset(dwi, patch);

  // Get particles mass, volume, and velocity
  constParticleVariable<double> pmass,
                                pvolume;
  constParticleVariable<long64> pParticleID;
  constParticleVariable<Vector> pvelocity;

  new_dw->get(pmass,       lb->pMassLabel,       pset);
  new_dw->get(pvolume,     lb->pVolumeLabel,     pset);
  new_dw->get(pParticleID, lb->pParticleIDLabel, pset);
  new_dw->get(pvelocity,   lb->pVelocityLabel,   pset);


  // loop over the particles in the patch
  for(ParticleSubset::iterator iter = pset->begin();
      iter != pset->end(); iter++){
    particleIndex idx = *iter;

    // Compute wave speed + particle velocity at each particle,
    // store the maximum
    c_dil = sqrt((bulk + four_third*shear)*(pvolume[idx]/pmass[idx]));

    WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
                     Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
                     Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));
  }

  // Compute the stable timestep based on maximum value of
  // "wave speed + particle velocity"
  WaveSpeed = dx/WaveSpeed;
  double delT_new = WaveSpeed.minComponent();
  new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
}

// ------------------------------------- BEGIN COMPUTE STRESS TENSOR FUNCTION
/**
Arenisca3::computeStressTensor is the core of the Arenisca3 model which computes
the updated stress at the end of the current timestep along with all other
required data such plastic strain, elastic strain, cap position, etc.

*/
void Arenisca3::computeStressTensor(const PatchSubset* patches,
                                           const MPMMaterial* matl,
                                           DataWarehouse* old_dw,
                                           DataWarehouse* new_dw)
{
  // Global loop over each patch
  for(int p=0;p<patches->size();p++){

    // Declare and initial value assignment for some variables
    const Patch* patch = patches->get(p);
    Matrix3 D;

    double c_dil=0.0,
           se=0.0;  // MH! Fix this, or the increment in strain energy gets saved as the strain energy.

    Vector WaveSpeed(1.e-12,1.e-12,1.e-12); //used to calc. stable timestep
    Vector dx = patch->dCell(); //used to calc. artificial viscosity and timestep

    // Get particle subset for the current patch
    int dwi = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

    // Get the particle variables
    delt_vartype                   delT;
    constParticleVariable<int>     pLocalized,
                                   pAreniscaFlag;
    constParticleVariable<double>  peakI1IDist;  // For variability
    constParticleVariable<double>  pScratchDouble1,
                                   pScratchDouble2,
                                   pPorePressure,
                                   pmass,           //used for stable timestep
                                   pevp,
                                   peve,
                                   pCapX,
                                   pZeta,
                                   pP3;
    constParticleVariable<long64>  pParticleID;
    constParticleVariable<Vector>  pvelocity;
    constParticleVariable<Matrix3> pScratchMatrix,
                                   pep,
                                   pDefGrad,
                                   pStress_old, pStressQS_old,
                                   pBackStress,
                                   pBackStressIso;

    old_dw->get(delT,            lb->delTLabel,   getLevel(patches));
    old_dw->get(peakI1IDist,     peakI1IDistLabel,             pset);  // For variability
    old_dw->get(pLocalized,      lb->pLocalizedMPMLabel,       pset); //initializeCMData()
    old_dw->get(pAreniscaFlag,   pAreniscaFlagLabel,           pset); //initializeCMData()
    old_dw->get(pScratchDouble1, pScratchDouble1Label,         pset); //initializeCMData()
    old_dw->get(pScratchDouble2, pScratchDouble2Label,         pset); //initializeCMData()
    old_dw->get(pPorePressure,   pPorePressureLabel,           pset); //initializeCMData()
    old_dw->get(pmass,           lb->pMassLabel,               pset);
    old_dw->get(pevp,            pevpLabel,                    pset); //initializeCMData()
    old_dw->get(peve,            peveLabel,                    pset); //initializeCMData()
    old_dw->get(pCapX,           pCapXLabel,                   pset); //initializeCMData()
    old_dw->get(pZeta,           pZetaLabel,                   pset); //initializeCMData()
    old_dw->get(pP3,             pP3Label,                     pset); //initializeCMData()
    old_dw->get(pParticleID,     lb->pParticleIDLabel,         pset);
    old_dw->get(pvelocity,       lb->pVelocityLabel,           pset);
    old_dw->get(pScratchMatrix,  pScratchMatrixLabel,          pset); //initializeCMData()
    old_dw->get(pep,             pepLabel,                     pset); //initializeCMData()
    old_dw->get(pDefGrad,        lb->pDeformationMeasureLabel, pset);
    old_dw->get(pStress_old,     lb->pStressLabel,             pset); //initializeCMData()
    old_dw->get(pStressQS_old,   pStressQSLabel,               pset); //initializeCMData()

    // Get the particle variables from interpolateToParticlesAndUpdate() in SerialMPM
    constParticleVariable<double>  pvolume;
    constParticleVariable<Matrix3> pVelGrad_new,
                                   pDefGrad_new;
    new_dw->get(pvolume,        lb->pVolumeLabel_preReloc,  pset);
    new_dw->get(pVelGrad_new,   lb->pVelGradLabel_preReloc, pset);
    new_dw->get(pDefGrad_new,   lb->pDeformationMeasureLabel_preReloc,      pset);

    // Get the particle variables from compute kinematics
    ParticleVariable<int>     pLocalized_new,
                              pAreniscaFlag_new;
    ParticleVariable<double>  peakI1IDist_new;  // For variability
    new_dw->allocateAndPut(peakI1IDist_new,   peakI1IDistLabel_preReloc,       pset); // For variability
    new_dw->allocateAndPut(pLocalized_new,    lb->pLocalizedMPMLabel_preReloc, pset);
    new_dw->allocateAndPut(pAreniscaFlag_new, pAreniscaFlagLabel_preReloc,     pset);

    // Allocate particle variables used in ComputeStressTensor
    ParticleVariable<double>  p_q,
                              pdTdt,
                              pScratchDouble1_new,
                              pScratchDouble2_new,
                              pPorePressure_new,
                              pevp_new,
                              peve_new,
                              pCapX_new,
                              pZeta_new,
                              pP3_new;
    ParticleVariable<Matrix3> pScratchMatrix_new,
                              pep_new,
                              pStress_new, pStressQS_new;

    new_dw->allocateAndPut(p_q,                 lb->p_qLabel_preReloc,         pset);
    new_dw->allocateAndPut(pdTdt,               lb->pdTdtLabel,                pset);
    new_dw->allocateAndPut(pScratchDouble1_new, pScratchDouble1Label_preReloc, pset);
    new_dw->allocateAndPut(pScratchDouble2_new, pScratchDouble2Label_preReloc, pset);
    new_dw->allocateAndPut(pPorePressure_new,   pPorePressureLabel_preReloc,   pset);
    new_dw->allocateAndPut(pevp_new,            pevpLabel_preReloc,            pset);
    new_dw->allocateAndPut(peve_new,            peveLabel_preReloc,            pset);
    new_dw->allocateAndPut(pCapX_new,           pCapXLabel_preReloc,           pset);
    new_dw->allocateAndPut(pZeta_new,           pZetaLabel_preReloc,           pset);
    new_dw->allocateAndPut(pP3_new,             pP3Label_preReloc,             pset);
    new_dw->allocateAndPut(pScratchMatrix_new,  pScratchMatrixLabel_preReloc,  pset);
    new_dw->allocateAndPut(pep_new,             pepLabel_preReloc,             pset);
    new_dw->allocateAndPut(pStress_new,         lb->pStressLabel_preReloc,     pset);
    new_dw->allocateAndPut(pStressQS_new,       pStressQSLabel_preReloc,       pset);

    // Loop over the particles of the current patch to update particle
    // stress at the end of the current timestep along with all other
    // required data such plastic strain, elastic strain, cap position, etc.

    for(ParticleSubset::iterator iter = pset->begin();iter!=pset->end();iter++){
      particleIndex idx = *iter;  //patch index
      //cout<<"pID="<<pParticleID[idx]<<endl;

      // A parameter to consider the thermal effects of the plastic work which
      // is not coded in the current source code. Further development of Arenisca
      // may ativate this feature.
      pdTdt[idx] = 0.0;

      // Particle deletion variable
      pLocalized_new[idx] = pLocalized[idx];

      //Set scratch parameters to old values
      pScratchDouble1_new[idx] = pScratchDouble1[idx];
      pScratchDouble2_new[idx] = pScratchDouble2[idx];
      pScratchMatrix_new[idx]  = pScratchMatrix[idx];

      // Compute the symmetric part of the velocity gradient
      Matrix3 D = (pVelGrad_new[idx] + pVelGrad_new[idx].Transpose())*.5;

      // Use polar decomposition to compute the rotation and stretch tensors
      Matrix3 tensorR, tensorU;

#ifdef MHdeleteBadF
      if(pDefGrad[idx].MaxAbsElem()>1.0e2){
		  pLocalized_new[idx]=-999;
		  cout<<"Large deformation gradient component: [F] = "<<pDefGrad[idx]<<endl;
		  cout<<"Resetting [F]=[I] for this step and deleting particle"<<endl;
		  Identity.polarDecompositionRMB(tensorU, tensorR);
      }
      else if(pDefGrad[idx].Determinant()<1.0e-3){
		  pLocalized_new[idx]=-999;
		  cout<<"Small deformation gradient determinant: [F] = "<<pDefGrad[idx]<<endl;
		  cout<<"Resetting [F]=[I] for this step and deleting particle"<<endl;
		  Identity.polarDecompositionRMB(tensorU, tensorR);
      }
	  else if(pDefGrad[idx].Determinant()>1.0e5){
	  	  pLocalized_new[idx]=-999;
	  	  cout<<"Large deformation gradient determinant: [F] = "<<pDefGrad[idx]<<endl;
	  	  cout<<"Resetting [F]=[I] for this step and deleting particle"<<endl;
	  	  Identity.polarDecompositionRMB(tensorU, tensorR);
      }
      else{
		  pDefGrad[idx].polarDecompositionRMB(tensorU, tensorR);
      }
#else
	  pDefGrad[idx].polarDecompositionRMB(tensorU, tensorR);
#endif
      // Compute the unrotated symmetric part of the velocity gradient
      D = (tensorR.Transpose())*(D*tensorR);

      // To support non-linear elastic properties and to allow for the fluid bulk modulus
      // model to increase elastic stiffness under compression, we allow for the bulk
      // modulus to vary for each substep.  To compute the required number of substeps
      // we use a conservative value for the bulk modulus (the high pressure limit B0+B1)
      // to compute the trial stress and use this to subdivide the strain increment into
      // appropriately sized substeps.  The strain increment is a product of the strain
      // rate and time step, so we pass the strain rate and subdivided time step (rather
      // than a subdivided trial stress) to the substep function.

      // Compute the unrotated stress at the first of the current timestep
      Matrix3 sigma_old = (tensorR.Transpose())*(pStress_old[idx]*tensorR),
              sigmaQS_old = (tensorR.Transpose())*(pStressQS_old[idx]*tensorR);

      // initial assignment for the updated values of plastic strains, volumetric
      // part of the plastic strain, volumetric part of the elastic strain, \kappa,
      // and the backstress. tentative assumption of elasticity

      // Weibull Distribution on PEAKI1 for variability is passed to the subroutines
      // as a single scalar coherence measure, which is 1 for a nominally intact material
      // and 0 for a fully damaged material.  It is possible for d>1, corresponding to
      // a stronger (either because of variability or scale effects) material than the
      // reference sample.
      peakI1IDist_new[idx] = peakI1IDist[idx];
      double coher = 1.0;
      if(d_cm.PEAKI1 > 0.0){
        coher = peakI1IDist[idx]/d_cm.PEAKI1;        // Scalar-valued Damage (XXX) 
      }

      // Divides the strain increment into substeps, and calls substep function
      int stepFlag = computeStep(D,                  // strain "rate"
                                 delT,               // time step (s)
                                 sigmaQS_old,        // unrotated stress at start of step
                                 pCapX[idx],         // hydrostatic comp. strength at start of step
                                 pZeta[idx],         // trace of isotropic backstress at start of step
                                 coher,              // Scalar-valued coherence (XXX)
                                 pP3[idx],           // Modified p3 for initial disaggregation strain.
                                 pep[idx],           // plastic strain at start of step
                                 pStressQS_new[idx], // unrotated stress at end of step
                                 pCapX_new[idx],     // hydrostatic compressive strength at end of step
                                 pZeta_new[idx],     // trace of isotropic backstress at end of step
                                 pep_new[idx],       // plastic strain at end of step
                                 pParticleID[idx]);

      //MH! add P3 as an input:
      pP3_new[idx] = pP3[idx];

      // If the computeStep function can't converge it will return a stepFlag!=1.  This indicates substepping
      // has failed, and the particle will be deleted.
      if(stepFlag!=0){
        pLocalized_new[idx]=-999;
        cout<<"bad step, deleting particle"<<endl;
      }

      // Plastic volumetric strain at end of step
      pevp_new[idx] = pep_new[idx].Trace();

      // Elastic volumetric strain at end of step, compute from updated deformatin gradient.
      // peve_new[idx] = peve[idx] + D.Trace()*delT - pevp_new[idx] + pevp[idx];  // Faster
      peve_new[idx] = log(pDefGrad_new[idx].Determinant()) - pevp_new[idx];           // More accurate

      // Set pore pressure (plotting variable)
      pPorePressure_new[idx] = computePorePressure(peve_new[idx]+pevp_new[idx]);

      // ======================================================================RATE DEPENDENCE CODE
      // Compute the new dynamic stress from the old dynamic stress and the new and old QS stress
      // using Duvaut-Lions rate dependence, as described in "Elements of Phenomenological Plasticity",
      // by RM Brannon.

      if (d_cm.T1_rate_dependence != 0.0 && d_cm.T2_rate_dependence != 0.0 ) {
        // This is not straightforward, due to nonlinear elasticity.  The equation requires that we
        // compute the trial stress for the step, but this is not known, since the bulk modulus is
        // evolving through the substeps.  It would be necessary to to loop through the substeps to
        // compute the trial stress assuming nonlinear elasticity, but instead we will approximate
        // the trial stress the average of the elastic moduli at the start and end of the step.
        double bulk_n, shear_n, bulk_p, shear_p;
        computeElasticProperties(sigmaQS_old,       pep[idx],    pP3[idx],bulk_n,shear_n);
        computeElasticProperties(pStressQS_new[idx],pep_new[idx],pP3[idx],bulk_p,shear_p);
 
        Matrix3 sigma_trial = computeTrialStress(sigma_old,  // Dynamic stress at the start of the step
                                                 D*delT,     // Total train increment over the step
                                                 0.5*(bulk_n + bulk_p),  // midstep bulk modulus
                                                 0.5*(shear_n + shear_p) ); // midstep shear modulus

        // The characteristic time is defined from the rate dependence input parameters and the
        // magnitude of the strain rate.  MH!: I don't have a reference for this equation.
        //
        // tau = T1*(epsdot)^(-T2) = T1*(1/epsdot)^T2, modified to avoid division by zero.
        double tau = d_cm.T1_rate_dependence*Pow(1.0/max(D.Norm(), 1.0e-15),d_cm.T2_rate_dependence);

        // RH and rh are defined by eq. 6.93 in the book chapter, but there seems to be a sign error
        // in the text, and I've rewritten it to avoid computing the exponential twice.
        double dtbytau = delT/tau;
        double rh  = exp(-dtbytau);
        double RH  = (1.0 - rh)/dtbytau;

        // sigma_new = sigmaQS_new + sigma_over_new, as defined by eq. 6.92
        // sigma_over_new = [(sigma_trial_new - sigma_old) - (sigmaQS_new-sigmaQS_old)]*RH + sigma_over_old*rh
        pStress_new[idx] = pStressQS_new[idx]
                           + ((sigma_trial - sigma_old) - (pStressQS_new[idx] - sigmaQS_old))*RH
                           + (sigma_old - sigmaQS_old)*rh;
      }
      else { // No rate dependence, the dynamic stress equals the static stress.
        pStress_new[idx] = pStressQS_new[idx];
      } // ==========================================================================================

      // Use polar decomposition to compute the rotation and stretch tensors.  These checks prevent
	  // failure of the polar decomposition algorithm if [F_new] has some extreme values.
#ifdef MHdeleteBadF
      if(pDefGrad_new[idx].MaxAbsElem()>1.0e16){
		  pLocalized_new[idx]=-999;
		  cout<<"Large deformation gradient component: [F_new] = "<<pDefGrad_new[idx]<<endl;
		  cout<<"Resetting [F_new]=[I] for this step and deleting particle"<<endl;
		  Identity.polarDecompositionRMB(tensorU, tensorR);
      }
      else if(pDefGrad_new[idx].Determinant()<1.0e-16){
		  pLocalized_new[idx]=-999;
		  cout<<"Small deformation gradient determinant: [F_new] = "<<pDefGrad_new[idx]<<endl;
		  cout<<"Resetting [F_new]=[I] for this step and deleting particle"<<endl;
		  Identity.polarDecompositionRMB(tensorU, tensorR);
      }
	  else if(pDefGrad_new[idx].Determinant()>1.0e16){
		  pLocalized_new[idx]=-999;
		  cout<<"Large deformation gradient determinant: [F_new] = "<<pDefGrad_new[idx]<<endl;
		  cout<<"Resetting [F_new]=[I] for this step and deleting particle"<<endl;
		  Identity.polarDecompositionRMB(tensorU, tensorR);
      }
      else{
		  pDefGrad_new[idx].polarDecompositionRMB(tensorU, tensorR);
      }
#else
      pDefGrad_new[idx].polarDecompositionRMB(tensorU, tensorR);
#endif

      // Compute the rotated dynamic and quasistatic stress at the end of the current timestep
      pStress_new[idx] = (tensorR*pStress_new[idx])*(tensorR.Transpose());
      pStressQS_new[idx] = (tensorR*pStressQS_new[idx])*(tensorR.Transpose());

      // Compute wave speed + particle velocity at each particle, store the maximum
      // Conservative elastic properties used to compute number of time steps:
      // Get the Arenisca model parameters.
      double bulk,
             shear;

		 
#ifdef MHdisaggregationStiffness
	  // Compute the wave speed for the particle based on the reduced stiffness, which
	  // is computed when the value of P3 is sent to computeElasticProperties.
		if(d_cm.Use_Disaggregation_Algorithm){
	    computeElasticProperties(pStressQS_new[idx],pep_new[idx],pP3[idx],bulk,shear);
		} else {
	    computeElasticProperties(bulk,shear); // High pressure bulk and shear moduli.
		}
#else
	  computeElasticProperties(bulk,shear); // High pressure bulk and shear moduli.
#endif
		
			 
      double rho_cur = pmass[idx]/pvolume[idx];
             c_dil = sqrt((bulk+four_third*shear)/rho_cur);

      WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
                       Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
                       Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));

      // Compute artificial viscosity term
      if (flag->d_artificial_viscosity) {
        double dx_ave = (dx.x() + dx.y() + dx.z())*one_third;
        double c_bulk = sqrt(bulk/rho_cur);
        p_q[idx] = artificialBulkViscosity(D.Trace(), c_bulk, rho_cur, dx_ave);
      } else {
        p_q[idx] = 0.;
      }

      // Compute the averaged stress
      Matrix3 AvgStress = (pStress_new[idx] + pStress_old[idx])*0.5;
      // Compute the strain energy increment associated with the particle
      double e = (D(0,0)*AvgStress(0,0) +
                  D(1,1)*AvgStress(1,1) +
                  D(2,2)*AvgStress(2,2) +
                  2.0*(D(0,1)*AvgStress(0,1) +
                       D(0,2)*AvgStress(0,2) +
                       D(1,2)*AvgStress(1,2))) * pvolume[idx]*delT;

      // Accumulate the total strain energy
      // MH! Note the initialization of se needs to be fixed as it is currently reset to 0
      se += e;
    }

    // Compute the stable timestep based on maximum value of "wave speed + particle velocity"
    WaveSpeed = dx/WaveSpeed; // Variable now holds critical timestep (not speed)

    double delT_new = WaveSpeed.minComponent();

    // Put the stable timestep and total strain enrgy
    new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
    if (flag->d_reductionVars->accStrainEnergy ||
        flag->d_reductionVars->strainEnergy) {
      new_dw->put(sum_vartype(se),        lb->StrainEnergyLabel);
    }
  }
} // -----------------------------------END OF COMPUTE STRESS TENSOR FUNCTION

// ****************************************************************************************************
// ****************************************************************************************************
// **** HOMEL's FUNCTIONS FOR GENERALIZED RETURN AND NONLINEAR ELASTICITY *****************************
// ****************************************************************************************************
// ****************************************************************************************************

// Divides the strain increment into substeps, and calls substep function
int Arenisca3::computeStep(const Matrix3& D,       // strain "rate"
                           const double & Dt,      // time step (s)
                           const Matrix3& sigma_n, // unrotated stress at start of step(t_n)
                           const double & X_n,     // hydrostatic comrpessive strength at start of step(t_n)
                           const double & Zeta_n,  // trace of isotropic backstress at start of step(t_n)
                           const double & coher,   // scalar-valued coherence XXX
                           const double & P3,      // initial disaggregation strain
                           const Matrix3& ep_n,    // plastic strain at start of step(t_n)
                           Matrix3& sigma_p,       // unrotated stress at end of step(t_n+1)
                           double & X_p,           // hydrostatic comrpessive strength at end of step(t_n+1)
                           double & Zeta_p,        // trace of isotropic backstress at end of step(t_n+1)
                           Matrix3& ep_p,          // plastic strain at end of step (t_n+1)
                           long64 ParticleID)      // ParticleID for debug purposes
{
 // All stress values within computeStep are quasistatic.
  int n,
      chimax = 1,                                  // max allowed subcycle multiplier
      // MH!: make this an input parameter for subcycle control
      chi = 1,                                      // subcycle multiplier
      stepFlag,                                     // 0/1 good/bad step
      substepFlag;                                  // 0/1 good/bad substep
  
  // MH! Need to initialize X_old and Zeta_old BEFORE compute StepDivisions!
  // currently this breaks the step division code.
  double dt,                                        // substep time increment
         X_old = X_n,                               // X at start of substep
         X_new,                                     // X at end of substep
         Zeta_old = Zeta_n,                         // Zeta at start of substep
         Zeta_new;                                  // Zeta at end of substep

  Matrix3 sigma_old,                                // sigma at start of substep
          sigma_new,                                // sigma at end of substep
          ep_old,                                   // plastic strain at start of substep
          ep_new;                                   // plastic strain at end of substep

// (1) Define conservative elastic properties based on the high-pressure
// limit of the bulk modulus function. The bulk modulus function can be
// without stress and plastic strain arguments to return the
// high pressure limit.  These values are used only for computing number
// of substeps and stable time steps.
double bulk, shear;

#ifdef MHdisaggregationStiffness
if(d_cm.Use_Disaggregation_Algorithm){
	computeElasticProperties(sigma_n,ep_n,P3,bulk,shear);
}
else{
	computeElasticProperties(bulk,shear);
}
#else	   
computeElasticProperties(bulk,shear);
#endif

//Compute the trial stress: [sigma_trial] = computeTrialStress(sigma_old,d_e,K,G)
Matrix3 sigma_trial = computeTrialStress(sigma_old,D*Dt,bulk,shear);

double  I1_trial,
        J2_trial,
        rJ2_trial;
Matrix3 S_trial,
        d_e;
computeInvariants(sigma_trial,S_trial,I1_trial,J2_trial,rJ2_trial);

// (2) Determine the number of substeps (nsub) based on the magnitude of
// the trial stress increment relative to the characteristic dimensions
// of the yield surface.  Also compare the value of the pressure dependent
// elastic properties as sigma_old and sigma_trial and adjust nsub if
// there is a large change to ensure an accurate solution for nonlinear
// elasticity even with fully elastic loading.
  int nsub = computeStepDivisions(X_old,Zeta_old,P3,ep_old,sigma_old,sigma_trial);
  if (nsub < 0) { // nsub > d_cm.subcycling_characteristic_number. Delete particle
    goto failedStep;
  }

// (3) Compute a subdivided time step:
//
  stepDivide:
    dt = Dt/(chi*nsub);
    d_e = D*dt;

// (4) Set {sigma_old,X_old,Zeta_old,ep_old}={sigma_n,X_n,Zeta_n,ep_n} to their
// values at the start of the step, and set n = 1:
  sigma_old = sigma_n;
  X_old     = X_n;
  Zeta_old  = Zeta_n;
  ep_old    = ep_n;
  n = 1;

// (5) Call substep function {sigma_new,ep_new,X_new,Zeta_new}
//                               = computeSubstep(D,dt,sigma_old,ep_old,X_old,Zeta_old)
computeSubstep:
  substepFlag = computeSubstep(d_e,sigma_old,ep_old,X_old,Zeta_old,coher,P3,
                               sigma_new,ep_new,X_new,Zeta_new);

// (6) Check error flag from substep calculation:
  if (substepFlag == 0) { // no errors in substep calculation
    if (n < (chi*nsub)) { // update and keep substepping
      sigma_old = sigma_new;
      X_old     = X_new;
      Zeta_old  = Zeta_new;
      ep_old    = ep_new;
      n++;
      goto computeSubstep;
    }
    else goto successfulStep; // n = chi*nsub, Step is done
  }
  else
  {  // failed substep
    if (chi < chimax)   {       // errors in substep calculation
      chi = 2*chi;
      goto stepDivide;
    }
    else goto failedStep;     // bad substep and chi>=chimax, Step failed even with substepping
  }

// (7) Successful step, set value at end of step to value at end of last substep.
successfulStep:
  sigma_p   = sigma_new;
  X_p       = X_new;
  Zeta_p    = Zeta_new;
  ep_p      = ep_new;
  stepFlag  = 0;
  return stepFlag;

// (8) Failed step, Send ParticleDelete Flag to Host Code, Store Inputs to particle data:
failedStep:
  // input values for sigma_new,X_new,Zeta_new,ep_new, along with error flag
  sigma_p   = sigma_n;
  X_p       = X_n;
  Zeta_p    = Zeta_n;
  ep_p      = ep_n;
  stepFlag  = 1;
#ifdef MHdebug
  cout << "995: Step Failed I1_n = "<<sigma_n.Trace() <<", I1_p = "<<sigma_p.Trace()<< endl;
  cout << "996: evp_p = "<<ep_p.Trace()<<", evp_n = "<<ep_n.Trace()<<endl;
  cout << "997: X_p = "<<X_p<<", X_n = "<<X_n<<endl;
#endif
  return stepFlag;

} //===================================================================

// [shear,bulk] = computeElasticProperties()
void Arenisca3::computeElasticProperties(double & bulk,
                                         double & shear)
{
// When computeElasticProperties() is called with two doubles as arguments, it
// computes the high pressure limit tangent elastic shear and bulk modulus
// This is used to esimate wave speeds and make conservative estimates of substepping.
  shear   = d_cm.G0;            // Linear elastic shear Modulus
  bulk    = d_cm.B0 + d_cm.B1;  // Bulk Modulus
  
  // If the user has specified a nonzero G1 and G2, these are used to define a pressure
  // dependent poisson ratio, which is used to adjust the shear modulus along with the
  // bulk modulus.  The high pressure limit has nu=G1+G2;
  //if ((d_cm.G1!=0.0)&&(d_cm.G2!=0.0)){
//	  // High pressure bulk modulus:
//	  double nu = d_cm.G1+d_cm.G2;
//	  shear = 1.5*bulk*(1.0-2.0*nu)/(1.0+nu);
  //}
  
} //===================================================================

// [shear,bulk] = computeElasticProperties(stress, ep)
void Arenisca3::computeElasticProperties(const Matrix3 stress,
                                         const Matrix3 ep,
										 const double& P3,
                                         double & bulk,
                                         double & shear)
{
// Compute the nonlinear elastic tangent stiffness as a function of the pressure
// plastic strain, and fluid parameters.
	double  b0 = d_cm.B0,
			b1 = d_cm.B1,
			b2 = d_cm.B2,
			b3 = d_cm.B3,
			b4 = d_cm.B4,
			g0 = d_cm.G0,
			I1 = stress.Trace(),
			evp = ep.Trace();

// ..........................................................Undrained
// The low pressure bulk and shear moduli are also used for the tensile response.
	bulk = b0;
	shear = g0;
	// To be thermodynamically consistent, the shear modulus in an isotropic model
	// must be constant, but the bulk modulus can depend on pressure.  However, this
	// leads to a Poisson's ratio that approaches 0.5 at high pressures, which is
	// inconsistent with experimental data for the Poisson's ratio, inferred from the 
	// Young's modulus.  Induced anisotropy is likely the cause of the discrepency, 
	// but it may be better to allow the shear modulus to vary so the Poisson's ratio
	// remains reasonable.
	//
	// If the user has specified a nonzero value of G1 and G2, the shear modulus will
	// vary with pressure so the drained Poisson's ratio transitions from G1 to G1+G2 as 
	// the bulk modulus varies from B0 to B0+B1.  The fluid model further affects the 
	// bulk modulus, but does not alter the shear modulus, so the pore fluid does
	// increase the Poisson's ratio.  
	if(evp <= 0.0){
#ifdef MHfastfcns
		// Elastic-plastic coupling
		if (evp < 0.0){bulk = bulk - b3*fasterexp(b4/evp);}
		
		
		// Pressure dependence
		if (I1 < 0.0){
			double expb2byI1 = fasterexp(b2/I1);
			bulk = bulk + b1*expb2byI1;
			if(d_cm.G1!=0.0 && d_cm.G2!=0.0){
				double nu = d_cm.G1 + d_cm.G2*expb2byI1;
				shear = 1.5*bulk*(1.0-2.0*nu)/(1.0+nu);
			}
		}
#else
		if (I1 < 0.0){
			double expb2byI1 = exp(b2/I1);
			bulk = bulk + b1*expb2byI1;
			if(d_cm.G1!=0.0 && d_cm.G2!=0.0){
				double nu = d_cm.G1 + d_cm.G2*expb2byI1;
				shear = 1.5*bulk*(1.0-2.0*nu)/(1.0+nu);
			}
		}
		// Elastic-plastic coupling
		if (evp < 0.0){bulk = bulk - b3*exp(b4/evp);}
#endif	
	}
	
#ifdef MHdisaggregationStiffness

	if(d_cm.Use_Disaggregation_Algorithm){
#ifdef MHfastfcns
		double fac = fasterexp(-(P3+evp));
#else
	    double fac = exp(-(P3+evp));
#endif
		double scale = max(fac,0.00001);
		bulk = bulk*scale;
		shear = shear*scale;
	}
#endif

// In  compression, or with fluid effects if the strain is more compressive
// than the zero fluid pressure volumetric strain:
	if (evp <= ev0 && Kf!=0.0){// ..........................................................Undrained
		// Compute the porosity from the strain using Homel's simplified model, and
		// then use this in the Biot-Gassmann formula to compute the bulk modulus.

		// The dry bulk modulus, taken as the low pressure limit of the nonlinear
		// formulation:
		double Kd = b0;
#ifdef MHfastfcns
		if (evp < 0.0){Kd = b0 - b3*fasterexp(b4/evp);}
		// Current unloaded porosity (phi):
		double C2 = fasterexp(evp*Km/C1)*phi_i;
		double phi = C2/(-fasterexp(evp*Kf/C1)*(phi_i-1.0) + C2);
#else
		if (evp < 0.0){Kd = b0 - b3*exp(b4/evp);}
		// Current unloaded porosity (phi):
		double C2 = exp(evp*Km/C1)*phi_i;
		double phi = C2/(-exp(evp*Kf/C1)*(phi_i-1.0) + C2);
#endif
		// Biot-Gassmann formula for the saturated bulk modulus, evaluated at the
		// current porosity.  This introduces some error since the Kd term is a
		// function of the initial porosity, but since large strains would also
		// modify the bulk modulus through damage
		double oneminusKdbyKm = 1.0 - Kd / Km;
		bulk = Kd + oneminusKdbyKm*oneminusKdbyKm/((oneminusKdbyKm - phi)/Km + (1.0/Kf - 1.0/Km)*phi);
	}
	
} //===================================================================

// [sigma_trial] = computeTrialStress(sigma_old,d_e,K,G)
Matrix3 Arenisca3::computeTrialStress(const Matrix3& sigma_old,  // old stress
                                      const Matrix3& d_e,        // Strain increment
                                      const double& bulk,        // bulk modulus
                                      const double& shear)       // shear modulus
{
// Compute the trial stress for some increment in strain assuming linear elasticity
// over the step.
  Matrix3 d_e_iso = one_third*d_e.Trace()*Identity;
  Matrix3 d_e_dev = d_e - d_e_iso;
  Matrix3 sigma_trial = sigma_old + (3.0*bulk*d_e_iso + 2.0*shear*d_e_dev);
  return sigma_trial;
} //===================================================================

// [nsub] = computeStepDivisions(X,Zeta,ep,sigma_n,sigma_trial)
int Arenisca3::computeStepDivisions(const double& X,
                                    const double& Zeta,
									const double& P3,
                                    const Matrix3& ep,
                                    const Matrix3& sigma_n,
                                    const Matrix3& sigma_trial)
{
// compute the number of step divisions (substeps) based on a comparison
// of the trial stress relative to the size of the yield surface, as well
// as change in elastic properties between sigma_n and sigma_trial.
	
  int nmax = ceil(d_cm.subcycling_characteristic_number);
  
  // Compute change in bulk modulus:
  double  bulk_n,shear_n,bulk_trial,shear_trial;
  computeElasticProperties(sigma_n,ep,P3,bulk_n,shear_n);
  computeElasticProperties(sigma_trial,ep,P3,bulk_trial,shear_trial);
  int n_bulk = ceil(fabs(bulk_n-bulk_trial)/bulk_n);  
  
  // Compute trial stress increment relative to yield surface size:
  Matrix3 d_sigma = sigma_trial - sigma_n;
  double size = 0.5*(d_cm.PEAKI1 - X);
	
  if (d_cm.STREN > 0.0){
	  size = min(size,d_cm.STREN);
  }  
  int n_yield = ceil(1.0e-4*d_sigma.Norm()/size);

  // nsub is the maximum of the two values.above.  If this exceeds allowable,
  // throw warning and delete particle.
  int nsub = max(n_bulk,n_yield);
 
  if (nsub>d_cm.subcycling_characteristic_number){
#ifdef MHdebug
    cout<<"\nstepDivide out of range."<<endl;
	cout<<"P3 = "<<P3<<endl;
	cout<<"d_sigma.Norm() = "<<d_sigma.Norm()<<endl;
	cout<<"X = "<<X<<endl;
	cout<<"size = "<<size<<endl;
	cout<<"n_yield = "<<n_yield<<endl;
	cout<<"bulk_n = "<<bulk_n<<endl;
	cout<<"bulk_trial = "<<bulk_trial<<endl;
	cout<<"n_bulk = "<<n_bulk<<endl;
	cout<<"nsub = "<<nsub<<" > "<< d_cm.subcycling_characteristic_number<<endl;
#endif
    nsub = -1;
  }
  else {
    nsub = min(max(nsub,1),nmax);
  }
  return nsub;
} //===================================================================

void Arenisca3::computeInvariants(const Matrix3& stress,
                                  Matrix3& S,
                                  double & I1,
                                  double & J2,
                                  double & rJ2)
{
  // Compute the first invariants
  I1 = stress.Trace();  //Pa

  // Compute the deviatoric part of the tensor
  S = stress - one_third*Identity*I1;  //Pa

  // Compute the second invariant
  J2 = 0.5*S.Contract(S);  //Pa^2

  if(J2 < 1e-16*(I1*I1+J2)){
    J2=0.0;
  };
  rJ2 = sqrt(J2);
} //===================================================================

// Computes the updated stress state for a substep
int Arenisca3::computeSubstep(const Matrix3& d_e,       // Total strain increment for the substep
                              const Matrix3& sigma_old, // stress at start of substep
                              const Matrix3& ep_old,    // plastic strain at start of substep
                              const double & X_old,     // hydrostatic compressive strength at start of substep
                              const double & Zeta_old,  // trace of isotropic backstress at start of substep
                              const double & coher,     // scalar valued coherence
                              const double & P3,        // initial disaggregation strain
                              Matrix3& sigma_new,       // stress at end of substep
                              Matrix3& ep_new,          // plastic strain at end of substep
                              double & X_new,           // hydrostatic compressive strength at end of substep
                              double & Zeta_new         // trace of isotropic backstress at end of substep
                             )
{
// Computes the updated stress state for a substep that may be either elastic, plastic, or
// partially elastic.   Returns an integer flag 0/1 for a good/bad update.
  int     substepFlag,
		  returnFlag;

// (1)  Compute the elastic properties based on the stress and plastic strain at
// the start of the substep.  These will be constant over the step unless elastic-plastic
// is used to modify the tangent stiffness in the consistency bisection iteration.
  double bulk,
         shear;
  computeElasticProperties(sigma_old,ep_old,P3,bulk,shear);

// (3) Compute the trial stress: [sigma_trail] = computeTrialStress(sigma_old,d_e,K,G)
  Matrix3 sigma_trial = computeTrialStress(sigma_old,d_e,bulk,shear),
          S_trial;

  double I1_trial,
         J2_trial,
         rJ2_trial;
  computeInvariants(sigma_trial,S_trial,I1_trial,J2_trial,rJ2_trial);

// (4) Evaluate the yield function at the trial stress:
  // Compute the limit parameters based on the value of coher.  These are then passed down
  // to the computeYieldFunction, to avoid the expense of repeatedly computing a3
  double limitParameters[4];  //double a1,a2,a3,a4;
  computeLimitParameters(limitParameters,coher);
  
  int YIELD = computeYieldFunction(I1_trial,rJ2_trial,X_old,Zeta_old,coher,limitParameters);
  if (YIELD == -1) { // elastic substep
    sigma_new = sigma_trial;
    X_new = X_old;
    Zeta_new = Zeta_old;
    ep_new = ep_old;
    substepFlag = 0;
    goto successfulSubstep;
  }
  if (YIELD == 1) {  // elastic-plastic or fully-plastic substep
// (5) Compute non-hardening return to initial yield surface:
//     [sigma_0,d_e_p,0] = (nonhardeningReturn(sigma_trial,sigma_old,X_old,Zeta_old,K,G)
    double  I1_0,       // I1 at stress update for non-hardening return
            rJ2_0,      // rJ2 at stress update for non-hardening return
            TOL = 1e-4; // bisection convergence tolerance on eta (if changed, change imax)
    Matrix3 S_0,        // S (deviator) at stress update for non-hardening return
            d_ep_0;     // increment in plastic strain for non-hardening return

    Matrix3 S_old;
    double I1_old,
           J2_old,
           rJ2_old,
           evp_old = ep_old.Trace();
    computeInvariants(sigma_old,S_old,I1_old,J2_old,rJ2_old);

    // returnFlag would be != 0 if there was an error in the nonHardeningReturn call, but
    // there are currently no tests in that function that could detect such an error.
    returnFlag = nonHardeningReturn(I1_trial,rJ2_trial,S_trial,
									I1_old,rJ2_old,S_old,
									d_e,X_old,Zeta_old,coher,bulk,shear,
									I1_0,rJ2_0,S_0,d_ep_0);
	if (returnFlag!=0){
#ifdef MHdebug
		cout << "1344: failed nonhardeningReturn in substep "<< endl;
#endif
		goto failedSubstep;
	}

	// If there is no porosity (p3=0) and no fluid effects (Kf=0) then the nonhardening
	// return will be the solution
	if ( (d_cm.p3_crush_curve == 0.0)&&(d_cm.fluid_B0==0.0) ){
        Zeta_new = Zeta_old,
		X_new = X_old;
		sigma_new = one_third*I1_0*Identity + S_0;
		ep_new = ep_old + d_ep_0;
		
		goto successfulSubstep;
	}
	
    double d_evp_0 = d_ep_0.Trace();
	
// (6) Iterate to solve for plastic volumetric strain consistent with the updated
//     values for the cap (X) and isotropic backstress (Zeta).  Use a bisection method
//     based on the multiplier eta,  where  0<eta<1

    double eta_out = 1.0,
           eta_in = 0.0,
           eta_mid,
           d_evp;
    int i = 0,
        imax = 93;  // imax = ceil(-10.0*log(TOL)); // Update this if TOL changes
	
    double dZetadevp = computedZetadevp(Zeta_old,evp_old);
    
// (7) Update Internal State Variables based on Last Non-Hardening Return:
//
updateISV:
    i++;
    eta_mid   = 0.5*(eta_out+eta_in);
    d_evp     = eta_mid*d_evp_0;

    // Update X exactly
    X_new     = computeX(evp_old + d_evp,P3);
    // Update zeta. min() eliminates tensile fluid pressure from explicit integration error
    Zeta_new = min(Zeta_old + dZetadevp*d_evp,0.0);

// (8) Check if the updated yield surface encloses trial stres.  If it does, there is too much
//     plastic strain for this iteration, so we adjust the bisection parameters and recompute
//     the state variable update.
	
    if( computeYieldFunction(I1_trial,rJ2_trial,X_new,Zeta_new,coher,limitParameters)!=1 ){
      eta_out = eta_mid;
      if( i >= imax ){
        // solution failed to converge within the allowable iterations, which means
        // the solution requires a plastic strain that is less than TOL*d_evp_0
        // In this case we are near the zero porosity limit, so the response should
        // be that of no porosity. By setting eta_out=eta_in, the next step will
        // converge with the cap position of the previous iteration.  In this case,
        // we set evp=-p3 (which corresponds to X=1e12*p0) so subsequent compressive
        // loading will respond as though there is no porosity.  If there is dilatation
        // in subsequent loading, the porosity will be recovered.
        eta_out=eta_in;
      }
      goto updateISV;
    }

// (9) Recompute the elastic properties based on the midpoint of the updated step:
//     [K,G] = computeElasticProperties( (sigma_old+sigma_new)/2,ep_old+d_ep/2 )
//     and compute return to updated surface.
//    MH! change this when elastic-plastic coupling is used.
//    Matrix3 sigma_new = ...,
//            ep_new    = ...,
//    computeElasticProperties((sigma_old+sigma_new)/2,(ep_old+ep_new)/2,bulk,shear);
    double  I1_new,
            rJ2_new,
            d_evp_new;
    Matrix3 S_new,
            d_ep_new;
    returnFlag = nonHardeningReturn(I1_trial,rJ2_trial,S_trial,
									I1_old,rJ2_old,S_old,
									d_e,X_new,Zeta_new,coher,bulk,shear,
									I1_new,rJ2_new,S_new,d_ep_new);
	if (returnFlag!=0){
#ifdef MHdebug
		cout << "1344: failed nonhardeningReturn in substep "<< endl;
#endif
		goto failedSubstep;
	}

// (10) Check whether the isotropic component of the return has changed sign, as this
//      would indicate that the cap apex has moved past the trial stress, indicating
//      too much plastic strain in the return.

    //if(fabs(I1_trial - I1_new)>(d_cm.B0*TOL) && Sign(I1_trial - I1_new)!=Sign(I1_trial - I1_0)){
    if(Sign(I1_trial - I1_new)!=Sign(I1_trial - I1_0)){
      eta_out = eta_mid;
      if( i >= imax ){
        // solution failed to converge within the allowable iterations, which means
        // the solution requires a plastic strain that is less than TOL*d_evp_0
        // In this case we are near the zero porosity limit, so the response should
        // be that of no porosity. By setting eta_out=eta_in, the next step will
        // converge with the cap position of the previous iteration.  In this case,
        // we set evp=-p3 (which corresponds to X=1e12*p0) so subsequent compressive
        // loading will respond as though there is no porosity.  If there is dilatation
        // in subsequent loading, the porosity will be recovered.
        eta_out = eta_in;
      }
      goto updateISV;
    }

    // Compare magnitude of plastic strain with prior update
    d_evp_new = d_ep_new.Trace();   // Increment in vol. plastic strain for return to new surface
    ep_new = ep_old + d_ep_new;

    // Check for convergence
    if( fabs(eta_out-eta_in) < TOL ){ // Solution is converged
      sigma_new = one_third*I1_new*Identity + S_new;

    // If out of range, scale back isotropic plastic strain.
      if(ep_new.Trace()<-P3){
        d_evp_new = -P3-ep_old.Trace();
        Matrix3 d_ep_new_iso = one_third*d_ep_new.Trace()*Identity,
                d_ep_new_dev = d_ep_new - d_ep_new_iso;
        ep_new = ep_old + d_ep_new_dev + one_third*d_evp_new*Identity;
      }

      // Update X exactly
      X_new = computeX(ep_new.Trace(),P3);
      // Update zeta. min() eliminates tensile fluid pressure from explicit integration error
      Zeta_new = min(Zeta_old + dZetadevp*d_evp_new,0.0);

      goto successfulSubstep;
    }
    if( i >= imax ){
      // Solution failed to converge but not because of too much plastic strain
      // (which would have been caught by the checks above).  In this case we
      // go to the failed substep return, which will trigger subcycling and
      // particle deletion (if subcycling doesn't work).
      //
      // This code was never reached in testing, but is here to catch
      // unforseen errors.
#ifdef MHdebug
      cout << "1273: i>=imax, failed substep "<< endl;
#endif
      goto failedSubstep;
    }

// (11) Compare magnitude of the volumetric plastic strain and bisect on eta
//
    if( fabs(d_evp_new) > eta_mid*fabs(d_evp_0) ){
      eta_in = eta_mid;
    }
    else {
      eta_out = eta_mid;
    }
    goto updateISV;
  }
  
// (12) Compute softening, collapse yield surface, and compute nonhardening return
// to collapse yield surface.
//
// This is not rigorous, since the treatment of softening is not included as a 
// hardenining mechanism in the computation of the increment in plastic strain.
//
// Figure out where to put this:
// coher = max(coher - d_ep_new.Norm()/d_cm.failure_strain);
//  //
//  returnFlag = nonHardeningReturn(I1_trial,rJ2_trial,S_trial,
//								  I1_old,rJ2_old,S_old,
//								  d_e,X_new,Zeta_new,coher,bulk,shear,
//								  I1_new,rJ2_new,S_new,d_ep_new);
//  if (returnFlag!=0){
//#ifdef MHdebug
//	  cout << "1344: failed nonhardeningReturn in substep "<< endl;
//#endif
//	  goto failedSubstep;
  
  
  
// (13) Return updated values for successful/unsuccessful steps
//
successfulSubstep:
  substepFlag = 0;
  return substepFlag;

failedSubstep:
  sigma_new = sigma_old;
  ep_new = ep_old;
  X_new = X_old;
  Zeta_new = Zeta_old;
  substepFlag = 1;
  return substepFlag;
} //===================================================================

// Compute state variable X, the Hydrostatic Compressive strength (cap position)
double Arenisca3::computeX(const double& evp,const double& P3)
{
  // X is the value of (I1 - Zeta) at which the cap function crosses
  // the hydrostat. For the drained material in compression. X(evp)
  // is derived from the emprical Kayenta crush curve, but with p2 = 0.
  // In tension, M. Homel's piecewsie formulation is used.

  // define and initialize some variables
  double p0  = d_cm.p0_crush_curve,
         p1  = d_cm.p1_crush_curve,
         X;
  

  if(evp<=-P3) { // ------------Plastic strain exceeds allowable limit--------------------------
    // The plastic strain for this iteration has exceed the allowable
    // value.  X is not defined in this region, so we set it to a large
    // negative number.
    //
    // The code should never have evp<-p3, but will have evp=-p3 if the
    // porosity approaches zero (within the specified tolerance).  By setting
    // X=1e12*p0, the material will respond as though there is no porosity.
    X = 1.0e12*p0;
  }
  else { // ------------------Plastic strain is within allowable domain------------------------
    // We first compute the drained response.  If there are fluid effects, this value will
    // be used in detemining the elastic volumetric strain to yield.
    if(evp <= 0.0){
      // This is an expensive calculation, but fasterlog() may cause errors.
      X = (p0*p1 + log((evp+P3)/P3))/p1;
    }
    else{
      // This is an expensive calculation, but fastpow() may cause errors.
      X = p0*Pow(1.0 + evp, 1.0/(p0*p1*P3));
    }

    if(Kf!=0.0 && evp<=ev0) { // ------------------------------------------- Fluid Effects
      // This is an expensive calculation, but fastpow() may cause errors.
      // First we evaluate the elastic volumetric strain to yield from the
      // empirical crush curve (Xfit) and bulk modulus (Kfit) formula for
      // the drained material.  Xfit was computed as X above.
      double b0 = d_cm.B0,
             b1 = d_cm.B1,
             b2 = d_cm.B2,
             b3 = d_cm.B3,
             b4 = d_cm.B4;

      // Kfit is the drained bulk modulus evaluated at evp, and for I1 = Xdry/2.
      double Kdry = b0 + b1*exp(2.0*b2/X);
      if (evp<0.0){Kdry = Kdry - b3*exp(b4/evp);}

      // Now we use our engineering model for the bulk modulus of the
      // saturated material (Keng) to compute the stress at our elastic strain to yield.
      // Since the stress and plastic strain tensors are not available in this scope, we call the
      // computeElasticProperties function with and isotropic matrices that will have the
      // correct values of evp. (The saturated bulk modulus doesn't depend on I1).
      double Ksat,Gsat;       // Not used, but needed to call computeElasticProperties()
      // This needs to be evaluated at the current value of pressure.
      computeElasticProperties(one_sixth*X*Identity,one_third*evp*Identity,P3,Ksat,Gsat); //Overwrites Geng & Keng

      // Compute the stress to hydrostatic yield.
      // We are only in this looop if(evp <= ev0)
      X = X*Ksat/Kdry;   // This is X_sat = K_sat*eve = K_sat*(X_dry/K_dry)
    } // End fluid effects
  } // End plastic strain in allowable domain
  return X;
} //===================================================================

// Compute the strain at zero pore pressure from initial pore pressure (Pf0)
double Arenisca3::computePorePressure(const double ev)
{
  // This compute the plotting variable pore pressure, which is defined from
  // input paramters and the current total volumetric strain (ev).
  double pf = 0.0;                          // pore fluid pressure

  if(ev<=ev0 && Kf!=0){ // ....................fluid effects are active
    //double Km = d_cm.B0 + d_cm.B1;                   // Matrix bulk modulus (inferred from high pressure limit of drained bulk modulus)
    //double pfi = d_cm.fluid_pressure_initial;        // initial pore pressure
    //double phi_i = 1.0 - exp(-d_cm.p3_crush_curve);  // Initial porosity (inferred from crush curve)
    //double C1 = Kf*(1.0 - phi_i) + Km*(phi_i);       // Term to simplify the expression below

    pf = d_cm.fluid_pressure_initial +
         Kf*log(exp(ev*(-1.0 - Km/C1))*(-exp((ev*Kf)/C1)*(phi_i-1.0) + exp((ev*Km)/C1)*phi_i));
  }
  return pf;
} //===================================================================

// Compute nonhardening return from trial stress to some yield surface
int Arenisca3::nonHardeningReturn(const double & I1_trial,    // Trial Stress
                                  const double & rJ2_trial,
                                  const Matrix3& S_trial,
                                  const double & I1_old,      // Stress at start of subtep
                                  const double & rJ2_old,
                                  const Matrix3& S_old,
                                  const Matrix3& d_e,         // increment in total strain
                                  const double & X,           // cap position
                                  const double & Zeta,        // isotropic bacstress
                                  const double & coher,
                                  const double & bulk,        // elastic bulk modulus
                                  const double & shear,       // elastic shear modulus
                                  double & I1_new,      // New stress state on yield surface
                                  double & rJ2_new,
                                  Matrix3& S_new,
                                  Matrix3& d_ep_new)    // increment in plastic strain for return
{
  // Computes a non-hardening return to the yield surface in the meridional profile
  // (constant Lode angle) based on the current values of the internal state variables
  // and elastic properties.  Returns the updated stress and  the increment in plastic
  // strain corresponding to this return.
  //
  // NOTE: all values of r and z in this function are transformed!

  const int nmax = 19;  // If this is changed, more entries may need to be added to sinV cosV.
  int n = 0,
	  returnFlag = 0,   // error flag = 0 for successful return.
      interior;

// (1) Define an interior point, (I1_0 = Zeta, also, J2_0 = 0 but no need to  create this variable.)
  double  I1_0,
		  I1trialMinusZeta = I1_trial-Zeta;
				

// It may be better to use an interior point at the center of the yield surface, rather than at zeta, in particular
// when PEAKI1=0.  Picking the midpoint between PEAKI1 and X would be problematic when the user has specified
// some no porosity condition (e.g. p0=-1e99)
  if( I1trialMinusZeta>= coher*d_cm.PEAKI1 ){ // Trial is past vertex
	  double lTrial = sqrt(I1trialMinusZeta*I1trialMinusZeta + rJ2_trial*rJ2_trial),
			 lYield = 0.5*(coher*d_cm.PEAKI1 - X);
	  I1_0 = Zeta + coher*d_cm.PEAKI1 - min(lTrial,lYield);
  }
  else if( (I1trialMinusZeta < coher*d_cm.PEAKI1)&&(I1trialMinusZeta > X) ){ // Trial is above yield surface
	  I1_0 = I1_trial;
  }
  else if( I1trialMinusZeta <= X ){ // Trial is past X, use yield midpoint as interior point
	  I1_0 = Zeta + 0.5*(coher*d_cm.PEAKI1 + X);
  }
  else { // Shouldn't get here
	  I1_0 = Zeta;
  }

// (2) Transform the trial and interior points as follows where beta defines the degree
//  of non-associativity.
  // multiplier to compute Lode R to sqrt(J2)
  double rJ2_to_r = sqrt_two*d_cm.BETA_nonassociativity*sqrt(1.5*bulk/shear);  
  // multiplier to compute sqrt(J2) to Lode R
  double r_to_rJ2 = 1.0/rJ2_to_r;                        
  double r_trial = rJ2_to_r*rJ2_trial,
         z_trial = I1_trial*one_sqrt_three,
         z_test,
         r_test,
         r_0     = 0.0,
         z_0     = I1_0*one_sqrt_three;

  // Lookup tables for computing the sin() and cos() of th rotation angle.
  double sinV[]={0.7071067811865475,-0.5,0.3420201433256687,-0.2306158707424402,0.1545187928078405,
                  -0.1032426220806015,0.06889665647555759,-0.04595133277786571,0.03064021661344469,
                  -0.02042858745187096,0.01361958465478159,-0.009079879062402308,0.006053298918749807,
                  -0.004035546304539714,0.002690368259933135,-0.001793580042002626,0.001195720384163988,
                  -0.0007971470283055577,0.0005314313834717263,-0.00035428759824575,0.0002361917349088998};
  double cosV[]={0.7071067811865475,0.8660254037844386,0.9396926207859084,0.9730448705798238,
                  0.987989849476809,0.9946562024066014,0.9976238022052647,0.9989436796015769,
                  0.9995304783376449,0.9997913146325693,0.999907249155556,0.9999587770484402,
                  0.9999816786182636,0.999991857149859,0.9999963809527642,0.9999983915340229,
                  0.9999992851261259,0.9999996822782572,0.9999998587903324,0.9999999372401469,
                  0.9999999721067318};
  double sinTheta = sinV[0],
         cosTheta = cosV[0];
  
  // Compute the a1,a2,a3,a4 parameters from FSLOPE,YSLOPE,STREN and PEAKI1,
  // which are perturbed by variability according to coher.  These are then 
  // passed down to the computeYieldFunction, to avoid the expense of computing a3
  double limitParameters[4];  //double a1,a2,a3,a4;
  computeLimitParameters(limitParameters,coher);

// (3) Perform Bisection between in transformed space, to find the new point on the
//  yield surface: [znew,rnew] = transformedBisection(z0,r0,z_trial,r_trial,X,Zeta,K,G)
  //int icount=1;
  
  // It may be getting stuck in this loop, perhaps bouncing back and forth so interior = 1, 
  // with with n<=2 iterations, so n never gets to nmax.
  int k = 0;
  while ( (n < nmax)&&(k < 10*nmax) ){
    // transformed bisection to find a new interior point, just inside the boundary of the
    // yield surface.  This function overwrites the inputs for z_0 and r_0
    //  [z_0,r_0] = transformedBisection(z_0,r_0,z_trial,r_trial,X_Zeta,bulk,shear)
    transformedBisection(z_0,r_0,z_trial,r_trial,X,Zeta,coher,limitParameters,r_to_rJ2);

// (4) Perform a rotation of {z_new,r_new} about {z_trial,r_trial} until a new interior point
// is found, set this as {z0,r0}
    interior = 0;
    n = max(n-4,0);  //
    // (5) Test for convergence:
    while ( (interior==0)&&(n < nmax) ){
		k++;
      // To avoid the cost of computing pow() to get theta, and then sin(), cos(),
      // we use a lookup table defined above by sinV and cosV.
      //
      // theta = pi_fourth*Pow(-two_third,n);
      // z_test = z_trial + cos(theta)*(z_0-z_trial) - sin(theta)*(r_0-r_trial);
      // r_test = r_trial + sin(theta)*(z_0-z_trial) + cos(theta)*(r_0-r_trial);
      sinTheta = sinV[n];
      cosTheta = cosV[n];
      z_test = z_trial + cosTheta*(z_0-z_trial) - sinTheta*(r_0-r_trial);
      r_test = r_trial + sinTheta*(z_0-z_trial) + cosTheta*(r_0-r_trial);

      if ( transformedYieldFunction(z_test,r_test,X,Zeta,coher,limitParameters,r_to_rJ2) == -1 ) { // new interior point
        interior = 1;
        z_0 = z_test;
        r_0 = r_test;
      }
      else { n++; }
    }
  }

  if (k>=10*nmax){
	  returnFlag = 1;
#ifdef MHdebug
	  cout<<"k >= 10*nmax, nonHardening return failed."<<endl;
#endif
  }
  
// (6) Solution Converged, Compute Untransformed Updated Stress:
  I1_new = sqrt_three*z_0;
  rJ2_new = r_to_rJ2*r_0;

  if ( rJ2_trial!=0.0 ){S_new = S_trial*rJ2_new/rJ2_trial;}
  else                 {S_new = S_trial;}

  Matrix3 sigma_new = I1_new*one_third*Identity + S_new,
          sigma_old = I1_old*one_third*Identity + S_old;
  Matrix3 d_sigma = sigma_new - sigma_old;

// (7) Compute increment in plastic strain for return:
//  d_ep0 = d_e - [C]^-1:(sigma_new-sigma_old)
  Matrix3 d_ee    = 0.5*d_sigma/shear + (one_ninth/bulk - one_sixth/shear)*d_sigma.Trace()*Identity;
  d_ep_new        = d_e - d_ee;

  return returnFlag;
} //===================================================================

// Computes bisection between two points in transformed space
void Arenisca3::transformedBisection(double& z_0,
                                     double& r_0,
                                     const double& z_trial,
                                     const double& r_trial,
                                     const double& X,
                                     const double& Zeta,
									 const double& coher,
                                     const double limitParameters[4],
                                     const double& r_to_rJ2
                                    )
{
// Computes a bisection in transformed stress space between point sigma_0 (interior to the
// yield surface) and sigma_trial (exterior to the yield surface).  Returns this new point,
// which will be just outside the yield surface, overwriting the input arguments for
// z_0 and r_0.

// After the first iteration of the nonhardening return, the subseqent bisections will likely
// converge with eta << 1.  It may be faster to put in some logic to try to start bisection
// with tighter bounds, and only expand them to 0<eta<1 if the first eta mid is too large.


// (1) initialize bisection
  double eta_out = 1.0,  // This is for the accerator.  Must be > TOL
         eta_in  = 0.0,
         eta_mid,
         TOL = 1.0e-6,
         r_test,
         z_test;

// (2) Test for convergence
  while (eta_out-eta_in > TOL){

// (3) Transformed test point
    eta_mid = 0.5*(eta_out+eta_in);
    z_test = z_0 + eta_mid*(z_trial-z_0);
    r_test = r_0 + eta_mid*(r_trial-r_0);
// (4) Check if test point is within the yield surface:
    if ( transformedYieldFunction(z_test,r_test,X,Zeta,coher,limitParameters,r_to_rJ2)!=1 ) {eta_in = eta_mid;}
    else {eta_out = eta_mid;}
  }
// (5) Converged, return {z_new,r_new}={z_test,r_test}
  z_0 = z_0 + eta_out*(z_trial-z_0); //z_0 = z_test;
  r_0 = r_0 + eta_out*(r_trial-r_0); //r_0 = r_test;

} //===================================================================

// computeTransformedYieldFunction from transformed inputs
int Arenisca3::transformedYieldFunction(const double& z,
                                        const double& r,
                                        const double& X,
                                        const double& Zeta,
										const double& coher,
                                        const double limitParameters[4],
                                        const double& r_to_rJ2
                                       )
{
// Evaluate the yield criteria and return:
//  -1: elastic
//   0: on yield surface within tolerance
//   1: plastic

// Untransformed values:
  double I1  = sqrt_three*z,
		 rJ2 = r_to_rJ2*r;
  int    YIELD = computeYieldFunction(I1,rJ2,X,Zeta,coher,limitParameters);
  return YIELD;
} //===================================================================

// computeYieldFunction from untransformed inputs
int Arenisca3::computeYieldFunction(const double& I1,
                                    const double& rJ2,
                                    const double& X,
                                    const double& Zeta,
									const double& coher,
									const double limitParameters[4]
                                   )
{
  // Evaluate the yield criteria and return:
  //  -1: elastic
  //   0: on yield surface within tolerance (not used)
  //   1: plastic
  //
  //                        *** Developer Note ***
  // THIS FUNCTION IS DEEP WITHIN A NESTED LOOP AND IS CALLED THOUSANDS
  // OF TIMES PER TIMESTEP.  EVERYTHING IN THIS FUNCTION SHOULD BE
  // OPTIMIZED FOR SPEED.
  //
  int YIELD = -1;
  double I1mZ = I1 - Zeta;    // Shifted stress to evalue yield criteria

// --------------------------------------------------------------------
// *** SHEAR LIMIT FUNCTION (Ff) ***
// --------------------------------------------------------------------
  // Read input parameters to specify strength model
  double  Ff,
		  a1 = limitParameters[0],
		  a2 = limitParameters[1],
		  a3 = limitParameters[2],
		  a4 = limitParameters[3];
  
#ifdef MHfastfcns
  Ff = a1 - a3*fasterexp(a2*I1mZ) - a4*I1mZ;
#else
  Ff = a1 - a3*exp(a2*I1mZ) - a4*I1mZ;
#endif

// --------------------------------------------------------------------
// *** Branch Point (Kappa) ***
// --------------------------------------------------------------------
  double  CR  = d_cm.CR,
          PEAKI1 = coher*d_cm.PEAKI1;    // Perturbed point for variability
  double  Kappa  = PEAKI1-CR*(PEAKI1-X); // Branch Point

// --------------------------------------------------------------------
// *** COMPOSITE YIELD FUNCTION ***
// --------------------------------------------------------------------
  // Evaluate Composite Yield Function F(I1) = Ff(I1)*fc(I1) in each region.
  // The elseif statements have nested if statements, which is not equivalent
  // to them having a single elseif(A&&B&&C)
  if( I1mZ<X ){//---------------------------------------------------(I1<X)
    YIELD = 1;
  }
  else if(( I1mZ < Kappa )&&( I1mZ >= X )) {// ---------------(X<I1<kappa)
    // p3 is the maximum achievable volumetric plastic strain in compresson
    // so if a value of 0 has been specified this indicates the user
    // wishes to run without porosity, and no cap function is used, i.e. fc=1

    // **Elliptical Cap Function: (fc)**
    // fc = sqrt(1.0 - Pow((Kappa-I1mZ)/(Kappa-X)),2.0);
    // faster version: fc2 = fc^2
    double fc2 = 1.0 - ((Kappa-I1mZ)/(Kappa-X))*((Kappa-I1mZ)/(Kappa-X));
    if(rJ2*rJ2 > Ff*Ff*fc2 ) YIELD = 1;
  }
  else if(( I1mZ <= PEAKI1 )&&( I1mZ >= Kappa )){// -----(kappa<I1<PEAKI1)
    if(rJ2 > Ff) YIELD = 1;
  }
  else if( I1mZ > PEAKI1 ) {// --------------------------------(peakI1<I1)
    YIELD = 1;
  };

  return YIELD;
} //===================================================================

// Compute (dZeta/devp) Zeta and vol. plastic strain
double Arenisca3::computedZetadevp(double Zeta, double evp)
{
  // Computes the partial derivative of the trace of the
  // isotropic backstress (Zeta) with respect to volumetric
  // plastic strain (evp).
  double dZetadevp = 0.0;           // Evolution rate of isotropic backstress

  if (evp <= ev0 && Kf != 0.0) { // .................................... Fluid effects are active
    double pfi = d_cm.fluid_pressure_initial; // initial fluid pressure

    // This is an expensive calculation, but fasterexp() seemed to cause errors.
    dZetadevp = (3.0*exp(evp)*Kf*Km)/(exp(evp)*(Kf + Km)
                                      + exp(Zeta/(3.0*Km))*Km*(-1.0 + phi_i)
                                      - exp((3.0*pfi + Zeta)/(3.0*Kf))*Kf*phi_i);
  }
  return dZetadevp;
  //
} //===================================================================

// Compute (dZeta/devp) Zeta and vol. plastic strain
void Arenisca3::computeLimitParameters(double limitParameters[4], const double& coher)
{ // The shear limit surface is defined in terms of the a1,a2,a3,a4 parameters, but
  // the user inputs are the more intuitive set of FSLOPE. YSLOPE, STREN, and PEAKI1.

  // This routine computes the a_i parameters from the user inputs.  The code was
  // originally written by R.M. Brannon, with modifications by M.S. Swan.
  double  FSLOPE = d_cm.FSLOPE,       // Slope at I1=PEAKI1
          STREN  = d_cm.STREN,        // Value of rootJ2 at I1=0
          YSLOPE = d_cm.YSLOPE,       // High pressure slope
          PEAKI1 = coher*d_cm.PEAKI1; // Value of I1 at strength=0 (Perturbed by variability)
  
  double a1,a2,a3,a4;

  if (FSLOPE > 0.0 && PEAKI1 >= 0.0 && STREN == 0.0 && YSLOPE == 0.0)
  {// ----------------------------------------------Linear Drucker Prager
    a1 = PEAKI1*FSLOPE;
    a2 = 0.0;
    a3 = 0.0;
    a4 = FSLOPE;
  }
  else if (FSLOPE == 0.0 && PEAKI1 == 0.0 && STREN > 0.0 && YSLOPE == 0.0)
  { // ------------------------------------------------------- Von Mises
    a1 = STREN;
    a2 = 0.0;
    a3 = 0.0;
    a4 = 0.0;
  }
  else if (FSLOPE > 0.0 && YSLOPE == 0.0 && STREN > 0.0 && PEAKI1 == 0.0)
  { // ------------------------------------------------------- 0 PEAKI1 to vonMises
    a1 = STREN;
    a2 = FSLOPE/STREN;
    a3 = STREN;
    a4 = 0.0;
  }
  else if (FSLOPE > YSLOPE && YSLOPE > 0.0 && STREN > YSLOPE*PEAKI1 && PEAKI1 >= 0.0)
  { // ------------------------------------------------------- Nonlinear Drucker-Prager
    a1 = STREN;
    a2 = (FSLOPE-YSLOPE)/(STREN-YSLOPE*PEAKI1);
#ifdef MHfastfcns
    a3 = (STREN-YSLOPE*PEAKI1)*fasterexp(-a2*PEAKI1);
#else
    a3 = (STREN-YSLOPE*PEAKI1)*exp(-a2*PEAKI1);
#endif
    a4 = YSLOPE;
  }
  else
  {
    // Bad inputs, call exception:
    ostringstream warn;
    warn << "Bad input parameters for shear limit surface. FSLOPE = "<<FSLOPE<<
    ", YSLOPE = "<<YSLOPE<<", PEAKI1 = "<<PEAKI1<<", STREN = "<<STREN<<endl;
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
  limitParameters[0] = a1;
  limitParameters[1] = a2;
  limitParameters[2] = a3;
  limitParameters[3] = a4;
} //===================================================================

void Arenisca3::checkInputParameters(){
	
  if(d_cm.PEAKI1<0.0){
	ostringstream warn;
    warn << "PEAKI1 must be nonnegative. PEAKI1 = "<<d_cm.PEAKI1<<endl;
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
  if(d_cm.FSLOPE<0.0){
	  ostringstream warn;
	  warn << "FSLOPE must be nonnegative. FSLOPE = "<<d_cm.FSLOPE<<endl;
	  throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
  if(d_cm.FSLOPE<d_cm.YSLOPE){
	  ostringstream warn;
	  warn << "FSLOPE must be greater than YSLOPE. FSLOPE = "<<d_cm.FSLOPE<<", YSLOPE = "<<d_cm.YSLOPE<<endl;
	  throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
  if(d_cm.BETA_nonassociativity <= 0.0){
	ostringstream warn;
    warn << "BETA_nonassociativity must be positive. BETA_nonassociativity = "<<d_cm.BETA_nonassociativity<<endl;
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
  if(d_cm.B0<=0.0){
	ostringstream warn;
    warn << "B0 must be positive. B0 = "<<d_cm.B0<<endl;
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
  if(d_cm.B1<0.0){
	  ostringstream warn;
	  warn << "B1 must be nonnegative. B1 = "<<d_cm.B1<<endl;
	  throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
  if(d_cm.B2<0.0){
	  ostringstream warn;
	  warn << "B2 must be nonnegative. B2 = "<<d_cm.B2<<endl;
	  throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
  if(d_cm.G0<=0.0){
	  ostringstream warn;
	  warn << "G0 must be positive. G0 = "<<d_cm.G0<<endl;
	  throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
  if(d_cm.p0_crush_curve>=0.0){
	  ostringstream warn;
	  warn << "p0 must be negative. p0 = "<<d_cm.p0_crush_curve<<endl;
	  throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
  if(d_cm.p1_crush_curve<=0.0){
	  ostringstream warn;
	  warn << "p1 must be positive. p1 = "<<d_cm.p1_crush_curve<<endl;
	  throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
  if(d_cm.p3_crush_curve<0.0){
	  ostringstream warn;
	  warn << "p3 must be positive. p3 = "<<d_cm.p3_crush_curve<<endl;
	  throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
  if(d_cm.CR>=1||d_cm.CR<=0.0){
	  ostringstream warn;
	  warn << "CR must be 0<CR<1. CR = "<<d_cm.CR<<endl;
	  throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
  if(d_cm.fluid_B0<0.0){
	  ostringstream warn;
	  warn << "fluid_b0 must be >=0. fluid_b0 = "<<d_cm.fluid_B0<<endl;
	  throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
  if(d_cm.fluid_pressure_initial<0.0){
	  ostringstream warn;
	  warn << "Negative pfi not supported. fluid_pressure_initial = "<<d_cm.fluid_pressure_initial<<endl;
	  throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
  if(d_cm.fluid_B0<0.0&&(d_cm.B0==0.0||d_cm.B1==0.0)){
	  ostringstream warn;
	  warn << "B0 and B1 must be positive to use fluid model."<<endl;
	  throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
  if(d_cm.T1_rate_dependence<0.0){
	  ostringstream warn;
	  warn << "T1 must be nonnegative. T1 = "<<d_cm.T1_rate_dependence<<endl;
	  throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
  if(d_cm.T2_rate_dependence<0.0){
	  ostringstream warn;
	  warn << "T2 must be nonnegative. T2 = "<<d_cm.T2_rate_dependence<<endl;
	  throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
  if( (d_cm.T1_rate_dependence>0.0||d_cm.T2_rate_dependence>0.0)
    !=(d_cm.T1_rate_dependence>0.0&&d_cm.T2_rate_dependence>0.0)  ){
	  ostringstream warn;
	  warn << "For rate dependence both T1 and T2 must be positive. T1 = "<<d_cm.T1_rate_dependence<<", T2 = "<<d_cm.T2_rate_dependence<<endl;
	  throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
  if(d_cm.subcycling_characteristic_number<1){
	  ostringstream warn;
	  warn << "subcycling characteristic number should be > 1. Default = 256"<<endl;
	  throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
  if(d_cm.Use_Disaggregation_Algorithm&&d_cm.fluid_B0!=0.0){
	  ostringstream warn;
	  warn << "Disaggregation algorithm not supported with fluid model"<<endl;
	  throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
  if(d_cm.Use_Disaggregation_Algorithm&&d_cm.PEAKI1!=0.0){
	  ostringstream warn;
	  warn << "Disaggregation algorithm not supported with PEAKI1 > 0.0"<<endl;
	  throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
}

// ****************************************************************************************************
// ****************************************************************************************************
// ************** PUBLIC Uintah MPM constitutive model specific functions *****************************
// ****************************************************************************************************
// ****************************************************************************************************

void Arenisca3::carryForward(const PatchSubset* patches,
                             const MPMMaterial* matl,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw)
{
  // Carry forward the data.
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    int dwi = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

    // Carry forward the data common to all constitutive models
    // when using RigidMPM.
    // This method is defined in the ConstitutiveModel base class.
    carryForwardSharedData(pset, old_dw, new_dw, matl);

    // Carry forward the data local to this constitutive model
    new_dw->put(delt_vartype(1.e10), lb->delTLabel, patch->getLevel());

    if (flag->d_reductionVars->accStrainEnergy ||
        flag->d_reductionVars->strainEnergy) {
      new_dw->put(sum_vartype(0.0),     lb->StrainEnergyLabel);
    }
  }
}

//When a particle is pushed from patch to patch, carry information needed for the particle
void Arenisca3::addParticleState(std::vector<const VarLabel*>& from,
                                 std::vector<const VarLabel*>& to)
{
  // Push back all the particle variables associated with Arenisca.
  // Important to keep from and to lists in same order!
  from.push_back(peakI1IDistLabel);  // For variability
  from.push_back(pAreniscaFlagLabel);
  from.push_back(pScratchDouble1Label);
  from.push_back(pScratchDouble2Label);
  from.push_back(pPorePressureLabel);
  from.push_back(pepLabel);
  from.push_back(pevpLabel);
  from.push_back(peveLabel);
  from.push_back(pCapXLabel);
  from.push_back(pZetaLabel);
  from.push_back(pP3Label);
  from.push_back(pStressQSLabel);
  from.push_back(pScratchMatrixLabel);
  to.push_back(  peakI1IDistLabel_preReloc);  // For variability
  to.push_back(  pAreniscaFlagLabel_preReloc);
  to.push_back(  pScratchDouble1Label_preReloc);
  to.push_back(  pScratchDouble2Label_preReloc);
  to.push_back(  pPorePressureLabel_preReloc);
  to.push_back(  pepLabel_preReloc);
  to.push_back(  pevpLabel_preReloc);
  to.push_back(  peveLabel_preReloc);
  to.push_back(  pCapXLabel_preReloc);
  to.push_back(  pZetaLabel_preReloc);
  to.push_back(  pP3Label_preReloc);
  to.push_back(  pStressQSLabel_preReloc);
  to.push_back(  pScratchMatrixLabel_preReloc);
}

void Arenisca3::addInitialComputesAndRequires(Task* task,
    const MPMMaterial* matl,
    const PatchSet* patch) const
{
  // Add the computes and requires that are common to all explicit
  // constitutive models.  The method is defined in the ConstitutiveModel
  // base class.
  const MaterialSubset* matlset = matl->thisMaterial();

  // Other constitutive model and input dependent computes and requires
  task->computes(peakI1IDistLabel,     matlset);  // For variability
  task->computes(pAreniscaFlagLabel,   matlset);
  task->computes(pScratchDouble1Label, matlset);
  task->computes(pScratchDouble2Label, matlset);
  task->computes(pPorePressureLabel,   matlset);
  task->computes(pepLabel,             matlset);
  task->computes(pevpLabel,            matlset);
  task->computes(peveLabel,            matlset);
  task->computes(pCapXLabel,           matlset);
  task->computes(pZetaLabel,           matlset);
  task->computes(pP3Label,             matlset);
  task->computes(pStressQSLabel,       matlset);
  task->computes(pScratchMatrixLabel,  matlset);
}

void Arenisca3::addComputesAndRequires(Task* task,
                                       const MPMMaterial* matl,
                                       const PatchSet* patches ) const
{
  // Add the computes and requires that are common to all explicit
  // constitutive models.  The method is defined in the ConstitutiveModel
  // base class.
  const MaterialSubset* matlset = matl->thisMaterial();
  addSharedCRForHypoExplicit(task, matlset, patches);
  task->requires(Task::OldDW, peakI1IDistLabel,       matlset, Ghost::None);  // For variability
  task->requires(Task::OldDW, lb->pLocalizedMPMLabel, matlset, Ghost::None);
  task->requires(Task::OldDW, pAreniscaFlagLabel,     matlset, Ghost::None);
  task->requires(Task::OldDW, pScratchDouble1Label,   matlset, Ghost::None);
  task->requires(Task::OldDW, pScratchDouble2Label,   matlset, Ghost::None);
  task->requires(Task::OldDW, pPorePressureLabel,     matlset, Ghost::None);
  task->requires(Task::OldDW, pepLabel,               matlset, Ghost::None);
  task->requires(Task::OldDW, pevpLabel,              matlset, Ghost::None);
  task->requires(Task::OldDW, peveLabel,              matlset, Ghost::None);
  task->requires(Task::OldDW, pCapXLabel,             matlset, Ghost::None);
  task->requires(Task::OldDW, pZetaLabel,             matlset, Ghost::None);
  task->requires(Task::OldDW, pP3Label,               matlset, Ghost::None);
  task->requires(Task::OldDW, pStressQSLabel,         matlset, Ghost::None);
  task->requires(Task::OldDW, pScratchMatrixLabel,    matlset, Ghost::None);
  task->requires(Task::OldDW, lb->pParticleIDLabel,   matlset, Ghost::None);
  task->computes(peakI1IDistLabel_preReloc,         matlset);  // For variability
  task->computes(lb->pLocalizedMPMLabel_preReloc,   matlset);
  task->computes(pAreniscaFlagLabel_preReloc,       matlset);
  task->computes(pScratchDouble1Label_preReloc,     matlset);
  task->computes(pScratchDouble2Label_preReloc,     matlset);
  task->computes(pPorePressureLabel_preReloc,       matlset);
  task->computes(pepLabel_preReloc,                 matlset);
  task->computes(pevpLabel_preReloc,                matlset);
  task->computes(peveLabel_preReloc,                matlset);
  task->computes(pCapXLabel_preReloc,               matlset);
  task->computes(pZetaLabel_preReloc,               matlset);
  task->computes(pP3Label_preReloc,                 matlset);
  task->computes(pStressQSLabel_preReloc,           matlset);
  task->computes(pScratchMatrixLabel_preReloc,      matlset);
}

//T2D: Throw exception that this is not supported
void Arenisca3::addComputesAndRequires(Task* ,
                                       const MPMMaterial* ,
                                       const PatchSet* ,
                                       const bool ) const
{
  cout << "NO VERSION OF addComputesAndRequires EXISTS YET FOR Arenisca3"<<endl;
}

//T2D: Throw exception that this is not supported
double Arenisca3::computeRhoMicroCM(double pressure,
                                    const double p_ref,
                                    const MPMMaterial* matl,
                                    double temperature,
                                    double rho_guess)
{
  double rho_orig = matl->getInitialDensity();
  double p_gauge = pressure - p_ref;
  double rho_cur;
  double bulk = d_cm.B0;

  rho_cur = rho_orig/(1.0-p_gauge/bulk);

  cout << "NO VERSION OF computeRhoMicroCM EXISTS YET FOR Arenisca3"<<endl;
  return rho_cur;
}

//T2D: Throw exception that this is not supported
void Arenisca3::computePressEOSCM(double rho_cur,double& pressure,
                                  double p_ref,
                                  double& dp_drho, double& tmp,
                                  const MPMMaterial* matl,
                                  double temperature)
{
  double bulk = d_cm.B0;
  double shear = d_cm.G0;
  double rho_orig = matl->getInitialDensity();

  double p_g = 0.5*bulk*(rho_cur/rho_orig - rho_orig/rho_cur);
  pressure = p_ref + p_g;
  dp_drho  = 0.5*bulk*(rho_orig/(rho_cur*rho_cur) + 1./rho_orig);
  tmp = (bulk + 4.0*shear/3.0)/rho_cur;  // speed of sound squared

  cout << "NO VERSION OF computePressEOSCM EXISTS YET FOR Arenisca3" << endl;
}

//T2D: Throw exception that this is not supported
double Arenisca3::getCompressibility()
{
  cout << "NO VERSION OF computePressEOSCM EXISTS YET FOR Arenisca3"
       << endl;
  return 1.0;
}

// Initialize all labels of the particle variables associated with Arenisca3.
void Arenisca3::initializeLocalMPMLabels()
{
  //peakI1Dist for variability
  peakI1IDistLabel = VarLabel::create("p.peakI1IDist",
                                      ParticleVariable<double>::getTypeDescription());
  peakI1IDistLabel_preReloc = VarLabel::create("p.peakI1IDist+",
                              ParticleVariable<double>::getTypeDescription());

  //pAreniscaFlag
  pAreniscaFlagLabel = VarLabel::create("p.AreniscaFlag",
                                        ParticleVariable<int>::getTypeDescription());
  pAreniscaFlagLabel_preReloc = VarLabel::create("p.AreniscaFlag+",
                                ParticleVariable<int>::getTypeDescription());
  //pScratchDouble1
  pScratchDouble1Label = VarLabel::create("p.ScratchDouble1",
                                          ParticleVariable<double>::getTypeDescription());
  pScratchDouble1Label_preReloc = VarLabel::create("p.ScratchDouble1+",
                                  ParticleVariable<double>::getTypeDescription());
  //pScratchDouble2
  pScratchDouble2Label = VarLabel::create("p.ScratchDouble2",
                                          ParticleVariable<double>::getTypeDescription());
  pScratchDouble2Label_preReloc = VarLabel::create("p.ScratchDouble2+",
                                  ParticleVariable<double>::getTypeDescription());
  //pPorePressure
  pPorePressureLabel = VarLabel::create("p.PorePressure",
                                        ParticleVariable<double>::getTypeDescription());
  pPorePressureLabel_preReloc = VarLabel::create("p.PorePressure+",
                                ParticleVariable<double>::getTypeDescription());
  //pep
  pepLabel = VarLabel::create("p.ep",
                              ParticleVariable<Matrix3>::getTypeDescription());
  pepLabel_preReloc = VarLabel::create("p.ep+",
                                       ParticleVariable<Matrix3>::getTypeDescription());
  //pevp
  pevpLabel = VarLabel::create("p.evp",
                               ParticleVariable<double>::getTypeDescription());
  pevpLabel_preReloc = VarLabel::create("p.evp+",
                                        ParticleVariable<double>::getTypeDescription());
  //peve
  peveLabel = VarLabel::create("p.eve",
                               ParticleVariable<double>::getTypeDescription());
  peveLabel_preReloc = VarLabel::create("p.eve+",
                                        ParticleVariable<double>::getTypeDescription());
  //pCapX
  pCapXLabel = VarLabel::create("p.CapX",
                                ParticleVariable<double>::getTypeDescription());
  pCapXLabel_preReloc = VarLabel::create("p.CapX+",
                                         ParticleVariable<double>::getTypeDescription());
  //pZeta
  pZetaLabel = VarLabel::create("p.Zeta",
                                ParticleVariable<double>::getTypeDescription());
  pZetaLabel_preReloc = VarLabel::create("p.Zeta+",
                                         ParticleVariable<double>::getTypeDescription());
  //pP3
  pP3Label = VarLabel::create("p.P3",
                                ParticleVariable<double>::getTypeDescription());
  pP3Label_preReloc = VarLabel::create("p.P3+",
                                         ParticleVariable<double>::getTypeDescription());
  //pStressQS
  pStressQSLabel = VarLabel::create("p.StressQS",
                                    ParticleVariable<Matrix3>::getTypeDescription());
  pStressQSLabel_preReloc = VarLabel::create("p.StressQS+",
                            ParticleVariable<Matrix3>::getTypeDescription());
  //pScratchMatrix
  pScratchMatrixLabel = VarLabel::create("p.ScratchMatrix",
                                         ParticleVariable<Matrix3>::getTypeDescription());
  pScratchMatrixLabel_preReloc = VarLabel::create("p.ScratchMatrix+",
                                 ParticleVariable<Matrix3>::getTypeDescription());
}

// For variability:
//
// Weibull input parser that accepts a structure of input
// parameters defined as:
//
// bool Perturb        'True' for perturbed parameter
// double WeibMed       Medain distrib. value OR const value
//                         depending on bool Perturb
// double WeibMod       Weibull modulus
// double WeibRefVol    Reference Volume
// int    WeibSeed      Seed for random number generator
// std::string WeibDist  String for Distribution
//
// the string 'WeibDist' accepts strings of the following form
// when a perturbed value is desired:
//
// --Distribution--|-Median-|-Modulus-|-Reference Vol -|- Seed -|
// "    weibull,      45e6,      4,        0.0001,          0"
//
// or simply a number if no perturbed value is desired.
void Arenisca3::WeibullParser(WeibParameters &iP)
{
  // Remove all unneeded characters
  // only remaining are alphanumeric '.' and ','
  for( int i = iP.WeibDist.length()-1; i >= 0; i--){
    iP.WeibDist[i] = tolower(iP.WeibDist[i]);
    if(!isalnum(iP.WeibDist[i]) &&
       iP.WeibDist[i] != '.' &&
       iP.WeibDist[i] != ',' &&
       iP.WeibDist[i] != '-' &&
       iP.WeibDist[i] != EOF) {
      iP.WeibDist.erase(i,1);
    }
  } // End for
  if(iP.WeibDist.substr(0,4) == "weib"){
    iP.Perturb = true;
  }
  else{
    iP.Perturb = false;
  }
  // ######
  // If perturbation is NOT desired
  // ######
  if( !iP.Perturb ){
    bool escape = false;
    int num_of_e = 0;
    int num_of_periods = 0;
    for( unsigned int i = 0; i < iP.WeibDist.length(); i++){
      if( iP.WeibDist[i] != '.'
          && iP.WeibDist[i] != 'e'
          && iP.WeibDist[i] != '-'
          && !isdigit(iP.WeibDist[i]))
        escape = true;
      if( iP.WeibDist[i] == 'e' )
        num_of_e += 1;
      if( iP.WeibDist[i] == '.' )
        num_of_periods += 1;
      if( num_of_e > 1 || num_of_periods > 1 || escape ){
        std::cerr << "\n\nERROR:\nInput value cannot be parsed. Please\n"
                     "check your input values.\n" << std::endl;
        exit (1);
      }
    } // end for(int i = 0;....)
    if( escape )
      exit (1);
    iP.WeibMed  = atof(iP.WeibDist.c_str());
  }
  // ######
  // If perturbation IS desired
  // ######
  if( iP.Perturb ){
    int weibValues[4];
    int weibValuesCounter = 0;
    for( unsigned int r = 0; r < iP.WeibDist.length(); r++){
      if( iP.WeibDist[r] == ',' ){
        weibValues[weibValuesCounter] = r;
        weibValuesCounter += 1;
      } // end if(iP.WeibDist[r] == ',')
    } // end for(int r = 0; ...... )
    if(weibValuesCounter != 4){
      std::cerr << "\n\nERROR:\nWeibull perturbed input string must contain\n"
                   "exactly 4 commas. Verify that your input string is\n"
                   "of the form 'weibull, 45e6, 4, 0.001, 1'.\n" << std::endl;
      exit (1);
    } // end if(weibValuesCounter != 4)
    std::string weibMedian;
    std::string weibModulus;
    std::string weibRefVol;
    std::string weibSeed;
    weibMedian  = iP.WeibDist.substr(weibValues[0]+1,weibValues[1]-weibValues[0]-1);
    weibModulus = iP.WeibDist.substr(weibValues[1]+1,weibValues[2]-weibValues[1]-1);
    weibRefVol  = iP.WeibDist.substr(weibValues[2]+1,weibValues[3]-weibValues[2]-1);
    weibSeed    = iP.WeibDist.substr(weibValues[3]+1);
    iP.WeibMed    = atof(weibMedian.c_str());
    iP.WeibMod    = atof(weibModulus.c_str());
    iP.WeibRefVol = atof(weibRefVol.c_str());
    iP.WeibSeed   = atoi(weibSeed.c_str());
  } // End if (iP.Perturb)
}
