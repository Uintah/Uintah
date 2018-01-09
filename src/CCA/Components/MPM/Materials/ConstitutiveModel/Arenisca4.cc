/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

  /* Arenisca4 INTRO

  This source code is for a simplified constitutive model, named ``Arenisca4'',
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
//#define MHfastfcns    // Use fast approximate exp(), log() and pow() in deep loops.
#define MHdisaggregationStiffness // reduce stiffness with disaggregation

// INCLUDE SECTION: tells the preprocessor to include the necessary files
#include <CCA/Components/MPM/Materials/ConstitutiveModel/Arenisca4.h>
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
#include <fstream>
#include <iostream>
#include <Core/Geometry/Vector.h>

#ifdef MHfastfcns
#include <CCA/Components/MPM/Materials/ConstitutiveModel/fastapproximatefunctions.h>
#endif

using std::cerr;
using namespace Uintah;
using namespace std;

// Requires the necessary input parameters CONSTRUCTORS
Arenisca4::Arenisca4(ProblemSpecP& ps, MPMFlags* Mflag)
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
  ps->getWithDefault("Use_Disaggregation_Algorithm",d_cm.Use_Disaggregation_Algorithm, false);          // CVD jet
  ps->getWithDefault("J3_type",d_cm.J3_type, 0);				// Lode angle dependence
  ps->getWithDefault("J3_psi",d_cm.J3_psi, 1.0);  				// Lode angle dependence
  ps->getWithDefault("principal_stress_cutoff",d_cm.principal_stress_cutoff, 1.0e99); 	// Principal stress cutoff
  ps->get("PEAKI1IDIST",wdist.WeibDist);        // Variability
  WeibullParser(wdist);                         // Variability
  proc0cout <<"WeibMed="<<wdist.WeibMed<<endl;  // Variability

  // These class variables are computed from input parameters and are used throughout the code
  // The are evaluates here to avoid repeated computation, or to simplify expressions.

  // This phi_i value is not modified by the disaggregation strain, because
  // it is the same for all particles.  Thus disaggregation strain is
  // not supported when there is a pore fluid.
  phi_i = 1.0 - exp(-d_cm.p3_crush_curve); // initial porosity (inferred from crush curve, used for fluid model/

  Km = d_cm.B0 + d_cm.B1;                       // Matrix bulk modulus
  Kf = d_cm.fluid_B0;                           // Fluid bulk modulus
  C1 = Kf*(1.0 - phi_i) + Km*(phi_i);           // Term to simplify the fluid model expressions
  ev0 = C1*d_cm.fluid_pressure_initial/(Kf*Km); // Zero fluid pressure vol. strain.  (will equal zero if pfi=0)

  initializeLocalMPMLabels();
}

// DESTRUCTOR
Arenisca4::~Arenisca4()
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
void Arenisca4::outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag)
{
  ProblemSpecP cm_ps = ps;
  if (output_cm_tag) {
    cm_ps = ps->appendChild("constitutive_model");
    cm_ps->setAttribute("type","Arenisca4");
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
  cm_ps->appendElement("G1",d_cm.G1);  // Not used
  cm_ps->appendElement("G2",d_cm.G2);  // Not used
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
  cm_ps->appendElement("J3_type",d_cm.J3_type);  
  cm_ps->appendElement("J3_psi",d_cm.J3_psi);
  cm_ps->appendElement("principal_stress_cutoff",d_cm.principal_stress_cutoff); 

  //    Uintah Variability Variables
  cm_ps->appendElement("peakI1IPerturb", wdist.Perturb);
  cm_ps->appendElement("peakI1IMed", wdist.WeibMed);
  cm_ps->appendElement("peakI1IMod", wdist.WeibMod);
  cm_ps->appendElement("peakI1IRefVol", wdist.WeibRefVol);
  cm_ps->appendElement("peakI1ISeed", wdist.WeibSeed);
  cm_ps->appendElement("PEAKI1IDIST", wdist.WeibDist);
}

Arenisca4* Arenisca4::clone()
{
  return scinew Arenisca4(*this);
}

void Arenisca4::initializeCMData(const Patch* patch,
                                   const MPMMaterial* matl,
                                   DataWarehouse* new_dw)
{
  // Initialize the variables shared by all constitutive models
  // This method is defined in the ConstitutiveModel base class.
  /////
  // Allocates memory for internal state variables at beginning of run.

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
  ParticleVariable<int> pAreniscaFlag;
  ParticleVariable<double>  peakI1IDist;     // Holder for particles PEAKI1 value for variability
  ParticleVariable<double>  pScratchDouble1, // Developer tool
                   pScratchDouble2, // Developer tool
                   pPorePressure,   // Plottable fluid pressure
                   pevp,            // Plastic Volumetric Strain
                   peve,            // Elastic Volumetric Strain
                   pCapX,           // I1 of cap intercept
                   pZeta,           // Trace of isotropic Backstress
                   pP3;             // Modified p3 for initial disaggregation strain.
  ParticleVariable<Matrix3>         pScratchMatrix,  // Developer tool
                                    pep;             // Plastic Strain Tensor

  new_dw->allocateAndPut(pAreniscaFlag,   pAreniscaFlagLabel,   pset);
  new_dw->allocateAndPut(pScratchDouble1, pScratchDouble1Label, pset);
  new_dw->allocateAndPut(pScratchDouble2, pScratchDouble2Label, pset);
  new_dw->allocateAndPut(pPorePressure,   pPorePressureLabel,   pset);
  new_dw->allocateAndPut(peakI1IDist,     peakI1IDistLabel,     pset); // For variability
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
    peakI1IDist[*iter] = d_cm.PEAKI1;  // For variability
    pevp[*iter] = 0.0;
    peve[*iter] = 0.0;
    pZeta[*iter] = -3.0 * d_cm.fluid_pressure_initial;
    if(d_cm.Use_Disaggregation_Algorithm){
      pP3[*iter] = log(pVolume[*iter]*(matl->getInitialDensity())/pMass[*iter]);
    }
    else{
      pP3[*iter] = d_cm.p3_crush_curve;
    }
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
void Arenisca4::computeStableTimeStep(const Patch* patch,
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
Arenisca4::computeStressTensor is the core of the Arenisca4 model which computes
the updated stress at the end of the current timestep along with all other
required data such plastic strain, elastic strain, cap position, etc.

*/
void Arenisca4::computeStressTensor(const PatchSubset* patches,
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
        coher = peakI1IDist[idx]/d_cm.PEAKI1;
      } // Scalar-valued Damage (XXX)

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

      // Use polar decomposition to compute the rotation and stretch tensors
#ifdef MHdeleteBadF
      if(pDefGrad_new[idx].MaxAbsElem()>1.0e2){
		  pLocalized_new[idx]=-999;
		  cout<<"Large deformation gradient component: [F_new] = "<<pDefGrad_new[idx]<<endl;
		  cout<<"Resetting [F_new]=[I] for this step and deleting particle"<<endl;
		  Identity.polarDecompositionRMB(tensorU, tensorR);
      }
      else if(pDefGrad_new[idx].Determinant()<1.0e-3){
		  pLocalized_new[idx]=-999;
		  cout<<"Small deformation gradient determinant: [F_new] = "<<pDefGrad_new[idx]<<endl;
		  cout<<"Resetting [F_new]=[I] for this step and deleting particle"<<endl;
		  Identity.polarDecompositionRMB(tensorU, tensorR);
      }
	  else if(pDefGrad_new[idx].Determinant()>1.0e2){
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
             computeElasticProperties(bulk,shear); // High pressure bulk and shear moduli.
		 
#ifdef MHdisaggregationStiffness
	  // Compute the wave speed for the particle based on the reduced stiffness, which
	  // is computed when the value of P3 is sent to computeElasticProperties.
		if(d_cm.Use_Disaggregation_Algorithm){
	    computeElasticProperties(pStressQS_new[idx],pep_new[idx],pP3[idx],bulk,shear);
		}
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
int Arenisca4::computeStep(const Matrix3& D,       // strain "rate"
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
  double dt,                                        // substep time increment
         X_old = X_n,                                     // X at start of substep
         X_new,                                     // X at end of substep
         Zeta_old = Zeta_n,                                  // Zeta at start of substep
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
double bulk,
       shear;
computeElasticProperties(bulk,shear);

//Compute the trial stress: [sigma_trial] = computeTrialStress(sigma_old,d_e,K,G)
Matrix3 sigma_trial = computeTrialStress(sigma_old,D*Dt,bulk,shear);

double  I1_trial,
        J2_trial,
        rJ2_trial,
		J3_trial;
Matrix3 S_trial,
        d_e;
computeInvariants(sigma_trial,S_trial,I1_trial,J2_trial,rJ2_trial,J3_trial);

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
void Arenisca4::computeElasticProperties(double & bulk,
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
void Arenisca4::computeElasticProperties(const Matrix3 stress,
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
				double nu = d_cm.G1 + d_cm.G1*expb2byI1;
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
		double scale = max(fac,0.001);
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
Matrix3 Arenisca4::computeTrialStress(const Matrix3& sigma_old,  // old stress
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
int Arenisca4::computeStepDivisions(const double& X,
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
  int n_yield = ceil(1.0e-5*d_sigma.Norm()/size);

  // nsub is the maximum of the two values.above.  If this exceeds allowable,
  // throw warning and delete particle.
  int nsub = max(n_bulk,n_yield);

  if (nsub>d_cm.subcycling_characteristic_number){
#ifdef MHdebug
    cout<<"\nstepDivide out of range."<<endl;
	cout<<"d_sigma.Norm() = "<<d_sigma.Norm()<<endl;
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

void Arenisca4::computeInvariants(const Matrix3& A,
                                  Matrix3& S,
                                  double & I1,
                                  double & J2,
                                  double & rJ2,
								  double & J3)
{
  // Compute the first invariants
  I1 = A(0,0) + A(1,1) + A(2,2);  //Pa

  // Compute the deviatoric part of the tensor
  S = A - one_third*I1*Identity;  //Pa

  // Compute the second invariant
  J2 = 0.5*(S(0,0)*S(0,0) + S(1,1)*S(1,1) + S(2,2)*S(2,2))
	     + (S(0,1)*S(0,1) + S(1,2)*S(1,2) + S(2,0)*S(2,0));
			
  if(J2 < 1e-16*(I1*I1+J2)){
    J2=0.0;
  };
  rJ2 = sqrt(J2);
  
  // Compute third invariant
  J3 = S(0,0)*S(1,1)*S(2,2) + 2.0*S(0,1)*S(1,2)*S(2,0) 
	   - (S(0,0)*S(1,2)*S(1,2) + S(1,1)*S(2,0)*S(2,0) + S(2,2)*S(0,1)*S(0,1));
} //===================================================================

// Computes the updated stress state for a substep
int Arenisca4::computeSubstep(const Matrix3& d_e,       // Total strain increment for the substep
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
  int     returnFlag,
		  substepFlag;

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
         rJ2_trial,
		 J3_trial;
  computeInvariants(sigma_trial,S_trial,I1_trial,J2_trial,rJ2_trial,J3_trial);

// (4) Evaluate the yield function at the trial stress:
  // Compute the limit parameters based on the value of coher.  These are then passed down
  // to the computeYieldFunction, to avoid the expense of repeatedly computing a3
  double limitParameters[4];  //double a1,a2,a3,a4;
  computeLimitParameters(limitParameters,coher);
  
  int YIELD = computeYieldFunction(sigma_trial,X_old,Zeta_old,coher,limitParameters);
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
    double  TOL = 1e-4; // bisection convergence tolerance on eta (if changed, change imax)

    Matrix3 d_ep_0;     // increment in plastic strain for non-hardening return
    Matrix3 sigma_0;
	double evp_old = ep_old.Trace();
    
    // returnFlag would be != 0 if there was an error in the nonHardeningReturn call, but
    // there are currently no tests in that function that could detect such an error.
    returnFlag = nonHardeningReturn(sigma_trial,
                       sigma_old,
                       d_e,X_old,Zeta_old,coher,bulk,shear,
                       sigma_0,d_ep_0);
	if (returnFlag!=0){
#ifdef MHdebug
		cout << "1344: failed nonhardeningReturn in substep "<< endl;
#endif
		goto failedSubstep;
	}
	//cout<<"\n First nonhardeningReturn() call"<<endl;
	//cout<<"sigma_trial = "<<sigma_trial<<endl;
	//cout<<"sigma_old = "<<sigma_old<<endl;
	//cout<<"sigma_0 = "<<sigma_0<<endl;
	//cout<<"d_ep_0 = "<<d_ep_0<<endl;
	
    double d_evp_0 = d_ep_0.Trace();
	
	double I1_0,	// I1 at stress update for non-hardening return
           J2_0,	// J2 at stress update for non-hardening return
           rJ2_0,	// rJ2 at stress update for non-hardening return
		   J3_0;	// J3 at stress update for non-hardening return
	Matrix3 S_0;    // S (deviator) at stress update for non-hardening return
    computeInvariants(sigma_0,S_0,I1_0,J2_0,rJ2_0,J3_0);
	
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
	
    if( computeYieldFunction(sigma_trial,X_new,Zeta_new,coher,limitParameters)!=1 ){
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
    Matrix3 d_ep_new;
    returnFlag = nonHardeningReturn(sigma_trial,
                       sigma_old,
                       d_e,X_new,Zeta_new,coher,bulk,shear,
                       sigma_new,d_ep_new);
	if (returnFlag!=0){
#ifdef MHdebug
		cout << "1344: failed nonhardeningReturn in substep "<< endl;
#endif
		goto failedSubstep;
	}	
// (10) Check whether the isotropic component of the return has changed sign, as this
//      would indicate that the cap apex has moved past the trial stress, indicating
//      too much plastic strain in the return.
    if(Sign(I1_trial - sigma_new.Trace())!=Sign(I1_trial - I1_0)){
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
    double d_evp_new = d_ep_new.Trace();   // Increment in vol. plastic strain for return to new surface
    ep_new = ep_old + d_ep_new;

    // Check for convergence
    if( fabs(eta_out-eta_in) < TOL ){ // Solution is converged
      //sigma_new = one_third*I1_new*Identity + S_new;

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
// (12) Return updated values for successful/unsuccessful steps
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
double Arenisca4::computeX(const double& evp,const double& P3)
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
double Arenisca4::computePorePressure(const double ev)
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
int Arenisca4::nonHardeningReturn(const Matrix3 & sigma_trial, // Trial Stress (untransformed)
                                   const Matrix3 & sigma_old,   // Stress at start of subtep (untransformed)
                                   const Matrix3& d_e,          // increment in total strain
                                   const double & X,            // cap position
                                   const double & Zeta,         // isotropic backstress
                                   const double & coher,
                                   const double & bulk,         // elastic bulk modulus
                                   const double & shear,        // elastic shear modulus
                                   Matrix3 & sigma_new,         // New stress state on yield surface
                                   Matrix3 & d_ep_new)          // increment in plastic strain for return
{
  // Computes a non-hardening return to the yield surface in the meridional profile
  // (constant Lode angle) based on the current values of the internal state variables
  // and elastic properties.  Returns the updated stress and  the increment in plastic
  // strain corresponding to this return.
	
  int returnFlag = 0;
  
// (1) Transform sigma_trial and define interior point sigma_0.
  double S_to_S_star = d_cm.BETA_nonassociativity*sqrt(1.5*bulk/shear);
  double S_star_to_S = 1.0/S_to_S_star;
	
  Matrix3 iso_sigma_trial = one_third*sigma_trial.Trace()*Identity;
  Matrix3 S_trial = sigma_trial - iso_sigma_trial;
  Matrix3 sigma_trial_star = iso_sigma_trial + S_to_S_star*S_trial;
  
  
  double  I1_trial = sigma_trial.Trace(),
		  rJ2_trial = S_trial.Norm(),
		  I1_0,
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
  
  Matrix3 sigma_0 = one_third*I1_0*Identity;
   
// (2) Convert 3x3 stresses to vectors of ordered eigenvalues and ordered eigen projectors:
  
  Matrix3  PL_0,		// Eigenprojector for interior point (Low)
		   PM_0,		// Eigenprojector for interior point (Mid)
		   PH_0,		// Eigenprojector for interior point (High)
		   PL_trial,	// Eigenprojector for trial stress (Low)
		   PM_trial,	// Eigenprojector for trial stress (Mid)
		   PH_trial;	// Eigenprojector for trial stress (High)
  
  Vector lambda_trial, // Ordered eigenvalues {L,M,H} for trial stress
		 lambda_0,     // Ordered eigenvalues {L,M,H} for interior point
		 lambda_test;  // Ordered eigenvalues {L,M,H} for test point
   
  computeEigenProjectors(sigma_trial_star,	// Input tensor (transformed trial stress)
						 lambda_trial,	    // Ordered eigenvalues {L,M,H}
						 PL_trial,		    // Low eigenprojector
						 PM_trial,		    // Mid eigenprojector
						 PH_trial		    // High eigenprojector
						);
 
  computeEigenProjectors(sigma_0, 	// Input tensor (interior point) Isotropic so doesn't need transformation
						 lambda_0,	// Ordered eigenvalues {L,M,H}
						 PL_0,		// Low eigenprojector
						 PM_0,		// Mid eigenprojector
						 PH_0		// High eigenprojector
						);
  
// (3) Perform Bisection between in transformed space, to find the new point on the
//  yield surface: [znew,rnew] = transformedBisection(z0,r0,z_trial,r_trial,X,Zeta,K,G)
  //int icount=1;
  const int nmax = 40; // If this is changed, more entries may need to be added to sinTheta,cosTheta,sinPhi,cosPhi
  int n = 0,
	  interior;
  
  // Compute the a1,a2,a3,a4 parameters from FSLOPE,YSLOPE,STREN and PEAKI1,
  // which are perturbed by variability according to coher.  These are then 
  // passed down to the computeYieldFunction, to avoid the expense of computing a3
  double limitParameters[4];  //double a1,a2,a3,a4;
  computeLimitParameters(limitParameters,coher);
  
  //// Modify this to use lookup tables for computing the sin() and cos() of the rotation angle.
  
  double sinPhi[] = {1.,0.9689323775011424,0.8960189359268066,0.8040055020688935,0.7071067811865475,
				   0.6134632047267948,0.5272495422822986,0.4502100305139085,0.3826834323650898,
				   0.3242504404584508,0.274125434819962,0.231384216250307,0.1950903220161283,
				   0.1643604661519454,0.1383944613955442,0.1164850872534101,0.0980171403295606,
				   0.0824610718723793,0.06936430182358691,0.05834191978071341,0.04906767432741801,
				   0.04126568568335606,0.03470305368435329,0.02918338984984145,0.02454122852291229,
				   0.02063723796402109,0.01735414027907056,0.01459324891806791,0.01227153828571993,
				   0.01031916841618379,0.008677396837596723,0.007296818715934582,0.006135884649154475,
				   0.005159652888734864,0.004338739256632612,0.003648433640322288,0.003067956762965976,
				   0.002579835029490658,0.002169374733063605,0.001824219855463094,0.001533980186284766};
				   //,0.001289918587887071,0.001084688004625778,0.0009121103071445277,0.0007669903187427045,
				   //0.0006449594280862953,0.0005423440820746535,0.0004560552009988938,0.0003834951875713956,
				   //0.0003224797308109939,0.0002711720510075479,0.0002280276064277759,0.0001917475973107033,
				   //0.0001612398675014778,0.0001355860267500516,0.0001140138039549291,0.00009587379909597735,
				   //0.00008061993401273649,0.00006779301353081051,0.00005700690207009467,0.00004793689960306688,
				   //0.00004030996703911794,0.00003389650678487834,0.0000285034510466261,0.00002396844980841822,
				   //0.00002015498352365268,0.00001694825339487331,0.0000142517255247604,0.00001198422490506971,
				   //0.00001007749176233806,8.474126697740921e-6,7.125862762561117e-6,5.992112452642428e-6,
				   //5.038745881232993e-6,4.237063348908494e-6,3.562931381303173e-6,2.996056226334661e-6,
				   //2.519372940624492e-6,2.118531674459001e-6,1.781465690654414e-6,1.498028113169011e-6,
				   //1.259686470313245e-6,1.059265837230095e-6,8.907328453275602e-7,7.490140565847157e-7,
				   //6.298432351567476e-7,5.296329186151217e-7,4.453664226638242e-7,3.745070282923841e-7,
				   //3.149216175783894e-7,2.648164593075701e-7,2.226832113319176e-7,1.872535141461953e-7,
				   //1.574608087891967e-7,1.324082296537862e-7,1.113416056659595e-7,9.362675707309808e-8,
				   //7.873040439459857e-8,6.620411482689326e-8,5.567080283297984e-8,4.681337853654909e-8};
										
  double cosPhi[] = {0,0.2473257928926614,0.4440158403262132,0.5946218568493312,0.7071067811865475,
				   0.7897233037250013,0.8497104919695335,0.8929226889404623,0.9238795325112868,
				   0.9459712743326304,0.9616939461100744,0.9728624488951309,0.9807852804032304,
				   0.9864003432513166,0.9903771872650527,0.9931924407926016,0.9951847266721969,
				   0.9965942863701649,0.9975913961299618,0.9982966595137443,0.9987954562051724,
				   0.9991482088184327,0.9993976676303487,0.9995740741720306,0.9996988186962042,
				   0.9997870295263969,0.9998494055682457,0.9998935128732536,0.9999247018391445,
				   0.9999467559641355,0.9999623506833259,0.9999733778639443,0.9999811752826011,
				   0.9999866889024412,0.9999905876265351,0.9999933444438379,0.9999952938095762,
				   0.9999966722200732,0.9999976469038652,0.9999983361095752,0.9999988234517019};
				   //,0.9999991680546722,0.9999994117257933,0.9999995840273073,0.9999997058628822,
				   //0.9999997920136464,0.9999998529314375,0.9999998960068214,0.9999999264657179,
				   //0.9999999480034103,0.9999999632328587,0.999999974001705,0.9999999816164293,
				   //0.9999999870008525,0.9999999908082146,0.9999999935004262,0.9999999954041073,
				   //0.9999999967502131,0.9999999977020537,0.9999999983751066,0.9999999988510268,
				   //0.9999999991875533,0.9999999994255134,0.999999999593777,0.999999999712757,
				   //0.999999999796888,0.999999999856378,0.999999999898444,0.999999999928189,
				   //0.999999999949222,0.999999999964095,0.999999999974611,0.999999999982047,
				   //0.999999999987306,0.999999999991024,0.999999999993653,0.999999999995512,
				   //0.999999999996826,0.999999999997756,0.999999999998413,0.999999999998878,
				   //0.999999999999207,0.999999999999439,0.999999999999603,0.999999999999719,
				   //0.999999999999802,0.99999999999986,0.999999999999901,0.99999999999993,
				   //0.99999999999995,0.999999999999965,0.999999999999975,0.999999999999982,
				   //0.999999999999988,0.999999999999991,0.999999999999994,0.999999999999996,
				   //0.999999999999997,0.999999999999998,0.999999999999998,0.999999999999999};
  
double sinTheta[] = {0,0.6530800292138481,0.9891405116101687,0.8450502273278872,0.2907537996606676,
					 -0.4046809781025658,-0.9036746237763955,-0.9640045423646659,-0.5563852518516206,
					 0.1213157935885842,0.7401274591903676,0.9996648227097819,0.7739426852667083,
					 0.1725315841887478,-0.5126301786527758,-0.9489498706119317,-0.9246282254043701,
					 -0.4514715096385484,0.2408395011359971,0.8162416705885972,0.9954220097471839,
					 0.6914024035604609,0.05176071915901047,-0.6130067675893461,-0.9802071584778202,
					 -0.8715932306494945,-0.3398885962773765,0.3568055094310588,0.8802982985443536,
					 0.9764747478750539,0.5986486734679712,-0.06977475911309192,-0.7043279743794996,
					 -0.9969847523632076,-0.8056829950374929,-0.2232848229694142,0.4675007598639339,
					 0.9313510937243198,0.9431029274729957,0.4970516600769691,-0.190279519387695};
					 //,-0.7852447952776354,-0.9990348123874233,-0.7278711495336058,-0.1033826694373081,
					 //0.5712900538569543,0.9686459003924745,0.8957995195857675,0.3881121621222028,
					 //-0.3079734562688337,-0.8545619209102743,-0.986327054909366,-0.6393071365809022,
					 //0.01804666067516924,0.6666402081411824,0.9916317968554326,0.8352632937178742,
					 //0.2734394420734792,-0.4211179860554405,-0.9112553934761619,-0.959049199879731,
					 //-0.5412992304499267,0.1392094041567632,0.7521427030551485,0.9999692337191986,
					 //0.7623884955578118,0.15472745398924,-0.5280417292307387,-0.9544877327280404,
					 //-0.9176041978277584,-0.4352952113126279,0.2583157358340115,0.8265344893033894,
					 //0.9935350497383921,0.6782516371133961,0.03372982029961126,-0.6271652002063631,
					 //-0.983620307293945,-0.8626042774456625,-0.322860978524602,0.373606208003528,
					 //0.8887166458157811,0.9724242911634413,0.5840955943952939,-0.08776607283622896,
					 //-0.7170241396024485,-0.9982227685879364,-0.7948619017002729,-0.2056574190418587,
					 //0.4833777411260093,0.9377706130923654,0.9369488077102126,0.4813112475589719,
					 //-0.207965478933031,-0.7962911443974866,-0.9980794079518292,-0.7153777660688891,
					 //-0.08541587267359811,0.586008781845477,0.9729717622334742,0.8876326458925173};
 
double cosTheta[] = {1.,0.7572888982693721,0.1469729508840787,-0.5346869301685671,-0.9567979034168524,
					 -0.9144579301214193,-0.4282197734138278,0.2658857692699838,0.8309244559657687,
					 0.9926139623368049,0.6724666119239012,0.02588903699677217,-0.63325565131482,
					 -0.9850039860108796,-0.8586095153994179,-0.3154269219099697,0.3808709030440175,
					 0.8922855350080994,0.9705649564519441,0.5777105981326082,-0.09557731169517839,
					 -0.7224698722789882,-0.9986595155267595,-0.7900776562399809,-0.1979745601556861,
					 0.4902297831486548,0.940465704914642,0.9341786919212205,0.4744205998688464,
					 -0.2156317851392739,-0.8010117138688038,-0.9975627714538619,-0.7098747104288692,
					 -0.07759770328607386,0.5923469519693802,0.9747532445862991,0.8839926693851281,
					 0.3641224247674308,-0.3325009296105281,-0.8677209500840285,-0.9817299549782454};
					 //,-0.6191854419230029,0.04392543270163517,0.6857139269962564,0.9946416559042841,
					 //0.8207482405489019,0.2484454057793229,-0.4444583453034808,-0.9216121470623252,
					 //-0.9513949496575167,-0.5193495194280458,0.1647996988887314,0.7689514842411891,
					 //0.9998371457584857,0.7453796568792852,0.1290983325425473,-0.5498501888401678,
					 //-0.9618892199825011,-0.9070058664753,-0.4118417267113732,0.2832395315100768,
					 //0.8408300322385715,0.99026296598142,0.6590002687714889,0.00784420901294187,
					 //-0.6471196039690781,-0.9879571928894529,-0.8492184243120325,-0.2982501769851695,
					 //0.397495328436544,0.9002877756630347,0.9660605470779423,0.5628860790532889,
					 //-0.1135257897630787,-0.7348297195629733,-0.9994309877237927,-0.7788862636162693,
					 //-0.1802528531784318,0.5058792944294583,0.9464464002499753,0.9275874090031756,
					 //0.4584568937751372,-0.2332187770212312,-0.8116848751874143,-0.9961411127239482,
					 //-0.697048336363676,-0.05959282064674551,0.6067903733789996,0.9786240473200326,
					 //0.875411879850803,0.3472553487282364,-0.3494666389376975,-0.8765497606942951,
					 //-0.9781361661712315,-0.6049135585801836,0.06194752152044655,0.698737899225658,
					 //0.9963453862468615,0.8103047004676575,0.2309241215124303,-0.4605521533397157};

	int k = 0;
  while ( (n < nmax)&&(k<10*nmax) ){
    // transformed bisection to find a new interior point, just inside the boundary of the
    // yield surface.  This function overwrites the inputs for lambda_0
    transformedBisection(lambda_0,lambda_trial,X,Zeta,coher,limitParameters,S_star_to_S);
	// rotation matrix to transform to spherical coordinates
	Matrix3 R; 
	computeRotationToSphericalCS(lambda_0,lambda_trial,R);
	// Find the radial coordinate of lambda_0 in the spherical CS centered around the trial stress.
	Vector r_test = lambda_0 - lambda_trial;
	double r_test_norm = sqrt(r_test[0]*r_test[0] + r_test[1]*r_test[1] + r_test[2]*r_test[2]);	

// (4) Perform a rotation of lamda_0 about lambda_trial until a new interior point is found, set this as lambda_0
    interior = 0;
    n = max(n-2,0);
	
    // (5) Test for convergence:
	while ( (interior==0)&&(n < nmax) ){
		k++;
      // To avoid the cost of computing pow() to get theta, and then sin(), cos(),
      // we use a lookup table defined above by sinV and cosV.
	  // phi = pi_half*Pow(2.0,-0.25*n);
	  // theta = pi_fourth*n;
	  // Cartesian vector for the spherical {r,theta,phi}  
	  double x = r_test_norm*cosTheta[n]*sinPhi[n],
			 y = r_test_norm*sinTheta[n]*sinPhi[n],
			 z = r_test_norm*cosPhi[n];
	  Vector cart = Vector(x,y,z); 
	  // Test point (by rotation of lambda_0)
	  lambda_test = lambda_trial + R*cart;  
	  Matrix3 sigma_test = Matrix3(lambda_test[0],0.,0.,0.,lambda_test[1],0.,0.,0.,lambda_test[2]);
      if ( transformedYieldFunction(sigma_test,X,Zeta,coher,limitParameters,S_star_to_S) == -1 ) { // new interior point
        interior = 1;
        lambda_0 = lambda_test;
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
  Matrix3 sigma_new_star = lambda_0[0]*PL_trial + lambda_0[1]*PM_trial + lambda_0[2]*PH_trial;
  Matrix3 iso_sigma_new = one_third*sigma_new_star.Trace()*Identity;
  sigma_new = iso_sigma_new + S_star_to_S*(sigma_new_star - iso_sigma_new);
  Matrix3 d_sigma = sigma_new - sigma_old;

// (7) Compute increment in plastic strain for return:
//  d_ep0 = d_e - [C]^-1:(sigma_new-sigma_old)
  Matrix3 d_ee    = 0.5*d_sigma/shear + (one_ninth/bulk - one_sixth/shear)*d_sigma.Trace()*Identity;
  d_ep_new        = d_e - d_ee;
  
  return returnFlag;

} //===================================================================

// Computes bisection between two points in transformed space
void Arenisca4::transformedBisection(Vector& sigma_0,			// {lamda_L,lamda_M,lamda_H}
                                     const Vector& sigma_trial,	// {lamda_L,lamda_M,lamda_H}
                                     const double& X,
                                     const double& Zeta,
									 const double& coher,
                                     const double limitParameters[4],
                                     const double& S_star_to_S
                                    )
{
// Computes a bisection in transformed stress space between point sigma_0 (interior to the
// yield surface) and sigma_trial (exterior to the yield surface).  Returns this new point,
// which will be just outside the yield surface, overwriting the input arguments for lambda_0

// After the first iteration of the nonhardening return, the subseqent bisections will likely
// converge with eta << 1.  It may be faster to put in some logic to try to start bisection
// with tighter bounds, and only expand them to 0<eta<1 if the first eta mid is too large.


// (1) initialize bisection
  double eta_out = 1.0,  // This is for the accerator.  Must be > TOL
         eta_in  = 0.0,
         eta_mid,
         TOL = 1.0e-6;
  Vector sigma_test;
  Matrix3 sigma_test_3x3; //diagonal matrix to call yield function:

// (2) Test for convergence
  while (eta_out-eta_in > TOL){

// (3) Transformed test point
    eta_mid = 0.5*(eta_out+eta_in);
	sigma_test = sigma_0 + eta_mid*(sigma_trial-sigma_0);
	
// (4) Check if test point is within the yield surface:
	sigma_test_3x3 = Matrix3(sigma_test[0],0.0,0.0,0.0,sigma_test[1],0.0,0.0,0.0,sigma_test[2]);
    if ( transformedYieldFunction(sigma_test_3x3,X,Zeta,coher,limitParameters,S_star_to_S)!=1 ) {eta_in = eta_mid;}
    else {eta_out = eta_mid;}
	
	//cout<<"eta_mid = "<<eta_mid<<", sigma_test = "<<sigma_test<<endl;
  }
// (5) Converged, return {z_new,r_new}={z_test,r_test}
  sigma_0 = sigma_test;

} //===================================================================

// computeTransformedYieldFunction from transformed inputs
int Arenisca4::transformedYieldFunction(const Matrix3& sigma_star, // transformed
                                        const double& X,
                                        const double& Zeta,
										const double& coher,
                                        const double limitParameters[4],
                                        const double& S_star_to_S // (1/beta)*sqrt(two_third*G/K)
                                       )
{
// Evaluate the yield criteria and return:
//  -1: elastic
//   0: on yield surface within tolerance
//   1: plastic
  Matrix3 iso_sigma_star = one_third*sigma_star.Trace()*Identity;
  Matrix3 dev_sigma_star = sigma_star - iso_sigma_star;
  
  // Untransformed values:
  Matrix3 sigma = iso_sigma_star + S_star_to_S*dev_sigma_star;
  int    YIELD = computeYieldFunction(sigma,X,Zeta,coher,limitParameters);
  return YIELD;
} //===================================================================

// computeYieldFunction from untransformed inputs
int Arenisca4::computeYieldFunction(const Matrix3& sigma,
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
	
  double I1,
		 J2,
		 rJ2,
		 J3;
  Matrix3 S;
  computeInvariants(sigma,S,I1,J2,rJ2,J3);

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
// *** Lode Angle Function (Gamma) ***
// --------------------------------------------------------------------
  
  // All of these are far too slow to be in computeYieldFunction
  double Gamma = 1.0;
  if(d_cm.J3_type==1){// Gudehus: valid for 7/9<psi<9/7
	  if(J2 != 0.0){
		  double psi = d_cm.J3_psi;
#ifdef MHfastfcns		  
		  double sin3theta = -0.5*J3*fasterpow(3.0/J2,1.5);
#else
		  //double sin3theta = -0.5*J3*Pow(3.0/J2,1.5);
		  double sin3theta = -0.5*J3*sqrt(27.0/(J2*J2*J2));
#endif		  
		  Gamma = 0.5*(1 + sin3theta + (1./psi)*(1. - sin3theta));
	  }
  }
  if(d_cm.J3_type==3){// Mohr Coulomb: convex for 1/2<=psi<=2
	  if(J2 != 0.0){
		  double psi = d_cm.J3_psi;
		  double sinphi = 3.0*(1.0-psi)/(1.0+psi);
#ifdef MHfastfcns		  
		  double theta = one_third*asin(-0.5*J3*fasterpow(3.0/J2,1.5));
#else
		  //double theta = one_third*asin(-0.5*J3*Pow(3.0/J2,1.5));
		  double theta = one_third*asin(-0.5*J3*sqrt(27.0/(J2*J2*J2)));
#endif		  
		  //double theta = one_third*asin(-0.5*J3*Pow(3.0/J2,1.5));
		  Gamma = 2*sqrt_three/(3.0-sinphi)*(cos(theta)-sinphi*sin(theta)/sqrt_three);
	  }
  }
  
// --------------------------------------------------------------------
// *** Branch Point (Kappa) ***
// --------------------------------------------------------------------
  double  CR  = d_cm.CR,
          PEAKI1 = coher*d_cm.PEAKI1;  // perturbed point for variability
  double  Kappa  = PEAKI1-CR*(PEAKI1-X);

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
    if(J2 > Gamma*Gamma*Ff*Ff*fc2 ) YIELD = 1;
  }
  else if(( I1mZ <= PEAKI1 )&&( I1mZ >= Kappa )){// -----(kappa<I1<PEAKI1)
    if(rJ2 > Gamma*Ff) YIELD = 1;
  }
  else if( I1mZ > PEAKI1 ) {// --------------------------------(peakI1<I1)
    YIELD = 1;
  };

  return YIELD;
} //===================================================================

// Compute (dZeta/devp) Zeta and vol. plastic strain
double Arenisca4::computedZetadevp(double Zeta, double evp)
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
void Arenisca4::computeLimitParameters(double limitParameters[4], const double& coher)
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

// Compute rotation matrix [R] to transform to new basis z' aligned with p_new-p0
void Arenisca4::computeRotationToSphericalCS(const Vector& pnew,// interior point
		                                     const Vector& p0,	// origin (i.e. trial stress)
		                                     Matrix3& R			// Rotation matrix
											)
{
	// The basis is rotated so that e3 is aligned with z' in the new spherical coordinate system
	// but the rotation around that axis is arbitrary.
	if(pnew == p0){ // This shouldn't happen, but will cause nan in computeR if it does.
		cout << "Error in computeRotationToSphericalCS: pnew = p0 " << endl;
	}
	Vector x,
		   y,
		   z = pnew - p0;
   z = z/sqrt(z[0]*z[0]+z[1]*z[1]+z[2]*z[2]);
	// Dyadic product zz_ij = z_i*z_j
	Matrix3 zz = Matrix3(z[0]*z[0],z[0]*z[1],z[0]*z[2],
						 z[1]*z[0],z[1]*z[1],z[1]*z[2],
						 z[2]*z[0],z[2]*z[1],z[2]*z[2]);
	if( z==Vector(1.0,0.0,0.0)||z==Vector(-1.0,0.0,0.0))
	{
		y = (Identity-zz)*Vector(0.0,1.0,0.0);
		y = y/sqrt(y[0]*y[0]+y[1]*y[1]+y[2]*y[2]);
		x = Cross(y,z);
	}
	else
	{
		x = (Identity-zz)*Vector(1.0,0.0,0.0);
		x = x/sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
		y = Cross(z,x);
	}
	R = Matrix3(x[0],y[0],z[0],
				x[1],y[1],z[1],
				x[2],y[2],z[2]);
} //===================================================================

// Compute the unique set of eigenvalues and eigenprojectors of [A]
void Arenisca4::computeEigenProjectors(const Matrix3& A,// Input tensor
									   Vector& lambda,	// Ordered eigenvalues {L,M,H}
									   Matrix3& PL,		// Low eigenprojector
									   Matrix3& PM,		// Mid eigenprojector
									   Matrix3& PH		// High eigenprojector
									  )
{
	// Find eigenvalues and eigenprojectors.  For duplicate eigenvalues, the doubled eigenprojectors
	// are set to zero so that A=lambda_k P_k, summed for k=1:3
	//
	// This only works on symmetric positive definite 3x3 tensors, i.e. the stress tensor.
	computeEigenValues(A,lambda);	// Ordered eigenvalues for [A] lambda = {L,M,H}

	// Number of unique eigenvalues:
	int d=1;
	if(lambda[1] != lambda[0]) d++;
	if(lambda[2] != lambda[1]) d++;
	
	if(d==3){ //-----------------------------------------------------------------------------------LMH
		PL = (A - lambda[1]*Identity)*(A-lambda[2]*Identity) / ((lambda[0]-lambda[1])*(lambda[0]-lambda[2]));
		PM = (A - lambda[2]*Identity)*(A-lambda[0]*Identity) / ((lambda[1]-lambda[2])*(lambda[1]-lambda[0]));
		PH = (A - lambda[0]*Identity)*(A-lambda[1]*Identity) / ((lambda[2]-lambda[0])*(lambda[2]-lambda[1]));
	}
	else if(d==2){
		if(lambda[1]>lambda[0]){ //--------------------------------------------------------------------LMM
			PH = 0.0*Identity;
			PM = (A - lambda[0]*Identity)/(lambda[1]-lambda[0]);
			PL = Identity - PM;			
		}
		else{ //-----------------------------------------------------------------------------------MMH
			PL = 0.0*Identity;
			PM = (A - lambda[2]*Identity)/(lambda[1]-lambda[2]);
			PH = Identity-PM;			
		}
	}
	else{ //---------------------------------------------------------------------------------------LLL
		PL = 0.0*Identity;
		PM = 0.0*Identity;
		PH = Identity;		
	}

} //===================================================================

// Compute Eigenvalues for some real symmetric 3x3 matrix [A]
void Arenisca4::computeEigenValues(const Matrix3& A,// Input tensor
								   Vector& lambda	// Ordered eigenvalues {L,M,H}
								  )
{
	// The Matrix3 eigen() option seems robust, but does not order the eigenvectors. Also,
	// it doesn't accept a const Matrix3 as an input so we copy A to B.
	Matrix3 B(A);
	Matrix3 eigVec;
	B.eigen(lambda, eigVec);
	
	// Sort the eigenvalues in lambda {L,M,H}
	Vector temp = lambda;
	for(int i=1;i<4;i++)
	{
		if(lambda[1]<lambda[0]){
			temp[0]=lambda[1];
			temp[1]=lambda[0];
			lambda = temp;
		}
		if(lambda[2]<lambda[1]){
			temp[2]=lambda[1];
			temp[1]=lambda[2];
			lambda = temp;
		}
	}
	
	// Analytical solution (causes nan due to roundoff in some cases)
	// ==============================================================
	//// Compute Lode coordinates {r,z,theta}
	//double I1 = A.Trace();
	//Matrix3 S = A - one_third*I1*Identity;
	//Matrix3 SdotS = S*S;
	//double J2 = 0.5*SdotS.Trace();
	//double z = one_sqrt_three*I1,
	//	   r = sqrt(2.0*J2);
	//double theta;
	//if(r>0.0){
	//	Matrix3 Shat = S/r;
	//	theta = one_third*asin(3.0*sqrt(6.0)*Shat.Determinant());
	//}
	//else{
	//	theta = 0.0;
	//}
	//// Analytical solution to eigenvalues:
	//double lambdaL = z*one_sqrt_three + (sqrt_two/sqrt_three)*r*cos(theta - one_sixth*pi),
	//	   lambdaM = z*one_sqrt_three - (sqrt_two/sqrt_three)*r*sin(theta),
	//	   lambdaH = z*one_sqrt_three - (sqrt_two/sqrt_three)*r*cos(theta + one_sixth*pi);
	
	//// Form a vector
	//lambda = Vector(lambdaL,lambdaM,lambdaH);
    
	//cout<<"[A] = "<<A<<endl;
	//cout<<"lambda = "<<lambda<<endl;
} //===================================================================


// Series of checks that input parameters are within allowable limits
void Arenisca4::checkInputParameters(){
	
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
  if(d_cm.p3_crush_curve<=0.0){
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
  if(d_cm.principal_stress_cutoff<0.0){
	  ostringstream warn;
	  warn << "Principal stress cutoff must be nonnegative"<<endl;
	  throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
} //===================================================================


// ****************************************************************************************************
// ****************************************************************************************************
// ************** PUBLIC Uintah MPM constitutive model specific functions *****************************
// ****************************************************************************************************
// ****************************************************************************************************


void Arenisca4::carryForward(const PatchSubset* patches,
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
void Arenisca4::addParticleState(std::vector<const VarLabel*>& from,
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

void Arenisca4::addInitialComputesAndRequires(Task* task,
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

void Arenisca4::addComputesAndRequires(Task* task,
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
  task->computes(peakI1IDistLabel_preReloc,       matlset);  // For variability
  task->computes(lb->pLocalizedMPMLabel_preReloc, matlset);
  task->computes(pAreniscaFlagLabel_preReloc,     matlset);
  task->computes(pScratchDouble1Label_preReloc,   matlset);
  task->computes(pScratchDouble2Label_preReloc,   matlset);
  task->computes(pPorePressureLabel_preReloc,     matlset);
  task->computes(pepLabel_preReloc,               matlset);
  task->computes(pevpLabel_preReloc,              matlset);
  task->computes(peveLabel_preReloc,              matlset);
  task->computes(pCapXLabel_preReloc,             matlset);
  task->computes(pZetaLabel_preReloc,             matlset);
  task->computes(pP3Label_preReloc,               matlset);
  task->computes(pStressQSLabel_preReloc,         matlset);
  task->computes(pScratchMatrixLabel_preReloc,    matlset);
}

//T2D: Throw exception that this is not supported
void Arenisca4::addComputesAndRequires(Task* ,
                                       const MPMMaterial* ,
                                       const PatchSet* ,
                                       const bool ) const
{
  cout << "NO VERSION OF addComputesAndRequires EXISTS YET FOR Arenisca4"<<endl;
}

//T2D: Throw exception that this is not supported
double Arenisca4::computeRhoMicroCM(double pressure,
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

  cout << "NO VERSION OF computeRhoMicroCM EXISTS YET FOR Arenisca4"<<endl;
  return rho_cur;
}

//T2D: Throw exception that this is not supported
void Arenisca4::computePressEOSCM(double rho_cur,double& pressure,
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

  cout << "NO VERSION OF computePressEOSCM EXISTS YET FOR Arenisca4" << endl;
}

//T2D: Throw exception that this is not supported
double Arenisca4::getCompressibility()
{
  cout << "NO VERSION OF computePressEOSCM EXISTS YET FOR Arenisca4"
       << endl;
  return 1.0;
}

// Initialize all labels of the particle variables associated with Arenisca4.
void Arenisca4::initializeLocalMPMLabels()
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
void Arenisca4::WeibullParser(WeibParameters &iP)
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
