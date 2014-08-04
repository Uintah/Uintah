/* MIT LICENSE

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI),
             University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included
  in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
  DEALINGS IN THE SOFTWARE.

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
//#define MH_VARIABILITY             // MH! Broken, not sure why since it works in Arenisca 2
#define MHdebug                      // Prints errors messages when particles are deleted or subcycling fails

// INCLUDE SECTION: tells the preprocessor to include the necessary files
#include <CCA/Components/MPM/ConstitutiveModel/Arenisca3.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/Matrix3.h>
#include <Core/Math/MiscMath.h>
#include <Core/Malloc/Allocator.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <sci_values.h>

#ifdef MH_VARIABILITY
#include <Core/Math/Weibull.h>
#include <fstream>
#endif

#include <iostream>

using std::cerr;
using namespace Uintah;
using namespace std;

// Requires the necessary input parameters CONSTRUCTORS
Arenisca3::Arenisca3(ProblemSpecP& ps, MPMFlags* Mflag)
  : ConstitutiveModel(Mflag)
{
  proc0cout << "In Arenisca ver 3.0"<< endl;
  proc0cout << endl
            << "                                        ;1BB@B@B@@@B@8u:                        " << endl
            << "                                   .Y@@@B@B@BB8GZMB@B@B@B@Mr                    " << endl
            << "                                 Y@B@BB7.              :S@@B@Bi                 " << endl
            << "                       BB.     EB@BG.                      rB@B@L               " << endl
            << "                    iBOF@    5@B@L                            NB@Bi             " << endl
            << "                     OB G  :B@Bv                                O@BM            " << endl
            << "                   .@B@B@B@B@B  ;irr77777r77vL, .Yv77777777r7rr  :@B@           " << endl
            << "                    B@BZS@@@2  :BBMMMMMMMMMMM@i ,@BMOMOMMMMMM@B    @@@          " << endl
            << "                   L@B  i@@L   ,@E0q0PNPqPNPGB: .BGP0PNP0P0P08O     @B@         " << endl
            << "                 uB5B. ,B@X    :B8qqXXSXSXkPNB: .@EqkXkXXPkPqO8      @@@        " << endl
            << "                     @Z BZ  B@B     i@M8PqkPkXkqPOBr :BMNPkXkPkPPGB@      v@Bi       " << endl
            << "              ;@r BN  7@B:        r8XXSPSPXZ5     :8PXkPkXkZU         B@B       " << endl
            << "             2@  u@   @B@         iONkPkPkqG1     .M0kPSPkqGu         XB@       " << endl
            << "            F@  :@    B@P         rMPXkXkXXOS     .BqqkXkXXO1         :@@i      " << endl
            << "           Y@   @v    @@L         7MNSXkPXNGX     ,M0kPkXSN8F         .B@7      " << endl
            << "          :@    B: v  B@7         rMPPSXkXPOk     ,BqXkPSPPO5         .@@7      " << endl
            << "          @r   @@  B. @BX         7ONkXSXXq8k     ,M0kXkXXNGS         rB@.      " << endl
            << "         @B  .BG   @. B@B         7MqPkPkXXOF     .BqPkXSPPO1         O@B       " << endl
            << "        :B   B@       uB@.        7MNkPSPkqG5     .O0kXkXSN8F         @BN       " << endl
            << "        BL   LB   E:   @@@        rMqPkXkPkG2     ,OPPSPkXPO5        MB@        " << endl
            << "       r@     @  u@Z   :@BY       7M0XPSPSXXZOBBBMONqSPSPk0ME       7B@v        " << endl
            << "       @v    .   @B     B@B7      v@ENXPSPkqX00Z0EPPSXkPXEO@i      i@@Z         " << endl
            << "      :B     GM  OM    B@0@Bu      J@80XPkPkPXqXPkqkqkqXZMZ       vB@8          " << endl
            << "      BM     B@  :B    .B i@BB      .OM800N0qEq0q0q0qE0OBY       MB@1           " << endl
            << "      @.     B    @,    Gq .@B@v      Y@@BBMBMBBBMBMBB@M,      L@@@:            " << endl
            << "     .B     .@    P@    F@i  UB@B2      .. ............      jB@BS              " << endl
            << "     2@  B.  P@    :    @B1    1@B@Br                     r@@B@F                " << endl
            << "     @u  @:   B@      B@Br       rB@B@Bqi.           ,78B@B@B7                  " << endl
            << "     @:  Gr    B2 ,8uB@B@           i0@B@B@B@B@B@B@@@@@@@Gr                     " << endl
            << "     @   7Y    XBUP@B@@@                .ru8B@B@B@MZjr.                         " << endl
            << "     B         B@B@B@B.                                                         " << endl
            << "     @02    ..BM U@@@@      :LLrjM           ,.           r8,       N@.         " << endl
            << "     B@@,r@ @@@   .B@     GB@B@B@BE      F@B@B@@@@7      :@B@      2@B@         " << endl
            << "     uB@B@B@B@.         Y@B@i   B@k    qB@8:   .ru.      @B@B;     @B@B         " << endl
            << "      U@@B@B@.         M@@7      .    NB@                B@@@O    :B@@@r        " << endl
            << "       2B@B@:         B@B             M@@7              7@BEB@    B@E0BO        " << endl
            << "        :B7          k@B               1@@@B@B@BF       @BE @B:  :@B .@B        " << endl
            << "                     @B7                  .:iLZ@B@X    :@@, B@B  @@O  B@.       " << endl
            << "                    :@@                         iB@;   B@@  r@@ :B@   @BG       " << endl
            << "                     @Bi        ur               @BJ  .@@U   @BO@@2   Y@B       " << endl
            << "                     P@BY    ;@B@B  iB@i       :@B@   8B@    u@B@B     B@5      " << endl
            << "                      7@@@B@B@B@:    BB@@@MOB@B@B5    B@@     B@B7     @B@      " << endl
            << "                        :Lk5v.         ;ZB@B@BU,      Z@r     :Ov      .@B.     " << endl
            << endl
            << "    University of Utah, Mechanical Engineering, Computational Solid Mechanics   " << endl << endl;

  cout << endl;
// ps->getWithDefault("STREN",d_cm.STREN,0.0);    // Shear Limit Surface Parameter
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
  pi_half = 0.5*pi;

  ps->require("PEAKI1",d_cm.PEAKI1);             // Shear Limit Surface Parameter
  ps->require("FSLOPE",d_cm.FSLOPE);             // Shear Limit Surface Parameter
  ps->require("STREN",d_cm.STREN);    // Shear Limit Surface Parameter
  ps->require("YSLOPE",d_cm.YSLOPE);  // Shear Limit Surface Parameter
  ps->require("BETA_nonassociativity",d_cm.BETA_nonassociativity);   // Nonassociativity Parameter
  ps->require("B0",d_cm.B0);                     // Tangent Elastic Bulk Modulus Parameter
  ps->require("B1",d_cm.B1);          // Tangent Elastic Bulk Modulus Parameter
  ps->require("B2",d_cm.B2);          // Tangent Elastic Bulk Modulus Parameter
  ps->require("B3",d_cm.B3);          // Tangent Elastic Bulk Modulus Parameter
  ps->require("B4",d_cm.B4);          // Tangent Elastic Bulk Modulus Parameter
  ps->require("G0",d_cm.G0);                     // Tangent Elastic Shear Modulus Parameter
  ps->require("G1",d_cm.G1);          // Tangent Elastic Shear Modulus Parameter
  ps->require("G2",d_cm.G2);          // Tangent Elastic Shear Modulus Parameter
  ps->require("G3",d_cm.G3);          // Tangent Elastic Shear Modulus Parameter
  ps->require("G4",d_cm.G4);          // Tangent Elastic Shear Modulus Parameter
  ps->require("p0_crush_curve",d_cm.p0_crush_curve);             // Crush Curve Parameter
  ps->require("p1_crush_curve",d_cm.p1_crush_curve);             // Crush Curve Parameter
  ps->require("p2_crush_curve",d_cm.p2_crush_curve);  // Crush Curve Parameter (not used)
  ps->require("p3_crush_curve",d_cm.p3_crush_curve);             // Crush Curve Parameter
  ps->require("CR",d_cm.CR);                                     // Cap Shape Parameter CR = (peakI1-kappa)/(peakI1-X)
  ps->require("fluid_B0",d_cm.fluid_B0);                                // Fluid bulk modulus (K_f)
  ps->require("fluid_pressure_initial",d_cm.fluid_pressure_initial);    // Zero strain Fluid Pressure (Pf0)
  ps->require("T1_rate_dependence",d_cm.T1_rate_dependence);    // Rate dependence parameter
  ps->require("T2_rate_dependence",d_cm.T2_rate_dependence);    // Rate dependence parameter
  ps->require("gruneisen_parameter",d_cm.gruneisen_parameter);  // Mie Gruneisen e.o.s. parameter
  ps->require("subcycling_characteristic_number",d_cm.subcycling_characteristic_number);  // Force subcycling for value > 1

#ifdef MH_VARIABILITY
  ps->get("PEAKI1IDIST",wdist.WeibDist);
  WeibullParser(wdist);
  proc0cout <<"WeibMed="<<wdist.WeibMed<<endl;
#endif

  initializeLocalMPMLabels();
}
Arenisca3::Arenisca3(const Arenisca3* cm)
  : ConstitutiveModel(cm)
{
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
  pi_half = 0.5*pi;

#ifdef MH_VARIABILITY
//Weibull distribution input
  wdist.WeibMed    = cm->wdist.WeibMed;
  wdist.WeibMod    = cm->wdist.WeibMod;
  wdist.WeibRefVol = cm->wdist.WeibRefVol;
  wdist.WeibSeed   = cm->wdist.WeibSeed;
  wdist.Perturb    = cm->wdist.Perturb;
  wdist.WeibDist   = cm->wdist.WeibDist;
  WeibullParser(wdist);
#endif

  // Shear Strength
  d_cm.PEAKI1 = cm->d_cm.PEAKI1;
  d_cm.FSLOPE = cm->d_cm.FSLOPE;
  d_cm.STREN = cm->d_cm.STREN;   // not used (except as temp input for damage model)
  d_cm.YSLOPE = cm->d_cm.YSLOPE; // not used (except as temp input for damage model)
  d_cm.BETA_nonassociativity = cm->d_cm.BETA_nonassociativity;
  // Bulk Modulus
  d_cm.B0 = cm->d_cm.B0;
  d_cm.B1 = cm->d_cm.B1;
  d_cm.B2 = cm->d_cm.B2;
  d_cm.B3 = cm->d_cm.B3;
  d_cm.B4 = cm->d_cm.B4;
  // Shear Modulus
  d_cm.G0 = cm->d_cm.G0;
  d_cm.G1 = cm->d_cm.G1; //not used
  d_cm.G2 = cm->d_cm.G2; //not used
  d_cm.G3 = cm->d_cm.G3; //not used
  d_cm.G4 = cm->d_cm.G4; //not used
  // Porosity (Crush Curve)
  d_cm.p0_crush_curve = cm->d_cm.p0_crush_curve;
  d_cm.p1_crush_curve = cm->d_cm.p1_crush_curve;
  d_cm.p2_crush_curve = cm->d_cm.p2_crush_curve; // not used
  d_cm.p3_crush_curve = cm->d_cm.p3_crush_curve;
  d_cm.CR = cm->d_cm.CR;  // not used
  // Fluid Effects
  d_cm.fluid_B0 = cm->d_cm.fluid_B0;
  d_cm.fluid_pressure_initial = cm->d_cm.fluid_pressure_initial; // pfi
  // Rate Dependence
  d_cm.T1_rate_dependence = cm->d_cm.T1_rate_dependence; //not used
  d_cm.T2_rate_dependence = cm->d_cm.T2_rate_dependence; //not used
  // Equation of State
  d_cm.gruneisen_parameter = cm->d_cm.gruneisen_parameter; //not used
  // Subcycling
  d_cm.subcycling_characteristic_number = cm->d_cm.subcycling_characteristic_number; // not used
  initializeLocalMPMLabels();
}
// DESTRUCTOR
Arenisca3::~Arenisca3()
{
#ifdef MH_VARIABILITY
  VarLabel::destroy(peakI1IDistLabel);
  VarLabel::destroy(peakI1IDistLabel_preReloc);
#endif
  VarLabel::destroy(pLocalizedLabel);
  VarLabel::destroy(pLocalizedLabel_preReloc);
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
  VarLabel::destroy(pCapXQSLabel);
  VarLabel::destroy(pCapXQSLabel_preReloc);
  VarLabel::destroy(pKappaLabel);
  VarLabel::destroy(pKappaLabel_preReloc);
  VarLabel::destroy(pStressQSLabel);
  VarLabel::destroy(pStressQSLabel_preReloc);
  VarLabel::destroy(pScratchMatrixLabel);
  VarLabel::destroy(pScratchMatrixLabel_preReloc);
  VarLabel::destroy(pZetaLabel);
  VarLabel::destroy(pZetaLabel_preReloc);
  VarLabel::destroy(pZetaQSLabel);
  VarLabel::destroy(pZetaQSLabel_preReloc);
  VarLabel::destroy(pIotaLabel);
  VarLabel::destroy(pIotaLabel_preReloc);
  VarLabel::destroy(pIotaQSLabel);
  VarLabel::destroy(pIotaQSLabel_preReloc);
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
  cm_ps->appendElement("G1",d_cm.G1);  // Not used
  cm_ps->appendElement("G2",d_cm.G2);  // Not used
  cm_ps->appendElement("G3",d_cm.G3);  // Not used
  cm_ps->appendElement("G4",d_cm.G4);  // Not used
  cm_ps->appendElement("p0_crush_curve",d_cm.p0_crush_curve);
  cm_ps->appendElement("p1_crush_curve",d_cm.p1_crush_curve);
  cm_ps->appendElement("p2_crush_curve",d_cm.p2_crush_curve);  // Not used
  cm_ps->appendElement("p3_crush_curve",d_cm.p3_crush_curve);
  cm_ps->appendElement("CR",d_cm.CR);
  cm_ps->appendElement("fluid_B0",d_cm.fluid_B0); // kf
  cm_ps->appendElement("fluid_pressure_initial",d_cm.fluid_pressure_initial); //Pf0
  cm_ps->appendElement("T1_rate_dependence",d_cm.T1_rate_dependence);
  cm_ps->appendElement("T2_rate_dependence",d_cm.T2_rate_dependence);
  cm_ps->appendElement("gruneisen_parameter",d_cm.gruneisen_parameter); //Pf0
  cm_ps->appendElement("subcycling_characteristic_number",d_cm.subcycling_characteristic_number);
#ifdef MH_VARIABILITY
//    Uintah Variability Variables
  cm_ps->appendElement("peakI1IPerturb", wdist.Perturb);
  cm_ps->appendElement("peakI1IMed", wdist.WeibMed);
  cm_ps->appendElement("peakI1IMod", wdist.WeibMod);
  cm_ps->appendElement("peakI1IRefVol", wdist.WeibRefVol);
  cm_ps->appendElement("peakI1ISeed", wdist.WeibSeed);
  cm_ps->appendElement("PEAKI1IDIST", wdist.WeibDist);
#endif
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
  /////
  // Allocates memory for internal state variables at beginning of run.

  // Get the particles in the current patch
  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(),patch);
  Matrix3 Identity; Identity.Identity();

  ParticleVariable<double>  pdTdt;
  ParticleVariable<Matrix3> pDefGrad,
                            pStress;

  new_dw->allocateAndPut(pdTdt,       lb->pdTdtLabel,               pset);
  new_dw->allocateAndPut(pStress,     lb->pStressLabel,             pset);
  new_dw->allocateAndPut(pDefGrad,    lb->pDeformationMeasureLabel, pset);

  // To fix : For a material that is initially stressed we need to
  // modify the stress tensors to comply with the initial stress state
  ParticleSubset::iterator iter = pset->begin();
  for(; iter != pset->end(); iter++){
    pdTdt[*iter] = 0.0;
    pDefGrad[*iter] = Identity;
    pStress[*iter] = - d_cm.fluid_pressure_initial * Identity;
  }

  // Allocate particle variables
  ParticleVariable<int> pLocalized,
                        pAreniscaFlag;

#ifdef MH_VARIABILITY
  ParticleVariable<double>  peakI1IDist;     // Holder for particles PEAKI1 value
#endif
  ParticleVariable<double>  pScratchDouble1, // Developer tool
                   pScratchDouble2, // Developer tool
                   pPorePressure,   // Plottable fluid pressure
                   pevp,            // Plastic Volumetric Strain
                   peve,            // Elastic Volumetric Strain
                   pCapX,           // I1 of cap intercept
                   pCapXQS,         // I1 of cap intercept, quasistatic
                   pKappa,          // Not used
                   pZeta,           // Trace of isotropic Backstress
                   pZetaQS,         // Trace of isotropic Backstress, quasistatic
                   pIota,           // void variable
                   pIotaQS;         // void variable, quasistatic
  ParticleVariable<Matrix3> pStressQS,       // stress, quasistatic
                   pScratchMatrix,  // Developer tool
                   pep;             // Plastic Strain Tensor

  new_dw->allocateAndPut(pLocalized,      pLocalizedLabel,      pset);
  new_dw->allocateAndPut(pAreniscaFlag,   pAreniscaFlagLabel,   pset);
  new_dw->allocateAndPut(pScratchDouble1, pScratchDouble1Label, pset);
  new_dw->allocateAndPut(pScratchDouble2, pScratchDouble2Label, pset);
  new_dw->allocateAndPut(pPorePressure,   pPorePressureLabel,   pset);
#ifdef MH_VARIABILITY
  new_dw->allocateAndPut(peakI1IDist,     peakI1IDistLabel,     pset);
#endif
  new_dw->allocateAndPut(pep,             pepLabel,             pset);
  new_dw->allocateAndPut(pevp,            pevpLabel,            pset);
  new_dw->allocateAndPut(peve,            peveLabel,            pset);
  new_dw->allocateAndPut(pCapX,           pCapXLabel,           pset);
  new_dw->allocateAndPut(pCapXQS,         pCapXQSLabel,         pset);
  new_dw->allocateAndPut(pKappa,          pKappaLabel,          pset);  //not used?
  new_dw->allocateAndPut(pZeta,           pZetaLabel,           pset);
  new_dw->allocateAndPut(pZetaQS,         pZetaQSLabel,         pset);
  new_dw->allocateAndPut(pIota,           pIotaLabel,           pset);
  new_dw->allocateAndPut(pIotaQS,         pIotaQSLabel,         pset);
  new_dw->allocateAndPut(pStressQS,       pStressQSLabel,  pset);
  new_dw->allocateAndPut(pScratchMatrix,  pScratchMatrixLabel,  pset);

  for(ParticleSubset::iterator iter = pset->begin();
      iter != pset->end();iter++){
    pLocalized[*iter] = 0;
    pAreniscaFlag[*iter] = 0;
    pScratchDouble1[*iter] = 0;
    pScratchDouble2[*iter] = 0;
    pPorePressure[*iter] = d_cm.fluid_pressure_initial;
#ifdef MH_VARIABILITY
    peakI1IDist[*iter] = d_cm.PEAKI1;
#endif
    pevp[*iter] = 0.0;
    peve[*iter] = 0.0;
    pCapX[*iter] = computeX(0.0);
    pCapXQS[*iter] = computeX(0.0);
    pKappa[*iter] = 0;                                   // MH: remove
    pZeta[*iter] = -3.0 * d_cm.fluid_pressure_initial;
    pZetaQS[*iter] = -3.0 * d_cm.fluid_pressure_initial;
    pIota[*iter] = 0.0;
    pIotaQS[*iter] = 0.0;
    pStressQS[*iter].set(0.0);
    pScratchMatrix[*iter].set(0.0);
    pep[*iter].set(0.0);
  }
#ifdef MH_VARIABILITY  // XXX if this is uncommented it seg faults!
//if ( wdist.Perturb){
//    // Make the seed differ for each patch, otherwise each patch gets the
//    // same set of random #s.
//    int patchID = patch->getID();
//    int patch_div_32 = patchID/32;
//    patchID = patchID%32;
//    unsigned int unique_seed = ((wdist.WeibSeed+patch_div_32+1) << patchID);
//    SCIRun::Weibull weibGen(wdist.WeibMed,wdist.WeibMod,wdist.WeibRefVol,
//                unique_seed,wdist.WeibMod);
//    //proc0cout << "Weibull Variables for PEAKI1I: (initialize CMData)\n"
//    //          << "Median:            " << wdist.WeibMed
//    //          << "\nModulus:         " << wdist.WeibMod
//    //          << "\nReference Vol:   " << wdist.WeibRefVol
//    //          << "\nSeed:            " << wdist.WeibSeed
//    //          << "\nPerturb?:        " << wdist.Perturb << std::endl;
//    constParticleVariable<double>pVolume;
//    new_dw->get(pVolume, lb->pVolumeLabel, pset);
//    ParticleSubset::iterator iter = pset->begin();
//    for(;iter != pset->end();iter++){
//      //set value with variability and scale effects
//      peakI1IDist[*iter] = weibGen.rand(pVolume[*iter]);
//
//      //set value with ONLY scale effects
//      if(wdist.WeibSeed==0)
//        peakI1IDist[*iter]= pow(wdist.WeibRefVol/pVolume[*iter],1./wdist.WeibMod)
//                  *wdist.WeibMed;
//    }
//}
#endif
  computeStableTimestep(patch, matl, new_dw);
}

// Compute stable timestep based on both the particle velocities
// and wave speed
void Arenisca3::computeStableTimestep(const Patch* patch,
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

  //MH! change this to call the computeElasticProperties co
  double  c_dil = 0.0;
  double  bulk,shear;                   // High pressure limit elastic properties
  computeElasticProperties(bulk,shear);

  Vector  dx = patch->dCell(),
          WaveSpeed(1.e-12,1.e-12,1.e-12); // MH!: what is this doing?

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
  // Define some constants
  Matrix3 Identity; Identity.Identity();

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
#ifdef MH_VARIABILITY
    constParticleVariable<double>  peakI1IDist;
#endif
    constParticleVariable<double>  pScratchDouble1,
                                   pScratchDouble2,
                                   pPorePressure,
                                   pmass,           //used for stable timestep
                                   pevp,
                                   peve,
                                   pCapX, pCapXQS,
                                   pKappa,
                                   pZeta, pZetaQS,
                                   pIota, pIotaQS;
    constParticleVariable<long64>  pParticleID;
    constParticleVariable<Vector>  pvelocity;
    constParticleVariable<Matrix3> pScratchMatrix,
                                   pep,
                                   pDefGrad,
                                   pStress_old, pStressQS_old,
                                   pBackStress,
                                   pBackStressIso;

    old_dw->get(delT,            lb->delTLabel,   getLevel(patches));
#ifdef MH_VARIABILITY
    old_dw->get(peakI1IDist,     peakI1IDistLabel,             pset);
#endif
    old_dw->get(pLocalized,      pLocalizedLabel,              pset); //initializeCMData()
    old_dw->get(pAreniscaFlag,   pAreniscaFlagLabel,           pset); //initializeCMData()
    old_dw->get(pScratchDouble1, pScratchDouble1Label,         pset); //initializeCMData()
    old_dw->get(pScratchDouble2, pScratchDouble2Label,         pset); //initializeCMData()
    old_dw->get(pPorePressure,   pPorePressureLabel,           pset); //initializeCMData()
    old_dw->get(pmass,           lb->pMassLabel,               pset);
    old_dw->get(pevp,            pevpLabel,                    pset); //initializeCMData()
    old_dw->get(peve,            peveLabel,                    pset); //initializeCMData()
    old_dw->get(pCapX,           pCapXLabel,                   pset); //initializeCMData()
    old_dw->get(pCapXQS,         pCapXQSLabel,                 pset); //initializeCMData()
    old_dw->get(pKappa,          pKappaLabel,                  pset); //initializeCMData()
    old_dw->get(pZeta,           pZetaLabel,                   pset); //initializeCMData()
    old_dw->get(pZetaQS,         pZetaQSLabel,                 pset); //initializeCMData()
    old_dw->get(pIota,           pIotaLabel,                   pset); //initializeCMData()
    old_dw->get(pIotaQS,         pIotaQSLabel,                 pset); //initializeCMData()
    old_dw->get(pParticleID,     lb->pParticleIDLabel,         pset);
    old_dw->get(pvelocity,       lb->pVelocityLabel,           pset);
    old_dw->get(pScratchMatrix,  pScratchMatrixLabel,          pset); //initializeCMData()
    old_dw->get(pep,             pepLabel,                     pset); //initializeCMData()
    old_dw->get(pDefGrad,        lb->pDeformationMeasureLabel, pset);
    old_dw->get(pStress_old,     lb->pStressLabel,             pset); //initializeCMData()
    old_dw->get(pStressQS_old,   pStressQSLabel,             pset); //initializeCMData()

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
#ifdef MH_VARIABILITY
    ParticleVariable<double>  peakI1IDist_new;
    new_dw->allocateAndPut(peakI1IDist_new, peakI1IDistLabel_preReloc,   pset);
#endif
    new_dw->allocateAndPut(pLocalized_new, pLocalizedLabel_preReloc,   pset);
    new_dw->allocateAndPut(pAreniscaFlag_new,   pAreniscaFlagLabel_preReloc,    pset);

    // Allocate particle variables used in ComputeStressTensor
    ParticleVariable<double>  p_q,
                              pdTdt,
                              pScratchDouble1_new,
                              pScratchDouble2_new,
                              pPorePressure_new,
                              pevp_new,
                              peve_new,
                              pCapX_new, pCapXQS_new,
                              pKappa_new,
                              pZeta_new, pZetaQS_new,
                              pIota_new, pIotaQS_new;
    ParticleVariable<Matrix3> pScratchMatrix_new,
                              pep_new,
                              pStress_new, pStressQS_new;

    new_dw->allocateAndPut(p_q,                 lb->p_qLabel_preReloc,         pset);
    new_dw->allocateAndPut(pdTdt,               lb->pdTdtLabel_preReloc,       pset);
    new_dw->allocateAndPut(pScratchDouble1_new, pScratchDouble1Label_preReloc, pset);
    new_dw->allocateAndPut(pScratchDouble2_new, pScratchDouble2Label_preReloc, pset);
    new_dw->allocateAndPut(pPorePressure_new,   pPorePressureLabel_preReloc,   pset);
    new_dw->allocateAndPut(pevp_new,            pevpLabel_preReloc,            pset);
    new_dw->allocateAndPut(peve_new,            peveLabel_preReloc,            pset);
    new_dw->allocateAndPut(pCapX_new,           pCapXLabel_preReloc,           pset);
    new_dw->allocateAndPut(pCapXQS_new,         pCapXQSLabel_preReloc,         pset);
    new_dw->allocateAndPut(pKappa_new,          pKappaLabel_preReloc,          pset);
    new_dw->allocateAndPut(pZeta_new,           pZetaLabel_preReloc,           pset);
    new_dw->allocateAndPut(pZetaQS_new,         pZetaQSLabel_preReloc,         pset);
    new_dw->allocateAndPut(pIota_new,           pIotaLabel_preReloc,           pset);
    new_dw->allocateAndPut(pIotaQS_new,         pIotaQSLabel_preReloc,         pset);
    new_dw->allocateAndPut(pScratchMatrix_new,  pScratchMatrixLabel_preReloc,  pset);
    new_dw->allocateAndPut(pep_new,             pepLabel_preReloc,             pset);
    new_dw->allocateAndPut(pStress_new,         lb->pStressLabel_preReloc,     pset);
    new_dw->allocateAndPut(pStressQS_new,       pStressQSLabel_preReloc,       pset);

    // Allocate temporary particle variables
    ParticleVariable<double>       f_trial_step;
    ParticleVariable<Matrix3>      rotation;

    new_dw->allocateTemporary(f_trial_step, pset);
    new_dw->allocateTemporary(rotation,     pset);

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

      // Use poYieldFxn:I1=-35.3311, J2=7516lar decomposition to compute the rotation and stretch tensors
      Matrix3 tensorR, tensorU;
      pDefGrad[idx].polarDecompositionRMB(tensorU, tensorR);
      rotation[idx]=tensorR;

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
      Matrix3 sigma_old = (tensorR.Transpose())*(pStress_old[idx]*tensorR);

      // initial assignment for the updated values of plastic strains, volumetric
      // part of the plastic strain, volumetric part of the elastic strain, \kappa,
      // and the backstress. tentative assumption of elasticity
      pevp_new[idx]   = pevp[idx];
      peve_new[idx]   = peve[idx] + D.Trace()*delT;
      pCapX_new[idx]  = pCapX[idx];
      pKappa_new[idx] = pKappa[idx];
      pZeta_new[idx]  = pZeta[idx];
      pep_new[idx]    = pep[idx];
#ifdef MH_VARIABILITY
      //Weibull Distribution on PEAKI1
      peakI1IDist_new[idx] = peakI1IDist[idx];
      double PEAKI1Dist    = peakI1IDist_new[idx]; //Weibull Distribution on PEAKI1
#endif

      // Divides the strain increment into substeps, and calls substep function
      int stepFlag = computeStep(D,                  // strain "rate"
                                 delT,               // time step (s)
                                 sigma_old,          // unrotated stress at start of step
                                 pCapX[idx],         // hydrostatic comrpessive strength at start of step
                                 pZeta[idx],         // trace of isotropic backstress at start of step
                                 pScratchDouble1[idx],  // Scalar-valued damage (XXX)
                                 pep[idx],           // plastic strain at start of step
                                 pStress_new[idx],   // unrotated stress at start of step
                                 pCapX_new[idx],     // hydrostatic comrpessive strength at start of step
                                 pZeta_new[idx],     // trace of isotropic backstress at start of step
                                 pScratchDouble1_new[idx], // Scalar-valued damage Damage (XXX)
                                 pep_new[idx],       // plastic strain at start of step
                                 pParticleID[idx]
                                );
      // If the computeStep function can't converge it will return a stepFlag!=1.  This indicates substepping
      // has failed, and the particle will be deleted.
      if(stepFlag!=0){
        pLocalized_new[idx]=-999;
#ifdef MHdebug
       cout<<"bad step, deleting particle"<<endl;
#endif
      }

      // Plastic volumetric strain at end of step
      pevp_new[idx] = pep_new[idx].Trace();
      // Elastic volumetric strain at end of step
      // peve_new[idx] = peve[idx] + D.Trace()*delT - pevp_new[idx] + pevp[idx];  // Faster
      peve_new[idx] = log(pDefGrad[idx].Determinant()) - pevp_new[idx];           // More accurate

      // Set pore pressure (plotting variable)
      pPorePressure_new[idx] = computePorePressure(peve_new[idx]+pevp_new[idx]);

      // Branch point at the start of the step (not used, set to zero)
      pKappa_new[idx] = 0.0;

      // MH!: ADD RATE DEPENDENCE CODE HERE
      pCapXQS_new[idx]   = pCapX_new[idx];
      pZetaQS_new[idx]   = pZeta_new[idx];
      pIotaQS_new[idx]   = pIota_new[idx];
      pStressQS_new[idx] = pStress_new[idx];

      // Use polar decomposition to compute the rotation and stretch tensors
      pDefGrad_new[idx].polarDecompositionRMB(tensorU, tensorR);
      rotation[idx]=tensorR;

      // Compute the rotated stress at the end of the current timestep
      pStress_new[idx] = (rotation[idx]*pStress_new[idx])*(rotation[idx].Transpose());

      // Compute wave speed + particle velocity at each particle, store the maximum
      // Conservative elastic properties used to compute number of time steps:
      // Get the Arenisca model parameters.
      double bulk,
             shear;
             computeElasticProperties(bulk,shear); // High pressure bulk and shear moduli.

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
                           const double & damage_n, // XXX
                           const Matrix3& ep_n,    // plastic strain at start of step(t_n)
                           Matrix3& sigma_p, // unrotated stress at end of step(t_n+1)
                           double & X_p,     // hydrostatic comrpessive strength at end of step(t_n+1)
                           double & Zeta_p,  // trace of isotropic backstress at end of step(t_n+1)
                           double & damage_p, // XXX
                           Matrix3& ep_p,    // plastic strain at end of step (t_n+1)
                           long64 ParticleID)// ParticleID for debug purposes
{
  int n,
      chimax = 16,                                  // max allowed subcycle multiplier
      // MH!: make this an input parameter for subcycle control
      chi = 1,                                      // subcycle multiplier
      stepFlag,                                     // 0/1 good/bad step
      substepFlag;                                  // 0/1 good/bad substep
  double dt,                                        // substep time increment
         X_old,                                     // X at start of substep
         X_new,                                     // X at end of substep
         Zeta_old,                                  // Zeta at start of substep
         Zeta_new,                                  // Zeta at end of substep
         damage_old,
         damage_new;
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
//
Matrix3 sigma_trial = computeTrialStress(sigma_old,D*Dt,bulk,shear);
double  I1_trial,
        J2_trial,
        rJ2_trial;
Matrix3 S_trial;
computeInvariants(sigma_trial,S_trial,I1_trial,J2_trial,rJ2_trial);

// (2) Determine the number of substeps (nsub) based on the magnitude of
// the trial stress increment relative to the characteristic dimensions
// of the yield surface.  Also compare the value of the pressure dependent
// elastic properties as sigma_old and sigma_trial and adjust nsub if
// there is a large change to ensure an accurate solution for nonlinear
// elasticity even with fully elastic loading.
int nsub = computeStepDivisions(X_old,Zeta_old,ep_old,sigma_old,sigma_trial);

// (3) Compute a subdivided time step:
//
  stepDivide:
    dt = Dt/(chi*nsub);

// (4) Set {sigma_old,X_old,Zeta_old,ep_old}={sigma_n,X_n,Zeta_n,ep_n} to their
// values at the start of the step, and set n = 1:
  sigma_old = sigma_n;
  X_old     = X_n;
  Zeta_old  = Zeta_n;
  damage_old = damage_n;
  ep_old    = ep_n;
  n = 1;

// (5) Call substep function {sigma_new,ep_new,X_new,Zeta_new}
//                               = computeSubstep(D,dt,sigma_old,ep_old,X_old,Zeta_old)
computeSubstep:
  substepFlag = computeSubstep(D,dt,sigma_old,ep_old,X_old,Zeta_old,damage_old,
                               sigma_new,ep_new,X_new,Zeta_new,damage_new);

// (6) Check error flag from substep calculation:
  if (substepFlag == 0) { // no errors in substep calculation
    if (n < (chi*nsub)) { // update and keep substepping
      sigma_old = sigma_new;
      X_old     = X_new;
      Zeta_old  = Zeta_new;
      damage_old= damage_new;
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
  damage_p  = damage_new;
  ep_p      = ep_new;
  stepFlag  = 0;
  return stepFlag;

// (8) Failed step, Send ParticleDelete Flag to Host Code, Store Inputs to particle data:
failedStep:
  // input values for sigma_new,X_new,Zeta_new,ep_new, along with error flag
  sigma_p   = sigma_n;
  X_p       = X_n;
  Zeta_p    = Zeta_n;
  damage_p  = damage_n;
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
  double  b0 = d_cm.B0,
          b1 = d_cm.B1,
          g0 = d_cm.G0;

  shear   = g0;       // Shear Modulus
  bulk    = b0 + b1;  // Bulk Modulus
} //===================================================================

// [shear,bulk] = computeElasticProperties(stress, ep)
void Arenisca3::computeElasticProperties(const Matrix3 stress,
    const Matrix3 ep,
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
          Kf  = d_cm.fluid_B0,
          I1 = stress.Trace(),
          evp = ep.Trace();

// SHEAR MODULUS -------------------------------------------------------
// Need to modify this to support nonlinear elasticity, but currently do
// not have parameterization for this feature
  shear = g0;     // Shear Modulus

// BULK MODULUS -------------------------------------------------------
// The low pressure bulk modulus is also used for the tensile response.
  bulk = b0;
  if(evp <= 0.0){// ...................................................Drained
    if (I1 < 0.0){bulk = bulk + b1*exp(b2/I1);}
    // Elastic-plastic coupling
    if (evp < 0.0){bulk = bulk - b3*exp(b4/evp);}
  }
// In compression the low pressure modulus is modified by pressure,
// plastic-strain, and fluid effects:
  double ev0 = computeev0();

// In  compression, or with fluid effects if the strain is more compressive
// than the zero fluid pressure volumetric strain:
  if (evp <= ev0 && Kf!=0.0){// ..........................................................Undrained

    // Compute the porosity from the strain using Homel's simplified model, and
    // then use this in the Biot-Gassmann formula to compute the bulk modulus.

    // The dry bulk modulus, taken as the low pressure limit of the nonlinear
    // formulation:
    double Kd = b0;
    if (evp < 0.0){Kd = b0 - b3*exp(b4/evp);}

    // The grain bulk modulus, interpreted as the high pressure limit of the
    // nonlinear elastic fit to the drained material.
    double Km = b0 + b1;

    // initial porosity, inferred from the p3 parameter in the crush curve

    double phi_i = 1.0 - exp(-d_cm.p3_crush_curve);

    // Current unloaded porosity (phi):
    double C1 = Kf*(1.0 - phi_i) + Km*(phi_i);  // term to simplify the expression below
    double phi = exp(evp*Km/C1)*phi_i/(-exp(evp*Kf/C1)*(phi_i-1.0) + exp(evp*Km/C1)*phi_i);

    // Biot-Gassmann formula for the saturated bulk modulus, evaluated at the
    // current porosity.  This introduces some error since the Kd term is a
    // function of the initial porosity, but since large strains would also
    // modify the bulk modulus through damage
    bulk = Kd + (1.0 - Kd/Km)*(1.0 - Kd/Km)/((1.0 - Kd/Km - phi)/Km + (1.0/Kf - 1.0/Km)*phi);
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
  Matrix3 Identity;             // Identity tensor
  Identity.Identity();  // Initialize identity tensor
  Matrix3 d_e_iso = one_third*d_e.Trace()*Identity;
  Matrix3 d_e_dev = d_e - d_e_iso;
  Matrix3 sigma_trial = sigma_old + (3.0*bulk*d_e_iso + 2.0*shear*d_e_dev);
  return sigma_trial;
} //===================================================================

// [nsub] = computeStepDivisions(X,Zeta,ep,sigma_n,sigma_trial)
int Arenisca3::computeStepDivisions(const double& X,
                                    const double& Zeta,
                                    const Matrix3& ep,
                                    const Matrix3& sigma_n,
                                    const Matrix3& sigma_trial)
{
// compute the number of step divisions (substeps) based on a comparison
// of the trial stress relative to the size of the yield surface, as well
// as change in elastic properties between sigma_n and sigma_trial.
  double PEAKI1 = d_cm.PEAKI1,
         FSLOPE = d_cm.FSLOPE;

  Matrix3 d_sigma = sigma_trial - sigma_n;

  double  bulk_n,shear_n,bulk_trial,shear_trial;
  computeElasticProperties(sigma_n,ep,bulk_n,shear_n);
  computeElasticProperties(sigma_trial,ep,bulk_trial,shear_trial);

  int n_bulk = ceil(abs(bulk_n-bulk_trial)/bulk_n),
      n_iso = ceil(.03125*abs(d_sigma.Trace())/(PEAKI1-X)),
      n_dev = ceil(.0625*d_sigma.Norm()/(FSLOPE*(PEAKI1-X)));

  int nsub = max(max(n_bulk,n_iso),n_dev);
#ifdef MHdebug
  if (nsub>256){    cout<<"stepDivide out of range. nsub = "<<nsub<<endl;}
#endif
  nsub = min(max(nsub,1),256);
  return nsub;
} //===================================================================

void Arenisca3::computeInvariants(const Matrix3& stress,
                                  Matrix3& S,
                                  double & I1,
                                  double & J2,
                                  double & rJ2)
{
  // Compute the invariants of a second-order tensor
  Matrix3 Identity;
  Identity.Identity();

  // Compute the first invariants
  I1 = stress.Trace();  //Pa

  // Compute the deviatoric part of the tensor
  S = stress - Identity*(I1/3.0);  //Pa

  // Compute the second invariant
  J2 = 0.5*S.Contract(S);  //Pa^2

  if(J2 < 1e-16*(I1*I1+J2)){
    J2=0.0;
  };
  rJ2 = sqrt(J2);
} //===================================================================

// Computes the updated stress state for a substep
int Arenisca3::computeSubstep(const Matrix3& D,         // Strain "rate"
                              const double & dt,        // time substep (s)
                              const Matrix3& sigma_old, // stress at start of substep
                              const Matrix3& ep_old,    // plastic strain at start of substep
                              const double & X_old,     // hydrostatic compressive strength at start of substep
                              const double & Zeta_old,  // trace of isotropic backstress at start of substep
                              const double & damage_old, // XXX damage at start of substep
                              Matrix3& sigma_new, // stress at end of substep
                              Matrix3& ep_new,    // plastic strain at end of substep
                              double & X_new,     // hydrostatic compressive strength at end of substep
                              double & Zeta_new,   // trace of isotropic backstress at end of substep
                              double & damage_new // XXX
                             )
{
// Computes the updated stress state for a substep that may be either elastic, plastic, or
// partially elastic.   Returns an integer flag 0/1 for a good/bad update.
  int     substepFlag,
          returnFlag;
  double  p3  = d_cm.p3_crush_curve;

// (1)  Compute the elastic properties based on the stress and plastic strain at
// the start of the substep.  These will be constant over the step unless elastic-plastic
// is used to modify the tangent stiffness in the consistency bisection iteration.
  double bulk,
         shear;
  computeElasticProperties(sigma_old,ep_old,bulk,shear);

// (2) Compute the increment in total strain:
  Matrix3 d_e = D*dt;

// (3) Compute the trial stress: [sigma_trail] = computeTrialStress(sigma_old,d_e,K,G)
  Matrix3 sigma_trial = computeTrialStress(sigma_old,d_e,bulk,shear),
          S_trial;

  double I1_trial,
         J2_trial,
         rJ2_trial;
  computeInvariants(sigma_trial,S_trial,I1_trial,J2_trial,rJ2_trial);

// (4) Evaluate the yield function at the trial stress:
  int YIELD = computeYieldFunction(I1_trial,rJ2_trial,X_old,Zeta_old,damage_old);
  if (YIELD == -1) { // elastic substep
    sigma_new = sigma_trial;
    X_new = X_old;
    Zeta_new = Zeta_old;
    damage_new = damage_old;
    ep_new = ep_old;
    substepFlag = 0;
    goto successfulSubstep;
  }
  if (YIELD == 1) {  // elastic-plastic or fully-plastic substep
// (5) Compute non-hardening return to initial yield surface:
//     [sigma_0,d_e_p,0] = (nonhardeningReturn(sigma_trial,sigma_old,X_old,Zeta_old,K,G)
    double  I1_0,       // I1 at stress update for non-hardening return
            rJ2_0,      // rJ2 at stress update for non-hardening return
            TOL = 1e-5; // bisection convergence tolerance on eta
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
    returnFlag = nonHardeningReturn(I1_trial,rJ2_trial,S_trial,I1_old,rJ2_old,S_old,
                                    d_e,X_old,Zeta_old,damage_old,bulk,shear,
                                    I1_0,rJ2_0,S_0,d_ep_0);

    double d_evp_0 = d_ep_0.Trace();

// (6) Iterate to solve for plastic volumetric strain consistent with the updated
//     values for the cap (X) and isotropic backstress (Zeta).  Use a bisection method
//     based on the multiplier eta,  where  0<eta<1
    double eta_out = 1.0,
           eta_in = 0.0,
           eta_mid,
           d_evp;
    int i = 0,
        imax=100;
    double dZetadevp = computedZetadevp(Zeta_old,evp_old);

// (7) Update Internal State Variables based on Last Non-Hardening Return:
//
updateISV:
    i++;
    eta_mid   = 0.5*(eta_out+eta_in);
    d_evp     = eta_mid*d_evp_0;

    if(evp_old + d_evp <= -p3 ){
      eta_out = eta_mid;
      if( i >= imax ){ // solution failed to converge
#ifdef MHdebug
        cout << "1296: i>=imax, failed substep (evp_old + d_evp <= -p3) "<< endl;
#endif
        goto failedSubstep;
      }
      goto updateISV;
    }

    // Update X exactly
    X_new     = computeX(evp_old + d_evp);
    // Update zeta. min() eliminates tensile fluid pressure from explicit integration error
    Zeta_new = min(Zeta_old + dZetadevp*d_evp,0.0);

// (8) Check if the updated yield surface encloses trial stres.  If it does, there is too much
//     plastic strain for this iteration, so we adjust the bisection parameters and recompute
//     the state variable update.
    if( computeYieldFunction(I1_trial,rJ2_trial,X_new,Zeta_new,damage_old)!=1 ){
      eta_out = eta_mid;
      if( i >= imax ){                                        // solution failed to converge
#ifdef MHdebug
        cout << "1310: i>=imax, (yield surface encloses trial stress) failed substep "<< endl;
#endif
        goto failedSubstep;
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
    int returnFlag = nonHardeningReturn(I1_trial,rJ2_trial,S_trial,
                                        I1_old,rJ2_old,S_old,
                                        d_e,X_new,Zeta_new,damage_old,bulk,shear,
                                        I1_new,rJ2_new,S_new,d_ep_new);

// (10) Check whether the isotropic component of the return has changed sign, as this
//      would indicate that the cap apex has moved past the trial stress, indicating
//      too much plastic strain in the return.

    //if(abs(I1_trial - I1_new)>(d_cm.B0*TOL) && Sign(I1_trial - I1_new)!=Sign(I1_trial - I1_0)){
    if(Sign(I1_trial - I1_new)!=Sign(I1_trial - I1_0)){
      eta_out = eta_mid;
      if( i >= imax ){                                        // solution failed to converge
#ifdef MHdebug
        cout << "1346: i>=imax, (isotropic return changed sign) failed substep "<< endl;
#endif
        goto failedSubstep;
      }
      goto updateISV;
    }
    // Good update, compare magnitude of plastic strain with prior update
    d_evp_new = d_ep_new.Trace();   // Increment in vol. plastic strain for return to new surface

    // Check for convergence
    if( abs(eta_out-eta_in) < TOL ){           // Solution is converged
      Matrix3 Identity;
      Identity.Identity();
      sigma_new = one_third*I1_new*Identity + S_new;
      ep_new = ep_old + d_ep_new;
      // Update X exactly
      X_new = computeX(ep_new.Trace());
      // Update zeta. min() eliminates tensile fluid pressure from explicit integration error
      Zeta_new = min(Zeta_old + dZetadevp*d_evp_new,0.0);

      // XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
      // XXX Damage XXX:
      // Recompute damage based on updated plastic strain.
      // compute a nonhardening return to the updated surface and store this
      // as the updated stress.
      // The plastic strain for this return is not used to evolve the state
      // variables.  This is certainly an error, but I suspect (though I still
      // need to show) that there will be multiple solutions for some cases
      // of combined hardening and softening).  If this is not the case, and
      // the combined problem is well posed, it will simply be necessary to
      // compute the damage above when X and Zeta are updated.
      //
      // Currently damage and porosity do not work together, because the plastic
      // strain is modeled after the correct X is computed, but the value for X
      // is not updated.  Thus on the next step it may not be possible to find
      // a solution based on the total plastic strain.  If an incremental update
      // to the cap function were used (rather than computing X(evp) exactly),
      // this would not be a problem, but then there might be problems with
      // nonphysical cap evolution.

      if( d_cm.STREN < 0.0) {
        damage_new = Min(1.0,damage_old + d_ep_new.Norm()/abs(d_cm.STREN));

        returnFlag = nonHardeningReturn(I1_new,rJ2_new,S_new,
                                        I1_old,rJ2_old,S_old,
                                        d_e,X_new,Zeta_new,damage_new,bulk,shear,
                                        I1_new,rJ2_new,S_new,d_ep_new);

        sigma_new = one_third*I1_new*Identity + S_new;
      }
      else {
        damage_new = 0.0;
      }
      // XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

      goto successfulSubstep;
    }
    if( i >= imax ){                                        // solution failed to converge
#ifdef MHdebug
      cout << "1306: i>=imax, failed substep "<< endl;
#endif
      goto failedSubstep;
    }

// (11) Compare magnitude of the volumetric plastic strain and bisect on eta
//
    if( abs(d_evp_new) > eta_mid*abs(d_evp_0) ){
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
double Arenisca3::computeX(double evp)
{
  // X is the value of (I1 - Zeta) at which the cap function crosses
  // the hydrostat. For the drained material in compression. X(evp)
  // is derived from the emprical Kayenta crush curve, but with p2 = 0.
  // In tension, M. Homel's piecewsie formulation is used.

  // define and initialize some variables
  double p0  = d_cm.p0_crush_curve,
         p1  = d_cm.p1_crush_curve,
         p3  = d_cm.p3_crush_curve, // max vol. plastic strain
         X;

  if(evp<=-p3)
  { // --------------------Plastic strain exceeds allowable limit--------------------------
    // The plastic strain for this iteration has exceed the allowable
    // value.  X is not defined in this region, so we set it to a large
    // negative number.  This will cause the plastic strain to be reduced
    // in subsequent iterations.
    //
    // MH!: This shouldn't be reached, but may allow for relaxed convergence
    // requirements (if this is reached it will be within an iteration on cap
    // position, and shouldn't end up as the final solution).
    X = 1.0e6*p0;
  }
  else
  { // --------------------Plastic strain is within allowable domain------------------------
    // We first compute the drained response.  If there are fluid effects, this value will
    // be used in detemining the elastic volumetric strain to yield.
    if(evp <= 0.0){
      X = (p0*p1 + log((evp+p3)/p3))/p1;
    }
    else{
      X = p0*Pow(1.0 + evp, 1.0/(p0*p1*p3));
    }

    double Kf  = d_cm.fluid_B0,       // fluid bulk modulus
           ev0 = computeev0();        // strain at zero pore pressure
    if(Kf!=0.0 && evp<=ev0)
    { // --------------------------------------------------------------------- Fluid Effects
      // First we evaluate the elastic volumetric strain to yield from the
      // empirical crush curve (Xfit) and bulk modulus (Kfit) formula for
      // the drained material.  Xfit was computed as X above.
      double b0 = d_cm.B0,
             b1 = d_cm.B1,
             b2 = d_cm.B2,
             b3 = d_cm.B3,
             b4 = d_cm.B4;

      // Kfit is the drained bulk modulus evaluated at evp, and for I1 = Xdry/2.
      double Kdry = b0;
      if (evp<=0.0){ // Pore Collapse
        Kdry = Kdry + b1*exp(2.0*b2/X);
        if (evp<0.0){Kdry = Kdry - b3*exp(b4/evp);}
      }

      // Now we use our engineering model for the bulk modulus of the
      // saturated material (Keng) to compute the stress at our elastic strain to yield.
      // Since the stress and plastic strain tensors are not available in this scope, we call the
      // computeElasticProperties function with and isotropic matrices that will have the
      // correct values of evp. (The saturated bulk modulus doesn't depend on I1).
      double Ksat,Gsat;       // Not used, but needed to call computeElasticProperties()
      Matrix3 Identity;
      Identity.Identity();    // Set this to the identity matrix
      // This needs to be evaluated at the current value of pressure.
      computeElasticProperties(one_sixth*X*Identity,one_third*evp*Identity,Ksat,Gsat); //Overwrites Geng & Keng

      // Compute the stress to hydrostatic yield.
      // We are only in this looop if(evp <= ev0)
      X = X*Ksat/Kdry;
    } // End fluid effects
  } // End plastic strain in allowable domain
  return X;
} //===================================================================

// Compute the strain at zero pore pressure from initial pore pressure (Pf0)
double Arenisca3::computeev0()
{
  // The user-supplied initial pore pressure (Pf0) is the pore pressure at zero
  // volumetric strain.  An estimate of the strain (ev0) at which the fluid pressure
  // is zero is derived from M. Homel's engineering model of matrix compressibility:

  //define and initialize some variables
  double Kf  = d_cm.fluid_B0,               // fluid bulk modulus
         pfi = d_cm.fluid_pressure_initial, // initial pore pressure
         ev0 = 0.0;                         // strain at zero pore pressure

  if(pfi!=0 && Kf!=0){ // Nonzero initial pore pressure
    double phi_i = 1.0 - exp(-d_cm.p3_crush_curve), // Initial porosity (inferred from crush curve)
           Km = d_cm.B0 + d_cm.B1;                  // Matrix bulk modulus (inferred from high pressure limit of drained bulk modulus)

    ev0 = (Kf*(1.0 - phi_i) + Km*phi_i)*pfi/(Kf*Km);
  }
  return ev0;
} //===================================================================

// Compute the strain at zero pore pressure from initial pore pressure (Pf0)
double Arenisca3::computePorePressure(const double ev)
{
  // This compute the plotting variable pore pressure, which is defined from
  // input paramters and the current total volumetric strain (ev).
  double Kf  = d_cm.fluid_B0,               // fluid bulk modulus
         ev0 = computeev0(),                // strain at zero pore pressure
         pf = 0.0;                          // pore fluid pressure

  if(ev<=ev0 && Kf!=0){ // ....................fluid effects are active
    double Km = d_cm.B0 + d_cm.B1,                   // Matrix bulk modulus (inferred from high pressure limit of drained bulk modulus)
           phi_i = 1.0 - exp(-d_cm.p3_crush_curve),  // Initial porosity (inferred from crush curve)
           pfi = d_cm.fluid_pressure_initial;        // initial pore pressure

    double C1 = Kf*(1.0 - phi_i) + Km*(phi_i);       // Term to simplify the expression below
    pf = pfi + Kf*log(exp(ev*(-1.0 - Km/C1))*(-exp((ev*Kf)/C1)*(phi_i-1.0) + exp((ev*Km)/C1)*phi_i));
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
                                  const double & damage,
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

  const double TOL = 1e-3;
  double theta = pi_half,
         n = 0.0;
  int interior,
      returnFlag;

// (1) Define an interior point, (I1_0 = Zeta, J2_0 = 0)
  double  I1_0 = Zeta,
          rJ2_0 = 0.0;

// (2) Transform the trial and interior points as follows where beta defines the degree
//  of non-associativity.
  double beta = d_cm.BETA_nonassociativity;  // MH! change this for nonassociativity in the meridional plane
  double fac = beta*sqrt(1.5*bulk/shear);
  double r_trial = fac*sqrt_two*rJ2_trial,
         z_trial = I1_trial*one_sqrt_three,
         z_test,
         r_test,
         r_0     = fac*sqrt_two*rJ2_0,
         z_0     = I1_0*one_sqrt_three;

// (3) Perform Bisection between in transformed space, to find the new point on the
//  yield surface: [znew,rnew] = transformedBisection(z0,r0,z_trial,r_trial,X,Zeta,K,G)
  //int icount=1;
  while ( abs(theta) > TOL ){
    // transformed bisection to find a new interior point, just inside the boundary of the
    // yield surface.  This function overwrites the inputs for z_0 and r_0
    //  [z_0,r_0] = transformedBisection(z_0,r_0,z_trial,r_trial,X_Zeta,bulk,shear)
    transformedBisection(z_0,r_0,z_trial,r_trial,X,Zeta,damage,bulk,shear);

// (4) Perform a rotation of {z_new,r_new} about {z_trial,r_trial} until a new interior point
// is found, set this as {z0,r0}
    interior = 0;
    n = max(n-2.0,0.0);
    // (5) Test for convergence:
    while ( (interior==0)&&(abs(theta)>TOL) ){
      //changed this to prevent the possibility of symmetric bouncing about a symmetric feature
      theta = (pi/2.0)*Pow(-1.0,n+2.0)*Pow(0.5,(n+2.0)/2.0);
      z_test = z_trial + cos(theta)*(z_0-z_trial) - sin(theta)*(r_0-r_trial);
      r_test = r_trial + sin(theta)*(z_0-z_trial) + cos(theta)*(r_0-r_trial);

      if ( transformedYieldFunction(z_test,r_test,X,Zeta,damage,bulk,shear) == -1 ) { // new interior point
        interior = 1;
        z_0 = z_test;
        r_0 = r_test;
      }
      else { n=n+1.0; }
    }
  }

// (6) Solution Converged, Compute Untransformed Updated Stress:
  I1_new = sqrt_three*z_0;
  rJ2_new = r_0/fac*one_sqrt_two;
  if ( rJ2_trial!=0.0 ){S_new = S_trial*rJ2_new/rJ2_trial;}
  else                 {S_new = S_trial;}
  Matrix3 Identity;
  Identity.Identity();
  Matrix3 sigma_new = one_third*I1_new*Identity + S_new,
          sigma_old = one_third*I1_old*Identity + S_old;
  Matrix3 d_sigma = sigma_new - sigma_old;

// (7) Compute increment in plastic strain for return:
//  d_ep0 = d_e - [C]^-1:(sigma_new-sigma_old)
  Matrix3 d_ee    = 0.5*d_sigma/shear + (one_ninth/bulk-one_sixth/shear)*d_sigma.Trace()*Identity;
  d_ep_new        = d_e - d_ee;

  //MH! add some error detection that returns a returnFlag=1
  returnFlag = 0;
  return returnFlag;
} //===================================================================

// Computes bisection between two points in transformed space
void Arenisca3::transformedBisection(double& z_0,
                                     double& r_0,
                                     const double& z_trial,
                                     const double& r_trial,
                                     const double& X,
                                     const double& Zeta,
                                     const double& damage,
                                     const double& bulk,
                                     const double& shear
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
  double eta_out=1.0,  // This is for the accerator.  Must be > TOL
         eta_in =0.0,
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
    if ( transformedYieldFunction(z_test,r_test,X,Zeta,damage,bulk,shear)!=1 ) {eta_in = eta_mid;}
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
                                        const double& damage,
                                        const double& bulk,
                                        const double& shear
                                       )
{
// Evaluate the yield criteria and return:
//  -1: elastic
//   0: on yield surface within tolerance
//   1: plastic

// Untransformed values:
  double beta = d_cm.BETA_nonassociativity;  //MH! modify this for nonassociativity.
  double fac = beta*sqrt(1.5*bulk/shear);
  double I1  = sqrt_three*z,
         rJ2 = (r/fac)*one_sqrt_two;
  int    YIELD = computeYieldFunction(I1,rJ2,X,Zeta,damage);
  return YIELD;
} //===================================================================

// computeYieldFunction from untransformed inputs
int Arenisca3::computeYieldFunction(const double& I1,
                                    const double& rJ2,
                                    const double& X,
                                    const double& Zeta,
                                    const double& damage
                                   )
{
  // Evaluate the yield criteria and return:
  //  -1: elastic
  //   0: on yield surface within tolerance
  //   1: plastic
  int YIELD = -1;
  double I1mZ = I1 - Zeta;    // Shifted stress to evalue yield criteria

// --------------------------------------------------------------------
// *** SHEAR LIMIT FUNCTION ***
// --------------------------------------------------------------------
  // Read input parameters to specify strength model
  double  FSLOPE = d_cm.FSLOPE,
          //STREN = d_cm.STREN,    //MH! add this user input
          YSLOPE = d_cm.YSLOPE,  //MH! add this user input
          PEAKI1 = d_cm.PEAKI1,
          Ff;

  // Damage
  FSLOPE = (1.0-damage)*d_cm.FSLOPE + damage*d_cm.YSLOPE;
  PEAKI1 = (1.0-damage)*d_cm.PEAKI1;

  //if (FSLOPE == 0.0) {// VON MISES-------------------------------------
  //  // If the user has specified an input set with FSLOPE = 0, this indicates
  //  // a von Mises plasticity model should be used.  In this case, the yield
  //  // stress is the input value for PEAKI1.
  //  // Need to modify the compute substepdivisions as well for the von mises case
  //  if( abs(rJ2) > PEAKI1 ) {YIELD=1;}
  //  return YIELD;
  //}
  if (YSLOPE == FSLOPE){// LINEAR DRUCKER-PRAGER SURFACE----------
    // If the user has specified an input set with FLSOPE=YSLOPE!=0, this
    // indicates a linear Drucker-Prager shear strength model.
    Ff = FSLOPE*(PEAKI1 - I1mZ);
  }
  else { //MH! hack until nonlinear is written in:
    Ff = FSLOPE*(PEAKI1 - I1mZ);
  }
  //else{// NONLINEAR DRUCKER PRAGER-------------------------------------
  //  // The general case for a non-linear Drucker-Prager surface.  We will
  //  // compute the a_i parameters from the user inputs and the evaluate the
  //  // non lniear shear limit function.
  //
  //  //MH! fix these...
  //  double a1 = STREN + 2*I1*YSLOPE,
  //         a2 = ProductLog(a2*Pow(E,a2*PEAKI1)*PEAKI1)/PEAKI1,
  //         a3 = (FSLOPE - YSLOPE)/(a2*Pow(E,a2*PEAKI1)),
  //         a4 = YSLOPE;
  //
  //  double  Ff = a1 - a3*Pow(E,a2*I1mZ) - a4*I1mZ;
  //}

// --------------------------------------------------------------------
// *** CAP FUNCTION ***
// --------------------------------------------------------------------
  double  p3  = d_cm.p3_crush_curve,
          CR  = d_cm.CR;
  double  Kappa  = PEAKI1-CR*(PEAKI1-X),
          fc = 1.0;

  if (p3 == 0.0){// No Cap---------------------------------------------
    // p3 is the maximum achievable volumetric plastic strain in compresson
    // so if a value of 0 has been specified this indicates the user
    // wishes to run without porosity, and no cap function is used.
    fc = 1.0;
  }
  else if( ( I1mZ < Kappa )&&( I1mZ >= X ) ){// Elliptical Cap Function
    fc = sqrt(1.0-Pow((Kappa-I1mZ)/(Kappa-X),2.0));
  }

// --------------------------------------------------------------------
// *** COMPOSITE YIELD FUNCTION ***
// --------------------------------------------------------------------
  // Evaluate Composite Yield Function F(I1) = Ff(I1)*fc(I1) in each region
  if( I1mZ<X ){//---------------------------------------------------(I1<X)
    YIELD=1;
  }

  else if( ( I1mZ < Kappa )&&( I1mZ >= X ) ){// -------------(X<I1< kappa)
    if( abs(rJ2) > Ff*fc ) {YIELD=1;}
    //else if(abs(rJ2)==Ff*fc){YIELD=0;}
  }

  else if( ( I1mZ <= PEAKI1 )&&( I1mZ >= Kappa ) ){// ---(kappa<I1<PEAKI1)
    if( abs(rJ2) > Ff ) {YIELD=1;}
    //else if(abs(rJ2)==Ff){YIELD=0;}
  }

  else if( I1mZ > PEAKI1 ) {// -------------------------------(peakI1<I1)
    YIELD=1;
  };

  return YIELD;
} //===================================================================

// Compute (dZeta/devp) Zeta and vol. plastic strain
double Arenisca3::computedZetadevp(double Zeta, double evp)
{
  // Computes the partial derivative of the trace of the
  // isotropic backstress (Zeta) with respect to volumetric
  // plastic strain (evp).
  //
  // From M. Homel's engineering model for matrix compressibility:
  double Kf  = d_cm.fluid_B0,       // fluid bulk modulus
         ev0  = computeev0(),       // volumetric strain at zero pore pressure
         dZetadevp = 0.0;           // Evolution rate of isotorpic backstress

  if (evp <= ev0 && Kf != 0.0) { // ............................................ Fluid effects are active
    double pfi = d_cm.fluid_pressure_initial,      // initial fluid pressure
           phi_i = 1.0 - exp(-1.0*d_cm.p3_crush_curve), // Initial porosity (inferred from crush curve
           Km = d_cm.B0 + d_cm.B1;                 // Matrix bulk modulus (inferred from high pressure elastic modulus)

    dZetadevp = (3.0*exp(evp)*Kf*Km)/(exp(evp)*(Kf + Km) + exp(Zeta/(3.0*Km))*Km*(-1.0 + phi_i) - exp((3.0*pfi + Zeta)/(3.0*Kf))*Kf*phi_i);
  }
  return dZetadevp;
  //
} //===================================================================


// ****************************************************************************************************
// ****************************************************************************************************
// ************** PUBLIC Uintah MPM constitutive model specific functions *****************************
// ****************************************************************************************************
// ****************************************************************************************************

void Arenisca3::addRequiresDamageParameter(Task* task,
    const MPMMaterial* matl,
    const PatchSet* ) const
{
  // Require the damage parameter
  const MaterialSubset* matlset = matl->thisMaterial();//T2D; what is this?
  task->requires(Task::NewDW, pLocalizedLabel_preReloc,matlset,Ghost::None);
}

void Arenisca3::getDamageParameter(const Patch* patch,
                                   ParticleVariable<int>& damage,
                                   int dwi,
                                   DataWarehouse* old_dw,
                                   DataWarehouse* new_dw)
{
  // Get the damage parameter
  ParticleSubset* pset = old_dw->getParticleSubset(dwi,patch);
  constParticleVariable<int> pLocalized;
  new_dw->get(pLocalized, pLocalizedLabel_preReloc, pset);

  ParticleSubset::iterator iter;
  // Loop over the particle in the current patch.
  for (iter = pset->begin(); iter != pset->end(); iter++) {
    damage[*iter] = pLocalized[*iter];
  }
}

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
      new_dw->put(sum_vartype(0.),     lb->StrainEnergyLabel);
    }
  }
}

//When a particle is pushed from patch to patch, carry information needed for the particle
void Arenisca3::addParticleState(std::vector<const VarLabel*>& from,
                                 std::vector<const VarLabel*>& to)
{
  // Push back all the particle variables associated with Arenisca.
  // Important to keep from and to lists in same order!
#ifdef MH_VARIABILITY
  from.push_back(peakI1IDistLabel);
#endif
  from.push_back(pLocalizedLabel);
  from.push_back(pAreniscaFlagLabel);
  from.push_back(pScratchDouble1Label);
  from.push_back(pScratchDouble2Label);
  from.push_back(pPorePressureLabel);
  from.push_back(pepLabel);
  from.push_back(pevpLabel);
  from.push_back(peveLabel);
  from.push_back(pCapXLabel);
  from.push_back(pCapXQSLabel);
  from.push_back(pKappaLabel);
  from.push_back(pZetaLabel);
  from.push_back(pZetaQSLabel);
  from.push_back(pIotaLabel);
  from.push_back(pIotaQSLabel);
  from.push_back(pStressQSLabel);
  from.push_back(pScratchMatrixLabel);
#ifdef MH_VARIABILITY
  to.push_back(  peakI1IDistLabel_preReloc);
#endif
  to.push_back(  pLocalizedLabel_preReloc);
  to.push_back(  pAreniscaFlagLabel_preReloc);
  to.push_back(  pScratchDouble1Label_preReloc);
  to.push_back(  pScratchDouble2Label_preReloc);
  to.push_back(  pPorePressureLabel_preReloc);
  to.push_back(  pepLabel_preReloc);
  to.push_back(  pevpLabel_preReloc);
  to.push_back(  peveLabel_preReloc);
  to.push_back(  pCapXLabel_preReloc);
  to.push_back(  pCapXQSLabel_preReloc);
  to.push_back(  pKappaLabel_preReloc);
  to.push_back(  pZetaLabel_preReloc);
  to.push_back(  pZetaQSLabel_preReloc);
  to.push_back(  pIotaLabel_preReloc);
  to.push_back(  pIotaQSLabel_preReloc);
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
#ifdef MH_VARIABILITY
  task->computes(peakI1IDistLabel, matlset);
#endif
  task->computes(pLocalizedLabel,      matlset);
  task->computes(pAreniscaFlagLabel,   matlset);
  task->computes(pScratchDouble1Label, matlset);
  task->computes(pScratchDouble2Label, matlset);
  task->computes(pPorePressureLabel,   matlset);
  task->computes(pepLabel,             matlset);
  task->computes(pevpLabel,            matlset);
  task->computes(peveLabel,            matlset);
  task->computes(pCapXLabel,           matlset);
  task->computes(pCapXQSLabel,         matlset);
  task->computes(pKappaLabel,          matlset);
  task->computes(pZetaLabel,           matlset);
  task->computes(pZetaQSLabel,         matlset);
  task->computes(pIotaLabel,           matlset);
  task->computes(pIotaQSLabel,         matlset);
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
#ifdef MH_VARIABILITY
  task->requires(Task::OldDW, peakI1IDistLabel,     matlset, Ghost::None);
#endif
  task->requires(Task::OldDW, pLocalizedLabel,      matlset, Ghost::None);
  task->requires(Task::OldDW, pAreniscaFlagLabel,   matlset, Ghost::None);
  task->requires(Task::OldDW, pScratchDouble1Label, matlset, Ghost::None);
  task->requires(Task::OldDW, pScratchDouble2Label, matlset, Ghost::None);
  task->requires(Task::OldDW, pPorePressureLabel,   matlset, Ghost::None);
  task->requires(Task::OldDW, pepLabel,             matlset, Ghost::None);
  task->requires(Task::OldDW, pevpLabel,            matlset, Ghost::None);
  task->requires(Task::OldDW, peveLabel,            matlset, Ghost::None);
  task->requires(Task::OldDW, pCapXLabel,           matlset, Ghost::None);
  task->requires(Task::OldDW, pCapXQSLabel,         matlset, Ghost::None);
  task->requires(Task::OldDW, pKappaLabel,          matlset, Ghost::None);
  task->requires(Task::OldDW, pZetaLabel,           matlset, Ghost::None);
  task->requires(Task::OldDW, pZetaQSLabel,         matlset, Ghost::None);
  task->requires(Task::OldDW, pIotaLabel,           matlset, Ghost::None);
  task->requires(Task::OldDW, pIotaQSLabel,         matlset, Ghost::None);
  task->requires(Task::OldDW, pStressQSLabel,       matlset, Ghost::None);
  task->requires(Task::OldDW, pScratchMatrixLabel,  matlset, Ghost::None);
  task->requires(Task::OldDW, lb->pParticleIDLabel, matlset, Ghost::None);
#ifdef MH_VARIABILITY
  task->computes(peakI1IDistLabel_preReloc,     matlset);
#endif
  task->computes(pLocalizedLabel_preReloc,      matlset);
  task->computes(pAreniscaFlagLabel_preReloc,   matlset);
  task->computes(pScratchDouble1Label_preReloc, matlset);
  task->computes(pScratchDouble2Label_preReloc, matlset);
  task->computes(pPorePressureLabel_preReloc,   matlset);
  task->computes(pepLabel_preReloc,             matlset);
  task->computes(pevpLabel_preReloc,            matlset);
  task->computes(peveLabel_preReloc,            matlset);
  task->computes(pCapXLabel_preReloc,           matlset);
  task->computes(pCapXQSLabel_preReloc,         matlset);
  task->computes(pKappaLabel_preReloc,          matlset);
  task->computes(pZetaLabel_preReloc,           matlset);
  task->computes(pZetaQSLabel_preReloc,         matlset);
  task->computes(pIotaLabel_preReloc,           matlset);
  task->computes(pIotaQSLabel_preReloc,         matlset);
  task->computes(pStressQSLabel_preReloc,       matlset);
  task->computes(pScratchMatrixLabel_preReloc,  matlset);
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

  rho_cur = rho_orig/(1-p_gauge/bulk);

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

  double p_g = .5*bulk*(rho_cur/rho_orig - rho_orig/rho_cur);
  pressure = p_ref + p_g;
  dp_drho  = .5*bulk*(rho_orig/(rho_cur*rho_cur) + 1./rho_orig);
  tmp = (bulk + 4.*shear/3.)/rho_cur;  // speed of sound squared

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
#ifdef MH_VARIABILITY
  //peakI1Dist
  peakI1IDistLabel = VarLabel::create("p.peakI1IDist",
                                      ParticleVariable<double>::getTypeDescription());
  peakI1IDistLabel_preReloc = VarLabel::create("p.peakI1IDist+",
                              ParticleVariable<double>::getTypeDescription());
#endif
  //pLocalized
  pLocalizedLabel = VarLabel::create("p.localized",
                                     ParticleVariable<int>::getTypeDescription());
  pLocalizedLabel_preReloc = VarLabel::create("p.localized+",
                             ParticleVariable<int>::getTypeDescription());
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
  //pKappa
  pKappaLabel = VarLabel::create("p.Kappa",
                                 ParticleVariable<double>::getTypeDescription());
  pKappaLabel_preReloc = VarLabel::create("p.Kappa+",
                                          ParticleVariable<double>::getTypeDescription());
  //pCapX
  pCapXLabel = VarLabel::create("p.CapX",
                                ParticleVariable<double>::getTypeDescription());
  pCapXLabel_preReloc = VarLabel::create("p.CapX+",
                                         ParticleVariable<double>::getTypeDescription());
  //pCapXQS
  pCapXQSLabel = VarLabel::create("p.CapXQS",
                                  ParticleVariable<double>::getTypeDescription());
  pCapXQSLabel_preReloc = VarLabel::create("p.CapXQS+",
                          ParticleVariable<double>::getTypeDescription());
  //pZeta
  pZetaLabel = VarLabel::create("p.Zeta",
                                ParticleVariable<double>::getTypeDescription());
  pZetaLabel_preReloc = VarLabel::create("p.Zeta+",
                                         ParticleVariable<double>::getTypeDescription());
  //pZetaQS
  pZetaQSLabel = VarLabel::create("p.ZetaQS",
                                  ParticleVariable<double>::getTypeDescription());
  pZetaQSLabel_preReloc = VarLabel::create("p.ZetaQS+",
                          ParticleVariable<double>::getTypeDescription());
  //pIota
  pIotaLabel = VarLabel::create("p.Iota",
                                ParticleVariable<double>::getTypeDescription());
  pIotaLabel_preReloc = VarLabel::create("p.Iota+",
                                         ParticleVariable<double>::getTypeDescription());
  //pIotaQS
  pIotaQSLabel = VarLabel::create("p.IotaQS",
                                  ParticleVariable<double>::getTypeDescription());
  pIotaQSLabel_preReloc = VarLabel::create("p.IotaQS+",
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

#ifdef MH_VARIABILITY
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
  for ( int i = iP.WeibDist.length()-1; i >= 0; i--) {
    iP.WeibDist[i] = tolower(iP.WeibDist[i]);
    if ( !isalnum(iP.WeibDist[i]) &&
         iP.WeibDist[i] != '.' &&
         iP.WeibDist[i] != ',' &&
         iP.WeibDist[i] != '-' &&
         iP.WeibDist[i] != EOF) {
      iP.WeibDist.erase(i,1);
    }
  } // End for
  if (iP.WeibDist.substr(0,4) == "weib") {
    iP.Perturb = true;
  } else {
    iP.Perturb = false;
  }
  // ######
  // If perturbation is NOT desired
  // ######
  if ( !iP.Perturb ) {
    bool escape = false;
    int num_of_e = 0;
    int num_of_periods = 0;
    for ( unsigned int i = 0; i < iP.WeibDist.length(); i++) {
      if ( iP.WeibDist[i] != '.'
           && iP.WeibDist[i] != 'e'
           && iP.WeibDist[i] != '-'
           && !isdigit(iP.WeibDist[i]) ) escape = true;
      if ( iP.WeibDist[i] == 'e' ) num_of_e += 1;
      if ( iP.WeibDist[i] == '.' ) num_of_periods += 1;
      if ( num_of_e > 1 || num_of_periods > 1 || escape ) {
        std::cerr << "\n\nERROR:\nInput value cannot be parsed. Please\n"
                  "check your input values.\n" << std::endl;
        exit (1);
      }
    } // end for(int i = 0;....)
    if ( escape ) exit (1);
    iP.WeibMed  = atof(iP.WeibDist.c_str());
  }
  // ######
  // If perturbation IS desired
  // ######
  if ( iP.Perturb ) {
    int weibValues[4];
    int weibValuesCounter = 0;
    for ( unsigned int r = 0; r < iP.WeibDist.length(); r++) {
      if ( iP.WeibDist[r] == ',' ) {
        weibValues[weibValuesCounter] = r;
        weibValuesCounter += 1;
      } // end if(iP.WeibDist[r] == ',')
    } // end for(int r = 0; ...... )
    if (weibValuesCounter != 4) {
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
    //T2D: is this needed? d_cm.PEAKI1=iP.WeibMed;  // Set this here to satisfy KAYENTA_CHK
  } // End if (iP.Perturb)
}
#endif
