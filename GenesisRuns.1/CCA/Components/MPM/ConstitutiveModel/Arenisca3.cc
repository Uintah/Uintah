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
2. R.M. Brannon and S. Leelavanichkul, "A multi-stage return algorithm for
   solving the classical damage component of constitutive models for rocks,
   ceramics, and other rock-like media", International Journal of Fracture,
   163, pp.133-149, 2010, and
3. R.M. Brannon, "Elements of Phenomenological Plasticity: Geometrical Insight,
   Computational Algorithms, and Topics in Shock Physics", Shock Wave Science
   and Technology Reference Library: Solids I, Springer 2: pp. 189-274, 2007.

As shown in "fig:AreniscaYieldSurface" of the Arenisca manual, Arenisca is
a two-surface plasticity model combining a linear Drucker-Prager
pressure-dependent strength (to model influence of friction at microscale
sliding surfaces) and a cap yield function (to model influence of microscale
porosity).

*/

/* NOTES
  1207 code control passed from Dr. Ali to James Colovos

  This is VERSION 0.1 120826.1339
*/

//----------------suggested max line width (72char)-------------------->

//----------JC DEFINE SECTION----------
#define JC_ZETA_HARDENING
#define JC_KAPPA_HARDENING
//#define JC_ARENISCA_VERSION 0.1  //120826.1339
//#define JC_ARENISCA_VERSION 0.2  //120826.0827
#define JC_ARENISCA_VERSION 3.0  //121215.2310 JC & MH
#define JC_USE_BB_DEFGRAD_UPDATE 2
//#define CSM_PRESSURE_STABILIZATION
//#define CSM_PORE_PRESSURE_INITIAL
//#define JC_DEBUG_SMALL_TIMESTEP
//#define JC_DEBUG_PARTICLE 562958543486976 //Test 02_uniaxialstrainrotate
//#define JC_DEBUG_PARTICLE 562958543486976 //Test 01_uniaxialstrainrotate
//#define JC_DEBUG_DEBUG_PARTICLE 42501762304
//#define JC_EPV
#define JC_FREEZE_PARTICLE
//#define JC_MAX_NESTED_RETURN
//#define CSM_DEBUG_BISECTION
//#define JC_LIMITER_PRINT
//#define CSM_FORCE_MASSIVE_SUBCYCLING 10

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
#include <iostream>

using std::cerr;

using namespace Uintah;
using namespace std;

// Requires the necessary input parameters CONSTRUCTORS
Arenisca3::Arenisca3(ProblemSpecP& ps, MPMFlags* Mflag)
  : ConstitutiveModel(Mflag)
{
  cout << "In Arenisca ver 3.0"<< endl;
  //cout << endl
  //     << "                                        ;1BB@B@B@@@B@8u:                        " << endl
  //     << "                                   .Y@@@B@B@BB8GZMB@B@B@B@Mr                    " << endl
  //     << "                                 Y@B@BB7.              :S@@B@Bi                 " << endl
  //     << "                       BB.     EB@BG.                      rB@B@L               " << endl
  //     << "                    iBOF@    5@B@L                            NB@Bi             " << endl
  //     << "                     OB G  :B@Bv                                O@BM            " << endl
  //     << "                   .@B@B@B@B@B  ;irr77777r77vL, .Yv77777777r7rr  :@B@           " << endl
  //     << "                    B@BZS@@@2  :BBMMMMMMMMMMM@i ,@BMOMOMMMMMM@B    @@@          " << endl
  //     << "                   L@B  i@@L   ,@E0q0PNPqPNPGB: .BGP0PNP0P0P08O     @B@         " << endl
  //     << "                 uB5B. ,B@X    :B8qqXXSXSXkPNB: .@EqkXkXXPkPqO8      @@@        " << endl
  //     << "                @Z BZ  B@B     i@M8PqkPkXkqPOBr :BMNPkXkPkPPGB@      v@Bi       " << endl
  //     << "              ;@r BN  7@B:        r8XXSPSPXZ5     :8PXkPkXkZU         B@B       " << endl
  //     << "             2@  u@   @B@         iONkPkPkqG1     .M0kPSPkqGu         XB@       " << endl
  //     << "            F@  :@    B@P         rMPXkXkXXOS     .BqqkXkXXO1         :@@i      " << endl
  //     << "           Y@   @v    @@L         7MNSXkPXNGX     ,M0kPkXSN8F         .B@7      " << endl
  //     << "          :@    B: v  B@7         rMPPSXkXPOk     ,BqXkPSPPO5         .@@7      " << endl
  //     << "          @r   @@  B. @BX         7ONkXSXXq8k     ,M0kXkXXNGS         rB@.      " << endl
  //     << "         @B  .BG   @. B@B         7MqPkPkXXOF     .BqPkXSPPO1         O@B       " << endl
  //     << "        :B   B@       uB@.        7MNkPSPkqG5     .O0kXkXSN8F         @BN       " << endl
  //     << "        BL   LB   E:   @@@        rMqPkXkPkG2     ,OPPSPkXPO5        MB@        " << endl
  //     << "       r@     @  u@Z   :@BY       7M0XPSPSXXZOBBBMONqSPSPk0ME       7B@v        " << endl
  //     << "       @v    .   @B     B@B7      v@ENXPSPkqX00Z0EPPSXkPXEO@i      i@@Z         " << endl
  //     << "      :B     GM  OM    B@0@Bu      J@80XPkPkPXqXPkqkqkqXZMZ       vB@8          " << endl
  //     << "      BM     B@  :B    .B i@BB      .OM800N0qEq0q0q0qE0OBY       MB@1           " << endl
  //     << "      @.     B    @,    Gq .@B@v      Y@@BBMBMBBBMBMBB@M,      L@@@:            " << endl
  //     << "     .B     .@    P@    F@i  UB@B2      .. ............      jB@BS              " << endl
  //     << "     2@  B.  P@    :    @B1    1@B@Br                     r@@B@F                " << endl
  //     << "     @u  @:   B@      B@Br       rB@B@Bqi.           ,78B@B@B7                  " << endl
  //     << "     @:  Gr    B2 ,8uB@B@           i0@B@B@B@B@B@B@@@@@@@Gr                     " << endl
  //     << "     @   7Y    XBUP@B@@@                .ru8B@B@B@MZjr.                         " << endl
  //     << "     B         B@B@B@B.                                                         " << endl
  //     << "     @02    ..BM U@@@@      :LLrjM           ,.           r8,       N@.         " << endl
  //     << "     B@@,r@ @@@   .B@     GB@B@B@BE      F@B@B@@@@7      :@B@      2@B@         " << endl
  //     << "     uB@B@B@B@.         Y@B@i   B@k    qB@8:   .ru.      @B@B;     @B@B         " << endl
  //     << "      U@@B@B@.         M@@7      .    NB@                B@@@O    :B@@@r        " << endl
  //     << "       2B@B@:         B@B             M@@7              7@BEB@    B@E0BO        " << endl
  //     << "        :B7          k@B               1@@@B@B@BF       @BE @B:  :@B .@B        " << endl
  //     << "                     @B7                  .:iLZ@B@X    :@@, B@B  @@O  B@.       " << endl
  //     << "                    :@@                         iB@;   B@@  r@@ :B@   @BG       " << endl
  //     << "                     @Bi        ur               @BJ  .@@U   @BO@@2   Y@B       " << endl
  //     << "                     P@BY    ;@B@B  iB@i       :@B@   8B@    u@B@B     B@5      " << endl
  //     << "                      7@@@B@B@B@:    BB@@@MOB@B@B5    B@@     B@B7     @B@      " << endl
  //     << "                        :Lk5v.         ;ZB@B@BU,      Z@r     :Ov      .@B.     " << endl
  //     << endl
  //     << "    University of Utah, Mechanical Engineering, Computational Solid Mechanics   " << endl << endl;

#ifdef JC_ZETA_HARDENING
  cout << ",JC_ZETA_HARDENING";
#endif
#ifdef JC_KAPPA_HARDENING
  cout << ",JC_KAPPA_HARDENING";
#endif
#ifdef JC_DEBUG_PARTICLE
  cout << ",JC_DEBUG_PARTICLE=" << JC_DEBUG_PARTICLE ;
#endif
#ifdef JC_USE_BB_DEFGRAD_UPDATE
  cout << ",JC_USE_BB_DEFGRAD_UPDATE=" << JC_USE_BB_DEFGRAD_UPDATE;
#endif
#ifdef CSM_PORE_PRESSURE_INITIAL
  cout << ",PORE_PRESSURE_INITIAL";
#endif
#ifdef JC_DEBUG_SMALL_TIMESTEP
  cout << ",JC_DEBUG_SMALL_TIMESTEP";
#endif
#ifdef JC_EPV
  cout << ",JC_EPV";
#endif
#ifdef JC_FREEZE_PARTICLE
  cout << ",JC_FREEZE_PARTICLE";
#endif
#ifdef JC_MAX_NESTED_RETURN
  cout << ",JC_MAX_NESTED_RETURN";
#endif
#ifdef JC_DEBUG_FR_OUTSIDE_CAP
  cout << ",JC_DEBUG_FR_OUTSIDE_CAP";
#endif
#ifdef CSM_DEBUG_BISECTION
  cout << ",CSM_DEBUG_BISECTION";
#endif
#ifdef JC_LIMITER_PRINT
  cout << ",JC_LIMITER_PRINT";
#endif
  cout << endl;
 // ps->getWithDefault("STREN",d_cm.STREN,0.0);    // Shear Limit Surface Parameter
  one_third      = 1.0/(3.0);
  two_third      = 2.0/(3.0);
  four_third     = 4.0/(3.0);
  sqrt_three     = sqrt(3.0);
  one_sqrt_three = 1.0/sqrt_three;
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
  initializeLocalMPMLabels();
}
Arenisca3::Arenisca3(const Arenisca3* cm)
  : ConstitutiveModel(cm)
{
  one_third      = 1.0/(3.0);
  two_third      = 2.0/(3.0);
  four_third     = 4.0/(3.0);
  sqrt_three     = sqrt(3.0);
  one_sqrt_three = 1.0/sqrt_three;
  // Shear Strength
  d_cm.PEAKI1 = cm->d_cm.PEAKI1;
  d_cm.FSLOPE = cm->d_cm.FSLOPE;
  d_cm.STREN = cm->d_cm.STREN;
  d_cm.YSLOPE = cm->d_cm.YSLOPE;
  d_cm.BETA_nonassociativity = cm->d_cm.BETA_nonassociativity;
  // Bulk Modulus
  d_cm.B0 = cm->d_cm.B0;
  d_cm.B1 = cm->d_cm.B1;
  d_cm.B2 = cm->d_cm.B2;
  d_cm.B3 = cm->d_cm.B3;
  d_cm.B4 = cm->d_cm.B4;
  // Shear Modulus
  d_cm.G0 = cm->d_cm.G0;
  d_cm.G1 = cm->d_cm.G1;
  d_cm.G2 = cm->d_cm.G2;
  d_cm.G3 = cm->d_cm.G3;
  d_cm.G4 = cm->d_cm.G4;
  // Porosity (Crush Curve)
  d_cm.p0_crush_curve = cm->d_cm.p0_crush_curve;
  d_cm.p1_crush_curve = cm->d_cm.p1_crush_curve;
  d_cm.p2_crush_curve = cm->d_cm.p2_crush_curve; // not used
  d_cm.p3_crush_curve = cm->d_cm.p3_crush_curve;
  d_cm.CR = cm->d_cm.CR;  // not used
  // Fluid Effects
  d_cm.fluid_B0 = cm->d_cm.fluid_B0;
  d_cm.fluid_pressure_initial = cm->d_cm.fluid_pressure_initial; //pf0
  // Rate Dependence
  d_cm.T1_rate_dependence = cm->d_cm.T1_rate_dependence;
  d_cm.T2_rate_dependence = cm->d_cm.T2_rate_dependence;
  // Equation of State
  d_cm.gruneisen_parameter = cm->d_cm.gruneisen_parameter; //pf0
  // Subcycling
  d_cm.subcycling_characteristic_number = cm->d_cm.subcycling_characteristic_number;
  initializeLocalMPMLabels();
}
// DESTRUCTOR
Arenisca3::~Arenisca3()
{
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


#ifdef CSM_PORE_PRESSURE_INITIAL
  ParticleVariable<double>  pdTdt;
  constParticleVariable<Matrix3> pDefGrad;
  ParticleVariable<Matrix3> //pDefGrad,
                            pStress;

  new_dw->allocateAndPut(pdTdt,       lb->pdTdtLabel,               pset);
  //new_dw->get(pDefGrad,    lb->pDeformationMeasureLabel, pset);
  new_dw->allocateAndPut(pStress,     lb->pStressLabel,             pset);

  // To fix : For a material that is initially stressed we need to
  // modify the stress tensors to comply with the initial stress state
  ParticleSubset::iterator iter = pset->begin();
  for(; iter != pset->end(); iter++){
    pdTdt[*iter] = 0.0;
    //pDefGrad[*iter] = Identity;
    pStress[*iter] = - one_third * d_cm.fluid_pressure_initial * Identity;
  }
#else
  initSharedDataForExplicit(patch, matl, new_dw);
#endif
  // Allocate particle variables
  ParticleVariable<int>     pLocalized,
                            pAreniscaFlag;

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
    pevp[*iter] = 0.0;
    peve[*iter] = 0.0;
    pCapX[*iter] = computeX(0.0);
    pCapXQS[*iter] = computeX(0.0);
    pKappa[*iter] = 0;//remove
    pZeta[*iter] = -3.0 * d_cm.fluid_pressure_initial;   //MH: Also need to initialize I1 to equal zeta
    pZetaQS[*iter] = -3.0 * d_cm.fluid_pressure_initial; //MH: Also need to initialize I1 to equal zeta
    pIota[*iter] = 0.0;
    pIotaQS[*iter] = 0.0;
    pStressQS[*iter].set(0.0);
    pScratchMatrix[*iter].set(0.0);
    pep[*iter].set(0.0);
  }
  computeStableTimestep(patch, matl, new_dw);
}

//May be used in the future
void Arenisca3::allocateCMDataAdd(DataWarehouse* new_dw,
                                 ParticleSubset* addset,
            map<const VarLabel*, ParticleVariableBase*>* newState,
                                 ParticleSubset* delset,
                                 DataWarehouse* old_dw)
{
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
  double  c_dil = 0.0,
          B0 = d_cm.B0,               // Low pressure bulk modulus
          B1 = d_cm.B1;               //
  double  bulk = B0 + B1,             // High pressure limit:  bulk = B0 + B1
          shear= d_cm.G0;             // shear modulus

  Vector  dx = patch->dCell(),
          WaveSpeed(1.e-12,1.e-12,1.e-12);       // what is this doing?

#ifdef JC_DEBUG_SMALL_TIMESTEP
  Vector  idvel(1,1,1),
          vbulk(1,1,1),
          vshear(1,1,1);
#endif
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

  // Allocate temporary particle variables
  ParticleVariable<double> rho_cur;
  new_dw->allocateTemporary(rho_cur,      pset);

  // loop over the particles in the patch
  for(ParticleSubset::iterator iter = pset->begin();
      iter != pset->end(); iter++){
    particleIndex idx = *iter;

    rho_cur[idx] = pmass[idx]/pvolume[idx];

    // Compute wave speed + particle velocity at each particle,
    // store the maximum
    c_dil = sqrt((bulk+4.0*shear/3.0)/rho_cur[idx]);

#ifdef JC_DEBUG_SMALL_TIMESTEP
     if(c_dil+fabs(pvelocity[idx].x()) > WaveSpeed.x()){
       idvel.x(idx); vbulk.x(bulk); vshear.x(shear);
     }
     if(c_dil+fabs(pvelocity[idx].y()) > WaveSpeed.y()){
       idvel.y(idx); vbulk.y(bulk); vshear.y(shear);
     }
     if(c_dil+fabs(pvelocity[idx].z()) > WaveSpeed.z()){
       idvel.z(idx); vbulk.z(bulk); vshear.z(shear);
     }
#endif
    WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
                     Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
                     Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));
  }

  // Compute the stable timestep based on maximum value of
  // "wave speed + particle velocity"
  WaveSpeed = dx/WaveSpeed;

#ifdef JC_DEBUG_SMALL_TIMESTEP
  //cout <<"delT_new="<<delT_new;
  //cout <<"dx="<<dx<<endl;
  if(delT_new==WaveSpeed.x()){
    cout << "pvel.x=" << pvelocity[idvel.x()].x()
         << ",wavespeed.x=" << WaveSpeed.x()
         << ",bulk=" << vbulk.x()
         << ",rho=" << rho_cur[idvel.x()]  << endl;
  }
  else if(delT_new==WaveSpeed.y()){
    cout << "pvel.y: " << pvelocity[idvel.y()].y()
         << ",wavespeed.y=" << WaveSpeed.y()
         << ",bulk=" << vbulk.y()
         << ",rho=" << rho_cur[idvel.y()] << endl;
  }
  else if(delT_new==WaveSpeed.z()){
    cout << "pvel.z: " << pvelocity[idvel.z()].z()
         << ",wavespeed.z=" << WaveSpeed.z()
         << ",bulk=" << vbulk.z()
         << ",rho=" << rho_cur[idvel.z()]  << endl;
  }
  else
    cout << "ERROR in JC_DEBUG_SMALL_TIMESTEP" <<endl;
#endif

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

  // Get the initial density
  double rho_orig = matl->getInitialDensity();
  // Compute kinematics variables (pDefGrad_new, pvolume, pLocalized_new, pVelGrad_new)
  // computeKinematics(patches, matl, old_dw, new_dw);

  // Global loop over each patch
  for(int p=0;p<patches->size();p++){

    // Declare and initial value assignment for some variables
    const Patch* patch = patches->get(p);
    Matrix3 D;

    double J,
           c_dil=0.0,
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
    new_dw->get(pDefGrad_new,
                lb->pDeformationMeasureLabel_preReloc,      pset);

    // Get the particle variables from compute kinematics

    ParticleVariable<int>     pLocalized_new,
                              pAreniscaFlag_new;

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
    ParticleVariable<double>       f_trial_step,
                                   rho_cur; //used for calc. of stable timestep
    ParticleVariable<Matrix3>      rotation;

    new_dw->allocateTemporary(f_trial_step, pset);
    new_dw->allocateTemporary(rho_cur,      pset);
    new_dw->allocateTemporary(rotation,     pset);

    // Loop over the particles of the current patch to compute particle density
    //T2D: remove once stable timestep is made into a modular function
    for(ParticleSubset::iterator iter=pset->begin();iter!=pset->end();iter++){
      particleIndex idx = *iter;

      // Update particle density
      J = pDefGrad_new[idx].Determinant();
      rho_cur[idx] = rho_orig/J;
    }

    // Loop over the particles of the current patch to update particle
    // stress at the end of the current timestep along with all other
    // required data such plastic strain, elastic strain, cap position, etc.
#ifdef JC_DEBUG_SMALL_TIMESTEP
    Vector idvel(1,1,1);   // temp
    Vector vbulk(1,1,1);   // temp
    Vector vshear(1,1,1);  // temp
#endif
    for(ParticleSubset::iterator iter = pset->begin();iter!=pset->end();iter++){
      particleIndex idx = *iter;  //patch index
        //cout<<"pID="<<pParticleID[idx]<<endl;

        // A parameter to consider the thermal effects of the plastic work which
        // is not coded in the current source code. Further development of Arenisca
        // may ativate this feature.
        pdTdt[idx] = 0.0;

        //Set scratch parameters to old values
        pScratchDouble1_new[idx] = pScratchDouble1[idx];
        pScratchDouble2_new[idx] = pScratchDouble2[idx];
        pScratchMatrix_new[idx]  = pScratchMatrix[idx];

        //Set currently unused porepressure value to previous value
        pPorePressure_new[idx]   = pPorePressure[idx];

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

        // Divides the strain increment into substeps, and calls substep function
        // cout<<"765: D = "<<D<<", delT = "<<delT<<", sigma_old = "<<sigma_old<<", pCapX[idx] = "<<pCapX[idx]<<", pZeta[idx] = "<<pZeta[idx]<<", pep[idx] = "<<pep[idx]<<endl;
        // cout<<"766: pStress_new[idx] = "<<pStress_new[idx]<<", pCapX_new[idx] = "<< pCapX_new[idx]<<", pZeta_new[idx] = "<<pZeta_new[idx]<<", pep_new[idx] = "<< pep_new[idx]<<endl;
        int stepFlag = computeStep(D,                  // strain "rate"
                                   delT,               // time step (s)
                                   sigma_old,          // unrotated stress at start of step
                                   pCapX[idx],         // hydrostatic comrpessive strength at start of step
                                   pZeta[idx],         // trace of isotropic backstress at start of step
                                   pep[idx],           // plastic strain at start of step
                                   pStress_new[idx],   // unrotated stress at start of step
                                   pCapX_new[idx],     // hydrostatic comrpessive strength at start of step
                                   pZeta_new[idx],     // trace of isotropic backstress at start of step
                                   pep_new[idx],       // plastic strain at start of step
                                   pParticleID[idx]
                                  );
        //cout<<"807: pStress_new[idx] = "<<pStress_new[idx]<<", pCapX_new[idx] = "<< pCapX_new[idx]<<", pZeta_new[idx] = "<<pZeta_new[idx]<<", pep_new[idx] = "<< pep_new[idx]<<endl;

        // Plastic volumetric strain at end of step
        pevp_new[idx] = pep_new[idx].Trace();
        // Elastic volumetric strain at end of step
        // MH! Change this to compute the total strain tensor from [F], subtract [ep] and take trace
        peve_new[idx] = peve[idx] + D.Trace()*delT - pevp_new[idx] + pevp[idx];
        // Branch point at the start of the step (not used, set to zero)
        pKappa_new[idx] = 0.0;

        // T2D: ADD RATE DEPENDENCE CODE HERE
        pCapXQS_new[idx]   = pCapX_new[idx];
        pZetaQS_new[idx]   = pZeta_new[idx];
        pIotaQS_new[idx]   = pIota_new[idx];
        pStressQS_new[idx] = pStress_new[idx];
        //d_cm.T1_rate_dependence
        //d_cm.T2_rate_dependence

        // Compute the total strain energy and the stable timestep based on both
        // the particle velocities and wave speed.

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

        c_dil = sqrt((bulk+four_third*shear)/(rho_cur[idx]));
#ifdef JC_DEBUG_SMALL_TIMESTEP
        if(c_dil+fabs(pvelocity[idx].x()) > WaveSpeed.x())
          {
          idvel.x(idx);
          vbulk.x(bulk);
          vshear.x(shear);
          }
        if(c_dil+fabs(pvelocity[idx].y()) > WaveSpeed.y())
        {
          idvel.y(idx);
          vbulk.y(bulk);
          vshear.y(shear);
        }
        if(c_dil+fabs(pvelocity[idx].z()) > WaveSpeed.z())
        {
          idvel.z(idx);
          vbulk.z(bulk);
          vshear.z(shear);
        }
#endif
        WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
                         Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
                         Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));

        // Compute artificial viscosity term
        if (flag->d_artificial_viscosity) {
          double dx_ave = (dx.x() + dx.y() + dx.z())*one_third;
          double c_bulk = sqrt(bulk/rho_cur[idx]);
          p_q[idx] = artificialBulkViscosity(D.Trace(), c_bulk, rho_cur[idx], dx_ave);
        } else {
          p_q[idx] = 0.;
        }

        // Compute the averaged stress
        Matrix3 AvgStress = (pStress_new[idx] + pStress_old[idx])*0.5;
        // Compute the strain energy increment associated with the particle
        double e = (D(0,0)*AvgStress(0,0) +
                    D(1,1)*AvgStress(1,1) +
                    D(2,2)*AvgStress(2,2) +
                2.*(D(0,1)*AvgStress(0,1) +
                    D(0,2)*AvgStress(0,2) +
                    D(1,2)*AvgStress(1,2))) * pvolume[idx]*delT;

        // Accumulate the total strain energy
        //MH! Note the initialization of se needs to be fixed as it is currently reset to 0
        se += e;
#ifdef JC_DEBUG_PARTICLE
      if(pParticleID[idx]==JC_DEBUG_PARTICLE)
        cout << endl;

#endif

    }

    // Compute the stable timestep based on maximum value of
    // "wave speed + particle velocity"
    WaveSpeed = dx/WaveSpeed; //Variable now holds critical timestep (not speed)

    double delT_new = WaveSpeed.minComponent();
    //cout<<"delT_new="<<delT_new<<endl;
    //computeStableTimestep(patch,pset,matl,new_dw);

#ifdef JC_DEBUG_SMALL_TIMESTEP
        //cout <<"delT_new="<<delT_new;
        //cout <<"dx="<<dx<<endl;
    if(delT_new==WaveSpeed.x()){
      cout << "pvel.x=" << pvelocity[idvel.x()].x()
           << ",wavespeed.x=" << WaveSpeed.x()
           << ",bulk=" << vbulk.x()
           << ",rho=" << rho_cur[idvel.x()]  << endl;
    }
    else if(delT_new==WaveSpeed.y()){
    cout << "pvel.y: " << pvelocity[idvel.y()].y()
         << ",wavespeed.y=" << WaveSpeed.y()
         << ",bulk=" << vbulk.y()
         << ",rho=" << rho_cur[idvel.y()] << endl;
    }
    else if(delT_new==WaveSpeed.z()){
      cout << "pvel.z: " << pvelocity[idvel.z()].z()
           << ",wavespeed.z=" << WaveSpeed.z()
           << ",bulk=" << vbulk.z()
           << ",rho=" << rho_cur[idvel.z()]  << endl;
    }
    else
      cout << "ERROR in JC_DEBUG_SMALL_TIMESTEP" <<endl;
#endif

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
                          const Matrix3& ep_n,    // plastic strain at start of step(t_n)
                                Matrix3& sigma_p, // unrotated stress at wnd of step(t_n+1)
                                double & X_p,     // hydrostatic comrpessive strength at end of step(t_n+1)
                                double & Zeta_p,  // trace of isotropic backstress at end of step(t_n+1)
                                Matrix3& ep_p,    // plastic strain at end of step (t_n+1)
                                long64 ParticleID)// ParticleID for debug purposes
{
int n,
    //chi = d_cm.subcycling_characteristic_number,// subcycle multiplier
    chi = 1, //MH! fix this
    chimax = 256,                                 // max allowed subcycle multiplier
    stepFlag,                                     // 0/1 good/bad step
    substepFlag;                                  // 0/1 good/bad substep
double dt,                                        // substep time increment
       X_old,                                     // X at start of substep
       X_new,                                     // X at end of substep
       Zeta_old,                                  // Zeta at start of substep
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
    ep_old    = ep_n;
    n = 1;

// (5) Call substep function {sigma_new,ep_new,X_new,Zeta_new}
//                               = computeSubstep(D,dt,sigma_old,ep_old,X_old,Zeta_old)
    computeSubstep:
      //cout<<"991: D = "<<D<<",   dt = "<<dt<<", sigma_old = "<<sigma_old<<", ep_old = "<<ep_old<<", X_old = "<<X_old<<", Zeta_old = "<<Zeta_old<<endl;
      substepFlag = computeSubstep(D,dt,sigma_old,ep_old,X_old,Zeta_old,
                                        sigma_new,ep_new,X_new,Zeta_new);
      //cout<<"994: sigma_new = "<<sigma_new<<", ep_new = "<<ep_new<<", X_new = "<<X_new<<", Zeta_new = "<<Zeta_new<<endl;
// (6) Check error flag from substep calculation:
      if (substepFlag == 0) {       // no errors in substep calculation
        if (n < (chi*nsub)) {       // update and keep substepping
          sigma_old = sigma_new;
          X_old     = X_new;
          Zeta_old  = Zeta_new;
          ep_old    = ep_new;
          n++;
          goto computeSubstep;
        } else goto successfulStep; // n = chi*nsub, Step is done
        if (chi < chimax)   {       // errors in substep calculation
          chi = 2*chi;
          goto stepDivide;
        } else goto failedStep;     // bad substep and chi>=chimax, Step failed even with substepping
      }

// (7) Successful step, set value at end of step to value at end of last substep.
    successfulStep:
      sigma_p   = sigma_new;
      X_p       = X_new;
      Zeta_p    = Zeta_new;
      ep_p      = ep_new;
      stepFlag  = 0;
      //cout << "1043: Step Succeeded I1_n = "<<sigma_n.Trace() <<", I1_p = "<<sigma_p.Trace()<< endl;
      //cout << "1044: evp_p = "<<ep_p.Trace()<<", evp_n = "<<ep_n.Trace()<<endl;
      //cout << "1045: X_p = "<<X_p<<", X_n = "<<X_n<<endl;
      return stepFlag;

// (8) Failed step, Send ParticleDelete Flag to Host Code, Store Inputs to particle data:
    failedStep:
      // input values for sigma_new,X_new,Zeta_new,ep_new, along with error flag
      sigma_p   = sigma_n;
      X_p       = X_n;
      Zeta_p    = Zeta_n;
      ep_p      = ep_n;
      stepFlag  = 1;
      cout << "1059: Step Failed I1_n = "<<sigma_n.Trace() <<", I1_p = "<<sigma_p.Trace()<< endl;
      cout << "1060: evp_p = "<<ep_p.Trace()<<", evp_n = "<<ep_n.Trace()<<endl;
      cout << "1061: X_p = "<<X_p<<", X_n = "<<X_n<<endl;
      return stepFlag;

} //===================================================================


// [shear,bulk] = computeElasticProperties()
void Arenisca3::computeElasticProperties(double & bulk,
                                        double & shear
                                       )
{
// When computeElasticProperties() is called with two doubles as arguments, it
// computes the high pressure limit tangent elastic shear and bulk modulus
// This is used to esimate wave speeds and make conservative estimates of substepping.
double  b0 = d_cm.B0,
        b1 = d_cm.B1,   // MH! Change this to B1 and add B2-B4 parameters
        g0 = d_cm.G0;

shear   = g0;     // Shear Modulus
bulk    = b0+b1;  // Bulk Modulus
} //===================================================================

// [shear,bulk] = computeElasticProperties(stress, ep)
void Arenisca3::computeElasticProperties(const Matrix3 stress,
                                        const Matrix3 ep,
                                        double & bulk,
                                        double & shear
                                       )
{
// When computeElasticProperties() is called with two Matrix3 and two doubles as arguments,
// it computes the nonlinear elastic properties.

//MH! fix this when input arguments are available.  For now just return constant values.
double  b0 = d_cm.B0,
        b1 = d_cm.B1,   // MH! Change this to B1 and add B2-B4 parameters
        b2 = d_cm.B2,
        b3 = d_cm.B3,
        b4 = d_cm.B4,
        g0 = d_cm.G0;

shear   = g0;     // Shear Modulus
bulk    = b0;     // Bulk Modulus

double  I1 = stress.Trace(),
        evp = ep.Trace();

if (evp<=0.0){
  if (I1!=0.0){bulk = bulk + b1*exp(-b2/abs(I1));}
  if (evp!=0.0){bulk = bulk - b3*exp(-b4/abs(evp));}
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
//cout<<"1085: sigma_old = "<<sigma_old<<", d_e = "<<d_e<<", bulk = "<<bulk<<", shear = "<<shear<<endl;
Matrix3 d_e_iso = (1.0/3.0)*d_e.Trace()*Identity;
//cout<<"1087: d_e_iso = "<<d_e_iso<<endl;
Matrix3 d_e_dev = d_e - d_e_iso;
//cout<<"1089: d_e_dev = "<<d_e_dev<<endl;
Matrix3 sigma_trial = sigma_old + (3.0*bulk*d_e_iso + 2.0*shear*d_e_dev);
//cout<<"1091: sigma_trial = "<<sigma_trial<<endl;
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

// MH! Make this work! For now just set nsub = 1
double PEAKI1 = d_cm.PEAKI1,
       FSLOPE = d_cm.FSLOPE;

Matrix3 d_sigma = sigma_trial - sigma_n;

double  bulk_n,shear_n,bulk_trial,shear_trial;
computeElasticProperties(sigma_n,ep,bulk_n,shear_n);
computeElasticProperties(sigma_trial,ep,bulk_trial,shear_trial);

int n_bulk = ceil(abs(bulk_n-bulk_trial)/bulk_n),
    n_iso = ceil(abs(d_sigma.Trace())/(PEAKI1-X)/20.),
    n_dev = ceil(d_sigma.Norm()/(FSLOPE*(PEAKI1-X))/20.);

//int nsub = 1;
int nsub = min(max(max(max(n_bulk,n_iso),n_dev),1),100);
//if (nsub>50){cout<<"1159 nsub = "<<nsub<<endl;}
//nsub=10;
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

  if(sqrt(J2) < 1e-8*sqrt(Pow(I1,2)+J2))
    J2=0.0;

  rJ2 = sqrt(J2);
} //===================================================================

// Computes the updated stress state for a substep
int Arenisca3::computeSubstep(const Matrix3& D,         // Strain "rate"
                             const double & dt,        // time substep (s)
                             const Matrix3& sigma_old, // stress at start of substep
                             const Matrix3& ep_old,    // plastic strain at start of substep
                             const double & X_old,     // hydrostatic compressive strength at start of substep
                             const double & Zeta_old,  // trace of isotropic backstress at start of substep
                                   Matrix3& sigma_new, // stress at end of substep
                                   Matrix3& ep_new,    // plastic strain at end of substep
                                   double & X_new,     // hydrostatic compressive strength at end of substep
                                   double & Zeta_new   // trace of isotropic backstress at end of substep
                            )
{
// Computes the updated stress state for a substep that may be either elastic, plastic, or
// partially elastic.   Returns an integer flag 0/1 for a good/bad update.
  int     substepFlag,
          returnFlag;
  double p3  = d_cm.p3_crush_curve;

// (1)  Compute the elastic properties based on the stress and plastic strain at
// the start of the substep.  These will be constant over the step unless elastic-plastic
// is used to modify the tangent stiffness in the consistency bisection iteration.
  double bulk,
         shear;
  computeElasticProperties(sigma_old,ep_old,bulk,shear);

// (2) Compute the increment in total strain:
  Matrix3 d_e = D*dt;

// (3) Compute the trial stress: [sigma_trail] = computeTrialStress(sigma_old,d_e,K,G)
  //cout<<"1178: sigma_old = "<<sigma_old<<", d_e = "<<d_e<<", bulk = "<<bulk<<", shear = "<<shear<<endl;
  Matrix3 sigma_trial = computeTrialStress(sigma_old,d_e,bulk,shear),
          S_trial;
  //cout<<"1181: sigma_trial = "<<sigma_trial<<endl;
  double I1_trial,
         J2_trial,
         rJ2_trial;
  computeInvariants(sigma_trial,S_trial,I1_trial,J2_trial,rJ2_trial);
  //cout<<"1186: sigma_trial = "<<sigma_trial<<", S_trial = "<<S_trial<<", I1_trial = "<<I1_trial<<", J2_trial = "<<J2_trial<<endl;
// (4) Evaluate the yield function at the trial stress:
  int YIELD = computeYieldFunction(I1_trial,rJ2_trial,X_old,Zeta_old);
  //cout << "1175: I1_trial = " << I1_trial << ", rJ2_trial = " << rJ2_trial <<"YIELD = "<<YIELD << endl;
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
            TOL = 1e-9; // bisection convergence tolerance on eta
    Matrix3 S_0,        // S (deviator) at stress update for non-hardening return
            d_ep_0;     // increment in plastic strain for non-hardening return

    Matrix3 S_old;
    double I1_old,
           J2_old,
           rJ2_old,
           evp_old = ep_old.Trace();
    computeInvariants(sigma_old,S_old,I1_old,J2_old,rJ2_old);
    returnFlag = nonHardeningReturn(I1_trial,rJ2_trial,S_trial,I1_old,rJ2_old,S_old,
                                    d_e,X_old,Zeta_old,bulk,shear,
                                    I1_0,rJ2_0,S_0,d_ep_0);
  //  cout << "1201: nonhardening return: I1_0 = " << I1_0 << ", rJ2_0 = " << rJ2_0 << endl;
    double d_evp_0 = d_ep_0.Trace();
  //cout << "1203: nonhardening return: d_vep_0 = " << d_evp_0 <<", evp_old = " <<evp_old<< endl;

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
      //cout << "1218: i = " << i << endl;
      eta_mid   = 0.5*(eta_out+eta_in);
      d_evp     = eta_mid*d_evp_0;

      if(evp_old + d_evp <= -p3 ){
        eta_out = eta_mid;
        //  cout << "1229: yield surface encloses trial stress, goto updateISV "<< endl;
        goto updateISV;
      }

      X_new     = computeX(evp_old + d_evp);
      Zeta_new  = Zeta_old + dZetadevp*d_evp;

// (8) Check if the updated yield surface encloses trial stres.  If it does, there is too much
//     plastic strain for this iteration, so we adjust the bisection parameters and recompute
//     the state variable update.
      if( computeYieldFunction(I1_trial,rJ2_trial,X_new,Zeta_new)!=1 ){
        eta_out = eta_mid;
      //  cout << "1229: yield surface encloses trial stress, goto updateISV "<< endl;
        goto updateISV;
      }

// (9) Recompute the elastic properties based on the midpoint of the updated step:
//     [K,G] = computeElasticProeprties( (sigma_old+sigma_new)/2,ep_old+d_ep/2 )
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
                                          d_e,X_new,Zeta_new,bulk,shear,
                                          I1_new,rJ2_new,S_new,d_ep_new);

// (10) Check whether the isotropic component of the return has changed sign, as this
//      would indicate that the cap apex has moved past the trial stress, indicating
//      too much plastic strain in the return.

      // MH! add code that will be robust if I1_trial - I1_new ~0, so the sign would be
      //     a matter of roundoff.  Otherwise there could be problems with a von Mises
      //     type model.

      if( Sign(I1_trial - I1_new)!=Sign(I1_trial - I1_0)){
        eta_out = eta_mid;
      //  cout << "1259: isotropic return changed sign, goto updateISV "<< endl;
      //  cout << "I1_trial = " << I1_trial << endl;
      //  cout << "I1_new = " << I1_new << endl;
      //  cout << "I1_old = " << I1_old << endl;
        if( i >= imax ){                                        // solution failed to converge
          cout << "1284: i>=imax, failed substep "<< endl;
          goto failedSubstep;
        }
        goto updateISV;
      }
      // Good update, compare magnitude of plastic strain with prior update
      d_evp_new = d_ep_new.Trace();   // Increment in vol. plastic strain for return to new surface

     // cout<<"1273, d_evp_new = "<<d_evp_new<<endl;
      //cout<<"1274, eta_mid*d_evp_0 = "<<eta_mid*d_evp_0<<endl;
      // Check for convergence
      if( abs(eta_out-eta_in) < TOL ){           // Solution is converged
        Matrix3 Identity;
        Identity.Identity();
        sigma_new = (1.0/3.0)*I1_new*Identity + S_new;
        ep_new = ep_old + d_ep_new;
        X_new = computeX(ep_new.Trace());
        Zeta_new = Zeta_old + dZetadevp*d_evp_new;
      //  cout << "1273: successful substep "<< endl;
        goto successfulSubstep;
      }
      if( i >= imax ){                                        // solution failed to converge
        cout << "1306: i>=imax, failed substep "<< endl;
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
      //cout << "1289: good substep, eta_in = "<<eta_in<<", eta_out = "<<eta_out<< endl;
      //cout << "1289: X_new = "<<X_new<<", ep_old = "<<ep_old<<", d_ep_new = "<<d_ep_new<<endl;
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
         B0  = d_cm.B0,             // low pressure bulk modulus
         B1  = d_cm.B1,             // additional high pressure bulk modulus
         Kf  = d_cm.fluid_B0,       // fluid bulk modulus
         //Pf0 = d_cm.fluid_pressure_initial,            // initial pore pressure
         ev0,                       // strain at zero pore pressure
         Kfit,
         Xfit,
         Keng,
         eveX,
         X;

  if(evp<=-p3)
  { // Plastic strain exceeds allowable limit.========================
    // The plastic strain for this iteration has exceed the allowable
    // value.  X is not defined in this region, so we set it to a large
    // negative number.  This will cause the plastic strain to be reduced
    // in subsequent iterations.
    X = 1.0e6*p0;
  }
  else
  { // Plastic strain is within allowable domain======================
    if(Kf==0.0)
    { // No Fluid Effects ---------------------------------------------
      if(evp <= 0.0)
        X = (p0*p1 + log((evp+p3)/p3))/p1;
      else
        X = p0*Pow(1.0+evp , 1.0/(p0*p1*p3));
    }
    else
    { // Fluid Effects ------------------------------------------------
      // First we evaluate the elastic volumetric strain to yield from the
      // empirical crush curve (Xfit) and bulk modulus (Kfit) formula for
      // the drained material.  These functions could be modified to use
      // the full non-linear and elastic-plastic coupled input paramters
      // without introducing the additional complexity of elastic-plastic
      // coupling in the plasticity solution.
      if(evp <= 0.0)
      { // pore collapse
        // Hack: for now we use a constant bulk modulus until we revise Arenisca to call
        // step with a strain rate so the bulk modulus can be adjusted with deformation.
        //
        //Kfit = B0 + B1;                     // drained bulk modulus function
        Kfit = B0;                            // drained bulk modulus function
        Xfit = (p0*p1+log((evp+p3)/p3))/p1;   // drained crush curve function
      }
      else
      { // pore expansion
        Kfit = B0;                                   // drained bulk modulus function
        Xfit = Pow(1.0 + evp , 1.0 / (p0*p1*p3))*p0; // drained crush curve function
      }

      // Now we use our linear engineering model for the bulk modulus of the
      // saturated material to compute the stress at our elastic strain to yield.
      ev0  = computeev0();                      // strain at zero pore pressure

      // Hack: for now we use a constant bulk modulus until we revise Arenisca to call
      // step with a strain rate so the bulk modulus can be adjusted with deformation.
      //
      //Keng = computeBulkModulus(ev0-1);       // Saturated bulk modulus
      Keng = B0;

      eveX = one_third*Xfit/Kfit;               // Elastic vol. strain to compressive yield

      // There are three regions depending on whether the elastic loading to yield
      // occurs within the domain of fluid effects (ev < ev0)
      if(evp <= ev0)                            // Fluid Effects
        X = 3.0*Keng*eveX;
      else if(evp > ev0 && evp+eveX < ev0)      // Transition
        // Hack: for now we use a constant bulk modulus until we revise Arenisca to call
        // step with a strain rate so the bulk modulus can be adjusted with deformation.
        // Also, check this, it might be wrong
        //
        // X = 3*B0*(evp-ev0) + 3*Keng*(evp+eveX-ev0);
        X = 3.0*B0*eveX;
      else                                      // No Fluid Effects
        X = 3.0*B0*eveX;
    } //end fluid effects
  } // end good/bad plastic strainno matching function for call to ‘Uintah::Arenisca3::computeYieldFunct

  return X;
} //===================================================================

// Compute the strain at zero pore pressure from initial pore pressure (Pf0)
double Arenisca3::computeev0()
{
  // The user-supplied initial pore pressure (Pf0) is the pore pressure at zero
  // volumetric strain.  An estimate of the strain (ev0) at which the fluid pressure
  // is zero is derived from M. Homel's engineering model of matrix compressibility:

  //define and initialize some variables
  double p3  = d_cm.p3_crush_curve, // max vol. plastic strain
         Kf  = d_cm.fluid_B0,       // fluid bulk modulus
         Pf0 = 3.0*d_cm.fluid_pressure_initial,            // initial pore pressure (-I1)
         ev0;                       // strain at zero pore pressure

   if(Pf0==0) // No initial pore pressure
    ev0 = 0.0;

   else       // Initial pore pressure
    ev0 = -p3 + log(1.0 - exp(Pf0/Kf) + exp(p3 + Pf0/Kf));

  return ev0;
} //===================================================================

// Compute nonhardening return from trial stress to some yield surface
int Arenisca3::nonHardeningReturn(const double & I1_trial,    // Trial Stress
                                 const double & rJ2_trial,
                                 const Matrix3& S_trial,
                                 const double & I1_old,      // Stress at start of subtep
                                 const double &rJ2_old,
                                 const Matrix3& S_old,
                                 const Matrix3& d_e,         // increment in total strain
                                 const double & X,           // cap position
                                 const double & Zeta,        // isotropic bacstress
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

  const double pi  = 3.141592653589793238462,
               TOL = 1e-6;
  double theta = pi/2.0;
  int n,
      interior,
      returnFlag;

// (1) Define an interior point, (I1_0 = Zeta, J2_0 = 0)
  double  I1_0 = Zeta,
          rJ2_0 = 0.0;

// (2) Transform the trial and interior points as follows where beta defines the degree
//  of non-associativity.
  double beta = d_cm.BETA_nonassociativity;  // MH! change this for nonassociativity in the meridional plane
  double r_trial = beta*(3.0*bulk)/(2.0*shear)*sqrt(2.0)*rJ2_trial,
         z_trial = I1_trial/sqrt(3.0),
         z_test,
         r_test,
         r_0     = beta*(3.0*bulk)/(2.0*shear)*sqrt(2.0)*rJ2_0,
         z_0     = I1_0/sqrt(3.0);

//  cout << "1480: r_trial = "<<r_trial<<", z_trial = "<<z_trial<<endl;
//  cout << "1480: r_0 = "<<r_0<<", z_0 = "<<z_0<<endl;
// (3) Perform Bisection between in transformed space, to find the new point on the
//  yield surface: [znew,rnew] = transformedBisection(z0,r0,z_trial,r_trial,X,Zeta,K,G)
  while ( abs(theta) > TOL ){
    // transformed bisection to find a new interior point, just inside the boundary of the
    // yield surface.  This function overwrites the inputs for z_0 and r_0
    //  [z_0,r_0] = transformedBisection(z_0,r_0,z_trial,r_trial,X_Zeta,bulk,shear)
    transformedBisection(z_0,r_0,z_trial,r_trial,X,Zeta,bulk,shear);
//    cout << "1489: r_0 = "<<r_0<<", z_0 = "<<z_0<<endl;
// (4) Perform a rotation of {z_new,r_new} about {z_trial,r_trial} until a new interior point
// is found, set this as {z0,r0}
    interior = 0;
    n = 0.0;
// (5) Test for convergence:
    while ( (interior==0)&&(abs(theta)>TOL) ){
      theta = (pi/2.0)*Pow(-1.0,n+2)*Pow(0.5,floor((n+2.0)/2.0));
      z_test = z_trial + cos(theta)*(z_0-z_trial) - sin(theta)*(r_0-r_trial);
      r_test = r_trial + sin(theta)*(z_0-z_trial) + cos(theta)*(r_0-r_trial);
//      cout << "1554: n = "<<n<<", theta = "<<theta<<endl;
//      cout << "1555: r_test = "<<r_test<<", z_test = "<<z_test<<endl;
      if ( transformedYieldFunction(z_test,r_test,X,Zeta,bulk,shear) == -1 ) { // new interior point
        interior = 1;
        z_0 = z_test;
        r_0 = r_test;
      }
      else { n++; }
      //if(n>50){cout<<"1562, n>40, n = "<<n<<endl;}
    }
  }

// (6) Solution Converged, Compute Untransformed Updated Stress:
  I1_new = sqrt(3.0)*z_0;
  rJ2_new = sqrt(2.0)*shear/(3.0*bulk*beta)*r_0;
  if ( rJ2_trial!=0.0 ){S_new = S_trial*rJ2_new/rJ2_trial;}
  else                 {S_new = S_trial;}
  Matrix3 Identity;
  Identity.Identity();
  Matrix3 sigma_new = (1.0/3.0)*I1_new*Identity + S_new,
          sigma_old = (1.0/3.0)*I1_old*Identity + S_old;
  Matrix3 d_sigma = sigma_new - sigma_old;

//  cout << "1515: sigma_new.Trace() = "<<sigma_new.Trace()<<endl;
//  cout << "1516: sigma_old.Trace() = "<<sigma_new.Trace()<<endl;
// (7) Compute increment in plastic strain for return:
//  d_ep0 = d_e - [C]^-1:(sigma_new-sigma_old)
  Matrix3 d_ee    = d_sigma/(2.0*shear) + (1.0/(9.0*bulk)-1.0/(6.0*shear))*d_sigma.Trace()*Identity;
  d_ep_new        = d_e - d_ee;

  //cout << "1532: d_ev = "<<d_e.Trace()<<", d_ee = "<<d_ee.Trace()<<", d_ep_new = "<<d_ep_new.Trace()<<endl;
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
                                    const double& bulk,
                                    const double& shear
                                   )
{
// Computes a bisection in transformed stress space between point sigma_0 (interior to the
// yield surface) and sigma_trial (exterior to the yield surface).  Returns this new point,
// which will be just inside the yield surface, overwriting the input arguments for
// z_0 and r_0.

// (1) initialize bisection
  double eta_out=1.0,
       eta_in =0.0,
       eta_mid,
       TOL = 1e-6,
       r_test,
       z_test;

// (2) Test for convergence
  while (eta_out-eta_in > TOL){
    eta_mid = (eta_out+eta_in)/2.0;

// (3) Transformed test point
    z_test = z_0 + eta_mid*(z_trial-z_0);
    r_test = r_0 + eta_mid*(r_trial-r_0);
// (4) Check if test point is within the yield surface:
    if ( transformedYieldFunction(z_test,r_test,X,Zeta,bulk,shear)!=1 ) {eta_in = eta_mid;}
    else {eta_out = eta_mid;}
  }

// (5) Converged, return {z_new,r_new}={z_test,r_test}
//z_0 = z_test;
//r_0 = r_test;
  z_0 = z_0 + eta_out*(z_trial-z_0);
  r_0 = r_0 + eta_out*(r_trial-r_0);

} //===================================================================

// computeTransformedYieldFunction from transformed inputs
int Arenisca3::transformedYieldFunction(const double& z,
                                       const double& r,
                                       const double& X,
                                       const double& Zeta,
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
double  I1  = sqrt(3.0)*z,
        rJ2 = sqrt(2.0)*shear/(3.0*bulk*beta)*r;
int    YIELD = computeYieldFunction(I1,rJ2,X,Zeta);
return YIELD;
} //===================================================================

// computeYieldFunction from untransformed inputs
int Arenisca3::computeYieldFunction(const double& I1,
                                   const double& rJ2,
                                   const double& X,
                                   const double& Zeta
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
          STREN = d_cm.STREN,    //MH! add this user input
          YSLOPE = d_cm.YSLOPE,  //MH! add this user input
          PEAKI1 = d_cm.PEAKI1,
          Ff;

  if (FSLOPE == 0.0) {// VON MISES-------------------------------------
    // If the user has specified an input set with FSLOPE = 0, this indicates
    // a von Mises plasticity model should be used.  In this case, the yield
    // stress is the input value for PEAKI1.
    Ff = PEAKI1;
  }
  else if (YSLOPE == FSLOPE){// LINEAR DRUCKER-PRAGER SURFACE----------
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
  //
  //define and initialize some variables
  double p3  = d_cm.p3_crush_curve,
         B0  = d_cm.B0,             // low pressure bulk modulus
         B1  = d_cm.B1,             // additional high pressure bulk modulus
         Kf  = d_cm.fluid_B0,       // fluid bulk modulus
         Pf0 = d_cm.fluid_pressure_initial,
         ev0,
         dZetadevp;

  ev0  = computeev0();                // strain at zero pore pressure

  if (evp <= ev0 && Kf != 0) // ev0, is material strain at zero fluid pressure
    // Fluid Effects
    dZetadevp = (3.0*(B0 + B1)*exp(evp + p3)*Kf)/
                (B0*(exp(evp + p3) - exp(Zeta/(3.0*(B0 + B1)))) +
                 B1*(exp(evp + p3) - exp(Zeta/(3.0*(B0 + B1)))) +
                (exp(evp + p3) + exp((3.0*Pf0 + Zeta)/(3.0*Kf)) -
                 exp(p3 + Pf0/Kf + Zeta/(3.0*Kf)))*Kf);
  else
    dZetadevp = 0.0;

  //dZetadevp=0.0;  //MH! For now supress backstress evolution
  //cout<<"1691: dZetadevp = "<<dZetadevp<<", Zeta = "<<Zeta<<", evp = "<<evp<<endl;
  return dZetadevp;
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

//T2D: move up
void Arenisca3::addInitialComputesAndRequires(Task* task,
    const MPMMaterial* matl,
    const PatchSet* patch) const
{
  // Add the computes and requires that are common to all explicit
  // constitutive models.  The method is defined in the ConstitutiveModel
  // base class.
  const MaterialSubset* matlset = matl->thisMaterial();

  // Other constitutive model and input dependent computes and requires
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

  return rho_cur;

//#if 1
  cout << "NO VERSION OF computeRhoMicroCM EXISTS YET FOR Arenisca3"<<endl;
//#endif

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

  cout << "NO VERSION OF computePressEOSCM EXISTS YET FOR Arenisca3"
  << endl;
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
