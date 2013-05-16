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

/* ARENISCA INTRO

This source code is for a simplified constitutive model, named ``Arenisca'',
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
#define JC_ARENISCA_VERSION 1.0  //121215.2310 JC & MH
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
#include <CCA/Components/MPM/ConstitutiveModel/Arenisca.h>
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
Arenisca::Arenisca(ProblemSpecP& ps, MPMFlags* Mflag)
  : ConstitutiveModel(Mflag)
{
  cout << "In Arenisca ver"<< JC_ARENISCA_VERSION;
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

  one_third      = 1.0/(3.0);
  two_third      = 2.0/(3.0);
  four_third     = 4.0/(3.0);
  sqrt_three     = sqrt(3.0);
  one_sqrt_three = 1.0/sqrt_three;

  ps->require("FSLOPE",d_cm.FSLOPE);
  ps->require("FSLOPE_p",d_cm.FSLOPE_p);  // not used
  ps->require("hardening_modulus",d_cm.hardening_modulus); //not used
  ps->require("CR",d_cm.CR); // not used
  ps->require("T1_rate_dependence",d_cm.T1_rate_dependence);
  ps->require("T2_rate_dependence",d_cm.T2_rate_dependence);
  ps->require("p0_crush_curve",d_cm.p0_crush_curve);
  ps->require("p1_crush_curve",d_cm.p1_crush_curve);
  ps->require("p3_crush_curve",d_cm.p3_crush_curve);
  ps->require("p4_fluid_effect",d_cm.p4_fluid_effect); // b1
  ps->require("fluid_B0",d_cm.fluid_B0);               // kf
  ps->require("fluid_pressure_initial",d_cm.fluid_pressure_initial);             // Pf0
  ps->require("gruneisen_parameter",d_cm.gruneisen_parameter);             // Pf0
  ps->require("subcycling_characteristic_number",d_cm.subcycling_characteristic_number);
  ps->require("kinematic_hardening_constant",d_cm.kinematic_hardening_constant); // not used
  ps->require("PEAKI1",d_cm.PEAKI1);
  ps->require("B0",d_cm.B0);
  ps->require("G0",d_cm.G0);

  initializeLocalMPMLabels();
}
Arenisca::Arenisca(const Arenisca* cm)
  : ConstitutiveModel(cm)
{
  one_third      = 1.0/(3.0);
  two_third      = 2.0/(3.0);
  four_third     = 4.0/(3.0);
  sqrt_three     = sqrt(3.0);
  one_sqrt_three = 1.0/sqrt_three;

  d_cm.FSLOPE = cm->d_cm.FSLOPE;
  d_cm.FSLOPE_p = cm->d_cm.FSLOPE_p; // not used
  d_cm.hardening_modulus = cm->d_cm.hardening_modulus;  // not used
  d_cm.CR = cm->d_cm.CR;  // not used
  d_cm.T1_rate_dependence = cm->d_cm.T1_rate_dependence;
  d_cm.T2_rate_dependence = cm->d_cm.T2_rate_dependence;
  d_cm.p0_crush_curve = cm->d_cm.p0_crush_curve;
  d_cm.p1_crush_curve = cm->d_cm.p1_crush_curve;
  d_cm.p3_crush_curve = cm->d_cm.p3_crush_curve;
  d_cm.p4_fluid_effect = cm->d_cm.p4_fluid_effect; // b1
  d_cm.fluid_B0 = cm->d_cm.fluid_B0;
  d_cm.fluid_pressure_initial = cm->d_cm.fluid_pressure_initial; //pf0
  d_cm.gruneisen_parameter = cm->d_cm.gruneisen_parameter; //pf0
  d_cm.subcycling_characteristic_number = cm->d_cm.subcycling_characteristic_number;
  d_cm.kinematic_hardening_constant = cm->d_cm.kinematic_hardening_constant;  // not supported
  d_cm.PEAKI1 = cm->d_cm.PEAKI1;
  d_cm.B0 = cm->d_cm.B0;
  d_cm.G0 = cm->d_cm.G0;

  initializeLocalMPMLabels();
}
// DESTRUCTOR
Arenisca::~Arenisca()
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
void Arenisca::outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag)
{
  ProblemSpecP cm_ps = ps;
  if (output_cm_tag) {
    cm_ps = ps->appendChild("constitutive_model");
    cm_ps->setAttribute("type","Arenisca");
  }
  cm_ps->appendElement("FSLOPE",d_cm.FSLOPE);
  cm_ps->appendElement("FSLOPE_p",d_cm.FSLOPE_p); //not used
  cm_ps->appendElement("hardening_modulus",d_cm.hardening_modulus); //not used
  cm_ps->appendElement("CR",d_cm.CR); //not used
  cm_ps->appendElement("T1_rate_dependence",d_cm.T1_rate_dependence);
  cm_ps->appendElement("T2_rate_dependence",d_cm.T2_rate_dependence);
  cm_ps->appendElement("p0_crush_curve",d_cm.p0_crush_curve);
  cm_ps->appendElement("p1_crush_curve",d_cm.p1_crush_curve);
  cm_ps->appendElement("p3_crush_curve",d_cm.p3_crush_curve);
  cm_ps->appendElement("p4_fluid_effect",d_cm.p4_fluid_effect); // b1
  cm_ps->appendElement("fluid_B0",d_cm.fluid_B0); // kf
  cm_ps->appendElement("fluid_pressure_initial",d_cm.fluid_pressure_initial); //Pf0
  cm_ps->appendElement("gruneisen_parameter",d_cm.gruneisen_parameter); //Pf0
  cm_ps->appendElement("subcycling_characteristic_number",d_cm.subcycling_characteristic_number);
  cm_ps->appendElement("kinematic_hardening_constant",d_cm.kinematic_hardening_constant); // not used
  cm_ps->appendElement("PEAKI1",d_cm.PEAKI1);
  cm_ps->appendElement("B0",d_cm.B0);
  cm_ps->appendElement("G0",d_cm.G0);
}

Arenisca* Arenisca::clone()
{
  return scinew Arenisca(*this);
}

void Arenisca::initializeCMData(const Patch* patch,
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
void Arenisca::allocateCMDataAdd(DataWarehouse* new_dw,
                                 ParticleSubset* addset,
            map<const VarLabel*, ParticleVariableBase*>* newState,
                                 ParticleSubset* delset,
                                 DataWarehouse* old_dw)
{
}

// Compute stable timestep based on both the particle velocities
// and wave speed
void Arenisca::computeStableTimestep(const Patch* patch,
                                     //ParticleSubset* pset, //T2D: this should be const
                                     const MPMMaterial* matl,
                                     DataWarehouse* new_dw)
{
  // MH: For fluid-effects or non-linear elasticity, the bulk modulus may is a function
  //     of strain and/or pressure.  We calculate this at the beginning of the step
  //     and hold it constant for all substeps.  This ensures that the stable time step
  //     as well as the the trial stress increment for the step are consistent with the
  //     calculations in each substep.  A more sophisticated approach would be to improve
  //     the algorithm to allow for a different value of the bulk modulus for each substep.
  //     To avoid needing the material state (i.e. the strain relative to ev0) for each
  //     particle, the stable time step is computed with the conservative estimate that
  //     bulk = B0 + B1, which should be greater than the K_eng model for the bulk modulus
  //     of the saturated material for most input properties.
  //
  //define and initialize some variables
  int     dwi = matl->getDWIndex();
  double  c_dil = 0.0,
          bulk = d_cm.B0 + d_cm.p4_fluid_effect, // bulk = B0 + B1
          shear= d_cm.G0;                        // shear modulus
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

  //double delT_new = WaveSpeed.minComponent();
  //new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());

  //cout<<"CST:delT_new="<<delT_new<<endl;

  //if(delT_new < 1.e-12) //T2D: Should this be here?
  //  new_dw->put(delt_vartype(DBL_MAX), lb->delTLabel, patch->getLevel());
  //else
  //  new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
}

// ------------------------------------- BEGIN COMPUTE STRESS TENSOR FUNCTION
/**
Arenisca::computeStressTensor is the core of the Arenisca model which computes
the updated stress at the end of the current timestep along with all other
required data such plastic strain, elastic strain, cap position, etc.

*/
void Arenisca::computeStressTensor(const PatchSubset* patches,
                                           const MPMMaterial* matl,
                                           DataWarehouse* old_dw,
                                           DataWarehouse* new_dw)
{
  // Define some constants
  Matrix3 Identity; Identity.Identity();

  // Get the initial density
  double rho_orig = matl->getInitialDensity();

  // Get the Arenisca model parameters
  const double FSLOPE = d_cm.FSLOPE,        //yield function
               //FSLOPE_p = d_cm.FSLOPE_p,  //flow function
               //hardening_modulus = d_cm.hardening_modulus,
               //CR = d_cm.CR,
               //XXp0 = d_cm.p0_crush_curve,    // initial value of X, used to compute characteristic length
               //p1 = d_cm.p1_crush_curve,
               //p3 = d_cm.p3_crush_curve,
               subcycling_characteristic_number = d_cm.subcycling_characteristic_number,
               //fluid_B0 = d_cm.fluid_B0,
               //kinematic_hardening_constant = d_cm.kinematic_hardening_constant,
               PEAKI1 = d_cm.PEAKI1,
               B0 = d_cm.B0,
               //XXB1 = d_cm.p4_fluid_effect,
               G0 = d_cm.G0;

  // Compute kinematics variables (pDefGrad_new, pvolume, pLocalized_new, pVelGrad_new)
  // computeKinematics(patches, matl, old_dw, new_dw);

  // Global loop over each patch
  for(int p=0;p<patches->size();p++){

    // Declare and initial value assignment for some variables
    const Patch* patch = patches->get(p);
    Matrix3 D;

    double J,
           c_dil=0.0,
           se=0.0;
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

        //MH: the bulk modulus function will be a bilinear function depending
        //    on the total strain relative to the zero fluid pressure strain, ev0
        //    this is a change from the old function that used plastic strain
        //    Since this bulk modulus is used to compute the substeps and trial
        //    stress, it should be held constant over the entire step, and thus
        //    is computed based on initial values.  We must also pass this value
        //    of the bulk modulus to the computeStep function, since the volumetric
        //    strain at the beginning of the step will not be available within the
        //    substep.

        // Hack: Need to modify Arenisca to call step with strain rate and dt, allowing for
        // different bulk modulus for each step.  Also need to modify this to use low
        // pressure bulk modulus in tension, and to use some mid range value between low
        // and high in compression.

        double bulk = B0,
               shear = G0;
/*
        if (d_cm.fluid_B0 != 0.)
        {
          double ev0  = computeev0();                // strain at zero pore pressure
          bulk  = computeBulkModulus( ev0 - 1 );     // Compute bulk modulus with fluid effects
        }
*/
        // Compute the lame constant using the bulk and shear moduli
        double lame       = bulk - two_third*shear,
               threeKby2G = (3.0 * bulk) / (2.0 * shear);

        // Compute the unrotated stress at the first of the current timestep
        Matrix3 unrotated_stress = (tensorR.Transpose())*(pStress_old[idx]*tensorR);

        // Compute the unrotated trial stress for the full timestep
        Matrix3 stress_diff_step  = (Identity*lame*(D.Trace()*delT) + D*delT*2.0*shear),
                trial_stress_step = unrotated_stress + stress_diff_step;

        if (isnan(trial_stress_step.Norm())) {  //Check stress_iteration for nan
          cerr << "pParticleID=" << pParticleID[idx];
          throw InvalidValue("**ERROR**: Nan in trial_stress_step", __FILE__, __LINE__);
        }

        // Compute stress invariants of the trial stress for the full timestep
        double I1_trial_step,
               J2_trial_step;
        Matrix3 S_trial_step;
        computeInvariants(trial_stress_step, S_trial_step, I1_trial_step, J2_trial_step);

        // Compute the value of the test yield function at the trial stress.  This will
        // return +/- 1 for plastic and elastic states, respectively, or 0 if the state
        // is on the yield surface.
        f_trial_step[idx] = YieldFunction(I1_trial_step,
                                          J2_trial_step,
                                          pCapX[idx],
                                          pZeta[idx],
                                          threeKby2G);

        // initial assignment for the updated values of plastic strains, volumetric
        // part of the plastic strain, volumetric part of the elastic strain, \kappa,
        // and the backstress. tentative assumption of elasticity
        pevp_new[idx]   = pevp[idx];
        peve_new[idx]   = peve[idx] + D.Trace()*delT;
        pCapX_new[idx]  = pCapX[idx];
        pKappa_new[idx] = pKappa[idx];
        pZeta_new[idx]  = pZeta[idx];
        pep_new[idx]    = pep[idx];

        // allocate and assign step values
        double  evp_new_step    = pevp_new[idx],
                eve_new_step    = peve_new[idx],
                X_new_step      = pCapX_new[idx],
                Kappa_new_step  = pKappa_new[idx],
                Zeta_new_step   = pZeta_new[idx];
        Matrix3 ep_new_step     = pep_new[idx],
                stress_new_step = pStress_new[idx];


        // MH: We now check if the entire step is elastic.  If it is, we update the
        //     new stress to be our trial stress and compute the new elastic strain.
        //     The plastic strain and internal state variables are unchanged.
        if (f_trial_step[idx]<=0){  // elastic

          // An elastic step: the updated stres at the end of the current time step
          // is equal to the trial stress. otherwise, the plasticity return algrithm would be used.
          stress_new_step = trial_stress_step;
          #ifdef JC_DEBUG_PARTICLE // print characteristic length of yeild surface
          if(pParticleID[idx]==JC_DEBUG_PARTICLE){
            cout << " elastic step";
          }
          #endif

        }else{  // plastic

          // An elasto-plasic/fully plastic step: the plasticity return algrithm should be used.
          // We first subdivide our trial stress into smaller increments.  To determine a suitable
          // subdivision, we compare the magnitude of the stress difference (trial stress - old stress)
          // over the entire step to a characteristic length of the yield surface.

          ////////////////////////////////////////////////////////////////
          //COMPUTE CHARACTERISTIC LENGTHS
          double clenI1,
                 clensqrtJ2,
                 clen;

          // the characteristic length for volumetric terms (units of I1, Pa) is the
          // distance from the vertex to the cap along the hydrostat.  To provide a
          // a measure in the case of no cap, we also compute the value of stress
          // corresponding to 0.1% volumetric strain.
          clenI1 = min( bulk/1000 , PEAKI1 - pCapX[idx] );

          // Similarly, for the deviator, the characteristic length the characteristic
          // length (units of sqrt(J2), Pa), is the value of linear drucker-prager
          // surface at X, or a stress corresponding to 0.1% shear strain.
          clensqrtJ2 = min( 2*G0/1000 , FSLOPE * (PEAKI1 - pCapX[idx]) );

          // the general characteristic length (units of I1 and sqrt(J2), Pa)
          clen = sqrt( clenI1*clenI1 + clensqrtJ2*clensqrtJ2 );

          #ifdef JC_DEBUG_PARTICLE // print characteristic length of yeild surface
          if(pParticleID[idx]==JC_DEBUG_PARTICLE){
            cout << " clen=" << clen << ", B0e-3=" << B0/1000
                 << ", PEAKI1-p0=" << PEAKI1-p0 << ", 2G0e-3=" <<2*G0/1000
                 << ", FSLOPE*(PEAKI1-p0)=" << FSLOPE*(PEAKI1-p0);
          }
          #endif

          /* MH: Removed this
          // If the characteristic length gets a negative value, it means that there is an issue
          // with the yield surface, which should be reported.
          if (clen<=0.0 || clenI1 <= 0 || clensqrtJ2<=0.0) {
            cout<<"ERROR! in clen"<<endl;
            cout<<"pParticleID="<<pParticleID[idx]<<endl;
            cout<<"clen="<<clen<<endl;
            throw InvalidValue("**ERROR**:in characteristic length of yield surface (clen)",
                               __FILE__, __LINE__);
          }
          */

          //////////////////////////////////////////////////////////////////////
          //SUBCYCLING

          // create and initialize flag variable for substep;
          int flag_substep = 0,
              massive_subcycling_flag = 1,
              massive_subcycling_counter = 1;

          // Compute total number of cycles in the plasticity subcycling
      //   Will be the subcycling characteristic number unless stress_diff>clen
          double num_steps = subcycling_characteristic_number*(
                             floor(stress_diff_step.Norm()/clen) + 1.0);

          #ifdef CSM_FORCE_MASSIVE_SUBCYCLING
          num_steps = num_steps * CSM_FORCE_MASSIVE_SUBCYCLING;
          #endif

          #ifdef JC_DEBUG_PARTICLE // print characteristic length of yield surface
          if(pParticleID[idx]==JC_DEBUG_PARTICLE){
            cout << ", num_steps=" << num_steps
                 << ", stress_diff_step.Norm()="<<stress_diff_step.Norm();
          }
          #endif
          if (isnan(num_steps)) {  //Check stress_iteration for nan
             cerr << "pParticleID=" << pParticleID[idx]
                  << ", num_steps=" << num_steps << endl;
            throw InvalidValue("**ERROR**: Nan in num_steps", __FILE__, __LINE__);
          }

          // define and initialize values for the substep
          double evp_new_substep   = evp_new_step,
                 eve_new_substep   = eve_new_step,
                 X_new_substep     = X_new_step,
                 Kappa_new_substep = Kappa_new_step,
                 Zeta_new_substep  = Zeta_new_step,
                 num_substeps;

          Matrix3 ep_new_substep = ep_new_step,
                  trial_stress_substep,
                  stress_diff_substep,
                  stress_new_substep = trial_stress_step - stress_diff_step;

          while(massive_subcycling_flag == 1
                && massive_subcycling_counter <= 4){

            // modify the number of subcycles depending on success
            //  1st time through, num_subcycles remains the same.
            //  2nd time through, num_subcycles is multiplied by 10
            //  3rd time trhrugh, num_subcycles is multiplied by 100
            //  4th and list time, num_subcycles is multiplied by 1000
            num_substeps = num_steps * Pow(10,massive_subcycling_counter-1);

            if(num_substeps > 15000){  //T2D: this might change, but keep high for node success
              cout << "WARNING: $num_subcycles=" << num_substeps
                   << " exceeds 15000 maximum for pID=" << pParticleID[idx] << endl;
//              #ifndef CSM_FORCE_MASSIVE_SUBCYCLING
              num_substeps=15000;
//              #endif
            }

            // Set initial values for the substep
            evp_new_substep   = evp_new_step,
            eve_new_substep   = eve_new_step,
            X_new_substep     = X_new_step,
            Kappa_new_substep = Kappa_new_step,
            Zeta_new_substep  = Zeta_new_step;
            ep_new_substep    = ep_new_step,

            // Remove the new changes from the trial stress so we can apply the changes in each sub-cycle
            trial_stress_substep = trial_stress_step - stress_diff_step; //unrotated stress

            // Changes in the trial stress in each sub-cycle assuming the elastic behavior
            stress_diff_substep = stress_diff_step / num_substeps;

            //initial assignment
            stress_new_substep = trial_stress_substep;

            // Loop over sub-cycles in the plasticity return algorithm
            for (int substep_counter=0 ; substep_counter<=num_substeps-1 ; substep_counter++){
              // Compute the trial stress for the current sub-cycle
              trial_stress_substep = stress_new_substep + stress_diff_substep;

              #ifdef JC_DEBUG_PARTICLE  // Print number of subcycles
              if(pParticleID[idx]==JC_DEBUG_PARTICLE)
                cout << endl << "  SCc=" << substep_counter << "/"<<num_substeps-1;
              #endif
              ///////////////////////////////////////
              // COMPUTE STRESS TENSOR FOR SUBSTEP
              //  flag_substep = 0 means no flag thrown (good)

              // MH: We call computeStressTensorStep with the bulk modulus for
              // the entire step, since this must remain constant between substeps
              // to be consistent with our intial trial stress definition:
              flag_substep = computeStressTensorStep(trial_stress_substep,
                                                     stress_new_substep,
                                                     ep_new_substep,
                                                     evp_new_substep,
                                                     eve_new_substep,
                                                     X_new_substep,
                                                     Kappa_new_substep,
                                                     Zeta_new_substep,
                                                     bulk, // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                                                     pParticleID[idx]);

              if(flag_substep!=0) //if flag thrown stop subcycling, end for loop
                substep_counter=num_substeps;

            } //End of subcycle loop

            //if no flag was thrown during computeStressTensorStep do not substep again
            if(flag_substep==0) //end while loop
              massive_subcycling_flag = 0;
            else{ //flag thrown, redo subcycling with more step
              massive_subcycling_flag = 1;
              massive_subcycling_counter++;

              //report warning
              cout<< "WARNING: massive subcycling needed with flag_substep="
                  << flag_substep <<endl;
            }
          }//end while loop

          //END OF SUBCYCLING ROUTINE for step
          evp_new_step    = evp_new_substep;
          eve_new_step    = eve_new_substep;
          X_new_step      = X_new_substep;
          Kappa_new_step  = Kappa_new_substep;
          Zeta_new_step   = Zeta_new_substep;
          ep_new_step     = ep_new_substep;
          stress_new_step = stress_new_substep;

          //Complete final check that after subcycling we are on the yield surface

          // Compute the invariants of the new stress for the step
          //double J2_new_step,I1_new_step;
          //Matrix3 S_new_step;
          //computeInvariants(stress_new_step, S_new_step, I1_new_step, J2_new_step);

          // Compute the yield function at the returned back stress to check
          // if it correctly returned back to the yield surface or not?
          //double f_new_step=YieldFunction(I1_new_step, J2_new_step, X_new_step,
          //                                Kappa_new_step, Zeta_new_step);

          //// If the new stress is not on the yield surface send error message to the host code.s
          //if (sqrt(abs(f_new_step))>1.0*clen)
          //{
          //  cerr<<"ERROR!  did not return to <1e-1 yield surface (Arenisca.cc)"<<endl;
          //  cerr<<"pParticleID="<<pParticleID[idx]<<endl;
          //  cerr<<"pDefGrad_new="<<pDefGrad_new[idx]<<endl<<endl;
          //  cerr<<"pDefGrad_old="<<pDefGrad[idx]<<endl;
          //  //cerr<<"condition_return_to_vertex="<<condition_return_to_vertex<<endl;
          //  cerr<<"J2_new_step= "<<J2_new_step<<endl;
          //  cerr<<"I1_new_step= "<<I1_new_step<<endl;
          //  cerr<<"f_new_step= "<<f_new_step<<endl;
          //  cerr<<"clen= "<<clen<<endl;
          //  cerr<<"pStress_old[idx]="<<pStress_old[idx]<<endl;
          //  cerr<<"pStress_new[idx]="<<pStress_new[idx]<<endl;
          //  throw InvalidValue("**ERROR**:did not return to yield surface ",
          //                     __FILE__, __LINE__);
          //}
        }

        pevp_new[idx]    = evp_new_step;
        peve_new[idx]    = eve_new_step;
        pCapX_new[idx]   = X_new_step;
        pKappa_new[idx]  = Kappa_new_step;
        pZeta_new[idx]   = Zeta_new_step;
        pIota_new[idx]   = 0.0; //T2D: Emad
        pep_new[idx]     = ep_new_step;
        pStress_new[idx] = stress_new_step;

        // T2D: Cindy ADD RATE DEPENDENCE CODE HERE
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

        // Compute wave speed + particle velocity at each particle,
        // store the maximum
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

  #ifdef JC_DEBUG_PARTICLE // Print plastic work //T2D UNFINISHED aka wrong!!
        // Verify plastic work is positive
        //Given: AvgStress, delT, stress_new, stress_old, bulk, shear,
        if(pParticleID[idx]==JC_DEBUG_PARTICLE){
          //Matrix3 AvgStressRate = (pStress_new[idx] + pStress_old[idx])/delT;
          //Matrix3 AvgStressRateIso = one_third*AvgStressRate.Trace()*Identity;
          //Matrix3 EdE = D-0.5/shear*(AvgStressRate-AvgStressRateIso)-one_third/bulk*AvgStressRateIso;
          //Matrix3 PSR = (pep_new[idx] + pep[idx])/delT;  //AvgepRate
          //double pPlasticWork = (PSR(0,0)*AvgStress(0,0) +
          //                       PSR(1,1)*AvgStress(1,1) +
          //                       PSR(2,2)*AvgStress(2,2) +
          //                   2.*(PSR(0,1)*AvgStress(0,1) +
          //                       PSR(0,2)*AvgStress(0,2) +
          //                       PSR(1,2)*AvgStress(1,2)));
          ////cout << ",pPlasticWork=" << pPlasticWork;
        }
  #endif

        // Compute the strain energy associated with the particle
        double e = (D(0,0)*AvgStress(0,0) +
                    D(1,1)*AvgStress(1,1) +
                    D(2,2)*AvgStress(2,2) +
                2.*(D(0,1)*AvgStress(0,1) +
                    D(0,2)*AvgStress(0,2) +
                    D(1,2)*AvgStress(1,2))) * pvolume[idx]*delT;

        // Accumulate the total strain energy
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

//
int Arenisca::computeStressTensorStep(const Matrix3& sigma_trial, // trial stress tensor
                                      Matrix3& sigma_new,         // stress tensor
                                      Matrix3& ep_new,            // plastic strain tensor
                                      double&  evp_new,           // vol plastic strain
                                      double&  eve_new,           // vol elastic strain
                                      double&  X_new,             // cap intercept (shifted)
                                      double&  Kappa_new,         // branch point (shifted)
                                      double&  Zeta_new,          // trace of isotropic backstress
                                      double&  bulk,              // bulk modulus for the step !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                                      long64   ParticleID)
{
  // Define and initialize some variables

  // Many of the inputs are pointers that will be overwritten with updated values.  The input values are
  // stored as _old
  Matrix3 sigma_old = sigma_new,
          ep_old    = ep_new;
  double  evp_old   = evp_new,
          //XXeve_old   = eve_new,
          X_old     = X_new,
          //XXKappa_old = Kappa_new,
          Zeta_old  = Zeta_new;

  double  FSLOPE = d_cm.FSLOPE,                         // slope of the linear drucker prager surface in rootJ2 vs. I1
          //XXPEAKI1 = d_cm.PEAKI1,                         // shifted I1 value of the vertex
          //bulk   = computeBulkModulus(eve_old+evp_old), // tangent bulk modulus for the step (stress)
          shear  = d_cm.G0;                             // tangent shear modulus for the step (stress)
  int     n      = 1,                                   // iteration counter
          nmax   = 100;                                 // allowable iterations
  Matrix3 ee_new,      // elastic strain tensor at the end of the step.
          Identity;    // identity tensor
  Identity.Identity(); // initialize identity tensor

  // Frequently used constants from elastic properties
  double  //XXtwoGby3K      = (2.0*shear) / (3.0*bulk),               // scale deviator from closest point space
          threeKby2G    = (3.0*bulk)  / (2.0*shear),              // scale deviator to closest point space
          oneby9k_1by6G = (1.0/(9.0*bulk)-1.0/(6.0*shear)),       // used to compute strain from stress
          oneby2G       = 1.0/(2.0*shear);                        // used to compute strain from stress

  // Compute invarients of the trial stress
  double  I1_trial,
          J2_trial;
  Matrix3 S_trial;

  // The computeInvarients function overwrites the input argumets for S, I1, and J2
  computeInvariants(sigma_trial,S_trial,I1_trial,J2_trial);

  // Shifted and transformed coordinates for closest-point space.
  // Note, the shifted value for z_trial will change as the backstress
  // is changed in the iterative solution
  double  Beta = FSLOPE*threeKby2G*sqrt(6.0),  //slope of r vs z in transformed space
          z_trial = (I1_trial - Zeta_old)*one_sqrt_three,
          r_trial = threeKby2G * sqrt(2.0*J2_trial);

  // Checking for elastic or plastic step:
  if( TransformedYieldFunction(r_trial,z_trial,X_old,Beta) <= 0 )
  {
    // =========================================================== ELASTIC STEP
    // Update stress to trial stress, and update volumetric elastic strain,
    // which is done after the end of the plastic loop.  All other pointers
    // are unchanged.
    sigma_new = sigma_trial;
  }
  else
  {
    // =========================================================== PLASTIC STEP
    // The step is at least partially plastic, so we use an iterative implicity
    // method to find the stress, compute the plastic strain and to update the
    // internal state variables.  The basic algorithm is as follows:
    //
    // (1) Compute the return to the initial yield surface assuming the yield
    //     surface is fixed (no hardening, softening, backstress evolution, etc.)
    // (2) Compute the increment in plastic strain for the non-hardening solution;
    //     the actual increment (with hardening) will be a multiple of this, and
    //     we will solve for the multiplier eta, 0<eta<1
    // (3) Compute the updated yield surface as a function of the scaled increment
    //     in plastic strain.
    // (4) Compute the increment in plastic strain for the return to the updated
    //     yield surface.
    // (5) Compare the new plastic strain increment to the scaled non-hardening
    //     and adjust eta using a bisection method; Return to (3) until converged.
    //
    // In computing the return to the yield surface, we map to a stress space in
    // which the deviator has been scaled by a ratio of 3K/2G.  In that space, the
    // solution is a closest-point return.  This is computed without iteration by
    // making use of a flow function (g) that is defined such that the value at
    // any point in the plastic domain is the distance in the transformed space to
    // the closest point on the yield surface.  As such, the gradient of the flow
    // function evaluated at the trial stress gives the return direction, and the
    // product grad(g)*g gives the stress increment that returns the trial stress
    // to the closest point.  Once the new stress is computed, it is transformed
    // back to the normal stress space.  With this method, no special treatment
    // of the vertex is needed.
    // ........................................................................

    // Initialize variables needed for plastic solution
    double  gfcn,          // value of the flow function
            r_new0 = r_trial,        // transformed r for non-hardening return
            z_new0 = z_trial,        // transformed, shifted z for non hardening return
            r_new = r_trial,         // transformed r for hardening return
            z_new = z_trial,         // transformed, shifted z for hardening return
            eta_out = 1.0, // inner bound for plastic scaler
            eta_in  = 0.0, // outer bound for plastic scaler
            eta_mid,       // solution for plastic scaler
            eps_eta = 1.0, // convergence measure: eta_out-eta_in
            TOL = 1.0e-11;  // convergence tolerance on eps_eta

    Matrix3 sigma_new0,   // non-hardening return stress
            d_sigma0,     // non-hardening increment stress over step
            d_sigmaT = sigma_trial - sigma_old,
            d_sigma,      // hardening increment stress over step
            d_e,          // total strain increment over the step
            d_ee0,        // non-hardening increment in elastic strain
            d_ep0,        // non-hardening increment in plastic strain
            d_ee,         // hardening increment in elastic strain
            d_ep;         // hardening increment in plastic strain

    // Non-hardening Closest Point Return in Transformed Space
    /*
    gfcn   = TransformedFlowFunction(r_trial,z_trial,X_old,Beta);
    r_new0 = r_trial - gfcn * dgdr(r_trial,z_trial,X_old,Beta);
    z_new0 = z_trial - gfcn * dgdz(r_trial,z_trial,X_old,Beta);
    */

    // Compute non-hardening return, (overwriting r_new0, and z_new0)
    gfcn = ComputeNonHardeningReturn(r_trial, z_trial, X_old, Beta, r_new0, z_new0);

    // Update unshifted untransformed stress
    sigma_new0 = one_third*(sqrt_three*z_new0+Zeta_old)*Identity;
    if ( r_trial != 0.0 )
      sigma_new0 = sigma_new0 + (r_new0/r_trial)*S_trial;

    // Stress increment for non-hardening return
    d_sigma0 = sigma_new0 - sigma_old;

    // Increment in total strain from sigma_old to sigma_trial
    d_e = oneby2G*d_sigmaT + oneby9k_1by6G*d_sigmaT.Trace()*Identity;

    // Increment in elastic strain for the non-hardening return
    d_ee0 = oneby2G*d_sigma0 + oneby9k_1by6G*d_sigma0.Trace()*Identity;

    // Increment in plastic strain for the non-hardening return
    d_ep0 = d_e - d_ee0;

    /*
    cout << endl << "Non-Hardening step, n = " << n
    << ", Trace d_ee = "  << d_e.Trace()
    << ", Trace d_ee0 = " << d_ee0.Trace()
    << ", Trace d_ep0 = " << d_ep0.Trace()
    << ", Trace d_sigma0 = " << d_sigma0.Trace()
    << ", r_trial = " << r_trial
    << ", r_new0 = " << r_new0
    << ", z_trial = " << z_trial
    << ", z_new0 = " << z_new0
    << ", gfcn = " << gfcn <<endl;
    */

    // loop until the value of eta_mid is converged or max iterations exceeded
    while( (eps_eta > TOL || evp_new <= -d_cm.p3_crush_curve) && (n <= nmax) )
    {
      n++;
      // mid-point bisection on plastic strain scaler
      eta_mid = 0.5*(eta_in + eta_out);

      // plastic strain increment is a multiple of the non-hardening value
      ep_new   = ep_old + eta_mid*d_ep0;
      evp_new  = ep_new.Trace();

      // Exact update of the cap:
      X_new    = computeX(evp_new);

      // Explicit update of zeta based on dZetadevp at beginning of step.
      Zeta_new = Zeta_old + computedZetadevp(Zeta_old,evp_old)*(eta_mid*d_ep0.Trace());

      // Recompute the shifted trial stress:
      z_trial = (I1_trial - Zeta_new) * one_sqrt_three;

      // Check if updated yield surface encloses the yield surface, or if the
      // z-component d_min_subcycles_for_Fof the return direction to the updated yield surface has
      // changed sign.  If either of these has occured the increment in plastic
      // strain was too large, so we scale back the multiplier eta.

      // Compute non-hardening return, (overwriting r_new, and z_new)
      gfcn = ComputeNonHardeningReturn(r_trial, z_trial, X_new, Beta, r_new, z_new);

      if( TransformedYieldFunction( r_trial,z_trial,X_new,Beta ) <= 0.0 ||
          //Sign(dgdz(r_trial,z_trial,X_old,Beta)) != Sign(dgdz(r_trial,z_trial,X_new,Beta))
          Sign(z_trial-z_new0) != Sign(z_trial-z_new)
        )
      {
        /*
        cout << "BAD step, n = " << n
        << ", eta_in = " << eta_in
        << ", eta_out = " << eta_out
        << ", eps_eta = " << eps_eta
        << ", evp_new = " << evp_new
        << ", eve_new = " << eve_new
        << ", ZcapX_new = " << X_new*sqrt_three
        << ", Zeta_new = " << Zeta_new<<endl
        << ", r_trial = " << r_trial
        << ", r_new = " << r_new
        << ", z_trial = " << z_trial
        << ", z_new = " << z_new
        << ", f(r,z,x,beta) = " << TransformedYieldFunction(r_trial,z_trial,X_new,Beta)<<endl;
        */
        eta_out = eta_mid;
      }

      else
      {
        // Our updated yield surface has passed the above basic checks for a bad
        // update so we compute the increment in plastic strain for a return to
        // the updated surface, compare this to our scaled value of the non-
        // hardening return, and adjust the scale parameter, eta, accordingly.

        // Hardening Closest Point Return in Transformed Space
        /* already computed before the last test
          gfcn  = TransformedFlowFunction(r_trial,z_trial,X_new,Beta);
          r_new = r_trial - gfcn * dgdr(r_trial,z_trial,X_new,Beta);
          z_new = z_trial - gfcn * dgdz(r_trial,z_trial,X_new,Beta);
        */

        // Update unshifted untransformed stress
        sigma_new = one_third*(sqrt_three*z_new+Zeta_new)*Identity;
        if (r_trial!=0)
          sigma_new = sigma_new + (r_new/r_trial)*S_trial;

        // Stress increment for non-hardening return
        d_sigma = sigma_new - sigma_old;

        // Increment in elastic strain for the hardening solution: strain = (C^-1) : stress
        d_ee =  oneby2G*d_sigma + oneby9k_1by6G*d_sigma.Trace()*Identity;

        // Increment in plastic strain for the non-hardening solution
        d_ep = d_e - d_ee;

        // Compare magnitude of the computed plastic strain to the scaled
        // non-hardening value and adjust eta_mid accordingly
        if ( d_ep.Norm() > eta_mid*d_ep0.Norm() )
        {
          eta_in  = eta_mid;  // too little plastic strain
        }
        else
        {
          eta_out = eta_mid;  // too much plastic strain
        }

        eps_eta = eta_out - eta_in; // Only allow convergence in a good step.

        /*
        cout << "GOOD step, n = " << n
             << ", eta_in = " << eta_in
             << ", eta_out = " << eta_out
             << ", eps_eta = " << eps_eta
             << ", evp_new = " << evp_new
             << ", eve_new = " << eve_new
             << ", ZcapX_new = " << X_new/sqrt_three
             << ", Zeta_new = " << Zeta_new<<endl
             << ", r_trial = " << r_trial
             << ", r_new = " << r_new
             << ", z_trial = " << z_trial
             << ", z_new = " << z_new
             << ", gfcn = " << gfcn
             << ", f = " << TransformedYieldFunction(r_trial,z_trial,X_new,Beta)<<endl;
        */

      } // end return to updated surface to adjust eta_mid
    } // end while loop to find eta_mid
  } // end plastic section

  // Update elastic volumetric strain from converged result for sigma_new.
  // All other variables that were called by reference in the function
  // have been updated in previous steps.
  ee_new = oneby2G*sigma_new + oneby9k_1by6G*sigma_new.Trace()*Identity;
  eve_new = ee_new.Trace();

  // Checks for a bad update
  // -----------------------
  if( n == nmax )
  {
    cout << "(1) Plastic strain scalar (eta) did not converge in nmax iterations"<< "@line:" << __LINE__;
    return 1;
  }
  else if( evp_new <= -d_cm.p3_crush_curve )
  {
    cout << "(2) exceeded max allowable volumetric plastic strain"<< "@line:" << __LINE__;
    return 2;
  }
  else if(isnan(sigma_new.Norm()) ||
          isnan(ep_new.Norm())    ||
          isnan(evp_new)          ||
          isnan(eve_new)          ||
          isnan(X_new)            ||
          isnan(Kappa_new)        ||
          isnan(Zeta_new) )
  {
    cout << "(3) NAN in output"<< "@line:" << __LINE__;
    return 3;
  }
  /*
  else if(delta > delta_TOL)
   {
    cout << "(4) magnitude difference of hardening and non-hardening returns exceeds allowable"<< "@line:" << __LINE__;
    return 4;
   }
  else if(theta > theta_TOL)
   {
    cout << "(5) angle between of hardening and non-hardening returns exceeds allowable"<< "@line:" << __LINE__;
    return 5;
   }
  */
  else
  {  // updated states has passed error checks
    return 0; //No flags thrown
  }
}

void Arenisca::computeInvariants(const Matrix3& stress, Matrix3& S,  double& I1, double& J2)
{
  // Compute the invariants of a second-order tensor

  Matrix3 Identity;
  Identity.Identity();

  // Compute the first invariants
  I1 = stress.Trace();  //Pa

  // Compute the deviatoric part of the tensor
  S = stress - Identity*(I1/3.0);  //Pa

  // Compute the first invariants
  J2 = 0.5*S.Contract(S);  //Pa^2

  if(sqrt(J2) < 1e-8*sqrt(Pow(I1,2)+J2))
    J2=0;
}

// MH! Note the input arguements have changed!!!!!
// Calls the Transformed Yield Function with Untransformed Arguments
double Arenisca::YieldFunction(const double& I1,   // Unshifted
                               const double& J2,   // Untransformed
                               const double& X,    // Shifted
                               const double& Zeta, // Trace of backstres
                               const double& threeKby2G) // (3*K)/(2*G)

{
  // Calls the transformed yield function with untransformed arguments.
  // This is a yield test function and is used only to compute whether
  // a trial stress state is elastic or plastic.  Thus only the sign of
  // the returned value is meaningful.  The function returns {-1,0,+1}
  // The flow function is to be used to quantify the distance to the
  // yield surface.

  // define and initialize some variables
  double FSLOPE = d_cm.FSLOPE,
         R,
         Z,
         Beta,
         f;

  R = sqrt(2*J2)*threeKby2G;
  Z = one_sqrt_three*(I1 - Zeta);
  Beta = FSLOPE*threeKby2G*sqrt(6.0);
  f = TransformedYieldFunction(R,Z,X,Beta);

  //cout << " YieldFxn:I1="<<I1<<", J2="<<J2<<", X="<<X
  //     <<", Zeta="<<Zeta<<", threeKby2G="<<threeKby2G
  //     <<", R="<<R<<", Z="<<Z<<", Beta="<<Beta
  //     <<",transf="<< f
  //     <<",signtransf="<<Sign( f ) << endl;

  f = Sign( f ); // + plastic, - elastic

  return f;
}

// MH! START Compute Non Hardening Return (Elliptical Cap)
double Arenisca::ComputeNonHardeningReturn(const double& R,   // Transformed Trial Stress
                                             const double& Z,   // Shifted Trial Stress
                                             const double& CapX,
                                             const double& Beta,
                                             double& r_new,
                                             double& z_new)
{
  // ===========================================================================
  // MH! START Compute Non Hardening Return (Elliptical Cap)
  // ===========================================================================
  // Computes return to a stationary yield surface.
  //
  // This function overwrites new values for r_new and z_new, and also returns
  // a value for the distance in the transformed space from the trial stress to
  // the closest point on the yield surface.
  //
  // This function is defined in a transformed and shifted stress space:
  //
  //       r = (3K/2G)*sqrt(2*J2)           z = (I1-Zeta)/sqrt(3)
  //
  //define and initialize some varialbes
  double Beta2   = Beta*Beta,
         CapR    =  d_cm.CR,
         ZVertex = d_cm.PEAKI1/sqrt(3),
         ZCapX   = CapX/sqrt(3),
         ZKappa,
         RKappa,
         ZApex,
         dgdr,
         dgdz,
         g;

  double CapR2 = CapR*CapR,
         R2 = R*R;

  // Shifted Z component of the branch point:
  ZKappa = ZCapX - (Beta*ZCapX)/Sqrt(Beta2 + CapR2) + (Beta*ZVertex) / Sqrt(Beta2 + CapR2);
    // Transformed R Component of the branch point
  RKappa = (Beta*(-Beta + Sqrt(Beta2 + CapR2))*(-ZCapX + ZVertex)) / Sqrt(Beta2 + CapR2);
    // Shifted Z component of the apex:
  ZApex = (CapR2*ZCapX + Beta*(-Beta + Sqrt(Beta2 + CapR2))*(-ZCapX + ZVertex)) / CapR2;

  // Region I - closest point return to vertex
  if(R <= (Z - ZVertex)/Beta)
  {
    g = Sqrt(R2 + Pow(Z - ZVertex,2));
    dgdr = R/Sqrt(R2 + Pow(Z - ZVertex,2));
    dgdz = (Z - ZVertex)/Sqrt(R2 + Pow(Z - ZVertex,2));

    r_new = R - g*dgdr;
    z_new = Z - g*dgdz;

  }
  // Region II - closest point return to the linear Drucker-Prager surface
  else if( (Z>ZKappa) && (R - RKappa <= ( Z - ZKappa ) / Beta) )
  {
    g = (R + Beta*Z - Beta*ZVertex)/Sqrt(1 + Beta2);
    dgdr = 1/Sqrt(1 + Beta2);
    dgdz = Beta/Sqrt(1 + Beta2);

    r_new = R - g*dgdr;
    z_new = Z - g*dgdz;
  }
  // Region III - closest point return to an elliptical cap
  else
  {
   //An iterative bisection method is used to compute the return to the elliptical cap.
   // Solution is found in the first quadrant then remapped to the correct region.

   double X = Abs(Z-ZApex), // Relative x-position
          Y = Abs(R), //Relative y-position
          a = (Beta*(-Beta + Sqrt(Beta2 + CapR2))*(-ZCapX + ZVertex)) / CapR2,
          b = (Beta*(-Beta + Sqrt(Beta2 + CapR2))*(-ZCapX + ZVertex)) / CapR,
          Theta = 0.0,
          Theta_in = 0.0,
          Theta_out = 1.570796326794897,
          TOL = 1.0e-11;

   //while (Abs( Sqrt( Pow(X-a*cos(Theta_out),2) + Pow(Y-b*sin(Theta_out),2) ) -
   //            Sqrt( Pow(X-a*cos(Theta_in),2) + Pow(Y-b*sin(Theta_in),2) ) ) > TOL)
   while(Abs(Theta_out-Theta_in)>TOL)
   {
     Theta = (Theta_out+Theta_in)/2;
     if ((2*a*(X-a*cos(Theta))*sin(Theta)-2*b*cos(Theta)*(Y-b*sin(Theta)))
                          /(2*Sqrt(Pow(X-a*cos(Theta),2)+Pow(Y-b*sin(Theta),2))) > 0)
       Theta_out = Theta;
     else
       Theta_in = Theta;
   }

   g = Sqrt( Pow(X-a*cos(Theta),2) + Pow(Y-b*sin(Theta),2) );
   r_new = b*sin(Theta);
   z_new = Sign(Z-ZApex)*a*cos(Theta) + ZApex;
  }

  // Return the value of the flow function, which equals the distance from the trial
  // stress to the closest point on the yield surface in the transformed space.
  return g;

  // ===========================================================================
  // MH! END Compute Non Hardening Return
  // ===========================================================================
}

//START Compute Tranformed Yield Function
double Arenisca::TransformedYieldFunction(const double& R,   // Transformed Trial Stress
                                                   const double& Z,   // Shifted Trial Stress
                                                   const double& CapX,
                                                   const double& Beta)
{
  // ===========================================================================
  // MH! START Compute Tranformed Yield Function
  // ===========================================================================
  // Computes The sign of the transformed yield function, which will be +1 in the plastic
  // region and 0 on the boundary of the yield function.
  //
  // This function is defined in a transformed and shifted stress space:
  //
  //       r = (3K/2G)*sqrt(2*J2)           z = (I1-Zeta)/sqrt(3)
  //
  // define and initialize some variables:

  double Beta2   = Beta*Beta,
         CapR    = d_cm.CR,
         ZVertex = d_cm.PEAKI1/sqrt(3),
         ZCapX   = CapX/sqrt(3),
         ZKappa,
         RKappa,
         ZApex,
         f = 0.0; // sign of the yield function

  double CapR2 = CapR*CapR;

  // Shifted Z component of the branch point:
  ZKappa = ZCapX - (Beta*ZCapX)/Sqrt(Beta2 + CapR2) + (Beta*ZVertex) / Sqrt(Beta2 + CapR2);
    // Transformed R Component of the branch point
  RKappa = (Beta*(-Beta + Sqrt(Beta2 + CapR2))*(-ZCapX + ZVertex)) / Sqrt(Beta2 + CapR2);
    // Shifted Z component of the apex:
  ZApex = (CapR2*ZCapX + Beta*(-Beta + Sqrt(Beta2 + CapR2))*(-ZCapX + ZVertex)) / CapR2;

  // Region I - closest point return to vertex
  if(R <= (Z - ZVertex)/Beta)
  {
    if( (R != 0.0) || (Z != ZVertex) )
      f = 1.0;
  }

  // Region II - closest point return to the linear Drucker-Prager surface
  else if( (Z>ZKappa)&&(R - RKappa <= ( Z - ZKappa ) / Beta) )
  {
    if( R > Beta*(ZVertex-Z) )
      f = 1.0;
    else if ( R < Beta*(ZVertex-Z) )
      f = -1.0;
  }

  // Region III - closest point return to an elliptical cap
  else
  {
   //An iterative bisection method is used to compute the return to the elliptical cap.
   // Solution is found in the first quadrant then remapped to the correct region.

    double X = Abs(Z-ZApex), // Relative x-position
           Y = Abs(R), //Relative y-position
           a = (Beta*(-Beta + Sqrt(Beta2 + CapR2))*(-ZCapX + ZVertex)) / CapR2,
           b = (Beta*(-Beta + Sqrt(Beta2 + CapR2))*(-ZCapX + ZVertex)) / CapR;

    if( (Z-ZApex)*(Z-ZApex)/(a*a)+R*R/(b*b) - 1 < 0 )
        f = -1.0;
    else if ( (X*X)/(a*a)+(Y*Y)/(b*b) - 1 > 0 )
        f = 1.0;
  }

  // Return the sign of the yield function
  return f;
  // ===========================================================================
  // MH! END Compute Tranformed Yield Function
  // ===========================================================================
}

void Arenisca::addRequiresDamageParameter(Task* task,
    const MPMMaterial* matl,
    const PatchSet* ) const
{
  // Require the damage parameter
  const MaterialSubset* matlset = matl->thisMaterial();//T2D; what is this?
  task->requires(Task::NewDW, pLocalizedLabel_preReloc,matlset,Ghost::None);
}

void Arenisca::getDamageParameter(const Patch* patch,
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

void Arenisca::carryForward(const PatchSubset* patches,
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
void Arenisca::addParticleState(std::vector<const VarLabel*>& from,
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
void Arenisca::addInitialComputesAndRequires(Task* task,
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

void Arenisca::addComputesAndRequires(Task* task,
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
void Arenisca::addComputesAndRequires(Task* ,
                                   const MPMMaterial* ,
                                   const PatchSet* ,
                                   const bool ) const
{

}

//T2D: Throw exception that this is not supported
double Arenisca::computeRhoMicroCM(double pressure,
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
  cout << "NO VERSION OF computeRhoMicroCM EXISTS YET FOR Arenisca"<<endl;
//#endif

}

//T2D: Throw exception that this is not supported
void Arenisca::computePressEOSCM(double rho_cur,double& pressure,
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

  cout << "NO VERSION OF computePressEOSCM EXISTS YET FOR Arenisca"
       << endl;
}

//T2D: Throw exception that this is not supported
double Arenisca::getCompressibility()
{
  cout << "NO VERSION OF computePressEOSCM EXISTS YET FOR Arenisca"
       << endl;
  return 1.0;
}

// Initialize all labels of the particle variables associated with Arenisca.
void Arenisca::initializeLocalMPMLabels()
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

// Compute the strain at zero pore pressure from initial pore pressure (Pf0)
double Arenisca::computeev0()
{
  // The user-supplied initial pore pressure (Pf0) is the pore pressure at zero
  // volumetric strain.  An estimate of the strain (ev0) at which the fluid pressure
  // is zero is derived from M. Homel's engineering model of matrix compressibility:

  //define and initialize some variables
  double p3  = d_cm.p3_crush_curve, // max vol. plastic strain
         Kf  = d_cm.fluid_B0,       // fluid bulk modulus
         Pf0 = 3*d_cm.fluid_pressure_initial,            // initial pore pressure (-I1)
         ev0;                       // strain at zero pore pressure

   if(Pf0==0) // No initial pore pressure
    ev0 = 0;

   else       // Initial pore pressure
    ev0 = -p3 + log(1 - exp(Pf0/Kf) + exp(p3 + Pf0/Kf));

  return ev0;
}

// Compute derivative of yield function w.r.t. branch point
double Arenisca::computedfdKappa(double I1,
                                double X,
                                double Kappa,
                                double Zeta)
{
  // The branch point (Kappa) is the value of I1 that divides the linear
  // Drucker-Prager region and the cap region of the yield function.
  // Kappa is related to the hydrostative compressive strength (X), which
  // evolves with plastic strain and governs the dependence of strength
  // on porosity.

  //define and initialize some variables
  double FSLOPE = d_cm.FSLOPE,
         PEAKI1 = d_cm.PEAKI1,
         dfdKappa;

 // Linear Drucker-Prager Region (I1-Zeta) >= Kappa
  if(I1-Zeta >= Kappa)
   dfdKappa = 0;

 // Cap Region (I1-Zeta) < Kappa
  else
   dfdKappa = ( 2*Pow(FSLOPE,2)*(I1 - X)*(I1 - Kappa - Zeta)*Pow( -I1+PEAKI1+Zeta , 2 ) )
              /Pow( -Kappa+X-Zeta , 3 );

  return dfdKappa;
}

//Compute derivative of yield function w.r.t. trace of iso backstress
double Arenisca::computedfdZeta(double I1,
                                double X,
                                double Kappa,
                                double Zeta)
{
  // The trace of the isotropic back stress tensor (Zeta) is the value of I1
  // corresponding to the fluid pressure evolved under volumetric plastic
  // deformation.  This backstress results in a shift of the yield surface
  // along the hydrostat.  Recall that the yield function is expressed in terms
  // of unshifted stress, and therefore evolves with Zeta.

  //define and initialize some varialbes
  double FSLOPE = d_cm.FSLOPE,
         Kf     = d_cm.fluid_B0, // fluid bulk modulus
         PEAKI1 = d_cm.PEAKI1,
         dfdZeta;

  if(Kf == 0)  // No Fluid Effects ---------------------------------------
   dfdZeta = 0;

  else{        // Fluid Effects ------------------------------------------

  // Linear Drucker-Prager Region (I1-Zeta) >= Kappa
   if(I1-Zeta >= Kappa)
    dfdZeta = -2*Pow(FSLOPE,2)*(-I1 + PEAKI1 + Zeta);

  // Cap Region (I1-Zeta) < Kappa
   else
    dfdZeta = (2*Pow(FSLOPE,2)*(I1 - X)*(I1 - PEAKI1 - Zeta)*
               ( Pow(I1,2) - Pow(X,2) + 3*X*(Kappa+Zeta) +
                (-2*Kappa+PEAKI1-Zeta)*(Kappa+Zeta) - I1*(PEAKI1+X+Zeta) ) )
               / Pow(-Kappa+X-Zeta , 3);

  } //end fluid effects

  return dfdZeta;
}

// Compute bulk modulus for tensile or compressive vol. strain
double Arenisca::computeBulkModulus(double ev)
{
  //define and initialize some variables
  double p3  = d_cm.p3_crush_curve, // max vol. plastic strain
         B0  = d_cm.B0,             // low pressure bulk modulus
         B1  = d_cm.p4_fluid_effect,// additional high pressure bulk modulus
         Kf  = d_cm.fluid_B0,       // fluid bulk modulus
         //Pf0 = d_cm.fluid_pressure_initial,            // initial pore pressure
         ev0,                       // strain at zero pore pressure
         K;                         // computed bulk modulus


  if(Kf == 0){  // No Fluid Effects ---------------------------------------
   // Bulk modulus is computed as a the low pressure value when in tension
   // and as the mean of the low and high pressure values in compression

   if(ev < 0)  // Compression
    K = B0 + B1/2;
   else        // Tension
    K = B0;
  }

  else{        // Fluid Effects ------------------------------------------
   // The linear elastic approximation to the bulk modulus is derived
   // using M. Homel's engineering model of matrix compressibility.

   ev0 = computeev0();            // strain at zero pore pressure

   if(ev < ev0)  // Compression
    K = B0 + ((B0 + B1)*exp(p3)*Kf)
             /(B0*(exp(p3) - 1) + B1*(exp(p3) - 1) + Kf);
   else        // Tension
    K = B0;
  }

  return K;
}

// Compute state variable X (defines cap position)
double Arenisca::computeX(double evp)
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
         B1  = d_cm.p4_fluid_effect,             // additional high pressure bulk modulus
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
    X = 1.0e6 * p0;
  }
  else
  { // Plastic strain is within allowable domain======================
    if(Kf==0)
    { // No Fluid Effects ---------------------------------------------
      if(evp <= 0)
        X = (p0*p1 + log((evp+p3)/p3))/p1;
      else
        X = p0*Pow(1+evp , 1/(p0*p1*p3));
    }
    else
    { // Fluid Effects ------------------------------------------------
      // First we evaluate the elastic volumetric strain to yield from the
      // empirical crush curve (Xfit) and bulk modulus (Kfit) formula for
      // the drained material.  These functions could be modified to use
      // the full non-linear and elastic-plastic coupled input paramters
      // without introducing the additional complexity of elastic-plastic
      // coupling in the plasticity solution.
      if(evp <= 0)
      { // pore collapse
        // Hack: for now we use a constant bulk modulus until we revise Arenisca to call
        // step with a strain rate so the bulk modulus can be adjusted with deformation.
        //
        //Kfit = B0 + B1;                     // drained bulk modulus function
        Kfit = B0;                     // drained bulk modulus function
        Xfit = (p0*p1+log((evp+p3)/p3))/p1; // drained crush curve function
      }
      else
      { // pore expansion
        Kfit = B0;                                 // drained bulk modulus function
        Xfit = Pow(1 + evp , 1 / (p0*p1*p3))*p0; // drained crush curve function
      }

      // Now we use our linear engineering model for the bulk modulus of the
      // saturated material to compute the stress at our elastic strain to yield.
      ev0  = computeev0();                // strain at zero pore pressure

      // Hack: for now we use a constant bulk modulus until we revise Arenisca to call
      // step with a strain rate so the bulk modulus can be adjusted with deformation.
      //
      //Keng = computeBulkModulus(ev0-1);   // Saturated bulk modulus
      Keng = B0;

      eveX = one_third*Xfit/Kfit;         // Elastic vol. strain to compressive yield

      // There are three regions depending on whether the elastic loading to yield
      // occurs within the domain of fluid effects (ev < ev0)
      if(evp <= ev0)                            // Fluid Effects
        X = 3*Keng*eveX;
      else if(evp > ev0 && evp+eveX < ev0)      // Transition
        // Hack: for now we use a constant bulk modulus until we revise Arenisca to call
        // step with a strain rate so the bulk modulus can be adjusted with deformation.
        // Also, check this, it might be wrong
        //
        //X = 3*B0*(evp-ev0) + 3*Keng*(evp+eveX-ev0);
        X = 3*B0*eveX;
      else                                      // No Fluid Effects
        X = 3*B0*eveX;
    } //end fluid effects
  } // end good/bad plastic strain
  return X;
}

// Compute (dZeta/devp) Zeta and vol. plastic strain
double Arenisca::computedZetadevp(double Zeta, double evp)
{
  // Computes the partial derivative of the trace of the
  // isotropic backstress (Zeta) with respect to volumetric
  // plastic strain (evp).
  //
  // From M. Homel's engineering model for matrix compressibility:
  //
  //define and initialize some varialbes
  double p3  = d_cm.p3_crush_curve,
         B0  = d_cm.B0,             // low pressure bulk modulus
         B1  = d_cm.p4_fluid_effect,             // additional high pressure bulk modulus
         Kf  = d_cm.fluid_B0,       // fluid bulk modulus
         Pf0 = d_cm.fluid_pressure_initial,
         ev0,
         dZetadevp;

  ev0  = computeev0();                // strain at zero pore pressure

  if (evp <= ev0 && Kf != 0) // ev0, is material strain at zero fluid pressure
    // Fluid Effects
    dZetadevp = (3*(B0 + B1)*exp(evp + p3)*Kf)/
                (B0*(exp(evp + p3) - exp(Zeta/(3*(B0 + B1)))) +
                 B1*(exp(evp + p3) - exp(Zeta/(3*(B0 + B1)))) +
                (exp(evp + p3) + exp((3*Pf0 + Zeta)/(3*Kf)) -
                 exp(p3 + Pf0/Kf + Zeta/(3*Kf)))*Kf);
  else
    dZetadevp=0;

  return dZetadevp;
}

#ifdef JC_DEBUG_PARTICLE // Undefine
#undef JC_DEBUG_PARTICLE
#endif
