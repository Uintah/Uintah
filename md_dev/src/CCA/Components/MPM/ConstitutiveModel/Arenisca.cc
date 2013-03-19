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
  ps->require("p0_crush_curve",d_cm.p0_crush_curve);
  ps->require("p1_crush_curve",d_cm.p1_crush_curve);
  ps->require("p3_crush_curve",d_cm.p3_crush_curve);
  ps->require("p4_fluid_effect",d_cm.p4_fluid_effect); // b1
  ps->require("fluid_B0",d_cm.fluid_B0);               // kf
  ps->require("fluid_pressure_initial",d_cm.fluid_pressure_initial);             // Pf0
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
  d_cm.p0_crush_curve = cm->d_cm.p0_crush_curve;
  d_cm.p1_crush_curve = cm->d_cm.p1_crush_curve;
  d_cm.p3_crush_curve = cm->d_cm.p3_crush_curve;
  d_cm.p4_fluid_effect = cm->d_cm.p4_fluid_effect; // b1
  d_cm.fluid_B0 = cm->d_cm.fluid_B0;
  d_cm.fluid_pressure_initial = cm->d_cm.fluid_pressure_initial; //pf0
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
  VarLabel::destroy(pKappaLabel);
  VarLabel::destroy(pKappaLabel_preReloc);
  VarLabel::destroy(pScratchMatrixLabel);
  VarLabel::destroy(pScratchMatrixLabel_preReloc);
  VarLabel::destroy(pZetaLabel);
  VarLabel::destroy(pZetaLabel_preReloc);
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
  cm_ps->appendElement("p0_crush_curve",d_cm.p0_crush_curve);
  cm_ps->appendElement("p1_crush_curve",d_cm.p1_crush_curve);
  cm_ps->appendElement("p3_crush_curve",d_cm.p3_crush_curve);
  cm_ps->appendElement("p4_fluid_effect",d_cm.p4_fluid_effect); // b1
  cm_ps->appendElement("fluid_B0",d_cm.fluid_B0); // kf
  cm_ps->appendElement("fluid_pressure_initial",d_cm.fluid_pressure_initial); //Pf0
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
                            pKappa,          // Not used
                            pZeta;           // Trace of isotropic Backstress
  ParticleVariable<Matrix3> pScratchMatrix,  // Developer tool
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
  new_dw->allocateAndPut(pKappa,          pKappaLabel,          pset);  //not used
  new_dw->allocateAndPut(pZeta,           pZetaLabel,           pset);
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
    pKappa[*iter] = computeKappa(pCapX[*iter]);
    pZeta[*iter] = -3.0 * d_cm.fluid_pressure_initial; //MH: Also need to initialize I1 to equal zeta
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
                                   pCapX,
                                   pKappa,
                                   pZeta;
    constParticleVariable<long64>  pParticleID;
    constParticleVariable<Vector>  pvelocity;
    constParticleVariable<Matrix3> pScratchMatrix,
                                   pep,
                                   pDefGrad,
                                   pStress_old,
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
    old_dw->get(pKappa,          pKappaLabel,                  pset); //initializeCMData()
    old_dw->get(pZeta,           pZetaLabel,                   pset); //initializeCMData()
    old_dw->get(pParticleID,     lb->pParticleIDLabel,         pset);
    old_dw->get(pvelocity,       lb->pVelocityLabel,           pset);
    old_dw->get(pScratchMatrix,  pScratchMatrixLabel,          pset); //initializeCMData()
    old_dw->get(pep,             pepLabel,                     pset); //initializeCMData()
    old_dw->get(pDefGrad,        lb->pDeformationMeasureLabel, pset);
    old_dw->get(pStress_old,     lb->pStressLabel,             pset); //initializeCMData()

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
                              pCapX_new,
                              pKappa_new,
                              pZeta_new;
    ParticleVariable<Matrix3> pScratchMatrix_new,
                              pep_new,
                              pStress_new;

    new_dw->allocateAndPut(p_q,                 lb->p_qLabel_preReloc,         pset);
    new_dw->allocateAndPut(pdTdt,               lb->pdTdtLabel_preReloc,       pset);
    new_dw->allocateAndPut(pScratchDouble1_new, pScratchDouble1Label_preReloc, pset);
    new_dw->allocateAndPut(pScratchDouble2_new, pScratchDouble2Label_preReloc, pset);
    new_dw->allocateAndPut(pPorePressure_new,   pPorePressureLabel_preReloc,   pset);
    new_dw->allocateAndPut(pevp_new,            pevpLabel_preReloc,            pset);
    new_dw->allocateAndPut(peve_new,            peveLabel_preReloc,            pset);
    new_dw->allocateAndPut(pCapX_new,           pCapXLabel_preReloc,           pset);
    new_dw->allocateAndPut(pKappa_new,          pKappaLabel_preReloc,          pset);
    new_dw->allocateAndPut(pZeta_new,           pZetaLabel_preReloc,           pset);
    new_dw->allocateAndPut(pScratchMatrix_new,  pScratchMatrixLabel_preReloc,  pset);
    new_dw->allocateAndPut(pep_new,             pepLabel_preReloc,             pset);
    new_dw->allocateAndPut(pStress_new,         lb->pStressLabel_preReloc,     pset);

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

        double bulk  = computeBulkModulus( peve[idx] + pevp[idx] ),
               shear = G0;

        bulk=B0;//Hack

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
        pep_new[idx]     = ep_new_step;
        pStress_new[idx] = stress_new_step;

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
            r_new0,        // transformed r for non-hardening return
            z_new0,        // transformed, shifted z for non hardening return
            r_new,         // transformed r for hardening return
            z_new,         // transformed, shifted z for hardening return
            eta_out = 1.0, // inner bound for plastic scaler
            eta_in  = 0.0, // outer bound for plastic scaler
            eta_mid,       // solution for plastic scaler
            eps_eta = 1.0, // convergence measure: eta_out-eta_in
            TOL = 1.0e-9;  // convergence tolerance on eps_eta

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
    gfcn   = TransformedFlowFunction(r_trial,z_trial,X_old,Beta);
    r_new0 = r_trial - gfcn * dgdr(r_trial,z_trial,X_old,Beta);
    z_new0 = z_trial - gfcn * dgdz(r_trial,z_trial,X_old,Beta);

    // Update unshifted untransformed stress
    sigma_new0 = one_third*(sqrt_three*z_new0+Zeta_old)*Identity;
    if ( r_trial != 0.0 )
      sigma_new0 = sigma_new0 + (r_new0/r_trial)*S_trial;

    // Stress increment for non-hardening return
    d_sigma0 = sigma_new0 - sigma_old;

    //
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
      // z-component of the return direction to the updated yield surface has
      // changed sign.  If either of these has occured the increment in plastic
      // strain was too large, so we scale back the multiplier eta.
      if( TransformedYieldFunction( r_trial,z_trial,X_new,Beta ) <= 0.0 ||
          Sign(dgdz(r_trial,z_trial,X_old,Beta)) != Sign(dgdz(r_trial,z_trial,X_new,Beta)) )
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
        gfcn  = TransformedFlowFunction(r_trial,z_trial,X_new,Beta);
        r_new = r_trial - gfcn * dgdr(r_trial,z_trial,X_new,Beta);
        z_new = z_trial - gfcn * dgdz(r_trial,z_trial,X_new,Beta);

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

// Compute the old yield function (Not Used)
/*
double Arenisca::YieldFunction(const double& I1, const double& J2, const double& X,
                               const double& Kappa, const double& Zeta)
{
  // See "fig:AreniscaYieldSurface" in the Arenisca manual.

  //define and initialize some varialbes
  double FSLOPE = d_cm.FSLOPE,
         PEAKI1 = d_cm.PEAKI1,
         f;

  if(I1 - Zeta >= Kappa)
    f = J2 - Pow(FSLOPE,2)*Pow(-I1 + PEAKI1 + Zeta,2);

  else // I1 - Zeta < Kappa
    f =  J2 - (1-Pow((I1-Zeta-Kappa)/(X-Kappa),2))
              *Pow(FSLOPE,2)*Pow(-I1 + PEAKI1 + Zeta,2);

  return f;
}
*/

// Compute the Yield Test Function
double Arenisca::TransformedYieldFunction(const double& R,
                                          const double& Z,
                                          const double& X,
                                          const double& Beta)
{
  // This function is defined in a transformed and shifted stress space:
  //
  //       R = (3K/2G)*sqrt(2*J2)           Z = (I1-Zeta)/sqrt(3)
  //
  // This function is used ONLY to compute whether a trial stress state is elastic or
  // plastic, and is not used to compute the plastic flow.  Thus only the sign of the
  // returned value for f is important.

  //define and initialize some varialbes
  double Beta2 = Beta*Beta,
         ZVertex = one_sqrt_three*d_cm.PEAKI1,
         ZCapX   = one_sqrt_three*X,
         //XXRKappa,
         ZKappa,
         ZApex,
         f;

  // Transformed R component of the branchpoint:
  //XXRKappa = (Beta*(1 + Beta2 - Beta*sqrt(1 + Beta2))*(-ZCapX + ZVertex))/(1 + Beta2);

  // Shifted Z component of the branchpoint:
  ZKappa = (ZCapX + Beta2*ZCapX + Beta*sqrt(1 + Beta2)*(-ZCapX + ZVertex))/(1 + Beta2);

  // Shifted Z component of the apex:
  ZApex = ((1 + Beta2)*(-Beta + sqrt(1 + Beta2))*ZCapX +
     Beta*(1 + Beta2 - Beta*sqrt(1 + Beta2))*ZVertex)/sqrt(1 + Beta2);

  //cout << " TransYieldFxn:Beta="<<Beta<<", ZCapX="<<ZCapX
  //     << ", ZVertex="<<ZVertex<<", RKappa="<<RKappa<<",ZKappa="<<ZKappa<<", ZApex="<<ZApex<< endl;

  // Region I - closest point return to vertex
  if( (Z <= ZCapX) || (Z >= ZVertex) )
    f = 1;
  else
  {
    if(Z >= ZKappa)
    { // Region II - Linear Drucker Prager
      if(R <= Beta*(ZVertex-Z))
        f = -1.0;
      else
        f = 1.0;
    }
    else
    { // Region III - Circular Cap
      if (R*R <= Pow(ZApex-ZCapX,2) - Pow(Z-ZApex,2))
        f = -1.0;
      else
        f = 1.0;
    }
  }
  return f;
}

// Compute the Flow Function.
double Arenisca::TransformedFlowFunction(const double& R,
                                         const double& Z,
                                         const double& X,
                                         const double& Beta)
{
  // This function is defined in a transformed and shifted stress space:
  //
  //       R = (3K/2G)*sqrt(2*J2)           Z = (I1-Zeta)/sqrt(3)

  //define and initialize some varialbes
  double Beta2   = Beta*Beta,
         ZVertex = d_cm.PEAKI1/sqrt(3),
         ZCapX   = X/sqrt(3),
         //XXRKappa,
         //XXZKappa,
         ZApex,
         g;

  // Transformed R component of the branchpoint:
  //XXRKappa = (Beta*(1 + Beta2 - Beta*sqrt(1 + Beta2))*(-ZCapX + ZVertex))/(1 + Beta2);

  // Shifted Z component of the branchpoint:
  //XXZKappa = (ZCapX + Beta2*ZCapX + Beta*sqrt(1 + Beta2)*(-ZCapX + ZVertex))/(1 + Beta2);

  // Shifted Z component of the apex:
  ZApex = ((1 + Beta2)*(-Beta + sqrt(1 + Beta2))*ZCapX +
          Beta*(1 + Beta2 - Beta*sqrt(1 + Beta2))*ZVertex)/sqrt(1 + Beta2);

  // Region I - closest point return to vertex
  if(R <= (Z - ZVertex) / Beta)
    g = sqrt(Pow(R,2) + Pow(Z - ZVertex,2));

  // Region II - closest point return to the linear Drucker-Prager surface
  else if(R <= (Z - ZApex)/Beta)
    g = (R + Beta*(Z - ZVertex))/sqrt(1 + Beta2);

  // Region III - closest point return to a circular cap
  else
    g = sqrt( R*R + Pow(Z - ZApex,2)) - ZApex + ZCapX;

  return g;
}

// Compute the R-component of the Gradient of the Transformed Flow Function.

double Arenisca::dgdr(const double& R,
                      const double& Z,
                      const double& X,
                      const double& Beta)
{
  // This function is defined in a transformed and shifted stress space:
  //
  //       R = (3K/2G)*sqrt(2*J2)           Z = (I1-Zeta)/sqrt(3)

  //define and initialize some varialbes
  double Beta2   = Beta*Beta,
         ZVertex = d_cm.PEAKI1/sqrt(3),
         ZCapX   = X/sqrt(3),
         //XXRKappa,
         //XXZKappa,
         ZApex,
         dgdr;

  // Transformed R component of the branchpoint:
  //XXRKappa = (Beta*(1 + Beta2 - Beta*sqrt(1 + Beta2))*(-ZCapX + ZVertex))/(1 + Beta2);

  // Shifted Z component of the branchpoint:
  //XXZKappa = (ZCapX + Beta2*ZCapX + Beta*sqrt(1 + Beta2)*(-ZCapX + ZVertex))/(1 + Beta2);

  // Shifted Z component of the apex:
  ZApex = ((1 + Beta2)*(-Beta + sqrt(1 + Beta2))*ZCapX +
     Beta*(1 + Beta2 - Beta*sqrt(1 + Beta2))*ZVertex)/sqrt(1 + Beta2);

  // Region I - closest point return to vertex
  if(R <= (Z - ZVertex)/Beta)
    dgdr = R/sqrt(Pow(R,2) + Pow(Z - ZVertex,2));

  // Region II - closest point return to the linear Drucker-Prager surface
  else if(R <= (Z - ZApex)/Beta)
    dgdr = 1/sqrt(1 + Beta2);

  // Region III - closest point return to a circular cap
  else
    dgdr = R/sqrt(Pow(R,2) + Pow(Z - ZApex,2));

  return dgdr;
}

// Compute the Z-component of the Gradient of the Transformed Flow Function.
double Arenisca::dgdz(const double& R,
                      const double& Z,
                      const double& X,
                      const double& Beta)
{
  // This function is defined in a transformed and shifted stress space:
  //
  //       r = (3K/2G)*sqrt(2*J2)           z = (I1-Zeta)/sqrt(3)

  //define and initialize some varialbes
  double Beta2   = Beta*Beta,
         ZVertex = d_cm.PEAKI1/sqrt(3),
         ZCapX   = X/sqrt(3),
         //XXRKappa,
         //XXZKappa,
         ZApex,
         dgdz;

  // Transformed R component of the branchpoint:
  //XXRKappa = (Beta*(1 + Beta2 - Beta*sqrt(1 + Beta2))*(-ZCapX + ZVertex))/(1 + Beta2);

  // Shifted Z component of the branchpoint:
  //XXZKappa = (ZCapX + Beta2*ZCapX + Beta*sqrt(1 + Beta2)*(-ZCapX + ZVertex))/(1 + Beta2);

  // Shifted Z component of the apex:
  ZApex = ((1 + Beta2)*(-Beta + sqrt(1 + Beta2))*ZCapX +
     Beta*(1 + Beta2 - Beta*sqrt(1 + Beta2))*ZVertex)/sqrt(1 + Beta2);

  // Region I - closest point return to vertex
  if(R <= (Z - ZVertex)/Beta)
    dgdz = (Z - ZVertex)/sqrt(Pow(R,2) + Pow(Z - ZVertex,2));

  // Region II - closest point return to the linear Drucker-Prager surface
  else if(R <= (Z - ZApex)/Beta)
    dgdz = Beta/sqrt(1 + Beta2);

  // Region III - closest point return to a circular cap
  else
    dgdz = (Z - ZApex)/sqrt(Pow(R,2) + Pow(Z - ZApex,2));

  return dgdz;
}

// Old Yield Function Gradient (Not Used)
Matrix3 Arenisca::YieldFunctionGradient(const Matrix3& S,
                             const double& I1,
                             const double& J2,
                             const Matrix3& S_trial,
                             const double& I1_trial,
                             const double& J2_trial,
                             const double& X,
                             const double& Kappa,
                             const double& Zeta)
{
  //define and initialize some varialbes
  double FSLOPE = d_cm.FSLOPE,
         PEAKI1 = d_cm.PEAKI1;
  Matrix3 Identity,
          G;
  Identity.Identity();

  if(I1-Zeta != PEAKI1 || J2 != 0){ //not at vertex
    if(I1 - Zeta > PEAKI1)
      G=-2*Pow(FSLOPE,2)*(I1 - PEAKI1 - Zeta);
    else if(I1 - Zeta >= Kappa && I1 - Zeta <= PEAKI1)
      G=2*Pow(FSLOPE,2)*(I1 - PEAKI1 - Zeta);
    else //if(I1 - Zeta < Kappa
      G=2*Pow(FSLOPE,2)*(1 - Pow(I1 - Kappa - Zeta,2)/Pow(-Kappa + X,2))*(I1 - PEAKI1 - Zeta) -
        (2*Pow(FSLOPE,2)*(I1 - Kappa - Zeta)*Pow(I1 - PEAKI1 - Zeta,2))/Pow(-Kappa + X,2);
  }
  else{ //at vertex
    if(I1_trial-Zeta != PEAKI1)
      G = YieldFunction(I1_trial, J2_trial, X, Kappa, Zeta)
          / (I1_trial - Zeta - PEAKI1) * Identity;
    else
      G = Identity;
  }
  //if(I1 - Zeta >= Kappa)
  //  G = S + 2*Pow(FSLOPE,2)*Identity*(-I1 + PEAKI1 + Zeta);
  //else // I1 - Zeta < Kappa
  //  G =  S + (2*Pow(FSLOPE,2)*Identity*(I1 - PEAKI1 - Zeta)*
  //          (2*Pow(I1,2) - Pow(X,2) + PEAKI1*Zeta + 2*X*Zeta +
  //           Pow(Zeta,2) + Kappa*(PEAKI1 + 2*X + Zeta) -
  //           I1*(3*Kappa + PEAKI1 + 4*Zeta)))/Pow(Kappa - X + Zeta,2);

  return G;
}

// Old Yield Function (Not Used)
Matrix3 Arenisca::YieldFunctionBisection(const Matrix3& sigma_in,
                               const Matrix3& sigma_out,
                               const double& X,
                               const double& Kappa,
                               const double& Zeta,
                               long64 ParticleID)
{
  int    counter_midpt=0,
         counter_midpt_max;
  double FSLOPE = d_cm.FSLOPE,
         PEAKI1 = d_cm.PEAKI1,
         I1_in,
         J2_in,
         I1_out,
         J2_out,
         I1_diff,
         J2_diff,
         r_in=0,
         r_mid,
         r_out,
         r_out_initial,
         r_max,
         n_I1,
         n_sqrtJ2,
         f_mid;
  Matrix3 S_in,
          S_out,
          S_diff,
          Identity,
          sigma_return;
  Identity.Identity();

  //compute invariants
  computeInvariants(sigma_in,S_in,I1_in,J2_in);
  computeInvariants(sigma_out,S_out,I1_out,J2_out);

  #ifdef JC_DEBUG_PARTICLE // Print shifted trial stress for current subcycle
  #ifdef CSM_DEBUG_BISECTION
  if(ParticleID==JC_DEBUG_PARTICLE)
    cout << ", I1_out=" << I1_out << ", J2_out=" << J2_out
         << ", X=" << X << ", Kappa=" << Kappa << ", Zeta=" << Zeta;
  #endif
  #endif

  if(I1_in == I1_out && J2_in == J2_out)
    sigma_return = sigma_out;
  else{
    I1_diff = I1_out - I1_in;
    J2_diff = J2_out - J2_in;
    S_diff = S_out - S_in;

    r_out = sqrt(I1_diff * I1_diff + J2_diff);
    n_I1 = I1_diff/r_out;
    n_sqrtJ2 = sqrt(J2_diff)/r_out;

    r_out_initial = r_out;
    r_max = sqrt(Pow(PEAKI1 - X, 2) + Pow(FSLOPE * (PEAKI1 - X), 2));

    if(r_out > r_max){
      r_out = r_max;
      cout << endl << "WARNING: r_out > r_max in bisection" << endl;
    }

    counter_midpt_max=Floor(5*log10(r_out)+1);

    #ifdef JC_DEBUG_PARTICLE
    #ifdef CSM_DEBUG_BISECTION
    if(ParticleID==JC_DEBUG_PARTICLE && 0)
      cout << endl << "    FAST RETURN";
    #endif
    #endif

    while(r_out-r_in > 1.0e-6 * r_out_initial &&
      counter_midpt < counter_midpt_max){

      counter_midpt++;

      r_mid = 0.5*(r_out+r_in);
      f_mid = YieldFunction(r_mid*n_I1 + I1_in,
                            Pow(r_mid*n_sqrtJ2,2) + J2_in,X,Kappa,Zeta);

      #ifdef JC_DEBUG_PARTICLE
      #ifdef CSM_DEBUG_BISECTION
      if(ParticleID==JC_DEBUG_PARTICLE)
        cout<< endl << "    nFR=" << counter_midpt << "/" << counter_midpt_max
        << ", r_in=" << r_in << ", r_out=" << r_out
        << ", r_diff=" << r_out-r_in << ", f_mid="<<f_mid;
        //<< ", I1_out=" << r_out*n_I1+I1_in
        //<< ", J2_out=" << Pow(r_out*n_sqrtJ2,2)+J2_in
        //<<", f_out="<<YieldFunction(r_out*n_I1+I1_in,Pow(r_out*n_sqrtJ2,2)+J2_in,X,Kappa,Zeta)
        //<<", f_in="<<YieldFunction(r_in*n_I1+I1_in,Pow(r_in*n_sqrtJ2,2)+J2_in,X,Kappa,Zeta);
      #endif
      #endif

      if(f_mid > 0)
        r_out = r_mid;
      else
        r_in = r_mid;
    }
    //f_out = YieldFunction(r_out*n_I1+Zeta+Kappa,Pow(r_out*n_sqrtJ2,2),X,Kappa,Zeta);
    sigma_return = sigma_in + (1/3.0) * r_out * n_I1 * Identity;
    if(J2_diff != 0)
      sigma_return += r_out * n_sqrtJ2 / sqrt(J2_diff) * S_diff;
  }
  return sigma_return;
}

// Old Yield Function Bisect (Not Used)
Matrix3 Arenisca::YieldFunctionFastRet(const Matrix3& S,
                                       const double& I1,
                                       const double& J2,
                                       const double& X,
                                       const double& Kappa,
                                       const double& Zeta,
                                       long64 ParticleID)
{
  //define and initialize some variables
  double FSLOPE = d_cm.FSLOPE,
         PEAKI1 = d_cm.PEAKI1;
  Matrix3 Identity,
          sigmaF;
  Identity.Identity();

  #ifdef JC_DEBUG_PARTICLE
  #ifdef CSM_DEBUG_FAST_RET
  if(ParticleID==JC_DEBUG_PARTICLE)
    cout << endl << "    FR: I1=" << I1  << ", J2=" << J2 << ", X=" << X
         << ", Kappa=" << Kappa << ", Zeta=" << Zeta;
  #endif
  #endif

  //HACK
  if(YieldFunction(I1, J2, X, Kappa, Zeta) < 0)  //elastic
    sigmaF = one_third * I1 * Identity + S;
  else{
    double apexI1=(3*Kappa + PEAKI1 - sqrt(9*Pow(Kappa,2) - 2*Kappa*PEAKI1 + Pow(PEAKI1,2)
                                     - 16*Kappa*X + 8*Pow(X,2)) + 4*Zeta)/4;
    #ifdef JC_DEBUG_PARTICLE
    #ifdef CSM_DEBUG_FAST_RET
    cout<<",apexI1="<<apexI1<<",X="<<X<<",Kappa="<<Kappa;
    #endif
    #endif
    //     -----------------------------------
    //     FASTRET Region III - Cap Region
    //     -----------------------------------
    //     This portion of the code could be replaced with a radial return to the point {kappa,0}
    //     with an initial return to the radius equal to r = ||{X-kappa,Ff(X)}||
    //
    //     Algorithm:
    //       1. calculate normalized unit vectors for radial line
    //       2. calculate rmax
    //       3. setup midpoint rules
    if(sqrt(J2) > 1/FSLOPE * (I1 - apexI1)) {

      #ifdef JC_DEBUG_PARTICLE // Print shifted trial stress for current subcycle
      #ifdef CSM_DEBUG_FAST_RET
      if(ParticleID==JC_DEBUG_PARTICLE)
        cout << ",FastRet Region III";
      #endif
      #endif

      sigmaF=YieldFunctionBisection(one_third * apexI1 * Identity,
                                    one_third * I1 * Identity
                                    + S, X, Kappa, Zeta, ParticleID);

#ifdef JC_DEBUG_PARTICLE // Print shifted trial stress for current subcycle
#ifdef CSM_DEBUG_FAST_RET
      //if(ParticleID==JC_DEBUG_PARTICLE)
      //  cout << ", sigma_out="<<sigma_out<<", I1=" << I1 << ", J2=" << J2;
#endif
#endif
      if (isnan(sigmaF.Trace())) {  //Check stress_iteration for nan
        cerr << "ParticleID = " << ParticleID << " sigmaF = " << sigmaF << endl;
        throw InvalidValue("**ERROR**: Nan in sigmaF value", __FILE__, __LINE__);
      }
    }

    //     -----------------------------------
    //     FASTRET Region II - Linear DP
    //     -----------------------------------
    //     Compute a radial return in the octahedral profile to the linear drucker-prager
    //     surface.  If (PEAKI1>I1Trial>Kappa), sigmaF = {I1Trial, (PeakI1-I1Trial)*FSLOPE}
    //     Note FSLOPE is the slope of rootJ2 vs. I1
    else if(sqrt(J2) > 1/FSLOPE * (I1 - Zeta - PEAKI1)){  //above normal drucker-p
      #ifdef JC_DEBUG_PARTICLE // Print shifted trial stress for current subcycle
      #ifdef CSM_DEBUG_FAST_RET
      if(ParticleID==JC_DEBUG_PARTICLE)
        cout << ",FastRet Region II";
      #endif
      #endif

      // Fast return algorithm in other cases (see "fig:AreniscaYieldSurface"
      // in the Arenisca manual). In this case, the radial fast returning is used.
      sigmaF=YieldFunctionBisection(one_third * Identity  * (I1 - FSLOPE * sqrt(J2)),
                                    one_third * I1 * Identity + S,
                                    X, Kappa, Zeta, ParticleID);

      //cout << ", I1="<<I1<<",Zeta="<<Zeta<<",J2="<<J2
      //<<",S="<<S;
      if (isnan(sigmaF.Trace())) {  //Check stress_iteration for nan
        cerr << "ParticleID = " << ParticleID << " sigmaF = " << sigmaF << endl;
        throw InvalidValue("**ERROR**: Nan in sigmaF value", __FILE__, __LINE__);
      }
    }

    //     -----------------------------------
    //     FASTRET Region I - Return to vertex
    //     -----------------------------------
    else{  //sqrt(J2) <= 1/FSLOPE * (I1_sigmaP - ZetaP - PEAKI1)
      #ifdef JC_DEBUG_PARTICLE // Print shifted trial stress for current subcycle
      #ifdef CSM_DEBUG_FAST_RET
      if(ParticleID==JC_DEBUG_PARTICLE)
        cout << ",FastRet Region I";
      #endif
      #endif

      // Fast return algorithm in the case of I1>PEAKI1 (see "fig:AreniscaYieldSurface"
      // in the Arenisca manual). In this case, the fast returned position is the vertex.
      sigmaF = one_third * Identity * (PEAKI1+Zeta);
    }
  }

  // Compute the invariants of the fast returned stress in the loop
  //computeInvariants(sigmaF, S_sigmaF, I1_sigmaF, J2_sigmaF);

  // Calculate value of yield function at fast return
  //f_new=YieldFunction(I1_sigmaF,J2_sigmaF,XP,KappaP,ZetaP);

  return sigmaF;
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
  from.push_back(pKappaLabel);
  from.push_back(pZetaLabel);
  from.push_back(pScratchMatrixLabel);
  //Xfrom.push_back(pVelGradLabel); //needed?
  to.push_back(  pLocalizedLabel_preReloc);
  to.push_back(  pAreniscaFlagLabel_preReloc);
  to.push_back(  pScratchDouble1Label_preReloc);
  to.push_back(  pScratchDouble2Label_preReloc);
  to.push_back(  pPorePressureLabel_preReloc);
  to.push_back(  pepLabel_preReloc);
  to.push_back(  pevpLabel_preReloc);
  to.push_back(  peveLabel_preReloc);
  to.push_back(  pCapXLabel_preReloc);
  to.push_back(  pKappaLabel_preReloc);
  to.push_back(  pZetaLabel_preReloc);
  to.push_back(  pScratchMatrixLabel_preReloc);
  //Xto.push_back(  pVelGradLabel_preReloc); //needed?
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
  task->computes(pKappaLabel,          matlset);
  task->computes(pZetaLabel,           matlset);
  task->computes(pScratchMatrixLabel,  matlset);
  //Xtask->computes(pVelGradLabel,        matlset);
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
  task->requires(Task::OldDW, pKappaLabel,          matlset, Ghost::None);
  task->requires(Task::OldDW, pZetaLabel,           matlset, Ghost::None);
  task->requires(Task::OldDW, pScratchMatrixLabel,  matlset, Ghost::None);
  //Xtask->requires(Task::OldDW, pVelGradLabel,        matlset, Ghost::None);
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
  task->computes(pKappaLabel_preReloc,          matlset);
  task->computes(pZetaLabel_preReloc,           matlset);
  task->computes(pScratchMatrixLabel_preReloc,  matlset);
  //Xtask->computes(pVelGradLabel_preReloc,        matlset);
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
  pScratchDouble1Label = VarLabel::create("p.ScratchDoubleOne",
    ParticleVariable<double>::getTypeDescription());
  pScratchDouble1Label_preReloc = VarLabel::create("p.ScratchDoubleOne+",
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
  pKappaLabel = VarLabel::create("p.kappa",
    ParticleVariable<double>::getTypeDescription());
  pKappaLabel_preReloc = VarLabel::create("p.kappa+",
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
  //pScratchMatrix
  pScratchMatrixLabel = VarLabel::create("p.ScratchMatrix",
    ParticleVariable<Matrix3>::getTypeDescription());
  pScratchMatrixLabel_preReloc = VarLabel::create("p.ScratchMatrix+",
    ParticleVariable<Matrix3>::getTypeDescription());
  //pVelGrad
  //XpVelGradLabel = VarLabel::create("p.velGrad",
  //X  ParticleVariable<Matrix3>::getTypeDescription());
  //XpVelGradLabel_preReloc = VarLabel::create("p.velGrad+",
  //X  ParticleVariable<Matrix3>::getTypeDescription());
}

//Compute Kinematics variables (Deformation Gradient, Velocity Gradient,
// Volume, Density) for the particles in a patch
// NO LONGER USED SINCE KINEMATICS MOVED TO SERIALMPM.cc
void Arenisca::computeKinematics(const PatchSubset* patches,
                                 const MPMMaterial* matl,
                                 DataWarehouse* old_dw,
                                 DataWarehouse* new_dw)
{
  // Define some constants
  Ghost::GhostType  gac   = Ghost::AroundCells;
  Matrix3 Identity,tensorL(0.0);//T2D: is Identity used?
  Identity.Identity();
  double J;

  // Get the initial density
  double rho_orig = matl->getInitialDensity();

  // Global loop over each patch
  for(int p=0;p<patches->size();p++){

    // Declare and initial value assignment for some variables
    const Patch* patch = patches->get(p);

    // Initialize for this patch
    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};

    // Declare the interpolator variables (CPDI, GIMP, linear, etc)
    ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());  ///< nodes affected by particle
    vector<Vector> d_S(interpolator->size());    ///< gradient of grid shape fxn
    vector<double> S(interpolator->size());      ///< grid shape fxn (T2D: needed?)

    // Get particle subset for the current patch
    int dwi = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

    // Get the particle variables
    delt_vartype                   delT;
    constParticleVariable<int>     pLocalized,
                                   pAreniscaFlag;
    constParticleVariable<double>  pmass;
    constParticleVariable<long64>  pParticleID;
    constParticleVariable<Point>   px;
    constParticleVariable<Vector>  pvelocity;
    constParticleVariable<Matrix3> psize,
                                   pVelGrad,
                                   pDefGrad;

    old_dw->get(delT, lb->delTLabel, getLevel(patches));
    old_dw->get(pLocalized,          pLocalizedLabel,              pset);
    old_dw->get(pAreniscaFlag,            pAreniscaFlagLabel,                pset);
    old_dw->get(pmass,               lb->pMassLabel,               pset);
    old_dw->get(pParticleID,         lb->pParticleIDLabel,         pset);
    old_dw->get(px,                  lb->pXLabel,                  pset);
    old_dw->get(pvelocity,           lb->pVelocityLabel,           pset);
    old_dw->get(psize,               lb->pSizeLabel,               pset);
    old_dw->get(pVelGrad,            lb->pVelGradLabel,                pset);
    old_dw->get(pDefGrad,            lb->pDeformationMeasureLabel, pset);

    // Allocate the particle variables
    ParticleVariable<int>          pLocalized_new,
                                   pAreniscaFlag_new;
    ParticleVariable<double>       pvolume;
    ParticleVariable<Matrix3>      pVelGrad_new;
    ParticleVariable<Matrix3>      pDefGrad_new;
    new_dw->allocateAndPut(pLocalized_new,        pLocalizedLabel_preReloc,              pset);
    new_dw->allocateAndPut(pAreniscaFlag_new,          pAreniscaFlagLabel_preReloc,                pset);
    new_dw->allocateAndPut(pvolume,               lb->pVolumeLabel_preReloc,             pset);
    new_dw->allocateAndPut(pVelGrad_new,          lb->pVelGradLabel_preReloc,                pset);
    new_dw->allocateAndPut(pDefGrad_new,          lb->pDeformationMeasureLabel_preReloc, pset);

    // Allocate some temporary particle variables
    ParticleVariable<double>  rho_cur;
    new_dw->allocateTemporary(rho_cur, pset);

    // Get the grid variables
    constNCVariable<Vector> gvelocity;  //NCV=Node Centered Variable
    new_dw->get(gvelocity, lb->gVelocityStarLabel,dwi,patch,gac,NGN);

    // loop over the particles
    for(ParticleSubset::iterator iter = pset->begin();
    iter != pset->end(); iter++){
      particleIndex idx = *iter;

      //re-zero the velocity gradient:
      pLocalized_new[idx]=pLocalized[idx];

      // pAreniscaFlag is a flag indicating if the particle has met any of the
      // limit values on evp (pPlasticStrainVol) based on CM parameter p3,
      // the maximum achieveable volumetric plastic strain in compression.
      pAreniscaFlag_new[idx]=0.0;

      tensorL.set(0.0);
      if(!flag->d_axisymmetric){
        // Get the node indices that surround the cell
        interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx],
            pDefGrad[idx]);
        computeVelocityGradient(tensorL,ni,d_S, oodx, gvelocity);
      } else {  // axi-symmetric kinematics
        // Get the node indices that surround the cell
        interpolator->findCellAndWeightsAndShapeDerivatives(px[idx],ni,S,d_S,
            psize[idx],pDefGrad[idx]);
        // x -> r, y -> z, z -> theta
        computeAxiSymVelocityGradient(tensorL,ni,d_S,S,oodx,gvelocity,px[idx]);
      }
      //pVelGrad=pVelGrad_new[idx];
      pVelGrad_new[idx]=tensorL;

      int num_scs = 4;
      Matrix3 one; one.Identity();
  #ifdef JC_USE_BB_DEFGRAD_UPDATE
      // Improve upon first order estimate of deformation gradient
      Matrix3 Amat = (pVelGrad[idx] + pVelGrad_new[idx])*(0.5*delT);
      //Matrix3 Amat = ( pVelGrad_new[idx])*(0.5*delT);//HACK BUG
      Matrix3 Finc = Amat.Exponential(JC_USE_BB_DEFGRAD_UPDATE);
      Matrix3 Fnew = Finc*pDefGrad[idx];
      pDefGrad_new[idx] = Fnew;
  #else
      // Update the deformation gradient in a new way using subcycling
      Matrix3 F=pDefGrad[idx];
      double Lnorm_dt = tensorL.Norm()*delT;
      num_scs = max(4,2*((int) Lnorm_dt));
      if(num_scs > 1000){
        cout << "NUM_SCS = " << num_scs << endl;
      }
      double dtsc = delT/(double (num_scs));
      Matrix3 OP_tensorL_DT = one + tensorL*dtsc;
      for(int n=0;n<num_scs;n++){
        F=OP_tensorL_DT*F;
      }
      pDefGrad_new[idx]=F;
      // Update the deformation gradient, Old First Order Way
      // pDefGrad_new[idx]=(tensorL*delT+Identity)*pDefGrad[idx];
  #endif

      // Compute the Jacobian and delete the particle in the case of negative Jacobian
      J = pDefGrad_new[idx].Determinant();
      if (J<=0 || J>10){
        cout<<"ERROR, negative J! "<<endl;
        cout<<"pParticleID="<<pParticleID[idx]<<endl;
        cout<<"pDefGrad= "<<pDefGrad_new[idx]<<endl;
        cout<<"pDefGrad_new="<<pDefGrad_new[idx]<<endl<<endl;
        cout<<"J= "<<J<<endl;
        cout<<"L= "<<tensorL<<endl;
        cout<<"num_scs= "<<num_scs<<endl;
        pLocalized_new[idx] = -999;
        cout<<"DELETING Arenisca particle " << endl;
        J=1;
        pDefGrad_new[idx] = one;
        //throw InvalidValue("**ERROR**:Negative Jacobian", __FILE__, __LINE__);
      }
  #ifdef JC_FREEZE_PARTICLE
      if (J>4000){
        cout << "WARNING, massive $J! " << "J=" << J
        //     <<", DELETING particleID=" <<pParticleID[idx]<< endl;
        //pLocalized_new[idx] = -999;
        <<",FREEZING particleID="<<pParticleID[idx]<<endl;
        J =pDefGrad[idx].Determinant();
        pDefGrad_new[idx] = pDefGrad[idx];
      }
  #endif

      // Update particle volume and density
      pvolume[idx]=(pmass[idx]/rho_orig)*J;
      //rho_cur[idx] = rho_orig/J;
    }

#ifdef CSM_PRESSURE_STABILIZATION
    // The following is used only for pressure stabilization
    CCVariable<double> J_CC;
    new_dw->allocateTemporary(J_CC,       patch);
    J_CC.initialize(0.);

    CCVariable<double> vol_0_CC;
    CCVariable<double> vol_CC;
    new_dw->allocateTemporary(vol_0_CC, patch);
    new_dw->allocateTemporary(vol_CC,   patch);

    vol_0_CC.initialize(0.);
    vol_CC.initialize(0.);
    for(ParticleSubset::iterator iter = pset->begin();
        iter != pset->end(); iter++){
      particleIndex idx = *iter;

      // get the volumetric part of the deformation
      double J = pDefGrad_new[idx].Determinant();

      // Get the deformed volume
      double rho_cur = rho_orig/J;
      pvolume[idx] = pmass[idx]/rho_cur;

      IntVector cell_index;
      patch->findCell(px[idx],cell_index);

      vol_CC[cell_index]  +=pvolume[idx];
      vol_0_CC[cell_index]+=pmass[idx]/rho_orig;
    }

    for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++){
      IntVector c = *iter;
      J_CC[c]=vol_CC[c]/vol_0_CC[c];
    }
    //end of pressureStabilization loop  at the patch level

    for(ParticleSubset::iterator iter = pset->begin();
        iter != pset->end(); iter++){
      particleIndex idx = *iter;

      double J = pDefGrad_new[idx].Determinant();

      IntVector cell_index;
      patch->findCell(px[idx],cell_index);

      // Change F such that the determinant is equal to the average for
      // the cell
      pDefGrad_new[idx]*=cbrt(J_CC[cell_index]/J);
      J=J_CC[cell_index];

      if (J<=0.0) {
        double Jold = pDefGrad[idx].Determinant();
        cout<<"negative J in ProgramBurn, J="<<J<<", Jold = " << Jold << endl;
        cout << "pos = " << px[idx] << endl;
        pLocalized_new[idx]=-999;
        cout<< "localizing (deleting) particle "<<pParticleID[idx]<<endl;
        cout<< "material = " << dwi << endl << "Momentum deleted = "
            << pvelocity[idx]*pmass[idx] <<endl;
        J=1;
      }
    }
#endif

  }
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
         Pf0 = d_cm.fluid_pressure_initial,            // initial pore pressure
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
        Kfit = B0 + B1;                     // drained bulk modulus function
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
      Keng = computeBulkModulus(ev0-1);   // Saturated bulk modulus
      eveX = one_third*Xfit/Kfit;         // Elastic vol. strain to compressive yield

      // There are three regions depending on whether the elastic loading to yield
      // occurs within the domain of fluid effects (ev < ev0)
      if(evp <= ev0)                            // Fluid Effects
        X = 3*Keng*eveX;
      else if(evp > ev0 && evp+eveX < ev0)      // Transition
        X = 3*B0*(evp-ev0) + 3*Keng*(evp+eveX-ev0);
      else                                      // No Fluid Effects
        X = 3*B0*eveX;
    } //end fluid effects
  } // end good/bad plastic strain
  return X;
}

// Compute state variable rate dXdevp (used in cap evolution)
double Arenisca::computedXdevp(double evp)
{
  //define and initialize some varialbes
  double p0  = d_cm.p0_crush_curve,
         p1  = d_cm.p1_crush_curve,
         p3  = d_cm.p3_crush_curve,
         B0  = d_cm.B0,             // low pressure bulk modulus
         B1  = d_cm.p4_fluid_effect,             // additional high pressure bulk modulus
         Kf  = d_cm.fluid_B0,       // fluid bulk modulus
         //XXPf0 = d_cm.fluid_pressure_initial,
         Kfit,
         Xfit,
         ev0,
         deveXdevp,
         Keng,
         eveX,
         pdXdevp;

  if(Kf==0){ // No Fluid Effects -------------------------------------
   if(evp <= 0)                        // pore collapse
    pdXdevp = 1/(p1*(evp + p3));
   else                                // pore expansion
    pdXdevp = Pow(1 + evp,-1 + 1/(p0*p1*p3))/(p1*p3);
  }

  else{      //Fluid Effects -----------------------------------------

  // First we evaluate the elastic volumetric strain to yield from the
  // empirical crush curve (Xfit) and bulk modulus (Kfit) formula for
  // the drained material.  These functions could be modified to use
  // the full non-linear and elastic-plastic coupled input paramters
  // without introducing the additional complexity of elastic-plastic
  // coupling in the plasticity solution.
  //

  if(evp <= 0){                      // pore collapse
    Kfit = B0 + B1;                     // drained bulk modulus function
    Xfit = (p0*p1+log((evp+p3)/p3))/p1; // drained crush curve function
    deveXdevp = 2/(3*(2*B0 + B1)*p1*(evp + p3));
  }
  else{                              // pore expansion
    Kfit = B0;                                 // drained bulk modulus function
    Xfit = Pow(1+evp , 1/(B0*p1*p3))*p0;       // drained crush curve function
    deveXdevp = Pow( 1+evp , -1 + 1/(p0*p1*p3) ) / (3*B0*p1*p3);
  }

   // Now we use our linear engineering model for the bulk modulus of the
   // saturated material to compute the stress at our elastic strain to yield.
   // Now we use our linear engineering model for the bulk modulus of the
   // saturated material to compute the stress at our elastic strain to yield.
   ev0  = computeev0();                // vol. strain at zero fluid pressure
   Keng = computeBulkModulus(ev0-1);   // Saturated bulk modulus
   eveX = one_third*Xfit/Kfit;         // Elastic vol. strain to compressive yield

   // There are three regions depending on whether the elastic loading to yield
   // occurs within the domain of fluid effects (ev < ev0)
   if(evp <= ev0)                            // Fluid Effects
    pdXdevp = 3*Keng*deveXdevp;              // X(evp) = 3*Keng*eveX

   else if(evp > ev0 && evp+eveX < ev0)      // Transition
    pdXdevp = 3*B0 + 3*Keng*(1 + deveXdevp); // X(evp) = 3*b0*(evp - ev0)
                                             //        + 3*Keng*(evp + eveX - ev0)
   else                                      // No Fluid Effects
    pdXdevp = 3*B0*deveXdevp;                // X(evp) = 3*b0*eveX
  } //end fluid effects

  return pdXdevp;
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

// Compute (dKappa/devp) from vol. plastic strain
double Arenisca::computedKappadevp(double evp)
{
  // The value of the partial derivative of the branch point
  // (Kappa) with respect to volumetric plastic strain (evp)
  // is computed using the chaing rule:
  //
  // dKappa    dKappa      dX
  // ------ = -------- * ------
  //  devp       dX       devp
  //
  //define and initialize some variables
  double CR     = d_cm.CR,
         FSLOPE = d_cm.FSLOPE,
         dKappadevp;

  dKappadevp= 1/(1+FSLOPE * CR) * computedXdevp(evp);

  return dKappadevp;
}

// Compute branch point, kappa exactly from cap intercept X
double Arenisca::computeKappa(double X)
{
  // The ratio of the branch point (Kappa) to the hydrostatic
  // compressive strength, (X) is defined by the input parameter
  // CR, defined such that:
  //
  //       FSLOPE*(PEAKI1-Kappa)
  // CR = -----------------------
  //           (Kappa - X)
  //
  //define and initialize some variables
  double FSLOPE = d_cm.FSLOPE,
         CR     = d_cm.CR,
         PEAKI1 = d_cm.PEAKI1,
         Kappa;

  Kappa = (FSLOPE * PEAKI1 * CR + X)
          /(1 + FSLOPE * CR);

  return Kappa;
}

// Compute unique projection direction P = C:M +Z
Matrix3 Arenisca::computeP(double lame,
                 Matrix3 M,
                 Matrix3 Z)
{
  //define and initialize some variables
  double  shear = d_cm.G0;
  Matrix3 Identity,
          P;
  Identity.Identity();

  //P=C:M+Z
  P = (Identity*lame*(M.Trace()) + M * 2.0*shear) + Z;

  return P;
}
#ifdef JC_DEBUG_PARTICLE // Undefine
#undef JC_DEBUG_PARTICLE
#endif
