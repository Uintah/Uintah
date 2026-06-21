/*
 * The MIT License
 *
 * Copyright (c) 1997-2026 The University of Utah
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

#include <CCA/Components/MPM/Materials/ConstitutiveModel/CeramicFiberDamage.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>
#include <CCA/Components/MPM/Core/MPMLabel.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Math/Matrix3.h>
#include <Core/Math/MinMax.h>
#include <Core/Malloc/Allocator.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>

#include <cmath>
#include <algorithm>
#include <string>
#include <sstream>

using namespace Uintah;
using std::string;

// ---------------------------------------------------------------------------
// Private micromechanics helpers (file scope only)
// ---------------------------------------------------------------------------

// Eshelby tensor component for a porous matrix with inclusion aspect ratio a.
// a < 0 or a == 1: spherical pores (Tandon-Weng 1984 isotropic limit).
// sub: four-digit string label, e.g. "1111", "1122", "2323".
static double eshelbyS(const char* sub, double vm, double a)
{
  string s(sub);

  if (a < 0.0 || std::abs(a - 1.0) < 1e-12) {
    // Spherical pore
    if (s=="1111"||s=="2222"||s=="3333") return (7-5*vm)/(15*(1-vm));
    if (s=="1122"||s=="1133"||s=="2211"||s=="2233"||s=="3311"||s=="3322")
      return (5*vm-1)/(15*(1-vm));
    if (s=="1212"||s=="2323"||s=="3131") return (4-5*vm)/(15*(1-vm));
    return 0.0;
  }

  // General prolate (a>1) or oblate (a<1) spheroid
  double a2 = a*a;
  double g;
  if (a > 1.0) {
    g = (a/std::pow(a2-1.0, 1.5))*(a*std::sqrt(a2-1.0) - std::acosh(a));
  } else {
    g = (a/std::pow(1.0-a2, 1.5))*(std::acos(a) - a*std::sqrt(1.0-a2));
  }

  if (s=="1111")
    return (1.0/(2*(1-vm)))*(1-2*vm+(3*a2-1)/(a2-1)-(1-2*vm+3*a2/(a2-1))*g);
  if (s=="2222"||s=="3333")
    return (3.0/(8*(1-vm)))*(a2/(a2-1))
         + (1.0/(4*(1-vm)))*(1-2*vm-9.0/(4*(a2-1)))*g;
  if (s=="2233"||s=="3322")
    return (1.0/(4*(1-vm)))*(a2/(2*(a2-1))-(1-2*vm+3.0/(4*(a2-1)))*g);
  if (s=="2211"||s=="3311")
    return -(1.0/(2*(1-vm)))*(a2/(a2-1))
           + (1.0/(4*(1-vm)))*(3*a2/(a2-1)-(1-2*vm))*g;
  if (s=="1122"||s=="1133")
    return -(1.0/(2*(1-vm)))*(1-2*vm+1.0/(a2-1))
           + (1.0/(2*(1-vm)))*(1-2*vm+3.0/(2*(a2-1)))*g;
  if (s=="2323"||s=="3232")
    return (1.0/(4*(1-vm)))*(a2/(2*(a2-1))+(1-2*vm-3.0/(4*(a2-1)))*g);
  if (s=="1212"||s=="1313")
    return (1.0/(4*(1-vm)))*(1-2*vm-(a2+1)/(a2-1)
           -(0.5)*(1-2*vm-3*(a2+1)/(a2-1))*g);
  return 0.0;
}

// ---------------------------------------------------------------------------
// Porous-matrix effective elastic constants (Tandon-Weng 1984 void limit).
// Pores treated as inclusions with zero stiffness.
// out[7] = [Epm11, Epm33, Gpm12, Gpm23, Kpm, vpm12, vpm23]
// ---------------------------------------------------------------------------
static bool getPorousMatrixElasticity(double Em, double vm, double c,
                                      double a, double out[7])
{
  double Gm = Em / (2*(1+vm));
  double lm = Em*vm / ((1+vm)*(1-2*vm));
  double Km = Gm + lm;

  // D constants for void inclusion (inclusion stiffness = 0)
  double D1 = 1.0 + 2*(-Gm)/(-lm);         // = 1 + 2*Gm/lm
  double D2 = (lm + 2*Gm)/(-lm);           // = -(1 + 2*Gm/lm)
  double D3 = lm/(-lm);                    // = -1

  double B1 = c*D1 + D2 + (1-c)*(D1*eshelbyS("1111",vm,a) + 2*eshelbyS("2211",vm,a));
  double B2 = c + D3 + (1-c)*(D1*eshelbyS("1122",vm,a)
                               + eshelbyS("2222",vm,a)
                               + eshelbyS("2233",vm,a));
  double B3 = c + D3 + (1-c)*(eshelbyS("1111",vm,a)
                               + (1+D1)*eshelbyS("2211",vm,a));
  double B4 = c*D1 + D2 + (1-c)*(eshelbyS("1122",vm,a)
                                  + D1*eshelbyS("2222",vm,a)
                                  + eshelbyS("2233",vm,a));
  double B5 = c + D3 + (1-c)*(eshelbyS("1122",vm,a)
                               + eshelbyS("2222",vm,a)
                               + D1*eshelbyS("2233",vm,a));

  double A1 = D1*(B4+B5) - 2*B2;
  double A2 = (1+D1)*B2 - (B4+B5);
  double A3 = B1 - D1*B3;
  double A4 = (1+D1)*B1 - 2*B3;
  double A5 = (1-D1)/(B4-B5);
  double A  = 2*B2*B3 - B1*(B4+B5);

  if (std::abs(A) < 1e-20) return false;

  double E11 = Em / (1 + c*(-2*vm*A3 + (1-vm)*A4 + (1+vm)*A5*A)/(2*A));
  double E33 = Em / (1 + c*(A1 + 2*vm*A2)/A);

  // Effective shear moduli: void inclusion → G_inclusion = 0 → ratio = -1
  double G12 = Gm*(1 + c/((-1.0) + 2*(1-c)*eshelbyS("2323",vm,a)));
  double G23 = Gm*(1 + c/((-1.0) + 2*(1-c)*eshelbyS("1212",vm,a)));

  // Iterative solve for plane-strain bulk modulus K12
  double K12 = 13e10;
  const double tol  = 0.01;
  const int maxIter = 20000;
  for (int iter = 0; iter < maxIter; iter++) {
    double inside = (E33/E11) - (E33/4)*((1.0/G12) + (1.0/K12));
    if (inside < 0.0) inside = 0.0;
    double v32   = std::sqrt(inside);
    double denom = 1 - vm*(1+2*v32) + c*(2*(v32-vm)*A3 + (1-vm*(1+2*v32))*A4)/A;
    double K12n  = Km*((1+vm)*(1-2*vm)/denom);
    if (std::abs(K12n - K12) < tol) { K12 = K12n; break; }
    K12 = K12n;
  }

  double inside = (E33/E11) - (E33/4)*((1.0/G12) + (1.0/K12));
  if (inside < 0.0) inside = 0.0;
  double v32 = std::sqrt(inside);
  double v23 = v32*(E11/E33);
  double v12 = E11/(2*G12) - 1.0;

  out[0] = E11;   // Epm11
  out[1] = E33;   // Epm33
  out[2] = G12;   // Gpm12
  out[3] = G23;   // Gpm23
  out[4] = K12;   // Kpm  (plane-strain bulk)
  out[5] = v12;   // vpm12
  out[6] = v23;   // vpm23
  return true;
}

// ---------------------------------------------------------------------------
// Build 6x6 Voigt stiffness from 9 orthotropic engineering constants.
// Input ordering: [E11, E22, E33, G23, G13, G12, v12, v13, v23]
// Voigt ordering: [11, 22, 33, 23, 13, 12]
// ---------------------------------------------------------------------------
static void buildVoigtStiffness(const double m[9], double C[6][6])
{
  double E11 = m[0], E22 = m[1], E33 = m[2];
  double G23 = m[3], G13 = m[4], G12 = m[5];
  double v12 = m[6], v13 = m[7], v23 = m[8];

  // Compliance matrix upper-left 3x3 block
  double S[3][3] = {
    {  1.0/E11,   -v12/E11,  -v13/E11 },
    { -v12/E11,    1.0/E22,  -v23/E22 },
    { -v13/E11,  -v23/E22,    1.0/E33 }
  };

  // Invert S by Cramer's rule (S is symmetric; inv(S)[i][j] = cofactor(j,i)/det)
  double det = S[0][0]*(S[1][1]*S[2][2] - S[1][2]*S[2][1])
             - S[0][1]*(S[1][0]*S[2][2] - S[1][2]*S[2][0])
             + S[0][2]*(S[1][0]*S[2][1] - S[1][1]*S[2][0]);
  double inv = 1.0/det;

  for (int i = 0; i < 6; i++) for (int j = 0; j < 6; j++) C[i][j] = 0.0;

  // Upper 3x3: inverse of S
  C[0][0] = inv*(S[1][1]*S[2][2] - S[1][2]*S[2][1]);
  C[0][1] = inv*(S[0][2]*S[2][1] - S[0][1]*S[2][2]);
  C[0][2] = inv*(S[0][1]*S[1][2] - S[0][2]*S[1][1]);
  C[1][0] = C[0][1];
  C[1][1] = inv*(S[0][0]*S[2][2] - S[0][2]*S[2][0]);
  C[1][2] = inv*(S[0][2]*S[1][0] - S[0][0]*S[1][2]);
  C[2][0] = C[0][2];
  C[2][1] = C[1][2];
  C[2][2] = inv*(S[0][0]*S[1][1] - S[0][1]*S[1][0]);

  // Shear diagonal
  C[3][3] = G23;
  C[4][4] = G13;
  C[5][5] = G12;
}

// ---------------------------------------------------------------------------
// Compute the 9 undamaged cross-ply composite moduli and store in moduli[9].
// Cross-ply = average of 0° ply and 90°-about-z ply (Voigt rotation shortcut).
// ---------------------------------------------------------------------------
static bool buildCompositeModuli(double Em, double vm,
                                 double Eft, double Efl, double Gfl,
                                 double Kf, double vf,
                                 double fiberVF, double porosity, double a,
                                 double moduli[9])
{
  double mvf = 1.0 - fiberVF;

  double porous[7];
  if (!getPorousMatrixElasticity(Em, vm, porosity, a, porous)) return false;
  double Epm11 = porous[0], Epm33 = porous[1], Gpm12 = porous[2], Gpm23 = porous[3];
  double Kpm   = porous[4], vpm12 = porous[5], vpm23 = porous[6];

  // Single-ply fiber-matrix composite (Hashin-Rosen mixing rules)
  double Es11 = Efl*fiberVF + Epm11*mvf
              + (4*(vpm12-vf)*(vpm12-vf)*fiberVF*mvf)
                / (fiberVF/Kpm + mvf/Kf + 1.0/Gfl);
  double Es33 = Eft*Epm33 / (Epm33*fiberVF + Eft*mvf);
  double Es22 = Eft*Epm11 / (Epm11*fiberVF + Eft*mvf);

  double vs12 = vpm12*mvf + vf*fiberVF
              + ((vf-vpm12)*(1.0/Kpm - 1.0/Kf)*mvf*fiberVF)
                / (mvf/Kf + fiberVF/Kpm + 1.0/Gpm12);
  double vs13 = vpm23*mvf + vf*fiberVF
              + ((vf-vpm23)*(1.0/Kpm - 1.0/Kf)*mvf*fiberVF)
                / (mvf/Kf + fiberVF/Kpm + 1.0/Gpm23);

  double Gs12 = Gpm12*(Gpm12*mvf + Gfl*(1+fiberVF))
                     / (Gpm12*(1+fiberVF) + Gfl*mvf);
  double Gs13 = Gpm23*(Gpm23*mvf + Gfl*(1+fiberVF))
                     / (Gpm23*(1+fiberVF) + Gfl*mvf);
  double Kc   = Kpm + fiberVF/(1.0/(Kf-Kpm) + mvf/(Kpm+Gpm12));
  double denom23 = 4.0/Es33 - 1.0/Kc - 4*vs12*vs12/Es11;
  if (std::abs(denom23) < 1e-30) return false;
  double Gs23 = 1.0/denom23;
  double vs23 = Es33/(2*Gs23) - 1.0;

  // Build Voigt stiffness for single 0° ply
  // Ordering: [E11, E22, E33, G23, G13, G12, v12, v13, v23]
  double singlePly[9] = { Es11, Es22, Es33, Gs23, Gs13, Gs12, vs12, vs13, vs23 };
  double C[6][6];
  buildVoigtStiffness(singlePly, C);

  // Cross-ply average: rotate 90° about z → x↔y, Voigt 0↔1, 3↔4.
  // For an orthotropic material this is exact without leaving the Voigt basis.
  double Ca[6][6] = {};
  Ca[0][0] = Ca[1][1] = 0.5*(C[0][0] + C[1][1]);
  Ca[2][2] = C[2][2];
  Ca[0][1] = Ca[1][0] = C[0][1];
  Ca[0][2] = Ca[2][0] = Ca[1][2] = Ca[2][1] = 0.5*(C[0][2] + C[1][2]);
  Ca[3][3] = Ca[4][4] = 0.5*(C[3][3] + C[4][4]);
  Ca[5][5] = C[5][5];

  // Invert upper-left 3x3 of Ca to extract engineering constants
  double A[3][3] = {
    {Ca[0][0], Ca[0][1], Ca[0][2]},
    {Ca[1][0], Ca[1][1], Ca[1][2]},
    {Ca[2][0], Ca[2][1], Ca[2][2]}
  };
  double det = A[0][0]*(A[1][1]*A[2][2] - A[1][2]*A[2][1])
             - A[0][1]*(A[1][0]*A[2][2] - A[1][2]*A[2][0])
             + A[0][2]*(A[1][0]*A[2][1] - A[1][1]*A[2][0]);
  if (std::abs(det) < 1e-20) return false;
  double inv = 1.0/det;

  double S[3][3];
  S[0][0] = inv*(A[1][1]*A[2][2] - A[1][2]*A[2][1]);
  S[0][1] = inv*(A[0][2]*A[2][1] - A[0][1]*A[2][2]);
  S[0][2] = inv*(A[0][1]*A[1][2] - A[0][2]*A[1][1]);
  S[1][0] = S[0][1];
  S[1][1] = inv*(A[0][0]*A[2][2] - A[0][2]*A[2][0]);
  S[1][2] = inv*(A[0][2]*A[1][0] - A[0][0]*A[1][2]);
  S[2][0] = S[0][2];
  S[2][1] = S[1][2];
  S[2][2] = inv*(A[0][0]*A[1][1] - A[0][1]*A[1][0]);

  double Ec11 = 1.0/S[0][0];
  double Ec22 = 1.0/S[1][1];
  double Ec33 = 1.0/S[2][2];
  double Gc23 = Ca[3][3];
  double Gc13 = Ca[4][4];
  double Gc12 = Ca[5][5];
  double vc12 = -S[0][1]*Ec11;
  double vc13 = -S[0][2]*Ec11;
  double vc23 = -S[1][2]*Ec22;

  moduli[0] = Ec11;
  moduli[1] = Ec22;
  moduli[2] = Ec33;
  moduli[3] = Gc23;
  moduli[4] = Gc13;
  moduli[5] = Gc12;
  moduli[6] = vc12;
  moduli[7] = vc13;
  moduli[8] = vc23;
  return true;
}

// ---------------------------------------------------------------------------
// CeramicFiberDamage implementation
// ---------------------------------------------------------------------------

CeramicFiberDamage::CeramicFiberDamage(ProblemSpecP& ps, MPMFlags* Mflag)
  : ConstitutiveModel(Mflag)
{
  // Matrix
  ps->require("E_matrix",            d_initialData.Em);
  ps->require("nu_matrix",           d_initialData.vm);
  // Fiber
  ps->require("E_fiber_transverse",  d_initialData.Eft);
  ps->require("E_fiber_longitudinal",d_initialData.Efl);
  ps->require("G_fiber_longitudinal",d_initialData.Gfl);
  ps->require("K_fiber",             d_initialData.Kf);
  ps->require("nu_fiber",            d_initialData.vf);
  // Microstructure
  ps->require("fiber_volume_fraction",d_initialData.fiberVF);
  ps->require("porosity",            d_initialData.porosity);
  d_initialData.poreAspectRatio = -1.0;  // default: spherical
  ps->get("pore_aspect_ratio",       d_initialData.poreAspectRatio);
  // Damage constants
  ps->require("alpha_matrix",        d_initialData.alphaM);
  ps->require("sigma_o",             d_initialData.sigO);
  ps->require("rho_o",               d_initialData.rhoO);
  ps->require("rho_c",               d_initialData.rhoC);
  ps->require("rho_n",               d_initialData.rhoN);
  ps->require("nu_o",                d_initialData.nuO);
  ps->require("nu_n",                d_initialData.nuN);
  ps->require("v_f_eff",             d_initialData.vFEff);
  ps->require("E_fiber_damage",      d_initialData.EFiber);
  ps->require("E_matrix_damage",     d_initialData.EMatrix);
  ps->require("residual_stress",     d_initialData.residualStress);
  ps->require("T_reference",         d_initialData.TRef);
  ps->require("psi_c",               d_initialData.psiC);
  ps->require("psi_n",               d_initialData.psiN);
  ps->require("eps_o",               d_initialData.epsO);
  ps->require("eps_f",               d_initialData.epsF);

  // Precompute undamaged cross-ply moduli
  const CMData& d = d_initialData;
  bool ok = buildCompositeModuli(d.Em, d.vm, d.Eft, d.Efl, d.Gfl, d.Kf, d.vf,
                                 d.fiberVF, d.porosity, d.poreAspectRatio,
                                 d_initialData.moduli);
  if (!ok) {
    throw ProblemSetupException(
      "CeramicFiberDamage: micromechanics initialization failed.",
      __FILE__, __LINE__);
  }

  // Effective bulk modulus for EOS from undamaged stiffness
  double C[6][6];
  buildVoigtStiffness(d_initialData.moduli, C);
  // Voigt average: K = (C11+C22+C33 + 2*(C12+C13+C23))/9
  d_initialData.bulkMod = (C[0][0]+C[1][1]+C[2][2]
                          + 2*(C[0][1]+C[0][2]+C[1][2])) / 9.0;

  // Create particle labels for damage state
  pCrackDensityLabel = VarLabel::create("p.crackDensity",
      ParticleVariable<Vector>::getTypeDescription());
  pCrackDensityLabel_preReloc = VarLabel::create("p.crackDensity+",
      ParticleVariable<Vector>::getTypeDescription());

  pCrackOpeningLabel = VarLabel::create("p.crackOpening",
      ParticleVariable<Vector>::getTypeDescription());
  pCrackOpeningLabel_preReloc = VarLabel::create("p.crackOpening+",
      ParticleVariable<Vector>::getTypeDescription());

  pFiberFractureLabel = VarLabel::create("p.fiberFracture",
      ParticleVariable<Vector>::getTypeDescription());
  pFiberFractureLabel_preReloc = VarLabel::create("p.fiberFracture+",
      ParticleVariable<Vector>::getTypeDescription());
}

CeramicFiberDamage::~CeramicFiberDamage()
{
  VarLabel::destroy(pCrackDensityLabel);
  VarLabel::destroy(pCrackDensityLabel_preReloc);
  VarLabel::destroy(pCrackOpeningLabel);
  VarLabel::destroy(pCrackOpeningLabel_preReloc);
  VarLabel::destroy(pFiberFractureLabel);
  VarLabel::destroy(pFiberFractureLabel_preReloc);
}

void CeramicFiberDamage::outputProblemSpec(ProblemSpecP& ps, bool output_cm_tag)
{
  ProblemSpecP cm_ps = ps;
  if (output_cm_tag) {
    cm_ps = ps->appendChild("constitutive_model");
    cm_ps->setAttribute("type","ceramic_fiber_damage");
  }
  const CMData& d = d_initialData;
  cm_ps->appendElement("E_matrix",             d.Em);
  cm_ps->appendElement("nu_matrix",            d.vm);
  cm_ps->appendElement("E_fiber_transverse",   d.Eft);
  cm_ps->appendElement("E_fiber_longitudinal", d.Efl);
  cm_ps->appendElement("G_fiber_longitudinal", d.Gfl);
  cm_ps->appendElement("K_fiber",              d.Kf);
  cm_ps->appendElement("nu_fiber",             d.vf);
  cm_ps->appendElement("fiber_volume_fraction",d.fiberVF);
  cm_ps->appendElement("porosity",             d.porosity);
  cm_ps->appendElement("pore_aspect_ratio",    d.poreAspectRatio);
  cm_ps->appendElement("alpha_matrix",         d.alphaM);
  cm_ps->appendElement("sigma_o",              d.sigO);
  cm_ps->appendElement("rho_o",                d.rhoO);
  cm_ps->appendElement("rho_c",                d.rhoC);
  cm_ps->appendElement("rho_n",                d.rhoN);
  cm_ps->appendElement("nu_o",                 d.nuO);
  cm_ps->appendElement("nu_n",                 d.nuN);
  cm_ps->appendElement("v_f_eff",              d.vFEff);
  cm_ps->appendElement("E_fiber_damage",       d.EFiber);
  cm_ps->appendElement("E_matrix_damage",      d.EMatrix);
  cm_ps->appendElement("residual_stress",      d.residualStress);
  cm_ps->appendElement("T_reference",          d.TRef);
  cm_ps->appendElement("psi_c",                d.psiC);
  cm_ps->appendElement("psi_n",                d.psiN);
  cm_ps->appendElement("eps_o",                d.epsO);
  cm_ps->appendElement("eps_f",                d.epsF);
}

CeramicFiberDamage* CeramicFiberDamage::clone()
{
  return scinew CeramicFiberDamage(*this);
}

void CeramicFiberDamage::initializeCMData(const Patch* patch,
                                          const MPMMaterial* matl,
                                          DataWarehouse* new_dw)
{
  initSharedDataForExplicit(patch, matl, new_dw);

  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);

  ParticleVariable<Vector> pCrackDensity, pCrackOpening, pFiberFracture;
  new_dw->allocateAndPut(pCrackDensity, pCrackDensityLabel, pset);
  new_dw->allocateAndPut(pCrackOpening, pCrackOpeningLabel, pset);
  new_dw->allocateAndPut(pFiberFracture, pFiberFractureLabel, pset);

  for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++) {
    particleIndex idx = *iter;
    pCrackDensity[idx] = Vector(0.0, 0.0, 0.0);  // rho_o initial (overridden by d.rhoO in stress)
    pCrackOpening[idx] = Vector(0.0, 0.0, 0.0);  // nu_o initial
    pFiberFracture[idx] = Vector(1.0, 1.0, 0.0); // psi: 1 = fully intact
  }

  computeStableTimeStep(patch, matl, new_dw);
}

void CeramicFiberDamage::addParticleState(std::vector<const VarLabel*>& from,
                                          std::vector<const VarLabel*>& to)
{
  from.push_back(pCrackDensityLabel);
  to.push_back(pCrackDensityLabel_preReloc);
  from.push_back(pCrackOpeningLabel);
  to.push_back(pCrackOpeningLabel_preReloc);
  from.push_back(pFiberFractureLabel);
  to.push_back(pFiberFractureLabel_preReloc);
}

void CeramicFiberDamage::computeStableTimeStep(const Patch* patch,
                                               const MPMMaterial* matl,
                                               DataWarehouse* new_dw)
{
  Vector dx = patch->dCell();
  int dwi = matl->getDWIndex();
  ParticleSubset* pset = new_dw->getParticleSubset(dwi, patch);

  constParticleVariable<double> pmass, pvolume;
  constParticleVariable<Vector> pvelocity;
  new_dw->get(pmass,     lb->pMassLabel,     pset);
  new_dw->get(pvolume,   lb->pVolumeLabel,   pset);
  new_dw->get(pvelocity, lb->pVelocityLabel, pset);

  // Maximum undamaged P-wave modulus over three axes
  double C[6][6];
  buildVoigtStiffness(d_initialData.moduli, C);
  double Mmax = std::max({C[0][0], C[1][1], C[2][2]});

  double c_dil = 0.0;
  Vector WaveSpeed(1.e-12, 1.e-12, 1.e-12);
  for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++) {
    particleIndex idx = *iter;
    c_dil = std::sqrt(Mmax * pvolume[idx] / pmass[idx]);
    WaveSpeed = Vector(Max(c_dil+fabs(pvelocity[idx].x()), WaveSpeed.x()),
                       Max(c_dil+fabs(pvelocity[idx].y()), WaveSpeed.y()),
                       Max(c_dil+fabs(pvelocity[idx].z()), WaveSpeed.z()));
  }
  WaveSpeed = dx/WaveSpeed;
  new_dw->put(delt_vartype(WaveSpeed.minComponent()), lb->delTLabel, patch->getLevel());
}

void CeramicFiberDamage::computeStressTensor(const PatchSubset* patches,
                                             const MPMMaterial* matl,
                                             DataWarehouse* old_dw,
                                             DataWarehouse* new_dw)
{
  double rho_orig = matl->getInitialDensity();
  const CMData& d = d_initialData;

  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    double se = 0.0;
    double c_dil = 0.0;
    Vector WaveSpeed(1.e-12, 1.e-12, 1.e-12);
    Vector dx = patch->dCell();

    int dwi = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

    // Read particle variables
    constParticleVariable<Matrix3> pDefGrad_old;
    constParticleVariable<Matrix3> pDefGrad_new;
    constParticleVariable<Matrix3> velGrad;
    constParticleVariable<double>  pmass, pvolume_new, ptemperature;
    constParticleVariable<Vector>  pvelocity;
    constParticleVariable<Vector>  pCrackDensity, pCrackOpening, pFiberFracture;
    constParticleVariable<Matrix3> pStress_old;

    old_dw->get(pDefGrad_old,   lb->pDeformationMeasureLabel,          pset);
    old_dw->get(ptemperature,   lb->pTemperatureLabel,                 pset);
    old_dw->get(pvelocity,      lb->pVelocityLabel,                    pset);
    old_dw->get(pStress_old,    lb->pStressLabel,                      pset);
    old_dw->get(pCrackDensity,  pCrackDensityLabel,                    pset);
    old_dw->get(pCrackOpening,  pCrackOpeningLabel,                    pset);
    old_dw->get(pFiberFracture, pFiberFractureLabel,                   pset);

    new_dw->get(pDefGrad_new,   lb->pDeformationMeasureLabel_preReloc, pset);
    new_dw->get(pvolume_new,    lb->pVolumeLabel_preReloc,             pset);
    new_dw->get(velGrad,        lb->pVelGradLabel_preReloc,            pset);

    // Allocate output
    ParticleVariable<Matrix3> pStress_new;
    ParticleVariable<double>  pdTdt, p_q;
    ParticleVariable<Vector>  pCrackDensity_new, pCrackOpening_new, pFiberFracture_new;

    new_dw->allocateAndPut(pStress_new,        lb->pStressLabel_preReloc,       pset);
    new_dw->allocateAndPut(pdTdt,              lb->pdTdtLabel,                  pset);
    new_dw->allocateAndPut(p_q,                lb->p_qLabel_preReloc,           pset);
    new_dw->allocateAndPut(pCrackDensity_new,  pCrackDensityLabel_preReloc,     pset);
    new_dw->allocateAndPut(pCrackOpening_new,  pCrackOpeningLabel_preReloc,     pset);
    new_dw->allocateAndPut(pFiberFracture_new, pFiberFractureLabel_preReloc,    pset);

    // Undamaged P-wave modulus for wave speed
    double C_undamaged[6][6];
    buildVoigtStiffness(d.moduli, C_undamaged);
    double Mmax = std::max({C_undamaged[0][0], C_undamaged[1][1], C_undamaged[2][2]});

    for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++) {
      particleIndex idx = *iter;

      pdTdt[idx] = 0.0;

      const Matrix3& F = pDefGrad_new[idx];
      double J = F.Determinant();

      // Green-Lagrange strain: E_GL = 0.5*(F^T F - I)
      Matrix3 Identity; Identity.Identity();
      Matrix3 EGL = (F.Transpose()*F - Identity)*0.5;

      // Current damage state from previous step
      double rho[2] = { pCrackDensity[idx].x(),  pCrackDensity[idx].y()  };
      double nu[2]  = { pCrackOpening[idx].x(),   pCrackOpening[idx].y()  };
      double psi[2] = { pFiberFracture[idx].x(),  pFiberFracture[idx].y() };

      // Copy undamaged moduli; only moduli[0] and moduli[1] are updated by damage
      double m[9];
      for (int k = 0; k < 9; k++) m[k] = d.moduli[k];
      double Ec[2] = { m[0], m[1] };  // undamaged E_c (never changes)

      double T = ptemperature[idx];
      double eps_dir[2] = { EGL(0,0), EGL(1,1) };

      for (int i = 0; i < 2; i++) {
        double sigM = d.EMatrix*(eps_dir[i] - d.alphaM*(T - d.TRef)) + d.residualStress;

        // Crack-opening fraction (monotonically non-decreasing)
        double nu_new = 0.0;
        if (sigM >= 0.0) {
          nu_new = d.nuO + (1-d.nuO)*(1 - std::exp(-0.693*std::pow(sigM/d.sigO, d.nuN)));
        }
        nu[i] = std::max(nu_new, nu[i]);

        // Matrix crack density (monotonically non-decreasing)
        double rho_new = 0.0;
        if (nu[i] > 0.0) {
          rho_new = d.rhoO + (1-d.rhoO)*(1 - std::exp(-d.rhoC*std::pow(sigM/d.sigO, d.rhoN)));
        }
        rho[i] = std::max(rho_new, rho[i]);

        // Fiber fracture factor (monotonically non-increasing)
        if (eps_dir[i] >= d.epsO && std::abs(d.epsF - d.epsO) > 1e-15) {
          double xi = (eps_dir[i] - d.epsO) / (d.epsF - d.epsO);
          double psi_new = std::exp(d.psiC*std::pow(xi, d.psiN));
          psi[i] = std::min(psi_new, psi[i]);
        }

        // Damaged effective modulus in direction i
        double vfl = d.vFEff * psi[i];
        double denom = (1-rho[i])*vfl*d.EFiber + rho[i]*Ec[i];
        if (std::abs(denom) < 1e-30) denom = 1e-30;
        m[i] = (1 - nu[i])*Ec[i]
              + nu[i]*((vfl*d.EFiber*Ec[i]) / denom);
      }

      // Build damaged Voigt stiffness from updated moduli
      double C[6][6];
      buildVoigtStiffness(m, C);

      // Voigt strain vector (engineering shear strains for off-diagonal)
      double eps[6] = {
        EGL(0,0), EGL(1,1), EGL(2,2),
        2*EGL(1,2), 2*EGL(0,2), 2*EGL(0,1)
      };

      // 2nd Piola-Kirchhoff stress in Voigt: sv = C * eps
      double sv[6] = {0,0,0,0,0,0};
      for (int i = 0; i < 6; i++)
        for (int j = 0; j < 6; j++)
          sv[i] += C[i][j]*eps[j];

      // Assemble 2nd PK stress tensor S
      // Voigt ordering [11,22,33,23,13,12] → indices [0,1,2,3,4,5]
      Matrix3 Spk2(sv[0], sv[5], sv[4],
                   sv[5], sv[1], sv[3],
                   sv[4], sv[3], sv[2]);

      // Push forward to Cauchy stress: sigma = (1/J) F S F^T
      pStress_new[idx] = (F * Spk2 * F.Transpose()) * (1.0/J);

      // Store updated damage state
      pCrackDensity_new[idx]  = Vector(rho[0], rho[1], 0.0);
      pCrackOpening_new[idx]  = Vector(nu[0],  nu[1],  0.0);
      pFiberFracture_new[idx] = Vector(psi[0], psi[1], 0.0);

      // Strain energy
      Matrix3 AvgStress = (pStress_new[idx] + pStress_old[idx])*0.5;
      // Rate of deformation approximation from incremental GL strain
      // (informational only, not used for anything time-critical)
      se += (EGL(0,0)*AvgStress(0,0) + EGL(1,1)*AvgStress(1,1) + EGL(2,2)*AvgStress(2,2)
           + 2*(EGL(0,1)*AvgStress(0,1) + EGL(0,2)*AvgStress(0,2) + EGL(1,2)*AvgStress(1,2)))
           * pvolume_new[idx];

      // Wave speed for CFL
      double rho_cur = rho_orig/J;
      c_dil = std::sqrt(Mmax/rho_cur);
      WaveSpeed = Vector(Max(c_dil+fabs(pvelocity[idx].x()), WaveSpeed.x()),
                         Max(c_dil+fabs(pvelocity[idx].y()), WaveSpeed.y()),
                         Max(c_dil+fabs(pvelocity[idx].z()), WaveSpeed.z()));

      // Artificial viscosity
      if (flag->d_artificial_viscosity) {
        double DTrace = (velGrad[idx](0,0) + velGrad[idx](1,1) + velGrad[idx](2,2));
        double c_bulk = std::sqrt(d.bulkMod/rho_cur);
        double dx_ave = (dx.x()+dx.y()+dx.z())/3.0;
        p_q[idx] = artificialBulkViscosity(DTrace, c_bulk, rho_cur, dx_ave);
      } else {
        p_q[idx] = 0.0;
      }
    } // particle loop

    WaveSpeed = dx/WaveSpeed;
    new_dw->put(delt_vartype(WaveSpeed.minComponent()), lb->delTLabel, patch->getLevel());

    if (flag->d_reductionVars->accStrainEnergy ||
        flag->d_reductionVars->strainEnergy) {
      new_dw->put(sum_vartype(se), lb->StrainEnergyLabel);
    }
  } // patch loop
}

void CeramicFiberDamage::carryForward(const PatchSubset* patches,
                                      const MPMMaterial* matl,
                                      DataWarehouse* old_dw,
                                      DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int dwi = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

    carryForwardSharedData(pset, old_dw, new_dw, matl);

    constParticleVariable<Vector> pCrackDensity, pCrackOpening, pFiberFracture;
    old_dw->get(pCrackDensity,  pCrackDensityLabel,  pset);
    old_dw->get(pCrackOpening,  pCrackOpeningLabel,  pset);
    old_dw->get(pFiberFracture, pFiberFractureLabel, pset);

    ParticleVariable<Vector> pCrackDensity_new, pCrackOpening_new, pFiberFracture_new;
    new_dw->allocateAndPut(pCrackDensity_new,  pCrackDensityLabel_preReloc,  pset);
    new_dw->allocateAndPut(pCrackOpening_new,  pCrackOpeningLabel_preReloc,  pset);
    new_dw->allocateAndPut(pFiberFracture_new, pFiberFractureLabel_preReloc, pset);

    for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++) {
      particleIndex idx = *iter;
      pCrackDensity_new[idx]  = pCrackDensity[idx];
      pCrackOpening_new[idx]  = pCrackOpening[idx];
      pFiberFracture_new[idx] = pFiberFracture[idx];
    }

    new_dw->put(delt_vartype(1.e10), lb->delTLabel, patch->getLevel());
    if (flag->d_reductionVars->accStrainEnergy ||
        flag->d_reductionVars->strainEnergy) {
      new_dw->put(sum_vartype(0.0), lb->StrainEnergyLabel);
    }
  }
}

void CeramicFiberDamage::addComputesAndRequires(Task* task,
                                                const MPMMaterial* matl,
                                                const PatchSet* patches) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  addSharedCRForExplicit(task, matlset, patches);

  Ghost::GhostType gnone = Ghost::None;
  task->requiresVar(Task::OldDW, lb->pStressLabel,       matlset, gnone);
  task->requiresVar(Task::OldDW, lb->pTemperatureLabel,  matlset, gnone);
  task->requiresVar(Task::OldDW, pCrackDensityLabel,     matlset, gnone);
  task->requiresVar(Task::OldDW, pCrackOpeningLabel,     matlset, gnone);
  task->requiresVar(Task::OldDW, pFiberFractureLabel,    matlset, gnone);

  task->computesVar(pCrackDensityLabel_preReloc,  matlset);
  task->computesVar(pCrackOpeningLabel_preReloc,  matlset);
  task->computesVar(pFiberFractureLabel_preReloc, matlset);
}

void CeramicFiberDamage::addComputesAndRequires(Task* task,
                                                const MPMMaterial* matl,
                                                const PatchSet* patches,
                                                const bool /*recursion*/,
                                                const bool /*SchedParent*/) const
{
  // Stub for implicit; not supported.
  (void)task; (void)matl; (void)patches;
}

double CeramicFiberDamage::computeRhoMicroCM(double pressure,
                                             const double p_ref,
                                             const MPMMaterial* matl,
                                             double /*temperature*/,
                                             double /*rho_guess*/)
{
  double rho_orig = matl->getInitialDensity();
  double K = d_initialData.bulkMod;
  return rho_orig / (1.0 - (pressure - p_ref)/K);
}

void CeramicFiberDamage::computePressEOSCM(double rho_cur,
                                           double& pressure, double p_ref,
                                           double& dp_drho, double& ss_new,
                                           const MPMMaterial* matl,
                                           double /*temperature*/)
{
  double K = d_initialData.bulkMod;
  double rho_orig = matl->getInitialDensity();
  pressure = p_ref + K*(1.0 - rho_orig/rho_cur);
  dp_drho  = K*rho_orig/(rho_cur*rho_cur);
  ss_new   = K/rho_cur;
}

double CeramicFiberDamage::getCompressibility()
{
  return 1.0/d_initialData.bulkMod;
}
