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

#include <CCA/Components/ICE/EOS/KnaussSeaWater.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/ProblemSpec/ProblemSpec.h>

using namespace Uintah;

KnaussSeaWater::KnaussSeaWater(ProblemSpecP& ps)
{
  // Constructor
  ps->getWithDefault("a", d_a,         -0.15);
  ps->getWithDefault("b", d_b,          0.0);
  ps->getWithDefault("K", d_k,          4.5e-7);
  ps->getWithDefault("T0",d_T0,       283.15);
  ps->getWithDefault("P0",d_P0,    101325.0);
  ps->getWithDefault("S", d_S,         35.0);
  ps->getWithDefault("S0",d_S0,        35.0);
  ps->getWithDefault("rho0", d_rho0, 1027.0);
}

KnaussSeaWater::~KnaussSeaWater()
{
}

void KnaussSeaWater::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP eos_ps = ps->appendChild("EOS");
  eos_ps->setAttribute("type","KnaussSeaWater");
  eos_ps->appendElement("a", d_a);
  eos_ps->appendElement("b", d_b);
  eos_ps->appendElement("k", d_k);
  eos_ps->appendElement("T0",d_T0);
  eos_ps->appendElement("P0",d_T0);
  eos_ps->appendElement("S", d_S);
  eos_ps->appendElement("S0",d_S0);
  eos_ps->appendElement("rho0", d_rho0);
}

//__________________________________
double KnaussSeaWater::computeRhoMicro(double press, double gamma,
                                 double cv, double Temp, double)
{
  // Pointwise computation of microscopic density
  return d_rho0 + d_a*(Temp - d_T0) + d_b*(d_S - d_S0) + d_k*(press-d_P0);
}

//__________________________________
void KnaussSeaWater::computeTempCC(const Patch* patch,
                             const string& comp_domain,
                             const CCVariable<double>& press, 
                             const CCVariable<double>& gamma,
                             const CCVariable<double>& cv,
                             const CCVariable<double>& rho_micro, 
                             CCVariable<double>& Temp,
                             Patch::FaceType face)
{
  if(comp_domain == "WholeDomain") {
    for (CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++){
      IntVector c = *iter;
      Temp[c]= d_T0 + (1./d_a)*
                 ((rho_micro[c]-d_rho0) - d_b*(d_S-d_S0) - d_k*(press[c]-d_P0));
    }
  } 
  // Although this isn't currently being used
  // keep it around it could be useful
  if(comp_domain == "FaceCells") { 
    Patch::FaceIteratorType MEC = Patch::ExtraMinusEdgeCells;    
    
    for (CellIterator iter = patch->getFaceIterator(face,MEC);
         !iter.done();iter++) {
      IntVector c = *iter;                    
      Temp[c]= d_T0 + (1./d_a)*
                 ((rho_micro[c]-d_rho0) - d_b*(d_S-d_S0) - d_k*(press[c]-d_P0));
    }
  }
}

//__________________________________
void KnaussSeaWater::computePressEOS(double rhoM, double gamma,
                            double cv, double Temp,
                            double& press, double& dp_drho, double& dp_de)
{
  // Pointwise computation of thermodynamic quantities
  press   = d_P0 + (1./d_k)*((rhoM-d_rho0) - d_a*(Temp-d_T0) - d_b*(d_S-d_S0));
  dp_drho = 1./d_k;
  dp_de   = -d_a/d_k;
}
//__________________________________
// Return (1/v)*(dv/dT)  (constant pressure thermal expansivity)
double KnaussSeaWater::getAlpha(double Temp, double , double press, double )
{
  return  -d_a/(d_rho0 + d_a*(Temp-d_T0) + d_b*(d_S-d_S0) + d_k*(press-d_P0));
}

//______________________________________________________________________
// Update temperature boundary conditions due to hydrostatic pressure gradient
// call this after set Dirchlet and Neuman BC
void KnaussSeaWater::hydrostaticTempAdjustment(Patch::FaceType face, 
                                         const Patch* patch,
                                         Iterator& bound_ptr,
                                         Vector& gravity,
                                         const CCVariable<double>& gamma,
                                         const CCVariable<double>& cv,
                                         const Vector& cell_dx,
                                         CCVariable<double>& Temp_CC)
{ 
   // needs to be filled in
//  IntVector axes = patch->getFaceAxes(face);
//  int P_dir = axes[0];  // principal direction
//  double plusMinusOne = patch->faceDirection(face)[P_dir];
  // On xPlus yPlus zPlus you add the increment 
  // on xminus yminus zminus you subtract the increment
//  double dx_grav = gravity[P_dir] * cell_dx[P_dir];
  
//   for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
//     IntVector c = *bound_ptr;
//     Temp_CC[c] += plusMinusOne * dx_grav/( (gamma[c] - 1.0) * cv[c] ); 
//  }
}
