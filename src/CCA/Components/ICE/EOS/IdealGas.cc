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

#include <CCA/Components/ICE/EOS/IdealGas.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/ProblemSpec/ProblemSpec.h>

using namespace Uintah;

IdealGas::IdealGas(ProblemSpecP& )
{
   // Constructor
}

IdealGas::~IdealGas()
{
}

void IdealGas::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP eos_ps = ps->appendChild("EOS");
  eos_ps->setAttribute("type","ideal_gas");
}


//__________________________________
double IdealGas::computeRhoMicro(double press, double gamma,
                                 double cv, double Temp, double)
{
  // Pointwise computation of microscopic density
  return  press/((gamma - 1.0)*cv*Temp);
}

//__________________________________
void IdealGas::computeTempCC(const Patch* patch,
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
      Temp[c]= press[c]/ ( (gamma[c] - 1.0) * cv[c] * rho_micro[c] );
    }
  } 
  // Although this isn't currently being used
  // keep it around it could be useful
  if(comp_domain == "FaceCells") { 
    Patch::FaceIteratorType MEC = Patch::ExtraMinusEdgeCells;    
    
    for (CellIterator iter = patch->getFaceIterator(face,MEC);
         !iter.done();iter++) {
      IntVector c = *iter;                    
      Temp[c]= press[c]/ ( (gamma[c] - 1.0) * cv[c] * rho_micro[c] );
    }
  }
}

//__________________________________
void IdealGas::computePressEOS(double rhoM, double gamma,
                            double cv, double Temp,
                            double& press, double& dp_drho, double& dp_de)
{
  // Pointwise computation of thermodynamic quantities
  press   = (gamma - 1.0)*rhoM*cv*Temp;
  dp_drho = (gamma - 1.0)*cv*Temp;
  dp_de   = (gamma - 1.0)*rhoM;
}
//__________________________________
// Return (1/v)*(dv/dT)  (constant pressure thermal expansivity)
double IdealGas::getAlpha(double Temp, double , double , double )
{
  return  1.0/Temp;
}

//______________________________________________________________________
// Update temperature boundary conditions due to hydrostatic pressure gradient
// call this after set Dirchlet and Neuman BC
void IdealGas::hydrostaticTempAdjustment(Patch::FaceType face, 
                                         const Patch* patch,
                                         Iterator& bound_ptr,
                                         Vector& gravity,
                                         const CCVariable<double>& gamma,
                                         const CCVariable<double>& cv,
                                         const Vector& cell_dx,
                                         CCVariable<double>& Temp_CC)
{ 
  IntVector axes = patch->getFaceAxes(face);
  int P_dir = axes[0];  // principal direction
  double plusMinusOne = patch->faceDirection(face)[P_dir];
  // On xPlus yPlus zPlus you add the increment 
  // on xminus yminus zminus you subtract the increment
  double dx_grav = gravity[P_dir] * cell_dx[P_dir];
  
   for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
     IntVector c = *bound_ptr;
     Temp_CC[c] += plusMinusOne * dx_grav/( (gamma[c] - 1.0) * cv[c] ); 
  }
}
