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

#include <CCA/Components/ICE/EOS/Gruneisen.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/InternalError.h>
#include <iostream>
#include <iomanip>

using namespace Uintah;

Gruneisen::Gruneisen(ProblemSpecP& ps)
{
   // Constructor
  ps->require("A",A);
  ps->require("B",B);
  ps->require("rho0",rho0);
  ps->require("T0",T0);
  ps->require("P0",P0);
}

Gruneisen::~Gruneisen()
{
}

void Gruneisen::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP eos_ps = ps->appendChild("EOS");
  eos_ps->setAttribute("type","Gruneisen");
}

//__________________________________
double Gruneisen::computeRhoMicro(double P, double,
                                 double , double T, double)
{
  // Pointwise computation of microscopic density
  double rhoM = rho0*((1./A)*((P-P0) - B*(T-T0)) + 1.);
  return rhoM;
}
//__________________________________
// Return (1/v)*(dv/dT)  (constant pressure thermal expansivity)
double Gruneisen::getAlpha(double T, double, double P, double)
{
  double alpha=B/((P-P0) - B*(T-T0)+A);
  return  alpha;
}

//__________________________________
void Gruneisen::computeTempCC(const Patch* patch,
                              const string& comp_domain,
                              const CCVariable<double>& P, 
                              const CCVariable<double>&,
                              const CCVariable<double>&,
                              const CCVariable<double>& rhoM, 
                              CCVariable<double>& Temp,
                              Patch::FaceType face)
{
  if(comp_domain == "WholeDomain") {
    for (CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++){
      IntVector c = *iter;
      Temp[c]= T0 + (1./B)*((P[c]-P0) - A*(rhoM[c]/rho0-1.));
    }
  } 
  // Although this isn't currently being used
  // keep it around it could be useful
  if(comp_domain == "FaceCells") {     
    Patch::FaceIteratorType MEC = Patch::ExtraMinusEdgeCells;
    for (CellIterator iter=patch->getFaceIterator(face,MEC);!iter.done();iter++){
      IntVector c = *iter;
      Temp[c]= T0 + (1./B)*((P[c]-P0) - A*(rhoM[c]/rho0-1.));
    }
  }
}

//__________________________________
void Gruneisen::computePressEOS(double rhoM, double,
                          double cv, double Temp,
                          double& press, double& dp_drho, double& dp_de)
{
  // Pointwise computation of thermodynamic quantities
  press   = P0 + A*(rhoM/rho0-1.) + B*(Temp-T0);
  dp_drho = A/rho0;
  dp_de   = B/cv;
}

//______________________________________________________________________
// Update temperature boundary conditions due to hydrostatic pressure gradient
// call this after set Dirchlet and Neuman BC
void Gruneisen::hydrostaticTempAdjustment(Patch::FaceType, 
                                          const Patch*,
                                          Iterator&,
                                          Vector&,
                                          const CCVariable<double>&,
                                          const CCVariable<double>&,
                                          const Vector&,
                                          CCVariable<double>&)
{ 
  throw InternalError( "ERROR:ICE:EOS:Gruneisen, hydrostaticTempAdj() \n"
                               " has not been implemented", __FILE__, __LINE__ );
}

