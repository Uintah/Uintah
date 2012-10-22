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

#include <CCA/Components/ICE/EOS/Thomsen_Hartka_water.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/ProblemSpec/ProblemSpec.h>

using namespace Uintah;

Thomsen_Hartka_water::Thomsen_Hartka_water(ProblemSpecP& ps)
{
   // Constructor
  ps->require("a", d_a);
  ps->require("b", d_b);
  ps->require("co",d_co);
  ps->require("ko",d_ko);
  ps->require("To",d_To);
  ps->require("L", d_L);
  ps->require("vo",d_vo);
}

Thomsen_Hartka_water::~Thomsen_Hartka_water()
{
}

void Thomsen_Hartka_water::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP eos_ps = ps->appendChild("EOS");
  eos_ps->setAttribute("type","Thomsen_Hartka_water");
  eos_ps->appendElement("a", d_a);
  eos_ps->appendElement("b", d_b);
  eos_ps->appendElement("co",d_co);
  eos_ps->appendElement("ko",d_ko);
  eos_ps->appendElement("To",d_To);
  eos_ps->appendElement("L", d_L);
  eos_ps->appendElement("vo",d_vo);
}


//__________________________________
  // Pointwise computation of microscopic density
double Thomsen_Hartka_water::computeRhoMicro(double press, double gamma,
                                 double cv, double Temp, double)
{
  double x = d_a * press + Temp - d_To;
  return  1./(d_vo*(1. - d_ko * press + d_L * x*x ) );
}

//__________________________________
// See "ICE/EOS/Thomsen&Hartka_notebook.pdf"
void Thomsen_Hartka_water::computeTempCC(const Patch* patch,
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
      double P           = press[c];
      double rhoM        = rho_micro[c];
      double vo_rhoM     = d_vo * rhoM;
      double numerator   = (1. + vo_rhoM * ( d_ko * P - 1. ));
      
      Temp[c] = -d_a * P + d_To + sqrt(numerator);
    }
  } 
  // Although this isn't currently being used
  // keep it around it could be useful  
  if(comp_domain == "FaceCells") {  
    Patch::FaceIteratorType MEC = Patch::ExtraMinusEdgeCells;   
    
    for (CellIterator iter = patch->getFaceIterator(face,MEC);
         !iter.done();iter++) {
      IntVector c = *iter;
      double P           = press[c];
      double rhoM        = rho_micro[c];
      double vo_rhoM     = d_vo * rhoM;
      double numerator   = (1. + vo_rhoM * ( d_ko * P - 1. ));
      
      Temp[c] = -d_a * P + d_To + sqrt(numerator);
    }
  }
}

//__________________________________
// See "ICE/EOS/Thomsen&Hartka_notebook.pdf"
void Thomsen_Hartka_water::computePressEOS(double rhoM, double gamma,
                                           double cv, double Temp,
                                           double& press, double& dp_drho, double& dp_de)
{
 // Pointwise computation of thermodynamic quantities
 
 
 double vo_rhoM = d_vo * rhoM;  // common
 double a_L     = d_a * d_L;
 
 double term1 = 1./(2. * a_L * d_a * vo_rhoM);
 double term2 = ( d_ko + 2. * a_L * (d_To - Temp) ) * vo_rhoM;
 double term3 = d_ko * d_ko * vo_rhoM;
 double term4 = 4. * a_L * (d_a - ( d_a + (Temp - d_To) * d_ko ) * vo_rhoM );
 
 press  = term1 * (term2 - sqrt( vo_rhoM * (term3 + term4 ) ) );
 

 //__________________________________
 // dp_drho
 double a_press      = d_a * press;
 double a_press_temp = a_press + Temp;

 double x = a_press_temp - d_To;
 double y = 1. - d_ko * press + x*x * d_L;
 double numerator    = y * y * d_vo;
 double denominator  = ( d_ko - 2. * a_L * (a_press_temp - d_To) );
 
 dp_drho = numerator/denominator; 
 
 //__________________________________
 //  dp_de
 numerator   = 2. * d_L * (a_press_temp - d_To);
 
 double d1   = d_ko - 2. * a_L * (a_press_temp - d_To);
 double d2   = d_co + d_b * ( d_To - Temp) + 2. * press * d_L * d_vo * (-a_press + d_To - 2. * Temp);
 denominator = (d1 * d2 );
 
 dp_de = numerator/denominator;
}


//__________________________________
// Return (1/v)*(dv/dT)  (constant pressure thermal expansivity)
// See "ICE/EOS/Thomsen&Hartka_notebook.pdf"
double Thomsen_Hartka_water::getAlpha(double Temp, double , double P, double )
{
  double x = d_a * P + Temp - d_To;
  double beta = 2. * d_L * (d_a * P + Temp - d_To)/( 1. - d_ko * P + d_L * x*x);

  return  beta;
}

//______________________________________________________________________
// Update temperature boundary conditions due to hydrostatic pressure gradient
// call this after set Dirchlet and Neuman BC
void Thomsen_Hartka_water::hydrostaticTempAdjustment(Patch::FaceType face, 
                                                     const Patch* patch,
                                                     Iterator& bound_ptr,
                                                     Vector& gravity,
                                                     const CCVariable<double>& gamma,
                                                     const CCVariable<double>& cv,
                                                     const Vector& cell_dx,
                                                     CCVariable<double>& Temp_CC)
{ 
// needs to be filled in
}
