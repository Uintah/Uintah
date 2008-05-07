#include <Packages/Uintah/CCA/Components/ICE/EOS/TST.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/InternalError.h>
#include <iostream>
#include <iomanip>
using namespace Uintah;


TST::TST(ProblemSpecP& ps){
  // Constructor
  ps->require("a", a);
  ps->require("b", b);
  ps->require("u", u);
  ps->require("w", w);
  ps->require("Gamma", Gamma);

  /* initialize all the constants */
  bb   = b*b;
  bu   = b*u;
  bw   = b*w;
  bbuw = bu*bw;
  b1uw = b*(1-u-w);
  bbuwuw = bb*(u+w-u*w);
  bubw = bu+bw;
  ab = a*b;
}


TST::~TST(){
}


void TST::outputProblemSpec(ProblemSpecP& ps){
  ProblemSpecP eos_ps = ps->appendChild("EOS");
  eos_ps->setAttribute("type","TST");
  eos_ps->appendElement("a", a);
  eos_ps->appendElement("b", b);
  eos_ps->appendElement("u", u);
  eos_ps->appendElement("w", w);
  eos_ps->appendElement("Gamma", Gamma);
}



//__________________________________
double TST::computeRhoMicro(double press, double gamma,
                            double cv, double Temp,double rho_guess){
  /*
   *       (gamma-1)*cv*Temp           a
   *   P = ------------------  -  ------------
   *             v - b            (v+ub)(v+wb)
   * 
   *  This is solved by use of cubic equation solver
   *  c's are the coefficients of the cubic equation
   *
   */
  double k  = (gamma-1)*cv*Temp;
  double c1 = -press;
  double c2 = press*b1uw + k;
  double c3 = k*bubw - a + press*bbuwuw;
  double c4 = (press*b+k)*bbuw + ab;

  double p = fabs(c3/c1 - pow(c2/c1, 2.)/3.);
  double q = pow((c2/c1),3.)/13.5 - c2*c3/(c1*c1)/3. + c4/c1;
  double cos_val = -q / pow(p,1.5) * 2.5980762113533159402911695122588;
  //2.59807621135331594 = sqrt(27)/2.0; 

  if(cos_val >  1.0){ 
    cos_val = 1.0;
  }
  if(cos_val < -1.0){ 
    cos_val = -1.0;  
  } 

  double phi = acos(cos_val);

  double sp_v = 2.*sqrt(p/3.)*cos(phi/3.) - c2/c1/3.;

  return  1./sp_v;
}


//__________________________________
// Return (1/v)*(dv/dT)  (constant pressure thermal expansivity)
double TST::getAlpha(double Temp, double sp_v, double P, double cv){
  // Cheating here a bit, computing v*(dT/dv) and returning the inverse of that
  double n1 = 2*P*sp_v + P*bw + P*bu;
  double n2 = sp_v - b;
  double n3 = P*sp_v*sp_v + P*sp_v*(bu+bw) + P*bbuw + a;

  double d1 = (sp_v+bu) * (sp_v+bw) * (Gamma-1) * cv;
  double d2 = sp_v + bu;
  double d3 = sp_v + bw;

  double dT_dv = n1*n2/d1 - n3*n2/d1*(1./d2+1./d3) + n3/d1;
  return 1./(sp_v*dT_dv);
}


//__________________________________
void TST::computeTempCC(const Patch* patch,
                        const string& comp_domain,
                        const CCVariable<double>& press, 
                        const CCVariable<double>& gamma,
                        const CCVariable<double>& cv,
                        const CCVariable<double>& rhoM, 
                        CCVariable<double>& Temp,
                        Patch::FaceType face){
  double sp_v, n1, n2, d1;  
  double P; 
  if(comp_domain == "WholeDomain") {
    for (CellIterator iter = patch->getExtraCellIterator__New();!iter.done();iter++){
      IntVector c = *iter;
      sp_v = 1.0/rhoM[c];
      P = press[c];
      n1 = sp_v - b;
      n2 = P*sp_v*sp_v + P*sp_v*(bu+bw) + P*bbuw + a;
      d1 = (sp_v+bu) * (sp_v+bw) * (Gamma-1) * cv[c];
      Temp[c]= n1*n2/d1;
    }
  } 
  // Although this isn't currently being used
  // keep it around it could be useful
  if(comp_domain == "FaceCells") {     
    for (CellIterator iter=patch->getFaceCellIterator(face);!iter.done();iter++){
      IntVector c = *iter;
      sp_v = 1.0/rhoM[c];
      P = press[c];
      n1 = sp_v - b;
      n2 = P*sp_v*sp_v + P*sp_v*(bu+bw) + P*bbuw + a;
      d1 = (sp_v+bu) * (sp_v+bw) * (Gamma-1) * cv[c];
      Temp[c]= n1*n2/d1;
    }
  }
  return;
}


//__________________________________
//
void TST::computePressEOS(double rhoM, double gamma,
                          double cv, double Temp,
                          double& press, double& dp_drho, double& dp_de){
  // Pointwise computation of thermodynamic quantities
  double sp_v = 1/rhoM;
  double p1 = (Gamma-1)*cv*Temp/(sp_v-b);
  double p2 = a/((sp_v+bu)*(sp_v+bw));
  double d1 = sp_v - b;
  double d2 = sp_v + bu;
  double d3 = sp_v + bw;

  press   = p1-p2;
  dp_drho = sp_v*sp_v * (p1/d1 - p2*(1/d2+1/d3));
  dp_de   = (Gamma-1)/(sp_v-b);
  return;
}


//______________________________________________________________________
// Update temperature boundary conditions due to hydrostatic pressure gradient
// call this after set Dirchlet and Neuman BC
void TST::hydrostaticTempAdjustment(Patch::FaceType, 
                                    const Patch*,
                                    const vector<IntVector>*,
                                    Vector&,
                                    const CCVariable<double>&,
                                    const CCVariable<double>&,
                                    const Vector&,
                                    CCVariable<double>&){ 
  throw InternalError( "ERROR:ICE:EOS:TST: hydrostaticTempAdj() \n"
                       " has not been implemented", __FILE__, __LINE__ );
}

