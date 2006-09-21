#include <Packages/Uintah/CCA/Components/ICE/EOS/TSTGas.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>

using namespace Uintah;

TSTGas::TSTGas(ProblemSpecP& ps)
{
   // Constructor
  ps->require("a",a);
  ps->require("b",b);
  ps->require("u",u);
  ps->require("w",w);
  ps->require("gas_constant", d_gas_constant);
}

TSTGas::~TSTGas()
{
}

void TSTGas::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP eos_ps = ps->appendChild("EOS");
  eos_ps->setAttribute("type","TST_gas");

  eos_ps->appendElement("a",a);
  eos_ps->appendElement("b",b);
  eos_ps->appendElement("u",u);
  eos_ps->appendElement("w",w);
  eos_ps->appendElement("gas_constant", d_gas_constant);
}

double TSTGas::getGasConstant() const
{
  return d_gas_constant;
}

//__________________________________
double TSTGas::computeRhoMicro(double press, double gamma,
                                 double cv, double Temp, double rho_guess)
{
  // Pointwise computation of microscopic density
  // P=P(rho,T) is not invertable to get rho=rho(P,T)
  // so here using Newton's method to find the rhoM
  // such that
  //press - (gamma-1)*Temp*cv*rhoM)/(1-b*rhoM)+
  //         a*rhoM/(1+u*b*rhoM)/(v+w*b*rho) = 0
  // First guess comes from inverting the last term of this equation

  double rhoM=rho_guess;

  double epsilon = 1.e-15;
  double delta = 1.;
  double f,df_drho,relfac=.9;
  int count = 0;

  while(fabs(delta/rhoM)>epsilon){
    f = (gamma-1.0)*Temp*cv*rhoM/(1.-b*rhoM)-a*rhoM/(1.0+u*b*rhoM)/(1.+w*b*rhoM)-press;

    df_drho = ((gamma-1.)*(1.-b*rhoM)+(gamma-1.)*rhoM*b)*cv*Temp/(1.-b*rhoM)/(1.-b*rhoM)
            -(a*(1.+u*b*rhoM)*(1.+w*b*rhoM)-a*rhoM*(u*b*(1.+w*b*rhoM)+(1.+u*b*rhoM)*w*b))/(1.+u*b*rhoM)/(1.+u*b*rhoM)/(1.+w*b*rhoM)/(1.+w*b*rhoM);

    delta = -relfac*(f/df_drho);
    rhoM+=delta;
    rhoM=fabs(rhoM);
    if(count>=100){
//      cout << setprecision(15);
      cout << "TSTGas::computeRhoMicro not converging." << endl;
      cout << "press = " << press << " temp = " << Temp << " cv = " << cv << endl;
      cout << "delta = " << delta << " rhoM = " << rhoM << " f = " << f << " df_drho = " << df_drho << endl;

      exit(1);
    }
    count++;
  }
  return rhoM;
}

//__________________________________
void TSTGas::computeTempCC(const Patch* patch,
                             const string& comp_domain,
                             const CCVariable<double>& press, 
                             const CCVariable<double>& gamma,
                             const CCVariable<double>& cv,
                             const CCVariable<double>& rhoM, 
                             CCVariable<double>& Temp,
                             Patch::FaceType face)
{
  if(comp_domain == "WholeDomain") {
    for (CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++){
      IntVector c = *iter;
      Temp[c]= (press[c]+a*rhoM[c]/(1.+u*b*rhoM[c])/(1.+w*b*rhoM[c]))*(1.-b*rhoM[c])/(gamma[c]-1.)/cv[c]/rhoM[c];
    }
  } 
  // Although this isn't currently being used
  // keep it around it could be useful
  if(comp_domain == "FaceCells") {     
    for (CellIterator iter = patch->getFaceCellIterator(face);
         !iter.done();iter++) {
      IntVector c = *iter;                    
      Temp[c]= (press[c]+a*rhoM[c]/(1.+u*b*rhoM[c])/(1.+w*b*rhoM[c]))*(1.-b*rhoM[c])/(gamma[c]-1.)/cv[c]/rhoM[c];
    }
  }
}

//__________________________________
void TSTGas::computePressEOS(double rhoM, double gamma,
                            double cv, double Temp,
                            double& press, double& dp_drho, double& dp_de)
{
  // Pointwise computation of thermodynamic quantities
  press   = (gamma-1.)*Temp*cv*rhoM/(1.-b*rhoM)-a*rhoM/(1.+u*b*rhoM)/(1.+w*b*rhoM);
  dp_drho = ((gamma-1.)*(1.-b*rhoM)+(gamma-1.)*rhoM*b)*cv*Temp/(1.-b*rhoM)/(1.-b*rhoM)
            -(a*(1.+u*b*rhoM)*(1.+w*b*rhoM)-a*rhoM*(u*b*(1.+w*b*rhoM)+(1.+u*b*rhoM)*w*b))/(1.+u*b*rhoM)/(1.+u*b*rhoM)/(1.+w*b*rhoM)/(1.+w*b*rhoM);
  dp_de   = (gamma-1.)*rhoM/(1.-b*rhoM);
}
//__________________________________
// Return (1/v)*(dv/dT)  (constant pressure thermal expansivity)
double TSTGas::getAlpha(double Temp, double sp_v, double P, double cv)
{
  // Cheating here a bit, computing v*(dT/dv) and returning the inverse of that
  double alpha;
  double dt_dv;
  double R=getGasConstant();    //gas constant
  dt_dv=(P+a/(sp_v+u*b)/(sp_v+w*b))/R+(-a*(2.*sp_v+w*b+u*b)/(sp_v+u*b)/(sp_v+u*b)/(sp_v+w*b)/(sp_v+w*b))*(sp_v-b)/R;
  alpha=1./sp_v/dt_dv;

  return alpha;
}

//______________________________________________________________________
// Update temperature boundary conditions due to hydrostatic pressure gradient
// call this after set Dirchlet and Neuman BC
void TSTGas::hydrostaticTempAdjustment(Patch::FaceType face, 
                                         const Patch* patch,
                                         const vector<IntVector>& bound,
                                         Vector& gravity,
                                         const CCVariable<double>& gamma,
                                         const CCVariable<double>& cv,
                                         const Vector& cell_dx,
                                         CCVariable<double>& Temp_CC)
{ 
  throw InternalError( "ERROR:ICE:EOS:TSTGas: hydrostaticTempAdj() \n"
                       " has not been implemented", __FILE__, __LINE__ );
}
