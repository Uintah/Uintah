#include <Packages/Uintah/CCA/Components/ICE/EOS/StiffGas.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>

using namespace Uintah;

StiffGas::StiffGas(ProblemSpecP& ps)
{
   // Constructor
  ps->get("gas_constant",d_gas_constant);
}

StiffGas::~StiffGas()
{
}

double StiffGas::getGasConstant() const
{
  return d_gas_constant;
}

//__________________________________
//
double StiffGas::computeRhoMicro(double press, double gamma,
                             double cv, double Temp)
{
  // Pointwise computation of microscopic density
  return  press/((gamma - 1.0)*cv*Temp);
}

//__________________________________
//
void StiffGas::computeTempCC(const Patch* patch,
                                const CCVariable<double>& press, 
                                const double& gamma,
                                const double& cv,
                                const CCVariable<double>& rho_micro, 
                                CCVariable<double>& Temp)
{
  const IntVector gc(1,1,1);  // include ghostcells in the calc.

  for (CellIterator iter = patch->getCellIterator(gc);!iter.done();iter++) {                     
    Temp[*iter]= press[*iter]/ ( (gamma - 1.0) * cv * rho_micro[*iter] );
  }
}

//__________________________________
//
void StiffGas::computePressEOS(double rhoM, double gamma,
                            double cv, double Temp,
                            double& press, double& dp_drho, double& dp_de)
{
  // Pointwise computation of thermodynamic quantities
  press   = (gamma - 1.0)*rhoM*cv*Temp;
  dp_drho = (gamma - 1.0)*cv*Temp;
  dp_de   = (gamma - 1.0)*rhoM;
}
