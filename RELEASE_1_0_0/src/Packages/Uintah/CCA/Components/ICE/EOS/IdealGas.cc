#include <Packages/Uintah/CCA/Components/ICE/EOS/IdealGas.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>

using namespace Uintah;

IdealGas::IdealGas(ProblemSpecP& ps)
{
   // Constructor
  ps->require("gas_constant",d_gas_constant);
  lb = scinew ICELabel();

}

IdealGas::~IdealGas()
{
  delete lb;
}


double IdealGas::getGasConstant() const
{
  return d_gas_constant;
}

//__________________________________
//
double IdealGas::computeRhoMicro(double& press, double& gamma,
				 double& cv, double& Temp)
{
  // Pointwise computation of microscopic density
  return  press/((gamma - 1.0)*cv*Temp);
}

//__________________________________
//
void IdealGas::computeTempCC(const Patch* patch,
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
void IdealGas::computePressEOS(double& rhoM, double& gamma,
			       double& cv, double& Temp,
			       double& press, double& dp_drho, double& dp_de)
{
  // Pointwise computation of thermodynamic quantities
  press   = (gamma - 1.0)*rhoM*cv*Temp;
  dp_drho = (gamma - 1.0)*cv*Temp;
  dp_de   = (gamma - 1.0)*rhoM;
}
