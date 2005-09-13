//----- StandardTable.cc --------------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/Mixing/StandardTable.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/InletStream.h>
#include <Packages/Uintah/CCA/Components/Models/test/TableFactory.h>
#include <Packages/Uintah/CCA/Components/Models/test/TableInterface.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <math.h>
#include <stdio.h>
#include <iostream>

using namespace std;
using namespace Uintah;
using namespace SCIRun;
//****************************************************************************
// Default constructor for StandardTable
//****************************************************************************
StandardTable::StandardTable():MixingModel()
{
	//srand ( time(NULL) );
}

//****************************************************************************
// Destructor
//****************************************************************************
StandardTable::~StandardTable()
{
}

//****************************************************************************
// Problem Setup for StandardTable
//****************************************************************************
void 
StandardTable::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP db = params->findBlock("StandardTable");
  db->require("adiabatic",d_adiabatic);
  db->require("rxnvars",d_numRxnVars);
  db->require("mixstatvars",d_numMixStatVars);
  if ((db->findBlock("h_fuel"))&&(db->findBlock("h_air"))) {
    db->require("h_fuel",d_H_fuel);
    db->require("h_air",d_H_air);
    d_adiab_enth_inputs = true;
  }
  else
    d_adiab_enth_inputs = false;
  d_numMixingVars = 1; // Always one scalar (so far)
  
  string tablename = "testtable";
  table = TableFactory::readTable(db, tablename);
  table->addIndependentVariable("F");
  if (!(d_adiabatic))
    table->addIndependentVariable("Hl");
  if (d_numMixStatVars > 0)
    table->addIndependentVariable("Fvar");

  Rho_index = -1;
  T_index = -1;
  Cp_index = -1;
  Enthalpy_index = -1;
  Hs_index = -1;
  co2_index = -1;
  h2o_index = -1;
  for (ProblemSpecP child = db->findBlock("tableValue"); child != 0;
       child = child->findNextBlock("tableValue")) {
    TableValue* tv = new TableValue;
    child->get(tv->name);
    tv->index = table->addDependentVariable(tv->name);
    if(tv->name == "density")
	    Rho_index = tv->index;
    else if(tv->name == "Temp")
	    T_index = tv->index;
    else if(tv->name == "heat_capac")
	    Cp_index = tv->index;
    else if(tv->name == "Entalpy")
	    Enthalpy_index = tv->index;
    else if(tv->name == "sensible_h")
	    Hs_index = tv->index;
    else if(tv->name == "CO2")
	    co2_index = tv->index;
    else if(tv->name == "H2O")
	    h2o_index = tv->index;
    tv->label = 0;
    tablevalues.push_back(tv);
  }
  table->setup();
}

      

//****************************************************************************
// Actually compute the properties here
//****************************************************************************
void 
StandardTable::computeProps(const InletStream& inStream,
			     Stream& outStream)
{
  double small=1.0e-10;
  // Extracting the independent variables from the in stream 
  double mixFrac = inStream.d_mixVars[0];
  double mixFracVars = 0.0;
  if (inStream.d_mixVarVariance.size() != 0)
    mixFracVars = inStream.d_mixVarVariance[0];
  if(mixFrac > 1.0)
	mixFrac=1.0;
  else if (mixFrac < small)
	mixFrac=0.0;
  if(mixFracVars < small)
	mixFracVars=0.0;
  double var_limit=mixFracVars/((mixFrac*(1.0-mixFrac))+small);
  if(var_limit > 0.9)
  	mixFracVars=(2.0/3.0)*mixFracVars;
  // normilizing varaince
  mixFracVars=min(mixFracVars,mixFrac*(1.0-mixFrac));
  if (mixFracVars <= small)
    mixFracVars=0.0;
  else
  mixFracVars/=((mixFrac*(1.0-mixFrac))+small);
  // Heat loss for adiabatic case
  double current_heat_loss=0.0;
  double zero_heat_loss=0.0;
  double zero_mixFracVars=0.0;
  //Absolute enthalpy
  double enthalpy=0.0;
  // Adiabatic enthalpy
  double adiab_enthalpy=0.0;
  double interp_adiab_enthalpy = 0.0;
  if (d_adiab_enth_inputs)
    interp_adiab_enthalpy = d_H_fuel*mixFrac+d_H_air*(1.0-mixFrac);
  // Sensible enthalpy
  double sensible_enthalpy=0.0;

  vector<double> ind_vars;
  ind_vars.push_back(mixFrac);

  vector<double> ind_vars_zero_heat_loss;
  vector<double> ind_vars_zero_heat_loss_zero_variance;

  if(!d_adiabatic){
        ind_vars_zero_heat_loss.push_back(mixFrac);
        ind_vars_zero_heat_loss.push_back(zero_heat_loss);
        if (d_numMixStatVars > 0)
          ind_vars_zero_heat_loss.push_back(mixFracVars);
	sensible_enthalpy=table->interpolate(Hs_index, ind_vars_zero_heat_loss);
	enthalpy=inStream.d_enthalpy;
	if (!(Enthalpy_index == -1)) {
          ind_vars_zero_heat_loss_zero_variance.push_back(mixFrac);
          ind_vars_zero_heat_loss_zero_variance.push_back(zero_heat_loss);
          if (d_numMixStatVars > 0)
            ind_vars_zero_heat_loss_zero_variance.push_back(zero_mixFracVars);
          adiab_enthalpy=table->interpolate(Enthalpy_index, ind_vars_zero_heat_loss_zero_variance);
	}
	else if (d_adiab_enth_inputs)
	  adiab_enthalpy=interp_adiab_enthalpy;
	else {
	  cout << "No way provided to compute adiabatic enthalpy" << endl;
	  exit (1);
	}

        if (inStream.d_initEnthalpy)
          	current_heat_loss = zero_heat_loss;
        else
  		current_heat_loss=(adiab_enthalpy-enthalpy)/(sensible_enthalpy+small);
        if(current_heat_loss < -1.0 || current_heat_loss > 1.0){
		cout<< "Heat loss is exceeding the bounds: "<<current_heat_loss << endl;
		cout<< "Absolute enthalpy is : "<< enthalpy << endl;
		cout<< "Adiabatic enthalpy is : "<< adiab_enthalpy << endl;
		cout<< "Sensible enthalpy is : "<< sensible_enthalpy << endl;
  		cout<< "Mixture fraction is :  "<< mixFrac << endl;
  		cout<< "Mixture fraction variance is :  "<< mixFracVars << endl;
	}
	if(fabs(current_heat_loss) < small)
		current_heat_loss=zero_heat_loss;
// Commented out heat loss table bounds clipping for now
/*        if(current_heat_loss > heatLoss[d_heatlosscount-1])
		current_heat_loss = heatLoss[d_heatlosscount-1];
	else if (current_heat_loss < heatLoss[0])
		current_heat_loss = heatLoss[0];*/

        ind_vars.push_back(current_heat_loss);
  }
  if (d_numMixStatVars > 0)
    ind_vars.push_back(mixFracVars);

  // Looking for the properties corresponding to the heat loss
  outStream.d_temperature=table->interpolate(T_index, ind_vars);  
  outStream.d_density=table->interpolate(Rho_index, ind_vars);  
  outStream.d_cp=table->interpolate(Cp_index, ind_vars);  
  if (!(Enthalpy_index == -1))
    outStream.d_enthalpy=table->interpolate(Enthalpy_index, ind_vars);  
  else if (d_adiab_enth_inputs)
    outStream.d_enthalpy=interp_adiab_enthalpy;
  else
    outStream.d_enthalpy=0.0;
  outStream.d_co2=table->interpolate(co2_index, ind_vars);  
  outStream.d_h2o=table->interpolate(h2o_index, ind_vars);  
}

