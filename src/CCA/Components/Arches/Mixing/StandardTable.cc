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

//----- StandardTable.cc --------------------------------------------------

#include <CCA/Components/Arches/Mixing/StandardTable.h>
#include <CCA/Components/Arches/Mixing/InletStream.h>
#include <CCA/Components/Models/FluidsBased/TableFactory.h>
#include <CCA/Components/Models/FluidsBased/TableInterface.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Math/MiscMath.h>
#include <cmath>
#include <cstdio>
#include <iostream>

using namespace std;
using namespace Uintah;
//****************************************************************************
// Default constructor for StandardTable
//****************************************************************************
StandardTable::StandardTable(bool calcReactingScalar,
                             bool calcEnthalpy,
                             bool calcVariance):
                             MixingModel(),
                             d_calcReactingScalar(calcReactingScalar),
                             d_calcEnthalpy(calcEnthalpy),
                             d_calcVariance(calcVariance)
{
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
  if (d_calcReactingScalar) 
    throw InvalidValue("Reacting scalar is unsupported parameter",
                         __FILE__, __LINE__);
  ProblemSpecP db = params->findBlock("StandardTable");
  if ((db->findBlock("h_fuel"))&&(db->findBlock("h_air"))) {
    db->require("h_fuel",d_H_fuel);
    db->require("h_air",d_H_air);
    d_adiab_enth_inputs = true;
  }
  else
    d_adiab_enth_inputs = false;
  
  string tablename = "testtable";
  table = TableFactory::readTable(db, tablename);
  table->addIndependentVariable("mixture_fraction");
  if (d_calcEnthalpy)
    table->addIndependentVariable("heat_loss");
  if (d_calcVariance)
    table->addIndependentVariable("mixture_fraction_variance");

  Rho_index = -1;
  T_index = -1;
  Cp_index = -1;
  Enthalpy_index = -1;
  Hs_index = -1;
  co2_index = -1;
  h2o_index = -1;
  for (ProblemSpecP child = db->findBlock("tableValue"); child != 0;
       child = child->findNextBlock("tableValue")) {
    TableValue* tv = scinew TableValue;
    child->get(tv->name);
    tv->index = table->addDependentVariable(tv->name);
    if(tv->name == "density"){
      Rho_index = tv->index;
    }else if(tv->name == "temperature"){
      T_index = tv->index;
    }else if(tv->name == "heat_capacity"){
      Cp_index = tv->index;
    }else if(tv->name == "entalpy"){
      Enthalpy_index = tv->index;
    }else if(tv->name == "sensible_heat"){
      Hs_index = tv->index;
    }else if(tv->name == "CO2"){
      co2_index = tv->index;
    }else if(tv->name == "H2O"){
      h2o_index = tv->index;
    }
    tv->label = 0;
    tablevalues.push_back(tv);
  }
  table->setup(false);
  if (!d_adiab_enth_inputs) {
    vector<double> ind_vars;
    ind_vars.push_back(0.0);
    if(d_calcEnthalpy) ind_vars.push_back(0.0);
    if (d_calcVariance) ind_vars.push_back(0.0);
    if (!(Enthalpy_index == -1))
      d_H_air=table->interpolate(Enthalpy_index, ind_vars);  
    else {
      throw ProblemSetupException("ERROR Arches::StandardTable::ProblemSetup \n"
                               "No way provided to compute adiabatic enthalpy", __FILE__, __LINE__);
    }
  }
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
  if (d_calcVariance){
    mixFracVars = inStream.d_mixVarVariance[0];
  }
  if(mixFrac > 1.0){
    mixFrac=1.0;
  }else if (mixFrac < small){
    mixFrac=0.0;
  }
  if(mixFracVars < small)
    mixFracVars=0.0;
    
  double var_limit=mixFracVars/((mixFrac*(1.0-mixFrac))+small);
  if(var_limit > 0.9){
    mixFracVars=(2.0/3.0)*mixFracVars;
  }
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
  if (d_adiab_enth_inputs){
    interp_adiab_enthalpy = d_H_fuel*mixFrac+d_H_air*(1.0-mixFrac);
  }
  // Sensible enthalpy
  double sensible_enthalpy=0.0;

  vector<double> ind_vars;
  ind_vars.push_back(mixFrac);

  vector<double> ind_vars_zero_heat_loss;
  vector<double> ind_vars_zero_heat_loss_zero_variance;

  if(d_calcEnthalpy){
    ind_vars_zero_heat_loss.push_back(mixFrac);
    ind_vars_zero_heat_loss.push_back(zero_heat_loss);
    if (d_calcVariance)
      ind_vars_zero_heat_loss.push_back(mixFracVars);
    sensible_enthalpy=table->interpolate(Hs_index, ind_vars_zero_heat_loss);
    enthalpy=inStream.d_enthalpy;
    if (!(Enthalpy_index == -1)) {
      ind_vars_zero_heat_loss_zero_variance.push_back(mixFrac);
      ind_vars_zero_heat_loss_zero_variance.push_back(zero_heat_loss);
      if (d_calcVariance)
        ind_vars_zero_heat_loss_zero_variance.push_back(zero_mixFracVars);
      adiab_enthalpy=table->interpolate(Enthalpy_index, ind_vars_zero_heat_loss_zero_variance);
    }
    else if (d_adiab_enth_inputs)
      adiab_enthalpy=interp_adiab_enthalpy;
    else {
      throw ProblemSetupException("ERROR Arches::StandardTable::computeProps \n"
                               "No way provided to compute adiabatic enthalpy", __FILE__, __LINE__);
    }

    if ((inStream.d_initEnthalpy)||
        ((Abs(adiab_enthalpy-enthalpy)/Abs(adiab_enthalpy) < 1.0e-4)&&
         (mixFrac < 1.0e-4))){
      current_heat_loss = zero_heat_loss;
    }else{
      current_heat_loss=(adiab_enthalpy-enthalpy)/(sensible_enthalpy+small);
    }
    
    if(current_heat_loss < -1.0 || current_heat_loss > 1.0){
      cout<< "Heat loss is exceeding the bounds: "<<current_heat_loss << endl;
      cout<< "Absolute enthalpy is :             "<< enthalpy << endl;
      cout<< "Adiabatic enthalpy is :            "<< adiab_enthalpy << endl;
      cout<< "Sensible enthalpy is :             "<< sensible_enthalpy << endl;
      cout<< "Mixture fraction is :              "<< mixFrac << endl;
      cout<< "Mixture fraction variance is :     "<< mixFracVars << endl;
    }
    if(fabs(current_heat_loss) < small){
      current_heat_loss=zero_heat_loss;
    }
// Commented out heat loss table bounds clipping for now
/*        if(current_heat_loss > heatLoss[d_heatlosscount-1])
                current_heat_loss = heatLoss[d_heatlosscount-1];
        else if (current_heat_loss < heatLoss[0])
                current_heat_loss = heatLoss[0];*/

     ind_vars.push_back(current_heat_loss);
  }
  
  if (d_calcVariance){
    ind_vars.push_back(mixFracVars);
  }
  
  // Looking for the properties corresponding to the heat loss
  outStream.d_temperature=table->interpolate(T_index, ind_vars);  
  outStream.d_density=table->interpolate(Rho_index, ind_vars);  
  outStream.d_cp=table->interpolate(Cp_index, ind_vars);  
  if (!(Enthalpy_index == -1)){
    outStream.d_enthalpy=table->interpolate(Enthalpy_index, ind_vars);  
  }else if (d_adiab_enth_inputs){
    outStream.d_enthalpy=interp_adiab_enthalpy;
  }else{
    outStream.d_enthalpy=0.0;
  }
  outStream.d_co2=table->interpolate(co2_index, ind_vars);  
  outStream.d_h2o=table->interpolate(h2o_index, ind_vars);  
}

