
#include <Packages/Uintah/CCA/Components/Arches/Mixing/Stream.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/ChemkinInterface.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <string>
#include <iostream>

using namespace Uintah;
using namespace std;
const int Stream::NUM_DEP_VARS;
Stream::Stream()
{
}

Stream::Stream(const int numSpecies)
{
  d_speciesConcn = vector<double>(numSpecies, 0.0); // initialize with 0
  d_pressure = 0.0;
  d_density = 0.0;
  d_temperature = 0.0;
  d_enthalpy = 0.0;
  d_sensibleEnthalpy = 0.0;
  d_moleWeight = 0.0;
  d_cp = 0.0;
  d_mole = false;
  d_numMixVars = 0;
  d_numRxnVars = 0;
  d_rxnVarRates = vector<double>(d_numRxnVars, 0.0); // initial with 0
  // NUM_DEP_VARS corresponds to pressure, density, temp, enthalpy, sensh, 
  // cp, molwt; total number of dependent state space variables also 
  // includes number of species and source terms (rxn rates) for each
  // rxn variable
  d_depStateSpaceVars = NUM_DEP_VARS + numSpecies + d_numRxnVars;
  // ***Add in enthalpy stuff***
}


Stream::Stream(const int numSpecies, const int numMixVars, const int numRxnVars): 
  d_numMixVars(numMixVars), d_numRxnVars(numRxnVars)
{
  d_speciesConcn = vector<double>(numSpecies, 0.0); // initialize with 0
  d_pressure = 0.0;
  d_density = 0.0;
  d_temperature = 0.0;
  d_enthalpy = 0.0;
  d_sensibleEnthalpy = 0.0;
  d_moleWeight = 0.0;
  d_cp = 0.0;
  d_mole = false;
  d_rxnVarRates = vector<double>(d_numRxnVars, 0.0); // initial with 0
  // NUM_DEP_VARS corresponds to pressure, density, temp, enthalpy, sensh, 
  // cp, molwt; total number of dependent state space variables also 
  // includes number of species and source terms (rxn rates) for each
  // rxn variable
  d_depStateSpaceVars = NUM_DEP_VARS + numSpecies + d_numRxnVars;
}


Stream::Stream(const Stream& strm) // copy constructor
{
  d_speciesConcn = strm.d_speciesConcn;
  d_pressure = strm.d_pressure;
  d_density = strm.d_density;
  d_temperature = strm.d_temperature;
  d_enthalpy = strm.d_enthalpy;
  d_sensibleEnthalpy = strm.d_sensibleEnthalpy;
  d_moleWeight = strm.d_moleWeight;
  d_cp = strm.d_cp;
  d_depStateSpaceVars = strm.d_depStateSpaceVars;
  d_mole = strm.d_mole;
  d_numMixVars = strm.d_numMixVars;
  d_numRxnVars = strm.d_numRxnVars;
  d_rxnVarRates = strm.d_rxnVarRates;
}

Stream&
Stream::operator=(const Stream &rhs)
{
  // Guard against self-assignment
  if (this != &rhs)
    {	
      d_speciesConcn = rhs.d_speciesConcn;
      d_pressure = rhs.d_pressure;
      d_density = rhs.d_density;
      d_temperature = rhs.d_temperature;
      d_enthalpy = rhs.d_enthalpy;
      d_sensibleEnthalpy = rhs.d_sensibleEnthalpy;
      d_moleWeight = rhs.d_moleWeight;
      d_cp = rhs.d_cp;
      d_depStateSpaceVars = rhs.d_depStateSpaceVars;
      d_mole = rhs.d_mole;
      d_numMixVars = rhs.d_numMixVars;
      d_numRxnVars = rhs.d_numRxnVars;
      d_rxnVarRates = rhs.d_rxnVarRates;
    }
  return *this;
  }

Stream::~Stream()
{
}

void
Stream::addSpecies(const ChemkinInterface* chemInterf,
		   const char* speciesName, double mfrac)
{
  int indx = speciesIndex(chemInterf, speciesName);
  d_speciesConcn[indx] = mfrac;
}

void 
Stream::addStream(const Stream& strm, ChemkinInterface* chemInterf,
		  const double factor) 
{
  vector<double> spec_mfrac;
  if (strm.d_mole) // convert to mass fraction
    spec_mfrac = chemInterf->convertMolestoMass(strm.d_speciesConcn);
  else
    spec_mfrac = strm.d_speciesConcn;
  for (int i = 0; i < spec_mfrac.size(); i++)
    d_speciesConcn[i] += factor*spec_mfrac[i];
  d_pressure += factor*strm.d_pressure;
  d_density += factor*strm.d_density;
  d_temperature += factor*strm.d_temperature;
  d_enthalpy += factor*strm.d_enthalpy;
  d_sensibleEnthalpy += factor*strm.d_sensibleEnthalpy; //Does this even make sense??
  d_moleWeight += factor*strm.d_moleWeight;
  d_cp += factor*strm.d_cp;
  d_mole = false;
}


int
Stream::speciesIndex(const ChemkinInterface* chemInterf, const char* speciesName) 
{
  for (int i = 0; i < d_speciesConcn.size(); i++) {
    if (strlen(speciesName) == strlen(chemInterf->d_speciesNames[i])) {
      if (strncmp(speciesName, chemInterf->d_speciesNames[i],
		  (size_t) strlen(speciesName)) == 0) 
	return i;
    }
  }
  throw InvalidValue("Species not found");
}

double
Stream::getValue(int count, bool lfavre) 
{
  int sumVars = NUM_DEP_VARS + d_speciesConcn.size();
  if ((count >= sumVars) && (count < sumVars + d_numRxnVars))
      return d_rxnVarRates[count-NUM_DEP_VARS-d_speciesConcn.size()];
  else
    if ((count >= NUM_DEP_VARS)&&(count < sumVars))
      return d_speciesConcn[count-NUM_DEP_VARS];
  else
    {
      switch (count) {
      case 0:
	return d_pressure;
      case 1:
	return d_density;
      case 2:
	if (lfavre){
	  cout<<"Stream::lfavre is true"<<endl;
	  return d_temperature/d_density;
	}
	else
	  //cout<<"Stream::temp = "<<d_temperature<<endl;
	  return d_temperature;
      case 3:
	return d_enthalpy;
      case 4:
	return d_sensibleEnthalpy;
      case 5:
	return d_moleWeight;
      case 6:
	return d_cp;
      default:
	cerr << "Invalid count value" << '/n';
	return 0;
      }
    }
}

void
Stream::normalizeStream() {
  double sum = 0.0;
  for (vector<double>::iterator iter = d_speciesConcn.begin();
       iter != d_speciesConcn.end(); ++iter) 
    sum += *iter;
  for (vector<double>::iterator iter = d_speciesConcn.begin();
       iter != d_speciesConcn.end(); ++iter) 
    *iter /= sum;
}  
	
void
Stream::convertVecToStream(const vector<double>& vec_stateSpace, const bool lfavre,
                           const int numMixVars, const int numRxnVars) {
  //cout<<"Stream::convertVecToStream"<<endl;
  d_depStateSpaceVars = vec_stateSpace.size();
  //cout<<"Stream::depStateSpace= "<<d_depStateSpaceVars<<endl;
  d_pressure = vec_stateSpace[0];
  d_density = vec_stateSpace[1];
  if (lfavre) 
    d_temperature = d_density*vec_stateSpace[2];
  else
    d_temperature = vec_stateSpace[2];
  d_enthalpy = vec_stateSpace[3];
  d_sensibleEnthalpy = vec_stateSpace[4];
  d_moleWeight = vec_stateSpace[5];
  d_cp = vec_stateSpace[6];
  // d_rxnVarRates = vector<double>(d_numRxnVars, 0.0);  
  //if (d_numRxnVars > 0) {
  // d_rxnVarRates =  vector<double> (vec_stateSpace.end()-d_numRxnVars, 
  //				  vec_stateSpace.end());
  //int sumVars = NUM_DEP_VARS + numSpecies;
  d_numMixVars = numMixVars;
  d_numRxnVars = numRxnVars;
  d_speciesConcn = vector<double> (vec_stateSpace.begin()+NUM_DEP_VARS,
				   vec_stateSpace.end()-d_numRxnVars);
  d_rxnVarRates = vector<double> (vec_stateSpace.end()-d_numRxnVars, 
				  vec_stateSpace.end());
}      

std::vector<double>
Stream::convertStreamToVec(const bool)
{
  vector<double> vec_stateSpace;
  vec_stateSpace.push_back(d_pressure);
  vec_stateSpace.push_back(d_density);
  vec_stateSpace.push_back(d_temperature);
  vec_stateSpace.push_back(d_enthalpy);
  vec_stateSpace.push_back(d_sensibleEnthalpy);
  vec_stateSpace.push_back(d_moleWeight);
  vec_stateSpace.push_back(d_cp);
  // copy d_speciesConcn to rest of the vector
  for (vector<double>::iterator iter = d_speciesConcn.begin(); 
       iter != d_speciesConcn.end(); ++iter)
   vec_stateSpace.push_back(*iter);
  for (vector<double>::iterator iter = d_rxnVarRates.begin(); 
       iter != d_rxnVarRates.end(); ++iter)
   vec_stateSpace.push_back(*iter);
  return vec_stateSpace;
}


Stream& Stream::linInterpolate(double upfactor, double lowfactor,
			       Stream& rightvalue) {
  d_pressure = upfactor*d_pressure+lowfactor*rightvalue.d_pressure;
  d_density = upfactor*d_density+lowfactor*rightvalue.d_density;
  d_temperature = upfactor*d_temperature+lowfactor*rightvalue.d_temperature;
  d_enthalpy = upfactor*d_enthalpy+lowfactor*rightvalue.d_enthalpy;
  d_sensibleEnthalpy = upfactor*d_sensibleEnthalpy+lowfactor*
                                           rightvalue.d_sensibleEnthalpy;
  d_moleWeight = upfactor*d_moleWeight+lowfactor*rightvalue.d_moleWeight;
  d_cp = upfactor*d_cp+lowfactor*rightvalue.d_cp;
  for (int i = 0; i < d_speciesConcn.size(); i++)
    d_speciesConcn[i] = upfactor*d_speciesConcn[i] +
                   lowfactor*rightvalue.d_speciesConcn[i];
  for (int i = 0; i < d_rxnVarRates.size(); i++)
    d_rxnVarRates[i] = upfactor*d_rxnVarRates[i] +
                   lowfactor*rightvalue.d_rxnVarRates[i];
  return *this;
}

void
Stream::print(std::ostream& out) const {
  out << "Integrated values"<< '\n';
  out << "Density: "<< d_density << endl;
  out << "Pressure: "<< d_pressure << endl;
  out << "Temperature: "<< d_temperature << endl;
  out << "Enthalpy: "<< d_enthalpy << endl;
  out << "Sensible Enthalpy: "<< d_sensibleEnthalpy << endl;
  out << "Molecular Weight: "<< d_moleWeight << endl;
  out << "CP: "<< d_cp << endl;
  out << "Species concentration in mass fraction: " << endl;
    for (int ii = 0; ii < d_speciesConcn.size(); ii++) {
      out.width(10);
      out << d_speciesConcn[ii] << " " ; 
      if (!(ii % 10)) out << endl; 
    }
    out << endl;
}

void
Stream::print(std::ostream& out, ChemkinInterface* chemInterf) {
  out << "Integrated values"<< '\n';
  out << "Density: "<< d_density << endl;
  out << "Pressure: "<< d_pressure << endl;
  out << "Temperature: "<< d_temperature << endl;
  out << "Enthalpy: "<< d_enthalpy << endl;
  out << "Sensible Enthalpy: "<< d_sensibleEnthalpy << endl;
  out << "Molecular Weight: "<< d_moleWeight << endl;
  out << "CP: "<< d_cp << endl;
  int numSpecies = chemInterf->getNumSpecies();
  double* specMW = new double[numSpecies];
  chemInterf->getMoleWeight(specMW);
  out << "Species concentration in mole fraction: " << endl;
    for (int ii = 0; ii < d_speciesConcn.size(); ii++) {
      out.width(10);
      out << d_speciesConcn[ii]/specMW[ii]*d_moleWeight << " " ; 
      if (!(ii % 10)) out << endl; 
    }
    out << endl;
}


//
// $Log$
// Revision 1.8  2001/10/11 18:48:59  divyar
// Made changes to Mixing
//
// Revision 1.6  2001/08/25 07:32:45  skumar
// Incorporated Jennifer's beta-PDF mixing model code with some
// corrections to the equilibrium code.
// Added computation of scalar variance for use in PDF model.
// Properties::computeInletProperties now uses speciesStateSpace
// instead of computeProps from d_mixingModel.
//
// Revision 1.5  2001/07/27 20:51:40  sparker
// Include file cleanup
// Fix uninitialized array element
//
// Revision 1.4  2001/07/16 21:15:38  rawat
// added enthalpy solver and Jennifer's changes in Mixing and Reaction model required for ILDM and non-adiabatic cases
//
// Revision 1.1  2001/01/31 16:35:30  rawat
// Implemented mixing and reaction models for fire.
//
// Revision 1.1  2001/01/15 23:38:21  rawat
// added some more classes for implementing mixing model
//
//
