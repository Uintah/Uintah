
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
// NUM_DEP_VARS corresponds to pressure, density, temp, enthalpy,snesh, cp, molwt
  d_depStateSpaceVars = NUM_DEP_VARS + numSpecies; 
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
  d_sensibleEnthalpy += factor*strm.d_sensibleEnthalpy;
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
  if (count >= NUM_DEP_VARS) 
    return d_speciesConcn[count];
  else
    {
      switch (count) {
      case 0:
	return d_pressure;
      case 1:
	return d_density;
      case 2:
	if (lfavre)
	  return d_temperature/d_density;
	else
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
Stream::convertVecToStream(const vector<double>& vec_stateSpace,
			   const bool lfavre) {
  d_depStateSpaceVars = vec_stateSpace.size();
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
  d_speciesConcn = vector<double> (vec_stateSpace.begin()+NUM_DEP_VARS,
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
  out << "Moleculer Weight: "<< d_moleWeight << endl;
  out << "CP: "<< d_cp << endl;
  out << "Species concentration in moles: " << endl;
    for (int ii = 0; ii < d_speciesConcn.size(); ii++) {
      out.width(10);
      out << d_speciesConcn[ii] << " " ; 
      if (!(ii % 10)) out << endl; 
    }
    out << endl;
}
//
// $Log$
// Revision 1.3  2001/06/28 21:59:50  divyar
// merged Arches with new UCF
//
// Revision 1.2  2001/04/25 18:02:16  rawat
// added capability to compute overall mass balance
// modified flow inlet geometry to do circular inlets
// removed some print statements
//
// Revision 1.1  2001/01/31 16:35:30  rawat
// Implemented mixing and reaction models for fire.
//
// Revision 1.1  2001/01/15 23:38:21  rawat
// added some more classes for implementing mixing model
//
//
