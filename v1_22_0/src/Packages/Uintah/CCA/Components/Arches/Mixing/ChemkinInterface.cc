
#include <Packages/Uintah/CCA/Components/Arches/Mixing/ChemkinInterface.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/Common.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/ChemkinPrototypes.h>
#include <iostream>
using namespace std;
using namespace Uintah;
  
ChemkinInterface::ChemkinInterface() {
  int linc=15;
  int d_leniwk=1;
  int d_lenrwk=1;
  int d_lencwk=1;
  // Name of Chemkin binary link file
  char cklinkfile[NAMELENGTH];
  d_lout=6;
  d_numElements=1;
  d_numRxns=1;
  d_numSpecies=1;
  d_nfit=1;  

  strcpy (cklinkfile, "chem.bin");

  // Initialize Chemkin by reading binary file into appropriate work
  // arrays. All MRH Fortran routines are declared in ChemkinPrototypes.h.
  // The MRH interface to Chemkin is needed because of how files are
  // opened in Fortran with a unit number and because of the unknown 
  // character*16 data format.

  // Find required array lengths
  int cklinkfilenamelength=strlen(cklinkfile);
  mrhlen(&d_leniwk, &d_lenrwk, &d_lencwk, &linc, &d_lout, cklinkfile, 
	  &cklinkfilenamelength);
  d_ickwrk = new int[d_leniwk];
  d_rckwrk = new double[d_lenrwk];
  //cout << "CK: lenrwk "<<d_lenrwk<<endl;
  //cout << "CK: leniwk "<<d_leniwk<<endl;
  if(!(d_ickwrk && d_rckwrk))
    {
      cout << "Error allocating ickwrk or rckwrk"<<endl; 
    }
  d_cckwrk = CharArray(CHARLENGTH, d_lencwk); 

  // Read the binary file and create the internal work arrays
  mrhinit(&d_leniwk, &d_lenrwk, &d_lencwk, &linc, &d_lout, d_ickwrk, 
	  d_rckwrk, d_cckwrk[0], cklinkfile, &cklinkfilenamelength);
  ckindx(d_ickwrk, d_rckwrk, &d_numElements, &d_numSpecies, &d_numRxns, 
	 &d_nfit);

  //Get array of element names
  d_elementNames = CharArray(CHARLENGTH, d_numElements);
  int ichar = CHARLENGTH;
  mrhsyme(d_cckwrk[0], &d_lout, d_elementNames[0], &ichar); 
  for(int i = 0; i< d_numElements; i++) {
    strtok(d_elementNames[i]," "); //Add terminator 
  }


  //Get array of species names
  d_speciesNames = CharArray(CHARLENGTH, d_numSpecies);
  // This routine calls cksyms which returns character strings of
  // species names
  mrhsyms(d_cckwrk[0], &d_lout, d_speciesNames[0], &ichar); 
  for(int i = 0; i < d_numSpecies; i++) {
    strtok(d_speciesNames[i]," "); //Add terminator 
    //cout<<"species = " << d_speciesNames[i] << endl;
  }

  // Get array of atomic weights
  d_atomicWeight = new double[d_numElements];
  ckawt(d_ickwrk, d_rckwrk, d_atomicWeight);

  // Get array of molecular weights
  d_moleWeight = new double[d_numSpecies];
  ckwt(d_ickwrk, d_rckwrk, d_moleWeight);

}


ChemkinInterface::~ChemkinInterface()
{
  DeleteCharArray(d_speciesNames, d_numSpecies);
  DeleteCharArray(d_elementNames, d_numElements);
  DeleteCharArray(d_cckwrk, d_lencwk);

  delete [] d_atomicWeight;
  delete [] d_moleWeight;
  delete [] d_rckwrk;
  delete [] d_ickwrk;
}

int 
ChemkinInterface::getElementIndex(char *name)
{
  for(int i = 0; i < d_numElements; i++) {
    if(strlen(name) == strlen(d_elementNames[i])) {
      if(strncmp(name, d_elementNames[i], (size_t) strlen(name)) == 0) 
	{
	return i;
	}
    }
  }
  return -1;
  //return 1;
}

int 
ChemkinInterface::getSpeciesIndex(char *name)
{
  for(int i = 0; i < d_numSpecies; i++) {
    if(strlen(name) == strlen(d_speciesNames[i])) {
      if(strncmp(name, d_speciesNames[i], (size_t) strlen(name)) == 0) 
	{
	return i;
	}
    }
  }
  cout << "Did not find index of species " << name << "\n";
  return -1;
  //return 1;
}

double
ChemkinInterface::getMixMoleWeight(vector<double> Yvec)
{
  double mixMoleWeight;
  ckmmwy(&Yvec[0], d_ickwrk, d_rckwrk, &mixMoleWeight);
  // Check to see if next line causes compile time error. This will
  // answer question about const
  // d_ickwrk[1] = 1;
  return mixMoleWeight;
}

double
ChemkinInterface::getMixEnthalpy(double temp, vector<double> Yvec)
{
  double mixEnthalpy; // Units of J/kg
  ckhbms(&temp, &Yvec[0], d_ickwrk, d_rckwrk, &mixEnthalpy);
  mixEnthalpy *= 1.e-4; // Convert to SI (erg/gm) -> (J/kg)
  return mixEnthalpy;
}

double
ChemkinInterface::getMixSpecificHeat(double temp, vector<double> Yvec)
{
  double mixSpecificHeat; // Units of J/kg*K
  ckcpbs(&temp, &Yvec[0], d_ickwrk, d_rckwrk, &mixSpecificHeat);
  mixSpecificHeat = mixSpecificHeat / 1e+7 * 1000; //Convert from
  // erg/(gm K) to J/(kg K)
  return mixSpecificHeat;
}

double 
ChemkinInterface::getMassDensity(double press, double temp, 
				 vector<double> Yvec)
{
  double massDensity; // Units of kg/m^3
  double cgsPressure = press*10; // Convert from Pa to dyne/cm^2
  ckrhoy(&cgsPressure, &temp, &Yvec[0], d_ickwrk, d_rckwrk, &massDensity);
  massDensity *= 1000; // Convert to SI
  return massDensity;
}

vector<double>
ChemkinInterface::convertMolestoMass(vector<double> Xvec)
{
  vector<double> Yvec(d_numSpecies);
  ckxty(&Xvec[0], d_ickwrk, d_rckwrk, &Yvec[0]);
  return Yvec;
}

vector<double>
ChemkinInterface::convertMasstoMoles(vector<double> Yvec)
{
  vector<double> Xvec(d_numSpecies); 
  ckytx(&Yvec[0], d_ickwrk, d_rckwrk, &Xvec[0]);
  return Xvec;
}

void
ChemkinInterface::getSpeciesEnthalpy(double temp, double *speciesEnthalpy)
{
  ckhms(&temp, d_ickwrk, d_rckwrk, speciesEnthalpy);
  for (int i = 0; i < d_numSpecies; i++)
    speciesEnthalpy[i]*= 1.0e+10;  //Convert from ergs/g to J/kg                      
}

void
ChemkinInterface::getMolarRates(double press, double temp, 
				vector<double> Yvec, double *wdot)
{
  double cgsPressure = press*10; // Convert from Pa to dyne/cm^2
  ckwyp(&cgsPressure, &temp, &Yvec[0], d_ickwrk, d_rckwrk, wdot);
  for (int i=0; i < d_numSpecies; i++)
    wdot[i] *= 1e+03; //Convert from moles/(cm^3*s) to kmoles/(m^3*s)
}
  

// Access Functions 
int
ChemkinInterface::getOutFile () const {
  return d_lout;
}  

int
ChemkinInterface::getNumElements () const {
  return d_numElements;
}

int
ChemkinInterface::getNumSpecies () const {
  return d_numSpecies;
}

int
ChemkinInterface::getNumRxns () const {
  return d_numRxns;
}

int
ChemkinInterface::getNFit () const {
  // Do I ever use this??
  return d_nfit;
}

/*void
ChemkinInterface::getElementNames (char** elementNames) {
 for (int i=0; i < d_numElements; i++)
   elementNames[i] = d_elementNames[i];
   }*/

/*void
ChemkinInterface::getSpeciesNames (char** speciesNames) {
  for (int i=0; i < d_numElements; i++)
    speciesNames[i] = d_speciesNames[i];
    }*/  

void
ChemkinInterface::getAtomicWeight(double *atomicWeight) {
  for (int i=0; i < d_numElements; i++)
    atomicWeight[i] = d_atomicWeight[i];
}

void
ChemkinInterface::getMoleWeight(double *moleWeight) {
  for (int i=0; i < d_numSpecies; i++)
    moleWeight[i] = d_moleWeight[i];
}

