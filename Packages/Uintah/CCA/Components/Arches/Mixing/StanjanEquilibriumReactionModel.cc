#include <Packages/Uintah/CCA/Components/Arches/Mixing/StanjanEquilibriumReactionModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/Stream.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/ChemkinInterface.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/ChemkinPrototypes.h>
#include <string>

using namespace Uintah;
  
StanjanEquilibriumReactionModel::StanjanEquilibriumReactionModel(bool adiabatic):
  ReactionModel(), d_adiabatic(adiabatic)
{
  d_reactionData = new ChemkinInterface();
  
}

StanjanEquilibriumReactionModel::~StanjanEquilibriumReactionModel() 
{
}
void
StanjanEquilibriumReactionModel::problemSetup(const ProblemSpecP& /* params*/)
{
}

Stream
StanjanEquilibriumReactionModel::computeRxnStateSpace(Stream& unreactedMixture)
{
    double adiabaticEnthalpy = unreactedMixture.d_enthalpy;
    double initTemp = unreactedMixture.d_temperature;
    double initPress = unreactedMixture.d_pressure;
    int nofspecies = d_reactionData->getNumSpecies();
    // Compute equilibrium for adiabatic system; if equilibrium calculation fails,
    // unreacted values are returned; the 0 in the computeEquilibrium call
    // indicates the adiabatic problem will be solved
    int heatLoss = 0;
    Stream equilStateSpace(nofspecies);
    vector<double> initMassFract;
    if (unreactedMixture.d_mole) 
      initMassFract = d_reactionData->convertMolestoMass(
				      unreactedMixture.d_speciesConcn);
    else
      initMassFract = unreactedMixture.d_speciesConcn;

    computeEquilibrium(initTemp, initPress, heatLoss, adiabaticEnthalpy, 
		       initMassFract, equilStateSpace);
    double sensibleEnthalpy = 0;
    // Compute equilibrium for nonadiabatic system
#if 0
    if (!(getAdiabatic())) {
      double* equilMassFract= new double[d_numSpecies];
      for  (int ii = 0; ii < d_numSpecies; ii++)
	equilMassFract[ii] = equilStateSpace.d_speciesMassFract[ii];        
      //Calculate the sensible enthalpy based on a reference temperature, TREF   
      double trefEnthalpy =  d_reactionData->getMixEnthalpy(TREF, equilMassFract);
      double sensibleEnthalpy =  adiabaticEnthalpy - trefEnthalpy;
      
      if (!d_normalizeEnthalpy) {
	double absoluteEnthalpy;
	// First variable in mixRxnVar is normalized enthalpy; use it to compute
	// absolute enthalpy
	absoluteEnthalpy = adiabaticEnthalpy + mixRxnVar[0]*sensibleEnthalpy;
	// Find temperature associated with absoluteEnthalpy
	double heatLossTemp = computeTemperature(absoluteEnthalpy);
	heatLoss = 1;
	equilStateSpace = computeEquilibrium(heatLossTemp, heatLoss, absoluteEnthalpy);
	//DEBUGGING COUT (enthalpies should be equal)
	//	cout<<"Absolute enthalpy = "<<absoluteEnthalpy<<endl
	//	    <<"Equilibrium enthalpy = "<<equilStateSpace.d_adiabaticEnthalpy
	//          <<endl;
	
	// Calculate radiation gas absorption coefficient and black body
	// emissivity for given mixture
	//computeRadiationProperties();
      }
      delete[] equilMassFract;
    }
#endif 
    equilStateSpace.d_sensibleEnthalpy = sensibleEnthalpy;
    
    //Assign radiation coefficients ???
    //for (i = 0; i < nofSpecies; i++){    
    //  equilStateSpace[count++] = abkg[i];
    //  equilStateSpace[count++] = emb[i];
    //}
    return equilStateSpace;
}

void
StanjanEquilibriumReactionModel::computeEquilibrium(double tin, 
					    double initPress,
					    int heatLoss, 
					    double enthalpy, 
					    const vector<double>& initMassFract,
					    Stream& equilSoln) 
{
  // Set parameters for equilibrium calculation
  int nop = 5; // Specified P and H problem is to be solved
  int lprnt = 0; // logical flag for printing results
  int lsave = 0; // unit number for binary solution file
  int leqst = 0; // logical flag for initializing Stanjan
  int lcntue = 0; // logical flag for continuation
  int kmon = 0; // integer monitor flag
  int ncon = 0; // integer number of species to be held constant
  int kcon[1]; // integer array of index numbers for the species to be 
               // held constant
  double xcon[1]; // real mole fractions input for the species to be 
                  // held constant
  int ierr = 0; // integer flag =0 if equilibrium calculation successful,
                // =1 if calculation unsuccessful
  int nphase = 2;
  //Both lengths computed here are kluged; I don't know wny formula doesn't work
  int nofSpecies = d_reactionData->getNumSpecies();
  int nofElements = d_reactionData->getNumElements();
  int lenieq = 3*(22 + 14*nofElements + 4*nphase + 8*nofSpecies + 
               2*nofElements*nofSpecies);
  int lenreq = 3*(24 + 16*nofElements + 12*nofElements*nofElements + 
               3*nphase*nofElements + 6*nphase + 18*nofSpecies + 4*nofElements
               *nofElements + 2*nofElements);
  int *ieqwrk = new int[lenieq]; // integer equilibrium work space
  double *reqwrk = new double[lenreq]; // real equilibrium work space

  // Determine other required inputs
  double patm = initPress * .9869e-05; // Pa -> atm
  double test = 2000; // estimated equilibrium temperature
  double pest = patm; // estimated equilibrium pressure
  int lout = d_reactionData->getOutFile();
  // Convert vector of species mass fractions to array of mole fractions
  vector<double> Xequil = d_reactionData->convertMasstoMoles(initMassFract);
  double *Yarray = new double[nofSpecies];
  double *Xarray = new double[nofSpecies];
  // convert vector to array
  for (int ii = 0; ii < nofSpecies; ii++) {
    Xarray[ii] = Xequil[ii];
    Yarray[ii] = initMassFract[ii];
  }
  double initTemp = tin;
  // Compute equilibrium using Stanjan
  equil(&lout, &lprnt, &lsave, &leqst, &lcntue, d_reactionData->d_ickwrk,
	d_reactionData->d_rckwrk, &lenieq, ieqwrk, &lenreq, reqwrk, 
	&nofElements, &nofSpecies, d_reactionData->d_elementNames[0], 
	d_reactionData->d_speciesNames[0], &nop, &kmon, Xarray, &tin, 
	&test, &patm, &pest, &ncon, kcon, xcon, 
	&heatLoss, &enthalpy, &ierr);

  if (ierr == 0) {
    // If no error detected in equilibrium solution, call eqsol, which
    // returns equilibrium solution in cgs units
    double *xeq = new double[nofSpecies]; // Equilibrium mole fractions
    double *yeq = new double[nofSpecies]; // Equilibrium mass fractions
    double equilVol; // Mixture specific volume; not returned
    double sout; // Mixture entropy; not returned
    double cout_s, cdet_out; // Not returned
    eqsol(&nofSpecies, reqwrk, xeq, yeq, &equilSoln.d_temperature, 
	  &equilSoln.d_pressure, &equilSoln.d_enthalpy, &equilVol, 
	  &sout, &equilSoln.d_moleWeight, &cout_s, &cdet_out);
    // Convert output to SI units (d_temperature and d_moleWeight do not 
    // require any units conversion)
    equilSoln.d_pressure *= 1.01325e+05; // atm -> Pa
    equilSoln.d_enthalpy *= 1.e-4; // Units of J/kg
    equilSoln.d_density = 1./equilVol*1.e+3; // Units of kg/m^3
    equilSoln.d_cp = d_reactionData->getMixSpecificHeat(equilSoln.d_temperature, 
							yeq); // Units of J/(kg-K)
    // Assign yeq array to output vector; ?? put zeros in vector if mass fractions 
    // within ATOL of zero??
    // store mass fraction
    equilSoln.d_mole = false;
    for (int i = 0; i < nofSpecies; i++)
      equilSoln.d_speciesConcn[i] = yeq[i]; 
    
    delete[] xeq;
    delete[] yeq;        
  }
  else {
    equilSoln.d_pressure = initPress;
    equilSoln.d_temperature = initTemp;
    equilSoln.d_mole = false;
    equilSoln.d_speciesConcn = initMassFract;
    equilSoln.d_density = d_reactionData->getMassDensity(initPress, initTemp,
							 Yarray);
    equilSoln.d_enthalpy = d_reactionData->getMixEnthalpy(initTemp, 
							  Yarray);
    equilSoln.d_moleWeight = d_reactionData->getMixMoleWeight(Yarray);
    equilSoln.d_cp = d_reactionData->getMixSpecificHeat(initTemp, Yarray);
  }

  delete[] ieqwrk;
  delete[] reqwrk;
  delete[] Xarray;
  delete[] Yarray;
}


