#include <Packages/Uintah/CCA/Components/Arches/Mixing/StanjanEquilibriumReactionModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/Stream.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/ChemkinInterface.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/ChemkinPrototypes.h>
#include <iostream>
#include <string>
#include <math.h>

using namespace std;
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
  d_normalizeEnthalpy = false;
}

Stream
StanjanEquilibriumReactionModel::computeEnthalpy(Stream& unreactedMixture, 
						 const vector<double>& mixRxnVar)
  {
    //This routine is needed to linearize the enthalpy. Both the adiabatic 
    //and sensible enthalpy are required to linearize the absolute enthalpy 
    //passed to the mixing model from the cfd. The sensible enthalpy is chosen 
    //as an arbitrary enthalpy whereas the adiabatic enthalpy is chosen for
    //the standardized constant enthalpy.
    int nofspecies = d_reactionData->getNumSpecies();
    Stream enthalpy(nofspecies);
    // Initialized in problemSetup to false
    d_normalizeEnthalpy = true;
    bool adiabatic = false;
    enthalpy = computeRxnStateSpace(unreactedMixture, mixRxnVar, adiabatic);
    d_normalizeEnthalpy = false;
    return enthalpy;
  }    

Stream
StanjanEquilibriumReactionModel::computeRxnStateSpace(Stream& unreactedMixture,
						      const vector<double>& mixRxnVar,
						      bool adiabatic)
{
    double adiabaticEnthalpy = unreactedMixture.d_enthalpy;
    double initTemp = unreactedMixture.d_temperature;
    double initPress = unreactedMixture.d_pressure;
    int nofspecies = d_reactionData->getNumSpecies();
    // Compute equilibrium for adiabatic system; if equilibrium calculation fails,
    // unreacted values are returned; the 0 in the computeEquilibrium call
    // indicates the adiabatic problem will be solved
    //int heatLoss = 0;
    Stream equilStateSpace(nofspecies);
    vector<double> initMassFract;
    if (unreactedMixture.d_mole) 
      initMassFract = d_reactionData->convertMolestoMass(
				      unreactedMixture.d_speciesConcn);
    else
      initMassFract = unreactedMixture.d_speciesConcn;

    //computeEquilibrium(initTemp, initPress, heatLoss, adiabaticEnthalpy, 
    //		       initMassFract, equilStateSpace);
    computeEquilibrium(initTemp, initPress, initMassFract, equilStateSpace);
    double sensibleEnthalpy = 0;
    equilStateSpace.d_sensibleEnthalpy = sensibleEnthalpy;
    // Compute equilibrium for nonadiabatic system
    if (!(adiabatic)) {       
      //Calculate the sensible enthalpy based on a reference temperature, TREF   
      double trefEnthalpy =  d_reactionData->getMixEnthalpy(TREF, initMassFract);\
      double sensibleEnthalpy =  adiabaticEnthalpy - trefEnthalpy;
      // This section can be skipped if computeRxnStateSpace was only called to
      // get enthalpies for normalization
      if (!d_normalizeEnthalpy) {
	double absoluteEnthalpy;
	// First variable in mixRxnVar is normalized enthalpy; use it to compute
	// absolute enthalpy
	absoluteEnthalpy = adiabaticEnthalpy + mixRxnVar[0]*sensibleEnthalpy;
	// Find temperature associated with absoluteEnthalpy
	double heatLossTemp = computeTemperature(absoluteEnthalpy, initMassFract, 
						 initTemp);
	//heatLoss = 1;
	computeEquilibrium(heatLossTemp, initPress, initMassFract, equilStateSpace);
	//DEBUGGING COUT (enthalpies should be equal)
	//	cout<<"Absolute enthalpy = "<<absoluteEnthalpy<<endl
	//	    <<"Equilibrium enthalpy = "<<equilStateSpace.d_adiabaticEnthalpy
	//          <<endl;
	
	// Calculate radiation gas absorption coefficient and black body
	// emissivity for given mixture
	//computeRadiationProperties();
      }
      equilStateSpace.d_sensibleEnthalpy = sensibleEnthalpy;
    }
    
    //Assign radiation coefficients ???
    //for (i = 0; i < nofSpecies; i++){    
    //  equilStateSpace[count++] = abkg[i];
    //  equilStateSpace[count++] = emb[i];
    //}
    return equilStateSpace;
}

void
StanjanEquilibriumReactionModel::computeEquilibrium(double initTemp, 
					    double initPress,
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

  // Compute equilibrium using Stanjan
  equil(&lout, &lprnt, &lsave, &leqst, &lcntue, d_reactionData->d_ickwrk,
	d_reactionData->d_rckwrk, &lenieq, ieqwrk, &lenreq, reqwrk, 
	&nofElements, &nofSpecies, d_reactionData->d_elementNames[0], 
	d_reactionData->d_speciesNames[0], &nop, &kmon, Xarray, &initTemp, 
	&test, &patm, &pest, &ncon, kcon, xcon, &ierr); 
	//	&heatLoss, &enthalpy, &ierr);

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
    //cout<<"equil temp = "<<equilSoln.d_temperature<<endl;
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
							  initMassFract);
    equilSoln.d_moleWeight = d_reactionData->getMixMoleWeight(Yarray);
    equilSoln.d_cp = d_reactionData->getMixSpecificHeat(initTemp, Yarray);
    cerr << "equilibrium failed for: " << endl;
    for (int ii = 0; ii < initMassFract.size(); ii++) {
      cerr << "  " << Xarray[ii];
      if (!(ii % 10))
	cerr << endl;
    }
    cerr << endl;
    cerr << "print Enthalpy" << equilSoln.d_enthalpy << endl;
    cerr << "print Pressure" << equilSoln.d_pressure << endl;
    cerr << "print Temperature" << equilSoln.d_temperature << endl;
    cerr << "nofspecies "<< nofSpecies << endl;
    cerr << "nofelements "<< nofElements << endl;
    cerr << d_reactionData->d_speciesNames[25] << endl;
    cerr << d_reactionData->d_elementNames[3] << endl;
  }

  delete[] ieqwrk;
  delete[] reqwrk;
  delete[] Xarray;
  delete[] Yarray;
}

double 
StanjanEquilibriumReactionModel::computeTemperature(const double absEnthalpy,
						    const vector<double>& massFract, 
						    double initTemp)
{
	double lowerTemp, upperTemp, lowerEnthalpy, upperEnthalpy, nonadTemp, del;
	lowerTemp = TLOW;
	upperTemp = THIGH;
	//cout<<"TLOW="<<TLOW<<" THIGH="<<THIGH<<" TREF="<<TREF<<endl;
	lowerEnthalpy = d_reactionData->getMixEnthalpy(lowerTemp,massFract);
	upperEnthalpy = d_reactionData->getMixEnthalpy(upperTemp,massFract);
        //cout<<"lowerEnthalpy= "<<lowerEnthalpy<<" upperEnthalpy="<<upperEnthalpy<<endl;
	int iter = 0;
	do {
	  iter += 1;
	  if (fabs(upperEnthalpy-lowerEnthalpy) < 0.0001) {
	    nonadTemp = upperTemp;
	  }
	  else {
	    nonadTemp = upperTemp-(upperEnthalpy-absEnthalpy)*
	      (upperTemp-lowerTemp)/(upperEnthalpy-lowerEnthalpy);
	  }
	  del = nonadTemp - upperTemp;
	  lowerTemp = upperTemp;
	  lowerEnthalpy = upperEnthalpy;
	  upperTemp = nonadTemp;
	  upperEnthalpy = d_reactionData->getMixEnthalpy(upperTemp, massFract);
	} while((fabs(del) > 0.0001) && (iter < MAXITER));
	// If secant method fails to find a solution, set nonadTemp to initTemp
	if (iter == MAXITER) {
	  nonadTemp = initTemp;
	  //cout<<"At f = "<<d_mixVar[0]<<" and h = "<<d_normH<<" the max number of"
	  //  <<" iterations was exceeded when computing the initial temperature"<<endl
	  //  <<"Using temperature for adiabatic system."<<endl; 
	}
	else if (nonadTemp < TLOW) {
	  nonadTemp = initTemp;
	  //cout<<"At f = "<<d_mixVar[0]<<" and h = "<<d_normH<<" the computed"
	  //  <<" initial temperature is less than "<<TLOW<<endl
	  //  <<"Using temperature for adiabatic system."<<endl;
	}
	else if (nonadTemp > THIGH) {
	  nonadTemp = initTemp;
	  //cout<<"At f = "<<d_mixVar[0]<<" and h = "<<d_normH<<" the computed"
	  //  <<" initial temperature exceeds "<<THIGH<<endl
	  //  <<"Using temperature for adiabatic system."<<endl;
	}	  
	//DEBUGGING COUT

	return nonadTemp;
}

void
StanjanEquilibriumReactionModel::computeRadiationProperties() {}
