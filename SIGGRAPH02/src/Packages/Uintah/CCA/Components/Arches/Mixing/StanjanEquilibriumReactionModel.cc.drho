#include <Packages/Uintah/CCA/Components/Arches/Mixing/StanjanEquilibriumReactionModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/Stream.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/ChemkinInterface.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/ChemkinPrototypes.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/MixRxnTableInfo.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/KDTree.h>
//Are these necessary???
#include <Packages/Uintah/CCA/Components/Arches/Mixing/MixingModel.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <iostream>
#include <string>
#include <math.h>

using namespace std;
using namespace Uintah;
  
StanjanEquilibriumReactionModel::StanjanEquilibriumReactionModel(bool adiabatic):
  ReactionModel(), DynamicTable(), d_adiabatic(adiabatic)
{
  d_reactionData = new ChemkinInterface();
  
}

StanjanEquilibriumReactionModel::~StanjanEquilibriumReactionModel() 
{
}

void
StanjanEquilibriumReactionModel::problemSetup(const ProblemSpecP& params, 
					      MixingModel* mixModel)
{
  ProblemSpecP rxn_db = params->findBlock("EquilibriumReactionModel");
  d_mixModel = mixModel;
  d_numMixVars = d_mixModel->getNumMixVars();
  d_numRxnVars =  d_mixModel->getNumRxnVars();
  d_rxnTableDimension = d_numMixVars + d_numRxnVars + !(d_adiabatic);
  d_rxnTableInfo = new MixRxnTableInfo(d_rxnTableDimension);
  bool mixTableFlag = false;
  d_rxnTableInfo->problemSetup(rxn_db, mixTableFlag, d_mixModel);
  d_depStateSpaceVars = NUM_DEP_VARS + d_reactionData->getNumSpecies();
  d_lsoot = false;
  d_rxnTable = new KD_Tree(d_rxnTableDimension, d_depStateSpaceVars);
  // tableSetup is a function in DynamicTable; it allocates memory for table
  tableSetup(d_rxnTableDimension, d_rxnTableInfo);
  d_indepVars = vector<double>(d_rxnTableDimension);
}

void
StanjanEquilibriumReactionModel::getRxnStateSpace(Stream& unreactedMixture,
						      vector<double>& varsHFPi,
						      Stream& reactedStream)
{
  reactedStream = getProps(varsHFPi);
}

Stream
StanjanEquilibriumReactionModel::tableLookUp(int* tableKeyIndex) 
{
  Stream equilStateSpace;
  vector<double> vec_stateSpaceVars;
  bool flag = false;
#if 0
  cout << "Stanjan::tableKeyIndex = " << endl;
  for (int ii = 0; ii < d_rxnTableDimension; ii++) {
    cout.width(10);
    cout << tableKeyIndex[ii] << " " ; 
    if (!(ii % 10)) cout << endl; 
  }
#endif
   if (!(d_rxnTable->Lookup(tableKeyIndex, vec_stateSpaceVars))) 
    {
      // Convert indeces to values, get unreacted mixture
      convertIndextoValues(tableKeyIndex); //computes d_indepVars
      int mixIndex = 0;
      if (!(d_adiabatic))
	++mixIndex;
      vector<double> mixVars(d_mixModel->getNumMixVars());
      for (int jj = 0; jj < d_mixModel->getNumMixVars(); jj++) {
	mixVars[jj] = d_indepVars[mixIndex+jj];
	//cout << "Stanjan::indepVars = " << d_indepVars[jj]<< endl;
      }
      Stream unreactedStream = d_mixModel->speciesStateSpace(mixVars);
      computeRxnStateSpace(unreactedStream, d_indepVars, equilStateSpace);
      // This next section is for computing the derivative of density with
      // respect to mixture fraction and only works for systems with ONE
      // mixture fraction
      // Change the value of f by eps
      double eps = 0.0001;
      vector <double> dfMixVars(1);
      if (mixVars[0] > 0.9999) 
        dfMixVars[0] = mixVars[0] - eps;
      else
        dfMixVars[0] = mixVars[0] + eps;
      Stream dfStream =  d_mixModel->speciesStateSpace(dfMixVars);
      Stream dfStateSpace;
      computeRxnStateSpace(dfStream, d_indepVars, dfStateSpace);
      if (mixVars[0] > 0.9999) 
        equilStateSpace.d_drhodf = (equilStateSpace.d_density - dfStateSpace.d_density)/eps;
      else    
        equilStateSpace.d_drhodf = (dfStateSpace.d_density - equilStateSpace.d_density)/eps;
      vec_stateSpaceVars = equilStateSpace.convertStreamToVec();
      // defined in K-D tree implementation
      d_rxnTable->Insert(tableKeyIndex, vec_stateSpaceVars);
      //equilStateSpace.print(cerr);
    }
  else {
    //cout<<"Stanjan::entry exists"<<endl;
    bool flag = false;
    equilStateSpace.convertVecToStream(vec_stateSpaceVars, flag, 
				       d_mixModel->getNumMixVars(),
				       d_mixModel->getNumRxnVars(), d_lsoot);
  }
  return equilStateSpace;  

}


void
StanjanEquilibriumReactionModel::convertIndextoValues(int tableKeyIndex[]) 
{
  for (int i = 0; i < d_rxnTableDimension; i++)
    if (tableKeyIndex[i] <= d_rxnTableInfo->getNumDivsBelow(i)) {
      d_indepVars[i] = tableKeyIndex[i]*d_rxnTableInfo->getIncrValueBelow(i)
	+ d_rxnTableInfo->getMinValue(i);
}
    else
      d_indepVars[i] = (tableKeyIndex[i]-d_rxnTableInfo->getNumDivsBelow(i))*	
	d_rxnTableInfo->getIncrValueAbove(i) + d_rxnTableInfo->getStoicValue(i);
  //return; ASK RAJESH
}


void
StanjanEquilibriumReactionModel::computeRxnStateSpace(Stream& unreactedMixture,
						      vector<double>& mixRxnVar,
						      Stream& equilStateSpace)
{
  equilStateSpace = unreactedMixture;
  double adiabaticEnthalpy = unreactedMixture.d_enthalpy;
  double initTemp = unreactedMixture.d_temperature;
  double initPress = unreactedMixture.d_pressure;
  // Compute equilibrium for adiabatic system; if equilibrium calculation fails,
  // unreacted values are returned; the 0 in the computeEquilibrium call
  // indicates the adiabatic problem will be solved
  vector<double> initMassFract(d_reactionData->getNumSpecies());
  if (unreactedMixture.d_mole) 
    initMassFract = d_reactionData->convertMolestoMass(
				      unreactedMixture.d_speciesConcn);
  else
    initMassFract = unreactedMixture.d_speciesConcn;   
  computeEquilibrium(initTemp, initPress, initMassFract, equilStateSpace);
  equilStateSpace.d_lsoot = d_lsoot;  // No soot in this model
  double sensibleEnthalpy = 0;
  equilStateSpace.d_sensibleEnthalpy = sensibleEnthalpy;
  // Compute equilibrium for nonadiabatic system
  if (!(d_adiabatic)) {       
    //Calculate the sensible enthalpy based on a reference temperature, TREF   
    double trefEnthalpy =  d_reactionData->getMixEnthalpy(TREF, initMassFract);
    //cout<<"Stanjan::trefH = "<<trefEnthalpy<<endl;
    //cout<<"Stanjan::adH = "<<adiabaticEnthalpy<<endl;
    double sensibleEnthalpy =  adiabaticEnthalpy - trefEnthalpy;
    //cout<<"Stanjan::sensH = "<<sensibleEnthalpy<<endl;
    double absoluteEnthalpy;
    // First variable in mixRxnVar is normalized enthalpy; use it to compute
    // absolute enthalpy
    absoluteEnthalpy = adiabaticEnthalpy + mixRxnVar[0]*sensibleEnthalpy;
    // Find temperature associated with absoluteEnthalpy
    double heatLossTemp = computeTemperature(absoluteEnthalpy, initMassFract, 
					     initTemp);
    computeEquilibrium(heatLossTemp, initPress, initMassFract, equilStateSpace);
    //DEBUGGING COUT (enthalpies should be equal)
    //	cout<<"Absolute enthalpy = "<<absoluteEnthalpy<<endl
    //	    <<"Equilibrium enthalpy = "<<equilStateSpace.d_adiabaticEnthalpy<<endl;

    // This next section is for computing the derivative of density with
    // respect to enthalpy
    // Change the value of h by eps
    double eps = 0.0001;
    double dhEnthalpy;
    if (mixRxnVar[0] > 0.9999)
      dhEnthalpy = adiabaticEnthalpy + (mixRxnVar[0]-eps)*sensibleEnthalpy;
    else
      dhEnthalpy = adiabaticEnthalpy + (mixRxnVar[0]+eps)*sensibleEnthalpy;
    // Find temperature associated with dhEnthalpy
    double dhTemp = computeTemperature(dhEnthalpy, initMassFract, 
					     initTemp);
    Stream dhStateSpace;
    computeEquilibrium(dhTemp, initPress, initMassFract, dhStateSpace);
    if (mixRxnVar[0] > 0.9999)
      equilStateSpace.d_drhodh = (equilStateSpace.d_density - dhStateSpace.d_density)/eps;
    else  
      equilStateSpace.d_drhodh = (dhStateSpace.d_density - equilStateSpace.d_density)/eps;
    
    // Calculate radiation gas absorption coefficient and black body
    // emissivity for given mixture
    //computeRadiationProperties();
    equilStateSpace.d_sensibleEnthalpy = sensibleEnthalpy;
  } // if !(d_adiabatic)
   
  //Assign radiation coefficients ???
  //for (i = 0; i < nofSpecies; i++){    
  //  equilStateSpace[count++] = abkg[i];
  //  equilStateSpace[count++] = emb[i];
  //}
}


void
StanjanEquilibriumReactionModel::computeEquilibrium(double initTemp, 
					    double initPress,
					    const vector<double> initMassFract,
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
  for (int i=0; i < lenreq; i++)
    reqwrk[i] = 0;

  // Determine other required inputs
  double patm = initPress * .9869e-05; // Pa -> atm
  double test = 2000; // estimated equilibrium temperature
  double pest = patm; // estimated equilibrium pressure
  int lout = d_reactionData->getOutFile();
  // Convert vector of species mass fractions to array of mole fractions
  vector<double> Xequil = d_reactionData->convertMasstoMoles(initMassFract);
  // Compute equilibrium using Stanjan
  equil(&lout, &lprnt, &lsave, &leqst, &lcntue, d_reactionData->d_ickwrk,
	d_reactionData->d_rckwrk, &lenieq, ieqwrk, &lenreq, reqwrk, 
	&nofElements, &nofSpecies, d_reactionData->d_elementNames[0], 
	d_reactionData->d_speciesNames[0], &nop, &kmon, &Xequil[0], &initTemp, 
	&test, &patm, &pest, &ncon, kcon, xcon, &ierr); 

  if (ierr == 0) {
    // If no error detected in equilibrium solution, call eqsol, which
    // returns equilibrium solution in cgs units
    double *xeq = new double[nofSpecies]; // Equilibrium mole fractions
    vector<double> yeq(nofSpecies); // Equilibrium mass fractions
    double equilVol; // Mixture specific volume; not returned
    double sout; // Mixture entropy; not returned
    double cout_s, cdet_out; // Not returned
    eqsol(&nofSpecies, reqwrk, xeq, &yeq[0], &equilSoln.d_temperature, 
	  &equilSoln.d_pressure, &equilSoln.d_enthalpy, &equilVol, 
	  &sout, &equilSoln.d_moleWeight, &cout_s, &cdet_out);
    // Convert output to SI units (d_temperature and d_moleWeight do not 
    // require any units conversion)
    //?? put zeros in vector if mass fractions 
    // within ATOL of zero??
    equilSoln.d_speciesConcn = yeq;
    equilSoln.d_pressure *= 1.01325e+05; // atm -> Pa
    equilSoln.d_enthalpy *= 1.e-4; // Units of J/kg
    equilSoln.d_density = 1./equilVol*1.e+3; // Units of kg/m^3
    equilSoln.d_cp = d_reactionData->getMixSpecificHeat(equilSoln.d_temperature, 
							equilSoln.d_speciesConcn); 
                                                        // Units of J/(kg-K)
    // store mass fraction
    equilSoln.d_mole = false;
   
    delete[] xeq;        
  }
  else {
    equilSoln.d_pressure = initPress;
    equilSoln.d_temperature = initTemp;
    equilSoln.d_mole = false;
    equilSoln.d_speciesConcn = initMassFract;
    equilSoln.d_density = d_reactionData->getMassDensity(initPress, initTemp,
							 initMassFract);
    equilSoln.d_enthalpy = d_reactionData->getMixEnthalpy(initTemp, 
							  initMassFract);
    equilSoln.d_moleWeight = d_reactionData->getMixMoleWeight(initMassFract);
    equilSoln.d_cp = d_reactionData->getMixSpecificHeat(initTemp, initMassFract);
    cerr << "equilibrium failed for: " << endl;
    for (int ii = 0; ii < initMassFract.size(); ii++) {
      cerr << "  " << initMassFract[ii];
      if (!(ii % 10))
	cerr << endl;
    }
    cerr << endl;
    cerr << "print Enthalpy" << equilSoln.d_enthalpy << endl;
    cerr << "print Pressure" << equilSoln.d_pressure << endl;
    cerr << "print Temperature" << equilSoln.d_temperature << endl;
    cerr << "nofspecies "<< nofSpecies << endl;
    cerr << "nofelements "<< nofElements << endl;
  }

  delete[] ieqwrk;
  delete[] reqwrk;
}

double 
StanjanEquilibriumReactionModel::computeTemperature(const double absEnthalpy,
						    const vector<double>& massFract, 
						    double initTemp)
{
  double lowerTemp, upperTemp, lowerEnthalpy, upperEnthalpy, nonadTemp, del;
  lowerTemp = TLOW;
  upperTemp = THIGH;
  lowerEnthalpy = d_reactionData->getMixEnthalpy(lowerTemp,massFract);
  upperEnthalpy = d_reactionData->getMixEnthalpy(upperTemp,massFract);
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
  
  return nonadTemp;
}


void
StanjanEquilibriumReactionModel::computeRadiationProperties() {}
