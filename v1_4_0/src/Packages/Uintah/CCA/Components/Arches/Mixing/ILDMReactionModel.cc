//----- PDFMixingModel.cc --------------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/Mixing/ILDMReactionModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/StanjanEquilibriumReactionModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/KDTree.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/MixRxnTableInfo.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/ChemkinInterface.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/Stream.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/PDFMixingModel.h>


#include <string>
#include <fstream>
#include <iostream>

using namespace std;
using namespace Uintah;
using namespace SCIRun;
//****************************************************************************
// Default constructor for ILDMReactionModel
//****************************************************************************
ILDMReactionModel::ILDMReactionModel(bool adiabatic):
  ReactionModel(), DynamicTable(), d_adiabatic(adiabatic)
  
							    
{
  d_reactionData = new ChemkinInterface();
}

//****************************************************************************
// Destructor
//****************************************************************************
ILDMReactionModel::~ILDMReactionModel()
{
}

//****************************************************************************
// Problem Setup for ILDMReactionModel
//****************************************************************************
void 
ILDMReactionModel::problemSetup(const ProblemSpecP& params, 
				MixingModel* mixModel)
{  
  ProblemSpecP rxn_db = params->findBlock("ILDMReactionModel");
  rxn_db->require("ComputeSoot", d_lsoot);
  int numSpecInTbl;
  rxn_db->require("SpeciesInTable", numSpecInTbl);
  d_mixModel = mixModel;
  d_numMixVars = mixModel->getNumMixVars();
  d_numRxnVars =  mixModel->getNumRxnVars();
  int rxnTableDimension = d_numMixVars + d_numRxnVars + !(d_adiabatic);
  d_rxnTableInfo = new MixRxnTableInfo(rxnTableDimension);
  bool mixTable = false;
  d_rxnTableInfo->problemSetup(rxn_db, mixTable, mixModel);
  //Include entries for minimum and maximum value of parameter in kdtree
  // ***Rajesh- it doesn't work to get totalVar from Stream, because it uses 
  // first stream constructor***
  int sootTrue = 0;
  if (d_lsoot)
    sootTrue = 1;
  d_depStateSpaceVars = NUM_DEP_VARS + d_reactionData->getNumSpecies() + 2*sootTrue 
    + 3*d_numRxnVars;
  //int depStateSpaceVars = mixModel->getTotalVars();
  d_rxnTable = new KD_Tree(rxnTableDimension, d_depStateSpaceVars);
  // tableSetup is a function in DynamicTable; it allocates memory for table
  tableSetup(rxnTableDimension, d_rxnTableInfo);
 
  //Compute total number of entries that should be in table 
  int totalEntries = 1; //Don't want to multiply by zero if problem adiabatic
  int count = 0;
  if (!(d_adiabatic)) {
    totalEntries = d_rxnTableInfo->getNumDivsBelow(count) +
      d_rxnTableInfo->getNumDivsAbove(count) + 1;
    count++;
  }
  for (int jj = 0; jj < d_numMixVars; jj++) {
    totalEntries *= d_rxnTableInfo->getNumDivsBelow(count) +
      d_rxnTableInfo->getNumDivsAbove(count) + 1;
    count++;
  }
  for (int jj = 0; jj < d_numRxnVars; jj++) {
    totalEntries *= d_rxnTableInfo->getNumDivsBelow(count) +
      d_rxnTableInfo->getNumDivsAbove(count) + 1;
    count++;
  }

  // This next section assumes there is only ONE reaction variable
  // !!!This section is specfically coded to read Diem's ILDM file format!!!
  //cout<<"Code is currently hard-wired to read in data for ONE rxn variable"
  //    <<endl;
  cout<<"MAKE SURE there are "<<d_rxnTableInfo->getNumDivsBelow(count-1)+
    d_rxnTableInfo->getNumDivsAbove(count-1)+1<<" entries for the reaction parameter"
      << endl;
  //cout<< "Otherwise, YOUR RESULTS WILL BE SCREWED UP" <<endl;
 
  //Data files being read in are set up as follows. The word ENTRY appear at the
  //start of each new data set , i.e. each (h,f) pair. The first line after ENTRY
  //contains four values: h (linearized), f, max value of parameter, min value of parameter.
  //The rest of the lines are the state space vector at each value of the parameter.
  ifstream ildmfile("ILDMData");  
  if (ildmfile.fail()) {
    cout<<"ERROR in ILDMReactionModel"<<endl
	<<"    Could not open ILDMData file."<<endl;
    exit(1);
  }
  //Look for flag, then read vector data
  char flagKey[6];
  vector<double> vec_stateSpaceVars(d_depStateSpaceVars, 0.0);
  vector<double> indepVars(rxnTableDimension, 0.0);
  double value = 0.0;
  double minParamValue = 0.0;
  double maxParamValue = 0.0;
  int dataCount = 0;
  int rxnCount = d_numMixVars + !(d_adiabatic);
  while(ildmfile)
    { 
      //Look for ILDM table input until the end of the file is reached.
      ildmfile.getline(flagKey, 6);
      //ildmfile.ignore(200,'\n');        //Move to next line
      if (strcmp(flagKey, "ENTRY") == 0) // If keyword is found, start reading
	//subsequent lines
	{
	  //First line after ENTRY line contains values for linearized enthalpy,
	  //mixture fractions(s), min/max values of parameter(s)
	  //Remove enthalpy from indepVars if problem adiabatic
	  int kk = 0;
	  ildmfile>>value;
	  if (!(d_adiabatic)) {
	    indepVars[kk] = value;
	    kk++;
	  }
	  for (int ii = 0; ii < d_numMixVars; ii++) {
	    ildmfile>>value;
	    indepVars[kk] = value;
	    kk++;
	  }
	  ildmfile>>minParamValue>>maxParamValue;
	  ildmfile.ignore(200,'\n');    //Move to next line
	  //Read lines containing state space information
	  for(int ii = 0; ii < (d_rxnTableInfo->getNumDivsBelow(rxnCount)+
				d_rxnTableInfo->getNumDivsAbove(rxnCount)+1); ii++)
	    {
	      double ignore, rxnParam;
	      int vecCount = 0;
	      dataCount++; // Counter for nunmber of total entries in data file
	      // First two entries are unnormalized value of parameter and entropy
	      ildmfile>>rxnParam>>ignore;
	      ildmfile>>value; //Read pressure in atm
	      vec_stateSpaceVars[vecCount++] = value*101325; //Convert pressure to Pa
	      ildmfile>>value; //Read density in g/cm^3
	      vec_stateSpaceVars[vecCount++] = value*1.e+3;// Units of kg/m^3
	      ildmfile>>value; //Read in temperature in K
	      vec_stateSpaceVars[vecCount++] = value;
	      ildmfile>>value; //Read in enthalpy in ergs/g
	      vec_stateSpaceVars[vecCount++] = value*1.e-4; // Units of J/kg
 
	      // Sensible enthalpy not included in data file, so compute it 
	      // here
	      double sensibleEnthalpy = 0;
	      int ivCount = 0;  
	      vector<double> mixVar(d_numMixVars); 
	      if (!(d_adiabatic)) { 
		 ivCount ++;
		 for (int i = 0; i < d_numMixVars; i++) {
		   mixVar[i] = indepVars[ivCount];
		   ivCount++;
		 }	 
		 Stream sensStream = mixModel->speciesStateSpace(mixVar);
		 double adiabaticEnthalpy = sensStream.d_enthalpy;
		 vector<double> initMassFract(d_reactionData->getNumSpecies());
		 if (sensStream.d_mole) 
		   initMassFract = d_reactionData->convertMolestoMass(
				   sensStream.d_speciesConcn);
		 else
		   initMassFract = sensStream.d_speciesConcn; 
		 // Calculate the sensible enthalpy based on a reference 
		 // temperature, TREF   
		 double trefEnthalpy =  
		   d_reactionData->getMixEnthalpy(TREF, initMassFract);
		 //cout<<"ILDM::trefH = "<<trefEnthalpy<<endl;
		 //cout<<"ILDM::adH = "<<adiabaticEnthalpy<<endl;
		 sensibleEnthalpy =  adiabaticEnthalpy - trefEnthalpy;
		 //cout<<"ILDM::sensH = "<<sensibleEnthalpy<<endl;
	      }	      
	      vec_stateSpaceVars[vecCount++] = sensibleEnthalpy;
	      ildmfile>>value; //Read in mix MW
	      vec_stateSpaceVars[vecCount++] = value;
	      ildmfile>>value; //Read in mix heat capacity in ergs/(g-K)
	      vec_stateSpaceVars[vecCount++] = value/1e+7*1000; //Convert from
	      // erg/(gm K) to J/(kg K)
	      // Read in species mass fractions
	      for (int jj = NUM_DEP_VARS; jj < NUM_DEP_VARS + numSpecInTbl; jj++) {
		ildmfile>>value;
		vec_stateSpaceVars[vecCount++] = value;
	      }
	      // Not all species in chem.inp are included in data file;
	      // set those species conc to zero.
	      for (int jj =  NUM_DEP_VARS + numSpecInTbl; jj < NUM_DEP_VARS 
		     + d_reactionData->getNumSpecies(); jj++)
		vec_stateSpaceVars[vecCount++] = 0.0; 
	      // Read in soot data
	      double sootDiam, sootFV;
	      ildmfile>>sootDiam>>sootFV;
	      // Read in rxn rate of parameter
	      ildmfile>>value;
	      vec_stateSpaceVars[vecCount++] = value;
	      // Now assign min, max values of parameter to proper vector location
	      vec_stateSpaceVars[vecCount++] = minParamValue;
	      vec_stateSpaceVars[vecCount++] = maxParamValue;
	      // Assign soot data to end of vector
	      vec_stateSpaceVars[vecCount++] = sootDiam;
	      vec_stateSpaceVars[vecCount++] = sootFV;
	      //ildmfile.ignore(200,'\n');    //Move to next line
	      indepVars[rxnCount] = (rxnParam - minParamValue)/
		(maxParamValue - minParamValue);
	      //for (int kk = 0; kk < vec_stateSpaceVars.size(); kk++) {
	      //	cout.width(10);
	      //	cout << vec_stateSpaceVars[kk] << " " ; 
	      //	if (!(kk % 10)) cout << endl; 
	      //     }
	      //cout << endl;
	      ildmfile.ignore(200,'\n');    //Move to next line

	      //Convert indepVars to tableKeyIndex
	      int* tableIndex = new int[rxnTableDimension];//??+1??
	      double tableValue;
	      for (int i = 0; i < rxnTableDimension; i++)
		{
		  // calculates index in the table
		  double midPt = d_rxnTableInfo->getStoicValue(i);
		  if (indepVars[i] <= midPt) 
		    tableValue = (indepVars[i] - d_rxnTableInfo->getMinValue(i))/  
		      d_rxnTableInfo->getIncrValueBelow(i);
		  else
		    {
		      tableValue = ((indepVars[i] - midPt)/d_rxnTableInfo->getIncrValueAbove(i))
			+ d_rxnTableInfo->getNumDivsBelow(i);
		    }
		  tableValue = tableValue + 0.5;
		  tableIndex[i] = (int) tableValue; // cast to int
		  if (tableIndex[i] > (d_rxnTableInfo->getNumDivsBelow(i)+
					d_rxnTableInfo->getNumDivsAbove(i))||
		      (tableIndex[i] < 0))	
		    cerr<<"Index value out of range in RxnTable"<<endl;		   
		}
	      d_rxnTable->Insert(tableIndex, vec_stateSpaceVars);
	      delete [] tableIndex;
	    } // for(i = 0 to total number of lines under ENTRY 
	} //if (strcmp(flagKey, "ENTRY") == 0)
    } //while(ildmfile)
      ildmfile.close();

      //Check to see if number of entries in datafile match the size of table 
      //specified above
      cerr << "dataCount = " << dataCount << " totalEntries = " << totalEntries << endl;
      assert(dataCount==totalEntries);
}


//****************************************************************************
// Read manifold data from kdtree here
//****************************************************************************
void
ILDMReactionModel::getRxnStateSpace(Stream& unreactedMixture, 
				    vector<double>& varsHFPi, Stream& reactedStream)
{
  // WARNING - The following line is hardcoded for one mixing variable and
  // one rxn variable!!!
  reactedStream = unreactedMixture;
  //Stream reactedStream(d_reactionData->getNumSpecies(), 
  //	       d_reactionData->getNumElements(), 1, 1, d_lsoot);
  // Define what happens if h, f or pi are outside tabulated range
  int count = 0;
  if (!(d_adiabatic)) {
    if (varsHFPi[count] <  d_rxnTableInfo->getMinValue(count)) {
      varsHFPi[count] =  d_rxnTableInfo->getMinValue(count);
      cout<< "Linearized enthalpy (=" << varsHFPi[count] << ") less than" <<
	" lowest value in table (=" << d_rxnTableInfo->getMinValue(count) << ")" << endl;
      cout << "So, value set to " <<  d_rxnTableInfo->getMinValue(count) << endl;
    }
    if (varsHFPi[count] >  d_rxnTableInfo->getMaxValue(count)) {
      varsHFPi[count] =  d_rxnTableInfo->getMaxValue(count);
      cout << "Linearized enthalpy (=" << varsHFPi[count] << ") greater than" <<
	" highest value in table (= " <<  d_rxnTableInfo->getMaxValue(count) << ")" << endl;
      cout << "So, value set to " << d_rxnTableInfo->getMaxValue(count) << endl;  
    }
    count++;
  }
  // If f outside tabulated range, return unreacted values
  if ((varsHFPi[count] < d_rxnTableInfo->getMinValue(count))||
      (varsHFPi[count] > d_rxnTableInfo->getMaxValue(count))) {
    return;
    //Should I change enthalpy if it was out of range???
  }
  else {
    //cerr << "ILDM::rxnStateSpace::getProps = " << varsHFPi[0] << " " <<
    //varsHFPi[1] << " " << varsHFPi[2] << endl;
    reactedStream = getProps(varsHFPi); //function in DynamicTable
    //What about radiation properties??
    return;
  }
}

Stream
ILDMReactionModel::tableLookUp(int* tableKeyIndex) 
{
  Stream stateSpaceVars;
  vector<double> vec_stateSpaceVars;
   if (d_rxnTable->Lookup(tableKeyIndex, vec_stateSpaceVars)) 
    {
      bool flag = false;
      stateSpaceVars.convertVecToStream(vec_stateSpaceVars, flag, 
					d_numMixVars, d_numRxnVars, d_lsoot);
      //stateSpaceVars.print(cout);
    
    } 
   else
     {
       cout << "Table entry not found in ILDM::tableLookup" <<endl;
       exit(1);
     }
  return stateSpaceVars;
  
}





