//----- ILDMReactionModel.cc --------------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/Mixing/ILDMReactionModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/StanjanEquilibriumReactionModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/KDTree.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/VectorTable.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/MixRxnTable.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/MixRxnTableInfo.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/ChemkinInterface.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/Stream.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/PDFMixingModel.h>

#include <string>
#include <fstream>
#include <iostream>

#include <math.h>
#include <Core/Math/MiscMath.h>

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
 cout << "in problemSetup" << endl; 
 ProblemSpecP rxn_db = params->findBlock("ILDMReactionModel");
 d_lsoot = true;
 d_numSpecInTbl = 13;
  d_mixModel = mixModel;
  d_numMixVars = d_mixModel->getNumMixVars();
  d_numRxnVars =  d_mixModel->getNumRxnVars(); 
  //Include entries for minimum and maximum value of parameter in kdtree
  // ***Rajesh- it doesn't work to get totalVar from Stream, because it uses 
  // first stream constructor***
  int sootTrue = 0;
  if (d_lsoot)
    sootTrue = 1;
  d_depStateSpaceVars = NUM_DEP_VARS + 4 + 2*sootTrue + 3*d_numRxnVars;
  // Only printing out four species: CO2, H2O, O2, CO 
  d_rxnTableDimension = d_numMixVars + d_numRxnVars + !(d_adiabatic);

  cout << "Calling StanjanEquilibriumReactionModel constructor" << endl;
  cout << "Equilibrium values will be returned outside ILDM range" << endl;
  d_equilModel = new StanjanEquilibriumReactionModel(d_adiabatic);

  // Only need to set up reaction table if mixing table is dynamic
  string mixTableType = mixModel->getMixTableType();
  if (mixTableType == "dynamic") {
    d_rxnTableInfo = new MixRxnTableInfo(d_rxnTableDimension);
    bool mixTable = false;
    d_rxnTableInfo->problemSetup(rxn_db, mixTable, d_mixModel);
    // Set up table, either as vectors or as kdtree
    string tableType, tableStorage;
    //tableType is always static
    if (rxn_db->findBlock("TableType")) {
      rxn_db->require("TableType", tableType);
      if (tableType != "static") 
	cerr << "ILDM table is static; must have file named ILDMData!!!" << endl;
    }
    else
      cerr << "ILDM table is static; must have file named ILDMData!!!" << endl;
    if (rxn_db->findBlock("TableStorage")) {
      rxn_db->require("TableStorage", tableStorage);
      if (tableStorage == "KDTree")
	d_rxnTable = new KD_Tree(d_rxnTableDimension, d_depStateSpaceVars);
      else if (tableStorage == "2DVector")
	d_rxnTable = new VectorTable(d_rxnTableDimension, d_rxnTableInfo);
      else {
	d_rxnTable = new VectorTable(d_rxnTableDimension, d_rxnTableInfo);
	cerr << "ILDM TABLE STORAGE is vectorTable" << endl;
	//throw InvalidValue("Table storage not supported" + tableStorage);
      }
    }
    else {
      d_rxnTable = new VectorTable(d_rxnTableDimension, d_rxnTableInfo);
      cout << "ILDM TABLE STORAGE is vectorTable" << endl;
    }
    // tableSetup is a function in DynamicTable; it allocates memory for 
    // table functions
    tableSetup(d_rxnTableDimension, d_rxnTableInfo); 

    // Read table in from data file
    readStaticTable();   
  } // IfmixTableType = "dynamic"
}

void 
ILDMReactionModel::readStaticTable() 
{
  // This next section assumes there is only ONE reaction variable
  // !!!This section is specfically coded to read Diem's ILDM file format!!!
  //Data files being read in are set up as follows. The word ENTRY appear at the
  //start of each new data set , i.e. each (h,f) pair. The first line after ENTRY
  //contains four values: h (linearized), f, max value of parameter, min value of parameter.
  //The rest of the lines are the state space vector at each value of the parameter.
  
  //Compute total number of entries that should be in data file 
  int totalEntries = 1; //Don't want to multiply by zero
  for (int ii = 0; ii < d_rxnTableDimension; ii++)
    totalEntries *= d_rxnTableInfo->getNumDivsBelow(ii) +
      d_rxnTableInfo->getNumDivsAbove(ii) + 1;
 
  // Fill in table entries not included in Diem's data file
  //vector<double> vec_stateSpaceVars(d_depStateSpaceVars, 0.0);
  int *tableIndex = new int[d_rxnTableDimension];
  vector<int> numEntries(d_rxnTableDimension, 0.0);
  Stream stateSpaceVars;
  vector<double> indepVars(d_rxnTableDimension, 0.0);
  int entryCount = 0;
  int count = 0;
 
  if (!(d_adiabatic)) {
    numEntries[count] = d_rxnTableInfo->getNumDivsBelow(count) + 
      d_rxnTableInfo->getNumDivsAbove(count) + 1;
    cout << "d_adiabatic num entries = " << numEntries[count] << endl;
    count++;
  }
  int mixCount = count;
  numEntries[count] = d_rxnTableInfo->getNumDivsBelow(count) + 
    d_rxnTableInfo->getNumDivsAbove(count) + 1; //Number of mixture fraction entries
  cout << "mix num entries = " << numEntries[count] << endl;
  count++;
  int rxnCount = count;
  numEntries[count] = d_rxnTableInfo->getNumDivsBelow(count) + 
    d_rxnTableInfo->getNumDivsAbove(count) + 1;
  cout << "rxn num entries = " << numEntries[count] << endl;

  int CO2index = 6;
  int H2Oindex = 5; 
  int O2index = 2;
  int COindex = 7;
  Stream mixedStream;
  Stream reactedStream(4,0,1,1,1);
  vector<double> mixVars(1); // Hardcoded for only 1 mixing variable!!!
  bool tableInsert = false;
  double fmin = 0.028599;
  double fmax = 0.156899;
  int minfIndex =  7;
  int maxfIndex = 23;
  int minhIndex = 3;
  int iimin, iimax;
  if (!(d_adiabatic)) {
    iimin = 0;
    iimax = numEntries[0];
  }
  else {
    iimin = 1;
    iimax = 2;
  }
  for (int ii = iimin; ii < iimax; ii++)
    {
      if (!(d_adiabatic)) {
	  tableIndex[0] = ii;
      }
      for (int jj = 0; jj < numEntries[mixCount]; jj++) 
        {
	  tableIndex[mixCount] = jj;
	  for (int kk = 0; kk < numEntries[rxnCount]; kk++) 
	    {
	      tableIndex[rxnCount] = kk;
	      cout << "createTable::tableIndex = " << endl;
	      for (int pp = 0; pp < d_rxnTableDimension; pp++) {
		cout.width(10);
		cout << tableIndex[pp] << " " ; 
		if (!(pp % 10)) cout << endl; 
	      }
	      cout << endl;
	      if (!(d_adiabatic)) {
		cout << "Shouldn't be in nonadiabatic loop" << endl;
		if (tableIndex[0] >= minhIndex) {
		  convertKeyToFloatValues(tableIndex, indepVars);
		  if (indepVars[mixCount] < 0.0 )
		    indepVars[mixCount] = 0.0;
		  if (indepVars[mixCount] > 1.0 )
		    indepVars[mixCount] = 1.0;
		  mixVars[0] = indepVars[mixCount];
		  mixedStream = d_mixModel->speciesStateSpace(mixVars);
		  d_equilModel->computeRxnStateSpace(mixedStream, indepVars, reactedStream);
		  reactedStream.d_lsoot = true;
		  reactedStream.d_sootData[0] = 0.0;
		  reactedStream.d_sootData[1] = 0.0;
		  //vec_stateSpaceVars = reactedStream.convertStreamToVec();
		  //d_rxnTable->Insert(tableIndex, vec_stateSpaceVars);
		  d_rxnTable->Insert(tableIndex, stateSpaceVars);
		  tableInsert = true;
		  cout << "tableIndex = " << tableIndex[0] << " " << tableIndex[1] << " " 
		       << tableIndex[2] << endl;
		}
	      }
	      double richLimit = 0.7;
	      // If f outside tabulated range, return equilibrium values; if f greater than
	      // rich limit, return unreacted values
	      if (!(tableInsert)) {
		if ((tableIndex[mixCount] < minfIndex)||(tableIndex[mixCount] > maxfIndex))
		  {
		    convertKeyToFloatValues(tableIndex, indepVars);
		    if (indepVars[mixCount] < 0.0 )
		      indepVars[mixCount] = 0.0;
		    if (indepVars[mixCount] > 1.0 )
		      indepVars[mixCount] = 1.0;
		    mixVars[0] = indepVars[mixCount];
		    mixedStream = d_mixModel->speciesStateSpace(mixVars);
		 //   mixedStream.print(cout);
		    cout << "indepVars = " << indepVars[0] << " " << indepVars[1] << endl;
		    if (indepVars[mixCount] < richLimit) {
		      d_equilModel->computeRxnStateSpace(mixedStream, indepVars, reactedStream);
		      reactedStream.d_lsoot = true; 
		      reactedStream.d_sootData[0] = 0.0;
		      reactedStream.d_sootData[1] = 0.0;
		      //vec_stateSpaceVars = reactedStream.convertStreamToVec();
		      //d_rxnTable->Insert(tableIndex, vec_stateSpaceVars);
		      d_rxnTable->Insert(tableIndex, stateSpaceVars);
		      cout << "tableIndex = " << tableIndex[0] << " " << tableIndex[1] << " " 
			   << tableIndex[2] << endl;
		      tableInsert = true;
		    }
		    else if (indepVars[mixCount] >= richLimit) {
		      reactedStream = mixedStream;
		      // Since only 4 species included in table, need to modify reactedStream
		      reactedStream.d_depStateSpaceVars = d_depStateSpaceVars;
		      reactedStream.d_speciesConcn = vector<double> (4);
		      reactedStream.d_speciesConcn[0] = mixedStream.d_speciesConcn[CO2index];
		      reactedStream.d_speciesConcn[1] = mixedStream.d_speciesConcn[H2Oindex];
		      reactedStream.d_speciesConcn[2] = mixedStream.d_speciesConcn[COindex];
		      reactedStream.d_speciesConcn[3] = mixedStream.d_speciesConcn[O2index];
		      //vec_stateSpaceVars = reactedStream.convertStreamToVec();
		      //d_rxnTable->Insert(tableIndex, vec_stateSpaceVars);
		      d_rxnTable->Insert(tableIndex, stateSpaceVars);
		      cout << "tableIndex = " << tableIndex[0] << " " << tableIndex[1] << " " 
			   << tableIndex[2] << endl;
		      tableInsert = true;
		    }
		    else {
		      d_equilModel->computeRxnStateSpace(mixedStream, indepVars, reactedStream);
		      reactedStream.d_lsoot = true;
		      reactedStream.d_sootData[0] = 0.0;
		      reactedStream.d_sootData[1] = 0.0;
		      //vec_stateSpaceVars = reactedStream.convertStreamToVec();
		      //d_rxnTable->Insert(tableIndex, vec_stateSpaceVars);
		      d_rxnTable->Insert(tableIndex, stateSpaceVars);
		      cout << "tableIndex = " << tableIndex[0] << " " << tableIndex[1] << " " 
			   << tableIndex[2] << endl;
		      tableInsert = true;
		    }
		  } // if ((tableIndex[mixCount] < minfIndex) 	
	      } // if (!(tableInsert)) 
	      if (tableInsert) {
		entryCount++;
		tableInsert = false;
	      }
	    } // for kk
	} // for jj
    } // for ii
 
  // Now read in Diem's data file
  ifstream ildmfile("ILDMData");  
  if (ildmfile.fail()) {
    cout<<"ERROR in ILDMReactionModel"<<endl
	<<"    Could not open ILDMData file."<<endl;
    exit(1);
  }
  //Look for flag, then read vector data
  char flagKey[6];
  double value = 0.0;
  double minParamValue = 0.0;
  double maxParamValue = 0.0;
  while(ildmfile)
    { 
      //Look for ILDM table input until the end of the file is reached.
      ildmfile.getline(flagKey, 6);
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
	      //int vecCount = 0;
	      entryCount++;
	      // First two entries are unnormalized value of parameter and entropy
	      ildmfile>>rxnParam>>ignore;
	      ildmfile>>value; //Read pressure in atm
	      double temp = value*101325; //Convert pressure to Pa
	      ildmfile>>value; //Read density in g/cm^3
#if 0
	      vec_stateSpaceVars[vecCount++] = value*1.e+3;// Units of kg/m^3
	      vec_stateSpaceVars[vecCount++] = temp; //Pressure, density switched places
	      ildmfile>>value; //Read in temperature in K
	      vec_stateSpaceVars[vecCount++] = value;
	      ildmfile>>value; //Read in enthalpy in ergs/g
	      vec_stateSpaceVars[vecCount++] = value*1.e-4; // Units of J/kg
#endif
	      stateSpaceVars.d_density = value*1.e+3; // Units of kg/m^3
	      stateSpaceVars.d_pressure = temp;
	      ildmfile>>value; //Read in temperature in K
	      stateSpaceVars.d_temperature = value;
	      ildmfile>>value; //Read in enthalpy in ergs/g
	      stateSpaceVars.d_enthalpy = value*1.e-4; // Units of J/kg
	      
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
		Stream sensStream = d_mixModel->speciesStateSpace(mixVar);
		double adiabaticEnthalpy = sensStream.getEnthalpy();
		vector<double> initMassFract(d_reactionData->getNumSpecies());
		if (sensStream.getMoleBool()) 
		  initMassFract = d_reactionData->convertMolestoMass
		    (sensStream.d_speciesConcn);
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
#if 0	      
	      vec_stateSpaceVars[vecCount++] = sensibleEnthalpy;
	      ildmfile>>value; //Read in mix MW
	      vec_stateSpaceVars[vecCount++] = value;
	      ildmfile>>value; //Read in mix heat capacity in ergs/(g-K)
	      vec_stateSpaceVars[vecCount++] = value/1e+7*1000; //Convert from
	      // erg/(gm K) to J/(kg K)
	      //vec_stateSpaceVars[vecCount++] = 1.0; // Test for BetaPDF model
	      vec_stateSpaceVars[vecCount++] = 0.0; // Placeholder for d_drhodf
	      vec_stateSpaceVars[vecCount++] = 0.0; // Placeholder for d_drhodh
#endif
	      stateSpaceVars.d_sensibleEnthalpy = sensibleEnthalpy;
	      ildmfile>>value; //Read in mix MW
	      stateSpaceVars.d_moleWeight = value;
	      ildmfile>>value; //Read in mix heat capacity in ergs/(g-K)
	      stateSpaceVars.d_cp = value/1e+7*1000; //Convert from
	      // erg/(gm K) to J/(kg K)
	      stateSpaceVars.d_drhodf = 0.0; // Placeholder for d_drhodf
	      stateSpaceVars.d_drhodh = 0.0; // Placeholder for d_drhodh      
	      // Read in species mass fractions
              //Kluge to write only 4 species to table: CO2, H2O, O2, CO
	      ildmfile >> value;
	      stateSpaceVars.d_speciesConcn[0] = value; //CO2
              ildmfile >> value;
              stateSpaceVars.d_speciesConcn[1] = value; //H2O
	      ildmfile >> value;
              stateSpaceVars.d_speciesConcn[2] = value; //O2
	      ildmfile >> value;
              stateSpaceVars.d_speciesConcn[3] = value; //CO
              
	      // Not all species in chem.inp are included in data file;
	      // set those species conc to zero.
	      //for (int jj =  NUM_DEP_VARS + d_numSpecInTbl; jj < NUM_DEP_VARS 
		 //    + d_reactionData->getNumSpecies(); jj++)
		//vec_stateSpaceVars[vecCount++] = 0.0; 
	      // Read in soot data
	      double sootDiam, sootFV;
	      ildmfile>>sootDiam>>sootFV;
	      // Read in rxn rate of parameter
	      ildmfile>>value;
	      //vec_stateSpaceVars[vecCount++] = value;
	      stateSpaceVars.d_rxnVarRates[0] = value;
	      // Now assign min, max values of parameter to proper vector location
	      //vec_stateSpaceVars[vecCount++] = minParamValue;
	      //vec_stateSpaceVars[vecCount++] = maxParamValue;
	      stateSpaceVars.d_rxnVarNorm[0] = minParamValue;
	      stateSpaceVars.d_rxnVarNorm[1] = maxParamValue;
	      // Assign soot data to end of vector
	      //vec_stateSpaceVars[vecCount++] = sootDiam;
	      //vec_stateSpaceVars[vecCount++] = sootFV;
	      stateSpaceVars.d_sootData[0] = sootDiam;
	      stateSpaceVars.d_sootData[1] = sootFV;
	      indepVars[rxnCount] = (rxnParam - minParamValue)/
		(maxParamValue - minParamValue);
#if 0
	      for (int kk = 0; kk < vec_stateSpaceVars.size(); kk++) {
	      	cout.width(10);
	      	cout << vec_stateSpaceVars[kk] << " " ; 
	      	if (!(kk % 10)) cout << endl; 
	      }
	      cout << endl;
	      stateSpaceVars.print(cerr);
#endif
	      ildmfile.ignore(200,'\n');    //Move to next line
	      
	      //Convert indepVars to tableKeyIndex
	      double tableValue;
	      for (int i = 0; i < d_rxnTableDimension; i++)
		{
		  // calculates index in the table
		  double midPt = d_rxnTableInfo->getStoicValue(i);
		  if (indepVars[i] <= midPt){
		    tableValue = (indepVars[i] - d_rxnTableInfo->getMinValue(i))/  
		      d_rxnTableInfo->getIncrValueBelow(i);
                    if (i==0)
                      cout << "tableValue = " << tableValue << endl;
		  }
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
	      //d_rxnTable->Insert(tableIndex, vec_stateSpaceVars);
	      d_rxnTable->Insert(tableIndex, stateSpaceVars);
	      cout << "ILDMData table index = " << tableIndex[0] << " " << 
		tableIndex[1] << " " << tableIndex[2] << endl;
	      //delete [] tableIndex;
	    } // for(i = 0 to total number of lines under ENTRY 
	} //if (strcmp(flagKey, "ENTRY") == 0)
    } //while(ildmfile)
  ildmfile.close();
  delete [] tableIndex;
  
  //Check to see if number of entries in datafile match the size of table 
  //specified above
  cerr << "entryCount = " << entryCount << " totalEntries = " << totalEntries << endl;
  ASSERT(entryCount==totalEntries);
}


//****************************************************************************
// Read manifold data from kdtree here
//****************************************************************************
void
ILDMReactionModel::getRxnStateSpace(const Stream& unreactedMixture, 
				    vector<double>& varsHFPi, Stream& reactedStream)
{

    getProps(varsHFPi, reactedStream); //function in DynamicTable
    return;
}


void
ILDMReactionModel::tableLookUp(int* tableKeyIndex, Stream& stateSpaceVars) 
{
  //vector<double> vec_stateSpaceVars;
  //if (d_rxnTable->Lookup(tableKeyIndex, vec_stateSpaceVars)) 
  if (d_rxnTable->Lookup(tableKeyIndex, stateSpaceVars)) 
    {
      //stateSpaceVars.convertVecToStream(vec_stateSpaceVars, 
      //				d_numMixVars, d_numRxnVars, 
      //					d_lsoot);
      //stateSpaceVars.print(cout);   
    } 
   else
     {
       cout << "Table entry not found in ILDM::tableLookup" <<endl;
       exit(1);
     }  
}

void
ILDMReactionModel::convertKeyToFloatValues(int tableKeyIndex[], vector<double>& indepVars) {
  for (int i = 0; i < d_rxnTableDimension; i++)
    if (tableKeyIndex[i] <= d_rxnTableInfo->getNumDivsBelow(i))
      indepVars[i] = tableKeyIndex[i]*d_rxnTableInfo->getIncrValueBelow(i) + 
	d_rxnTableInfo->getMinValue(i);
    else
      indepVars[i] = (tableKeyIndex[i]-d_rxnTableInfo->getNumDivsBelow(i))*	
	d_rxnTableInfo->getIncrValueAbove(i) + d_rxnTableInfo->getStoicValue(i);
  return;
}

void
ILDMReactionModel::computeRxnStateSpace(const Stream& unreactedMixture, 
					const vector<double>& mixRxnVar, 
					Stream& equilStateSpace) {}

double
ILDMReactionModel::computeTemperature(const double absEnthalpy, 
				      const vector<double>& massFract, 
				      double initTemp) {} 




