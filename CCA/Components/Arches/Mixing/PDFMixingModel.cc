//----- PDFMixingModel.cc --------------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/Mixing/PDFMixingModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/InletStream.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/Integrator.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/KDTree.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/VectorTable.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/ReactionModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/StanjanEquilibriumReactionModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/ILDMReactionModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/ChemkinInterface.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/MixRxnTable.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/MixRxnTableInfo.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <Core/Math/MiscMath.h>

using namespace std;
using namespace Uintah;
using namespace SCIRun;
//****************************************************************************
// Default constructor for PDFMixingModel
//****************************************************************************
PDFMixingModel::PDFMixingModel():MixingModel(), DynamicTable()
{
}

//****************************************************************************
// Destructor
//****************************************************************************
PDFMixingModel::~PDFMixingModel()
{
}

//****************************************************************************
// Problem Setup for PDFMixingModel
//****************************************************************************
void 
PDFMixingModel::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP db = params->findBlock("PDFMixingModel");
  db->require("adiabatic",d_adiabatic);
  db->require("mixstatvars",d_numMixStatVars);
  db->require("rxnvars",d_numRxnVars);
  if (d_numMixStatVars < 1) {
    cout << "numMixStatVars must be > 0 for PDF mixing models" << endl;
    cout << "WARNING: Setting numMixStatVars = " << endl;
    d_numMixStatVars = 1;
  }
  db->require("PDFShape",d_pdfShape);
  if ((d_pdfShape != "Beta")&&(d_pdfShape != "ClippedGaussian")) {
    cout << "PDFShape is Beta" << endl;
    d_pdfShape = "Beta";
    //throw InvalidValue("PDF shape not implemented " + d_pdfShape);
  }
  // read and initialize reaction model with chemkin interface
  string rxnModel;
  db->require("reaction_model",rxnModel);
  if (rxnModel == "EquilibriumReactionModel")
    d_rxnModel = new StanjanEquilibriumReactionModel(d_adiabatic);
  else if (rxnModel == "ILDMReactionModel")
    d_rxnModel = new ILDMReactionModel(d_adiabatic);
  else {
    d_rxnModel = new StanjanEquilibriumReactionModel(d_adiabatic);
    cout << "REACTION MODEL is Equilibrium" << endl;
    //throw InvalidValue("Reaction Model not supported" + rxnModel);
  }
  ChemkinInterface* chemInterf = d_rxnModel->getChemkinInterface();
  int nofSpecies = chemInterf->getNumSpecies();
  int nofElements = chemInterf->getNumElements();
  // Read the mixing variable streams, total is nofstreams
  int nofstrm = 0;
  string speciesName;
  double mfrac; //mole or mass fraction
  for (ProblemSpecP stream_db = db->findBlock("Stream");
       stream_db != 0; stream_db = stream_db->findNextBlock("Stream")) {
    // Create the stream and add it to the vector
    d_streams.push_back(Stream(nofSpecies, nofElements));
    stream_db->require("pressure", d_streams[nofstrm].d_pressure);
    stream_db->require("temperature", d_streams[nofstrm].d_temperature);
    // mole fraction or mass fraction
    stream_db->require("mole",d_streams[nofstrm].d_mole);
    for (ProblemSpecP species_db = stream_db->findBlock("Species");
	 species_db !=0; species_db = species_db->findNextBlock("Species")) {
      species_db->require("symbol",speciesName);
      species_db->require("mfraction",mfrac);
      // convert c++ string to c string
      d_streams[nofstrm].addSpecies(d_rxnModel->getChemkinInterface(),
					    speciesName.c_str(), mfrac);
    }
    d_streams[nofstrm].normalizeStream(); // normalize sum to be 1
    vector<double> ymassVec;
    if (d_streams[nofstrm].getMoleBool()) {
      ymassVec = chemInterf->convertMolestoMass(
					  d_streams[nofstrm].d_speciesConcn);
      d_streams[nofstrm].d_mole = false;
      d_streams[nofstrm].d_speciesConcn = ymassVec;
    }
    else
      ymassVec = d_streams[nofstrm].d_speciesConcn;
    double strmTemp = d_streams[nofstrm].getTemperature();
    double strmPress = d_streams[nofstrm].getPressure();
    d_streams[nofstrm].d_density=chemInterf->getMassDensity(strmPress,
						     strmTemp, ymassVec);
    d_streams[nofstrm].d_enthalpy=chemInterf->getMixEnthalpy(strmTemp, ymassVec);
    d_streams[nofstrm].d_moleWeight=chemInterf->getMixMoleWeight(ymassVec);
    d_streams[nofstrm].d_cp=chemInterf->getMixSpecificHeat(strmTemp, ymassVec);
    // store as mass fraction
    //d_streams[nofstrm].print(cerr );
    ++nofstrm;
  }
  d_numMixingVars = nofstrm - 1;
  //cout << "PDFMixingModel::numMixVars = " << d_numMixingVars << endl;
  //cout << "PDFMixingModel::numMixStatVars = " << d_numMixStatVars << endl;
  //cout <<"PDF::numRxnVars = "<<d_numRxnVars<<endl;

  // Define MixingModel table, which includes calling reaction model constructor
  d_tableDimension = d_numMixingVars + d_numMixStatVars + d_numRxnVars + !(d_adiabatic);
  d_tableInfo = new MixRxnTableInfo(d_tableDimension);
  bool varFlag = true; //Table has variance
  d_tableInfo->problemSetup(db, varFlag, this);
  // Define table type (static or dynamic). Set up table storage, either as 
  // vectors or as kdtree. 
  if (db->findBlock("TableType")) {
    // d_tableType must be class variable because of inline function
    db->require("TableType", d_tableType);
    if (d_tableType == "dynamic") {
      d_dynamic = true;
      if (d_pdfShape == "ClippedGaussian") {
	cout << "Dynamic table for clipped gaussian PDF not implemented yet" << endl;
	cout << "TABLE TYPE is static" << endl;
	d_dynamic = false;
      }
    }
    else if (d_tableType == "static") {
      d_dynamic = false;
      if (d_pdfShape == "Beta") {
	cerr << "Static table for BetaPDF not implemented yet." << endl;
	cerr << "TABLE TYPE is dynamic" << endl;
	d_dynamic = true;
      }
    }
    else {
      throw InvalidValue("Table type not supported" + d_tableType);
    }
  }
  else {
    // Default tableType is dynamic
    d_dynamic = true;
    cout << "TABLE TYPE is dynamic" << endl;
  }
  // Call reaction model constructor, get total number of dependent  vars;
  // can't call it sooner because d_numMixingVars  and mixTableType are needed
  d_rxnModel->problemSetup(db, this); 
  d_depStateSpaceVars = d_rxnModel->getTotalDepVars();
  string tableStorage;
  if (db->findBlock("TableStorage")) {
    db->require("TableStorage", tableStorage);
    if (tableStorage == "KDTree")
      d_mixTable = new KD_Tree(d_tableDimension, d_depStateSpaceVars);
    else if (tableStorage == "2DVector")
      d_mixTable = new VectorTable(d_tableDimension, d_tableInfo);
    else {
      throw InvalidValue("Table storage not supported" + tableStorage);
    }
  }
  else {
    d_mixTable = new VectorTable(d_tableDimension, d_tableInfo);
    cout << "TABLE STORAGE is vectorTable" << endl;
  }
  // tableSetup is a function in DynamicTable; it allocates memory for 
  // table functions
  tableSetup(d_tableDimension, d_tableInfo);

  // If table is dynamic, need to call integrator constructor
  if (d_dynamic) {
    d_integrator = new Integrator(d_tableDimension, this, d_rxnModel, d_tableInfo);
    d_integrator->problemSetup(db);
  }
  else {
    // If table is static, read in table from data file
    readStaticTable();
  }

}

Stream
PDFMixingModel::speciesStateSpace(const vector<double>& mixVar) 
{
  ChemkinInterface* chemInterf = d_rxnModel->getChemkinInterface();
  int nofElements = chemInterf->getNumElements();
  int nofSpecies = chemInterf->getNumSpecies();
  bool lsoot = d_rxnModel->getSootBool();
  Stream mixedStream(nofSpecies,nofElements, d_numMixingVars, d_numRxnVars,
		     lsoot);
  // if adiabatic
  int count = 0;
  // store species as massfraction
  mixedStream.d_mole = false;
  double* sumMixVarFrac = new double[d_numMixingVars+1];
  double sum = 0.0;
  for (int ii = 0; ii < d_numMixingVars; ii ++) {
    sum += mixVar[ii];
    sumMixVarFrac[ii] = mixVar[ii];
  }
  sumMixVarFrac[d_numMixingVars] = 1.0 - sum;
  for (vector<Stream>::iterator iter = d_streams.begin();
       iter != d_streams.end(); ++iter) {
    Stream& inlstrm = *iter;
    mixedStream.addStream(inlstrm, chemInterf, sumMixVarFrac[count]);
    ++count;
  }
  --count; // nummixvars = numstreams -1
  ASSERT(count==d_numMixingVars);
  delete[] sumMixVarFrac;
  
  return mixedStream;

}
      

//****************************************************************************
// Actually compute the properties here
//****************************************************************************
void 
PDFMixingModel::computeProps(const InletStream& inStream,
			     Stream& outStream)
{
  // convert inStream to array
  std::vector<double> mixRxnVar(d_tableDimension); 
  std::vector<double> normVar(d_tableDimension);
  int count = 0;
  double absEnthalpy = 0.0;
  if (!(d_adiabatic)) {
    absEnthalpy = inStream.d_enthalpy;
    mixRxnVar[count] = 0.0;
    normVar[count] = 0.0;
    count ++;
  }
  for (int i = 0; i < d_numMixingVars; i++) {
    //Check to make sure mixing vars within proper range
    if (inStream.d_mixVars[i] < d_tableInfo->getMinValue(count)) {
      mixRxnVar[count] = d_tableInfo->getMinValue(count);
      normVar[count] = d_tableInfo->getMinValue(count);
    }
    else if (inStream.d_mixVars[i] > d_tableInfo->getMaxValue(count)) {
      mixRxnVar[count] = d_tableInfo->getMaxValue(count);
      normVar[count] = d_tableInfo->getMaxValue(count); 
    }
    else {
      mixRxnVar[count] = inStream.d_mixVars[i];
      normVar[count] = inStream.d_mixVars[i];
    }
    count++;
  }
  int statcount = count;
  for (int i = 0; i < d_numMixStatVars; i++) {
    mixRxnVar[count] = inStream.d_mixVarVariance[i];   
    normVar[count] = 0.0;
    count++;
  }
  int rxncount = count;
  for (int i = 0; i < d_numRxnVars;i++) {
    mixRxnVar[count] = inStream.d_rxnVars[i];
    normVar[count] = 0.0;
    count++;
  }
  // count and d_tableDimension should be equal
  ASSERT(count==d_tableDimension);
  // Normalize enthalpy
  if (!(d_adiabatic)) {
    Stream normStream;
    getProps(normVar, normStream);
    double adiabaticEnthalpy = normStream.getEnthalpy();
    double sensEnthalpy = normStream.getSensEnthalpy(); 
    double normEnthalpy;
    if (Abs(absEnthalpy) < 1e-10)
      normEnthalpy = 0.0;
    else 
      normEnthalpy = (absEnthalpy - adiabaticEnthalpy)/sensEnthalpy;
    if (normEnthalpy < d_tableInfo->getMinValue(0))
      normEnthalpy =  d_tableInfo->getMinValue(0);
    if (normEnthalpy > d_tableInfo->getMaxValue(0))
      normEnthalpy =  d_tableInfo->getMaxValue(0);
    mixRxnVar[0] = normEnthalpy;
    normVar[0] = normEnthalpy; //Need to normalize rxn variable next, so 
                               //normalized enthalpy must be known
  }
  //Normalize mixture fraction variance
  if (d_numMixStatVars > 0) {
    //Compute max gf; min value is 0.0
    double maxStatValue;
    for (int ii = 0; ii < d_numMixStatVars; ii++) {
      maxStatValue = inStream.d_mixVars[ii]*(1-inStream.d_mixVars[ii]);
      if (mixRxnVar[statcount+ii] < 0.0)
	mixRxnVar[statcount+ii] = 0.0;
      if (mixRxnVar[statcount+ii] > maxStatValue)
	mixRxnVar[statcount+ii] = maxStatValue;
      double normStatVar;
      if (maxStatValue < 1e-10)
	normStatVar = 0.0;
      else
	normStatVar = mixRxnVar[statcount+ii]/maxStatValue;
      mixRxnVar[statcount+ii] = normStatVar;
      normVar[statcount+ii] = normStatVar;
      //cout << "PDF:NormVar = " << normVar[statcount+ii] << endl;
    }
  }
  //Normalize reaction variables
  if (d_numRxnVars > 0) { 
    //Since min/max rxn parameter values for a given (h/f) combo are the 
    //same for every rxn parameter entry, look up the first entry; 
    Stream paramValues;
    for (int ii = 0; ii < d_numRxnVars; ii++) {
      getProps(normVar, paramValues);
      double minParamValue = paramValues.d_rxnVarNorm[0];
      double maxParamValue = paramValues.d_rxnVarNorm[1];
      if (mixRxnVar[rxncount+ii] < minParamValue)
	mixRxnVar[rxncount+ii] = minParamValue;
      if (mixRxnVar[rxncount+ii] > maxParamValue)
	mixRxnVar[rxncount+ii] = maxParamValue;
      double normParam;
      if ((maxParamValue - minParamValue) < 1e-10)
	normParam = 0.0;
      else
	normParam = (mixRxnVar[rxncount+ii] - minParamValue)/
	  (maxParamValue - minParamValue);
      mixRxnVar[rxncount+ii] = normParam;
      normVar[rxncount+ii] = normParam;
      //mixRxnVar[rxncount+ii] = 1.0; 
      //normVar[rxncount+ii] = 1.0; 
    }
  }
  getProps(mixRxnVar, outStream); //function in DynamicTable
  //outStream.print(cout);
#if 0
  cout << "PDF::getProps mixRxnVar = " << endl;
  for (int ii = 0; ii < mixRxnVar.size(); ii++) {
    cout.width(10);
    cout << mixRxnVar[ii] << " " ; 
    if (!(ii % 10)) cout << endl; 
  }
  cout << endl;
#endif
 
}


void
PDFMixingModel::tableLookUp(int* tableKeyIndex, Stream& stateSpaceVars) {
  vector<double> vec_stateSpaceVars;
  bool lsoot = d_rxnModel->getSootBool();
  bool flag = false;
#if 0
  cout << "PDF::tableKeyIndex = " << endl;
  for (int ii = 0; ii < d_tableDimension; ii++) {
    cout.width(10);
    cout << tableKeyIndex[ii] << " " ; 
    if (!(ii % 10)) cout << endl; 
  }
  cout << endl;
#endif
  if (d_dynamic) 
    {  //Table is dynamic
      if (!(d_mixTable->Lookup(tableKeyIndex, vec_stateSpaceVars))) 
	{
	  // Call to integrator
	  // Don't need "if (d_numMixStatVars)" because it is set to 1 in 
	  // problemSetup
	  stateSpaceVars = d_integrator->integrate(tableKeyIndex);
	  vec_stateSpaceVars = stateSpaceVars.convertStreamToVec();
	  // defined in K-D tree or 2D vector implementation
	  d_mixTable->Insert(tableKeyIndex, vec_stateSpaceVars);
	  //stateSpaceVars.print(cerr);
	}
      else {
	bool flag = false;
	stateSpaceVars.convertVecToStream(vec_stateSpaceVars, flag, d_numMixingVars,
					  d_numRxnVars, lsoot);
#if 0
	cout<<"PDF::entry exists"<<endl;
	for (int ii = 0; ii < vec_stateSpaceVars.size(); ii++) {
	  cout.width(10);
	  cout << vec_stateSpaceVars[ii] << " " ; 
	  if (!(ii % 10)) cout << endl; 
	}
	cout << endl;
#endif
      }
    }
  else 
    {  //Table is static
      if (d_mixTable->Lookup(tableKeyIndex, vec_stateSpaceVars)) 
	{
	  bool flag = false;
	  stateSpaceVars.convertVecToStream(vec_stateSpaceVars, flag, 
					    d_numMixingVars, d_numRxnVars, 
					    lsoot);
	  //stateSpaceVars.print(cout);	  
	} 
      else
	{
	  cout << "Static table entry not found in PDF::tableLookup" <<endl;
	  exit(1);
	}
    }
  
}

void
PDFMixingModel::readStaticTable() {
  // This function will read data files created by James' tableGenerator program
  // Data files being read in are set up as follows. The first line contains the 
  // number of f divisions, number of f variance divisions, number of species,
  // and df (f spacing in table). The first line of each entry has f and dg.
  // The second line has g, species mass fractions, mixture temperature (K), 
  // mixture density (kg/m^3), mixture enthalpy (J/kg), and mixture heat capacity 
  // (J/kg-K).

  ifstream mixfile("stateTable");  
  if (mixfile.fail()) {
    cout<<"ERROR in PDFMixingModel"<<endl
	<<"    Could not open stateTable file."<<endl;
    exit(1);
  }

  int dataCount = 0;
  int numMixDiv, numVarDiv;
  int numSpecies;
  vector<double> vec_stateSpaceVars(d_depStateSpaceVars, 0.0);
  vector<double> indepVars(d_tableDimension, 0.0);
  double value;
  // Read in header information
  // ***Put in check for numSpecies???***
  mixfile >> numMixDiv >> numVarDiv >> numSpecies>> value;
  mixfile.ignore(200,'\n');    //Move to next line
  mixfile.ignore(200,'\n');    //Move to next line
  //mixfile.ignore();    //Move to next line
  vector<double> speciesMassFract(numSpecies, 0.0);

  //Look for mixing table input until the end of the file is reached.
  //while(mixfile)
  for (int nn = 0; nn < numVarDiv; nn++)  	 
    { 
      // Read f,dg pair and then subsequent data. Number of entries = 
      // numvarDiv (***need to add h later***)
      int kk = 0;
#if 0
      mixfile>>value;
      if (!(d_adiabatic)) {
	indepVars[kk] = value;
	kk++;
      }
#endif
      for (int jj = 0; jj < d_numMixingVars; jj++) {
	mixfile>>value;
	indepVars[kk] = value;
	kk++;
      }
      double dg;
      mixfile >> dg;
      mixfile.ignore(50,'\n');    //Move to next line
      mixfile.ignore(50,'\n');    //Move to next line
      //mixfile.ignore();    //Move to next line
      //Read lines containing g, state space information
      //***Not set up to have more than one gf
      // gf must be normalized so it scales from 0-1; min value of g is 0.0, 
      // need to calculate max value
      double maxg = dg * (numVarDiv-1);
      bool gflag = false;
      double normg;
      int gcount = 0;
      for (int ii = 0; ii < numVarDiv; ii++)
	{
	  mixfile>>value;  
	  if (maxg == 0.0) {
	    normg = value;
	    gflag = true;
	  }
	  else
	    normg = value / maxg;
	  if (normg < 0.0) {
	    cout << "WARNING: Normalized g is less than 0  (=" << normg << ")" << endl;
	    cout << "Check problem with input file; setting it = 0.0" << endl;
	    normg = 0.0;
	  }
	  if (normg > 1.0) {
	    cout << "WARNING: Normalized g is greater than 1 (=" << normg << ")" << endl;
	    cout << "Setting it = 1.0" << endl;
	    normg = 1.0;
	  }
	  indepVars[kk] = normg;  
	  dataCount++; // Counter for number of total entries in data file
	  // Read in species mass fractions
	  for (int jj = 0; jj < numSpecies; jj++) {
	    mixfile>>value;
	    speciesMassFract[jj] = value;
	  }
	  int vecCount = 0; 
	  double temp;
	  // Ordering in output vector: pressure, density, temp, enthalpy, 
	  // sensh, cp, molwt, species mass fractions
	  vec_stateSpaceVars[vecCount++] = 1.0; //Pressure in atm
	  mixfile>>temp; //Read in temperature in K
	  mixfile>>value; //Read in density in kg/m^3
	  vec_stateSpaceVars[vecCount++] = value;
	  vec_stateSpaceVars[vecCount++] = temp;
	  mixfile>>value; //Read in enthalpy in J/kg
	  vec_stateSpaceVars[vecCount++] = value;
	  vec_stateSpaceVars[vecCount++] = 0.0; //Set sensible enthalpy to 0
	  vec_stateSpaceVars[vecCount++] = 0.0; //Set molecular weight to 0
	  mixfile>>value; //Read in mix heat capacity in J/(kg-K)
	  vec_stateSpaceVars[vecCount++] = value;
	  vec_stateSpaceVars[vecCount++]= 0; //drhodf place holder
	  vec_stateSpaceVars[vecCount++] = 0; //drhodh place holder
	  // Assign species mass fractions to vector
	  for (int jj = 0; jj < numSpecies; jj++) {
	    vec_stateSpaceVars[vecCount++] = speciesMassFract[jj];
	    // ***Can I do this in one line???
	  }
	  mixfile.ignore(50,'\n');    //Move to next line
  
	  //Convert indepVars to tableKeyIndex, but first check table dimensions in stateTable
	  //file against dimensions in input file. ***This will only work for one f- I do check
	  //at end that may be adequate
	  if ((d_tableInfo->getNumDivsBelow(0)+d_tableInfo->getNumDivsAbove(0)+1) != 
	      numMixDiv) {
	    cout << "WARNING: Number of f entries in stateTable does not match table size";
	    cout << " specified in input file" << endl;
	    exit(1);
	  }
	  if ((d_tableInfo->getNumDivsBelow(1)+d_tableInfo->getNumDivsAbove(1)+1) != 
	      numVarDiv) {
	    cout << "WARNING: Number of g entries in stateTable does not match table size";
	    cout << " specified in input file" << endl;
	    exit(1);
	  }
	  int* tableIndex = new int[d_tableDimension];//??+1??
	  double tableValue;
	  for (int ll = 0; ll < d_tableDimension; ll++)
	    {
	      // calculates index in the table
	      double midPt = d_tableInfo->getStoicValue(ll);
	      if (indepVars[ll] <= midPt) 
		tableValue = (indepVars[ll] - d_tableInfo->getMinValue(ll))/  
		  d_tableInfo->getIncrValueBelow(ll);
	      else
		tableValue = ((indepVars[ll] - midPt)/d_tableInfo->getIncrValueAbove(ll))
		  + d_tableInfo->getNumDivsBelow(ll);
	      tableValue = tableValue + 0.5;
	      tableIndex[ll] = (int) tableValue; // cast to int
	      if (tableIndex[ll] > (d_tableInfo->getNumDivsBelow(ll)+
				    d_tableInfo->getNumDivsAbove(ll))||
		  (tableIndex[ll] < 0))	
		cerr<<"Index value out of range in MixingTable"<<endl;
	      //If max value for gf=0 (gflag=true) for a set of table
	      //entries, i.e. f=0 or 1, table indeces won't increment. 
	      //Force them to increment here 
	      //***Warning-this only works for one f and one gf***
	      if ((ll == 1)&&(gflag)) {
		tableIndex[ll] = gcount;
		gcount++;
		gflag = false;
	      }		   
	    }
	    d_mixTable->Insert(tableIndex, vec_stateSpaceVars);
	    delete [] tableIndex;
	} // for(ii = 0 to numVarDiv)
    } //while(mixfile) - end of for loop
  mixfile.close();

  //Check to see if number of entries in datafile match the specified size of table 
  //Compute total number of entries that should be in table 
  int totalEntries = 1; //Don't want to multiply by zero
  for (int ii = 0; ii < d_tableDimension; ii++)
    totalEntries *= d_tableInfo->getNumDivsBelow(ii) + d_tableInfo->getNumDivsAbove(ii) + 1;
  cerr << "dataCount = " << dataCount << " totalEntries = " << totalEntries << endl;
  ASSERT(dataCount==totalEntries);
}

