//----- PDFMixingModel.cc --------------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/Mixing/PDFMixingModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/BetaPDFShape.h>
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
  if (db->findBlock("PDFShape")) {
    db->require("PDFShape",d_pdfShape);
    if ((d_pdfShape != "Beta")&&(d_pdfShape != "ClippedGaussian")) {
      cout << "PDFShape is Beta" << endl;
      d_pdfShape = "Beta";
      //throw InvalidValue("PDF shape not implemented " + d_pdfShape);
    }
  }
  else {
    cout << "PDFShape is Beta" << endl;
    d_pdfShape = "Beta";
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
  if (db->findBlock("StoicValue")) {
    db->require("StoicValue", d_stoicValue);
  }
  else {
    d_stoicValue = 0.05516706; //Default is stoichiometric mixture 
                               // fraction for methane
    cout << "Using stoichiometric value for methane" << endl;
  }
  //double fstoic = 0.05516706 ; // Kluge to run methane
  //fstoic = 6.218489e-02; // Kluge to run heptane
  d_chemInterf = d_rxnModel->getChemkinInterface();
  int nofSpecies = d_chemInterf->getNumSpecies();
  int nofElements = d_chemInterf->getNumElements();
  //d_CO2index = d_chemInterf->getSpeciesIndex("CO2");
  //d_H2Oindex = d_chemInterf->getSpeciesIndex("H2O");
  d_CO2index = 0;
  d_H2Oindex = 1; // Four species output: CO2, H2O, O2, CO
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
      ymassVec = d_chemInterf->convertMolestoMass(
					  d_streams[nofstrm].d_speciesConcn);
      d_streams[nofstrm].d_mole = false;
      d_streams[nofstrm].d_speciesConcn = ymassVec;
    }
    else
      ymassVec = d_streams[nofstrm].d_speciesConcn;
    double strmTemp = d_streams[nofstrm].getTemperature();
    double strmPress = d_streams[nofstrm].getPressure();
    d_streams[nofstrm].d_density=d_chemInterf->getMassDensity(strmPress,
						     strmTemp, ymassVec);
    d_streams[nofstrm].d_enthalpy=d_chemInterf->getMixEnthalpy(strmTemp, ymassVec);
    d_streams[nofstrm].d_moleWeight=d_chemInterf->getMixMoleWeight(ymassVec);
    d_streams[nofstrm].d_cp=d_chemInterf->getMixSpecificHeat(strmTemp, ymassVec);
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
    }
    else {
      // Default tableType is dynamic
      d_dynamic = true;
      cout << "PDFMixing TABLE TYPE is dynamic" << endl;
    }
  }
  else {
    // Default tableType is dynamic
    d_dynamic = true;
    cout << "PDFMixing TABLE TYPE is dynamic" << endl;
  }
  // Call reaction model constructor, get total number of dependent  vars;
  // can't call it sooner because d_numMixingVars  and mixTableType are needed
  d_rxnModel->problemSetup(db, this); 
  d_depStateSpaceVars = d_rxnModel->getTotalDepVars();
  //cout << "totalDepVars = " << d_depStateSpaceVars << endl;
  string tableStorage;
  if (db->findBlock("TableStorage")) {
    db->require("TableStorage", tableStorage);
    if (tableStorage == "KDTree")
      d_mixTable = new KD_Tree(d_tableDimension, d_depStateSpaceVars);
    else if (tableStorage == "2DVector")
      d_mixTable = new VectorTable(d_tableDimension, d_tableInfo);
    else {
      d_mixTable = new VectorTable(d_tableDimension, d_tableInfo);
      cout << "PDFMixing TABLE STORAGE is vector table" << endl;
      //throw InvalidValue("Table storage not supported" + tableStorage);
    }
  }
  else {
    d_mixTable = new VectorTable(d_tableDimension, d_tableInfo);
    cout << "PDFMixing TABLE STORAGE is vectorTable" << endl;
  }
  // tableSetup is a function in DynamicTable; it allocates memory for 
  // table functions
  tableSetup(d_tableDimension, d_tableInfo);

  // If table is dynamic, need to call integrator constructor, create table
  if (d_dynamic) {
    d_integrator = new Integrator(d_tableDimension, this, d_rxnModel, d_tableInfo);
    d_integrator->problemSetup(db);
    createBetaTable(); // Nullifies concept of dynamic table, but let's see if this works!!!
    d_dynamic = false;
  }
  else {
    // If table is static, read in table from data file
    if ( d_pdfShape == "Beta")
      readBetaTable();
    else
      readStaticTable();
  }
}

Stream
PDFMixingModel::speciesStateSpace(const vector<double>& mixVar) 
{
  int nofElements = d_chemInterf->getNumElements();
  int nofSpecies = d_chemInterf->getNumSpecies();
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
    mixedStream.addStream(inlstrm, d_chemInterf, sumMixVarFrac[count]);
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
    if (inStream.d_initEnthalpy)
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
      //maxStatValue = inStream.d_mixVars[ii]*(1-inStream.d_mixVars[ii]);
      maxStatValue = 0.7*inStream.d_mixVars[ii]*(1-inStream.d_mixVars[ii]);
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
  outStream.d_co2 = outStream.d_speciesConcn[d_CO2index]; //Needed for radiation model
  outStream.d_h2o = outStream.d_speciesConcn[d_H2Oindex]; //Needed for radiation model
  //cout << "Here are the results" << endl; 
  // outStream.print(cout);
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
  //vector<double> vec_stateSpaceVars;
  bool lsoot = d_rxnModel->getSootBool();
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
      //if (!(d_mixTable->Lookup(tableKeyIndex, vec_stateSpaceVars)))
      if (!(d_mixTable->Lookup(tableKeyIndex, stateSpaceVars)))
        {
          // Call to integrator
          // Don't need "if (d_numMixStatVars)" because it is set to 1 in 
          // problemSetup
          stateSpaceVars = d_integrator->integrate(tableKeyIndex);
          //vec_stateSpaceVars = stateSpaceVars.convertStreamToVec();
          // defined in K-D tree or 2D vector implementation
          //d_mixTable->Insert(tableKeyIndex, vec_stateSpaceVars);
	  d_mixTable->Insert(tableKeyIndex, stateSpaceVars);
          //stateSpaceVars.print(cerr);
        }
      else {
        //stateSpaceVars.convertVecToStream(vec_stateSpaceVars, d_numMixingVars,
	//                                       d_numRxnVars, lsoot);
#if 0
        cout<<"PDF::entry exists"<<endl;
        for (int ii = 0; ii < vec_stateSpaceVars.size(); ii++) {
          cout.width(10);
          cout << vec_stateSpaceVars[ii] << " " ; 
          if (!(ii % 10)) cout << endl; 
        }
        cout << endl;
	stateSpaceVars.print(cerr);
#endif
      }
    }
  else 
    {  //Table is static
      //if (d_mixTable->Lookup(tableKeyIndex, vec_stateSpaceVars)) 
      if (d_mixTable->Lookup(tableKeyIndex, stateSpaceVars))
        {
          //stateSpaceVars.convertVecToStream(vec_stateSpaceVars, d_numMixingVars, 
	  //                                   d_numRxnVars, lsoot);
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
PDFMixingModel::convertKeyToFloatValues(int tableKeyIndex[], vector<double>& indepVars) {
  for (int i = 0; i < d_tableDimension; i++)
    if (tableKeyIndex[i] <= d_tableInfo->getNumDivsBelow(i))
      indepVars[i] = tableKeyIndex[i]*d_tableInfo->getIncrValueBelow(i) + 
	d_tableInfo->getMinValue(i);
    else
      indepVars[i] = (tableKeyIndex[i]-d_tableInfo->getNumDivsBelow(i))*	
	d_tableInfo->getIncrValueAbove(i) + d_tableInfo->getStoicValue(i);
  return;
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
  //vector<double> vec_stateSpaceVars(d_depStateSpaceVars, 0.0);
  int nofElements = d_chemInterf->getNumElements();
  //int nofSpecies = d_chemInterf->getNumSpecies();
  int nofSpecies = 4; //Only CO2,H2O,O2,CO in output vector
  bool lsoot = d_rxnModel->getSootBool();
  Stream stateSpaceVars(nofSpecies,nofElements, d_numMixingVars, d_numRxnVars,
		     lsoot);
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
  for (int nn = 0; nn < numMixDiv; nn++)  	 
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
	indepVars[jj] = value;
	jj++;
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
	  //int vecCount = 0; 
	  double temp;
	  // Ordering in output vector: density, pressure, temp, enthalpy, 
	  // sensh, cp, molwt, species mass fractions
	  mixfile>>temp; //Read in temperature in K
	  mixfile>>value; //Read in density in kg/m^3
#if 0
	  vec_stateSpaceVars[vecCount++] = value;
	  vec_stateSpaceVars[vecCount++] = 1.0; //Pressure in atm
	  vec_stateSpaceVars[vecCount++] = temp;
	  mixfile>>value; //Read in enthalpy in J/kg
	  vec_stateSpaceVars[vecCount++] = value;
	  vec_stateSpaceVars[vecCount++] = 0.0; //Set sensible enthalpy to 0
	  vec_stateSpaceVars[vecCount++] = 0.0; //Set molecular weight to 0
	  mixfile>>value; //Read in mix heat capacity in J/(kg-K)
	  vec_stateSpaceVars[vecCount++] = value;
	  vec_stateSpaceVars[vecCount++]= 0; //drhodf place holder
	  vec_stateSpaceVars[vecCount++] = 0; //drhodh place holder
#endif
	  stateSpaceVars.d_density = value;
	  stateSpaceVars.d_pressure = 1.0; //Pressure in atm
	  stateSpaceVars.d_temperature = temp;
	  mixfile>>value; //Read in enthalpy in J/kg
	  stateSpaceVars.d_enthalpy = value;
	  stateSpaceVars.d_sensibleEnthalpy = 0.0; //Set sensible enthalpy to 0
	  stateSpaceVars.d_moleWeight = 0.0; //Set molecular weight to 0
	  mixfile>>value; //Read in mix heat capacity in J/(kg-K)
	  stateSpaceVars.d_cp = value;
	  stateSpaceVars.d_drhodf= 0; //drhodf place holder
	  stateSpaceVars.d_drhodh = 0; //drhodh place holder
	  stateSpaceVars.d_speciesConcn = speciesMassFract;  // Species mass fractions
      
	  mixfile.ignore(50,'\n');    //Move to next line
  
	  //Convert indepVars to tableKeyIndex, but first check table dimensions in 
	  //stateTable file against dimensions in input file. ***This will only work 
	  //for one f- I do check at end that may be adequate
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
	  //d_mixTable->Insert(tableIndex, vec_stateSpaceVars);
	  d_mixTable->Insert(tableIndex, stateSpaceVars);
	    delete [] tableIndex;
	} // for(ii = 0 to numVarDiv)
    } // for(nn = 0 to numMixDiv)
  mixfile.close();

  //Check to see if number of entries in datafile match the specified size of table 
  //Compute total number of entries that should be in table 
  int totalEntries = 1; //Don't want to multiply by zero
  for (int ii = 0; ii < d_tableDimension; ii++)
    totalEntries *= d_tableInfo->getNumDivsBelow(ii) + d_tableInfo->getNumDivsAbove(ii) + 1;
  cerr << "dataCount = " << dataCount << " totalEntries = " << totalEntries << endl;
  ASSERT(dataCount==totalEntries);
}

void
PDFMixingModel::readBetaTable() {
  // This function will read data files created by the class function createBetaTable.
  // Data files being read in are set up as follows. The first line contains the 
  // number of table entries for each independent variable in the following order:
  // h, f, gf, pi. Depending on the case being run, some variables might not be used.
  // All subsequent lines contain table entries and are written as pairs. The first  
  // line contains: values of all independent variables for that entry (h, f, gf, pi)
  // The second line contains: mixture density (kg/m^3), pressure (),  mixture 
  // temperature (K), mixture enthalpy (J/kg), mixture sensible enthalpy (J/kg), 
  // mixture mol weight, mixture heat capacity (J/kg-K), drhodf, drhodh, species mass 
  // fractions for CO2, H2O, O2, and CO, source terms for rxn variables, soot volume 
  // fraction and soot diameter. Source terms are present only if numrxnvars > 0
  // and soot data if lsoot = true.

  ifstream mixfile("betaTable");  
  if (mixfile.fail()) {
    cout<<"ERROR in PDFMixingModel"<<endl
	<<"    Could not open betaTable file."<<endl;
    exit(1);
  }

  int dataCount = 0;
  int numHDiv, numMixDiv, numVarDiv, numPiDiv;
  //vector<double> vec_stateSpaceVars(d_depStateSpaceVars, 0.0);
  int nofElements = d_chemInterf->getNumElements();
  //int nofSpecies = d_chemInterf->getNumSpecies();
  int nofSpecies = 4; //Only CO2,H2O,O2,CO in output vector
  bool lsoot = d_rxnModel->getSootBool();
  Stream stateSpaceVars(nofSpecies,nofElements, d_numMixingVars, d_numRxnVars,
		     lsoot);
  vector<double> indepVars(d_tableDimension, 0.0);
  double value;
  int mixCount = 0;
  int varCount;
  int gcount = 0;

  // Read in header information
  if (!d_adiabatic) {
    mixfile >> numHDiv;
    mixCount++;
  }
  varCount = mixCount+1;
  mixfile >> numMixDiv >> numVarDiv;
  if (d_numRxnVars > 0)
    mixfile >> numPiDiv;
  mixfile.ignore(200,'\n');    //Move to next line
  mixfile.ignore(200,'\n');    //Move to next line
  vector<double> speciesMassFract(4, 0.0);

  // Reset gcount if necessary
  if (gcount >= numVarDiv)
    gcount = 0;
  int totalEntries = 1; //Don't want to multiply by zero
  for (int ii = 0; ii < d_tableDimension; ii++)
    totalEntries *= d_tableInfo->getNumDivsBelow(ii) + d_tableInfo->getNumDivsAbove(ii) + 1;
  //Look for mixing table input until the end of the file is reached.
  for (int nn = 0; nn < totalEntries; nn++)  	 
    { 
      // Read values for independent variables
      for (int jj = 0; jj < d_tableDimension; jj++) {
	mixfile>>value;
	indepVars[jj] = value;
      }
      bool gflag = false;
      if ((indepVars[mixCount] < CLOSETOZERO)||(indepVars[mixCount] > CLOSETOONE))
	gflag = true;
      mixfile.ignore(50,'\n');    //Move to next line
      // Read line containing state space information
      // gf is already normalized.
      //for (int kk = 0; kk < d_depStateSpaceVars; kk++) {
      //mixfile >> value;
      //vec_stateSpaceVars[kk] = value;
      mixfile >> value;
      stateSpaceVars.d_density = value;
      mixfile >> value;
      stateSpaceVars.d_pressure = value;
      mixfile >> value;
      stateSpaceVars.d_temperature = value;
      mixfile >> value;
      stateSpaceVars.d_enthalpy = value;
      mixfile >> value;
      stateSpaceVars.d_sensibleEnthalpy = value;
      mixfile >> value;
      stateSpaceVars.d_moleWeight = value;
      mixfile >> value;
      stateSpaceVars.d_cp = value;
      mixfile >> value;
      stateSpaceVars.d_drhodf = value;
      mixfile >> value;
      stateSpaceVars.d_drhodh = value;
      for (unsigned int iter = 0; 
	   iter < stateSpaceVars.d_speciesConcn.size(); ++iter) {
	mixfile >> value;
	stateSpaceVars.d_speciesConcn[iter] = value;
      }
      if (d_numRxnVars > 0) {
	for (unsigned int iter = 0; 
	     iter < stateSpaceVars.d_rxnVarRates.size(); ++iter) {
	  mixfile >> value;
	  stateSpaceVars.d_rxnVarRates[iter] = value;
	}
	for (unsigned int iter = 0; 
	     iter < stateSpaceVars.d_rxnVarNorm.size(); ++iter){
	  mixfile >> value;
	  stateSpaceVars.d_rxnVarNorm[iter] = value;
	}
      }
      if (lsoot) {
	for (unsigned int iter = 0; 
	     iter < stateSpaceVars.d_sootData.size(); ++iter) {
	  mixfile >> value;
	  stateSpaceVars.d_sootData[iter] = value;
	}
      }
      mixfile.ignore(50,'\n');    //Move to next line
      dataCount++; // Counter for number of total entries in data file
      //cout << "dataCount = " << dataCount << endl;
  
      //Convert indepVars to tableKeyIndex
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
	  if ((ll == varCount)&&(gflag)) {
	    tableIndex[ll] = gcount;
	    gcount++;
	    if (gcount == numVarDiv)
	      gcount = 0;
	    gflag = false;
	  }		   
	} // for (int ll)
      //cout << "Indeces = " << tableIndex[0] << " " << tableIndex[1] << " " << tableIndex[2]<< endl;
      //d_mixTable->Insert(tableIndex, vec_stateSpaceVars);
      d_mixTable->Insert(tableIndex, stateSpaceVars); 
      delete [] tableIndex;
    } // for(nn = 0 to totalEntries)
  mixfile.close();

  //Check to see if number of entries in datafile match the specified size of table 
  cerr << "dataCount = " << dataCount << " totalEntries = " << totalEntries << endl;
  ASSERT(dataCount==totalEntries);
}

void
PDFMixingModel::createBetaTable() {
  // This function creates a mixing model table using the beta PDF model. Assumption
  // is that only one mixture fraction and one variance are used.
  // The table is written out to the file "betaTable"

  //vector<double> vec_stateSpaceVars;
  int *tableIndex = new int[d_tableDimension];
  vector<int> numEntries(d_tableDimension, 0);
  Stream stateSpaceVars, prevStateSpaceVars;
  vector<double> indepVars(d_tableDimension, 0.0);
  int entryCount = 0;
  int count = 0;

  ofstream betafile("betaTable");
  //ofstream comparefile("compareTable");
  if (betafile.fail()) {
    cout<<"ERROR in PDFMixingModel"<<endl
	<<"    Could not open betaTable file."<<endl;
    exit(1);
  }
  if (!(d_adiabatic)) {
    numEntries[count] = d_tableInfo->getNumDivsBelow(count) + 
      d_tableInfo->getNumDivsAbove(count) + 1;
    betafile << numEntries[count] << " ";
    //cout << "d_adiabatic num entries = " << numEntries[count] << endl;
    count++;
  }
  int mixCount = count;
  numEntries[count] = d_tableInfo->getNumDivsBelow(count) + 
    d_tableInfo->getNumDivsAbove(count) + 1; //Number of mixture fraction entries
  //cout << "mix num entries = " << numEntries[count] << endl;
  betafile << numEntries[count] << " ";
  count++;
  int varCount = count;
  numEntries[count] = d_tableInfo->getNumDivsBelow(count) + 
    d_tableInfo->getNumDivsAbove(count) + 1; // Number of variance entries
  betafile << numEntries[count] << " ";
  //cout << "var num entries = " << numEntries[count] << endl;
  count++;
  int rxnCount = 0;
  if (d_numRxnVars > 0) {
    rxnCount = count;
    numEntries[count] = d_tableInfo->getNumDivsBelow(count) + 
      d_tableInfo->getNumDivsAbove(count) + 1;
    betafile << numEntries[count] << " ";
    //cout << "rxn num entries = " << numEntries[count] << endl;
  }
  betafile << endl;
  betafile << endl;

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
	  double interpVar = VARLIMIT;
	  for (int kk = 0; kk < numEntries[varCount]; kk++) 
	    {
	      tableIndex[varCount] = kk;
	      int llmax;
	      if (d_numRxnVars > 0)
		llmax = numEntries[rxnCount];
	      else
		llmax = 1;
	      for (int ll = 0; ll < llmax; ll++)
		{
		  if (d_numRxnVars > 0)
		    tableIndex[rxnCount] = ll;
		  cout << "createTable::tableIndex = " << endl;
		  for (int pp = 0; pp < d_tableDimension; pp++) {
		    cout.width(10);
		    cout << tableIndex[pp] << " " ; 
		    if (!(pp % 10)) cout << endl; 
		  }
		  cout << endl;
		  convertKeyToFloatValues(tableIndex, indepVars);
		  if (indepVars[varCount] > VARLIMIT) {
		    // interpolate between double delta functions and the last 
		    // integrated variance
		    vector<double> mixVars(d_numMixingVars, 0.0);
		    Stream unreactedStream;
		    for (int zz = 0; zz < d_numMixingVars; zz++)
		      mixVars[zz] = indepVars[mixCount+zz];
		    unreactedStream = speciesStateSpace(mixVars);
		    if (!(d_adiabatic)) {
		      unreactedStream.d_temperature = TREF;
		      vector<double> massFract = unreactedStream.d_speciesConcn;
		      unreactedStream.d_density = 
			d_chemInterf->getMassDensity(unreactedStream.getPressure(),
						     unreactedStream.getTemperature(),
						     massFract); 
		    }
		    double dg = d_tableInfo->getIncrValueBelow(varCount);
		    double maxg = 1.0;
		    interpVar += dg;
		    double prevFactor = (interpVar - maxg)/(interpVar - dg - maxg);
		    double unFactor = -dg/(interpVar - dg - maxg);
		    if (prevFactor < 0) prevFactor = 0.0;
		    if (unFactor < 0) unFactor = 0.0;
		    stateSpaceVars = prevStateSpaceVars.linInterpolate(prevFactor, 
				      unFactor, unreactedStream);
		    //cout << "UnreactedStream Temp = " << unreactedStream.d_temperature << " " << unFactor << endl;
                    //cout << "prevTemp = " << prevStateSpaceVars.d_temperature << " "  << prevFactor << endl;
		    //cout << "stateSpaceVars Temp = " << stateSpaceVars.d_temperature << endl;
		    //cout << "interpVar = " << interpVar << endl;
                  }
		  else {
		    stateSpaceVars = d_integrator->integrate(tableIndex);
		  }
		  prevStateSpaceVars = stateSpaceVars;
		  //stateSpaceVars.drhodf = 0.0;
		  //vec_stateSpaceVars = stateSpaceVars.convertStreamToVec();
		  // defined in K-D tree or 2D vector implementation
		  //d_mixTable->Insert(tableIndex, vec_stateSpaceVars);	
		  d_mixTable->Insert(tableIndex, stateSpaceVars);	
		  ++entryCount;

		  //convertKeyToFloatValues(tableIndex, indepVars);
		  for (int nn = 0; nn < d_tableDimension; nn++) {
		    betafile << indepVars[nn] << " " ;  
		  }
		  betafile << endl;
		  //for (int nn = 0; nn < vec_stateSpaceVars.size(); nn++) {
		  //  betafile << vec_stateSpaceVars[nn] << " " ; 
		  //}
		  stateSpaceVars.print_oneline(betafile);
		  betafile << endl;
		  //if (kk = 0)
		  //  comparefile << indepVars[0] << endl;
		  //comparefile << vec_stateSpaceVars[0] << " " <<vec_stateSpaceVars[2] << endl;
		} // for ll 
	    } // for kk
	} // for jj
    } // for ii
  betafile.close();
  //comparefile.close();
      
  //Check to see if number of entries in datafile match the specified size of table 
  //Compute total number of entries that should be in table 
  int totalEntries = 1; //Don't want to multiply by zero
  for (int ii = 0; ii < d_tableDimension; ii++)
    totalEntries *= d_tableInfo->getNumDivsBelow(ii) + d_tableInfo->getNumDivsAbove(ii) + 1;
  cerr << "entryCount = " << entryCount << " totalEntries = " << totalEntries << endl;
  ASSERT(entryCount==totalEntries);

  delete [] tableIndex;
}









