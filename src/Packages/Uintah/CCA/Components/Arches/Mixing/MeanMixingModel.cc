//----- MeanMixingModel.cc --------------------------------------------------

#include <TauProfilerForSCIRun.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/MeanMixingModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/InletStream.h>
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
#include <math.h>
#include <Core/Math/MiscMath.h>

using namespace std;
using namespace Uintah;
using namespace SCIRun;
//****************************************************************************
// Default constructor for MeanMixingModel
//****************************************************************************
MeanMixingModel::MeanMixingModel():MixingModel(), DynamicTable()
{
}

//****************************************************************************
// Destructor
//****************************************************************************
MeanMixingModel::~MeanMixingModel()
{
}

//****************************************************************************
// Problem Setup for MeanMixingModel
//****************************************************************************
void 
MeanMixingModel::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP db = params->findBlock("MeanMixingModel");
  db->require("adiabatic",d_adiabatic);
  db->require("rxnvars",d_numRxnVars);
  // Set up for MixingModel table
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
  //d_CO2index = chemInterf->getSpeciesIndex("CO2");
  //d_H2Oindex = chemInterf->getSpeciesIndex("H2O");
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
    if (d_streams[nofstrm].d_mole) {
      ymassVec = chemInterf->convertMolestoMass(
					  d_streams[nofstrm].d_speciesConcn);
      d_streams[nofstrm].d_mole = false;
      d_streams[nofstrm].d_speciesConcn = ymassVec;
    }
    else
      ymassVec = d_streams[nofstrm].d_speciesConcn;
    double strmTemp = d_streams[nofstrm].d_temperature;
    double strmPress = d_streams[nofstrm].d_pressure;
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
  //cout << "Mean::numMixVars = " << d_numMixingVars << endl;
  //cout <<"Mean::numRxnVars = "<<d_numRxnVars<<endl;
 
  // Define mixing table, which includes call reaction model constructor
  d_tableDimension = d_numMixingVars + d_numRxnVars + !(d_adiabatic);
  d_tableInfo = new MixRxnTableInfo(d_tableDimension);
  bool mixTableFlag = false; //Table does not have variance
  d_tableInfo->problemSetup(db, mixTableFlag, this);
  // Define table type (static or dynamic). Set up table storage, either as 
  // vectors or as kdtree 
  if (db->findBlock("TableType")) {
    // d_tableType must be class variable because of inline function
    db->require("TableType", d_tableType);
    if (d_tableType != "dynamic") {
      cerr << "MeanMixing TABLE TYPE is dynamic" << endl;
      d_tableType = "dynamic";
    }
  }
  else {
    d_tableType = "dynamic";
    // Default tableType is dynamic
    cout << "MeanMixing TABLE TYPE is dynamic" << endl;
  }
  // Call reaction model constructor, get total number of dependent  vars;
  // can't call it sooner because d_numMixingVars and mixTableType are needed
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
      d_mixTable = new VectorTable(d_tableDimension, d_tableInfo);
      cerr << "MeanMixing TABLE STORAGE is vectorTable" << endl;
      //throw InvalidValue("Table storage not supported" + tableStorage);
    }
  }
  else {
    d_mixTable = new VectorTable(d_tableDimension, d_tableInfo);
    cout << "MeanMixing TABLE STORAGE is vectorTable" << endl;
  }
  // tableSetup is a function in DynamicTable; it allocates memory for 
  // table functions
  tableSetup(d_tableDimension, d_tableInfo);

}

Stream
MeanMixingModel::speciesStateSpace(const vector<double>& mixVar) 
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
MeanMixingModel::computeProps(const InletStream& inStream,
			     Stream& outStream)
{
  TAU_PROFILE_TIMER(mixing, "Mixing", "[Mixing::mixing]" , TAU_USER);
  TAU_PROFILE_TIMER(reaction, "Reaction", "[Mixing::reaction]" , TAU_USER);
  TAU_PROFILE_START(mixing);
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
    // Check to make sure mixing vars within proper range
    if (inStream.d_mixVars[i] < d_tableInfo->getMinValue(count)) {
      //if (Abs(inStream.d_mixVars[i]) < 1e-10) {
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
  TAU_PROFILE_START(reaction);
    getProps(normVar, normStream);
  TAU_PROFILE_STOP(reaction);
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
  //Normalize reaction variables
  if (d_numRxnVars > 0) { 
    //Since min/max rxn parameter values for a given (h/f) combo are the 
    //same for every rxn parameter entry, look up the first entry;
    Stream paramValues;
    for (int ii = 0; ii < d_numRxnVars; ii++) {
      TAU_PROFILE_START(reaction);
      getProps(normVar, paramValues);
  TAU_PROFILE_STOP(reaction);
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
#if 0
  cout << "Mean::getProps mixRxnVar = " << endl;
  for (int ii = 0; ii < mixRxnVar.size(); ii++) {
    cout.width(10);
    cout << mixRxnVar[ii] << " " ; 
    if (!(ii % 10)) cout << endl; 
  }
  cout << endl;
#endif
  TAU_PROFILE_STOP(mixing);
  TAU_PROFILE_START(reaction);
  getProps(mixRxnVar, outStream); //function in DynamicTable
  TAU_PROFILE_STOP(reaction);
  outStream.d_co2 = outStream.d_speciesConcn[d_CO2index]; //Needed for radiation model
  outStream.d_h2o = outStream.d_speciesConcn[d_H2Oindex]; //Needed for radiation model
  //outStream.print(cout);
 
 
}


void
MeanMixingModel::tableLookUp(int* tableKeyIndex, Stream& stateSpaceVars) {
  TAU_PROFILE("lookup", "[Properties::lookup]" , TAU_USER);
  TAU_PROFILE_TIMER(compute, "Compute", "[Mixing::compute]" , TAU_USER);
  //vector<double> vec_stateSpaceVars;
  bool lsoot = d_rxnModel->getSootBool();
#if 0
  cout << "Mean::tableKeyIndex = " << endl;
  for (int ii = 0; ii < d_tableDimension; ii++) {
    cout.width(10);
    cout << tableKeyIndex[ii] << " " ; 
    if (!(ii % 10)) cout << endl; 
  }
  cout << endl;
#endif
  //if (!(d_mixTable->Lookup(tableKeyIndex, vec_stateSpaceVars))) 
  if (!(d_mixTable->Lookup(tableKeyIndex, stateSpaceVars))) 
    {
  TAU_PROFILE_START(compute);
      computeMeanValues(tableKeyIndex, stateSpaceVars);
      //vec_stateSpaceVars = stateSpaceVars.convertStreamToVec();
      // defined in K-D tree or 2D vector implementation
      //d_mixTable->Insert(tableKeyIndex, vec_stateSpaceVars);
      d_mixTable->Insert(tableKeyIndex, stateSpaceVars);
      //stateSpaceVars.print(cerr);
  TAU_PROFILE_STOP(compute);
    }
  else {
    //stateSpaceVars.convertVecToStream(vec_stateSpaceVars, d_numMixingVars,
    //			      d_numRxnVars, lsoot);
#if 0
    cout<<"Mean::entry exists"<<endl;
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

void
MeanMixingModel::computeMeanValues(int* tableKeyIndex, Stream& meanStream)
{
  // Compute floating point values of independent variables
  // from tableKeyIndex
  vector<double> indepVars(d_tableDimension);
  convertKeytoFloatValues(tableKeyIndex, indepVars);
  int mixIndex = 0;
  if (!(d_adiabatic))
    ++mixIndex;
  vector<double> mixVars(d_numMixingVars);
  for (int jj = 0; jj < d_numMixingVars; jj++)
    mixVars[jj] = indepVars[mixIndex+jj];
  Stream unreactedStream = speciesStateSpace(mixVars);
  //unreactedStream.print(cout);
  d_rxnModel->getRxnStateSpace(unreactedStream, indepVars, meanStream);
}

void
MeanMixingModel::convertKeytoFloatValues(int tableKeyIndex[], vector<double>& indepVars) {
  for (int i = 0; i < d_tableDimension; i++)
    if (tableKeyIndex[i] <= d_tableInfo->getNumDivsBelow(i))
      indepVars[i] = tableKeyIndex[i]*d_tableInfo->getIncrValueBelow(i) + 
	d_tableInfo->getMinValue(i);
    else
      indepVars[i] = (tableKeyIndex[i]-d_tableInfo->getNumDivsBelow(i))*	
	d_tableInfo->getIncrValueAbove(i) + d_tableInfo->getStoicValue(i);
  return;
}
