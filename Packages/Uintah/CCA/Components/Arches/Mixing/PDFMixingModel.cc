//----- PDFMixingModel.cc --------------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/Mixing/PDFMixingModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/Integrator.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/KDTree.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/ReactionModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/StanjanEquilibriumReactionModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/ChemkinInterface.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/MixRxnTableInfo.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <string>
#include <iostream>

using namespace std;
using namespace Uintah;
using namespace SCIRun;
//****************************************************************************
// Default constructor for PDFMixingModel
//****************************************************************************
PDFMixingModel::PDFMixingModel():MixingModel()
{
}

//****************************************************************************
// Destructor
//****************************************************************************
PDFMixingModel::~PDFMixingModel()
{
  for (int i = 0; i < 2; i++)
    {
      delete [] d_tableBoundsVec[i];
      delete [] d_tableIndexVec[i];
    }
  delete [] d_tableBoundsVec;
  delete [] d_tableIndexVec;
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
  db->require("rxnvars",d_rxnVars);
  // read and initialize reaction model with chemkin interface
  string rxnModel;
  db->require("reaction_model",rxnModel);
  if (rxnModel == "EquilibriumReactionModel")
    d_rxnModel = new StanjanEquilibriumReactionModel(d_adiabatic);
  else
    throw InvalidValue("Reaction Model not supported" + rxnModel);
  d_rxnModel->problemSetup(db);
  cerr << "Made it upto pdfmix model" << std::endl;
  // number of species
  ChemkinInterface* chemInterf = d_rxnModel->getChemkinInterface();
  int nofSpecies = chemInterf->getNumSpecies();
  // Read the mixing variable streams, total is noofStreams 0 
  int nofstrm = 0;
  string speciesName;
  double mfrac; //mole or mass fraction
  for (ProblemSpecP stream_db = db->findBlock("Stream");
       stream_db != 0; stream_db = stream_db->findNextBlock("Stream")) {
    // Create the stream and add it to the vector
    d_streams.push_back(Stream(nofSpecies));
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
    double* ymassFrac = new double[nofSpecies];
    if (d_streams[nofstrm].d_mole) {
      ymassVec = chemInterf->convertMolestoMass(
					  d_streams[nofstrm].d_speciesConcn);
      d_streams[nofstrm].d_mole = false;
      d_streams[nofstrm].d_speciesConcn = ymassVec;
    }
    else
      ymassVec = d_streams[nofstrm].d_speciesConcn;
    // convert vec to array
    for (int ii = 0; ii < nofSpecies; ii++)
      ymassFrac[ii] = ymassVec[ii];
    double strmTemp = d_streams[nofstrm].d_temperature;
    double strmPress = d_streams[nofstrm].d_pressure;
    d_streams[nofstrm].d_density=chemInterf->getMassDensity(strmPress,
						     strmTemp, ymassFrac);
    d_streams[nofstrm].d_enthalpy=chemInterf->getMixEnthalpy(
							   strmTemp, ymassFrac);
    d_streams[nofstrm].d_moleWeight=chemInterf->getMixMoleWeight(ymassFrac);
    d_streams[nofstrm].d_cp=chemInterf->getMixSpecificHeat(strmTemp, ymassFrac);
    // store as mass fraction
    //    d_streams[nofstrm].print(cerr);
    delete[] ymassFrac;
    ++nofstrm;
  }
  // num_mix_scalars = num_streams -1
  d_numMixingVars = nofstrm - 1;
  d_tableDimension = d_numMixingVars + d_numMixStatVars + d_rxnVars + !(d_adiabatic);
  d_tableInfo = new MixRxnTableInfo(d_tableDimension);
  d_tableInfo->problemSetup(db, this);
  d_depStateSpaceVars = d_streams[0].getDepStateSpaceVars();
  d_dynamicTable = new KD_Tree(d_tableDimension, d_depStateSpaceVars);
  // allocating memory for two dimensional arrays
  d_tableBoundsVec = new double*[2];
  d_tableIndexVec = new int*[2];
  for (int i = 0; i < 2; i++)
    {
      d_tableBoundsVec[i] = new double[2*d_tableDimension];
      d_tableIndexVec[i] = new int[2*d_tableDimension];
    }
  d_integrator = new Integrator(d_tableDimension, this, d_rxnModel, d_tableInfo);
  d_integrator->problemSetup(db);
  

}

Stream
PDFMixingModel::speciesStateSpace(const vector<double>& mixVar) 
{
  ChemkinInterface* chemInterf = d_rxnModel->getChemkinInterface();
  int nofSpecies = chemInterf->getNumSpecies();
  Stream mixedStream(nofSpecies);
  // if adiabatic
  int count = 0;
  // store species as massfraction
  mixedStream.d_mole = false;
  if (d_adiabatic) {
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
    assert(count==d_numMixingVars);
    delete[] sumMixVarFrac;
  }
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
  int count = 0;
  // ***warning** change it
#if 0
  if (!(d_adiabatic)) {
    double absEnthalpy = inStream.d_enthalpy;
    vector<double> enthalpyValues = d_rxnModel->computeEnthalpy(mixRxnVar);
    double adiabaticEnthalpy = enthalpyValues[0];
    double sensEnthalpy = enthalpyValues[1];
    double normEnthalpy = (absEnthalpy - adiabaticEnthalpy)/sensEnthalpy;
    cout << endl;
    cout << "Absolute enthalpy: " << normEnthalpy*sensEnthalpy+adiabaticEnthalpy << endl;
    mixRxnVar[0] = normEnthalpy;
    count ++;
  }
#endif
  for (int i = 0; i < d_numMixingVars; i++)
    mixRxnVar[count++] = inStream.d_mixVars[i];
  for (int i = 0; i < d_numMixStatVars; i++)
    mixRxnVar[count++] = inStream.d_mixVarVariance[i];
  for (int i = 0; i < d_rxnVars; i++)
    mixRxnVar[count++] = inStream.d_rxnVars[i];
  // count and d_tableDimension should be equal
  assert(count==d_tableDimension);
  
  for (int i = 0; i < d_tableDimension; i++)
    {
	// calculates index in the table
      double tableIndex = (mixRxnVar[i] - d_tableInfo->getMinValue(i))/  
	d_tableInfo->getIncrValue(i);
      // can use floor(tableIndex)
      int tableLowIndex = (int) tableIndex; // cast to int
      int tableUpIndex = tableLowIndex + 1;
      if (tableLowIndex >= d_tableInfo->getNumDivisions(i))
	{
	  tableLowIndex = d_tableInfo->getNumDivisions(i);
	  tableIndex = tableLowIndex;
	  tableUpIndex = tableLowIndex;
	}
      d_tableIndexVec[0][i] = tableLowIndex;
      d_tableIndexVec[1][i] = tableUpIndex;
      // cast to float to compute the weighting factor for linear interpolation
      d_tableBoundsVec[0][i] = tableIndex - (double) tableLowIndex;
      d_tableBoundsVec[1][i] = 1.0 - d_tableBoundsVec[0][i];
    }
  int *lowIndex = new int[d_tableDimension + 1];
  int *upIndex = new int[d_tableDimension + 1];
  double *lowFactor = new double[d_tableDimension + 1];
  double *upFactor = new double[d_tableDimension + 1];
  outStream = interpolate(0, lowIndex, upIndex, lowFactor, upFactor);
  delete[] lowIndex;
  delete[] upIndex;
  delete[] lowFactor;
  delete[] upFactor;
}


Stream
PDFMixingModel::interpolate(int currentDim, int* lowIndex, int* upIndex,
			      double* lowFactor, double* upFactor) {
  if (currentDim == (d_tableDimension - 1))
    {
      lowFactor[currentDim] = d_tableBoundsVec[0][currentDim];
      upFactor[currentDim] = d_tableBoundsVec[1][currentDim];
      lowIndex[currentDim] = d_tableIndexVec[0][currentDim];
      Stream lowValue = tableLookUp(lowIndex);
      lowIndex[currentDim] = d_tableIndexVec[1][currentDim];
      Stream upValue = tableLookUp(lowIndex);
      return lowValue.linInterpolate(upFactor[currentDim],
				     lowFactor[currentDim], upValue);
    }
  else
    {
      lowIndex[currentDim] = d_tableIndexVec[0][currentDim];
      upIndex[currentDim] = d_tableIndexVec[1][currentDim];
      lowFactor[currentDim] = d_tableBoundsVec[0][currentDim];
      upFactor[currentDim] = d_tableBoundsVec[1][currentDim];
      for (int i = 0; i < currentDim; i++)
	upIndex[i] = lowIndex[i];
      Stream leftValue = interpolate(currentDim+1,lowIndex,upIndex,
				     lowFactor, upFactor);
      if (currentDim < (d_tableDimension - 2))
	{
	  lowIndex[currentDim] = d_tableIndexVec[0][currentDim];
	  upIndex[currentDim] = d_tableIndexVec[1][currentDim];
	}
      Stream rightValue =  interpolate(currentDim+1,upIndex,lowIndex,
				       lowFactor, upFactor);
      return leftValue.linInterpolate(upFactor[currentDim], 
				      lowFactor[currentDim], rightValue);
    }
}


Stream
PDFMixingModel::tableLookUp(int* tableKeyIndex) {
  Stream stateSpaceVars;
  vector<double> vec_stateSpaceVars;
  if (!(d_dynamicTable->Lookup(tableKeyIndex, vec_stateSpaceVars))) 
    {
      // call to integrator
      // calling Integrator constructor
      // it can be a local variable
      if (d_numMixStatVars)
	stateSpaceVars = d_integrator->integrate(tableKeyIndex);
      else
	stateSpaceVars = d_integrator->computeMeanValues(tableKeyIndex);
      bool flag = false;
      vec_stateSpaceVars = stateSpaceVars.convertStreamToVec(flag);
      // defined in K-D tree implementation
      d_dynamicTable->Insert(tableKeyIndex, vec_stateSpaceVars);
      //      cerr << "state space vars in PDFMixModel: " << tableKeyIndex[0] << endl;
      //      stateSpaceVars.print(cerr);
    }
  else {
    bool flag = false;
    stateSpaceVars.convertVecToStream(vec_stateSpaceVars, flag);
  }
  return stateSpaceVars;
  
}




