//----- PDFMixingModel.cc --------------------------------------------------

/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Components/Arches/PDFMixingModel.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Interface/Scheduler.h>
#include <Uintah/Exceptions/InvalidValue.h>

using namespace Uintah::ArchesSpace;

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
  db->require("rxnvars",d_numRxnVars);
  // Read the mixing variable streams, total is noofStreams 0 
  d_numMixingVars = 0;
  char speciesName[16];
  double mfrac; //mole or mass fraction
  for (ProblemSpecP stream_db = db->findBlock("Stream");
       stream_db != 0; stream_db = stream_db->findNextBlock("Stream")) {
    // Create the stream and add it to the vector
    d_streams.push_back(Stream());
    stream_db->require("pressure", d_streams[d_numMixingVars].d_pressure);
    stream_db->require("temperature", d_streams[d_numMixingVars].d_temperature);
    // mole fraction or mass fraction
    stream_db->require("mole",d_streams[d_numMixingVars].d_mole);
    for (ProblemSpecP species_db = db->findBlock("Species");
	 species_db !=0; species_db = species_db->findNextBlock("Species")) {
      species_db->require("symbol",speciesName);
      species_db->require("mfraction",mfrac);
      d_streams[d_numMixingVars].addSpecies(speciesName, mfrac);
    }
    ++d_numMixingVars;
  }
  // num_mix_scalars = num_streams -1
  d_numMixingVars--;
  d_tableDimension = d_numMixingVars + d_numMixStatVars + d_rxnVars + !(d_adiabatic);
  d_tableInfo = new MixRxnTableInfo(d_tableDimension);
  d_tableInfo->problemSetup(db, this);
  d_dynamicTable = new KD_Tree(d_tableDimension);
  d_integrator = new Integrator(d_tableDimension, this);
  d_integrator->problemSetup(db);
  // allocating memory for two dimensional arrays
  d_tableBoundsVec = new double*[2];
  d_tableIndexVec = new int*[2];
  for (int i = 0; i < 2; i++)
    {
      d_tableBoundsVec[i] = new double[2*d_tableDimension];
      d_tableIndexVec[i] = new int[2*d_tableDimension];
    }
  

}


//****************************************************************************
// Actually compute the properties here
//****************************************************************************
void 
PDFMixingModel::computeProps(InletStream& inStream,
			     Stream& outStream)
{
  // convert inStream to array
  std::vector<double> mixRxnVar(d_tableDimension);  
  int count = 0;
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
  for (int i = 0; i < d_numMixingVars; i++)
    mixRxnVar[count++] = inStream.d_mixVars[i];
  for (int i = 0; i < d_numMixStatVars; i++)
    mixRxnVar[count++] = inStream.d_mixVarVariance[i];
  for (int i = 0; i < d_numRxnVars; i++)
    mixRxnVar[count++] = inStream.d_rxnVars[i];
  // count and d_tableDimension should be equal
  assert(count, d_tableDimension);
  
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
#if 0
  vector<double> stateSpaceVar = interpolate(0, lowIndex, upIndex,
					    lowFactor, upFactor);
#endif
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
#if 0
      vector <double> lowValue = tableLookUp(lowIndex);
#endif
      Stream lowValue = tableLookUp(lowIndex);
      lowIndex[currentDim] = d_tableIndexVec[1][currentDim];
#if 0
      vector <double> upValue = tableLookUp(lowIndex);
#endif
      Stream upValue = tableLookUp(lowIndex);
#if 0
      for (int i = 0; i < lowValue.size(); i++)
	lowValue[i] = upFactor[currentDim]*lowValue[i] +
	  lowFactor[currentDim]*upValue[i];
      // return (upFactor[currentDim]*lowValue + lowFactor[currentDim]*upValue);
      return lowValue;
#endif
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
#if 0
      vector<double> leftValue = interpolate(currentDim+1,lowIndex,upIndex,
					    lowFactor, upFactor);
#endif
      Stream leftValue = interpolate(currentDim+1,lowIndex,upIndex,
				     lowFactor, upFactor);
      if (currentDim < (d_tableDimension - 2))
	{
	  lowIndex[currentDim] = d_tableIndexVec[0][currentDim];
	  upIndex[currentDim] = d_tableIndexVec[1][currentDim];
	}
#if 0
      vector<double> rightValue =  interpolate(currentDim+1,upIndex,lowIndex,
					      lowFactor, upFactor);
#endif
      Stream rightValue =  interpolate(currentDim+1,upIndex,lowIndex,
				       lowFactor, upFactor);
#if 0
      for ( i = 0; i < leftValue.size(); i++)
	leftValue[i] = upFactor[currentDim]*leftValue[i] +
	  lowFactor[currentDim]*rightValue[i];
      return leftValue;
#endif
      return leftValue.linInterpolate(upFactor[currentDim], 
				      lowFactor[currentDim], rightValue);
    }
}


Stream
PDFMixingModel::tableLookUp(int* tableKeyIndex) {
#if 0
  vector<double> stateSpaceVars(d_numStateSpaceVar);
#endif
  Stream stateSpaceVars;
  vector<double> vec_stateSpaceVars;
  if (!(d_dynamicTable->Lookup(tableKeyIndex, vec_stateSpaceVars))) 
    {
      // call to integrator
      // calling Integrator constructor
      // it can be a local variable
#if 0
      integratorFunction = new Integrator(d_tableDimension, tableKeyIndex, *this);
      integratorFunction->problemSetup();
      if (d_numMixStatVar)
	stateSpaceVars = integratorFunction->integrate();
      else
	stateSpaceVars = integratorFunction->computeMeanValues();
#endif
      if (d_numMixStatVar)
	stateSpaceVars = d_integrator->integrate(tableKeyIndex);
      else
	stateSpaceVars = d_integrator->computeMeanValues(tableKeyIndex);
      bool flag = false;
      vec_stateSpaceVars = stateSpaceVars.convertStreamToVec(flag);
      // defined in K-D tree implementation
      d_dynamicTable->Insert(tableKeyIndex, vec_stateSpaceVars);
      stateSpaceVars.print(cerr);
    }
  else {
    bool flag = false;
    stateSpaceVars.convertVecToStream(vec_stateSpaceVars, flag);
  }
  return stateSpaceVars;
  
}




//
// $Log$
// Revision 1.1  2001/01/15 23:38:21  rawat
// added some more classes for implementing mixing model
//
//

