//----- DynamicTable.cc --------------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/Mixing/DynamicTable.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/MixRxnTableInfo.h>
#include <iostream>

using namespace std;
using namespace Uintah;

//****************************************************************************
// Default constructor for DynamicTable
//****************************************************************************
DynamicTable::DynamicTable()
{
}

//****************************************************************************
// Destructor
//****************************************************************************
DynamicTable::~DynamicTable()
{
  for (int i = 0; i < 2; i++)
    {
      delete [] d_tableBoundsVec[i];
      delete [] d_tableIndexVec[i];
    }
  delete [] d_tableBoundsVec;
  delete [] d_tableIndexVec;
}

void
DynamicTable::tableSetup(int numTableDim, MixRxnTableInfo* tableInfo)
{
  d_tableDim = numTableDim;
  d_tableInfo = tableInfo;
  // allocating memory for two dimensional arrays
  d_tableBoundsVec = new double*[2];
  d_tableIndexVec = new int*[2];
  for (int i = 0; i < 2; i++)
    {
      d_tableBoundsVec[i] = new double[2*d_tableDim];
      d_tableIndexVec[i] = new int[2*d_tableDim];
    } 
}
  
Stream
DynamicTable::getProps(const vector<double> mixRxnVar)
{
  for (int i = 0; i < d_tableDim; i++)
    {
	// calculates index in the table
      double midPt = d_tableInfo->getStoicValue(i);
      double tableIndex;
      if (mixRxnVar[i] <= midPt) {
	tableIndex = (mixRxnVar[i] - d_tableInfo->getMinValue(i))/  
	  d_tableInfo->getIncrValueBelow(i);
	//cout << mixRxnVar[i] << " " << d_tableInfo->getMinValue(i) << " " 
	//   << d_tableInfo->getIncrValueBelow(i) << endl;
      }
      else {
	tableIndex = ((mixRxnVar[i] - midPt)/  
	  d_tableInfo->getIncrValueAbove(i))+d_tableInfo->getNumDivsBelow(i);
	//cout << mixRxnVar[i] << " " << midPt << " " 
	//     << d_tableInfo->getIncrValueAbove(i) << " "  << 
	//  d_tableInfo->getNumDivsBelow(i)<< endl;
      }
      // can use floor(tableIndex)
      //tableIndex = tableIndex + 0.5;
      int tableLowIndex = (int) tableIndex; // cast to int
      int tableUpIndex = tableLowIndex + 1;
      if (tableLowIndex >= (d_tableInfo->getNumDivsBelow(i)+
	  d_tableInfo->getNumDivsAbove(i)))
	{
	  tableLowIndex = d_tableInfo->getNumDivsBelow(i)+
	    d_tableInfo->getNumDivsAbove(i);
	  tableIndex = tableLowIndex;
	  tableUpIndex = tableLowIndex;
	}
      d_tableIndexVec[0][i] = tableLowIndex;
      d_tableIndexVec[1][i] = tableUpIndex;
      // cast to float to compute the weighting factor for linear interpolation
      d_tableBoundsVec[0][i] = tableIndex - (double) tableLowIndex;
      d_tableBoundsVec[1][i] = 1.0 - d_tableBoundsVec[0][i];
    }
  int *lowIndex = new int[d_tableDim + 1];
  int *upIndex = new int[d_tableDim + 1];
  double *lowFactor = new double[d_tableDim + 1];
  double *upFactor = new double[d_tableDim + 1];
  Stream outStream = interpolate(0, lowIndex, upIndex, lowFactor, upFactor);
  delete[] lowIndex;
  delete[] upIndex;
  delete[] lowFactor;
  delete[] upFactor;
  return outStream;
}

Stream
DynamicTable::interpolate(int currentDim, int* lowIndex, int* upIndex,
			      double* lowFactor, double* upFactor) {
  if (currentDim == (d_tableDim- 1))
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
      if (currentDim < (d_tableDim - 2))
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


 
//
// $Log$
// Revision 1.5  2001/11/08 19:13:44  spinti
// 1. Corrected minor problems in ILDMReactionModel.cc
// 2. Added tabulation capability to StanjanEquilibriumReactionModel.cc. Now,
//    a reaction table is created dynamically. The limits and spacing in the
//    table are specified in the *.ups file.
// 3. Corrected the mixture temperature computation in Stream::addStream. It
//    now is computed using a Newton search.
// 4. Made other minor corrections to various reaction model files.
//
// Revision 1.2  2001/08/03 18:08:09  witzel
// Removed an extraneous semi-colon at the end of a #include.
//
// Revision 1.1  2001/07/16 21:15:38  rawat
// added enthalpy solver and Jennifer's changes in Mixing and Reaction model required for ILDM and non-adiabatic cases
//
// 

