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
      double tableIndex = (mixRxnVar[i] - d_tableInfo->getMinValue(i))/  
	d_tableInfo->getIncrValue(i);
      //cout << "DynamicTable: tableIndex = " << tableIndex << endl;
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
      //cout << "DynamicTable: lowValue complete for lowIndex = " << lowIndex[currentDim] << endl;
      //cout<< "LowFactor = "<<lowFactor[currentDim]<<" UpFactor = "<<
      //	upFactor[currentDim]<<endl;
      lowIndex[currentDim] = d_tableIndexVec[1][currentDim];
      //cout<< "lowIndex = " << lowIndex[currentDim] << endl;
      Stream upValue = tableLookUp(lowIndex);
      //cout << "Dynamic Table: upValue complete for upIndex = " << lowIndex[currentDim] <<endl;
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
// Revision 1.2  2001/08/03 18:08:09  witzel
// Removed an extraneous semi-colon at the end of a #include.
//
// Revision 1.1  2001/07/16 21:15:38  rawat
// added enthalpy solver and Jennifer's changes in Mixing and Reaction model required for ILDM and non-adiabatic cases
//
// 

