//----- VectorTable.cc --------------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/Mixing/VectorTable.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/MixRxnTableInfo.h>

#include<iostream>

using namespace std;
using namespace Uintah;

//****************************************************************************
// Default constructor for VectorTable
//****************************************************************************
VectorTable::VectorTable(int numIndepVars, MixRxnTableInfo* tableInfo)
{
  d_numIndepVars = numIndepVars;
  d_tableInfo = tableInfo;
  // Calculate total number of entries in table
  int totalEntries = 1; //Don't want to multiply by zero
  for (int ii = 0; ii < numIndepVars; ii++)
    totalEntries *= d_tableInfo->getNumDivsBelow(ii) + d_tableInfo->getNumDivsAbove(ii) 
      + 1;
  d_tableVec = vector< vector <double> > (totalEntries);
  cout << "vectorTable total entries = " << totalEntries << endl;
  // Initialize to 0
}

//****************************************************************************
// Destructor
//****************************************************************************
VectorTable::~VectorTable()
{
}


bool
VectorTable::Lookup(int keyIndex[], vector<double>& stateSpaceVars)
{
  //Returns true if node is nonzero vector
  int tableIndex = 0;
  for (int ii = 0; ii < d_numIndepVars; ii++) {
    int nextTerm = 1;
    for (int jj = ii; jj < d_numIndepVars-1; jj++) {
      nextTerm *=  d_tableInfo->getNumDivsBelow(jj+1) + 
	d_tableInfo->getNumDivsAbove(jj+1)+1;
    }
    nextTerm *= keyIndex[ii];
    tableIndex += nextTerm;
  }
  if (d_tableVec[tableIndex].size() == 0) {
    return(false);
  }
  stateSpaceVars = d_tableVec[tableIndex];
  return(true);
}

bool
VectorTable::Insert(int keyIndex[], vector<double>& stateSpaceVars) 
{
    // Returns true if the inserted node is nonzero vector
  int tableIndex = 0;
  for (int ii = 0; ii < d_numIndepVars; ii++) {
    int nextTerm = 1;
    for (int jj = ii; jj < d_numIndepVars-1; jj++)
      nextTerm *=  d_tableInfo->getNumDivsBelow(jj+1) + 
	d_tableInfo->getNumDivsAbove(jj+1)+1;
    nextTerm *= keyIndex[ii];
    tableIndex += nextTerm;
  }

  if (d_tableVec[tableIndex].size() != 0) {
    cout << "Tried to insert but failed" << d_tableVec[tableIndex].size()<< endl;
    return(false);
  }
  d_tableVec[tableIndex] = stateSpaceVars;
  //cout << "Insertion complete" << endl;
  return(true);
}
  







































