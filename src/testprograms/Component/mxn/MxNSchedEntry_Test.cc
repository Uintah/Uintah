/*!
    Test the following:
    	MxNScheduleEntry Creation and Usage
*/

#include <iostream>
#include <assert.h>
#include <Core/CCA/PIDL/MxNScheduleEntry.h>
#include <Core/CCA/SSIDL/array.h>
using namespace SCIRun;
using namespace std;

int main()
{

  MxNScheduleEntry* mxnentry1;
  
  //Create an MxNScheduleEntry
  mxnentry1 = new MxNScheduleEntry("K",callee);

  //isCaller() & isCallee
  assert(mxnentry1->isCaller() == false);
  assert(mxnentry1->isCallee() == true);

  //Create a couple of ArrayReps and add them
  Index** dr1 = new Index* [1];
  dr1[0] = new Index(0,98,2);
  MxNArrayRep* mxnarep1 = new MxNArrayRep(1,dr1);
  mxnarep1->setRank(0);
  Index** dr2 = new Index* [1];
  dr2[0] = new Index(0,50,1);
  MxNArrayRep* mxnarep2 = new MxNArrayRep(1,dr2);
  mxnarep1->setRank(0);
  //*****
  mxnentry1->addCalleeRep(mxnarep1);
  mxnentry1->addCallerRep(mxnarep2);
  
  //Report that the Meta Data is received
  mxnentry1->reportMetaRecvDone(1);

  //Set the Array
  SSIDL::array1<int>* main_arr = new  SSIDL::array1<int>(5);
  mxnentry1->setArray((void**) &main_arr);

  //Report that the Actual Data is received
  mxnentry1->doReceive(0);

  //Retrieve the Complete Array
  void* arr = mxnentry1->waitCompleteArray();
  delete ((SSIDL::array1<int>*) arr);

  delete mxnentry1;
}


