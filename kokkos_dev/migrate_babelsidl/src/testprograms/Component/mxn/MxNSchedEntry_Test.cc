/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/

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


