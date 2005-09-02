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



/*
 *  MxNMetaSynch.cc 
 *
 *  Written by:
 *   Keming Zhang 
 *   Department of Computer Science
 *   University of Utah
 *   August 2003 
 *
 *  Copyright (C) 2003 SCI Group
 */

#include <Core/CCA/PIDL/MxNMetaSynch.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/ConditionVariable.h>
#include <assert.h>
using namespace SCIRun;   

MxNMetaSynch::MxNMetaSynch(int size)
  : count_mutex("Receieved Flag lock"), metaCondition("setCallerDistribution Wait")
{
  //initialize the receiving flags to false.
  for (int i=0; i < size; i++) {
    recv_flags.push_back(0);
  }
  //initially, no setCallerDistribution is called, so completed should be false.
  completed=false;

  //initially, no body entered yet.
  numInside=0;
}

MxNMetaSynch::~MxNMetaSynch()
{
}

void MxNMetaSynch::enter()
{
  count_mutex.lock();
  numInside++;
  completed=false;
  count_mutex.unlock();
}


void MxNMetaSynch::waitForCompletion()
{
  //when a RPC invoke this method, if completed==true,
  //that means setCallerDistribution is done, and this 
  //status will not change until all the peer RPCs complete
  //their data distribution and only after that new 
  //setCallerDistribution can change this status. 
  count_mutex.lock();
  if(!completed) metaCondition.wait(count_mutex);
  count_mutex.unlock();
}



void MxNMetaSynch::leave(int rank)
{
  count_mutex.lock();

  if (rank < 0 || rank > recv_flags.size()){
    //TODO: throw an exception here
    ::std::cout << "Error: Unexpected meta distribution from rank=" << rank << "\n";
  } else {
    //same rank cannot enter at the same time, so this is threadsafe.
    ::std::cout << "got meta distribution from rank=" << rank << "\n";
    recv_flags[rank]++;
  }
  
  bool allReceived=true;

  for(unsigned int i = 1; i < recv_flags.size(); i++){
    if (recv_flags[0] != recv_flags[i]) {
      allReceived=false;
      break;
    }
  }
  
  numInside--;

  if(allReceived && numInside==0){
    completed=true;
    metaCondition.conditionBroadcast();
  }
  
  count_mutex.unlock();
}


