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
 *  MxNScheduleEntry.cc 
 *
 *  Written by:
 *   Kostadin Damevski & Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   May 2002 
 *
 *  Copyright (C) 2002 SCI Group
 */

#include <Core/CCA/PIDL/MxNScheduleEntry.h>
#include <Core/Thread/Thread.h>
#include <assert.h>
using namespace SCIRun;   

MxNScheduleEntry::MxNScheduleEntry()
{
  madeSched=false;
  meta_sync=NULL;
}

MxNScheduleEntry::~MxNScheduleEntry()
{
  clear();
}

void MxNScheduleEntry::addRep(MxNArrayRep* arr_rep)
{
  //Check if a representation with the same rank exists 
  int myrank = arr_rep->getRank();
  descriptorList::iterator iter = rep.begin();
  for(unsigned int i=0; i < rep.size(); i++, iter++){ 
    if(myrank == (*iter)->getRank()) { 
      delete (*iter); 
      rep.erase(iter); 
      break;
    }
  }
  rep.push_back(arr_rep);
}

descriptorList* MxNScheduleEntry::makeSchedule(MxNArrayRep* this_rep)
{
  if(madeSched) return (&sched);

  //Create the schedule 
  MxNArrayRep* i_rep;
  std::cerr<<"rep.size()="<<rep.size()<<"\n";

  for(unsigned int i=0; i < rep.size() ;i++)  {
    i_rep = rep[i];
    assert(i_rep->getDimNum() == this_rep->getDimNum());
    //Determine an intersect
    if(this_rep->isIntersect(i_rep)){
      std::cerr<<"sched.push_back "<<i<<"\n";
      sched.push_back(this_rep->Intersect(i_rep));
    }

  }
  //Set flag to indicate that schedule is now created
  madeSched = true;
  return (&sched);
}

MxNArrayRep* MxNScheduleEntry::getRep(unsigned int index)
{
  if (index < rep.size())
    return rep[index];
  else
    return NULL;
}

void MxNScheduleEntry::clear()
{
  unsigned int i;
  descriptorList::iterator iter;
  for(iter=rep.begin(),i=0; i < rep.size(); i++,iter++) {
    delete (*iter);
  }
  rep.clear();
}

void MxNScheduleEntry::print(std::ostream& dbg)
{
  for(unsigned int i=0; i < rep.size(); i++) {
    dbg << "------------------------------\n";
    rep[i]->print(dbg);
  }
}





