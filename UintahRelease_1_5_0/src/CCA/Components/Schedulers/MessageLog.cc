/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


#include <CCA/Components/Schedulers/MessageLog.h>
#include <Core/Grid/Patch.h>
#include <CCA/Ports/Output.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Time.h>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

// Debug: Used to sync cerr so it is readable (when output by
// multiple threads at the same time)  From sus.cc:
extern SCIRun::Mutex       cerrLock;

MessageLog::MessageLog(const ProcessorGroup* myworld, Output* oport)
   : d_enabled(false), d_myworld(myworld), d_oport(oport)
{
}

MessageLog::~MessageLog()
{
}

void MessageLog::problemSetup(const ProblemSpecP& prob_spec)
{
   ProblemSpecP mess = prob_spec->findBlock("MessageLog");
   if(mess){
      d_enabled = true;
      ostringstream outname;
      outname << d_oport->getOutputLocation() << "/message_" << setw(5) << setfill('0') << d_myworld->myrank() << ".log";
      out.open(outname.str().c_str());
      if(!out){
	 cerr << "Message log disabled, cannot open file\n";
	 d_enabled=false;
      }
      out << "\t\t\t\t  patch\t\t processor\n";
      out << "event\time\t\tsize\tfrom\tto\tfrom\tto\tvariable\totherinfo\n";
   } else {
      d_enabled=false;
   }
}

#if 0
void MessageLog::logSend(const DetailedReq* dep, int bytes,
			 const char* msg)
{
  if(!d_enabled)
    return;
  out << "send\t";
  out << setprecision(8) << Time::currentSeconds() << "\t";
  out << bytes << "\t";
  if(dep){
#if 0
      if(dep->d_task->getPatch())
	 out << dep->d_task->getPatch()->getID();
      else
	 out << "-";
      out << "\t";
      if(dep->d_patch)
	 out << dep->d_patch->getID();
      else
	 out << "-";
      out << "\t" << d_myworld->myrank() << "\t" << dep->d_task->getAssignedResourceIndex() << "\t";
      if(dep->d_var)
	 out << dep->d_var->getName();
      else
	 out << "-";
#else
      cerrLock.lock()
      NOT_FINISHED("new task stuff");
      cerrLock.unlock()
#endif
   } else {
      out << "-\t-\t-\t-\t-";
   }
   if(msg)
      out << "\t" << msg;
   out << '\n';
}

void MessageLog::logRecv(const DetailedReq* /*dep*/, int /*bytes*/,
			 const char* /*msg*/)
{
   if(!d_enabled)
      return;
}
#endif

void MessageLog::finishTimestep()
{
   if(!d_enabled)
      return;
   out.flush();
}

