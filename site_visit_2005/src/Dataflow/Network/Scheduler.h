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
 *  Scheduler.h: 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

#ifndef SCIRun_Dataflow_Network_Scheduler_h
#define SCIRun_Dataflow_Network_Scheduler_h

#include <Dataflow/Comm/MessageBase.h>
#include <Core/GuiInterface/GuiCallback.h>
#include <Core/Thread/Mailbox.h>
#include <Core/Thread/Runnable.h>
#include <string>
#include <list>

namespace SCIRun {

using std::string;

class Connection;
class MessageBase;
class Module;
class Network;
class OPort;


struct SerialSet
{
  unsigned int base;
  int size;
  int callback_count;

  SerialSet(unsigned int base, int s) :
    base(base), size(s), callback_count(0)
  {}
};

typedef void (*SchedulerCallback)(void *);


class Scheduler : public Runnable
{
  Network* net;
  bool first_schedule;
  bool schedule;
  unsigned int serial_id;
  std::list<SerialSet> serial_set;
  std::vector<std::pair<SchedulerCallback, void *> > callbacks_;

  virtual void run();
  void main_loop();
  void multisend_real(OPort*);
  void do_scheduling_real(Module*);
  void report_execution_finished_real(unsigned int serial);

  Mailbox<MessageBase*> mailbox;

public:

  Scheduler(Network*);
  ~Scheduler();
    
  // Turns scheduler on and off
  // Returns: true if scheduler is now on, false if scheduler is now off
  bool toggleOnOffScheduling();

  void do_scheduling();
  void request_multisend(OPort*);

  // msg must be of type ModuleExecute.
  void report_execution_finished(const MessageBase *msg);
  void report_execution_finished(unsigned int serial);

  void add_callback(SchedulerCallback cv, void *data);
  void remove_callback(SchedulerCallback cv, void *data);
};


class Scheduler_Module_Message : public MessageBase {
public:
  Connection* conn;
  unsigned int serial;
  Scheduler_Module_Message(unsigned int serial, bool ignored);
  Scheduler_Module_Message(unsigned int serial);
  Scheduler_Module_Message(Connection* conn);
  virtual ~Scheduler_Module_Message();
};


class Module_Scheduler_Message : public MessageBase {
public:
  OPort* p1;
  unsigned int serial;
  Module_Scheduler_Message();
  Module_Scheduler_Message(OPort*);
  Module_Scheduler_Message(unsigned int serial);
  virtual ~Module_Scheduler_Message();
};


} // End namespace SCIRun

#endif

