/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  SimplePort.h:  Ports that use only the Atomic protocol
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_SimplePort_h
#define SCI_project_SimplePort_h 1

#include <Dataflow/Network/Port.h>
#include <Core/Thread/Mailbox.h>
#include <Dataflow/Network/Connection.h>
#include <Dataflow/Network/Module.h>
#include <Core/Datatypes/Field.h>

namespace SCIRun {


class Module;

template<class T>
struct SimplePortComm {
  SimplePortComm();
  SimplePortComm(const T&);

  T data_;
  int have_data_;
};

template<class T> class SimpleOPort;

template<class T>
class SimpleIPort : public IPort {
public:
  friend class SimpleOPort<T>;
  Mailbox<SimplePortComm<T>*> mailbox;

  static string port_type_;
  static string port_color_;

  SimpleIPort(Module*, const string& name);
  virtual ~SimpleIPort();
  virtual void reset();
  virtual void finish();

  int get(T&);
private:
  int recvd_;
};

template<class T>
class SimpleOPort : public OPort {
public:
  SimpleOPort(Module*, const string& name);
  virtual ~SimpleOPort();

  virtual void reset();
  virtual void finish();

  void send(const T&);
  void send_intermediate(const T&);

  virtual bool have_data();
  virtual void resend(Connection* conn);
private:
  void do_send(const T&);
  void do_send_intermediate(const T&);

  bool sent_something_;
  T handle_;
};



template<class T>
SimpleIPort<T>::SimpleIPort(Module* module, 
			    const string& portname)
  : IPort(module, port_type_, portname, port_color_),
    mailbox("Port mailbox (SimpleIPort)", 2)
{
}

template<class T>
SimpleIPort<T>::~SimpleIPort()
{
}

template<class T>
SimpleOPort<T>::SimpleOPort(Module* module, 
			    const string& portname)
  : OPort(module, SimpleIPort<T>::port_type_, portname,
	  SimpleIPort<T>::port_color_),
    sent_something_(false),
    handle_(0)
{
}

template<class T>
SimpleOPort<T>::~SimpleOPort()
{
}

template<class T>
void SimpleIPort<T>::reset()
{
  recvd_=0;
}

template<class T>
void SimpleIPort<T>::finish()
{
  if(!recvd_ && num_unblocked_connections() > 0) {
    if (module->showStats()) turn_on(Finishing);
    SimplePortComm<T>* msg=mailbox.receive();
    delete msg;
    if (module->showStats()) { turn_off(); }
  }
}

template<class T>
void SimpleOPort<T>::reset()
{
  sent_something_ = false;
  handle_ = 0;
}

template<class T>
void SimpleOPort<T>::finish()
{
  if(!sent_something_ && num_unblocked_connections() > 0){
    // Tell them that we didn't send anything...
    if (module->showStats()) { turn_on(Finishing); }
    for(int i=0;i<nconnections();i++) {
      SimplePortComm<T>* msg = new SimplePortComm<T>();
      Connection* conn = connections[i];
      if (! conn->is_blocked()) {
	((SimpleIPort<T>*)conn->iport)->mailbox.send(msg);
      }
    }
    
    if (module->showStats())
       turn_off();
  }
}

//! Declare specialization for field ports.
//! Field ports must only send const fields i.e. frozen fields.
//! Definition in FieldPort.cc
template<>
void SimpleOPort<FieldHandle>::send(const FieldHandle& data);

template<class T>
void SimpleOPort<T>::send(const T& data)
{
  do_send(data);
}

template<class T>
void SimpleOPort<T>::do_send(const T& data)
{
  handle_ = data;
  if (num_unblocked_connections() == 0) { return; }

  // Change oport state and colors on screen.
  if (module->showStats()) { turn_on(); }

  for (int i = 0; i < nconnections(); i++) {
    // Add the new message.
    SimplePortComm<T>* msg = new SimplePortComm<T>(data);
    Connection* conn = connections[i];
    if (! conn->is_blocked()) {
      ((SimpleIPort<T>*)conn->iport)->mailbox.send(msg);
    }
  }
  sent_something_ = true;

  if (module->showStats()) { turn_off(); }
}

template<>
void SimpleOPort<FieldHandle>::send_intermediate(const FieldHandle& data);

template<class T>
void SimpleOPort<T>::send_intermediate(const T& data)
{
  do_send_intermediate(data);
}

template<class T>
void SimpleOPort<T>::do_send_intermediate(const T& data)
{
  handle_ = data;
  if(num_unblocked_connections() == 0) { return; }

  if (module->showStats()) { turn_on(); }

  module->request_multisend(this);

  for(int i=0;i<nconnections();i++){
    // Add the new message.
    SimplePortComm<T>* msg=new SimplePortComm<T>(data);
    Connection* conn = connections[i];
    if (! conn->is_blocked()) {
      ((SimpleIPort<T>*)conn->iport)->mailbox.send(msg);
    }
  }
  sent_something_ = true;

  if (module->showStats()) { turn_off(); }
}

template<class T>
int SimpleIPort<T>::get(T& data)
{
  if(num_unblocked_connections()==0) { return 0; }
  if (module->showStats()) { turn_on(); }

  // Wait for the data...
  SimplePortComm<T>* comm=mailbox.receive();
  recvd_=1;
  if(comm->have_data_){
    data=comm->data_;
    delete comm;
    if (module->showStats())
      turn_off();
    return 1;
  } else {
    delete comm;
    if (module->showStats())
      turn_off();
    return 0;
  }
}

template<class T>
bool SimpleOPort<T>::have_data()
{
  if(handle_.get_rep())
    return true;
  else 
    return false;
}

template<class T>
void SimpleOPort<T>::resend(Connection* conn)
{
  if (module->showStats()) { turn_on(); }
  for(int i=0;i<nconnections();i++){
    Connection* c = connections[i];
    if (! c->is_blocked()) {
      if(c == conn){
	SimplePortComm<T>* msg=new SimplePortComm<T>(handle_);
	((SimpleIPort<T>*)c->iport)->mailbox.send(msg);
      }
    }
  }
  if (module->showStats()) { turn_off(); }
}

template<class T>
SimplePortComm<T>::SimplePortComm() : 
  have_data_(0)
{
}

template<class T>
SimplePortComm<T>::SimplePortComm(const T& data) : 
  data_(data), 
  have_data_(1)
{
}

} // End namespace SCIRun


#endif /* SCI_project_SimplePort_h */
