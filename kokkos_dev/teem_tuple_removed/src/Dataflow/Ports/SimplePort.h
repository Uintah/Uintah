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
struct SimplePortComm
{
  SimplePortComm();
  SimplePortComm(const T&);

  T data_;
  bool have_data_;
};


template<class T>
class SimpleIPort : public IPort
{
public:
  Mailbox<SimplePortComm<T>*> mailbox;

  static string port_type_;
  static string port_color_;

  SimpleIPort(Module*, const string& name);
  virtual ~SimpleIPort();
  virtual void reset();
  virtual void finish();

  int get(T&);

private:
  bool got_something_;
};


template<class T>
class SimpleOPort : public OPort
{
public:
  SimpleOPort(Module*, const string& name);
  virtual ~SimpleOPort();

  virtual void reset();
  virtual void finish();
  virtual void detach(Connection *conn, bool blocked);

  void send(const T&);
  void send_intermediate(const T&);
  void set_cache( bool cache = true )
  {
    cache_ = cache;
    if ( !cache )
    {
      handle_ = 0;
    }
  }

  virtual bool have_data();
  virtual void resend(Connection* conn);

private:
  void do_send(const T&);
  void do_send_intermediate(const T&);

  bool cache_;
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
    cache_(true),
    sent_something_(true),
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
  got_something_ = false;
}


template<class T>
void
SimpleIPort<T>::finish()
{
  if (!got_something_ && nconnections() > 0)
  {
    if (module->show_stats()) { turn_on(Finishing); }
    SimplePortComm<T>* msg = mailbox.receive();
    delete msg;
    if (module->show_stats()) { turn_off(); }
  }
  got_something_ = true;
}


template<class T>
void
SimpleOPort<T>::reset()
{
  sent_something_ = false;
  handle_ = 0;
}


template<class T>
void
SimpleOPort<T>::finish()
{
  if (!sent_something_ && nconnections() > 0)
  {
    // Tell them that we didn't send anything.
    if (module->show_stats()) { turn_on(Finishing); }
    for (int i = 0; i < nconnections(); i++)
    {
      Connection* conn = connections[i];
      SimplePortComm<T>* msg = scinew SimplePortComm<T>();
      ((SimpleIPort<T>*)conn->iport)->mailbox.send(msg);
    }

    if (module->show_stats()) { turn_off(); }
  }

  sent_something_ = true;
}


template<class T>
void
SimpleOPort<T>::detach(Connection *conn, bool blocked)
{
  if (!sent_something_)
  {
    SimplePortComm<T>* msg = scinew SimplePortComm<T>(0);
    ((SimpleIPort<T>*)conn->iport)->mailbox.send(msg);
  }
  //sent_something_ = true;  // Only sent something on the one port.
  OPort::detach(conn, blocked);
}


//! Declare specialization for field ports.
//! Field ports must only send const fields i.e. frozen fields.
//! Definition in FieldPort.cc
template<>
void SimpleOPort<FieldHandle>::send(const FieldHandle& data);

template<class T>
void
SimpleOPort<T>::send(const T& data)
{
  do_send(data);
}


template<class T>
void
SimpleOPort<T>::do_send(const T& data)
{
  handle_ = cache_ ? data : 0;

  if (nconnections() == 0) { return; }

  // Change oport state and colors on screen.
  if (module->show_stats()) { turn_on(); }

  for (int i = 0; i < nconnections(); i++)
  {
    // Add the new message.
    Connection* conn = connections[i];
    SimplePortComm<T>* msg = scinew SimplePortComm<T>(data);
    ((SimpleIPort<T>*)conn->iport)->mailbox.send(msg);
  }
  sent_something_ = true;

  if (module->show_stats()) { turn_off(); }
}


template<>
void SimpleOPort<FieldHandle>::send_intermediate(const FieldHandle& data);

template<class T>
void
SimpleOPort<T>::send_intermediate(const T& data)
{
  do_send_intermediate(data);
}


template<class T>
void
SimpleOPort<T>::do_send_intermediate(const T& data)
{
  handle_ = cache_ ? data : 0;

  if (nconnections() == 0) { return; }

  if (module->show_stats()) { turn_on(); }

  module->request_multisend(this);

  for (int i = 0; i < nconnections(); i++)
  {
    // Add the new message.
    Connection* conn = connections[i];
    SimplePortComm<T>* msg = scinew SimplePortComm<T>(data);
    ((SimpleIPort<T>*)conn->iport)->mailbox.send(msg);
  }
  sent_something_ = true;

  if (module->show_stats()) { turn_off(); }
}


template<class T>
int
SimpleIPort<T>::get(T& data)
{
  if (nconnections() == 0) { return 0; }
  if (module->show_stats()) { turn_on(); }

  // Wait for the data.
  SimplePortComm<T>* comm = mailbox.receive();
  got_something_ = true;
  if (comm->have_data_)
  {
    data = comm->data_;
    delete comm;
    if (module->show_stats()) { turn_off(); }
    return 1;
  }
  else
  {
    delete comm;
    if (module->show_stats()) { turn_off(); }
    return 0;
  }
}


template<class T>
bool
SimpleOPort<T>::have_data()
{
  return (bool)(handle_.get_rep());
}


template<class T>
void
SimpleOPort<T>::resend(Connection* conn)
{
  if (module->show_stats()) { turn_on(); }
  for (int i = 0; i < nconnections(); i++)
  {
    Connection* c = connections[i];
    if (c == conn)
    {
      SimplePortComm<T>* msg = scinew SimplePortComm<T>(handle_);
      ((SimpleIPort<T>*)c->iport)->mailbox.send(msg);
    }
  }
  if (module->show_stats()) { turn_off(); }
}


template<class T>
SimplePortComm<T>::SimplePortComm() :
  have_data_(false)
{
}


template<class T>
SimplePortComm<T>::SimplePortComm(const T& data) :
  data_(data),
  have_data_(true)
{
}


} // End namespace SCIRun


#endif /* SCI_project_SimplePort_h */
