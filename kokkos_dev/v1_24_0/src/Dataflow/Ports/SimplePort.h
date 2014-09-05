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
#include <Core/Util/Environment.h>

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
  enum SendType {SEND_NORMAL=0, SEND_INTERMEDIATE=1};

  void do_send(const T&, SendType type = SEND_NORMAL);

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
  if (sci_getenv_p("SCIRUN_NO_PORT_CACHING"))
  {
    issue_no_port_caching_warning();
    cache_ = false;
  }
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
  //handle_ = 0;
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
      SimplePortComm<T>* msg = scinew SimplePortComm<T>(handle_);
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
  do_send(data, SEND_NORMAL);
}

template<class T>
void
SimpleOPort<T>::send_intermediate(const T& data)
{
  do_send(data, SEND_INTERMEDIATE);
}

template<class T>
void
SimpleOPort<T>::do_send(const T& data, SendType type)
{
  handle_ = cache_ ? data : 0;

  if (nconnections() == 0) { return; }

  // Change oport state and colors on screen.
  if (module->show_stats()) { turn_on(); }

  if( type == SEND_INTERMEDIATE )
    module->request_multisend(this);

  for (int i = 0; i < nconnections(); i++)
  {
    // Add the new message.
    Connection* conn = connections[i];
    SimplePortComm<T>* msg = scinew SimplePortComm<T>(data);
    ((SimpleIPort<T>*)conn->iport)->mailbox.send(msg);
  }

  sent_something_ = true;

  // Change oport state and colors on screen.
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
