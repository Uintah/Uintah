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
#include <Core/Util/Timer.h>    
#include <Core/Persistent/Pstreams.h>
#include <Core/Util/Assert.h>
#include <Dataflow/Network/Connection.h>
#include <Dataflow/Network/Module.h>
#include <Core/GuiInterface/Remote.h>
#include <Core/Datatypes/Field.h>
#include <iostream>
#include <unistd.h> // for the call to sleep

using std::cerr;
using std::endl;

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
  enum Protocol {
    Atomic=0x01
  };

  friend class SimpleOPort<T>;
  Mailbox<SimplePortComm<T>*> mailbox;

  static string port_type_;
  static string port_color_;

  SimpleIPort(Module*, const string& name, int protocol=Atomic);
  virtual ~SimpleIPort();
  virtual void reset();
  virtual void finish();

  int get(T&);
  int special_get(T&);
private:
  int recvd_;
};

template<class T>
class SimpleOPort : public OPort {
public:
  SimpleOPort(Module*, const string& name, 
	      int protocol=SimpleIPort<T>::Atomic);
  virtual ~SimpleOPort();

  virtual void reset();
  virtual void finish();

  void send(const T&);
  void send_intermediate(const T&);

  virtual int have_data();
  virtual void resend(Connection* conn);
private:
  void do_send(const T&);
  void do_send_intermediate(const T&);

  int sent_something_;
  T handle_;
#ifdef DEBUG
  WallClockTimer timer1_;
#endif

};

} // End namespace SCIRun

extern char** global_argv;

namespace SCIRun {


template<class T>
SimpleIPort<T>::SimpleIPort(Module* module, 
			    const string& portname,
			    int protocol) : 
  IPort(module, port_type_, portname, port_color_, protocol),
  mailbox("Port mailbox (SimpleIPort)", 2)
{
}

template<class T>
SimpleIPort<T>::~SimpleIPort()
{
}

template<class T>
SimpleOPort<T>::SimpleOPort(Module* module, 
			    const string& portname,
			    int protocol) : 
  OPort(module, SimpleIPort<T>::port_type_, portname,
	SimpleIPort<T>::port_color_, protocol)
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
  if(!recvd_ && nconnections() > 0) {
    if (module->show_stat) turn_on(Finishing);
    SimplePortComm<T>* msg=mailbox.receive();
    delete msg;
    if (module->show_stat) { turn_off(); }
  }
}

template<class T>
void SimpleOPort<T>::reset()
{
  handle_=0;
  sent_something_=0;
}

template<class T>
void SimpleOPort<T>::finish()
{
  // get timestamp here to measure communication time, print to screen
#ifdef DEBUG
  timer1.stop();
  double time = timer1.time();
#endif

#ifdef DEBUG
  cerr << "Done in " << time << " seconds\n"; 
  cerr << "Entering SimpleOPort<T>::finish()\n";
#endif

  if(!sent_something_ && nconnections() > 0){
    // Tell them that we didn't send anything...
    if (module->show_stat) { turn_on(Finishing); }
    for(int i=0;i<nconnections();i++) {
      SimplePortComm<T>* msg = new SimplePortComm<T>();
      ((SimpleIPort<T>*)connections[i]->iport)->mailbox.send(msg);
    }
    
#ifdef DEBUG
    cerr << "Exiting SimpleOPort<T>::finish()\n";
#endif
    if (module->show_stat) { turn_off(); }
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

#ifdef DEBUG
  cerr << "Entering SimpleOPort<T>::send (data)\n";
#endif

  handle_ = data;
  if (nconnections() == 0) { return; }

  // change oport state and colors on screen
  if (module->show_stat) { turn_on(); }

  for (int i = 0; i < nconnections(); i++) {
    if (connections[i]->isRemote()) {
      
      // start timer here
#ifdef DEBUG
      timer1.clear();
      timer1.start();
#endif
      
      // send data - must only be binary files, text truncates and causes
      // problems when diffing outputs 
      Piostream *outstream= new BinaryPiostream(connections[i]->remSocket,
						Piostream::Write);
      if (!outstream) {
	perror ("Couldn't open outfile");
	exit (-1);
      }

      // stream data out
      Pio (*outstream, handle_);
      delete outstream;

    } else {
      SimplePortComm<T>* msg = new SimplePortComm<T>(data);
      ((SimpleIPort<T>*)connections[i]->iport)->mailbox.send(msg);
    }
  }
  sent_something_ = 1;

#ifdef DEBUG
  cerr << "Exiting SimpleOPort<T>::send (data)\n";
#endif

  if (module->show_stat) { turn_off(); }
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
  handle_=data;
  if(nconnections() == 0) { return; }
  if (module->show_stat) { turn_on(); }

  module->multisend(this);
  for(int i=0;i<nconnections();i++){
    SimplePortComm<T>* msg=new SimplePortComm<T>(data);
    ((SimpleIPort<T>*)connections[i]->iport)->mailbox.send(msg);
  }
  if (module->show_stat) { turn_off(); }
}

template<class T>
int SimpleIPort<T>::get(T& data)
{

#ifdef DEBUG
  cerr << "Entering SimpleIPort<T>::get (data)\n";
#endif

  if(nconnections()==0) { return 0; }
  if (module->show_stat) { turn_on(); }

  if (connections[0]->isRemote()) {

    // receive data - unmarshal data read from socket. no auto_istream as
    // it could try to mmap a file, which doesn't apply here.
    Piostream *instream = new BinaryPiostream(connections[0]->remSocket,
					      Piostream::Read);
    if (!instream) {
      perror ("Couldn't open infile");
      exit (-1);
    }
    Pio (*instream, data);
    delete instream;

    if (!data.get_rep()) {
      perror ("Error reading data from socket");
      exit (-1);
    }
	
#ifdef DEBUG
    cerr << "SimpleIPort<T>::get (data) read data from socket\n";
#endif
    if (module->show_stat) { turn_off(); }
    return 1;

  } else {

    // Wait for the data...
    SimplePortComm<T>* comm=mailbox.receive();
    recvd_=1;
    if(comm->have_data_){
      data=comm->data_;
      delete comm;
#ifdef DEBUG
      cerr << "SimpleIPort<T>::get (data) has data\n";
#endif
      if (module->show_stat) { turn_off(); }
      return 1;
    } else {
#ifdef DEBUG
      cerr << "SimpleIPort<T>::get (data) mailbox has no data\n";
#endif
      delete comm;
      if (module->show_stat) { turn_off(); }
      return 0;
    }
  }
}

template<class T>
int SimpleIPort<T>::special_get(T& data)
{


  if(nconnections()==0) { return 0; }
  if (module->show_stat) { turn_on(); }

  // Wait for the data...
  SimplePortComm<T>* comm;
  while(!mailbox.tryReceive(comm)){
    MessageBase* msg;
    if(module->mailbox.tryReceive(msg)){
      switch(msg->type){
      case MessageTypes::ExecuteModule:
	cerr << "Dropping execute...\n";
	break;
      case MessageTypes::TriggerPort:
	cerr << "Dropping trigger...\n";
	break;
      case MessageTypes::Demand:
	{
	  Demand_Message* dmsg=(Demand_Message*)msg;
	  if(dmsg->conn->oport->have_data()){
	    dmsg->conn->oport->resend(dmsg->conn);
	  } else {
	    cerr << "Dropping demand...\n";
	  }
	}
	break;
      default:
	cerr << "Illegal Message type: " << msg->type << endl;
	break;
      }
      delete msg;
    } else {
      sleep(1);
      // sginap(1);
    }
  }
	
  recvd_=1;
  if(comm->have_data_){
    data=comm->data_;
    delete comm;
    if (module->show_stat) turn_off();
    return 1;
  } else {
    delete comm;
    if (module->show_stat) turn_off();
    return 0;
  }
}

template<class T>
int SimpleOPort<T>::have_data()
{
  if(handle_.get_rep()) { return 1; }
  return 0;
}

template<class T>
void SimpleOPort<T>::resend(Connection* conn)
{
  if (module->show_stat) { turn_on(); }
  for(int i=0;i<nconnections();i++){
    if(connections[i] == conn){
      SimplePortComm<T>* msg=new SimplePortComm<T>(handle_);
      ((SimpleIPort<T>*)connections[i]->iport)->mailbox.send(msg);
    }
  }
  if (module->show_stat) { turn_off(); }
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
