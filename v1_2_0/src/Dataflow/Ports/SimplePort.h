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
    T data;
    int have_data;
};

template<class T> class SimpleOPort;

template<class T>
class SimpleIPort : public IPort {
    int recvd;
public:
    enum Protocol {
	Atomic=0x01
    };

public:
    friend class SimpleOPort<T>;
    Mailbox<SimplePortComm<T>*> mailbox;

    static string port_type;
    static string port_color;
public:
    SimpleIPort(Module*, const string& name, int protocol=Atomic);
    virtual ~SimpleIPort();
    virtual void reset();
    virtual void finish();

    int get(T&);
    int special_get(T&);
};

template<class T>
class SimpleOPort : public OPort {
    int sent_something;
    T handle;
#ifdef DEBUG
    WallClockTimer timer1;
#endif
public:
    SimpleOPort(Module*, const string& name, int protocol=SimpleIPort<T>::Atomic);
    virtual ~SimpleOPort();

    virtual void reset();
    virtual void finish();

    void send(const T&);
    void send_intermediate(const T&);

    virtual int have_data();
    virtual void resend(Connection* conn);
};

} // End namespace SCIRun

extern char** global_argv;

namespace SCIRun {


template<class T>
SimpleIPort<T>::SimpleIPort(Module* module, const string& portname,
			    int protocol)
: IPort(module, port_type, portname, port_color, protocol),
  mailbox("Port mailbox (SimpleIPort)", 2)
{
}

template<class T>
SimpleIPort<T>::~SimpleIPort()
{
}

template<class T>
SimpleOPort<T>::SimpleOPort(Module* module, const string& portname,
			    int protocol)
: OPort(module, SimpleIPort<T>::port_type, portname,
	SimpleIPort<T>::port_color, protocol)
{
}

template<class T>
SimpleOPort<T>::~SimpleOPort()
{
}

template<class T>
void SimpleIPort<T>::reset()
{
    recvd=0;
}

template<class T>
void SimpleIPort<T>::finish()
{
    if(!recvd && nconnections() > 0){
	if (module->show_stat) turn_on(Finishing);
	SimplePortComm<T>* msg=mailbox.receive();
	delete msg;
	if (module->show_stat) turn_off();
    }
}

template<class T>
void SimpleOPort<T>::reset()
{
    handle=0;
    sent_something=0;
}

template<class T>
void SimpleOPort<T>::finish()
{
    // get timestamp here to measure communication time, print to screen
#ifdef DEBUG
    timer1.stop();
    double time = 
                  timer1.time();
#endif

#ifdef DEBUG
    cerr << "Done in " << time << " seconds\n"; 
    cerr << "Entering SimpleOPort<T>::finish()\n";
#endif

    if(!sent_something && nconnections() > 0){
	// Tell them that we didn't send anything...
	if (module->show_stat) turn_on(Finishing);
	for(int i=0;i<nconnections();i++){
#if 0
	    if(connections[i]->demand){
#endif
	        SimplePortComm<T>* msg=new SimplePortComm<T>();
		((SimpleIPort<T>*)connections[i]->iport)->mailbox.send(msg);
#if 0
		connections[i]->demand--;
	    }
#endif
	}

#ifdef DEBUG
	cerr << "Exiting SimpleOPort<T>::finish()\n";
#endif
	if (module->show_stat) turn_off();
    }
}

template<class T>
void SimpleOPort<T>::send(const T& data)
{

#ifdef DEBUG
    cerr << "Entering SimpleOPort<T>::send (data)\n";
#endif

    handle = data;
    if (nconnections() == 0)
	return;
#if 0
    if(sent_something){
      // Tell the scheduler that we are going to do this...
      cerr << "The data got sent twice - ignoring second one...\n";
    }
#endif

    // change oport state and colors on screen
    if (module->show_stat) turn_on();

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
            Pio (*outstream, handle);
   	    delete outstream;

	} else {
            SimplePortComm<T>* msg = new SimplePortComm<T>(data);
            ((SimpleIPort<T>*)connections[i]->iport)->mailbox.send(msg);
	}
    }
    sent_something = 1;

#ifdef DEBUG
    cerr << "Exiting SimpleOPort<T>::send (data)\n";
#endif

    if (module->show_stat) turn_off();
}

template<class T>
void SimpleOPort<T>::send_intermediate(const T& data)
{
    handle=data;
    if(nconnections() == 0)
	return;
    if (module->show_stat) turn_on();
    module->multisend(this);
    for(int i=0;i<nconnections();i++){
	SimplePortComm<T>* msg=new SimplePortComm<T>(data);
	((SimpleIPort<T>*)connections[i]->iport)->mailbox.send(msg);
    }
    if (module->show_stat) turn_off();
}

template<class T>
int SimpleIPort<T>::get(T& data)
{

#ifdef DEBUG
    cerr << "Entering SimpleIPort<T>::get (data)\n";
#endif

    if(nconnections()==0)
	return 0;
    if (module->show_stat) turn_on();

#if 0
    // Send the demand token...
    Connection* conn=connections[0];
    conn->oport->get_module()->mailbox.send(new Demand_Message(conn));
#endif

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
	if (module->show_stat) turn_off();
	return 1;
    } else {

    	// Wait for the data...
        SimplePortComm<T>* comm=mailbox.receive();
        recvd=1;
        if(comm->have_data){
            data=comm->data;
            delete comm;
#ifdef DEBUG
	    cerr << "SimpleIPort<T>::get (data) has data\n";
#endif
            if (module->show_stat) turn_off();
            return 1;
        } else {
#ifdef DEBUG
	    cerr << "SimpleIPort<T>::get (data) mailbox has no data\n";
#endif
            delete comm;
            if (module->show_stat) turn_off();
            return 0;
        }
    }
}

template<class T>
int SimpleIPort<T>::special_get(T& data)
{


    if(nconnections()==0)
	return 0;
    if (module->show_stat) turn_on();
#if 0
    // Send the demand token...
    Connection* conn=connections[0];
    conn->oport->get_module()->mailbox.send(new Demand_Message(conn));
#endif

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
	
    recvd=1;
    if(comm->have_data){
       data=comm->data;
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
    if(handle.get_rep())
	return 1;
    else
	return 0;
}

template<class T>
void SimpleOPort<T>::resend(Connection* conn)
{
    if (module->show_stat) turn_on();
    for(int i=0;i<nconnections();i++){
	if(connections[i] == conn){
	    SimplePortComm<T>* msg=new SimplePortComm<T>(handle);
	    ((SimpleIPort<T>*)connections[i]->iport)->mailbox.send(msg);
	}
    }
    if (module->show_stat) turn_off();
}

template<class T>
SimplePortComm<T>::SimplePortComm()
: have_data(0)
{
}

template<class T>
SimplePortComm<T>::SimplePortComm(const T& data)
: data(data), have_data(1)
{
}

} // End namespace SCIRun


#endif /* SCI_project_SimplePort_h */
