#ifndef SCI_project_SimplePort_h
#define SCI_project_SimplePort_h 1

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

#include <Dataflow/Port.h>
#include <Multitask/ITC.h>
#include <Util/Timer.h>

namespace PSECommon {

namespace Dataflow {
  class Module;
}

namespace CommonDatatypes {

using PSECommon::Dataflow::Module;
using PSECommon::Dataflow::IPort;
using PSECommon::Dataflow::OPort;
using PSECommon::Dataflow::Connection;
using SCICore::Multitask::Mailbox;
using SCICore::Containers::clString;

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

    static clString port_type;
    static clString port_color;
public:
    SimpleIPort(Module*, const clString& name, int protocol=Atomic);
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
    WallClockTimer timer1;
public:
    SimpleOPort(Module*, const clString& name, int protocol=SimpleIPort<T>::Atomic);
    virtual ~SimpleOPort();

    virtual void reset();
    virtual void finish();

    void send(const T&);
    void send_intermediate(const T&);

    virtual int have_data();
    virtual void resend(Connection* conn);
};

} // End namespace CommonDatatypes
} // End namespace PSECommon

////////////////////////////////////////////////////////////
//
// Start of included SimplePort.cc
//

#include <Persistent/Pstreams.h>

#include <Util/Assert.h>
#include <Dataflow/Connection.h>
#include <Dataflow/Module.h>
#include <Malloc/Allocator.h>
#include <TclInterface/Remote.h>

#include <stdio.h>
#include <iostream.h>
#include <fstream.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <fcntl.h>
#include <unistd.h>

//#define DEBUG 1

extern char** global_argv;

namespace PSECommon {
namespace CommonDatatypes {

using namespace SCICore::PersistentSpace;

template<class T>
SimpleIPort<T>::SimpleIPort(Module* module, const clString& portname,
			    int protocol)
: IPort(module, port_type, portname, port_color, protocol),
  mailbox(2)
{
}

template<class T>
SimpleIPort<T>::~SimpleIPort()
{
}

template<class T>
SimpleOPort<T>::SimpleOPort(Module* module, const clString& portname,
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
	turn_on(Finishing);
	SimplePortComm<T>* msg=mailbox.receive();
	delete msg;
	turn_off();
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
    using SCICore::PersistentSpace::Pio;
    using SCICore::Containers::Pio;

    // get timestamp here to measure communication time, print to screen
    timer1.stop();
#ifdef DEBUG
    double time = 
#endif
                  timer1.time();

#ifdef DEBUG
    cerr << "Done in " << time << " seconds\n"; 
    cerr << "Entering SimpleOPort<T>::finish()\n";
#endif

    if(!sent_something && nconnections() > 0){
	// Tell them that we didn't send anything...
	turn_on(Finishing);
	for(int i=0;i<nconnections();i++){
#if 0
	    if(connections[i]->demand){
#endif
	        SimplePortComm<T>* msg=scinew SimplePortComm<T>();
		((SimpleIPort<T>*)connections[i]->iport)->mailbox.send(msg);
#if 0
		connections[i]->demand--;
	    }
#endif
	}

#ifdef DEBUG
	cerr << "Exiting SimpleOPort<T>::finish()\n";
#endif
	turn_off();
    }
}

template<class T>
void SimpleOPort<T>::send(const T& data)
{
    using SCICore::PersistentSpace::Pio;
    using SCICore::Containers::Pio;

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
    turn_on();

    for (int i = 0; i < nconnections(); i++) {
	if (connections[i]->isRemote()) {

	    // start timer here
	    timer1.clear();
	    timer1.start();

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
            SimplePortComm<T>* msg = scinew SimplePortComm<T>(data);
            ((SimpleIPort<T>*)connections[i]->iport)->mailbox.send(msg);
	}
    }
    sent_something = 1;

#ifdef DEBUG
    cerr << "Exiting SimpleOPort<T>::send (data)\n";
#endif

    turn_off();
}

template<class T>
void SimpleOPort<T>::send_intermediate(const T& data)
{
    handle=data;
    if(nconnections() == 0)
	return;
    turn_on();
    module->multisend(this);
    for(int i=0;i<nconnections();i++){
	SimplePortComm<T>* msg=scinew SimplePortComm<T>(data);
	((SimpleIPort<T>*)connections[i]->iport)->mailbox.send(msg);
    }
    turn_off();
}

template<class T>
int SimpleIPort<T>::get(T& data)
{
    using SCICore::PersistentSpace::Pio;
    using SCICore::Containers::Pio;

#ifdef DEBUG
    cerr << "Entering SimpleIPort<T>::get (data)\n";
#endif

    if(nconnections()==0)
	return 0;
    turn_on();

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
	turn_off();
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
            turn_off();
            return 1;
        } else {
#ifdef DEBUG
	    cerr << "SimpleIPort<T>::get (data) mailbox has no data\n";
#endif
            delete comm;
            turn_off();
            return 0;
        }
    }
}

template<class T>
int SimpleIPort<T>::special_get(T& data)
{
    using SCICore::PersistentSpace::Pio;
    using SCICore::Containers::Pio;

    using PSECommon::Comm::MessageTypes;
    using PSECommon::Dataflow::Demand_Message;

    if(nconnections()==0)
	return 0;
    turn_on();
#if 0
    // Send the demand token...
    Connection* conn=connections[0];
    conn->oport->get_module()->mailbox.send(new Demand_Message(conn));
#endif

    // Wait for the data...
    SimplePortComm<T>* comm;
    while(!mailbox.try_receive(comm)){
      MessageBase* msg;
      if(module->mailbox.try_receive(msg)){
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
       turn_off();
       return 1;
   } else {
       delete comm;
       turn_off();
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
    turn_on();
    for(int i=0;i<nconnections();i++){
	if(connections[i] == conn){
	    SimplePortComm<T>* msg=scinew SimplePortComm<T>(handle);
	    ((SimpleIPort<T>*)connections[i]->iport)->mailbox.send(msg);
	}
    }
    turn_off();
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

} // End namespace CommonDatatypes
} // End namespace PSECommon

//
// $Log$
// Revision 1.1  1999/07/27 16:55:50  mcq
// Initial commit
//
// Revision 1.4  1999/07/07 21:10:20  dav
// added beginnings of support for g++ compilation
//
// Revision 1.3  1999/05/06 20:17:03  dav
// added back PSECommon .h files
//
// Revision 1.1.1.1  1999/04/24 23:12:47  dav
// Import sources
//
//

#endif /* SCI_project_SimplePort_h */
