
/*
 *  SimplePort.cc: Handle to the Simple Data type
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <SimplePort.h>
#include <Connection.h>
#include <Classlib/Assert.h>
#include <iostream.h>

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
	SimpleIPort<T>::port_color, protocol), in(0)
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
    sent_something=0;
}

template<class T>
void SimpleOPort<T>::finish()
{
    if(!sent_something && nconnections() > 0){
	// Tell them that we didn't send anything...
	turn_on(Finishing);
	if(!in){
	    Connection* connection=connections[0];
	    in=(SimpleIPort<T>*)connection->iport;
	}
	SimplePortComm<T>* msg=new SimplePortComm<T>();
	in->mailbox.send(msg);
	turn_off();
    }
}

template<class T>
void SimpleOPort<T>::send(const T& data)
{
    if(!in){
	Connection* connection=connections[0];
	in=(SimpleIPort<T>*)connection->iport;
    }
    if(sent_something){
	cerr << "The data got sent twice - ignoring second one...\n";
	return;
    }
    turn_on();
    SimplePortComm<T>* msg=new SimplePortComm<T>(data);
    in->mailbox.send(msg);
    sent_something=1;
    turn_off();
}

template<class T>
int SimpleIPort<T>::get(T& data)
{
    turn_on();
    SimplePortComm<T>* comm=mailbox.receive();
    if(comm->have_data){
       data=comm->data;
       recvd=1;
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
SimplePortComm<T>::SimplePortComm()
: have_data(0)
{
}

template<class T>
SimplePortComm<T>::SimplePortComm(const T& data)
: data(data), have_data(1)
{
}
