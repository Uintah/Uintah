
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

#include <Datatypes/SimplePort.h>
#include <Classlib/Assert.h>
#include <Dataflow/Connection.h>
#include <Malloc/Allocator.h>

#include <iostream.h>

#ifdef __GNUG__
#pragma interface
#endif

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
    sent_something=0;
}

template<class T>
void SimpleOPort<T>::finish()
{
    if(!sent_something && nconnections() > 0){
	// Tell them that we didn't send anything...
	turn_on(Finishing);
	for(int i=0;i<nconnections();i++){
	    SimplePortComm<T>* msg=scinew SimplePortComm<T>();
	    ((SimpleIPort<T>*)connections[i]->iport)->mailbox.send(msg);
	}
	turn_off();
    }
}

template<class T>
void SimpleOPort<T>::send(const T& data)
{
    if(nconnections() == 0)
	return;
    if(sent_something){
	cerr << "The data got sent twice - ignoring second one...\n";
	return;
    }
    turn_on();
    for(int i=0;i<nconnections();i++){
	SimplePortComm<T>* msg=scinew SimplePortComm<T>(data);
	((SimpleIPort<T>*)connections[i]->iport)->mailbox.send(msg);
    }
    sent_something=1;
    turn_off();
}

template<class T>
int SimpleIPort<T>::get(T& data)
{
    if(nconnections()==0)
	return 0;
    turn_on();
    SimplePortComm<T>* comm=mailbox.receive();
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
SimplePortComm<T>::SimplePortComm()
: have_data(0)
{
}

template<class T>
SimplePortComm<T>::SimplePortComm(const T& data)
: data(data), have_data(1)
{
}
