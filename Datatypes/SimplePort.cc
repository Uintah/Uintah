
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
#include <Dataflow/Module.h>
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
#if 0
    if(!recvd && nconnections() > 0){
	turn_on(Finishing);
	SimplePortComm<T>* msg=mailbox.receive();
	delete msg;
	turn_off();
    }
#endif
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
	turn_off();
    }
}

template<class T>
void SimpleOPort<T>::send(const T& data)
{
    handle=data;
    if(nconnections() == 0)
	return;
#if 0
    if(sent_something){
	// Tell the scheduler that we are going to do this...
	cerr << "The data got sent twice - ignoring second one...\n";
	return;
    }
#endif
    turn_on();
    for(int i=0;i<nconnections();i++){
#if 0
        if(connections[i]->demand){
#endif
	    SimplePortComm<T>* msg=scinew SimplePortComm<T>(data);
	    ((SimpleIPort<T>*)connections[i]->iport)->mailbox.send(msg);
#if 0
	    connections[i]->demand--;
	} else {
	    // Advise the module of the change...
	  //connections[i]->iport->get_module()->mailbox.send(msg);
	}
#endif
    }
    sent_something=1;
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
    if(nconnections()==0)
	return 0;
    turn_on();
#if 0
    // Send the demand token...
    Connection* conn=connections[0];
    conn->oport->get_module()->mailbox.send(new Demand_Message(conn));
#endif

    // Wait for the data...
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
int SimpleIPort<T>::special_get(T& data)
{
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
	sginap(1);
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
