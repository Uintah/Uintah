
/*
 *  Port.h: Classes for module ports
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_Port_h
#define SCI_project_Port_h 1

#include <Classlib/Array1.h>
#include <Classlib/String.h>
class Connection;
class Module;

class Port {
    clString typename;
    clString portname;
    clString colorname;
    int protocols;
    int u_proto;
protected:
    Module* module;
    int which_port;
    Array1<Connection*> connections;
    int xlight, ylight;
    enum PortState {
	Off,
	Resetting,
	Finishing,
	On,
    };
    PortState portstate;
public:
    Port(Module*, const clString&, const clString&,
	 const clString&, int protocols);
    virtual ~Port();
    void set_port(int which_port);
    int using_protocol();
    int nconnections();
    Connection* connection(int);
    Module* get_module();
    int get_which_port();
    void set_which_port(int);
    void attach(Connection*);
    void detach(Connection*);
    virtual void reset()=0;
    virtual void finish()=0;
    clString get_typename();
    clString get_portname();
    clString get_colorname();
};

class IPort : public Port {
    void update_light();
protected:
    IPort(Module*, const clString&, const clString&,
	  const clString&, int protocols);
    void turn_on(PortState st=On);
    void turn_off();
public:
    virtual ~IPort();
};

class OPort : public Port {    
    void update_light();
protected:
    OPort(Module*, const clString&, const clString&,
	  const clString&, int protocols);
    void turn_on(PortState st=On);
    void turn_off();
public:
    virtual ~OPort();
};

#endif /* SCI_project_Port_h */
