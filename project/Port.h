
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
class ColorManager;
class Connection;
class Module;
class XQColor;

class Port {
    Module* module;
    int which_port;
    clString typename;
    clString portname;
    clString colorname;
    int protocols;
    int u_proto;
protected:
    Array1<Connection*> connections;
public:
    Port(Module*, const clString&, const clString&,
	 const clString&, int protocols);
    void set_port(int which_port);
    int using_protocol();
    int nconnections();
    Connection* connection(int);
    Module* get_module();
    void attach(Connection*);
    void detach(Connection*);
    virtual void reset()=0;
    virtual void finish()=0;
    void get_colors(ColorManager*);
    XQColor* bgcolor;
    XQColor* top_shadow;
    XQColor* bottom_shadow;
};

class IPort : public Port {
public:
    IPort(Module*, const clString&, const clString&,
	  const clString&, int protocols);
};

class OPort : public Port {
public:
    OPort(Module*, const clString&, const clString&,
	  const clString&, int protocols);
};

#endif /* SCI_project_Port_h */
