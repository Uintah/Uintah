
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
class Datatype;
class Module;
class InData;
class OutData;

class Port {
    Array1<Connection*> connections;
public:
    Port(Module*, int, const clString&, const clString&);
    clString name;
    Module* module;
    int which_port;
    void attach(Connection*);
    int nconnections();
    Connection* connection(int);
    Datatype* datatype;
};

class IPort : public Port {
public:
    IPort(Module*, int, InData*, const clString&);
    InData* data;
};

class OPort : public Port {
public:
    OPort(Module*, int, OutData*, const clString&);
    OutData* data;
};


#endif /* SCI_project_Port_h */
