
/*
 *  Connection.h: A Connection between two modules
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_Connection_h
#define SCI_project_Connection_h 1

#include <Classlib/String.h>
class IPort;
class Module;
class OPort;

class Connection {
    int connected;
public:
    Connection(Module*, int, Module*, int);
    ~Connection();
    void attach(OPort*);
    void attach(IPort*);

    OPort* oport;
    IPort* iport;
    int local;
    clString id;

    void wait_ready();

    void connect();
};

#endif /* SCI_project_Connection_h */
