
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

class IPort;
class OPort;

class Connection {
public:
    Connection(OPort*, IPort*);
    ~Connection();

    OPort* oport;
    IPort* iport;
    int local;

    // Data members for the NetworkEdtitor
    void* drawing_a[5];
};

#endif /* SCI_project_Connection_h */

