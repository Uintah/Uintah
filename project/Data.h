
/*
 *  Data.h: Base class for data items
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_Data_h
#define SCI_project_Data_h 1

class Connection;
class clString;

class InData {
protected:
    int protocol;
public:
    Connection* connection;
    virtual clString typename()=0;
    InData(int protocol);
    int using_protocol();

    virtual void reset()=0;
    virtual void finish()=0;
};

class OutData {
protected:
    int protocol;
public:
    Connection* connection;
    virtual clString typename()=0;
    OutData(int protocol);
    int using_protocol();

    virtual void reset()=0;
    virtual void finish()=0;
};


#endif /* SCI_project_Data_h */
