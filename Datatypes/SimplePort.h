
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

#ifndef SCI_project_SimplePort_h
#define SCI_project_SimplePort_h 1

#ifdef __GNUG__
#pragma interface
#endif

#include <Dataflow/Port.h>
#include <Multitask/ITC.h>
#include <Classlib/Timer.h>

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

#endif /* SCI_project_SimplePort_h */
