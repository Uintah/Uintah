
/*
 *  SoundPort.h: Handle to the Sound Data type
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_SoundPort_h
#define SCI_project_SoundPort_h 1

#include <Port.h>
#include <Multitask/ITC.h>

struct SoundComm {
    enum Action {
	Parameters,
	SoundData,
	EndOfStream,
    };
    Action action;
    double sample_rate;
    int nsamples;
    int sbufsize;
    double* samples;
};

class SoundIPort : public IPort {
public:
    enum Protocol {
	Atomic=0x01,
	Stream=0x02,
	NotNegotiated=0x10000,
    };

private:
    enum State {
	Begin,
	Done,
	NeedSamples,
	HaveSamples,
    };
    State state;
    int total_samples;
    double rate;
    int sbufsize;
    double* sample_buf;
    int bufp;
    int recvd_samples;
    void do_read();
protected:
    friend class SoundOPort;
    Mailbox<SoundComm*> mailbox;
public:
    SoundIPort(Module*, const clString& name, int protocol);
    virtual ~SoundIPort();
    int nsamples();
    double sample_rate();
    double next_sample();
    int end_of_stream();

    virtual void reset();
    virtual void finish();
};

class SoundOPort : public OPort {
    int total_samples;
    double rate;
    enum State {
	Begin,
	Transmitting,
	End,
    };
    State state;
    int sbufsize;
    double* sbuf;
    int ptr;
    SoundIPort* in;
public:
    SoundOPort(Module*, const clString& name, int protocol);
    virtual ~SoundOPort();
    void set_nsamples(int);
    void set_sample_rate(double);
    void put_sample(double);

    virtual void reset();
    virtual void finish();
};

#endif /* SCI_project_SoundPort_h */
