
/*
 *  SoundData.h: Handle to the Sound Data type
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_SoundData_h
#define SCI_project_SoundData_h 1

#include <Data.h>
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

class SoundData {
public:
    enum Protocol {
	Atomic=0x01,
	Stream=0x02,
	NotNegotiated=0x10000,
    };
};

class InSoundData : public InData {
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
    friend class OutSoundData;
    Mailbox<SoundComm*> mailbox;
public:
    InSoundData();
    ~InSoundData();
    virtual clString typename();
    int nsamples();
    double sample_rate();
    double next_sample();
    int end_of_stream();

    virtual void reset();
    virtual void finish();
};

class OutSoundData : public OutData {
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
    InSoundData* in;
public:
    OutSoundData();
    ~OutSoundData();
    virtual clString typename();
    void set_nsamples(int);
    void set_sample_rate(double);
    void put_sample(double);

    virtual void reset();
    virtual void finish();
};

#endif /* SCI_project_SoundData_h */
