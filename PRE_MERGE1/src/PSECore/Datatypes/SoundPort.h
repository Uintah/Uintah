
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

#include <Dataflow/Port.h>
#include <Multitask/ITC.h>

namespace PSECommon {
namespace CommonDatatypes {

using PSECommon::Dataflow::IPort;
using PSECommon::Dataflow::OPort;
using PSECommon::Dataflow::Module;
using PSECommon::Dataflow::Connection;
using SCICore::Multitask::Mailbox;
using SCICore::Containers::clString;

struct SoundComm {
    enum Action {
	Parameters,
	SoundData,
	EndOfStream
    };
    Action action;
    double sample_rate;
    int nsamples;
    int stereo;
    int sbufsize;
    double* samples;
};

class SoundIPort : public IPort {
public:
    enum Protocol {
	Atomic=0x01,
	Stream=0x02,
	NotNegotiated=0x10000
    };

private:
    enum State {
	Begin,
	Done,
	NeedSamples,
	HaveSamples,
	Flushing
    };
    State state;
    int total_samples;
    double rate;
    int stereo;
    int sbufsize;
    double* sample_buf;
    int bufp;
    int recvd_samples;
    void do_read(int fin);
protected:
    friend class SoundOPort;
    Mailbox<SoundComm*> mailbox;
public:
    SoundIPort(Module*, const clString& name, int protocol);
    virtual ~SoundIPort();
    int nsamples();
    double sample_rate();
    inline double next_sample();
    inline int end_of_stream();
    int is_stereo();

    virtual void reset();
    virtual void finish();
};

class SoundOPort : public OPort {
    int total_samples;
    double rate;
    int stereo;
    enum State {
	Begin,
	Transmitting,
	End
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
    void set_stereo(int);
    void put_sample(double);
    void put_sample(double, double);

    virtual void reset();
    virtual void finish();

    virtual int have_data();
    virtual void resend(Connection*);
};

inline double SoundIPort::next_sample()
{
    if(state != HaveSamples)
	do_read(0);
    double s=sample_buf[bufp++];
    if(bufp>=sbufsize){
	state=NeedSamples;
	if(state != HaveSamples)
	    do_read(0);
	bufp=0;
    }
    return s;
}

inline int SoundIPort::end_of_stream()
{
    return state==Done;
}

} // End namespace CommonDatatypes
} // End namespace PSECommon

//
// $Log$
// Revision 1.1  1999/07/27 16:55:50  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 20:17:03  dav
// added back PSECommon .h files
//
// Revision 1.1.1.1  1999/04/24 23:12:47  dav
// Import sources
//
//

#endif /* SCI_project_SoundPort_h */
