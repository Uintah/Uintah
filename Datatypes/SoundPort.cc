
/*
 *  SoundPort.cc: Handle to the Sound Data type
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Datatypes/SoundPort.h>

#include <Classlib/Assert.h>
#include <Classlib/NotFinished.h>
#include <Classlib/String.h>
#include <Dataflow/Connection.h>
#include <Dataflow/Port.h>
#include <Malloc/Allocator.h>

#include <iostream.h>

static clString sound_type("Sound");
static clString sound_color("aquamarine4");

SoundIPort::SoundIPort(Module* module, const clString& portname, int protocol)
: IPort(module, sound_type, portname, sound_color, protocol),
  state(Begin), sample_buf(0), mailbox(10)
{
}

SoundIPort::~SoundIPort()
{
    if(sample_buf)
	delete[] sample_buf;
}

void SoundOPort::reset()
{
    state=Begin;
    total_samples=0;
    rate=-1;
    stereo=0;
}

SoundOPort::SoundOPort(Module* module, const clString& portname, int protocol)
: OPort(module, sound_type, portname, sound_color, protocol),
  sbuf(0), ptr(0), in(0)
{
}

SoundOPort::~SoundOPort()
{
    if(sbuf)
	delete[] sbuf;
}

void SoundIPort::reset()
{
    state=Begin;
    bufp=0;
}

void SoundIPort::finish()
{
    if(state != Done){
	cerr << "Not all of sound was read...\n";
	state=Flushing;
	while(state != Done)
	    do_read(1);
    }
    if(sample_buf){
	delete[] sample_buf;
	sample_buf=0;
    }
}

int SoundIPort::nsamples()
{
    if(using_protocol() == SoundIPort::Stream){
	cerr << "SoundIPort error: nsamples requested when using Stream protocol\n";
	return 0;
    }
    while(state == Begin)
	do_read(0);
    return total_samples;
}

double SoundIPort::sample_rate()
{
    while(state == Begin)
	do_read(0);

    return rate;
}

int SoundIPort::is_stereo()
{
    while(state == Begin)
	do_read(0);

    return stereo;
}

void SoundIPort::do_read(int fin)
{
    if(fin)
	turn_on(Finishing);
    else
	turn_on();
    SoundComm* comm;
    comm=mailbox.receive();
    switch(comm->action){
    case SoundComm::Parameters:
	ASSERT(state==Begin);
	total_samples=comm->nsamples;
	rate=comm->sample_rate;
	stereo=comm->stereo;
	recvd_samples=0;
	state=NeedSamples;
	break;
    case SoundComm::SoundData:
	ASSERT(state==NeedSamples || state==Flushing);
	if(sample_buf)
	    delete[] sample_buf;
	sample_buf=comm->samples;
	sbufsize=comm->sbufsize;
	recvd_samples+=sbufsize;
	if(state==NeedSamples)state=HaveSamples;
	break;
    case SoundComm::EndOfStream:
	ASSERT(state==NeedSamples || state==Flushing || state==Begin);
	state=Done;
	total_samples=recvd_samples;
	break;
    }
    delete comm;
    turn_off();
}

void SoundOPort::finish()
{
    // Flush the stream and send an end of stream marker...
    turn_on();
    if(!in){
	Connection* connection=connections[0];
	in=(SoundIPort*)connection->iport;
    }
    if(ptr != 0){
	SoundComm* comm=scinew SoundComm;
	comm->action=SoundComm::SoundData;
	comm->sbufsize=ptr;
	comm->samples=sbuf;
	in->mailbox.send(comm);
	sbuf=0;
	ptr=0;
    }
    SoundComm* comm=scinew SoundComm;
    comm->action=SoundComm::EndOfStream;
    in->mailbox.send(comm);
    state=End;
    turn_off();
}

void SoundOPort::set_nsamples(int s)
{
    ASSERT(state == Begin);
    total_samples=s;
}

void SoundOPort::set_sample_rate(double r)
{
    ASSERT(state == Begin);
    rate=r;
}

void SoundOPort::set_stereo(int s)
{
    ASSERT(state == Begin);
    stereo=s;
}

void SoundOPort::put_sample(double s)
{
    ASSERT(state != End);
    if(state == Begin){
	// Send the Parameters message...
	turn_on();
	SoundComm* comm=scinew SoundComm;
	comm->action=SoundComm::Parameters;
	comm->sample_rate=rate;
	comm->nsamples=total_samples;
	comm->stereo=stereo;
	if(!in){
	    Connection* connection=connections[0];
	    in=(SoundIPort*)connection->iport;
	}
	in->mailbox.send(comm);
	state=Transmitting;
	sbufsize=(int)(rate/10);
	ptr=0;
	turn_off();
    }
    if(!sbuf){
	sbuf=scinew double[sbufsize];
	ptr=0;
    }
    sbuf[ptr++]=s;
    if(ptr >= sbufsize){
	// Send it away...
	turn_on();
	SoundComm* comm=scinew SoundComm;
	comm->action=SoundComm::SoundData;
	comm->sbufsize=sbufsize;
	comm->samples=sbuf;
	ASSERT(in != 0);
	in->mailbox.send(comm);
	sbuf=0;
	ptr=0;
	turn_off();
    }
}

void SoundOPort::put_sample(double sl, double sr)
{
    put_sample(sl);
    put_sample(sr);
}

int SoundOPort::have_data()
{
    return 0;
}

void SoundOPort::resend(Connection*)
{
    cerr << "SoundOPort can't resend and shouldn't be asked to!\n";
}

#ifdef __GNUG__

#include <Multitask/Mailbox.cc>

template class Mailbox<SoundComm*>;

#endif

