
/*
 *  SoundData.cc: Handle to the Sound Data type
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <SoundData.h>
#include <Connection.h>
#include <NotFinished.h>
#include <Port.h>
#include <Classlib/Assert.h>
#include <Classlib/String.h>
#include <iostream.h>

static clString sound_type("Sound");

clString InSoundData::typename()
{
    return sound_type;
}

clString OutSoundData::typename()
{
    return sound_type;
}


InSoundData::InSoundData()
: InData(SoundData::NotNegotiated), state(Begin),
  mailbox(10),
  sample_buf(0)
{
}

InSoundData::~InSoundData()
{
    if(sample_buf)
	delete[] sample_buf;
}

void InSoundData::reset()
{
    state=Begin;
}

void InSoundData::finish()
{
    if(state != Done){
	cerr << "Not all of sound was read...\n";
    }
}

int InSoundData::nsamples()
{
    if(protocol == SoundData::Stream){
	cerr << "SoundData error: nsamples requested when using Stream protocol\n";
	return 0;
    }
    while(state == Begin)
	do_read();
    return total_samples;
}

double InSoundData::sample_rate()
{
    while(state == Begin)
	do_read();

    return rate;
}

double InSoundData::next_sample()
{
    if(state != HaveSamples)
	do_read();
    double s=sample_buf[bufp++];
    if(bufp>=sbufsize){
	state=NeedSamples;
	if(state != HaveSamples)
	    do_read();
	bufp=0;
    }
    return s;
}

int InSoundData::end_of_stream()
{
    return state==Done;
}

void InSoundData::do_read()
{
    SoundComm* comm;
    if(connection->local){
	comm=mailbox.receive();
    } else {
	NOT_FINISHED("read from nonlocal connection");
    }
    switch(comm->action){
    case SoundComm::Parameters:
	ASSERT(state==Begin);
	total_samples=comm->nsamples;
	rate=comm->sample_rate;
	recvd_samples=0;
	state=NeedSamples;
	break;
    case SoundComm::SoundData:
	ASSERT(state==NeedSamples);
	if(sample_buf)
	    delete[] sample_buf;
	sample_buf=comm->samples;
	sbufsize=comm->sbufsize;
	recvd_samples+=sbufsize;
	state=HaveSamples;
	break;
    case SoundComm::EndOfStream:
	ASSERT(state==NeedSamples);
	state=Done;
	total_samples=recvd_samples;
	break;
    }
}

OutSoundData::OutSoundData()
: OutData(SoundData::NotNegotiated), state(Begin),
  sbuf(0), in(0), ptr(0)
{
}

OutSoundData::~OutSoundData()
{
    if(sbuf)
	delete[] sbuf;
}

void OutSoundData::reset()
{
    state=Begin;
    total_samples=0;
    rate=-1;
}

void OutSoundData::finish()
{
    // Flush the stream and send an end of stream marker...
    if(ptr != 0){
	if(connection->local){
	    SoundComm* comm=new SoundComm;
	    comm->action=SoundComm::SoundData;
	    comm->sbufsize=ptr;
	    comm->samples=sbuf;
	    ASSERT(in != 0);
	    in->mailbox.send(comm);
	} else {
	    NOT_FINISHED("Nonlocal send");
	}
	sbuf=0;
	ptr=0;
    }
    SoundComm* comm=new SoundComm;
    if(connection->local){
	comm->action=SoundComm::EndOfStream;
	in->mailbox.send(comm);
    } else {
	NOT_FINISHED("Nonlocal send");
    }
    state=End;
}

void OutSoundData::set_nsamples(int s)
{
    ASSERT(state == Begin);
    total_samples=s;
}

void OutSoundData::set_sample_rate(double r)
{
    ASSERT(state == Begin);
    rate=r;
}

void OutSoundData::put_sample(double s)
{
    ASSERT(state != End);
    if(state == Begin){
	// Send the Parameters message...
	if(connection->local){
	    SoundComm* comm=new SoundComm;
	    comm->action=SoundComm::Parameters;
	    comm->sample_rate=rate;
	    comm->nsamples=total_samples;
	    if(!in){
		in=(InSoundData*)connection->iport->data;
	    }
	    in->mailbox.send(comm);
	} else {
	    NOT_FINISHED("Non-local send");
	}
	state=Transmitting;
	sbufsize=(int)(rate/20);
	ptr=0;
    }
    if(!sbuf){
	sbuf=new double[sbufsize];
	ptr=0;
    }
    sbuf[ptr++]=s;
    if(ptr >= sbufsize){
	// Send it away...
	if(connection->local){
	    SoundComm* comm=new SoundComm;
	    comm->action=SoundComm::SoundData;
	    comm->sbufsize=sbufsize;
	    comm->samples=sbuf;
	    ASSERT(in != 0);
	    in->mailbox.send(comm);
	} else {
	    NOT_FINISHED("Nonlocal send");
	}
	sbuf=0;
	ptr=0;
    }
}
