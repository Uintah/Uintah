//static char *id="@(#) $Id$";

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

#include <PSECore/Datatypes/SoundPort.h>

#include <SCICore/Util/Assert.h>
#include <SCICore/Containers/String.h>
#include <PSECore/Dataflow/Connection.h>
#include <PSECore/Dataflow/Port.h>
#include <PSECore/Dataflow/Module.h>
#include <SCICore/Malloc/Allocator.h>

#include <iostream>
using std::cerr;

namespace PSECore {
namespace Datatypes {

static clString sound_type("Sound");
static clString sound_color("aquamarine4");

SoundIPort::SoundIPort(Module* module, const clString& portname, int protocol)
: IPort(module, sound_type, portname, sound_color, protocol),
  state(Begin), sample_buf(0), mailbox("Sound port FIFO", 10)
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
	if (module->show_status) turn_on(Finishing);
    else
	if (module->show_status) turn_on();
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
    if (module->show_status) turn_off();
}

void SoundOPort::finish()
{
    // Flush the stream and send an end of stream marker...
    if (module->show_status) turn_on();
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
    if (module->show_status) turn_off();
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
	if (module->show_status) turn_on();
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
	if (module->show_status) turn_off();
    }
    if(!sbuf){
	sbuf=scinew double[sbufsize];
	ptr=0;
    }
    sbuf[ptr++]=s;
    if(ptr >= sbufsize){
	// Send it away...
	if (module->show_status) turn_on();
	SoundComm* comm=scinew SoundComm;
	comm->action=SoundComm::SoundData;
	comm->sbufsize=sbufsize;
	comm->samples=sbuf;
	ASSERT(in != 0);
	in->mailbox.send(comm);
	sbuf=0;
	ptr=0;
	if (module->show_status) turn_off();
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

} // End namespace Datatypes
} // End namespace PSECore

//
// $Log$
// Revision 1.8  1999/11/12 00:57:33  dmw
// had to include Module.h
//
// Revision 1.7  1999/11/11 19:56:37  dmw
// added show_status check for GeometryPort and SoundPort
//
// Revision 1.6  1999/10/07 02:07:21  sparker
// use standard iostreams and complex type
//
// Revision 1.5  1999/08/28 17:54:32  sparker
// Integrated new Thread library
//
// Revision 1.4  1999/08/25 03:48:23  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.3  1999/08/19 23:18:03  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.2  1999/08/17 06:38:12  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:55:50  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:50  dav
// Import sources
//
//

