/* TODO:
  FFT should FFT the whole thing or packetize - a switch
    UI for packet length
  Multiple output ports.
    1 - Graph - or SoundSpectrum2Graph module???
    2 - SoundSpectrum.
 */
/*
 *  SoundFFT.cc:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Dataflow/ModuleList.h>
#include <Datatypes/SoundPort.h>
#include <Math/MinMax.h>
#include <Math/Trig.h>
#include <iostream.h>

class SoundFFT : public Module {
    SoundOPort* outsound;
    SoundIPort* insound;
public:
    SoundFFT(const clString& id);
    SoundFFT(const SoundFFT&, int deep);
    virtual ~SoundFFT();
    virtual Module* clone(int deep);
    virtual void execute();
};

static Module* make_SoundFFT(const clString& id)
{
    return new SoundFFT(id);
}

static RegisterModule db1("Sound", "SoundFFT", make_SoundFFT);

SoundFFT::SoundFFT(const clString& id)
: Module("SoundFFT", id, Filter)
{
    // Create the output data handle and port
#if 0
    add_oport(&outsound, "Sound", SoundData::Stream|SoundData::Atomic);

    // Create the input port
    add_iport(&isound, "Sound", SoundData::Stream|SoundData::Atomic);

    // Setup the execute condtion...
    execute_condition(NewDataOnAllConnectedPorts);
#endif
    NOT_FINISHED("SoundFFT::SoundFFT");
}

SoundFFT::SoundFFT(const SoundFFT& copy, int deep)
: Module(copy, deep)
{
    NOT_FINISHED("SoundFFT::SoundFFT");
}

SoundFFT::~SoundFFT()
{
}

Module* SoundFFT::clone(int deep)
{
    return new SoundFFT(*this, deep);
}

void SoundFFT::execute()
{
#if 0
    double rate=isound.sample_rate();
    int nsamples;
    if(isound.using_protocol() == SoundData::Atomic){
	nsamples=isound.nsamples();
    } else {
	nsamples=(int)(rate*10);
    }

    int window=512;
    double* samps=new double[window];
    int wp=0;
    int fill=0;
    while(!isound.end_of_stream() || (fill=1 && wp != 0)){
	// Read a sound sample
	double s;
	if(!fill){
	    s=isound.next_sample();
	} else {
	    s=0;
	}

	// Put it in the buffer...
	samps[wp++]=s;
	if(wp==window){
	    // Do the FFT and output it...
	    cerr << "FFT not finished!\n";
	    // Bit reverse the buffer...

	    wp=0;
	}

	// Tell everyone how we are doing...
	update_progress(wp, window);
    }
#endif
}
