
/*
 *  SoundReader.cc:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <SoundReader/SoundReader.h>
#include <Math/MinMax.h>
#include <MUI.h>
#include <NotFinished.h>
#include <SoundPort.h>
#include <iostream.h>
#include <audiofile.h>
#include <fstream.h>

SoundReader::SoundReader()
: UserModule("SoundReader", Source)
{
    // Create the output data handle and port
    osound=new SoundOPort(this, "Sound", SoundIPort::Stream);
    add_oport(osound);

    // Set up the interface
    MUI_onoff_switch* oo=new MUI_onoff_switch("Input",
					      &onoff, 0);
    add_ui(oo);
    filename="isound.aifc";
}

SoundReader::SoundReader(const SoundReader& copy, int deep)
: UserModule(copy, deep)
{
    NOT_FINISHED("SoundReader::SoundReader");
}

SoundReader::~SoundReader()
{
}

Module* make_SoundReader()
{
    return new SoundReader;
}

Module* SoundReader::clone(int deep)
{
    return new SoundReader(*this, deep);
}

void SoundReader::execute()
{
    // Open the sound port...
#if 0
    AFfilehandle afile=AFopenfile(filename(), "r", 0);
    if(!afile){
	cerr << "Error opening file: " << afile << endl;
	return;
    }
    long nchannels=AFgetchannels(afile, AF_DEFAULT_TRACK);
    // Setup the sampling rate...
    double rate=AFgetrate(afile, AF_DEFAULT_TRACK);
    outsound.set_sample_rate(rate);

    long nsamples=AFgetframecnt(afile, AF_DEFAULT_TRACK);
    outsound.set_nsamples((int)nsamples);
    long sampfmt, sampwidth;
    AFgetsampfmt(afile, AF_DEFAULT_TRACK, &sampfmt, &sampwidth);
    double mx=1./double(1<<(sampwidth-1));
    mx/=double(nchannels);
    long sampl[4];
    short samps[4];
    signed char sampc[4];
    void* sampp;
    int which;
    if(sampwidth <= 8){
	sampp=(void*)sampc;
	which=0;
    } else if(sampwidth <= 16){
	sampp=(void*)samps;
	which=1;
    } else {
	sampp=(void*)sampl;
	which=2;
    }
#endif
    double rate=8000;
    osound->set_sample_rate(rate);
    ifstream in(filename());
    int nsamples=int(10*rate);
    int sample=0;
    while(onoff){
	// Read a sound sample
#if 0
	long status=AFreadframes(afile, AF_DEFAULT_TRACK,
				 sampp, 1);
	if(status != 1)
	    break;
	double s=0;
	int i;
	switch(which){
	case 0:
	    for(i=0;i<nchannels;i++)
		s+=double(sampc[i])*mx;
	    break;
	case 1:
	    for(i=0;i<nchannels;i++)
		s+=double(samps[i])*mx;
	    break;
	case 2:
	    for(i=0;i<nchannels;i++)
		s+=double(sampl[i])*mx;
	    break;
	}
#endif
	char c1, c2;
	in.get(c1); in.get(c2);
	if(!in)break;
	short m=(c1<<8)|c2;
	double s=m/32768.0;
	osound->put_sample(s);

	// Tell everyone how we are doing...
	update_progress(sample++, (int)nsamples);
	if(sample >= nsamples)
	    sample=0;
    }
    update_progress(1.0);
}

int SoundReader::should_execute()
{
    if(!onoff){
	if(sched_state == SchedDormant)
	    return 0;
	sched_state=SchedDormant;
	return 1;
    }
    return 0;
}
