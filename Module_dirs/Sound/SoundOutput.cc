
/*
 *  SoundOutput.cc:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <SoundOutput/SoundOutput.h>
#include <Math/MinMax.h>
#include <ModuleList.h>
#include <MUI.h>
#include <NotFinished.h>
#include <SoundPort.h>
#include <iostream.h>
#include <fstream.h>
#include <audio.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

static Module* make_SoundOutput()
{
    return new SoundOutput;
}

static RegisterModule db1("Sound", "SoundOutput", make_SoundOutput);

SoundOutput::SoundOutput()
: Module("SoundOutput", Sink)
{
    // Create the input data handle and port
    isound=new SoundIPort(this, "Input",
			  SoundIPort::Atomic|SoundIPort::Stream);
    add_iport(isound);
}

SoundOutput::SoundOutput(const SoundOutput& copy, int deep)
: Module(copy, deep)
{
    NOT_FINISHED("SoundOutput::SoundOutput");
}

SoundOutput::~SoundOutput()
{
}

Module* SoundOutput::clone(int deep)
{
    return new SoundOutput(*this, deep);
}

void SoundOutput::execute()
{
    // Setup the sampling rate and other parameters...
    ALconfig config=ALnewconfig();
    if(!config){
	perror("ALnewconfig");
	return;
    }
    if(ALsetsampfmt(config, AL_SAMPFMT_DOUBLE) == -1){
	perror("ALsetsampfmt");
	return;
    }
    if(ALsetfloatmax(config, 1.0) == -1){
	perror("ALsetfloatmax");
	return;
    }
    long nchan=1;
    if(isound->is_stereo())
	nchan=2;
    if(ALsetchannels(config, nchan) == -1){
	perror("ALsetchannels");	
	return;
    }
    
    double rate=isound->sample_rate();
    long pvbuf[2];
    pvbuf[0]=AL_OUTPUT_RATE;
    pvbuf[1]=(long)rate;
    if(ALsetparams(AL_DEFAULT_DEVICE, pvbuf, 2) == -1){
	perror("ALsetparams");
	return;
    }
    long qsize=(long)(rate/10);
    if(qsize<10)qsize=10;
    if(ALsetqueuesize(config, qsize)){
	perror("ALsetqueuesize");
	return;
    }

    // Open the sound port...
    ALport port=ALopenport("SoundOutput", "w", config);
    if(!port){
	cerr << "Error opening sound port\n";
	perror("ALopenport");
	return;
    }
    
    int nsamples=isound->nsamples();
    if(nsamples==-1){
	nsamples=(int)(rate*10);
    }

    int sample=0;
    int size=(int)(rate/20);
    double* buf=new double[size];
    while(!isound->end_of_stream()){
	// Read a sound sample
	double* p=buf;
	for(int i=0;i<size;i++ && !isound->end_of_stream()){
	    double s=isound->next_sample();
	    *p++=s;
	}
	if(ALwritesamps(port, (void*)buf, i) != 0){
	    perror("ALwritesamps");
	    error("Error writing to audio port!");
	    break;
	}

	// Tell everyone how we are doing...
	sample+=(int)(size/nchan);
	update_progress(sample, nsamples);
	if(sample >= nsamples)
	    sample=0;
    }
    while(ALgetfilled(port) != 0){
        sginap(10);
    }
    ALcloseport(port);
    ALfreeconfig(config);
}
