
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
#include <MUI.h>
#include <NotFinished.h>
#include <SoundPort.h>
#include <iostream.h>
#include <fstream.h>
#include <audio.h>
#include <stdlib.h>
#include <stdio.h>

SoundOutput::SoundOutput()
: UserModule("SoundOutput", Sink)
{
    // Create the input data handle and port
    isound=new SoundIPort(this, "Input",
			  SoundIPort::Atomic|SoundIPort::Stream);
    add_iport(isound);
}

SoundOutput::SoundOutput(const SoundOutput& copy, int deep)
: UserModule(copy, deep)
{
    NOT_FINISHED("SoundOutput::SoundOutput");
}

SoundOutput::~SoundOutput()
{
}

Module* make_SoundOutput()
{
    return new SoundOutput;
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
	exit(-1);
    }
    if(ALsetsampfmt(config, AL_SAMPFMT_DOUBLE) == -1){
	perror("ALsetsampfmt");
	exit(-1);
    }
    if(ALsetfloatmax(config, 1.0) == -1){
	perror("ALsetfloatmax");
	exit(-1);
    }
    if(ALsetchannels(config, 1L) == -1){
	perror("ALsetchannels");	
	exit(-1);
    }
    
    double rate=isound->sample_rate();
    long pvbuf[2];
    pvbuf[0]=AL_OUTPUT_RATE;
    pvbuf[1]=(long)rate;
    ALsetparams(AL_DEFAULT_DEVICE, pvbuf, 2);

    // Open the sound port...
    ALport port=ALopenport("SoundOutput", "w", config);
    if(!port){
	cerr << "Error opening sound port\n";
	perror("ALopenport");
	exit(-1);
    }
    
    int nsamples=(int)(rate*10);
    if(isound->using_protocol() == SoundIPort::Atomic){
	nsamples=isound->nsamples();
    }
    int sample=0;
    while(!isound->end_of_stream()){
	// Read a sound sample
	double s=isound->next_sample();
	ALwritesamps(port, (void*)&s, 1);

	// Tell everyone how we are doing...
	update_progress(sample++, nsamples);
	if(sample >= nsamples)
	    sample=0;
    }
    ALcloseport(port);
    ALfreeconfig(config);
}
