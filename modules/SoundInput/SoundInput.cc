
/*
 *  SoundInput.cc:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <SoundInput/SoundInput.h>
#include <Math/MinMax.h>
#include <MUI.h>
#include <NotFinished.h>
#include <Port.h>
#include <SoundPort.h>
#include <iostream.h>
#include <fstream.h>
#include <audio.h>
#include <stdlib.h>
#include <stdio.h>

SoundInput::SoundInput()
: UserModule("SoundInput", Source)
{
    // Create the output data handle and port
    osound=new SoundOPort(this, "Sound Output", SoundIPort::Stream);
    add_oport(osound);

    // Set up the interface
    MUI_onoff_switch* oo=new MUI_onoff_switch("Input",
					      &onoff, 0);
    add_ui(oo);
}

SoundInput::SoundInput(const SoundInput& copy, int deep)
: UserModule(copy, deep)
{
    NOT_FINISHED("SoundInput::SoundInput");
}

SoundInput::~SoundInput()
{
}

Module* make_SoundInput()
{
    return new SoundInput;
}

Module* SoundInput::clone(int deep)
{
    return new SoundInput(*this, deep);
}

void SoundInput::execute()
{
    // Open the sound port...
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
    
    
    // Setup the sampling rate...
    rate=8000;
    long pvbuf[2];
    pvbuf[0]=AL_OUTPUT_RATE;
    pvbuf[1]=(long)rate;
    ALsetparams(AL_DEFAULT_DEVICE, pvbuf, 2);
    osound->set_sample_rate(rate);

    ALport port=ALopenport("SoundInput", "r", config);
    if(!port){
	cerr << "Error opening sound port\n";
	perror("ALopenport");
	exit(-1);
    }

    int sample=0;
    int nsamples=(int)(rate*10);
    onoff=1;
    while(onoff){
	// Read a sound sample
	double s;
	if(ALreadsamps(port, (void*)&s, 1) == -1){
	    perror("ALreadsamps");
	    exit(-1);
	}
	osound->put_sample(s);

	// Tell everyone how we are doing...
	update_progress(sample++, nsamples);
	if(sample >= nsamples)
	    sample=0;
    }
}

int SoundInput::should_execute()
{
    if(!onoff){
	if(sched_state == SchedDormant)
	    return 0;
	sched_state=SchedDormant;
	return 1;
    }
    return 0;
}
