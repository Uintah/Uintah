
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
#include <Port.h>
#include <iostream.h>
#include <fstream.h>

SoundOutput::SoundOutput()
: UserModule("SoundOutput")
{
    // Create the input data handle and port
    add_iport(&insound, "Input",
	      SoundData::Atomic|SoundData::Stream);

    // Setup the execute condtion...
    execute_condition(NewDataOnAllConnectedPorts);
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
    // Open the sound port...
    ofstream out("osound");
    
    // Setup the sampling rate...
    double rate=insound.sample_rate();
    NOT_FINISHED("Set sampling rate");

    int nsamples=(int)(rate*10);
    if(insound.using_protocol() == SoundData::Atomic){
	nsamples=insound.nsamples();
    }
    int sample=0;
    while(!insound.end_of_stream()){
	// Read a sound sample
	double s=insound.next_sample();
	s=s<-1?-1:s>1?1:s;
	short sample=(short)(s*32768.0+(s<0?-0.5:0.5));
	char c1=sample>>8;
	char c2=sample&0xff;
	out.put(c1);
	out.put(c2);

	// Tell everyone how we are doing...
	update_progress(sample++, nsamples);
	if(sample >= nsamples)
	    sample=0;
    }
}
