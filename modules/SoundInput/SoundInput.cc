
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
#include <iostream.h>
#include <fstream.h>

SoundInput::SoundInput()
: UserModule("SoundInput")
{
    // Create the output data handle and port
    add_oport(&outsound, "Sound", SoundData::Stream);

    // Set up the interface
    MUI_onoff_switch* oo=new MUI_onoff_switch("Input",
					      &onoff, 0);
    add_ui(oo);

    // Setup the execute condtion...
    execute_condition(oo, 1);
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
    ifstream in("isound");
    
    // Setup the sampling rate...
    double rate=8000;
    NOT_FINISHED("Set sampling rate");
    outsound.set_sample_rate(rate);

    int sample=0;
    int nsamples=(int)(rate*10);
    onoff=1;
    while(onoff){
	// Read a sound sample
	short sample;
	char c1, c2;
	in.get(c1); in.get(c2);
	sample=(c1<<8)|c2;
	double s=sample/32768.0;
	if(!in)break;
	outsound.put_sample(s);

	// Tell everyone how we are doing...
	update_progress(sample++, nsamples);
	if(sample >= nsamples)
	    sample=0;
    }
}
