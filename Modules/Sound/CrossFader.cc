
/*
 *  CrossFader.cc:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   May 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <CrossFader/CrossFader.h>
#include <ModuleList.h>
#include <NotFinished.h>
#include <SoundPort.h>
#include <Math/MinMax.h>
#include <iostream.h>

static Module* make_CrossFader(const clString& id)
{
    return new CrossFader(id);
}

static RegisterModule db1("Sound", "CrossFader", make_CrossFader);

CrossFader::CrossFader(const clString& id)
: Module("CrossFader", id, Sink)
{
    // Create the input ports
    isound1=new SoundIPort(this, "Input",
			  SoundIPort::Atomic|SoundIPort::Stream);
    add_iport(isound1);
    isound2=new SoundIPort(this, "Input",
			  SoundIPort::Atomic|SoundIPort::Stream);
    add_iport(isound2);

    // Create the output port
    osound=new SoundOPort(this, "Output",
			  SoundIPort::Atomic|SoundIPort::Stream);
    add_oport(osound);

    // Make the UI
    gain=1.0;
    fade=0.5;
#ifdef OLDUI
    MUI_slider_real* slider1=new MUI_slider_real("gain", &gain,
						 MUI_widget::Immediate, 1);
    add_ui(slider1);
    MUI_slider_real* slider2=new MUI_slider_real("fade", &fade,
						 MUI_widget::Immediate, 1);
    add_ui(slider2);
#endif
}

CrossFader::CrossFader(const CrossFader& copy, int deep)
: Module(copy, deep)
{
    NOT_FINISHED("CrossFader::CrossFader");
}

CrossFader::~CrossFader()
{
}

Module* CrossFader::clone(int deep)
{
    return new CrossFader(*this, deep);
}

void CrossFader::execute()
{
    int stereo=0;
    int st1=isound1->is_stereo();
    int st2=isound2->is_stereo();
    if(!isound1->is_stereo() || isound2->is_stereo()){
	stereo=1;
	osound->set_stereo(1);
    }
    double rate=isound1->sample_rate();
    if(isound2->sample_rate() != rate){
	error("Both ports must have the same sample rate");
	return;
    }
    int nsamples;
    if(isound1->nsamples() == -1 || isound2->nsamples() == -1){
	nsamples=(int)(rate*10);
    } else {
	nsamples=Max(isound1->nsamples(), isound2->nsamples());
	osound->set_nsamples(nsamples);
    }
    osound->set_sample_rate(rate);

    int sample=0;
    if(st1 && st2){
	while(!isound1->end_of_stream() && !isound2->end_of_stream()){
	    double l1=isound1->next_sample();
	    double l2=isound2->next_sample();
	    double fade1=1-fade;
	    double l=gain*(fade*l1+fade1*l2);
	    osound->put_sample(l);
	
	    double r1=isound1->next_sample();
	    double r2=isound2->next_sample();
	    double r=gain*(fade*r1+fade1*r2);
	    osound->put_sample(r);
	
	    // Tell everyone how we are doing...
	    update_progress(sample++, nsamples);
	    if(sample >= nsamples)
		sample=0;
	}
    } else if(st1){
	// Only channel 1 is stereo
	while(!isound1->end_of_stream() && !isound2->end_of_stream()){
	    double l1=isound1->next_sample();
	    double s2=isound2->next_sample();
	    double fade1=1-fade;
	    double l=gain*(fade*l1+fade1*s2);
	    osound->put_sample(l);

	    double r1=isound1->next_sample();
	    double r=gain*(fade*r1+fade1*s2);
	    osound->put_sample(r);

	    // Tell everyone how we are doing...
	    update_progress(sample++, nsamples);
	    if(sample >= nsamples)
		sample=0;
	}
    } else if(st2){
	// Only channel 2 is stereo
	while(!isound1->end_of_stream() && !isound2->end_of_stream()){
	    double s1=isound1->next_sample();
	    double l2=isound2->next_sample();
	    double fade1=1-fade;
	    double l=gain*(fade*s1+fade1*l2);
	    osound->put_sample(l);

	    double r2=isound2->next_sample();
	    double r=gain*(fade*s1+fade1*r2);
	    osound->put_sample(r);

	    // Tell everyone how we are doing...
	    update_progress(sample++, nsamples);
	    if(sample >= nsamples)
		sample=0;
	}
    } else {
	while(!isound1->end_of_stream() && !isound2->end_of_stream()){
	    double s1=isound1->next_sample();
	    double s2=isound2->next_sample();
	    double s=gain*(fade*s1+(1-fade)*s2);
	    osound->put_sample(s);
	    // Tell everyone how we are doing...
	    update_progress(sample++, nsamples);
	    if(sample >= nsamples)
		sample=0;
	}
    }
}
 
