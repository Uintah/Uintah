
/*
 *  SoundMixer.cc:  The first module!
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
#include <TCL/TCLvar.h>

struct SoundMixer_PortInfo {
    TCLvardouble* gain;
    SoundIPort* isound;
};

class SoundMixer : public Module {
    TCLvardouble overall_gain;
    SoundOPort* osound;
    Array1<SoundMixer_PortInfo*> portinfo;
public:
    SoundMixer(const clString& id);
    SoundMixer(const SoundMixer&, int deep);
    virtual ~SoundMixer();
    virtual Module* clone(int deep);
    virtual void connection(Module::ConnectionMode, int, int);
    virtual void execute();
};

static Module* make_SoundMixer(const clString& id)
{
    return new SoundMixer(id);
}

static RegisterModule db1("Sound", "SoundMixer", make_SoundMixer);

SoundMixer::SoundMixer(const clString& id)
: Module("SoundMixer", id, Filter), overall_gain("overall_gain", id, this)
{
    // Create the output data handle and port
    osound=new SoundOPort(this, "Sound Output",
			  SoundIPort::Stream|SoundIPort::Atomic);
    add_oport(osound);

    // Create the input port
    SoundMixer_PortInfo* pi=new SoundMixer_PortInfo;
    pi->isound=new SoundIPort(this, "Sound(0)",
			      SoundIPort::Stream|SoundIPort::Atomic);
    add_iport(pi->isound);
    portinfo.add(pi);
}

SoundMixer::SoundMixer(const SoundMixer& copy, int deep)
: Module(copy, deep), overall_gain("overall_gain", id, this)
{
    NOT_FINISHED("SoundMixer::SoundMixer");
}

SoundMixer::~SoundMixer()
{
}

Module* SoundMixer::clone(int deep)
{
    return new SoundMixer(*this, deep);
}

void SoundMixer::execute()
{
    // If all of the inputs are not the same sampling rate, flag
    // an error...
    int ni=portinfo.size()-1;
    double rate=portinfo[0]->isound->sample_rate();
    for(int i=1;i<ni;i++){
	if(portinfo[i]->isound->sample_rate() != rate){
	    error("All inputs must have the same sampling rate");
	    return;
	}
    }

    // If any of the inputs are stereo, then the output will be
    // in stereo
    int stereo=0;
    for(i=0;i<ni;i++){
	if(portinfo[i]->isound->is_stereo()){
	    stereo=1;
	    break;
	}
    }
    osound->set_stereo(stereo);

    // If any of the protocols are atomic, then we can figure
    // out the maximum number of samples.  Otherwise, we make
    // a random guess as 10X the sampling rate
    int nsamples=-1;
    for(i=0;i<ni;i++){
	SoundIPort* isound=portinfo[i]->isound;
	if(isound->using_protocol() == SoundIPort::Atomic){
	    nsamples=Max(nsamples, isound->nsamples());
	}
    }
    if(nsamples == -1)
	nsamples=(int)(rate*10);

    // Setup the output sampling rate
    osound->set_sample_rate(rate);

    int sample=0;
    int nend=0;
    double buf[20];
    int bs=20;
    for(i=0;i<bs;i++)
	buf[i]=0;
    while(nend < ni){
	// Get the samples from the input stream,
	// combine them and pass them along...
	double sum=0;
	nend=0;
	for(int i=0;i<ni;i++){
	    double gain=portinfo[i]->gain->get();
	    SoundIPort* isound=portinfo[i]->isound;
	    double sample;
	    if(isound->end_of_stream()){
		nend++;
		sample=0;
	    } else {
		sample=isound->next_sample();
	    }
	    sum+=gain*sample;
	}
	if(nend != ni){
	    sum*=overall_gain.get();
	    osound->put_sample(sum);

	    // Tell everyone how we are doing...
	    update_progress(sample++, nsamples);
	    if(sample >= nsamples)
		sample=0;
	}
    }
    update_progress(1.0);
}

void SoundMixer::connection(ConnectionMode mode, int which_port,
			    int output)
{
    if(output)return; // Don't care about connections on output ports...
    if(mode==Disconnected){
	// Remove the associated port...
	remove_iport(which_port);

	// Delete the slider...
#ifdef OLDUI
	remove_ui(portinfo[which_port]->interface);
#endif
	delete portinfo[which_port];

	// Update all of the port and slider names...
	int ni=portinfo.size()-1;
	for(int i=0;i<ni;i++){
	    clString pname(clString("Sound (")+to_string(i)+clString(")"));
	    rename_iport(i, pname);
#ifdef OLDUI
	    if(portinfo[i]->interface){
		clString sname(clString("Gain (")+to_string(i)+clString(")"));
		portinfo[i]->interface->set_title(sname);
	    }
#endif
	}
#ifdef OLDUI
	reconfigure_ui();
#endif
    } else {
	// Add a new data handle
	SoundMixer_PortInfo* pi=new SoundMixer_PortInfo;
	portinfo.add(pi);
	
	// Add a new port...
	clString pname(clString("Sound (")+to_string(which_port)+clString(")"));
	pi->isound=new SoundIPort(this, pname,
				  SoundIPort::Stream|SoundIPort::Atomic);
	add_iport(pi->isound);

	// Add a slider for the one that we just connected;
	pi=portinfo[which_port];
	clString sname(clString("Gain (")+to_string(which_port)+clString(")"));
#ifdef OLDUI
	MUI_slider_real* slider=new MUI_slider_real(sname, &pi->gain,
						    MUI_widget::Immediate, 1);
	pi->interface=slider;
	add_ui(slider);
	reconfigure_ui();
#endif
    }
}
