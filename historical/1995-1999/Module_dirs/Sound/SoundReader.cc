
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

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Dataflow/ModuleList.h>
#include <Datatypes/SoundPort.h>
#include <Math/MinMax.h>
#include <TCL/TCLvar.h>

#include <iostream.h>
#include <audiofile.h>

class SoundReader : public Module {
    SoundOPort* osound;
    TCLstring filename;
public:
    SoundReader(const clString& id);
    SoundReader(const SoundReader&, int deep);
    virtual ~SoundReader();
    virtual Module* clone(int deep);
    virtual void execute();
};

static Module* make_SoundReader(const clString& id)
{
    return new SoundReader(id);
}

static RegisterModule db1("Sound", "SoundReader", make_SoundReader);

SoundReader::SoundReader(const clString& id)
: Module("SoundReader", id, Source), filename("filename", id, this)
{
    // Create the output data handle and port
    osound=new SoundOPort(this, "Sound", SoundIPort::Stream);
    add_oport(osound);
}

SoundReader::SoundReader(const SoundReader& copy, int deep)
: Module(copy, deep), filename("filename", id, this)
{
    NOT_FINISHED("SoundReader::SoundReader");
}

SoundReader::~SoundReader()
{
}

Module* SoundReader::clone(int deep)
{
    return new SoundReader(*this, deep);
}

void SoundReader::execute()
{
    // Open the sound port...
    AFfilehandle afile=AFopenfile(filename.get()(), "r", 0);
    if(!afile){
	cerr << "Error opening file: " << afile << endl;
	return;
    }
    long nchannels=AFgetchannels(afile, AF_DEFAULT_TRACK);
    if(nchannels==1){
	osound->set_stereo(0);
    } else if(nchannels==2){
	osound->set_stereo(1);
    } else {
	error("Four channel mode not supported\n");
	return;
    }
    // Setup the sampling rate...
    double rate=AFgetrate(afile, AF_DEFAULT_TRACK);
    osound->set_sample_rate(rate);

    long nsamples=AFgetframecnt(afile, AF_DEFAULT_TRACK);
    osound->set_nsamples((int)nsamples);
    long sampfmt, sampwidth;
    AFgetsampfmt(afile, AF_DEFAULT_TRACK, &sampfmt, &sampwidth);
    double mx=1./double(1<<(sampwidth-1));
    mx/=double(nchannels);
    int which;
    if(sampwidth <= 8){
	which=0;
    } else if(sampwidth <= 16){
	which=1;
    } else {
	which=2;
    }
    long sample=0;
    int size=(int)(rate/20);
    signed char* sampc=new char[size*nchannels];
    short* samps=new short[size*nchannels];
    long* sampl=new long[size*nchannels];
    int done=0;
    switch(which){
    case 0:
	while(!done){
	    long status=AFreadframes(afile, AF_DEFAULT_TRACK,
				     (void*)sampc, size);
	    if(status==0){
		done=1;
	    } else if(status < 0){
		error("Error in AFreadsamps");
		done=1;
	    } else {
		signed char* p=sampc;
		long ns=status*nchannels;
		for(int i=0;i<ns;i++){
		    double s=double(*p++)*mx;
		    osound->put_sample(s);
		}
		sample+=status;
		update_progress((int)sample++, (int)nsamples);
		while(sample >= nsamples)
		    sample-=nsamples;
	    }
	}
	break;
    case 1:
	while(!done){
	    long status=AFreadframes(afile, AF_DEFAULT_TRACK,
				     (void*)samps, size);
	    if(status==0){
		done=1;
	    } else if(status < 0){
		error("Error in AFreadsamps");
		done=1;
	    } else {
		short* p=samps;
		long ns=status*nchannels;
		for(int i=0;i<ns;i++){
		    double s=double(*p++)*mx;
		    osound->put_sample(s);
		}
		sample+=status;
		update_progress((int)sample++, (int)nsamples);
		while(sample >= nsamples)
		    sample-=nsamples;
	    }
	}
	break;
    case 2:
	while(!done){
	    long status=AFreadframes(afile, AF_DEFAULT_TRACK,
				     (void*)sampl, size);
	    if(status==0){
		done=1;
	    } else if(status < 0){
		error("Error in AFreadsamps");
		done=1;
	    } else {
		long* p=sampl;
		long ns=status*nchannels;
		for(int i=0;i<ns;i++){
		    double s=double(*p++)*mx;
		    osound->put_sample(s);
		}
		sample+=status;
		update_progress((int)sample++, (int)nsamples);
		while(sample >= nsamples)
		    sample-=nsamples;
	    }
	}
	break;
    }
    delete samps;
    delete sampl;
    delete sampc;
}
