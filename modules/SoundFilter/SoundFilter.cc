
/*
 *  SoundFilter.cc:  The first module!
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <SoundFilter/SoundFilter.h>
#include <Math/MinMax.h>
#include <Math/Trig.h>
#include <MUI.h>
#include <NotFinished.h>
#include <Port.h>
#include <iostream.h>
#include <fstream.h>

SoundFilter::SoundFilter()
: UserModule("SoundFilter")
{
    lower_cutoff=2500;
    upper_cutoff=3500;
    // Create the output data handle and port
    add_oport(&outsound, "Sound", SoundData::Stream|SoundData::Atomic);

    // Create the input port
    add_iport(&isound, "Sound", SoundData::Stream|SoundData::Atomic);

    // Setup the execute condtion...
    execute_condition(NewDataOnAllConnectedPorts);
}

SoundFilter::SoundFilter(const SoundFilter& copy, int deep)
: UserModule(copy, deep)
{
    NOT_FINISHED("SoundFilter::SoundFilter");
}

SoundFilter::~SoundFilter()
{
}

Module* make_SoundFilter()
{
    return new SoundFilter;
}

Module* SoundFilter::clone(int deep)
{
    return new SoundFilter(*this, deep);
}

void SoundFilter::execute()
{
    double rate=isound.sample_rate();
    int nsamples;
    if(isound.using_protocol() == SoundData::Atomic){
	nsamples=isound.nsamples();
    } else {
	nsamples=(int)(rate*10);
    }

    // Setup the output sampling rate
    outsound.set_sample_rate(rate);

    // Design the filter
    double wp1=2*PI*lower_cutoff/rate;
    double wp2=2*PI*upper_cutoff/rate;
    double cs=Cos((wp2+wp1)/2);
    double cd=Cos((wp2-wp1)/2);
    double alpha=cs/cd;
    double theta_p=0.2*PI;
    double kappa=Cot((wp2-wp1)/2)*Tan(theta_p);
    double X=-2*alpha*kappa/(kappa+1);
    double X_2=X*X;
    double Y=(kappa-1)/(kappa+1);
    cerr << "cd=" << cd << ", kappa+1=" << kappa+1 << endl;
    cerr << "X=" << X << ", Y=" << Y << endl;
    double Y_2=Y*Y;
    double A=0.001836;
    double B=-1.5548;
    double C=0.6493;
    double D=-1.4996;
    double F=0.8482;
    double bb[3];
    // Numberator is b
    // b=(Y*z^2+2*X*z+1+z^2+Y)^4*A;
    bb[0]=Y+1;
    bb[1]=2*X;
    bb[2]=Y+1;
    double bbb[5];
    for(int i=0;i<5;i++)
	bbb[i]=0;
    // Square bb to get bbb
    for(i=0;i<3;i++){
	for(int j=0;j<3;j++){
	    int k=i+j;
	    bbb[k]+=bb[i]*bb[j];
	}
    }

    // square again to get b
    double b[9];
    for(i=0;i<9;i++)
	b[i]=0;
    for(i=0;i<5;i++){
	for(int j=0;j<5;j++){
	    int k=i+j;
	    b[k]+=bbb[i]*bbb[j];
	}
    }
    // Multiply by A
    for(i=0;i<9;i++)
	b[i]*=A;

    // Compute denominator...
    double a1[5];
    a1[0]=D*Y+F*Y_2+1;
    a1[1]=D*X*Y+2*F*X+2*X*Y+D*X;
    a1[2]=2*Y+D+2*F*Y+D*X_2+X_2+D*Y_2+F*X_2;
    a1[3]=D*X*Y+2*F*X+2*X*Y+D*X;
    a1[4]=F+Y_2+D*Y;
    double a2[5];
    a2[0]=B*Y+C*Y_2+1;
    a2[1]=B*X*Y+2*C*X*Y+2*X+B*X;
    a2[2]=2*Y+B+2*C*Y+B*X_2+X_2+B*Y_2+C*X_2;
    a2[3]=B*X*Y+2*C*X+2*X*Y+B*X;
    a2[4]=C+Y_2+B*Y;
    double a[9];
    for(i=0;i<9;i++)
	a[i]=0;
    for(i=0;i<5;i++){
	for(int j=0;j<5;j++){
	    int k=i+j;
	    a[k]=a1[i]*a2[j];
	}
    }
    for(i=0;i<9;i++){
	cerr << "a[" << i << "]=" << a[i] << ", b[" << i << "]=" << b[i] << endl;
    }

    int sample=0;
    double delay[9];
    for(i=0;i<9;i++)
	delay[i]=0;
    int dcount=0;
    ofstream out("osamp");
    while(!isound.end_of_stream() || dcount++<9){
	// Read a sound sample
	double s;
	if(dcount==0){
	    s=isound.next_sample();
	} else {
	    s=0;
	}

	// Apply the filter...
	for(int i=1;i<9;i++)
	    s+=a[i]*delay[i];
	s*=a[0];
	delay[0]=s;
	s=0;
	for(i=0;i<9;i++)
	    s+=b[i]*delay[i];
	for(i=8;i>0;i--)
	    delay[i]=delay[i-1];

	// Output the sound...
	outsound.put_sample(s);
	out << s << "\n";

	// Tell everyone how we are doing...
	update_progress(sample++, nsamples);
	if(sample >= nsamples)
	    sample=0;
    }
}
