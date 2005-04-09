
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

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/SoundPort.h>
#include <Math/Complex.h>
#include <Math/Expon.h>
#include <Math/MinMax.h>
#include <Math/MiscMath.h>
#include <Math/Trig.h>
#include <iostream.h>

class SoundFilter : public Module {
    SoundIPort* isound;
    SoundOPort* osound;
    double lower_cutoff;
    double upper_cutoff;
public:
    SoundFilter(const clString& id);
    SoundFilter(const SoundFilter&, int deep);
    virtual ~SoundFilter();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_SoundFilter(const clString& id)
{
    return new SoundFilter(id);
}
};

SoundFilter::SoundFilter(const clString& id)
: Module("SoundFilter", id, Filter)
{
    lower_cutoff=2500;
    upper_cutoff=3500;
    // Create the output data handle and port
    isound=new SoundIPort(this, "Filter Input",
			  SoundIPort::Stream|SoundIPort::Atomic);
    add_iport(isound);

    // Create the input port
    osound=new SoundOPort(this, "Filter Output",
			  SoundIPort::Stream|SoundIPort::Atomic);
    add_oport(osound);
}

SoundFilter::SoundFilter(const SoundFilter& copy, int deep)
: Module(copy, deep)
{
    NOT_FINISHED("SoundFilter::SoundFilter");
}

SoundFilter::~SoundFilter()
{
}

Module* SoundFilter::clone(int deep)
{
    return new SoundFilter(*this, deep);
}

void SoundFilter::execute()
{
    double rate=isound->sample_rate();
    int nsamples;
    if(isound->using_protocol() == SoundIPort::Atomic){
	nsamples=isound->nsamples();
    } else {
	nsamples=(int)(rate*10);
    }

    // Setup the output sampling rate
    osound->set_sample_rate(rate);

    // Defaults (put in UI...)
    double ripple_db=0.15;
    double atten_db=17;
    double close=1.2;

    // Figure out the order of the filter.
    double A2=Exp10(atten_db/10.0);
    double e=Sqrt(Exp10(ripple_db/10.0)-1);
    double nr=ACosh(Sqrt(A2-1)/e)/ACosh(close);
    int N=(int)nr;
    double rem=nr-N;
    if(rem > 1.e-4)N++; // Round up...
    N=1;
    cerr << "Designing " << N << "th order Chebyshev type I filter\n";

    // Misc calculations...
    double alpha=1./e+Sqrt(1+1./(e*e));
    double U=upper_cutoff*(2*PI);
    double L=lower_cutoff*(2*PI);
    U=2000;
    L=1000;
    double T=1./rate;

    // Find the poles of the Low Pass chebyshev analog filter
    Complex* root=new Complex[N];
    double major_axis=0.5*(Pow(alpha, 1./N)-Pow(alpha, -1./N));
    double minor_axis=0.5*(Pow(alpha, 1./N)+Pow(alpha, -1./N));
    int i;
    for(i=0;i<N;i++){
	double theta=(i-0.5)*PI/N;
	root[i]=Complex(minor_axis*Sin(theta), major_axis*Cos(theta));
	cerr << "root[i]=(" << root[i].re() << "," << root[i].im() << ")" << endl;
    }

    // Perform Partial Fraction Expansion
    Complex* AK=new Complex[N];
    for(i=0;i<N;i++){
	Complex A(1.0, 0.0);
	Complex s(root[i]);
	for(int j=0;j<N;j++){
	    if(i != j) A/=s-root[j];
	}
	AK[i]=A;
	cerr << "AK[i]=(" << A.re() << "," << A.im() << ")" << endl;
    }

    // Transform to band pass filter, find roots and PFE again...
    Complex* Q=new Complex[2*N];
    Complex* BK=new Complex[2*N];
    for(i=0;i<N;i++){
	// Find roots of:
	// s^2 -Rk(U-L)s+UL
	Complex A(1.0, 0.0);
	Complex B(-root[i]*(U-L));
	Complex C(U*L, 0.0);
	Complex disc(B*B-A*C*4.0);
	Complex sdisc=Sqrt(disc);
	Complex r1=(-B+sdisc)/(A*2.0);
	Complex r2=(-B-sdisc)/(A*2.0);
	// Now: Ak*(U-L)*s/((s-r1)*(s-r2))
	Complex w=AK[i]*(U-L);
	Complex w1=w*r1/(r1-r2);
	Complex w2=w*r2/(r2-r1);
	Q[2*i+0]=r1;
	Q[2*i+1]=r2;
	BK[2*i+0]=w1;
	BK[2*i+1]=w2;
	cerr << "Q[" << 2*i+0 << "]=" << Q[2*i+0] << endl;
	cerr << "Q[" << 2*i+1 << "]=" << Q[2*i+1] << endl;
	cerr << "BK[" << 2*i+0 << "]=" << BK[2*i+0] << endl;
	cerr << "BK[" << 2*i+1 << "]=" << BK[2*i+1] << endl;
    }

    // Now take Z transform, while expanding...
    int fsize=2*N+1;
    Complex* top=new Complex[fsize];
    Complex* bot=new Complex[fsize];
    top[0]=Complex(0.0, 0.0);
    bot[0]=Complex(1.0, 0.0);
    for(i=1;i<fsize;i++){
	top[i]=Complex(0.0, 0.0);
    }
    for(int ii=0;ii<fsize;ii++)
	cerr << "b[" << ii << "]=" << top[ii] << ", " << "a[" << ii << "]=" << bot[ii] << endl;
    for(i=0;i<2*N;i++){
	// Multiply in Bk/(1-Exp(-Qk*T)*z^-1)

	// Multiply old numerator by 1-Exp(-Qk*T)*z^-1
	Complex E=Exp(Q[i]*T);
	cerr << "E[" << i << "]=" << E << endl;

	int j;
	for(j=fsize-1;j>=1;j--){
	    top[j]=top[j]-E*top[j-1];
	}
	// Add old denominator*Bk to numerator
	for(j=fsize-1;j>=0;j--){
	    top[j]+=bot[j]*BK[i];
	}

	// Multiply old denominator by 1-Exp(-Qk*T)*z^-1
	for(j=fsize-1;j>=1;j--){
	    bot[j]=bot[j]-E*bot[j-1];
	}
	for(int ii=0;ii<fsize;ii++)
	    cerr << "b[" << ii << "]=" << top[ii] << ", " << "a[" << ii << "]=" << bot[ii] << endl;

    }

    // Make the filter...
    double* b=new double[fsize];
    double* a=new double[fsize];
    for(i=0;i<fsize;i++){
	b[i]=top[i].re();
	a[i]=bot[i].re();
	if(Abs(top[i].im()) > 1.e-6
	   || Abs(bot[i].im()) > 1.e-6){
	    cerr << "Warning: Filter design failed!\n";
	}
	cerr << "b[" << i << "]=" << b[i] << ", " << "a[" << i << "]=" << a[i] << endl;
    }
    delete[] root;
    delete[] AK;
    delete[] Q;
    delete[] BK;
    delete[] top;
    delete[] bot;
#if 0    
    double b[10];
    b[0]=0.0346;
    b[1]=0.0016;
    b[2]=-0.0293;
    b[3]=-0.0362;
    b[4]=0.0090;
    b[5]=0.0427;
    b[6]=0.0177;
    b[7]=-0.0155;
    b[8]=-0.0323;
    b[9]=0.0076;
    double a[10];
    a[0]=1.0000;
    a[1]=-2.2954;
    a[2]=4.0755;
    a[3]=-4.9622;
    a[4]=5.0056;
    a[5]=-3.8148;
    a[6]=2.4054;
    a[7]=-1.1407;
    a[8]=0.3916;
    a[9]=-0.0817;
#endif

    int sample=0;
    double* delay=new double[fsize];
    for(i=0;i<fsize;i++)
	delay[fsize]=0;
    int dcount=0;
    while(!isound->end_of_stream() || dcount++<10){
	// Read a sound sample
	double s;
	if(dcount==0){
	    s=isound->next_sample();
	} else {
	    s=0;
	}

	// Apply the filter...
	for(int i=1;i<fsize;i++)
	    s-=b[i]*delay[i];
	delay[0]=s;
	s=0;
	for(i=0;i<fsize;i++)
	    s+=a[i]*delay[i];
	for(i=fsize-1;i>0;i--)
	    delay[i]=delay[i-1];

	// Output the sound...
	osound->put_sample(s);

	// Tell everyone how we are doing...
	update_progress(sample++, nsamples);
	if(sample >= nsamples)
	    sample=0;
    }
    delete[] a;
    delete[] b;
}
