//static char *id="@(#) $Id"

/*
 *  PMFilterImage.cc:  Unfinished modules
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Util/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/Image/ImagePort.h>
#include <Malloc/Allocator.h>
#include <Math/Trig.h>
#include <TclInterface/TCLvar.h>
#include <math.h>

namespace SCIRun {
namespace Modules {

using namespace SCICore::TclInterface;

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;

using namespace SCIRun::Datatypes;

#define	G	9.81

class PMFilterImage : public Module {
    TCLdouble inphi;
    TCLdouble inu10;
    TCLdouble inbegtime;
    TCLdouble intstep;
    TCLint innsteps;
    TCLint inres;
    ImageOPort* oport;
    Image* make_filter(int, double, double, double);
    Image* phaseimage;
    double old_phi;
    double old_u10;
    int old_res;
public:
    PMFilterImage(const clString& id);
    PMFilterImage(const PMFilterImage&, int deep);
    virtual ~PMFilterImage();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_PMFilterImage(const clString& id)
{
    return scinew PMFilterImage(id);
}
}

PMFilterImage::PMFilterImage(const clString& id)
: Module("PMFilterImage", id, Source), inphi("phi", id, this),
  inu10("u10", id, this), inbegtime("begtime", id, this),
  inres("res", id, this),
  innsteps("nsteps", id, this), intstep("tstep", id, this)
{

    // Create the output port
    oport=scinew ImageOPort(this, "Frequency domain Pierson-Moskowitz filter", ImageIPort::Atomic);
    add_oport(oport);
    phaseimage=0;
    old_res=-1;
}

PMFilterImage::PMFilterImage(const PMFilterImage& copy, int deep)
: Module(copy, deep), inphi("phi", id, this),
  inu10("u10", id, this), inbegtime("begtime", id, this),
  inres("res", id, this),
  innsteps("nsteps", id, this), intstep("tstep", id, this)
{
    NOT_FINISHED("PMFilterImage::PMFilterImage");
}

PMFilterImage::~PMFilterImage()
{
}

Module* PMFilterImage::clone(int deep)
{
    return scinew PMFilterImage(*this, deep);
}

static double g(double x)
{
    double lg = lgamma(x);
    double g = signgam * exp(lg);

    return g;
}


Image* PMFilterImage::make_filter(int res, double time, double phi, double u10)
{
    if(res != old_res || phi != old_phi || u10 != old_u10){
	old_res=res;
	old_phi=phi;
	old_u10=u10;
	if(phaseimage)
	    delete phaseimage;
	phaseimage=new Image(res, res);
	phi = DtoR(phi);

	double fm = 0.13 * G / u10;

	int res2=res/2;
	{
	    for(int y = 0; y < res; y++){
		for(int x = 0; x < res; x++) {
		    phaseimage->set(x,y,0,0);
		}
	    }
	}
	for(int y = 0; y < res; y++){
	    for(int x = 0; x <= res2; x++) {
		if( ( x == 0) && ( y == 0) ){
		    phaseimage->set(x, y, 0, 0);
		    phaseimage->set(res-x-1, y, 0, 0);
		    continue;
		}
		
		double xx = x / (double)res2;
		//double yy = (y - res2) / (double)res2;
		double yy = y / (double)res;
		double f = sqrt(xx*xx + yy*yy);
		double theta = atan( yy / xx);
		theta = theta + phi;
		
		double mu;
		if( f < fm )
		    mu = 4.06;
		else
		    mu = -2.34;
		double p = 9.77 * pow( (f/fm), mu);
		double n = pow(2., 1 - 2 * p) * PI *
		    g(2 * p + 1) / 
			( g(p + 1) * g(p + 1) );
		double d = 1 / n * pow( cos(theta/2), 2 * p);
		double FPM = .0081 * G * G / ( pow(2*PI, 4.) * pow(f, 5.) ) *
		    exp( -(5./4.) * pow( (fm/f), 4.) );
		double F = FPM * d;
	    
/* 
 * F is the amplitute of the filter.  Change the phase if we want
 * to animate the waves.  The frequency is sqrt(x*x+y*y).
 */
		phaseimage->set(x, y, F, f);
	    }
	}
    }
	    

    float* p=phaseimage->rows[0];
    Image* image=new Image(res, res);
    for(int y = 0; y < res; y++){
	for(int x = 0; x < res; x++) {
	    float F=*p++;
	    float f=*p++;

	    float Ftheta = time * fsqrt(f) * PI/180;
	    float Fre = F * fcos(Ftheta);
	    float Fim = F * fsin(Ftheta);

	    image->set(x, y, Fre, Fim);
	    //image->set(res-x-1, y, Fre, Fim);
	    //image->set(res-x-1, y, 0, 0);
	}
    }
    return image;
}

void PMFilterImage::execute()
{
    double phi=inphi.get();
    double u10=inu10.get();
    double time=inbegtime.get();
    double dt=intstep.get();
    int res=inres.get();
    int nsteps=innsteps.get();
    
    for(int i=0;i<nsteps;i++){
	Image* image=make_filter(res, time, phi, u10);
	oport->send_intermediate(image);
	time+=dt;
    }
    Image* image=make_filter(res, time, phi, u10);
    oport->send(image);
}

} // End namespace Modules
} // End namespace SCIRun

//
// $Log$
// Revision 1.3  1999/08/25 03:48:57  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:40:01  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:58:54  mcq
// Initial commit
//
// Revision 1.2  1999/04/30 01:11:53  dav
// moved TiffReader to SCIRun from PSECore
//
// Revision 1.1  1999/04/29 22:26:33  dav
// Added image files to SCIRun
//
//
