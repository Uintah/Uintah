/*
 *  IFFTImage.cc:  Unfinished modules
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
#include <Datatypes/ImagePort.h>
#include <Malloc/Allocator.h>
#include <Math/fft.h>

class IFFTImage : public Module {
    ImageIPort* iport;
    ImageOPort* oport;
public:
    IFFTImage(const clString& id);
    IFFTImage(const IFFTImage&, int deep);
    virtual ~IFFTImage();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_IFFTImage(const clString& id)
{
    return scinew IFFTImage(id);
}
};

IFFTImage::IFFTImage(const clString& id)
: Module("IFFTImage", id, Filter)
{
    iport=scinew ImageIPort(this, "Frequency domain", ImageIPort::Atomic);
    add_iport(iport);
    // Create the output port
    oport=scinew ImageOPort(this, "Spatial domain", ImageIPort::Atomic);
    add_oport(oport);
}

IFFTImage::IFFTImage(const IFFTImage& copy, int deep)
: Module(copy, deep)
{
    NOT_FINISHED("IFFTImage::IFFTImage");
}

IFFTImage::~IFFTImage()
{
}

Module* IFFTImage::clone(int deep)
{
    return scinew IFFTImage(*this, deep);
}

void IFFTImage::execute()
{
    ImageHandle in;
    if(!iport->get(in))
	return;
    ImageHandle out=in->clone();
    unsigned long flops, refs;
    int xres=out->xres();
    int yres=out->yres();
    fft2d_float(out->rows[0], out->xres(), -1, &flops, &refs);
    float* p=out->rows[0];
    float scale=1./(xres*yres);
    //cout << "scale=" << scale << endl;
    for(int y=0;y<yres;y++){
	for(int x=0;x<xres;x++){
	    *p++ *= scale;
	    *p++ *= scale;
	}
    }
    oport->send(out);
}
