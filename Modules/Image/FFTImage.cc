/*
 *  FFTImage.cc:  Unfinished modules
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

class FFTImage : public Module {
    ImageIPort* iport;
    ImageOPort* oport;
public:
    FFTImage(const clString& id);
    FFTImage(const FFTImage&, int deep);
    virtual ~FFTImage();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_FFTImage(const clString& id)
{
    return scinew FFTImage(id);
}
};

FFTImage::FFTImage(const clString& id)
: Module("FFTImage", id, Filter)
{
    iport=scinew ImageIPort(this, "Spatial domain", ImageIPort::Atomic);
    add_iport(iport);
    // Create the output port
    oport=scinew ImageOPort(this, "Frequency domain", ImageIPort::Atomic);
    add_oport(oport);
}

FFTImage::FFTImage(const FFTImage& copy, int deep)
: Module(copy, deep)
{
    NOT_FINISHED("FFTImage::FFTImage");
}

FFTImage::~FFTImage()
{
}

Module* FFTImage::clone(int deep)
{
    return scinew FFTImage(*this, deep);
}

void FFTImage::execute()
{
    ImageHandle in;
    if(!iport->get(in))
	return;
    ImageHandle out=in->clone();
    unsigned long flops, refs;

    int xres=out->xres();
    int yres=out->yres();
    fft2d_float(out->rows[0], out->xres(), 1, &flops, &refs);
    oport->send(out);
}



