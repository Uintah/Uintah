/*
 *  FilterImage.cc:  Unfinished modules
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

class FilterImage : public Module {
    ImageIPort* iimage;
    ImageIPort* ifilter;
    ImageOPort* oport;
public:
    FilterImage(const clString& id);
    FilterImage(const FilterImage&, int deep);
    virtual ~FilterImage();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_FilterImage(const clString& id)
{
    return scinew FilterImage(id);
}
};

FilterImage::FilterImage(const clString& id)
: Module("FilterImage", id, Filter)
{
    iimage=scinew ImageIPort(this, "Freq. domain image", ImageIPort::Atomic);
    add_iport(iimage);
    ifilter=scinew ImageIPort(this, "Freq. domain filter", ImageIPort::Atomic);
    add_iport(ifilter);
    // Create the output port
    oport=scinew ImageOPort(this, "Freq. domain filtered image", ImageIPort::Atomic);
    add_oport(oport);
}

FilterImage::FilterImage(const FilterImage& copy, int deep)
: Module(copy, deep)
{
    NOT_FINISHED("FilterImage::FilterImage");
}

FilterImage::~FilterImage()
{
}

Module* FilterImage::clone(int deep)
{
    return scinew FilterImage(*this, deep);
}

void filter_image(int xres, int yres, float* out, float* image, float* filter)
{
    for(int y=0;y<yres;y++){
	for(int x=0;x<xres;x++){
	    float ir=*image++;
	    float ii=*image++;
	    float fr=*filter++;
	    float fi=*filter++;
	    float or=ir*fr-ii*fi;
	    float oi=ii*fr-ir*fi;
	    *out++=or;
	    *out++=oi;
	}
    }
}

void FilterImage::execute()
{
    ImageHandle image;
    ImageHandle filter;
    if(!iimage->get(image))
	return;
    if(!ifilter->get(filter))
	return;
    ImageHandle out=new Image(image->xres(), image->yres());
    filter_image(image->xres(), image->yres(),
		 out->rows[0], image->rows[0], filter->rows[0]);
    oport->send(out);
}
