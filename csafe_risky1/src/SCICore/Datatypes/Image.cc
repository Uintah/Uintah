//static char *id="@(#) $Id$";

/*
 *  Image.cc
 *
 *  Written by:
 *   Author: ?
 *   Sourced from MeshPort.cc by David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */


#include <SCICore/Datatypes/Image.h>
#include <SCICore/Persistent/Persistent.h>
#include <SCICore/Containers/String.h>

namespace SCICore {
namespace Datatypes {

PersistentTypeID Image::type_id("Image", "Datatype", 0);

ColorImage::ColorImage(int xres, int yres)
: imagedata(yres, xres)
{
}

ColorImage::~ColorImage()
{
}

DepthImage::DepthImage(int xres, int yres)
: depthdata(yres, xres)
{
}

DepthImage::~DepthImage()
{
}

Image::Image(int xres, int yres)
: xr(xres), yr(yres)
{
    if(xres && yres){
	rows=new float*[yres];
	float* p=new float[xres*yres*2];
	for(int y=0;y<yres;y++){
	    rows[y]=p;
	    p+=xres*2;
	}
    } else {
	rows=0;
    }
}

Image::~Image()
{
    if(rows){
	delete[] rows[0];
	delete[] rows;
    }
}

int Image::xres() const
{
    return xr;
}

int Image::yres() const
{
    return yr;
}

Image* Image::clone()
{
    return new Image(*this);
}

Image::Image(const Image& copy)
: xr(copy.xr), yr(copy.yr)
{
    if(xr && yr){
	rows=new float*[yr];
	float* p=new float[xr*yr*2];
	for(int y=0;y<yr;y++){
	    rows[y]=p;
	    p+=xr*2;
	    for(int x=0;x<xr*2;x++)
		rows[y][x]=copy.rows[y][x];
	}
    }
}

#define IMAGE_VERSION 1

void Image::io(Piostream& stream)
{
    stream.begin_class("Image", IMAGE_VERSION);
    stream.io(xr);
    stream.io(yr);
    if(stream.reading()){
	if(rows){
	    delete[] rows[0];
	    delete[] rows;
	}
	    
	rows=new float*[yr];
	float* p=new float[xr*yr*2];
	for(int y=0;y<yr;y++){
	    rows[y]=p;
	    p+=xr*2;
	}
    }
    int n=xr*yr*2;
    float* p=rows[0];
    for(int i=0;i<n;i++)
	stream.io(*p++);
    stream.end_class();
}

float Image::max_abs()
{
    float max=0;
    float* p=rows[0];
    for(int y=0;y<yr;y++){
	for(int x=0;x<xr;x++){
	    float r=*p;
	    if(r>max)
		max=r;
	    p+=2;
	}
    }
    return max;
}

} // End namespace Datatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.3  1999/08/25 03:48:33  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:38:46  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:21  mcq
// Initial commit
//
// Revision 1.1  1999/04/25 04:07:07  dav
// Moved files into Datatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:49  dav
// Import sources
//
//
