//static char *id="@(#) $Id"

/*
 *  WhiteNoiseImage.cc:  Unfinished modules
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/ImagePort.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MusilRNG.h>
#include <Core/TclInterface/TCLvar.h>

namespace SCIRun {

class WhiteNoiseImage : public Module {
  ImageOPort* oport;
  TCLint xres;
  TCLint yres;
public:
  WhiteNoiseImage(const clString& id);
  virtual ~WhiteNoiseImage();
  virtual void execute();
};

extern "C" Module* make_WhiteNoiseImage(const clString& id)
{
  return scinew WhiteNoiseImage(id);
}

WhiteNoiseImage::WhiteNoiseImage(const clString& id)
  : Module("WhiteNoiseImage", id, Source), xres("xres", id, this),
    yres("yres", id, this)
{
  // Create the output port
  oport=scinew ImageOPort(this, "Image", ImageIPort::Atomic);
  add_oport(oport);
}

WhiteNoiseImage::~WhiteNoiseImage()
{
}

void WhiteNoiseImage::execute()
{
  MusilRNG rng;
  int xr=xres.get();
  int yr=yres.get();
  Image* image=new Image(xr, yr);
  for(int y=0;y<yr;y++){
    float* p=image->rows[y];
    for(int x=0;x<xr;x++){
      *p++=rng()*2-1;
      *p++=0;
    }
  }
  oport->send(image);
}

} // End namespace SCIRun

