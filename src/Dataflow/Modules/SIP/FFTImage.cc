//static char *id="@(#) $Id"

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

#include <Dataflow/Ports/ImagePort.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/fft.h>

namespace SCIRun {

class FFTImage : public Module {
  ImageIPort* iport;
  ImageOPort* oport;
public:
  FFTImage(const clString& id);
  virtual ~FFTImage();
  virtual void execute();
};

extern "C" Module* make_FFTImage(const clString& id)
{
  return scinew FFTImage(id);
}

FFTImage::FFTImage(const clString& id)
  : Module("FFTImage", id, Filter)
{
  iport=scinew ImageIPort(this, "Spatial domain", ImageIPort::Atomic);
  add_iport(iport);
  // Create the output port
  oport=scinew ImageOPort(this, "Frequency domain", ImageIPort::Atomic);
  add_oport(oport);
}

FFTImage::~FFTImage()
{
}

void FFTImage::execute()
{
  ImageHandle in;
  if(!iport->get(in))
    return;
  ImageHandle out=in->clone();
  unsigned long flops, refs;

  fft2d_float(out->rows[0], out->xres(), 1, &flops, &refs);
  oport->send(out);
}

} // End namespace SCIRun


