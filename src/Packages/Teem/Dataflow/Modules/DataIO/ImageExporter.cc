/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  ImageExporter.cc: Use ImageMagick to write various image formats.
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   April 2004
 *
 *  Copyright (C) 2004 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Teem/Dataflow/Ports/NrrdPort.h>
#include <Core/GuiInterface/GuiVar.h>

#ifdef HAVE_MAGICK
namespace C_Magick {
#include <magick/api.h>
}
#endif

using namespace SCIRun;
using namespace SCITeem;


class ImageExporter : public Module {
  GuiString filename_;
  GuiString filetype_;
public:
  ImageExporter(GuiContext *ctx);
  virtual ~ImageExporter();
  virtual void execute();
};


DECLARE_MAKER(ImageExporter)

ImageExporter::ImageExporter(GuiContext *ctx)
  : Module("ImageExporter", ctx, Filter, "DataIO", "Teem"), 
    filename_(ctx->subVar("filename")),
    filetype_(ctx->subVar("filetype"))
{
}


ImageExporter::~ImageExporter()
{
}


void
ImageExporter::execute()
{
  // Read data from the input port
  NrrdDataHandle handle;
  NrrdIPort* inport = (NrrdIPort *)get_iport("Input Data");
  if (!inport) {
    error("Unable to initialize iport 'Input Data'.");
    return;
  }

  if(!inport->get(handle))
  {
    return;
  }
  
  if (!handle.get_rep())
  {
    error("Null input");
    return;
  }

#ifdef HAVE_MAGICK
  // If no name is provided, return.
  string fn(filename_.get());
  if(fn == "")
  {
    error("Warning: no filename in ImageExporter");
    return;
  }

  const Nrrd *nrrd = handle->nrrd;

  if (nrrd->dim != 3)
  {
    error("Only 3 dimensional nrrds at this time (1-4, height, width).");
    return;
  }
  
  bool grey = false;
  bool alpha = true;
  if (nrrd->axis[0].size == 1)
  {
    grey = true;
    alpha = false;
  }
  if (nrrd->axis[0].size == 2)
  {
    grey = true;
    alpha = true;
  }
  else if (nrrd->axis[0].size == 3)
  {
    grey = false;
    alpha = false;
  }
  else if (nrrd->axis[0].size == 4)
  {
    grey = false;
    alpha = true;
  }
  else
  {
    error("Nrrd axis zero must contain Grayscale, RGB or RGBA data.");
    return;
  }

  if (!(nrrd->type == nrrdTypeUShort || nrrd->type == nrrdTypeUChar))
  {
    error("Only Nrrds of type UShort and UChar are currently supported.");
    return;
  }
  
  C_Magick::InitializeMagick(0);

  C_Magick::ImageInfo *image_info =
    C_Magick::CloneImageInfo((C_Magick::ImageInfo *)0);
  strncpy(image_info->filename, fn.c_str(), MaxTextExtent);
  image_info->colorspace = C_Magick::RGBColorspace;
  image_info->quality = 90;
  C_Magick::Image *image = C_Magick::AllocateImage(image_info);
  const unsigned int w = image->columns = nrrd->axis[1].size;
  const unsigned int h = image->rows = nrrd->axis[2].size;

  C_Magick::PixelPacket *pixels =
    C_Magick::SetImagePixels(image, 0, 0, w, h);

  if (nrrd->type == nrrdTypeUShort)
  {
    unsigned short *data = (unsigned short *)nrrd->data;

    // Copy pixels from nrrd to Image.
    for (unsigned int j = 0; j < h; j++)
    {
      for (unsigned int i = 0; i < w; i++)
      {
	if (grey)
	{
	  pixels[j * w + i].red = *data;
	  pixels[j * w + i].green = *data;
	  pixels[j * w + i].blue = *data++;
	}
	else
	{
	  pixels[j * w + i].blue = *data++;
	  pixels[j * w + i].green = *data++;
	  pixels[j * w + i].red = *data++;
	}

	if (alpha)
	{
	  pixels[j * w + i].opacity = *data++;
	}
	else
	{
	  pixels[j * w + i].opacity = 0xffff;
	}
      }
    }
  }
  else
  {
    unsigned char *data = (unsigned char *)nrrd->data;

    // Copy pixels from nrrd to Image.
    for (unsigned int j = 0; j < h; j++)
    {
      for (unsigned int i = 0; i < w; i++)
      {
	if (grey)
	{
	  pixels[j * w + i].red = *data << 8;
	  pixels[j * w + i].green = *data << 8;
	  pixels[j * w + i].blue = *data++ << 8;
	}
	else
	{
	  pixels[j * w + i].blue = *data++ << 8;
	  pixels[j * w + i].green = *data++ << 8;
	  pixels[j * w + i].red = *data++ << 8;
	}

	if (alpha)
	{
	  pixels[j * w + i].opacity = *data++ << 8;
	}
	else
	{
	  pixels[j * w + i].opacity = 0xffff;
	}
      }
    }
  }

  C_Magick::SyncImagePixels(image);

  C_Magick::WriteImage(image_info, image);
  
  C_Magick::DestroyImage(image);
  C_Magick::DestroyImageInfo(image_info);
  C_Magick::DestroyMagick();
#else
  error("ImageMagick not found.  Please verify that you have the application development installation of ImageMagick.");
  return;
#endif
}


