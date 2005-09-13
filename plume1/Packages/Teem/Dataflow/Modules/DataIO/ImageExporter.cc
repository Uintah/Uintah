/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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
#include <Dataflow/Ports/NrrdPort.h>
#include <Core/GuiInterface/GuiVar.h>
#include <sci_defs/image_defs.h>

#ifdef HAVE_MAGICK
namespace C_Magick {
#include <magick/api.h>
}
#endif

using namespace SCIRun;


class ImageExporter : public Module {
  GuiFilename filename_;
  GuiString   filetype_;
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


#ifdef HAVE_MAGICK
static
C_Magick::Quantum
TO_QUANTUM(double f)
{
  if (sizeof(C_Magick::Quantum) == 1)
  {
    int tmp = (int)(f * 0xff);
    if (tmp > 0xff) return 0xff;
    if (tmp < 0) return 0;
    return tmp;
  }
  else
  {
    int tmp = (int)(f * 0xffff);
    if (tmp > 0xffff) return 0xffff;
    if (tmp < 0) return 0;
    return tmp;
  }
}
#endif

void
ImageExporter::execute()
{
  // Read data from the input port
  NrrdDataHandle handle;
  NrrdIPort* inport = (NrrdIPort *)get_iport("Input Data");

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

  if (!(nrrd->type == nrrdTypeUShort || nrrd->type == nrrdTypeUChar ||
	nrrd->type == nrrdTypeFloat || nrrd->type == nrrdTypeDouble))
  {
    error("Only Nrrds of type UShort and UChar, Float, and Double are currently supported.");
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
  image->matte = alpha;

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
	  pixels[j * w + i].red = *data++;
	  pixels[j * w + i].green = *data++;
	  pixels[j * w + i].blue = *data++;
	}

	if (alpha)
	{
	  pixels[j * w + i].opacity = 0xffff - *data++;
	}
	else
	{
	  pixels[j * w + i].opacity = 0;
	}
      }
    }
  }
  else if (nrrd->type == nrrdTypeUChar)
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
	  pixels[j * w + i].red = *data++ << 8;
	  pixels[j * w + i].green = *data++ << 8;
	  pixels[j * w + i].blue = *data++ << 8;
	}

	if (alpha)
	{
	  pixels[j * w + i].opacity = 0xffff - (*data++ << 8);
	}
	else
	{
	  pixels[j * w + i].opacity = 0;
	}
      }
    }
  }
  else if (nrrd->type == nrrdTypeFloat)
  {
    float *data = (float *)nrrd->data;

    // Copy pixels from nrrd to Image.
    for (unsigned int j = 0; j < h; j++)
    {
      for (unsigned int i = 0; i < w; i++)
      {
	if (grey)
	{
	  pixels[j * w + i].red = TO_QUANTUM(*data);
	  pixels[j * w + i].green = TO_QUANTUM(*data);
	  pixels[j * w + i].blue = TO_QUANTUM(*data++);
	}
	else
	{
	  pixels[j * w + i].red = TO_QUANTUM(*data++);
	  pixels[j * w + i].green = TO_QUANTUM(*data++);
	  pixels[j * w + i].blue = TO_QUANTUM(*data++);
	}

	if (alpha)
	{
	  pixels[j * w + i].opacity = 0xffff - TO_QUANTUM(*data++);
	}
	else
	{
	  pixels[j * w + i].opacity = 0;
	}
      }
    }
  }
  else if (nrrd->type == nrrdTypeDouble)
  {
    double *data = (double *)nrrd->data;

    // Copy pixels from nrrd to Image.
    for (unsigned int j = 0; j < h; j++)
    {
      for (unsigned int i = 0; i < w; i++)
      {
	if (grey)
	{
	  pixels[j * w + i].red = TO_QUANTUM(*data);
	  pixels[j * w + i].green = TO_QUANTUM(*data);
	  pixels[j * w + i].blue = TO_QUANTUM(*data++);
	}
	else
	{
	  pixels[j * w + i].red = TO_QUANTUM(*data++);
	  pixels[j * w + i].green = TO_QUANTUM(*data++);
	  pixels[j * w + i].blue = TO_QUANTUM(*data++);
	}

	if (alpha)
	{
	  pixels[j * w + i].opacity = 0xffff - TO_QUANTUM(*data++);
	}
	else
	{
	  pixels[j * w + i].opacity = 0;
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


