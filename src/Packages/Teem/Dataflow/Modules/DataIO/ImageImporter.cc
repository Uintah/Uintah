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
 *  ImageImporter.cc: Use ImageMagick to import numerous image formats.
 *
 *  Written by:
 *   Michael Callahan
 *   School of Computing
 *   University of Utah
 *   April 2004
 *
 *  Copyright (C) 2004 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Containers/StringUtil.h>
#include <Dataflow/Ports/NrrdPort.h>
#include <sci_defs/image_defs.h>
#include <sys/stat.h>

#ifdef HAVE_MAGICK
namespace C_Magick {
#include <magick/api.h>
}
#endif


using namespace SCIRun;


class ImageImporter : public Module {
public:
  ImageImporter(SCIRun::GuiContext* ctx);
  virtual ~ImageImporter();
  virtual void execute();

private:
  GuiFilename     filename_;
  string          old_filename_;
  time_t          old_filemodification_;
  NrrdDataHandle  handle_;
};


DECLARE_MAKER(ImageImporter)


ImageImporter::ImageImporter(SCIRun::GuiContext* ctx) : 
  Module("ImageImporter", ctx, Filter, "DataIO", "Teem"),
  filename_(ctx->subVar("filename")),
  old_filemodification_(0),
  handle_(0)
{
}


ImageImporter::~ImageImporter()
{
}


void
ImageImporter::execute()
{
  const string fn(filename_.get());

  // Read the status of this file so we can compare modification timestamps
  struct stat buf;
  if( fn == "" ) {
    error("No file has been selected.  Please choose a file.");
    return;
  } else if (stat(fn.c_str(), &buf) == -1) {
    error("File '" + fn + "' not found.");
    return;
  }

  // If we haven't read yet, or if it's a new filename, 
  //  or if the datestamp has changed -- then read...
#ifdef __sgi
  time_t new_filemodification = buf.st_mtim.tv_sec;
#else
  time_t new_filemodification = buf.st_mtime;
#endif

#ifdef HAVE_MAGICK
  if (!handle_.get_rep() || 
      fn != old_filename_ || 
      new_filemodification != old_filemodification_)
  {
    old_filemodification_ = new_filemodification;
    old_filename_ = fn;
    // Read in the imagemagic image.
    C_Magick::ImageInfo *image_info;
    C_Magick::Image *image;
    C_Magick::ExceptionInfo exception;

    C_Magick::InitializeMagick(0);
    C_Magick::GetExceptionInfo(&exception);
    image_info = C_Magick::CloneImageInfo((C_Magick::ImageInfo *) NULL);
    strncpy(image_info->filename, fn.c_str(), MaxTextExtent);
    image = C_Magick::ReadImage(image_info, &exception);
    if (image == 0 || exception.severity != C_Magick::UndefinedException)
    {
      warning("Unable to read image.");
      //C_Magick::CatchException(&exception);
      return;
    }

    C_Magick::SyncImage(image); // RGBA (maybe CMYK)
    const C_Magick::PixelPacket *pixels =
      C_Magick::AcquireImagePixels(image, 0, 0, image->columns, image->rows,
				   &exception);

    if (pixels == 0)
    {
      error("Unable to Acquire Image Pixels for some reason.");
      return;
    }

    size_t dsize = image->rows * image->columns;
    float *dptr = scinew float[dsize*3];
    const float iqsize = (sizeof(C_Magick::Quantum) == 1)?1.0/0xff:1.0/0xffff;
    for (unsigned int i=0; i < dsize; i++)
    {
      dptr[i*3+0] = pixels[i].red * iqsize;
      dptr[i*3+1] = pixels[i].green *iqsize;
      dptr[i*3+2] = pixels[i].blue *iqsize;
    }
    Nrrd *nrrd = nrrdNew();
    if (nrrdWrap(nrrd, dptr, nrrdTypeFloat,
		 3, 3, image->columns, image->rows))
    {
      char *err = biffGetDone(NRRD);
      error(string("Error creating NRRD: ") + err + "\n");
      free(err);
    }
    nrrd->axis[0].kind = nrrdKind3Vector;
    handle_ = scinew NrrdData();
    handle_->nrrd = nrrd;
    
    C_Magick::DestroyImage(image);

    // Clean up Imagemagick
    C_Magick::DestroyImageInfo(image_info);
    C_Magick::DestroyExceptionInfo(&exception);
    C_Magick::DestroyMagick();
  }

  // Send the data downstream.
  NrrdOPort *outport = (NrrdOPort *)getOPort(0);
  outport->send(handle_);
#else
  error("ImageMagick not found.  Please verify that you have the application development installation of ImageMagick.");
  return;
#endif
}

