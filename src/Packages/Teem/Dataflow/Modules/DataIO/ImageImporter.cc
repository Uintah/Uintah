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
 *  NrrdReader.cc: Read in a Nrrd
 *
 *  Written by:
 *   David Weinstein
 *   School of Computing
 *   University of Utah
 *   February 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Containers/StringUtil.h>
#include <Teem/Dataflow/Ports/NrrdPort.h>
#include <sys/stat.h>

#ifdef HAVE_MAGICK
namespace C_Magick {
#include <magick/api.h>
}
#endif


using namespace SCIRun;
using namespace SCITeem;


class ImageImporter : public Module {
public:
  ImageImporter(SCIRun::GuiContext* ctx);
  virtual ~ImageImporter();
  virtual void execute();

private:
  GuiString       filename_;
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
  } else if (stat(fn.c_str(), &buf)) {
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

    if (sizeof(C_Magick::Quantum) == 1)
    {
      // Unsigned char
      size_t dsize = image->rows * image->columns * 4;
      unsigned char *dptr = scinew unsigned char[dsize];
      memcpy(dptr, pixels, dsize);
      Nrrd *nrrd = nrrdNew();
      if (nrrdWrap(nrrd, dptr, nrrdTypeUChar,
		   3, 4, image->columns, image->rows))
      {
	char *err = biffGetDone(NRRD);
	error(string("Error creating NRRD: ") + err + "\n");
	free(err);
      }
      handle_ = scinew NrrdData();
      handle_->nrrd = nrrd;
    }
    else if (sizeof(C_Magick::Quantum) == 2)
    {
      // Unsigned short
      size_t dsize = image->rows * image->columns * 4;
      unsigned short *dptr = scinew unsigned short[dsize];
      memcpy(dptr, pixels, dsize * 2);
      Nrrd *nrrd = nrrdNew();
      if (nrrdWrap(nrrd, dptr, nrrdTypeUShort,
		   3, 4, image->columns, image->rows))
      {
	char *err = biffGetDone(NRRD);
	error(string("Error creating NRRD: ") + err + "\n");
	free(err);
      }
      handle_ = scinew NrrdData();
      handle_->nrrd = nrrd;
    }
    else
    {
      remark("Quantum = " + to_string(sizeof(C_Magick::Quantum)));
      handle_ = 0;
    }

    
    C_Magick::DestroyImage(image);

    // Clean up Imagemagick
    C_Magick::DestroyImageInfo(image_info);
    C_Magick::DestroyExceptionInfo(&exception);
    C_Magick::DestroyMagick();
  }

  // Send the data downstream.
  NrrdOPort *outport = (NrrdOPort *)getOPort(0);
  if (!outport)
  {
    error("Unable to initialize oport.");
    return;
  }
  outport->send(handle_);
}

