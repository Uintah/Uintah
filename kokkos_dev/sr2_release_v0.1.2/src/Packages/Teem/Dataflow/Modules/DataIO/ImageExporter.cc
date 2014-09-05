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
 *  ImageExporter.cc: Use PNG and ImageMagick's convert to write 
 *                    various image formats.
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
#include <Dataflow/Network/Ports/NrrdPort.h>
#include <Core/GuiInterface/GuiVar.h>
#include <sci_defs/image_defs.h>

#if defined HAVE_PNG && HAVE_PNG
#include <png.h>
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
    filename_(get_ctx()->subVar("filename"), ""),
    filetype_(get_ctx()->subVar("filetype"), "Binary")
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

  if(!inport->get(handle))
  {
    return;
  }
  
  if (!handle.get_rep())
  {
    error("Null input");
    return;
  }
  
#if defined HAVE_PNG && HAVE_PNG

  // If no name is provided, return.
  string fn(filename_.get());
  if(fn == "")
    {
      error("Warning: no filename in ImageExporter");
      return;
    }
  
  const Nrrd *nrrd = handle->nrrd_;
  
  if (nrrd->dim != 3)
    {
      error("Only 3 dimensional nrrds at this time (1-4, height, width).");
      return;
    }

  const unsigned int w = nrrd->axis[1].size;
  const unsigned int h = nrrd->axis[2].size;


  bool grey = false;
  bool alpha = true;
  unsigned int color_type = PNG_COLOR_TYPE_GRAY;
  if (nrrd->axis[0].size == 1)
    {
      grey = true;
      alpha = false;
      color_type = PNG_COLOR_TYPE_GRAY;
    }
  if (nrrd->axis[0].size == 2)
    {
      grey = true;
      alpha = true;
      color_type = PNG_COLOR_TYPE_GRAY_ALPHA;
    }
  else if (nrrd->axis[0].size == 3)
    {
      grey = false;
      alpha = false;
      color_type = PNG_COLOR_TYPE_RGB;
    }
  else if (nrrd->axis[0].size == 4)
    {
      grey = false;
      alpha = true;
      color_type = PNG_COLOR_TYPE_RGB_ALPHA;
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
  
  unsigned int type_size = 8;
  Nrrd *convert_nrrd = handle->nrrd_;

  if (nrrd->type == nrrdTypeUShort) {
    NrrdRange *range = nrrdRangeNewSet(handle->nrrd_, nrrdBlind8BitRangeState);
    convert_nrrd = nrrdNew();
    nrrdQuantize(convert_nrrd, nrrd, range, 8);
    type_size = 8;
  }
  else if (nrrd->type == nrrdTypeUChar) {
    type_size = 8;
  }
  else if (nrrd->type == nrrdTypeFloat) {
    NrrdRange *range = nrrdRangeNewSet(handle->nrrd_, nrrdBlind8BitRangeState);
    convert_nrrd = nrrdNew();
    nrrdQuantize(convert_nrrd, nrrd, range, 8);
    type_size = 8;
  }
  else {
    NrrdRange *range = nrrdRangeNewSet(handle->nrrd_, nrrdBlind8BitRangeState);
    convert_nrrd = nrrdNew();
    nrrdQuantize(convert_nrrd, nrrd, range, 8);
    type_size = 8;
  }

  png_structp png;
  png_infop info;
  png_bytep *row;

  /* create png struct */
  png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  
  if (png == NULL) {
    error("ERROR - failed to create PNG write struct\n");
    return;
  }
  
  /* create image info struct */
  info = png_create_info_struct(png);
  
  if (info == NULL) {
    error("ERROR - Failed to create PNG info struct\n");
    png_destroy_write_struct(&png, NULL);
    return;
  }
  
  ASSERT(sci_getenv("SCIRUN_TMP_DIR"));
  
  const char * tmp_dir(sci_getenv("SCIRUN_TMP_DIR"));
  
  const string tmp_file = string (tmp_dir + string("/scirun_temp_png.png")); 


  // write out a temporary png file
  FILE *fp;
  bool use_convert = false;
  string ext = fn.substr(fn.find(".",0)+1, fn.length());

  for(int i=0; i<(int)ext.size(); i++)
    ext[i] = tolower(ext[i]);
  
  if (ext != "png") {
    // test for convert program
    if (system("convert -version") != 0) {
      error(string("Unsupported extension " + ext + ". Program convert not found in path."));
      return;
    }
    use_convert = true;
    fp = fopen(tmp_file.c_str(), "wb");
  } else {
    fp = fopen(fn.c_str(), "wb");
  }

  // initialize IO 
  png_init_io(png, fp);

  png_set_IHDR(png, info, w, h,
	       type_size, color_type, PNG_INTERLACE_NONE,
	       PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
  
  /* write header */
  png_write_info(png, info);
  
  row = (png_bytep*)malloc(sizeof(png_bytep)*h);
  for (int hi=0; hi<(int)h; hi++) {
    row[hi] = &((png_bytep)convert_nrrd->data)[hi * w * 
					       convert_nrrd->axis[0].size];
  }
  png_set_rows(png, info, row);
  
  /* write the entire image in one pass */
  png_write_image(png, row);
  
  /* finish writing */
  png_write_end(png, info);
  
  /* clean up */
  row = (png_bytep*)airFree(row);   
  png_destroy_write_struct(&png, &info);
  fclose(fp);

  if (use_convert) {
    // convert from temporary png to correct type 
    // and remove temporary png
    if (system(string("convert " + tmp_file + " " + fn).c_str()) != 0) {
      error("Error using convert to write image.");
    }
    system(string("rm " + tmp_file).c_str());
  }
  
#else
  error("PNG library not found.");
  return;
#endif
}


