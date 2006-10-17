/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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
 *  ImportImage.cc: Use PNG and ImageMagick's convert to import 
 *                    numerous image formats.
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
#include <Dataflow/GuiInterface/GuiVar.h>
#include <Core/Containers/StringUtil.h>
#include <Dataflow/Network/Ports/NrrdPort.h>
#include <sci_defs/image_defs.h>
#include <sys/stat.h>

#if defined HAVE_PNG && HAVE_PNG
#include <png.h>
#endif


using namespace SCIRun;


class ImportImage : public Module {
public:
  ImportImage(SCIRun::GuiContext* ctx);
  virtual ~ImportImage();
  virtual void execute();

private:
  GuiFilename     filename_;
  string          old_filename_;
  time_t          old_filemodification_;
  NrrdDataHandle  handle_;
};


DECLARE_MAKER(ImportImage)


ImportImage::ImportImage(SCIRun::GuiContext* ctx) : 
  Module("ImportImage", ctx, Filter, "DataIO", "Teem"),
  filename_(get_ctx()->subVar("filename"), ""),
  old_filemodification_(0),
  handle_(0)
{
}


ImportImage::~ImportImage()
{
}


void
ImportImage::execute()
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


#if defined HAVE_PNG && HAVE_PNG
  if (!handle_.get_rep() || 
      fn != old_filename_ || 
      new_filemodification != old_filemodification_)
    {
      old_filemodification_ = new_filemodification;
      old_filename_ = fn;

      ASSERT(sci_getenv("SCIRUN_TMP_DIR"));
      
      const char * tmp_dir(sci_getenv("SCIRUN_TMP_DIR"));
      
      const string tmp_file = string (tmp_dir + string("/scirun_temp_png.png")); 
      FILE *fp = NULL;
      string ext = fn.substr(fn.find(".",0)+1, fn.length());

      for(int i=0; i<(int)ext.size(); i++) {
	ext[i] = tolower(ext[i]);
      }
      
      if (ext != "png") {
	// test for convert program
	if (system("convert -version") != 0) {
	  error(string("Unsupported extension " + ext + ". Program convert not found in path."));
	  return;
	}
	if (system(string("convert " + fn + " " + tmp_file).c_str())!= 0) {
	  error("Error using convert to write temporary png");
	  return;
	}
	fp = fopen(tmp_file.c_str(), "rb");
      }
      else
      {
        fp = fopen(fn.c_str(), "rb");
      }

      png_structp png;
      png_infop info;
      png_bytep *row;
      
      /* create png struct */
      png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

      if (!png) {
	error("Error creating PNG read struct.");
	system(string("rm " + tmp_file).c_str());
	return;
      }
      
      /* create info struct */
      info = png_create_info_struct(png);
      if (!info) {
	png_destroy_read_struct(&png, NULL, NULL);
	error("Error creating PNG ingo struct.");
	system(string("rm " + tmp_file).c_str());
	return;
      }
      
      /* init io */
      png_init_io(png, fp);

      png_read_info(png, info);
      
      png_uint_32 w, h;
      int depth, type, interlace_type;
      png_get_IHDR(png, info, &w, &h, &depth, &type, &interlace_type, 
		   NULL, NULL);
      
      if (interlace_type == PNG_INTERLACE_ADAM7) {
	error("Interlaced images not currently supported.");
	png_destroy_read_struct(&png, &info, NULL);
	system(string("rm " + tmp_file).c_str());
	return;
      }

      /* create Nrrd */
      unsigned int nrrd_type = nrrdTypeUChar;
      if (depth == 8) {
	nrrd_type = nrrdTypeUChar;
      }
      else if (depth == 16) {
	nrrd_type = nrrdTypeUShort;
      } 

      int type_size = 3;
      if (type == PNG_COLOR_TYPE_GRAY) 
	type_size = 1;
      else if (type == PNG_COLOR_TYPE_GRAY_ALPHA)
	type_size = 2;
      else if (type == PNG_COLOR_TYPE_RGB)
	type_size = 3;
      else if (type == PNG_COLOR_TYPE_RGB_ALPHA)
	type_size = 4;
      else {
	error("Unknown type");
	png_destroy_read_struct(&png, &info, NULL);
	system(string("rm " + tmp_file).c_str());
	return;
      }
      
      Nrrd *nrrd = nrrdNew();

      size_t size[NRRD_DIM_MAX];
      size[0] = type_size; size[1] = w;
      size[2] = h;
      if (nrrdAlloc_nva(nrrd,nrrd_type, 3, size)) {
	error("Error allocating NRRD.");
	png_destroy_read_struct(&png, &info, NULL);
	system(string("rm " + tmp_file).c_str());
	return;
      }

      if (nrrd->axis[0].size == 3)
	nrrd->axis[0].kind = nrrdKind3Vector;
      
      png_uint_32 rowsize = png_get_rowbytes(png, info);

      row = (png_bytep*)malloc(sizeof(png_bytep)*h);
      for(int y = 0; y < (int)h; y++) {
	row[y] = &((png_bytep)nrrd->data)[y*rowsize];
      }

      png_read_image(png, row); 

      png_read_end(png, 0);

      png_destroy_read_struct(&png, &info, NULL);
      row = (png_bytep*)airFree(row);

      fclose(fp);
      system(string("rm " + tmp_file).c_str());

      
      //   Send the data downstream.
      handle_ = scinew NrrdData();
      handle_->nrrd_ = nrrd;
      
      NrrdOPort *outport = (NrrdOPort *)get_output_port(0);
      outport->send(handle_);
      
    }
#else
  error("PNG library not found.");
#endif

}

