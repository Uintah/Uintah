//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//    File   : GeoTiffToImageFld.cc
//    Author : Martin Cole
//    Date   : Mon Nov  6 08:51:11 2006

#include <Core/Basis/Constant.h>
#include <Core/Basis/QuadBilinearLgn.h>
#include <Core/Datatypes/ImageMesh.h>
#include <Core/Containers/FData.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Persistent/Pstreams.h>

#include <geotiff.h>
#include <geo_normalize.h>
#include <geovalues.h>
#include <tiffio.h>
#include <xtiffio.h>


#include <iostream>
#include <fstream>
#include <stack>
#include <stdlib.h>

using std::cerr;
using std::ifstream;
using std::endl;
using std::stack;

using namespace SCIRun;

typedef ImageMesh<QuadBilinearLgn<Point> >                   IMesh;
typedef QuadBilinearLgn<Vector>                              DBasisrgb;
typedef GenericField<IMesh, DBasisrgb, FData2d<Vector, IMesh> > IFieldrgb;
typedef QuadBilinearLgn<double>                              DBasisgs;
typedef GenericField<IMesh, DBasisgs, FData2d<double, IMesh> > IFieldgs;




BBox 
get_bounds(GTIF *gtif, GTIFDefn *defn, int xsize, int ysize) 
{
  double x = 0.0;
  double y = 0.0;
  double tx = x;
  double ty = y;

  cerr << "Corner Coordinates:" << endl;
  if(!GTIFImageToPCS(gtif, &tx, &ty)) 
  {
    cerr << "unable to transform points between pixel/line and PCS space"
	 << endl;
    //return BBox();
  }
  BBox bb;

  tx = 0.0;
  ty = ysize;
  GTIFImageToPCS(gtif, &tx, &ty);
  cerr << "LL: " << tx << ", " << ty << endl;
  bb.extend(Point(tx, ty, 0.0));

  tx = xsize;
  ty = 0.0;
  GTIFImageToPCS(gtif, &tx, &ty);
  cerr << "UR: " << tx << ", " << ty << endl;
  bb.extend(Point(tx, ty, 0.0));

//   tx = xsize;
//   ty = ysize;
//   GTIFImageToPCS(gtif, &tx, &ty);
//   cerr << "LR" << tx << ", " << ty << endl;

  return bb;
}

Vector
get_value(uint32* buf, unsigned int idx, int bpp, int spp) 
{  
  if (spp == 1 && bpp == 8) {
    uint32 p = buf[idx];
    unsigned char r = TIFFGetR(p);
    unsigned char g = TIFFGetG(p);
    unsigned char b = TIFFGetB(p);
    //cerr << "rgb: " << (int)r << ", " << (int)g << ", " << (int)b << endl;
    return Vector(r / 255., g / 255., b / 255.);
  }

  cerr << "WARNING: default get_value is 0" << endl;
  return Vector(0,0,0);
}


template <class Fld>
bool 
fill_image(Fld *fld, IMesh *im, int bpp, int spp, uint32 *buf)
{
  IMesh::Node::iterator iter, end;
  im->begin(iter);
  im->end(end);
  
  unsigned int idx = 0;
  while (iter != end) {
    typename Fld::value_type val = get_value(buf, idx, bpp, spp);
    IMesh::Node::index_type ni = *iter;
    fld->set_value(val, ni);

    ++iter;
    ++idx;
  }
  
  return true;
}


int
main(int argc, char **argv) {

  const char* tmpfn = "/tmp/dddas.tif";
  TIFF* tif = XTIFFOpen(tmpfn, "r");
  if (!tif) {
    cerr << "Error opening: " << tmpfn << endl;
    return 1<< 2;
  }

  GTIF* gtif = GTIFNew(tif);
  GTIFDefn	defn;
  FieldHandle out_fld_h;        
  if(GTIFGetDefn(gtif, &defn))
  {
    int xsize, ysize;
    uint16 spp, bpp, photo;

    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &xsize);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &ysize);
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bpp);
    TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &spp);
    TIFFGetField(tif, TIFFTAG_PHOTOMETRIC, &photo);
    cerr << "tiff size: (" << xsize << ", " << ysize << ")" << endl; 
    cerr << "bits/pixel: " << bpp << endl;
    cerr << "samples/pixel: " << spp << endl;
    cerr << "photo: " << photo << endl;


    int npixels = xsize * ysize;
    uint32* raster = (uint32*) _TIFFmalloc(npixels * sizeof (uint32));
    if (raster != NULL) {
      if (TIFFReadRGBAImage(tif, xsize, ysize, raster, 0)) {
	BBox bb = get_bounds(gtif, &defn, xsize, ysize);
	cerr << "bb.min: " << bb.min() << endl;
	cerr << "bb.max: " << bb.max() << endl;
	IMesh* im = new IMesh(xsize, ysize, bb.min(), bb.max());
	BBox cbb = im->get_bounding_box();
	cerr << "cbb.min: " << cbb.min() << endl;
	cerr << "cbb.max: " << cbb.max() << endl;	
	IFieldrgb *ifld = new IFieldrgb(im);
	fill_image(ifld, im, bpp, spp, raster);
	out_fld_h = ifld;
	cerr << "read image" << endl;
      } else {
	cerr << "could not read image" << endl;
	return 0;
      }
      //_TIFFfree(raster);
    }
    TIFFClose(tif);


  } else {
    cerr << "GTIFDefn Failed " << endl;
      
  }

  //   TVMesh::handle_type tvmH(tvm);
  //   TVField *tv = scinew TVField(tvmH);

  //   for (i=0; i<nnodes; i++)
  //     tv->fdata()[i]=vals[i];
  //   FieldHandle tvH(tv);

  TextPiostream out_stream("/tmp/dddas.img.fld", Piostream::Write);
  Pio(out_stream, out_fld_h);
  return 0;
}
