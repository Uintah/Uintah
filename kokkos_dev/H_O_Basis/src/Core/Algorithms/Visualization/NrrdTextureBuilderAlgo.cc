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
//    File   : NrrdTextureBuilderAlgo.cc
//    Author : Milan Ikits
//    Date   : Fri Jul 16 02:56:39 2004

#include <Core/Algorithms/Visualization/NrrdTextureBuilderAlgo.h>

namespace SCIRun {

using namespace std;

void
nrrd_build_bricks(vector<TextureBrickHandle>& bricks,
		  int nx, int ny, int nz,
		  int nc, int* nb,
		  const BBox& ignored, int card_mem)
{
  const int brick_mem = card_mem*1024*1024/2;
  
  const unsigned int max_texture_size =
    (nb[0] == 1)?
    ShaderProgramARB::max_texture_size_1() :
    ShaderProgramARB::max_texture_size_4();

  // Initial brick size
  int bsize[3];
  bsize[0] = Min(NextPowerOf2(nx), max_texture_size);
  bsize[1] = Min(NextPowerOf2(ny), max_texture_size);
  bsize[2] = Min(NextPowerOf2(nz), max_texture_size);
  
  // Determine brick size here.

  // Slice largest axis, weighted by fastest/slowest memory access
  // axis so that our cuts leave us with contiguous blocks of data.
  // Currently set at 4x2x1 blocks.
  while (bsize[0] * bsize[1] * bsize[2] * nb[0] > brick_mem)
  {
    if (bsize[1] / bsize[2] >= 4 || bsize[2] < 4)
    {
      if (bsize[0] / bsize[1] >= 2 || bsize[1] < 4)
      {
	bsize[0] /= 2;
      }
      else
      {
	bsize[1] /= 2;
      }
    }
    else
    {
      bsize[2] /= 2;
    }
  }

  bricks.clear();

  for (int k = 0; k < nz; k += bsize[2])
  {
    if (k) k--;
    for (int j = 0; j < ny; j += bsize[1])
    {
      if (j) j--;
      for (int i = 0; i < nx; i += bsize[0])
      {
	if (i) i--;
	const int mx = Min(bsize[0], nx - i);
	const int my = Min(bsize[1], ny - j);
	const int mz = Min(bsize[2], nz - k);

	// Compute Texture Box.
	const double tx0 = i?(0.5 / bsize[0]): 0.0;
	const double ty0 = j?(0.5 / bsize[1]): 0.0;
	const double tz0 = k?(0.5 / bsize[2]): 0.0;
	
	double tx1 = 1.0 - 0.5 / bsize[0];
	if (mx < bsize[0]) tx1 = 1.0; //(mx + 0.5) / (double)bsize[0];
	if (nx - i == bsize[0]) tx1 = 1.0;

	double ty1 = 1.0 - 0.5 / bsize[1];
	if (my < bsize[1]) ty1 = 1.0; //(my + 0.5) / (double)bsize[1];
	if (ny - j == bsize[1]) ty1 = 1.0;

	double tz1 = 1.0 - 0.5 / bsize[2];
	if (mz < bsize[2]) tz1 = 1.0; //(mz + 0.5) / (double)bsize[2];
	if (nz - k == bsize[2]) tz1 = 1.0;

	BBox tbox(Point(tx0, ty0, tz0), Point(tx1, ty1, tz1));

	// Compute BBox.
	double bx1 = Min((i + bsize[0] - 0.5) / (double)nx, 1.0);
	if (nx - i == bsize[0]) bx1 = 1.0;

	double by1 = Min((j + bsize[1] - 0.5) / (double)ny, 1.0);
	if (ny - j == bsize[1]) by1 = 1.0;

	double bz1 = Min((k + bsize[2] - 0.5) / (double)nz, 1.0);
	if (nz - k == bsize[2]) bz1 = 1.0;

	BBox bbox(Point(i==0?0:(i+0.5) / (double)nx,
			j==0?0:(j+0.5) / (double)ny,
			k==0?0:(k+0.5) / (double)nz),
		  Point(bx1, by1, bz1));

	NrrdTextureBrick *b =
	  scinew NrrdTextureBrick(0, 0,
				  mx, my, mz, //bsize[0], bsize[1], bsize[2],
				  nc, nb,
				  i, j, k,
				  mx, my, mz,
				  bbox, tbox);
	bricks.push_back(b);
      }
    }
  }
}


CompileInfoHandle
NrrdTextureBuilderAlgo::get_compile_info( const unsigned int vtype,
					  const unsigned int gtype)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("NrrdTextureBuilderAlgoT");
  static const string base_class_name("NrrdTextureBuilderAlgo");

  string vTypeStr,  gTypeStr;
  string vTypeName, gTypeName;

  get_nrrd_compile_type( vtype, vTypeStr, vTypeName );
  get_nrrd_compile_type( gtype, gTypeStr, gTypeName );

  CompileInfo *rval =
    scinew CompileInfo(template_class_name + "." +
		       vTypeName + "." + gTypeName + ".",
		       base_class_name, 
		       template_class_name, 
		       vTypeStr + ", " + gTypeStr );
  
  // Add in the include path to compile this obj
  rval->add_include(include_path);  
  rval->add_namespace("SCIRun");

  return rval;
}

} // namespace SCIRun
