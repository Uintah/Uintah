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

#include <iostream>

namespace SCIRun {


using namespace std;

void
nrrd_build_bricks(vector<TextureBrickHandle>& bricks,
                  int nx, int ny, int nz,
                  int nc, int* nb, int card_mem)
{
  const bool force_pow2 = !ShaderProgramARB::texture_non_power_of_two();

  const int brick_mem = card_mem*1024*1024/2;
  
  const unsigned int max_texture_size =
    (nb[0] == 1)?
    ShaderProgramARB::max_texture_size_1() :
    ShaderProgramARB::max_texture_size_4();

  // Initial brick size
  int bsize[3];
  bsize[0] = Min(Pow2(nx), max_texture_size);
  bsize[1] = Min(Pow2(ny), max_texture_size);
  bsize[2] = Min(Pow2(nz), max_texture_size);
  if (force_pow2)
  {
    if (Pow2(nx) > nx) bsize[0] = Min(Pow2(nx)/2, max_texture_size);
    if (Pow2(ny) > ny) bsize[1] = Min(Pow2(ny)/2, max_texture_size);
    if (Pow2(nz) > nz) bsize[2] = Min(Pow2(nz)/2, max_texture_size);
  }
  
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
        
        int mx2 = mx;
        int my2 = my;
        int mz2 = mz;
        if (force_pow2)
        {
          mx2 = Pow2(mx);
          my2 = Pow2(my);
          mz2 = Pow2(mz);
        }

        // Compute Texture Box.
        const double tx0 = i?((mx2 - mx + 0.5) / mx2): 0.0;
        const double ty0 = j?((my2 - my + 0.5) / my2): 0.0;
        const double tz0 = k?((mz2 - mz + 0.5) / mz2): 0.0;
        
        double tx1 = 1.0 - 0.5 / mx2;
        if (mx < bsize[0]) tx1 = 1.0;
        if (nx - i == bsize[0]) tx1 = 1.0;

        double ty1 = 1.0 - 0.5 / my2;
        if (my < bsize[1]) ty1 = 1.0;
        if (ny - j == bsize[1]) ty1 = 1.0;

        double tz1 = 1.0 - 0.5 / mz2;
        if (mz < bsize[2]) tz1 = 1.0;
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
                                  mx2, my2, mz2,
                                  nc, nb,
                                  i-(mx2-mx), j-(my2-my), k-(mz2-mz),
                                  mx2, my2, mz2,
                                  bbox, tbox);
        bricks.push_back(b);
      }
    }
  }
}



void
NrrdTextureBuilderAlgo::build(TextureHandle tHandle,
                              NrrdDataHandle vHandle,
                              double vmin, double vmax,
                              NrrdDataHandle gHandle,
                              double gmin, double gmax,
                              int card_mem,
                              int is_uchar)
{
  Nrrd* nv_nrrd = vHandle->nrrd_;
  Nrrd* gm_nrrd = (gHandle.get_rep() ? gHandle->nrrd_ : 0);

  size_t axis_size[4];
  nrrdAxisInfoGet_nva(nv_nrrd, nrrdAxisInfoSize, axis_size);
  double axis_min[4];
  nrrdAxisInfoGet_nva(nv_nrrd, nrrdAxisInfoMin, axis_min);
  double axis_max[4];
  nrrdAxisInfoGet_nva(nv_nrrd, nrrdAxisInfoMax, axis_max);

  const int nx = axis_size[nv_nrrd->dim-3];
  const int ny = axis_size[nv_nrrd->dim-2];
  const int nz = axis_size[nv_nrrd->dim-1];

  int nc = gm_nrrd ? 2 : 1;
  int nb[2];
  nb[0] = nv_nrrd->dim == 4 ? axis_size[0] : 1;
  nb[1] = gm_nrrd ? 1 : 0;

  const BBox bbox(Point(0,0,0), Point(1,1,1)); 

  Transform tform;
  string trans_str;
  // See if it's stored in the nrrd first.
  if (vHandle->get_property("Transform", trans_str) && trans_str != "Unknown")
  {
    double t[16];
    int old_index=0, new_index=0;
    for(int i=0; i<16; i++)
    {
      new_index = trans_str.find(" ", old_index);
      string temp = trans_str.substr(old_index, new_index-old_index);
      old_index = new_index+1;
      string_to_double(temp, t[i]);
    }
    tform.set(t);
  } 
  else
  {
    // Reconstruct the axis aligned transform.
    const Point nmin(axis_min[nv_nrrd->dim-3],
                     axis_min[nv_nrrd->dim-2],
                     axis_min[nv_nrrd->dim-1]);
    const Point nmax(axis_max[nv_nrrd->dim-3],
                     axis_max[nv_nrrd->dim-2],
                     axis_max[nv_nrrd->dim-1]);
    tform.pre_scale(nmax - nmin);
    tform.pre_translate(nmin.asVector());
  }

  tHandle->lock_bricks();
  vector<TextureBrickHandle>& bricks = tHandle->bricks();
  if (nx != tHandle->nx() || ny != tHandle->ny() || nz != tHandle->nz() ||
      nc != tHandle->nc() || nb[0] != tHandle->nb(0) ||
      card_mem != tHandle->card_mem() ||
      bbox.min() != tHandle->bbox().min() ||
      bbox.max() != tHandle->bbox().max() ||
      vmin != tHandle->vmin() ||
      vmax != tHandle->vmax() ||
      gmin != tHandle->gmin() ||
      gmax != tHandle->gmax() )
  {
    // NrrdTextureBricks can be used if specifically requested or if
    // the data is unisgned chars with no rescaling.
    bool use_nrrd_brick =
      is_uchar || (vHandle->nrrd_->type == nrrdTypeUChar &&
		   vmin == 0 && vmax == 255 &&
		   gHandle->nrrd_->type == nrrdTypeUChar &&
		   gmin == 0 && gmax == 255);

    if( use_nrrd_brick &&
        ShaderProgramARB::shaders_supported() )
    {
      nrrd_build_bricks(bricks, nx, ny, nz, nc, nb, card_mem);
    }
    else
    {
      if( vHandle->nrrd_->type == nrrdTypeUChar &&
	  vmin == 0 && vmax == 255 &&
	  gHandle->nrrd_->type == nrrdTypeUChar &&
	  gmin == 0 && gmax == 255 )
      texture_build_bricks(bricks, nx, ny, nz, nc, nb, bbox, card_mem,
			   use_nrrd_brick );

    }
    tHandle->set_size(nx, ny, nz, nc, nb);
    tHandle->set_card_mem(card_mem);
  }
  tHandle->set_bbox(bbox);
  tHandle->set_minmax(vmin, vmax, gmin, gmax);
  tHandle->set_transform(tform);
  for (unsigned int i=0; i<bricks.size(); i++)
  {
    fill_brick(bricks[i], vHandle, vmin, vmax, gHandle, gmin, gmax,
	       nx, ny, nz);

    bricks[i]->set_dirty(true);
  }
  tHandle->unlock_bricks();
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
