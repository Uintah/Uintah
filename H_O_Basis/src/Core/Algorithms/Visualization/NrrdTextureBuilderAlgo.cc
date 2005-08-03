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
#include <Core/Util/DebugStream.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Geom/ShaderProgramARB.h>

#include <iostream>

namespace SCIRun {

using namespace std;


void
texture_build_bricks(vector<TextureBrickHandle>& bricks,
                     int nx, int ny, int nz,
                     int nc, int* nb,
                     const BBox& bbox, int card_mem,
                     bool use_nrrd_brick)
{
  int brick_mem = card_mem*1024*1024/2;
  const int data_size[3] = { nx, ny, nz };
  int brick_size[3];
  int brick_offset[3];
  int brick_pad[3];
  int num_brick[3];
  int brick_nb;
  // figure out largest possible brick size
  int size[3];
  size[0] = data_size[0]; size[1] = data_size[1]; size[2] = data_size[2];

  const unsigned int max_texture_size =
    (nb[0] == 1)?
    ShaderProgramARB::max_texture_size_1() :
    ShaderProgramARB::max_texture_size_4();
  // initial brick size
  brick_size[0] = Min(NextPowerOf2(data_size[0]), max_texture_size);
  brick_size[1] = Min(NextPowerOf2(data_size[1]), max_texture_size);
  brick_size[2] = Min(NextPowerOf2(data_size[2]), max_texture_size);
  // number of bytes per brick
  brick_nb = brick_size[0]*brick_size[1]*brick_size[2]*nb[0];
  
  // find brick size
  // subdivide until fits
  while (brick_nb > brick_mem)
  {
    // number of bricks along the axes
    // the divisions have to satisfy the equation:
    // data_size <= (brick_size-1)*num_brick + 1
    // we assume that brick_size > 1
    num_brick[0] = (int)ceil((double)(data_size[0]-1)/(brick_size[0]-1));
    num_brick[1] = (int)ceil((double)(data_size[1]-1)/(brick_size[1]-1));
    num_brick[2] = (int)ceil((double)(data_size[2]-1)/(brick_size[2]-1));
    // size of leftover volumes
    int sp[3];
    sp[0] = data_size[0] - (num_brick[0]-1)*(brick_size[0]-1);
    sp[1] = data_size[1] - (num_brick[1]-1)*(brick_size[1]-1);
    sp[2] = data_size[2] - (num_brick[2]-1)*(brick_size[2]-1);
    // size of padding
    brick_pad[0] = NextPowerOf2(sp[0]) - sp[0];
    brick_pad[1] = NextPowerOf2(sp[1]) - sp[1];
    brick_pad[2] = NextPowerOf2(sp[2]) - sp[2];
    // sort padding
    int idx[3];
    SortIndex(brick_pad, idx);
    // split largest one
    size[idx[2]] = (int)ceil(size[idx[2]]/2.0);
    brick_size[idx[2]] = NextPowerOf2(size[idx[2]]);
    brick_nb = brick_size[0]*brick_size[1]*brick_size[2]*nb[0];
  }

  // number of bricks along the axes
  // the divisions have to satisfy the equation:
  // data_size <= (brick_size-1)*num_brick + 1
  // we assume that brick_size > 1
  num_brick[0] = (int)ceil((double)(data_size[0]-1)/(brick_size[0]-1));
  num_brick[1] = (int)ceil((double)(data_size[1]-1)/(brick_size[1]-1));
  num_brick[2] = (int)ceil((double)(data_size[2]-1)/(brick_size[2]-1));
  // padded sizes of last bricks
  brick_pad[0] = NextPowerOf2(data_size[0] - (num_brick[0]-1)*(brick_size[0]-1));
  brick_pad[1] = NextPowerOf2(data_size[1] - (num_brick[1]-1)*(brick_size[1]-1));
  brick_pad[2] = NextPowerOf2(data_size[2] - (num_brick[2]-1)*(brick_size[2]-1));
  // delete previous bricks (if any)
  bricks.clear();
  bricks.reserve(num_brick[0]*num_brick[1]*num_brick[2]);

  // create bricks
  // data bbox
  double data_bbmin[3], data_bbmax[3];
  data_bbmin[0] = bbox.min().x();
  data_bbmin[1] = bbox.min().y();
  data_bbmin[2] = bbox.min().z();
  data_bbmax[0] = bbox.max().x();
  data_bbmax[1] = bbox.max().y();
  data_bbmax[2] = bbox.max().z();
  // bbox and tbox parameters
  double bmin[3], bmax[3], tmin[3], tmax[3];
  brick_offset[2] = 0;
  bmin[2] = data_bbmin[2];
  bmax[2] = num_brick[2] > 1 ?
    (double)(brick_size[2] - 0.5)/(double)data_size[2] *
    (data_bbmax[2] - data_bbmin[2]) + data_bbmin[2] : data_bbmax[2];
  tmin[2] = 0.0;
  tmax[2] = num_brick[2] > 1 ? (double)(brick_size[2] - 0.5)/(double)brick_size[2] :
    (double)data_size[2]/(double)brick_size[2];
  for (int k=0; k<num_brick[2]; k++)
  {
    brick_offset[1] = 0;
    bmin[1] = data_bbmin[1];
    bmax[1] = num_brick[1] > 1 ?
      (double)(brick_size[1] - 0.5)/(double)data_size[1] *
      (data_bbmax[1] - data_bbmin[1]) + data_bbmin[1] : data_bbmax[1];
    tmin[1] = 0.0;
    tmax[1] = num_brick[1] > 1 ?
      (double)(brick_size[1] - 0.5)/(double)brick_size[1] :
      (double)data_size[1]/(double)brick_size[1];
    for (int j=0; j<num_brick[1]; j++)
    {
      brick_offset[0] = 0;
      bmin[0] = data_bbmin[0];
      bmax[0] = num_brick[0] > 1 ?
        (double)(brick_size[0] - 0.5)/(double)data_size[0] *
        (data_bbmax[0] - data_bbmin[0]) + data_bbmin[0] : data_bbmax[0];
      tmin[0] = 0.0;
      tmax[0] = num_brick[0] > 1 ? (double)(brick_size[0] - 0.5)/(double)brick_size[0] :
        (double)data_size[0]/(double)brick_size[0];
      for (int i=0; i<num_brick[0]; i++)
      {
        if (use_nrrd_brick &&
            num_brick[0] * num_brick[1] * num_brick[2] == 1 &&
	    brick_pad[0] == nx && brick_pad[1] == ny && brick_pad[2] == nz)
	{
	  NrrdTextureBrick *b = scinew NrrdTextureBrick(0, 0,
            i < num_brick[0]-1 ? brick_size[0] : brick_pad[0],
            j < num_brick[1]-1 ? brick_size[1] : brick_pad[1],
            k < num_brick[2]-1 ? brick_size[2] : brick_pad[2],
            nc, nb,
            brick_offset[0], brick_offset[1], brick_offset[2],
            i < num_brick[0]-1 ? brick_size[0] : data_size[0] - (num_brick[0]-1)*(brick_size[0]-1),
            j < num_brick[1]-1 ? brick_size[1] : data_size[1] - (num_brick[1]-1)*(brick_size[1]-1),
            k < num_brick[2]-1 ? brick_size[2] : data_size[2] - (num_brick[2]-1)*(brick_size[2]-1),
            BBox(Point(bmin[0], bmin[1], bmin[2]),
                 Point(bmax[0], bmax[1], bmax[2])),
            BBox(Point(tmin[0], tmin[1], tmin[2]),
                 Point(tmax[0], tmax[1], tmax[2])));
          bricks.push_back(b);
	}
	else
	{
	  TextureBrick* b = scinew TextureBrickT<unsigned char>(
            i < num_brick[0]-1 ? brick_size[0] : brick_pad[0],
            j < num_brick[1]-1 ? brick_size[1] : brick_pad[1],
            k < num_brick[2]-1 ? brick_size[2] : brick_pad[2],
            nc, nb,
            brick_offset[0], brick_offset[1], brick_offset[2],
            i < num_brick[0]-1 ? brick_size[0] : data_size[0] - (num_brick[0]-1)*(brick_size[0]-1),
            j < num_brick[1]-1 ? brick_size[1] : data_size[1] - (num_brick[1]-1)*(brick_size[1]-1),
            k < num_brick[2]-1 ? brick_size[2] : data_size[2] - (num_brick[2]-1)*(brick_size[2]-1),
            BBox(Point(bmin[0], bmin[1], bmin[2]),
                 Point(bmax[0], bmax[1], bmax[2])),
            BBox(Point(tmin[0], tmin[1], tmin[2]),
                 Point(tmax[0], tmax[1], tmax[2])));
          bricks.push_back(b);
	}

        // update x parameters                     
        brick_offset[0] += brick_size[0]-1;
        bmin[0] = bmax[0];
        bmax[0] += i < num_brick[0]-2 ?
          (double)(brick_size[0]-1)/(double)data_size[0]*(data_bbmax[0] - data_bbmin[0]) :
          (double)(data_size[0]-brick_offset[0]-0.5)/(double)data_size[0]*(data_bbmax[0]-data_bbmin[0]);
        tmin[0] = i < num_brick[0]-2 ? 0.5/(double)brick_size[0] : 0.5/(double)brick_pad[0];
        tmax[0] = i < num_brick[0]-2 ? (double)(brick_size[0]-0.5)/(double)brick_size[0] : (data_size[0] - brick_offset[0])/(double)brick_pad[0];
      }
      // update y parameters
      brick_offset[1] += brick_size[1]-1;
      bmin[1] = bmax[1];
      bmax[1] += j < num_brick[1]-2 ?
        (double)(brick_size[1]-1)/(double)data_size[1] * (data_bbmax[1]-data_bbmin[1]) :
        (double)(data_size[1]-brick_offset[1]-0.5)/(double)data_size[1] * (data_bbmax[1]-data_bbmin[1]);
      tmin[1] = j < num_brick[1]-2 ? 0.5/(double)brick_size[1] : 0.5/(double)brick_pad[1];
      tmax[1] = j < num_brick[1]-2 ? (double)(brick_size[1]-0.5)/(double)brick_size[1] : (data_size[1]-brick_offset[1])/(double)brick_pad[1];
    } // j
    // update z parameters
    brick_offset[2] += brick_size[2]-1;
    bmin[2] = bmax[2];
    bmax[2] += k < num_brick[2]-2 ?
      (double)(brick_size[2]-1)/(double)data_size[2] * (data_bbmax[2]-data_bbmin[2]) :
      (double)(data_size[2]-brick_offset[2]-0.5)/(double)data_size[2] * (data_bbmax[2]-data_bbmin[2]);
    tmin[2] = k < num_brick[2]-2 ? 0.5/(double)brick_size[2] : 0.5/(double)brick_pad[2];
    tmax[2] = k < num_brick[2]-2 ? (double)(brick_size[2]-0.5)/(double)brick_size[2] : (data_size[2] - brick_offset[2])/(double)brick_pad[2];
  } // k
}



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



NrrdTextureBuilderAlgo::NrrdTextureBuilderAlgo()
{}

NrrdTextureBuilderAlgo::~NrrdTextureBuilderAlgo() 
{}

bool
NrrdTextureBuilderAlgo::build(ProgressReporter *report,
			      TextureHandle texture,
                              const NrrdDataHandle &nvn,
			      const NrrdDataHandle &gmn,
                              int card_mem)
{
  Nrrd* nv_nrrd = nvn->nrrd;
  if (nv_nrrd->dim != 3 && nv_nrrd->dim != 4)
  {
    report->error("Invalid dimension for input value nrrd.");
    return false;
  }

  Nrrd* gm_nrrd = 0;
  int axis_size[4];
  int nc;
  if (gmn.get_rep())
  {
    gm_nrrd = gmn->nrrd;
    if (gm_nrrd)
    {
      if (gm_nrrd->dim != 3 && gm_nrrd->dim != 4)
      {
        report->error("Invalid dimension for input gradient magnitude nrrd.");
        return false;
      }
      nrrdAxisInfoGet_nva(gm_nrrd, nrrdAxisInfoSize, axis_size);
      if (gm_nrrd->dim > 3 && axis_size[0] != 1)
      {
        report->error("Invalid size for gradient magnitude nrrd.");
        return false;
      }
      nc = 2;
    }
    else
    {
      nc = 1;
    }
  }
  else
  {
    nc = 1;
  }

  nrrdAxisInfoGet_nva(nv_nrrd, nrrdAxisInfoSize, axis_size);
  double axis_min[4];
  nrrdAxisInfoGet_nva(nv_nrrd, nrrdAxisInfoMin, axis_min);
  double axis_max[4];
  nrrdAxisInfoGet_nva(nv_nrrd, nrrdAxisInfoMax, axis_max);

  const int nx = axis_size[nv_nrrd->dim-3];
  const int ny = axis_size[nv_nrrd->dim-2];
  const int nz = axis_size[nv_nrrd->dim-1];
  int nb[2];
  nb[0] = nv_nrrd->dim > 3 ? axis_size[0] : 1;
  nb[1] = gm_nrrd ? 1 : 0;

  const BBox bbox(Point(0,0,0), Point(1,1,1)); 

  Transform tform;
  string trans_str;
  // See if it's stored in the nrrd first.
  if (nvn->get_property("Transform", trans_str) && trans_str != "Unknown")
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

  texture->lock_bricks();
  vector<TextureBrickHandle>& bricks = texture->bricks();
  if (nx != texture->nx() || ny != texture->ny() || nz != texture->nz() ||
      nc != texture->nc() || nb[0] != texture->nb(0) ||
      card_mem != texture->card_mem() ||
      bbox.min() != texture->bbox().min() ||
      bbox.max() != texture->bbox().max())
  {
    if (ShaderProgramARB::shaders_supported() &&
	ShaderProgramARB::texture_non_power_of_two())
    {
      nrrd_build_bricks(bricks, nx, ny, nz, nc, nb, bbox, card_mem);
    }
    else
    {
      texture_build_bricks(bricks, nx, ny, nz, nc, nb, bbox, card_mem, true);
    }
    texture->set_size(nx, ny, nz, nc, nb);
    texture->set_card_mem(card_mem);
  }
  texture->set_bbox(bbox);
  texture->set_minmax(0.0, 255.0, 0.0, 255.0);
  texture->set_transform(tform);
  for (unsigned int i=0; i<bricks.size(); i++)
  {
    fill_brick(bricks[i], nvn, gmn, nx, ny, nz);
    bricks[i]->set_dirty(true);
  }
  texture->unlock_bricks();
  return true;
}


void
NrrdTextureBuilderAlgo::fill_brick(TextureBrickHandle &brick,
				   const NrrdDataHandle &nvn,
				   const NrrdDataHandle &gmn,
                                   int ni, int nj, int /*nk*/)
{
  NrrdTextureBrick *nbrick =
    dynamic_cast<NrrdTextureBrick *>(brick.get_rep());
  if (nbrick)
  {
    nbrick->set_nrrds(nvn, gmn);
    return;
  }
  Nrrd *nv_nrrd = 0;
  if (nvn.get_rep()) { nv_nrrd = nvn->nrrd; }
  Nrrd *gm_nrrd = 0;
  if (gmn.get_rep()) { gm_nrrd = gmn->nrrd; }

  TextureBrickT<unsigned char>* br =
    dynamic_cast<TextureBrickT<unsigned char>*>(brick.get_rep());
  int nc = brick->nc();
  int nx = brick->nx();
  int ny = brick->ny();
  int nz = brick->nz();
  int x0 = brick->ox();
  int y0 = brick->oy();
  int z0 = brick->oz();
  int x1 = x0+brick->mx();
  int y1 = y0+brick->my();
  int z1 = z0+brick->mz();
  int i, j, k, jj, kk;
  if (nv_nrrd && ((gm_nrrd && nc == 2) || nc == 1)) {
    if (!gm_nrrd) {
      int nb = brick->nb(0);
      unsigned char* tex = br->data(0);
      unsigned char* data = (unsigned char*)nv_nrrd->data;
      for (k=0, kk=z0; kk<z1; kk++,k++) {
        for (j=0, jj=y0; jj<y1; jj++,j++) {
	  const size_t tex_idx = (k*ny*nx+j*nx)*nb;
	  const size_t data_idx = (kk*ni*nj+jj*ni+x0)*nb;
	  memcpy(tex + tex_idx, data + data_idx, (x1 - x0)*nb);
	  i = x1 - x0;
          if (nx != brick->mx()) {
            for (int b=0; b<nb; b++) {
              const size_t idx0 = (k*ny*nx+j*nx+i)*nb+b;
              const size_t idx1 = idx0-nb;
              tex[idx0] = tex[idx1];
            }
          }
        }
        if (ny != brick->my()) {
          for (i=0; i<Min(nx, brick->mx()+1); i++) {
            for (int b=0; b<nb; b++) {
              const size_t idx0 = (k*ny*nx+j*nx+i)*nb+b;
              const size_t idx1 = (k*ny*nx+(brick->my()-1)*nx+i)*nb+b;
              tex[idx0] = tex[idx1];
            }
          }
        }
      }
      if (nz != brick->mz()) {
        for (j=0; j<Min(ny, brick->my()+1); j++) {
          for (i=0; i<Min(nx, brick->mx()+1); i++) {
            for (int b=0; b<nb; b++) {
              const size_t idx0 = (k*ny*nx+j*nx+i)*nb+b;
              const size_t idx1 = ((brick->mz()-1)*ny*nx+j*nx+i)*nb+b;
              tex[idx0] = tex[idx1];
            }
          }
        }
      }
    } else {
      int nb0 = brick->nb(0);
      int nb1 = brick->nb(1);
      unsigned char* tex0 = br->data(0);
      unsigned char* tex1 = br->data(1);
      unsigned char* data0 = (unsigned char*)nv_nrrd->data;
      unsigned char* data1 = (unsigned char*)gm_nrrd->data;
      for (k=0, kk=z0; kk<z1; kk++, k++) {
        for (j=0, jj=y0; jj<y1; jj++,j++) {
	  const size_t tex_idx0 = (k*ny*nx+j*nx)*nb0;
	  const size_t data_idx0 = (kk*ni*nj+jj*ni+x0)*nb0;
	  memcpy(tex0 + tex_idx0, data0 + data_idx0, (x1 - x0)*nb0);
	  const size_t tex_idx1 = (k*ny*nx+j*nx)*nb1;
	  const size_t data_idx1 = (kk*ni*nj+jj*ni+x0)*nb1;
	  memcpy(tex1 + tex_idx1, data1 + data_idx1, (x1 - x0)*nb1);
	  i = x1 - x0;
          if (nx != brick->mx()) {
            for (int b=0; b<nb0; b++) {
              const size_t idx0 = (k*ny*nx+j*nx+i)*nb0+b;
              const size_t idx1 = idx0-nb0;
              tex0[idx0] = tex0[idx1];
            }
            for (int b=0; b<nb1; b++) {
              const size_t idx0 = (k*ny*nx+j*nx+i)*nb1+b;
              const size_t idx1 = idx0-nb1;
              tex1[idx0] = tex1[idx1];
            }
          }
        }
        if (ny != brick->my()) {
          for (i=0; i<Min(nx, brick->mx()+1); i++) {
            for (int b=0; b<nb0; b++) {
              const size_t idx0 = (k*ny*nx+j*nx+i)*nb0+b;
              const size_t idx1 = (k*ny*nx+(brick->my()-1)*nx+i)*nb0+b;
              tex0[idx0] = tex0[idx1];
            }
            for (int b=0; b<nb1; b++) {
              const size_t idx0 = (k*ny*nx+j*nx+i)*nb1+b;
              const size_t idx1 = (k*ny*nx+(brick->my()-1)*nx+i)*nb1+b;
              tex1[idx0] = tex1[idx1];
            }
          }
        }
      }
      if (nz != brick->mz()) {
        for (j=0; j<Min(ny, brick->my()+1); j++) {
          for (i=0; i<Min(nx, brick->mx()+1); i++) {
            for (int b=0; b<nb0; b++) {
              const size_t idx0 = (k*ny*nx+j*nx+i)*nb0+b;
              const size_t idx1 = ((brick->mz()-1)*ny*nx+j*nx+i)*nb0+b;
              tex0[idx0] = tex0[idx1];
            }
            for (int b=0; b<nb1; b++) {
              const size_t idx0 = (k*ny*nx+j*nx+i)*nb1+b;
              const size_t idx1 = ((brick->mz()-1)*ny*nx+j*nx+i)*nb1+b;
              tex1[idx0] = tex1[idx1];
            }
          }
        }
      }
    }
  }
}

} // namespace SCIRun
