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

#include <Packages/Volume/Core/Algorithms/NrrdTextureBuilderAlgo.h>
#include <Core/Util/DebugStream.h>

#include <iostream>

namespace Volume {

using namespace SCIRun;
using namespace std;

static DebugStream dbg("NrrdTextureBuilderAlgo", false);

NrrdTextureBuilderAlgo::NrrdTextureBuilderAlgo()
{}

NrrdTextureBuilderAlgo::~NrrdTextureBuilderAlgo() 
{}

void
NrrdTextureBuilderAlgo::build(TextureHandle texture,
                              NrrdDataHandle nvn, NrrdDataHandle gmn,
                              int card_mem)
{
  Nrrd* nv_nrrd = nvn->nrrd;
  if (nv_nrrd->dim != 3 && nv_nrrd->dim != 4) {
    cerr << "Invalid dimension for input value nrrd" << endl;
    return;
  }

  Nrrd* gm_nrrd = 0;
  int axis_size[4];
  int nc;
  if(gmn.get_rep()) {
    gm_nrrd = gmn->nrrd;
    if(gm_nrrd) {
      if (gm_nrrd->dim != 3 && gm_nrrd->dim != 4) {
        cerr << "Invalid dimension for input gradient magnitude nrrd" << endl;
        return;
      }
      nrrdAxisInfoGet_nva(gm_nrrd, nrrdAxisInfoSize, axis_size);
      if(gm_nrrd->dim > 3 && axis_size[0] != 1) {
        cerr << "Invalid size for gradient magnitude nrrd" << endl;
        return;
      }
      nc = 2;
    } else {
      nc = 1;
    }
  } else {
    nc = 1;
  }
  nrrdAxisInfoGet_nva(nv_nrrd, nrrdAxisInfoSize, axis_size);
  double axis_min[4];
  nrrdAxisInfoGet_nva(nv_nrrd, nrrdAxisInfoMin, axis_min);
  double axis_max[4];
  nrrdAxisInfoGet_nva(nv_nrrd, nrrdAxisInfoMax, axis_max);

  int nx = axis_size[nv_nrrd->dim-3];
  int ny = axis_size[nv_nrrd->dim-2];
  int nz = axis_size[nv_nrrd->dim-1];
  int nb[2];
  nb[0] = nv_nrrd->dim > 3 ? axis_size[0] : 1;
  nb[1] = gm_nrrd ? 1 : 0;

  BBox bbox;
  bbox.extend(Point(axis_min[nv_nrrd->dim-3],
                    axis_min[nv_nrrd->dim-2],
                    axis_min[nv_nrrd->dim-1]));
  bbox.extend(Point(axis_max[nv_nrrd->dim-3],
                    axis_max[nv_nrrd->dim-2],
                    axis_max[nv_nrrd->dim-1]));

  //
  texture->lock_bricks();
  vector<Brick*>& bricks = texture->bricks();
  if(nx != texture->nx() || ny != texture->ny() || nz != texture->nz()
     || nc != texture->nc() || card_mem != texture->card_mem()) {
    build_bricks(bricks, nx, ny, nz, nc, nb, bbox, card_mem);
    texture->set_size(nx, ny, nz, nc, nb);
    texture->set_card_mem(card_mem);
  }
  texture->set_bbox(bbox);
  texture->set_minmax(0.0, 255.0, 0.0, 255.0);
  for(unsigned int i=0; i<bricks.size(); i++) {
    fill_brick(bricks[i], nv_nrrd, gm_nrrd, nx, ny, nz);
    bricks[i]->set_dirty(true);
  }
  texture->unlock_bricks();
}

void
NrrdTextureBuilderAlgo::build_bricks(vector<Brick*>& bricks, int nx, int ny,
                                     int nz, int nc, int* nb,
                                     const BBox& bbox, int card_mem)
{
  int brick_mem = card_mem*1024*1024/2;
  int data_size[3] = { nx, ny, nz };
  int brick_size[3];
  int brick_offset[3];
  int brick_pad[3];
  int num_brick[3];
  int brick_nb;
  // figure out largest possible brick size
  int size[3];
  size[0] = data_size[0]; size[1] = data_size[1]; size[2] = data_size[2];
  // initial brick size
  brick_size[0] = NextPowerOf2(size[0]);
  brick_size[1] = NextPowerOf2(size[1]);
  brick_size[2] = NextPowerOf2(size[2]);
  // number of bytes per brick
  brick_nb = brick_size[0]*brick_size[1]*brick_size[2]*nb[0];
  // find brick size
  if(brick_nb > brick_mem)
  {
    // subdivide until fits
    while(brick_nb > brick_mem)
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
  for(unsigned int i=0; i<bricks.size(); i++) delete bricks[i];
  bricks.resize(0);
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
        Brick* b = scinew BrickT<unsigned char>(
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
               Point(tmax[0], tmax[1], tmax[2])),
          true);
        bricks.push_back(b);

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
NrrdTextureBuilderAlgo::fill_brick(Brick* brick, Nrrd* nv_nrrd, Nrrd* gm_nrrd,
                                   int ni, int nj, int nk)
{
  BrickT<unsigned char>* br =
    dynamic_cast<BrickT<unsigned char>*>(brick);
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
  int i, j, k, ii, jj, kk;
  if(nv_nrrd && ((gm_nrrd && nc == 2) || nc == 1)) {
    if(!gm_nrrd) {
      int nb = brick->nb(0);
      unsigned char* tex = br->data(0);
      unsigned char* data = (unsigned char*)nv_nrrd->data;
      for(k=0, kk=z0; kk<z1; kk++,k++) {
        for(j=0, jj=y0; jj<y1; jj++,j++) {
          for(i=0, ii=x0; ii<x1; ii++,i++) {
            for(int b=0; b<nb; b++) {
              int tex_idx = (k*ny*nx+j*nx+i)*nb+b;
              int data_idx = (kk*ni*nj+jj*ni+ii)*nb+b;
              tex[tex_idx] = data[data_idx];
            }
          }
          if(nx != brick->mx()) {
            for(int b=0; b<nb; b++) {
              int tex_idx = (k*ny*nx+j*nx+i)*nb+b;
              tex[tex_idx] = 0;
            }
          }
        }
        if(ny != brick->my()) {
          for(i=0; i<Min(nx, brick->mx()+1); i++) {
            for(int b=0; b<nb; b++) {
              int tex_idx = (k*ny*nx+j*nx+i)*nb+b;
              tex[tex_idx] = 0;
            }
          }
        }
      }
      if(nz != brick->mz()) {
        for(j=0; j<Min(ny, brick->my()+1); j++) {
          for(i=0; i<Min(nx, brick->mx()+1); i++) {
            for(int b=0; b<nb; b++) {
              int tex_idx = (k*ny*nx+j*nx+i)*nb+b;
              tex[tex_idx] = 0;
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
      for(k=0, kk=z0; kk<z1; kk++, k++) {
        for(j=0, jj=y0; jj<y1; jj++,j++) {
          for(i=0, ii=x0; ii<x1; ii++, i++) {
            for(int b=0; b<nb0; b++) {
              int tex_idx = (k*ny*nx+j*nx+i)*nb0+b;
              int data_idx = (kk*ni*nj+jj*ni+ii)*nb0+b;
              tex0[tex_idx] = data0[data_idx];
            }
            for(int b=0; b<nb1; b++) {
              int tex_idx = (k*ny*nx+j*nx+i)*nb1+b;
              int data_idx = (kk*ni*nj+jj*ni+ii)*nb1+b;
              tex1[tex_idx] = data1[data_idx];
            }
          }
          if(nx != brick->mx()) {
            for(int b=0; b<nb0; b++) {
              int tex_idx = (k*ny*nx+j*nx+i)*nb0+b;
              tex0[tex_idx] = 0;
            }
            for(int b=0; b<nb1; b++) {
              int tex_idx = (k*ny*nx+j*nx+i)*nb1+b;
              tex1[tex_idx] = 0;
            }
          }
        }
        if(ny != brick->my()) {
          for(i=0; i<Min(nx, brick->mx()+1); i++) {
            for(int b=0; b<nb0; b++) {
              int tex_idx = (k*ny*nx+j*nx+i)*nb0+b;
              tex0[tex_idx] = 0;
            }
            for(int b=0; b<nb1; b++) {
              int tex_idx = (k*ny*nx+j*nx+i)*nb1+b;
              tex1[tex_idx] = 0;
            }
          }
        }
      }
      if(nz != brick->mz()) {
        for(j=0; j<Min(ny, brick->my()+1); j++) {
          for(i=0; i<Min(nx, brick->mx()+1); i++) {
            for(int b=0; b<nb0; b++) {
              int tex_idx = (k*ny*nx+j*nx+i)*nb0+b;
              tex0[tex_idx] = 0;
            }
            for(int b=0; b<nb1; b++) {
              int tex_idx = (k*ny*nx+j*nx+i)*nb1+b;
              tex1[tex_idx] = 0;
            }
          }
        }
      }
    }
  }
}

} // namespace Volume
