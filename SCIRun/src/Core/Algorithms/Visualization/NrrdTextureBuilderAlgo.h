//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  
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
//    File   : NrrdTextureBuilderAlgo.h
//    Author : Milan Ikits
//    Date   : Fri Jul 16 02:48:14 2004

#ifndef Volume_NrrdTextureBuilderAlgo_h
#define Volume_NrrdTextureBuilderAlgo_h

#include <vector>
#include <Core/Datatypes/Datatype.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Volume/Texture.h>
#include <Core/Volume/TextureBrick.h>
#include <Core/Datatypes/NrrdData.h>
#include <Core/Geom/ShaderProgramARB.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Math/MiscMath.h>

#include <Core/Algorithms/Visualization/share.h>

namespace SCIRun {

// Currently located in TextureBuilderAlgo.cc
void texture_build_bricks(vector<TextureBrickHandle>& bricks,
                          int nx, int ny, int nz,
                          int nc, int* nb,
                          const BBox& bbox, int brick_mem,
                          bool use_nrrd_brick);


// Currently located in NrrdTextureBuilderAlgo.cc
SCISHARE void nrrd_build_bricks(vector<TextureBrickHandle>& bricks,
                                int nx, int ny, int nz,
                                int nc, int* nb, int card_mem);

class SCISHARE NrrdTextureBuilderAlgo : public SCIRun::DynamicAlgoBase
{
public:
  // The min/max values are currently ignored.  tHandle/vHandle should be
  // pre-scaled by this point.
  static void build_static(TextureHandle tHandle,
                           NrrdDataHandle vHandle, double vmin, double vmax,
                           NrrdDataHandle gHandle, double gmin, double gmax,
                           int card_mem);

  virtual void build(TextureHandle tHandle,
		     NrrdDataHandle vHandle, double vmin, double vmax,
		     NrrdDataHandle gHandle, double gmin, double gmax,
		     int card_mem, int is_uchar);

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const unsigned int vtype,
					    const unsigned int gtype);
protected:
  static void build_aux(TextureHandle tHandle,
                        NrrdDataHandle vHandle, double vmin, double vmax,
                        NrrdDataHandle gHandle, double gmin, double gmax,
                        int card_mem, int is_uchar);

  virtual void fill_brick(TextureBrickHandle &tHandle,
			  NrrdDataHandle vHandle, double vmin, double vmax,
			  NrrdDataHandle gHandle, double gmin, double gmax,
			  int ni, int nj, int nk) = 0;
};


template < class VTYPE, class GTYPE >
class NrrdTextureBuilderAlgoT : public NrrdTextureBuilderAlgo
{
protected:
  virtual void fill_brick(TextureBrickHandle &tHandle,
			  NrrdDataHandle vHandle, double vmin, double vmax,
			  NrrdDataHandle gHandle, double gmin, double gmax,
			  int ni, int nj, int nk);
}; 


template< class VTYPE, class GTYPE >
void
NrrdTextureBuilderAlgoT<VTYPE, GTYPE>::fill_brick(TextureBrickHandle &brick,
                                                  NrrdDataHandle vHandle,
                                                  double vmin, double vmax,
                                                  NrrdDataHandle gHandle,
                                                  double gmin, double gmax,
                                                  int ni, int nj, int /*nk*/)
{
  if (brick->get_type_description(TextureBrick::TB_NAME_ONLY_E)->get_name() == 
      "NrrdTextureBrick") 
  {
    NrrdTextureBrick *nbrick = (NrrdTextureBrick *) brick.get_rep();
    nbrick->set_nrrds(vHandle, gHandle);
    return;
  }
  
  Nrrd *nv_nrrd = vHandle->nrrd_;
  Nrrd *gm_nrrd = (gHandle.get_rep() ? gHandle->nrrd_ : 0);
  
  TextureBrickT<unsigned char>* br =
    (TextureBrickT<unsigned char>*) brick.get_rep();

  // Direct memory copies can be done if the data is unisgned chars
  // with no rescaling.
  bool use_mem_copy = (vHandle->nrrd_->type == nrrdTypeUChar &&
		       vmin == 0 && vmax == 255 &&
		       gHandle->nrrd_->type == nrrdTypeUChar &&
		       gmin == 0 && gmax == 255);

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
  if (nc == 1 || (gm_nrrd && nc == 2) ) {
    if (!gm_nrrd) { // fill only values
      int nb = brick->nb(0);
      int boff = nb - 1;
      unsigned char* tex = br->data(0);
      VTYPE* data = (VTYPE*)nv_nrrd->data;
      VTYPE* iter = data;
      for (k=0, kk=z0; kk<z1; kk++,k++) {
        for (j=0, jj=y0; jj<y1; jj++,j++) {

	  if( use_mem_copy ) {
	    const size_t tex_idx = (k*ny*nx+j*nx)*nb;
	    const size_t data_idx = (kk*ni*nj+jj*ni+x0)*nb;
	    memcpy(tex + tex_idx, data + data_idx, (x1 - x0)*nb);

	  } else {

	    for(i=0, ii=x0; ii<x1; ii++, i++) {

	      double v = *iter;

	      for (int b=0; b<boff; b++) {
		tex[(k*ny*nx+j*nx+i)*nb+b] = (unsigned char) v;
		++iter; v = *iter;
 	      }

	      tex[(k*ny*nx+j*nx+i)*nb+boff] =
		(unsigned char)(Clamp((v - vmin)/(vmax-vmin), 0.0, 1.0)*255.0);
	      ++iter;
	    }
	  }

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
    } else { // fill values + gradient
      int nb0 = brick->nb(0);
      int nb1 = brick->nb(1);
      int boff0 = nb0 - 1;
      int boff1 = nb1 - 1;
      unsigned char* tex0 = br->data(0);
      unsigned char* tex1 = br->data(1);
      VTYPE* data0 = (VTYPE*)nv_nrrd->data;
      GTYPE* data1 = (GTYPE*)gm_nrrd->data;
      VTYPE* iter0 = data0;
      GTYPE* iter1 = data1;

      for (k=0, kk=z0; kk<z1; kk++, k++) {
        for (j=0, jj=y0; jj<y1; jj++,j++) {

	  if( use_mem_copy ) {
	    const size_t tex_idx0 = (k*ny*nx+j*nx)*nb0;
	    const size_t data_idx0 = (kk*ni*nj+jj*ni+x0)*nb0;
	    memcpy(tex0 + tex_idx0, data0 + data_idx0, (x1 - x0)*nb0);
	    const size_t tex_idx1 = (k*ny*nx+j*nx)*nb1;
	    const size_t data_idx1 = (kk*ni*nj+jj*ni+x0)*nb1;
	    memcpy(tex1 + tex_idx1, data1 + data_idx1, (x1 - x0)*nb1);

	  } else {
	    for(i=0, ii=x0; ii<x1; ii++, i++) {
	      double v = *iter0;

	      for (int b=0; b<boff0; b++) {
		tex0[(k*ny*nx+j*nx+i)*nb0+b] = (unsigned char) v;
		++iter0; v = *iter0;
	      }

	      tex0[(k*ny*nx+j*nx+i)*nb0+boff0] =
		(unsigned char)(Clamp((v - vmin)/(vmax-vmin), 0.0, 1.0)*255.0);
	      ++iter0;

	      
	      double g = *iter1;

// 	      for (int b=0; b<boff1; b++) {
// 		tex1[(k*ny*nx+j*nx+i)*nb1+b] =
// 		  (unsigned char)(Clamp((g - gmin)/(gmax-gmin), 0.0, 1.0)*255.0);
// 		++iter1; g = *iter1;
// 	      }

 	      tex1[(k*ny*nx+j*nx+i)*nb1+boff1] =
 		(unsigned char)(Clamp((g - gmin)/(gmax-gmin), 0.0, 1.0)*255.0);
 	      ++iter1;
	    }
	  }

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

#endif // Volume_NrrdTextureBuilderAlgo_h
