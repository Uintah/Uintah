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

namespace SCIRun {

// Currently located in TextureBuilderAlgo.cc
void texture_build_bricks(vector<TextureBrickHandle>& bricks,
                          int nx, int ny, int nz,
                          int nc, int* nb,
                          const BBox& bbox, int brick_mem,
                          bool use_nrrd_brick);


// Currently located in NrrdTextureBuilderAlgo.cc
void nrrd_build_bricks(vector<TextureBrickHandle>& bricks,
		       int nx, int ny, int nz,
		       int nc, int* nb,
		       const BBox& ignored, int card_mem);


class NrrdTextureBuilderAlgo : public SCIRun::DynamicAlgoBase
{
public:
  virtual void build(TextureHandle tHandle,
		     NrrdDataHandle vHandle, double vmin, double vmax,
		     NrrdDataHandle gHandle, double gmin, double gmax,
		     int card_mem, int is_uchar) = 0;

protected:
  virtual void fill_brick(TextureBrickHandle &tHandle,
			  NrrdDataHandle vHandle, double vmin, double vmax,
			  NrrdDataHandle gHandle, double gmin, double gmax,
			  int ni, int nj, int nk) = 0;
public:
  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const unsigned int vtype,
					    const unsigned int gtype);
};


template < class VTYPE, class GTYPE >
class NrrdTextureBuilderAlgoT : public NrrdTextureBuilderAlgo
{
public:
  virtual void build(TextureHandle tHandle,
		     NrrdDataHandle vHandle, double vmin, double vmax,
		     NrrdDataHandle gHandle, double gmin, double gmax,
		     int card_mem,
		     int is_uchar);

protected:
  virtual void fill_brick(TextureBrickHandle &tHandle,
			  NrrdDataHandle vHandle, double vmin, double vmax,
			  NrrdDataHandle gHandle, double gmin, double gmax,
			  int ni, int nj, int nk);
}; 


template< class VTYPE, class GTYPE >
void
NrrdTextureBuilderAlgoT<VTYPE,
			GTYPE>::build(TextureHandle tHandle,
				      NrrdDataHandle vHandle,
				      double vmin, double vmax,
				      NrrdDataHandle gHandle,
				      double gmin, double gmax,
				      int card_mem,
				      int is_uchar)
{
  Nrrd* nv_nrrd = vHandle->nrrd;
  Nrrd* gm_nrrd = (gHandle.get_rep() ? gHandle->nrrd : 0);

  int axis_size[4];
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
      is_uchar || (vHandle->nrrd->type == nrrdTypeUChar &&
		   vmin == 0 && vmax == 255 &&
		   gHandle->nrrd->type == nrrdTypeUChar &&
		   gmin == 0 && gmax == 255);

    if ( use_nrrd_brick &&
	ShaderProgramARB::shaders_supported() &&
	ShaderProgramARB::texture_non_power_of_two())
    {
      cerr << "Building NrrdTextureBricks\n";

      nrrd_build_bricks(bricks, nx, ny, nz, nc, nb, bbox, card_mem);
    }
    else
    {
      if( vHandle->nrrd->type == nrrdTypeUChar &&
	  vmin == 0 && vmax == 255 &&
	  gHandle->nrrd->type == nrrdTypeUChar &&
	  gmin == 0 && gmax == 255 )
	cerr << "Asking for NrrdTextureBricks\n";

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


template< class VTYPE, class GTYPE >
void
NrrdTextureBuilderAlgoT<VTYPE,
			GTYPE>::fill_brick(TextureBrickHandle &brick,
					   NrrdDataHandle vHandle,
					   double vmin, double vmax,
					   NrrdDataHandle gHandle,
					   double gmin, double gmax,
					   int ni, int nj, int /*nk*/)
{
  cerr << "Getting brick type name " << brick->type_name() << endl;
  cerr << "Getting brick type desc " << brick->get_type_description(0)->get_name() << endl;

  if (brick->get_type_description(0)->get_name() == "NrrdTextureBrick") {

    cerr << "Filling NrrdTextureBricks\n";
    
    NrrdTextureBrick *nbrick = (NrrdTextureBrick *) brick.get_rep();
    nbrick->set_nrrds(vHandle, gHandle);
    return;
  }
  
  Nrrd *nv_nrrd = vHandle->nrrd;
  Nrrd *gm_nrrd = (gHandle.get_rep() ? gHandle->nrrd : 0);
  
  cerr << "Filling TextureBrickT<unsigned char>\n";

  TextureBrickT<unsigned char>* br =
    (TextureBrickT<unsigned char>*) brick.get_rep();

  // Direct memory copies can be done if the data is unisgned chars
  // with no rescaling.
  bool use_mem_copy = (vHandle->nrrd->type == nrrdTypeUChar &&
		       vmin == 0 && vmax == 255 &&
		       gHandle->nrrd->type == nrrdTypeUChar &&
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
