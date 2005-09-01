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
//    File   : TextureBuilderAlgo.h
//    Author : Milan Ikits
//    Date   : Thu Jul 15 00:47:30 2004

#ifndef Volume_TextureBuilderAlgo_h
#define Volume_TextureBuilderAlgo_h

#include <Core/Datatypes/Datatype.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Volume/Texture.h>
#include <Core/Datatypes/MRLatVolField.h>
#include <Core/Volume/TextureBrick.h>

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

using std::cerr;
using std::endl;

using std::string;
using std::vector;


namespace SCIRun {

// Currently located in TextureBuilderAlgo.cc
void texture_build_bricks(vector<TextureBrickHandle>& bricks,
                          int nx, int ny, int nz,
                          int nc, int* nb,
                          const BBox& bbox, int brick_mem,
                          bool use_nrrd_brick);


class TextureBuilderAlgo : public SCIRun::DynamicAlgoBase
{
public:
  virtual void build(TextureHandle texture,
                     FieldHandle vHandle, double vmin, double vmax,
                     FieldHandle gHandle, double gmin, double gmax,
                     int card_mem) = 0;

protected:
  virtual void fill_brick(TextureBrickHandle &brick,
                          FieldHandle vHandle, double vmin, double vmax,
                          FieldHandle gHandle, double gmin, double gmax) = 0;

public:
  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription* vftd,
					    const TypeDescription* gftd);
};


template < class VFIELD, class GFIELD, class TEXTUREBRICK >
class TextureBuilderAlgoT : public TextureBuilderAlgo
{
public:
  typedef typename VFIELD::value_type value_type;

  virtual void build(TextureHandle tHandle,
                     FieldHandle vHandle, double vmin, double vmax,
                     FieldHandle gHandle, double gmin, double gmax,
                     int card_mem);
  
protected:
  virtual void fill_brick(TextureBrickHandle &brick,
                          FieldHandle vHandle, double vmin, double vmax,
                          FieldHandle gHandle, double gmin, double gmax);
}; 

template< class VFIELD, class GFIELD, class TEXTUREBRICK >
void
TextureBuilderAlgoT<VFIELD,
		    GFIELD,
		    TEXTUREBRICK>::build(TextureHandle tHandle,
					 FieldHandle vHandle,
					 double vmin, double vmax,
					 FieldHandle gHandle,
					 double gmin, double gmax,
					 int card_mem)
{
  VFIELD *vfld = (VFIELD *) vHandle.get_rep();
  GFIELD *gfld = (GFIELD *) gHandle.get_rep();


  MRLatVolField<value_type> *mrvfld = 0;
  MRLatVolField<Vector>     *mrgfld = 0;

  if( vHandle->get_type_description(0)->get_name() == "MRLatVolField" )
    mrvfld = (MRLatVolField<value_type>*) vfld;

  if( gfld && gHandle->get_type_description(0)->get_name() == "MRLatVolField" )
    mrgfld = (MRLatVolField<Vector>*) gfld;

  if( mrvfld ) {
    // temporary
    int nc = 1;
    int nb[2] = { 1, 0 };

    if( mrgfld ) {
      // In order to use the gradient field, it must have the exact
      // structure as the value field.
      bool same = true;
      // Same number of levels?
      if( mrvfld->nlevels() == mrgfld->nlevels() ) {
	for(int i = 0; i < mrvfld->nlevels(); i++){
	  const MultiResLevel<value_type>*  lev = mrvfld->level(i);
	  const MultiResLevel<Vector>    * glev = mrgfld->level(i);
	  // Does each level have the same number of patches?
	  if( lev->patches.size() == glev->patches.size() ){
	    for(unsigned int j = 0; j < lev->patches.size(); j++ ){
	      LatVolField<value_type>* vmr =  lev->patches[j].get_rep(); 
	      LatVolField<Vector>*     gmr = glev->patches[j].get_rep();
	      
	      typename LatVolField<value_type>::mesh_handle_type mesh =
		vmr->get_typed_mesh();
	      typename LatVolField<Vector>::mesh_handle_type gmesh =
		gmr->get_typed_mesh();
	      
	      // Is each patch the same size?
	      if( mesh->get_ni() != gmesh->get_ni() ||
		  mesh->get_nj() != gmesh->get_nj() ||
		  mesh->get_nk() != gmesh->get_nk()) {
		same = false;
		break;
	      }
	    }
	    if (!same) {
	      break;
	    }
	  } else {
	    same = false;
	    break;
	  }
	}
      } else {
	same = false;
      }
      // If same is still true, we can use the gradient field
      if( same ) {
	nc = 2;
	nb[0] = 4;
	nb[1] = 1;
      }
    }

    
    if( tHandle->nlevels() > 1 ){
      tHandle->lock_bricks();
      tHandle->clear();
      tHandle->unlock_bricks();
    }
    
    // Grab the transform from the lowest resolution field
    Transform tform;
    mrvfld->level(0)->patches[0].get_rep()->get_typed_mesh()->
      get_canonical_transform(tform);
    
    for(int i = 0 ; i < mrvfld->nlevels(); i++ ){
      const MultiResLevel<value_type> *lev =  mrvfld->level(i);
      const MultiResLevel<Vector>    *glev = (mrgfld ? mrgfld->level(i) : 0);
      vector<TextureBrickHandle> new_level;
      if( i == tHandle->nlevels() ) {
	tHandle->add_level(new_level);
      }
      vector<TextureBrickHandle>& bricks = tHandle->bricks(i);
      unsigned int k = 0;
      for(unsigned int j = 0; j < lev->patches.size(); j++ ){
	LatVolField<value_type>* vmr =  lev->patches[j].get_rep(); 
	LatVolField<Vector>*     gmr = (glev ? glev->patches[j].get_rep() : 0);

	typename LatVolField<value_type>::mesh_handle_type mesh = vmr->get_typed_mesh();

	int nx = mesh->get_ni();
	int ny = mesh->get_nj();
	int nz = mesh->get_nk();
	 if(vHandle->basis_order() == 0) {
	  --nx; --ny; --nz;
	}
	
	// make sure each sub level has a corrected bounding box
	BBox bbox(tform.unproject( mesh->get_bounding_box().min() ),
		  tform.unproject( mesh->get_bounding_box().max() ));

       	vector<TextureBrickHandle> patch_bricks;
  	texture_build_bricks(patch_bricks, nx, ny, nz, nc, nb, bbox,
                             card_mem, false);
	
	if( i == 0 ){
	  tHandle->set_size(nx, ny, nz, nc, nb);
	  tHandle->set_card_mem(card_mem);
	  tHandle->set_bbox(bbox);
	  tHandle->set_minmax(vmin, vmax, gmin, gmax);
	  tHandle->set_transform(tform);
	}

	tHandle->lock_bricks();
	for(k = 0; k < patch_bricks.size(); k++){
	  fill_brick(patch_bricks[k], vmr, vmin, vmax, gmr, gmin, gmax);
	  patch_bricks[k]->set_dirty(true);
	  bricks.push_back( patch_bricks[k] );
	}

	tHandle->unlock_bricks();
      }
    }
  } else {
    
    typename VFIELD::mesh_handle_type mesh = vfld->get_typed_mesh();
    int nx = mesh->get_ni();
    int ny = mesh->get_nj();
    int nz = mesh->get_nk();
    if(vHandle->basis_order() == 0) {
      --nx; --ny; --nz;
    }
    int nc = gHandle.get_rep() ? 2 : 1;
    int nb[2];
    nb[0] = gHandle.get_rep() ? 4 : 1;
    nb[1] = gHandle.get_rep() ? 1 : 0;
    Transform tform;
    mesh->get_canonical_transform(tform);

    tHandle->lock_bricks();
    tHandle->clear();
    vector<TextureBrickHandle>& bricks = tHandle->bricks();
    const BBox bbox(Point(0,0,0), Point(1,1,1)); 
    if(nx != tHandle->nx() || ny != tHandle->ny() || nz != tHandle->nz()
       || nc != tHandle->nc() || card_mem != tHandle->card_mem() ||
       bbox.min() != tHandle->bbox().min())
    {
      texture_build_bricks(bricks, nx, ny, nz, nc, nb, bbox, card_mem, false);
      tHandle->set_size(nx, ny, nz, nc, nb);
      tHandle->set_card_mem(card_mem);
    }
    tHandle->set_bbox(bbox);
    tHandle->set_minmax(vmin, vmax, gmin, gmax);
    tHandle->set_transform(tform);
    for(unsigned int i=0; i<bricks.size(); i++) {
      fill_brick(bricks[i], vHandle, vmin, vmax, gHandle, gmin, gmax);
      bricks[i]->set_dirty(true);
    }
    tHandle->unlock_bricks();
  }
}


template <class VFIELD, class GFIELD, class TEXTUREBRICK>
void 
TextureBuilderAlgoT<VFIELD,
		    GFIELD,
		    TEXTUREBRICK>::fill_brick(TextureBrickHandle &brick,
					      FieldHandle vHandle,
					      double vmin, double vmax,
					      FieldHandle gHandle,
					      double gmin, double gmax)
{
  VFIELD *vfld = (VFIELD *) vHandle.get_rep();
  GFIELD *gfld = (GFIELD *) gHandle.get_rep();
  TEXTUREBRICK *br = (TEXTUREBRICK*) brick.get_rep();

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
    
  if (nc == 1 || (nc == 2 && gfld)) {
    typename VFIELD::mesh_type* mesh = vfld->get_typed_mesh().get_rep();

    if (!gfld) { // fill only values

      unsigned char* tex = br->data(0);
      if(vHandle->basis_order() == 0) {
        typename VFIELD::mesh_type::Cell::range_iter iter(mesh,
							  x0, y0, z0,
							  x1, y1, z1);
        for(k=0, kk=z0; kk<z1; kk++, k++) {
          for(j=0, jj=y0; jj<y1; jj++, j++) {
            for(i=0, ii=x0; ii<x1; ii++, i++) {
              double v = vfld->fdata()[*iter];
              tex[k*ny*nx+j*nx+i] =
                (unsigned char)(Clamp((v - vmin)/(vmax-vmin), 0.0, 1.0)*255.0);
              ++iter;
            }
            if(nx != brick->mx()) {
              tex[k*ny*nx+j*nx+i] = tex[k*ny*nx+j*nx+(brick->mx()-1)];
            }
          }
          if(ny != brick->my()) {
            for(i=0; i<Min(nx, brick->mx()+1); i++) {
              tex[k*ny*nx+j*nx+i] = tex[k*ny*nx+(brick->my()-1)*nx+i];
            }
          }
        }
        if(nz != brick->mz()) {
          for(j=0; j<Min(ny, brick->my()+1); j++) {
            for(i=0; i<Min(nx, brick->mx()+1); i++) {
              tex[k*ny*nx+j*nx+i] = tex[(brick->mz()-1)*ny*nx+j*nx+i];
            }
          }
        }
      } else {
        typename VFIELD::mesh_type::Node::range_iter iter(mesh,
							  x0, y0, z0,
							  x1, y1, z1);
        for(k=0, kk=z0; kk<z1; kk++, k++) {
          for(j=0, jj=y0; jj<y1; jj++, j++) {
            for(i=0, ii=x0; ii<x1; ii++, i++) {
              double v = vfld->fdata()[*iter];
              tex[k*ny*nx+j*nx+i] =
                (unsigned char)(Clamp((v - vmin)/(vmax-vmin), 0.0, 1.0)*255.0);
              ++iter;
            }
            if(nx != brick->mx()) {
              tex[k*ny*nx+j*nx+i] = tex[k*ny*nx+j*nx+(brick->mx()-1)];
            }
          }
          if(ny != brick->my()) {
            for(i=0; i<Min(nx, brick->mx()+1); i++) {
              tex[k*ny*nx+j*nx+i] = tex[k*ny*nx+(brick->my()-1)*nx+i];
            }
          }
        }
        if(nz != brick->mz()) {
          for(j=0; j<Min(ny, brick->my()+1); j++) {
            for(i=0; i<Min(nx, brick->mx()+1); i++) {
              tex[k*ny*nx+j*nx+i] = tex[(brick->mz()-1)*ny*nx+j*nx+i];
            }
          }
        }
      }
    } else { // fill values + gradient
      unsigned char* tex0 = br->data(0);
      unsigned char* tex1 = br->data(1);
      
      if(vHandle->basis_order() == 0) {
        typename VFIELD::mesh_type::Cell::range_iter iter(mesh,
							  x0, y0, z0,
							  x1, y1, z1);
        for(k=0, kk=z0; kk<z1; kk++, k++) {
          for(j=0, jj=y0; jj<y1; jj++, j++) {
            for(i=0, ii=x0; ii<x1; ii++, i++) {
              double v = vfld->fdata()[*iter];
              int idx = k*ny*nx+j*nx+i;
              tex0[idx*4+3] =
                (unsigned char)(Clamp((v - vmin)/(vmax-vmin), 0.0, 1.0)*255.0);
              Vector g = gfld->fdata()[*iter];
              const double gn = g.safe_normalize();
              tex0[idx*4+0] = (unsigned char)((g.x()*0.5 + 0.5)*255);
              tex0[idx*4+1] = (unsigned char)((g.y()*0.5 + 0.5)*255);
              tex0[idx*4+2] = (unsigned char)((g.z()*0.5 + 0.5)*255);
              tex1[idx] = (unsigned char)(((gn-gmin)/(gmax-gmin))*255);
              ++iter;
            }
            if(nx != brick->mx()) {
              const int idx = k*ny*nx+j*nx+i;
              const int idx1 = k*ny*nx+j*nx+(brick->mx()-1);
              tex0[idx*4+0] = tex0[idx1*4+0];
              tex0[idx*4+1] = tex0[idx1*4+1];
              tex0[idx*4+2] = tex0[idx1*4+2];
              tex0[idx*4+3] = tex0[idx1*4+3];
              tex1[idx] = tex1[idx1];
            }
          }
          if(ny != brick->my()) {
            for(i=0; i<Min(nx, brick->mx()+1); i++) {
              const int idx = k*ny*nx+j*nx+i;
              const int idx1 = k*ny*nx+(brick->my()-1)*nx+i;
              tex0[idx*4+0] = tex0[idx1*4+0];
              tex0[idx*4+1] = tex0[idx1*4+1];
              tex0[idx*4+2] = tex0[idx1*4+2];
              tex0[idx*4+3] = tex0[idx1*4+3];
              tex1[idx] = tex1[idx1];
            }
          }
        }
        if(nz != brick->mz()) {
          for(j=0; j<Min(ny, brick->my()+1); j++) {
            for(i=0; i<Min(nx, brick->mx()+1); i++) {
              const int idx = k*ny*nx+j*nx+i;
              const int idx1 = (brick->mz()-1)*ny*nx+j*nx+i;
              tex0[idx*4+0] = tex0[idx1*4+0];
              tex0[idx*4+1] = tex0[idx1*4+1];
              tex0[idx*4+2] = tex0[idx1*4+2];
              tex0[idx*4+3] = tex0[idx1*4+3];
              tex1[idx] = tex1[idx1];
            }
          }
        }
      } else {
        typename VFIELD::mesh_type::Node::range_iter iter(mesh,
							  x0, y0, z0,
							  x1, y1, z1);
        for(k=0, kk=z0; kk<z1; kk++, k++) {
          for(j=0, jj=y0; jj<y1; jj++, j++) {
            for(i=0, ii=x0; ii<x1; ii++, i++) {
              double v = vfld->fdata()[*iter];
              const int idx = k*ny*nx+j*nx+i;
              tex0[idx*4+3] =
                (unsigned char)(Clamp((v - vmin)/(vmax-vmin), 0.0, 1.0)*255.0);
              Vector g = gfld->fdata()[*iter];
              const double gn = g.safe_normalize();
              tex0[idx*4+0] = (unsigned char)((g.x()*0.5 + 0.5)*255);
              tex0[idx*4+1] = (unsigned char)((g.y()*0.5 + 0.5)*255);
              tex0[idx*4+2] = (unsigned char)((g.z()*0.5 + 0.5)*255);
              tex1[idx] = (unsigned char)(((gn-gmin)/(gmax-gmin))*255);
              ++iter;
            }
            if(nx != brick->mx()) {
              const int idx = k*ny*nx+j*nx+i;
              const int idx1 = k*ny*nx+j*nx+(brick->mx()-1);
              tex0[idx*4+0] = tex0[idx1*4+0];
              tex0[idx*4+1] = tex0[idx1*4+1];
              tex0[idx*4+2] = tex0[idx1*4+2];
              tex0[idx*4+3] = tex0[idx1*4+3];
              tex1[idx] = tex1[idx1];
            }
          }
          if(ny != brick->my()) {
            for(i=0; i<Min(nx, brick->mx()+1); i++) {
              const int idx = k*ny*nx+j*nx+i;
              const int idx1 = k*ny*nx+(brick->my()-1)*nx+i;
              tex0[idx*4+0] = tex0[idx1*4+0];
              tex0[idx*4+1] = tex0[idx1*4+1];
              tex0[idx*4+2] = tex0[idx1*4+2];
              tex0[idx*4+3] = tex0[idx1*4+3];
              tex1[idx] = tex1[idx1];
            }
          }
        }
        if(nz != brick->mz()) {
          for(j=0; j<Min(ny, brick->my()+1); j++) {
            for(i=0; i<Min(nx, brick->mx()+1); i++) {
              const int idx = k*ny*nx+j*nx+i;
              const int idx1 = (brick->mz()-1)*ny*nx+j*nx+i;
              tex0[idx*4+0] = tex0[idx1*4+0];
              tex0[idx*4+1] = tex0[idx1*4+1];
              tex0[idx*4+2] = tex0[idx1*4+2];
              tex0[idx*4+3] = tex0[idx1*4+3];
              tex1[idx] = tex1[idx1];
            }
          }
        }
      }
    }    
  }
}


} // namespace SCIRun

#endif // Volume_TextureBuilderAlgo_h
