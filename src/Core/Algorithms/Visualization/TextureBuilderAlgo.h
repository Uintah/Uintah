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

void break_in_me();

// Currently located in NrrdTextureBuilderAlgo.cc
void texture_build_bricks(vector<TextureBrickHandle>& bricks,
                          int nx, int ny, int nz,
                          int nc, int* nb,
                          const BBox& bbox, int brick_mem,
                          bool use_nrrd_brick);

class TextureBuilderAlgoBase : public SCIRun::DynamicAlgoBase
{
public:
  TextureBuilderAlgoBase();
  virtual ~TextureBuilderAlgoBase();
  virtual void build(TextureHandle texture,
                     FieldHandle vfield, double vmin, double vmax,
                     FieldHandle gfield, double gmin, double gmax,
                     int card_mem) = 0;
  //! support the dynamically compiled algorithm concept
  static const string& get_h_file_path();
  static CompileInfoHandle get_compile_info(const TypeDescription* td);

protected:
  virtual void fill_brick(TextureBrickHandle &brick,
                          FieldHandle vfield, double vmin, double vmax,
                          FieldHandle gfield, double gmin, double gmax) = 0;
};


template <class FieldType>
class TextureBuilderAlgo : public TextureBuilderAlgoBase
{
public:
  typedef typename FieldType::value_type value_type;

  TextureBuilderAlgo() {}
  virtual ~TextureBuilderAlgo() {}
  
  virtual void build(TextureHandle texture,
                     FieldHandle vfield, double vmin, double vmax,
                     FieldHandle gfield, double gmin, double gmax,
                     int card_mem);
  
protected:
  virtual void fill_brick(TextureBrickHandle &brick,
                          FieldHandle vfield, double vmin, double vmax,
                          FieldHandle gfield, double gmin, double gmax);
}; 

template<typename FieldType>
void
TextureBuilderAlgo<FieldType>::build(TextureHandle texture,
                                     FieldHandle vfield,
				     double vmin, double vmax,
                                     FieldHandle gfield,
				     double gmin, double gmax,
                                     int card_mem)
{
  if(   MRLatVolField<value_type>* vmrfield =
      dynamic_cast< MRLatVolField< value_type >* > (vfield.get_rep()) ) {

    // temporary
    int nc = 1;
    int nb[2] = { 1, 0 };
    MRLatVolField<Vector>* gmrfield = 0;
    if( gfield.get_rep() ){
      if( gmrfield = 
	  dynamic_cast< MRLatVolField<Vector>* > (gfield.get_rep()) ){
	// In order to use the gradient field, it have the exact structure
	// as the value field.
	bool same = true;
	// Same number of levels?
	if( vmrfield->nlevels() == gmrfield->nlevels() ) {
	  for(int i = 0; i < vmrfield->nlevels(); ++i){
	    const MultiResLevel<value_type>* lev = vmrfield->level( i );
	    const MultiResLevel<Vector> * glev = gmrfield->level(i);
	    // Does each level have the same number of patches?
	    if( lev->patches.size() == glev->patches.size() ){
	      for(unsigned int j = 0; j < lev->patches.size(); j++ ){
		LatVolField<value_type>* vmr = lev->patches[j].get_rep(); 
		LatVolField< Vector >* gmr = glev->patches[j].get_rep();
	
		LatVolMeshHandle mesh = vmr->get_typed_mesh();
		LatVolMeshHandle gmesh = gmr->get_typed_mesh();
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
    }

    
    if( texture->nlevels() > 1 ){
      texture->lock_bricks();
      texture->clear();
      texture->unlock_bricks();
    }
    
    // Grab the transform from the lowest resolution field
    Transform tform;
      vmrfield->level(0)->patches[0].get_rep()->get_typed_mesh()->
	get_canonical_transform(tform);

    for(int i = 0 ; i < vmrfield->nlevels(); i++ ){
      const MultiResLevel<value_type>* lev = vmrfield->level( i );
      const MultiResLevel<Vector> * glev =
	( gmrfield ? gmrfield->level( i ) : 0 );
      vector<TextureBrickHandle> new_level;
      if( i == texture->nlevels() ) {
	texture->add_level(new_level);
      }
      vector<TextureBrickHandle>& bricks = texture->bricks(i);
      unsigned int k = 0;
      for(unsigned int j = 0; j < lev->patches.size(); j++ ){
	LatVolField<value_type>* vmr = lev->patches[j].get_rep(); 
	LatVolField< Vector >* gmr = ( glev ? glev->patches[j].get_rep() : 0 );
	LatVolMeshHandle mesh = vmr->get_typed_mesh();

	int nx = mesh->get_ni();
	int ny = mesh->get_nj();
	int nz = mesh->get_nk();
	 if(vfield->basis_order() == 0) {
	  --nx; --ny; --nz;
	}
	
	// make sure each sub level has a corrected bounding box
	BBox bbox(tform.unproject( mesh->get_bounding_box().min() ),
		  tform.unproject( mesh->get_bounding_box().max() ));

       	vector<TextureBrickHandle> patch_bricks;
  	texture_build_bricks(patch_bricks, nx, ny, nz, nc, nb, bbox,
                             card_mem, false);
	
	if( i == 0 ){
	  texture->set_size(nx, ny, nz, nc, nb);
	  texture->set_card_mem(card_mem);
	  texture->set_bbox(bbox);
	  texture->set_minmax(vmin, vmax, gmin, gmax);
	  texture->set_transform(tform);
	}

	texture->lock_bricks();
	for(k = 0; k < patch_bricks.size(); k++){
	  fill_brick(patch_bricks[k], vmr, vmin, vmax, gmr, gmin, gmax);
	  patch_bricks[k]->set_dirty(true);
	  bricks.push_back( patch_bricks[k] );
	}

	texture->unlock_bricks();
      }
    }
  } else {
    
    LatVolMeshHandle mesh = (LatVolMesh*)(vfield->mesh().get_rep());
    int nx = mesh->get_ni();
    int ny = mesh->get_nj();
    int nz = mesh->get_nk();
    if(vfield->basis_order() == 0) {
      --nx; --ny; --nz;
    }
    int nc = gfield.get_rep() ? 2 : 1;
    int nb[2];
    nb[0] = gfield.get_rep() ? 4 : 1;
    nb[1] = gfield.get_rep() ? 1 : 0;
    Transform tform;
    mesh->get_canonical_transform(tform);

    texture->lock_bricks();
    texture->clear();
    vector<TextureBrickHandle>& bricks = texture->bricks();
    const BBox bbox(Point(0,0,0), Point(1,1,1)); 
    if(nx != texture->nx() || ny != texture->ny() || nz != texture->nz()
       || nc != texture->nc() || card_mem != texture->card_mem() ||
       bbox.min() != texture->bbox().min())
    {
      texture_build_bricks(bricks, nx, ny, nz, nc, nb, bbox, card_mem, false);
      texture->set_size(nx, ny, nz, nc, nb);
      texture->set_card_mem(card_mem);
    }
    texture->set_bbox(bbox);
    texture->set_minmax(vmin, vmax, gmin, gmax);
    texture->set_transform(tform);
    for(unsigned int i=0; i<bricks.size(); i++) {
      fill_brick(bricks[i], vfield, vmin, vmax, gfield, gmin, gmax);
      bricks[i]->set_dirty(true);
    }
    texture->unlock_bricks();
  }
}


template <class FieldType>
void 
TextureBuilderAlgo<FieldType>::fill_brick(TextureBrickHandle &brick,
                                          FieldHandle vfield,
					  double vmin, double vmax,
                                          FieldHandle gfield,
					  double gmin, double gmax)
{
  LatVolField<value_type>* vfld = 
    dynamic_cast<LatVolField<value_type>*>(vfield.get_rep());

  if (! vfld) { 
    cerr << "dynamic cast failed! : value field" << endl;
    return;
  }

  LatVolField<Vector>* gfld = 
    dynamic_cast<LatVolField<Vector>*>(gfield.get_rep());

  if (gfield.get_rep() && !gfld) { 
    cerr << "dynamic cast failed! : gradient field" << endl;
    return;
  }

  int nc = brick->nc();
  TextureBrickT<unsigned char>* br =
    dynamic_cast<TextureBrickT<unsigned char>*>(brick.get_rep());

  if (! br) { 
    cerr << "dynamic cast failed! : brick" << endl;
    return;
  }

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
    
  if (br && vfld && ((gfld && nc == 2) || nc == 1))
  {
    typename FieldType::mesh_type* mesh = vfld->get_typed_mesh().get_rep();

    if (!gfld) { // fill only values
      unsigned char* tex = br->data(0);
      if(vfield->basis_order() == 0) {
        typename FieldType::mesh_type::Cell::range_iter iter(mesh, x0, y0, z0,
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
        typename FieldType::mesh_type::Node::range_iter iter(mesh, x0, y0, z0,
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
      
      if(vfield->basis_order() == 0) {
        typename FieldType::mesh_type::Cell::range_iter iter(mesh, x0, y0, z0,
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
        typename FieldType::mesh_type::Node::range_iter iter(mesh, x0, y0, z0,
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
  else
  {
    cerr<<"Not a Lattice type---should not be here\n";
  }
}


} // namespace SCIRun

#endif // Volume_TextureBuilderAlgo_h
