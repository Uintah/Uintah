/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  GLTexture3DBuilder:
 *  Written by:
 *   Kurt Zimmerman
 *   SCI Institute
 *   University of Utah
 *   Jan 2004
 *
 *  Copyright (C) 2003 SCI Group
 */


#ifndef TextureBuilderAlgo_h
#define TextureBuilderAlgo_h

#include <Core/Datatypes/Datatype.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Transform.h>
#include <Core/Containers/BinaryTree.h>
#include <Core/Thread/ThreadGroup.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Runnable.h>
#include <sgi_stl_warnings_off.h>
#include <sstream>
#include <sgi_stl_warnings_on.h>

#include <Core/Util/DynamicLoader.h>
#include <Packages/Volume/Core/Datatypes/Texture.h>
#include <Packages/Volume/Core/Datatypes/TypedBrickData.h>
#include <Packages/Volume/Core/Datatypes/BrickWindow.h>

namespace Volume {
using std::ostringstream;

using SCIRun::Transform;
using SCIRun::BBox;
class BrickNode;

class TextureBuilderAlgoBase : public DynamicAlgoBase {
public:
  TextureBuilderAlgoBase();
  virtual ~TextureBuilderAlgoBase();
  virtual Texture*  build(FieldHandle, FieldHandle, int,
                          double, double, double, double) = 0;
  
  virtual void replace_data() = 0;

  //! support the dynamically compiled algorithm concept
  static const string& get_h_file_path();
  static CompileInfoHandle get_compile_info(const TypeDescription *td );

protected:
  double vmin_, vmax_, gmin_, gmax_;
  int ni_, nj_, nk_;
  int nc_, nb_[2];
  int vi_, vj_, vk_;
  Transform transform_;
  BBox bbox_;
  void set_minmax(double vmin, double vmax, double gmin, double gmax)
  { vmin_ = vmin; vmax_ = vmax; gmin_ = gmin; gmax_ = gmax; }

  double SETVAL(double val)
  {
    double v = (val - vmin_)*255/(vmax_ - vmin_);
    if ( v < 0 ) return 0;
    else if (v > 255) return 255;
    else return v;
  }

  unsigned char SETVALC(double val)
  {
    return (unsigned char)SETVAL(val);
  }

  void computeDivisions(int nx, int ny, int nz, int nb,
                        int& max_tex, int& sx, int& sy, int& sz);
  void buildBricks(BinaryTree<BrickNode*>*& tree,
                   FieldHandle vfield, FieldHandle gfield,
                   int max_tex, int sx, int sy, int sz, int nc, int* nb);
  BinaryTree<BrickNode*>* buildTree(int& mi, int& mj, int& mk,
                                    int& ni, int& nj, int& nk, int nc, int* nb,
                                    BBox bbox,
                                    int& mvi, int &mvj, int& mvk,
                                    int& vi, int& vj, int& vk, BBox vbox,
                                    const double& di, const double& dj,
                                    const double& dk, const int& max_tex, 
                                    int axis, int& index);
  virtual void fillTree(BinaryTree<BrickNode*>* tree,
                        FieldHandle vfield,
                        FieldHandle gfield) = 0;
  virtual void filldata(BrickData* bdata,
			BrickWindow* bw,
                        FieldHandle vfield,
                        FieldHandle gfield) = 0;
};

// TextureBuilderAlgo<T>

template <class TexField>
class TextureBuilderAlgo: public TextureBuilderAlgoBase
{
public:
  typedef typename TexField::value_type value_type;
  
public:
  TextureBuilderAlgo( ){}
  virtual ~TextureBuilderAlgo(){}
  
  virtual Texture* build(FieldHandle, FieldHandle, int,
                         double, double, double, double);
  virtual void replace_data();
  
protected:
  virtual void fillTree(BinaryTree<BrickNode*>* tree,
                        FieldHandle vfield,
                        FieldHandle gfield);
  virtual void filldata(BrickData* bdata,
			BrickWindow* bw,
                        FieldHandle vfield,
                        FieldHandle gfield);
}; 

template<class TexField>
Texture*
TextureBuilderAlgo<TexField>::build(FieldHandle vfield, FieldHandle gfield,
                                    int card_mem, double vminval, double vmaxval,
                                    double gminval, double gmaxval)
				     
{
  Texture* myTexture = 0;

  set_minmax(vminval, vmaxval, gminval, gmaxval);
  // by now we've established that this is some sort of LatVolMesh
  // so this should work.
  LatVolMeshHandle mesh = (LatVolMesh*)(vfield->mesh().get_rep());
  transform_ = mesh->get_transform();
  
  ni_ = mesh->get_ni();
  nj_ = mesh->get_nj();
  nk_ = mesh->get_nk();
  
//   bbox_.extend(Point(mesh->get_min_i(),
// 		     mesh->get_min_k(), mesh->get_min_j()));

  if(vfield->data_at() == Field::CELL) {
    --ni_; --nj_; --nk_;
  }

  bbox_.reset();
  bbox_.extend(mesh->get_bounding_box().min());
  bbox_.extend(mesh->get_bounding_box().max());

  // compute subdivision
  int brick_mem = card_mem*1024*1024/2;
  nb_[0] = gfield.get_rep() ? 4 : 1;
  nb_[1] = gfield.get_rep() ? 1 : 0;
  nc_ = gfield.get_rep() ? 2 : 1;
  int sx = 0, sy = 0, sz = 0;
  computeDivisions(ni_, nj_, nk_, nb_[0], brick_mem, sx, sy, sz);

  // build brick tree
  BinaryTree<BrickNode*>* root = 0; 
  buildBricks(root, vfield, gfield, brick_mem, sx, sy, sz, nc_, nb_);

  //
  Transform mytrans;
  mesh->transform(mytrans);
  myTexture = new Texture(root, bbox_.min(), bbox_.max(), mytrans,
			  vminval, vmaxval, gminval, gmaxval);
  //BinaryTree<BrickNode*>* tree = myTexture->getTree();
  return myTexture;
}

template <class TexField>
void
TextureBuilderAlgo<TexField>::replace_data()
{
}

template <class TexField>
void
TextureBuilderAlgo<TexField>::fillTree(BinaryTree< BrickNode *> *tree,
                                       FieldHandle vfield,
                                       FieldHandle gfield)
{
  if( tree->type() == BinaryTree<BrickNode*>::PARENT ){
    fillTree(tree->child(0), vfield, gfield);
    fillTree(tree->child(1), vfield, gfield);
  } else {
    BrickNode* bn = tree->stored();
    BrickWindow* bw = bn->brickWindow();
    BrickData* bd = bn->brick()->data();
    filldata(bd, bw, vfield, gfield);
  }
}

template <class TexField>
void 
TextureBuilderAlgo<TexField>::filldata(BrickData* bdata,
                                       BrickWindow* bw,
                                       FieldHandle vfield,
                                       FieldHandle gfield)
{
  LatVolField<value_type>* vfld = 
    dynamic_cast<LatVolField<value_type>*>(vfield.get_rep());
  LatVolField<Vector>* gfld = 
    dynamic_cast<LatVolField<Vector>*>(gfield.get_rep());

  int nc = bdata->nc();
  
  if (vfld && ((gfld && nc == 2) || nc == 1)) {
   
    typename TexField::mesh_type* m = vfld->get_typed_mesh().get_rep();

    TypedBrickData<unsigned char>* tbd =
      dynamic_cast<TypedBrickData<unsigned char>*>(bdata);
    
    if(!tbd) {
      cerr << "Some sort of error in TextureBuilderAlgo<TexField>::filldata() \n";
      return;
    }

    int x0, y0, z0, x1, y1, z1;
    bw->getBoundingIndices(x0, y0, z0, x1, y1, z1);
    
    if (!gfld) {
      unsigned char*** tex = tbd->data(0);
      if(vfield->data_at() == Field::CELL) {
        typename TexField::mesh_type::RangeCellIter iter(m, x0, y0, z0,
                                                         x1+1, y1+1, z1+1);
        typename TexField::mesh_type::CellIter iter_end;
        iter.end(iter_end);
    
        //double tmp = 0;
        for(int k=0, kk=z0; kk<=z1; kk++, k++) {
          for(int j=0, jj=y0; jj<=y1; jj++, j++) {
            for(int i=0, ii=x0; ii<=x1; ii++, i++) {
              tex[k][j][i] = SETVALC(vfld->fdata()[*iter]);
              ++iter;
            }
          }
        }
      } else {
        typename TexField::mesh_type::RangeNodeIter iter(m, x0, y0, z0,
                                                         x1+1, y1+1, z1+1);
        typename TexField::mesh_type::NodeIter iter_end;
        iter.end(iter_end);
      
        //double tmp = 0;
        for(int k=0, kk=z0; kk<=z1; kk++, k++) {
          for(int j=0, jj=y0; jj<=y1; jj++, j++) {
            for(int i=0, ii=x0; ii<=x1; ii++, i++) {
              tex[k][j][i] = SETVALC(vfld->fdata()[*iter]);
              ++iter;
            }
          }
        }
      }
    } else {
      unsigned char*** tex0 = tbd->data(0);
      unsigned char*** tex1 = tbd->data(1);
      
      if(vfield->data_at() == Field::CELL) {
        typename TexField::mesh_type::RangeCellIter iter(m, x0, y0, z0,
                                                         x1+1, y1+1, z1+1);
        typename TexField::mesh_type::CellIter iter_end;
        iter.end(iter_end);
    
        //double tmp = 0;
        for(int k=0, kk=z0; kk<=z1; kk++, k++) {
          for(int j=0, jj=y0; jj<=y1; jj++, j++) {
            for(int i=0, ii=x0; ii<=x1; ii++, i++) {
              tex0[k][j][i*4+3] = SETVALC(vfld->fdata()[*iter]);
              Vector v = gfld->fdata()[*iter];
              double vlen = v.length();
              if(vlen > std::numeric_limits<float>::epsilon())
                v.normalize();
              else
                v = Vector(0.0, 0.0, 0.0);
              tex0[k][j][i*4+0] = (unsigned char)((v.x()*0.5 + 0.5)*255);
              tex0[k][j][i*4+1] = (unsigned char)((v.y()*0.5 + 0.5)*255);
              tex0[k][j][i*4+2] = (unsigned char)((v.z()*0.5 + 0.5)*255);
              tex1[k][j][i] = (unsigned char)((vlen-gmin_)/(gmax_-gmin_))*255;
              ++iter;
            }
          }
        }
      } else {
        typename TexField::mesh_type::RangeNodeIter iter(m, x0, y0, z0,
                                                         x1+1, y1+1, z1+1);
        typename TexField::mesh_type::NodeIter iter_end;
        iter.end(iter_end);
      
        //double tmp = 0;
        for(int k=0, kk=z0; kk<=z1; kk++, k++) {
          for(int j=0, jj=y0; jj<=y1; jj++, j++) {
            for(int i=0, ii=x0; ii<=x1; ii++, i++) {
              tex0[k][j][i*4+3] = SETVALC(vfld->fdata()[*iter]);
              Vector v = gfld->fdata()[*iter];
              double vlen = v.length();
              if(vlen > std::numeric_limits<float>::epsilon())
                v.normalize();
              else
                v = Vector(0.0, 0.0, 0.0);
              tex0[k][j][i*4+0] = (unsigned char)((v.x()*0.5 + 0.5)*255);
              tex0[k][j][i*4+1] = (unsigned char)((v.y()*0.5 + 0.5)*255);
              tex0[k][j][i*4+2] = (unsigned char)((v.z()*0.5 + 0.5)*255);
              tex1[k][j][i] = (unsigned char)((vlen-gmin_)/(gmax_-gmin_))*255;
              ++iter;
            }
          }
        }
      }
      
    }      
  } else {
    cerr<<"Not a Lattice type---should not be here\n";
  }
}

} // end namespace Volume

#endif // GLTexture3DBuilder_h
