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
  virtual Texture*  build(FieldHandle, int, double, double) = 0;
  
  virtual void replace_data() = 0;

  //! support the dynamically compiled algorithm concept
  static const string& get_h_file_path();
  static CompileInfoHandle get_compile_info(const TypeDescription *td );

protected:
  double min_, max_;
  int ni_, nj_, nk_;
  int vi_, vj_, vk_;
  Transform transform_;
  BBox bbox_;
  void set_minmax( double min, double max ) { min_ = min; max_ = max; }

  double SETVAL(double val)
  {
    double v = (val - min_)*255/(max_ - min_);
    if ( v < 0 ) return 0;
    else if (v > 255) return 255;
    else return v;
  }

  unsigned char SETVALC(double val)
  {
    return (unsigned char)SETVAL(val);
  }

  void computeDivisions( int nx, int ny, int nz, int nbytes,
		    int& max_tex, int& sx, int& sy, int& sz);
  void buildBricks( BinaryTree<BrickNode *>*& tree, FieldHandle texfld,
		    int max_tex, int sx, int sy, int sz);
  BinaryTree<BrickNode *> *buildTree( int& mi, int& mj, int& mk,
				   int& ni, int& nj, int& nk, BBox bbox,
				   int& mvi, int &mvj, int& mvk,
				   int& vi, int& vj, int& vk, BBox vbox,
				   const double& di, const double& dj,
				   const double& dk, const int& max_tex, 
				   int axis, int& index);
  virtual void fillTree( BinaryTree< BrickNode *> *tree, FieldHandle texfld) = 0;
  virtual void filldata( BrickData* bdata,
			 BrickWindow *bw, FieldHandle texfld) = 0;
};

// TextureBuilderAlgo<T>

template <class TexField>
class TextureBuilderAlgo: public TextureBuilderAlgoBase
{
public:
  typedef typename TexField::value_type       value_type;
  
public:
  TextureBuilderAlgo( ){}
  virtual ~TextureBuilderAlgo(){}
  
  virtual Texture* build(FieldHandle, int, double, double);
  virtual void replace_data();
  
protected:
  virtual void fillTree( BinaryTree< BrickNode *> *tree, FieldHandle texfld);
  virtual void filldata( BrickData* bdata,
			 BrickWindow *bw, FieldHandle texfld);

private:

}; 

template<class TexField>
Texture*
TextureBuilderAlgo<TexField>::build( FieldHandle fieldH, int card_mem,
				     double minval, double maxval)
				     
{
  Texture* myTexture = 0;

  set_minmax(minval, maxval);
  // by now we've established that this is some sort of LatVolMesh
  // so this should work.
  LatVolMeshHandle mesh = (LatVolMesh *)(fieldH->mesh().get_rep());
  transform_ = mesh->get_transform();
  ni_ = mesh->get_ni();
  nj_ = mesh->get_nj();
  nk_ = mesh->get_nk();

  
//   bbox_.extend(Point(mesh->get_min_i(),
// 		     mesh->get_min_k(), mesh->get_min_j()));

  if( fieldH->data_at() == Field::CELL ){
    --ni_; --nj_; --nk_;
  }

  bbox_.reset();
  bbox_.extend( mesh->get_bounding_box().min() );
  bbox_.extend( mesh->get_bounding_box().max() );

//   BrickNode *mybn = scinew BrickNode(0,0,0);
  BinaryTree<BrickNode *> *root = 0; 
//    = scinew BinaryTree<BrickNode *>( mybn, BinaryTree<BrickNode *>::PARENT);
				    
  int brick_mem = (int)((card_mem*1024*1024)*0.5);
  int brick_size = brick_mem*8;

  int sx = 0, sy = 0, sz = 0;
  computeDivisions( ni_, nj_, nk_, 8, brick_size, sx, sy, sz);
  buildBricks( root, fieldH, brick_mem, sx, sy, sz);

  Transform mytrans;
  mesh->transform(mytrans);
  myTexture = new Texture( root, bbox_.min(), bbox_.max(), mytrans,
			   minval, maxval);
  BinaryTree<BrickNode*>* tree = myTexture->getTree();
  return myTexture;
}

template <class TexField>
void
TextureBuilderAlgo<TexField>::replace_data()
{
}


template <class TexField>
void
TextureBuilderAlgo<TexField>::fillTree( BinaryTree< BrickNode *> *tree,
			      FieldHandle texfld)
{
  if( tree->type() == BinaryTree<BrickNode*>::PARENT ){
    fillTree( tree->child(0), texfld);
    fillTree( tree->child(1), texfld );
  } else {
    BrickNode* bn = tree->stored();
    BrickWindow* bw = bn->brickWindow();
    BrickData* bd = bn->brick()->data();
    filldata( bd, bw, texfld);
  }
}

template <class TexField>
void 
TextureBuilderAlgo<TexField>::filldata( BrickData* bdata,
				  BrickWindow *bw,
				  FieldHandle field)
{

  if( LatVolField< value_type >* fld = 
      dynamic_cast<LatVolField< value_type >* >(field.get_rep())){
   
    typename TexField::mesh_type *m = fld->get_typed_mesh().get_rep();
    unsigned char*** tex;
    TypedBrickData< unsigned char >* tbd;
    if( tbd =  dynamic_cast<TypedBrickData< unsigned char > *>( bdata )) {
      tex = tbd->data();
    } else {
      cerr<<"Some sort of error in TextureBuilderAlgo<TexField>::filldata() \n";
      return;
    }

    int i,j,k, ii, jj, kk;
    int x0, y0, z0, x1, y1, z1;
    bw->getBoundingIndices(x0,y0,z0,x1,y1,z1);
  
    if( field->data_at() == Field::CELL ){
      typename TexField::mesh_type::RangeCellIter iter(m, x0, y0, z0,
						       x1+1, y1+1, z1+1);
      typename TexField::mesh_type::CellIter iter_end; iter.end( iter_end );
    
      double tmp = 0;
      for( k = 0, kk = z0; kk <= z1; kk++,k++){
	for(j = 0, jj = y0; jj <= y1; jj++, j++){
	  for(i = 0, ii = x0; ii <= x1; ii++, i++){
	    tex[k][j][i] = SETVALC( fld->fdata()[*iter] );
	    ++iter;
	  }
	}
      }
    } else {
      typename TexField::mesh_type::RangeNodeIter iter(m, x0, y0, z0,
						       x1+1, y1+1, z1+1);
      typename TexField::mesh_type::NodeIter iter_end; iter.end( iter_end );
      
      double tmp = 0;
      for( k = 0, kk = z0; kk <= z1; kk++,k++){
	for(j = 0, jj = y0; jj <= y1; jj++, j++){
	  for(i = 0, ii = x0; ii <= x1; ii++, i++){
	    tex[k][j][i] = SETVALC( fld->fdata()[*iter] );
	    ++iter;
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
