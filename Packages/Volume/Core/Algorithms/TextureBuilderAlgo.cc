//  The contents of this file are subject to the University of Utah Public
//  License (the "License"); you may not use this file except in compliance
//  with the License.
//  
//  Software distributed under the License is distributed on an "AS IS"
//  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
//  License for the specific language governing rights and limitations under
//  the License.
//  
//  The Original Source Code is SCIRun, released March 12, 2001.
//  
//  The Original Source Code was developed by the University of Utah.
//  Portions created by UNIVERSITY are Copyright (C) 2001, 1994
//  University of Utah. All Rights Reserved.
//  
//    File   : GLTexture3DBuilder.cc
//    Author : Martin Cole
//    Date   : Fri Jun 15 17:09:17 2001

#include <Packages/Volume/Core/Datatypes/BrickWindow.h>
#include <Packages/Volume/Core/Datatypes/TypedBrickData.h>
#include <Packages/Volume/Core/Algorithms/TextureBuilderAlgo.h>
#include <Core/Util/DebugStream.h>

#include <iostream>

namespace Volume {

using namespace SCIRun;

using namespace std;


static DebugStream dbg("TextureBuilderAlgo", false);


TextureBuilderAlgoBase::TextureBuilderAlgoBase()
{}

TextureBuilderAlgoBase::~TextureBuilderAlgoBase() 
{}

const string& 
TextureBuilderAlgoBase::get_h_file_path() 
{
  static const string path(TypeDescription::cc_to_h(__FILE__));
  return path;
}

CompileInfoHandle
TextureBuilderAlgoBase::get_compile_info(const TypeDescription *td )
{
  string subname;
  string subinc;
  string sname = td->get_name("", "");

  //Test for LatVolField inheritance...
  if (sname.find("LatVol") != string::npos ) {
    // we are dealing with a lattice vol or inherited version
    //subname.append("TextureBuilderAlgo<" + td->get_name() + "> ");
    subname.append(td->get_name());
    subinc.append(get_h_file_path());
  } else {
    cerr << "Unsupported Geometry, needs to be of Lattice type." << endl;
    subname.append("Cannot compile this unsupported type");
  }

  string fname("TextureBuilderAlgo." + td->get_filename() + ".");
  CompileInfo *rval = scinew CompileInfo(fname, "TextureBuilderAlgoBase", 
					 "TextureBuilderAlgo",
					 subname);
  rval->add_include(get_h_file_path());
  rval->add_include(subinc);
  rval->add_namespace("Volume");
  td->fill_compile_info(rval);
  return rval;
}

// Compute the divisions needed along each axis
void
TextureBuilderAlgoBase::computeDivisions( int nx, int ny, int nz, 
					  int nbytes, int& max_tex,
					  int& sx, int& sy, int& sz)
{
  int width, height, depth;
  int u,v,w;
  if( isPowerOf2( nx ) ) u = nx; else u = nextPowerOf2( nx );
  if( isPowerOf2( ny ) ) v = ny; else v = nextPowerOf2( ny );
  if( isPowerOf2( nz ) ) w = nz; else w = nextPowerOf2( nz );

  char str[400];
//   sprintf( str, "u, v, w = %d, %d, %d;  nx, ny, nx = %d, %d, %d\n",
// 	   u, v, w, nx, ny, nz);
//   dbg<<str;

  if( u * v * w * nbytes < max_tex ){
    return;
  } else {
    int padx, pady, padz;
    padx = u - nx;
    pady = v - ny;
    padz = w - nz;

    if( padx > pady && padx > padz ){
      nx = u = u/2;
      sx++;
    } else if( pady > padz){
      ny = v = v/2;
      sy++;
    } else if( padz > 0 ){
      nz = w = w/2;
      sz++;
    } else if( u > v && u > w && u > 256 ){
      nx = u = u/2;
      sx++;
    } else if( v > w && v > 256){
      ny = v = v/2;
      sy++;
    } else {
      nz = w = w/2;
      sz++;
    }
    computeDivisions( nx, ny, nz, nbytes, max_tex, sx, sy, sz );
  }
}


void
TextureBuilderAlgoBase::buildBricks(  BinaryTree<BrickNode *>*& tree, FieldHandle texfld,
				     int max_tex, int sx, int sy, int sz)
{
  char str[400];  

  sprintf(str, " division size is %dx%dx%d\n", sx, sy, sz );
  dbg<<str;

  if( !tree ) { // build one
    // compute the virtual box. First, compute the next largest
    // power of two box:  sx, sy, sz represent the number of
    // slices in each direction that will be necessary.  Make
    // sure that the virtual domain is large enough to cover all 
    // of those divisions and account for overlap.
    int u,v,w, nx, ny, nz;
    nx = ni_ + ((sx == 0) ? 0 : ((int)pow(2.0, double(sx)) - 1));
    ny = nj_ + ((sy == 0) ? 0 : ((int)pow(2.0, double(sy)) - 1));
    nz = nk_ + ((sz == 0) ? 0 : ((int)pow(2.0, double(sz)) - 1));
    if( isPowerOf2( nx )) u = nx; else u = nextPowerOf2( nx );
    if( isPowerOf2( ny )) v = ny; else v = nextPowerOf2( ny );
    if( isPowerOf2( nz )) w = nz; else w = nextPowerOf2( nz );

    // then compute the size of each voxel
    double dx, dy, dz;
    Vector range( bbox_.max() - bbox_.min() );
    dbg<<"range = "<< range <<"\n";;
    dx = range.x()/(ni_);
    dy = range.y()/(nj_);
    dz = range.z()/(nk_);

    // now we have the virtual box
    BBox vbox( bbox_.min() + Vector( dx*0.5, dy*0.5, dz*0.5),
               bbox_.min() + Vector( dx * (u - 1),
                                     dy * (v - 1), 
                                     dz * (w - 1)));
    //bricks_.clear();
    int brickIndex = 0;
    nx = ni_, ny = nj_, nz = nk_;
    vi_ = u; vj_ - v; vk_ + w;
    int mx = 0, mvx = 0, my = 0, mvy = 0, mz = 0, mvz = 0;
    int bx = 0, bvx = 0, by = 0, bvy = 0, bz = 0, bvz = 0;
    
    
    sprintf(str, "texture size is %dx%dx%d\n", nx, ny, nz );
    dbg<<str;
    
    sprintf(str, "\n #:    size     |  tex_size   |   min idx   |   max idx   |              bbmin             ->             bbmax               |           tmin            ->           tmax\n");
    dbg<<str;
    tree = buildTree( mx, my, mz, nx, ny, nz, bbox_,
		      mvx, mvy, mvz, u, v, w, vbox,
		      dx, dy, dz, max_tex, 0, brickIndex );
  }
  
  dbg<<"Fill the tree with data ... ";
  // if we made it this far, we must either fill a new tree
  // or refill the existing one, with or without a mask
  fillTree( tree, texfld );
  dbg<<"Tree filled\n";
  
}
/// Helper Functions for BuildTree
Point cc_point_min( Point min, double dx, double dy, double dz,
                int i, int j, int k, int nx, int ny, int nz)
{
  return min + Vector( dx * (i == 0 ? 0 : (i + 0.5)),
                       dy * (j == 0 ? 0 : (j + 0.5)),
                       dz * (k == 0 ? 0 : (k + 0.5)));
}
Point cc_point_max( Point min, double dx, double dy, double dz,
		    int i, int j, int k, int nx, int ny, int nz)
{
  return min + Vector( dx * (i == nx -1 ? nx : (i + 0.5)),
                       dy * (j == ny -1 ? ny : (j + 0.5)),
                       dz * (k == nz -1 ? nz : (k + 0.5)));
}

Point t_min_point( Point min, double dx, double dy, double dz,
                int i, int j, int k, int nx, int ny, int nz) {
  return min + Vector( dx * (i == 0 ? 0 : 0.5 ),
                       dy * (j == 0 ? 0 : 0.5 ),
                       dz * (k == 0 ? 0 : 0.5 ));
}
Point t_max_point( Point min, double dx, double dy, double dz,
		   int padi, int padj, int padk,
                int i, int j, int k, int nx, int ny, int nz)
{
  return min + Vector( ( i == nx - 1 ? 1.0 -(padi*dx) : 1.0 - (0.5+padi)*dx ),
                       ( j == ny - 1 ? 1.0 -(padj*dy) : 1.0 - (0.5+padj)*dy ),
                       ( k == nz - 1 ? 1.0 -(padk*dz) : 1.0 - (0.5+padk)*dz ));
}

Point node_point(Point min, double dx, double dy, double dz,
                int i, int j, int k )
{
  return min + Vector( dx * i,
                       dy * j,
                       dz * k);
}

BinaryTree<BrickNode *>*
TextureBuilderAlgoBase::buildTree( int& mi, int& mj, int& mk,
				   int& ni, int& nj, int& nk, BBox bbox,
				   int& mvi, int &mvj, int& mvk,
				   int& vi, int& vj, int& vk, BBox vbox,
                   const double& di, const double& dj, const double& dk,
                   const int& max_tex, int axis, int& index)
{
  char str[400];
  BinaryTree<BrickNode *> *tree;
  int ti, tj, tk;
  if( isPowerOf2( ni )) ti = ni; else ti = nextPowerOf2(ni); 
  if( isPowerOf2( nj )) tj = nj; else tj = nextPowerOf2(nj);  
  if( isPowerOf2( nk )) tk = nk; else tk = nextPowerOf2(nk);  
  if( ti * tj * tk  < max_tex ) {
    BBox bb( cc_point_min(bbox.min(), di, dj, dk, mi, mj, mk, ni_, nj_, nk_),
             cc_point_max(bbox.min(), di, dj, dk, mi+ni-1,mj+nj-1,mk+nk-1,
                      ni_, nj_, nk_));
//     BBox vb( node_point(vbox.min(), di, dj, dk, mvi, mvj, mvk),
//              node_point(vbox.min(), di, dj, dk,
//                         mvi+vi-1, mvj+vj-1, mvk+vk-1));
    BBox vb( cc_point_min(bbox.min(), di, dj, dk, mvi, mvj, mvk, vi_, vj_, vk_),
             cc_point_max(bbox.min(), di, dj, dk, mvi+ti-1, mvj+tj-1, mvk+tk-1,
                      vi_, vj_, vk_));
    
    Point tmin(0,0,0);
    BBox tb( t_min_point( tmin, 1.0/ti, 1.0/tj, 1.0/tk,
                          mi, mj, mk, ni_, nj_, nk_),
             t_max_point( tmin, 1.0/ti, 1.0/tj, 1.0/tk,
			  ti - ni, tj - nj, tk - nk,
                          mi+ni-1, mj+nj-1, mk+nk-1, ni_, nj_, nk_));


    sprintf(str, "%2d: %3d %3d %3d | %3d %3d %3d | %3d %3d %3d | %3d %3d %3d | %9.5f %9.5f %9.5f  ->  %9.5f %9.5f %9.5f  |  %.5f %.5f %.5f  ->  %.5f %.5f %.5f\n", index, ni, nj, nk, ti, tj, tk, mi, mj, mk, mi+ni-1, mj+nj-1, mk+nk-1,bb.min().x(), bb.min().y(), bb.min().z(), bb.max().x(), bb.max().y(), bb.max().z(), tb.min().x(), tb.min().y(), tb.min().z(), tb.max().x(), tb.max().y(), tb.max().z());
    dbg<<str;

    BrickWindow *bw = new BrickWindow( mi, mj, mk,
                                       mi + ni - 1,
                                       mj + nj - 1,
                                       mk + nk - 1, 
                                       vb);


    TypedBrickData<unsigned char> *bd =
      new TypedBrickData<unsigned char>(ti, tj, tk, sizeof(unsigned char));
    Brick *b = new Brick(bd, ti - ni, tj - nj, tk - nk, &bb, &tb);
    BrickNode *bn = new BrickNode(b, bw, index++);
    tree = new BinaryTree<BrickNode *>( bn, BinaryTree<BrickNode *>::LEAF);
  } else {
    int child_axis;
    if( vi - ni > vj - nj &&  vi - ni > vk - nk ) {
      child_axis = 1;
    } else if( vj - nj > vk - nk) {
      child_axis = 2;
    } else if( vk - nk > 0 ){
      child_axis = 3;
    } else if( ni > nj && ni > nk){
      child_axis = 1;
    } else if( nj >= nk) {
      child_axis = 2;
    } else {
      child_axis = 3;
    }

    BinaryTree<BrickNode *> *child0;
    BinaryTree<BrickNode *> *child1;
    int _ni = ni, _nj = nj, _nk = nk;
    int _vi = vi, _vj = vj, _vk = vk;
    switch( child_axis ){
    case 1:
//       dbg << "_ni and _vi are switched from "<<_ni<<" and "<<_vi;
      _ni = Min(ni_, vi/2);
      _vi = vi/2;
//       dbg <<" to "<< _ni<<" and "<<_vi<<"\n";
      break;
    case 2:
//       dbg << "_nj and _vj are switched from "<<_nj<<" and "<<_vj;
      _nj = Min(nj_, vj/2);
      _vj = vj/2;
//       dbg <<" to "<< _nj<<" and "<<_vj<<"\n";
      break;
    case 3:
//       dbg << "_nk and _vk are switched from "<<_nk<<" and "<<_vk;
      _nk = Min(nk_, vk/2);
      _vk = vk/2;
//       dbg <<" to "<< _nk<<" and "<<_vk<<"\n";
      break;
    }
    child0 = buildTree(mi, mj, mk, _ni, _nj, _nk, bbox,
                       mvi, mvj, mvk, _vi, _vj, _vk, vbox,
                       di, dj, dk, max_tex, child_axis, index);

    BrickWindow *bw0 = child0->stored()->brickWindow();

    switch( child_axis ){
    case 1:
//       dbg << "_ni and mi are switched from "<<_ni<<" and "<<mi;
      mi = mvi = bw0->max_i();
      _ni = Min( _ni, ni_ - mi);
//       dbg <<" to "<< _ni<<" and "<<mi<<"\n";
      break;      break;
    case 2:
//       dbg << "_nj and mj are switched from "<<_nj<<" and "<<mj;
      mj = mvj = bw0->max_j();
      _nj = Min( _nj, nj_ - mj);
//       dbg <<" to "<< _nj<<" and "<<mj<<"\n";
      break;      break;
    case 3:
//       dbg << "_nk and mk are switched from "<<_nk<<" and "<<mk;
      mk = mvk = bw0->max_k();
      _nk = Min( _nk, nk_ - mk);
//       dbg <<" to "<< _nk<<" and "<<mk<<"\n";
      break;      break;
    }

    child1 = buildTree(mi, mj, mk, _ni, _nj, _nk, bbox,
                       mvi, mvj, mvk, _vi, _vj, _vk, vbox,
                       di, dj, dk, max_tex, axis, index );


    BBox vb;
    BrickWindow *bw1 = child1->stored()->brickWindow();
    vb.extend( bw0->vbox().min() );
    vb.extend( bw1->vbox().max() );
    BrickWindow *bw =
      new BrickWindow( bw0->min_i(), bw0->min_j(), bw0->min_k(),
                       bw1->max_i(), bw1->max_j(), bw1->max_k(),
                       vb);
    switch( child_axis ){
    case 1:
//       dbg<<"vi is switched to from "<<vi<<" to ";
      vi = 2*_vi;
//       dbg<<vi<<"\n";
      break;
    case 2:
//       dbg<<"vj is switched to from "<<vj<<" to ";
      vj = 2*_vj;
//       dbg<<vj<<"\n";
      break;
    case 3:
//       dbg<<"vk is switched to from "<<vk<<" to ";
      vk = 2*_vk;
//       dbg<<vj<<"\n";
      break;
    }  
    
//     sprintf(str,"mi, mj, mk, mvi, mvj, mvk, are being switched from %d, %d, %d, %d, %d, %d to ", mi, mj, mk, mvi, mvj, mvk); 
//     dbg<<str;
    switch( axis ){
    case 1:
//       mi =  mvi = bw1->max_i();
      mj =  mvj = bw0->min_j();
      mk =  mvk = bw0->min_k();
      break;
    case 2:
      mi =  mvi = bw0->min_i();
//       mj =  mvj = bw1->max_j();
      mk =  mvk = bw0->min_k();
      break;
    case 3:
      mi =  mvi = bw0->min_i();
      mj =  mvj = bw0->min_j();
//       mk =  mvk = bw1->max_k();
      break;
    }
//     sprintf(str," %d, %d, %d, %d, %d, %d\n", mi, mj, mk, mvi, mvj, mvk);
//     dbg<<str;
    
//     sprintf(str,"ni, nj, nk are being switched from %d, %d, %d to ", ni, nj, nk);
//     dbg << str;
    ni = bw1->max_i() - bw0->min_i() + 1;
    nj = bw1->max_j() - bw0->min_j() + 1;
    nk = bw1->max_k() - bw0->min_k() + 1;
//     sprintf(str,"%d, %d, %d\n", ni, nj, nk);
//     dbg<< str;

    BrickNode *bn = new BrickNode( 0, bw, -1);
    tree = new BinaryTree<BrickNode *>(bn, BinaryTree<BrickNode*>::PARENT);
    tree->AddChild( child0, 0 );
    tree->AddChild( child1, 1 );
  }
  return tree;
}

} // end namespace Volume
