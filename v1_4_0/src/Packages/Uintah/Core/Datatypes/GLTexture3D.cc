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

#include <Packages/Uintah/Core/Datatypes/GLTexture3D.h>
#include <Packages/Uintah/Core/Datatypes/LevelField.h>
#include <pair.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Semaphore.h>
#include <Core/Thread/ThreadGroup.h>

using std::pair;


namespace SCIRun{
template<> 
bool SCIRun::GLTexture3D::get_dimensions(Uintah::LevelMeshHandle m,
				 int& nx, int& ny, int& nz)
  {
    nx = m->get_nx();
    ny = m->get_ny();
    nz = m->get_nz();
    return true;
  }
} // End namespace SCIRun


namespace Uintah {

using SCIRun::Thread;
using SCIRun::ThreadGroup;
using SCIRun::Semaphore;


GLTexture3D::GLTexture3D() :
  SCIRun::GLTexture3D()
{
}



GLTexture3D::GLTexture3D(FieldHandle texfld, double &min, double &max, 
			 int use_minmax)
  :  SCIRun::GLTexture3D()
{
  texfld_ = texfld;
  if (texfld_->get_type_name(0) != "LevelField") {
        cerr << "GLTexture3D constructor error - can only make a GLTexture3D from a LevelField\n";
    return;
  }

  pair<double,double> minmax;
  LevelMeshHandle mesh_;
  const string type = texfld_->get_type_name(1);
  if (type == "double") {
    LevelField<double> *fld =
      dynamic_cast<LevelField<double>*>(texfld_.get_rep());
    if( !use_minmax ) fld->minmax(minmax);
    mesh_ = fld->get_typed_mesh();
  } else if (type == "float") {
    LevelField<float> *fld =
      dynamic_cast<LevelField<float>*>(texfld_.get_rep());
    if( !use_minmax ) fld->minmax(minmax);
    mesh_ = fld->get_typed_mesh();
  } else if (type == "long") {
    LevelField<long> *fld =
      dynamic_cast<LevelField<long>*>(texfld_.get_rep());
    if( !use_minmax ) fld->minmax(minmax);
    mesh_ = fld->get_typed_mesh();
//    } else if (type == "int") {
//      LevelField<int> *fld =
//        dynamic_cast<LevelField<int>*>(texfld_.get_rep());
//      if( !use_minmax ) fld->minmax(minmax);
//      mesh_ = fld->get_typed_mesh();
//    } else if (type == "unsigned_char") {
//      LevelField<unsigned char> *fld =
//        dynamic_cast<LevelField<unsigned char>*>(texfld_.get_rep());
//      if( !use_minmax ) fld->minmax(minmax);
//      mesh_ = fld->get_typed_mesh();
  } else {
    cerr << "GLTexture3D constructor error - unknown LevelField type: " << type << endl;
    return;
  }
  xmax_=ymax_=zmax_=128;
   if( texfld_->data_at() == Field::CELL ){
     isCC_=true;
     X_ = mesh_->get_nx()-1;
     Y_ = mesh_->get_ny()-1;
     Z_ = mesh_->get_nz()-1;
   } else {
    isCC_=false;
    X_ = mesh_->get_nx();
    Y_ = mesh_->get_ny();
    Z_ = mesh_->get_nz();
   }    
  minP_ = mesh_->get_min();
  maxP_ = mesh_->get_max();
  cerr <<"X_, Y_, Z_ = "<<X_<<", "<<Y_<<", "<<Z_<<endl;
  cerr << "use_minmax = "<<use_minmax<<"  min="<<min<<" max="<<max<<"\n";
  cerr << "    fieldminmax: min="<<minmax.first<<" max="<<minmax.second<<"\n";
  if (use_minmax) {
    min_ = min;
    max_ = max;
  } else {
    min = min_ = minmax.first;
    max = max_ = minmax.second;
  }
  cerr << "    texture: min="<<min<<"  max="<<max<<"\n";
  set_bounds();
  compute_tree_depth(); 
  build_texture();
}

GLTexture3D::~GLTexture3D()
{
}

void GLTexture3D::build_texture()
{
  max_workers = Max(Thread::numProcessors()/2, 8);
  Semaphore* thread_sema = scinew Semaphore( "worker count semhpore",
					  max_workers);  

  string group_name( "thread group ");
  group_name = group_name + "0";
  tg = scinew ThreadGroup( group_name.c_str() );

  string type = texfld_->get_type_name(1);
  cerr << "Type = " << type << endl;

  if (type == "double") {
    bontree_ = build_bon_tree(minP_, maxP_, 0, 0, 0, X_, Y_, Z_, 0, 
	       dynamic_cast<LevelField<double>*>(texfld_.get_rep()), 0, 
			   thread_sema, tg);
  } else if (type == "float") {
    bontree_ = build_bon_tree(minP_, maxP_, 0, 0, 0, X_, Y_, Z_, 0, 
	       dynamic_cast<LevelField<float>*>(texfld_.get_rep()), 0, 
			   thread_sema, tg);
  } else if (type == "long") {
    bontree_ = build_bon_tree(minP_, maxP_, 0, 0, 0, X_, Y_, Z_, 0, 
	       dynamic_cast<LevelField<long>*>(texfld_.get_rep()), 0, 
			   thread_sema, tg);
//   } else if (type == "int") {
//     bontree_ = build_bon_tree(minP_, maxP_, 0, 0, 0, X_, Y_, Z_, 0, 
// 	       dynamic_cast<LevelField<int>*>(texfld_.get_rep()), 0);
//   } else if (type == "unsigned short") {
//     bontree_ = build_bon_tree(minP_, maxP_, 0, 0, 0, X_, Y_, Z_, 0, 
// 	       dynamic_cast<LevelField<unsigned short>*>(texfld_.get_rep()),0);
//   } else if (type == "short") {
//     bontree_ = build_bon_tree(minP_, maxP_, 0, 0, 0, X_, Y_, Z_, 0, 
// 	       dynamic_cast<LevelField<short>*>(texfld_.get_rep()), 0);
//   } else if (type == "unsigned char") {
//     bontree_ = build_bon_tree(minP_, maxP_, 0, 0, 0, X_, Y_, Z_, 0, 
// 	       dynamic_cast<LevelField<unsigned char>*>(texfld_.get_rep()), 0);
//   } else if (type == "char") {
//     bontree_ = build_bon_tree(minP_, maxP_, 0, 0, 0, X_, Y_, Z_, 0, 
// 	       dynamic_cast<LevelField<char>*>(texfld_.get_rep()), 0);
  } else {
    cerr<<"Error: cast didn't work!\n";
  }
  tg->join();
  delete tg;
  thread_sema->down(max_workers);
  ASSERT(bontree_ != 0x0);
}

void
GLTexture3D::set_field( FieldHandle texfld )
{
  if (texfld_->get_type_name(0) != "LevelField") {
    cerr << "GLTexture3D constructor error - can only make a GLTexture3D from a LevelField\n";
    return;
  }
  texfld_=texfld;

  int size = std::max(X_,Y_);
  size = std::max(size,Z_);

  xmax_ = ymax_ = zmax_ = 128;

  set_bounds();
  compute_tree_depth(); 
  build_texture();
} 

bool
GLTexture3D::set_brick_size(int bsize)
{
  xmax_ = ymax_ = zmax_ = bsize;
  int x,y,z;
  if (get_dimensions( x, y, z) ){
    if( texfld_->data_at() == Field::CELL ){
      isCC_=true;
      X_ = x-1;
      Y_ = y-1;
      Z_ = z-1;
    } else {
      isCC_=false;
      X_ = x;
      Y_ = y;
      Z_ = z;
    }    
  if( bontree_ ) delete bontree_;
  compute_tree_depth();
  build_texture();
  return true;
  } else {
    return false;
  }
}

bool
GLTexture3D::get_dimensions( int& nx, int& ny, int& nz)
{
  LevelMeshHandle mesh_;
  const string type = texfld_->get_type_name(1);
  if (type == "double") {
    LevelField<double> *fld =
      dynamic_cast<LevelField<double>*>(texfld_.get_rep());
    mesh_ = fld->get_typed_mesh();
  } else if (type == "float") {
    LevelField<float> *fld =
      dynamic_cast<LevelField<float>*>(texfld_.get_rep());
    mesh_ = fld->get_typed_mesh();
  } else if (type == "unsigned_int") {
    LevelField<long> *fld =
      dynamic_cast<LevelField<long>*>(texfld_.get_rep());
    mesh_ = fld->get_typed_mesh();
  } else {
    cerr << "GLTexture3D constructor error - unknown LevelMesh type: " <<
      type << endl;
    return false;
  }
  return SCIRun::GLTexture3D::get_dimensions( mesh_, nx, ny, nz );
}

} // End namespace Uintah

