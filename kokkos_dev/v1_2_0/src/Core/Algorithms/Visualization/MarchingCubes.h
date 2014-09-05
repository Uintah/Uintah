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
//    File   : MarchingCubes.h
//    Author : Yarden Livnat
//    Date   : Fri Jun 15 16:20:11 2001

#if !defined(Visualization_MarchingCubes_h)
#define Visualization_MarchingCubes_h

#include <iostream>
#include <string>
#include <vector>

#include <Core/Thread/Thread.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/GeomObj.h>
#include <Core/Disclosure/DynamicLoader.h>
#include <Core/Datatypes/TriSurf.h>

namespace SCIRun {
class Field;

class MarchingCubesAlg : public DynamicAlgoBase {
protected:
  int np_;
public:

  MarchingCubesAlg();
  virtual ~MarchingCubesAlg();

  virtual void set_np( int np ) { np_ = np; }
  virtual void release() = 0;
  virtual void set_field( Field * ) = 0;
  virtual void search( double, bool ) = 0;
  virtual GeomObj* get_geom() = 0;
  virtual TriSurfMeshHandle get_trisurf() = 0;

  //! support the dynamically compiled algorithm concept
  static const string& get_h_file_path();
  static CompileInfo *get_compile_info(const TypeDescription *td);
};

  
template <class Tesselator>
class MarchingCubes : public MarchingCubesAlg
{
  typedef typename Tesselator::field_type       field_type;
  typedef typename Tesselator::mesh_type        mesh_type;
  typedef typename Tesselator::mesh_handle_type mesh_handle_type;

protected:
  vector<Tesselator *> tess_;
  mesh_handle_type mesh_;
  GeomObj *geom_;
public:
  MarchingCubes() : mesh_(0) { tess_.resize( np_, 0 ); }
  virtual ~MarchingCubes() {}

  virtual void set_np( int );
  virtual void release();
  virtual void set_field( Field * );
  virtual void search( double, bool=false );
  virtual GeomObj* get_geom() { return geom_; } 
  virtual TriSurfMeshHandle get_trisurf() { return TriSurfMeshHandle(0); }

  void parallel_search( int, double, bool );
};
    

    
// MarchingCubes

template<class Tesselator>
void 
MarchingCubes<Tesselator>::release() 
{
  for (int i=0; i<np_; i++)
    if ( tess_[i] ) { delete tess_[i]; tess_[i] = 0; }
  if (geom_) geom_=0;
}

template<class Tesselator>
void 
MarchingCubes<Tesselator>::set_np( int np )
{
  if ( np > np_ ) 
    tess_.resize( np, 0 );
  if ( geom_ ) 
    geom_ = 0;

  np_ = np;
}  

template<class Tesselator>
void 
MarchingCubes<Tesselator>::set_field( Field *f )
{
  if ( field_type *field = dynamic_cast<field_type *>(f) ) {
    for (int i=0; i<np_; i++) {
      if ( tess_[i] ) delete tess_[i];
      tess_[i] = new Tesselator( field );
    }
    mesh_ = field->get_typed_mesh();
  }
}

template<class Tesselator>
void
MarchingCubes<Tesselator>::search( double iso, bool build_trisurf )
{
  if ( np_ == 1 ) {
    tess_[0]->reset(0, build_trisurf);
    typename mesh_type::Cell::iterator cell = mesh_->cell_begin(); 
    while ( cell != mesh_->cell_end() )
    {
      tess_[0]->extract( *cell, iso );
      ++cell;
    }
    geom_ = tess_[0]->get_geom();
  }
  else {
    Thread::parallel( this,  
		      &MarchingCubes<Tesselator>::parallel_search, 
		      np_, 
		      true,    // block
		      iso, 
		      false ); // for now build_trisurf is off for parallel mc
    GeomGroup *group = new GeomGroup;
    for (int i=0; i<np_; i++) {
      GeomObj *obj = tess_[i]->get_geom();
      if ( obj ) 
	group->add( obj );
    }
    if ( group->size() > 0 )
      geom_ = group;
    else {
      delete group;
      geom_ = 0;
    }
  }
}

template<class Tesselator>
void 
MarchingCubes<Tesselator>::parallel_search( int proc, 
					       double iso, bool build_trisurf)
{
  tess_[proc]->reset(0, build_trisurf);
  int n = mesh_->cells_size();
  
  typename mesh_type::Cell::iterator from = mesh_->cell_begin();
  int i;
  for ( i=0; i<(proc*(n/np_)); i++) ++from;
  
  for ( int last = (proc < np_-1) ? (proc+1)*(n/np_) : n; i<last; i++) {
    tess_[proc]->extract( *from, iso );
    ++from;
  }
}

} // End namespace SCIRun

#endif // Visualization_MarchingCubes_h
