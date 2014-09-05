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

#include <Core/Thread/Thread.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/Matrix.h>

#include <sgi_stl_warnings_off.h>
#include <string>
#include <vector>
#include <sgi_stl_warnings_on.h>

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
  virtual void search( double ival, bool field, bool geom ) = 0;
  virtual GeomHandle get_geom() = 0;
  virtual FieldHandle get_field(int n) = 0;
  virtual MatrixHandle get_interpolant(int n) = 0;

  //! support the dynamically compiled algorithm concept
  static const string& get_h_file_path();
  static CompileInfoHandle get_compile_info(const TypeDescription *td);
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
  GeomHandle geom_;
  vector<FieldHandle> output_field_;
  vector<MatrixHandle> output_matrix_;
public:
  MarchingCubes() : mesh_(0) { tess_.resize( np_, 0 ); }
  virtual ~MarchingCubes() { release(); }

  virtual void set_np( int );
  virtual void release();
  virtual void set_field( Field * );
  virtual void search( double ival, bool field, bool geom );
  virtual GeomHandle get_geom() { return geom_; } 
  virtual FieldHandle get_field(int n)
  { ASSERT((unsigned int)n < output_field_.size()); return output_field_[n]; }
  virtual MatrixHandle get_interpolant(int n)
  { ASSERT((unsigned int)n < output_matrix_.size()); return output_matrix_[n];}


  void parallel_search( int, double, bool field, bool geom);
};
    

    
// MarchingCubes

template<class Tesselator>
void 
MarchingCubes<Tesselator>::release() 
{
  for (int i=0; i<np_; i++)
    if ( tess_[i] ) { delete tess_[i]; tess_[i] = 0; }
  if (geom_.get_rep()) geom_=0;
}

template<class Tesselator>
void 
MarchingCubes<Tesselator>::set_np( int np )
{
  if ( np > np_ ) 
    tess_.resize( np, 0 );
  if ( geom_.get_rep() ) 
    geom_ = 0;
  
  output_field_.resize(np);
  output_matrix_.resize(np);

  np_ = np;
}  

template<class Tesselator>
void 
MarchingCubes<Tesselator>::set_field( Field *f )
{
  if ( field_type *field = dynamic_cast<field_type *>(f) ) {
    for (int i=0; i<np_; i++) {
      if ( tess_[i] ) delete tess_[i];
      tess_[i] = scinew Tesselator( field );
    }
    mesh_ = field->get_typed_mesh();
  }
}

template<class Tesselator>
void
MarchingCubes<Tesselator>::search( double iso, bool bf, bool bg )
{
  if ( np_ == 1 ) {
    tess_[0]->reset(0, bf, bg);
    typename mesh_type::Elem::iterator cell; mesh_->begin(cell); 
    typename mesh_type::Elem::iterator cell_end; mesh_->end(cell_end); 
    while ( cell != cell_end)
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
		      iso, bf, bg );
    GeomGroup *group = scinew GeomGroup;
    for (int i=0; i<np_; i++) {
      GeomHandle obj = tess_[i]->get_geom();
      if ( obj.get_rep() ) 
	group->add( obj );
    }
    if ( group->size() > 0 )
      geom_ = group;
    else {
      delete group;
      geom_ = 0;
    }
  }
  output_field_.resize(np_);
  output_matrix_.resize(np_);
  for (int i = 0; i < np_; i++)
  {
    output_field_[i] = tess_[i]->get_field(iso);
    output_matrix_[i] = tess_[i]->get_interpolant();
  }
}

template<class Tesselator>
void 
MarchingCubes<Tesselator>::parallel_search( int proc,
					    double iso, bool bf, bool bg )
{
  tess_[proc]->reset(0, bf, bg);
  typename mesh_type::Elem::size_type csize;
  mesh_->size(csize);
  unsigned int n = csize;
  
  typename mesh_type::Elem::iterator from; mesh_->begin(from);
  unsigned int i;
  for ( i=0; i<(proc*(n/np_)); i++) { ++from; }
  
  for ( unsigned int last = (proc < np_-1) ? (proc+1)*(n/np_) : n; i<last; i++)
  {
    tess_[proc]->extract( *from, iso );
    ++from;
  }
}

} // End namespace SCIRun

#endif // Visualization_MarchingCubes_h
