/*!
 *  MarchingCubes.h
 *      Isosurface extraction based on Marching Cubes
 *
 *   \author Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   \date Feb 2001
 *
 *  Copyright (C) 2001 SCI Group
 */


#ifndef MarchingCubes_h
#define MarchingCubes_h

#include <stdio.h>
#include <iostream>

#include <Core/Geom/GeomObj.h>

namespace SCIRun {

class MarchingCubesAlg {
public:
  MarchingCubesAlg() {}
  virtual ~MarchingCubesAlg() {}

  virtual void release() = 0;
  virtual void set_field( Field * ) = 0;
  virtual GeomObj* search( double ) = 0;
};

  
template < class AI, class Tesselator>
class MarchingCubes : public MarchingCubesAlg
{
  typedef typename Tesselator::field_type       field_type;
  typedef typename Tesselator::mesh_type        mesh_type;
  typedef typename Tesselator::mesh_handle_type mesh_handle_type;

protected:
  AI *ai_;
  Tesselator *tess_;
  mesh_handle_type mesh_;
public:
  MarchingCubes() {}
  MarchingCubes( AI *ai ) : ai_(ai), tess_(0), mesh_(0) {}
  virtual ~MarchingCubes() {}

  virtual void release();
  virtual void set_field( Field * );
  virtual GeomObj* search( double );
};
    

    
// MarchingCubes

template<class AI, class Tesselator>
void 
MarchingCubes<AI,Tesselator>::release() 
{
  if ( tess_ ) { delete tess_; tess_ = 0; }
}

template<class AI, class Tesselator>
void 
MarchingCubes<AI,Tesselator>::set_field( Field *f )
{
  if ( field_type *field = dynamic_cast<field_type *>(f) ) {
    if ( tess_ ) delete tess_;
    tess_ = new Tesselator( field );
    mesh_ = field->get_typed_mesh();
  }
}

template<class AI, class Tesselator>
GeomObj *
MarchingCubes<AI,Tesselator>::search( double iso ) 
{
  ASSERT(tess_ != 0);

  tess_->reset(0);
  typename mesh_type::cell_iterator cell = mesh_->cell_begin(); 
  while ( cell != mesh_->cell_end() )
  {
    tess_->extract( *cell, iso );
    ++cell;
  }
  return tess_->get_geom();
}


} // End namespace SCIRun

#endif
