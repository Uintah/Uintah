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

#include <iostream>
#include <string>

#include <Core/Geom/GeomObj.h>

namespace SCIRun {

class MarchingCubesAlg {
public:

  MarchingCubesAlg() {}
  virtual ~MarchingCubesAlg() {}

  virtual void release() = 0;
  virtual void set_field( Field * ) = 0;
  virtual void search( double, bool ) = 0;
  virtual GeomObj* get_geom() = 0;
  virtual TriSurfMeshHandle get_trisurf() = 0;
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
  MarchingCubes(AI *ai) : ai_(ai), tess_(0), mesh_(0) {}
  virtual ~MarchingCubes() {}

  virtual void release();
  virtual void set_field( Field * );
  virtual void search( double, bool=false );
  virtual GeomObj* get_geom() { 
    if (tess_) return tess_->get_geom(); else return 0; }
  virtual TriSurfMeshHandle get_trisurf() { 
    if (tess_) return tess_->get_trisurf(); else return 0; }
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
void
MarchingCubes<AI,Tesselator>::search( double iso, bool build_trisurf )
{
  ASSERT(tess_ != 0);

  tess_->reset(0, build_trisurf);
  typename mesh_type::cell_iterator cell = mesh_->cell_begin(); 
  while ( cell != mesh_->cell_end() )
  {
    tess_->extract( *cell, iso );
    ++cell;
  }
}

} // End namespace SCIRun

#endif // MarchingCubes_h
