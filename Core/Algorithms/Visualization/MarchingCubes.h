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

  virtual GeomObj* search( double ) = 0;
};

  
template < class AI, class Tesselator, class Mesh >
class MarchingCubes : public MarchingCubesAlg
{
  typedef LockingHandle<Mesh> MeshHandle;

protected:
  AI *ai_;
  Tesselator *tess_;
  MeshHandle mesh_;
public:
  MarchingCubes( AI *ai, Tesselator *tess, MeshHandle mesh) 
    : ai_(ai), tess_(tess), mesh_(mesh) {}
  virtual ~MarchingCubes() {}

  virtual GeomObj* search( double );
};
    

    
// MarchingCubes

template <class AI, class Tesselator, class Mesh>
GeomObj *MarchingCubes<AI,Tesselator, Mesh>::search( double iso ) 
{
  tess_->reset(0);
  typename Mesh::cell_iterator cell = mesh_->cell_begin(); 
  while ( cell != mesh_->cell_end() )
  {
    tess_->extract( *cell, iso );
    ++cell;
  }
  return tess_->get_geom();
}


template<class AI, class Field> MarchingCubesAlg* 
make_tet_mc_alg(AI *ai, Field *field)
{
  typedef typename Field::mesh_type Mesh;

  TetMC<Field> *tmc = new TetMC<Field>(field);
  MarchingCubesAlg *alg 
    = new MarchingCubes<AI, TetMC<Field>, Mesh>( ai, tmc, 
						 field->get_typed_mesh());
  return alg;
}

template<class AI, class Field> MarchingCubesAlg* 
make_lattice_mc_alg(AI *ai, Field *field)
{
  typedef typename Field::mesh_type Mesh;

  HexMC<Field> *tmc = new HexMC<Field>(field);
  MarchingCubesAlg *alg 
    = new MarchingCubes<AI, HexMC<Field>, Mesh>( ai, tmc, 
						 field->get_typed_mesh());
  return alg;
}
} // End namespace SCIRun

#endif
