/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/

//    File   : InterfaceWithCamal.cc
//    Author : Jason Shepherd
//    Date   : Dec 8 2004

#include <Core/Thread/Mutex.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Util/ProgressReporter.h>
#include <Core/Datatypes/Field.h>
#include <Core/Basis/NoData.h>
#include <Core/Basis/TetLinearLgn.h>
#include <Core/Basis/TriLinearLgn.h>
#include <Core/Datatypes/TetVolMesh.h>
#include <Core/Datatypes/TriSurfMesh.h>
#include <Core/Datatypes/GenericField.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <CMLTetMesher.hpp>

namespace SCIRun {

using std::cerr;
using std::endl;

// Mutes for guarding the tetmesher: CAMAL is not threadsafe
SCIRun::Mutex InterfaceWithCamalMutex("InterfaceWithCamalMutex");

class InterfaceWithCamal : public Module
{
public:
  InterfaceWithCamal(GuiContext* ctx);
  virtual ~InterfaceWithCamal();

  virtual void execute();

private:
  bool read_tri_file(int &npoints, double *&points, int &ntris, int *&tris);
  bool write_tet_file(const int &npoints, double* const points,
                    const int &ntets, int* const tets, 
                    const int &noldpoints, double* const oldpoints);
};


DECLARE_MAKER(InterfaceWithCamal)
  
InterfaceWithCamal::InterfaceWithCamal(GuiContext* ctx)
  : Module("InterfaceWithCamal", ctx, Filter, "NewField", "SCIRun")
{ 
}


InterfaceWithCamal::~InterfaceWithCamal()
{
}


void
InterfaceWithCamal::execute() 
{
  
  bool ret_value = true;
  
    // read input file
  int   *tris   = NULL; // array allocated in read_tri_file
  int   *tets   = NULL;
  double *points = NULL; // array allocated in read_tri_file
  double *tetpoints = NULL;
  int num_tris = 0, num_points = 0;

  ret_value = read_tri_file(num_points, points, num_tris, tris);
  if (!ret_value) 
  {
    // printf("Failed read input\n");
    error("Failed read input.");
  }

  // FIX: Camal is NOT THreadsafe hence we need to make it thread safe by adding a mutex
  // Lock the tetmesher so no other module can address the tetmesher at the same time
  InterfaceWithCamalMutex.lock();

    // mesh the volume

  int new_points = 0, num_tets = 0;
  if (ret_value) 
  {
    CMLTetMesher tet_mesher;
    ret_value = tet_mesher.set_boundary_mesh(num_points, points,
                                             num_tris, tris);
    if (!ret_value) 
    {
      // printf("Failed setting boundary mesh\n");
      error("Failed setting boundary mesh.");
    }

      // generate the mesh
    if (ret_value) 
    {
      ret_value = tet_mesher.generate_mesh(new_points, num_tets);
      if (!ret_value) 
      {
        // printf("Failed generating mesh\n");
        error("Failed generating mesh.");
      }
    }

      // allocate memory to accept tet mesh and retrieve it
    if (ret_value) 
    {
      tets   = new int [num_tets * 4];
      tetpoints = new double [new_points * 3];
      ret_value = tet_mesher.get_mesh(new_points, tetpoints, num_tets, tets);
      if (!ret_value) {
        // printf("Failed reading tet mesh\n");
        error("Failed reading tet mesh.");
      }
    }
  }
  
    // write output file
  if (ret_value) 
  {    
    ret_value = write_tet_file(new_points, tetpoints, num_tets, tets, num_points, points);
    if (!ret_value) 
    {
      // printf("Failed writing tet mesh file\n");
      error("Failed writing tet mesh file.");
    }
  }

    // delete remaining arrays
  delete [] tetpoints;  
  delete [] points;
  delete [] tets;
  delete [] tris;
  
  // Unlock the InterfaceWithCamal, so the nextr module can use it
  InterfaceWithCamalMutex.unlock();
}


bool
InterfaceWithCamal::read_tri_file(int &npoints, double *&points, int &ntris, int *&tris)
{
  FieldHandle trisurfH;
  if (!get_input_handle("TriSurf", trisurfH)) return false;
  
  TriSurfMesh<TriLinearLgn<Point> > *tsm = dynamic_cast<TriSurfMesh<TriLinearLgn<Point> >* >(trisurfH->mesh().get_rep());
  if (!tsm)
  {
    error("Currently only Linear TriSurf Meshes are supported.");
    return(false);
  }
  
  TriSurfMesh<TriLinearLgn<Point> >::Node::iterator niter; 
  TriSurfMesh<TriLinearLgn<Point> >::Node::iterator niter_end; 
  TriSurfMesh<TriLinearLgn<Point> >::Node::size_type nsize; 
  tsm->begin(niter);
  tsm->end(niter_end);
  tsm->size(nsize);

  TriSurfMesh<TriLinearLgn<Point> >::Face::size_type fsize; 
  TriSurfMesh<TriLinearLgn<Point> >::Face::iterator fiter; 
  TriSurfMesh<TriLinearLgn<Point> >::Face::iterator fiter_end; 
  TriSurfMesh<TriLinearLgn<Point> >::Node::array_type fac_nodes(3);
  tsm->size(fsize);
  tsm->begin(fiter);
  tsm->end(fiter_end);

  npoints = nsize;
  ntris = fsize;

  int counter=0;
  
  points = new double[npoints * 3];
  tris   = new int[ntris * 3];
  if (points == NULL || tris == NULL) 
  {
    npoints = 0;
    ntris   = 0;
    return false;
  }

  while(niter != niter_end) 
  {
    Point p;
    tsm->get_center(p, *niter);
    points[counter*3]=p.x();
    points[counter*3+1]=p.y();
    points[counter*3+2]=p.z();
    ++niter;
    counter++;
  }

  counter=0;
  while(fiter != fiter_end) {
    tsm->get_nodes(fac_nodes, *fiter);
    tris[counter*3]=(int)fac_nodes[0];
    tris[counter*3+1]=(int)fac_nodes[1];
    tris[counter*3+2]=(int)fac_nodes[2];
    ++fiter;
    counter++;
  }

  return true;
}


bool
InterfaceWithCamal::write_tet_file(const int &npoints, double* const points, 
                          const int &ntets, int* const tets,
                          const int &noldpoints, double* const oldpoints)
{
  typedef TetVolMesh<TetLinearLgn<Point> > my_tetvolmesh;
  typedef GenericField<TetVolMesh<TetLinearLgn<Point> >, NoDataBasis<double>, vector<double> > my_tetvolfield;

  int i;
  my_tetvolmesh *tvm = new my_tetvolmesh();
  // THis code should fix the following problems:
  // (1) The number of nodes in a mesh should be reserved before inserting them
  //       Can SOMEONE clean up the Field classes so it will become clear to people 
  //       not familiar with the mehs classes
  // (2) Somehow Camal uses a different numeric format inside hence nodes in fixed boundaries
  //     end up somewhere else it is close to the original position but not exactly, which is
  //     annoying if you want to merge different meshes together.
  // The following lines of code fix the problem            
  
  tvm->node_reserve(npoints);
  for ( i=0; i<noldpoints; i++)
  {
    tvm->add_point( Point( oldpoints[i*3], oldpoints[i*3+1], oldpoints[i*3+2] ));    
  }
  
  for ( i=noldpoints; i<npoints; i++) 
  {
    tvm->add_point( Point( points[i*3], points[i*3+1], points[i*3+2] ));
  }

  for (i=0; i<ntets; i++) 
  {
    tvm->add_tet( tets[i*4], tets[i*4+1], tets[i*4+2], tets[i*4+3] );
  }
  
  my_tetvolfield* tvf = scinew my_tetvolfield(tvm);
  FieldHandle tvH = dynamic_cast<Field*>(tvf);

  cout << "Finished loading " << ntets << " tetrahedrons.\n\n";

  send_output_handle("TetVol", tvH);

  return true;
}


} // end namespace SCIRun

