/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
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

//    File   : UnitElementMesh.cc
//    Author : Frank B. Sachse, Martin Cole
//    Date   : 5 Dec 2005

// Standalone program for creating meshes with single unit element
// todo: output file name as parameter, support for scale factor basis types, dynamic compilation?,
// standard names for meshes and basis types.

#include <Core/Basis/Bases.h>

#include <Core/Datatypes/CurveMesh.h>
#include <Core/Datatypes/QuadSurfMesh.h>
#include <Core/Datatypes/TriSurfMesh.h>
#include <Core/Datatypes/TetVolMesh.h>
#include <Core/Datatypes/PrismVolMesh.h>
#include <Core/Datatypes/HexVolMesh.h>
#include <Core/Datatypes/GenericField.h>

#include <Core/Persistent/Pstreams.h>

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <assert.h>

using std::cerr;
using std::ifstream;
using std::endl;

using namespace SCIRun;

// !!!WARNING!!! The 'test_vector' is not (and SHOULD NOT be) used.
// It is here so that on the SGI, a vector of vector of Point will be
// instantiated and the compilation will go through... otherwise, the
// compiler can't find it (while deep in template instantiation code)
// and throws an error.  This most likely could (and should) be
// wrapped in #if SGI but that will have to wait until later.
vector< vector< SCIRun::Point > > test_vector;

template<typename MESH, typename FBASIS>
void create_unit_element_mesh()
{
  MESH *mesh = new MESH();

  typename MESH::Node::array_type n;
  typename MESH::basis_type mb;
  n.resize(mb.number_of_mesh_vertices());
  int domain_dimension=mb.domain_dimension();

  for(int i=0; i<mb.number_of_vertices(); i++) {
    Point p;
    p.x(mb.unit_vertices[i][0]);
    if (domain_dimension>1) p.y(mb.unit_vertices[i][1]);
    if (domain_dimension>2) p.z(mb.unit_vertices[i][2]);
    if ((unsigned)i<n.size()) {
      mesh->add_point(p);
      n[i]=i;
      if (mb.polynomial_order()==3) {
	vector<Point> d(3);
	d[0]=d[1]=d[2]=Point(0,0,0);
	d.resize(domain_dimension);
	mesh->get_basis().add_derivatives(d);
      }
    }
    else
      mesh->get_basis().add_node_value(p);
  }
  typename MESH::Elem::index_type ei=mesh->add_elem(n);
  assert(ei==0);

  typedef GenericField<MESH, FBASIS, vector<double> > FIELD;
  FIELD *field = scinew FIELD(mesh);
  field->fdata().clear();
  FBASIS f;

  int local_dimension_elem=(f.number_of_mesh_vertices() || !f.dofs() ? 0 : 1);
  int local_dimension_nodes=f.number_of_mesh_vertices();
  int local_dimension_add_nodes=f.number_of_vertices()-f.number_of_mesh_vertices();
  //  int local_dimension_derivatives=f.dofs()-local_dimension_nodes-local_dimension_add_nodes-local_dimension_elem;

  for(int i=0; i<f.dofs(); i++) {
    if (i<local_dimension_nodes+local_dimension_elem)
      field->fdata().push_back(1);
    else if (i<local_dimension_nodes+local_dimension_elem+local_dimension_add_nodes)
      field->get_basis().add_node_value(1);
    else {
      vector<double> d(3);
      d[0]=d[1]=d[2]=0;
      d.resize(domain_dimension);
      field->get_basis().add_derivatives(d);
    }
  }
  FieldHandle fH(field);
  TextPiostream out_stream("a.fld", Piostream::Write);
  Pio(out_stream, fH);

  //  cerr << "ao " << field->get_basis().size_node_values() << endl;
}


int
main(int argc, char **argv) 
{
  if (argc<=1) {
    cerr << argv[0] << "[-CurveMeshLinear][-CurveMeshQuadratic][-CurveMeshCubic]" << endl;
    cerr << "\t[-TriSurfMeshLinear][-TriSurfMeshQuadratic][-TriSurfMeshCubic]" << endl;
    cerr << "\t[-QuadSurfMeshLinear][-QuadSurfMeshQuadratic][-QuadSurfMeshCubic]" << endl;
    cerr << "\t[-TetVolMeshConstant][-TetVolMeshLinear][-TetVolMeshQuadratic][-TetVolMeshCubic]" << endl;
    cerr << "\t[-PrismVolMeshLinear][-PrismVolMeshQuadratic][-PrismVolMeshCubic]" << endl;
    cerr << "\t[-HexVolMeshLinear][-HexVolMeshQuadratic][-HexVolMeshCubic]" << endl;
    exit(-1);
  }

  try {
    for (int currArg = 1; currArg < argc; currArg++)   
      if (!strcmp(argv[currArg],"-CurveMeshLinear")) 
      create_unit_element_mesh<CurveMesh<CrvLinearLgn<Point> >, CrvLinearLgn<double> >();
    else if (!strcmp(argv[currArg],"-CurveMeshQuadratic")) 
      create_unit_element_mesh<CurveMesh<CrvQuadraticLgn<Point> >, CrvQuadraticLgn<double> >();
    else if (!strcmp(argv[currArg],"-CurveMeshCubic")) 
      create_unit_element_mesh<CurveMesh<CrvCubicHmt<Point> >, CrvCubicHmt<double> >();

    else if (!strcmp(argv[currArg],"-TriSurfMeshLinear")) 
      create_unit_element_mesh<TriSurfMesh<TriLinearLgn<Point> >, TriLinearLgn<double> >();
    else if (!strcmp(argv[currArg],"-TriSurfMeshQuadratic")) 
      create_unit_element_mesh<TriSurfMesh<TriQuadraticLgn<Point> >, TriQuadraticLgn<double> >();
    else if (!strcmp(argv[currArg],"-TriSurfMeshCubic")) 
      create_unit_element_mesh<TriSurfMesh<TriCubicHmt<Point> >, TriCubicHmt<double> >();

    else if (!strcmp(argv[currArg],"-QuadSurfMeshLinear")) 
      create_unit_element_mesh<QuadSurfMesh<QuadBilinearLgn<Point> >, QuadBilinearLgn<double> >();
    else if (!strcmp(argv[currArg],"-QuadSurfMeshQuadratic")) 
      create_unit_element_mesh<QuadSurfMesh<QuadBilinearLgn<Point> >, QuadBilinearLgn<double> >();
    else if (!strcmp(argv[currArg],"-QuadSurfMeshCubic")) 
      create_unit_element_mesh<QuadSurfMesh<QuadBicubicHmt<Point> >, QuadBicubicHmt<double> >();

    else if (!strcmp(argv[currArg],"-TetVolMeshConstant")) 
      create_unit_element_mesh<TetVolMesh<TetLinearLgn<Point> >, ConstantBasis<double> >();
        else if (!strcmp(argv[currArg],"-TetVolMeshLinear")) 
      create_unit_element_mesh<TetVolMesh<TetLinearLgn<Point> >, TetLinearLgn<double> >();
    else if (!strcmp(argv[currArg],"-TetVolMeshQuadratic")) 
      create_unit_element_mesh<TetVolMesh<TetQuadraticLgn<Point> >, TetQuadraticLgn<double> >();
    else if (!strcmp(argv[currArg],"-TetVolMeshCubic")) 
      create_unit_element_mesh<TetVolMesh<TetCubicHmt<Point> >, TetCubicHmt<double> >();
 
    else if (!strcmp(argv[currArg],"-PrismVolMeshConstant")) 
      create_unit_element_mesh<PrismVolMesh<PrismLinearLgn<Point> >, ConstantBasis<double> >();
    else if (!strcmp(argv[currArg],"-PrismVolMeshLinear")) 
      create_unit_element_mesh<PrismVolMesh<PrismLinearLgn<Point> >, PrismLinearLgn<double> >();
    else if (!strcmp(argv[currArg],"-PrismVolMeshQuadratic")) 
      create_unit_element_mesh<PrismVolMesh<PrismLinearLgn<Point> >, PrismQuadraticLgn<double> >();
    else if (!strcmp(argv[currArg],"-PrismVolMeshCubic")) 
      create_unit_element_mesh<PrismVolMesh<PrismCubicHmt<Point> >, PrismCubicHmt<double> >();
  
    else if (!strcmp(argv[currArg],"-HexVolMeshConstant")) 
      create_unit_element_mesh<HexVolMesh<HexTrilinearLgn<Point> >, ConstantBasis<double> >();
    else if (!strcmp(argv[currArg],"-HexVolMeshLinear")) 
      create_unit_element_mesh<HexVolMesh<HexTrilinearLgn<Point> >, HexTrilinearLgn<double> >();
    else if (!strcmp(argv[currArg],"-HexVolMeshQuadratic")) 
      create_unit_element_mesh<HexVolMesh<HexTriquadraticLgn<Point> >, HexTriquadraticLgn<double> >();
    else if (!strcmp(argv[currArg],"-HexVolMeshCubic")) 
      create_unit_element_mesh<HexVolMesh<HexTricubicHmt<Point> >, HexTricubicHmt<double> >();
  
    else
      cerr << argv[0] << ": Invalid argument " << argv[currArg] << endl;
  }

  catch(const Exception& e) {
    std::cerr << "Caught exception:\n";
    std::cerr << e.message() << std::endl;
    abort();
  }
  catch(...) {
    std::cerr << "Caught unexpected exception!\n";
    abort();
  }

  return 0;  
}    
