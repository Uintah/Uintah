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

//    File   : TestBasis.cc
//    Author : Frank B. Sachse
//    Date   : 13 OCT 2005

#include <iostream>
#include <fstream>
#include <stdlib.h>

#include <Core/Basis/Bases.h>

#include <Core/Geometry/Point.h>

#include <Core/Datatypes/CurveMesh.h>
#include <Core/Datatypes/QuadSurfMesh.h>
#include <Core/Datatypes/TriSurfMesh.h>
#include <Core/Datatypes/TetVolMesh.h>
#include <Core/Datatypes/PrismVolMesh.h>
#include <Core/Datatypes/HexVolMesh.h>
#include <Core/Datatypes/GenericField.h>

// Standalone program for testing meshes, fields and basis classes
// Strategy: Create mesh with single element
//           Transform local to global 
//           Transform global to local
//           Integrate constant value over volume 

using std::cout;
using std::ifstream;
using std::endl;

#ifdef _WIN32
// this is terrible, I know...
#define srand48 srand
#define drand48 rand
#endif

using namespace SCIRun;

// !!!WARNING!!! The 'test_vector' is not (and SHOULD NOT be) used.
// It is here so that on the SGI, a vector of vector of Point will be
// instantiated and the compilation will go through... otherwise, the
// compiler can't find it (while deep in template instantiation code)
// and throws an error.  This most likely could (and should) be
// wrapped in #if SGI but that will have to wait until later.
vector< vector< SCIRun::Point > > test_vector;


template<class FIELD, class FBASIS>
double CrvIntegral(FIELD *field, FBASIS& f)
{ 
  vector<double> coords(1);

  double vol=0;
  for(int i=0; i<f.GaussianNum; i++) {
    double val;
    coords[0]=f.GaussianPoints[i][0];
    field->interpolate(val, coords, typename FIELD::mesh_type::Elem::index_type(0));
    vol+=f.GaussianWeights[i]*val;
  }
  return vol;
}


template<class FIELD, class FBASIS>
double FaceIntegral(FIELD *field, FBASIS& f)
{ 
  vector<double> coords(2);

  double vol=0;
  for(int i=0; i<f.GaussianNum; i++) {
    double val;
    coords[0]=f.GaussianPoints[i][0];
    coords[1]=f.GaussianPoints[i][1];
    field->interpolate(val, coords, typename FIELD::mesh_type::Elem::index_type(0));
    vol+=f.GaussianWeights[i]*val;
  }
  return vol;
}


template<class FIELD, class FBASIS>
double CellIntegral(FIELD *field, FBASIS& f)
{ 
  vector<double> coords(3);

  double vol=0;
  for(int i=0; i<f.GaussianNum; i++) {
    double val;
    coords[0]=f.GaussianPoints[i][0];
    coords[1]=f.GaussianPoints[i][1];
    coords[2]=f.GaussianPoints[i][2];
    field->interpolate(val, coords, typename FIELD::mesh_type::Elem::index_type(0));
    vol+=f.GaussianWeights[i]*val;
  }
  return vol;
}

typedef TriSurfMesh<TriCubicHmtScaleFactors<Point> > TSFMesh;
typedef HexVolMesh<HexTricubicHmtScaleFactors<Point> > HVSFMesh;
typedef HexVolMesh<HexTricubicHmtScaleFactorsEdges<Point> > HVSFEMesh;


template <class MSH>
void add_extra_basis_data(MSH *mesh);

template <>
void add_extra_basis_data(TSFMesh *mesh)
{
  // xy scale factor per elem.
  TSFMesh::Elem::size_type sz;
  mesh->size(sz);
  for (unsigned i = 0; i < sz; i++) {
    vector<double> sf(2, 1.0L);
    mesh->get_basis().add_scalefactors(sf);
  }
}

template <>
void add_extra_basis_data(HVSFMesh *mesh)
{
  // xyz scale factor per node.
  HVSFMesh::Node::size_type sz;
  mesh->size(sz);
  for (unsigned i = 0; i < sz; i++) {
    vector<double> sf(3, 1.0L);
    mesh->get_basis().add_scalefactors(sf);
  }
}

template <>
void add_extra_basis_data(HVSFEMesh *mesh)
{
  // single scale factor per edge.
  HVSFEMesh::basis_type bas;
  unsigned sz = bas.number_of_edges();
  for (unsigned i = 0; i < sz; i++) {
    vector<double> sf(1, 1.0L);
    mesh->get_basis().add_scalefactors(sf);
  }
}

template<class MSH>
void add_extra_basis_data(MSH *mesh) 
{
}

template<typename MESH, typename FBASIS>
void Test()
{   
  MESH *mesh = new MESH();
  const TypeDescription *ftd = mesh->get_type_description();
  cout << "Test: " << ftd->get_name() << endl;

  //Transform t;
  //t.rotate(Vector(1,0,0), Vector(0,1,0));
  //t.pre_scale(Vector(1,2,3));
  //  t.print();
  typedef typename MESH::basis_type MBASIS;
  typename MESH::Node::array_type n;
  n.resize(MBASIS::number_of_mesh_vertices());

  for(int i=0; i<MBASIS::number_of_vertices(); i++) 
  {
    Point p;
    switch(MBASIS::domain_dimension()) 
    {
    case 1:
      p=Point(MBASIS::unit_vertices[i][0], 0, 0);
      break;
    case 2:
      p=Point(MBASIS::unit_vertices[i][0], MBASIS::unit_vertices[i][1], 0);
      break;
    case 3:
      p=Point(MBASIS::unit_vertices[i][0], MBASIS::unit_vertices[i][1], 
	      MBASIS::unit_vertices[i][2]);
      break;
    default:
      ASSERTFAIL("unknown dimension");
    }

    if ((unsigned)i<n.size()) 
    {
      mesh->add_point(p);
      n[i]=i;
      if (MBASIS::polynomial_order()==3) {
	// number of derivative values is dofs / num verts - 1;
	// putting too many in does not harm tests...
	unsigned sz = MBASIS::dofs() / MBASIS::number_of_vertices();
	cerr << "size of deriv array: " << sz << endl;
	vector<Point> d(sz);
	for(unsigned int i=0; i < d.size(); i++) {
	  d[i].x(0.0);
	  d[i].y(0.0);
	  d[i].z(0.0);
	}
	mesh->get_basis().add_derivatives(d);
      }
    }
    else
      mesh->get_basis().add_node_value(p);
  }
  
  add_extra_basis_data(mesh);

  typename MESH::Elem::index_type e0=mesh->add_elem(n); 
  ASSERT(e0==0);
  //  cout<<"Element index: " << e0 << "\n"; 
  mesh->synchronize(MESH::EDGES_E);

  vector<double> coords(MBASIS::domain_dimension());
  for(unsigned int i=0; i<coords.size();i++)
    coords[i]=drand48();
  Point p;
  
  mesh->interpolate(p, coords, 0);
  cout << "Transform L->G ";
  for(unsigned int i=0; i<coords.size();i++)
    cout << coords[0];
  cout << " => " << p << endl;

  vector<double> lc(MBASIS::domain_dimension());
  bool rc=mesh->get_coords(lc, p, 0);
  cout << "Transform G->L " << p << " => ";
  if (rc) {
    for(unsigned int i=0; i<coords.size();i++)
      cout << lc[0];
    cout << endl;
  }
  else
    cout << " not found" << endl;

  typename MESH::ElemData cmcd(*mesh, e0);
  for(int i=0; i<MBASIS::number_of_edges(); i++)
    cout << "Edge " << i << " length " << mesh->get_basis().get_arc_length(i, cmcd) << endl;

  for(int i=0; i<MBASIS::faces_of_cell(); i++)
    cout << "Face " << i << " area " << mesh->get_basis().get_area(i, cmcd) << endl;


  typedef GenericField<MESH, FBASIS, vector<double> > FIELD;
  FIELD *field = scinew FIELD(mesh);
  field->resize_fdata();
  FBASIS f;
  typename FIELD::fdata_type &d = field->fdata();
  for(int i=0; i<f.number_of_vertices(); i++)
    d[i] = 1;

  switch(MBASIS::domain_dimension()) {
  case 1:
    cout << "Crv integral " << CrvIntegral(field, f) << endl; 
    break;
  case 2:
    cout << "Face integral " << CellIntegral(field, f) << endl; 
    break;
  case 3:
    cout << "Cell integral " << CellIntegral(field, f) << endl;  
    break;
  default:
    ASSERTFAIL("unknown dimension");
  }

  delete field;
}


int
main(int argc, char **argv) 
{
  {
    srand48(0);
    Test<CurveMesh<CrvLinearLgn<Point> >, CrvLinearLgn<double> >();
    srand48(0);
    Test<CurveMesh<CrvQuadraticLgn<Point> >, CrvLinearLgn<double> >();
    srand48(0);
    Test<CurveMesh<CrvCubicHmt<Point> >, CrvLinearLgn<double> >(); 
   }

  {
    srand48(0);
    Test<TriSurfMesh<TriLinearLgn<Point> >, TriLinearLgn<double> >();
    srand48(0);
    Test<TriSurfMesh<TriQuadraticLgn<Point> >, TriLinearLgn<double> >();
    srand48(0);
    Test<TriSurfMesh<TriCubicHmt<Point> >, TriLinearLgn<double> >();
    
    //The following basis test crashes.
    //srand48(0);
    //Test<TriSurfMesh<TriCubicHmtScaleFactors<Point> >,TriLinearLgn<double> >();
  }

  {
    srand48(0);
    Test<QuadSurfMesh<QuadBilinearLgn<Point> >, QuadBilinearLgn<double> >();
    srand48(0);
    Test<QuadSurfMesh<QuadBiquadraticLgn<Point> >, QuadBilinearLgn<double> >();
    srand48(0);
    Test<QuadSurfMesh<QuadBicubicHmt<Point> >, QuadBilinearLgn<double> >();
  }
 
  {
    srand48(0);
    //    for(int i=0; i<1000; i++)
    Test<TetVolMesh<TetLinearLgn<Point> >, TetLinearLgn<double> >();
    srand48(0);
    Test<TetVolMesh<TetQuadraticLgn<Point> >, TetLinearLgn<double> >();
    srand48(0);
    Test<TetVolMesh<TetCubicHmt<Point> >, TetLinearLgn<double> >(); 
  }

  {
    srand48(0);
    Test<PrismVolMesh<PrismLinearLgn<Point> >, PrismLinearLgn<double> >();
    srand48(0);
    Test<PrismVolMesh<PrismQuadraticLgn<Point> >, PrismLinearLgn<double> >();
    srand48(0);
    //Test<PrismVolMesh<PrismCubicHmt<Point> >, PrismLinearLgn<double> >(); 
  }

  {
    srand48(0);
    Test<HexVolMesh<HexTrilinearLgn<Point> >, HexTrilinearLgn<double> >();
    srand48(0);
    Test<HexVolMesh<HexTriquadraticLgn<Point> >, HexTrilinearLgn<double> >();
    srand48(0);
    Test<HexVolMesh<HexTricubicHmt<Point> >, HexTrilinearLgn<double> >();  
    srand48(0);
    Test<HexVolMesh<HexTricubicHmtScaleFactors<Point> >, HexTrilinearLgn<double> >();   
    srand48(0);
    Test<HexVolMesh<HexTricubicHmtScaleFactorsEdges<Point> >, HexTrilinearLgn<double> >();    
  }
  
  return 0;  
}    
