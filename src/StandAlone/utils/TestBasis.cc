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

//    File   : TestBasis.cc
//    Author : Frank B. Sachse
//    Date   : 13 OCT 2005

#include <iostream>
#include <fstream>
#include <stdlib.h>

#include <Core/Basis/CrvLinearLgn.h>
#include <Core/Basis/CrvQuadraticLgn.h>
#include <Core/Basis/CrvCubicHmt.h>
#include <Core/Basis/QuadBilinearLgn.h>
#include <Core/Basis/QuadBiquadraticLgn.h>
#include <Core/Basis/QuadBicubicHmt.h>
#include <Core/Basis/TriLinearLgn.h>
#include <Core/Basis/TriQuadraticLgn.h>
#include <Core/Basis/TriCubicHmt.h>
#include <Core/Basis/TriCubicHmtScaleFactors.h>
#include <Core/Basis/TetLinearLgn.h>
#include <Core/Basis/TetQuadraticLgn.h>
#include <Core/Basis/TetCubicHmt.h>
#include <Core/Basis/PrismLinearLgn.h>
#include <Core/Basis/PrismQuadraticLgn.h>
#include <Core/Basis/PrismCubicHmt.h>
#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Basis/HexTriquadraticLgn.h>
#include <Core/Basis/HexTricubicHmt.h>
#include <Core/Basis/HexTricubicHmtScaleFactors.h>


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

using std::cerr;
using std::ifstream;
using std::endl;

using namespace SCIRun;


template<class FIELD, class FBASIS>
double CrvIntegral(FIELD *field, FBASIS& f)
{ 
  vector<double> coords(1);

  typename FIELD::fdata_type &d = field->fdata();
  for(int i=0; i<f.number_of_vertices(); i++)
    d[i] = 1;

  double vol=0;
  for(int i=0; i<f.GaussianNum; i++) {
    double val;
    coords[0]=f.GaussianPoints[i][0];
    field->interpolate(val, coords, 0);
    vol+=f.GaussianWeights[i]*val;
  }
  return vol;
}


template<class FIELD, class FBASIS>
double FaceIntegral(FIELD *field, FBASIS& f)
{ 
  vector<double> coords(2);

  typename FIELD::fdata_type &d = field->fdata();
  for(int i=0; i<f.number_of_vertices(); i++)
    d[i] = 1;

  double vol=0;
  for(int i=0; i<f.GaussianNum; i++) {
    double val;
    coords[0]=f.GaussianPoints[i][0];
    coords[1]=f.GaussianPoints[i][1];
    field->interpolate(val, coords, 0);
    vol+=f.GaussianWeights[i]*val;
  }
  return vol;
}


template<class FIELD, class FBASIS>
double CellIntegral(FIELD *field, FBASIS& f)
{ 
  vector<double> coords(3);

  typename FIELD::fdata_type &d = field->fdata();
  for(int i=0; i<f.number_of_vertices(); i++)
    d[i] = 1;

  double vol=0;
  for(int i=0; i<f.GaussianNum; i++) {
    double val;
    coords[0]=f.GaussianPoints[i][0];
    coords[1]=f.GaussianPoints[i][1];
    coords[2]=f.GaussianPoints[i][2];
    field->interpolate(val, coords, 0);
    vol+=f.GaussianWeights[i]*val;
  }
  return vol;
}

template<typename MESH, typename FBASIS, const int nnode>
void Test1D()
{   
  MESH *mesh = new MESH();
  mesh->synchronize(MESH::EDGES_E);

  typename MESH::Node::array_type n;
  n.resize(nnode);
  typename MESH::basis_type u;

  for(int i=0; i<u.number_of_vertices(); i++) {
    Point p(u.unit_vertices[i][0]+1, u.unit_vertices[i][0]+2, u.unit_vertices[i][0]+3);
    if (i<n.size()) {
      mesh->add_point(p);
      n[i]=i;
    }
    else
      mesh->get_basis().add_node_value(p);
  }

  typename MESH::Elem::index_type ei=mesh->add_elem(n); 
  cerr<<"Element index: " << ei << "\n"; 

  vector<double> coords(1);
  coords[0]=drand48();
  Point p;
  
  mesh->interpolate(p, coords, 0);
  cerr << "Transform L->G " << coords[0] << " => " << p << endl;


  vector<double> lc(u.domain_dimension());
  bool rc=mesh->get_coords(lc, p, 0);
  cerr << "Transform G->L " << p << " => ";
  if (rc) 
    cerr << lc[0] << endl;
  else
    cerr << " not found" << endl;
  
  typedef GenericField<MESH, FBASIS, vector<double> > FIELD;
  FIELD *field = scinew FIELD(mesh);
  field->resize_fdata();

  FBASIS f;
  cerr << "Crv integral " << CrvIntegral(field, f) << endl; 
}

template<typename MESH, typename FBASIS, const int nnode>
void Test2D()
{
  MESH *mesh = new MESH();
  mesh->synchronize(MESH::EDGES_E);

  typename MESH::Node::array_type n;
  n.resize(3);
  typename MESH::basis_type u;
 
  for(int i=0; i<u.number_of_vertices(); i++) {
    Point p(u.unit_vertices[i][0]+1, u.unit_vertices[i][1]+2, 3);
    if (i<n.size()) {
      mesh->add_point(p);
      n[i]=i;
    }
    else
      mesh->get_basis().add_node_value(p);
  }

  typename MESH::Elem::index_type ei=mesh->add_elem(n); 
  cerr<<"Element index: " << ei << "\n"; 
  
  vector<double> coords(2);
  coords[0]=drand48();
  coords[1]=drand48();
  Point p;

  mesh->interpolate(p, coords, 0);

  cerr << "Transform L->G " << coords[0] <<", " << coords[1] << " => " << p << endl;

  vector<double> lc(u.domain_dimension());
  
  bool rc=mesh->get_coords(lc, p, 0);
  cerr << "Transform G->L " << p << " => ";
  if (rc) 
    cerr << lc[0] <<", " << lc[1] << endl;
  else
    cerr << " not found" << endl;
    
  typedef GenericField<MESH, FBASIS, vector<double> > FIELD;
  FIELD *field = scinew FIELD(mesh);
  field->resize_fdata();

  FBASIS f;
  cerr << "Face integral " << CellIntegral(field, f) << endl; 
}

template<typename MESH, typename FBASIS, const int nnode>
void Test3D()
{
  MESH *mesh = new MESH();
  mesh->synchronize(MESH::EDGES_E);

  typename MESH::Node::array_type n;
  n.resize(4);
  typename MESH::basis_type u;
 
  for(int i=0; i<u.number_of_vertices(); i++) {
    Point p(u.unit_vertices[i][0]+1, u.unit_vertices[i][1]+2, u.unit_vertices[i][2]+3);
    if (i<n.size()) {
      mesh->add_point(p);
      n[i]=i;
    }
    else
      mesh->get_basis().add_node_value(p);
  }
  typename  MESH::Elem::index_type ei=mesh->add_elem(n); 

  cerr<<"Element index: " << ei << "\n"; 
  
  vector<double> coords(3);
  coords[0]=drand48();
  coords[1]=drand48();
  coords[2]=drand48();
  Point p;

  mesh->interpolate(p, coords, 0);
 
  cerr << "Transform L->G " << coords[0] <<", " << coords[1] <<", " << coords[2] << " => " << p << endl;

  vector<double> lc(u.domain_dimension());
  
  bool rc=mesh->get_coords(lc, p, 0);
  cerr << "Transform G->L " << p << " => ";
  if (rc) 
    cerr << lc[0] <<", " << lc[1] <<", " << lc[2] << endl;
  else
    cerr << " not found" << endl;
    
  typedef GenericField<MESH, FBASIS, vector<double> > FIELD;
  FIELD *field = scinew FIELD(mesh);
  field->resize_fdata();

  FBASIS f;
  cerr << "Cell integral " << CellIntegral(field, f) << endl;  
}


int
main(int argc, char **argv) 
{
  {
    cerr<<"TestCrvMesh\n";
    
    srand48(0);
    Test1D<CurveMesh<CrvLinearLgn<Point> >, CrvLinearLgn<double>, 2 >();
    srand48(0);
    Test1D<CurveMesh<CrvQuadraticLgn<Point> >, CrvLinearLgn<double>, 2 >();
    srand48(0);
    Test1D<CurveMesh<CrvCubicHmt<Point> >, CrvLinearLgn<double>, 2 >(); 
   }

  {
    cerr<<"TestTriSurfMesh\n";
    
    srand48(0);
    Test2D<TriSurfMesh<TriLinearLgn<Point> >, TriLinearLgn<double>, 3 >();
    srand48(0);
    Test2D<TriSurfMesh<TriQuadraticLgn<Point> >, TriLinearLgn<double>, 3 >();
    srand48(0);
    Test2D<TriSurfMesh<TriCubicHmt<Point> >, TriLinearLgn<double>, 3 >();
    srand48(0);
    Test2D<TriSurfMesh<TriCubicHmtScaleFactors<Point> >, TriLinearLgn<double>, 3 >();
  }

  {
    cerr<<"TestQuadSurfMesh\n";
    
    srand48(0);
    Test2D<QuadSurfMesh<QuadBilinearLgn<Point> >, QuadBilinearLgn<double>, 4 >();
    srand48(0);
    Test2D<QuadSurfMesh<QuadBilinearLgn<Point> >, QuadBilinearLgn<double>, 4 >();
    srand48(0);
    Test2D<QuadSurfMesh<QuadBilinearLgn<Point> >, QuadBilinearLgn<double>, 4 >();
  }

  {
    cerr<<"TestTetVolMesh\n";
    
    srand48(0);
    Test3D<TetVolMesh<TetLinearLgn<Point> >, TetLinearLgn<double>, 4 >();
    srand48(0);
    Test3D<TetVolMesh<TetQuadraticLgn<Point> >, TetLinearLgn<double>, 4 >();
    srand48(0);
    Test3D<TetVolMesh<TetCubicHmt<Point> >, TetLinearLgn<double>, 4 >(); 
  }
  {
    cerr<<"TestPrismVolMesh\n";
    
    srand48(0);
    Test3D<PrismVolMesh<PrismLinearLgn<Point> >, PrismLinearLgn<double>, 6 >();
    srand48(0);
    Test3D<PrismVolMesh<PrismQuadraticLgn<Point> >, PrismLinearLgn<double>, 6 >();
    srand48(0);
    Test3D<PrismVolMesh<PrismCubicHmt<Point> >, PrismLinearLgn<double>, 6 >(); 
  }

  {
    cerr<<"TestHexVolMesh\n";
    
    srand48(0);
    Test3D<HexVolMesh<HexTrilinearLgn<Point> >, HexTrilinearLgn<double>, 8 >();
    srand48(0);
    Test3D<HexVolMesh<HexTriquadraticLgn<Point> >, HexTrilinearLgn<double>, 8 >();
    srand48(0);
    Test3D<HexVolMesh<HexTricubicHmt<Point> >, HexTrilinearLgn<double>, 8 >();  
    srand48(0);
    Test3D<HexVolMesh<HexTricubicHmtScaleFactors<Point> >, HexTrilinearLgn<double>, 8 >();    
  }

  return 0;  
}    
