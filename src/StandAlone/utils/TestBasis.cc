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
#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Basis/HexTriquadraticLgn.h>
#include <Core/Basis/HexTricubicHmt.h>
#include <Core/Basis/HexTricubicHmtScaleFactors.h>


#include <Core/Datatypes/CurveMesh.h>
#include <Core/Datatypes/QuadSurfMesh.h>
#include <Core/Datatypes/TetVolMesh.h>
#include <Core/Datatypes/GenericField.h>



using std::cerr;
using std::ifstream;
using std::endl;

using namespace SCIRun;

void TestCrvLinearLgn()
{
  cerr<<"TestCrvMeshLinearLgn\n";

  typedef CrvLinearLgn<Point> MBASIS;
  typedef CurveMesh<MBASIS > MESH;
  MESH *mesh = new MESH();

  MBASIS u;
  MESH::Node::array_type n;
  n.resize(u.number_of_vertices());

  for(int i=0; i<u.number_of_vertices(); i++) {
    Point p(u.unit_vertices[i][0]+1, u.unit_vertices[i][0]+2, u.unit_vertices[i][0]+3);
    mesh->add_point(p);
    n[i]=i;
  }
  MESH::Elem::index_type ei=mesh->add_elem(n); 

  cerr<<"Element index: " << ei << "\n"; 

  vector<double> coords;
  coords.push_back(0.2);
  Point p;
  
  mesh->interpolate(p, coords, 0);

  //  if (sqrt(pow(p.x()-coords[0],2.0)+pow(p.y()-coords[1],2.0)+pow(p.z()-coords[1],2.0))>1e-7)    
   cerr << "Transform L->G " << coords[0] << " => " << p << endl;

  vector<double> lc(u.domain_dimension());
  bool rc=mesh->get_coords(lc, p, 0);
  cerr << "Transform G->L " << p << " => ";
  if (rc) 
    cerr << lc[0] << endl;
  else
    cerr << " not found" << endl;
  
  typedef CrvLinearLgn<double>  FBASIS;
  typedef GenericField<MESH, FBASIS, vector<double> > FIELD;
  FIELD *field = scinew FIELD(mesh);
  field->resize_fdata();

  FIELD::fdata_type &d = field->fdata();
  d[0] = 1;
  d[1] = 1;
}


void TestQuadBilinearLgn()
{
  cerr<<"TestQuadBilinearLgn\n";

  typedef QuadBilinearLgn<Point> MBASIS;
  typedef QuadSurfMesh<QuadBilinearLgn<Point> > MESH;
  MESH *mesh = new MESH();

  MBASIS u;
  MESH::Node::array_type n;
  n.resize(u.number_of_vertices());
 
  for(int i=0; i<u.number_of_vertices(); i++) {
    Point p(u.unit_vertices[i][0]+1, u.unit_vertices[i][1]+2, 3);
    mesh->add_point(p);
    n[i]=i;
  }
  MESH::Elem::index_type ei=mesh->add_elem(n); 

  cerr<<"Element index: " << ei << "\n"; 
  
  vector<double> coords;
  coords.push_back(.2);
  coords.push_back(.2);
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
    
  typedef QuadBilinearLgn<double>  FBASIS;
  typedef GenericField<MESH, FBASIS, vector<double> > FIELD;
  FIELD *field = scinew FIELD(mesh);
  field->resize_fdata();

  FIELD::fdata_type &d = field->fdata();
  d[0] = 1;
  d[1] = 1;
  d[2] = 1;
  d[3] = 1;
}

void TestTetLinearLgn()
{
  cerr<<"TestTetLinearLgn\n";

  typedef TetLinearLgn<Point> MBASIS;
  typedef TetVolMesh<MBASIS > MESH;
  MESH *mesh = new MESH();

  MBASIS u;
  MESH::Node::array_type n;
  n.resize(u.number_of_vertices());
 
 for(int i=0; i<u.number_of_vertices(); i++) {
    Point p(u.unit_vertices[i][0]+1, u.unit_vertices[i][1]+2, u.unit_vertices[i][2]+3);
    mesh->add_point(p);
    n[i]=i;
  }
  MESH::Elem::index_type ei=mesh->add_elem(n); 

  cerr<<"Element index: " << ei << "\n"; 
  
  vector<double> coords;
  coords.push_back(.2);
  coords.push_back(.2);
  coords.push_back(.2);
  Point p;

  mesh->interpolate(p, coords, 0);

  //  if (sqrt(pow(p.x()-coords[0],2.0)+pow(p.y()-coords[1],2.0)+pow(p.z()-coords[1],2.0))>1e-7)    
  cerr << "Transform L->G " << coords[0] <<", " << coords[1] <<", " << coords[2] << " => " << p << endl;

  vector<double> lc(u.domain_dimension());
  
  bool rc=mesh->get_coords(lc, p, 0);
  cerr << "Transform G->L " << p << " => ";
  if (rc) 
    cerr << lc[0] <<", " << lc[1] <<", " << lc[2] << endl;
  else
    cerr << " not found" << endl;
    
  typedef TetLinearLgn<double>  FBASIS;
  typedef GenericField<MESH, FBASIS, vector<double> > FIELD;
  FIELD *field = scinew FIELD(mesh);
  field->resize_fdata();

  FIELD::fdata_type &d = field->fdata();
  d[0] = 1;
  d[1] = 1;
  d[2] = 1;
  d[3] = 1;
}

int
main(int argc, char **argv) 
{
  //TestCrvLinearLgn();
  TestQuadBilinearLgn();
  //TestTetLinearLgn();
 
  return 0;  
}    
