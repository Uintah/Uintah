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

//    File   : Lin2QuadMesh.cc
//    Author : Frank B. Sachse
//    Date   : 24 APR 2006

// Standalone program for converting linear meshes to quadratic


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



template<typename FieldLin, typename FieldQuad>
void lin2quad(FieldLin *lf)
{
  typedef typename FieldLin::mesh_type MeshLin;
  typedef typename FieldQuad::mesh_type MeshQuad;
  typedef typename FieldLin::basis_type BasisLin;

  FieldQuad *qf=new FieldQuad();
  
  MeshLin *lm=dynamic_cast<MeshLin *>(lf->mesh().get_rep());
  ASSERT(lm);
  MeshQuad *qm=dynamic_cast<MeshQuad *>(qf->mesh().get_rep());
  ASSERT(qm);
 
  {
    cerr << "Copying node points from linear to quad mesh\n";
    typename MeshLin::Node::iterator na, nb;
    lm->begin(na);
    lm->end(nb);
    
    while(na!=nb) {
      Point p;
      lm->get_point(p, *na);
      //    cerr << p << endl; 
      qm->add_point(p);
      ++na;
    }
  }
  
  {
    cerr << "Creating elements in quadratic mesh\n";
    typename MeshLin::Cell::iterator ca, cb;
    lm->begin(ca);
    lm->end(cb);
    
    while(ca!=cb) {
      typename MeshQuad::Node::array_type na;
      //   na.resize(mb.number_of_mesh_vertices());
      lm->get_nodes(na, *ca);
      //    cerr << na[0] << endl; 
      qm->add_elem(na);
      ++ca;
    }
    lm->synchronize(MeshLin::EDGES_E);
    qm->synchronize(MeshQuad::EDGES_E);
  }

  {
    cerr << "Creating quadratic node points\n";
    typename MeshLin::Edge::iterator ea, eb;
    lm->begin(ea);
    lm->end(eb);
    
    while(ea!=eb) {
      Point p;
      lm->get_center(p, *ea);
      //    cerr << p << endl;
      qm->get_basis().add_node_value(p);
      ++ea;
    }
  }

  { 
    cerr << "Copying node values from linear to quad mesh\n";
    if (BasisLin::dofs())    
      qf->fdata()=lf->fdata();
    else
      qf->resize_fdata();
  }

  {
    cerr << "Creating quadratic node points\n";
    vector<double> coord(1,0.5L);
    
    typename MeshLin::Edge::iterator ea, eb;
    lm->begin(ea);
    lm->end(eb);
    
    int errorprinted=0;
    int i=0;
    while(ea!=eb) {
      Point p;
      lm->get_center(p, *ea);
      cerr << i++ << ' ' << errorprinted << '\r';
  
      typename MeshLin::Elem::index_type e;
      vector<double> coords(lm->dimensionality());
      bool found=false;
      if (lm->locate(e, p)) 
	if (lm->get_coords(coords, p, e)) 
	  found=true;

      typename FieldLin::value_type v=0;
      if (!errorprinted && !found) {
	cerr << "Interpolation failed\n";
	errorprinted++;
      }
      else if (BasisLin::dofs()) {
 	lf->interpolate(v, coords, e);
      }
      
      qf->get_basis().add_node_value(v);
      ++ea;
    }
  }

  FieldHandle fH(qf);
  TextPiostream out_stream("a.fld", Piostream::Write);
  Pio(out_stream, fH);
}


int
main(int argc, char **argv) 
{
  if (argc<=1) {
    cerr << argv[0] << " LinearField" << endl;
    exit(-1);
  }

  string fn=argv[1];

  Piostream *stream = auto_istream(fn);
  if (!stream)
  {
    cerr << argv[0] << ": Error reading file '" << fn << "'" << endl;
    return -1;
  }
  
  // Read the file
  FieldHandle field;
  Pio(*stream, field);
  if (!field.get_rep() || stream->error())
  {
    cerr << argv[0] << ": Error reading data from file '" << fn << "'" << endl;
    delete stream;
    return -1;
  }
  delete stream;


  typedef GenericField<CurveMesh<CrvLinearLgn<Point> >, CrvLinearLgn<double>, vector<double> >   CrvFLin;
  typedef GenericField<CurveMesh<CrvQuadraticLgn<Point> >, CrvQuadraticLgn<double>, vector<double> > CrvFQuad;

  typedef GenericField<TetVolMesh<TetLinearLgn<Point> >, TetLinearLgn<double>, vector<double> >   TetFLin;
  typedef GenericField<TetVolMesh<TetQuadraticLgn<Point> >, TetQuadraticLgn<double>, vector<double> > TetFQuad;

  typedef GenericField<TetVolMesh<TetLinearLgn<Point> >, NoDataBasis<double>, vector<double> >   TetNFLin;
  typedef GenericField<TetVolMesh<TetQuadraticLgn<Point> >, TetQuadraticLgn<double>, vector<double> > TetNFQuad;

  if (dynamic_cast<CrvFLin *>(field.get_rep()))
    lin2quad<CrvFLin, CrvFQuad>((CrvFLin *)field.get_rep());
  else if (dynamic_cast<TetFLin *>(field.get_rep()))
    lin2quad<TetFLin, TetFQuad>((TetFLin *)field.get_rep());
  else if (dynamic_cast<TetNFLin *>(field.get_rep()))
    lin2quad<TetNFLin, TetNFQuad>((TetNFLin *)field.get_rep());
  else
    cerr << argv[0] << ": Invalid field '" << fn << "'" << endl;

  return 0;  
}    
