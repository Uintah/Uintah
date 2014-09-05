//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//    File   : CalcLVVol.cc
//    Author : Martin Cole
//    Date   : Mon May 16 15:48:48 2005

#include <Core/Datatypes/HexVolField.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Containers/HashTable.h>
#include <Core/Init/init.h>
#include <StandAlone/convert/FileUtils.h>

#include <iostream>
#include <fstream>
#include <stdlib.h>

using namespace SCIRun;

using std::cerr;
using std::ifstream;
using std::endl;


double
tet_vol6(const Point &p1, const Point &p2, const Point &p3, const Point &p4)
{
  return fabs( Dot(Cross(p2-p1,p3-p1),p4-p1) );
}

void 
check_lv_vol(FieldHandle fld) 
{
  int base_perim[] = {143, 339, 158, 331, 160, 333, 162, 335, 164, 337, 155, 
		      322, 152, 328, 149, 325, 146, 319};
  int bp_len = 18;
  
  LockingHandle<HexVolMesh> mesh = 
    ((HexVolField<double>*)fld.get_rep())->get_typed_mesh();

  Vector tot(0.0, 0.0, 0.0);
  for (int i = 0; i < bp_len; ++i) {
    Point p;
    mesh->get_center(p, (HexVolMesh::Node::index_type)base_perim[i]);
    //    cerr << "point: " << p << std::endl;
    tot += Vector(p);
    //cerr << "at: " << i << " " << tot << std::endl;
  }

  tot *= 1. / (double)bp_len;
  
  //tot has the commont point for all tets...

  mesh->synchronize(Mesh::FACE_NEIGHBORS_E | Mesh::FACES_E);

  int lv_elems[] = {1, 3, 5, 7, 17, 19, 21, 23, 33, 35, 37, 39, 49, 51, 53, 55, 65, 67, 69, 71, 81, 83, 85, 87, 97, 99, 101, 103, 113, 115, 117, 119, 129, 131, 133, 135, 145, 147, 149, 151, 161, 163, 165, 167, 177, 179, 181, 183, 193, 195, 197, 199, 209, 211, 213, 215, 225, 227, 229, 231, 241, 243, 245, 247, 417, 419, 421, 423, 425, 427, 429, 431, 433, 435, 437, 439, 441, 443, 445, 447, 449, 451, 453, 455, 457, 459, 461, 463, 465, 467, 469, 471, 473, 475, 477, 479, 481, 483, 485, 487, 489, 491, 493, 495, 497, 499, 501, 503, 505, 507, 509, 511, 513, 515, 517, 519, 521, 523, 525, 527, 529, 531, 533, 535, 537, 539, 541, 543, 545, 547, 549, 551, 553, 555, 557, 559, 561, 563, 565, 567, 569, 571, 573, 575, 577, 579, 581, 583, 593, 595, 597, 599, 609, 611, 613, 615, 625, 627, 629, 631, 641, 643, 645, 647, 649, 651, 653, 655, 657, 659, 661, 663, 665, 667, 669, 671, 673, 675, 677, 679};
  
  int sz_lv_elems = 180;
  double tot_vol = 0.0L;
  for (int i = 0; i < sz_lv_elems; ++i) {
    HexVolMesh::Face::array_type faces;
    HexVolMesh::Cell::index_type ci =
      (HexVolMesh::Cell::index_type)(lv_elems[i] - 1);
    mesh->get_faces(faces, ci);

    // Check each face for neighbors.
    HexVolMesh::Face::array_type::iterator fiter = faces.begin();

    while (fiter != faces.end())
    {
      HexVolMesh::Cell::index_type nci;
      HexVolMesh::Face::index_type fi = *fiter;
      ++fiter;

      if (! mesh->get_neighbor(nci , ci, fi))
      {
	// Faces with no neighbors are on the boundary, build 2 tets.
	HexVolMesh::Node::array_type nodes;
	mesh->get_nodes(nodes, fi);
	// the 4 points

	Point p0;
	Point p1;
	Point p2;
	Point p3;
	mesh->get_center(p0, nodes[0]);
	mesh->get_center(p1, nodes[1]);
	mesh->get_center(p2, nodes[2]);
	mesh->get_center(p3, nodes[3]);
	double vol1 = tet_vol6(p0, p1, p2, Point(tot)) / 6.;
	double vol2 = tet_vol6(p0, p2, p3, Point(tot)) / 6.;
	//cerr << "v1: " << vol1 <<  " v2: " << vol2 << std::endl;
	tot_vol += (vol1 + vol2);
	break;
      }
    }
  }
  cerr << "lv_vol: " << tot_vol / 1000. << " ml " 
       << " w/ pivot: " << tot << std::endl;
}

int
main(int argc, char **argv) 
{

  SCIRunInit();

  char *fieldName = argv[1];

  FieldHandle handle;
  Piostream* stream=auto_istream(fieldName);
  if (!stream) {
    cerr << "Couldn't open file " << fieldName << ".  Exiting..." << std::endl;
    return 2;
  }
  Pio(*stream, handle);
  if (!handle.get_rep()) {
    cerr << "Error reading surface from file " << fieldName 
	 << ".  Exiting..." << std::endl;
    return 2;
  }
  if (handle->get_type_description(0)->get_name() != "HexVolField") 
  {
    cerr << "Error -- input field wasn't a HexVolField (type_name=" 
	 << handle->get_type_description(0)->get_name() << std::endl;
    return 2;
  }

  check_lv_vol(handle);
  return 0;
}    
