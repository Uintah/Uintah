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


/*
 *  HexVolFieldToExotxt.cc
 *
 *  Written by:
 *   Jason Shepherd
 *   Department of Computer Science
 *   University of Utah
 *   May 2006
 *
 *  Copyright (C) 2006 SCI Group
 */

// This program will read in a SCIRun HexVolField, and will save
// out the HexVolMesh into a very basic file that can be converted to an exodusII file
// using the exotxt translator.

#include <Core/Datatypes/GenericField.h>
#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Datatypes/HexVolMesh.h>
#include <Core/Containers/HashTable.h>
#include <Core/Init/init.h>
#include <Core/Persistent/Pstreams.h>

#include <StandAlone/convert/FileUtils.h>

#include <iostream>
#include <fstream>

#include <stdlib.h>

using std::cerr;
using std::ifstream;
using std::endl;

using namespace SCIRun;

int baseIndex;

void setDefaults() {
  baseIndex=1;
}

int parseArgs(int argc, char *argv[]) {
  int currArg = 3;
  while (currArg < argc) {
    cerr << "Error - unrecognized argument: "<<argv[currArg]<<"\n";
    return 0;
  }
  return 1;
}

void printUsageInfo(char *progName) {
  cerr << "\n Usage: "<<progName<<" HexVolField Exotxt_file\n\n";
  cerr << "\t This program will read in a SCIRun HexVolField, and will \n";
  cerr << "\t save out the HexVolMesh into a very basic file that can be \n";
  cerr << "\t converted to an exodusII file using the exotxt translator. \n";
}

int
main(int argc, char **argv) {
  if (argc < 2 || argc > 3) {
    printUsageInfo(argv[0]);
    return 2;
  }

  SCIRunInit();
  setDefaults();

  char *fieldName = argv[1];
  char *outputName = argv[2];
  if (!parseArgs(argc, argv)) {
    printUsageInfo(argv[0]);
    return 2;
  }

  FieldHandle handle;
  Piostream* stream=auto_istream(fieldName);
  if (!stream) {
    cerr << "Couldn't open file "<<fieldName<<".  Exiting...\n";
    return 2;
  }
  Pio(*stream, handle);
  if (!handle.get_rep()) {
    cerr << "Error reading surface from file "<<fieldName<<".  Exiting...\n";
    return 2;
  }
  if (handle->get_type_description(Field::MESH_TD_E)->get_name().find("HexVolField") != 
      string::npos) {
    cerr << "Error -- input field wasn't a HexVolField (type_name="
	 << handle->get_type_description(Field::MESH_TD_E)->get_name() << std::endl;
    return 2;
  }

  typedef HexVolMesh<HexTrilinearLgn<Point> > HVMesh;

  MeshHandle mH = handle->mesh();
  HVMesh *hvm = dynamic_cast<HVMesh *>(mH.get_rep());
  HVMesh::Node::iterator niter; 
  HVMesh::Node::iterator niter_end; 
  HVMesh::Node::size_type nsize; 
  hvm->begin(niter);
  hvm->end(niter_end);
  hvm->size(nsize);  
  HVMesh::Cell::size_type csize; 
  HVMesh::Cell::iterator citer; 
  HVMesh::Cell::iterator citer_end; 
  HVMesh::Node::array_type cell_nodes(8);
  hvm->size(csize);
  hvm->begin(citer);
  hvm->end(citer_end);

  FILE *f_out = fopen(outputName, "wt");
  if (!f_out) {
    cerr << "Error opening output file "<<outputName<<"\n";
    return 2;
  }

  int node_size = (unsigned)(nsize);
  int hex_size = (unsigned)(csize);
  cerr << "Number of points = "<< nsize <<"\n";

  fprintf( f_out, "! Database Title                             exo2txt                                                                 \n" );
  fprintf( f_out, "cubit(temp.g): 05/06/2005: 16:52:36                                             \n" );
  fprintf( f_out, "! Database initial variables\n" );
  fprintf( f_out, "         3      3.01               ! dimensions, version number\n" );
  fprintf( f_out, "      %d      %d         1     ! nodes, elements, element blocks\n", node_size, hex_size );
  fprintf( f_out, "           0         0               ! #node sets, #side sets\n" );
  fprintf( f_out, "         0         0               ! len: node set list, dist fact length\n" );
  fprintf( f_out, "         0         0         0     ! side sets len: element, node , dist fact\n" );
  fprintf( f_out, "! Coordinate names\n" );
  fprintf( f_out, "x                                y                                z                               \n" );
  fprintf( f_out, "! Coordinates\n" );

  while(niter != niter_end) 
  {
    Point p;
    hvm->get_center(p, *niter);
    fprintf( f_out, "%lf %lf %lf\n", p.x(), p.y(), p.z() );
    ++niter;
  }

  fprintf( f_out, "! Node number map\n" );
  fprintf( f_out, "sequence 1..numnp\n" );
  fprintf( f_out, "! Element number map\n" );
  fprintf( f_out, "sequence 1..numel\n" );
  fprintf( f_out, "! Element order map\n" );
  fprintf( f_out, "sequence 1..numel\n" );
  fprintf( f_out, "! Element block    1\n" );
  fprintf( f_out, "         1%10d      HEX8      ! ID, elements, name\n", hex_size );
  fprintf( f_out, "         8         0      ! nodes per element, attributes\n" );
  fprintf( f_out, "! Connectivity\n" );
  
  cerr << "Number of hexes = "<< csize <<"\n";
  while(citer != citer_end) 
  {
    hvm->get_nodes(cell_nodes, *citer);
    fprintf(f_out, "%d %d %d %d %d %d %d %d\n", 
            (int)cell_nodes[0]+baseIndex,
            (int)cell_nodes[1]+baseIndex,
            (int)cell_nodes[2]+baseIndex,
            (int)cell_nodes[3]+baseIndex,
            (int)cell_nodes[4]+baseIndex,
            (int)cell_nodes[5]+baseIndex,
            (int)cell_nodes[6]+baseIndex,
            (int)cell_nodes[7]+baseIndex);
    ++citer;
  }

  fprintf( f_out, "! Properties\n" );
  fprintf( f_out, "         1            ! Number of ELEMENT BLOCK Properties\n" );
  fprintf( f_out, "! Property Name: \n" );
  fprintf( f_out, "ID                              \n" );
  fprintf( f_out, "! Property Value(s): \n" );
  fprintf( f_out, "         1\n" );
  fprintf( f_out, "         0            ! Number of NODE SET Properties\n" );
  fprintf( f_out, "         0            ! Number of SIDE SET Properties\n" );
  fprintf( f_out, "! QA Records\n" );
  fprintf( f_out, "         1      ! QA records\n" );
  fprintf( f_out, "exo2txt                         \n" );
  fprintf( f_out, " 1.13                           \n" );
  fprintf( f_out, "20050506                        \n" );
  fprintf( f_out, "16:54:44                        \n" );
  fprintf( f_out, "! Information Records\n" );
  fprintf( f_out, "         0      ! information records\n" );
  fprintf( f_out, "! Variable names\n" );
  fprintf( f_out, "         0         0         0      ! global, nodal, element variables\n" );
  
  fclose(f_out);

  return 0;  
}    
