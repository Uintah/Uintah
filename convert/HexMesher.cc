/*
 *   HexMesher.cc: Read nodes/element files and dump a HexMesh
 *
 *  Written by:
 *   Peter A. Jensen
 *   Department of Computer Science
 *   University of Utah
 *   March 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include <Classlib/String.h>
#include <Datatypes/HexMesh.h>
#include <iostream.h>
#include <fstream.h>
#include <Classlib/Pstreams.h>
#include <stdio.h>
#include <stdlib.h>

/******************************************************************************
*
* Forward declarations
*
******************************************************************************/

void read_nodes (HexMesh * mesh, char * infile);
void read_elems (HexMesh * mesh, char * infile);
void write_mesh (HexMesh * mesh, char * outfile);
HexMesh * read_mesh (char * infile);


/******************************************************************************
* Main
*
*     Check for the proper arguments, then call the functions to build
* a HexMesh and write it out.
******************************************************************************/

int main(int argc, char **argv)
{
  HexMesh * mesh;
  clString dir, out;   
        
  // Make sure the number of parameters is correct.
    
  if (argc != 3 || strlen(argv[1]) < 1 || strlen(argv[2]) < 1)
  {
    cerr << "usage: " << argv[0] << " data_dir output_file\n";
    exit(0);
  }
    
  // Make sure filename(s) are ok.
  
  dir = argv[1];
  out = argv[2];
  
  if (dir(dir.len () - 1) != '/')
    dir = dir + "/";  
    
  // Allocate memory for the mesh.
    
  mesh = new HexMesh;
    
  // Read in the points.

  read_nodes (mesh, (dir + "NODES")());
        
  // Read in the elements
  
  read_elems (mesh, (dir + "ELEMS")());  
      
  // cout << endl;
  
  // Print out the mesh for debugging purposes.
    
  // cout << * mesh;
  
  // Do test.
  
  // Write out the mesh to a file.
  
  write_mesh (mesh, (out)());
  
  // Exit -- don't worry about cleaning up.  :)

  return 0;
}


/******************************************************************************
* read_nodes
*
*     Read the nodes (and their given index) from the node file and add the
* nodes to the mesh.
******************************************************************************/

void read_nodes (HexMesh * mesh, char * infile)
{
  ifstream node_file (infile);
  int index;
  double x, y, z; 
    
  // Make sure the file opened.  
    
  if (! node_file)
  {
    cerr << "Cannot open node file: " << infile << endl;
    exit (-1);
  }
  
  // Read in points.
  
  while (! node_file.eof ())
  {
    node_file >> index >> x >> y >> z;
    mesh->add_node (index, x, y, z);
    //printf (".");
    //fflush (stdout);
  }
}


/******************************************************************************
* read_elems
*
*     Read the connectivities for the various elements.  (Each element is
* paired with an index in the element file.)  Add each element to the mesh.
******************************************************************************/

void read_elems (HexMesh * mesh, char * infile)
{
  ifstream elem_file (infile);
  EightHexNodes e;
  int index, c;
  
  // Make sure the file opened.  
    
  if (! elem_file)
  {
    cerr << "Cannot open element file: " << infile << endl;
    exit (-1);
  }
  
  // Read in node values and build an element from them.
  
  while (! elem_file.eof ())
  {
    elem_file >> index;
    for (c = 0; c < 8; c++)
      elem_file >> e.index[c];
    if(e.index[7] == 0){
	e.index[7]=e.index[3];
	e.index[6]=e.index[2];
	e.index[5]=e.index[1];
	e.index[4]=e.index[1];
	e.index[3]=e.index[0];
	e.index[2]=e.index[0];
	e.index[1]=e.index[0];
	e.index[0]=e.index[0];
   }
    mesh->add_element (index, e);
    //printf ("*");
    //fflush (stdout);
  }  
}


/******************************************************************************
* write_mesh
*
*     Use the standard Pio call to write this mesh out to a text file.
******************************************************************************/

void write_mesh(HexMesh * mesh, char* outfile)
{
  TextPiostream tfile (outfile, Piostream::Write);
               
  Pio (tfile, *mesh);
}


/******************************************************************************
* read_mesh
*
*     Reads a mesh in from a file -- used to debug this convert routine.  (A
* written mesh should be able to be read and rewritten, the second write being
* identical to the first.)
******************************************************************************/

HexMesh * read_mesh(char* infile)
{
  Piostream * file = auto_istream (infile);

  HexMesh * mesh = new HexMesh ();

  Pio (*file, *mesh);
      
  delete file;
    
  return mesh;
}
