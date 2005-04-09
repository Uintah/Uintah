/*
 *   HexUG.cc: Read a HexMesh and data values and dump a ScalarFieldHUG
 *
 *  Written by:
 *   Peter A. Jensen
 *   Department of Computer Science
 *   University of Utah
 *   April 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include <Classlib/String.h>
#include <Datatypes/ScalarFieldHUG.h>
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

void read_data (Array1<double> & data, char * infile);
void write_HUG (ScalarFieldHUG * grid, char * outfile);
HexMesh * read_mesh (char * infile);


/******************************************************************************
* Main
*
*     Check for the proper arguments, then call the functions to build
* a ScalarFieldHUG and write it out.
******************************************************************************/

int main(int argc, char **argv)
{
  HexMesh * mesh;
  ScalarFieldHUG * grid;
  clString dir, out, meshfile;   
        
  // Make sure the number of parameters is correct.
    
  if (argc != 4 || strlen(argv[1]) < 1 || strlen(argv[2]) < 1 || strlen(argv[3]) < 1)
  {
    cerr << "usage: " << argv[0] << " data_dir hex_mesh output_file\n";
    exit(0);
  }
    
  // Make sure filename(s) are ok.
  
  dir = argv[1];
  meshfile = argv[2];
  out = argv[3];
  
  if (dir(dir.len () - 1) != '/')
    dir = dir + "/";  
    
  // Get the mesh.
    
  mesh = read_mesh (meshfile());
  
  // Make the scalar field.
  
  grid = new ScalarFieldHUG (mesh);
    
  // Read in the data values

  grid->data.resize (mesh->high_node()+1);
  read_data (grid->data, (dir + "S5")());
          
  // Write out the scalar field to a file.
  
  write_HUG (grid, (out)());
  
  // Exit -- don't worry about cleaning up.  :)

  return 0;
}


/******************************************************************************
* read_data
*
*     Read the data values (with their index) and store them in an array.
******************************************************************************/

void read_data (Array1<double> & data, char * infile)
{
  ifstream data_file (infile);
  int index;
  double v; 
    
  // Make sure the file opened.  
    
  if (! data_file)
  {
    cerr << "Cannot open data file: " << infile << endl;
    exit (-1);
  }
  
  // Read in point values.
  
  while (! data_file.eof ())
  {
    data_file >> index >> v;
    data[index] = v;
  }
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


/******************************************************************************
* write_HUG
*
******************************************************************************/

void write_HUG(ScalarFieldHUG * grid, char* outfile)
{
  TextPiostream tfile (outfile, Piostream::Write);
               
  ScalarFieldHandle gridhandle(grid);
  Pio (tfile, gridhandle);
}
