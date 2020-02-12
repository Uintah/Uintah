/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

/*
 *  grid_reader.cc: Reads in a binary grid.xml file and displays the actual XML
 *
 *  Written by:
 *   J. Davison de St. Germain
 *   SCI Institute
 *   University of Utah
 *   Aug. 2016
 *
 */

#include <Core/DataArchive/DataArchive.h>

#include <iostream>

#include <unistd.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>

using namespace std;

/////////////////////////////////////////////////////////////////
// Forward declarations:

void parse_file( const string & filename );

/////////////////////////////////////////////////////////////////

void
usage( const std::string& badarg, const std::string& progname )
{
  cout << "\n";
  if(badarg != "") {
    cout << "Error parsing argument: " << badarg << "\n";
  }
  cout << "Usage: " << progname << " [options] <grid.xml>\n\n";
  cout << "Valid options are:\n";
  cout << "  [-]-help\n";
  cout << "\n";
  exit( 1 );
}

int
main( int argc, char** argv )
{
  string filename;

  /*
   * Parse arguments
   */
  for( int pos = 1; pos < argc; pos++ ) {
    string arg = argv[ pos ];
    if( (arg == "-help") || (arg == "-h") || (arg == "--help") ) {
      usage( "", argv[0] );
    } 
    else if( arg[0] != '-' )  {
      filename = arg;
    }
    else {
      cout << "\n";
      cout << "Bad command line argument: '" + arg + "'\n";
      usage( "", argv[0] );
    }
  }

  if( filename == "" ) {
    cout << "No grid file specified\n";
    usage( "", argv[0] );
  }

  parse_file( filename );

}

void
parse_file( const string & filename )
{
  FILE         * fp;
  unsigned int   marker = -1;

  fp = fopen( filename.c_str(), "rb" );

  if( !fp ) {
    cout << "\n";
    cout << "Error: Failed to open file: " << filename << "\n";
    printf( "  (errno is %d: %s)\n", errno, strerror(errno) );
    cout << "\n";
    exit( 1 );
  }

  size_t num = fread( &marker, sizeof(int), 1, fp );

  if( num != 1 || marker != Uintah::DataArchive::GRID_MAGIC_NUMBER ) {
    cout << "\n";
    cout << "Error: " << filename << " does not appear to be a binary grid.xml file: " << marker << "\n";
    cout << "\n";
    fclose( fp );
    exit( 1 );
  }

  // Number of Levels
  int    numLevels, num_patches;
  long   num_cells;
  int    extra_cells[3], period[3];
  double anchor[3], cell_spacing[3];
  int    l_id;
  
  fread( & numLevels,    sizeof(int),    1, fp );

  printf( "<Grid>\n" );
  printf( " <numLevels>%d</numLevels>\n",              numLevels );

  for( int lev = 0; lev < numLevels; lev++ ) {

    fread( & num_patches,  sizeof(int),    1, fp );    // Number of Patches -  100
    fread( & num_cells,    sizeof(long),   1, fp );    // Number of Cells   - 8000
    fread(   extra_cells,  sizeof(int),    3, fp );    // Extra Cell Info   - [1,1,1]
    fread(   anchor,       sizeof(double), 3, fp );    // Anchor Info       - [0,0,0]
    fread(   period,       sizeof(int),    3, fp );    // 
    fread( & l_id,         sizeof(int),    1, fp );    // ID of Level       -    0
    fread(   cell_spacing, sizeof(double), 3, fp );    // Cell Spacing      - [0.1,0.1,0.1]

    string meta_period = "";
    if( period[0] == 0 && period[0] == 0 && period[0] == 0 ) {
      meta_period = "  <!-- Note: A [0,0,0] period is ignored by code. -->";
    }

    printf( " <Level>\n" );
    printf( "  <numPatches>%d</numPatches>\n",                    num_patches );
    printf( "  <totalCells>%ld</totalCells>\n",                   num_cells );
    printf( "  <extraCells>[%d,%d,%d]</extraCells>\n",            extra_cells[0],  extra_cells[1],  extra_cells[2] );
    printf( "  <anchor>[%f,%f,%f]</anchor>\n",                    anchor[0],       anchor[1],       anchor[2] );

    printf( "  <periodic>[%d,%d,%d]</periodic>%s\n",                period[0],       period[1],       period[2], meta_period.c_str() );

    printf( "  <id>%d</id>\n",                                    l_id );
    printf( "  <cellspacing>[%.18f,%.18f,%.18f]</cellspacing>\n", cell_spacing[0], cell_spacing[1], cell_spacing[2] );

    for( int patch = 0; patch < num_patches; patch++ ) {
      int    p_id, rank, nnodes, total_cells;
      int    low_index[3], high_index[3], i_low_index[3], i_high_index[3];
      double lower[3], upper[3];
     
      fread( & p_id,         sizeof(int),    1, fp );
      fread( & rank,         sizeof(int),    1, fp );
      fread(   low_index,    sizeof(int),    3, fp );    // <lowIndex>[-1,-1,-1]</lowIndex>
      fread(   high_index,   sizeof(int),    3, fp );    // <highIndex>[20,20,4]</highIndex>
      fread(   i_low_index,  sizeof(int),    3, fp );    // <interiorLowIndex></interiorLowIndex>
      fread(   i_high_index, sizeof(int),    3, fp );    // <interiorHighIndex>[20,20,3]</interiorHighIndex>
      fread( & nnodes,       sizeof(int),    1, fp );    // <nnodes>2646</nnodes>
      fread(   lower,        sizeof(double), 3, fp );    // <lower>[-0.025000000000000001,-0.025000000000000001,-0.049999999999999996]</lower>
      fread(   upper,        sizeof(double), 3, fp );    // <upper>[0.5,0.5,0.19999999999999998]</upper>
      fread( & total_cells,  sizeof(int),    1, fp );    // <totalCells>2205</totalCells>
      
      string meta_interior_low = "";
      if( i_low_index[0] == low_index[0] && i_low_index[1] == low_index[1] && i_low_index[2] == low_index[2] ) {
        meta_interior_low = "  <!-- Note: Code will ignore as same as low index. -->";
      }
      string meta_interior_high = "";
      if( i_high_index[0] == high_index[0] && i_high_index[1] == high_index[1] && i_high_index[2] == high_index[2] ) {
        meta_interior_high = "  <!-- Note: Code will ignore as same as high index. -->";
      }


      printf( "  <Patch>\n" );
      printf( "   <id>%d</id>\n",                                       p_id );
      printf( "   <proc>%d</proc>\n",                                   rank );
      printf( "   <lowIndex>[%d,%d,%d]</lowIndex>\n",                   low_index[0],    low_index[1],    low_index[2]  );
      printf( "   <highIndex>[%d,%d,%d]</highIndex>\n",                 high_index[0],   high_index[1],   high_index[2]  );
      printf( "   <interiorLowIndex>[%d,%d,%d]</interiorLowIndex>%s\n",   i_low_index[0],  i_low_index[1],  i_low_index[2], meta_interior_low.c_str() );
      printf( "   <interiorHighIndex>[%d,%d,%d]</interiorHighIndex>%s\n", i_high_index[0], i_high_index[1], i_high_index[2], meta_interior_high.c_str() );
      printf( "   <nnodes>%d</nnodes>\n",                               nnodes );
      printf( "   <lower>[%.18f,%.18f,%.18f]</lower>\n",                lower[0],        lower[1],        lower[2]  );
      printf( "   <upper>[%.18f,%.18f,%.18f]</upper>\n",                upper[0],        upper[1],        upper[2]  );
      printf( "   <totalCells>%d</totalCells>\n",                       total_cells );
      printf( "  </Patch>\n" );
    }
    printf(   " </Level>\n" );
  }
  printf(     "</Grid>\n" );
  
  fclose( fp );
  
}
