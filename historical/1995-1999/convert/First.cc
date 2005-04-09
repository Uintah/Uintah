
/*
 *  ScFldToIves: Read in a scalar field, and output a .dat file for ives
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   October 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Geom/View.h>
#include <Geom/TCLView.h>
#include <Geom/Color.h>
#include <Geometry/Point.h>
#include <Geometry/Vector.h>
#include <Math/Trig.h>

#include <Classlib/String.h>
#include <Classlib/Pstreams.h>
#include <Datatypes/ScalarFieldRG.h>
#include <Datatypes/ColormapPort.h>

#include <Classlib/Array1.h>
#include <Classlib/Array2.h>

#include <Classlib/Timer.h>


#include <Modules/Visualization/LevoyVis.h>

#include <iostream.h>
#include <fstream.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#if 0
#include <RLE/rledump.h>
#endif

#define ELTSIZE      32
#define LINESIZE     64
#define FILENAMESIZE 128


char *
ParseCommandLine ( int argc, char ** argv, FILE ** ViewFile )
{
  *ViewFile = NULL;
  
  int parsing;
  extern char *optarg;
  extern int optind;

  while ((parsing = getopt(argc, argv, "")) != -1)
    switch (parsing) {
    }

  if ( argv[optind] == '\0' )
    {
      printf("Usage: prog ViewFileName\n");
      exit( -1);
    }

  *ViewFile = fopen ( argv[optind], "r" );

  if ( *ViewFile == NULL )
    {
      perror( "open" );
      exit(-1);
    }
  
  return ( argv[optind] );
}    



/**********************************************************
*
**********************************************************/
int
Split ( FILE * ViewFile, char *  LineElts[] )
{
  char line[LINESIZE];
  int loop;

  int count;

  // initialize all LineElts to NULL

  for ( loop = 0; loop <= 10; loop++ )
    LineElts[loop][0] = '\0';

  if ( fgets(line, LINESIZE, ViewFile) == NULL )
    return FALSE;

  count = sscanf( line, "%s%s%s%s%s%s%s%s%s%s%s", LineElts[0], LineElts[1], LineElts[2],
	 LineElts[3], LineElts[4], LineElts[5], LineElts[6], LineElts[7],
	 LineElts[8], LineElts[9], LineElts[10] );

  if ( LineElts[0][0] == '#' )
    return Split( ViewFile, LineElts );
  else
    return TRUE;
}



/**********************************************************
*
**********************************************************/

void
ParseViewFile ( FILE * ViewFile, ExtendedView& v, char * SFn,
	       Array1<double> *Xvals, Array1<double> *Yvals, int& repeat,
	       int& steps )
{
  typedef char * joyous;

  joyous LineElts[11];

  int j;
  for ( j = 0; j<11; j++ )
    LineElts[j] = new char[ELTSIZE];

//  char LineElts[11][ELTSIZE];
  
  LineElts[0][0] = 'a';
  LineElts[0][1] = '\0';

  // traverse through entire file

  while ( Split( ViewFile, LineElts ) )
    {
      switch ( LineElts[0][0] )
	{
	case 'a':
	  if ( ! strcmp( "at", LineElts[0] ) )
	    {
	      Point temp( atof(LineElts[1]), atof(LineElts[2]),
			  atof(LineElts[3]) );
	      v.lookat( temp );
	    }
	  
	  break;
	  
	case 'b':
	  if ( ( ! strcmp( "background", LineElts[0] ) ) ||
	      ( ! strcmp( "bg", LineElts[0] ) ) )
	    {
	      Color tempC ( atof(LineElts[1]), atof(LineElts[2]),
			   atof(LineElts[3]) );
	      v.bg( tempC );
	    }
	  break;

	case 'e':
	  if ( ! strcmp( "eye", LineElts[0] ) )
	    {
	      Point temp( atof(LineElts[1]), atof(LineElts[2]),
			  atof(LineElts[3]) );
	      v.eyep( temp );
	    }
	  
	  break;
	  
	case 'f':
	  if ( !strcmp ( "fov", LineElts[0] ) )
	    {
	      v.fov( atof( LineElts[1] ) );
	    }
	  break;

	case 'n':
	  if ( ! strcmp( "nodes", LineElts[0] ) )
	    {
	      int i;
	      switch ( LineElts[1][0] )
		{
		case 'o':
		  i = 0;
		  break;
		case 'r':
		  i = 1;
		  break;
		case 'g':
		  i = 2;
		  break;
		case 'b':
		  i = 3;
		  break;
		default:
		  i = -1;
		}

	      if ( i != -1 )
		while ( Split( ViewFile, LineElts ) &&
		       strcmp( "end", LineElts[0] ) )
		  {
		    Xvals[i].add( atof( LineElts[0] ) );
		    Yvals[i].add( atof( LineElts[1] ) );
		  }
	    }
	  break;
		
	case 'r':
	  if ( ! strcmp( "rasterX", LineElts[0] ) )
	    {
	      v.xres( atoi( LineElts[1] ) );
	    }
	  
	  if ( ! strcmp( "rasterY", LineElts[0] ) )
	    {
	      v.yres( atoi( LineElts[1] ) );
	    }

	  if ( ! strcmp( "repeat", LineElts[0] ) )
	    repeat = atoi( LineElts[1] );
	  
	  break;

	case 's':
	  if ( ! strcmp( "sf", LineElts[0] ) )
	    {
	      strcpy( SFn, LineElts[1] );
	    }

	  if ( ! strcmp( "steps", LineElts[0] ) )
	    {
	      steps = atoi( LineElts[1] );
	    }
	  
	  break;
	  
	case 'u':
	  if ( strcmp( "up", LineElts[0]) == 0 )
	    {
	      Vector temp( atof(LineElts[1]), atof(LineElts[2]),
			  atof(LineElts[3]) );
	      v.up( temp );
	    }
	  break;
	  
	  
	}
    }
  
}


void
MinMax ( char * SFname, ScalarFieldRG * sfrg, double& min, double& max )
{
  char name[128];
  
  sprintf( name, "%s.dat", SFname );
  ofstream fout(name);
  
  max = 0.0;
  min = 999.0;
  
  for (int i=0; i<sfrg->nx; i++)
    for (int j=0; j<sfrg->ny; j++)
      for (int k=0; k<sfrg->nz; k++) {
	char c=(char)sfrg->grid(i,j,k);
	if ( c > max )
	  max = c;
	if ( c < min )
	  min = c;
      }

  cout << "The min and max are: " << min << "  " << max << endl;
}

#if 0
void
ConvertToURT( RLEdump *truth, Array2<CharColor>* im, int rast )
{
  int loop, pool;
  CharColor f;

  for ( loop = 0; loop < rast; loop++ )
    for ( pool = 0; pool < rast; pool++ )
      {
	f = (*im)(loop,pool);
	truth->WriteCharPixel( pool, loop, f.red, f.green, f.blue );
      }
}
#endif

void
DummyInit( Array2<CharColor>& arr )
{
  int i,j;

  CharColor fff( (unsigned char)101, (unsigned char)101, (unsigned char)101 );

  for( i = 0; i < 100; i++ )
    for( j = 0; j < 100; j++ )
      arr(i,j) = fff;

  CharColor ggg( (unsigned char)255, (unsigned char)255, (unsigned char)255 );
  
  for( i = 0; i < 100; i++ )
    arr(i,i) = ggg;
}


main(int argc, char **argv) {

  FILE* VF;

  ExtendedView myview( Point( 2, 0.5, 0.5 ), Point( 2, 0.5, 0.5 ), Vector( 0.0, 0.0, 1.0 ), 45, 100, 100, Color(0.,0.,0.) );

  char SFname[FILENAMESIZE];

  Color BackgroundColor;

  int repeat, steps;

  repeat = 10;

  Array1<double> Xvals[4];
  Array1<double> Yvals[4];

  ParseCommandLine ( argc, argv, &VF );
  ParseViewFile ( VF, myview, SFname,
		 Xvals, Yvals, repeat, steps );

  // check if got values correctly...

#if 0  
  cout << myview.eyep() << endl;
  cout << myview.lookat() << endl;
  cout << myview.up() << endl;
  cout << myview.fov() << endl;
#endif  

  // read scalar field information

    ScalarFieldHandle SFHandle;

    Piostream* SFstream=auto_istream(SFname);
    if (!SFstream) {
	printf("Couldn't open file %s.  Exiting...\n", SFname);
	exit(0);
    }
    Pio(*SFstream, SFHandle);
    if (!SFHandle.get_rep()) {
	printf("Error reading surface from file %s.  Exiting...\n", SFname);
	exit(0);
    }
    ScalarField *sf = SFHandle.get_rep();
    ScalarFieldRG * sfrg = sf->getRG();

  // figure out the min and max of the scalar field; this is
  // what i'll base my colors on for now...

  double mini, maxi;
  MinMax( SFname, sfrg, mini, maxi );

  // get some work done

//  Levoy levoyModule( (ScalarFieldRG*)sfrg, myview.bg(),
//		    Xvals, Yvals );
  
//  levoyModule.SetUp( myview, steps );

  Levoy levoyModule;
  levoyModule.SetUp( (ScalarFieldRG*) sfrg, Xvals, Yvals, myview,
	      steps );

  // trace rays perspectively

  WallClockTimer watch;
  watch.start();
  
  levoyModule.TraceRays( 1 );

  watch.stop();

  cerr<< "The time is: " << watch.time() << endl;

  // dummy initialization of the array to see if it will work at all

#if 0  
  // initialize the rle dump stuff
  
  RLEdump TheImage;
  
  TheImage.Initialize( rasterX );

  // dump an rle image to the screen

  ConvertToURT( &TheImage, levoyModule.Image, rasterX );

  // write the image to a file

  TheImage.Display( "image" );

  // make it be displayed on the screen

  char command[510];
  
  sprintf( command, "/usr/local/apps/urt/bin/getx11 image\n");

  system ( command );
#endif  

  return 1;
}    

