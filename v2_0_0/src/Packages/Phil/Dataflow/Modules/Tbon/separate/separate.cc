/* separate.cc
 * reorder HTVolumeBrick data for use in the unstructured T-BON
 *
 * Packages/Philip Sutton
 * May 1999
 */

#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <limits.h>
#include <math.h>

typedef float T;
const float ERROR = 1e-5;

struct Point {
  float x, y, z;
};

struct Tetra {
  int v[4];
};

void checkfile( FILE* fp, const char* filename ) {
  if( !fp ) {
    printf("Error: cannot open file %s\n",filename);
    exit(1);
  }
}

void main( int argc, char** argv ) {
  int npts, ntets;
  int ncells, size;
  int i, j, k;
  char filename[80];
  Point* parray;
  Tetra* tarray;
  int* cells;
  int* tetlist;
  int nx, ny, nz;

  // files of filenames
  FILE* htvolfiles;
  FILE* datafiles;

  // files for constant data
  FILE* listfile;
  FILE* geom;

  // varying files
  FILE* htvol;
  FILE* data;
  
  if( argc != 8 ) {
    printf("usage: separate <<htvol files>> <lists file> <geometry file> <<data files>> nx ny nz\n");
    printf("  ( << >> indicates a file of filenames )\n");
    exit(1);
  } 

  // open files
  htvolfiles = fopen( argv[1], "r" );
  checkfile( htvolfiles, argv[1] );

  listfile = fopen( argv[2], "r" );
  checkfile( listfile, argv[2] );

  geom  = fopen( argv[3], "w" );
  checkfile( geom, argv[3] );

  datafiles  = fopen( argv[4], "r" );
  checkfile( datafiles, argv[4] );

  nx = atoi( argv[5] );
  ny = atoi( argv[6] );
  nz = atoi( argv[7] );

  // extract geometry information
  fscanf( htvolfiles, "%s", filename );
  htvol = fopen( filename, "r" );
  checkfile( htvol, filename );
  fseek( htvolfiles, 0, SEEK_SET );

  // strip header
  fscanf( htvol, "%*s %*s\n" );
  fscanf( htvol, "%d %d\n", &npts, &ntets );
  parray = new Point[npts];
  tarray = new Tetra[ntets];

  // read points
  float minx, maxx, miny, maxy, minz, maxz;
  minx = miny = minz = FLT_MAX;
  maxx = maxy = maxz = -FLT_MAX;
  for( i = 0; i < npts; i++ ) {
    T val;
    fread( &parray[i], sizeof(float), 3, htvol );
    fread( &val, sizeof(T), 1, htvol );
    if( parray[i].x < minx ) minx = parray[i].x;
    if( parray[i].x > maxx ) maxx = parray[i].x;
    if( parray[i].y < miny ) miny = parray[i].y;
    if( parray[i].y > maxy ) maxy = parray[i].y;
    if( parray[i].z < minz ) minz = parray[i].z;
    if( parray[i].z > maxz ) maxz = parray[i].z;
  }
  float dx, dy, dz;
  dx = (maxx - minx) / (float)(nx-1);
  dy = (maxy - miny) / (float)(ny-1);
  dz = (maxz - minz) / (float)(nz-1);

  printf("min = (%f %f %f)\nmax = (%f %f %f)\n",minx,miny,minz,maxx,maxy,maxz);
  printf("d = (%f %f %f)\n",dx,dy,dz);

  // read tets
  for( i = 0; i < ntets; i++ ) {
    fread( &tarray[i], sizeof(int), 4, htvol );
  }
  fclose( htvol );
  
  // read list data
  fscanf( listfile, "%d\n%d\n", &ncells, &size );
  cells = new int[ncells];
  tetlist = new int[size];
  fread( cells, sizeof(int), ncells, listfile );
  fread( tetlist, sizeof(int), size, listfile );
  fclose( listfile );

  printf("reordering points...\n");
  
  // reorder points
  int* order = new int[npts];
  int* invorder = new int[npts];
  int curr = 0;
  int* done = new int[npts];
  bzero( done, sizeof(int)*npts );

  for( int z = 0; z < nz; z++ ) {
    printf("\nstarting z = %d\n",z);
    for( int y = 0; y < ny; y++ ) {
      printf("X");
      fflush(NULL);
      for( int x = 0; x < nx; x++ ) {
	float cx = minx + (float)x * dx;
	float cy = miny + (float)y * dy;
	float cz = minz + (float)z * dz;
	for( i = 0; i < npts; i++ ) {
	  if( done[i] )
	    continue;
	  if( parray[i].x >= cx && parray[i].x <= cx+dx &&
	      parray[i].y >= cy && parray[i].y <= cy+dy &&
	      parray[i].z >= cz && parray[i].z <= cz+dz ) {
	    order[curr] = i;
	    invorder[i] = curr;
	    curr++;
	    done[i] = 1;
	  } else if( fabs(parray[i].x-cx) < ERROR && 
		     parray[i].y >= cy && parray[i].y <= cy+dy &&
		     parray[i].z >= cz && parray[i].z <= cz+dz ) {
	    order[curr] = i;
	    invorder[i] = curr;
	    curr++;
	    done[i] = 1;
	  } else if( fabs(parray[i].x-(cx+dx)) < ERROR &&
		     parray[i].y >= cy && parray[i].y <= cy+dy &&
		     parray[i].z >= cz && parray[i].z <= cz+dz ) {
	    order[curr] = i;
	    invorder[i] = curr;
	    curr++;
	    done[i] = 1;
	  } else if( fabs(parray[i].y-cy) < ERROR && 
		     parray[i].x >= cx && parray[i].x <= cx+dx &&
		     parray[i].z >= cz && parray[i].z <= cz+dz ) {
	    order[curr] = i;
	    invorder[i] = curr;
	    curr++;
	    done[i] = 1;
	  } else if( fabs(parray[i].y-(cy+dy)) < ERROR &&
		     parray[i].x >= cx && parray[i].x <= cx+dx &&
		     parray[i].z >= cz && parray[i].z <= cz+dz ) {
	    order[curr] = i;
	    invorder[i] = curr;
	    curr++;
	    done[i] = 1;
	  } else if( fabs(parray[i].z-cz) < ERROR && 
		     parray[i].x >= cx && parray[i].x <= cx+dx &&
		     parray[i].y >= cy && parray[i].y <= cy+dy ) {
	    order[curr] = i;
	    invorder[i] = curr;
	    curr++;
	    done[i] = 1;
	  } else if( fabs(parray[i].z-(cz+dz)) < ERROR &&
		     parray[i].x >= cx && parray[i].x <= cx+dx &&
		     parray[i].y >= cy && parray[i].y <= cy+dy ) {
	    order[curr] = i;
	    invorder[i] = curr;
	    curr++;
	    done[i] = 1;
	  } 
	} 
      }
    }
  }

  printf("\nchecking...\n");
  for( i = 0; i < npts; i++ )
    if( done[i] != 1 ) 
      printf("point %d (%f %f %f) is not assigned!\n",i,parray[i].x,
	     parray[i].y,parray[i].z);
  
  printf("reassigning points...\n");

  // reassign points
  Point* newpts = new Point[npts];
  for( i = 0; i < npts; i++ )
    newpts[i] = parray[ order[i] ];
  delete [] parray;
  parray = newpts;

  printf("reassigning tets...\n");

  // reassign tets
  Tetra* newtets = new Tetra[ntets];
  for( i = 0; i < ntets; i++ ) {
    for( j = 0; j < 4; j++ ) 
      newtets[i].v[j] = invorder[ tarray[i].v[j] ];
  }
  delete [] tarray;
  tarray = newtets;
  
  printf("finding extremes...\n");

  // find min, max points for each cell
  int* extremes = new int[2*ncells];
  for( i = 0; i < ncells; i++ ) {
    int min = INT_MAX;
    int max = -INT_MAX;
    int n = tetlist[ cells[i] ];
    if( n > 0 ) {
      for( j = 1; j <= n; j++ ) {
	for( k = 0; k < 4; k++ ) {
	  if( tarray[ tetlist[ cells[i] +j ] ].v[k] < min )
	    min = tarray[ tetlist[ cells[i] +j ] ].v[k];
	  if( tarray[ tetlist[ cells[i] +j ] ].v[k] > max )
	    max = tarray[ tetlist[ cells[i] +j ] ].v[k];
	}
      }
      extremes[2*i] = min;
      extremes[2*i+1] = max;
    } else {
      extremes[2*i] = -1;
      extremes[2*i+1] = -1;
    }   // if( n > 0 )

  } // i = 0 .. ncells-1

  printf("writing geometry...\n");

  // write geometry
  fprintf( geom, "%d %d\n", npts, ntets );
  fprintf( geom, "%d %d\n", ncells, size );
  for( i = 0; i < npts; i++ )
    fwrite( &parray[i], sizeof(float), 3, geom );
  for( i = 0; i < ntets; i++ )
    fwrite( &tarray[i].v, sizeof(int), 4, geom );
  fwrite( tetlist, sizeof(int), size, geom );
  fwrite( extremes, sizeof(int), ncells*2, geom );
  fwrite( cells, sizeof(int), ncells, geom );
  fclose(geom);

  // extract data
  while( fscanf( htvolfiles, "%s", filename ) != EOF ) {

    printf("extracting data from file %s\n",filename);

    htvol = fopen( filename, "r" );
    checkfile( htvol, filename );

    fscanf( datafiles, "%s", filename );
    data = fopen( filename, "w" );
    checkfile( data, filename );
    
    T* values = new T[npts];

    // read points
    fscanf( htvol, "%*s %*s\n" );
    fscanf( htvol, "%*d %*d\n" );
    for( i = 0; i < npts; i++ ) {
      float buf[3];
      fread( buf, sizeof(float), 3, htvol );
      fread( &values[i], sizeof(T), 1, htvol );
    }

    // reorder points
    T* newvalues = new T[npts];
    for( i = 0; i < npts; i++ )
      newvalues[i] = values[ order[i] ];
    
    // write points
    fwrite( newvalues, sizeof(T), npts, data );

    delete [] values;
    delete [] newvalues;

    fclose( htvol );
    fclose( data );
  }

  delete [] extremes;
  delete [] order;
  delete [] invorder;
  delete [] parray;
  delete [] tarray;
  delete [] cells; 
  delete [] tetlist;
  delete [] done;

  fclose( htvolfiles );
  fclose( datafiles );
}
