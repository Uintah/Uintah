/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  MDSPlusAPI.cc:
 *
 *  Written by:
 *   Allen Sanderson
 *   School of Computing
 *   University of Utah
 *   March 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

/* This is a C/C++ interface for fetching data from a MDSPlus Server.
   It also contains many helper functions for fetching the data.

   This interface is geared towards fetching NIMROD data from the server.
   It is not a complete general purpose interface although several of
   the functions are.
*/

#include <sci_defs.h>

#ifdef HAVE_MDSPLUS

#include "mdsPlusAPI.h"

#include <mdslib.h>

/* These are not defined in mdslib.h so define them here. */
int MdsOpen(char *tree, int* shot);
void MdsDisconnect();
int MdsSetSocket( int *socket );

/* If an MDSPlus call is successful the first bit is set. */
#define status_ok( status ) ((status & 1) == 1)

/* Simple interface to interface bewteen the C++ and C calls. */
int MDS_Connect( const char *server )
{
  /* Connect to MDSplus */
  int retVal = MdsConnect((char*)server);

  return retVal;
}

/* Simple interface to interface bewteen the C++ and C calls. */
int MDS_Open( const char *tree, int shot )
{
  /* Open tree */
  int retVal = status_ok( MdsOpen((char*)tree,&shot) ) ? 0 : -1;

  return retVal;
}

/* Simple interface to interface bewteen the C++ and C calls. */
void MDS_Disconnect()
{
  /* Disconnect to MDSplus */
  MdsDisconnect();
}

/* Simple interface to interface bewteen the C++ and C calls. */
void MDS_SetSocket( int socket )
{
  /* Disconnect to MDSplus */
  MdsSetSocket( &socket );
}

/*  Query the rank of the node - as in the number of dimensions. */
int get_rank( const char *signal ) {

  /* Local vars */
  char buf[1024];                     /* Buffer for MDS+ exp */

  /* Put RANK() TDI function around signal name. */
  memset(buf,  0,sizeof(buf));
  sprintf(buf, "RANK(%s)", signal);
    
  return get_value( buf );
}

/*  Query the size of the node  - as in the number of elements. */
int get_size( const char *signal) {

  /* local vars */
  char buf[1024];                     /* Buffer for MDS+ exp */

  /* Put SIZE() TDI function around signal name */
  memset(buf, 0, sizeof(buf));
  sprintf(buf, "SIZE(%s)", signal);

  return get_value( buf );
}

/*  Query the long value of the node - as in the number of slices. */
int get_value( const char *signal ) {

  /* Local vars */
  char buf[1024];                     /* Buffer for MDS+ exp */   
  int len, dsc;                       /* Used in MDS+ calls  */
  int null = 0;
  int dtype_long = DTYPE_LONG;        /* MDS+ Descripter type def, long */

  int value = 0;
  int retVal;

  /* Build a descriptor for fectching the data. */
  dsc = descr(&dtype_long, &value, &null);

  /* Put the signal name in the buffer passed to MDS. */
  memset(buf,0,sizeof(buf));
  sprintf(buf,"%s", signal);
    
  /* Use MdsValue to get the value */
  retVal = status_ok( MdsValue(buf, &dsc, &null, &len) ) ? value : -1;

  return retVal;
}

/*  Query the long values of the node - as in the nids the of slices. */
int* get_values( const char *signal, int size ) {

  /* Local vars */
  char buf[1024];                     /* buffer for MDS+ exp */  
  int len, dsc;                       /* Used in MDS+ calls  */
  int null = 0;
  int dtype_long = DTYPE_LONG;        /* MDS+ Descripter type def, long */

  int* values = (int *) malloc(size * sizeof(int));
  int* retVal;

  /* Build a descriptor for fectching the data. */
  dsc = descr(&dtype_long, values, &size, &null);

  /* Put the signal in the buffer passed to mds. */
  memset(buf,0,sizeof(buf));
  sprintf(buf,"%s", signal);

  /* Use MdsValue to get the value */
  retVal = status_ok( MdsValue(buf, &dsc, &null, &len) ) ? values : NULL;

  return retVal;
}

/*  Query the double value of the node - as in the number of slices. */
double get_datum( const char *signal ) {

  /* Local vars */
  char buf[1024];                     /* Buffer for MDS+ exp */   
  int len, dsc;                       /* Used in MDS+ calls  */
  int null = 0;
  int dtype_long = DTYPE_DOUBLE;      /* MDS+ Descripter type def, double */

  double datum = 0;
  double retVal;

  /* Build a descriptor for fectching the data. */
  dsc = descr(&dtype_long, &datum, &null);

  /* Put the signal name in the buffer passed to MDS. */
  memset(buf,0,sizeof(buf));
  sprintf(buf,"%s", signal);
    
  /* Use MdsValue to get the value */
  retVal = status_ok( MdsValue(buf, &dsc, &null, &len) ) ? datum : -1;

  return retVal;
}

/*  Query the double values of the node - as in the signal for the of slices. */
double* get_data( const char *signal, int size )
{
  /* local vars */
  char buf[1024];                     /* buffer for MDS+ exp */   
  int len, dsc;                       /* Used in MDS+ calls  */
  int null = 0;
  int dtype_double = DTYPE_DOUBLE;    /* MDS+ Descripter type def, long */

  /* Create a data array of sufficient size.  Note: Tried to use a
     * multidimensional array, which works for static allocation.
     * However, MdsValue seems to get confused about using dynamically
     * allocated multidimensional arrays. */
  double *data = (double *) malloc(size * sizeof(double));
  double* retVal;

  memset(data,0,sizeof(data));
  dsc = descr(&dtype_double, data, &size, &null);

  /* Put the signal in the buffer passed to mds. */
  memset(buf,0,sizeof(buf));
  sprintf(buf,"%s", signal);

  /* Use MdsValue to get the value */
  retVal = status_ok( MdsValue(buf, &dsc, &null, &len) ) ? data : NULL;

  return retVal;
}

/*  Query the string values of the node - as in the name of the slices. */
char* get_string( const char *signal )
{
  /* local vars */
  char buf[1024];                     /* buffer for MDS+ exp */
  int len, dsc;                       /* Used in MDS+ calls  */
  int null = 0;
  int dtype_cstring = DTYPE_CSTRING;  /* MDS+ Descripter type def, cstring */

  int size = 32;

    /* Create a data array of sufficient size. */
  char *data = (char *) malloc(size * sizeof(char));
  char* retVal;

  memset(data,0,sizeof(data));
  dsc = descr(&dtype_cstring, data, &null, &size);

  /* Put the signal in the buffer passed to Mds. */
  memset(buf,0,sizeof(buf));
  sprintf(buf,"%s", signal);

  /* Use MdsValue to get the value */
  retVal = status_ok( MdsValue(buf, &dsc, &null, &len) ) ? data : NULL;

  return retVal;
}

/* Get the rank and number of dimensions for a particular signal. */
int get_dims( const char *node, int *dims )
{
  char buf[1024];                 /* buffer for MDS+ exp */

  int i, rank;

  /* Fetch the rank of the signal. */
  sprintf( buf, "%s", node );
  rank = get_rank( buf );

  if( 0 < rank && rank < 4 ) { 
    /* Fetch the dimensions of the signal. */
    for( i=0; i<rank; i++ )
      {
	sprintf( buf, "%s,%d", node, i );

	dims[i] = get_size( buf );
      }
  }

  return rank;
}

/*  Query the server for the cylindrical or cartesian components of the grid. */
double* get_grid( const char *axis, int *dims )
{
  char *gridStr = "\\NIMROD::TOP.OUTPUTS.CODE.GRID";
  char buf[1024];                 /* buffer for MDS+ exp */

  int i, rank, size = 1;

  /* Fetch the Grid data from the node. */
  sprintf( buf, "%s:%s", gridStr, axis );
  rank = get_dims( buf, dims );

  /* Make sure the rank is correct for the axis specified. */
  if( ( strcmp( axis, "R"        ) == 0 && rank == 2 ) ||
      ( strcmp( axis, "PHI"      ) == 0 && rank == 1 ) ||
      ( strcmp( axis, "X"        ) == 0 && rank == 3 ) ||
      ( strcmp( axis, "Y"        ) == 0 && rank == 3 ) ||
      ( strcmp( axis, "Z"        ) == 0 && rank == 2 ) ||
      ( strcmp( axis, "K"        ) == 0 && rank == 1 ) ||
      ( strcmp( axis, "RADIAL"   ) == 0 && rank == 1 ) ||
      ( strcmp( axis, "POLOIDAL" ) == 0 && rank == 1 ) )
  {
    /* The dimensions for the axis. */
    for( i=0; i<rank; i++ )
      size *= dims[i];

    /* Get the data for the axis. */
    sprintf( buf, "%s:%s", gridStr, axis );

    return get_data( buf, size );
  }
  else
    return NULL;
}

/*  Query the server for the slice nids for the shot. */
int get_slice_ids( int **nids )
{
  char *sliceStr = "\\NIMROD::TOP.OUTPUTS.CODE.SLICES";
  char buf[1024];                 // buffer for MDS+ exp     

  int nSlices;

  /* Query the server for the number of slices in the tree. */
  sprintf( buf, "getnci(\'\\%s\',\'NUMBER_OF_CHILDREN\')", sliceStr );
  nSlices = get_value(  buf );

  /* Query the nid numbers of the slices. */
  sprintf( buf, "getnci(\"\\%s.*\",\"NID_NUMBER\")", sliceStr );

  *nids = get_values( buf, nSlices );

  return nSlices;
}

/*  Query the server for the slice name. */
char* get_slice_name( const int *nids, int slice )
{
  char buf[1024];                 // buffer for MDS+ exp     
  char *name = NULL;              // Used to hold the name of the slice 

  int i;

  /* Query the name of the slice node, that nids[slice] refers to. */
  sprintf( buf, "getnci(%i,\"NODE_NAME\")", nids[slice] );
  name = get_string( buf );

  /* Trim the white spaces off of the node name. */
  if( name ) {
    i = 0;
    while (name[i] != ' ') i++;
    name[i] = '\0';
  }

  return name;
}

/*  Query the server for the slice time. */
double get_slice_time( const char *name )
{
  char *sliceStr = "\\NIMROD::TOP.OUTPUTS.CODE.SLICES";
  char buf[1024];                 // buffer for MDS+ exp     

  // Get the time of the slice.
  sprintf(buf,"%s.%s:TIME", sliceStr, name );

  return get_datum( buf );
}


/*  Query the server for the slice real space data. */
double *get_slice_data( const char *name,
			const char *space,
			const char *node,
			int *dims )
{
  char *sliceStr = "\\NIMROD::TOP.OUTPUTS.CODE.SLICES";

  char buf[1024];                 // buffer for MDS+ exp     

  int i, rank, size = 1;

  /*  */
  sprintf(buf, "%s.%s.%s.%s", sliceStr, name, space, node);

  /* Fetch the rank and size of the signal. */
  rank = get_dims( buf, dims );

  if( rank == 3 ) {

    /* Get the total size of the signal. */
    for( i=0; i<rank; i++ )
      size *= dims[i];

    /* Fetch the data from the node */
    sprintf(buf,"%s.%s.%s.%s", sliceStr, name, space, node);

    return get_data( buf, size );
  }
  else
    return NULL;
}
#endif  // HAVE_MDSPLUS
