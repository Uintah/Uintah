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

#include <mdslib.h>

#include "mdsPlusAPI.h"

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

/*  Query to see if the node is valid - does it really exist. */
int is_valid( const char *signal) {

  /* local vars */
  char buf[1024];                     /* Buffer for MDS+ exp */

  int *nid = NULL;
  int *len = NULL;

  /* Put GETNCI() TDI function around signal name */
  memset(buf, 0, sizeof(buf));
  sprintf( buf, "GETNCI(%s,\"NID_NUMBER\")", signal );

  nid = (int*) get_value( buf, DTYPE_LONG );

  /* Put GETNCI() TDI function around signal name */
  memset(buf, 0, sizeof(buf));
  sprintf( buf, "GETNCI(%s,\"LENGTH\")", signal );

  len = (int*) get_value( buf, DTYPE_LONG );

  return (nid && len) ? (*len>1) : 0;
}

/*  Query the type of the node - as in the data type. */
int get_type( const char *signal ) {

  /* Local vars */
  char buf[1024];                     /* Buffer for MDS+ exp */

  int *type = NULL;

  /* Put KIND and DATA TDI functions around signal name */
  memset(buf,  0,sizeof(buf));
  sprintf(buf, "KIND(DATA(%s))", signal);

  type = (int*) get_value( buf, DTYPE_LONG );

  return type ? *type : -1;
}

/*  Query the size of the node  - as in the number of elements. */
int get_size( const char *signal) {

  /* local vars */
  char buf[1024];                     /* Buffer for MDS+ exp */

  int *size = NULL;

  /* Put SIZE() TDI function around signal name */
  memset(buf, 0, sizeof(buf));
  sprintf(buf, "SIZE(%s)", signal);

  size = (int*) get_value( buf, DTYPE_LONG );

  return size ? *size : -1;
}

/*  Query the rank of the node - as in the number of dimensions. */
int get_rank( const char *signal ) {

  /* Local vars */
  char buf[1024];                     /* Buffer for MDS+ exp */

  int *rank = NULL;

  /* Put RANK() TDI function around signal name. */
  memset(buf,  0,sizeof(buf));
  sprintf(buf, "RANK(%s)", signal);
    
  rank = (int*) get_value( buf, DTYPE_LONG );

  return rank ? *rank : -1;
}

/* Get the rank and number of dimensions for a particular signal. */
int get_dims( const char *signal, int **dims )
{
  char buf[1024];                 /* buffer for MDS+ exp */
  int i;

  /* Fetch the rank of the signal. */
  int rank = get_rank( signal );

  if( rank > 0 ) {
    *dims = malloc( rank * sizeof( int ) );

    /* Fetch the dimensions of the signal. */
    for( i=0; i<rank; i++ ) {
      sprintf( buf, "%s,%d", signal, i );
    
      (*dims)[i] = get_size( buf );
    }
  }

  return rank;
}


/*  Query the long value of the node - as in the number of slices. */
void* get_value( const char *signal, int dtype ) {

  /* Local vars */
  char buf[1024];                     /* Buffer for MDS+ exp */   
  int len, dsc;                       /* Used in MDS+ calls  */
  int null = 0;

  int size = 1;
  void *value;

  switch( dtype ) {
  case DTYPE_UCHAR:
  case DTYPE_USHORT:
  case DTYPE_ULONG:
  case DTYPE_ULONGLONG:
  case DTYPE_CHAR:
  case DTYPE_SHORT:
  case DTYPE_LONG:
  case DTYPE_LONGLONG:
    dtype = DTYPE_LONG;
    value = (void*) malloc( sizeof( int ) );
    break;

  case DTYPE_FLOAT:
  case DTYPE_FS:
    dtype = DTYPE_FLOAT;
    value = (void*) malloc( sizeof( float ) );
    break;

  case DTYPE_DOUBLE:
  case DTYPE_FT:
    dtype = DTYPE_DOUBLE;
    value = (void*) malloc( sizeof( double ) );
    break;

  case DTYPE_CSTRING:
    /* Put SIZEOF() TDI function around signal name */
    memset(buf, 0, sizeof(buf));
    sprintf( buf, "SIZEOF(%s)", signal );
  
    size = (int) (*(int*) get_value( buf, DTYPE_LONG ));

    dtype = DTYPE_CSTRING;
    value = (void*) malloc( size * sizeof( char ) );
    break;

  default:
    return NULL;
    break;
  }

  /* Build a descriptor for fectching the data. */
  if( dtype == DTYPE_CSTRING )
    dsc = descr(&dtype, value, &null, &size);
  else
    dsc = descr(&dtype, value, &null);

  /* Put the signal name in the buffer passed to MDS. */
  memset(buf,0,sizeof(buf));
  sprintf(buf,"%s", signal);
    
  /* Use MdsValue to get the value */
  return status_ok( MdsValue(buf, &dsc, &null, &len) ) ? value : NULL;
}

/*  Query the long values of the node - as in the nids the of slices. */
void* get_values( const char *signal, int dtype ) {

  is_valid( signal );

  /* Local vars */
  char buf[1024];                     /* buffer for MDS+ exp */  
  int len, dsc;                       /* Used in MDS+ calls  */
  int null = 0;
  int *dims;
  int i, size = 1;

  void* values;

  int rank = get_dims( signal, &dims );

  if( rank < 0 )
    return NULL;

  if( rank == 0 )
    return get_value( signal, dtype );

  
  for( i=0; i<rank; i++ )
    size *= dims[i];
  
  free( dims );

  switch( dtype ) {
  case DTYPE_UCHAR:
  case DTYPE_USHORT:
  case DTYPE_ULONG:
  case DTYPE_ULONGLONG:
  case DTYPE_CHAR:
  case DTYPE_SHORT:
  case DTYPE_LONG:
  case DTYPE_LONGLONG:
    dtype = DTYPE_LONG;
    values = (void*) malloc( size * sizeof( int ) );
    break;

  case DTYPE_FLOAT:
  case DTYPE_FS:
    dtype = DTYPE_FLOAT;
    values = (void*) malloc( size * sizeof( float ) );
    break;

  case DTYPE_DOUBLE:
  case DTYPE_FT:
    dtype = DTYPE_DOUBLE;
    values = (void*) malloc( size * sizeof( double ) );
    break;

  case DTYPE_CSTRING:
    dtype = DTYPE_CSTRING;
    values = (void*) malloc( size * sizeof( char ) );
    break;

  default:
    return NULL;
    break;
  }

  memset(values,0,sizeof(values));

  /* Build a descriptor for fectching the data. */
  if( dtype == DTYPE_CSTRING )
    dsc = descr(&dtype, values, &null, &size);
  else
    dsc = descr(&dtype, values, &size, &null);

  /* Put the signal in the buffer passed to mds. */
  memset(buf,0,sizeof(buf));
  sprintf(buf,"%s", signal);

  /* Use MdsValue to get the value */
  return status_ok( MdsValue(buf, &dsc, &null, &len) ) ? values : NULL;
}


/*  Query the server for the slice name. */
char* get_name( const int nid )
{
  char buf[1024];                 // buffer for MDS+ exp     
  char *name = NULL;              // Used to hold the name of the slice 

  /* Query the name of the slice node, that nids[slice] refers to. */
  memset(buf,0,sizeof(buf));
  sprintf( buf, "GETNCI(%i,\"NODE_NAME\")", nid );

  name = get_value( buf, DTYPE_CSTRING );

  /* Trim the white spaces off of the node name. */
  if( name ) {
    int i = 0;
    while (name[i] != ' ') i++;
    name[i] = '\0';
  }

  return name;
}


/*  Query the server for the components of the grid. */
double* get_grid( const char *axis, int **dims )
{
  char *gridStr = "\\NIMROD::TOP.OUTPUTS.CODE.GRID";
  char buf[1024];                 /* buffer for MDS+ exp */

  /* Fetch the Grid data from the node. */
  sprintf( buf, "%s:%s", gridStr, axis );

  get_dims( buf, dims );

  /* Get the data for the axis. */
  sprintf( buf, "%s:%s", gridStr, axis );
  
  return (double*) get_values( buf, DTYPE_DOUBLE );
}

/*  Query the server for the slice nids for the shot. */
int get_slice_ids( int **nids )
{
  char *sliceStr = "\\NIMROD::TOP.OUTPUTS.CODE.SLICES";
  char buf[1024];                 /* buffer for MDS+ exp */

  int *nSlices = NULL;

  /* Query the server for the number of slices in the tree. */
  sprintf( buf, "GETNCI(%s,\"NUMBER_OF_CHILDREN\")", sliceStr );
  
  nSlices = (int*) get_value( buf, DTYPE_LONG );

  /* Query the nid numbers of the slices. */
  sprintf( buf, "GETNCI(\"\\%s.*\",\"NID_NUMBER\")", sliceStr );

  *nids = (int*) get_values( buf, DTYPE_LONG );

  return nSlices ? *nSlices : 0;
}

/*  Query the server for the slice time. */
double get_slice_time( const char *name )
{
  char *sliceStr = "\\NIMROD::TOP.OUTPUTS.CODE.SLICES";
  char buf[1024];                 /* buffer for MDS+ exp */

  double *time;

  // Get the time of the slice.
  sprintf(buf,"%s.%s:TIME", sliceStr, name );

  time = (double*) get_value( buf, DTYPE_DOUBLE );

  return time ? *time : 0.0;
}


/*  Query the server for the slice real space data. */
double *get_slice_data( const char *name,
			const char *space,
			const char *node,
			int **dims )
{
  char *sliceStr = "\\NIMROD::TOP.OUTPUTS.CODE.SLICES";

  char buf[1024];                 /* buffer for MDS+ exp */

  /* Fetch the data from the node */
  sprintf(buf, "%s.%s.%s.%s", sliceStr, name, space, node);

  get_dims( buf, dims );
  
  return (double*) get_values( buf, DTYPE_DOUBLE );
}
#endif  // HAVE_MDSPLUS
