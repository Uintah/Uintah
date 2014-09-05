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

#include <sci_defs/mdsplus_defs.h>

#ifdef HAVE_MDSPLUS

#include <mdslib.h>

#include "mdsPlusAPI.h"


/* If an MDSPlus call is successful the first bit is set. */
#define status_ok( status ) ((status & 1) == 1)


/* Simple interface to interface bewteen the C++ and C calls. */
int
MDS_Connect( const char *server )
{
  /* Connect to MDSplus */
  int retVal = MdsConnect((char*)server);

  return retVal;
}


/* Simple interface to interface bewteen the C++ and C calls. */
int
MDS_Open( const char *tree, unsigned int shot )
{
  /* Open tree */
  int retVal = status_ok( MdsOpen((char*)tree,&shot) ) ? 0 : -1;

  return retVal;
}


/* Simple interface to interface bewteen the C++ and C calls. */
void
MDS_Disconnect()
{
  /* Disconnect to MDSplus */
  MdsDisconnect();
}


/* Simple interface to interface bewteen the C++ and C calls. */
void
MDS_SetSocket( int socket )
{
  /* Disconnect to MDSplus */
  MdsSetSocket( &socket );
}


/*  Query to see if the node is valid - does it really exist. */
unsigned int
is_valid( const char *signal) {

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
int
get_type( const char *signal ) {

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
int
get_size( const char *signal) {

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
int
get_rank( const char *signal ) {

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
int
get_dims( const char *signal, unsigned int **dims )
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


/*  Query the server for the slice name. */
char * 
get_name( const unsigned int nid )
{
  char buf[1024];                 // buffer for MDS+ exp     

  /* Query the name of the slice node, that nids[slice] refers to. */
  memset(buf,0,sizeof(buf));
  sprintf( buf, "GETNCI(%i,\"NODE_NAME\")", nid );

  return (char*) get_value( buf, DTYPE_CSTRING );
}


/*  Query the server for the child names of a signal. */
unsigned int 
get_names( const char *signal, 
	   char **names, char recurse, char absolute, char type )
{
  char buf[1024];                 /* buffer for MDS+ exp */

  char wildcard[8];
  char path[12];
  char item[12];

  if( recurse ) strcpy( wildcard, "***" );
  else          strcpy( wildcard, ":*" );

  if( absolute ) strcpy( path, "fullpath" );
  else           strcpy( path, "node_name" );

  if( type ) strcpy( item, "text" );
  else       strcpy( item, "signal" );

  sprintf( buf, "GETNCI(\"\\%s%s\", \"%s\", \"%s\")",
	   signal, wildcard, path, item );

  unsigned int *dims;
  int rank = get_dims( buf, &dims );

  if( rank > 0 ) {
    int i, nnames = 1;

    for( i=0; i<rank; i++ )
      nnames *= dims[i];

    *names = (char*) get_values( buf, DTYPE_CSTRING );

    return nnames;
  } else {
    return 0;
  }
}


/*  Query the server for the nids of a signal. */
unsigned int
get_nids( const char *signal, unsigned int **nids )
{
  char buf[1024];                 /* buffer for MDS+ exp */

  int *nSlices = NULL;

  /* Query the server for the number of slices in the tree. */
  sprintf( buf, "GETNCI(\"\\%s\",\"NUMBER_OF_CHILDREN\")", signal );
  
  nSlices = (int*) get_value( buf, DTYPE_LONG );

  /* Query the nid numbers of the slices. */
  sprintf( buf, "GETNCI(\"\\%s.*\",\"NID_NUMBER\")", signal );

  *nids = (int*) get_values( buf, DTYPE_LONG );

  return nSlices ? *nSlices : 0;
}


/*  Query the long value of the node - as in the number of slices. */
void *
get_value( const char *signal, int dtype ) {

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
  case 52: /*DTYPE_FS:*/
    dtype = DTYPE_FLOAT;
    value = (void*) malloc( sizeof( float ) );
    break;

  case DTYPE_DOUBLE:
  case 53: /*DTYPE_FT:*/
    dtype = DTYPE_DOUBLE;
    value = (void*) malloc( sizeof( double ) );
    break;

  case DTYPE_CSTRING:
    /* Put SIZEOF() TDI function around signal name */
    memset(buf, 0, sizeof(buf));
    sprintf( buf, "SIZEOF(%s)", signal );
  
     value = (void*) get_value( buf, DTYPE_LONG );

    if( value )
      size = (int) (*(int*) value) + 1;
    else
      return NULL;

    dtype = DTYPE_CSTRING;
    value = (void*) malloc( size * sizeof( char ) );
    break;

  default:
    return NULL;
  }

  if( value == 0 ) {
    return NULL;
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
  if( status_ok( MdsValue(buf, &dsc, &null, &len) ) ) {
    if( dtype == DTYPE_CSTRING ) {

      /* Trim the white spaces off of the strings. */
      if( value ) {
	char *name = value;

	int i = size - 1;
	while (name[i] == ' ')
	  name[i--] = '\0';

	value = (void*) realloc( name, strlen(name) * sizeof( char ) );
      }
    }
    return value;
  } else {
    free( value );
    return NULL;
  }
}


/*  Query the long values of the node - as in the nids the of slices. */
void *
get_values( const char *signal, int dtype ) {

  int max_str = 1024;

  /* Local vars */
  char buf[1024];                     /* buffer for MDS+ exp */  
  int len, dsc;                       /* Used in MDS+ calls  */
  unsigned int null = 0;
  unsigned int *dims;
  unsigned int i, size = 1;

  void* values;

  is_valid( signal );

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
  case 52: /*DTYPE_FS:*/
    dtype = DTYPE_FLOAT;
    values = (void*) malloc( size * sizeof( float ) );
    break;

  case DTYPE_DOUBLE:
  case 53: /*DTYPE_FT:*/
    dtype = DTYPE_DOUBLE;
    values = (void*) malloc( size * sizeof( double ) );
    break;

  case DTYPE_CSTRING:
    dtype = DTYPE_CSTRING;
    values = (void*) malloc( size * max_str * sizeof( char ) );
    break;

  default:
    return NULL;
  }

  if( values == 0 ) {
    return NULL;
  }

  memset(values,0,sizeof(values));

  /* Build a descriptor for fectching the data. */
  if( dtype == DTYPE_CSTRING )
    dsc = descr(&dtype, values, &size, &null, &max_str );
  else
    dsc = descr(&dtype, values, &size, &null);

  /* Put the signal in the buffer passed to mds. */
  memset(buf,0,sizeof(buf));
  sprintf(buf,"%s", signal);

  /* Use MdsValue to get the value */
  if( status_ok( MdsValue(buf, &dsc, &null, &len) ) ) {
    if( dtype == DTYPE_CSTRING ) {

      /* Trim the white spaces off of the strings. */
      int n, cc = 0;
      char *names = values;

      for( n=0; n<size; n++ ) {
	char *name = (char*) &(names[n*max_str]);

	if( name ) {
	  int i = max_str - 1;
	  while (name[i] == ' ')
	    name[i--] = '\0';
	}

	strcpy( &(names[cc]), name );
	cc += strlen(name) + 1;
      }
      /*
      names = (void*) realloc( values, cc * sizeof( char ) );

      values = names;
      */
    }

    return values;

  } else {
    free( values );
    return NULL;
  }
}

#endif  // HAVE_MDSPLUS
