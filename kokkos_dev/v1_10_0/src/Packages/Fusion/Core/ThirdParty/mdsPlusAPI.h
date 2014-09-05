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
 *  MDSPlusAPI.h:
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

#ifndef MDS_PLUS_API

#define MDS_PLUS_API

#include <sci_defs.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef HAVE_MDSPLUS
    int MDS_Connect( const char *server );
    int MDS_Open( const char *tree, int shot );
    void MDS_SetSocket( int socket );
    void MDS_Disconnect();

    int get_rank( const char *signal );
    int get_value( const char *signal );
    int *get_values( const char *signal, int size );
    int get_size( const char *signal);
    double *get_data( const char *signal, int size );
    char *get_string( const char *signal );

    int get_dims( const char *node, int *dims );
    double *get_grid( const char *axis, int *dims );
    int get_slice_ids( int **nids );
    char *get_slice_name( const int *nids, int slice );
    double get_slice_time( const char *name );
    double *get_slice_data( const char *name,
			    const char *space,
			    const char *node,
			    int *dims );
#endif  // HAVE_MDSPLUS

#ifdef __cplusplus
} // extern "C"
#endif

#endif  // MDS_PLUS_API

