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

#include <sci_defs/mdsplus_defs.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef HAVE_MDSPLUS
  int MDS_Connect( const char *server );
  int MDS_Open( const char *tree, unsigned int shot );
  void MDS_SetSocket( int socket );
  void MDS_Disconnect();

  unsigned int is_valid( const char *signal );

  int get_rank( const char *signal );
  int get_size( const char *signal );
  int get_type( const char *signal );

  int get_dims( const char *signal, unsigned int **dims );
  char *get_name( const unsigned int nid );

  void *get_value( const char *signal, int dtype);
  void *get_values( const char *signal, int dtype );

  unsigned int get_names( const char *signal, char **names,
			  char recurse, char absolute, char type );
  unsigned int get_nids ( const char *signal, unsigned int **nids  );

#endif  // HAVE_MDSPLUS

#ifdef __cplusplus
} // extern "C"
#endif

#endif  // MDS_PLUS_API

