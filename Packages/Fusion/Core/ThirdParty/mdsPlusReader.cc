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
*/

#include <Packages/Fusion/Core/ThirdParty/mdsPlusReader.h>

#include "mdsPlusAPI.h"

namespace Fusion {

using namespace SCIRun;

Mutex MDSPlusReader::mdsPlusLock_( "MDS Plus Mutex Lock" );

MDSPlusReader::MDSPlusReader() : socket_(-1)
{
}

/* Simple interface to interface bewteen the C and C++ calls. */
int MDSPlusReader::connect( std::string server )
{
#ifdef HAVE_MDSPLUS
  int retVal;

  mdsPlusLock_.lock();

  /* Connect to MDSplus */
  MDS_SetSocket( -1 );  // Insure that there will not be a disconnect.

  socket_ = MDS_Connect((char*)server.c_str());

  if( socket_ > 0 )
    retVal = 0;
  else
    retVal = -1;

  mdsPlusLock_.unlock();

  return retVal;
#else
  return -1;
#endif
}

/* simple interface to interface bewteen the C and C++ calls. */
int MDSPlusReader::open( const std::string tree,
			 const int shot )
{
#ifdef HAVE_MDSPLUS
  int retVal;

  mdsPlusLock_.lock();

  MDS_SetSocket( socket_ );

  /* Open tree */
  retVal = MDS_Open((char*)tree.c_str(), shot);

  mdsPlusLock_.unlock();

  return retVal;
#else
  return -1;
#endif
}

/* Simple interface to interface bewteen the C and C++ calls. */
void MDSPlusReader::disconnect()
{
#ifdef HAVE_MDSPLUS
  mdsPlusLock_.lock();

  MDS_SetSocket( socket_ );

  /* Disconnect to MDSplus */
  MDS_Disconnect();

  socket_ = -1;

  mdsPlusLock_.unlock();
#endif
}

/*  Query the rank of the node - as in the number of dimensions. */
int MDSPlusReader::rank( const std::string signal )
{
#ifdef HAVE_MDSPLUS
  mdsPlusLock_.lock();

  MDS_SetSocket( socket_ );

  int rank = get_rank( signal.c_str() );

  mdsPlusLock_.unlock();

  return rank;
#else
  return 0;
#endif
}

/*  Query the size of the node  - as in the number of elements. */
int MDSPlusReader::size( const std::string signal )
{
#ifdef HAVE_MDSPLUS
  mdsPlusLock_.lock();

  MDS_SetSocket( socket_ );

  int size = get_size( signal.c_str() );

  mdsPlusLock_.unlock();

  return size;
#else
  return 0;
#endif
}

/*  Query the server for the cylindrical or cartesian components of the grid. */
double* MDSPlusReader::grid( const std::string axis, int *dims )
{
#ifdef HAVE_MDSPLUS
  mdsPlusLock_.lock();

  MDS_SetSocket( socket_ );

  double* data = get_grid( axis.c_str(), dims );

  mdsPlusLock_.unlock();

  return data;
#else
  return NULL;
#endif
}

/*  Query the server for the slice nids for the shot. */
int MDSPlusReader::slice_ids( int **nids ) 
{
#ifdef HAVE_MDSPLUS
  mdsPlusLock_.lock();

  MDS_SetSocket( socket_ );

  int nSlices = get_slice_ids( nids );

  mdsPlusLock_.unlock();

  return nSlices;
#else
  return 0;
#endif
}

/*  Query the server for the slice name. */
std::string MDSPlusReader::slice_name( const int *nids,
				       int slice )
{
#ifdef HAVE_MDSPLUS
  mdsPlusLock_.lock();

  MDS_SetSocket( socket_ );

  std::string name( get_slice_name(nids, slice) );

  mdsPlusLock_.unlock();

  return name;
#else
  return "";
#endif
}

/*  Query the server for the slice time. */
double MDSPlusReader::slice_time( const std::string name )
{
#ifdef HAVE_MDSPLUS
  mdsPlusLock_.lock();

  MDS_SetSocket( socket_ );

  double time = get_slice_time( name.c_str() );

  mdsPlusLock_.unlock();

  return time;
#else
  return 0;
#endif
}


/*  Query the server for the slice real space data. */
double *MDSPlusReader::slice_data( const std::string name,
				   const std::string space,
				   const std::string node,
				   int *dims )
{
#ifdef HAVE_MDSPLUS
  mdsPlusLock_.lock();

  MDS_SetSocket( socket_ );

  double* data =
    get_slice_data( name.c_str(), space.c_str(), node.c_str(), dims );
  mdsPlusLock_.unlock();

  return data;
#else
  return NULL;
#endif
}
}
