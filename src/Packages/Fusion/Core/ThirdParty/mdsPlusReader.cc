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
std::string MDSPlusReader::server_;
std::string MDSPlusReader::tree_;
int MDSPlusReader::shot_ = 0;
int MDSPlusReader::connects_ = 0;
int MDSPlusReader::opens_ = 0;

MDSPlusReader::MDSPlusReader()
{
}

/* Simple interface to interface bewteen the C and C++ calls. */
int MDSPlusReader::connect( std::string server )
{
  mdsPlusLock_.lock();

  int retVal;

  if( connects_ ) {
    if( server == server_ ) {
      connects_++;
      retVal = 0;
    }
    else {
      retVal = -2;
    }
  } else {

#ifdef HAVE_MDSPLUS
    /* Connect to MDSplus */
    retVal = MDS_Connect((char*)server.c_str());
#endif

    if( retVal == 0 ) {
      server_ = server_;
      connects_++;
    }
  }

  mdsPlusLock_.unlock();

  return retVal;
}

/* simple interface to interface bewteen the C and C++ calls. */
int MDSPlusReader::open( std::string tree, int shot )
{
  mdsPlusLock_.lock();

  int retVal;

  if( opens_ ) {
    if( tree == tree_ && shot == shot_ ) {
      opens_++;
      retVal = 0;
    }
    else {
      retVal = -2;
    }
  } else {

#ifdef HAVE_MDSPLUS
    /* Open tree */
    retVal = MDS_Open((char*)tree.c_str(), shot);
#endif
    if( retVal == 0 ) {
      tree_ = tree;
      shot_ = shot;
      opens_++;
    }
  }

  mdsPlusLock_.unlock();

  return retVal;
}

/* Simple interface to interface bewteen the C and C++ calls. */
void MDSPlusReader::disconnect()
{
  mdsPlusLock_.lock();

  if( opens_ && --opens_ == 0 ) {

    tree_ = "";
    shot_ = 0;
  }

  if( connects_ && --connects_ == 0 ) {

#ifdef HAVE_MDSPLUS
    /* Disconnect to MDSplus */
    MDS_Disconnect();
#endif
    server_ = "";
  }

  mdsPlusLock_.unlock();
}

/*  Query the rank of the node - as in the number of dimensions. */
int MDSPlusReader::rank( const std::string signal ) {
#ifdef HAVE_MDSPLUS

  mdsPlusLock_.lock();

  int rank = get_rank( signal.c_str() );

  mdsPlusLock_.unlock();

  return rank;
#else
  return 0;
#endif
}

/*  Query the size of the node  - as in the number of elements. */
int MDSPlusReader::size( const std::string signal ) {

  mdsPlusLock_.lock();

#ifdef HAVE_MDSPLUS
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

  int nSlices = get_slice_ids( nids );

  mdsPlusLock_.unlock();

  return nSlices;
#else
  return 0;
#endif
}

/*  Query the server for the slice name. */
std::string MDSPlusReader::slice_name( const int *nids, int slice )
{
#ifdef HAVE_MDSPLUS
  mdsPlusLock_.lock();

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

  double* data =
    get_slice_data( name.c_str(), space.c_str(), node.c_str(), dims );
  mdsPlusLock_.unlock();

  return data;
#else
  return NULL;
#endif
}
}
