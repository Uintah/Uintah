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
 * HEADER (H) FILE : PointCloudWrapper.h
 *
 * DESCRIPTION     : This is a wrapper class that contains methods for 
 *                   dynamically adding and modifying node/value pairs in a 
 *                   point cloud mesh.  Since the mesh and data are stored 
 *                   separately (the mesh is stored in a PointCloudMesh and the
 *                   data is stored in a PointCloudField), it is convenient to
 *                   have some wrapper functions that deal with both the mesh
 *                   and data simultaneously.  In this case, I want to treat
 *                   a node and its data as a unit.  Since a single mesh can 
 *                   have multiple fields applied to it, there is a vector of 
 *                   fields as a member variable. 
 *                     
 * AUTHOR(S)       : Jenny Simpson
 *                   SCI Institute
 *                   University of Utah
 *                 
 * CREATED         : Mon Dec 29 15:00:37 MST 2003
 * MODIFIED        : Mon Dec 29 15:00:37 MST 2003
 * DOCUMENTATION   :
 * NOTES           : 
 *
 * Copyright (C) 2003 SCI Group
*/

#ifndef PointCloudWrapper_h
#define PointCloudWrapper_h

// SCIRun includes
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/PointCloudField.h>
#include <sci_hash_map.h>

// Standard lib includes
#include <iostream>
#include <assert.h>

namespace SCIRun {

// ****************************************************************************
// ************************** Class: PointCloudWrapper ************************
// ****************************************************************************

typedef LockingHandle< PointCloudField<double> > PCField;

// Hash structs for hashing the node locations (key) and indices (value) for
// fast lookup

// Comparison function
struct NodeIDCompare
{
  bool operator()( const char * id1, const char * id2 ) const
  {
    return strcmp( id1, id2 ) == 0;
  }
};

class PointCloudWrapper
{

public:

  // !Constructors
  PointCloudWrapper();

  // !Copy constructor
  PointCloudWrapper(const PointCloudWrapper& w);

  // !Destructor
  ~PointCloudWrapper();

  // !Member functions
  PCField create_field( string data_name );
  PCField get_field( string data_name );
  void update_node_value( string id, Point pt, double value, 
                          string data_name );
  void remove_node( Point pt, string data_name ); 
  void freeze( string data_name );

private:

  LockingHandle<PointCloudMesh> mesh_; // Single point cloud mesh
  // Set of fields than can be applied to the point cloud mesh
  vector<PCField> fields_; 

  int mesh_size_; // Numbef of nodes in the mesh.  I'm using this because
                  // I can't figure out how to get this info from the mesh

#ifdef HAVE_HASH_MAP

  hash_map<const char *, int, hash<const char *>, NodeIDCompare> node_hash_;

#else

  map<const char *, int, NodeIDCompare> node_hash_;

#endif

};

} // End namespace SCIRun
 
#endif // PointCloudWrapper_h

