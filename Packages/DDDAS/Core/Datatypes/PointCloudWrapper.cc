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
 * C++ (CC) FILE : PointCloudWrapper.cc
 *
 * DESCRIPTION   : This is a wrapper class that contains methods for 
 *                 dynamically adding and modifying node/value pairs in a 
 *                 point cloud mesh.  Since the mesh and data are stored 
 *                 separately (the mesh is stored in a PointCloudMesh and the
 *                 data is stored in a PointCloudField), it is convenient to
 *                 have some wrapper functions that deal with both the mesh
 *                 and data simultaneously.  In this case, I want to treat
 *                 a node and its data as a unit.  Since a single mesh can 
 *                 have multiple fields applied to it, there is a vector of 
 *                 fields as a member variable. 
 *                       
 * AUTHOR(S)     : Jenny Simpson
 *                 SCI Institute
 *                 University of Utah
 *                 
 * CREATED       : Mon Dec 29 15:00:37 MST 2003
 * MODIFIED      : Fri Feb 27 11:41:18 MST 2004
 * DOCUMENTATION :
 * NOTES         : 
 *
 * Copyright (C) 2003 SCI Group
*/
 
// SCIRun includes
#include <Packages/DDDAS/Core/Datatypes/PointCloudWrapper.h>

using namespace std;

namespace SCIRun {

/*===========================================================================*/
// 
// PointCloudWrapper
//
// Description : Constructor
//
// Arguments   : none
//
PointCloudWrapper::PointCloudWrapper()
{
  // Initialize PointCloudMesh
  mesh_ = scinew PointCloudMesh;
  mesh_size_ = 0;
} 


/*===========================================================================*/
// 
// PointCloudWrapper
//
// Description : Destructor
//
// Arguments   : none
//
PointCloudWrapper::~PointCloudWrapper()
{
}

/*===========================================================================*/
// 
// create_field
//
// Description : Adds a field to the vector of fields that can be applied to 
//               this mesh.  Returns a handle to the field.
//
// Arguments   : 
//
// string data_name - Name of the data contained in the field (i.e. "Pressure,
//                    "Temperature", etc.)
//
PCField PointCloudWrapper::create_field( string data_name )
{
  // Create a new PCField
  PCField new_field =
    scinew PointCloudField<double>(mesh_, 0); 

  // Set the name of the field
  new_field->set_property( "name", data_name, false );

  // Add the PCField to the vector of fields
  fields_.push_back( new_field );

  return fields_[fields_.size() - 1];
}

/*===========================================================================*/
// 
// get_field
//
// Description : Gets the field with the given value for data_name.  If no
//               such field exists, returns null (0).
//
// Arguments   : 
//
// string data_name - Name of the data contained in the field (i.e. "Pressure,
//                    "Temperature", etc.)
//
PCField PointCloudWrapper::get_field( string data_name )
{
  //cout << "(PointCloudWrapper::get_field) Inside" << endl;
 
  // Loop through the vector of fields and return a pointer to 
  int num_fields = fields_.size();

  //cout << "(PointCloudWrapper::get_field) num_fields = " << num_fields 
  //     << endl;

  string dn;
  for( int i = 0; i < num_fields; i++ ) 
  {
    
    if( (fields_[i])->get_property("name", dn) && dn == data_name )
    {
      return fields_[i];
    }
  }
  //cout << "(PointCloudWrapper::get_field) returning 0" << endl;
  return 0;
}

/*===========================================================================*/
// 
// update_node_value
//
// Description : If this node is already in the mesh, updates the value in the
//               appropriate field.  Otherwise, adds this node to the mesh and 
//               sets the value in the appropriate field.
//
// Arguments   :
//  
// string id - Unique id of the sensor
//             (i.e. "VirTelem2-0.000000-0.850000-0.650000)
//
// Point pt - 3D location of the sensor
// 
// double value - Data value at the point
//
// string data_name - Name of the field that this data corresponds to 
//                    (i.e. "Pressure")
//
void PointCloudWrapper::update_node_value( string id, Point pt, 
                                           double value, string data_name )
{
  // Get the appropriate field
  PCField fld = get_field( data_name );

  // If the field didn't exist, create a new one
  if( fld.get_rep() == 0 )
  {
    cout << "(PointCloudWrapper::update_node_value) Creating field "  
         << data_name << endl;
    fld = create_field( data_name );
  }

  // Get the field data (data at nodes) for the PointCloudField
  PointCloudField<double>::fdata_type &fdata = fld->fdata();

  // Check to make sure this field has a value for every node (and no extra
  // values)
  if( (int) fdata.size() != mesh_size_ )
  {
    cerr << "(PointCloudWrapper::update_node_value) ERROR: Field and mesh "
         << "sizes don't match" << endl;
  }

  // Check to see if this node is already in the mesh, and get the index
  // of the node
  Point p;
  int node_index = node_hash_[id.c_str()] - 1;

  cout << "(PointCloudWrapper::update_node_value) node_index = " 
       << node_index << endl;

  // If the node is not in the mesh, add it and its new value in the 
  // appropriate field
  if( node_index < 0 )
  {
    mesh_->add_node( pt );
    mesh_size_++;
    fdata.push_back( value );
    node_hash_[id.c_str()] = fdata.size();
  }
  else
  {
    fdata[node_index] == value;
  }
 
}

/*===========================================================================*/
// 
// freeze
//
// Description : Freezes the mesh and the appropriate field.
//
// Arguments   : 
//
// string data_name - Name of the data contained in the field (i.e. "Pressure,
//                    "Temperature", etc.)
//
void PointCloudWrapper::freeze( string data_name )
{
  mesh_->freeze();
  
  // Get the appropriate field
  PCField fld = get_field( data_name );

  if( fld.get_rep() != 0 )
  {
    fld->freeze();
  }
}

} // End namespace SCIRun
