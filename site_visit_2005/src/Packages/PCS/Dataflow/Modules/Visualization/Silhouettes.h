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
 *  Silhouettes.h:
 *
 *  Written by:
 *   Allen Sanderson
 *   School of Computering
 *   University of Utah
 *   September 2004
 *
 *  Copyright (C) 2004 SCI Group
 */

#if !defined(Silhouettes_h)
#define Silhouettes_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>

#include <Core/Geom/View.h>
#include <Core/Datatypes/Mesh.h>
#include <Core/Datatypes/CurveField.h>

#include <Core/Geom/GeomLine.h>

namespace PCS {

using namespace SCIRun;

class SilhouettesAlgo : public DynamicAlgoBase
{
public:
  virtual void execute(FieldHandle& src,
		       View &view,
		       bool build_field,
		       bool build_geom) = 0;
  
  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *ftd,
					    const TypeDescription *ttd);

  virtual FieldHandle get_field() = 0;
  virtual GeomHandle get_geom( double &isoval ) = 0;
};


template< class IFIELD, class OFIELD, class OMESH >
class SilhouettesAlgoT : public SilhouettesAlgo
{
public:
  //! virtual interface. 
  virtual void execute(FieldHandle& src,
		       View &view,
		       bool build_field,
		       bool build_geom );

  FieldHandle get_field();
  GeomHandle get_geom( double &isoval );

private:
  OMESH     *omesh_;

  // Storage for the field nodes and assocaited values. 
  vector<typename OFIELD::value_type> values_;
  vector<typename OFIELD::mesh_type::Node::index_type> nodes_;

  GeomLines *lines_;

  double isoval_;
};

// The goal dump all all of the unshared edges into a new field.
template< class IFIELD, class OFIELD, class OMESH >
void
SilhouettesAlgoT<IFIELD, OFIELD, OMESH >::execute(FieldHandle& field_h,
						  View &view,
						  bool build_field,
						  bool build_geom )
{
  // Get the input field and mesh.
  IFIELD *ifield = (IFIELD *) field_h.get_rep();
  typename IFIELD::mesh_handle_type imesh = ifield->get_typed_mesh();

  // Create the output mesh.
  nodes_.clear();
  values_.clear();

  if( build_field )
    omesh_ = scinew OMESH();
  else
    omesh_ = 0;

  // Create the output geometry.
  if( build_geom )
    lines_ = scinew GeomLines;
  else
    lines_ = 0;

  // Iterators and arrays.
  typename IFIELD::mesh_type::Face::iterator in, end;
  typename IFIELD::mesh_type::Edge::array_type edgeArray;
  typename IFIELD::mesh_type::Node::array_type nodeArray;
  typename IFIELD::mesh_type::Face::index_type neighbor;

  imesh->begin( in );
  imesh->end( end );

  // Make sure all of the edges are up todate.
  imesh->synchronize(Mesh::EDGES_E);
  imesh->synchronize(Mesh::EDGE_NEIGHBORS_E);

  // Loop through all of the faces.
  while (in != end) {

    // For each face get the edges.
    imesh->get_edges(edgeArray, *in);

    // Loop through all of the edges.
    for( unsigned int i=0; i<edgeArray.size(); i++ ) {

      // For each edge and face see if there is neighboring face.
      if( !imesh->get_neighbor( neighbor, *in, edgeArray[i] ) ) {

	// Get the node indexs for this edge.
	imesh->get_nodes(nodeArray, edgeArray[i]);

	Point p0, p1;

	// Get the points for each node index.
	imesh->get_point( p0, nodeArray[0] );
	imesh->get_point( p1, nodeArray[1] );

	if( build_field ) {
	  // Add the points to the new mesh.
	  typename OFIELD::mesh_type::Node::index_type n0 = omesh_->add_node(p0);
	  typename OFIELD::mesh_type::Node::index_type n1 = omesh_->add_node(p1);

	  // Add the edge to the new mesh.
	  omesh_->add_edge( n0, n1 );

	  // Save the nodes for each value for the field.
	  nodes_.push_back( n0 );
	  nodes_.push_back( n1 );

	  // Save the values for the field.
	  values_.push_back( ifield->value( nodeArray[0] ) );
	  values_.push_back( ifield->value( nodeArray[1] ) );
	}

	// Build the edges for the geometry.
	if( build_geom ) {
	  isoval_ = ifield->value( nodeArray[0] );
	  lines_->add( p0, p1 );
	}
      }
    }

    ++in;
  }
}

template< class IFIELD, class OFIELD, class OMESH >
FieldHandle
SilhouettesAlgoT<IFIELD, OFIELD, OMESH >::get_field() {

  // Create the field after the mesh so that the data is allocated properly.
  // Uncomment for 1.24
  OFIELD *ofield = scinew OFIELD(omesh_, 1);

  // Uncomment for 1.22
  //OFIELD *ofield = scinew OFIELD(omesh_, Field::NODE);

  // Stuff all of the values into the field.
  for( unsigned int i; i<values_.size(); i++ )
    ofield->set_value( values_[i], nodes_[i] );

  ofield->freeze();

  return FieldHandle( ofield );
}

template< class IFIELD, class OFIELD, class OMESH >
GeomHandle
SilhouettesAlgoT<IFIELD, OFIELD, OMESH >::get_geom( double &isoval ) {

  isoval = isoval_;

  return GeomHandle( lines_ );
}


} // end namespace PCS

#endif // Silhouettes_h
