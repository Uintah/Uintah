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


//    File   : FieldFrenet.h
//    Author : Allen Sanderson
//             School of Computing
//             University of Utah
//    Date   : April 2005

#if !defined(FieldFrenet_h)
#define FieldFrenet_h

#include <Core/Containers/Handle.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Math/Trig.h>


namespace SCIRun {

class FieldFrenetAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(FieldHandle& src,
			      int direction,
			      int axis) = 0;
  
  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *iftd,
					    const TypeDescription *oftd,
					    const unsigned int dim);
};

template< class IFIELD, class OFIELD >
class FieldFrenetAlgoT_1D : public FieldFrenetAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(FieldHandle& src,
			      int direction,
			      int axis);
};


template< class IFIELD, class OFIELD >
class FieldFrenetAlgoT_2D : public FieldFrenetAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(FieldHandle& src,
			      int direction,
			      int axis);
};


template< class IFIELD, class OFIELD >
class FieldFrenetAlgoT_3D : public FieldFrenetAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(FieldHandle& src,
			      int direction,
			      int axis);
};


template< class IFIELD, class OFIELD >
FieldHandle
FieldFrenetAlgoT_1D<IFIELD, OFIELD>::execute(FieldHandle& ifield_h,
					      int direction,
					      int axis)
{
  IFIELD *ifield = (IFIELD *) ifield_h.get_rep();
  typename IFIELD::mesh_handle_type imesh = ifield->get_typed_mesh();


  typename OFIELD::mesh_type *omesh = scinew typename OFIELD::mesh_type();
  omesh->copy_properties(imesh.get_rep());

  vector<unsigned int> dim;
  imesh->get_dim( dim );

  vector<unsigned int> min;
  imesh->get_min( min );

  omesh->set_dim( dim );
  omesh->set_min( min );

  OFIELD *ofield = scinew OFIELD(omesh, ifield->basis_order());
  ofield->copy_properties(ifield);

  FieldHandle ofield_h = ofield;

  typename IFIELD::mesh_type::Node::index_type inode1;
  typename IFIELD::mesh_type::Node::index_type inodeIndex;

  typename IFIELD::mesh_type::Node::iterator inodeItr;
  typename OFIELD::mesh_type::Node::iterator onodeItr;

  imesh->begin( inodeItr );
  omesh->begin( onodeItr );

  Point p, p_1, p1;
  Vector t, b, n;

#ifdef SET_POINT_DEFINED

  inode1 = 1;

  Vector *d1 = scinew Vector[ dim[0] ];
  Vector d2;

  
  for (unsigned int i=0; i<dim[0]; i++) {

    // Get the point and value at this location
    imesh->get_center(p, *inodeItr);
    omesh->set_point(p, *onodeItr);

    if( i == 0 ) {
	  
      inodeIndex_ = *inodeItr + inode1;
      imesh->get_center(p1,  inodeIndex);

      t = (Vector) (p1-p);

    } else if( i == dim[0]-1 ) {
      inodeIndex = *inodeItr - inode1;
      imesh->get_center(p_1,  inodeIndex);

      t = (Vector) (p-p_1);

    } else {
      inodeIndex = *inodeItr - inode1;
      imesh->get_center(p_1,  inodeIndex);

      inodeIndex = *inodeItr + inode1;
      imesh->get_center(p1,  inodeIndex);

      t = ((Vector) (p1-p) + (Vector) (p-p_1)) / 2.0;
    }

    if( direction == 0 ) {  // Tangent
      t.safe_normalize();
      ofield->set_value(t, *onodeItr);
    } else {
      d1[i] = t;
    }

    ++inodeItr;
    ++onodeItr;
  }

  if( direction > 0 ) {
    omesh->begin( onodeItr );

    for (unsigned int i=0; i<dim[0]; i++) {
      if( i == 0 ) {

	d2 = d1[i+1] - d1[i];

      } else if( i == dim[0]-1 ) {

	d2 = d1[i] - d1[i-1];

      } else {

	d2 = ((d1[i+1] - d1[i]) + (d1[i] - d1[i-1])) / 2.0;
      }

      b = Cross( d1[i], d2 );
      b.safe_normalize();

      if( direction == 2 ) {  // Binormal
	ofield->set_value(b, *onodeItr);

      } else if( direction == 1 ) {  // Normal
	t = d1[i];
	t.safe_normalize();

	n = Cross(b,t);
	n.safe_normalize();

	ofield->set_value(n, *onodeItr);
      }

      ++onodeItr;
    }
  }

  delete[] d1;

#endif       
  return ofield_h;
}


template< class IFIELD, class OFIELD >
FieldHandle
FieldFrenetAlgoT_2D<IFIELD, OFIELD>::execute(FieldHandle& ifield_h,
					      int direction,
					      int axis)
{
  IFIELD *ifield = (IFIELD *) ifield_h.get_rep();
  typename IFIELD::mesh_handle_type imesh = ifield->get_typed_mesh();


  typename OFIELD::mesh_type *omesh = scinew typename OFIELD::mesh_type();
  omesh->copy_properties(imesh.get_rep());

  vector<unsigned int> dim;
  imesh->get_dim( dim );

  vector<unsigned int> min;
  imesh->get_min( min );

  omesh->set_dim( dim );
  omesh->set_min( min );

  OFIELD *ofield = scinew OFIELD(omesh, ifield->basis_order());
  ofield->copy_properties(ifield);

  FieldHandle ofield_h = ofield;

  typename IFIELD::mesh_type::Node::index_type inode1;
  typename IFIELD::mesh_type::Node::index_type inodeIndex;

  typename IFIELD::mesh_type::Node::iterator inodeItr;
  typename OFIELD::mesh_type::Node::iterator onodeItr;

  imesh->begin( inodeItr );
  omesh->begin( onodeItr );

  Point p, p_1, p1;
  Vector t, b, n;

#ifdef SET_POINT_DEFINED

  Vector **d1 = scinew Vector*[ dim[1] ];
  Vector d2;

  if( axis == 0 )      { inode1.i_ = 1; inode1.j_ = 0;}
  else if( axis == 1 ) { inode1.i_ = 0; inode1.j_ = 1;}

  for (unsigned int j=0; j<dim[1]; j++) {

    d1[j] = scinew Vector[ dim[0] ];

    for (unsigned int i=0; i<dim[0]; i++) {

      // Get the point and value at this location
      imesh->get_center(p, *inodeItr);
      omesh->set_point(p, *onodeItr);

      if( ( axis == 0 && i == 0 ) ||
	  ( axis == 1 && j == 0 ) ) {
	  
	inodeIndex.i_ = (*inodeItr).i_ + inode1.i_;
	inodeIndex.j_ = (*inodeItr).j_ + inode1.j_;

	imesh->get_center(p1, inodeIndex);
	
	t = (Vector) (p1-p);
      }

      else if( ( axis == 1 && j == dim[1]-1 ) ||
	       ( axis == 0 && i == dim[0]-1 ) ) {

	inodeIndex.i_ = (*inodeItr).i_ - inode1.i_;
	inodeIndex.j_ = (*inodeItr).j_ - inode1.j_;
	
	imesh->get_center(p_1, inodeIndex);

	t = (Vector) (p-p_1);

      } else if( ( axis == 0 && 0<i && i<dim[0]-1 ) ||
		 ( axis == 1 && 0<j && j<dim[1]-1 ) ) {

	inodeIndex.i_ = (*inodeItr).i_ - inode1.i_;
	inodeIndex.j_ = (*inodeItr).j_ - inode1.j_;
	
	imesh->get_center(p_1, inodeIndex);

	inodeIndex.i_ = (*inodeItr).i_ + inode1.i_;
	inodeIndex.j_ = (*inodeItr).j_ + inode1.j_;

	imesh->get_center(p1, inodeIndex);
	
	t = ((Vector) (p1-p) + (Vector) (p-p_1)) / 2.0;
      }

      if( direction == 0 ) {  // Tangent
	t.safe_normalize();
	ofield->set_value(t, *onodeItr);
      } else {
	d1[j][i] = t;
      }

      ++inodeItr;
      ++onodeItr;
    }
  }

  if( direction > 0 ) {
    omesh->begin( onodeItr );

    for (unsigned int j=0; j<dim[1]; j++) {
      for (unsigned int i=0; i<dim[0]; i++) {

	if( ( axis == 0 && i == 0 ) ||
	    ( axis == 1 && j == 0 ) ) {
	  
	  d2 = d1[j+inode1.j_][i+inode1.i_] - d1[j][i];
	  
	} else if( ( axis == 0 && i == dim[0]-1 ) ||
		   ( axis == 1 && j == dim[1]-1 ) ) {
	  
	  d2 = d1[j][i] - d1[j-inode1.j_][i-inode1.i_];
	  
	} else if( ( axis == 0 && 0<i && i<dim[0]-1 ) ||
		   ( axis == 1 && 0<j && j<dim[1]-1 ) ) {
	  
	  d2 = ((d1[j+inode1.j_][i+inode1.i_] - d1[j][i]) +
		(d1[j][i] - d1[j-inode1.j_][i-inode1.i_])) / 2.0;
	}
	
	b = Cross( d1[j][i], d2 );
	b.safe_normalize();
	
	if( direction == 2 ) {  // Binormal
	  ofield->set_value(b, *onodeItr);
	  
	} else if( direction == 1 ) {  // Normal
	  t = d1[j][i];
	  t.safe_normalize();
	  
	  n = Cross(b,t);
	  n.safe_normalize();
	  
	  ofield->set_value(n, *onodeItr);
	}
	
	++onodeItr;
      }
    }
  }
#endif       
  return ofield_h;
}


template< class IFIELD, class OFIELD >
FieldHandle
FieldFrenetAlgoT_3D<IFIELD, OFIELD>::execute(FieldHandle& ifield_h,
					      int direction,
					      int axis)
{
  IFIELD *ifield = (IFIELD *) ifield_h.get_rep();
  typename IFIELD::mesh_handle_type imesh = ifield->get_typed_mesh();


  typename OFIELD::mesh_type *omesh = scinew typename OFIELD::mesh_type();
  omesh->copy_properties(imesh.get_rep());

  vector<unsigned int> dim;
  imesh->get_dim( dim );

  vector<unsigned int> min;
  imesh->get_min( min );

  omesh->set_dim( dim );
  omesh->set_min( min );

  OFIELD *ofield = scinew OFIELD(omesh, ifield->basis_order());
  ofield->copy_properties(ifield);

  FieldHandle ofield_h = ofield;

  typename IFIELD::mesh_type::Node::index_type inodeTangent;
  typename IFIELD::mesh_type::Node::index_type inodeIndex;

  typename IFIELD::mesh_type::Node::iterator inodeItr;
  typename OFIELD::mesh_type::Node::iterator onodeItr;

  imesh->begin( inodeItr );
  omesh->begin( onodeItr );

  Point p, p_1, p1;
  Vector t, b;
 
#ifdef SET_POINT_DEFINED

  inodeTangent.i_ = inodeTangent.j_ = inodeTangent.k_ = 0;

  if( axis == 2 )      inodeTangent.k_ = 1;
  else if( axis == 1 ) inodeTangent.j_ = 1;
  else if( axis == 0 ) inodeTangent.i_ = 1;

  for (unsigned int k=0; k<dim[2]; k++) {	
    for (unsigned int j=0; j<dim[1]; j++) {
      for (unsigned int i=0; i<dim[0]; i++) {

	// Get the point and value at this location
	imesh->get_center(p, *inodeItr);
	omesh->set_point(p, *onodeItr);

	if( ( axis == 2 && 0<k && k<dim[2]-1 ) ||
	    ( axis == 1 && 0<j && j<dim[1]-1 ) ||
	    ( axis == 0 && 0<i && i<dim[0]-1 ) ) {

	  inodeIndex.i_ = (*inodeItr).i_ + inodeTangent.i_;
	  inodeIndex.j_ = (*inodeItr).j_ + inodeTangent.j_;
	  inodeIndex.k_ = (*inodeItr).k_ + inodeTangent.k_;
	  imesh->get_center(p1,  inodeIndex);

	  inodeIndex.i_ = (*inodeItr).i_ - inodeTangent.i_;
	  inodeIndex.j_ = (*inodeItr).j_ - inodeTangent.j_;
	  inodeIndex.k_ = (*inodeItr).k_ - inodeTangent.k_;
	  imesh->get_center(p_1, inodeIndex);
	  
	  t = ((Vector) (p-p_1) + (Vector) (p1-p) ) / 2.0;
	}

	else if( ( axis == 2 && k == 0 ) ||
		 ( axis == 1 && j == 0 ) ||
		 ( axis == 0 && i == 0 ) ) {
	  
	  inodeIndex.i_ = (*inodeItr).i_ + inodeTangent.i_;
	  inodeIndex.j_ = (*inodeItr).j_ + inodeTangent.j_;
	  inodeIndex.k_ = (*inodeItr).k_ + inodeTangent.k_;
	  imesh->get_center(p1,  inodeIndex);

	  t = (Vector) (p1-p);
	}

	else if( ( axis == 2 && k == dim[2]-1 ) ||
		 ( axis == 1 && j == dim[1]-1 ) ||
		 ( axis == 0 && i == dim[0]-1 ) ) {

	  inodeIndex.i_ = (*inodeItr).i_ - inodeTangent.i_;
	  inodeIndex.j_ = (*inodeItr).j_ - inodeTangent.j_;
	  inodeIndex.k_ = (*inodeItr).k_ - inodeTangent.k_;
	  imesh->get_center(p_1, inodeIndex);
	  
	  t = (Vector) (p-p_1);
	}
	  
	if( direction == 0 )
	  ofield->set_value(t, *onodeItr);
	else if( direction == 1 )
	  ofield->set_value( Cross(b,t), *onodeItr);
	else if( direction == 2 )
	  ofield->set_value(b, *onodeItr);

	++inodeItr;
	++onodeItr;
      }
    }
  }
#endif       
  return ofield_h;
}

} // end namespace SCIRun

#endif // FieldFrenet_h
