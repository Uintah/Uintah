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

//    File   : NrrdToField.h
//    Author : Allen Sanderson
//             School of Computing
//             University of Utah
//    Date   : July 2003

#if !defined(NrrdToField_h)
#define NrrdToField_h

#include <Dataflow/Network/Module.h>

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>

#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Tensor.h>

#include <Core/Containers/StringUtil.h>

#include <Core/Datatypes/NrrdData.h>

#include <teem/ten.h>

#include <iostream>

namespace SCITeem {

using namespace SCIRun;
using std::cerr;
class NrrdToFieldMeshAlgo : public DynamicAlgoBase
{
public:
  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info( const std::string topoStr,
					     const TypeDescription *mtd,
					     const unsigned int ptype,
					     const unsigned int ctype);

};



class RegularNrrdToFieldMeshAlgo : public NrrdToFieldMeshAlgo
{
public:
  virtual void execute(MeshHandle& mHandle,
		       NrrdDataHandle dataH, int data_size);
};


template< class MESH, class DNTYPE, class CNTYPE>
class RegularNrrdToFieldMeshAlgoT : 
  public RegularNrrdToFieldMeshAlgo
{
public:
  virtual void execute(MeshHandle& mHandle,
		       NrrdDataHandle dataH, int data_size) = 0;
};

template< class MESH, class DNTYPE, class CNTYPE>
void
RegularNrrdToFieldMeshAlgoT< MESH, DNTYPE, CNTYPE>::
execute(MeshHandle& mHandle, NrrdDataHandle dataH, int data_size)
{
  Point minpt, maxpt;

  DNTYPE *ptr = (DNTYPE *)(dataH->nrrd->data);

  int rank = data_size;

  float xVal = 0, yVal = 0, zVal = 0;

  if( rank >= 1 ) xVal = ptr[0];
  if( rank >= 2 ) yVal = ptr[1];
  if( rank >= 3 ) zVal = ptr[2];

  minpt = Point( xVal, yVal, zVal );

  xVal = 0; yVal = 0; zVal = 0;

  if( rank >= 1 ) xVal = ptr[rank + 0];
  if( rank >= 2 ) yVal = ptr[rank + 1];
  if( rank >= 3 ) zVal = ptr[rank + 2];

  maxpt = Point( xVal, yVal, zVal );

  MESH *imesh = (MESH *) mHandle.get_rep();

  vector<unsigned int> array;
  imesh->get_dim(array);

  Transform trans;

  if( array.size() == 1 )
    trans.pre_scale(Vector(1.0 / (array[0]-1.0),
			   1.0,
			   1.0));
  
  else if( array.size() == 2 )
    trans.pre_scale(Vector(1.0 / (array[0]-1.0),
			   1.0 / (array[1]-1.0),
			   1.0));
  
  else  if( array.size() == 3 )
    trans.pre_scale(Vector(1.0 / (array[0]-1.0),
			   1.0 / (array[1]-1.0),
			   1.0 / (array[2]-1.0)));
  
  trans.pre_scale(maxpt - minpt);
  
  trans.pre_translate(minpt.asVector());
  trans.compute_imat();

  imesh->set_transform(trans);
}


class StructuredNrrdToFieldMeshAlgo : public NrrdToFieldMeshAlgo
{
public:
  virtual void execute(MeshHandle& mHandle,
		       NrrdDataHandle pointsH,
		       int idim, int jdim, int kdim) = 0;
};


template< class MESH, class PNTYPE, class CNTYPE >
class StructuredNrrdToFieldMeshAlgoT : 
  public StructuredNrrdToFieldMeshAlgo
{
public:

  virtual void execute(MeshHandle& mHandle,
		       NrrdDataHandle pointsH,
		       int idim, int jdim, int kdim);
};


template< class MESH, class PNTYPE, class CNTYPE >
void
StructuredNrrdToFieldMeshAlgoT< MESH, PNTYPE, CNTYPE >::
execute(MeshHandle& mHandle,
	NrrdDataHandle pointsH,
	int idim, int jdim, int kdim)
{
  MESH *imesh = (MESH *) mHandle.get_rep();
  typename MESH::Node::iterator inodeItr;
  
  imesh->begin( inodeItr );
  
  register int i, j, k;
  int rank = pointsH->nrrd->axis[0].size;
  
  PNTYPE *ptr = (PNTYPE *)(pointsH->nrrd->data);
  for( i=0; i<idim; i++ ) {
    for( j=0; j<jdim; j++ ) {
      for( k=0; k<kdim; k++ ) {
	
	int index = (i * jdim + j) * kdim + k;
	
	float xVal = 0, yVal = 0, zVal = 0;
	
	// Mesh
	if( rank >= 1 ) xVal = ptr[index*rank + 0];
	if( rank >= 2 ) yVal = ptr[index*rank + 1];
	if( rank >= 3 ) zVal = ptr[index*rank + 2];
	
	imesh->set_point(Point(xVal, yVal, zVal), *inodeItr);
	
	++inodeItr;
      }
    }
  }
}



class UnstructuredNrrdToFieldMeshAlgo : 
  public NrrdToFieldMeshAlgo
{
public:
  virtual void execute(MeshHandle& mHandle,
		       NrrdDataHandle pointsH,
		       NrrdDataHandle connectH,
		       unsigned int connectivity, int which) = 0;
};

template< class MESH, class PNTYPE, class CNTYPE >
class UnstructuredNrrdToFieldMeshAlgoT : 
  public UnstructuredNrrdToFieldMeshAlgo
{
public:
  virtual void execute(MeshHandle& mHandle,
		       NrrdDataHandle pointsH,
		       NrrdDataHandle connectH,
		       unsigned int connectivity, int which);
};


template< class MESH, class PNTYPE, class CNTYPE >
void
UnstructuredNrrdToFieldMeshAlgoT< MESH, PNTYPE, CNTYPE >::
execute(MeshHandle& mHandle,
	NrrdDataHandle pointsH,
	NrrdDataHandle connectH,
	unsigned int connectivity, int which)
{
  MESH *imesh = (MESH *) mHandle.get_rep();
  typename MESH::Node::iterator inodeItr;
  bool single_element = false;
  if (connectH != 0 && connectH->nrrd->dim == 1)
    single_element = true;

  imesh->begin( inodeItr );

  int npts = pointsH->nrrd->axis[1].size;
  int rank = pointsH->nrrd->axis[0].size;

  PNTYPE *pPtr = (PNTYPE *)(pointsH->nrrd->data);
  
  for( int index=0; index<npts; index++ ) {
    float xVal = 0, yVal = 0, zVal = 0;
    
    // Mesh
    if( rank >= 1 ) xVal = pPtr[index*rank + 0];
    if( rank >= 2 ) yVal = pPtr[index*rank + 1];
    if( rank >= 3 ) zVal = pPtr[index*rank + 2];
    imesh->add_point( Point(xVal, yVal, zVal) );
  }

  if( connectivity > 0 ) {

    CNTYPE *cPtr = cPtr = (CNTYPE *)(connectH->nrrd->data);

    int nelements = 0;
    if (single_element)
      nelements = 1;
    else if (which == 0) {
      // p x n
      nelements = connectH->nrrd->axis[1].size;
    } else {
      // n x p
      nelements = connectH->nrrd->axis[0].size;
    }

    typename MESH::Node::array_type array(connectivity);

    if (which == 0) {
      // p x n
      for( int i=0; i<nelements; i++ ) {
	for( unsigned int j=0; j<connectivity; j++ ) {
	  array[j] = (int) cPtr[i*connectivity+j];
	}
	imesh->add_elem( array );
      }
    } else {
      // n x p
      for( int i=0; i<nelements; i++ ) {
	for( unsigned int j=0; j<connectivity; j++ ) {
	  array[j] = (int) cPtr[j*connectivity+i];
	}
	imesh->add_elem( array );
      }
    }
  }
}




class NrrdToFieldFieldAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(MeshHandle& mHandle,
			      NrrdDataHandle dataH,
			      int build_eigens,
			      int idim, int jdim, int kdim, int permute) = 0;
  
  virtual FieldHandle execute(MeshHandle& mHandle,
			      NrrdDataHandle dataH,
			      int build_eigens) = 0;
  
   //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *mtd,
					    const std::string fname,
					    const unsigned int ntype,
					    int rank);
};

template< class FIELD, class MESH, class NTYPE >
class NrrdToFieldFieldAlgoScalar : public NrrdToFieldFieldAlgo
{
public:
  //! virtual interface.
  virtual FieldHandle execute(MeshHandle& mHandle,
			      NrrdDataHandle dataH,
			      int build_eigens,
			      int idim, int jdim, int kdim, int permute);

  virtual FieldHandle execute(MeshHandle& mHandle,
			      NrrdDataHandle dataH,
			      int build_eigens);  
};


template< class FIELD, class MESH, class NTYPE >
FieldHandle
NrrdToFieldFieldAlgoScalar<FIELD, MESH, NTYPE>::
execute(MeshHandle& mHandle,
	NrrdDataHandle dataH,
	int build_eigens,
	int idim, int jdim, int kdim, int permute)
{
  MESH *imesh = (MESH *) mHandle.get_rep();
  FIELD *ifield = 0;
  
  if (dataH != 0) {
    // determine if nrrd is unknown, node or cell centered
    int data_center = nrrdCenterUnknown;
    for (int a = 0; a<dataH->nrrd->dim; a++) {
      if (dataH->nrrd->axis[a].center != nrrdCenterUnknown)
	data_center = dataH->nrrd->axis[a].center;
    }
    if (data_center == nrrdCenterCell) {
      ifield = (FIELD *) scinew FIELD((MESH *) imesh, 0);
      typename FIELD::mesh_type::Elem::iterator ielemItr;
      
      imesh->begin( ielemItr );
      
      NTYPE *ptr = (NTYPE *)(dataH->nrrd->data);
      
      register int i, j, k, index;
      
      for( i=0; i<idim; i++ ) {
	for( j=0; j<jdim; j++ ) {
	  for( k=0; k<kdim; k++ ) {
	    
	    if (permute)
	      index = (k * jdim + j) * idim + i;
	    else
	      index = (i * jdim + j) * kdim + k;
	    
	    // Value
	    ifield->set_value( ptr[index], *ielemItr);
	    
	    ++ielemItr;
	  }
	}
      }    
    }
    else {
      ifield = (FIELD *) scinew FIELD((MESH *) imesh, 1);
      typename FIELD::mesh_type::Node::iterator inodeItr;
      
      imesh->begin( inodeItr );
      
      NTYPE *ptr = (NTYPE *)(dataH->nrrd->data);
      
      register int i, j, k, index;
      
      for( i=0; i<idim; i++ ) {
	for( j=0; j<jdim; j++ ) {
	  for( k=0; k<kdim; k++ ) {
	    
	    if (permute)
	      index = (k * jdim + j) * idim + i;
	    else
	      index = (i * jdim + j) * kdim + k;
	    
	    // Value
	    ifield->set_value( ptr[index], *inodeItr);
	    
	    ++inodeItr;
	  }
	}
      }

    }
  } else {
    ifield = (FIELD *) scinew FIELD((MESH *) imesh, 1);
  }
  
  return FieldHandle( ifield );
}



template< class FIELD, class MESH, class NTYPE >
FieldHandle
NrrdToFieldFieldAlgoScalar<FIELD, MESH, NTYPE>::
execute(MeshHandle& mHandle,
	NrrdDataHandle dataH,
	int build_eigens)

{
  MESH *imesh = (MESH *) mHandle.get_rep();
  FIELD *ifield = 0;

  FieldHandle fH = 0;

  if (dataH != 0) {
    // determine if nrrd is unknown, node or cell centered
    int data_center = nrrdCenterUnknown;
    for (int a = 0; a<dataH->nrrd->dim; a++) {
      if (dataH->nrrd->axis[a].center != nrrdCenterUnknown)
	data_center = dataH->nrrd->axis[a].center;
    }
    if (data_center == nrrdCenterCell) {
      ifield = (FIELD *) scinew FIELD((MESH *) imesh, 0);

      typename FIELD::mesh_type::Elem::iterator ielemItr, end;
      
      imesh->begin( ielemItr );
      imesh->end( end );
      
      int i = 0;
      NTYPE *ptr = (NTYPE *)(dataH->nrrd->data);
      
      while (ielemItr != end) {
	ifield->set_value( ptr[i], *ielemItr);
	++ielemItr;
	i++;
      }
    }
    else {
      ifield = (FIELD *) scinew FIELD((MESH *) imesh, 1);

      typename FIELD::mesh_type::Node::iterator inodeItr, end;
      
      imesh->begin( inodeItr );
      imesh->end( end );
      
      int i = 0;
      NTYPE *ptr = (NTYPE *)(dataH->nrrd->data);
      
      while (inodeItr != end) {
	ifield->set_value( ptr[i], *inodeItr);
	++inodeItr;
	i++;
      }
    }

    // set transform if one of the nrrd properties
    fH = ifield;
    const string meshstr =
      fH->get_type_description(0)->get_name().substr(0, 6);
    
    if (!(imesh->is_editable() && meshstr != "Struct"))
      {
	string trans_string;
	if (dataH->get_property("Transform", trans_string) && trans_string != "Unknown") {
	  double t[16];
	  Transform trans;
	  int old_index=0, new_index=0;
	  for(int i=0; i<16; i++) {
	    new_index = trans_string.find(" ", old_index);
	    string temp = trans_string.substr(old_index, new_index-old_index);
	    old_index = new_index+1;
	    string_to_double(temp, t[i]);
	  }
	  trans.set(t);
	  imesh->transform(trans);
	} 
      }	        
  } else {
    ifield = (FIELD *) scinew FIELD((MESH *) imesh, 1);
    fH = ifield;
  }
  
  return fH;
}





template< class FIELD, class MESH, class NTYPE >
class NrrdToFieldFieldAlgoVector : public NrrdToFieldFieldAlgo
{
public:
  //! virtual interface.
  virtual FieldHandle execute(MeshHandle& mHandle,
			      NrrdDataHandle dataH,
			      int build_eigens,
			      int idim, int jdim, int kdim, int permute);

  virtual FieldHandle execute(MeshHandle& mHandle,
			      NrrdDataHandle dataH,
			      int build_eigens);  
};


template< class FIELD, class MESH, class NTYPE >
FieldHandle
NrrdToFieldFieldAlgoVector<FIELD, MESH, NTYPE>::
execute(MeshHandle& mHandle,
	NrrdDataHandle dataH,
	int build_eigens,
	int idim, int jdim, int kdim, int permute)
{
  MESH *imesh = (MESH *) mHandle.get_rep();
  FIELD *ifield = 0;

  if (dataH != 0 ) {
    int data_center = nrrdCenterUnknown;
    for (int a = 0; a<dataH->nrrd->dim; a++) {
      if (dataH->nrrd->axis[a].center != nrrdCenterUnknown)
	data_center = dataH->nrrd->axis[a].center;
    }
    if (data_center == nrrdCenterCell) {
      ifield = (FIELD *) scinew FIELD((MESH *) imesh, 0);

      typename FIELD::mesh_type::Elem::iterator ielemItr;
      
      imesh->begin( ielemItr );
      
      register int i, j, k, index;
      
      NTYPE *ptr = (NTYPE *)(dataH->nrrd->data);
      
      for( i=0; i<idim; i++ ) {
	for( j=0; j<jdim; j++ ) {
	  for( k=0; k<kdim; k++ ) {
	    
	    if( permute )
	      index = (k * jdim + j) * idim + i;
	    else 
	      index = (i * jdim + j) * kdim + k;
	    
	    ifield->set_value( Vector( ptr[index*3  ],
				       ptr[index*3+1],
				       ptr[index*3+2] ), *ielemItr);
	    
	    ++ielemItr;
	  }
	}
      }
    }
    else {
      ifield = (FIELD *) scinew FIELD((MESH *) imesh, 1);

      typename FIELD::mesh_type::Node::iterator inodeItr;
      
      imesh->begin( inodeItr );
      
      register int i, j, k, index;
      
      NTYPE *ptr = (NTYPE *)(dataH->nrrd->data);
      
      for( i=0; i<idim; i++ ) {
	for( j=0; j<jdim; j++ ) {
	  for( k=0; k<kdim; k++ ) {
	    
	    if( permute )
	      index = (k * jdim + j) * idim + i;
	    else 
	      index = (i * jdim + j) * kdim + k;
	    
	    ifield->set_value( Vector( ptr[index*3  ],
				       ptr[index*3+1],
				       ptr[index*3+2] ), *inodeItr);
	    
	    ++inodeItr;
	  }
	}
      }
    }
  } else {
    ifield = (FIELD *) scinew FIELD((MESH *) imesh, 1);
  }

  return FieldHandle( ifield );
  
}

template< class FIELD, class MESH, class NTYPE >
FieldHandle
NrrdToFieldFieldAlgoVector<FIELD, MESH, NTYPE>::
execute(MeshHandle& mHandle,
	NrrdDataHandle dataH,
	int build_eigens)
{

  MESH *imesh = (MESH *) mHandle.get_rep();
  FIELD *ifield = 0;
  FieldHandle fH = 0;

  if (dataH != 0) {
    // determine if nrrd is unknown, node or cell centered
    int data_center = nrrdCenterUnknown;
    for (int a = 0; a<dataH->nrrd->dim; a++) {
      if (dataH->nrrd->axis[a].center != nrrdCenterUnknown)
	data_center = dataH->nrrd->axis[a].center;
    }
    if (data_center == nrrdCenterCell) {
      ifield = (FIELD *) scinew FIELD((MESH *) imesh, 0);

      typename FIELD::mesh_type::Elem::iterator ielemItr, end;
      
      imesh->begin( ielemItr );
      imesh->end( end );
      
      NTYPE *ptr = (NTYPE *)(dataH->nrrd->data);
      int i = 0;
      while (ielemItr != end) {
	NTYPE x = *ptr;
	++ptr;
	NTYPE y = *ptr;
	++ptr;
	NTYPE z = *ptr;
	++ptr;
	ifield->set_value( Vector( x, y, z ),
			   *ielemItr);
	++ielemItr;
	i++;
      }

    }
    else {
      ifield = (FIELD *) scinew FIELD((MESH *) imesh, 1);

      typename FIELD::mesh_type::Node::iterator inodeItr, end;
      
      imesh->begin( inodeItr );
      imesh->end( end );
      
      NTYPE *ptr = (NTYPE *)(dataH->nrrd->data);
      int i = 0;
      while (inodeItr != end) {
	NTYPE x = *ptr;
	++ptr;
	NTYPE y = *ptr;
	++ptr;
	NTYPE z = *ptr;
	++ptr;
	ifield->set_value( Vector( x, y, z ),
			   *inodeItr);
	++inodeItr;
	i++;
      }
    }

    // set transform if one of the nrrd properties
    fH = ifield;
    const string meshstr =
      fH->get_type_description(0)->get_name().substr(0, 6);
    
    if (!(imesh->is_editable() && meshstr != "Struct"))
      {
	string trans_string;
	if (dataH->get_property("Transform", trans_string) && trans_string != "Unknown") {
	  double t[16];
	  Transform trans;
	  int old_index=0, new_index=0;
	  for(int i=0; i<16; i++) {
	    new_index = trans_string.find(" ", old_index);
	    string temp = trans_string.substr(old_index, new_index-old_index);
	    old_index = new_index+1;
	    string_to_double(temp, t[i]);
	  }
	  trans.set(t);
	  imesh->transform(trans);
	} 
      }	  
  } else {
    ifield = (FIELD *) scinew FIELD((MESH *) imesh, 1);
    fH = ifield;
  }

  return fH;
}




template< class FIELD, class MESH, class NTYPE >
class NrrdToFieldFieldAlgoTensor : public NrrdToFieldFieldAlgo
{
public:
  //! virtual interface.
  virtual FieldHandle execute(MeshHandle& mHandle,
			      NrrdDataHandle dataH,
			      int build_eigens,
			      int idim, int jdim, int kdim, int permute);

  virtual FieldHandle execute(MeshHandle& mHandle,
			      NrrdDataHandle dataH,
			      int build_eigens);  
};


template< class FIELD, class MESH, class NTYPE >
FieldHandle
NrrdToFieldFieldAlgoTensor<FIELD, MESH, NTYPE>::
execute(MeshHandle& mHandle,
	NrrdDataHandle dataH,
	int build_eigens,
	int idim, int jdim, int kdim, int permute)
{
  MESH *imesh = (MESH *) mHandle.get_rep();
  FIELD *ifield = 0;
  
  if (dataH != 0 ) {
    // determine if nrrd is unknown, node or cell centered
    int data_center = nrrdCenterUnknown;
    for (int a = 0; a<dataH->nrrd->dim; a++) {
      if (dataH->nrrd->axis[a].center != nrrdCenterUnknown)
	data_center = dataH->nrrd->axis[a].center;
    }
    if (data_center == nrrdCenterCell) {
      ifield = (FIELD *) scinew FIELD((MESH *) imesh, 0);

      typename FIELD::mesh_type::Elem::iterator iter, end;
      
      imesh->begin( iter );
      imesh->end( end );
      
      register int i, j, k, index;
      
      NTYPE *ptr = (NTYPE *)(dataH->nrrd->data);
      for( i=0; i<idim; i++ ) {
	for( j=0; j<jdim; j++ ) {
	  for( k=0; k<kdim; k++ ) {
	    if( permute )
	      index = (k * jdim + j) * idim + i;
	    else 
	      index = (i * jdim + j) * kdim + k;
	    
	    Tensor tmp;
	    
	    if (nrrdKindSize( dataH->nrrd->axis[0].kind) == 6) {
	      // 3D symmetric tensor
	      
	      if (build_eigens == 1) {
		float eval[3], evec[9], eval_scl[3], evec_scl[9];
		tenEigensolve_f(eval, evec, ptr);
		
		//float scl = ptr[0] > 0.5;
		float scl = 1.0;
		for (int cc=0; cc<3; cc++) {
		  ELL_3V_SCALE(evec_scl+3*cc, scl, evec+3*cc);
		  eval_scl[cc] = scl*eval[cc];
		}
		Vector e1(evec_scl[0], evec_scl[1], evec_scl[2]);
		Vector e2(evec_scl[3], evec_scl[4], evec_scl[5]);
		Vector e3(evec_scl[6], evec_scl[7], evec_scl[8]);
		
		tmp.set_outside_eigens(e1, e2, e3, eval_scl[0], eval_scl[1], eval_scl[2]);
		tmp.mat_[0][0] = (*ptr); 
		++ptr;
		tmp.mat_[0][1] = (*ptr); 
		tmp.mat_[1][0] = (*ptr);
		++ptr;
		tmp.mat_[0][2] = (*ptr); 
		tmp.mat_[2][0] = (*ptr);
		++ptr;
		tmp.mat_[1][1] = (*ptr); 
		++ptr;
		tmp.mat_[1][2] = (*ptr); 
		tmp.mat_[2][1] = (*ptr);
		++ptr;
		tmp.mat_[2][2] = (*ptr); 
		++ptr;
		ifield->set_value( tmp, *iter);
	      } else {
		tmp.mat_[0][0] = (*ptr); 
		++ptr;
		tmp.mat_[0][1] = (*ptr); 
		tmp.mat_[1][0] = (*ptr);
		++ptr;
		tmp.mat_[0][2] = (*ptr); 
		tmp.mat_[2][0] = (*ptr);
		++ptr;
		tmp.mat_[1][1] = (*ptr); 
		++ptr;
		tmp.mat_[1][2] = (*ptr); 
		tmp.mat_[2][1] = (*ptr);
		++ptr;
		tmp.mat_[2][2] = (*ptr); 
		++ptr;
		ifield->set_value( tmp, *iter);
	      }
	    } else if (nrrdKindSize( dataH->nrrd->axis[0].kind) == 7) {
	      // 3D symmetric tensor with mask
	      
	      if (build_eigens == 1) {
		float eval[3], evec[9], eval_scl[3], evec_scl[9];
		tenEigensolve_f(eval, evec, ptr);
		
		float scl = ptr[0] > 0.5;
		for (int cc=0; cc<3; cc++) {
		  ELL_3V_SCALE(evec_scl+3*cc, scl, evec+3*cc);
		  eval_scl[cc] = scl*eval[cc];
		}
		Vector e1(evec_scl[0], evec_scl[1], evec_scl[2]);
		Vector e2(evec_scl[3], evec_scl[4], evec_scl[5]);
		Vector e3(evec_scl[6], evec_scl[7], evec_scl[8]);
		
		tmp.set_outside_eigens(e1, e2, e3, eval_scl[0], eval_scl[1], eval_scl[2]);
		++ptr; // skip first value (confidence)
		tmp.mat_[0][0] = (*ptr); 
		++ptr;
		tmp.mat_[0][1] = (*ptr); 
		tmp.mat_[1][0] = (*ptr);
		++ptr;
		tmp.mat_[0][2] = (*ptr); 
		tmp.mat_[2][0] = (*ptr);
		++ptr;
		tmp.mat_[1][1] = (*ptr); 
		++ptr;
		tmp.mat_[1][2] = (*ptr); 
		tmp.mat_[2][1] = (*ptr);
		++ptr;
		tmp.mat_[2][2] = (*ptr); 
		++ptr;
		ifield->set_value( tmp, *iter);
	      } else {
		// if mask < 0.5, all tensor values are 0
		if (ptr[index*3] < 0.5) {
		  for (int x=0; x<3; x++) 
		    for (int y=0; y<3; y++)
		      tmp.mat_[x][y]=0;
		  
		} else {
		  ++ptr; // skip first value (confidence)
		  tmp.mat_[0][0] = (*ptr); 
		  ++ptr;
		  tmp.mat_[0][1] = (*ptr); 
		  tmp.mat_[1][0] = (*ptr);
		  ++ptr;
		  tmp.mat_[0][2] = (*ptr); 
		  tmp.mat_[2][0] = (*ptr);
		  ++ptr;
		  tmp.mat_[1][1] = (*ptr); 
		  ++ptr;
		  tmp.mat_[1][2] = (*ptr); 
		  tmp.mat_[2][1] = (*ptr);
		  ++ptr;
		  tmp.mat_[2][2] = (*ptr); 
		  ++ptr;
		}
		ifield->set_value( tmp, *iter);
	      }
	    } else if (nrrdKindSize( dataH->nrrd->axis[0].kind) == 9) {
	      // not symmetric, do a straight across copy
	      
	      if (build_eigens == 1) {
		float eval[3], evec[9], eval_scl[3], evec_scl[9];
		tenEigensolve_f(eval, evec, ptr);
		
		//float scl = ptr[0] > 0.5;
		float scl = 1.0;
		for (int cc=0; cc<3; cc++) {
		  ELL_3V_SCALE(evec_scl+3*cc, scl, evec+3*cc);
		  eval_scl[cc] = scl*eval[cc];
		}
		Vector e1(evec_scl[0], evec_scl[1], evec_scl[2]);
		Vector e2(evec_scl[3], evec_scl[4], evec_scl[5]);
		Vector e3(evec_scl[6], evec_scl[7], evec_scl[8]);
		
		tmp.set_outside_eigens(e1, e2, e3, eval_scl[0], eval_scl[1], eval_scl[2]);
		tmp.mat_[0][0] = (*ptr); 
		++ptr;
		tmp.mat_[0][1] = (*ptr); 
		++ptr;
		tmp.mat_[0][2] = (*ptr); 
		++ptr;
		tmp.mat_[1][0] = (*ptr); 
		++ptr;
		tmp.mat_[1][1] = (*ptr); 
		++ptr;
		tmp.mat_[1][2] = (*ptr); 
		++ptr;
		tmp.mat_[2][0] = (*ptr); 
		++ptr;
		tmp.mat_[2][1] = (*ptr); 
		++ptr;
		tmp.mat_[2][2] = (*ptr); 
		++ptr;
		ifield->set_value( tmp, *iter);
	      } else {
		tmp.mat_[0][0] = (*ptr);
		++ptr;
		tmp.mat_[0][1] = (*ptr);
		++ptr;
		tmp.mat_[0][2] = (*ptr);
		++ptr;
		tmp.mat_[1][0] = (*ptr);
		++ptr;
		tmp.mat_[1][1] = (*ptr);
		++ptr;
		tmp.mat_[1][2] = (*ptr);
		++ptr;
		tmp.mat_[2][0] = (*ptr);
		++ptr;
		tmp.mat_[2][1] = (*ptr);
		++ptr;
		tmp.mat_[2][2] = (*ptr);
		++ptr;
		ifield->set_value( tmp, *iter);
	      }
	    } else {
	      return 0;
	    }
	    ++iter;
	  }
	}
      }
    }
    else {
      ifield = (FIELD *) scinew FIELD((MESH *) imesh, 1);
      
      typename FIELD::mesh_type::Node::iterator iter, end;
      
      imesh->begin( iter );
      imesh->end( end );
      
      register int i, j, k, index;
      
      NTYPE *ptr = (NTYPE *)(dataH->nrrd->data);
      for( i=0; i<idim; i++ ) {
	for( j=0; j<jdim; j++ ) {
	  for( k=0; k<kdim; k++ ) {
	    if( permute )
	      index = (k * jdim + j) * idim + i;
	    else 
	      index = (i * jdim + j) * kdim + k;
	    
	    Tensor tmp;
	    
	    if (nrrdKindSize( dataH->nrrd->axis[0].kind) == 6) {
	      // 3D symmetric tensor
	      
	      if (build_eigens == 1) {
		float eval[3], evec[9], eval_scl[3], evec_scl[9];
		tenEigensolve_f(eval, evec, ptr);
		
		//float scl = ptr[0] > 0.5;
		float scl = 1.0;
		for (int cc=0; cc<3; cc++) {
		  ELL_3V_SCALE(evec_scl+3*cc, scl, evec+3*cc);
		  eval_scl[cc] = scl*eval[cc];
		}
		Vector e1(evec_scl[0], evec_scl[1], evec_scl[2]);
		Vector e2(evec_scl[3], evec_scl[4], evec_scl[5]);
		Vector e3(evec_scl[6], evec_scl[7], evec_scl[8]);
		
		tmp.set_outside_eigens(e1, e2, e3, eval_scl[0], eval_scl[1], eval_scl[2]);
		tmp.mat_[0][0] = (*ptr); 
		++ptr;
		tmp.mat_[0][1] = (*ptr); 
		tmp.mat_[1][0] = (*ptr);
		++ptr;
		tmp.mat_[0][2] = (*ptr); 
		tmp.mat_[2][0] = (*ptr);
		++ptr;
		tmp.mat_[1][1] = (*ptr); 
		++ptr;
		tmp.mat_[1][2] = (*ptr); 
		tmp.mat_[2][1] = (*ptr);
		++ptr;
		tmp.mat_[2][2] = (*ptr); 
		++ptr;
		ifield->set_value( tmp, *iter);
	      } else {
		tmp.mat_[0][0] = (*ptr); 
		++ptr;
		tmp.mat_[0][1] = (*ptr); 
		tmp.mat_[1][0] = (*ptr);
		++ptr;
		tmp.mat_[0][2] = (*ptr); 
		tmp.mat_[2][0] = (*ptr);
		++ptr;
		tmp.mat_[1][1] = (*ptr); 
		++ptr;
		tmp.mat_[1][2] = (*ptr); 
		tmp.mat_[2][1] = (*ptr);
		++ptr;
		tmp.mat_[2][2] = (*ptr); 
		++ptr;
		ifield->set_value( tmp, *iter);
	      }
	    } else if (nrrdKindSize( dataH->nrrd->axis[0].kind) == 7) {
	      // 3D symmetric tensor with mask
	      
	      if (build_eigens == 1) {
		float eval[3], evec[9], eval_scl[3], evec_scl[9];
		tenEigensolve_f(eval, evec, ptr);
		
		float scl = ptr[0] > 0.5;
		for (int cc=0; cc<3; cc++) {
		  ELL_3V_SCALE(evec_scl+3*cc, scl, evec+3*cc);
		  eval_scl[cc] = scl*eval[cc];
		}
		Vector e1(evec_scl[0], evec_scl[1], evec_scl[2]);
		Vector e2(evec_scl[3], evec_scl[4], evec_scl[5]);
		Vector e3(evec_scl[6], evec_scl[7], evec_scl[8]);
		
		tmp.set_outside_eigens(e1, e2, e3, eval_scl[0], eval_scl[1], eval_scl[2]);
		++ptr; // skip first value (confidence)
		tmp.mat_[0][0] = (*ptr); 
		++ptr;
		tmp.mat_[0][1] = (*ptr); 
		tmp.mat_[1][0] = (*ptr);
		++ptr;
		tmp.mat_[0][2] = (*ptr); 
		tmp.mat_[2][0] = (*ptr);
		++ptr;
		tmp.mat_[1][1] = (*ptr); 
		++ptr;
		tmp.mat_[1][2] = (*ptr); 
		tmp.mat_[2][1] = (*ptr);
		++ptr;
		tmp.mat_[2][2] = (*ptr); 
		++ptr;
		ifield->set_value( tmp, *iter);
	      } else {
		// if mask < 0.5, all tensor values are 0
		if (ptr[index*3] < 0.5) {
		  for (int x=0; x<3; x++) 
		    for (int y=0; y<3; y++)
		      tmp.mat_[x][y]=0;
		  
		} else {
		  ++ptr; // skip first value (confidence)
		  tmp.mat_[0][0] = (*ptr); 
		  ++ptr;
		  tmp.mat_[0][1] = (*ptr); 
		  tmp.mat_[1][0] = (*ptr);
		  ++ptr;
		  tmp.mat_[0][2] = (*ptr); 
		  tmp.mat_[2][0] = (*ptr);
		  ++ptr;
		  tmp.mat_[1][1] = (*ptr); 
		  ++ptr;
		  tmp.mat_[1][2] = (*ptr); 
		  tmp.mat_[2][1] = (*ptr);
		  ++ptr;
		  tmp.mat_[2][2] = (*ptr); 
		  ++ptr;
		}
		ifield->set_value( tmp, *iter);
	      }
	    } else if (nrrdKindSize( dataH->nrrd->axis[0].kind) == 9) {
	      // not symmetric, do a straight across copy
	      
	      if (build_eigens == 1) {
		float eval[3], evec[9], eval_scl[3], evec_scl[9];
		tenEigensolve_f(eval, evec, ptr);
		
		//float scl = ptr[0] > 0.5;
		float scl = 1.0;
		for (int cc=0; cc<3; cc++) {
		  ELL_3V_SCALE(evec_scl+3*cc, scl, evec+3*cc);
		  eval_scl[cc] = scl*eval[cc];
		}
		Vector e1(evec_scl[0], evec_scl[1], evec_scl[2]);
		Vector e2(evec_scl[3], evec_scl[4], evec_scl[5]);
		Vector e3(evec_scl[6], evec_scl[7], evec_scl[8]);
		
		tmp.set_outside_eigens(e1, e2, e3, eval_scl[0], eval_scl[1], eval_scl[2]);
		tmp.mat_[0][0] = (*ptr); 
		++ptr;
		tmp.mat_[0][1] = (*ptr); 
		++ptr;
		tmp.mat_[0][2] = (*ptr); 
		++ptr;
		tmp.mat_[1][0] = (*ptr); 
		++ptr;
		tmp.mat_[1][1] = (*ptr); 
		++ptr;
		tmp.mat_[1][2] = (*ptr); 
		++ptr;
		tmp.mat_[2][0] = (*ptr); 
		++ptr;
		tmp.mat_[2][1] = (*ptr); 
		++ptr;
		tmp.mat_[2][2] = (*ptr); 
		++ptr;
		ifield->set_value( tmp, *iter);
	      } else {
		tmp.mat_[0][0] = (*ptr);
		++ptr;
		tmp.mat_[0][1] = (*ptr);
		++ptr;
		tmp.mat_[0][2] = (*ptr);
		++ptr;
		tmp.mat_[1][0] = (*ptr);
		++ptr;
		tmp.mat_[1][1] = (*ptr);
		++ptr;
		tmp.mat_[1][2] = (*ptr);
		++ptr;
		tmp.mat_[2][0] = (*ptr);
		++ptr;
		tmp.mat_[2][1] = (*ptr);
		++ptr;
		tmp.mat_[2][2] = (*ptr);
		++ptr;
		ifield->set_value( tmp, *iter);
	      }
	    } else {
	      return 0;
	    }
	    ++iter;
	  }
	}
      }
    }
  } else {
    ifield = (FIELD *) scinew FIELD((MESH *) imesh, 1);
  }
  return FieldHandle( ifield );
}

template< class FIELD, class MESH, class NTYPE >
FieldHandle
NrrdToFieldFieldAlgoTensor<FIELD, MESH, NTYPE>::
execute(MeshHandle& mHandle,
	NrrdDataHandle dataH,
	int build_eigens)
{
  MESH *imesh = (MESH *) mHandle.get_rep();
  FIELD *ifield = 0;
  FieldHandle fH;

  if (dataH != 0) {
    // determine if nrrd is unknown, node or cell centered
    int data_center = nrrdCenterUnknown;
    for (int a = 0; a<dataH->nrrd->dim; a++) {
      if (dataH->nrrd->axis[a].center != nrrdCenterUnknown)
	data_center = dataH->nrrd->axis[a].center;
    }
    if (data_center == nrrdCenterCell) {
      ifield = (FIELD *) scinew FIELD((MESH *) imesh, 0);

      typename FIELD::mesh_type::Elem::iterator iter, end;
      
      imesh->begin( iter );
      imesh->end( end );
      
      NTYPE *ptr = (NTYPE *)(dataH->nrrd->data);
      int i = 0;
      while (iter != end) {
	Tensor tmp;
	if (nrrdKindSize( dataH->nrrd->axis[0].kind) == 6) {
	  // 3D symmetric tensor
	  
	  if (build_eigens == 1) {
	    float eval[3], evec[9], eval_scl[3], evec_scl[9];
	    tenEigensolve_f(eval, evec, ptr);
	    
	    float scl = 1.0;
	    for (int cc=0; cc<3; cc++) {
	      ELL_3V_SCALE(evec_scl+3*cc, scl, evec+3*cc);
	      eval_scl[cc] = scl*eval[cc];
	    }
	    Vector e1(evec_scl[0], evec_scl[1], evec_scl[2]);
	    Vector e2(evec_scl[3], evec_scl[4], evec_scl[5]);
	    Vector e3(evec_scl[6], evec_scl[7], evec_scl[8]);
	    
	    tmp.set_outside_eigens(e1, e2, e3, eval_scl[0], eval_scl[1], eval_scl[2]);
	  }
	  tmp.mat_[0][0] = (*ptr); 
	  ++ptr;
	  tmp.mat_[0][1] = (*ptr); 
	  tmp.mat_[1][0] = (*ptr);
	  ++ptr;
	  tmp.mat_[0][2] = (*ptr); 
	  tmp.mat_[2][0] = (*ptr);
	  ++ptr;
	  tmp.mat_[1][1] = (*ptr); 
	  ++ptr;
	  tmp.mat_[1][2] = (*ptr); 
	  tmp.mat_[2][1] = (*ptr);
	  ++ptr;
	  tmp.mat_[2][2] = (*ptr); 
	  ++ptr;
	  ifield->set_value( tmp, *iter);
	} else if (nrrdKindSize( dataH->nrrd->axis[0].kind) == 7) {
	  // 3D symmetric tensor with mask
	  
	  if (build_eigens == 1) {
	    float eval[3], evec[9], eval_scl[3], evec_scl[9];
	    tenEigensolve_f(eval, evec, ptr);
	    
	    float scl = ptr[0] > 0.5;
	    for (int cc=0; cc<3; cc++) {
	      ELL_3V_SCALE(evec_scl+3*cc, scl, evec+3*cc);
	      eval_scl[cc] = scl*eval[cc];
	    }
	    Vector e1(evec_scl[0], evec_scl[1], evec_scl[2]);
	    Vector e2(evec_scl[3], evec_scl[4], evec_scl[5]);
	    Vector e3(evec_scl[6], evec_scl[7], evec_scl[8]);
	    
	    tmp.set_outside_eigens(e1, e2, e3, eval_scl[0], eval_scl[1], eval_scl[2]);
	    
	    ++ptr; // skip mask
	    tmp.mat_[0][0] = (*ptr); 
	    ++ptr;
	    tmp.mat_[0][1] = (*ptr); 
	    tmp.mat_[1][0] = (*ptr);
	    ++ptr;
	    tmp.mat_[0][2] = (*ptr); 
	    tmp.mat_[2][0] = (*ptr);
	    ++ptr;
	    tmp.mat_[1][1] = (*ptr); 
	    ++ptr;
	    tmp.mat_[1][2] = (*ptr); 
	    tmp.mat_[2][1] = (*ptr);
	    ++ptr;
	    tmp.mat_[2][2] = (*ptr); 
	    ++ptr;
	    ifield->set_value( tmp, *iter);
	  } else {
	    // if mask < 0.5, all tensor values are 0
	    if (*ptr < 0.5) {
	      for (int x=0; x<3; x++) 
		for (int y=0; y<3; y++)
		  tmp.mat_[x][y]=0;
	      
	    } else {
	      // skip mask
	      ++ptr; // skip mask
	      tmp.mat_[0][0] = (*ptr); 
	      ++ptr;
	      tmp.mat_[0][1] = (*ptr); 
	      tmp.mat_[1][0] = (*ptr);
	      ++ptr;
	      tmp.mat_[0][2] = (*ptr); 
	      tmp.mat_[2][0] = (*ptr);
	      ++ptr;
	      tmp.mat_[1][1] = (*ptr); 
	      ++ptr;
	      tmp.mat_[1][2] = (*ptr); 
	      tmp.mat_[2][1] = (*ptr);
	      ++ptr;
	      tmp.mat_[2][2] = (*ptr); 
	      ++ptr;
	    }
	    ifield->set_value( tmp, *iter);
	  }
	} else if (nrrdKindSize( dataH->nrrd->axis[0].kind) == 9) {
	  // not symmetric, do a straight across copy
	  
	  if (build_eigens == 1) {
	    float eval[3], evec[9], eval_scl[3], evec_scl[9];
	    tenEigensolve_f(eval, evec, ptr);
	    
	    //float scl = ptr[0] > 0.5;
	    float scl = 1.0;
	    for (int cc=0; cc<3; cc++) {
	      ELL_3V_SCALE(evec_scl+3*cc, scl, evec+3*cc);
	      eval_scl[cc] = scl*eval[cc];
	    }
	    Vector e1(evec_scl[0], evec_scl[1], evec_scl[2]);
	    Vector e2(evec_scl[3], evec_scl[4], evec_scl[5]);
	    Vector e3(evec_scl[6], evec_scl[7], evec_scl[8]);
	    
	    tmp.set_outside_eigens(e1, e2, e3, eval_scl[0], eval_scl[1], eval_scl[2]);
	    tmp.mat_[0][0] = (*ptr); 
	    ++ptr;
	    tmp.mat_[0][1] = (*ptr); 
	    ++ptr;
	    tmp.mat_[0][2] = (*ptr); 
	    ++ptr;
	    tmp.mat_[1][0] = (*ptr); 
	    ++ptr;
	    tmp.mat_[1][1] = (*ptr); 
	    ++ptr;
	    tmp.mat_[1][2] = (*ptr); 
	    ++ptr;
	    tmp.mat_[2][0] = (*ptr); 
	    ++ptr;
	    tmp.mat_[2][1] = (*ptr); 
	    ++ptr;
	    tmp.mat_[2][2] = (*ptr); 
	    ++ptr;
	    ifield->set_value( tmp, *iter);
	  } else {
	    tmp.mat_[0][0] = (*ptr); 
	    ++ptr;
	    tmp.mat_[0][1] = (*ptr); 
	    ++ptr;
	    tmp.mat_[0][2] = (*ptr); 
	    ++ptr;
	    tmp.mat_[1][0] = (*ptr); 
	    ++ptr;
	    tmp.mat_[1][1] = (*ptr); 
	    ++ptr;
	    tmp.mat_[1][2] = (*ptr); 
	    ++ptr;
	    tmp.mat_[2][0] = (*ptr); 
	    ++ptr;
	    tmp.mat_[2][1] = (*ptr); 
	    ++ptr;
	    tmp.mat_[2][2] = (*ptr); 
	    ++ptr;
	    ifield->set_value( tmp, *iter);
	  }
	} else {
	  fH = ifield;
	  return fH;
	}
	++iter;
	i++;
      }    
    }
    else {
      ifield = (FIELD *) scinew FIELD((MESH *) imesh, 1);
      
      typename FIELD::mesh_type::Node::iterator iter, end;
      
      imesh->begin( iter );
      imesh->end( end );
      
      NTYPE *ptr = (NTYPE *)(dataH->nrrd->data);
      int i = 0;
      while (iter != end) {
	Tensor tmp;
	if (nrrdKindSize( dataH->nrrd->axis[0].kind) == 6) {
	  // 3D symmetric tensor
	  
	  if (build_eigens == 1) {
	    float eval[3], evec[9], eval_scl[3], evec_scl[9];
	    tenEigensolve_f(eval, evec, ptr);
	    
	    float scl = 1.0;
	    for (int cc=0; cc<3; cc++) {
	      ELL_3V_SCALE(evec_scl+3*cc, scl, evec+3*cc);
	      eval_scl[cc] = scl*eval[cc];
	    }
	    Vector e1(evec_scl[0], evec_scl[1], evec_scl[2]);
	    Vector e2(evec_scl[3], evec_scl[4], evec_scl[5]);
	    Vector e3(evec_scl[6], evec_scl[7], evec_scl[8]);
	    
	    tmp.set_outside_eigens(e1, e2, e3, eval_scl[0], eval_scl[1], eval_scl[2]);
	  }
	  tmp.mat_[0][0] = (*ptr); 
	  ++ptr;
	  tmp.mat_[0][1] = (*ptr); 
	  tmp.mat_[1][0] = (*ptr);
	  ++ptr;
	  tmp.mat_[0][2] = (*ptr); 
	  tmp.mat_[2][0] = (*ptr);
	  ++ptr;
	  tmp.mat_[1][1] = (*ptr); 
	  ++ptr;
	  tmp.mat_[1][2] = (*ptr); 
	  tmp.mat_[2][1] = (*ptr);
	  ++ptr;
	  tmp.mat_[2][2] = (*ptr); 
	  ++ptr;
	  ifield->set_value( tmp, *iter);
	} else if (nrrdKindSize( dataH->nrrd->axis[0].kind) == 7) {
	  // 3D symmetric tensor with mask
	  
	  if (build_eigens == 1) {
	    float eval[3], evec[9], eval_scl[3], evec_scl[9];
	    tenEigensolve_f(eval, evec, ptr);
	    
	    float scl = ptr[0] > 0.5;
	    for (int cc=0; cc<3; cc++) {
	      ELL_3V_SCALE(evec_scl+3*cc, scl, evec+3*cc);
	      eval_scl[cc] = scl*eval[cc];
	    }
	    Vector e1(evec_scl[0], evec_scl[1], evec_scl[2]);
	    Vector e2(evec_scl[3], evec_scl[4], evec_scl[5]);
	    Vector e3(evec_scl[6], evec_scl[7], evec_scl[8]);
	    
	    tmp.set_outside_eigens(e1, e2, e3, eval_scl[0], eval_scl[1], eval_scl[2]);
	    
	    ++ptr; // skip mask
	    tmp.mat_[0][0] = (*ptr); 
	    ++ptr;
	    tmp.mat_[0][1] = (*ptr); 
	    tmp.mat_[1][0] = (*ptr);
	    ++ptr;
	    tmp.mat_[0][2] = (*ptr); 
	    tmp.mat_[2][0] = (*ptr);
	    ++ptr;
	    tmp.mat_[1][1] = (*ptr); 
	    ++ptr;
	    tmp.mat_[1][2] = (*ptr); 
	    tmp.mat_[2][1] = (*ptr);
	    ++ptr;
	    tmp.mat_[2][2] = (*ptr); 
	    ++ptr;
	    ifield->set_value( tmp, *iter);
	  } else {
	    // if mask < 0.5, all tensor values are 0
	    if (*ptr < 0.5) {
	      for (int x=0; x<3; x++) 
		for (int y=0; y<3; y++)
		  tmp.mat_[x][y]=0;
	      
	    } else {
	      // skip mask
	      ++ptr; // skip mask
	      tmp.mat_[0][0] = (*ptr); 
	      ++ptr;
	      tmp.mat_[0][1] = (*ptr); 
	      tmp.mat_[1][0] = (*ptr);
	      ++ptr;
	      tmp.mat_[0][2] = (*ptr); 
	      tmp.mat_[2][0] = (*ptr);
	      ++ptr;
	      tmp.mat_[1][1] = (*ptr); 
	      ++ptr;
	      tmp.mat_[1][2] = (*ptr); 
	      tmp.mat_[2][1] = (*ptr);
	      ++ptr;
	      tmp.mat_[2][2] = (*ptr); 
	      ++ptr;
	    }
	    ifield->set_value( tmp, *iter);
	  }
	} else if (nrrdKindSize( dataH->nrrd->axis[0].kind) == 9) {
	  // not symmetric, do a straight across copy
	  
	  if (build_eigens == 1) {
	    float eval[3], evec[9], eval_scl[3], evec_scl[9];
	    tenEigensolve_f(eval, evec, ptr);
	    
	    //float scl = ptr[0] > 0.5;
	    float scl = 1.0;
	    for (int cc=0; cc<3; cc++) {
	      ELL_3V_SCALE(evec_scl+3*cc, scl, evec+3*cc);
	      eval_scl[cc] = scl*eval[cc];
	    }
	    Vector e1(evec_scl[0], evec_scl[1], evec_scl[2]);
	    Vector e2(evec_scl[3], evec_scl[4], evec_scl[5]);
	    Vector e3(evec_scl[6], evec_scl[7], evec_scl[8]);
	    
	    tmp.set_outside_eigens(e1, e2, e3, eval_scl[0], eval_scl[1], eval_scl[2]);
	    tmp.mat_[0][0] = (*ptr); 
	    ++ptr;
	    tmp.mat_[0][1] = (*ptr); 
	    ++ptr;
	    tmp.mat_[0][2] = (*ptr); 
	    ++ptr;
	    tmp.mat_[1][0] = (*ptr); 
	    ++ptr;
	    tmp.mat_[1][1] = (*ptr); 
	    ++ptr;
	    tmp.mat_[1][2] = (*ptr); 
	    ++ptr;
	    tmp.mat_[2][0] = (*ptr); 
	    ++ptr;
	    tmp.mat_[2][1] = (*ptr); 
	    ++ptr;
	    tmp.mat_[2][2] = (*ptr); 
	    ++ptr;
	    ifield->set_value( tmp, *iter);
	  } else {
	    tmp.mat_[0][0] = (*ptr); 
	    ++ptr;
	    tmp.mat_[0][1] = (*ptr); 
	    ++ptr;
	    tmp.mat_[0][2] = (*ptr); 
	    ++ptr;
	    tmp.mat_[1][0] = (*ptr); 
	    ++ptr;
	    tmp.mat_[1][1] = (*ptr); 
	    ++ptr;
	    tmp.mat_[1][2] = (*ptr); 
	    ++ptr;
	    tmp.mat_[2][0] = (*ptr); 
	    ++ptr;
	    tmp.mat_[2][1] = (*ptr); 
	    ++ptr;
	    tmp.mat_[2][2] = (*ptr); 
	    ++ptr;
	    ifield->set_value( tmp, *iter);
	  }
	} else {
	  fH = ifield;
	  return fH;
	}
	++iter;
	i++;
      }    
    }

    // set transform if one of the nrrd properties
    fH = ifield;
    const string meshstr =
      fH->get_type_description(0)->get_name().substr(0, 6);
    
    if (!(imesh->is_editable() && meshstr != "Struct"))
      {
	string trans_string;
	if (dataH->get_property("Transform", trans_string) && trans_string != "Unknown") {
	  double t[16];
	  Transform trans;
	  int old_index=0, new_index=0;
	  for(int i=0; i<16; i++) {
	    new_index = trans_string.find(" ", old_index);
	    string temp = trans_string.substr(old_index, new_index-old_index);
	    old_index = new_index+1;
	    string_to_double(temp, t[i]);
	  }
	  trans.set(t);
	  imesh->transform(trans);
	} 
      }	        
  } else {
    ifield = (FIELD *) scinew FIELD((MESH *) imesh, 1);
    fH = ifield;
  }
  return fH;
}


class NrrdToFieldTestMeshAlgo : public DynamicAlgoBase
{
public:
  virtual bool execute(SCIRun::FieldHandle, NrrdDataHandle, 
		       SCIRun::FieldHandle &, const int a0_size) = 0;
  virtual ~NrrdToFieldTestMeshAlgo();

  static const string& get_h_file_path();
  static string dyn_file_name(const TypeDescription *td) {
    // add no extension.
    return template_class_name() + "." + td->get_filename() + ".";
  }

  static const string base_class_name() {
    static string name("NrrdToFieldTestMeshAlgo");
    return name;
  }

  static const string template_class_name() {
    static string name("NrrdToFieldTestMesh");
    return name;
  }

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *td);
};

template< class Fld >
class NrrdToFieldTestMesh : public NrrdToFieldTestMeshAlgo
{
public:
  //! virtual interface.
  bool execute(SCIRun::FieldHandle fld, 
	       NrrdDataHandle      in,
	       SCIRun::FieldHandle &out,
	       const int a0_size);
};


template< class Fld>
bool
NrrdToFieldTestMesh<Fld>::execute(SCIRun::FieldHandle fld, 
				  NrrdDataHandle      in,
				  SCIRun::FieldHandle &out,
				  const int a0_size)
{
  Nrrd *inrrd = in->nrrd;

  vector<unsigned int> dims;

  typedef typename Fld::mesh_type Msh;
  Msh *mesh = dynamic_cast<Msh*>(fld->mesh().get_rep());
  ASSERT(mesh != 0);
  int off = 0;
  bool uns = false;
  if (! mesh->get_dim(dims)) {
    // Unstructured fields fall into this category, for them we create nrrds
    // of dimension 1 (2 if vector or scalar data).
    uns = true;
    switch (fld->basis_order()) {
    case 1:
      {
	typename Fld::mesh_type::Node::size_type sz;
	mesh->size(sz);
	dims.push_back(sz);
      }
      break;
    case 0:
      {
	if (mesh->dimensionality() == 0) {
	  typename Fld::mesh_type::Node::size_type sz;
	  mesh->size(sz);
	  dims.push_back(sz);
	} else if (mesh->dimensionality() == 1) {
	  typename Fld::mesh_type::Edge::size_type sz;
	  mesh->size(sz);
	  dims.push_back(sz);
	} else if (mesh->dimensionality() == 2) {
	  typename Fld::mesh_type::Face::size_type sz;
	  mesh->size(sz);
	  dims.push_back(sz);
	} else if (mesh->dimensionality() == 3) {
	  typename Fld::mesh_type::Elem::size_type sz;
	  mesh->size(sz);
	  dims.push_back(sz);
	}
      }
      break;
    default:
      cerr << "Location of data not defined.  Assuming at Nodes\n";
      typename Fld::mesh_type::Node::size_type sz;
      mesh->size(sz);
      dims.push_back(sz);
    }    
  }

  // if vector/tensor data store 3 or 7 at the end of dims vector
  if (a0_size > 1) 
    dims.push_back(a0_size);
  
  if ((!uns) && fld->basis_order() == 0) {
    off = 1;
  }

  // If the data was vector or tensor it will have an extra axis.
  // It is axis 0.  Make sure sizes along each dim still match.
  if (inrrd->dim != (int)dims.size()) {
    return false;
  }

  // If a0_size equals 3 or 7 then the first axis contains
  // vector or tensor data and a ND nrrd would convert
  // to a (N-1)D type field. 

  int field_dim = inrrd->dim;
  if (a0_size > 1) // tensor or vector data in first dimension
    field_dim -= 1;
  switch (field_dim) {
  case 1:
    {
      // make sure size of dimensions match up
      unsigned int nx = 0;
      if (a0_size > 1) {
	nx = inrrd->axis[1].size + off;
      } else {
	nx = inrrd->axis[0].size + off;
      }
      if (nx != dims[0]) { return false; }
    }
    break;
  case 2:
    {
      unsigned int nx = 0, ny = 0;
      if (a0_size > 1) {
	nx = inrrd->axis[1].size + off;
	ny = inrrd->axis[2].size + off;
      } else {
	nx = inrrd->axis[0].size + off;
	ny = inrrd->axis[1].size + off;
      }
      if ((nx != dims[0]) || (ny != dims[1])) {
	return false;
      }
    }
    break;
  case 3:
    {
      unsigned int nx = 0, ny = 0, nz = 0;
      if (a0_size > 1) {
	nx = inrrd->axis[1].size + off;
	ny = inrrd->axis[2].size + off;
        nz = inrrd->axis[3].size + off;
      } else {
	nx = inrrd->axis[0].size + off;
	ny = inrrd->axis[1].size + off;
        nz = inrrd->axis[2].size + off;
      }
      if ((nx != dims[0]) || (ny != dims[1]) || (nz != dims[2])) {
	return false;
      }
    }
    break;
  default:   // anything else is invalid.
    return false;
  }

  // Things match up, create the new output field.
  return true;
}
  


} // end namespace SCITeem

#endif // NrrdFieldConverter_h
