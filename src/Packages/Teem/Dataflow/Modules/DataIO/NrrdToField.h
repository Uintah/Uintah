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

//    File   : NrrdFieldConverter.h
//    Author : Allen Sanderson
//             School of Computing
//             University of Utah
//    Date   : July 2003

#if !defined(NrrdToField_h)
#define NrrdToField_h

#include <Dataflow/Network/Module.h>

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>

#include <Core/Datatypes/PointCloudField.h>
#include <Core/Datatypes/CurveField.h>
#include <Core/Datatypes/TriSurfField.h>
#include <Core/Datatypes/QuadSurfField.h>

#include <Core/Datatypes/TetVolField.h>
#include <Core/Datatypes/PrismVolField.h>
#include <Core/Datatypes/HexVolField.h>

#include <Core/Datatypes/LatVolField.h>
#include <Core/Datatypes/ImageField.h>
#include <Core/Datatypes/ScanlineField.h>

#include <Core/Datatypes/StructHexVolField.h>
#include <Core/Datatypes/StructQuadSurfField.h>
#include <Core/Datatypes/StructCurveField.h>

#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Tensor.h>

#include <Teem/Core/Datatypes/NrrdData.h>

namespace SCITeem {

using namespace SCIRun;

class NrrdFieldConverterMeshAlgo : public DynamicAlgoBase
{
public:
  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info( const std::string topoStr,
					     const TypeDescription *mtd,
					     const unsigned int ptype,
					     const unsigned int ctype);

};



class RegularNrrdFieldConverterMeshAlgo : public NrrdFieldConverterMeshAlgo
{
public:
  virtual void execute(MeshHandle& mHandle,
		       NrrdDataHandle dataH, int data_size);
};


template< class MESH, class DNTYPE, class CNTYPE>
class RegularNrrdFieldConverterMeshAlgoT : 
  public RegularNrrdFieldConverterMeshAlgo
{
public:
  virtual void execute(MeshHandle& mHandle,
		       NrrdDataHandle dataH, int data_size) = 0;
};

template< class MESH, class DNTYPE, class CNTYPE>
void
RegularNrrdFieldConverterMeshAlgoT< MESH, DNTYPE, CNTYPE>::
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


class StructuredNrrdFieldConverterMeshAlgo : public NrrdFieldConverterMeshAlgo
{
public:
  virtual void execute(MeshHandle& mHandle,
		       NrrdDataHandle pointsH,
		       int idim, int jdim, int kdim) = 0;
};


template< class MESH, class PNTYPE, class CNTYPE >
class StructuredNrrdFieldConverterMeshAlgoT : 
  public StructuredNrrdFieldConverterMeshAlgo
{
public:

  virtual void execute(MeshHandle& mHandle,
		       NrrdDataHandle pointsH,
		       int idim, int jdim, int kdim);
};


template< class MESH, class PNTYPE, class CNTYPE >
void
StructuredNrrdFieldConverterMeshAlgoT< MESH, PNTYPE, CNTYPE >::
execute(MeshHandle& mHandle,
	NrrdDataHandle pointsH,
	int idim, int jdim, int kdim)
{
  MESH *imesh = (MESH *) mHandle.get_rep();
  typename MESH::Node::iterator inodeItr;

  imesh->begin( inodeItr );

  register int i, j, k;
  int rank = pointsH->nrrd->axis[1].size;

    PNTYPE *ptr = (PNTYPE *)(pointsH->nrrd->data);
    
    for( k=0; k<kdim; k++ ) {
      for( j=0; j<jdim; j++ ) {
	for( i=0; i<idim; i++ ) {
	  
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



class UnstructuredNrrdFieldConverterMeshAlgo : 
  public NrrdFieldConverterMeshAlgo
{
public:
  virtual void execute(MeshHandle& mHandle,
		       NrrdDataHandle pointsH,
		       NrrdDataHandle connectH,
		       unsigned int connectivity) = 0;
};

template< class MESH, class PNTYPE, class CNTYPE >
class UnstructuredNrrdFieldConverterMeshAlgoT : 
  public UnstructuredNrrdFieldConverterMeshAlgo
{
public:
  virtual void execute(MeshHandle& mHandle,
		       NrrdDataHandle pointsH,
		       NrrdDataHandle connectH,
		       unsigned int connectivity);
};


template< class MESH, class PNTYPE, class CNTYPE >
void
UnstructuredNrrdFieldConverterMeshAlgoT< MESH, PNTYPE, CNTYPE >::
execute(MeshHandle& mHandle,
	NrrdDataHandle pointsH,
	NrrdDataHandle connectH,
	unsigned int connectivity)
{
  MESH *imesh = (MESH *) mHandle.get_rep();
  typename MESH::Node::iterator inodeItr;

  imesh->begin( inodeItr );

  int npts = pointsH->nrrd->axis[0].size;
  int rank = pointsH->nrrd->axis[1].size;

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

    int nelements = connectH->nrrd->axis[0].size;

    typename MESH::Node::array_type array(connectivity);

    for( int i=0; i<nelements; i++ ) {
      for( unsigned int j=0; j<connectivity; j++ ) {
	array[j] = (int) cPtr[i*connectivity+j];
      }

      imesh->add_elem( array );
    }
  }
}




class NrrdFieldConverterFieldAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(MeshHandle& mHandle,
			      NrrdDataHandle dataH,
			      int idim, int jdim, int kdim, int permute) = 0;
  
  virtual FieldHandle execute(MeshHandle& mHandle,
			      NrrdDataHandle dataH) = 0;
  
   //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *mtd,
					    const std::string fname,
					    const unsigned int ntype,
					    int rank);
};

template< class FIELD, class MESH, class NTYPE >
class NrrdFieldConverterFieldAlgoScalar : public NrrdFieldConverterFieldAlgo
{
public:
  //! virtual interface.
  virtual FieldHandle execute(MeshHandle& mHandle,
			      NrrdDataHandle dataH,
			      int idim, int jdim, int kdim, int permute);

  virtual FieldHandle execute(MeshHandle& mHandle,
			      NrrdDataHandle dataH);  
};


template< class FIELD, class MESH, class NTYPE >
FieldHandle
NrrdFieldConverterFieldAlgoScalar<FIELD, MESH, NTYPE>::
execute(MeshHandle& mHandle,
	NrrdDataHandle dataH,
	int idim, int jdim, int kdim, int permute)
{
  cerr << "Inside i,j,k execute\n";
  MESH *imesh = (MESH *) mHandle.get_rep();
  FIELD *ifield = 0;
  
  if (dataH != 0) {
    // determine if nrrd is unknown, node or cell centered
    int data_center = nrrdCenterUnknown;
    for (int a = 0; a<dataH->nrrd->dim; a++) {
      if (dataH->nrrd->axis[a].center != nrrdCenterUnknown)
	data_center = dataH->nrrd->axis[a].center;
    }
    if (data_center == nrrdCenterCell)
      ifield = (FIELD *) scinew FIELD((MESH *) imesh, Field::FACE);
    else
      ifield = (FIELD *) scinew FIELD((MESH *) imesh, Field::NODE);

    cerr << "has data!\n";
    typename FIELD::mesh_type::Node::iterator inodeItr;
    
    imesh->begin( inodeItr );
    
    NTYPE *ptr = (NTYPE *)(dataH->nrrd->data);
    
    register int i, j, k, index;
    
    for( k=0; k<kdim; k++ ) {
      for( j=0; j<jdim; j++ ) {
	for( i=0; i<idim; i++ ) {
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
  } else {
    ifield = (FIELD *) scinew FIELD((MESH *) imesh, Field::NODE);
  }
  
  return FieldHandle( ifield );
}



template< class FIELD, class MESH, class NTYPE >
FieldHandle
NrrdFieldConverterFieldAlgoScalar<FIELD, MESH, NTYPE>::
execute(MeshHandle& mHandle,
	NrrdDataHandle dataH)

{
  cerr << "Inside NORMAL execute\n";
  MESH *imesh = (MESH *) mHandle.get_rep();
  FIELD *ifield = 0;

  if (dataH != 0) {
    // determine if nrrd is unknown, node or cell centered
    int data_center = nrrdCenterUnknown;
    for (int a = 0; a<dataH->nrrd->dim; a++) {
      if (dataH->nrrd->axis[a].center != nrrdCenterUnknown)
	data_center = dataH->nrrd->axis[a].center;
    }
    if (data_center == nrrdCenterCell)
      ifield = (FIELD *) scinew FIELD((MESH *) imesh, Field::FACE);
    else
      ifield = (FIELD *) scinew FIELD((MESH *) imesh, Field::NODE);

    cerr << "has data...\n";
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
  } else {
    ifield = (FIELD *) scinew FIELD((MESH *) imesh, Field::NODE);
  }
  
  return FieldHandle( ifield );
}





template< class FIELD, class MESH, class NTYPE >
class NrrdFieldConverterFieldAlgoVector : public NrrdFieldConverterFieldAlgo
{
public:
  //! virtual interface.
  virtual FieldHandle execute(MeshHandle& mHandle,
		       NrrdDataHandle dataH,
		       int idim, int jdim, int kdim, int permute);

  virtual FieldHandle execute(MeshHandle& mHandle,
			      NrrdDataHandle dataH);  
};


template< class FIELD, class MESH, class NTYPE >
FieldHandle
NrrdFieldConverterFieldAlgoVector<FIELD, MESH, NTYPE>::
execute(MeshHandle& mHandle,
	NrrdDataHandle dataH,
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
    if (data_center == nrrdCenterCell)
      ifield = (FIELD *) scinew FIELD((MESH *) imesh, Field::FACE);
    else
      ifield = (FIELD *) scinew FIELD((MESH *) imesh, Field::NODE);

    typename FIELD::mesh_type::Node::iterator inodeItr;

    imesh->begin( inodeItr );

    register int i, j, k, index;
				  
    NTYPE *ptr = (NTYPE *)(dataH->nrrd->data);

    for( k=0; k<kdim; k++ ) {
      for( j=0; j<jdim; j++ ) {
	for( i=0; i<idim; i++ ) {
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
  } else {
    ifield = (FIELD *) scinew FIELD((MESH *) imesh, Field::NODE);
  }

  return FieldHandle( ifield );
  
}

template< class FIELD, class MESH, class NTYPE >
FieldHandle
NrrdFieldConverterFieldAlgoVector<FIELD, MESH, NTYPE>::
execute(MeshHandle& mHandle,
	NrrdDataHandle dataH)
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
    if (data_center == nrrdCenterCell)
      ifield = (FIELD *) scinew FIELD((MESH *) imesh, Field::FACE);
    else
      ifield = (FIELD *) scinew FIELD((MESH *) imesh, Field::NODE);

    typename FIELD::mesh_type::Node::iterator inodeItr, end;
    
    imesh->begin( inodeItr );
    imesh->end( end );
    
    NTYPE *ptr = (NTYPE *)(dataH->nrrd->data);
    int i = 0;
    while (inodeItr != end) {
      ifield->set_value( Vector( ptr[i*3],
				 ptr[i*3+1],
				 ptr[i*3+2]),
			 *inodeItr);
      ++inodeItr;
      i++;
    }
  } else {
    ifield = (FIELD *) scinew FIELD((MESH *) imesh, Field::NODE);
  }

  return FieldHandle( ifield );
}




template< class FIELD, class MESH, class NTYPE >
class NrrdFieldConverterFieldAlgoTensor : public NrrdFieldConverterFieldAlgo
{
public:
  //! virtual interface.
  virtual FieldHandle execute(MeshHandle& mHandle,
		       NrrdDataHandle dataH,
		       int idim, int jdim, int kdim, int permute);

  virtual FieldHandle execute(MeshHandle& mHandle,
			      NrrdDataHandle dataH);  
};


template< class FIELD, class MESH, class NTYPE >
FieldHandle
NrrdFieldConverterFieldAlgoTensor<FIELD, MESH, NTYPE>::
execute(MeshHandle& mHandle,
	NrrdDataHandle dataH,
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
    if (data_center == nrrdCenterCell)
      ifield = (FIELD *) scinew FIELD((MESH *) imesh, Field::FACE);
    else
      ifield = (FIELD *) scinew FIELD((MESH *) imesh, Field::NODE);

    typename FIELD::mesh_type::Node::iterator inodeItr;

    imesh->begin( inodeItr );

    register int i, j, k, index;
				  
    NTYPE *ptr = (NTYPE *)(dataH->nrrd->data);

    for( k=0; k<kdim; k++ ) {
      for( j=0; j<jdim; j++ ) {
	for( i=0; i<idim; i++ ) {
	  if( permute )
	    index = (k * jdim + j) * idim + i;
	  else 
	    index = (i * jdim + j) * kdim + k;
	
	  Tensor tmp;
	  if (nrrdKindSize( dataH->nrrd->axis[0].kind) == 6) {
	    // 3D symmetric tensor
	    tmp.mat_[0][0] = ptr[index*3];
	    tmp.mat_[0][1] = tmp.mat_[1][0] = ptr[index*3+1];
	    tmp.mat_[0][2] = tmp.mat_[2][0] = ptr[index*3+2];
	    tmp.mat_[1][1] = ptr[index*3+3];
	    tmp.mat_[1][2] = tmp.mat_[2][1] = ptr[index*3+4];
	    tmp.mat_[2][2] = ptr[index*3+5];
	    ifield->set_value( tmp, *inodeItr);
	  } else if (nrrdKindSize( dataH->nrrd->axis[0].kind) == 7) {
	    // 3D symmetric tensor with mask
	    // if mask < 0.5, all tensor values are 0
	    if (ptr[index*3] < 0.5) {
	      for (int x=0; x<3; x++) 
		for (int y=0; y<3; y++)
		  tmp.mat_[x][y]=0;

	    } else {
	      // skip mask
	      tmp.mat_[0][0] = ptr[index*3+1];
	      tmp.mat_[0][1] = tmp.mat_[1][0] = ptr[index*3+2];
	      tmp.mat_[0][2] = tmp.mat_[2][0] = ptr[index*3+3];
	      tmp.mat_[1][1] = ptr[index*3+4];
	      tmp.mat_[1][2] = tmp.mat_[2][1] = ptr[index*3+5];
	      tmp.mat_[2][2] = ptr[index*3+6];	      
	    }
	    ifield->set_value( tmp, *inodeItr);
	  } else if (nrrdKindSize( dataH->nrrd->axis[0].kind) == 9) {
	    // not symmetric, do a straight across copy
	    tmp.mat_[0][0] = ptr[index*3];
	    tmp.mat_[0][1] = ptr[index*3+1];
	    tmp.mat_[0][2] = ptr[index*3+2];
	    tmp.mat_[1][0] = ptr[index*3+3];
	    tmp.mat_[1][1] = ptr[index*3+4];
	    tmp.mat_[1][2] = ptr[index*3+4];
	    tmp.mat_[2][0] = ptr[index*3+5];
	    tmp.mat_[2][1] = ptr[index*3+6];
	    tmp.mat_[2][2] = ptr[index*3+7];
	  } else {
	    return FieldHandle( ifield );
	  }
	
	  ++inodeItr;
	}
      }
    }
  } else {
    ifield = (FIELD *) scinew FIELD((MESH *) imesh, Field::NODE);
  }

  return FieldHandle( ifield );
  
}

template< class FIELD, class MESH, class NTYPE >
FieldHandle
NrrdFieldConverterFieldAlgoTensor<FIELD, MESH, NTYPE>::
execute(MeshHandle& mHandle,
	NrrdDataHandle dataH)
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
    if (data_center == nrrdCenterCell)
      ifield = (FIELD *) scinew FIELD((MESH *) imesh, Field::FACE);
    else
      ifield = (FIELD *) scinew FIELD((MESH *) imesh, Field::NODE);

    typename FIELD::mesh_type::Node::iterator inodeItr, end;
    
    imesh->begin( inodeItr );
    imesh->end( end );
    
    NTYPE *ptr = (NTYPE *)(dataH->nrrd->data);
    int i = 0;
    while (inodeItr != end) {
      Tensor tmp;
      if (nrrdKindSize( dataH->nrrd->axis[0].kind) == 6) {
	// 3D symmetric tensor
	tmp.mat_[0][0] = ptr[index*3];
	tmp.mat_[0][1] = tmp.mat_[1][0] = ptr[index*3+1];
	tmp.mat_[0][2] = tmp.mat_[2][0] = ptr[index*3+2];
	tmp.mat_[1][1] = ptr[index*3+3];
	tmp.mat_[1][2] = tmp.mat_[2][1] = ptr[index*3+4];
	tmp.mat_[2][2] = ptr[index*3+5];
	ifield->set_value( tmp, *inodeItr);
      } else if (nrrdKindSize( dataH->nrrd->axis[0].kind) == 7) {
	// 3D symmetric tensor with mask
	// if mask < 0.5, all tensor values are 0
	if (ptr[index*3] < 0.5) {
	  for (int x=0; x<3; x++) 
	    for (int y=0; y<3; y++)
	      tmp.mat_[x][y]=0;
	  
	} else {
	  // skip mask
	  tmp.mat_[0][0] = ptr[index*3+1];
	  tmp.mat_[0][1] = tmp.mat_[1][0] = ptr[index*3+2];
	  tmp.mat_[0][2] = tmp.mat_[2][0] = ptr[index*3+3];
	  tmp.mat_[1][1] = ptr[index*3+4];
	  tmp.mat_[1][2] = tmp.mat_[2][1] = ptr[index*3+5];
	  tmp.mat_[2][2] = ptr[index*3+6];	      
	}
	ifield->set_value( tmp, *inodeItr);
      } else if (nrrdKindSize( dataH->nrrd->axis[0].kind) == 9) {
	// not symmetric, do a straight across copy
	tmp.mat_[0][0] = ptr[index*3];
	tmp.mat_[0][1] = ptr[index*3+1];
	tmp.mat_[0][2] = ptr[index*3+2];
	tmp.mat_[1][0] = ptr[index*3+3];
	tmp.mat_[1][1] = ptr[index*3+4];
	tmp.mat_[1][2] = ptr[index*3+4];
	tmp.mat_[2][0] = ptr[index*3+5];
	tmp.mat_[2][1] = ptr[index*3+6];
	tmp.mat_[2][2] = ptr[index*3+7];
      } else {
	return FieldHandle( ifield );
      }
      ++inodeItr;
      i++;
    }    
  } else {
    ifield = (FIELD *) scinew FIELD((MESH *) imesh, Field::NODE);
  }
  return FieldHandle( ifield );
}
  


} // end namespace SCITeem

#endif // NrrdFieldConverter_h
