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

//    File   : PPPLHDF5FieldReader.h
//    Author : Allen Sanderson
//             School of Computing
//             University of Utah
//    Date   : May 2003

#if !defined(PPPLHDF5FieldReader_h)
#define PPPLHDF5FieldReader_h

#include <Dataflow/Network/Module.h>

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>

#include <Core/Datatypes/StructHexVolField.h>
#include <Core/Datatypes/StructQuadSurfField.h>
#include <Core/Math/Trig.h>

#include <Core/GuiInterface/GuiVar.h>

#include <Packages/Fusion/share/share.h>

namespace Fusion {

using namespace SCIRun;

class FusionSHARE PPPLHDF5FieldReader : public Module {
public:
  PPPLHDF5FieldReader(GuiContext *context);

  virtual ~PPPLHDF5FieldReader();

  virtual void execute();

  float* readGrid( string filename );
  float* readData( string filename );

  virtual void tcl_command(GuiArgs&, void*);

private:
  GuiString filename_;
  GuiString dumpname_;

  GuiInt  nTimeSteps_;
  GuiInt  timeStep_;

  GuiInt  nDataSets_;
  GuiInt  dataSet_;

  GuiInt nDims_;

  GuiInt iDim_;
  GuiInt jDim_;
  GuiInt kDim_;

  GuiInt iStart_;
  GuiInt jStart_;
  GuiInt kStart_;

  GuiInt iCount_;
  GuiInt jCount_;
  GuiInt kCount_;

  GuiInt iStride_;
  GuiInt jStride_;
  GuiInt kStride_;

  GuiInt iWrap_;
  GuiInt jWrap_;
  GuiInt kWrap_;

  string old_filename_;
  time_t old_filemodification_;

  int timestep_;
  int dataset_;

  int idim_;
  int jdim_;
  int kdim_;

  int istart_;
  int jstart_;
  int kstart_;

  int icount_;
  int jcount_;
  int kcount_;

  int istride_;
  int jstride_;
  int kstride_;

  int iwrap_;
  int jwrap_;
  int kwrap_;

  int rank_;

  int fGeneration_;
  FieldHandle  pHandle_;
};


class PPPLHDF5FieldReaderMeshAlgo : public DynamicAlgoBase
{
public:
  virtual void execute(MeshHandle src,
		       int idim, int jdim, int kdim,
		       int iwrap, int jwrap, int kwrap,
		       float *grid) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *mtd);
};


template< class MESH >
class PPPLHDF5FieldReaderMeshAlgoT : public PPPLHDF5FieldReaderMeshAlgo
{
public:
  virtual void execute(MeshHandle src,
		       int idim, int jdim, int kdim,
		       int iwrap, int jwrap, int kwrap,
		       float *grid);
};

template< class MESH >
void
PPPLHDF5FieldReaderMeshAlgoT< MESH >::execute(MeshHandle src,
					      int idim, int jdim, int kdim,
					      int iwrap, int jwrap, int kwrap,
					      float *grid)
{
  MESH *imesh = (MESH *) src.get_rep();
  typename MESH::Node::iterator inodeItr;

  imesh->begin( inodeItr );

  register int i, j, k;
  
  for( k=0; k<kdim + kwrap; k++ ) {
    for( j=0; j<jdim + jwrap; j++ ) {
      for( i=0; i<idim + iwrap; i++ ) {
	
	int index = ((i%idim) * jdim + (j%jdim)) * kdim + (k%kdim);

	// Grid
	float xVal = grid[index*3 + 0];
	float yVal = grid[index*3 + 1];
	float zVal = grid[index*3 + 2];
	
	imesh->set_point(Point(xVal, yVal, zVal), *inodeItr);

	++inodeItr;
      }
    }
  }
}


class PPPLHDF5FieldReaderFieldAlgo : public DynamicAlgoBase
{
public:
  virtual void execute(FieldHandle src,
		       int idim, int jdim, int kdim,
		       int iwrap, int jwrap, int kwrap,
		       float *data) = 0;
  
  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *ftd,
					    int rank);
};

template< class FIELD >
class PPPLHDF5FieldReaderFieldAlgoScalar : public PPPLHDF5FieldReaderFieldAlgo
{
public:
  //! virtual interface.
  virtual void execute(FieldHandle src,
		       int idim, int jdim, int kdim,
		       int iwrap, int jwrap, int kwrap,
		       float *data);
};


template< class FIELD >
void
PPPLHDF5FieldReaderFieldAlgoScalar<FIELD>::execute(FieldHandle src,
						   int idim,
						   int jdim,
						   int kdim,
						   int iwrap,
						   int jwrap,
						   int kwrap,
						   float *data)
{
  FIELD *ifield = (FIELD *) src.get_rep();
  typename FIELD::mesh_handle_type imesh = ifield->get_typed_mesh();
  typename FIELD::mesh_type::Node::iterator inodeItr;

  imesh->begin( inodeItr );

  register int i, j, k;
  
  for( k=0; k<kdim + kwrap; k++ ) {
    for( j=0; j<jdim + jwrap; j++ ) {
      for( i=0; i<idim + iwrap; i++ ) {
	
	int index = ((i%idim) * jdim + (j%jdim)) * kdim + (k%kdim);
	
	// Value
	ifield->set_value( data[index], *inodeItr);
	
	++inodeItr;
      }
    }
  }
}

template< class FIELD >
class PPPLHDF5FieldReaderFieldAlgoVector : public PPPLHDF5FieldReaderFieldAlgo
{
public:
  //! virtual interface.
  virtual void execute(FieldHandle src,
		       int idim, int jdim, int kdim,
		       int iwrap, int jwrap, int kwrap,
		       float *data);
};


template< class FIELD >
void
PPPLHDF5FieldReaderFieldAlgoVector<FIELD>::execute(FieldHandle src,
						   int idim,
						   int jdim,
						   int kdim,
						   int iwrap,
						   int jwrap,
						   int kwrap,
						   float *data)
{
  FIELD *ifield = (FIELD *) src.get_rep();
  typename FIELD::mesh_handle_type imesh = ifield->get_typed_mesh();
  typename FIELD::mesh_type::Node::iterator inodeItr;

  imesh->begin( inodeItr );

  register int i, j, k;
  
  for( k=0; k<kdim + kwrap; k++ ) {
    for( j=0; j<jdim + jwrap; j++ ) {
      for( i=0; i<idim + iwrap; i++ ) {
	
	int index = (((i%idim) * jdim + (j%jdim)) * kdim + (k%kdim)) * 3;
	
	// Value
	ifield->set_value( Vector( data[index], data[index+1], data[index+2]),
			   *inodeItr);
	
	++inodeItr;
      }
    }
  }
}

} // end namespace SCIRun

#endif // PPPLHDF5FieldReader_h
