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
 * FILE: matlabconverter.h
 * AUTH: Jeroen G Stinstra
 * DATE: 18 MAR 2004
 */
 
#ifndef JGS_MATLABIO_MATLABCONVERTER_H
#define JGS_MATLABIO_MATLABCONVERTER_H 1

/*
 *  This class converts matlab matrices into SCIRun objects and vice versa.
 *  The class is more a collection of functions then a real object.
 */

#define HAVE_TEEM_PACKAGE	1

#include <vector>
#include <string>
#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/PointCloudField.h>
#include <Core/Datatypes/PointCloudMesh.h>
#include <Core/Datatypes/CurveField.h>
#include <Core/Datatypes/CurveMesh.h>
#include <Core/Datatypes/StructCurveField.h>
#include <Core/Datatypes/StructCurveMesh.h>
#include <Core/Datatypes/TriSurfField.h>
#include <Core/Datatypes/TriSurfMesh.h>
#include <Core/Datatypes/QuadSurfField.h>
#include <Core/Datatypes/QuadSurfMesh.h>
#include <Core/Datatypes/StructQuadSurfField.h>
#include <Core/Datatypes/StructQuadSurfMesh.h>
#include <Core/Datatypes/TetVolField.h>
#include <Core/Datatypes/TetVolMesh.h>
#include <Core/Datatypes/PrismVolField.h>
#include <Core/Datatypes/PrismVolMesh.h>
#include <Core/Datatypes/StructHexVolField.h>
#include <Core/Datatypes/StructHexVolMesh.h>
#include <Core/Datatypes/HexVolField.h>
#include <Core/Datatypes/HexVolMesh.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Tensor.h>

#ifdef HAVE_TEEM_PACKAGE
#include <Packages/Teem/Dataflow/Ports/NrrdPort.h>
#include <Packages/Teem/Core/Datatypes/NrrdData.h>
#endif

#include <Packages/MatlabInterface/Core/Datatypes/matlabfile.h>
#include <Packages/MatlabInterface/Core/Datatypes/matlabarray.h>


 
/*
 * CLASS DESCRIPTION
 * Class for dealing with the conversion of "matlab" objects into
 * SCIRun objects.
 * 
 * MEMORY MODEL
 * Functions use the matlabarray / matfiledata classes to allocate data.
 * All memory allocate is associated with the objects and is deallocated
 * by their destructors
 *
 * ERROR HANDLING
 * All errors are reported as exceptions:
 * the matlabconverter_error class.
 * 
 * COPYING/ASSIGNMENT
 * There is no data stored in the object, so it can be copied indefinitely
 *
 * RESOURCE ALLOCATION
 * no external resource are used
 *
 */ 


 namespace MatlabIO {
 
 class matlabconverter : public matfilebase {
 
 public:
 
 // Exception class indicating an error in the converter
 // class itself.
	class matlabconverter_error : public matfileerror {};
 
 public:

	// Functions for converting back and forward of 
	// Compatible function:
	// indicates whether a matrix can be converted, for this purpose only
	// the header of the matlabmatrix is read into memory. The function also
	// returns a inforamation string describing the object for use in the GUI
	// This function returns a value 0 if the object cannot be converted and
	// a positive number if it is compatible. The higher the number the more
	// likely it is the users wants to read this matlab array. 
	// The latter classification is based on some simple rules, like matrices
	// are probably of the double format etc.
	//
	// mlArrayTO.... function:
	// Convert a matlab array into a SCIRun object. If the object is compatible 
	// this function should deal with the conversion. Depending on the information
	// inthe object, fields like the property manager will be filled out.
	//
	// ....TOmlMatrix function:
	// Convert a SCIRun object into a matlabarray. This version will produce a 
	// pure numeric array, with only the numeric values, every other field will
	// be stripped and does not reappear in matlab.
	//
	// ....TOmlArray function:
	// Try to convert a SCIRun object as complete as possible into a matlab field
	// properties are stored and as well axis names, units etc. This function will
	// create a matlab structured arrazy, with each field representing different
	// parts of the original SCIRun object.
	//
	// Limitations:
	// PropertyManager:
	// Current implementation only allows for translation of strings. Any numeric
	// property like vectors and scalars are still ignored. (Limitation is partly
	// due to inavailablity of type information in the property manager)
	// Complex numbers:
	// Currently only the real parts of complex numbers are taken. SCIRun does not
	// support complex numbers!
	// Nrrd key value pairs:
	// These key value pairs are not supported yet, like in the rest of SCIRun

	// SCIRun MATRICES
	long sciMatrixCompatible(matlabarray &mlarray, std::string &infostring);
	void mlArrayTOsciMatrix(matlabarray &mlmat,SCIRun::MatrixHandle &scimat);
	void sciMatrixTOmlMatrix(SCIRun::MatrixHandle &scimat,matlabarray &mlmat,matlabarray::mitype dataformat = matlabarray::miDOUBLE);
	void sciMatrixTOmlArray(SCIRun::MatrixHandle &scimat,matlabarray &mlmat,matlabarray::mitype dataformat = matlabarray::miDOUBLE);

#ifdef HAVE_TEEM_PACKAGE
	// SCIRun NRRDS
	long sciNrrdDataCompatible(matlabarray &mlarray, std::string &infostring);
	void mlArrayTOsciNrrdData(matlabarray &mlmat,SCITeem::NrrdDataHandle &scinrrd);
	void sciNrrdDataTOmlMatrix(SCITeem::NrrdDataHandle &scinrrd, matlabarray &mlmat,matlabarray::mitype dataformat = matlabarray::miDOUBLE);
	void sciNrrdDataTOmlArray(SCITeem::NrrdDataHandle &scinrrd, matlabarray &mlmat,matlabarray::mitype dataformat = matlabarray::miDOUBLE);
#endif

	// SCIRun Fields/Meshs
	long sciFieldCompatible(matlabarray &mlarray,std::string &infostring);
	void mlArrayTOsciField(matlabarray &mlarray,SCIRun::FieldHandle &scifield);


	// SUPPORT FUNCTIONS
	// Test whether the proposed name of a matlab matrix is valid.
	bool isvalidmatrixname(std::string name);

 private:
 
	// FUNCTIONS FOR TRANSLATING THE PROPERTY MANAGER
	// add the field "property" in a matlabarray to a scirun property manager
	void mlPropertyTOsciProperty(matlabarray &ma,SCIRun::PropertyManager *handle);
	// the other way around
	void sciPropertyTOmlProperty(SCIRun::PropertyManager *handle,matlabarray &ma);


#ifdef HAVE_TEEM_PACKAGE
	// support functions for converting matlab arrays to nrrds and vice versa
	unsigned int convertmitype(matlabarray::mitype type);

	matlabarray::mitype convertnrrdtype(int type);
#endif

	// FUNCTIONS FOR CONVERTING FIELDS

private:
	
	struct fieldstruct
	{
		// mesh information
		matlabarray node; 
		matlabarray snode;  // structured mesh version
		matlabarray edge;
		matlabarray face;
		matlabarray cell;
		
		// structured mesh submatrices
		matlabarray x;
		matlabarray y;
		matlabarray z;
		
		// field information
		matlabarray scalarfield;
		matlabarray vectorfield;
		matlabarray tensorfield;
		matlabarray fieldlocation;
		
		// element definition (to be extended for more mesh classes)
		matlabarray elemtype;
		
		// In SCIRun field can have a name, so this is a simple way to add that
		matlabarray name;
		
		// Property matrix to set properties
		matlabarray property;
	};

	// analyse a matlab matrix and sort out all the different fieldname
	// combinations
	fieldstruct analyzefieldstruct(matlabarray &ma);
	template<class FIELDPTR> void addscalardata(FIELDPTR fieldptr,matlabarray mlarray);
	template<class FIELDPTR> void addvectordata(FIELDPTR fieldptr,matlabarray mlarray);
	template<class FIELDPTR> void addtensordata(FIELDPTR fieldptr,matlabarray mlarray);
	template<class MESH>   void addnodes(SCIRun::LockingHandle<MESH> meshH,matlabarray mlarray);
	template<class MESH>   void addedges(SCIRun::LockingHandle<MESH> meshH,matlabarray mlarray);
	template<class MESH>   void addfaces(SCIRun::LockingHandle<MESH> meshH,matlabarray mlarray);
	template<class MESH>   void addcells(SCIRun::LockingHandle<MESH> meshH,matlabarray mlarray);


 };
 
 } // end namespace

#endif
