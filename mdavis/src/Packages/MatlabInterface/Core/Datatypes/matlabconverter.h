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
 *
 *  The functions in this class are an attempt to bridge between the C++ kind
 *  of Object Oriented data management, towards a more classical way orginazing
 *  data in an ensamble of arrays of differrnt types
 *
 *  As SCIRun was not designed to easily extract and insert data the conversion
 *  algorithms are far from perfect and will need constant updating. A different
 *  way of managing data within SCIRun would greatly enhance the usability of
 *  SCIRun and make the conversions less cumbersome
 * 
 */

/* 
 * SCIRun data types have a lot of different classes, hence we need to include
 * a large number of class definitions......
 */
 
#define HAVE_BUNDLE 1
 
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <string>
#include <sgi_stl_warnings_on.h>

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
#include <Core/Datatypes/ScanlineField.h>
#include <Core/Datatypes/StructHexVolField.h>
#include <Core/Datatypes/StructHexVolMesh.h>
#include <Core/Datatypes/HexVolField.h>
#include <Core/Datatypes/HexVolMesh.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Geom/ColorMap.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Util/DynamicCompilation.h>
#include <Core/Util/ProgressReporter.h>

#include <Dataflow/Ports/NrrdPort.h>

#ifdef HAVE_BUNDLE
#include <Core/Bundle/Bundle.h>
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
 * Only the converter options are stored in the object and thence the 
 * object can be copied without any problems.
 *
 * RESOURCE ALLOCATION
 * no external resources are used
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

   // Constructor
   matlabconverter();

   // SET CONVERTER OPTIONS:
   // Data type sets the export type of the data
   void setdatatype(matlabarray::mitype dataformat);
   // Index base sets the index base used for indices in for example geometries
   void setindexbase(long indexbase);
   // In a numericmatrix all data will be stripped and the data will be saved as
   // a plain dense or sparse matrix.
   void setdisabletranspose(bool dt);
   void converttonumericmatrix();
   void converttostructmatrix();
   
   // The following options are for controlling the conversion to bundles
   // In case prefernrrds is set, numerical data is converted into nrrds
   // only sparse matrices become matrices. If prefermatrices is set, the
   // behavior is opposite and only ND (N>2) matrices become nrrds.
   
   void prefernrrds();
   void prefermatrices();
   
   // Since Bundles can be bundled, a choice needs to be made whether structured
   // matlab matrices should become bundles or if possible should be converted into
   // matrices/nrrds or fields. In case prefer bundles is set, a matlab structure will
   // be decomposed into bundles of sub bundles and of nrrds and matrices. In case
   // prefersciobjects is set each structure is read and if it can be translated into
   // a sciobject it will be come a field, nrrd or matrix and only at the last
   // resort it will be a bundle. Note that the comparison is done to see whether the
   // required number of fields is there if so other fields are ignored.
   void preferbundles();
   void prefersciobjects();

   // SCIRun MATRICES
   long sciMatrixCompatible(matlabarray &mlarray, std::string &infostring, SCIRun::ProgressReporter* pr);
   void mlArrayTOsciMatrix(matlabarray &mlmat,SCIRun::MatrixHandle &scimat, SCIRun::ProgressReporter* pr);
   void sciMatrixTOmlArray(SCIRun::MatrixHandle &scimat,matlabarray &mlmat, SCIRun::ProgressReporter* pr);

   // SCIRun NRRDS
   long sciNrrdDataCompatible(matlabarray &mlarray, std::string &infostring, SCIRun::ProgressReporter* pr);
   void mlArrayTOsciNrrdData(matlabarray &mlmat,SCIRun::NrrdDataHandle &scinrrd, SCIRun::ProgressReporter* pr);
   void sciNrrdDataTOmlArray(SCIRun::NrrdDataHandle &scinrrd, matlabarray &mlmat, SCIRun::ProgressReporter* pr);

#ifdef HAVE_BUNDLE
   // SCIRun Bundles (Currently contained in the CardioWave Package)
   long sciBundleCompatible(matlabarray &mlarray, std::string &infostring, SCIRun::ProgressReporter* pr);
   void mlArrayTOsciBundle(matlabarray &mlmat, SCIRun::BundleHandle &scibundle, SCIRun::ProgressReporter* pr);
   void sciBundleTOmlArray(SCIRun::BundleHandle &scibundle, matlabarray &mlmat,SCIRun::ProgressReporter* pr);
#endif

    // Reading of Matlabn colormaps
   long sciColorMapCompatible(matlabarray &mlarray, std::string &infostring, SCIRun::ProgressReporter* pr);
   void mlArrayTOsciColorMap(matlabarray &mlmat,SCIRun::ColorMapHandle &scinrrd, SCIRun::ProgressReporter* pr);


   // The reference status of the reader/compatible prs has been changed.
   // So I can change the contents without affecting the matrices at the input
   // This is a cleaner solution. 

   // SCIRun Fields/Meshes
   long sciFieldCompatible(matlabarray mlarray,std::string &infostring, SCIRun::ProgressReporter* pr);

   // DYNAMICALLY COMPILING CONVERTERS
   // Note: add the pointer from the Module which makes the call, so the user sees
   // the ouput of the dynamic compilation phase, for example
   //   matlabconverter translate;
   //   translate.sciFieldTOmlArray(scifield,mlarray,this);
   // ALL the dynamic code is in the matlabconverter class it just needs a pointer to the pr
   // class.
	
   void mlArrayTOsciField(matlabarray mlarray,SCIRun::FieldHandle &scifield,SCIRun::ProgressReporter* pr);
   void sciFieldTOmlArray(SCIRun::FieldHandle &scifield,matlabarray &mlarray,SCIRun::ProgressReporter* pr);


   // SUPPORT FUNCTIONS
   // Test whether the proposed name of a matlab matrix is valid.

   bool	isvalidmatrixname(std::string name);
   void	setpostmsg(bool val);

 private:
   // FUNCTIONS FOR COMMUNICATING WITH THE USER
   void	postmsg(SCIRun::ProgressReporter* pr, std::string msg);
   bool	postmsg_;

   // THE REST OF THE FUNCTIONS ARE PRIVATE
 private:
   // FUNCTION FOR TRANSLATING THE CONTENTS OF A MATRIX (THE NUMERIC PART OF THE DATA)
   void sciMatrixTOmlMatrix(SCIRun::MatrixHandle &scimat,matlabarray &mlmat);

   // FUNCTIONS FOR TRANSLATING THE PROPERTY MANAGER
   // add the field "property" in a matlabarray to a scirun property manager
   void mlPropertyTOsciProperty(matlabarray &ma,SCIRun::PropertyManager *handle);
   // the other way around
   void sciPropertyTOmlProperty(SCIRun::PropertyManager *handle,matlabarray &ma);


   // FUNCTIONS FOR TRANSLATING THE CONTENTS OF A NRRD (THE NUMERIC PART OF THE DATA)
   void sciNrrdDataTOmlMatrix(SCIRun::NrrdDataHandle &scinrrd, matlabarray &mlmat);
   unsigned int convertmitype(matlabarray::mitype type);
   matlabarray::mitype convertnrrdtype(int type);

   // ALTHOUGH FIELDSTRUCT IS PUBLIC DO NOT USE IT, IT NEEDS TO BE PUBLIC FOR THE DYNAMIC COMPILER
   // FRIENDS STATEMENTS WITH TEMPLATED CLASSES SEEM TO COMPLAIN LOT, HENCE I MADE THEM JUST PUBLIC
 public:

   struct fieldstruct
   {
     // unstructured mesh submatrices
     matlabarray node; 
     matlabarray edge;
     matlabarray face;
     matlabarray cell;
		
     // structured mesh submatrices
     matlabarray x;
     matlabarray y;
     matlabarray z;
		
     // structured regular meshes
		
     matlabarray dims;
     matlabarray transform;
		
     // field information
     matlabarray scalarfield;
     matlabarray vectorfield;
     matlabarray tensorfield;

     matlabarray field;
     matlabarray fieldlocation;
     matlabarray fieldtype;
		
     // element definition (to be extended for more mesh classes)
     matlabarray meshclass;
		
     // In SCIRun field can have a name, so this is a simple way to add that
     matlabarray name;
		
     // Property matrix to set properties
     matlabarray property;
		
     // Interpolation matrices as used in CVRTI
     matlabarray interp;
    
     int  basis_order;
   };



 private:

   // CONVERTER OPTIONS:
	
   // Matrix should be translated as a numeric matrix directly
   bool numericarray_;
   // Specify the indexbase for the output
   long indexbase_;
   // Specify the data of output data
   matlabarray::mitype datatype_;
   // Disable transposing matrices from Fortran format to C++ format
   bool disable_transpose_;
	
   // Options for translation of structures into bundled objects
   bool prefer_nrrds;
   bool prefer_bundles;
     
   // FUNCTIONS FOR CONVERTING FIELDS:
	
   // analyse a matlab matrix and sort out all the different fieldname
   // combinations
   fieldstruct analyzefieldstruct(matlabarray &ma);

   // These need to be public for the dynamic compilation, but do not use these
	
   void uncompressedtensor(std::vector<double> &fielddata,SCIRun::Tensor &tens, unsigned int p);
   void compressedtensor(std::vector<double> &fielddata,SCIRun::Tensor &tens, unsigned int p);	
	
	
   // make sure the dynamic code has access
   // Do NOT use any of the folowing functions, they are for dynamic compiled code only

 public:

   template <class T>		bool addfield(T &fdata,matlabarray mlarray);
   template <class T>		bool addfield(std::vector<T> &fdata,matlabarray mlarray);
   template <class T>		bool addfield(SCIRun::FData2d<T> &fdata,matlabarray mlarray);
   template <class T>		bool addfield(SCIRun::FData3d<T> &fdata,matlabarray mlarray);

   bool addfield(std::vector<SCIRun::Vector> &fdata,matlabarray mlarray);
   bool addfield(SCIRun::FData2d<SCIRun::Vector> &fdata,matlabarray mlarray);
   bool addfield(SCIRun::FData3d<SCIRun::Vector> &fdata,matlabarray mlarray);
   bool addfield(std::vector<SCIRun::Tensor> &fdata,matlabarray mlarray);
   bool addfield(SCIRun::FData2d<SCIRun::Tensor> &fdata,matlabarray mlarray);
   bool addfield(SCIRun::FData3d<SCIRun::Tensor> &fdata,matlabarray mlarray);

   template<class MESH>	bool createmesh(SCIRun::LockingHandle<MESH> &meshH,fieldstruct &fs);
   bool createmesh(SCIRun::LockingHandle<SCIRun::PointCloudMesh> &mesH, fieldstruct &fs);
   bool createmesh(SCIRun::LockingHandle<SCIRun::CurveMesh> &mesH, fieldstruct &fs);
   bool createmesh(SCIRun::LockingHandle<SCIRun::TriSurfMesh> &mesH, fieldstruct &fs);
   bool createmesh(SCIRun::LockingHandle<SCIRun::QuadSurfMesh> &mesH, fieldstruct &fs);
   bool createmesh(SCIRun::LockingHandle<SCIRun::TetVolMesh> &mesH, fieldstruct &fs);
   bool createmesh(SCIRun::LockingHandle<SCIRun::HexVolMesh> &mesH, fieldstruct &fs);
   bool createmesh(SCIRun::LockingHandle<SCIRun::PrismVolMesh> &mesH, fieldstruct &fs);
   bool createmesh(SCIRun::LockingHandle<SCIRun::StructCurveMesh> &mesH, fieldstruct &fs);
   bool createmesh(SCIRun::LockingHandle<SCIRun::StructQuadSurfMesh> &mesH, fieldstruct &fs);
   bool createmesh(SCIRun::LockingHandle<SCIRun::StructHexVolMesh> &mesH, fieldstruct &fs);
   bool createmesh(SCIRun::LockingHandle<SCIRun::ScanlineMesh> &mesH, fieldstruct &fs);
   bool createmesh(SCIRun::LockingHandle<SCIRun::ImageMesh> &mesH, fieldstruct &fs);
   bool createmesh(SCIRun::LockingHandle<SCIRun::LatVolMesh> &mesH, fieldstruct &fs);


   template<class MESH>	void addnodes(SCIRun::LockingHandle<MESH> meshH,matlabarray mlarray);
   template<class MESH>	void addedges(SCIRun::LockingHandle<MESH> meshH,matlabarray mlarray);
   template<class MESH>	void addfaces(SCIRun::LockingHandle<MESH> meshH,matlabarray mlarray);
   template<class MESH>	void addcells(SCIRun::LockingHandle<MESH> meshH,matlabarray mlarray);


							

   // Functions for dynamic translation on the WRITER
   // At least as far as you can call it dynamic compilation ....
   // The difference between the meshes is to big to have one polymorphic class
   // dealing with them all. 
	
   void mladdmeshclass(std::string meshclass,matlabarray mlarray);	
   template<class FIELD>	void mladdfieldat(FIELD *scifield,matlabarray mlarray);

   // Converters for each mesh class. These converters are precompiled as they all need different
   // functions for conversion. As soon as a function is not a template it will be precompiled
   // Since geometry converters are individually tuned, they cannot be templated. The current
   // dynamic compilation model does not account for this
	
   template<class MESH>    bool mladdmesh(SCIRun::LockingHandle<MESH> meshH, matlabarray mlarray);
   bool mladdmesh(SCIRun::LockingHandle<SCIRun::PointCloudMesh> meshH,matlabarray mlarray);
   bool mladdmesh(SCIRun::LockingHandle<SCIRun::CurveMesh> meshH,matlabarray mlarray);
   bool mladdmesh(SCIRun::LockingHandle<SCIRun::TriSurfMesh> meshH,matlabarray mlarray);
   bool mladdmesh(SCIRun::LockingHandle<SCIRun::QuadSurfMesh> meshH,matlabarray mlarray);
   bool mladdmesh(SCIRun::LockingHandle<SCIRun::TetVolMesh>meshH,matlabarray mlarray);
   bool mladdmesh(SCIRun::LockingHandle<SCIRun::HexVolMesh> meshH,matlabarray mlarray);
   bool mladdmesh(SCIRun::LockingHandle<SCIRun::PrismVolMesh>meshH,matlabarray mlarray);
   bool mladdmesh(SCIRun::LockingHandle<SCIRun::StructHexVolMesh> meshH,matlabarray mlarray);
   bool mladdmesh(SCIRun::LockingHandle<SCIRun::StructQuadSurfMesh> meshH,matlabarray mlarray);							
   bool mladdmesh(SCIRun::LockingHandle<SCIRun::StructCurveMesh> meshH,matlabarray mlarray);
   bool mladdmesh(SCIRun::LockingHandle<SCIRun::ScanlineMesh> meshH,matlabarray mlarray);
   bool mladdmesh(SCIRun::LockingHandle<SCIRun::ImageMesh> meshH,matlabarray mlarray);							
   bool mladdmesh(SCIRun::LockingHandle<SCIRun::LatVolMesh> meshH,matlabarray mlarray);	
	
   // Templates for the mesh generation																
   template<class MESH>    void mladdnodesfield(SCIRun::LockingHandle<MESH> meshH,matlabarray mlarray);
   template<class MESH>    void mladdedgesfield(SCIRun::LockingHandle<MESH> meshH,matlabarray mlarray,unsigned int num);
   template<class MESH>    void mladdfacesfield(SCIRun::LockingHandle<MESH> meshH,matlabarray mlarray,unsigned int num);
   template<class MESH>    void mladdcellsfield(SCIRun::LockingHandle<MESH> meshH,matlabarray mlarray,unsigned int num);
   template<class MESH>	void mladdtransform(SCIRun::LockingHandle<MESH> meshH,matlabarray mlarray);
	
   void mladdxyznodes(SCIRun::LockingHandle<SCIRun::StructCurveMesh> meshH,matlabarray mlarray);
   void mladdxyznodes(SCIRun::LockingHandle<SCIRun::StructQuadSurfMesh> meshH,matlabarray mlarray);
   void mladdxyznodes(SCIRun::LockingHandle<SCIRun::StructHexVolMesh> meshH,matlabarray mlarray);							
	
   // Converters for the different field classes in SCIRun
   // Some are precompiled, others are compiled on the fly
   // The Tensor and Vector ones are precompiled and are currently not templated						
   template<class T>		bool mladdfield(std::vector<T> &fdata,matlabarray mlarray);
   bool mladdfield(std::vector<SCIRun::Vector> &fdata,matlabarray mlarray);
   bool mladdfield(std::vector<SCIRun::Tensor> &fdata,matlabarray mlarray);
   template<class T>		bool mladdfield(SCIRun::FData2d<T> &fdata,matlabarray mlarray);		
   bool mladdfield(SCIRun::FData2d<SCIRun::Vector> &fdata,matlabarray mlarray);	
   bool mladdfield(SCIRun::FData2d<SCIRun::Tensor> &fdata,matlabarray mlarray);	
   template<class T>		bool mladdfield(SCIRun::FData3d<T> &fdata,matlabarray mlarray);							
   bool mladdfield(SCIRun::FData3d<SCIRun::Vector> &fdata,matlabarray mlarray);	
   bool mladdfield(SCIRun::FData3d<SCIRun::Tensor> &fdata,matlabarray mlarray);	

 };
 
 ////////////// HERE CODE FOR DYNAMIC FIELDWRITER STARTS ///////////////

// Default function for mladdmesh, in case no suitable converter is found,
// this one will inform the code that none is available. The return(false) will return in an
// error message for the user
template<class MESH>   bool matlabconverter::mladdmesh(SCIRun::LockingHandle<MESH> meshH, matlabarray mlarray)
{
	return(false);
}
 
// Check all possible positions of the field data
template<class FIELD> void matlabconverter::mladdfieldat(FIELD *scifield,matlabarray mlarray)
{
	matlabarray fieldat;
    if (scifield->basis_order() == 1) fieldat.createstringarray("node");
    if (scifield->basis_order() == 0) fieldat.createstringarray("cell");
    if (scifield->basis_order() == -1) fieldat.createstringarray("none");
    if (scifield->basis_order() > 1) fieldat.createstringarray("higher order");
	mlarray.setfield(0,"fieldat",fieldat);
	
	matlabarray basisorder;
	basisorder.createdoublescalar(static_cast<double>(scifield->basis_order()));
	mlarray.setfield(0,"basisorder",basisorder);
}
 
template <class MESH> void matlabconverter::mladdtransform(SCIRun::LockingHandle<MESH> meshH,matlabarray mlarray)
{
	matlabarray dim;
	matlabarray transform;
	
	// Obtain dimensions and make a matrix with the dimensions of the mesh
	// This one is needed when we have an image/latvol without any data
	// Then dim defines the mesh
	std::vector<unsigned int> dims;
	meshH->get_dim(dims);
	dim.createdensearray(1,dims.size(),matlabarray::miDOUBLE);
	dim.setnumericarray(dims,matlabarray::miDOUBLE);
	
	// Obtain transform matrix. This is the affine transformation
	// matrix.
	SCIRun::Transform T = meshH->get_transform();
	transform.createdensearray(4,4,matlabarray::miDOUBLE);
	// Transform does not have a mechanism to access its internal fields
	// Hence it is copied to a separate data field
	std::vector<double> Tbuf(16);
	// Copy the data to the buffer. Thanks to OpenGL, it is done in the
	// same order matlab, fortran and most sane programs use, yeahhh
	// Hence we do not need to reorder any data
	T.get(&(Tbuf[0]));
	// Dump the data in the matfile generation classes
	transform.setnumericarray(Tbuf);

	mlarray.setfield(0,"dims",dims);
	mlarray.setfield(0,"transform",transform);
	
} 
 
template <class MESH> void matlabconverter::mladdnodesfield(SCIRun::LockingHandle<MESH> meshH,matlabarray mlarray)
{
	// A lot of pointless casting, but that is the way SCIRun was setup .....
	// Iterators and Index classes to make the code really complicated 
	// The next code tries to get away with minimal use of all this overhead

	matlabarray node;
	typename MESH::Node::size_type size;
	meshH->size(size);
	unsigned int numnodes = static_cast<unsigned int>(size);

	meshH->synchronize(SCIRun::Mesh::NODES_E); 

	SCIRun::Point P;
	std::vector<double> nodes(3*numnodes);
	std::vector<long> dims(2);
	dims[0] = 3; dims[1] = static_cast<long>(numnodes);

	// Extracting data from the SCIRun classes is a painfull process
	// and we end up with at least three function calls per node.
	// I'd like to change this, but hey a lot of code should be rewritten
	// This works, it might not be really efficient, at least it does not
	// hack into the object.
	unsigned int q = 0;
	for (unsigned int p=0; p<numnodes; p++)
	{
		meshH->get_point(P,typename MESH::Node::index_type(p));
		nodes[q++] = P.x(); nodes[q++] = P.y(); nodes[q++] = P.z(); 
	}
	node.createdoublematrix(nodes,dims);
	mlarray.setfield(0,"node",node);
	
}

template <class MESH>
void matlabconverter::mladdedgesfield(SCIRun::LockingHandle<MESH> meshH,matlabarray mlarray,unsigned int num)
{
	// A lot of pointless casting, but that is the way SCIRun was setup .....
	// Iterators and Index classes to make the code really complicated 
	// The next code tries to get away with minimal use of all this overhead

	matlabarray edge;
	typename MESH::Edge::size_type size;
	meshH->size(size);
	unsigned int numedges = static_cast<unsigned int>(size);

	meshH->synchronize(SCIRun::Mesh::EDGES_E); 
	
	typename MESH::Node::array_type a;
	std::vector<typename MESH::Node::index_type> edges(num*numedges);
	std::vector<long> dims(2);	
	dims[0] = static_cast<long>(num); dims[1] = static_cast<long>(numedges);
	
	// SCIRun iterators are limited in supporting any index management
	// Hence I prefer to do it with integer and convert to the required
	// class at the last moment. Hopefully the compiler is smart and
	// has a fast translation. Why do we use these iterator classes anyway
	// they just slow down our simulations......		
	unsigned int q = 0;
	for (unsigned int p = 0; p < numedges; p++)
	{
		meshH->get_nodes(a,typename MESH::Edge::index_type(p));
		for (unsigned int r = 0; r < num; r++) edges[q++] = a[r]+1;
	}

	edge.createdensearray(dims,matlabarray::miUINT32);
	edge.setnumericarray(edges); // store them as UINT32 but treat them as doubles
	mlarray.setfield(0,"edge",edge);
	
}

template <class MESH>
void matlabconverter::mladdfacesfield(SCIRun::LockingHandle<MESH> meshH,matlabarray mlarray,unsigned int num)
{
	// A lot of pointless casting, but that is the way SCIRun was setup .....
	// Iterators and Index classes to make the code really complicated 
	// The next code tries to get away with minimal use of all this overhead

	matlabarray face;
	typename MESH::Face::size_type size;
	meshH->size(size);
	unsigned int numfaces = static_cast<unsigned int>(size);

	meshH->synchronize(SCIRun::Mesh::FACES_E);

	typename MESH::Node::array_type a;
	std::vector<typename MESH::Node::index_type> faces(num*numfaces);
	std::vector<long> dims(2);	
	dims[0] = static_cast<long>(num); dims[1] = static_cast<long>(numfaces);
		
	// Another painfull and slow conversion process.....
	unsigned int q = 0;
	for (unsigned int p = 0; p < numfaces; p++)
	{
		meshH->get_nodes(a,typename MESH::Face::index_type(p));
		for (unsigned int r = 0; r < num; r++) faces[q++] = a[r]+1;
	}

	face.createdensearray(dims,matlabarray::miUINT32);
	face.setnumericarray(faces); // store them as UINT32 but treat them as doubles
	mlarray.setfield(0,"face",face);

}


template <class MESH>
void matlabconverter::mladdcellsfield(SCIRun::LockingHandle<MESH> meshH,matlabarray mlarray,unsigned int num)
{
	// A lot of pointless casting, but that is the way SCIRun was setup .....
	// Iterators and Index classes to make the code really complicated 
	// The next code tries to get away with minimal use of all this overhead

	matlabarray cell;
	typename MESH::Cell::size_type size;
	meshH->size(size);
	unsigned int numcells = static_cast<unsigned int>(size);

	meshH->synchronize(SCIRun::Mesh::CELLS_E);

	typename MESH::Node::array_type a;
	std::vector<typename MESH::Node::index_type> cells(num*numcells);
	std::vector<long> dims(2);	
	dims[0] = static_cast<long>(num); dims[1] = static_cast<long>(numcells);
		
	// ..............	
	unsigned int q = 0;
	for (unsigned int p = 0; p < numcells; p++)
	{
		meshH->get_nodes(a,typename MESH::Cell::index_type(p));
		for (unsigned int r = 0; r < num; r++) cells[q++] = a[r]+1;
	}

	cell.createdensearray(dims,matlabarray::miUINT32);
	cell.setnumericarray(cells); // store them as UINT32 but treat them as doubles
	mlarray.setfield(0,"cell",cell);
	
}

template<class T>
bool matlabconverter::mladdfield(std::vector<T> &fdata,matlabarray mlarray)
{
	matlabarray field;
	matlabarray fieldtype;

	T dummy;
	matlabarray::mitype	type = field.getmitype(dummy);
	if (type == matlabarray::miUNKNOWN) return(false);
	fieldtype.createstringarray("scalar");
	field.createdensearray(fdata.size(),1,type);
	field.setnumericarray(fdata);		
	mlarray.setfield(0,"field",field);
	mlarray.setfield(0,"fieldtype",fieldtype);

	return(true);
}



template<class T>
bool matlabconverter::mladdfield(SCIRun::FData2d<T> &fdata,matlabarray mlarray)
{
	matlabarray field;
	matlabarray fieldtype;

	T dummy;
	matlabarray::mitype	type = field.getmitype(dummy);
	if (type == matlabarray::miUNKNOWN) return(false);
	fieldtype.createstringarray("scalar");
	
	std::vector<long> dims(2);
	dims[0] = fdata.dim2(); dims[1] = fdata.dim1();
	field.createdensearray(dims,type);
	field.setnumericarray(fdata.get_dataptr(),fdata.dim2(),fdata.dim1());		
	mlarray.setfield(0,"field",field);
	mlarray.setfield(0,"fieldtype",fieldtype);

	return(true);
}

template<class T>
bool matlabconverter::mladdfield(SCIRun::FData3d<T> &fdata,matlabarray mlarray)
{
	matlabarray field;
	matlabarray fieldtype;

	T dummy;
	matlabarray::mitype	type = field.getmitype(dummy);
	if (type == matlabarray::miUNKNOWN) return(false);
	fieldtype.createstringarray("scalar");
	
	std::vector<long> dims(3);
	dims[0] = fdata.dim3(); dims[1] = fdata.dim2(); dims[2] = fdata.dim3();
	field.createdensearray(dims,type);
	field.setnumericarray(fdata.get_dataptr(),fdata.dim3(),fdata.dim2(),fdata.dim1());		
	mlarray.setfield(0,"field",field);
	mlarray.setfield(0,"fieldtype",fieldtype);
	
	return(true);
}

//////// CLASSES FOR DYNAMIC READER ////////////////




template<class MESH>
bool matlabconverter::createmesh(SCIRun::LockingHandle<MESH> &meshH,fieldstruct &fs)
{
	return(false);
}

// Templates for adding mesh components

template <class MESH>
void matlabconverter::addnodes(SCIRun::LockingHandle<MESH> meshH,matlabarray mlarray)
{
	// Get the data from the matlab file, which has been buffered
	// but whose format can be anything. The next piece of code
	// copies and casts the data
	
	std::vector<double> mldata;
	mlarray.getnumericarray(mldata);
		
	// Again the data is copied but now reorganised into
	// a vector of Point objects
	
	long numnodes = mlarray.getn();	
	std::vector<SCIRun::Point> points(numnodes);

	meshH->node_reserve(numnodes);
	
	long p,q;
	for (p = 0, q = 0; p < numnodes; p++, q+=3)
	{ meshH->add_point(SCIRun::Point(mldata[q],mldata[q+1],mldata[q+2])); } 
}

template <class MESH>
void matlabconverter::addedges(SCIRun::LockingHandle<MESH> meshH,matlabarray mlarray)
{
	// Get the data from the matlab file, which has been buffered
	// but whose format can be anything. The next piece of code
	// copies and casts the data
	
	std::vector<unsigned int> mldata;
	mlarray.getnumericarray(mldata);		
	
	// check whether it is zero based indexing 
	// In short if there is a zero it must be zero
	// based numbering right ??
	// If not we assume one based numbering
	
	long p,q;
	
	bool zerobased = false;  
	long size = static_cast<long>(mldata.size());
	for (p = 0; p < size; p++) { if (mldata[p] == 0) {zerobased = true; break;} }
	
	if (zerobased == false)
	{   // renumber to go from matlab indexing to C++ indexing
		for (p = 0; p < size; p++) { mldata[p]--;}
	}
	
	meshH->elem_reserve(mlarray.getn());
	
	for (p = 0, q = 0; p < mlarray.getn(); p++, q += 2)
	{
		meshH->add_edge(typename MESH::Node::index_type(mldata[q]), typename MESH::Node::index_type(mldata[q+1]));
	}
		  
 }

 template <class MESH>
 void matlabconverter::addfaces(SCIRun::LockingHandle<MESH> meshH,matlabarray mlarray)
 {
   // Get the data from the matlab file, which has been buffered
   // but whose format can be anything. The next piece of code
   // copies and casts the data
	
   std::vector<unsigned int> mldata;
   mlarray.getnumericarray(mldata);		
	
   // check whether it is zero based indexing 
   // In short if there is a zero it must be zero
   // based numbering right ??
   // If not we assume one based numbering
	
   bool zerobased = false;  
   long size = static_cast<long>(mldata.size());
   for (long p = 0; p < size; p++) { if (mldata[p] == 0) {zerobased = true; break;} }
	
   if (zerobased == false)
   {   // renumber to go from matlab indexing to C++ indexing
     for (long p = 0; p < size; p++) { mldata[p]--;}
   }
	
   long m,n;
   m = mlarray.getm();
   n = mlarray.getn();
	
   meshH->elem_reserve(n);	  
			  	  
   typename MESH::Node::array_type face(m);  
	
   long r;
   r = 0;
	
   for (long p = 0; p < n; p++)
   {
     for (long q = 0 ; q < m; q++)
     {
       face[q] = mldata[r]; r++; 
     }
     meshH->add_elem(face);
   }
 }

 template <class MESH>
 void matlabconverter::addcells(SCIRun::LockingHandle<MESH> meshH,matlabarray mlarray)
 {
   // Get the data from the matlab file, which has been buffered
   // but whose format can be anything. The next piece of code
   // copies and casts the data
	
   std::vector<unsigned int> mldata;
   mlarray.getnumericarray(mldata);		
	
   // check whether it is zero based indexing 
   // In short if there is a zero it must be zero
   // based numbering right ??
   // If not we assume one based numbering
	
   bool zerobased = false;  
   long size = static_cast<long>(mldata.size());
   for (long p = 0; p < size; p++) { if (mldata[p] == 0) {zerobased = true; break;} }
	
   if (zerobased == false)
   {   // renumber to go from matlab indexing to C++ indexing
     for (long p = 0; p < size; p++) { mldata[p]--;}
   }
	  
   long m,n;
   m = mlarray.getm();
   n = mlarray.getn();
	
   meshH->elem_reserve(n);	  
			  	  
   typename MESH::Node::array_type cell(m);  
	
   long r;
   r = 0;
	
   for (long p = 0; p < n; p++)
   {
     for (long q = 0 ; q < m; q++)
     {
       cell[q] = mldata[r]; r++; 
     }
     meshH->add_elem(cell);
   }
 }


 template<class T>
 bool matlabconverter::addfield(T &fdata,matlabarray mlarray)
 {
   return(false);
 }


 template <class T> 
 bool matlabconverter::addfield(std::vector<T> &fdata,matlabarray mlarray)
 {
   mlarray.getnumericarray(fdata);
   return(true);
 }


 template <class T> 
 bool matlabconverter::addfield(SCIRun::FData2d<T> &fdata,matlabarray mlarray)
 {
   mlarray.getnumericarray(fdata.get_dataptr(),fdata.dim2(),fdata.dim1());
   return(true);
 }


 template <class T> 
 bool matlabconverter::addfield(SCIRun::FData3d<T> &fdata,matlabarray mlarray)
 {
   mlarray.getnumericarray(fdata.get_dataptr(),fdata.dim3(),fdata.dim2(),fdata.dim1());
   return(true);
 }


 inline void matlabconverter::compressedtensor(std::vector<double> &fielddata,SCIRun::Tensor &tens, unsigned int p)
 {
   tens.mat_[0][0] = fielddata[p];
   tens.mat_[0][1] = fielddata[p+3];
   tens.mat_[0][2] = fielddata[p+4];
   tens.mat_[1][1] = fielddata[p+1];
   tens.mat_[1][2] = fielddata[p+5];
   tens.mat_[2][2] = fielddata[p+2];
 }

 inline void matlabconverter::uncompressedtensor(std::vector<double> &fielddata,SCIRun::Tensor &tens, unsigned int p)
 {
   tens.mat_[0][0] = fielddata[p];
   tens.mat_[0][1] = fielddata[p+1];
   tens.mat_[0][2] = fielddata[p+2];
   tens.mat_[1][0] = fielddata[p+3];
   tens.mat_[1][1] = fielddata[p+4];
   tens.mat_[1][2] = fielddata[p+5];
   tens.mat_[2][0] = fielddata[p+6];
   tens.mat_[2][1] = fielddata[p+7];
   tens.mat_[2][2] = fielddata[p+8];
 }



 // Dynamic compilation classes
 // vvvvvvvvvvvvvvvvvvvvvvvvvvv 
 
 class MatlabFieldReaderAlgo : public SCIRun::DynamicAlgoBase
 {
 public:
   // Place holder for the dynamically compiled code 
   virtual bool execute(SCIRun::FieldHandle &scifield, matlabconverter::fieldstruct &fs, matlabconverter &translate) =0;
	
   // support the dynamically compiled algorithm concept
   static SCIRun::CompileInfoHandle get_compile_info(const SCIRun::TypeDescription *fieldTD);
	
   // The maker of this class will be used to create the templated class
 };
 
 template <class FIELD> class MatlabFieldReaderAlgoT : public MatlabFieldReaderAlgo
 {
 public:
   // The actual dynamic code definitiongoes in here -> 
   virtual bool execute(SCIRun::FieldHandle &scifield,  matlabconverter::fieldstruct &fs, matlabconverter &translate);
 };


 // Dynamically defined function starts here -> 
 template <class FIELD>  bool MatlabFieldReaderAlgoT<FIELD>::execute(SCIRun::FieldHandle &scifield, matlabconverter::fieldstruct &fs, matlabconverter &translate)
 {
   typename FIELD::mesh_handle_type meshH;
	
   if (!(translate.createmesh(meshH,fs)))
   {
     // Somehow my dymanic compilation for a meshgenerator failed
     return(false);
   }
	
   // Here we finally create the field
   FIELD *fieldptr = scinew FIELD(meshH,fs.basis_order);
   scifield = static_cast<SCIRun::Field *>(fieldptr);
	
   fieldptr->resize_fdata();   // make sure it is resized to number of nodes/edges/faces/cells or whatever
   if (fs.field.isempty()) return(true);	// NEED TO CHECK THIS NOW AS data_at changed into basis_order
   typename FIELD::fdata_type& fdata = fieldptr->fdata();
   if (!(translate.addfield(fdata,fs.field)))
   {
     return(false);
   }
   // everything went ok, so report this to the function doing the actual implementation
   return(true);
 } 
 
 
 
 
 class MatlabFieldWriterAlgo : public SCIRun::DynamicAlgoBase
 {
 public:
   // Place holder for the dynamically compiled code 
   virtual bool execute(SCIRun::FieldHandle fieldH, matlabarray &mlarray, matlabconverter &translate) =0;
	
   // support the dynamically compiled algorithm concept
   static SCIRun::CompileInfoHandle get_compile_info(const SCIRun::TypeDescription *fieldTD);
	
   // The maker of this class will be used to create the templated class
 };

 template <class FIELD> class MatlabFieldWriterAlgoT : public MatlabFieldWriterAlgo
 {
 public:
   // The actual dynamic code definitiongoes in here -> 
   virtual bool execute(SCIRun::LockingHandle<SCIRun::Field> fieldH, matlabarray &mlarray, matlabconverter &translate);
 };

 // Dynamically defined function starts here -> 
 template <class FIELD>  bool MatlabFieldWriterAlgoT<FIELD>::execute(SCIRun::FieldHandle scifield, matlabarray &mlarray, matlabconverter &translate)
 {
	
   // input is a general FieldHandle, cast this to the specific one
   FIELD *field = dynamic_cast<FIELD *>(scifield.get_rep());
	
   // get the specific mesh class as well
   typename FIELD::mesh_handle_type meshH;
	
   // Start translation with adding where the field is located
   // Thhe next function is templated and will be inserted for every dynamic version of the code
   translate.mladdfieldat(field,mlarray);

   // Get the meshHandle, mesh() will only give a non-specific MeshHandle and canot be used
   meshH = field->get_typed_mesh();
	
   // Dynamically translate the mesh class
   // mladdmesh is both a templated class as well a collection of specifically written functions
   // for a certain mesh class. The precompiled version that overload the dynamix ones are already
   // in the matlabconverter class. The latter were needed as SCIRun's mesh classes are not as polymorphic
   // as they should be. Hence to deal with local details often a speciliased function is better
   if (translate.mladdmesh(meshH,mlarray) == false)
   {   // could not translate mesh
     // Currently the templated function return a false and tells this function that not a proper
     // converter could be found.
     return(false);
   }

   // Dynamically do the field contents. Luckily this is better templated and hence a couple
   // of templated functions do the tricks

   if (field->basis_order() > -1)
   {
     typename FIELD::fdata_type fdata = field->fdata();
     if (translate.mladdfield(fdata,mlarray) == false)
     {
       // Could not translate field but continuing anyway
       return(false);
     }
   }
   // everything went ok, so report this to the function doing the actual implementation
   return(true);
 } 
 
 
 } // end namespace

#endif
