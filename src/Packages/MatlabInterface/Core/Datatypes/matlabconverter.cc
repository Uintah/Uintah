
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
 * FILE: matlabconverter.cc
 * AUTH: Jeroen G Stinstra
 * DATE: 18 MAR 2004
 */

#include "matlabconverter.h"


namespace MatlabIO {

// Currently the property converter only manages strings
// all other data is ignored both on matlabside as well
// as on the property manager side.

/* DESIGN NOTES : */

/* 
 * The BIG Problem in SCIRun is that it is hard to know what
 * kind of object a SCIRun property is and how it should be 
 * translated. For example there are countless array classes and
 * no attempt has been made to standardize the data formats
 * which leads to a huge conversion problem. Hence only the useful
 * objects are translated the rest is just discarded. Until there
 * is a better data management structure within SCIRun, the conversion
 * process is basically a big switch statemet scanning for each possible
 * object structure
 */

/* 
 * With the hope in mind that a better data management system will be
 * in place in the future, all conversion algorithms are grouped in
 * this one object. All the other matlab classes function independent
 * of the main SCIRun structure, it has its own memory and data management
 * making it easier to adapt to future changes. The separation of matlab
 * and SCIRun data management has the advantage that the matlab side can be
 * further enhanced to deal with different future matlab fileformats, without
 * having to comb through the conversion modules. Though there is a little 
 * memory overhead. Especially with the V7 compressed files, more memory
 * is needed to maintain the integrity of the matlab reader. Some changes 
 * in this converter may be needed to enhance performance. Currently the
 * a compressed file will be decompressed and scanned for suitable objects.
 * Upon loading the matrix, the matrix will be decompressed again as after
 * scanning the file, nothing will remain in memory
 */


// Manage converter options

// Set defaults in the constructor
matlabconverter::matlabconverter()
: numericarray_(false), indexbase_(1), datatype_(matlabarray::miSAMEASDATA), disable_transpose_(false)
{
}

void matlabconverter::setdatatype(matlabarray::mitype dataformat)
{
	datatype_ = dataformat;
}

void matlabconverter::setindexbase(long indexbase)
{
	indexbase_ = indexbase;
}

void matlabconverter::converttonumericmatrix()
{
	numericarray_ = true;
}

void matlabconverter::converttostructmatrix()
{
	numericarray_ = false;
}

void matlabconverter::setdisabletranspose(bool dt)
{
	disable_transpose_ = dt;
}



void matlabconverter::mlPropertyTOsciProperty(matlabarray &ma,SCIRun::PropertyManager *handle)
{
	long numfields;
	matlabarray::mlclass mclass;
	matlabarray subarray;
	std::string propname;
	std::string propval;
	matlabarray proparray;

	// properties are stored in field property
	long propindex = ma.getfieldnameindexCI("property");
	
	if (propindex > -1)
	{ // field property exists
	
		proparray = ma.getfield(0,propindex);
		if (proparray.isempty()) return;
		
		numfields = proparray.getnumfields();
	
		for (long p=0; p<numfields; p++)
		{
			subarray = proparray.getfield(0,p);
			mclass = subarray.getclass();
			if (mclass == matlabarray::mlSTRING)
			{   // only string arrays are converted
				propname = proparray.getfieldname(p);
				propval = subarray.getstring();
				handle->set_property(propname,propval,false);
			}
		}
	}
}

void matlabconverter::sciPropertyTOmlProperty(SCIRun::PropertyManager *handle,matlabarray &ma)
{
	long numfields;
	matlabarray proparray;
	std::string propname;
	std::string propvalue;
	matlabarray subarray;
	
	proparray.createstructarray();
	numfields = handle->nproperties();
	
	for (long p=0;p<numfields;p++)
	{
		propname = handle->get_property_name(p);
		if (handle->get_property(propname,propvalue))
		{
			subarray.createstringarray(propvalue);
			proparray.setfield(0,propname,subarray);
		}
	}
	
	ma.setfield(0,"property",proparray);
}

// The next function checks whether
// the program knows how to convert 
// the matlabarray into a scirun matrix

long matlabconverter::sciMatrixCompatible(matlabarray &ma, std::string &infotext)
{
	infotext = "";

	matlabarray::mlclass mclass;
	mclass = ma.getclass();
	
	switch (mclass)
	{
		case matlabarray::mlDENSE:
		case matlabarray::mlSPARSE:
		{
			// check whether the data is of a proper format
	
			std::vector<long> dims;	
			dims = ma.getdims();
			if (dims.size() > 2)
			{   
				return(0); // no multidimensional arrays supported yet in the SCIRun Matrix classes
			}
	
			matlabarray::mitype type;
			type = ma.gettype();
	
			infotext = ma.getinfotext(); 
			
			// We expect a double array.		
			// This classification is pure for convenience
			// Any integer array can be dealt with as well
	
			// doubles are most likely to be the data wanted by the users
			if (type == matlabarray::miDOUBLE) return(3);
			// though singles should work as well
			if (type == matlabarray::miSINGLE) return(2);
			// all other numeric formats should be integer types, which
			// can be converted using a simple cast
			return(1);			
		}		
		break;
		case matlabarray::mlSTRUCT:
		{
			long index;
			/* A lot of different names can be used for the data:
			   This has mainly historical reasons: a lot of different
			   names have been used at CVRTI to store data, to be 
			   compatible with all, we allow all of them. Though we
			   suggest the use of : "data" 
			*/
			index = ma.getfieldnameindexCI("data");
			if (index == -1) index = ma.getfieldnameindexCI("potvals");	// in case it is a saved TSDF file
			if (index == -1) index = ma.getfieldnameindexCI("field");
			if (index == -1) index = ma.getfieldnameindexCI("scalarfield");
			if (index == -1) index = ma.getfieldnameindexCI("vectorfield");
			if (index == -1) index = ma.getfieldnameindexCI("tensorfield");
			if (index == -1) return(0); // incompatible
		
			long numel;
			numel = ma.getnumelements();
			if (numel > 1) return(0); // incompatible	
					
			matlabarray subarray;
			subarray = ma.getfield(0,index);
			
			// check whether the data is of a proper format
	
			if (subarray.isempty()) return(0); // not compatible
	
			std::vector<long> dims;	
			dims = subarray.getdims();
			if (dims.size() > 2)
			{   
				return(0); // no multidimensional arrays supported yet in the SCIRun Matrix classes
			}
	
			matlabarray::mitype type;
			type = subarray.gettype();
	
			infotext = subarray.getinfotext(ma.getname());
			// We expect a double array.		
			// This classification is pure for convenience
			// Any integer array can be dealt with as well
	
			// doubles are most likely to be the data wanted by the users
			if (type == matlabarray::miDOUBLE) return(3);
			// though singles should work as well
			if (type == matlabarray::miSINGLE) return(2);
			// all other numeric formats should be integer types, which
			// can be converted using a simple cast
			return(1);			
		} 
		break;
	}
	return (0);
}
		



void matlabconverter::mlArrayTOsciMatrix(matlabarray &ma,SCIRun::MatrixHandle &handle)
{
	matlabarray::mlclass mclass = ma.getclass();
	
	switch(mclass)
	{
		case matlabarray::mlDENSE:
			{   // new environment so I can create new variables
			
				if (disable_transpose_)
				{
					SCIRun::DenseMatrix* dmptr;							// pointer to a new dense matrix
						
					int m = static_cast<int>(ma.getm());
					int n = static_cast<int>(ma.getn());
					
					dmptr = new SCIRun::DenseMatrix(n,m);   // create dense matrix
						// copy and cast elements:
						// getnumericarray is a templated function that casts the data to the supplied pointer
						// type. It needs the dimensions of the memory block (in elements) to make sure
						// everything is still OK. 
					ma.getnumericarray(dmptr->getData(),(dmptr->nrows())*(dmptr->ncols()));  
					
					handle = static_cast<SCIRun::Matrix *>(dmptr); // cast it to a general matrix pointer
				}
				else
				{
				
					SCIRun::DenseMatrix* dmptr;							// pointer to a new dense matrix
					
					int m = static_cast<int>(ma.getm());
					int n = static_cast<int>(ma.getn());
					
					SCIRun::DenseMatrix  dm(n,m);   // create dense matrix
						// copy and cast elements:
						// getnumericarray is a templated function that casts the data to the supplied pointer
						// type. It needs the dimensions of the memory block (in elements) to make sure
						// everything is still OK. 
					ma.getnumericarray(dm.getData(),(dm.nrows())*(dm.ncols()));  
					
					// There is no transpose function to operate on the same memory block
					// Hence, it is a little memory inefficient.
					
					dmptr = dm.transpose();	// SCIRun has a C++-style matrix and matlab a FORTRAN-style matrix
					handle = static_cast<SCIRun::Matrix *>(dmptr); // cast it to a general matrix pointer
				}
			}
			break;
			
		case matlabarray::mlSPARSE:
			{
				if (disable_transpose_)
				{
					SCIRun::SparseRowMatrix* smptr;
					
					// Since the SparseRowMatrix does not allocate memory but on the 
					// otherhand frees it in the destructor. The memory needs to be
					// allocated outside of the object and then linked to the object
					// to have it freed lateron by the object.
					
					// in the matlabio classes they are defined as long, hence
					// the casting operators
					int nnz = static_cast<int>(ma.getnnz());
					int m = static_cast<int>(ma.getm());
					int n = static_cast<int>(ma.getn()); 
					
					double *values = scinew double[nnz];
					int *rows   = scinew int[nnz];
					int *cols   = scinew int[n+1];
					
					ma.getnumericarray(values,nnz);
					ma.getrowsarray(rows,nnz); // automatically casts longs to ints
					ma.getcolsarray(cols,(n+1));
					
					smptr = new SCIRun::SparseRowMatrix(n,m,cols,rows,nnz,values);
					
					handle = static_cast<SCIRun::Matrix *>(smptr); // cast it to a general matrix pointer
				}
				else
				{
					SCIRun::SparseRowMatrix* smptr;
					
					// Since the SparseRowMatrix does not allocate memory but on the 
					// otherhand frees it in the destructor. The memory needs to be
					// allocated outside of the object and then linked to the object
					// to have it freed lateron by the object.
					
					// in the matlabio classes they are defined as long, hence
					// the casting operators
					int nnz = static_cast<int>(ma.getnnz());
					int m = static_cast<int>(ma.getm());
					int n = static_cast<int>(ma.getn()); 
					
					double *values = scinew double[nnz];
					int *rows   = scinew int[nnz];
					int *cols   = scinew int[n+1];
					
					ma.getnumericarray(values,nnz);
					ma.getrowsarray(rows,nnz); // automatically casts longs to ints
					ma.getcolsarray(cols,(n+1));
					
					SCIRun::SparseRowMatrix  sm(n,m,cols,rows,nnz,values);
					
					smptr = sm.transpose(); // SCIRun uses Row sparse matrices and matlab Column sparse matrices
					handle = static_cast<SCIRun::Matrix *>(smptr); // cast it to a general matrix pointer
				}
			}
			break;
			
		case matlabarray::mlSTRUCT:
			{
				// A compatible struct has the following fields
				// - data:     submatrix with the actual matrix in it
				// - property: optional extra struct array with key
				//             /value pairs for the property manager
				
				long dataindex, propertyindex;
				dataindex = ma.getfieldnameindexCI("data");
				if (dataindex == -1) dataindex = ma.getfieldnameindex("potvals");
				if (dataindex == -1) dataindex = ma.getfieldnameindexCI("field");
				if (dataindex == -1) dataindex = ma.getfieldnameindexCI("scalarfield");
				if (dataindex == -1) dataindex = ma.getfieldnameindexCI("vectorfield");
				if (dataindex == -1) dataindex = ma.getfieldnameindexCI("tensorfield");

				propertyindex = ma.getfieldnameindexCI("property");
				
				if (dataindex == -1)
				{
					throw matlabconverter_error();
				}
				
				matlabarray subarray;
				subarray = ma.getfield(0,dataindex);
				mlArrayTOsciMatrix(subarray,handle);
				
				if (propertyindex != -1)
				{
					mlPropertyTOsciProperty(ma,static_cast<SCIRun::PropertyManager *>(handle.get_rep()));
				}
				
			}
			break;
		default:
			{   // The program should not get here
				throw matlabconverter_error();
			}
	}
}


void matlabconverter::sciMatrixTOmlMatrix(SCIRun::MatrixHandle &scimat,matlabarray &mlmat)
{
	// Get the format for exporting data
	matlabarray::mitype dataformat = datatype_;

	// SCIRun matrices are always (up till now) doubles
	if (dataformat == matlabarray::miSAMEASDATA) dataformat = matlabarray::miDOUBLE;
	
	if (scimat->is_dense())
	{
		SCIRun::DenseMatrix* dmatrix;
		SCIRun::DenseMatrix* tmatrix;
		dmatrix = scimat->as_dense();
		tmatrix = dmatrix->transpose();
		
		std::vector<long> dims(2);
		dims[1] = tmatrix->nrows();
		dims[0] = tmatrix->ncols();
		mlmat.createdensearray(dims,dataformat);
		mlmat.setnumericarray(tmatrix->getData(),mlmat.getnumelements());
	}
	if (scimat->is_column())
	{
		SCIRun::ColumnMatrix* cmatrix;
		std::vector<long> dims(2);
		cmatrix = scimat->as_column();
		dims[0] = cmatrix->nrows();
		dims[1] = cmatrix->ncols();
		mlmat.createdensearray(dims,dataformat);
		mlmat.setnumericarray(cmatrix->get_data(),mlmat.getnumelements());
	}
	if (scimat->is_sparse())
	{
		SCIRun::SparseRowMatrix* smatrix;
		SCIRun::SparseRowMatrix* tmatrix;
		smatrix = scimat->as_sparse();
		tmatrix = smatrix->transpose();
		
		std::vector<long> dims(2);
		dims[1] = tmatrix->nrows();
		dims[0] = tmatrix->ncols();
		mlmat.createsparsearray(dims,dataformat);
		
		mlmat.setnumericarray(tmatrix->get_val(),tmatrix->get_nnz());
		mlmat.setrowsarray(tmatrix->get_col(),tmatrix->get_nnz());
		mlmat.setcolsarray(tmatrix->get_row(),tmatrix->nrows()+1);
	}
}


void matlabconverter::sciMatrixTOmlArray(SCIRun::MatrixHandle &scimat,matlabarray &mlmat)
{
	if (numericarray_ == true)
	{
		sciMatrixTOmlMatrix(scimat,mlmat);
	}
	else
	{
		matlabarray dataarray;
		mlmat.createstructarray();
		sciMatrixTOmlMatrix(scimat,dataarray);
		mlmat.setfield(0,"data",dataarray);
		sciPropertyTOmlProperty(static_cast<SCIRun::PropertyManager *>(scimat.get_rep()),mlmat);
	}
}

// Support function to check whether the names supplied are within the 
// rules that matlab allows. Otherwise we could save the file, but matlab 
// would complain it could not read the file.

bool matlabconverter::isvalidmatrixname(std::string name)
{
	const std::string validchar("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_");
	const std::string validstartchar("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ");


	bool valid = true;
	bool foundchar = false;

	for (long p=0; p < name.size(); p++)
	{
		if (p == 0)
		{   
			// A variable name is not allowed to start with a number
			foundchar = false;
			for (long q = 0; q < validstartchar.size(); q++) 
			{
				if (name[p] == validstartchar[q]) { foundchar = true; break; }
			}
		}
		else
		{
			foundchar = false;
			for (long q = 0; q < validchar.size(); q++) 
			{
				if (name[p] == validchar[q]) { foundchar = true; break; }
			}
		}
		if (foundchar == false) { valid = false; break; }
	}
	return(valid);
}


#ifdef HAVE_TEEM_PACKAGE

// Test the compatibility of the matlabarray witha nrrd structure
// in case it is compatible return a positive value and write
// out an infostring with a summary of the contents of the matrix

long matlabconverter::sciNrrdDataCompatible(matlabarray &mlarray, std::string &infostring)
{
	matlabarray::mlclass mclass;
	mclass = mlarray.getclass();
	
	// parse matrices are dealt with in a separate 
	// module as the the data needs to be divided over
	// three separate Nrrds

	infostring = "";
	
	if ((mclass == matlabarray::mlSTRUCT)||(mclass == matlabarray::mlOBJECT))
	{
		long fieldnameindex;
		matlabarray subarray;
		
		fieldnameindex = mlarray.getfieldnameindexCI("data");
		if (fieldnameindex == -1) fieldnameindex = mlarray.getfieldnameindexCI("potvals");
		if (fieldnameindex == -1) fieldnameindex = mlarray.getfieldnameindexCI("field");
		if (fieldnameindex == -1) fieldnameindex = mlarray.getfieldnameindexCI("scalarfield");
		if (fieldnameindex == -1) fieldnameindex = mlarray.getfieldnameindexCI("vectorfield");
		if (fieldnameindex == -1) fieldnameindex = mlarray.getfieldnameindexCI("tensorfield");

		if (fieldnameindex == -1) return(0);
		
		subarray = mlarray.getfield(0,fieldnameindex);
	
		if (subarray.isempty()) return(0);
				
		infostring = subarray.getinfotext(mlarray.getname());
		matlabarray::mitype type;
		type = subarray.gettype();
	
		matlabarray::mlclass mclass;
		mclass = subarray.getclass();
		
		if ((mclass != matlabarray::mlDENSE)&&(mclass != matlabarray::mlSPARSE)) return(0);
	
		// We expect a double array.
		// This classification is pure for convenience
		// Any integer array can be dealt with as well
	
		// doubles are most likely to be the data wanted by the users
		if (type == matlabarray::miDOUBLE) return(3);
		// though singles should work as well
		if (type == matlabarray::miSINGLE) return(2);
		// all other numeric formats should be integer types, which
		// can be converted using a simple cast
		return(1);	
	}
	
	if ((mclass != matlabarray::mlDENSE))
	{
		return(0); // incompatible for the moment, no converter written for this type yet
	}

	// Need to enhance this code to squeeze out dimensions of size one

	// Nrrds can be multi dimensional hence no limit on the dimensions is
	// needed
	
	if (mlarray.isempty()) return(0);
	
	matlabarray::mitype type;
	type = mlarray.gettype();
	
	infostring = mlarray.getinfotext();
	
	// We expect a double array.
	// This classification is pure for convenience
	// Any integer array can be dealt with as well
	
	// doubles are most likely to be the data wanted by the users
	if (type == matlabarray::miDOUBLE) return(3);
	// though singles should work as well
	if (type == matlabarray::miSINGLE) return(2);
	// all other numeric formats should be integer types, which
	// can be converted using a simple cast
	return(1);	
}


unsigned int matlabconverter::convertmitype(matlabarray::mitype type)
{
	switch (type)
	{
		case matlabarray::miINT8:   return(nrrdTypeChar);
		case matlabarray::miUINT8:  return(nrrdTypeUChar);
		case matlabarray::miINT16:  return(nrrdTypeShort);
		case matlabarray::miUINT16: return(nrrdTypeUShort);
		case matlabarray::miINT32:  return(nrrdTypeInt);
		case matlabarray::miUINT32: return(nrrdTypeUInt);
		case matlabarray::miINT64:  return(nrrdTypeLLong);
		case matlabarray::miUINT64: return(nrrdTypeULLong);
		case matlabarray::miSINGLE: return(nrrdTypeFloat);
		case matlabarray::miDOUBLE: return(nrrdTypeDouble);
		default: return(nrrdTypeUnknown);
	}
}


// This function converts nrrds into matlab matrices
// only the datais being transformed into a matlab array

matlabarray::mitype matlabconverter::convertnrrdtype(int type)
{
	switch (type)
	{
		case nrrdTypeChar : return(matlabarray::miINT8);
		case nrrdTypeUChar : return(matlabarray::miUINT8);
		case nrrdTypeShort : return(matlabarray::miINT16);
		case nrrdTypeUShort : return(matlabarray::miUINT16);		
		case nrrdTypeInt : return(matlabarray::miINT32);
		case nrrdTypeUInt : return(matlabarray::miUINT32);
		case nrrdTypeLLong : return(matlabarray::miINT64);
		case nrrdTypeULLong : return(matlabarray::miUINT64);
		case nrrdTypeFloat : return(matlabarray::miSINGLE);
		case nrrdTypeDouble : return(matlabarray::miDOUBLE);
		default: return(matlabarray::miUNKNOWN);
	}
}

void matlabconverter::mlArrayTOsciNrrdData(matlabarray &mlarray,SCITeem::NrrdDataHandle &scinrrd)
{
	// Depending on the matlabclass there are several converters
	// for converting the data from matlab into a SCIRun Nrrd object
	
	matlabarray::mlclass mclass;
	mclass = mlarray.getclass();
	
	// In case no converter is found return 0 
	// Hence initialise scinrrd as a NULL ptr
	
	scinrrd = 0; 
	
	// Pointer to a new SCIRun Nrrd Data object
	SCITeem::NrrdData* nrrddataptr = 0;
					
	switch(mclass)
	{
		case matlabarray::mlDENSE:
			{   // new environment so I can create new variables

				try
				{
					// new nrrd data handle
					nrrddataptr = new SCITeem::NrrdData(true); // nrrd is owned by the object
					nrrddataptr->nrrd = nrrdNew();
				
					// obtain the type of the new nrrd
					// we want to keep the nrrd type the same
					// as the original matlab type
					
					unsigned int nrrdtype = convertmitype(mlarray.gettype());
				
					// obtain the dimensions of the new nrrd
					int nrrddims[NRRD_DIM_MAX];
					std::vector<long> dims = mlarray.getdims();
					long nrrddim = dims.size();
					for (long p=0;p<nrrddim;p++) nrrddims[p] = dims[p];
				
					nrrdAlloc_nva(nrrddataptr->nrrd,nrrdtype,nrrddim,nrrddims);
					
					switch (nrrdtype)
					{
						case nrrdTypeChar:
							mlarray.getnumericarray(static_cast<signed char *>(nrrddataptr->nrrd->data),nrrdElementNumber(nrrddataptr->nrrd));
							break;
						case nrrdTypeUChar:
							mlarray.getnumericarray(static_cast<unsigned char *>(nrrddataptr->nrrd->data),nrrdElementNumber(nrrddataptr->nrrd));
							break;
						case nrrdTypeShort:
							mlarray.getnumericarray(static_cast<signed short *>(nrrddataptr->nrrd->data),nrrdElementNumber(nrrddataptr->nrrd));
							break;
						case nrrdTypeUShort:
							mlarray.getnumericarray(static_cast<unsigned short *>(nrrddataptr->nrrd->data),nrrdElementNumber(nrrddataptr->nrrd));
							break;
						case nrrdTypeInt:
							mlarray.getnumericarray(static_cast<signed long *>(nrrddataptr->nrrd->data),nrrdElementNumber(nrrddataptr->nrrd));
							break;
						case nrrdTypeUInt:
							mlarray.getnumericarray(static_cast<unsigned long *>(nrrddataptr->nrrd->data),nrrdElementNumber(nrrddataptr->nrrd));
							break;
#ifdef JGS_MATLABIO_USE_64INTS
						case nrrdTypeLLong:
							mlarray.getnumericarray(static_cast<int64 *>(nrrddataptr->nrrd->data),nrrdElementNumber(nrrddataptr->nrrd));
							break;
						case nrrdTypeULLong:
							mlarray.getnumericarray(static_cast<uint64 *>(nrrddataptr->nrrd->data),nrrdElementNumber(nrrddataptr->nrrd));
							break;
#endif
						case nrrdTypeFloat:
							mlarray.getnumericarray(static_cast<float *>(nrrddataptr->nrrd->data),nrrdElementNumber(nrrddataptr->nrrd));
							break;
						case nrrdTypeDouble:
							mlarray.getnumericarray(static_cast<double *>(nrrddataptr->nrrd->data),nrrdElementNumber(nrrddataptr->nrrd));
							break;
						default:
							throw matlabconverter_error();
							break;
					}
	
					// set some info on the axis as not all SCIRun modules check whether there is any
					// data and may crash if there is no label
					
					// Nrrd lib is C and needs a list with pointers to C-style strings
					// The following C++ code does this without the need of the need for
					// explicit dynamic memory allocation.
					
					std::vector<std::string> labels;
					labels.resize(nrrddim);
					const char *labelptr[NRRD_DIM_MAX];
					
					for (long p=0;p<nrrddim;p++)
					{
						std::ostringstream oss; 
						oss << "dimension " << (p+1);
						labels[p] = oss.str();
						labelptr[p] = labels[p].c_str();
						// labelptr contains ptrs into labels and
						// the memory it points to will be destroyed with
						// the labels object. The const cast is needed as
						// nrrd
					}
									
					nrrdAxisInfoSet_nva(nrrddataptr->nrrd,nrrdAxisInfoLabel,labelptr);
																													
					scinrrd = nrrddataptr;
				}
				catch (...)
				{
					// in case something went wrong
					// release the datablock attached to
					// the nrrdhandle
					
					delete nrrddataptr;
					scinrrd = 0;
					
					throw;
				}
			}
			
			if (scinrrd != 0)
			{
				std::string str = mlarray.getname();
				scinrrd->set_filename(str);
			}
			
			break;
			// END CONVERSION OF MATLAB MATRIX
			
		case matlabarray::mlSTRUCT:
		case matlabarray::mlOBJECT:
			{
				long dataindex;
				dataindex = mlarray.getfieldnameindexCI("data");
				if (dataindex == -1) dataindex = mlarray.getfieldnameindexCI("potvals");
				if (dataindex == -1) dataindex = mlarray.getfieldnameindexCI("field");
				if (dataindex == -1) dataindex = mlarray.getfieldnameindexCI("scalarfield");
				if (dataindex == -1) dataindex = mlarray.getfieldnameindexCI("vectorfield");
				if (dataindex == -1) dataindex = mlarray.getfieldnameindexCI("tensorfield");

			
				// We need data to create an object
				// if no data field is found return
				// an error
				
				if (dataindex == -1)
				{
					throw matlabconverter_error();
				}
			
				matlabarray subarray;
				subarray = mlarray.getfieldCI(0,"data");
				
				matlabarray::mlclass subclass;
				subclass = subarray.getclass();
				
				if (subclass != matlabarray::mlDENSE)
				{
					throw matlabconverter_error();
					return;
				}
				
				mlArrayTOsciNrrdData(subarray,scinrrd);
				
				if (scinrrd == 0)
				{
					throw matlabconverter_error();
					return;
				}
				
				// Add axes properties if they are specified
				
				long axisindex;
				axisindex = mlarray.getfieldnameindexCI("axis");
				
				if (axisindex != -1)
				{
					matlabarray::mlclass axisarrayclass;
					matlabarray axisarray;
					long		numaxis;
					long		fnindex;
					matlabarray farray;
					
					axisarray = mlarray.getfieldCI(0,"axis");
					
					if (!axisarray.isempty())
					{
						numaxis = axisarray.getm();
						axisarrayclass =axisarray.getclass();
					
						if ((axisarrayclass != matlabarray::mlSTRUCT)&&(axisarrayclass != matlabarray::mlOBJECT))
						{
							throw matlabconverter_error();
							return;
						}
				
				
						// insert labels into nnrd
						// labels can be defined in axis(n).label
				
						fnindex = axisarray.getfieldnameindexCI("label");
						
						if (fnindex != -1)
						{
							std::vector<std::string> labels(NRRD_DIM_MAX);
							const char *clabels[NRRD_DIM_MAX];
					
							// Set some default values in case the
							// label is not defined by the matlabarray
						
							for (long p=0;p<NRRD_DIM_MAX;p++)
							{
								std::ostringstream oss; 
								oss << "dimension " << (p+1);
								labels[p] = oss.str();
								clabels[p] = labels[p].c_str();
							}
						
							// In case the label is set in the matlabarray
							// add this label to the clabels array
						
							for (long p=0;p<numaxis;p++)
							{
								farray = axisarray.getfield(p,fnindex);
								if (farray.getclass() == matlabarray::mlSTRING)
								{
									labels[p] = farray.getstring();
									clabels[p] = labels[p].c_str();
								} 
							}
						
							nrrdAxisInfoSet_nva(scinrrd->nrrd,nrrdAxisInfoLabel,clabels);
						}

						
						// insert unit names
						// into nrrd object
						
						fnindex = axisarray.getfieldnameindexCI("unit");
						if (fnindex != -1)
						{
							std::vector<std::string> units(NRRD_DIM_MAX);
							const char *cunits[NRRD_DIM_MAX];
					
							// Set some default values in case the
							// unit is not defined by the matlabarray
						
							for (long p=0;p<NRRD_DIM_MAX;p++)
							{
								units[p] = "unknown";
								cunits[p] = units[p].c_str();
							}
						
							// In case the unit is set in the matlabarray
							// add this unit to the clabels array
						
							for (long p=0;p<numaxis;p++)
							{
								farray = axisarray.getfield(p,fnindex);
								if (farray.getclass() == matlabarray::mlSTRING)
								{
									units[p] = farray.getstring();
									cunits[p] = units[p].c_str();
								}
							}
						
							nrrdAxisInfoSet_nva(scinrrd->nrrd,nrrdAxisInfoUnit,cunits);
						}
					
						// insert spacing information
				
						fnindex = axisarray.getfieldnameindexCI("spacing");
						if (fnindex != -1)
						{
							double spacing[NRRD_DIM_MAX];
							std::vector<double> data(1);
						
							// Set some default values in case the
							// spacing is not defined by the matlabarray
						
							for (long p=0;p<NRRD_DIM_MAX;p++)
							{
								spacing[p] = AIR_NAN;
							}
						
							for (long p=0;p<numaxis;p++)
							{
								farray = axisarray.getfield(p,fnindex);
								if ((farray.getclass() == matlabarray::mlDENSE)&&(farray.getnumelements() > 0))
								{
									farray.getnumericarray(data);
									spacing[p] = data[0];
								}
							}
						
							nrrdAxisInfoSet_nva(scinrrd->nrrd,nrrdAxisInfoSpacing,spacing);
						}

						// insert minimum information
					
						fnindex = axisarray.getfieldnameindexCI("min");
						if (fnindex != -1)
						{
							double mindata[NRRD_DIM_MAX];
							std::vector<double> data;
						
							// Set some default values in case the
							// minimum is not defined by the matlabarray
						
							for (long p=0;p<NRRD_DIM_MAX;p++)
							{
								mindata[p] = AIR_NAN;
							}
						
							for (long p=0;p<numaxis;p++)
							{
								farray = axisarray.getfield(p,fnindex);
								if ((farray.getclass() == matlabarray::mlDENSE)&&(farray.getnumelements() > 0))
								{
									farray.getnumericarray(data);
									mindata[p] = data[0];
								}
							}
						
							nrrdAxisInfoSet_nva(scinrrd->nrrd,nrrdAxisInfoMin,mindata);
						}
					
					
						// insert maximum information
					
						fnindex = axisarray.getfieldnameindexCI("max");
						if (fnindex != -1)
						{
							double maxdata[NRRD_DIM_MAX];
							std::vector<double> data;
						
							// Set some default values in case the
							// maximum is not defined by the matlabarray
						
							for (long p=0;p<NRRD_DIM_MAX;p++)
							{
								maxdata[p] = AIR_NAN;
							}
						
							for (long p=0;p<numaxis;p++)
							{
								farray = axisarray.getfield(p,fnindex);
								if ((farray.getclass() == matlabarray::mlDENSE)&&(farray.getnumelements() > 0))
								{
									farray.getnumericarray(data);
									maxdata[p] = data[0];
								}
							}
						
							nrrdAxisInfoSet_nva(scinrrd->nrrd,nrrdAxisInfoMax,maxdata);
						}
				
						// insert centering information
					
						fnindex = axisarray.getfieldnameindexCI("center");
						if (fnindex != -1)
						{
							long centerdata[NRRD_DIM_MAX];
							std::vector<long> data;
						
							// Set some default values in case the
							// maximum is not defined by the matlabarray
						
							for (long p=0;p<NRRD_DIM_MAX;p++)
							{
								centerdata[p] = 0;
							}
						
							for (long p=0;p<numaxis;p++)
							{
								farray = axisarray.getfield(p,fnindex);
								if ((farray.getclass() == matlabarray::mlDENSE)&&(farray.getnumelements() > 0))
								{
									farray.getnumericarray(data);
									centerdata[p] = data[0];
								}
							}
						
							nrrdAxisInfoSet_nva(scinrrd->nrrd,nrrdAxisInfoCenter,centerdata);
						}
					}
				}
	
				
				long propertyindex;
				propertyindex = mlarray.getfieldnameindexCI("property");
				
				if (propertyindex != -1)
				{
					mlPropertyTOsciProperty(mlarray,static_cast<SCIRun::PropertyManager *>(scinrrd.get_rep()));
				}

				if (mlarray.isfieldCI("name"))
				{
					if (scinrrd != 0)
					{	
						matlabarray matname;
						matname = mlarray.getfieldCI(0,"name");
						std::string str = matname.getstring();
						if (matname.isstring())	scinrrd->set_filename(str);
					}
		
				}
				else
				{
					if (scinrrd != 0)
					{
						std::string str = mlarray.getname();
						scinrrd->set_filename(str);
					}
				}
			
			}	
			
			break;
			
		default:
			{   // The program should not get here
				throw matlabconverter_error(); 
			}
	}
}


void matlabconverter::sciNrrdDataTOmlMatrix(SCITeem::NrrdDataHandle &scinrrd, matlabarray &mlarray)
{

	Nrrd	    *nrrdptr;
	matlabarray::mitype dataformat = datatype_;

	mlarray.clear();

	// first determine the size of a nrrd
	std::vector<long> dims;
	
	nrrdptr = scinrrd->nrrd;

	// check if there is any data 
	if (nrrdptr->dim == 0) return;
	
	dims.resize(nrrdptr->dim);
	
	long totsize = 1; // this one is used as an internal check and is handy to have
	for (long p=0; p<(nrrdptr->dim);p++)
	{
		dims[p] = nrrdptr->axis[p].size;
		totsize *= dims[p];
	}
	
	// if there is no data leave the object empty
	if (totsize == 0) return;
	
	// we now have to determine the type of the matlab array
	// It can be either the same as in the nrrd array or casted
	// to a more appropriate type
	// type will store the new matlab array type
	
	if(dataformat == matlabarray::miSAMEASDATA) dataformat = convertnrrdtype(nrrdptr->type);
	
	// create the matlab array structure
    mlarray.createdensearray(dims,dataformat);
	
	// having the correct pointer type will automatically invoke
	// the proper template function for casting and storing the data
	
	switch (nrrdptr->type)
	{
		case nrrdTypeDouble  : mlarray.setnumericarray(static_cast<double *>(nrrdptr->data),totsize,dataformat); break;
		case nrrdTypeFloat   : mlarray.setnumericarray(static_cast<float *>(nrrdptr->data),totsize,dataformat); break;
		case nrrdTypeChar    : mlarray.setnumericarray(static_cast<signed char *>(nrrdptr->data),totsize,dataformat); break;
		case nrrdTypeUChar   : mlarray.setnumericarray(static_cast<unsigned char *>(nrrdptr->data),totsize,dataformat); break;
		case nrrdTypeShort   : mlarray.setnumericarray(static_cast<signed short *>(nrrdptr->data),totsize,dataformat); break;
		case nrrdTypeUShort  : mlarray.setnumericarray(static_cast<unsigned short *>(nrrdptr->data),totsize,dataformat); break;
		case nrrdTypeInt	 : mlarray.setnumericarray(static_cast<signed long *>(nrrdptr->data),totsize,dataformat); break;
		case nrrdTypeUInt    : mlarray.setnumericarray(static_cast<unsigned long *>(nrrdptr->data),totsize,dataformat); break;
		case nrrdTypeLLong   : mlarray.setnumericarray(static_cast<int64 *>(nrrdptr->data),totsize,dataformat); break;
		case nrrdTypeULLong  : mlarray.setnumericarray(static_cast<uint64 *>(nrrdptr->data),totsize,dataformat); break;	
	}
}


void matlabconverter::sciNrrdDataTOmlArray(SCITeem::NrrdDataHandle &scinrrd, matlabarray &mlarray)
{
	matlabarray matrix;
	sciNrrdDataTOmlMatrix(scinrrd,matrix);
		
	mlarray.createstructarray();
	mlarray.setfield(0,"data",matrix);
		
	// Set the properies of the axis
	std::vector<std::string> axisfieldnames(7);
	axisfieldnames[0] = "size";
	axisfieldnames[1] = "spacing";
	axisfieldnames[2] = "min";
	axisfieldnames[3] = "max";
	axisfieldnames[4] = "center";
	axisfieldnames[5] = "label";
	axisfieldnames[6] = "unit";
	
	Nrrd	*nrrdptr;
	nrrdptr = scinrrd->nrrd;
				
	matlabarray axisma;
	std::vector<long> dims(2);
	dims[0] = nrrdptr->dim;
	dims[1] = 1;
	axisma.createstructarray(dims,axisfieldnames);
		
		
	for (long p=0; p<nrrdptr->dim; p++ )
	{
		matlabarray sizema;
		matlabarray spacingma;
		matlabarray minma;
		matlabarray maxma;
		matlabarray centerma;
		matlabarray labelma;
		matlabarray unitma;
			
		sizema.createdoublescalar(static_cast<double>(nrrdptr->axis[p].size));
		axisma.setfield(p,0,sizema);
		spacingma.createdoublescalar(nrrdptr->axis[p].spacing);
		axisma.setfield(p,1,spacingma);
		minma.createdoublescalar(nrrdptr->axis[p].min);
		axisma.setfield(p,2,minma);
		maxma.createdoublescalar(nrrdptr->axis[p].max);
		axisma.setfield(p,3,maxma);
		centerma.createdoublescalar(static_cast<double>(nrrdptr->axis[p].center));
		axisma.setfield(p,4,centerma);
		if (nrrdptr->axis[p].label == 0)
		{
			labelma.createstringarray();
		}
		else
		{
			labelma.createstringarray(nrrdptr->axis[p].label);
		}
		axisma.setfield(p,5,labelma);
		if (nrrdptr->axis[p].unit == 0)
		{
			unitma.createstringarray();
		}
		else
		{
			unitma.createstringarray(nrrdptr->axis[p].unit);
		}
		axisma.setfield(p,6,unitma);
	}
	
	mlarray.setfield(0,"axis",axisma);
	sciPropertyTOmlProperty(static_cast<SCIRun::PropertyManager *>(scinrrd.get_rep()),mlarray);
}

#endif



// Routine for discovering which kind of mesh is being supplied
// It reads the matlabarray and looks for certain fields to see whether
// it is convertible into a SCIRun mesh.
// Currently there are dozens of mesh and field types being used in
// SCIRun, each having a similar but different interface. This function
// only supports the most used ones:
//   PointCloudMesh
//   CurveMesh
//   TriSurfMesh
//   QuadSurfMesh
//   TetVolMesh
//   HexVolMesh
//   PrismVolMesh
//   StructCurveMesh
//   StructQuadSurfMesh
//   StructHexVolMesh
//   Scanline
//   Image
//   LatVol
//   any suggestions for other types that need support ??

long matlabconverter::sciFieldCompatible(matlabarray &mlarray,std::string &infostring)
{

	// If it is regular matrix translate it to a image or a latvol
	// The following section of code rewrites the matlab matrix into a
	// structure and then the normal routine picks up the field and translates it
	// properly.

	if (mlarray.isdense())
	{
		long numdims = mlarray.getnumdims();
		if ((numdims >0)&&(numdims < 4))
		{
			matlabarray ml;
			matlabarray dimsarray;
			std::vector<long> d = mlarray.getdims();
			if ((d[0]==1)||(d[1]==1))
			{
				if (d[0]==1) d[0] = d[1];
				long temp = d[0];
				d.resize(1);
				d[0] = temp;
			}			
			dimsarray.createlongvector(d);
			ml.createstructarray();
			ml.setfield(0,"dims",dimsarray);
			ml.setfield(0,"field",mlarray);
			ml.setname(mlarray.getname());			
			mlarray = ml;
		}
	}

	if (!mlarray.isstruct()) return(0); // not compatible if it is not structured data
	fieldstruct fs = analyzefieldstruct(mlarray); // read the main structure of the object
	
	// Check what kind of field has been supplied
	
	// The fieldtype string, contains the type of field, so it can be listed in the
	// infostring. Just to supply the user with a little bit more data.
	std::string fieldtype;
	fieldtype = "NO FIELD DATA";
	
	// The next step will incorporate a new way of dealing with fields
	// Basically we alter the way fields are processed:
	// instead of specifying vectorfield as a field, it is now allowed and recommended to use
	// two fields: .field describing the data and .fieldtype for the type of data
	// The next piece of code translates the new notation back to the old one.
	
	if (!(fs.fieldtype.isempty()))
	{
		if ((fs.fieldtype.compareCI("vector"))&&(fs.vectorfield.isempty())&&(fs.scalarfield.isdense()))
		{   
			fs.vectorfield = fs.scalarfield;
			fs.scalarfield.clear();
		}

		if ((fs.fieldtype.compareCI("tensor"))&&(fs.tensorfield.isempty())&&(fs.scalarfield.isdense()))
		{   
			fs.tensorfield = fs.scalarfield;
			fs.scalarfield.clear();
		}
	}
	
	if (fs.scalarfield.isdense()) fieldtype = "SCALAR FIELD";
	if (fs.vectorfield.isdense()) fieldtype = "VECTOR FIELD";
	if (fs.tensorfield.isdense()) fieldtype = "TENSOR FIELD";
	
	// Field data has been analysed, now analyse the connectivity data
	// Connectivity data needs to be or edge data, or face data, or cell data,
	// or no data in which case it is a point cloud.
	
	
	// Tests for images/latvols/scanlines
	// vvvvvvvvvvvvvvvvvvvvvvvvvv
	
	// Test whether transform is a 4 by 4 matrix
	if (fs.transform.isdense())
	{
		if (fs.transform.getnumdims() != 2) return(0);
		if ((fs.transform.getn() != 4)&&(fs.transform.getm() != 4)) return(0);
	}
	
	// Test whether rotation is a 3 x 3 matrix
	if (fs.rotation.isdense())
	{
		if (fs.rotation.getnumdims() != 2) return(0);
		if ((fs.rotation.getn()!=3)&&(fs.rotation.getm()!=3)) return(0);
	}
	
	// Test whether offset is a 1x3 or 3x1 vector
	if (fs.offset.isdense())
	{
		if (fs.offset.getnumdims() != 2) return(0);
		if ((fs.offset.getn()*fs.offset.getm())!=3) return(0);
	}

	// Test whether size is a 1x3 or 3x1 vector
	if (fs.size.isdense())
	{
		if (fs.size.getnumdims() != 2) return(0);
		if ((fs.size.getn()*fs.size.getm())!=3) return(0);
	}
	
	// In case one of the components above is given and dims is not given,
	// derive this one from the size of the data 
	if (((fs.rotation.isdense())||(fs.offset.isdense())||(fs.size.isdense())||(fs.transform.isdense())
		||(fs.elemtype.compareCI("scanline"))||(fs.elemtype.compareCI("image"))||(fs.elemtype.compareCI("latvol")))&&(fs.dims.isempty()))
	{
		if (fs.scalarfield.isdense()) 
			{  std::vector<long> dims = fs.scalarfield.getdims();
			   fs.dims.createlongvector(dims);
			}
		if (fs.vectorfield.isdense()) 
			{  std::vector<long> dims = fs.vectorfield.getdims();
			   fs.dims.createlongvector((dims.size()-1),&(dims[1]));
			}
		if (fs.tensorfield.isdense()) 
			{  std::vector<long> dims = fs.tensorfield.getdims();
			   fs.dims.createlongvector((dims.size()-1),&(dims[1]));
			}
		if ((fs.data_at == SCIRun::Field::CELL)||(fs.data_at == SCIRun::Field::FACE)||(fs.data_at == SCIRun::Field::EDGE))
		{
			std::vector<long> dims = fs.scalarfield.getdims();
			for (long p = 0; p<dims.size(); p++) dims[p] = dims[p]+1;
			fs.dims.createlongvector(dims);
		}
	}
	
	// if dims is not present it is not a regular mesh
	// Data at edges, faces, or cells is only possible if the data
	// is of that dimension otherwise skip it and declare the data not usable
	
	if (fs.dims.isdense())
	{
		if ((fs.dims.getnumelements()==1)&&(fs.data_at!=SCIRun::Field::NONE)&&(fs.data_at!=SCIRun::Field::NODE)&&(fs.data_at!=SCIRun::Field::EDGE)) return(0);
		if ((fs.dims.getnumelements()==2)&&(fs.data_at!=SCIRun::Field::NONE)&&(fs.data_at!=SCIRun::Field::NODE)&&(fs.data_at!=SCIRun::Field::FACE)) return(0);
		if ((fs.dims.getnumelements()==3)&&(fs.data_at!=SCIRun::Field::NONE)&&(fs.data_at!=SCIRun::Field::NODE)&&(fs.data_at!=SCIRun::Field::CELL)) return(0);
	}
	
	// If it survived until here it should be translatable or not a regular mesh at all
	
	if (fs.dims.isdense())
	{
		long size = fs.dims.getnumelements();
		
		if ((size > 0)&&(size < 4))
		{
			std::ostringstream oss;
			std::string name = mlarray.getname();
			oss << name << " ";
			if (name.length() < 30) oss << std::string(30-(name.length()),' '); // add some form of spacing		
		
			if (fs.elemtype.isstring())
			{   // explicitly stated type (check whether type confirms the guessed type, otherwise someone supplied us with improper data)
				if ((fs.elemtype.compareCI("scanline"))&&(size!=1)) return(0);
				if ((fs.elemtype.compareCI("image"))&&(size!=2)) return(0);
				if ((fs.elemtype.compareCI("latvolmesh"))&&(size!=3)) return(0);
			}	

			switch (size)
			{
				case 1:
					oss << "[SCANLINE - " << fieldtype << "]";
					break;
				case 2:
					oss << "[IMAGE - " << fieldtype << "]";
					break;
				case 3:	
					oss << "[LATVOLMESH - " << fieldtype << "]";
					break;
			}
			infostring = oss.str();
			return(1);					
		}
		else
		{
			return(0);
		}
	
	}

	
	// Test for structured meshes
	// vvvvvvvvvvvvvvvvvvvvvvv
	
	if ((fs.x.isdense())&&(fs.y.isdense())&(fs.z.isdense()))
	{
	
		long numdims = fs.x.getnumdims();
		if (fs.y.getnumdims() != numdims) return(0);
		if (fs.z.getnumdims() != numdims) return(0);
		
		std::vector<long> dimsx = fs.x.getdims();
		std::vector<long> dimsy = fs.y.getdims();
		std::vector<long> dimsz = fs.z.getdims();
		
		for (long p=0 ; p < numdims ; p++)
		{
			if(dimsx[p] != dimsy[p]) return(0);
			if(dimsx[p] != dimsz[p]) return(0);
		}

		
		std::ostringstream oss;
		std::string name = mlarray.getname();
		oss << name << " ";
		if (name.length() < 30) oss << std::string(30-(name.length()),' '); // add some form of spacing		
		
		// Minimum dimensions is in matlab is 2 and hence detect any empty dimension
		
		if (numdims == 2)
		{
			if ((dimsx[0] == 1)||(dimsx[1] == 1)) numdims = 1;
		}

		// Disregard data at odd locations. The translation function for those is not straight forward
		// Hence disregard those data locations.

		if ((numdims==1)&&(fs.data_at!=SCIRun::Field::NONE)&&(fs.data_at!=SCIRun::Field::NODE)&&(fs.data_at!=SCIRun::Field::EDGE)) return(0);
		if ((numdims==2)&&(fs.data_at!=SCIRun::Field::NONE)&&(fs.data_at!=SCIRun::Field::NODE)&&(fs.data_at!=SCIRun::Field::FACE)) return(0);
		if ((numdims==3)&&(fs.data_at!=SCIRun::Field::NONE)&&(fs.data_at!=SCIRun::Field::NODE)&&(fs.data_at!=SCIRun::Field::CELL)) return(0);
	
		if (fs.elemtype.isstring())
		{   // explicitly stated type (check whether type confirms the guessed type, otherwise someone supplied us with improper data)
			if ((fs.elemtype.compareCI("structcurve"))&&(numdims!=1)) return(0);
			if ((fs.elemtype.compareCI("structquadsurf"))&&(numdims!=2)) return(0);
			if ((fs.elemtype.compareCI("structhexvol"))&&(numdims!=3)) return(0);
		}		
			
		switch (numdims)
		{
			case 1:
				oss << "[STRUCTURED CURVEMESH (" << dimsx[0] << " nodes) - " << fieldtype << "]";
				break;
			case 2:
				oss << "[STRUCTURED QUADSURFMESH (" << dimsx[0] << "x" << dimsx[1] << " nodes) - " << fieldtype << "]";
				break;
			case 3:
				oss << "[STRUCTURED HEXVOLMESH (" << dimsx[0] << "x" << dimsx[1] << "x" << dimsx[2] << " nodes) - " << fieldtype << "]";
				break;
			default:
				return(0);  // matrix is not compatible
		}
		infostring = oss.str();
		return(1);		
	}
	

	if (fs.node.isempty()) return(0); // a node matrix is always required
	
	if (fs.node.getnumdims() > 2) return(0); // Currently N dimensional arrays are not supported here


	// Check the dimensions of the NODE array supplied only [3xM] or [Mx3] are supported
	long m,n;
	m = fs.node.getm();
	n = fs.node.getn();
	
	long numpoints;
	long numel;
	
	if ((n==0)||(m==0)) return(0); //empty matrix, no nodes => no mesh => no field......
	if ((n != 3)&&(m != 3)) return(0); // SCIRun is ONLY 3D data, no 2D, or 1D
	
	numpoints = n;
	if ((m!=3)&&(n==3)) numpoints = m;
	
	if ((fs.edge.isempty())&&(fs.face.isempty())&&(fs.cell.isempty()))
	{
		// These is no connectivity data => it must be a pointcloud ;)
		// Supported mesh/field types here:
		// PointCloudField
		
		if (fs.elemtype.isstring())
		{   // explicitly stated type (check whether type confirms the guessed type, otherwise someone supplied us with improper data)
			if (fs.elemtype.compareCI("pointcloud")) return(0);
		}
		
		// Data at edges, faces, and cells is nonsense for point clouds 
		if ((fs.data_at!=SCIRun::Field::NONE)&&(fs.data_at!=SCIRun::Field::NODE)) return(0);		
		
		// Create an information string for the GUI
		std::ostringstream oss;
		std::string name = mlarray.getname();	
		oss << name << "  ";
		if (name.length() < 30) oss << std::string(30-(name.length()),' '); // add some form of spacing
		oss << "[POINTCLOUD (" << numpoints << " nodes) - " << fieldtype << "]";
		infostring = oss.str();
		return(1);
	}
	
	if (fs.edge.isdense())
	{
		// Edge data is provide hence it must be some line element!
		// Supported mesh/field types here:
		//  CurveField
		if (fs.elemtype.isstring())
		{   // explicitly stated type 
			if (fs.elemtype.compareCI("curve")) return(0);
		}

		// Data at faces, and cells is nonsense for  curves
		if ((fs.data_at!=SCIRun::Field::NONE)&&(fs.data_at!=SCIRun::Field::NODE)&&(fs.data_at!=SCIRun::Field::EDGE)) return(0);	

		// Test whether someone made it into a line/surface/volume element type.
		// Since multiple connectivity matrices do not make sense (at least at this point)
		// we do not allow them to be used, hence they are not compatible
		if ((!fs.face.isempty())||(!fs.cell.isempty())) return(0); // a matrix with multiple connectivities is not yet allowed
		
		// Connectivity should be 2D
		if (fs.edge.getnumdims() > 2) return(0);
		
		// Check whether the connectivity data makes any sense, if not one of the dimensions is 2, the data is some not
		// yet supported higher order element
		m = fs.edge.getm();
		n = fs.edge.getn();
		
		if ((n!=2)&&(m!=2)) return(0); 
	
		numel = n;
		if ((m!=2)&&(n==2)) numel = m;
				
		std::ostringstream oss;	
		std::string name = mlarray.getname();
		oss << name << "  ";
		if (name.length() < 30) oss << std::string(30-(name.length()),' '); // add some form of spacing
		oss << "[CURVEFIELD (" << numpoints << " nodes, " << numel << " edges) - " << fieldtype << "]";
		infostring = oss.str();
		return(1);
	}

	if (fs.face.isdense())
	{
		// Supported mesh/field types here:
		// TriSurfField

		// The connectivity data should be 2D
		if (fs.face.getnumdims() > 2) return(0);
		
		m = fs.face.getm();
		n = fs.face.getn();
	
		// if the cell matrix is not empty, the mesh is both surface and volume, which
		// we do not support at the moment.
		if((!fs.cell.isempty())) return(0);

		// Data at scells is nonsense for surface elements
		if ((fs.data_at!=SCIRun::Field::NONE)&&(fs.data_at!=SCIRun::Field::NODE)&&(fs.data_at!=SCIRun::Field::EDGE)&&(fs.data_at!=SCIRun::Field::FACE)) return(0);	

		if ((m==3)||((n==3)&&(n!=4)))
		{
			// check whether the element type is explicitly given
			if (fs.elemtype.isstring())
			{   // explicitly stated type 
				if (fs.elemtype.compareCI("trisurf")) return(0);
			}
	
			numel = n;
			if ((m!=3)&&(n==3)) numel = m;

							
			// Generate an output string describing the field we just reckonized
			std::ostringstream oss;	
			std::string name = mlarray.getname();
			oss << name << "  ";
			if (name.length() < 30) oss << std::string(30-(name.length()),' '); // add some form of spacing
			oss << "[TRISURFFIELD (" << numpoints << " nodes, " << numel << " faces) - " << fieldtype << "]";
			infostring = oss.str();		
			return(1);
		}
		
		if ((m==4)||((n==4)&&(n!=3)))
		{
			// check whether the element type is explicitly given
			if (fs.elemtype.isstring())
			{   // explicitly stated type 
				if (fs.elemtype.compareCI("quadsurf")) return(0);
			}
			
			numel = n;
			if ((m!=4)&&(n==4)) numel = m;
			
			
			// Generate an output string describing the field we just reckonized
			std::ostringstream oss;	
			std::string name = mlarray.getname();
			oss << name << "  ";
			if (name.length() < 30) oss << std::string(30-(name.length()),' '); // add some form of spacing
			oss << "[QUADSURFFIELD (" << numpoints << "nodes , " << numel << " faces) - " << fieldtype << "]";
			infostring = oss.str();		
			return(1);
		}
		
		return(0);
	}

	if (fs.cell.isdense())
	{
		// Supported mesh/field types here:
		// TetVolField
		// HexVolField
		
		if (fs.cell.getnumdims() > 2) return(0);
		
		m = fs.cell.getm();
		n = fs.cell.getn();
		
		// m is the preferred direction for the element node indices of one element, n is the number of elements
		// However we except a transposed matrix as long as m != 8 , which would indicate hexahedral element
		if ((m==4)||((n==4)&&(m!=8)&&(m!=6)))
		{
			if (fs.elemtype.isstring())
			{   // explicitly stated type 
				if (fs.elemtype.compareCI("tetvol")) return(0);
			}	
			
			numel = n;
			if ((m!=4)&&(n==4)) numel = m;

			
			std::ostringstream oss;	
			std::string name = mlarray.getname();			
			oss << name << "  ";
			if (name.length() < 30) oss << std::string(30-(name.length()),' '); // add some form of spacing
			oss << "[TETVOLFIELD (" << numpoints << " nodes, " << numel << " cells) - " << fieldtype << "]";
			infostring = oss.str();		
			return(1);
		}
		// In case it is a hexahedral mesh
		else if((m==8)||((n==8)&&(m!=4)&&(m!=6)))
		{
			if (fs.elemtype.isstring())
			{   // explicitly stated type 
				if (fs.elemtype.compareCI("hexvol")) return(0);
			}	
			
			numel = n;
			if ((m!=8)&&(n==8)) numel = m;
			
			std::ostringstream oss;	
			std::string name = mlarray.getname();		
			oss << name << "  ";
			if (name.length() < 30) oss << std::string(30-(name.length()),' '); // add some form of spacing
			oss << "[HEXVOLFIELD (" << numpoints << " nodes, " << numel << " cells) - " << fieldtype << "]";
			infostring = oss.str();		
			return(1);		
		}
		
		else if((m==6)||((n==6)&&(m!=4)&&(m!=8)))
		{
			if (fs.elemtype.isstring())
			{   // explicitly stated type 
				if (fs.elemtype.compareCI("prismvol")) return(0);
			}	
			
			numel = n;
			if ((m!=6)&&(n==6)) numel = m;

			
			std::ostringstream oss;	
			std::string name = mlarray.getname();		
			oss << name << "  ";
			if (name.length() < 30) oss << std::string(30-(name.length()),' '); // add some form of spacing
			oss << "[PRISMVOLFIELD (" << numpoints << " nodes, " << numel << " cells) - " << fieldtype << "]";
			infostring = oss.str();		
			return(1);		
		}
		
		return(0);
	}
	return(0);
}



void matlabconverter::mlArrayTOsciField(matlabarray &mlarray,SCIRun::FieldHandle &scifield)
{

	// If it is regular matrix translate it to a image or a latvol
	// The following section of code rewrites the matlab matrix into a
	// structure and then the normal routine picks up the field and translates it
	// properly.
	if (mlarray.isdense())
	{
		long numdims = mlarray.getnumdims();
		if ((numdims >0)&&(numdims < 4))
		{
			matlabarray ml;
			matlabarray dimsarray;
			std::vector<long> d = mlarray.getdims();
			if ((d[0]==1)||(d[1]==1))
			{
				if (d[0]==1) d[0] = d[1];
				long temp = d[0];
				d.resize(1);
				d[0] = temp;
			}
			dimsarray.createlongvector(d);
			ml.createstructarray();
			ml.setfield(0,"dims",dimsarray);
			ml.setfield(0,"field",mlarray);
			ml.setname(mlarray.getname());
			mlarray = ml;
		}
	}

	if (!mlarray.isstruct()) throw matlabconverter_error(); // not compatible if it is not structured data
	fieldstruct fs = analyzefieldstruct(mlarray); // read the main structure of the object
	
	// Convert the node information
	// Each field needs to have the position of the nodes defined
	// Currently SCIRun only accepts nodes in a 3D cartesian coordinate
	// system.
	// Dimensions are checked and the matrix is transposed if it is necessary


	// The next step will incorporate a new way of dealing with fields
	// Basically we alter the way fields are processed:
	// instead of specifying vectorfield as a field, it is now allowed and recommended to use
	// two fields: .field describing the data and .fieldtype for the type of data
	// The next piece of code translates the new notation back to the old one.
	
	// In case one of teh above components is given and dims is not given,
	// derive this one from the size of the data 
	if (((fs.rotation.isdense())||(fs.offset.isdense())||(fs.size.isdense())||(fs.transform.isdense())
		||(fs.elemtype.compareCI("scanline"))||(fs.elemtype.compareCI("image"))||(fs.elemtype.compareCI("latvol")))&&(fs.dims.isempty()))
	{
		if (fs.scalarfield.isdense()) 
			{  std::vector<long> dims = fs.scalarfield.getdims();
			   fs.dims.createlongvector(dims);
			}
		if (fs.vectorfield.isdense()) 
			{  std::vector<long> dims = fs.vectorfield.getdims();
			   fs.dims.createlongvector((dims.size()-1),&(dims[1]));
			}
		if (fs.tensorfield.isdense()) 
			{  std::vector<long> dims = fs.tensorfield.getdims();
			   fs.dims.createlongvector((dims.size()-1),&(dims[1]));
			}
		if ((fs.data_at == SCIRun::Field::CELL)||(fs.data_at == SCIRun::Field::FACE)||(fs.data_at == SCIRun::Field::EDGE))
		{
			std::vector<long> dims = fs.scalarfield.getdims();
			for (long p = 0; p<dims.size(); p++) dims[p] = dims[p]+1;
			fs.dims.createlongvector(dims);
		}
	}
	
	
	if ((fs.node.isempty())&&(fs.x.isempty())&&(fs.dims.isempty())) throw matlabconverter_error(); // a node matrix is always required
	
	long m,n,numnodes;
	
	if (fs.node.isdense())
	{
		m = fs.node.getm();
		n = fs.node.getn();
	
		// This condition should have been checked by the compatibility algorithm
		if ((m != 3)&&(n != 3)) throw matlabconverter_error();
	
		// In case the matrix is transposed, reorder it in the proper order for this converter
		if (m != 3) fs.node.transpose();
		numnodes = fs.node.getn();
	}
	
	if (fs.edge.isdense())
	{
		m = fs.node.getm();
		n = fs.node.getn();
	
		if (fs.elemtype.isstring())
		{   // explicitly stated type 
			if (fs.elemtype.compareCI("curve"))
			{
				if ((n!=2)&&(m!=2)) throw matlabconverter_error();
				if (m != 2) fs.edge.transpose();
			}
		}
		else
		{
			if ((n!=2)&&(m!=2)) throw matlabconverter_error();
			if (m != 2) fs.edge.transpose();
		}
	}
	
	if (fs.face.isdense())
	{
		m = fs.face.getm();
		n = fs.face.getn();
		
		if (fs.elemtype.isstring())
		{   // explicitly stated type 
			if (fs.elemtype.compareCI("trisurf"))
			{
				if ((n!=3)&&(m!=3)) throw matlabconverter_error();
				if (m!=3) fs.face.transpose();
			}
			
			if (fs.elemtype.compareCI("quadsurf"))
			{
				if ((n!=4)&&(m!=4)) throw matlabconverter_error();
				if (m!=4) fs.face.transpose();
			}
		}
		else
		{
			if ((n!=3)&&(m!=3)&&(n!=4)&&(m!=4)) throw matlabconverter_error();
			if ((m!=3)&&(m!=4)) fs.face.transpose();
		}
	}
	
	if (fs.cell.isdense())
	{
		m = fs.cell.getm();
		n = fs.cell.getn();
		
		if (fs.elemtype.isstring())
		{   // explicitly stated type 
			if (fs.elemtype.compareCI("tetvol"))
			{
				if ((n!=4)&&(m!=4)) throw matlabconverter_error();
				if (m!=4) fs.cell.transpose();
			}
			
			if (fs.elemtype.compareCI("hexvol"))
			{
				if ((n!=8)&&(m!=8)) throw matlabconverter_error();
				if (m!=8) fs.cell.transpose();
			}
			
			if (fs.elemtype.compareCI("prismvol"))
			{
				if ((n!=6)&&(m!=6)) throw matlabconverter_error();
				if (m!=6) fs.cell.transpose();
			}
		}
		else
		{
			if ((n!=4)&&(m!=4)&&(n!=6)&&(m!=6)&&(n!=8)&&(m!=8)) throw matlabconverter_error();
			if ((m!=4)&&(m!=6)&&(m!=8)) fs.cell.transpose();
		}
	}
	
	
	
	// Check the field information and preprocess the field information

	// detect whether there is data and where it is
	SCIRun::Field::data_location data_at = SCIRun::Field::NONE;

	// Currently Only SCALAR, VECTOR, and TENSOR objects are supported
	// As there is an infinite amount of possibilities, it is impossible
	// to write a converter for each type, so a selection had to be made
	
	// The default location for data is at the nodes
	// at least when there is a field defined
	
	if ((fs.scalarfield.isdense())||(fs.vectorfield.isdense())||(fs.tensorfield.isdense())) data_at = SCIRun::Field::NODE;
	if (!(fs.fieldlocation.isempty())) data_at = fs.data_at;
	
	// In case the location has been supplied but no actual data has been supplied, reset the location to NONE
	if ((fs.scalarfield.isempty())&&(fs.vectorfield.isempty())&&(fs.tensorfield.isempty())) data_at = SCIRun::Field::NONE;

	// Currently the following assumption on the data is made:
	// first dimension is scalar/vector/tensor (in case of scalar this dimension does not need to exist)
	// second dimension is the number of nodes/edges/faces/cells (this one always exists)
	
	// This algorithm tries to transpose the connectivity data if needed
	if (data_at != SCIRun::Field::NONE)
	{
		if (fs.scalarfield.isdense())
		{
			if ((fs.x.isempty())&&(fs.y.isempty())&&(fs.z.isempty())&&(fs.dims.isempty()))
			{
		
				if (fs.scalarfield.getnumdims() == 2)
				{ 
					m = fs.scalarfield.getm();
					n = fs.scalarfield.getn();
					if ((m != 1)&&(n != 1)) 
					{   // ignore the data and only load the mesh
						data_at = SCIRun::Field::NONE;
					}
					else
					{
						if (m != 1) fs.scalarfield.transpose();
					}
				}
				else
				{   // ignore the data and only load the mesh 
					data_at = SCIRun::Field::NONE;
				}
			}
		}
		if (fs.vectorfield.isdense())
		{
			if ((fs.x.isempty())&&(fs.y.isempty())&&(fs.z.isempty())&&(fs.dims.isempty()))
			{

				if (fs.vectorfield.getnumdims() == 2)
				{   
					m = fs.vectorfield.getm();
					n = fs.vectorfield.getn();
					if ((m != 3)&&(n != 3)) 
					{
						data_at = SCIRun::Field::NONE;
					}
					else
					{
						if (m != 3) fs.vectorfield.transpose();
					}
				}
				else
				{
					data_at = SCIRun::Field::NONE;
				}
			}
		}
		if (fs.tensorfield.isdense())
		{
			
			if ((fs.x.isempty())&&(fs.y.isempty())&&(fs.z.isempty())&&(fs.dims.isempty()))
			{

				if (fs.tensorfield.getnumdims() == 2)
				{
					m = fs.tensorfield.getm();
					n = fs.tensorfield.getn();
					if ((m != 6)&&(n != 6)&&(m != 9)&&(n != 9))
					{
						data_at = SCIRun::Field::NONE;
					}
					else
					{
						if ((m != 6)&&(m != 9)) fs.tensorfield.transpose();
					}
				}
				else
				{
					data_at = SCIRun::Field::NONE;
				}
			}
		}
	}
	
	if (fs.dims.isdense())
	{
		long numdims = fs.dims.getnumelements();
		std::vector<long> dims; 
		fs.dims.getnumericarray(dims);
		
		switch (numdims)
		{
			case 1:
				{
					SCIRun::ScanlineMeshHandle meshH;
					SCIRun::Point PointO(0.0,0.0,0.0);
					SCIRun::Point PointP(static_cast<double>(dims[0]),0.0,0.0);
					meshH = new SCIRun::ScanlineMesh(static_cast<unsigned int>(dims[0]),PointO,PointP);
					if (fs.transform.isdense())
					{
						SCIRun::Transform T;
						double trans[16];
						fs.transform.getnumericarray(trans,16);
						T.set_trans(trans);
						meshH->transform(T);
					}
					else
					{
						SCIRun::Transform T;
						double trans[16];
						for (long p = 0; p<16;p++) trans[p] = 0;
						trans[0] = 1.0; trans[5]=1.0; trans[10]=1.0; trans[15]=1.0;
						if (fs.rotation.isdense())
						{
							double rot[9];
							fs.rotation.getnumericarray(rot,9);
							trans[0] = rot[0]; trans[1] = rot[1]; trans[2] = rot[2];
							trans[4] = rot[3]; trans[5] = rot[4]; trans[6] = rot[5];
							trans[8] = rot[7]; trans[9] = rot[8]; trans[10] = rot[9];

						}
						if (fs.offset.isdense())
						{
							double offset[3];
							offset[0] = 0.0; offset[1] = 0.0; offset[2] = 0.0;
							fs.offset.getnumericarray(offset,3);
							trans[12] = offset[0]; trans[13] = offset[1]; trans[14] = offset[2];
						}
						if (fs.size.isdense())
						{
							double scale[3];
							scale[0] = 1.0; scale[1] = 1.0; scale[2] = 1.0;
							fs.size.getnumericarray(scale,3);
							trans[0] *= scale[0]; trans[4] *= scale[0]; trans[8] *= scale[0];
							trans[1] *= scale[1]; trans[5] *= scale[1]; trans[9] *= scale[1];
							trans[2] *= scale[2]; trans[6] *= scale[2]; trans[10] *= scale[2];
						}
						T.set_trans(trans);
						meshH->transform(T);						
					}
						

					if ((fs.scalarfield.isempty())&&(fs.vectorfield.isempty())&&(fs.tensorfield.isempty()))
					{
						SCIRun::ScanlineField<double> *fieldptr;
						fieldptr = new SCIRun::ScanlineField<double>(meshH,data_at);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
					}
					if (fs.scalarfield.isdense())
					{
						
						switch (fs.scalarfield.gettype())
						{
							case miINT8:
							case miUINT8:
								{
								SCIRun::ScanlineField<char> *fieldptr = new SCIRun::ScanlineField<char>(meshH,data_at);
								addscalardata(fieldptr,fs.scalarfield);
								scifield = static_cast<SCIRun::Field *>(fieldptr);
								}
								break;
							case miINT16:
								{
								SCIRun::ScanlineField<signed short> *fieldptr = new SCIRun::ScanlineField<signed short>(meshH,data_at);
								addscalardata(fieldptr,fs.scalarfield);
								scifield = static_cast<SCIRun::Field *>(fieldptr);					
								}
								break;
							case miUINT16:
								{
								SCIRun::ScanlineField<unsigned short> *fieldptr = new SCIRun::ScanlineField<unsigned short>(meshH,data_at);
								addscalardata(fieldptr,fs.scalarfield);
								scifield = static_cast<SCIRun::Field *>(fieldptr);
								}
								break;
							case miINT32:
								{
								SCIRun::ScanlineField<signed long> *fieldptr = new SCIRun::ScanlineField<signed long>(meshH,data_at);
								addscalardata(fieldptr,fs.scalarfield);
								scifield = static_cast<SCIRun::Field *>(fieldptr);
								}
								break;
							case miUINT32:
								{
								SCIRun::ScanlineField<unsigned long> *fieldptr = new SCIRun::ScanlineField<unsigned long>(meshH,data_at);
								addscalardata(fieldptr,fs.scalarfield);
								scifield = static_cast<SCIRun::Field *>(fieldptr);
								}
								break;
							case miSINGLE:
								{
								SCIRun::ScanlineField<float> *fieldptr = new SCIRun::ScanlineField<float>(meshH,data_at);
								addscalardata(fieldptr,fs.scalarfield);
								scifield = static_cast<SCIRun::Field *>(fieldptr);
								}
								break;
							case miDOUBLE:
							default:
								{
								SCIRun::ScanlineField<double> *fieldptr = new SCIRun::ScanlineField<double>(meshH,data_at);
								addscalardata(fieldptr,fs.scalarfield);
								scifield = static_cast<SCIRun::Field *>(fieldptr);				
								}
						}	

					}
					if (fs.vectorfield.isdense())
					{
						SCIRun::ScanlineField<SCIRun::Vector> *fieldptr;
						fieldptr = new SCIRun::ScanlineField<SCIRun::Vector>(meshH,data_at);
						addvectordata(fieldptr,fs.vectorfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
					}
					if (fs.tensorfield.isdense())
					{
						SCIRun::ScanlineField<SCIRun::Tensor> *fieldptr;
						fieldptr = new SCIRun::ScanlineField<SCIRun::Tensor>(meshH,data_at);
						addtensordata(fieldptr,fs.tensorfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
					}		
						
						
			}
			break;

			
			case 2:
				{
					SCIRun::ImageMeshHandle meshH;
					SCIRun::Point PointO(0.0,0.0,0.0);
					SCIRun::Point PointP(static_cast<double>(dims[0]),static_cast<double>(dims[1]),0.0);
					meshH = new SCIRun::ImageMesh(static_cast<unsigned int>(dims[0]),static_cast<unsigned int>(dims[1]),
						PointO,PointP);
					if (fs.transform.isdense())
					{
						SCIRun::Transform T;
						double trans[16];
						fs.transform.getnumericarray(trans,16);
						T.set_trans(trans);
						meshH->transform(T);
					}
					else
					{
						SCIRun::Transform T;
						double trans[16];
						for (long p = 0; p<16;p++) trans[p] = 0;
						trans[0] = 1.0; trans[5]=1.0; trans[10]=1.0; trans[15]=1.0;
						if (fs.rotation.isdense())
						{
							double rot[9];
							fs.rotation.getnumericarray(rot,9);
							trans[0] = rot[0]; trans[1] = rot[1]; trans[2] = rot[2];
							trans[4] = rot[3]; trans[5] = rot[4]; trans[6] = rot[5];
							trans[8] = rot[7]; trans[9] = rot[8]; trans[10] = rot[9];

						}
						if (fs.offset.isdense())
						{
							double offset[3];
							offset[0] = 0.0; offset[1] = 0.0; offset[2] = 0.0;
							fs.offset.getnumericarray(offset,3);
							trans[12] = offset[0]; trans[13] = offset[1]; trans[14] = offset[2];
						}
						if (fs.size.isdense())
						{
							double scale[3];
							scale[0] = 1.0; scale[1] = 1.0; scale[2] = 1.0;
							fs.size.getnumericarray(scale,3);
							trans[0] *= scale[0]; trans[4] *= scale[0]; trans[8] *= scale[0];
							trans[1] *= scale[1]; trans[5] *= scale[1]; trans[9] *= scale[1];
							trans[2] *= scale[2]; trans[6] *= scale[2]; trans[10] *= scale[2];
						}
						T.set_trans(trans);
						meshH->transform(T);						
					}

					if ((fs.scalarfield.isempty())&&(fs.vectorfield.isempty())&&(fs.tensorfield.isempty()))
					{
						SCIRun::ImageField<double> *fieldptr;
						fieldptr = new SCIRun::ImageField<double>(meshH,data_at);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
					}
					if (fs.scalarfield.isdense())
					{
						switch (fs.scalarfield.gettype())
						{
							case miINT8:
							case miUINT8:
								{
								SCIRun::ImageField<char> *fieldptr = new SCIRun::ImageField<char>(meshH,data_at);
								addscalardata2d(fieldptr,fs.scalarfield);
								scifield = static_cast<SCIRun::Field *>(fieldptr);
								}
								break;
							case miINT16:
								{
								SCIRun::ImageField<signed short> *fieldptr = new SCIRun::ImageField<signed short>(meshH,data_at);
								addscalardata2d(fieldptr,fs.scalarfield);
								scifield = static_cast<SCIRun::Field *>(fieldptr);					
								}
								break;
							case miUINT16:
								{
								SCIRun::ImageField<unsigned short> *fieldptr = new SCIRun::ImageField<unsigned short>(meshH,data_at);
								addscalardata2d(fieldptr,fs.scalarfield);
								scifield = static_cast<SCIRun::Field *>(fieldptr);
								}
								break;
							case miINT32:
								{
								SCIRun::ImageField<signed long> *fieldptr = new SCIRun::ImageField<signed long>(meshH,data_at);
								addscalardata2d(fieldptr,fs.scalarfield);
								scifield = static_cast<SCIRun::Field *>(fieldptr);
								}
								break;
							case miUINT32:
								{
								SCIRun::ImageField<unsigned long> *fieldptr = new SCIRun::ImageField<unsigned long>(meshH,data_at);
								addscalardata2d(fieldptr,fs.scalarfield);
								scifield = static_cast<SCIRun::Field *>(fieldptr);
								}
								break;
							case miSINGLE:
								{
								SCIRun::ImageField<float> *fieldptr = new SCIRun::ImageField<float>(meshH,data_at);
								addscalardata2d(fieldptr,fs.scalarfield);
								scifield = static_cast<SCIRun::Field *>(fieldptr);
								}
								break;
							case miDOUBLE:
							default:
								{
								SCIRun::ImageField<double> *fieldptr = new SCIRun::ImageField<double>(meshH,data_at);
								addscalardata2d(fieldptr,fs.scalarfield);
								scifield = static_cast<SCIRun::Field *>(fieldptr);				
								}
						}	

					}
					if (fs.vectorfield.isdense())
					{
						SCIRun::ImageField<SCIRun::Vector> *fieldptr;
						fieldptr = new SCIRun::ImageField<SCIRun::Vector>(meshH,data_at);
						addvectordata2d(fieldptr,fs.vectorfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
					}
					if (fs.tensorfield.isdense())
					{
						SCIRun::ImageField<SCIRun::Tensor> *fieldptr;
						fieldptr = new SCIRun::ImageField<SCIRun::Tensor>(meshH,data_at);
						addtensordata2d(fieldptr,fs.tensorfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
					}		
			}
			break;
			case 3:
			{
					SCIRun::LatVolMeshHandle meshH;
					SCIRun::Point PointO(0.0,0.0,0.0);
					SCIRun::Point PointP(static_cast<double>(dims[0]),static_cast<double>(dims[1]),static_cast<double>(dims[2]));
					meshH = new SCIRun::LatVolMesh(static_cast<unsigned int>(dims[0]),static_cast<unsigned int>(dims[1]),
						static_cast<unsigned int>(dims[2]),PointO,PointP);
					if (fs.transform.isdense())
					{
						SCIRun::Transform T;
						double trans[16];
						fs.transform.getnumericarray(trans,16);
						T.set_trans(trans);
						meshH->transform(T);
					}
					else
					{
						SCIRun::Transform T;
						double trans[16];
						for (long p = 0; p<16;p++) trans[p] = 0;
						trans[0] = 1.0; trans[5]=1.0; trans[10]=1.0; trans[15]=1.0;
						if (fs.rotation.isdense())
						{
							double rot[9];
							fs.rotation.getnumericarray(rot,9);
							trans[0] = rot[0]; trans[1] = rot[1]; trans[2] = rot[2];
							trans[4] = rot[3]; trans[5] = rot[4]; trans[6] = rot[5];
							trans[8] = rot[7]; trans[9] = rot[8]; trans[10] = rot[9];

						}
						if (fs.offset.isdense())
						{
							double offset[3];
							offset[0] = 0.0; offset[1] = 0.0; offset[2] = 0.0;
							fs.offset.getnumericarray(offset,3);
							trans[12] = offset[0]; trans[13] = offset[1]; trans[14] = offset[2];
						}
						if (fs.size.isdense())
						{
							double scale[3];
							scale[0] = 1.0; scale[1] = 1.0; scale[2] = 1.0;
							fs.size.getnumericarray(scale,3);
							trans[0] *= scale[0]; trans[4] *= scale[0]; trans[8] *= scale[0];
							trans[1] *= scale[1]; trans[5] *= scale[1]; trans[9] *= scale[1];
							trans[2] *= scale[2]; trans[6] *= scale[2]; trans[10] *= scale[2];
						}
						T.set_trans(trans);
						meshH->transform(T);						
					}
						

					if ((fs.scalarfield.isempty())&&(fs.vectorfield.isempty())&&(fs.tensorfield.isempty()))
					{
						SCIRun::LatVolField<double> *fieldptr;
						fieldptr = new SCIRun::LatVolField<double>(meshH,data_at);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
					}
					if (fs.scalarfield.isdense())
					{
						switch (fs.scalarfield.gettype())
						{
							case miINT8:
							case miUINT8:
								{
								SCIRun::LatVolField<char> *fieldptr = new SCIRun::LatVolField<char>(meshH,data_at);
								addscalardata3d(fieldptr,fs.scalarfield);
								scifield = static_cast<SCIRun::Field *>(fieldptr);
								}
								break;
							case miINT16:
								{
								SCIRun::LatVolField<signed short> *fieldptr = new SCIRun::LatVolField<signed short>(meshH,data_at);
								addscalardata3d(fieldptr,fs.scalarfield);
								scifield = static_cast<SCIRun::Field *>(fieldptr);					
								}
								break;
							case miUINT16:
								{
								SCIRun::LatVolField<unsigned short> *fieldptr = new SCIRun::LatVolField<unsigned short>(meshH,data_at);
								addscalardata3d(fieldptr,fs.scalarfield);
								scifield = static_cast<SCIRun::Field *>(fieldptr);
								}
								break;
							case miINT32:
								{
								SCIRun::LatVolField<signed long> *fieldptr = new SCIRun::LatVolField<signed long>(meshH,data_at);
								addscalardata3d(fieldptr,fs.scalarfield);
								scifield = static_cast<SCIRun::Field *>(fieldptr);
								}
								break;
							case miUINT32:
								{
								SCIRun::LatVolField<unsigned long> *fieldptr = new SCIRun::LatVolField<unsigned long>(meshH,data_at);
								addscalardata3d(fieldptr,fs.scalarfield);
								scifield = static_cast<SCIRun::Field *>(fieldptr);
								}
								break;
							case miSINGLE:
								{
								SCIRun::LatVolField<float> *fieldptr = new SCIRun::LatVolField<float>(meshH,data_at);
								addscalardata3d(fieldptr,fs.scalarfield);
								scifield = static_cast<SCIRun::Field *>(fieldptr);
								}
								break;
							case miDOUBLE:
							default:
								{
								SCIRun::LatVolField<double> *fieldptr = new SCIRun::LatVolField<double>(meshH,data_at);
								addscalardata3d(fieldptr,fs.scalarfield);
								scifield = static_cast<SCIRun::Field *>(fieldptr);				
								}
						}	

					}
					if (fs.vectorfield.isdense())
					{
						SCIRun::LatVolField<SCIRun::Vector> *fieldptr;
						fieldptr = new SCIRun::LatVolField<SCIRun::Vector>(meshH,data_at);
						addvectordata3d(fieldptr,fs.vectorfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
					}
					if (fs.tensorfield.isdense())
					{
						SCIRun::LatVolField<SCIRun::Tensor> *fieldptr;
						fieldptr = new SCIRun::LatVolField<SCIRun::Tensor>(meshH,data_at);
						addtensordata3d(fieldptr,fs.tensorfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
					}		
						
						
			}
			break;
			default:
				throw matlabconverter_error();
		
			
		}		
	}
				

	
	if ((fs.x.isdense())&&(fs.y.isdense())&&(fs.z.isdense()))
	{
		long numdim = fs.x.getnumdims();
	
		std::vector<long> dims;
		std::vector<unsigned int> mdims;
		dims = fs.x.getdims();
		
		mdims.resize(numdim);
		numnodes = 1;
		for (long p=0; p < numdim; p++) { numnodes *= dims[p]; mdims[p] = static_cast<unsigned int>(dims[p]); }
	
		if ((numdim == 2)&&((fs.x.getm() == 1)||(fs.x.getn() == 1)))
		{
			numdim = 1;
			if (fs.x.getm() == 1)
			{
				fs.x.transpose();
				fs.y.transpose();
				fs.z.transpose();
			}
			mdims.resize(1);
			mdims[0] = fs.x.getm();
		}
	
		switch (numdim)
		{
			case 1:
				{
					// Process a structured curve mesh
					SCIRun::StructCurveMeshHandle meshH;
					meshH = new SCIRun::StructCurveMesh;
					
					std::vector<double> X;
					std::vector<double> Y;
					std::vector<double> Z;
					fs.x.getnumericarray(X);
					fs.y.getnumericarray(Y);
					fs.z.getnumericarray(Z);
					
					meshH->set_dim(mdims);
					long p;
					for (p = 0; p < numnodes; p++)
					{
						meshH->set_point(SCIRun::Point(X[p],Y[p],Z[p]),static_cast<SCIRun::StructCurveMesh::Node::index_type>(p));
					}
					
					
					if ((fs.scalarfield.isempty())&&(fs.vectorfield.isempty())&&(fs.tensorfield.isempty()))
					{
						SCIRun::StructCurveField<double> *fieldptr;
						fieldptr = new SCIRun::StructCurveField<double>(meshH,data_at);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
					}
					if (fs.scalarfield.isdense())
					{
	
						switch (fs.scalarfield.gettype())
						{
							case miINT8:
							case miUINT8:
								{
								SCIRun::StructCurveField<char> *fieldptr = new SCIRun::StructCurveField<char>(meshH,data_at);
								addscalardata(fieldptr,fs.scalarfield);
								scifield = static_cast<SCIRun::Field *>(fieldptr);
								}
								break;
							case miINT16:
								{
								SCIRun::StructCurveField<signed short> *fieldptr = new SCIRun::StructCurveField<signed short>(meshH,data_at);
								addscalardata(fieldptr,fs.scalarfield);
								scifield = static_cast<SCIRun::Field *>(fieldptr);					
								}
								break;
							case miUINT16:
								{
								SCIRun::StructCurveField<unsigned short> *fieldptr = new SCIRun::StructCurveField<unsigned short>(meshH,data_at);
								addscalardata(fieldptr,fs.scalarfield);
								scifield = static_cast<SCIRun::Field *>(fieldptr);
								}
								break;
							case miINT32:
								{
								SCIRun::StructCurveField<signed long> *fieldptr = new SCIRun::StructCurveField<signed long>(meshH,data_at);
								addscalardata(fieldptr,fs.scalarfield);
								scifield = static_cast<SCIRun::Field *>(fieldptr);
								}
								break;
							case miUINT32:
								{
								SCIRun::StructCurveField<unsigned long> *fieldptr = new SCIRun::StructCurveField<unsigned long>(meshH,data_at);
								addscalardata(fieldptr,fs.scalarfield);
								scifield = static_cast<SCIRun::Field *>(fieldptr);
								}
								break;
							case miSINGLE:
								{
								SCIRun::StructCurveField<float> *fieldptr = new SCIRun::StructCurveField<float>(meshH,data_at);
								addscalardata(fieldptr,fs.scalarfield);
								scifield = static_cast<SCIRun::Field *>(fieldptr);
								}
								break;
							case miDOUBLE:
							default:
								{
								SCIRun::StructCurveField<double> *fieldptr = new SCIRun::StructCurveField<double>(meshH,data_at);
								addscalardata(fieldptr,fs.scalarfield);
								scifield = static_cast<SCIRun::Field *>(fieldptr);				
								}
						}	
	
					}
					if (fs.vectorfield.isdense())
					{
						SCIRun::StructCurveField<SCIRun::Vector> *fieldptr;
						fieldptr = new SCIRun::StructCurveField<SCIRun::Vector>(meshH,data_at);
						addvectordata(fieldptr,fs.vectorfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
					}
					if (fs.tensorfield.isdense())
					{
						SCIRun::StructCurveField<SCIRun::Tensor> *fieldptr;
						fieldptr = new SCIRun::StructCurveField<SCIRun::Tensor>(meshH,data_at);
						addtensordata(fieldptr,fs.tensorfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
					}		
				}
				break;
			case 2:
				{
					// Process a structured quadsurf mesh
					SCIRun::StructQuadSurfMeshHandle meshH;
					meshH = new SCIRun::StructQuadSurfMesh;
					
					std::vector<double> X;
					std::vector<double> Y;
					std::vector<double> Z;
					fs.x.getnumericarray(X);
					fs.y.getnumericarray(Y);
					fs.z.getnumericarray(Z);
					
					meshH->set_dim(mdims);
					unsigned p,r,q;
					q = 0;
					for (r = 0; r < mdims[1]; r++)
					for (p = 0; p < mdims[0]; p++)
					{
						meshH->set_point(SCIRun::Point(X[q],Y[q],Z[q]),SCIRun::StructQuadSurfMesh::Node::index_type(static_cast<SCIRun::ImageMesh *>(meshH.get_rep()),p,r));
						q++;
					}
					
					if ((fs.scalarfield.isempty())&&(fs.vectorfield.isempty())&&(fs.tensorfield.isempty()))
					{
						SCIRun::StructQuadSurfField<double> *fieldptr;
						fieldptr = new SCIRun::StructQuadSurfField<double>(meshH,data_at);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
					}
					if (fs.scalarfield.isdense())
					{
			
						switch (fs.scalarfield.gettype())
						{
							case miINT8:
							case miUINT8:
								{
								SCIRun::StructQuadSurfField<char> *fieldptr = new SCIRun::StructQuadSurfField<char>(meshH,data_at);
								addscalardata2d(fieldptr,fs.scalarfield);
								scifield = static_cast<SCIRun::Field *>(fieldptr);
								}
								break;
							case miINT16:
								{
								SCIRun::StructQuadSurfField<signed short> *fieldptr = new SCIRun::StructQuadSurfField<signed short>(meshH,data_at);
								addscalardata2d(fieldptr,fs.scalarfield);
								scifield = static_cast<SCIRun::Field *>(fieldptr);					
								}
								break;
							case miUINT16:
								{
								SCIRun::StructQuadSurfField<unsigned short> *fieldptr = new SCIRun::StructQuadSurfField<unsigned short>(meshH,data_at);
								addscalardata2d(fieldptr,fs.scalarfield);
								scifield = static_cast<SCIRun::Field *>(fieldptr);
								}
								break;
							case miINT32:
								{
								SCIRun::StructQuadSurfField<signed long> *fieldptr = new SCIRun::StructQuadSurfField<signed long>(meshH,data_at);
								addscalardata2d(fieldptr,fs.scalarfield);
								scifield = static_cast<SCIRun::Field *>(fieldptr);
								}
								break;
							case miUINT32:
								{
								SCIRun::StructQuadSurfField<unsigned long> *fieldptr = new SCIRun::StructQuadSurfField<unsigned long>(meshH,data_at);
								addscalardata2d(fieldptr,fs.scalarfield);
								scifield = static_cast<SCIRun::Field *>(fieldptr);
								}
								break;
							case miSINGLE:
								{
								SCIRun::StructQuadSurfField<float> *fieldptr = new SCIRun::StructQuadSurfField<float>(meshH,data_at);
								addscalardata2d(fieldptr,fs.scalarfield);
								scifield = static_cast<SCIRun::Field *>(fieldptr);
								}
								break;
							case miDOUBLE:
							default:
								{
								SCIRun::StructQuadSurfField<double> *fieldptr = new SCIRun::StructQuadSurfField<double>(meshH,data_at);
								addscalardata2d(fieldptr,fs.scalarfield);
								scifield = static_cast<SCIRun::Field *>(fieldptr);				
								}
							}				
					}
					if (fs.vectorfield.isdense())
					{
						SCIRun::StructQuadSurfField<SCIRun::Vector> *fieldptr;
						fieldptr = new SCIRun::StructQuadSurfField<SCIRun::Vector>(meshH,data_at);
						addvectordata2d(fieldptr,fs.vectorfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
					}
					if (fs.tensorfield.isdense())
					{
						SCIRun::StructQuadSurfField<SCIRun::Tensor> *fieldptr;
						fieldptr = new SCIRun::StructQuadSurfField<SCIRun::Tensor>(meshH,data_at);
						addtensordata2d(fieldptr,fs.tensorfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
					}		
				}			
				break;
			case 3:
				{
					// Process a structured quadsurf mesh
					SCIRun::StructHexVolMeshHandle meshH;
					meshH = new SCIRun::StructHexVolMesh;
					
					std::vector<double> X;
					std::vector<double> Y;
					std::vector<double> Z;
					fs.x.getnumericarray(X);
					fs.y.getnumericarray(Y);
					fs.z.getnumericarray(Z);
					
					meshH->set_dim(mdims);
					unsigned p,r,s,q;
					q= 0;
					for (s = 0; s < mdims[2]; s++)
					for (r = 0; r < mdims[1]; r++)
					for (p = 0; p < mdims[0]; p++)
					{
						meshH->set_point(SCIRun::Point(X[q],Y[q],Z[q]),SCIRun::StructHexVolMesh::Node::index_type(static_cast<SCIRun::LatVolMesh *>(meshH.get_rep()),p,r,s));
						q++;
					}

					if ((fs.scalarfield.isempty())&&(fs.vectorfield.isempty())&&(fs.tensorfield.isempty()))
					{
						SCIRun::StructHexVolField<double> *fieldptr;
						fieldptr = new SCIRun::StructHexVolField<double>(meshH,data_at);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
					}
					if (fs.scalarfield.isdense())
					{
					switch (fs.scalarfield.gettype())
					{
						case miINT8:
						case miUINT8:
							{
							SCIRun::StructHexVolField<char> *fieldptr = new SCIRun::StructHexVolField<char>(meshH,data_at);
							addscalardata3d(fieldptr,fs.scalarfield);
							scifield = static_cast<SCIRun::Field *>(fieldptr);
							}
							break;
						case miINT16:
							{
							SCIRun::StructHexVolField<signed short> *fieldptr = new SCIRun::StructHexVolField<signed short>(meshH,data_at);
							addscalardata3d(fieldptr,fs.scalarfield);
							scifield = static_cast<SCIRun::Field *>(fieldptr);					
							}
							break;
						case miUINT16:
							{
							SCIRun::StructHexVolField<unsigned short> *fieldptr = new SCIRun::StructHexVolField<unsigned short>(meshH,data_at);
							addscalardata3d(fieldptr,fs.scalarfield);
							scifield = static_cast<SCIRun::Field *>(fieldptr);
							}
							break;
						case miINT32:
							{
							SCIRun::StructHexVolField<signed long> *fieldptr = new SCIRun::StructHexVolField<signed long>(meshH,data_at);
							addscalardata3d(fieldptr,fs.scalarfield);
							scifield = static_cast<SCIRun::Field *>(fieldptr);
							}
							break;
						case miUINT32:
							{
							SCIRun::StructHexVolField<unsigned long> *fieldptr = new SCIRun::StructHexVolField<unsigned long>(meshH,data_at);
							addscalardata3d(fieldptr,fs.scalarfield);
							scifield = static_cast<SCIRun::Field *>(fieldptr);
							}
							break;
						case miSINGLE:
							{
							SCIRun::StructHexVolField<float> *fieldptr = new SCIRun::StructHexVolField<float>(meshH,data_at);
							addscalardata3d(fieldptr,fs.scalarfield);
							scifield = static_cast<SCIRun::Field *>(fieldptr);
							}
							break;
						case miDOUBLE:
						default:
							{
							SCIRun::StructHexVolField<double> *fieldptr = new SCIRun::StructHexVolField<double>(meshH,data_at);
							addscalardata3d(fieldptr,fs.scalarfield);
							scifield = static_cast<SCIRun::Field *>(fieldptr);				
							}
						}	

					}
					if (fs.vectorfield.isdense())
					{
						SCIRun::StructHexVolField<SCIRun::Vector> *fieldptr;
						fieldptr = new SCIRun::StructHexVolField<SCIRun::Vector>(meshH,data_at);
						addvectordata3d(fieldptr,fs.vectorfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
					}
					if (fs.tensorfield.isdense())
					{
						SCIRun::StructHexVolField<SCIRun::Tensor> *fieldptr;
						fieldptr = new SCIRun::StructHexVolField<SCIRun::Tensor>(meshH,data_at);
						addtensordata3d(fieldptr,fs.tensorfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
					}		
				}			
				break;
		}	
	}
		
		
	if ((fs.edge.isempty())&&(fs.face.isempty())&&(fs.cell.isempty())&&(fs.x.isempty())&&(fs.dims.isempty()))
	{
		// These is no connectivity data => it must be a pointcloud ;)
		// Supported mesh/field types here:
		// PointCloudField
		
		// Generate the mesh and add the nodes
		SCIRun::PointCloudMeshHandle meshH;
		meshH = new SCIRun::PointCloudMesh;
		addnodes(meshH,fs.node);
		
		// Depending on the type generate a new 
		// SCIRun Field object
		
		
		if ((fs.scalarfield.isempty())&&(fs.vectorfield.isempty())&&(fs.tensorfield.isempty()))
		{
			SCIRun::PointCloudField<double> *fieldptr;
			fieldptr = new SCIRun::PointCloudField<double>(meshH,data_at);
			scifield = static_cast<SCIRun::Field *>(fieldptr);
		}
		if (fs.scalarfield.isdense())
		{
				switch (fs.scalarfield.gettype())
				{
					case miINT8:
					case miUINT8:
						{
						SCIRun::PointCloudField<char> *fieldptr = new SCIRun::PointCloudField<char>(meshH,data_at);
						addscalardata(fieldptr,fs.scalarfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
						}
						break;
					case miINT16:
						{
						SCIRun::PointCloudField<signed short> *fieldptr = new SCIRun::PointCloudField<signed short>(meshH,data_at);
						addscalardata(fieldptr,fs.scalarfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);					
						}
						break;
					case miUINT16:
						{
						SCIRun::PointCloudField<unsigned short> *fieldptr = new SCIRun::PointCloudField<unsigned short>(meshH,data_at);
						addscalardata(fieldptr,fs.scalarfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
						}
						break;
					case miINT32:
						{
						SCIRun::PointCloudField<signed long> *fieldptr = new SCIRun::PointCloudField<signed long>(meshH,data_at);
						addscalardata(fieldptr,fs.scalarfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
						}
						break;
					case miUINT32:
						{
						SCIRun::PointCloudField<unsigned long> *fieldptr = new SCIRun::PointCloudField<unsigned long>(meshH,data_at);
						addscalardata(fieldptr,fs.scalarfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
						}
						break;
					case miSINGLE:
						{
						SCIRun::PointCloudField<float> *fieldptr = new SCIRun::PointCloudField<float>(meshH,data_at);
						addscalardata(fieldptr,fs.scalarfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
						}
						break;
					case miDOUBLE:
					default:
						{
						SCIRun::PointCloudField<double> *fieldptr = new SCIRun::PointCloudField<double>(meshH,data_at);
						addscalardata(fieldptr,fs.scalarfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);				
						}
				}	
			
		}
		if (fs.vectorfield.isdense())
		{
			SCIRun::PointCloudField<SCIRun::Vector> *fieldptr;
			fieldptr = new SCIRun::PointCloudField<SCIRun::Vector>(meshH,data_at);
			addvectordata(fieldptr,fs.vectorfield);
			scifield = static_cast<SCIRun::Field *>(fieldptr);
		}
		if (fs.tensorfield.isdense())
		{
			SCIRun::PointCloudField<SCIRun::Tensor> *fieldptr;
			fieldptr = new SCIRun::PointCloudField<SCIRun::Tensor>(meshH,data_at);
			addtensordata(fieldptr,fs.tensorfield);
			scifield = static_cast<SCIRun::Field *>(fieldptr);
		}		
	}
	
	
	if ((fs.edge.isdense())&&(fs.node.isdense()))
	{
		// This object must be a curvefield object
		
		m = fs.edge.getm();
		
		if (m == 2)
		{
			SCIRun::CurveMeshHandle meshH;
			meshH = new SCIRun::CurveMesh;
			addnodes(meshH,fs.node);
			addedges(meshH,fs.edge);
		
			if ((fs.scalarfield.isempty())&&(fs.vectorfield.isempty())&&(fs.tensorfield.isempty()))
			{
				SCIRun::CurveField<double> *fieldptr;
				fieldptr = new SCIRun::CurveField<double>(meshH,data_at);
				scifield = static_cast<SCIRun::Field *>(fieldptr);
			}
		
			if (fs.scalarfield.isdense())
			{
				switch (fs.scalarfield.gettype())
				{
					case miINT8:
					case miUINT8:
						{
						SCIRun::CurveField<char> *fieldptr = new SCIRun::CurveField<char>(meshH,data_at);
						addscalardata(fieldptr,fs.scalarfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
						}
						break;
					case miINT16:
						{
						SCIRun::CurveField<signed short> *fieldptr = new SCIRun::CurveField<signed short>(meshH,data_at);
						addscalardata(fieldptr,fs.scalarfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);					
						}
						break;
					case miUINT16:
						{
						SCIRun::CurveField<unsigned short> *fieldptr = new SCIRun::CurveField<unsigned short>(meshH,data_at);
						addscalardata(fieldptr,fs.scalarfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
						}
						break;
					case miINT32:
						{
						SCIRun::CurveField<signed long> *fieldptr = new SCIRun::CurveField<signed long>(meshH,data_at);
						addscalardata(fieldptr,fs.scalarfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
						}
						break;
					case miUINT32:
						{
						SCIRun::CurveField<unsigned long> *fieldptr = new SCIRun::CurveField<unsigned long>(meshH,data_at);
						addscalardata(fieldptr,fs.scalarfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
						}
						break;
					case miSINGLE:
						{
						SCIRun::CurveField<float> *fieldptr = new SCIRun::CurveField<float>(meshH,data_at);
						addscalardata(fieldptr,fs.scalarfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
						}
						break;
					case miDOUBLE:
					default:
						{
						SCIRun::CurveField<double> *fieldptr = new SCIRun::CurveField<double>(meshH,data_at);
						addscalardata(fieldptr,fs.scalarfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);				
						}
				}	
			}
			if (fs.vectorfield.isdense())
			{
				SCIRun::CurveField<SCIRun::Vector> *fieldptr;
				fieldptr = new SCIRun::CurveField<SCIRun::Vector>(meshH,data_at);
				addvectordata(fieldptr,fs.vectorfield);
				scifield = static_cast<SCIRun::Field *>(fieldptr);
			}
			if (fs.tensorfield.isdense())
			{
				SCIRun::CurveField<SCIRun::Tensor> *fieldptr;
				fieldptr = new SCIRun::CurveField<SCIRun::Tensor>(meshH,data_at);
				addtensordata(fieldptr,fs.tensorfield);
				scifield = static_cast<SCIRun::Field *>(fieldptr);
			}
		}
	}

	if ((fs.face.isdense())&&(fs.node.isdense()))
	{
		// This object must be a curvefield object
		
		m = fs.face.getm();
		if (m == 3)
		{
			SCIRun::TriSurfMeshHandle meshH;
			meshH = new SCIRun::TriSurfMesh;
			addnodes(meshH,fs.node);
			addfaces(meshH,fs.face);
		
			if ((fs.scalarfield.isempty())&&(fs.vectorfield.isempty())&&(fs.tensorfield.isempty()))
			{
				SCIRun::TriSurfField<double> *fieldptr;
				fieldptr = new SCIRun::TriSurfField<double>(meshH,data_at);
				scifield = static_cast<SCIRun::Field *>(fieldptr);
			}
			if (fs.scalarfield.isdense())
			{
				switch (fs.scalarfield.gettype())
				{
					case miINT8:
					case miUINT8:
						{
						SCIRun::TriSurfField<char> *fieldptr = new SCIRun::TriSurfField<char>(meshH,data_at);
						addscalardata(fieldptr,fs.scalarfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
						}
						break;
					case miINT16:
						{
						SCIRun::TriSurfField<signed short> *fieldptr = new SCIRun::TriSurfField<signed short>(meshH,data_at);
						addscalardata(fieldptr,fs.scalarfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);					
						}
						break;
					case miUINT16:
						{
						SCIRun::TriSurfField<unsigned short> *fieldptr = new SCIRun::TriSurfField<unsigned short>(meshH,data_at);
						addscalardata(fieldptr,fs.scalarfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
						}
						break;
					case miINT32:
						{
						SCIRun::TriSurfField<signed long> *fieldptr = new SCIRun::TriSurfField<signed long>(meshH,data_at);
						addscalardata(fieldptr,fs.scalarfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
						}
						break;
					case miUINT32:
						{
						SCIRun::TriSurfField<unsigned long> *fieldptr = new SCIRun::TriSurfField<unsigned long>(meshH,data_at);
						addscalardata(fieldptr,fs.scalarfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
						}
						break;
					case miSINGLE:
						{
						SCIRun::TriSurfField<float> *fieldptr = new SCIRun::TriSurfField<float>(meshH,data_at);
						addscalardata(fieldptr,fs.scalarfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
						}
						break;
					case miDOUBLE:
					default:
						{
						SCIRun::TriSurfField<double> *fieldptr = new SCIRun::TriSurfField<double>(meshH,data_at);
						addscalardata(fieldptr,fs.scalarfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);				
						}
				}	

			}
			if (fs.vectorfield.isdense())
			{
				SCIRun::TriSurfField<SCIRun::Vector> *fieldptr;
				fieldptr = new SCIRun::TriSurfField<SCIRun::Vector>(meshH,data_at);
				addvectordata(fieldptr,fs.vectorfield);
				scifield = static_cast<SCIRun::Field *>(fieldptr);
			}
			if (fs.tensorfield.isdense())
			{
				SCIRun::TriSurfField<SCIRun::Tensor> *fieldptr;
				fieldptr = new SCIRun::TriSurfField<SCIRun::Tensor>(meshH,data_at);
				addtensordata(fieldptr,fs.tensorfield);
				scifield = static_cast<SCIRun::Field *>(fieldptr);
			}		
		}
		
		if (m == 4)
		{
			SCIRun::QuadSurfMeshHandle meshH;
			meshH = new SCIRun::QuadSurfMesh;
			addnodes(meshH,fs.node);
			addfaces(meshH,fs.face);
		
			if ((fs.scalarfield.isempty())&&(fs.vectorfield.isempty())&&(fs.tensorfield.isempty()))
			{
				SCIRun::QuadSurfField<double> *fieldptr;
				fieldptr = new SCIRun::QuadSurfField<double>(meshH,data_at);
				scifield = static_cast<SCIRun::Field *>(fieldptr);
			}
			if (fs.scalarfield.isdense())
			{
				switch (fs.scalarfield.gettype())
				{
					case miINT8:
					case miUINT8:
						{
						SCIRun::QuadSurfField<char> *fieldptr = new SCIRun::QuadSurfField<char>(meshH,data_at);
						addscalardata(fieldptr,fs.scalarfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
						}
						break;
					case miINT16:
						{
						SCIRun::QuadSurfField<signed short> *fieldptr = new SCIRun::QuadSurfField<signed short>(meshH,data_at);
						addscalardata(fieldptr,fs.scalarfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);					
						}
						break;
					case miUINT16:
						{
						SCIRun::QuadSurfField<unsigned short> *fieldptr = new SCIRun::QuadSurfField<unsigned short>(meshH,data_at);
						addscalardata(fieldptr,fs.scalarfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
						}
						break;
					case miINT32:
						{
						SCIRun::QuadSurfField<signed long> *fieldptr = new SCIRun::QuadSurfField<signed long>(meshH,data_at);
						addscalardata(fieldptr,fs.scalarfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
						}
						break;
					case miUINT32:
						{
						SCIRun::QuadSurfField<unsigned long> *fieldptr = new SCIRun::QuadSurfField<unsigned long>(meshH,data_at);
						addscalardata(fieldptr,fs.scalarfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
						}
						break;
					case miSINGLE:
						{
						SCIRun::QuadSurfField<float> *fieldptr = new SCIRun::QuadSurfField<float>(meshH,data_at);
						addscalardata(fieldptr,fs.scalarfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
						}
						break;
					case miDOUBLE:
					default:
						{
						SCIRun::QuadSurfField<double> *fieldptr = new SCIRun::QuadSurfField<double>(meshH,data_at);
						addscalardata(fieldptr,fs.scalarfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);				
						}
				}	
			}
			if (fs.vectorfield.isdense())
			{
				SCIRun::QuadSurfField<SCIRun::Vector> *fieldptr;
				fieldptr = new SCIRun::QuadSurfField<SCIRun::Vector>(meshH,data_at);
				addvectordata(fieldptr,fs.vectorfield);
				scifield = static_cast<SCIRun::Field *>(fieldptr);
			}
			if (fs.tensorfield.isdense())
			{
				SCIRun::QuadSurfField<SCIRun::Tensor> *fieldptr;
				fieldptr = new SCIRun::QuadSurfField<SCIRun::Tensor>(meshH,data_at);
				addtensordata(fieldptr,fs.tensorfield);
				scifield = static_cast<SCIRun::Field *>(fieldptr);
			}		
		}		
	}	

	if ((fs.cell.isdense())&(fs.node.isdense()))
	{
		// This object must be a curvefield object
		
		m = fs.cell.getm();
		if (m == 4)
		{
			SCIRun::TetVolMeshHandle meshH;
			meshH = new SCIRun::TetVolMesh;
			addnodes(meshH,fs.node);
			addcells(meshH,fs.cell);
		
			if ((fs.scalarfield.isempty())&&(fs.vectorfield.isempty())&&(fs.tensorfield.isempty()))
			{
				SCIRun::TetVolField<double> *fieldptr;
				fieldptr = new SCIRun::TetVolField<double>(meshH,data_at);
				scifield = static_cast<SCIRun::Field *>(fieldptr);
			}
			if (fs.scalarfield.isdense())
			{
				switch (fs.scalarfield.gettype())
				{
					case miINT8:
					case miUINT8:
						{
						SCIRun::TetVolField<char> *fieldptr = new SCIRun::TetVolField<char>(meshH,data_at);
						addscalardata(fieldptr,fs.scalarfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
						}
						break;
					case miINT16:
						{
						SCIRun::TetVolField<signed short> *fieldptr = new SCIRun::TetVolField<signed short>(meshH,data_at);
						addscalardata(fieldptr,fs.scalarfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);					
						}
						break;
					case miUINT16:
						{
						SCIRun::TetVolField<unsigned short> *fieldptr = new SCIRun::TetVolField<unsigned short>(meshH,data_at);
						addscalardata(fieldptr,fs.scalarfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
						}
						break;
					case miINT32:
						{
						SCIRun::TetVolField<signed long> *fieldptr = new SCIRun::TetVolField<signed long>(meshH,data_at);
						addscalardata(fieldptr,fs.scalarfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
						}
						break;
					case miUINT32:
						{
						SCIRun::TetVolField<unsigned long> *fieldptr = new SCIRun::TetVolField<unsigned long>(meshH,data_at);
						addscalardata(fieldptr,fs.scalarfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
						}
						break;
					case miSINGLE:
						{
						SCIRun::TetVolField<float> *fieldptr = new SCIRun::TetVolField<float>(meshH,data_at);
						addscalardata(fieldptr,fs.scalarfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
						}
						break;
					case miDOUBLE:
					default:
						{
						SCIRun::TetVolField<double> *fieldptr = new SCIRun::TetVolField<double>(meshH,data_at);
						addscalardata(fieldptr,fs.scalarfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);				
						}
				}	
				
			}
			if (fs.vectorfield.isdense())
			{
				SCIRun::TetVolField<SCIRun::Vector> *fieldptr;
				fieldptr = new SCIRun::TetVolField<SCIRun::Vector>(meshH,data_at);
				addvectordata(fieldptr,fs.vectorfield);
				scifield = static_cast<SCIRun::Field *>(fieldptr);
			}
			if (fs.tensorfield.isdense())
			{
				SCIRun::TetVolField<SCIRun::Tensor> *fieldptr;
				fieldptr = new SCIRun::TetVolField<SCIRun::Tensor>(meshH,data_at);
				addtensordata(fieldptr,fs.tensorfield);
				scifield = static_cast<SCIRun::Field *>(fieldptr);
			}		
		}
		
		if (m == 6)
		{
			SCIRun::PrismVolMeshHandle meshH;
			meshH = new SCIRun::PrismVolMesh;
			addnodes(meshH,fs.node);
			addcells(meshH,fs.cell);
		
			if ((fs.scalarfield.isempty())&&(fs.vectorfield.isempty())&&(fs.tensorfield.isempty()))
			{
				SCIRun::PrismVolField<double> *fieldptr;
				fieldptr = new SCIRun::PrismVolField<double>(meshH,data_at);
				scifield = static_cast<SCIRun::Field *>(fieldptr);
			}
			if (fs.scalarfield.isdense())
			{
				switch (fs.scalarfield.gettype())
				{
					case miINT8:
					case miUINT8:
						{
						SCIRun::PrismVolField<char> *fieldptr = new SCIRun::PrismVolField<char>(meshH,data_at);
						addscalardata(fieldptr,fs.scalarfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
						}
						break;
					case miINT16:
						{
						SCIRun::PrismVolField<signed short> *fieldptr = new SCIRun::PrismVolField<signed short>(meshH,data_at);
						addscalardata(fieldptr,fs.scalarfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);					
						}
						break;
					case miUINT16:
						{
						SCIRun::PrismVolField<unsigned short> *fieldptr = new SCIRun::PrismVolField<unsigned short>(meshH,data_at);
						addscalardata(fieldptr,fs.scalarfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
						}
						break;
					case miINT32:
						{
						SCIRun::PrismVolField<signed long> *fieldptr = new SCIRun::PrismVolField<signed long>(meshH,data_at);
						addscalardata(fieldptr,fs.scalarfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
						}
						break;
					case miUINT32:
						{
						SCIRun::PrismVolField<unsigned long> *fieldptr = new SCIRun::PrismVolField<unsigned long>(meshH,data_at);
						addscalardata(fieldptr,fs.scalarfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
						}
						break;
					case miSINGLE:
						{
						SCIRun::PrismVolField<float> *fieldptr = new SCIRun::PrismVolField<float>(meshH,data_at);
						addscalardata(fieldptr,fs.scalarfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
						}
						break;
					case miDOUBLE:
					default:
						{
						SCIRun::PrismVolField<double> *fieldptr = new SCIRun::PrismVolField<double>(meshH,data_at);
						addscalardata(fieldptr,fs.scalarfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);				
						}
				}	
				
			}
			if (fs.vectorfield.isdense())
			{
				SCIRun::PrismVolField<SCIRun::Vector> *fieldptr;
				fieldptr = new SCIRun::PrismVolField<SCIRun::Vector>(meshH,data_at);
				addvectordata(fieldptr,fs.vectorfield);
				scifield = static_cast<SCIRun::Field *>(fieldptr);
			}
			if (fs.tensorfield.isdense())
			{
				SCIRun::PrismVolField<SCIRun::Tensor> *fieldptr;
				fieldptr = new SCIRun::PrismVolField<SCIRun::Tensor>(meshH,data_at);
				addtensordata(fieldptr,fs.tensorfield);
				scifield = static_cast<SCIRun::Field *>(fieldptr);
			}		
		}
		
		if (m == 8)
		{
			SCIRun::HexVolMeshHandle meshH;
			meshH = new SCIRun::HexVolMesh;
			addnodes(meshH,fs.node);
			addcells(meshH,fs.cell);
		
			if ((fs.scalarfield.isempty())&&(fs.vectorfield.isempty())&&(fs.tensorfield.isempty()))
			{
				SCIRun::HexVolField<double> *fieldptr;
				fieldptr = new SCIRun::HexVolField<double>(meshH,data_at);
				scifield = static_cast<SCIRun::Field *>(fieldptr);
			}
			if (fs.scalarfield.isdense())
			{
				switch (fs.scalarfield.gettype())
				{
					case miINT8:
					case miUINT8:
						{
						SCIRun::HexVolField<char> *fieldptr = new SCIRun::HexVolField<char>(meshH,data_at);
						addscalardata(fieldptr,fs.scalarfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
						}
						break;
					case miINT16:
						{
						SCIRun::HexVolField<signed short> *fieldptr = new SCIRun::HexVolField<signed short>(meshH,data_at);
						addscalardata(fieldptr,fs.scalarfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);					
						}
						break;
					case miUINT16:
						{
						SCIRun::HexVolField<unsigned short> *fieldptr = new SCIRun::HexVolField<unsigned short>(meshH,data_at);
						addscalardata(fieldptr,fs.scalarfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
						}
						break;
					case miINT32:
						{
						SCIRun::HexVolField<signed long> *fieldptr = new SCIRun::HexVolField<signed long>(meshH,data_at);
						addscalardata(fieldptr,fs.scalarfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
						}
						break;
					case miUINT32:
						{
						SCIRun::HexVolField<unsigned long> *fieldptr = new SCIRun::HexVolField<unsigned long>(meshH,data_at);
						addscalardata(fieldptr,fs.scalarfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
						}
						break;
					case miSINGLE:
						{
						SCIRun::HexVolField<float> *fieldptr = new SCIRun::HexVolField<float>(meshH,data_at);
						addscalardata(fieldptr,fs.scalarfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
						}
						break;
					case miDOUBLE:
					default:
						{
						SCIRun::HexVolField<double> *fieldptr = new SCIRun::HexVolField<double>(meshH,data_at);
						addscalardata(fieldptr,fs.scalarfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);				
						}
				}	
				
			}
			if (fs.vectorfield.isdense())
			{
				SCIRun::HexVolField<SCIRun::Vector> *fieldptr;
				fieldptr = new SCIRun::HexVolField<SCIRun::Vector>(meshH,data_at);
				addvectordata(fieldptr,fs.vectorfield);
				scifield = static_cast<SCIRun::Field *>(fieldptr);
			}
			if (fs.tensorfield.isdense())
			{
				SCIRun::HexVolField<SCIRun::Tensor> *fieldptr;
				fieldptr = new SCIRun::HexVolField<SCIRun::Tensor>(meshH,data_at);
				addtensordata(fieldptr,fs.tensorfield);
				scifield = static_cast<SCIRun::Field *>(fieldptr);
			}		
		}
		
	}

	if (fs.property.isstruct())
	{
		mlPropertyTOsciProperty(mlarray,static_cast<SCIRun::PropertyManager *>(scifield.get_rep()));
	}
	
	if (fs.name.isstring())
	{
		if (scifield != 0)
		{		
			if (fs.name.isstring())
			{
				scifield->set_property("name",fs.name.getstring(),false);
			}
		}
	}
	else
	{
		if (scifield != 0)
		{
			scifield->set_property("name",mlarray.getname(),false);
		}
	}
	
}


// Templates for adding data in the field


template <class FIELD> 
void matlabconverter::addscalardata(FIELD *fieldptr,matlabarray mlarray)
{

	fieldptr->resize_fdata();   // make sure it is resized to number of nodes/edges/faces/cells or whatever
	mlarray.getnumericarray(fieldptr->fdata());
}


template <class FIELD> 
void matlabconverter::addscalardata2d(FIELD *fieldptr,matlabarray mlarray)
{
	fieldptr->resize_fdata();   // make sure it is resized to number of nodes/edges/faces/cells or whatever
	typename FIELD::fdata_type& fdata = fieldptr->fdata();
	mlarray.getnumericarray(fdata.get_dataptr(),fdata.dim2(),fdata.dim1());
}


template <class FIELD> 
void matlabconverter::addscalardata3d(FIELD *fieldptr,matlabarray mlarray)
{
	fieldptr->resize_fdata();   // make sure it is resized to number of nodes/edges/faces/cells or whatever
	typename FIELD::fdata_type& fdata = fieldptr->fdata();
	mlarray.getnumericarray(fdata.get_dataptr(),fdata.dim3(),fdata.dim2(),fdata.dim1());
}


template <class FIELDPTR> 
void matlabconverter::addvectordata(FIELDPTR fieldptr,matlabarray mlarray)
{
	std::vector<double> fielddata;
	mlarray.getnumericarray(fielddata); // cast and copy the real part of the data

	fieldptr->resize_fdata();   // make sure it is resized to number of nodes/edges/faces/cells or whatever
	std::vector<SCIRun::Vector>& fdata = fieldptr->fdata();  // get a reference to the actual data
	
	long numdata = fielddata.size();
	if (numdata > (3*fdata.size())) numdata = (3*fdata.size()); // make sure we do not copy more data than there are elements
	
	long p,q;
	for (p = 0, q = 0; p < numdata; p+=3) 
	{ 
		fdata[q++] = SCIRun::Vector(fielddata[p],fielddata[p+1],fielddata[p+2]); 
	}
}

template <class FIELDPTR> 
void matlabconverter::addvectordata2d(FIELDPTR fieldptr,matlabarray mlarray)
{
	std::vector<double> fielddata;
	mlarray.getnumericarray(fielddata); // cast and copy the real part of the data

	fieldptr->resize_fdata();   // make sure it is resized to number of nodes/edges/faces/cells or whatever
	SCIRun::FData2d<SCIRun::Vector>& fdata = fieldptr->fdata();  // get a reference to the actual data
	
	long numdata = fielddata.size();
	if (numdata > (3*fdata.size())) numdata = (3*fdata.size()); // make sure we do not copy more data than there are elements
	
	
	SCIRun::Vector **data;
	int dim1,dim2;
	
	data = fdata.get_dataptr();
	dim1 = fdata.dim1();
	dim2 = fdata.dim2();
	
	long q,r,p;
	p = 0;
	for (q=0;(q<dim1)&&(p < numdata);q++)
		for (r=0;(r<dim2)&&(p < numdata);r++)
		{
			data[q][r] = SCIRun::Vector(fielddata[p],fielddata[p+1],fielddata[p+2]); 
			p+=3;
		}
	
}


template <class FIELDPTR> 
void matlabconverter::addvectordata3d(FIELDPTR fieldptr,matlabarray mlarray)
{
	std::vector<double> fielddata;
	mlarray.getnumericarray(fielddata); // cast and copy the real part of the data

	fieldptr->resize_fdata();   // make sure it is resized to number of nodes/edges/faces/cells or whatever
	SCIRun::FData3d<SCIRun::Vector>& fdata = fieldptr->fdata();  // get a reference to the actual data
	
	long numdata = fielddata.size();
	if (numdata > (3*fdata.size())) numdata = (3*fdata.size()); // make sure we do not copy more data than there are elements
	
	SCIRun::Vector ***data;
	int dim1,dim2,dim3;
	
	data = fdata.get_dataptr();
	dim1 = fdata.dim1();
	dim2 = fdata.dim2();
	dim3 = fdata.dim3();
	
	long q,r,s,p;
	p = 0;
	for (q=0;(q<dim1)&&(p < numdata);q++)
		for (r=0;(r<dim2)&&(p < numdata);r++)
			for (s=0;(s<dim3)&&(p <numdata);s++)
			{
				data[q][r][s] = SCIRun::Vector(fielddata[p],fielddata[p+1],fielddata[p+2]); 
				p+=3;
			}
	
}

template <class FIELDPTR> 
void matlabconverter::addtensordata(FIELDPTR fieldptr,matlabarray mlarray)
{
	std::vector<double> fielddata;
	mlarray.getnumericarray(fielddata); // cast and copy the real part of the data

	fieldptr->resize_fdata();   // make sure it is resized to number of nodes/edges/faces/cells or whatever
	std::vector<SCIRun::Tensor>& fdata = fieldptr->fdata();  // get a reference to the actual data
	
	long numdata = fielddata.size();
	if (mlarray.getm() == 6)
	{ // Compressed tensor data : xx,yy,zz,xy,xz,yz
		if (numdata > (6*fdata.size())) numdata = (6*fdata.size()); // make sure we do not copy more data than there are elements
	
		SCIRun::Tensor dummy;
		long p,q;
		for (p = 0, q = 0; p < numdata; p +=6) 
		{ 
			dummy.mat_[0][0] = fielddata[p];
			dummy.mat_[0][1] = fielddata[p+3];
			dummy.mat_[0][2] = fielddata[p+4];
			dummy.mat_[1][1] = fielddata[p+1];
			dummy.mat_[1][2] = fielddata[p+5];
			dummy.mat_[2][2] = fielddata[p+2];
			fdata[q++] = dummy; 
		}
	}
	else
	{  // UnCompressed tensor data : xx,xy,xz,yx,yy,yz,zx,zy,zz 
		if (numdata > (9*fdata.size())) numdata = (9*fdata.size()); // make sure we do not copy more data than there are elements
		SCIRun::Tensor dummy;
		long p,q;
		for (p = 0, q = 0; p < numdata;) 
		{ 
			dummy.mat_[0][0] = fielddata[p++];
			dummy.mat_[0][1] = fielddata[p++];
			dummy.mat_[0][2] = fielddata[p++];
			dummy.mat_[1][0] = fielddata[p++];
			dummy.mat_[1][1] = fielddata[p++];
			dummy.mat_[1][2] = fielddata[p++];
			dummy.mat_[2][0] = fielddata[p++];
			dummy.mat_[2][1] = fielddata[p++];
			dummy.mat_[2][2] = fielddata[p++];
			fdata[q++] = dummy; 
		}
	}
}



template <class FIELDPTR> 
void matlabconverter::addtensordata2d(FIELDPTR fieldptr,matlabarray mlarray)
{
	std::vector<double> fielddata;
	mlarray.getnumericarray(fielddata); // cast and copy the real part of the data

	fieldptr->resize_fdata();   // make sure it is resized to number of nodes/edges/faces/cells or whatever
	SCIRun::FData2d<SCIRun::Tensor>& fdata = fieldptr->fdata();  // get a reference to the actual data
	
	SCIRun::ImageMesh::Cell::iterator r(0);
	
	long numdata = fielddata.size();
	if (mlarray.getm() == 6)
	{ // Compressed tensor data : xx,yy,zz,xy,xz,yz
		if (numdata > (6*fdata.size())) numdata = (6*fdata.size()); // make sure we do not copy more data than there are elements

		SCIRun::Tensor dummy;

		SCIRun::Tensor **data;
		int dim1,dim2;
	
		data = fdata.get_dataptr();
		dim1 = fdata.dim1();
		dim2 = fdata.dim2();
				
		long q,r,p;
		p = 0;
		for (q=0;(q<dim1)&&(p < numdata);q++)
			for (r=0;(r<dim2)&&(p < numdata);r++)
			{
				dummy.mat_[0][0] = fielddata[p];
				dummy.mat_[0][1] = fielddata[p+3];
				dummy.mat_[0][2] = fielddata[p+4];
				dummy.mat_[1][1] = fielddata[p+1];
				dummy.mat_[1][2] = fielddata[p+5];
				dummy.mat_[2][2] = fielddata[p+2];
			
				data[q][r] = dummy;
				p+=6;
			}

	}
	else
	{  // UnCompressed tensor data : xx,xy,xz,yx,yy,yz,zx,zy,zz 
		if (numdata > (9*fdata.size())) numdata = (9*fdata.size()); // make sure we do not copy more data than there are elements
		SCIRun::Tensor dummy;
		
		SCIRun::Tensor **data;
		int dim1,dim2;
	
		data = fdata.get_dataptr();
		dim1 = fdata.dim1();
		dim2 = fdata.dim2();
				
		long q,r,p;
		p = 0;
		for (q=0;(q<dim1)&&(p < numdata);q++)
			for (r=0;(r<dim2)&&(p < numdata);r++)
			{	
				dummy.mat_[0][0] = fielddata[p++];
				dummy.mat_[0][1] = fielddata[p++];
				dummy.mat_[0][2] = fielddata[p++];
				dummy.mat_[1][0] = fielddata[p++];
				dummy.mat_[1][1] = fielddata[p++];
				dummy.mat_[1][2] = fielddata[p++];
				dummy.mat_[2][0] = fielddata[p++];
				dummy.mat_[2][1] = fielddata[p++];
				dummy.mat_[2][2] = fielddata[p++];
				data[q][r] = dummy; 
			}
	}
}


template <class FIELDPTR> 
void matlabconverter::addtensordata3d(FIELDPTR fieldptr,matlabarray mlarray)
{
	std::vector<double> fielddata;
	mlarray.getnumericarray(fielddata); // cast and copy the real part of the data

	fieldptr->resize_fdata();   // make sure it is resized to number of nodes/edges/faces/cells or whatever
	SCIRun::FData3d<SCIRun::Tensor>& fdata = fieldptr->fdata();  // get a reference to the actual data
	
	SCIRun::LatVolMesh::Cell::iterator r;
	
	long numdata = fielddata.size();
	if (mlarray.getm() == 6)
	{ // Compressed tensor data : xx,yy,zz,xy,xz,yz
		if (numdata > (6*fdata.size())) numdata = (6*fdata.size()); // make sure we do not copy more data than there are elements

		SCIRun::Tensor dummy;

		SCIRun::Tensor ***data;
		int dim1,dim2,dim3;
	
		data = fdata.get_dataptr();
		dim1 = fdata.dim1();
		dim2 = fdata.dim2();
		dim3 = fdata.dim3();
				
		long q,r,s,p;
		p = 0;
		for (q=0;(q<dim1)&&(p < numdata);q++)
			for (r=0;(r<dim2)&&(p < numdata);r++)
				for (s=0;(s<dim3)&&(p < numdata);s++)
				{
					dummy.mat_[0][0] = fielddata[p];
					dummy.mat_[0][1] = fielddata[p+3];
					dummy.mat_[0][2] = fielddata[p+4];
					dummy.mat_[1][1] = fielddata[p+1];
					dummy.mat_[1][2] = fielddata[p+5];
					dummy.mat_[2][2] = fielddata[p+2];
			
					data[q][r][s] = dummy;
					p+=6;
				}

	}
	else
	{  // UnCompressed tensor data : xx,xy,xz,yx,yy,yz,zx,zy,zz 
		if (numdata > (9*fdata.size())) numdata = (9*fdata.size()); // make sure we do not copy more data than there are elements
		SCIRun::Tensor dummy;
		
		SCIRun::Tensor ***data;
		int dim1,dim2,dim3;
	
		data = fdata.get_dataptr();
		dim1 = fdata.dim1();
		dim2 = fdata.dim2();
		dim3 = fdata.dim3();
				
		long q,r,s,p;
		p = 0;
		for (q=0;(q<dim1)&&(p < numdata);q++)
			for (r=0;(r<dim2)&&(p < numdata);r++)
				for (s=0; (s<dim3)&&(p <numdata); s++)
				{	
					dummy.mat_[0][0] = fielddata[p++];
					dummy.mat_[0][1] = fielddata[p++];
					dummy.mat_[0][2] = fielddata[p++];
					dummy.mat_[1][0] = fielddata[p++];
					dummy.mat_[1][1] = fielddata[p++];
					dummy.mat_[1][2] = fielddata[p++];
					dummy.mat_[2][0] = fielddata[p++];
					dummy.mat_[2][1] = fielddata[p++];
					dummy.mat_[2][2] = fielddata[p++];
					data[q][r][s] = dummy; 
				}
	}
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
	
	// FUNCTION TO BE ADDED
	meshH->node_reserve(numnodes);
	
	long p,q;
	for (p = 0, q = 0; p < numnodes; p++, q+=3)
	{ meshH->add_point(SCIRun::Point(mldata[q],mldata[q+1],mldata[q+2])); } 
	
	// Add these points to the mesh
}



template <class MESH>
void matlabconverter::addedges(SCIRun::LockingHandle<MESH> meshH,matlabarray mlarray)
{
	// Get the data from the matlab file, which has been buffered
	// but whose format can be anything. The next piece of code
	// copies and casts the data
	
	std::vector<typename MESH::under_type> mldata;
	mlarray.getnumericarray(mldata);		
	
	// check whether it is zero based indexing 
	// In short if there is a zero it must be zero
	// based numbering right ??
	// If not we assume one based numbering
	
	long p,q;
	
	bool zerobased = false;  
	long size = mldata.size();
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
	
	std::vector<typename MESH::under_type> mldata;
	mlarray.getnumericarray(mldata);		
	
	// check whether it is zero based indexing 
	// In short if there is a zero it must be zero
	// based numbering right ??
	// If not we assume one based numbering
	
	bool zerobased = false;  
	long size = mldata.size();
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
	
	std::vector<typename MESH::under_type> mldata;
	mlarray.getnumericarray(mldata);		
	
	// check whether it is zero based indexing 
	// In short if there is a zero it must be zero
	// based numbering right ??
	// If not we assume one based numbering
	
	bool zerobased = false;  
	long size = mldata.size();
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



matlabconverter::fieldstruct matlabconverter::analyzefieldstruct(matlabarray &ma)
{
	// define possible fieldnames
	// This function searches through the matlab structure and identifies which fields
	// can be used in the construction of a field.
	// The last name in each list is the recommended name. When multiple fields are 
	// defined which suppose to have the same contents, the last one is chosen, which
	// is the recommended name listed in the documentation.
	
	fieldstruct		fs;
	long			index;
	
	if (!ma.isstruct()) return(fs);
	
	// NODE MATRIX
	index = ma.getfieldnameindexCI("pts");
	if (index > -1) fs.node = ma.getfield(0,index);
	index = ma.getfieldnameindexCI("node");
	if (index > -1) fs.node = ma.getfield(0,index);

	// STRUCTURE MATRICES IN SUBMATRICES
	index = ma.getfieldnameindexCI("x");
	if (index > -1) fs.x = ma.getfield(0,index);
	index = ma.getfieldnameindexCI("y");
	if (index > -1) fs.y = ma.getfield(0,index);
	index = ma.getfieldnameindexCI("z");
	if (index > -1) fs.z = ma.getfield(0,index);

	// EDGE MATRIX
	index = ma.getfieldnameindexCI("line");
	if (index > -1) fs.edge = ma.getfield(0,index);
	index = ma.getfieldnameindexCI("edge");
	if (index > -1) fs.edge = ma.getfield(0,index);

	// FACE MATRIX
	index = ma.getfieldnameindexCI("fac");
	if (index > -1) fs.face = ma.getfield(0,index);
	index = ma.getfieldnameindexCI("quad");
	if (index > -1) fs.face = ma.getfield(0,index);
	index = ma.getfieldnameindexCI("face");
	if (index > -1) fs.face = ma.getfield(0,index);
	
	// CELL MATRIX
	index = ma.getfieldnameindexCI("tet");
	if (index > -1) fs.cell = ma.getfield(0,index);
	index = ma.getfieldnameindexCI("hex");
	if (index > -1) fs.cell = ma.getfield(0,index);
	index = ma.getfieldnameindexCI("prism");
	if (index > -1) fs.cell = ma.getfield(0,index);
	index = ma.getfieldnameindexCI("cell");
	if (index > -1) fs.cell = ma.getfield(0,index);
	
	// FIELDNODE MATRIX
	index = ma.getfieldnameindexCI("data");
	if (index > -1) fs.scalarfield = ma.getfield(0,index);
	index = ma.getfieldnameindexCI("potvals");
	if (index > -1) fs.scalarfield = ma.getfield(0,index);
	index = ma.getfieldnameindexCI("scalarfield");
	if (index > -1) fs.scalarfield = ma.getfield(0,index);
	index = ma.getfieldnameindexCI("scalardata");
	if (index > -1) fs.scalarfield = ma.getfield(0,index);
	index = ma.getfieldnameindexCI("field");
	if (index > -1) fs.scalarfield = ma.getfield(0,index);

	// VECTOR FIELD MATRIX
	index = ma.getfieldnameindexCI("vectordata");
	if (index > -1) fs.vectorfield = ma.getfield(0,index);
	index = ma.getfieldnameindexCI("vectorfield");
	if (index > -1) fs.vectorfield = ma.getfield(0,index);

	// TENSOR FIELD MATRIX
	index = ma.getfieldnameindexCI("tensordata");
	if (index > -1) fs.tensorfield = ma.getfield(0,index);
	index = ma.getfieldnameindexCI("tensorfield");
	if (index > -1) fs.tensorfield = ma.getfield(0,index);

	// FIELD LOCATION MATRIX
	index = ma.getfieldnameindexCI("dataat");
	if (index > -1) fs.fieldlocation = ma.getfield(0,index);
	index = ma.getfieldnameindexCI("fieldlocation");
	if (index > -1) fs.fieldlocation = ma.getfield(0,index);
	index = ma.getfieldnameindexCI("fieldat");
	if (index > -1) fs.fieldlocation = ma.getfield(0,index);

	// FIELD TYPE MATRIX
	index = ma.getfieldnameindexCI("datatype");
	if (index > -1) fs.fieldtype = ma.getfield(0,index);
	index = ma.getfieldnameindexCI("fieldtype");
	if (index > -1) fs.fieldtype = ma.getfield(0,index);

	// ELEMTYPE MATRIX
	index = ma.getfieldnameindexCI("elemtype");
	if (index > -1) fs.elemtype = ma.getfield(0,index);
	index = ma.getfieldnameindexCI("meshclass");
	if (index > -1) fs.elemtype = ma.getfield(0,index);

	// NAME OF THE MESH/FIELD
	index = ma.getfieldnameindexCI("name");
	if (index > -1) fs.name = ma.getfield(0,index);

	// PROPERTY FIELD
	index = ma.getfieldnameindexCI("property");
	if (index > -1) fs.property = ma.getfield(0,index);

	// REGULAR MATRICES
	index = ma.getfieldnameindexCI("dim");
	if (index > -1) fs.dims = ma.getfield(0,index);
	index = ma.getfieldnameindexCI("dims");
	if (index > -1) fs.dims = ma.getfield(0,index);
	
	index = ma.getfieldnameindexCI("translation");
	if (index > -1) fs.offset = ma.getfield(0,index);
	index = ma.getfieldnameindexCI("offset");
	if (index > -1) fs.offset = ma.getfield(0,index);
	index = ma.getfieldnameindexCI("scale");
	if (index > -1) fs.size = ma.getfield(0,index);
	index = ma.getfieldnameindexCI("size");
	if (index > -1) fs.size = ma.getfield(0,index);
	index = ma.getfieldnameindexCI("rotation");
	if (index > -1) fs.rotation = ma.getfield(0,index);
	index = ma.getfieldnameindexCI("transform");
	if (index > -1) fs.transform = ma.getfield(0,index);
	
	fs.data_at = SCIRun::Field::NONE;
	
	if (!(fs.fieldlocation.isempty()))
	{   // converter table for the string in the field "fieldlocation" array
		// These are case insensitive comparisons.
		if (!(fs.fieldlocation.isstring())) throw matlabconverter_error();
		if (fs.fieldlocation.compareCI("node")||fs.fieldlocation.compareCI("pts")) fs.data_at = SCIRun::Field::NODE;
		if (fs.fieldlocation.compareCI("egde")||fs.fieldlocation.compareCI("line")) fs.data_at = SCIRun::Field::EDGE;
		if (fs.fieldlocation.compareCI("face")||fs.fieldlocation.compareCI("fac")) fs.data_at = SCIRun::Field::FACE;
		if (fs.fieldlocation.compareCI("cell")||fs.fieldlocation.compareCI("tet")
			||fs.fieldlocation.compareCI("hex")||fs.fieldlocation.compareCI("prism")) fs.data_at = SCIRun::Field::CELL;
	}
	
	
	return(fs);
}



} // end namespace


