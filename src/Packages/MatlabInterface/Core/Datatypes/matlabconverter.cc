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
 * FILE: matlabconverter.cc
 * AUTH: Jeroen G Stinstra
 * DATE: 18 MAR 2004
 */

#include "matlabconverter.h"


namespace MatlabIO {

// Currently the property converter only manages strings
// all other data is ignored both on matlabside as well
// as on the property manager side.

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
			index = ma.getfieldnameindexCI("data");
			if (index == -1) index = ma.getfieldnameindexCI("potvals");	// in case it is a saved TSDF file
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
			break;
			
		case matlabarray::mlSPARSE:
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


void matlabconverter::sciMatrixTOmlMatrix(SCIRun::MatrixHandle &scimat,matlabarray &mlmat,matlabarray::mitype dataformat)
{
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


void matlabconverter::sciMatrixTOmlArray(SCIRun::MatrixHandle &scimat,matlabarray &mlmat,matlabarray::mitype dataformat)
{
	matlabarray dataarray;
	mlmat.createstructarray();
	sciMatrixTOmlMatrix(scimat,dataarray,dataformat);
	mlmat.setfield(0,"data",dataarray);
	sciPropertyTOmlProperty(static_cast<SCIRun::PropertyManager *>(scimat.get_rep()),mlmat);
}


bool matlabconverter::isvalidmatrixname(std::string name)
{
	const std::string validchar("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789");
	const std::string validstartchar("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ");


	bool valid = true;
	bool foundchar = false;

	for (long p=0; p < name.size(); p++)
	{
		if (p == 0)
		{
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
		if (fieldnameindex == -1) return(0);
		
		subarray = mlarray.getfield(0,"data");
	
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
			break;
			// END CONVERSION OF MATLAB MATRIX
			
		case matlabarray::mlSTRUCT:
		case matlabarray::mlOBJECT:
			{
				long dataindex;
				dataindex = mlarray.getfieldnameindexCI("data");
			
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
			
			}	
			break;
			
		default:
			{   // The program should not get here
				throw matlabconverter_error(); 
			}
	}
}


void matlabconverter::sciNrrdDataTOmlMatrix(SCITeem::NrrdDataHandle &scinrrd, matlabarray &mlarray,matlabarray::mitype dataformat)
{

	Nrrd	    *nrrdptr;

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


void matlabconverter::sciNrrdDataTOmlArray(SCITeem::NrrdDataHandle &scinrrd, matlabarray &mlarray,matlabarray::mitype dataformat)
{
	matlabarray matrix;
	sciNrrdDataTOmlMatrix(scinrrd,matrix,dataformat);
		
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
//   any suggestions for other types that need support ??

long matlabconverter::sciFieldCompatible(matlabarray &mlarray,std::string &infostring)
{

	if (!mlarray.isstruct()) return(0); // not compatible if it is not structured data
	fieldstruct fs = analyzefieldstruct(mlarray); // read the main structure of the object
	
	// Check what kind of field has been supplied
	
	// The fieldtype string, contains the type of field, so it can be listed in the
	// infostring. Just to supply the user with a little bit more data.
	std::string fieldtype;
	fieldtype = "NO FIELD DATA";
	
	if (fs.scalarfield.isdense()) fieldtype = "SCALAR FIELD";
	if (fs.vectorfield.isdense()) fieldtype = "VECTOR FIELD";
	if (fs.tensorfield.isdense()) fieldtype = "TENSOR FIELD";
	
	// Field data has been analysed, now analyse the connectivity data
	// Connectivity data needs to be or edge data, or face data, or cell data,
	// or no data in which case it is a point cloud.

	if (fs.snode.isdense())
	{
		// Structured mesh:
		// StructCurveMesh, StructQuadSurfMesh, StructHexVolMesh
		
		long numdims = fs.snode.getnumdims();
		
		if (fs.snode.getm() !=  3) return(0);
		
		std::ostringstream oss;
		std::string name = mlarray.getname();
		oss << name << " ";
		if (name.length() < 40) oss << std::string(40-(name.length()),' '); // add some form of spacing
		
		switch (numdims)
		{
			case 2:
				oss << "[STRUCTURED CURVEMESH - " << fieldtype << "]";
				break;
			case 3:
				oss << "[STRUCTURED QUADSURFMESH - " << fieldtype << "]";
				break;
			case 4:
				oss << "[STRUCTURED HEXVOLMESH - " << fieldtype << "]";
				break;
			default:
				return(0);  // matrix is not compatible
		}
		infostring = oss.str();
		return(1);
	}
	
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
		if (name.length() < 40) oss << std::string(40-(name.length()),' '); // add some form of spacing		
		if (numdims == 2)
		{
			if ((dimsx[0] == 1)||(dimsx[1] == 1)) numdims = 1;
		}
		
		switch (numdims)
		{
			case 1:
				oss << "[STRUCTURED CURVEMESH - " << fieldtype << "]";
				break;
			case 2:
				oss << "[STRUCTURED QUADSURFMESH - " << fieldtype << "]";
				break;
			case 3:
				oss << "[STRUCTURED HEXVOLMESH - " << fieldtype << "]";
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
	
	if ((n==0)||(m==0)) return(0); //empty matrix, no nodes => no mesh => no field......
	if ((n != 3)&&(m != 3)) return(0); // SCIRun is ONLY 3D data, no 2D, or 1D
	
	if ((fs.edge.isempty())&&(fs.face.isempty())&&(fs.cell.isempty()))
	{
		// These is no connectivity data => it must be a pointcloud ;)
		// Supported mesh/field types here:
		// PointCloudField
		
		if (fs.elemtype.isstring())
		{   // explicitly stated type (check whether type confirms the guessed type, otherwise someone supplied us with improper data)
			if (fs.elemtype.compareCI("pointcloud")) return(0);
		}
		
		// Create an information string for the GUI
		std::ostringstream oss;
		std::string name = mlarray.getname();	
		oss << name << "  ";
		if (name.length() < 40) oss << std::string(40-(name.length()),' '); // add some form of spacing
		oss << "[POINTCLOUD - " << fieldtype << "]";
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
		
		std::ostringstream oss;	
		std::string name = mlarray.getname();
		oss << name << "  ";
		if (name.length() < 40) oss << std::string(40-(name.length()),' '); // add some form of spacing
		oss << "[CURVEFIELD - " << fieldtype << "]";
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

		if ((m==3)||((n==3)&&(n!=4)))
		{
			// check whether the element type is explicitly given
			if (fs.elemtype.isstring())
			{   // explicitly stated type 
				if (fs.elemtype.compareCI("trisurf")) return(0);
			}
			
			// Generate an output string describing the field we just reckonized
			std::ostringstream oss;	
			std::string name = mlarray.getname();
			oss << name << "  ";
			if (name.length() < 40) oss << std::string(40-(name.length()),' '); // add some form of spacing
			oss << "[TRISURFFIELD - " << fieldtype << "]";
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
			
			// Generate an output string describing the field we just reckonized
			std::ostringstream oss;	
			std::string name = mlarray.getname();
			oss << name << "  ";
			if (name.length() < 40) oss << std::string(40-(name.length()),' '); // add some form of spacing
			oss << "[QUADSURFFIELD - " << fieldtype << "]";
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
			
			std::ostringstream oss;	
			std::string name = mlarray.getname();			
			oss << name << "  ";
			if (name.length() < 40) oss << std::string(40-(name.length()),' '); // add some form of spacing
			oss << "[TETVOLFIELD - " << fieldtype << "]";
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
			
			std::ostringstream oss;	
			std::string name = mlarray.getname();		
			oss << name << "  ";
			if (name.length() < 40) oss << std::string(40-(name.length()),' '); // add some form of spacing
			oss << "[HEXVOLFIELD - " << fieldtype << "]";
			infostring = oss.str();		
			return(1);		
		}
		
		else if((m==6)||((n==6)&&(m!=4)&&(m!=8)))
		{
			if (fs.elemtype.isstring())
			{   // explicitly stated type 
				if (fs.elemtype.compareCI("prismvol")) return(0);
			}	
			
			std::ostringstream oss;	
			std::string name = mlarray.getname();		
			oss << name << "  ";
			if (name.length() < 40) oss << std::string(40-(name.length()),' '); // add some form of spacing
			oss << "[PRISMVOLFIELD - " << fieldtype << "]";
			infostring = oss.str();		
			return(1);		
		}
		
		return(0);
	}
	return(0);
}



void matlabconverter::mlArrayTOsciField(matlabarray &mlarray,SCIRun::FieldHandle &scifield)
{

	if (!mlarray.isstruct()) throw matlabconverter_error(); // not compatible if it is not structured data
	fieldstruct fs = analyzefieldstruct(mlarray); // read the main structure of the object
	
	// Convert the node information
	// Each field needs to have the position of the nodes defined
	// Currently SCIRun only accepts nodes in a 3D cartesian coordinate
	// system.
	// Dimensions are checked and the matrix is transposed if it is necessary
	
	if ((fs.node.isempty())&&(fs.snode.isempty())&&(fs.x.isempty())) throw matlabconverter_error(); // a node matrix is always required
		
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
	
	// Process structured mesh data
	if (fs.snode.isdense())
	{
		m  = fs.snode.getm();
		if (m != 3) throw matlabconverter_error();
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
	if (!(fs.fieldlocation.isempty()))
	{   // converter table for the string in the field "fieldlocation" array
		// These are case insensitive comparisons.
		if (!(fs.fieldlocation.isstring())) throw matlabconverter_error();
		if (fs.fieldlocation.compareCI("node")||fs.fieldlocation.compareCI("pts")) data_at = SCIRun::Field::NODE;
		if (fs.fieldlocation.compareCI("egde")||fs.fieldlocation.compareCI("line")) data_at = SCIRun::Field::EDGE;
		if (fs.fieldlocation.compareCI("face")||fs.fieldlocation.compareCI("fac")) data_at = SCIRun::Field::FACE;
		if (fs.fieldlocation.compareCI("cell")||fs.fieldlocation.compareCI("tet")||fs.fieldlocation.compareCI("hex")||fs.fieldlocation.compareCI("prism")) data_at = SCIRun::Field::CELL;
	}
	
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
		if (fs.vectorfield.isdense())
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
		if (fs.tensorfield.isdense())
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
						SCIRun::StructCurveField<double> *fieldptr;
						fieldptr = new SCIRun::StructCurveField<double>(meshH,data_at);
						addscalardata(fieldptr,fs.scalarfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
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
						SCIRun::StructQuadSurfField<double> *fieldptr;
						fieldptr = new SCIRun::StructQuadSurfField<double>(meshH,data_at);
//						addscalardata(fieldptr,fs.scalarfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
					}
					if (fs.vectorfield.isdense())
					{
						SCIRun::StructQuadSurfField<SCIRun::Vector> *fieldptr;
						fieldptr = new SCIRun::StructQuadSurfField<SCIRun::Vector>(meshH,data_at);
//						addvectordata(fieldptr,fs.vectorfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
					}
					if (fs.tensorfield.isdense())
					{
						SCIRun::StructQuadSurfField<SCIRun::Tensor> *fieldptr;
						fieldptr = new SCIRun::StructQuadSurfField<SCIRun::Tensor>(meshH,data_at);
//						addtensordata(fieldptr,fs.tensorfield);
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
						SCIRun::StructHexVolField<double> *fieldptr;
						fieldptr = new SCIRun::StructHexVolField<double>(meshH,data_at);
//						addscalardata(fieldptr,fs.scalarfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
					}
					if (fs.vectorfield.isdense())
					{
						SCIRun::StructHexVolField<SCIRun::Vector> *fieldptr;
						fieldptr = new SCIRun::StructHexVolField<SCIRun::Vector>(meshH,data_at);
//						addvectordata(fieldptr,fs.vectorfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
					}
					if (fs.tensorfield.isdense())
					{
						SCIRun::StructHexVolField<SCIRun::Tensor> *fieldptr;
						fieldptr = new SCIRun::StructHexVolField<SCIRun::Tensor>(meshH,data_at);
//						addtensordata(fieldptr,fs.tensorfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
					}		
				}			
				break;
		}
	
	}
		
	
	
	if (fs.snode.isdense())
	{
		long numdim = fs.snode.getnumdims();
		
		if ((numdim > 4)||(numdim < 2)) throw matlabconverter_error();
		
		std::vector<long> dims;
		std::vector<unsigned int> mdims;
		dims = fs.snode.getdims();
		
		numnodes = 1;
		mdims.resize(numdim-1);
		for (long p=1; p < numdim; p++) { numnodes *= dims[p]; mdims[p-1] = static_cast<unsigned int>(dims[p]); }
		
		switch (numdim)
		{
			case 2:
				{
					// Process a structured curve mesh
					SCIRun::StructCurveMeshHandle meshH;
					meshH = new SCIRun::StructCurveMesh;
					
					std::vector<double> nodebuffer;
					fs.snode.getnumericarray(nodebuffer);
					
					meshH->set_dim(mdims);
					long p,q;
					for (p = 0, q = 0; p < numnodes; p++, q+=3)
					{
						meshH->set_point(SCIRun::Point(nodebuffer[q],nodebuffer[q+1],nodebuffer[q+2]),static_cast<SCIRun::StructCurveMesh::Node::index_type>(p));
					}
					
					
					if ((fs.scalarfield.isempty())&&(fs.vectorfield.isempty())&&(fs.tensorfield.isempty()))
					{
						SCIRun::StructCurveField<double> *fieldptr;
						fieldptr = new SCIRun::StructCurveField<double>(meshH,data_at);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
					}
					if (fs.scalarfield.isdense())
					{
						SCIRun::StructCurveField<double> *fieldptr;
						fieldptr = new SCIRun::StructCurveField<double>(meshH,data_at);
						addscalardata(fieldptr,fs.scalarfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
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
			case 3:
				{
					// Process a structured quadsurf mesh
					SCIRun::StructQuadSurfMeshHandle meshH;
					meshH = new SCIRun::StructQuadSurfMesh;
					
					std::vector<double> nodebuffer;
					fs.snode.getnumericarray(nodebuffer);
					
					meshH->set_dim(mdims);
					unsigned int p,r;
					long q = 0;
					for (r = 0; r < mdims[1]; r++)
					for (p = 0; p < mdims[0]; p++)
					{
						meshH->set_point(SCIRun::Point(nodebuffer[q],nodebuffer[q+1],nodebuffer[q+2]),SCIRun::StructQuadSurfMesh::Node::index_type(static_cast<SCIRun::ImageMesh *>(meshH.get_rep()),p,r));
						q += 3;
					}
					
					
					if ((fs.scalarfield.isempty())&&(fs.vectorfield.isempty())&&(fs.tensorfield.isempty()))
					{
						SCIRun::StructQuadSurfField<double> *fieldptr;
						fieldptr = new SCIRun::StructQuadSurfField<double>(meshH,data_at);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
					}
					if (fs.scalarfield.isdense())
					{
						SCIRun::StructQuadSurfField<double> *fieldptr;
						fieldptr = new SCIRun::StructQuadSurfField<double>(meshH,data_at);
//						addscalardata(fieldptr,fs.scalarfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
					}
					if (fs.vectorfield.isdense())
					{
						SCIRun::StructQuadSurfField<SCIRun::Vector> *fieldptr;
						fieldptr = new SCIRun::StructQuadSurfField<SCIRun::Vector>(meshH,data_at);
//						addvectordata(fieldptr,fs.vectorfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
					}
					if (fs.tensorfield.isdense())
					{
						SCIRun::StructQuadSurfField<SCIRun::Tensor> *fieldptr;
						fieldptr = new SCIRun::StructQuadSurfField<SCIRun::Tensor>(meshH,data_at);
//						addtensordata(fieldptr,fs.tensorfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
					}		
				}			
				break;
			case 4:
				{
					// Process a structured quadsurf mesh
					SCIRun::StructHexVolMeshHandle meshH;
					meshH = new SCIRun::StructHexVolMesh;
					
					std::vector<double> nodebuffer;
					fs.snode.getnumericarray(nodebuffer);
					
					meshH->set_dim(mdims);

					unsigned int p,r,s;
					long q = 0;
					for (s = 0; s < mdims[2]; s++)
					for (r = 0; r < mdims[1]; r++)
					for (p = 0; p < mdims[0]; p++)
					{
						meshH->set_point(SCIRun::Point(nodebuffer[q],nodebuffer[q+1],nodebuffer[q+2]),SCIRun::StructHexVolMesh::Node::index_type(static_cast<SCIRun::LatVolMesh *>(meshH.get_rep()),p,r,s));
						q += 3;
					}

					if ((fs.scalarfield.isempty())&&(fs.vectorfield.isempty())&&(fs.tensorfield.isempty()))
					{
						SCIRun::StructHexVolField<double> *fieldptr;
						fieldptr = new SCIRun::StructHexVolField<double>(meshH,data_at);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
					}
					if (fs.scalarfield.isdense())
					{
						SCIRun::StructHexVolField<double> *fieldptr;
						fieldptr = new SCIRun::StructHexVolField<double>(meshH,data_at);
//						addscalardata(fieldptr,fs.scalarfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
					}
					if (fs.vectorfield.isdense())
					{
						SCIRun::StructHexVolField<SCIRun::Vector> *fieldptr;
						fieldptr = new SCIRun::StructHexVolField<SCIRun::Vector>(meshH,data_at);
//						addvectordata(fieldptr,fs.vectorfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
					}
					if (fs.tensorfield.isdense())
					{
						SCIRun::StructHexVolField<SCIRun::Tensor> *fieldptr;
						fieldptr = new SCIRun::StructHexVolField<SCIRun::Tensor>(meshH,data_at);
//						addtensordata(fieldptr,fs.tensorfield);
						scifield = static_cast<SCIRun::Field *>(fieldptr);
					}		
				}			
				break;
		}
	
	}
	
	if ((fs.edge.isempty())&&(fs.face.isempty())&&(fs.cell.isempty())&&(fs.snode.isempty())&&(fs.x.isempty()))
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
			SCIRun::PointCloudField<double> *fieldptr;
			fieldptr = new SCIRun::PointCloudField<double>(meshH,data_at);
			addscalardata(fieldptr,fs.scalarfield);
			scifield = static_cast<SCIRun::Field *>(fieldptr);
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
				SCIRun::CurveField<double> *fieldptr;
				fieldptr = new SCIRun::CurveField<double>(meshH,data_at);
				addscalardata(fieldptr,fs.scalarfield);
				scifield = static_cast<SCIRun::Field *>(fieldptr);
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
				SCIRun::TriSurfField<double> *fieldptr;
				fieldptr = new SCIRun::TriSurfField<double>(meshH,data_at);
				addscalardata(fieldptr,fs.scalarfield);
				scifield = static_cast<SCIRun::Field *>(fieldptr);
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
				SCIRun::QuadSurfField<double> *fieldptr;
				fieldptr = new SCIRun::QuadSurfField<double>(meshH,data_at);
				addscalardata(fieldptr,fs.scalarfield);
				scifield = static_cast<SCIRun::Field *>(fieldptr);
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
				SCIRun::TetVolField<double> *fieldptr;
				fieldptr = new SCIRun::TetVolField<double>(meshH,data_at);
				addscalardata(fieldptr,fs.scalarfield);
				scifield = static_cast<SCIRun::Field *>(fieldptr);
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
				SCIRun::PrismVolField<double> *fieldptr;
				fieldptr = new SCIRun::PrismVolField<double>(meshH,data_at);
				addscalardata(fieldptr,fs.scalarfield);
				scifield = static_cast<SCIRun::Field *>(fieldptr);
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
				SCIRun::HexVolField<double> *fieldptr;
				fieldptr = new SCIRun::HexVolField<double>(meshH,data_at);
				addscalardata(fieldptr,fs.scalarfield);
				scifield = static_cast<SCIRun::Field *>(fieldptr);
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
			scifield->set_property("name",fs.name.getstring(),false);
		}
	}
	
}


// Templates for adding data in the field


template <class FIELDPTR> 
void matlabconverter::addscalardata(FIELDPTR fieldptr,matlabarray mlarray)
{
	std::vector<double> fielddata;
	mlarray.getnumericarray(fielddata); // cast and copy the real part of the data

	fieldptr->resize_fdata();   // make sure it is resized to number of nodes/edges/faces/cells or whatever
	std::vector<double>& fdata = fieldptr->fdata();  // get a reference to the actual data
	
	long numdata = fielddata.size();
	if (numdata > fdata.size()) numdata = fdata.size(); // make sure we do not copy more data than there are elements
	
	for (long p=0; p < numdata; p++) { fdata[p] = fielddata[p]; }
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
	// meshH->reserve_nodes(numnodes);
	
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
	
	// meshH->reserve_edges(mldata.getn());
	
	for (p = 0, q = 0; p < mlarray.getn(); p++, q += 2)
	{
		meshH->add_edge(typename MESH::Node::index_type(mldata[q]), typename MESH::Node::index_type(mldata[q+1]));
	}
		  
	// meshH->add_edges(mldata);
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
	
	// meshH->reserve_faces(n);	  
			  	  
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
	
	// meshH->reserve_faces(n);	  
			  	  
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
	
	fieldstruct		fs;
	long			index;
	
	if (!ma.isstruct()) return(fs);
	
	// NODE MATRIX
	index = ma.getfieldnameindexCI("node");
	if (index > -1) fs.node = ma.getfield(0,index);
	index = ma.getfieldnameindexCI("pts");
	if (index > -1) fs.node = ma.getfield(0,index);

	// STRUCTURED NODE MATRIX
	index = ma.getfieldnameindexCI("snode");
	if (index > -1) fs.snode = ma.getfield(0,index);
	index = ma.getfieldnameindexCI("spts");
	if (index > -1) fs.snode = ma.getfield(0,index);
	index = ma.getfieldnameindexCI("xyz");
	if (index > -1) fs.snode = ma.getfield(0,index);

	// STRUCTURE MATRICES IN SUBMATRICES
	index = ma.getfieldnameindexCI("x");
	if (index > -1) fs.x = ma.getfield(0,index);
	index = ma.getfieldnameindexCI("y");
	if (index > -1) fs.y = ma.getfield(0,index);
	index = ma.getfieldnameindexCI("z");
	if (index > -1) fs.z = ma.getfield(0,index);

	// EDGE MATRIX
	index = ma.getfieldnameindexCI("edge");
	if (index > -1) fs.edge = ma.getfield(0,index);
	index = ma.getfieldnameindexCI("line");
	if (index > -1) fs.edge = ma.getfield(0,index);

	// FACE MATRIX
	index = ma.getfieldnameindexCI("face");
	if (index > -1) fs.face = ma.getfield(0,index);
	index = ma.getfieldnameindexCI("fac");
	if (index > -1) fs.face = ma.getfield(0,index);
	index = ma.getfieldnameindexCI("quad");
	if (index > -1) fs.face = ma.getfield(0,index);
	
	// CELL MATRIX
	index = ma.getfieldnameindexCI("cell");
	if (index > -1) fs.cell = ma.getfield(0,index);
	index = ma.getfieldnameindexCI("tet");
	if (index > -1) fs.cell = ma.getfield(0,index);
	index = ma.getfieldnameindexCI("hex");
	if (index > -1) fs.cell = ma.getfield(0,index);
	index = ma.getfieldnameindexCI("prism");
	if (index > -1) fs.cell = ma.getfield(0,index);
	
	// FIELDNODE MATRIX
	index = ma.getfieldnameindexCI("field");
	if (index > -1) fs.scalarfield = ma.getfield(0,index);
	index = ma.getfieldnameindexCI("scalarfield");
	if (index > -1) fs.scalarfield = ma.getfield(0,index);

	// FIELDEDGE MATRIX
	index = ma.getfieldnameindexCI("vectorfield");
	if (index > -1) fs.vectorfield = ma.getfield(0,index);

	// FIELDFACE MATRIX
	index = ma.getfieldnameindexCI("tensorfield");
	if (index > -1) fs.tensorfield = ma.getfield(0,index);

	// FIELDCELL MATRIX
	index = ma.getfieldnameindexCI("fieldlocation");
	if (index > -1) fs.fieldlocation = ma.getfield(0,index);

	// ELEMTYPE MATRIX
	index = ma.getfieldnameindexCI("elemtype");
	if (index > -1) fs.elemtype = ma.getfield(0,index);
	index = ma.getfieldnameindexCI("type");
	if (index > -1) fs.elemtype = ma.getfield(0,index);

	// NAME OF THE MESH/FIELD
	index = ma.getfieldnameindexCI("name");
	if (index > -1) fs.name = ma.getfield(0,index);

	// PROPERTY FIELD
	index = ma.getfieldnameindexCI("property");
	if (index > -1) fs.property = ma.getfield(0,index);

	return(fs);
}



} // end namespace


