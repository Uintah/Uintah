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


// NrrdData.cc - Interface to Gordon's Nrrd package
//
//  Written by:
//   David Weinstein
//   School of Computing
//   University of Utah
//   February 2001
//
//  Copyright (C) 2001 SCI Institute

#include <Core/Datatypes/NrrdData.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>

using std::cerr;

namespace SCIRun {

static Persistent* make_NrrdData() {
  return scinew NrrdData;
}

PersistentTypeID NrrdData::type_id("NrrdData", "PropertyManager", make_NrrdData);

//vector<string> NrrdData::valid_tup_types_;


NrrdData::NrrdData(bool owned) : 
  nrrd(nrrdNew()),
  embed_object_(false),
  data_owned_(owned)
{
  //if (valid_tup_types_.size() == 0) {
  //load_valid_tuple_types();
  //}
}

NrrdData::NrrdData(const NrrdData &copy) :
  nrrd_fname_(copy.nrrd_fname_) 
{
  nrrd = nrrdNew();
  nrrdCopy(nrrd, copy.nrrd);
  //copy_sci_data(copy);
}

NrrdData::~NrrdData() {
  if(data_owned_) {
    nrrdNuke(nrrd);
  } else {
    nrrdNix(nrrd);
  }
}

NrrdData* 
NrrdData::clone() 
{
  return new NrrdData(*this);
}

// void 
// NrrdData::load_valid_tuple_types() 
// {
//   valid_tup_types_.push_back("Scalar");
//   valid_tup_types_.push_back("Vector");
//   valid_tup_types_.push_back("Tensor");
// }

// // This needs to parse axis 0 and see if the label is tuple as well...
// bool
// NrrdData::is_sci_nrrd() const 
// {
//   return (originating_field_.get_rep() != 0);
// }

// void 
// NrrdData::copy_sci_data(const NrrdData &cp)
// {
//   originating_field_ = cp.originating_field_;
// }

// int
// NrrdData::get_tuple_axis_size() const
// {
//   vector<string> elems;
//   get_tuple_indecies(elems);
//   return elems.size();
// }


// This would be much easier to check with a regular expression lib
// A valid label has the following format:
// type = one of the valid types (Scalar, Vector, Tensor)
// elem = [A-Za-z0-9\-]+:type
// (elem,?)+

bool 
NrrdData::in_name_set(const string &s) const
{
  const string 
    word("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_");
  
  //cout << "checking in_name " << s << endl;
  // test against valid char set.

  for(string::size_type i = 0; i < s.size(); i++) {
    bool in_set = false;
    for (unsigned int c = 0; c < word.size(); c++) {
      if (s[i] == word[c]) {
	in_set = true;
	break;
      }
    }
    if (! in_set) {
      //cout << "in_name_set failing" << endl;
      return false;
    }
  }

  return true;
}

// bool 
// NrrdData::in_type_set(const string &s) const
// {
//   // test against valid char set.
//   vector<string>::iterator iter = valid_tup_types_.begin();
//   while (iter != valid_tup_types_.end()) {
//     //cout << "comp " << s << " " << *iter << endl;
//     if (s == *iter) {
//       return true;
//     }
//     ++iter;
//   }
//   //cout << "in_type_set failing" << endl;
//   return false;
// }


// bool
// NrrdData::verify_tuple_label(const string &s, vector<string> &elems) const
// {

//   // first char must be part of name set
//   string::size_type nm_idx = 0;
//   string::size_type type_idx = s.size();

//   if (! s.size()) return false;

//   //cout << "label is: " << s << endl;
//   for(string::size_type i = 0; i < s.size(); i++) {
//     //cout << s[i] << endl;
//     if (s[i] == ':') {
//       // substring up until here must be a name
//       string sub = s.substr(nm_idx, i - nm_idx);
//       if (! in_name_set(sub)) return false;
//       // set nm_idx to something invalid for now.
//       type_idx = i+1;

//     } else if (s[i] == ',' || (i == s.size() - 1)) {
//       int off = 0;
//       if (i == s.size() - 1) {
// 	off = 1;
//       }
//       // substring up until here must be an elem
//       //cout << "sub from : " << type_idx << " to: " << i << endl;
//       string sub = s.substr(type_idx, i - type_idx + off);
//       if (! in_type_set(sub)) return false;
//       // the valid elem is from nm_idx to i-1
//       string elem = s.substr(nm_idx, i - nm_idx + off);
//       elems.push_back(elem);

//       // start looking for next valid elem
//       nm_idx = i+1;
//       // set type_idx to something invalid for now.
//       type_idx = s.size();
//     }
//   }
//   return true;
// }


// // return a comma separated list of just the type names along the tuple axis.
// string
// NrrdData::concat_tuple_types() const
// {
//   string rval;
//   const string s(nrrd->axis[0].label);

//   // first char must be part of name set
//   string::size_type nm_idx = 0;
//   string::size_type type_idx = s.size();

//   if (s.size()) {

//     //cout << "label is: " << s << endl;
//     for(string::size_type i = 0; i < s.size(); i++) {
//       //cout << s[i] << endl;
//       if (s[i] == ':') {
// 	// substring up until here must be a name
// 	string sub = s.substr(nm_idx, i - nm_idx);

// 	// set nm_idx to something invalid for now.
// 	if ( in_name_set(sub) )
// 	  type_idx = i+1;

//       } else if (s[i] == ',' || (i == s.size() - 1)) {
// 	int off = 0;

// 	if (i == s.size() - 1)
// 	  off = 1;

// 	// substring up until here must be an elem
// 	//cout << "sub from : " << type_idx << " to: " << i << endl;
// 	string sub = s.substr(type_idx, i - type_idx + off);

// 	if (rval.size() == 0)
// 	  rval = sub;
// 	else
// 	  rval += string(",") + sub;

// 	// start looking for next valid elem
// 	nm_idx = i+1;
// 	// set type_idx to something invalid for now.
// 	type_idx = s.size();
//       }
//     }
//   }

//   return rval;
// }

// bool
// NrrdData::get_tuple_indecies(vector<string> &elems) const
// {
//   if (!nrrd) return false;
//   string tup(nrrd->axis[0].label);
//   return verify_tuple_label(tup, elems);
// }

// bool 
// NrrdData::get_tuple_index_info(int tmin, int tmax, int &min, int &max) const
// {
//   if (!nrrd || !nrrd->axis[0].label) return false;
//   string tup(nrrd->axis[0].label);
//   vector<string> elems;
//   get_tuple_indecies(elems);

//   if (tmin < 0 || tmin > (int)elems.size() - 1 || 
//       tmax < 0 || tmax > (int)elems.size() - 1 ) return false;

//   min = 0;
//   max = 0;
//   for (int i = 0; i <= tmax; i++) {
    
//     string &s = elems[i];
//     int inc = 0;
//     if (s.find(string("Scalar")) <= s.size() - 1) {
//       inc = 1;
//     } else if (s.find(string("Vector")) <= s.size() - 1) {
//       inc = 3;
//     } else if (s.find(string("Tensor")) <= s.size() - 1) {
//       inc = 7;
//     }
//     if (tmin > i) min+=inc;
//     if (tmax > i) max+=inc;
//     if (tmax == i) max+= inc - 1;
    
//   } 
//   return true;
// }

#define NRRDDATA_VERSION 4

//////////
// PIO for NrrdData objects
void NrrdData::io(Piostream& stream) 
{
  int version =  stream.begin_class("NrrdData", NRRDDATA_VERSION);
  // Do the base class first...
  if (version > 2) 
  {
    PropertyManager::io(stream);
  }

  // In version 4 and higher we denote by a bool whether the object
  // is embedded or not. In case it is it is handled by the new
  // reader and writer version that comes with version 4.
  if (version > 3) stream.io(embed_object_);

  if (stream.reading()) {
  
	if ((version < 4)||(!embed_object_))
	{
		
		// Added a check against dumping a pointer without deallocation
		// memory. 
		if (nrrd)
		{   // make sure we free any existing Nrrd Data set
			if(data_owned_) 
			{
				nrrdNuke(nrrd);
			} 
			else 
			{
				nrrdNix(nrrd);
			}
			// Make sure we put a zero pointer in the fiedl. There is no nrrd
			nrrd = 0;
		}

		// This is the old code, which needs some update in the way
		// errors are reported
		// Functions have been added to supply a filename for external
		// nrrds 
		
		Pio(stream, nrrd_fname_);
		if (nrrdLoad(nrrd = nrrdNew(), nrrd_fname_.c_str(), 0)) 
		{
			// Need to upgade error reporting
			char *err = biffGet(NRRD);
			cerr << "Error reading nrrd " << nrrd_fname_ << ": " << err << endl;
			free(err);
			biffDone(NRRD);
			return;
		}
	}
	else
	{   // Allow for raw embedded nrrds in SCIRun streams

		// Added a check against dumping a pointer without deallocation
		// memory.
		// Any existing data will be purged, so we do not have a memory leak
		
		if (nrrd)
		{   // make sure we free any existing Nrrd Data set
			if(data_owned_) {
				nrrdNuke(nrrd);
			} else {
				nrrdNix(nrrd);
			}
			nrrd = 0;
		}

		// Create a new nrrd structure
		if (!(nrrd = nrrdNew())) 
		{   // Needs to be replaced with proper exception code
			std::cerr << "Error allocating memory for nrrd" << std::endl;
		}
		
		stream.begin_cheap_delim();
		stream.io(nrrd->type);  // the type of the nrrd
		
		// We dump the dimensions right at the start, so when reading
		// the data we can directly allocate the proper amount of memory
		
		stream.begin_cheap_delim();
		stream.io(nrrd->dim);
		
		int nrrddims[NRRD_DIM_MAX]; // defined in nrrd.h
		for (int p = 0; p<nrrd->dim; p++)
		{
			stream.io(nrrddims[p]);
		}	
		stream.end_cheap_delim();

		// Allocate memory using the nrrd allocator
		// Need some error checking here
		 
					// Need to upgade error reporting

		 
		if(nrrdAlloc_nva(nrrd,nrrd->type,nrrd->dim,nrrddims))	
		{
			char *err = biffGet(NRRD);
			std::cerr << "Error reading nrrd: " << err << std::endl;
			free(err);
			biffDone(NRRD); 
		}
		
		data_owned_ = true; // which is true of course as I just allocated the memory for the nrrddata
		
		stream.begin_cheap_delim();
		// Read the contents of the axis
		
		// Pio uses std::string and nrrd char*
		// These object are used as intermediates
		std::string label, unit;
		
		for (int q=0; q< nrrd->dim; q++)
		{
			stream.begin_cheap_delim();
			stream.io(nrrd->axis[q].size);
			stream.io(nrrd->axis[q].spacing);
			stream.io(nrrd->axis[q].min);
			stream.io(nrrd->axis[q].max);
			stream.io(nrrd->axis[q].center);
			stream.io(nrrd->axis[q].kind);
			stream.io(label);
			stream.io(unit);
			// dupiclate the strings so they are not deallocated when label and
			// unit are destroyed. This uses the nrrd allocato for obtaining memory
			// for the strings, we should not mix malloc and scinew..
			
			// Need error checking here as well
			nrrd->axis[q].label= airStrdup(label.c_str());
			nrrd->axis[q].unit= airStrdup(unit.c_str());
			stream.end_cheap_delim();
		}
		stream.end_cheap_delim();
		

		// Same construct as above for label and unit
		std::string content;
		stream.io(content);
		nrrd->content = airStrdup(content.c_str());
		stream.io(nrrd->blockSize);
		stream.io(nrrd->oldMin);
		stream.io(nrrd->oldMax);

		// Dummies for the moment until I figure out how to read
		// AirArrays
		int numcmts;
		int numkeys;
		
		
		
		// Place holders for extending the reader to the comment and
		// keyvalue pair arrays. Currently zeros are written to indicate
		// the length of the arrays
		stream.begin_cheap_delim();
		stream.io(numcmts);
		stream.end_cheap_delim();
		
		stream.begin_cheap_delim();
		stream.io(numkeys);
		stream.end_cheap_delim();
		
		stream.begin_cheap_delim();	
		int size;
		stream.io(size);
		
		// Ugly but necessary:
		// big switch statement going over every type of the nrrd structure
		switch(nrrd->type)
		{
			case nrrdTypeChar:
				{
					char *ptr = static_cast<char *>(nrrd->data);
					for (int p=0; p <size; p ++) stream.io(ptr[p]);
				}
				break;
			case nrrdTypeUChar:
				{
					unsigned char *ptr = static_cast<unsigned char *>(nrrd->data);
					for (int p=0; p <size; p ++) stream.io(ptr[p]);
				}
				break;
			case nrrdTypeShort:
				{
					short *ptr = static_cast<short *>(nrrd->data);
					for (int p=0; p <size; p ++) stream.io(ptr[p]);
				}
				break;
			case nrrdTypeUShort:
				{
					unsigned short *ptr = static_cast<unsigned short *>(nrrd->data);
					for (int p=0; p <size; p ++) stream.io(ptr[p]);
				}
				break;
			case nrrdTypeInt:
				{
					int *ptr = static_cast<int *>(nrrd->data);
					for (int p=0; p <size; p ++) stream.io(ptr[p]);
				}
				break;
			case nrrdTypeUInt:
				{
					unsigned short *ptr = static_cast<unsigned short *>(nrrd->data);
					for (int p=0; p <size; p ++) stream.io(ptr[p]);
				}
				break;
			case nrrdTypeLLong:
				{
					long long *ptr = static_cast<long long *>(nrrd->data);
					for (int p=0; p <size; p ++) stream.io(ptr[p]);
				}
				break;
			case nrrdTypeULLong:
				{
					// Currently PIO does not support unsigned long long
					// Need to fix this bug in the Persistent.h
					long long *ptr = static_cast<long long *>(nrrd->data);
					for (int p=0; p <size; p ++) stream.io(ptr[p]);
				}
				break;			
			case nrrdTypeFloat:
				{
					float *ptr = static_cast<float *>(nrrd->data);
					for (int p=0; p <size; p ++) stream.io(ptr[p]);
				}
				break;
			case nrrdTypeDouble:
				{
					double *ptr = static_cast<double *>(nrrd->data);
					for (int p=0; p <size; p ++) stream.io(ptr[p]);
				}
				break;			
			case nrrdTypeBlock:
				{
					char *ptr = static_cast<char *>(nrrd->data);
					for (int p=0; p < (size*nrrd->blockSize); p ++) stream.io(ptr[p]);
				}
				break;			
			default:
				// We should not get here, but it outputs a statement in case we reach this one
				// due to some other bug elsewhere
				std::cerr << "Error embedding nrrd, unknown datatype in nrrd " << std::endl;
		}
		stream.end_cheap_delim();
		stream.end_cheap_delim();
	}
	
  } else { // writing

    // the nrrd file name will just append .nrrd
  
	if ((version < 4)||(!embed_object_))
	{    
		nrrd_fname_ = stream.file_name + string(".nrrd");
		Pio(stream, nrrd_fname_);
		NrrdIoState *no = 0;
		TextPiostream *text = dynamic_cast<TextPiostream*>(&stream);
		if (text) {
		  no = nrrdIoStateNew();
		  no->encoding = nrrdEncodingAscii;
		} 
		if (nrrdSave(nrrd_fname_.c_str(), nrrd, no)) {
		  char *err = biffGet(NRRD);      
		  cerr << "Error writing nrrd " << nrrd_fname_ << ": "<< err << endl;
		  free(err);
		  biffDone(NRRD);
		  return;
		}
		if (text) { nrrdIoStateNix(no); }
	}
	else
	{
		// Save the type of data
		stream.begin_cheap_delim();
		stream.io(nrrd->type);

		// We dump the dimensions right at the start, so when reading
		// the data we can directly allocate the proper amount of memory
				
		stream.begin_cheap_delim();
		stream.io(nrrd->dim);
		for (int q=0; q < nrrd->dim; q++)
		{
			stream.io(nrrd->axis[q].size);
		}
		
		stream.end_cheap_delim();		
		// Save the contents of the axis

		stream.begin_cheap_delim();		
		for (int q=0; q< nrrd->dim; q++)
		{
			stream.begin_cheap_delim();
			stream.io(nrrd->axis[q].size);
			stream.io(nrrd->axis[q].spacing);
			stream.io(nrrd->axis[q].min);
			stream.io(nrrd->axis[q].max);
			stream.io(nrrd->axis[q].center);
			stream.io(nrrd->axis[q].kind);
			std::string label, unit;
			if ( nrrd->axis[q].label) { label = nrrd->axis[q].label; } else { label = ""; };
			if ( nrrd->axis[q].unit) { label = nrrd->axis[q].unit; } else { unit = ""; };
			stream.io(label);
			stream.io(unit);
			stream.end_cheap_delim();
		}
		stream.end_cheap_delim();
		
		if (nrrd->content)
		{
			std::string content = nrrd->content;
			stream.io(content);
		}
		else
		{
			std::string content = "";
			stream.io(content);
		}
		stream.io(nrrd->blockSize);
		stream.io(nrrd->oldMin);
		stream.io(nrrd->oldMax);

		// Make entry point for comments and keyvalue pair
		// arrays
		
		int numcmts = 0;
		int numkeys = 0;
		
		stream.begin_cheap_delim();
		stream.io(numcmts);
		stream.end_cheap_delim();
		
		stream.begin_cheap_delim();
		stream.io(numkeys);
		stream.end_cheap_delim();
		
		// Figure out how many data bytes we have
		
		int dim = nrrd->dim;
		int size = 1;
		for (int p = 0; p < dim ; p++)
		{
			size *= nrrd->axis[p].size;
		}
		
		stream.begin_cheap_delim();	
		stream.io(size);
		switch(nrrd->type)
		{
			case nrrdTypeChar:
				{
					char *ptr = static_cast<char *>(nrrd->data);
					for (int p=0; p <size; p ++) stream.io(ptr[p]);
				}
				break;
			case nrrdTypeUChar:
				{
					unsigned char *ptr = static_cast<unsigned char *>(nrrd->data);
					for (int p=0; p <size; p ++) stream.io(ptr[p]);
				}
				break;
			case nrrdTypeShort:
				{
					short *ptr = static_cast<short *>(nrrd->data);
					for (int p=0; p <size; p ++) stream.io(ptr[p]);
				}
				break;
			case nrrdTypeUShort:
				{
					unsigned short *ptr = static_cast<unsigned short *>(nrrd->data);
					for (int p=0; p <size; p ++) stream.io(ptr[p]);
				}
				break;
			case nrrdTypeInt:
				{
					int *ptr = static_cast<int *>(nrrd->data);
					for (int p=0; p <size; p ++) stream.io(ptr[p]);
				}
				break;
			case nrrdTypeUInt:
				{
					unsigned short *ptr = static_cast<unsigned short *>(nrrd->data);
					for (int p=0; p <size; p ++) stream.io(ptr[p]);
				}
				break;
			case nrrdTypeLLong:
				{
					long long *ptr = static_cast<long long *>(nrrd->data);
					for (int p=0; p <size; p ++) stream.io(ptr[p]);
				}
				break;
			case nrrdTypeULLong:
				{
					long long *ptr = static_cast<long long *>(nrrd->data);
					for (int p=0; p <size; p ++) stream.io(ptr[p]);
				}
				break;			
			case nrrdTypeFloat:
				{
					float *ptr = static_cast<float *>(nrrd->data);
					for (int p=0; p <size; p ++) stream.io(ptr[p]);
				}
				break;
			case nrrdTypeDouble:
				{
					double *ptr = static_cast<double *>(nrrd->data);
					for (int p=0; p <size; p ++) stream.io(ptr[p]);
				}
				break;			
			case nrrdTypeBlock:
				{
					char *ptr = static_cast<char *>(nrrd->data);
					for (int p=0; p < (size*nrrd->blockSize); p ++) stream.io(ptr[p]);
				}
				break;			
			default:
				std::cerr << "Error embedding nrrd, unknown datatype in nrrd " << std::endl;
		}
		stream.end_cheap_delim();
		stream.end_cheap_delim();
	}
  }
  if (version > 1) {
	// Somehow a statement got saved whether the nrrd owned the data or not. Although it might not
	// own the data while writing, when creating a new object in case of reading the data, it always
	// will be own by that nrrd, who else would own it. Hence in the new version it will write a dummy
	// variable and as well read a dummy. This dummy is set to one, so when older versions of SCIRun read
	// the data, they properly assume they own the data
  
	bool own_data = true;
    Pio(stream, own_data);   // Always true, this field was mistakenly added in a previous version
  }
  stream.end_class();
}

template <>
unsigned int get_nrrd_type<Tensor>() {
  return nrrdTypeFloat;
}

template <>
unsigned int get_nrrd_type<char>() {
  return nrrdTypeChar;
}


template <>
unsigned int get_nrrd_type<unsigned char>()
{
  return nrrdTypeUChar;
}

template <>
unsigned int get_nrrd_type<short>()
{
  return nrrdTypeShort;
}

template <>
unsigned int get_nrrd_type<unsigned short>()
{
  return nrrdTypeUShort;
}

template <>
unsigned int get_nrrd_type<int>()
{
  return nrrdTypeInt;
}

template <>
unsigned int get_nrrd_type<unsigned int>()
{
  return nrrdTypeUInt;
}

template <>
unsigned int get_nrrd_type<long long>()
{
  return nrrdTypeLLong;
}

template <>
unsigned int get_nrrd_type<unsigned long long>()
{
  return nrrdTypeULLong;
}

template <>
unsigned int get_nrrd_type<float>()
{
  return nrrdTypeFloat;
}



}  // end namespace SCIRun
