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


// AUTH: Jeroen Stinstra

#ifndef SCIRUN_CORE_BUNDLE_h
#define SCIRUN_CORE_BUNDLE_h 1

#include <Core/Datatypes/Datatype.h>
#include <Core/Datatypes/Field.h>
#include <Core/Geom/ColorMap.h>
#include <Core/Volume/Colormap2.h>
#include <Core/Geom/Path.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/NrrdData.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/NrrdString.h>
#include <Core/Datatypes/NrrdScalar.h>

#include <sgi_stl_warnings_off.h>
#include <deque>
#include <string>
#include <sgi_stl_warnings_on.h>

#ifdef HAVE_BUNDLE
#include <Core/Bundle/Bundle.h>
#endif

namespace SCIRun{

/////////
// Structure to hold NrrdData
class Bundle : public PropertyManager {
public:  
  
  Bundle();
  virtual ~Bundle();

  // SCIRun's way of copying
  virtual Bundle* clone();

  // For writning bundles to file
  virtual void io(Piostream&);
  static PersistentTypeID type_id;

   // Basic functionality
   
   // NOTE: For each type added on, a new entry in  io must be made
   // It does not do dynamic compilation currently
   
   // Get handle to an object from the bundle
    template<class T> inline LockingHandle<T> get(std::string name);
    
   // Add or replace an object in the bundle
    template<class T> inline void set(std::string name, LockingHandle<T> &handle);
    
   // Check whether an object is present 
    template<class T> inline bool is(std::string name);
    
   // Get the number of objects of a certain type
    template<class T> inline int  num();
    
   // Get the name of the objects 
    template<class T> inline std::string getname(int index);

   // remove object
    inline void rem(std::string name);
    
   // Merge two bundles together
    void merge(SCIRun::LockingHandle<Bundle> C);

    // transpose when doing a matrix to nrrd conversion
    void transpose_nrrd(bool on);

public:
  
  // The basic functions for managing fields
  //  getfield     -> retrieve a Handle to a field stored in the bundle
  //  setfield     -> add a field with a name, if it already exists the old one is overwritten
  //  remfield     -> remove a handle from the bundle
  //  isfield      -> Test whether a field is present in the bundle
  //  numfields    -> The number of fields stored in the bundle 
  //  getfieldname -> Get the nth name in the bundle for building a contents list
  
  LockingHandle<Field> getfield(std::string name) { return(get<Field>(name)); }
  void setfield(std::string name, LockingHandle<Field> &field) { set<Field>(name,field); }
  void remfield(std::string name) { rem(name); }
  bool isfield(std::string name)  { return(is<Field>(name)); }
  int  numfields() { return(num<Field>()); }
  std::string getfieldname(int index) {return(getname<Field>(index)); }
  
  // The basic functions for managing matrices
  //  getmatrix     -> retrieve a Handle to a matrix stored in the bundle
  //  setmatrix     -> add a matrix with a name, if it already exists the old one is overwritten
  //  remmatrix     -> remove a handle from the bundle
  //  ismatrix      -> Test whether a matrix is present in the bundle
  //  nummatrices   -> The number of matrices stored in the bundle 
  //  getmatrixname -> Get the nth name in the bundle for building a contents list
  
  LockingHandle<Matrix> getmatrix(std::string name);
  void setmatrix(std::string name, LockingHandle<Matrix> &matrix) { set<Matrix>(name,matrix); }
  void remmatrix(std::string name) { rem(name); }
  bool ismatrix(std::string name); 
  int  nummatrices();
  std::string getmatrixname(int index);

  // The basic functions for managing nrrds
  //  getnrrd     -> retrieve a Handle to a matrix stored in the bundle
  //  setnrrd     -> add a nrrd with a name, if it already exists the old one is overwritten
  //  remnrrd     -> remove a handle from the bundle
  //  isnrrd      -> Test whether a nrrd is present in the bundle
  //  numnrrds   -> The number of nrrds stored in the bundle 
  //  getnrrdname -> Get the nth name in the bundle for building a contents list
  
  LockingHandle<NrrdData> getnrrd(std::string name) { return(get<NrrdData>(name)); }
  NrrdScalar getnrrdscalar(std::string name) {return(NrrdScalar(get<NrrdData>(name))); }
  NrrdString getnrrdstring(std::string name) {return(NrrdString(get<NrrdData>(name))); }  
  void setnrrd(std::string name, LockingHandle<NrrdData> &nrrd) { set<NrrdData>(name,nrrd); }
  void setnrrdscalar(std::string name, NrrdScalar scalar) { NrrdDataHandle nrrd = scalar.gethandle(); set<NrrdData>(name,nrrd); }
  void setnrrdstring(std::string name, NrrdString string) { NrrdDataHandle nrrd = string.gethandle(); set<NrrdData>(name,nrrd); }

  void remnrrd(std::string name) { rem(name); }
  bool isnrrd(std::string name)  { return(is<NrrdData>(name)); }
  int  numnrrds() { return(num<NrrdData>()); }
  std::string getnrrdname(int index) {return(getname<NrrdData>(index)); }
  
  
  // The basic functions for managing colormaps
  //  getcolormap     -> retrieve a Handle to a colormap stored in the bundle
  //  setcolormap     -> add a colormap with a name, if it already exists the old one is overwritten
  //  remcolormap     -> remove a handle from the bundle
  //  iscolormap      -> Test whether a colormap is present in the bundle
  //  numcolormaps   -> The number of colormaps stored in the bundle 
  //  getcolormapname -> Get the nth name in the bundle for building a contents list

  LockingHandle<ColorMap> getcolormap(std::string name) { return(get<ColorMap>(name)); }
  void setcolormap(std::string name, LockingHandle<ColorMap> &colormap) { set<ColorMap>(name,colormap); }
  void remcolormap(std::string name) { rem(name); }
  bool iscolormap(std::string name)  { return(is<ColorMap>(name)); }
  int  numcolormaps() { return(num<ColorMap>()); }
  std::string getcolormapname(int index) {return(getname<ColorMap>(index)); }

  // The basic functions for managing colormap2s
  //  getcolormap2     -> retrieve a Handle to a colormap2 stored in the bundle
  //  setcolormap2     -> add a colormap2 with a name, if it already exists the old one is overwritten
  //  remcolormap2     -> remove a handle from the bundle
  //  iscolormap2      -> Test whether a colormap2 is present in the bundle
  //  numcolormap2s   -> The number of colormap2s stored in the bundle 
  //  getcolormap2name -> Get the nth name in the bundle for building a contents list

  LockingHandle<ColorMap2> getcolormap2(std::string name) { return(get<ColorMap2>(name)); }
  void setcolormap2(std::string name, LockingHandle<ColorMap2> &colormap2) { set<ColorMap2>(name,colormap2); }
  void remcolormap2(std::string name) { rem(name); }
  bool iscolormap2(std::string name)  { return(is<ColorMap2>(name)); }
  int  numcolormap2s() { return(num<ColorMap2>()); }
  std::string getcolormap2name(int index) {return(getname<ColorMap2>(index)); }

  // The basic functions for managing paths
  //  getpath     -> retrieve a Handle to a path stored in the bundle
  //  setpath     -> add a path with a name, if it already exists the old one is overwritten
  //  rempath     -> remove a handle from the bundle
  //  ispath      -> Test whether a path is present in the bundle
  //  numpaths   -> The number of paths stored in the bundle 
  //  getpathname -> Get the nth name in the bundle for building a contents list

  LockingHandle<Path> getpath(std::string name) { return(get<Path>(name)); }
  void setpath(std::string name, LockingHandle<Path> &path) { set<Path>(name,path); }
  void rempath(std::string name) { rem(name); }
  bool ispath(std::string name)  { return(is<Path>(name)); }
  int  numpaths() { return(num<Path>()); }
  std::string getpathname(int index) {return(getname<Path>(index)); }
  
  // The basic functions for managing bundles
  //  getbundle     -> retrieve a Handle to a bundle stored in the bundle
  //  setbundle     -> add a bundle with a name, if it already exists the old one is overwritten
  //  rembundle     -> remove a handle from the bundle
  //  isbundle      -> Test whether a bundle is present in the bundle
  //  numbundles   -> The number of bundles stored in the bundle 
  //  getbundlename -> Get the nth name in the bundle for building a contents list

  LockingHandle<Bundle> getbundle(std::string name) { return(get<Bundle>(name)); }
  void setbundle(std::string name, LockingHandle<Bundle> &bundle) { set<Bundle>(name,bundle); }
  void rembundle(std::string name) { rem(name); }
  bool isbundle(std::string name)  { return(is<Bundle>(name)); }
  int  numbundles() { return(num<Bundle>()); }
  std::string getbundlename(int index) {return(getname<Bundle>(index)); }
 
 int            getnumhandles() { return(static_cast<int>(bundle_.size())); }
 std::string    gethandlename(int index) { return(bundlename_[static_cast<size_t>(index)]); }
 LockingHandle<PropertyManager> gethandle(int index) { return(bundle_[static_cast<size_t>(index)]); }
 std::string    gethandletype(int index);
 
private:

  int findname(std::deque<std::string> &deq, std::string name);
  int cmp_nocase(const std::string &s1,const std::string &s2);

  template<class PTYPE> inline bool NrrdTOMatrixHelper(NrrdDataHandle dataH, MatrixHandle& matH);
  bool NrrdTOMatrixConvertible(NrrdDataHandle nrrdH);
  bool NrrdTOMatrix(NrrdDataHandle dataH,MatrixHandle& matH);
  bool MatrixTONrrdConvertible(MatrixHandle matH);
  bool MatrixTONrrd(MatrixHandle matH,NrrdDataHandle &nrrdH);

private:

  std::deque<std::string> bundlename_;
  std::deque<LockingHandle<PropertyManager> > bundle_;
  
  bool transposenrrd;

  
};


inline void Bundle::transpose_nrrd(bool transpose)
{
    transposenrrd = transpose;
}

typedef LockingHandle<Bundle> BundleHandle;

template<class T> inline LockingHandle<T> Bundle::get(std::string name)
{
	int index;
	index = findname(bundlename_,name);
	if (index == -1) return(0);
    LockingHandle<T> handle = dynamic_cast<T*>(bundle_[index].get_rep());
	return(handle);
}

template<class T> inline void Bundle::set(std::string name, LockingHandle<T> &handle)
{
	int index;
	index = findname(bundlename_,name);
	if (index == -1)
	{
        LockingHandle<PropertyManager> lhandle = dynamic_cast<PropertyManager*>(handle.get_rep());
		bundle_.push_back(lhandle);
		bundlename_.push_back(name);
	}
	else
	{
		bundle_[index] = dynamic_cast<PropertyManager*>(handle.get_rep());
		bundlename_[index] = name;
	}
}

template<class T> inline bool Bundle::is(std::string name)
{
    int index;
	if ((index = findname(bundlename_,name)) > -1)
    {
        if (dynamic_cast<T*>(bundle_[index].get_rep()) != 0) return(true);
    }
	return(false);
}

template<class T> inline int  Bundle::num()
{
    int cnt = 0;
    for (size_t p=0;p<bundlename_.size();p++)
         if (dynamic_cast<T*>(bundle_[p].get_rep()) != 0) cnt++;
	return(cnt);
}

template<class T> inline std::string Bundle::getname(int index)
{
    int cnt = 0;
    size_t p;
    for (p=0;(p<bundlename_.size())&&(cnt != index);p++)
         if (dynamic_cast<T*>(bundle_[p].get_rep()) != 0) cnt++;
    if (p < bundlename_.size()) return(bundlename_[p]);
    return(std::string(""));
}

inline void Bundle::rem(std::string name)
{
	int index;
	index = findname(bundlename_,name);
	if (index > -1)
	{
		bundle_.erase(bundle_.begin()+index);
		bundlename_.erase(bundlename_.begin()+index);
    }
}

template<class PTYPE> inline bool Bundle::NrrdTOMatrixHelper(NrrdDataHandle dataH, MatrixHandle& matH)
{
    if (dataH->nrrd->dim == 1)
    {
        int cols = dataH->nrrd->axis[0].size;

        ColumnMatrix* matrix = scinew ColumnMatrix(cols);
  
        PTYPE *val = (PTYPE*)dataH->nrrd->data;
        double *data = matrix->get_data();

        for(int c=0; c<cols; c++) 
        {
            *data = *val;
            data++;
            val++;
        }
        matH = matrix;
        return(true);
    }

    if (dataH->nrrd->dim == 2)
    {
        if (transposenrrd)
        {
            int rows = dataH->nrrd->axis[1].size;
            int cols = dataH->nrrd->axis[0].size;

            DenseMatrix* matrix = scinew DenseMatrix(rows,cols);
      
            PTYPE *val = (PTYPE*)dataH->nrrd->data;
            double *data = matrix->getData();

            int i,j;
            i = 0; j = 0;
            for(int r=0; r<rows; r++) 
            {
                for(int c=0; c<cols; c++) 
                {
                    i = c + cols*r;
                    data[j++] = val[i];
                }
            }
            matH = matrix;
        }
        else
        {
            int cols = dataH->nrrd->axis[1].size;
            int rows = dataH->nrrd->axis[0].size;

            DenseMatrix* matrix = scinew DenseMatrix(cols,rows);
      
            PTYPE *val = (PTYPE*)dataH->nrrd->data;
            double *data = matrix->getData();
            
            for(int c=0; c<cols; c++) 
            {
            for(int r=0; r<rows; r++) 
            {
                    *data++ = *val++;
                }
            }
            matH = matrix;
        
        }
        return(true);
    }
    // Improper dimensions
    return(false);
}



} // end namespace 
#endif 
