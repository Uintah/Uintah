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
#include <Core/Datatypes/String.h>
#include <Core/Datatypes/NrrdData.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Datatypes/DenseMatrix.h>

#include <sgi_stl_warnings_off.h>
#include <deque>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {

class Bundle : public PropertyManager {
  
  public:  
  
    //! Constructor
    Bundle();
    
    //! Destructor
    virtual ~Bundle();

    //! SCIRun's way of copying
    virtual Bundle* clone();

    //! For writning bundles to file
    virtual void io(Piostream&);
    static PersistentTypeID type_id;

    //! Basic functionality
      
    //! Get handle to an object from the bundle
    template<class T> inline LockingHandle<T> get(std::string name);
    
    //! Add or replace an object in the bundle
    template<class T> inline void set(std::string name, 
                                        LockingHandle<T> &handle);
    
    //! Check whether an object is present 
    template<class T> inline bool is(std::string name);
    
    //! Get the number of objects of a certain type
    template<class T> inline int  num();
    
    //! Get the name of an object
    template<class T> inline std::string getName(int index);

    //! Remove object
    inline void rem(std::string name);
    
    //! Merge two bundles together
    void merge(SCIRun::LockingHandle<Bundle> C);

    //! Transpose when doing a matrix to nrrd conversion
    void transposeNrrd(bool on);

    // The basic functions for managing fields
    //!  getfield     -> Retrieve a Handle to a field stored in the bundle
    //!  setfield     -> Add a field with a name, if it already exists the 
    //!                  old one is overwritten
    //!  remfield     -> Remove a handle from the bundle
    //!  isfield      -> Test whether a field is present in the bundle
    //!  numfields    -> The number of fields stored in the bundle 
    //!  getfieldname -> Get the nth name in the bundle for building a contents 
    //!                  list
  
    LockingHandle<Field> getField(std::string name) 
      { return(get<Field>(name)); }
      
    void setField(std::string name, LockingHandle<Field> &field) 
      { set<Field>(name,field); }
      
    void remField(std::string name)
      { rem(name); }
      
    bool isField(std::string name)  
      { return(is<Field>(name)); }
      
    int  numFields() 
      { return(num<Field>()); }
          
    std::string getFieldName(int index) 
      { return(getName<Field>(index)); }
  
    //! The basic functions for managing matrices
    //!  getmatrix     -> Retrieve a Handle to a matrix stored in the bundle
    //!  setmatrix     -> Add a matrix with a name, if it already exists the old 
    //!                   one is overwritten
    //!  remmatrix     -> Remove a handle from the bundle
    //!  ismatrix      -> Test whether a matrix is present in the bundle
    //!  nummatrices   -> The number of matrices stored in the bundle 
    //!  getmatrixname -> Get the nth name in the bundle for building a contents 
    //!                   list
  
    LockingHandle<Matrix> getMatrix(std::string name);
      // Implementation in cc file, with NRRD/MATRIX compatibility
      
    void setMatrix(std::string name, LockingHandle<Matrix> &matrix)
      { set<Matrix>(name,matrix); }
    
    void remMatrix(std::string name) 
      { rem(name); }
    
    bool isMatrix(std::string name); 
      // Implementation in cc file, with NRRD/MATRIX compatibility
    
    int  numMatrices();
      // Implementation in cc file, with NRRD/MATRIX compatibility
    
    std::string getMatrixName(int index);
      // Implementation in cc file, with NRRD/MATRIX compatibility

    //! The basic functions for managing matrices
    //!  getstring     -> Retrieve a Handle to a matrix stored in the bundle
    //!  setstring     -> Add a matrix with a name, if it already exists the old
    //!                   one is overwritten
    //!  remstring     -> Remove a handle from the bundle
    //!  isstring      -> Test whether a matrix is present in the bundle
    //!  numstrings    -> The number of matrices stored in the bundle 
    //!  getstringname -> Get the nth name in the bundle for building a contents 
    //!                   list
  
    LockingHandle<String> getString(std::string name) 
      { return(get<String>(name)); }
      
    void setString(std::string name, LockingHandle<String> &str) 
      { set<String>(name,str); }
      
    void remString(std::string name) 
      { rem(name); }
    
    bool isString(std::string name)  
      { return(is<String>(name)); }
    
    int  numStrings() 
      { return(num<String>()); }
    
    std::string getStringName(int index) 
      { return(getName<String>(index)); }


    //! The basic functions for managing nrrds
    //!  getnrrd     -> Retrieve a Handle to a matrix stored in the bundle
    //!  setnrrd     -> Add a nrrd with a name, if it already exists the old one 
    //!                 is overwritten
    //!  remnrrd     -> remove a handle from the bundle
    //!  isnrrd      -> Test whether a nrrd is present in the bundle
    //!  numnrrds    -> The number of nrrds stored in the bundle 
    //!  getnrrdname -> Get the nth name in the bundle for building a contents 
    //!                 list
  
    LockingHandle<NrrdData> getNrrd(std::string name);
      // Implementation in cc file, with NRRD/MATRIX compatibility    
    
    void setNrrd(std::string name, LockingHandle<NrrdData> &nrrd) 
      { set<NrrdData>(name,nrrd); }

    void remNrrd(std::string name) 
      { rem(name); }
      
    bool isNrrd(std::string name);
      // Implementation in cc file, with NRRD/MATRIX compatibility

    int  numNrrds();
      // Implementation in cc file, with NRRD/MATRIX compatibility
    
    std::string getNrrdName(int index);
      // Implementation in cc file, with NRRD/MATRIX compatibility

//    NrrdScalar getNrrdScalar(std::string name) {return NrrdScalar(getNrrd(name)); }
//    NrrdString getNrrdString(std::string name) {return NrrdString(getNrrd(name)); }  
//    void setNrrdScalar(std::string name, NrrdScalar scalar) { NrrdDataHandle nrrd = scalar.gethandle(); set<NrrdData>(name,nrrd); }
//    void setNrrdString(std::string name, NrrdString string) { NrrdDataHandle nrrd = string.gethandle(); set<NrrdData>(name,nrrd); }

  
    //! The basic functions for managing colormaps
    //!  getcolormap     -> Retrieve a Handle to a colormap stored in the bundle
    //!  setcolormap     -> Add a colormap with a name, if it already exists the
    //!                     old one is overwritten
    //!  remcolormap     -> Remove a handle from the bundle
    //!  iscolormap      -> Test whether a colormap is present in the bundle
    //!  numcolormaps    -> The number of colormaps stored in the bundle 
    //!  getcolormapname -> Get the nth name in the bundle for building a 
    //!                     contents list

    LockingHandle<ColorMap> getColormap(std::string name) 
      { return(get<ColorMap>(name)); }
    
    void setColormap(std::string name, LockingHandle<ColorMap> &colormap) 
      { set<ColorMap>(name,colormap); }
    
    void remColormap(std::string name) 
      { rem(name); }
    
    bool isColormap(std::string name)  
      { return(is<ColorMap>(name)); }
      
    int numColormaps() 
      { return(num<ColorMap>()); }
      
    std::string getColormapName(int index) 
      { return(getName<ColorMap>(index)); }

    //! The basic functions for managing colormap2s
    //!  getcolormap2     -> Retrieve a Handle to a colormap2 stored in the
    //!                      bundle
    //!  setcolormap2     -> Add a colormap2 with a name, if it already exists
    //!                      the old one is overwritten
    //!  remcolormap2     -> Remove a handle from the bundle
    //!  iscolormap2      -> Test whether a colormap2 is present in the bundle
    //!  numcolormap2s    -> The number of colormap2s stored in the bundle 
    //!  getcolormap2name -> Get the nth name in the bundle for building a
    //!                      contents list

    LockingHandle<ColorMap2> getColormap2(std::string name) 
      { return(get<ColorMap2>(name)); }
      
    void setColormap2(std::string name, LockingHandle<ColorMap2> &colormap2) 
      { set<ColorMap2>(name,colormap2); }
      
    void remColormap2(std::string name) 
      { rem(name); }
      
    bool isColormap2(std::string name)  
      { return(is<ColorMap2>(name)); }
      
    int  numColormap2s() 
      { return(num<ColorMap2>()); }
      
    std::string getColormap2Name(int index) 
      { return(getName<ColorMap2>(index)); }

    //! The basic functions for managing paths
    //!  getpath     -> Retrieve a Handle to a path stored in the bundle
    //!  setpath     -> Add a path with a name, if it already exists the old one 
    //!                 is overwritten
    //!  rempath     -> Remove a handle from the bundle
    //!  ispath      -> Test whether a path is present in the bundle
    //!  numpaths    -> The number of paths stored in the bundle 
    //!  getpathname -> Get the nth name in the bundle for building a contents 
    //!                 list

    LockingHandle<Path> getPath(std::string name) 
      { return(get<Path>(name)); }
      
    void setPath(std::string name, LockingHandle<Path> &path) 
      { set<Path>(name,path); }
      
    void remPath(std::string name) 
      { rem(name); }
      
    bool isPath(std::string name)  
      { return(is<Path>(name)); }
      
    int  numPaths() 
      { return(num<Path>()); }
    
    std::string getPathName(int index) 
      {return(getName<Path>(index)); }
  
    //! The basic functions for managing bundles
    //!  getbundle     -> Retrieve a Handle to a bundle stored in the bundle
    //!  setbundle     -> Add a bundle with a name, if it already exists the old
    //!                   one is overwritten
    //!  rembundle     -> Remove a handle from the bundle
    //!  isbundle      -> Test whether a bundle is present in the bundle
    //!  numbundles    -> The number of bundles stored in the bundle 
    //!  getbundleName -> Get the nth name in the bundle for building a contents 
    //!                   list

    LockingHandle<Bundle> getBundle(std::string name) 
      { return(get<Bundle>(name)); }
      
    void setBundle(std::string name, LockingHandle<Bundle> &bundle) 
      { set<Bundle>(name,bundle); }
      
    void remBundle(std::string name) 
      { rem(name); }
      
    bool isBundle(std::string name)  
      { return(is<Bundle>(name)); }
    
    int  numBundles() 
      { return(num<Bundle>()); }
      
    std::string getBundleName(int index) 
      { return(getName<Bundle>(index)); }
 
    
    //! Get the number of elements in the Bundle
    int getNumHandles() 
      { return static_cast<int>(bundle_.size()); }
    
    //! Get the name of the handles in the bundle     
    std::string getHandleName(int index) 
      { return bundleName_[static_cast<size_t>(index)]; }
      
    //! Get one of the handles in the bundle  
    LockingHandle<PropertyManager> gethandle(int index) 
      { return bundle_[static_cast<size_t>(index)]; }
    
    //! Get the type of a handle (for BundleInfo)
    std::string    getHandleType(int index);
 
  private:
    
    //! find the handle in the bundle
    int findName(std::deque<std::string> &deq, std::string name);
    
    //! Compare two strings 
    int cmp_nocase(const std::string &s1,const std::string &s2);

    //! Functions for doing NRRD/MATRIX conversion
    template<class PTYPE> 
      inline bool NrrdToMatrixHelper(NrrdDataHandle dataH, MatrixHandle& matH);
    
    bool NrrdToMatrixConvertible(NrrdDataHandle nrrdH);
    bool NrrdToMatrix(NrrdDataHandle dataH,MatrixHandle& matH);
    bool MatrixToNrrdConvertible(MatrixHandle matH);
    bool MatrixToNrrd(MatrixHandle matH,NrrdDataHandle &nrrdH);

    //! The names of the elements in the bundle
    std::deque<std::string> bundleName_;
    
    //! An array with handle to all the objects in the bundle
    std::deque<LockingHandle<PropertyManager> > bundle_;
  
    //! Setting for the converion between NRRD/MATRIX
    bool transposeNrrd_;
  
};

typedef LockingHandle<Bundle> BundleHandle;

inline void Bundle::transposeNrrd(bool transpose)
{
  transposeNrrd_ = transpose;
}


template<class T> inline LockingHandle<T> Bundle::get(std::string name)
{
  int index;
  index = findName(bundleName_,name);
  if (index == -1) return 0;
  LockingHandle<T> handle = dynamic_cast<T*>(bundle_[index].get_rep());
  return(handle);
}


template<class T> inline void Bundle::set(std::string name,
                                              LockingHandle<T> &handle)
{
  int index;
  index = findName(bundleName_,name);
  if (index == -1)
  {
    LockingHandle<PropertyManager> lhandle = 
                            dynamic_cast<PropertyManager*>(handle.get_rep());
    bundle_.push_back(lhandle);
    bundleName_.push_back(name);
  }
  else
  {
    bundle_[index] = dynamic_cast<PropertyManager*>(handle.get_rep());
    bundleName_[index] = name;
  }
}


template<class T> inline bool Bundle::is(std::string name)
{
  int index;
  if ((index = findName(bundleName_,name)) > -1)
  {
    if (dynamic_cast<T*>(bundle_[index].get_rep()) != 0) return(true);
  }
  return(false);
}


template<class T> inline int Bundle::num()
{
  int cnt = 0;
  for (size_t p=0;p<bundleName_.size();p++)
    if (dynamic_cast<T*>(bundle_[p].get_rep()) != 0) cnt++;
  return(cnt);
}


template<class T> inline std::string Bundle::getName(int index)
{
  int cnt = -1;
  size_t p;
  for (p=0;p<bundleName_.size();p++)
  {
    if (dynamic_cast<T*>(bundle_[p].get_rep()) != 0) cnt++;
    if (index == cnt) break;
  }
  if ((p < bundleName_.size())&&(cnt==index)) return bundleName_[p];
  return(std::string(""));
}


inline void Bundle::rem(std::string name)
{
  int index;
  index = findName(bundleName_,name);
  if (index > -1)
  {
    bundle_.erase(bundle_.begin()+index);
    bundleName_.erase(bundleName_.begin()+index);
  }
}

  
template<class PTYPE> inline bool Bundle::NrrdToMatrixHelper(
                                    NrrdDataHandle dataH, MatrixHandle& matH)
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
    if (transposeNrrd_)
      {
        int rows = dataH->nrrd->axis[1].size;
        int cols = dataH->nrrd->axis[0].size;

        DenseMatrix* matrix = scinew DenseMatrix(rows,cols);
  
        PTYPE *val = (PTYPE*)dataH->nrrd->data;
        double *data = matrix->get_data_pointer();

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
      double *data = matrix->get_data_pointer();
      
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

} // end namespace SCIRun

#endif 
