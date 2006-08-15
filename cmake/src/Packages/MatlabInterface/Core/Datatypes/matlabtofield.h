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

// THIS FILE CONTAINS MOST OF THE FUNCTIONALITY TO MAP A SCIRUN::FIELD OBJECT
// INTO A MATLAB ARRAY
//
// UNFORTUNATELY SCIRUN::FIELD OBJECT ARE NOT FULLY POLYMORPHIC, MEANING SOME
// FUNCTIONALITY IS MISSING IN CERTAIN CLASSES AND DATA STRUCTURES HAVE BEEN
// NAMED DIFFERENT OVER THE VARIOUS FIELDS.
//
// HENCE THIS CONVERTER IS HUGE AND HAS SPECIFIC CODE SNIPPITS FOR EVERY WEIRD
// SCIRUN DEFINITION. 
//
// THE CONVERTER IS COMPLETELY TEMPLATED AND USES TEMPLATE OVER LOADING TO DIRECT
// THE COMPILER TO INCLUDE TO PROPER PIECES OF CODE AT EACH POSITION. UNLIKE 
// MOST SCIRUN CODE, IT ONLY DOES ONE DYNAMIC COMPILATION AND THAN RELIES ON
// OVERLOADING TEMPALTED FUNCTIONS TO DEFINE A SPECIFIC CONVERTER.
// THE ADVANTAGE OF THIS METHODOLOGY IS THAT PIECES OF CODE CAN BE REUSED AND NOT
// EVERY MESH TYPE NEEDS SPECIALLY DESIGNED CODE.
// 
// THE CURRENT SYSTEM NEEDS A SPECIFIC OVERLOADED FUNCTION FOR EACH MESH TYPE WHICH
// TELLS WHICH PIECES TOGETHER FORM A NEW CONVERTER. IF THE COMPILER IS SMART ENOUGH
// IT ONLY COMPILES THE PIECES IT NEEDS. SINCE IT IS ALL TEMPLATED, THE COMPILER
// CANNOT INSTANTIATE EVERY PIECE, ALTHOUGH NEWER COMPILERS LIKE GCC4, WILL CHECK
// THE CODE EVEN IF IT IS NOT USED.....


// STL STUFF
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <string>
#include <algorithm>
#include <cctype>
#include <sgi_stl_warnings_on.h>

// Class for reading matlab files
#include <Packages/MatlabInterface/Core/Datatypes/matlabfile.h>
#include <Packages/MatlabInterface/Core/Datatypes/matlabarray.h>

// FData classes
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Tensor.h>


// Basis classes
#include <Core/Basis/NoData.h>
#include <Core/Basis/Constant.h>
#include <Core/Basis/CrvLinearLgn.h>
#include <Core/Basis/CrvQuadraticLgn.h>
#include <Core/Basis/HexTricubicHmt.h>
#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Basis/HexTriquadraticLgn.h>
#include <Core/Basis/PrismCubicHmt.h>
#include <Core/Basis/PrismLinearLgn.h>
#include <Core/Basis/PrismQuadraticLgn.h>
#include <Core/Basis/QuadBicubicHmt.h>
#include <Core/Basis/QuadBilinearLgn.h>
#include <Core/Basis/QuadBiquadraticLgn.h>
#include <Core/Basis/TetCubicHmt.h>
#include <Core/Basis/TetLinearLgn.h>
#include <Core/Basis/TetQuadraticLgn.h>
#include <Core/Basis/TriCubicHmt.h>
#include <Core/Basis/TriLinearLgn.h>
#include <Core/Basis/TriQuadraticLgn.h>

// MeshClasses
#include <Core/Datatypes/Mesh.h>
#include <Core/Datatypes/PointCloudMesh.h>
#include <Core/Datatypes/CurveMesh.h>
#include <Core/Datatypes/ImageMesh.h>
#include <Core/Datatypes/StructCurveMesh.h>
#include <Core/Datatypes/TriSurfMesh.h>
#include <Core/Datatypes/QuadSurfMesh.h>
#include <Core/Datatypes/StructQuadSurfMesh.h>
#include <Core/Datatypes/TetVolMesh.h>
#include <Core/Datatypes/PrismVolMesh.h>
#include <Core/Datatypes/StructHexVolMesh.h>
#include <Core/Datatypes/HexVolMesh.h>
#include <Core/Datatypes/LatVolMesh.h>

#include <Core/Containers/FData.h>

// Field class files
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/GenericField.h>



/*
 * FILE: matlabconverter_fieldtomatlab.h
 * AUTH: Jeroen G Stinstra
 * DATE: 18 MAR 2004
 */
 
#ifndef JGS_MATLABIO_MATLABCONVERTER_MATLABTOFIELD_H
#define JGS_MATLABIO_MATLABCONVERTER_MATLABTOFIELD_H 1

namespace MatlabIO {

class MatlabToFieldAlgo : public SCIRun::DynamicAlgoBase, public matfilebase
{
  public:

    //////// CONSTRUCTOR ///////////////////////////////////
    inline MatlabToFieldAlgo();

    //////// DYNAMIC ALGO ENTRY POINT /////////////////////

    virtual bool execute(SCIRun::FieldHandle& fieldhandle, matlabarray& mlarray);
    static SCIRun::CompileInfoHandle get_compile_info(std::string fielddesc);

    //////// ANALYZE INPUT FUNCTIONS //////////////////////

    int analyze_iscompatible(matlabarray mlarray, std::string& infotext, bool postremark = true);
    int analyze_fieldtype(matlabarray mlarray, std::string& fielddesc);

    inline void setreporter(SCIRun::ProgressReporter* pr);
    
  protected:

    int mlanalyze(matlabarray mlarray, bool postremark);  
    matlabarray findfield(matlabarray mlarray,std::string fieldnames);
  
    matlabarray mlnode; 
    matlabarray mledge;
    matlabarray mlface;
    matlabarray mlcell;

    matlabarray mlmeshderivatives; 
    matlabarray mlmeshscalefactors; 

    matlabarray mlx;
    matlabarray mly;
    matlabarray mlz;

    matlabarray mldims;
    matlabarray mltransform;

    matlabarray mlmeshbasis;
    matlabarray mlmeshbasisorder;
    matlabarray mlmeshtype;

    // Field description classes
    matlabarray mlfield;
    matlabarray mlfieldedge;
    matlabarray mlfieldface;
    matlabarray mlfieldcell;

    matlabarray mlfieldderivatives; 
    matlabarray mlfieldscalefactors; 

    matlabarray mlfieldbasis;
    matlabarray mlfieldbasisorder;
    matlabarray mlfieldtype;

    matlabarray mlchannels;

    std::string fdatatype;
    std::string fieldtype;
    std::string meshtype;
    std::string fieldbasis;
    std::string meshbasis;
    
    std::string meshbasistype;
    std::string fieldbasistype;

    std::vector<int> numnodesvec;
    std::vector<int> numelementsvec;
    
    int              numnodes;
    int              numelements;
    int              numfield;
    int              datasize;

    //////// FUNCTIONS FOR BUIDLIGN THE MESH //////////////

  protected:

    template <class MESH>  bool createmesh(SCIRun::LockingHandle<MESH>& handle);
    template <class BASIS> bool createmesh(SCIRun::LockingHandle<SCIRun::PointCloudMesh<BASIS> >& handle);
    template <class BASIS> bool createmesh(SCIRun::LockingHandle<SCIRun::CurveMesh<BASIS> >& handle);
    template <class BASIS> bool createmesh(SCIRun::LockingHandle<SCIRun::TriSurfMesh<BASIS> >& handle);
    template <class BASIS> bool createmesh(SCIRun::LockingHandle<SCIRun::QuadSurfMesh<BASIS> >& handle);
    template <class BASIS> bool createmesh(SCIRun::LockingHandle<SCIRun::TetVolMesh<BASIS> >& handle);
    template <class BASIS> bool createmesh(SCIRun::LockingHandle<SCIRun::PrismVolMesh<BASIS> >& handle);
    template <class BASIS> bool createmesh(SCIRun::LockingHandle<SCIRun::HexVolMesh<BASIS> >& handle);
    template <class BASIS> bool createmesh(SCIRun::LockingHandle<SCIRun::ScanlineMesh<BASIS> >& handle);
    template <class BASIS> bool createmesh(SCIRun::LockingHandle<SCIRun::ImageMesh<BASIS> >& handle);
    template <class BASIS> bool createmesh(SCIRun::LockingHandle<SCIRun::LatVolMesh<BASIS> >& handle);
    template <class BASIS> bool createmesh(SCIRun::LockingHandle<SCIRun::StructCurveMesh<BASIS> >& handle);
    template <class BASIS> bool createmesh(SCIRun::LockingHandle<SCIRun::StructQuadSurfMesh<BASIS> >& handle);
    template <class BASIS> bool createmesh(SCIRun::LockingHandle<SCIRun::StructHexVolMesh<BASIS> >& handle);

    template <class MESH> bool addtransform(SCIRun::LockingHandle<MESH>& handle);
    template <class MESH> bool addderivatives(SCIRun::LockingHandle<MESH>& handle);
    template <class MESH> bool addscalefactors(SCIRun::LockingHandle<MESH>& handle);    
    template <class MESH> bool addnodes(SCIRun::LockingHandle<MESH>& handle);
    template <class MESH> bool addedges(SCIRun::LockingHandle<MESH>& handle);
    template <class MESH> bool addfaces(SCIRun::LockingHandle<MESH>& handle);
    template <class MESH> bool addcells(SCIRun::LockingHandle<MESH>& handle);

    template <class T> bool addfield(T &fdata);
    template <class T> bool addfield(std::vector<T> &fdata);
    template <class T, class MESH> bool addfield(SCIRun::FData2d<T,MESH> &fdata);
    template <class T, class MESH> bool addfield(SCIRun::FData3d<T,MESH> &fdata);

    bool addfield(std::vector<SCIRun::Vector> &fdata);
    template <class MESH> bool addfield(SCIRun::FData2d<SCIRun::Vector,MESH> &fdata);
    template <class MESH> bool addfield(SCIRun::FData3d<SCIRun::Vector,MESH> &fdata);
    bool addfield(std::vector<SCIRun::Tensor> &fdata);
    template <class MESH> bool addfield(SCIRun::FData2d<SCIRun::Tensor,MESH> &fdata);
    template <class MESH> bool addfield(SCIRun::FData3d<SCIRun::Tensor,MESH> &fdata);

    inline void uncompressedtensor(std::vector<double> &fielddata,SCIRun::Tensor &tens, unsigned int p);
    inline void compressedtensor(std::vector<double> &fielddata,SCIRun::Tensor &tens, unsigned int p);	

    //////// ERROR REPORTERING MECHANISM /////////////////

  protected:
    
    inline void error(std::string error);
    inline void warning(std::string warning);
    inline void remark(std::string remark);
    
  private:
    SCIRun::ProgressReporter *pr_;
    
};

inline MatlabToFieldAlgo::MatlabToFieldAlgo() :
  pr_(0)
{
}

inline void MatlabToFieldAlgo::setreporter(SCIRun::ProgressReporter* pr)
{
  pr_ = pr;
}

inline void MatlabToFieldAlgo::error(std::string error)
{
  if(pr_) pr_->error(error);
}

inline void MatlabToFieldAlgo::warning(std::string warning)
{
  if(pr_) pr_->warning(warning);
}

inline void MatlabToFieldAlgo::remark(std::string remark)
{
  if(pr_) pr_->remark(remark);
}

///// DYNAMIC ALGORITHM STARTS HERE ///////////////////

template <class FIELD> 
class MatlabToFieldAlgoT : public MatlabToFieldAlgo
{
  public:
    //////// DYNAMIC ALGO ENTRY POINT /////////////////////
    virtual bool execute(SCIRun::FieldHandle& fieldhandle, matlabarray &mlarray);
};

template <class FIELD>  
bool MatlabToFieldAlgoT<FIELD>::execute(SCIRun::FieldHandle& fieldhandle, matlabarray &mlarray)
{
  // Create the type of mesh that needs to be generated
  SCIRun::LockingHandle<typename FIELD::mesh_type> meshhandle;

  if (!(createmesh(meshhandle)))
  {
    error("Error occured while generating mesh");
    return (false);
  }

  FIELD *field = scinew FIELD(meshhandle);
  
  if (field == 0)
  {
    error("Error occured while generating field");
    return (false);  
  }
  
  fieldhandle = dynamic_cast<SCIRun::Field *>(field);

  if (fieldhandle.get_rep() == 0)
  {
    error("Error occured while generating field");
    return (false);  
  }

  if (fieldbasistype == "constant")
  {
    field->resize_fdata();
    typename FIELD::fdata_type& fdata = field->fdata();
    if (!(addfield(fdata)))
    {
      error("The conversion of the field data failed");
      return (false);    
    }    
  }

  if (fieldbasistype == "linear")
  {
    field->resize_fdata();
    typename FIELD::fdata_type& fdata = field->fdata();
    if (!(addfield(fdata)))
    {
      error("The conversion of the field data failed");
      return (false);    
    }
  }

  if (fieldbasistype == "quadratic")
  {
    error("There is no converter available for quadratic field data");
    return (false);
  }

  if (fieldbasistype == "cubic")
  {
    error("There is no converter available for cubic field data");
    return (false);
  }

  return (true);
}


template <class MESH>
bool MatlabToFieldAlgo::createmesh(SCIRun::LockingHandle<MESH>& handle)
{
  error("There is no converter available for this kind of mesh");
  return (false);
}

template <class BASIS> 
bool MatlabToFieldAlgo::createmesh(SCIRun::LockingHandle<SCIRun::PointCloudMesh<BASIS> >& handle)
{
  handle = dynamic_cast<SCIRun::PointCloudMesh<BASIS>* >(scinew SCIRun::PointCloudMesh<BASIS>);
  if(!(addnodes(handle))) return (false);
  if(!(addderivatives(handle))) return (false);
  if(!(addscalefactors(handle))) return (false);
  
  return (true);
}

template <class BASIS> 
bool MatlabToFieldAlgo::createmesh(SCIRun::LockingHandle<SCIRun::CurveMesh<BASIS> >& handle)
{
  handle = dynamic_cast<SCIRun::CurveMesh<BASIS>* >(scinew SCIRun::CurveMesh<BASIS>);
  if(!(addnodes(handle))) return (false);
  if(!(addedges(handle))) return (false);
  if(!(addderivatives(handle))) return (false);
  if(!(addscalefactors(handle))) return (false);

  return (true);
}

template <class BASIS> 
bool MatlabToFieldAlgo::createmesh(SCIRun::LockingHandle<SCIRun::TriSurfMesh<BASIS> >& handle)
{
  handle = dynamic_cast<SCIRun::TriSurfMesh<BASIS>* >(scinew SCIRun::TriSurfMesh<BASIS>);
  if(!(addnodes(handle))) return (false);
  if(!(addfaces(handle))) return (false);
  if(!(addderivatives(handle))) return (false);
  if(!(addscalefactors(handle))) return (false);
  
  return (true);
}

template <class BASIS> 
bool MatlabToFieldAlgo::createmesh(SCIRun::LockingHandle<SCIRun::QuadSurfMesh<BASIS> >& handle)
{
  handle = dynamic_cast<SCIRun::QuadSurfMesh<BASIS>* >(scinew SCIRun::QuadSurfMesh<BASIS>);
  if(!(addnodes(handle))) return (false);
  if(!(addfaces(handle))) return (false);
  if(!(addderivatives(handle))) return (false);
  if(!(addscalefactors(handle))) return (false);
  
  return (true);
}

template <class BASIS> 
bool MatlabToFieldAlgo::createmesh(SCIRun::LockingHandle<SCIRun::TetVolMesh<BASIS> >& handle)
{
  handle = dynamic_cast<SCIRun::TetVolMesh<BASIS>* >(scinew SCIRun::TetVolMesh<BASIS>);
  if(!(addnodes(handle))) return (false);
  if(!(addcells(handle))) return (false);
  if(!(addderivatives(handle))) return (false);
  if(!(addscalefactors(handle))) return (false);  
  return (true);
}

template <class BASIS> 
bool MatlabToFieldAlgo::createmesh(SCIRun::LockingHandle<SCIRun::PrismVolMesh<BASIS> >& handle)
{
  handle = dynamic_cast<SCIRun::PrismVolMesh<BASIS>* >(scinew SCIRun::PrismVolMesh<BASIS>);
  if(!(addnodes(handle))) return (false);
  if(!(addcells(handle))) return (false);
  if(!(addderivatives(handle))) return (false);
  if(!(addscalefactors(handle))) return (false);  
  return (true);
}

template <class BASIS> 
bool MatlabToFieldAlgo::createmesh(SCIRun::LockingHandle<SCIRun::HexVolMesh<BASIS> >& handle)
{
  handle = dynamic_cast<SCIRun::HexVolMesh<BASIS>* >(scinew SCIRun::HexVolMesh<BASIS>);
  if(!(addnodes(handle))) return (false);
  if(!(addcells(handle))) return (false);
  if(!(addderivatives(handle))) return (false);
  if(!(addscalefactors(handle))) return (false);  
  return (true);
}

template <class BASIS> 
bool MatlabToFieldAlgo::createmesh(SCIRun::LockingHandle<SCIRun::ScanlineMesh<BASIS> >& handle)
{
  if (!(mldims.isdense()))
  {
    return (false);
  }
  
  if (mldims.getnumelements() != 1)
  {
    return (false);
  }
  
  std::vector<int> dims; 
  mldims.getnumericarray(dims);

  SCIRun::Point PointO(0.0,0.0,0.0);
  SCIRun::Point PointP(static_cast<double>(dims[0]),0.0,0.0);

  handle = dynamic_cast<SCIRun::ScanlineMesh<BASIS>* >(scinew SCIRun::ScanlineMesh<BASIS>(static_cast<unsigned int>(dims[0]),PointO,PointP));
  addtransform(handle);

  return(true);
}

template <class BASIS> 
bool MatlabToFieldAlgo::createmesh(SCIRun::LockingHandle<SCIRun::ImageMesh<BASIS> >& handle)
{
  if (!(mldims.isdense()))
  {
    return (false);
  }

  if (mldims.getnumelements() != 2)
  {
    return (false);
  }

  std::vector<int> dims; 
  mldims.getnumericarray(dims);

  SCIRun::Point PointO(0.0,0.0,0.0);
  SCIRun::Point PointP(static_cast<double>(dims[0]),static_cast<double>(dims[1]),0.0);
  handle = dynamic_cast<SCIRun::ImageMesh<BASIS>* >(scinew SCIRun::ImageMesh<BASIS>(static_cast<unsigned int>(dims[0]),static_cast<unsigned int>(dims[1]),PointO,PointP));
  addtransform(handle);

  return(true);
}

template <class BASIS> 
bool MatlabToFieldAlgo::createmesh(SCIRun::LockingHandle<SCIRun::LatVolMesh<BASIS> >& handle)
{
  if (!(mldims.isdense()))
  {
    return (false);
  }

  if (mldims.getnumelements() != 3)
  {
    return (false);
  }

  std::vector<int> dims; 
  mldims.getnumericarray(dims);

  SCIRun::Point PointO(0.0,0.0,0.0);
  SCIRun::Point PointP(static_cast<double>(dims[0]),static_cast<double>(dims[1]),static_cast<double>(dims[2]));
  handle = dynamic_cast<SCIRun::LatVolMesh<BASIS>* >(scinew SCIRun::LatVolMesh<BASIS>(static_cast<unsigned int>(dims[0]),static_cast<unsigned int>(dims[1]),static_cast<unsigned int>(dims[2]),PointO,PointP));
  addtransform(handle);

  return(true);
}

template <class BASIS> bool MatlabToFieldAlgo::createmesh(SCIRun::LockingHandle<SCIRun::StructCurveMesh<BASIS> >& handle)
{
  std::vector<int> dims;
  std::vector<unsigned int> mdims;
  int numdim = mlx.getnumdims();
  dims = mlx.getdims();
        
  mdims.resize(numdim); 
  for (int p=0; p < numdim; p++)  mdims[p] = static_cast<unsigned int>(dims[p]); 
        
  if ((numdim == 2)&&(mlx.getn() == 1))
  {
    numdim = 1;
    mdims.resize(1);
    mdims[0] = mlx.getm();
  }

  handle = dynamic_cast<SCIRun::StructCurveMesh<BASIS>* >(scinew SCIRun::StructCurveMesh<BASIS>);
  int numnodes = mlx.getnumelements();
        
  std::vector<double> X;
  std::vector<double> Y;
  std::vector<double> Z;
  mlx.getnumericarray(X);
  mly.getnumericarray(Y);
  mlz.getnumericarray(Z);
        
  handle->set_dim(mdims);
  int p;
  for (p = 0; p < numnodes; p++)
  {
    handle->set_point(SCIRun::Point(X[p],Y[p],Z[p]),static_cast<typename SCIRun::StructCurveMesh<BASIS>::Node::index_type>(p));
  }
                                        
  return(true);
}

template <class BASIS> 
bool MatlabToFieldAlgo::createmesh(SCIRun::LockingHandle<SCIRun::StructQuadSurfMesh<BASIS> >& handle)
{
  std::vector<int> dims;
  std::vector<unsigned int> mdims;
  int numdim = mlx.getnumdims();
  dims = mlx.getdims();
        
  mdims.resize(numdim); 
  for (int p=0; p < numdim; p++)  mdims[p] = static_cast<unsigned int>(dims[p]); 

  handle = dynamic_cast<SCIRun::StructQuadSurfMesh<BASIS>* >(scinew SCIRun::StructQuadSurfMesh<BASIS>);
        
  std::vector<double> X;
  std::vector<double> Y;
  std::vector<double> Z;
  mlx.getnumericarray(X);
  mly.getnumericarray(Y);
  mlz.getnumericarray(Z);
        
  handle->set_dim(mdims);

  unsigned p,r,q;
  q = 0;
  for (r = 0; r < mdims[1]; r++)
    for (p = 0; p < mdims[0]; p++)
    {
      handle->set_point(SCIRun::Point(X[q],Y[q],Z[q]),typename SCIRun::StructQuadSurfMesh<BASIS>::Node::index_type(handle.get_rep(),p,r));
      q++;
    }
                                        
  return(true);
}

template <class BASIS> bool MatlabToFieldAlgo::createmesh(SCIRun::LockingHandle<SCIRun::StructHexVolMesh<BASIS> >& handle)
{
  std::vector<int> dims;
  std::vector<unsigned int> mdims;
  int numdim = mlx.getnumdims();
  dims = mlx.getdims();
        
  mdims.resize(numdim); 
  for (int p=0; p < numdim; p++)  mdims[p] = static_cast<unsigned int>(dims[p]); 

  handle = dynamic_cast<SCIRun::StructHexVolMesh<BASIS>* >(scinew SCIRun::StructHexVolMesh<BASIS>);
        
  std::vector<double> X;
  std::vector<double> Y;
  std::vector<double> Z;
  mlx.getnumericarray(X);
  mly.getnumericarray(Y);
  mlz.getnumericarray(Z);
        
  handle->set_dim(mdims);

  unsigned p,r,s,q;
  q= 0;
  for (s = 0; s < mdims[2]; s++)
    for (r = 0; r < mdims[1]; r++)
      for (p = 0; p < mdims[0]; p++)
      {
        handle->set_point(SCIRun::Point(X[q],Y[q],Z[q]),typename SCIRun::StructHexVolMesh<BASIS>::Node::index_type(handle.get_rep(),p,r,s));
        q++;
      }
        
  return(true);
}


template <class MESH> bool MatlabToFieldAlgo::addtransform(SCIRun::LockingHandle<MESH>& handle)
{
  if (mltransform.isdense())
  {
    SCIRun::Transform T;
    double trans[16];
    mltransform.getnumericarray(trans,16);
    T.set_trans(trans);
    handle->transform(T);
  }
  return(true);  
}


template <class MESH>
bool MatlabToFieldAlgo::addnodes(SCIRun::LockingHandle<MESH>& handle)
{
	// Get the data from the matlab file, which has been buffered
	// but whose format can be anything. The next piece of code
	// copies and casts the data
	
  if (meshbasistype == "quadratic")
  {
    error("The converter misses code to add quadratic nodes to mesh");
    return (false);
  }
  
	std::vector<double> mldata;
	mlnode.getnumericarray(mldata);
		
	// Again the data is copied but now reorganised into
	// a vector of Point objects
	
	int numnodes = mlnode.getn();	
	handle->node_reserve(numnodes);
	
	int p,q;
	for (p = 0, q = 0; p < numnodes; p++, q+=3)
	{ 
    handle->add_point(SCIRun::Point(mldata[q],mldata[q+1],mldata[q+2]));
  }
  
  return (true);
}

template <class MESH>
bool MatlabToFieldAlgo::addedges(SCIRun::LockingHandle<MESH>& handle)
{
	// Get the data from the matlab file, which has been buffered
	// but whose format can be anything. The next piece of code
	// copies and casts the data
	
  if (meshbasistype == "quadratic")
  {
    error("The converter misses code to add quadratic edges to mesh");
    return (false);
  }

	std::vector<unsigned int> mldata;
	mledge.getnumericarray(mldata);		
	
	// check whether it is zero based indexing 
	// In short if there is a zero it must be zero
	// based numbering right ??
	// If not we assume one based numbering
	
	int p,q;
	
	bool zerobased = false;  
	int size = static_cast<int>(mldata.size());
	for (p = 0; p < size; p++) { if (mldata[p] == 0) {zerobased = true; break;} }
	
	if (zerobased == false)
	{   // renumber to go from matlab indexing to C++ indexing
		for (p = 0; p < size; p++) { mldata[p]--;}
	}
	
  int m,n;
   m = mledge.getm();
   n = mledge.getn();
  
	handle->elem_reserve(n);
	
  typename MESH::Node::array_type edge(m); 
  
  int r;
  r = 0;
     
	for (p = 0, q = 0; p < n; p++)
	{
     for (int q = 0 ; q < m; q++)
     {
       edge[q] = mldata[r]; r++; 
     }
     
		handle->add_elem(edge);
	}

  return (true);
}

template <class MESH>
bool MatlabToFieldAlgo::addfaces(SCIRun::LockingHandle<MESH>& handle)
{
   // Get the data from the matlab file, which has been buffered
   // but whose format can be anything. The next piece of code
   // copies and casts the data

  if (meshbasistype == "quadratic")
  {
    error("The converter misses code to add quadratic edges to mesh");
    return (false);
  }
	
  std::vector<unsigned int> mldata;
  mlface.getnumericarray(mldata);		

  // check whether it is zero based indexing 
  // In short if there is a zero it must be zero
  // based numbering right ??
  // If not we assume one based numbering

  bool zerobased = false;  
  int size = static_cast<int>(mldata.size());
  for (int p = 0; p < size; p++) { if (mldata[p] == 0) {zerobased = true; break;} }

  if (zerobased == false)
  {   // renumber to go from matlab indexing to C++ indexing
    for (int p = 0; p < size; p++) { mldata[p]--;}
  }

  int m,n;
  m = mlface.getm();
  n = mlface.getn();

  handle->elem_reserve(n);	  
          
  typename MESH::Node::array_type face(m);  

  int r;
  r = 0;

  for (int p = 0; p < n; p++)
  {
    for (int q = 0 ; q < m; q++)
    {
      face[q] = mldata[r]; r++; 
    }
    handle->add_elem(face);
  }
  
  return (true);
}

template <class MESH>
bool MatlabToFieldAlgo::addcells(SCIRun::LockingHandle<MESH>& handle)
{
  // Get the data from the matlab file, which has been buffered
  // but whose format can be anything. The next piece of code
  // copies and casts the data

  if (meshbasistype == "quadratic")
  {
    error("The converter misses code to add quadratic edges to mesh");
    return (false);
  }
  
  std::vector<unsigned int> mldata;
  mlcell.getnumericarray(mldata);		

  // check whether it is zero based indexing 
  // In short if there is a zero it must be zero
  // based numbering right ??
  // If not we assume one based numbering

  bool zerobased = false;  
  int size = static_cast<int>(mldata.size());
  for (int p = 0; p < size; p++) { if (mldata[p] == 0) {zerobased = true; break;} }

  if (zerobased == false)
  {   // renumber to go from matlab indexing to C++ indexing
    for (int p = 0; p < size; p++) { mldata[p]--;}
  }

  int m,n;
  m = mlcell.getm();
  n = mlcell.getn();

  handle->elem_reserve(n);	  
          
  typename MESH::Node::array_type cell(m);  

  int r;
  r = 0;

  for (int p = 0; p < n; p++)
  {
    for (int q = 0 ; q < m; q++)
    {
      cell[q] = mldata[r]; r++; 
    }
    handle->add_elem(cell);
  }

  return (true);
}

template <class MESH> 
bool MatlabToFieldAlgo::addderivatives(SCIRun::LockingHandle<MESH>& handle)
{
  if (meshbasistype == "cubic")
  {
    error("The converter misses code to add cubic hermitian derivatives edges to mesh");
    return (false);
  }
  return (true);
}
    
template <class MESH> 
bool MatlabToFieldAlgo::addscalefactors(SCIRun::LockingHandle<MESH>& handle)
{
  if (meshbasistype == "cubic")
  {
    error("The converter misses code to add cubic hermitian scalefactors edges to mesh");
    return (false);
  }
  return (true);
}


template<class FDATA>
bool MatlabToFieldAlgo::addfield(FDATA &fdata)
{
  return(false);
}


template <class T> 
bool MatlabToFieldAlgo::addfield(std::vector<T> &fdata)
{
  mlfield.getnumericarray(fdata);
  return(true);
}

template <class T, class MESH> 
bool MatlabToFieldAlgo::addfield(SCIRun::FData2d<T,MESH> &fdata)
{
  mlfield.getnumericarray(fdata.get_dataptr(),fdata.dim2(),fdata.dim1());
  return(true);
}

template <class T,class MESH> 
bool MatlabToFieldAlgo::addfield(SCIRun::FData3d<T,MESH> &fdata)
{
  mlfield.getnumericarray(fdata.get_dataptr(),fdata.dim3(),fdata.dim2(),fdata.dim1());
  return(true);
}

template <class MESH>
bool MatlabToFieldAlgo::addfield(SCIRun::FData2d<SCIRun::Vector,MESH> &fdata)
{
  std::vector<double> fielddata;
  mlfield.getnumericarray(fielddata); // cast and copy the real part of the data

  unsigned int numdata = fielddata.size();
  if (numdata > (3*fdata.size())) numdata = (3*fdata.size()); // make sure we do not copy more data than there are elements
        
  SCIRun::Vector **data = fdata.get_dataptr();
  unsigned int dim1 = fdata.dim1();
  unsigned int dim2 = fdata.dim2();
        
  unsigned int q,r,p;
  for (p=0,q=0;(q<dim1)&&(p < numdata);q++)
    for (r=0;(r<dim2)&&(p < numdata);r++)
    {
      data[q][r][0] = fielddata[p++];
      data[q][r][1] = fielddata[p++];
      data[q][r][2] = fielddata[p++];
    }
  
  return(true);
}


template <class MESH>
bool MatlabToFieldAlgo::addfield(SCIRun::FData3d<SCIRun::Vector,MESH> &fdata)
{
  std::vector<double> fielddata;
  mlfield.getnumericarray(fielddata); // cast and copy the real part of the data
        
  unsigned int numdata = fielddata.size();
  if (numdata > (3*fdata.size())) numdata = (3*fdata.size()); // make sure we do not copy more data than there are elements
        
  SCIRun::Vector ***data = fdata.get_dataptr();
  unsigned int dim1 = fdata.dim1();
  unsigned int dim2 = fdata.dim2();
  unsigned int dim3 = fdata.dim3();
        
  unsigned int q,r,s,p;
  for (p=0,q=0;(q<dim1)&&(p < numdata);q++)
    for (r=0;(r<dim2)&&(p < numdata);r++)
      for (s=0;(s<dim3)&&(p <numdata);s++)
      {
        data[q][r][s][0] = fielddata[p++];
        data[q][r][s][1] = fielddata[p++];
        data[q][r][s][2] = fielddata[p++];
      }
  
  return(true);
}


template <class MESH>
bool MatlabToFieldAlgo::addfield(SCIRun::FData2d<SCIRun::Tensor,MESH> &fdata)
{
  std::vector<double> fielddata;
  mlfield.getnumericarray(fielddata); // cast and copy the real part of the data
        
  unsigned int numdata = fielddata.size();

  SCIRun::Tensor tens;
  SCIRun::Tensor **data = fdata.get_dataptr();
  unsigned int dim1 = fdata.dim1();
  unsigned int dim2 = fdata.dim2();

  if (mlfield.getm() == 6)
  { // Compressed tensor data : xx,yy,zz,xy,xz,yz
    if (numdata > (6*fdata.size())) numdata = (6*fdata.size()); // make sure we do not copy more data than there are elements
    unsigned int q,r,p;
    for (p=0,q=0;(q<dim1)&&(p < numdata);q++)
      for (r=0;(r<dim2)&&(p < numdata);r++, p+=6)
      {   
        compressedtensor(fielddata,tens,p);
        data[q][r] = tens; 
      }
  }
  else
  {  // UnCompressed tensor data : xx,xy,xz,yx,yy,yz,zx,zy,zz 
    if (numdata > (9*fdata.size())) numdata = (9*fdata.size()); // make sure we do not copy more data than there are elements
    unsigned int q,r,p;
    for (p=0,q=0;(q<dim1)&&(p < numdata);q++)
      for (r=0;(r<dim2)&&(p < numdata);r++, p+=9)
      {   
        uncompressedtensor(fielddata,tens,p); 
        data[q][r] = tens; 
      }
  }
  return(true);
}

template <class MESH>
bool MatlabToFieldAlgo::addfield(SCIRun::FData3d<SCIRun::Tensor,MESH> &fdata)
{
  std::vector<double> fielddata;
  mlfield.getnumericarray(fielddata); // cast and copy the real part of the data

  SCIRun::Tensor tens;
  SCIRun::Tensor ***data = fdata.get_dataptr();
  unsigned int dim1 = fdata.dim1();
  unsigned int dim2 = fdata.dim2();
  unsigned int dim3 = fdata.dim3();
        
  unsigned int numdata = fielddata.size();
  if (mlfield.getm() == 6)
  { // Compressed tensor data : xx,yy,zz,xy,xz,yz
    if (numdata > (6*fdata.size())) numdata = (6*fdata.size()); // make sure we do not copy more data than there are elements
    unsigned int q,r,s,p;
    for (p=0,q=0;(q<dim1)&&(p < numdata);q++)
      for (r=0;(r<dim2)&&(p < numdata);r++)
        for (s=0;(s<dim3)&&(p < numdata);s++,p +=6)
        {   
          compressedtensor(fielddata,tens,p); 
          data[q][r][s] = tens; 
        }
  }
  else
  {  // UnCompressed tensor data : xx,xy,xz,yx,yy,yz,zx,zy,zz 
    if (numdata > (9*fdata.size())) numdata = (9*fdata.size()); // make sure we do not copy more data than there are elements
    unsigned int q,r,s,p;
    for (p=0,q=0;(q<dim1)&&(p < numdata);q++)
      for (r=0;(r<dim2)&&(p < numdata);r++)
        for (s=0; (s<dim3)&&(p <numdata); s++, p+= 9)
        {   
          uncompressedtensor(fielddata,tens,p); 
          data[q][r][s] = tens; 
        }
  }
  return(true);
}


inline void MatlabToFieldAlgo::compressedtensor(std::vector<double> &fielddata,SCIRun::Tensor &tens, unsigned int p)
{
   tens.mat_[0][0] = fielddata[p+0];
   tens.mat_[0][1] = fielddata[p+1];
   tens.mat_[0][2] = fielddata[p+2];
   tens.mat_[1][0] = fielddata[p+1];
   tens.mat_[1][1] = fielddata[p+3];
   tens.mat_[1][2] = fielddata[p+4];
   tens.mat_[2][0] = fielddata[p+2];
   tens.mat_[2][1] = fielddata[p+4];
   tens.mat_[2][2] = fielddata[p+5];
}

inline void MatlabToFieldAlgo::uncompressedtensor(std::vector<double> &fielddata,SCIRun::Tensor &tens, unsigned int p)
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


} // end namespace


#endif


