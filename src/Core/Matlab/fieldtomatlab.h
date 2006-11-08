/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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

#include <math.h>

// STL STUFF
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <string>
#include <sgi_stl_warnings_on.h>

// Class for reading matlab files
#include <Core/Matlab/matlabfile.h>
#include <Core/Matlab/matlabarray.h>

// Field class files
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/GenericField.h>

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


/*
 * FILE: matlabconverter_fieldtomatlab.h
 * AUTH: Jeroen G Stinstra
 * DATE: 18 MAR 2004
 */
 
#ifndef JGS_MATLABIO_MATLABCONVERTER_FIELDTOMATLAB_H
#define JGS_MATLABIO_MATLABCONVERTER_FIELDTOMATLAB_H 1

namespace MatlabIO {


class FieldToMatlabAlgo : public SCIRun::DynamicAlgoBase, public matfilebase
{
  public:

    //////// CONSTRUCTOR ///////////////////////////////////
    inline FieldToMatlabAlgo();

    //////// OPTIONS FOR CONVERTER ////////////////////////

    // Set the index base for the matlab code, normally htis one is 1
    inline void option_indexbase(int indexbase);
    // Option to switch on the old way of dealing with tensors, the ordering
    // is different. When switched off, ordering will be same as SCIRun 
    // ordering.
    inline void option_oldtensor(bool value);
    // When isoparametric the connectivity matrix of mesh and field should
    // be the same hence one can specify to remove it if not wanted
    inline void option_nofieldconnectivity(bool value);
    // Force old naming scheme when exporting, this will not work for H.O.
    // elements.
    inline void option_forceoldnames(bool value);
    
    //////// DYNAMIC ALGO ENTRY POINT /////////////////////

    virtual bool execute(SCIRun::FieldHandle fieldH, matlabarray &mlarray) =0;

    static SCIRun::CompileInfoHandle 
                      get_compile_info(SCIRun::FieldHandle field);

    inline void setreporter(SCIRun::ProgressReporter* pr);

  protected:    

    //////// CONVERSION TOOLS /////////////////////////////

    template <class BASIS>  bool isnodata(BASIS& basis);
    template <class BASIS>  bool isconstant(BASIS& basis);
    template <class BASIS>  bool islinear(BASIS& basis);    
    template <class BASIS>  bool ishigherorder(BASIS& basis);    
    template <class BASIS>  bool ishermitian(BASIS& basis);
    template <class BASIS>  bool islagrangian(BASIS& basis);

    template <class BASIS>  std::string get_basis_name(BASIS& basis);
    template <class FDATA>  std::string get_fdata_name(FDATA& fdata);    
    template <class MESH>   std::string get_mesh_name(MESH* mesh);

    template <class MESH> bool mladdmeshheader(MESH* mesh, matlabarray mlarray);
    template <class MESH> bool mladdnodes(MESH* mesh,matlabarray mlarray);
    template <class MESH> bool mladdedges(MESH* mesh,matlabarray mlarray);
    template <class MESH> bool mladdfaces(MESH* mesh,matlabarray mlarray);    
    template <class MESH> bool mladdcells(MESH* mesh,matlabarray mlarray);
    template <class MESH> bool mladdmeshderivatives(MESH* mesh,matlabarray mlarray);

    template <class MESH> bool mladdtransform(MESH* mesh,matlabarray mlarray);
    template <class MESH> bool mladdxyzmesh1d(MESH* mesh,matlabarray mlarray);
    template <class MESH> bool mladdxyzmesh2d(MESH* mesh,matlabarray mlarray);
    template <class MESH> bool mladdxyzmesh3d(MESH* mesh,matlabarray mlarray);

    template <class MESH> bool mladddimension1d(MESH* mesh,matlabarray mlarray);
    template <class MESH> bool mladddimension2d(MESH* mesh,matlabarray mlarray);
    template <class MESH> bool mladddimension3d(MESH* mesh,matlabarray mlarray);

    template <class FIELD> bool mladdfieldheader(FIELD* field, matlabarray mlarray);                          

    template <class MESH, class BASIS, class T> bool mladdfielddata(SCIRun::GenericField<MESH,BASIS,std::vector<T> >* field,MESH* mesh,matlabarray mlarray);
    template <class MESH, class BASIS> bool mladdfielddata(SCIRun::GenericField<MESH,BASIS,std::vector<SCIRun::Vector> >* field,MESH* mesh,matlabarray mlarray);
    template <class MESH, class BASIS> bool mladdfielddata(SCIRun::GenericField<MESH,BASIS,std::vector<SCIRun::Tensor> >* field,MESH* mesh,matlabarray mlarray);

    template <class MESH, class BASIS, class T> bool mladdfielddata2d(SCIRun::GenericField<MESH,BASIS,SCIRun::FData2d<T,MESH> >* field,MESH* mesh,matlabarray mlarray);
    template <class MESH, class BASIS> bool mladdfielddata2d(SCIRun::GenericField<MESH,BASIS,SCIRun::FData2d<SCIRun::Vector,MESH> >* field,MESH* mesh,matlabarray mlarray);
    template <class MESH, class BASIS> bool mladdfielddata2d(SCIRun::GenericField<MESH,BASIS,SCIRun::FData2d<SCIRun::Tensor,MESH> >* field,MESH* mesh,matlabarray mlarray);

    template <class MESH, class BASIS, class T> bool mladdfielddata3d(SCIRun::GenericField<MESH,BASIS,SCIRun::FData3d<T,MESH> >* field,MESH* mesh,matlabarray mlarray);
    template <class MESH, class BASIS> bool mladdfielddata3d(SCIRun::GenericField<MESH,BASIS,SCIRun::FData3d<SCIRun::Vector,MESH> >* field,MESH* mesh,matlabarray mlarray);
    template <class MESH, class BASIS> bool mladdfielddata3d(SCIRun::GenericField<MESH,BASIS,SCIRun::FData3d<SCIRun::Tensor,MESH> >* field,MESH* mesh,matlabarray mlarray);

    template <class FIELD, class MESH> bool mladdfieldedges(FIELD *field,MESH *mesh,matlabarray mlarray);
    template <class FIELD, class MESH> bool mladdfieldfaces(FIELD *field,MESH *mesh,matlabarray mlarray);
    template <class FIELD, class MESH> bool mladdfieldcells(FIELD *field,MESH *mesh,matlabarray mlarray);
    
    template <class FIELD, class MESH> bool mladdfieldedgederivatives(FIELD *field,MESH *mesh,matlabarray mlarray);
    template <class FIELD, class MESH> bool mladdfieldfacederivatives(FIELD *field,MESH *mesh,matlabarray mlarray);
    template <class FIELD, class MESH> bool mladdfieldcellderivatives(FIELD *field,MESH *mesh,matlabarray mlarray);

    template <class FIELD, class MESH> bool mladdfieldderivatives1d(FIELD *field,MESH *mesh,matlabarray mlarray);
    template <class FIELD, class MESH> bool mladdfieldderivatives2d(FIELD *field,MESH *mesh,matlabarray mlarray);
    template <class FIELD, class MESH> bool mladdfieldderivatives3d(FIELD *field,MESH *mesh,matlabarray mlarray);

    template <class FIELD, class MESH>  bool mladdfield(FIELD* field, MESH* mesh, matlabarray mlarray);
    template <class FIELD, class BASIS> bool mladdfield(FIELD* field, SCIRun::ScanlineMesh<BASIS>* mesh,matlabarray mlarray);
    template <class FIELD, class BASIS> bool mladdfield(FIELD* field, SCIRun::ImageMesh<BASIS>* mesh,matlabarray mlarray);
    template <class FIELD, class BASIS> bool mladdfield(FIELD* field, SCIRun::LatVolMesh<BASIS>* mesh,matlabarray mlarray);
    template <class FIELD, class BASIS> bool mladdfield(FIELD* field, SCIRun::StructCurveMesh<BASIS>* mesh,matlabarray mlarray);
    template <class FIELD, class BASIS> bool mladdfield(FIELD* field, SCIRun::StructQuadSurfMesh<BASIS>* mesh,matlabarray mlarray);
    template <class FIELD, class BASIS> bool mladdfield(FIELD* field, SCIRun::StructHexVolMesh<BASIS>* mesh,matlabarray mlarray);
    template <class FIELD, class BASIS> bool mladdfield(FIELD* field, SCIRun::PointCloudMesh<BASIS>* mesh,matlabarray mlarray);
    template <class FIELD, class BASIS> bool mladdfield(FIELD* field, SCIRun::CurveMesh<BASIS>* mesh,matlabarray mlarray);
    template <class FIELD, class BASIS> bool mladdfield(FIELD* field, SCIRun::TriSurfMesh<BASIS>* mesh,matlabarray mlarray);
    template <class FIELD, class BASIS> bool mladdfield(FIELD* field, SCIRun::QuadSurfMesh<BASIS>* mesh,matlabarray mlarray);
    template <class FIELD, class BASIS> bool mladdfield(FIELD* field, SCIRun::TetVolMesh<BASIS>* mesh,matlabarray mlarray);
    template <class FIELD, class BASIS> bool mladdfield(FIELD* field, SCIRun::PrismVolMesh<BASIS>* mesh,matlabarray mlarray);
    template <class FIELD, class BASIS> bool mladdfield(FIELD* field, SCIRun::HexVolMesh<BASIS>* mesh,matlabarray mlarray);

    //////// ERROR REPORTERING MECHANISM /////////////////

    inline void error(std::string error);
    inline void warning(std::string warning);

    //////// OPTION PARAMETERS //////////////////////////

    bool option_forceoldnames_;
    bool option_nofieldconnectivity_;
    int  option_indexbase_;
    
  private:
    SCIRun::ProgressReporter *pr_;
    
};

inline FieldToMatlabAlgo::FieldToMatlabAlgo() :
  option_forceoldnames_(false),
  option_nofieldconnectivity_(false),  
  option_indexbase_(1),
  pr_(0)
{
}

inline void FieldToMatlabAlgo::setreporter(SCIRun::ProgressReporter* pr)
{
  pr_ = pr;
}


inline void FieldToMatlabAlgo::option_forceoldnames(bool value) 
{
  option_forceoldnames_ = value;
}

inline void FieldToMatlabAlgo::option_nofieldconnectivity(bool value) 
{
  option_nofieldconnectivity_ = value;
}

inline void FieldToMatlabAlgo::option_indexbase(int indexbase) 
{
  option_indexbase_ = indexbase;
}

inline void FieldToMatlabAlgo::error(std::string error)
{
  if(pr_) pr_->error(error);
}

inline void FieldToMatlabAlgo::warning(std::string warning)
{
  if(pr_) pr_->warning(warning);
}



///// DYNAMIC ALGORITHM STARTS HERE ///////////////////


template <class FIELD> 
class FieldToMatlabAlgoT : public FieldToMatlabAlgo
{
  public:
    //////// DYNAMIC ALGO ENTRY POINT /////////////////////
    virtual bool execute(SCIRun::FieldHandle fieldH, matlabarray &mlarray);
                          
};

template <class FIELD>  
bool FieldToMatlabAlgoT<FIELD>::execute(SCIRun::FieldHandle fieldH, matlabarray &mlarray)
{

  // Check whether the field actually contains a field;
  if (fieldH.get_rep() == 0)
  {
    error("FieldToMatlab: Field is empty.");
    return(false);
  }

  // input is a general FieldHandle, cast this to the specific one
  FIELD *field = dynamic_cast<FIELD *>(fieldH.get_rep());
  if (field == 0)
  {
    error("FieldToMatlab: This algorithm cannot handle this kind of field.");
    return(false);  
  }

  typename FIELD::mesh_handle_type meshH = field->get_typed_mesh();
  typename FIELD::mesh_type* mesh = meshH.get_rep();

  if (mesh == 0)
  {
    error("FieldToMatlab: This algorithm cannot handle this kind of mesh.");
    return (false);
  }

  // Get the basis of the field and mesh and get the data in the field
  typename FIELD::mesh_type::basis_type meshbasis = meshH->get_basis();
  typename FIELD::basis_type fieldbasis = field->get_basis();
  typename FIELD::fdata_type fdata = field->fdata();
  
  std::string meshtype_name = get_mesh_name(mesh);
  std::string fieldtype_name = get_fdata_name(fdata);
  
  // DEAL WITH REGULAR FIELDS
  
  // Filter out some non-sense classes as people are going to construct these
  // oddities, we better check whether we are dealing with a non existing topology
  if (isnodata(meshbasis))  
  {
    error("FieldToMatlab: Algorithm does not deal with basis elements containing no data.");
    return (false);
  }
        
  if (meshtype_name == "PointCloudMesh")
  {
    if (!isconstant(meshbasis))
    {
      error("FieldToMatlab: Point clouds should have data solely confined to the nodes.");
      error("FieldToMatlab: No linear or higher order interpolation in point clouds is supported.");
      return (false);
    }
    
    if (!(isconstant(fieldbasis)||(isnodata(fieldbasis))))
    {
      error("FieldToMatlab: Point clouds should have data solely confined to the nodes.");
      error("FieldToMatlab: No linear or higher order interpolation in point clouds is supported.");
      return (false);
    }
  }      
        
  if ((meshtype_name == "ScanlineMesh")||(meshtype_name == "ImageMesh")||
      (meshtype_name == "LatVolMesh"))
  {
    // Check whether we have a second or higher order mesh. As these require additional
    // nodes and data in the scheme it is not regular anymore. Hence in the matlab
    // interface package we DO NOT support these. These would lead to such ugly 
    // translations that one better first adapts the data.
    // Higher order data itself is allowed, although ugly it is allowed here.
    
    if (!islinear(meshbasis))
    {
      error("FieldToMatlab: Only regular meshes that are not higher order are supported.");
      error("FieldToMatlab: Convert the mesh to a structured Curve, QuadSurf, or HexVol  for higher order support");
      return (false);
    }
  }
  
  //////////////////////////////////////
  //vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv//

  if (ishigherorder(fieldbasis)||ishigherorder(meshbasis))
  {
    error("FieldToMatlab: Exporting higher fields is currently disabled.");
    error("FieldToMatlab: Higher order support is work in progress, but is currently disabled.");
    return (false);
  }
        
  // OK, we should have a topology that can be constructed somehow 

  if (!(mladdfield(field,mesh,mlarray)))
  {
    error("FieldToMatlab: The dynamic compiled algorithm was not able to generate a representation of the field in matlab matrices.");
    return (false);
  }
  
  return(true);
}


///////////////////////////////////////////////////////////////////
// We have an implementation per mesh. The problem is that the SCIRun
// meshes are not pure polymorphic. Some meshes have incompatible 
// functionality overloaded on the same concept. Mainly the 2D and 3D
// iterators are troublesome and alter functionality. Similarly the
// FData2D and FData3D are trouble some.
//
// Due to these small incompatibilities we have a per mesh type 
// converter. Hence new mesh types need to have new converters
// This is inconvenient, but is due to a bad design in SCIRun.

template <class FIELD, class MESH>
bool FieldToMatlabAlgo::mladdfield(FIELD* field, MESH* mesh,matlabarray mlarray)
{
  error("FieldToMatlab: Encountered an unknown mesh type.");
  error("FieldToMatlab: Cannot translate field into matlabarray.");
  return(false);
}

////////////// REGULAR MESHES //////////////////////////////////////////

template <class FIELD, class BASIS>
bool FieldToMatlabAlgo::mladdfield(FIELD* field, SCIRun::ScanlineMesh<BASIS>* mesh,matlabarray mlarray)
{
  return ( mladdfieldheader(field,mlarray) && mladdfielddata(field,mesh,mlarray) &&
           mladdfieldderivatives1d(field,mesh,mlarray) &&
           mladdmeshheader(mesh,mlarray) && mladdtransform(mesh,mlarray) && 
           mladddimension1d(mesh,mlarray) );
}

template <class FIELD, class BASIS>
bool FieldToMatlabAlgo::mladdfield(FIELD* field, SCIRun::ImageMesh<BASIS>* mesh,matlabarray mlarray)
{
  return ( mladdfieldheader(field,mlarray) && mladdfielddata2d(field,mesh,mlarray) &&
           mladdfieldderivatives2d(field,mesh,mlarray) &&
           mladdmeshheader(mesh,mlarray) && mladdtransform(mesh,mlarray) && 
           mladddimension2d(mesh,mlarray) );
}

template <class FIELD, class BASIS>
bool FieldToMatlabAlgo::mladdfield(FIELD* field, SCIRun::LatVolMesh<BASIS>* mesh,matlabarray mlarray)
{
  return ( mladdfieldheader(field,mlarray) && mladdfielddata3d(field,mesh,mlarray) &&
           mladdfieldderivatives3d(field,mesh,mlarray) &&
           mladdmeshheader(mesh,mlarray) && mladdtransform(mesh,mlarray) && 
           mladddimension3d(mesh,mlarray) );
}

////////////// STRUCTURED MESHES //////////////////////////////////////////

template <class FIELD, class BASIS>
bool FieldToMatlabAlgo::mladdfield(FIELD* field, SCIRun::StructCurveMesh<BASIS>* mesh,matlabarray mlarray)
{
  return ( mladdfieldheader(field,mlarray) && mladdfielddata(field,mesh,mlarray) &&
           mladdfieldderivatives1d(field,mesh,mlarray) &&
           mladdmeshheader(mesh,mlarray) && mladdxyzmesh1d(mesh,mlarray) );
}

template <class FIELD, class BASIS>
bool FieldToMatlabAlgo::mladdfield(FIELD* field, SCIRun::StructQuadSurfMesh<BASIS>* mesh,matlabarray mlarray)
{
  return ( mladdfieldheader(field,mlarray) && mladdfielddata2d(field,mesh,mlarray) &&
           mladdfieldderivatives2d(field,mesh,mlarray) &&
           mladdmeshheader(mesh,mlarray) && mladdxyzmesh2d(mesh,mlarray) );
}

template <class FIELD, class BASIS>
bool FieldToMatlabAlgo::mladdfield(FIELD* field, SCIRun::StructHexVolMesh<BASIS>* mesh,matlabarray mlarray)
{
  return ( mladdfieldheader(field,mlarray) && mladdfielddata3d(field,mesh,mlarray) &&
           mladdfieldderivatives3d(field,mesh,mlarray) &&
           mladdmeshheader(mesh,mlarray) && mladdxyzmesh3d(mesh,mlarray) );
}

////////////// UNSTRUCTURED MESHES /////////////////////////////////////

template <class FIELD, class BASIS>
bool FieldToMatlabAlgo::mladdfield(FIELD* field, SCIRun::PointCloudMesh<BASIS>* mesh,matlabarray mlarray)
{
  return ( mladdfieldheader(field,mlarray) && mladdfielddata(field,mesh,mlarray) &&
           mladdmeshheader(mesh,mlarray) && mladdnodes(mesh,mlarray) );
}

template <class FIELD, class BASIS>
bool FieldToMatlabAlgo::mladdfield(FIELD* field, SCIRun::CurveMesh<BASIS>* mesh,matlabarray mlarray)
{
  return ( mladdfieldheader(field,mlarray) && mladdfielddata(field,mesh,mlarray) &&
           mladdfieldedges(field,mesh,mlarray) && mladdfieldedgederivatives(field,mesh,mlarray) &&
           mladdmeshheader(mesh,mlarray) && mladdnodes(mesh,mlarray) &&
           mladdedges(mesh,mlarray) && mladdmeshderivatives(mesh,mlarray) );
}

template <class FIELD, class BASIS>
bool FieldToMatlabAlgo::mladdfield(FIELD* field, SCIRun::TriSurfMesh<BASIS>* mesh,matlabarray mlarray)
{
  return ( mladdfieldheader(field,mlarray) && mladdfielddata(field,mesh,mlarray) &&
           mladdfieldfaces(field,mesh,mlarray) && mladdfieldfacederivatives(field,mesh,mlarray) &&
           mladdmeshheader(mesh,mlarray) && mladdnodes(mesh,mlarray) &&
           mladdfaces(mesh,mlarray) && mladdmeshderivatives(mesh,mlarray) );
}

template <class FIELD, class BASIS>
bool FieldToMatlabAlgo::mladdfield(FIELD* field, SCIRun::QuadSurfMesh<BASIS>* mesh,matlabarray mlarray)
{
  return ( mladdfieldheader(field,mlarray) && mladdfielddata(field,mesh,mlarray) &&
           mladdfieldfaces(field,mesh,mlarray) && mladdfieldfacederivatives(field,mesh,mlarray) &&
           mladdmeshheader(mesh,mlarray) && mladdnodes(mesh,mlarray) &&
           mladdfaces(mesh,mlarray) && mladdmeshderivatives(mesh,mlarray) );
}

template <class FIELD, class BASIS>
bool FieldToMatlabAlgo::mladdfield(FIELD* field, SCIRun::TetVolMesh<BASIS>* mesh,matlabarray mlarray)
{
 return ( mladdfieldheader(field,mlarray) && mladdfielddata(field,mesh,mlarray) &&
           mladdfieldcells(field,mesh,mlarray) && mladdfieldcellderivatives(field,mesh,mlarray) &&
           mladdmeshheader(mesh,mlarray) && mladdnodes(mesh,mlarray) &&
           mladdcells(mesh,mlarray) && mladdmeshderivatives(mesh,mlarray) );
}

template <class FIELD, class BASIS>
bool FieldToMatlabAlgo::mladdfield(FIELD* field, SCIRun::PrismVolMesh<BASIS>* mesh,matlabarray mlarray)
{
  return ( mladdfieldheader(field,mlarray) && mladdfielddata(field,mesh,mlarray) &&
           mladdfieldcells(field,mesh,mlarray) && mladdfieldcellderivatives(field,mesh,mlarray) &&
           mladdmeshheader(mesh,mlarray) && mladdnodes(mesh,mlarray) &&
           mladdcells(mesh,mlarray) && mladdmeshderivatives(mesh,mlarray) );
}

template <class FIELD, class BASIS>
bool FieldToMatlabAlgo::mladdfield(FIELD* field, SCIRun::HexVolMesh<BASIS>* mesh ,matlabarray mlarray)
{
  return ( mladdfieldheader(field,mlarray) && mladdfielddata(field,mesh,mlarray) &&
           mladdfieldcells(field,mesh,mlarray) && mladdfieldcellderivatives(field,mesh,mlarray) &&
           mladdmeshheader(mesh,mlarray) && mladdnodes(mesh,mlarray) &&
           mladdcells(mesh,mlarray) && mladdmeshderivatives(mesh,mlarray) );
}


///////////////////////////////////////////////////////////////
// Functions for determining the type of basis independent of
// the actual name of the algorithm

template <class BASIS>
bool FieldToMatlabAlgo::isnodata(BASIS& basis)
{ 
  if (basis.polynomial_order() == -1) return(true);
  return(false);
}

template <class BASIS>
bool FieldToMatlabAlgo::isconstant(BASIS& basis)
{ 
  if (basis.polynomial_order() == 0) return(true);
  return(false);
}

template <class BASIS>
bool FieldToMatlabAlgo::islinear(BASIS& basis)
{ 
  if (basis.polynomial_order() == 1) return(true);
  return(false);
}

template <class BASIS>
bool FieldToMatlabAlgo::ishigherorder(BASIS& basis)
{ 
  if (basis.polynomial_order() > 1) return(true);
  return(false);
}

template <class BASIS>
bool FieldToMatlabAlgo::ishermitian(BASIS& basis)
{ 
  if (basis.polynomial_order() < 2) return(false);
  std::vector<int> test(basis.number_of_vertices());
  for (size_t p=0; p<test.size(); p++) test[p] = 0;
  for (size_t p=0; p<2; p++)
    for (size_t q=0; q<basis.number_of_edges(); q++)
      test[basis.unit_edges[q][p]] = 1;
  
  size_t q = 0;
  for (size_t p=0; p<test.size(); p++) if(test[p]) q++;
 
  if (q == basis.number_of_vertices()) return(true);
  return(false);
}

template <class BASIS>
bool FieldToMatlabAlgo::islagrangian(BASIS& basis)
{ 
  if (basis.polynomial_order() < 2) return(false);
  std::vector<int> test(basis.number_of_vertices());
  for (size_t p=0; p<test.size(); p++) test[p] = 0;
  for (size_t p=0; p<2; p++)
    for (size_t q=0; q<basis.number_of_edges(); q++)
      test[basis.unit_edges[q][p]] = 1;
  
  size_t q = 0;
  for (size_t p=0; p<test.size(); p++) if(test[p]) q++;
 
  if (q == basis.number_of_vertices()) return(false);
  return(true);
}

///////////////////////////////////////////////////////////////


template <class BASIS>
std::string FieldToMatlabAlgo::get_basis_name(BASIS& basis)
{
  return(basis.type_name(0));
}

template <class FDATA>
std::string FieldToMatlabAlgo::get_fdata_name(FDATA& fdata)
{
  typename FDATA::value_type* val = 0;
  const SCIRun::TypeDescription *td = get_type_description(val);
  std::string type = td->get_name();
  
  if (type.find("unsigned char") < type.size()) return("unsigned char"); 
  if (type.find("char") < type.size()) return("char"); 
  if (type.find("unsigned short") < type.size()) return("unsigned short"); 
  if (type.find("short") < type.size()) return("short"); 
  if (type.find("unsigned int") < type.size()) return("unsigned int"); 
  if (type.find("int") < type.size()) return("int"); 
  if (type.find("unsigned long") < type.size()) return("unsigned long"); 
  if (type.find("long") < type.size()) return("long"); 
  if (type.find("float") < type.size()) return("float"); 
  if (type.find("double") < type.size()) return("double"); 
  if (type.find("Vector") < type.size()) return("vector"); 
  if (type.find("Tensor") < type.size()) return("tensor"); 

  return("unknown");
}

template <class MESH>
std::string FieldToMatlabAlgo::get_mesh_name(MESH* mesh)
{
  std::string type = mesh->get_type_description()->get_name(); 
  if (type.find("StructCurve") < type.size()) return("StructCurveMesh");  
  if (type.find("StructQuadSurf") < type.size()) return("StructQuadsurfMesh");  
  if (type.find("StructHexVol") < type.size()) return("StructHexVolMesh");  
  if (type.find("Scanline") < type.size()) return("ScanlineMesh");  
  if (type.find("Image") < type.size()) return("ImageMesh");  
  if (type.find("LatVol") < type.size()) return("LatVolMesh");  
  if (type.find("PointCloud") < type.size()) return("PointCloudMesh");  
  if (type.find("Curve") < type.size()) return("CurveMesh");  
  if (type.find("TriSurf") < type.size()) return("TriSurfMesh");  
  if (type.find("QuadSurf") < type.size()) return("QuadSurfMesh");  
  if (type.find("TetVol") < type.size()) return("TetVolMesh");  
  if (type.find("PrismVol") < type.size()) return("PrismVolMesh");  
  if (type.find("HexVol") < type.size()) return("HexVolMesh");  
  
  return("unknown");
}

///////////// FUNCTIONS FOR GENERATING MESHES /////////////////

template <class MESH> 
bool FieldToMatlabAlgo::mladdmeshheader(MESH* mesh, matlabarray mlarray)
{
  typename MESH::basis_type meshbasis = mesh->get_basis();
  
  std::string meshbasis_name = get_basis_name(meshbasis);
  std::string meshtype_name  = get_mesh_name(mesh);
  int         meshbasisorder = mesh->get_basis().polynomial_order();
    
  matlabarray mlmeshbasis;
  mlmeshbasis.createstringarray(meshbasis_name);
  mlarray.setfield(0,"meshbasis",mlmeshbasis);

  matlabarray mlmeshbasisorder;
  mlmeshbasisorder.createdoublescalar(static_cast<double>(mesh->get_basis().polynomial_order()));
  mlarray.setfield(0,"meshbasisorder",mlmeshbasisorder);

  matlabarray mlmeshtype;
  mlmeshtype.createstringarray(meshtype_name);
  mlarray.setfield(0,"meshtype",mlmeshtype);
  
  return(true);
}

template <class MESH> 
bool FieldToMatlabAlgo::mladdnodes(MESH* mesh,matlabarray mlarray)
{
  matlabarray node;

  typename MESH::basis_type& basis = mesh->get_basis();
  
  if (islinear(basis)||ishermitian(basis)||isconstant(basis))
  {
  
    // A lot of pointless casting, but that is the way SCIRun was setup .....
    // Iterators and Index classes to make the code really complicated 
    // The next code tries to get away with minimal use of all this overhead

    typename MESH::Node::size_type size;
    mesh->size(size);
    unsigned int numnodes = static_cast<unsigned int>(size);

    // Request that it generates the node matrix
    mesh->synchronize(SCIRun::Mesh::NODES_E); 

    // Buffers for exporting the data to matlab.
    // The MatlabIO does not use SCIRun style iterators hence we need to extract
    // the data first. 
    std::vector<double> nodes(3*numnodes);
    std::vector<int> dims(2);

    // Setup the dimensions of the matlab array
    dims[0] = 3; dims[1] = static_cast<int>(numnodes);

    // Extracting data from the SCIRun classes is a painfull process.
    // I'd like to change this, but hey a lot of code should be rewritten
    // This works, it might not be really efficient, at least it does not
    // hack into the object.

    SCIRun::Point P;
    unsigned int q = 0;

    typename MESH::Node::iterator it, it_end;
    mesh->begin(it);
    mesh->end(it_end);
    
    while(it != it_end)
    {
      mesh->get_point(P,*(it));
      nodes[q++] = P.x(); nodes[q++] = P.y(); nodes[q++] = P.z(); 
      ++it;
    }

    node.createdoublematrix(nodes,dims);
    mlarray.setfield(0,"node",node);
    
    return (true);
  }
  else if (islagrangian(basis))
  {
    // NEED ACCESS FUNCTION TO HO ELEMENTS
    // TO MAKE THIS FUNCTION WORK
    
    error("FieldToMatlab: Currently no access function available to get higher order node locations.");
    return(false);
  }
  else
  {
    error("FieldToMatlab: Unknown Basis class type encountered.");
    return(false);
  }

}

template <class MESH>
bool FieldToMatlabAlgo::mladdedges(MESH *mesh,matlabarray mlarray)
{
	matlabarray edge;

  typename MESH::basis_type& basis = mesh->get_basis();

  if (islinear(basis)||ishermitian(basis))
  {

    typename MESH::Edge::size_type size;
    mesh->size(size);

    typename MESH::basis_type& basis = mesh->get_basis();
    size_t num = basis.number_of_vertices();

    size_t numedges = static_cast<size_t>(size);
    mesh->synchronize(SCIRun::Mesh::EDGES_E); 

    typename MESH::Node::array_type a;
    std::vector<typename MESH::Node::index_type> edges(num*numedges);
    std::vector<int> dims(2);	
    dims[0] = static_cast<int>(num); dims[1] = static_cast<int>(numedges);
  
    
    // SCIRun iterators are limited in supporting any index management
    // Hence I prefer to do it with integer and convert to the required
    // class at the last moment. Hopefully the compiler is smart and
    // has a fast translation. 	
    typename MESH::Edge::iterator it, it_end;
    mesh->begin(it);
    mesh->end(it_end);
    size_t q = 0;

    while (it != it_end)
    {
      mesh->get_nodes(a,*(it));
      for (size_t r = 0; r < num; r++) edges[q++] = a[r] + option_indexbase_;
      ++it;
    }

    edge.createdensearray(dims,matlabarray::miUINT32);
    edge.setnumericarray(edges); // store them as UINT32 but treat them as doubles
    mlarray.setfield(0,"edge",edge);

    return (true);
  }
  else if (islagrangian(basis))
  {
    // NEED ACCESS FUNCTION TO HO ELEMENTS
    // TO MAKE THIS FUNCTION WORK
    
    error("FieldToMatlab: Currently no access function available to get higher order node locations.");
    return(false);
  }
  else
  {
    error("FieldToMatlab: Unknown Basis class type encountered.");
    return(false);
  }
  

}

template <class MESH>
bool FieldToMatlabAlgo::mladdfaces(MESH *mesh,matlabarray mlarray)
{
	// A lot of pointless casting, but that is the way SCIRun was setup .....
	// Iterators and Index classes to make the code really complicated 
	// The next code tries to get away with minimal use of all this overhead

	matlabarray face;
  
  typename MESH::basis_type& basis = mesh->get_basis();
  
  if (islinear(basis)||ishermitian(basis))
  {
    typename MESH::Face::size_type size;
    mesh->size(size);
    size_t numfaces = static_cast<size_t>(size);

    typename MESH::basis_type& basis = mesh->get_basis();
    size_t num = basis.number_of_vertices();

    mesh->synchronize(SCIRun::Mesh::FACES_E);

    typename MESH::Node::array_type a;
    std::vector<typename MESH::Node::index_type> faces(num*numfaces);
    std::vector<int> dims(2);	
    dims[0] = static_cast<int>(num); dims[1] = static_cast<int>(numfaces);
      
    size_t q = 0;
    typename MESH::Face::iterator it, it_end;
    mesh->begin(it);
    mesh->end(it_end);
    while (it != it_end)
    {
      mesh->get_nodes(a,*(it));
      for (size_t r = 0; r < num; r++) faces[q++] = a[r] + option_indexbase_;
      ++it;
    }

    face.createdensearray(dims,matlabarray::miUINT32);
    face.setnumericarray(faces); // store them as UINT32 but treat them as doubles
    mlarray.setfield(0,"face",face);

    return (true);
  }
  else if (islagrangian(basis))
  {
    // NEED ACCESS FUNCTION TO HO ELEMENTS
    // TO MAKE THIS FUNCTION WORK
    
    error("FieldToMatlab: Currently no access function available to get higher order node locations.");
    return(false);
  }
  else
  {
    error("FieldToMatlab: Unknown Basis class type encountered.");
    return(false);
  }

}


template <class MESH>
bool FieldToMatlabAlgo::mladdcells(MESH* mesh,matlabarray mlarray)
{
  // A lot of pointless casting, but that is the way SCIRun was setup .....
  // Iterators and Index classes to make the code really complicated 
  // The next code tries to get away with minimal use of all this overhead

  matlabarray cell;
  
  typename MESH::basis_type& basis = mesh->get_basis();
  
  if (islinear(basis)||ishermitian(basis))  
  {
    typename MESH::Cell::size_type size;
    mesh->size(size);
    size_t numcells = static_cast<size_t>(size);

    typename MESH::basis_type& basis = mesh->get_basis();
    size_t num = basis.number_of_vertices();

    mesh->synchronize(SCIRun::Mesh::CELLS_E);

    typename MESH::Node::array_type a;
    std::vector<typename MESH::Node::index_type> cells(num*numcells);
    std::vector<int> dims(2);	
    dims[0] = static_cast<int>(num); dims[1] = static_cast<int>(numcells);

          
    typename MESH::Cell::iterator it, it_end;
    mesh->begin(it);
    mesh->end(it_end);
    size_t q = 0;
    
    while(it != it_end)
    {
      mesh->get_nodes(a,*(it));
      for (size_t r = 0; r < num; r++) cells[q++] = a[r] + option_indexbase_;
      ++it;
    }

    cell.createdensearray(dims,matlabarray::miUINT32);
    cell.setnumericarray(cells); // store them as UINT32 but treat them as doubles
    mlarray.setfield(0,"cell",cell);

    return (true);
  }
  else if (islagrangian(basis))
  {
    // NEED ACCESS FUNCTION TO HO ELEMENTS
    // TO MAKE THIS FUNCTION WORK
    
    error("FieldToMatlab: Currently no access function available to get higher order node locations.");
    return(false);
  }
  else
  {
    error("FieldToMatlab: Unknown Basis class type encountered.");
    return(false);
  }
  

}

template <class MESH>
bool FieldToMatlabAlgo::mladdmeshderivatives(MESH* mesh,matlabarray mlarray)
{

  typename MESH::basis_type& basis = mesh->get_basis();
  
  if (ishermitian(basis))
  {
    // CANNOT DO THIS NEITHER, NO ACCESS FUNCTIONS TO DATA
    // UNLESS I HACK INTO THE BASIS CLASSS
    
    error("FieldToMatlab: Currently no access function available to get higher order node derivatives.");
    return (false);
  }
  else
  {
    return (true);
  }
}



template <class MESH>
bool FieldToMatlabAlgo::mladdtransform(MESH* mesh,matlabarray mlarray)
{
  typename MESH::basis_type& basis = mesh->get_basis();
  
  if (ishigherorder(basis))
  {
    error("FieldToMatlab: No higher order elements are supported for regular meshes");
    return(false);
  }

  SCIRun::Transform T;
  matlabarray transform;
  double data[16];
  
  T =mesh->get_transform();
  T.get_trans(data);
  transform.createdensearray(4,4,matlabarray::miDOUBLE);
  transform.setnumericarray(data,16);
  mlarray.setfield(0,"tranform",transform);
  return(true);
}

template <class MESH>
bool FieldToMatlabAlgo::mladddimension1d(MESH* mesh,matlabarray mlarray)
{
  matlabarray dim;
  dim.createdensearray(1,1,matlabarray::miDOUBLE);
  std::vector<double> dims(1);

  typename MESH::Node::size_type size;
  mesh->size(size);
  dims[0] = static_cast<double>(size);

  dim.setnumericarray(dims);
  mlarray.setfield(0,"dims",dim);
  
  return(true);
}

template <class MESH>
bool FieldToMatlabAlgo::mladddimension2d(MESH* mesh,matlabarray mlarray)
{
  matlabarray dim;
  dim.createdensearray(1,2,matlabarray::miDOUBLE);
  std::vector<double> dims(2);

  dims[0] = static_cast<double>(mesh->get_ni());
  dims[1] = static_cast<double>(mesh->get_nj());

  dim.setnumericarray(dims);
  mlarray.setfield(0,"dims",dim);
  
  return(true);
}

template <class MESH>
bool FieldToMatlabAlgo::mladddimension3d(MESH* mesh,matlabarray mlarray)
{
  matlabarray dim;
  dim.createdensearray(1,3,matlabarray::miDOUBLE);
  std::vector<double> dims(3);

  dims[0] = static_cast<double>(mesh->get_ni());
  dims[1] = static_cast<double>(mesh->get_nj());
  dims[2] = static_cast<double>(mesh->get_nk());

  dim.setnumericarray(dims);
  mlarray.setfield(0,"dims",dim);
  
  return(true);
}


template <class MESH>
bool FieldToMatlabAlgo::mladdxyzmesh1d(MESH* mesh,matlabarray mlarray)
{
  typename MESH::basis_type& basis = mesh->get_basis();

  if (islinear(basis))
  {
    matlabarray x,y,z;
    mesh->synchronize(SCIRun::Mesh::NODES_E);
    typename MESH::Node::size_type size;

    mesh->size(size);
    unsigned int numnodes = static_cast<unsigned int>(size);
    x.createdensearray(static_cast<int>(numnodes),1,matlabarray::miDOUBLE);
    y.createdensearray(static_cast<int>(numnodes),1,matlabarray::miDOUBLE);
    z.createdensearray(static_cast<int>(numnodes),1,matlabarray::miDOUBLE);
          
    std::vector<double> xbuffer(numnodes);
    std::vector<double> ybuffer(numnodes);
    std::vector<double> zbuffer(numnodes);
          
    SCIRun::Point P;
    for (unsigned int p = 0; p < numnodes ; p++)
    {
      mesh->get_point(P,typename MESH::Node::index_type(p));
      xbuffer[p] = P.x();
      ybuffer[p] = P.y();
      zbuffer[p] = P.z();
    }

    x.setnumericarray(xbuffer);
    y.setnumericarray(ybuffer);
    z.setnumericarray(zbuffer);
          
    mlarray.setfield(0,"x",x);
    mlarray.setfield(0,"y",y);
    mlarray.setfield(0,"z",z);
    
    return (true);
  }

  error("FieldToMatlab: Currently no higher order geometry available for structured meshes.");
  return (false);
}


template <class MESH>
bool FieldToMatlabAlgo::mladdxyzmesh2d(MESH* mesh,matlabarray mlarray)
{
  typename MESH::basis_type& basis = mesh->get_basis();
  
  if (islinear(basis))
  {
    matlabarray x,y,z;
    mesh->synchronize(SCIRun::Mesh::NODES_E);  

    unsigned int dim1 = static_cast<unsigned int>(mesh->get_ni());
    unsigned int dim2 = static_cast<unsigned int>(mesh->get_nj());
    unsigned int numnodes = dim1*dim2;
        
    // Note: the dimensions are in reverse order as SCIRun uses C++
    // ordering
    x.createdensearray(static_cast<int>(dim2),static_cast<int>(dim1),matlabarray::miDOUBLE);
    y.createdensearray(static_cast<int>(dim2),static_cast<int>(dim1),matlabarray::miDOUBLE);
    z.createdensearray(static_cast<int>(dim2),static_cast<int>(dim1),matlabarray::miDOUBLE);
        
    // We use temp buffers to store all the values before committing them to the matlab
    // classes, this takes up more memory, but should decrease the number of actual function
    // calls, which should be boost performance 
    std::vector<double> xbuffer(numnodes);
    std::vector<double> ybuffer(numnodes);
    std::vector<double> zbuffer(numnodes);
        
    SCIRun::Point P;
    unsigned int r = 0;
    for (unsigned int p = 0; p < dim1 ; p++)
      for (unsigned int q = 0; q < dim2 ; q++)
      {   
        // It's ulgy, it's SCIRun ......
        typename MESH::Node::index_type idx(mesh,p,q);
        mesh->get_point(P,idx);
        xbuffer[r] = P.x();
        ybuffer[r] = P.y();
        zbuffer[r] = P.z();
        r++;
      }

    x.setnumericarray(xbuffer);
    y.setnumericarray(ybuffer);
    z.setnumericarray(zbuffer);
          
    mlarray.setfield(0,"x",x);
    mlarray.setfield(0,"y",y);
    mlarray.setfield(0,"z",z);
    
    return (true);
  }

  error("FieldToMatlab: Currently no higher order geometry available for structured meshes.");
  return (false);
}

template <class MESH>
bool FieldToMatlabAlgo::mladdxyzmesh3d(MESH* mesh,matlabarray mlarray)
{
  typename MESH::basis_type& basis = mesh->get_basis();
  
  if (islinear(basis))
  {
    matlabarray x,y,z;
    mesh->synchronize(SCIRun::Mesh::NODES_E);  

    unsigned int dim1 = static_cast<unsigned int>(mesh->get_ni());
    unsigned int dim2 = static_cast<unsigned int>(mesh->get_nj());
    unsigned int dim3 = static_cast<unsigned int>(mesh->get_nk());
    unsigned int numnodes = dim1*dim2*dim3;
        
    // Note: the dimensions are in reverse order as SCIRun uses C++
    // ordering
    std::vector<int> dims(3); dims[0] = static_cast<int>(dim3); 
    dims[1] = static_cast<int>(dim2); dims[2] = static_cast<int>(dim1);    
    x.createdensearray(dims,matlabarray::miDOUBLE);
    y.createdensearray(dims,matlabarray::miDOUBLE);
    z.createdensearray(dims,matlabarray::miDOUBLE);
        
    // We use temp buffers to store all the values before committing them to the matlab
    // classes, this takes up more memory, but should decrease the number of actual function
    // calls, which should be boost performance 
    std::vector<double> xbuffer(numnodes);
    std::vector<double> ybuffer(numnodes);
    std::vector<double> zbuffer(numnodes);
        
    SCIRun::Point P;
    unsigned int r = 0;
    for (unsigned int p = 0; p < dim1 ; p++)
      for (unsigned int q = 0; q < dim2 ; q++)
        for (unsigned int s = 0; s < dim3 ; s++)
        {   
          // It's ulgy, it's SCIRun ......
          typename MESH::Node::index_type idx(mesh,p,q,s);
          mesh->get_point(P,idx);
          xbuffer[r] = P.x();
          ybuffer[r] = P.y();
          zbuffer[r] = P.z();
          r++;
        }

    x.setnumericarray(xbuffer);
    y.setnumericarray(ybuffer);
    z.setnumericarray(zbuffer);
          
    mlarray.setfield(0,"x",x);
    mlarray.setfield(0,"y",y);
    mlarray.setfield(0,"z",z);
    
    return (true);
  }

  error("FieldToMatlab: Currently no higher order geometry available for structured meshes.");
  return (false);
}


//////////////////////////////////////////////////////////////////////////////

template <class FIELD> 
bool FieldToMatlabAlgo::mladdfieldheader(FIELD* field, matlabarray mlarray)
{
  typename FIELD::basis_type fieldbasis = field->get_basis();
  typename FIELD::fdata_type fdata = field->fdata();
 
  std::string fieldbasis_name = get_basis_name(fieldbasis);
  std::string fieldtype_name  = get_fdata_name(fdata);
  int         fieldbasisorder = field->basis_order();

  matlabarray mlfieldbasis;
  mlfieldbasis.createstringarray(fieldbasis_name);
  mlarray.setfield(0,"fieldbasis",mlfieldbasis);

  matlabarray mlfieldbasisorder;
  mlfieldbasisorder.createdoublescalar(static_cast<double>(field->basis_order()));
  mlarray.setfield(0,"fieldbasisorder",mlfieldbasisorder);

  matlabarray mlfieldtype;
  mlfieldtype.createstringarray(fieldtype_name);
  mlarray.setfield(0,"fieldtype",mlfieldtype);
  
  return(true);
}

////////////////////////////////////////

template <class MESH, class BASIS, class T> 
bool FieldToMatlabAlgo::mladdfielddata(SCIRun::GenericField<MESH,BASIS,std::vector<T> >* field,MESH* mesh,matlabarray mlarray)
{
  BASIS basis = field->get_basis();
  matlabarray mlfield;
  
  if (isnodata(basis))
  {
    return (true);
  }
  
  if (islinear(basis)||ishermitian(basis)||isconstant(basis))
  {
  	T dummy; matlabarray::mitype type = mlfield.getmitype(dummy);
    if (type == matlabarray::miUNKNOWN) 
    {
      error("FieldToMatlab: The field has a datatype not know to Matlab, so the data cannot be translated");
      return(false);
    }
    
    std::vector<T> &fdata = field->fdata();
    mlfield.createdensearray(1,static_cast<int>(fdata.size()),type);
    mlfield.setnumericarray(fdata);   
    mlarray.setfield(0,"field",mlfield);
    return (true);
  }
  
  if (islagrangian(basis))
  {
    error("FieldToMatlab: Cannot access the data in the field properly, hence cannot retrieve the data");
    return (false);
  }
  
  error("FieldToMatlab: Unknonw basis type");
  return (false);
}


template <class MESH, class BASIS> 
bool FieldToMatlabAlgo::mladdfielddata(SCIRun::GenericField<MESH,BASIS,std::vector<SCIRun::Vector> >* field,MESH* mesh,matlabarray mlarray)
{
  BASIS basis = field->get_basis();
  matlabarray mlfield;
  
  if (isnodata(basis))
  {
    return (true);
  }
  
  if (islinear(basis)||ishermitian(basis)||isconstant(basis))
  {
    std::vector<SCIRun::Vector> &fdata = field->fdata(); 
    mlfield.createdensearray(3,static_cast<int>(fdata.size()),matlabarray::miDOUBLE);
                
    unsigned int size = fdata.size();
    unsigned int p,q;
    std::vector<double> data(size*3);
    for (p = 0, q = 0; p < size; p++) 
    { 
      data[q++] = fdata[p][0];
      data[q++] = fdata[p][1];
      data[q++] = fdata[p][2];
    }             
    mlfield.setnumericarray(data);          
    mlarray.setfield(0,"field",mlfield);
    return(true);
  }
  
  if (islagrangian(basis))
  {
    error("FieldToMatlab: Cannot access the data in the field properly, hence cannot retrieve the data");
    return (false);
  }
  
  error("FieldToMatlab: Unknow basis type");
  return (false);
}

template <class MESH, class BASIS> 
bool FieldToMatlabAlgo::mladdfielddata(SCIRun::GenericField<MESH,BASIS,std::vector<SCIRun::Tensor> >* field,MESH* mesh,matlabarray mlarray)
{
  BASIS basis = field->get_basis();
  matlabarray mlfield;
  
  if (isnodata(basis))
  {
    return (true);
  }
  
  if (islinear(basis)||ishermitian(basis)||(isconstant(basis)))
  {
    std::vector<SCIRun::Tensor> &fdata = field->fdata(); 
    mlfield.createdensearray(9,static_cast<int>(fdata.size()),matlabarray::miDOUBLE);
                
    unsigned int size = fdata.size();
    unsigned int p,q;
    std::vector<double> data(size*9);
    for (p = 0, q = 0; p < size; p++) 
    { 
      data[q++] = fdata[p].mat_[0][0];
      data[q++] = fdata[p].mat_[0][1];
      data[q++] = fdata[p].mat_[0][2];
      data[q++] = fdata[p].mat_[1][0];
      data[q++] = fdata[p].mat_[1][1];
      data[q++] = fdata[p].mat_[1][2];
      data[q++] = fdata[p].mat_[2][0];
      data[q++] = fdata[p].mat_[2][1];
      data[q++] = fdata[p].mat_[2][2];
    }             
    mlfield.setnumericarray(data);          
    mlarray.setfield(0,"field",mlfield);
    return(true);
  }
  
  if (islagrangian(basis))
  {
    error("FieldToMatlab: Cannot access the data in the field properly, hence cannot retrieve the data");
    return (false);
  }
  
  error("FieldToMatlab: Unknow basis type");
  return (false);
}


template <class MESH, class BASIS, class T> 
bool FieldToMatlabAlgo::mladdfielddata2d(SCIRun::GenericField<MESH,BASIS,SCIRun::FData2d<T,MESH> >* field,MESH* mesh,matlabarray mlarray)
{
  BASIS basis = field->get_basis();
  matlabarray mlfield;
  
  if (isnodata(basis))
  {
    return (true);
  }
  
  if (islinear(basis)||ishermitian(basis)||isconstant(basis))
  {
  	T dummy; matlabarray::mitype type = mlfield.getmitype(dummy);
    if (type == matlabarray::miUNKNOWN) 
    {
      error("FieldToMatlab: The field has a datatype not know to Matlab, so the data cannot be translated");
      return(false);
    }

    SCIRun::FData2d<T,MESH> &fdata = field->fdata(); 
    
    std::vector<int> dims(2);
    // Note: the dimensions are in reverse order as SCIRun uses C++
    // ordering
    dims[0] = fdata.dim2(); dims[1] = fdata.dim1();
    mlfield.createdensearray(dims,type);
      
    unsigned int p,q,r;
    unsigned int dim1 = fdata.dim1();
    unsigned int dim2 = fdata.dim2();
    T **dataptr = fdata.get_dataptr();
          
    std::vector<T> data(dim1*dim2);
    for (p=0,q=0;q<dim1;q++)
      for (r=0;r<dim2;r++)
      {
        data[p++] = dataptr[q][r];
      }    
    
    mlfield.setnumericarray(data);   
    mlarray.setfield(0,"field",mlfield);
    return (true);
  }
  
  if (islagrangian(basis))
  {
    error("FieldToMatlab: Cannot access the data in the field properly, hence cannot retrieve the data");
    return (false);
  }
  
  error("FieldToMatlab: Unknow basis type");
  return (false);
}


template <class MESH, class BASIS> 
bool FieldToMatlabAlgo::mladdfielddata2d(SCIRun::GenericField<MESH,BASIS,SCIRun::FData2d<SCIRun::Vector,MESH> >* field,MESH* mesh,matlabarray mlarray)
{
  BASIS basis = field->get_basis();
  matlabarray mlfield;
  
  if (isnodata(basis))
  {
    return (true);
  }
  
  if (islinear(basis)||ishermitian(basis)||isconstant(basis))
  {
    SCIRun::FData2d<SCIRun::Vector,MESH> &fdata = field->fdata(); 
    
    std::vector<int> dims(3);
    // Note: the dimensions are in reverse order as SCIRun uses C++
    // ordering
    dims[0] = 3; dims[1] = fdata.dim2(); dims[2] = fdata.dim1();
    mlfield.createdensearray(dims,matlabarray::miDOUBLE);
                
    unsigned int p,q,r;
    unsigned int dim1 = fdata.dim1();
    unsigned int dim2 = fdata.dim2();
    SCIRun::Vector **dataptr = fdata.get_dataptr();
          
    std::vector<double> data(dim1*dim2*3);
    for (p=0,q=0;q<dim1;q++)
      for (r=0;r<dim2;r++)
      {
        data[p++] = dataptr[q][r][0];
        data[p++] = dataptr[q][r][1];
        data[p++] = dataptr[q][r][2];
      }           
           
    mlfield.setnumericarray(data);          
    mlarray.setfield(0,"field",mlfield);
    return(true);
  }
  
  if (islagrangian(basis))
  {
    error("FieldToMatlab: Cannot access the data in the field properly, hence cannot retrieve the data");
    return (false);
  }
  
  error("FieldToMatlab: Unknow basis type");
  return (false);
}

template <class MESH, class BASIS> 
bool FieldToMatlabAlgo::mladdfielddata2d(SCIRun::GenericField<MESH,BASIS,SCIRun::FData2d<SCIRun::Tensor,MESH> >* field,MESH* mesh,matlabarray mlarray)
{
  BASIS basis = field->get_basis();
  matlabarray mlfield;
  
  if (isnodata(basis))
  {
    return (true);
  }
  
  if (islinear(basis)||ishermitian(basis)||isconstant(basis))
  {
    SCIRun::FData2d<SCIRun::Tensor,MESH> &fdata = field->fdata(); 
    
  std::vector<int> dims(3);
  // Note: the dimensions are in reverse order as SCIRun uses C++
  // ordering
  dims[0] = 9; dims[1] = fdata.dim2(); dims[2] = fdata.dim1();
  mlfield.createdensearray(dims,matlabarray::miDOUBLE);
                
  unsigned int p,q,r;
  unsigned int dim1 = fdata.dim1();
  unsigned int dim2 = fdata.dim2();

  SCIRun::Tensor **dataptr = fdata.get_dataptr();

  std::vector<double> data(dim1*dim2*9);
  for (p=0,q=0;q<dim1;q++)
    for (r=0;r<dim2;r++)
    {
      data[p++] = dataptr[q][r].mat_[0][0];
      data[p++] = dataptr[q][r].mat_[0][1];
      data[p++] = dataptr[q][r].mat_[0][2];
      data[p++] = dataptr[q][r].mat_[1][0];
      data[p++] = dataptr[q][r].mat_[1][1];
      data[p++] = dataptr[q][r].mat_[1][2];
      data[p++] = dataptr[q][r].mat_[2][0];
      data[p++] = dataptr[q][r].mat_[2][1];
      data[p++] = dataptr[q][r].mat_[2][2];
    }              
                                           
    mlfield.setnumericarray(data);          
    mlarray.setfield(0,"field",mlfield);
    return(true);
  }
  
  if (islagrangian(basis))
  {
    error("FieldToMatlab: Cannot access the data in the field properly, hence cannot retrieve the data");
    return (false);
  }
  
  error("FieldToMatlab: Unknow basis type");
  return (false);
}


template <class MESH, class BASIS, class T> 
bool FieldToMatlabAlgo::mladdfielddata3d(SCIRun::GenericField<MESH,BASIS,SCIRun::FData3d<T,MESH> >* field,MESH* mesh,matlabarray mlarray)
{
  BASIS basis = field->get_basis();
  matlabarray mlfield;
  
  if (isnodata(basis))
  {
    return (true);
  }
  
  if (islinear(basis)||ishermitian(basis)||isconstant(basis))
  {
  	T dummy; matlabarray::mitype type = mlfield.getmitype(dummy);
    if (type == matlabarray::miUNKNOWN) 
    {
      error("FieldToMatlab: The field has a datatype not know to Matlab, so the data cannot be translated");
      return(false);
    }

    SCIRun::FData3d<T,MESH> &fdata = field->fdata(); 
    
    std::vector<int> dims(3);
    // Note: the dimensions are in reverse order as SCIRun uses C++
    // ordering
    dims[0] = fdata.dim3(); dims[1] = fdata.dim2(); dims[2] = fdata.dim1();
    mlfield.createdensearray(dims,type);
          
    unsigned int p,q,r,s;
    unsigned int dim1 = fdata.dim1();
    unsigned int dim2 = fdata.dim2();
    unsigned int dim3 = fdata.dim3();

   T ***dataptr = fdata.get_dataptr();
          
    std::vector<T> data(dim1*dim2*dim3);
    for (p=0,q=0;q<dim1;q++)
      for (r=0;r<dim2;r++)
        for (s=0;s<dim3;s++)
        {
          data[p++] = dataptr[q][r][s];
        }    
    
    mlfield.setnumericarray(data);   
    mlarray.setfield(0,"field",mlfield);
    return (true);
  }
  
  if (islagrangian(basis))
  {
    error("FieldToMatlab: Cannot access the data in the field properly, hence cannot retrieve the data");
    return (false);
  }
  
  error("FieldToMatlab: Unknow basis type");
  return (false);
}


template <class MESH, class BASIS> 
bool FieldToMatlabAlgo::mladdfielddata3d(SCIRun::GenericField<MESH,BASIS,SCIRun::FData3d<SCIRun::Vector,MESH> >* field,MESH* mesh,matlabarray mlarray)
{
  BASIS basis = field->get_basis();
  matlabarray mlfield;
  
  if (isnodata(basis))
  {
    return (true);
  }
  
  if (islinear(basis)||ishermitian(basis)||isconstant(basis))
  {
    SCIRun::FData3d<SCIRun::Vector,MESH> &fdata = field->fdata(); 
    
    std::vector<int> dims(4);
    // Note: the dimensions are in reverse order as SCIRun uses C++
    // ordering
    dims[0] = 3; dims[1] = fdata.dim3(); dims[2] = fdata.dim2(); dims[3] = fdata.dim1();
    mlfield.createdensearray(dims,matlabarray::miDOUBLE);
                
    unsigned int p,q,r,s;
    unsigned int dim1 = fdata.dim1();
    unsigned int dim2 = fdata.dim2();
    unsigned int dim3 = fdata.dim3();

    SCIRun::Vector ***dataptr = fdata.get_dataptr();

    std::vector<double> data(dim1*dim2*dim3*3);
    for (p=0,q=0;q<dim1;q++)
      for (r=0;r<dim2;r++)
        for (s=0;s<dim3;s++)
          {
            data[p++] = dataptr[q][r][s][0];
            data[p++] = dataptr[q][r][s][1];
            data[p++] = dataptr[q][r][s][2];
          }                 
           
    mlfield.setnumericarray(data);          
    mlarray.setfield(0,"field",mlfield);
    return(true);
  }
  
  if (islagrangian(basis))
  {
    error("FieldToMatlab: Cannot access the data in the field properly, hence cannot retrieve the data");
    return (false);
  }
  
  error("FieldToMatlab: Unknow basis type");
  return (false);
}

template <class MESH, class BASIS> 
bool FieldToMatlabAlgo::mladdfielddata3d(SCIRun::GenericField<MESH,BASIS,SCIRun::FData3d<SCIRun::Tensor,MESH> >* field,MESH* mesh,matlabarray mlarray)
{
  BASIS basis = field->get_basis();
  matlabarray mlfield;
  
  if (isnodata(basis))
  {
    return (true);
  }
  
  if (islinear(basis)||ishermitian(basis)||isconstant(basis))
  {
    SCIRun::FData3d<SCIRun::Tensor,MESH> &fdata = field->fdata(); 
    
    std::vector<int> dims(4);
    // Note: the dimensions are in reverse order as SCIRun uses C++
    // ordering
    dims[0] = 9; dims[1] = fdata.dim3(); dims[2] = fdata.dim2(); dims[1] = fdata.dim1();
    mlfield.createdensearray(dims,matlabarray::miDOUBLE);
                  
    unsigned int p,q,r,s;
    unsigned int dim1 = fdata.dim1();
    unsigned int dim2 = fdata.dim2();
    unsigned int dim3 = fdata.dim3();  

    SCIRun::Tensor ***dataptr = fdata.get_dataptr();

    std::vector<double> data(dim1*dim2*dim3*9);
    for (p=0,q=0;q<dim1;q++)
      for (r=0;r<dim2;r++)
        for (s=0;s<dim3;s++)
          {
            data[p++] = dataptr[q][r][s].mat_[0][0];
            data[p++] = dataptr[q][r][s].mat_[0][1];
            data[p++] = dataptr[q][r][s].mat_[0][2];
            data[p++] = dataptr[q][r][s].mat_[1][0];
            data[p++] = dataptr[q][r][s].mat_[1][1];
            data[p++] = dataptr[q][r][s].mat_[1][2];
            data[p++] = dataptr[q][r][s].mat_[2][0];
            data[p++] = dataptr[q][r][s].mat_[2][1];
            data[p++] = dataptr[q][r][s].mat_[2][2];
          }         
           
                                           
    mlfield.setnumericarray(data);          
    mlarray.setfield(0,"field",mlfield);
    return(true);
  }
  
  if (islagrangian(basis))
  {
    error("FieldToMatlab: Cannot access the data in the field properly, hence cannot retrieve the data");
    return (false);
  }
  
  error("FieldToMatlab: Unknow basis type");
  return (false);
}

///////////////////////////////////////////////////////////////////

template <class FIELD, class MESH> 
bool FieldToMatlabAlgo::mladdfieldedges(FIELD *field,MESH *mesh,matlabarray mlarray)
{
  typename FIELD::basis_type basis = field->get_basis();
  typename FIELD::mesh_type::basis_type meshbasis = mesh->get_basis();

  matlabarray fieldedge;
  
  if (option_nofieldconnectivity_)
  {
    if ((get_basis_name(meshbasis) == get_basis_name(basis))||(isnodata(basis))||(isconstant(basis)))
    {
      return (true);
    }
  }
    
  if (isnodata(basis))
  {
    return (true);
  }
  
  if (isconstant(basis))
  {
    std::vector<typename FIELD::value_type> &fdata = field->fdata(); 
    fieldedge.createdensearray(1,static_cast<int>(fdata.size()),matlabarray::miUINT32);
    std::vector<unsigned int> mapping(fdata.size());
    for (size_t p = 0; p < fdata.size(); p++)
    {
      mapping[p] = static_cast<unsigned int>(p) + option_indexbase_;
    }
    fieldedge.setnumericarray(mapping);          
    mlarray.setfield(0,"fieldedge",fieldedge);

    return (true);
  }

  if (islinear(basis)||ishermitian(basis))
  {
    typename MESH::Edge::size_type size;
    mesh->size(size);

    typename MESH::basis_type& basis = mesh->get_basis();
    size_t num = basis.number_of_vertices();

    size_t numedges = static_cast<size_t>(size);
    mesh->synchronize(SCIRun::Mesh::EDGES_E); 

    typename MESH::Node::array_type a;
    std::vector<typename MESH::Node::index_type> edges(num*numedges);
    std::vector<int> dims(2);	
    dims[0] = static_cast<int>(num); dims[1] = static_cast<int>(numedges);
    fieldedge.createdensearray(dims,matlabarray::miUINT32);
        
    // SCIRun iterators are limited in supporting any index management
    // Hence I prefer to do it with integer and convert to the required
    // class at the last moment. Hopefully the compiler is smart and
    // has a fast translation. 	
    typename MESH::Edge::iterator it, it_end;
    mesh->begin(it);
    mesh->end(it_end);
    size_t q = 0;

    while (it != it_end)
    {
      mesh->get_nodes(a,*(it));
      for (size_t r = 0; r < num; r++) edges[q++] = a[r] + option_indexbase_;
      ++it;
    }
    
    fieldedge.setnumericarray(edges);          
    mlarray.setfield(0,"fieldedge",fieldedge);
    return (true);
  }

  if (islagrangian(basis))
  {
    error("FieldToMatlab: Cannot access the data in the field properly, hence cannot retrieve the data");
    return (false);
  }
  
  error("FieldToMatlab: Unknow basis type");
  return (false);  
}

template <class FIELD, class MESH> 
bool FieldToMatlabAlgo::mladdfieldfaces(FIELD *field,MESH *mesh,matlabarray mlarray)
{
  typename FIELD::basis_type basis = field->get_basis();
  typename FIELD::mesh_type::basis_type meshbasis = mesh->get_basis();

  matlabarray fieldface;
  
  if (option_nofieldconnectivity_)
  {
    if ((get_basis_name(meshbasis) == get_basis_name(basis))||(isnodata(basis))||(isconstant(basis)))
    {
      return (true);
    }
  }
    
  if (isnodata(basis))
  {
    return (true);
  }
  
  if (isconstant(basis))
  {
    std::vector<typename FIELD::value_type> &fdata = field->fdata(); 
    fieldface.createdensearray(1,static_cast<int>(fdata.size()),matlabarray::miUINT32);
    std::vector<unsigned int> mapping(fdata.size());
    for (size_t p = 0; p < fdata.size(); p++)
    {
      mapping[p] = static_cast<unsigned int>(p) + option_indexbase_;
    }
    fieldface.setnumericarray(mapping);          
    mlarray.setfield(0,"fieldface",fieldface);

    return (true);
  }

  if (islinear(basis)||ishermitian(basis))
  {
    typename MESH::Face::size_type size;
    mesh->size(size);

    typename MESH::basis_type& basis = mesh->get_basis();
    size_t num = basis.number_of_vertices();

    size_t numfaces = static_cast<size_t>(size);
    mesh->synchronize(SCIRun::Mesh::FACES_E); 

    typename MESH::Node::array_type a;
    std::vector<typename MESH::Node::index_type> faces(num*numfaces);
    std::vector<int> dims(2);	
    dims[0] = static_cast<int>(num); dims[1] = static_cast<int>(numfaces);
    fieldface.createdensearray(dims,matlabarray::miUINT32);    
    
    // SCIRun iterators are limited in supporting any index management
    // Hence I prefer to do it with integer and convert to the required
    // class at the last moment. Hopefully the compiler is smart and
    // has a fast translation. 	
    typename MESH::Face::iterator it, it_end;
    mesh->begin(it);
    mesh->end(it_end);
    size_t q = 0;

    while (it != it_end)
    {
      mesh->get_nodes(a,*(it));
      for (size_t r = 0; r < num; r++) faces[q++] = a[r] + option_indexbase_;
      ++it;
    }
    
    fieldface.setnumericarray(faces);          
    mlarray.setfield(0,"fieldface",fieldface);
    return (true);
  }

  if (islagrangian(basis))
  {
    error("FieldToMatlab: Cannot access the data in the field properly, hence cannot retrieve the data");
    return (false);
  }
  
  error("FieldToMatlab: Unknow basis type");
  return (false);  
}


template <class FIELD, class MESH> 
bool FieldToMatlabAlgo::mladdfieldcells(FIELD *field,MESH *mesh,matlabarray mlarray)
{
  typename FIELD::basis_type basis = field->get_basis();
  typename FIELD::mesh_type::basis_type meshbasis = mesh->get_basis();

  matlabarray fieldcell;
  
  if (option_nofieldconnectivity_)
  {
    if ((get_basis_name(meshbasis) == get_basis_name(basis))||(isnodata(basis))||(isconstant(basis)))
    {
      return (true);
    }
  }
    
  if (isnodata(basis))
  {
    return (true);
  }
  
  if (isconstant(basis))
  {
    std::vector<typename FIELD::value_type> &fdata = field->fdata(); 
    fieldcell.createdensearray(1,static_cast<int>(fdata.size()),matlabarray::miUINT32);
    std::vector<unsigned int> mapping(fdata.size());
    for (size_t p = 0; p < fdata.size(); p++)
    {
      mapping[p] = static_cast<unsigned int>(p) + option_indexbase_;
    }
    fieldcell.setnumericarray(mapping);          
    mlarray.setfield(0,"fieldcell",fieldcell);

    return (true);
  }

  if (islinear(basis)||ishermitian(basis))
  {
    typename MESH::Cell::size_type size;
    mesh->size(size);

    typename MESH::basis_type& basis = mesh->get_basis();
    size_t num = basis.number_of_vertices();

    size_t numcells = static_cast<size_t>(size);
    mesh->synchronize(SCIRun::Mesh::CELLS_E); 

    typename MESH::Node::array_type a;
    std::vector<typename MESH::Node::index_type> cells(num*numcells);
    std::vector<int> dims(2);	
    dims[0] = static_cast<int>(num); dims[1] = static_cast<int>(numcells);
    fieldcell.createdensearray(dims,matlabarray::miUINT32);    
    
    // SCIRun iterators are limited in supporting any index management
    // Hence I prefer to do it with integer and convert to the required
    // class at the last moment. Hopefully the compiler is smart and
    // has a fast translation. 	
    typename MESH::Cell::iterator it, it_end;
    mesh->begin(it);
    mesh->end(it_end);
    size_t q = 0;

    while (it != it_end)
    {
      mesh->get_nodes(a,*(it));
      for (size_t r = 0; r < num; r++) cells[q++] = a[r] + option_indexbase_;
      ++it;
    }
    
    fieldcell.setnumericarray(cells);          
    mlarray.setfield(0,"fieldcell",fieldcell);
    return (true);
  }

  if (islagrangian(basis))
  {
    error("FieldToMatlab: Cannot access the data in the field properly, hence cannot retrieve the data");
    return (false);
  }
  
  error("FieldToMatlab: Unknow basis type");
  return (false);  
}

template <class FIELD, class MESH> 
bool FieldToMatlabAlgo::mladdfieldedgederivatives(FIELD *field,MESH *mesh,matlabarray mlarray)
{
  typename FIELD::basis_type basis = field->get_basis();
  
  if (ishermitian(basis))
  {
    error("FieldToMatlab: Cannot access the data in the field properly, hence cannot retrieve the data");
    return (false);
  }

  return (true);
}

template <class FIELD, class MESH> 
bool FieldToMatlabAlgo::mladdfieldfacederivatives(FIELD *field,MESH *mesh,matlabarray mlarray)
{
  typename FIELD::basis_type basis = field->get_basis();
  
  if (ishermitian(basis))
  {
    error("FieldToMatlab: Cannot access the data in the field properly, hence cannot retrieve the data");
    return (false);
  }

  return (true);
}

template <class FIELD, class MESH>
bool FieldToMatlabAlgo::mladdfieldcellderivatives(FIELD *field,MESH *mesh,matlabarray mlarray)
{
  typename FIELD::basis_type basis = field->get_basis();
  
  if (ishermitian(basis))
  {
    error("FieldToMatlab: Cannot access the data in the field properly, hence cannot retrieve the data");
    return (false);
  }

  return (true);
}

template <class FIELD, class MESH> 
bool FieldToMatlabAlgo::mladdfieldderivatives1d(FIELD *field,MESH *mesh,matlabarray mlarray)
{
  typename FIELD::basis_type basis = field->get_basis();
  
  if (ishermitian(basis))
  {
    error("FieldToMatlab: Cannot access the data in the field properly, hence cannot retrieve the data");
    return (false);
  }

  return (true);
}

template <class FIELD, class MESH> 
bool FieldToMatlabAlgo::mladdfieldderivatives2d(FIELD *field,MESH *mesh,matlabarray mlarray)
{
  typename FIELD::basis_type basis = field->get_basis();
  
  if (ishermitian(basis))
  {
    error("FieldToMatlab: Cannot access the data in the field properly, hence cannot retrieve the data");
    return (false);
  }

  return (true);
}

template <class FIELD, class MESH>
bool FieldToMatlabAlgo::mladdfieldderivatives3d(FIELD *field,MESH *mesh,matlabarray mlarray)
{
  typename FIELD::basis_type basis = field->get_basis();
  
  if (ishermitian(basis))
  {
    error("FieldToMatlab: Cannot access the data in the field properly, hence cannot retrieve the data");
    return (false);
  }

  return (true);
}

} // end namespace

#endif
