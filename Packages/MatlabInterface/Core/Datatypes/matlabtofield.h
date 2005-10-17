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

    virtual bool execute(SCIRun::FieldHandle fieldH, matlabarray mlarray);
    static SCIRun::CompileInfoHandle get_compile_info(std::string fielddesc);

    //////// ANALYZE INPUT FUNCTIONS //////////////////////

    long analyze_iscompatible(matlabarray mlarray, std::string& infotext, bool postremark = true);
    long analyze_fieldtype(matlabarray mlarray, std::string& fielddesc);

    inline void setreporter(SCIRun::ProgressReporter* pr);

    
  protected:

    long mlanalyze(matlabarray mlarray, bool postremark);  
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

    std::string fdatatype;
    std::string fieldtype;
    std::string meshtype;
    std::string fieldbasis;
    std::string meshbasis;
    
    std::string meshbasistype;
    std::string fieldbasistype;

    std::vector<long> numnodesvec;
    std::vector<long> numelementsvec;
    
    long              numnodes;
    long              numelements;
    long              numfield;
    long              datasize;

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
    virtual bool execute(SCIRun::FieldHandle fieldH, matlabarray &mlarray);
};

template <class FIELD>  
bool MatlabToFieldAlgoT<FIELD>::execute(SCIRun::FieldHandle fieldH, matlabarray &mlarray)
{
  error("This functionality has disabled for now, new code coming soon");
  return (false);
}


} // end namespace

#endif


