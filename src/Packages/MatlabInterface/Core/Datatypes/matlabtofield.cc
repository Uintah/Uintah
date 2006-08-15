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
 * DATE: 17 OCT 2004
 */

#include <Packages/MatlabInterface/Core/Datatypes/matlabtofield.h>

namespace MatlabIO {

using namespace std;
using namespace SCIRun;

SCIRun::CompileInfoHandle MatlabToFieldAlgo::get_compile_info(std::string fielddesc)
{
  SCIRun::CompileInfoHandle cinfo;
  
  static const std::string include_path(TypeDescription::cc_to_h(__FILE__));
  static const std::string algo_name("MatlabToFieldAlgoT");
  static const std::string base_name("MatlabToFieldAlgo");
  static const std::string name_space("MatlabIO");
  static const std::string name_space2("SCIRun");

  std::string fieldname = DynamicAlgoBase::to_filename(fielddesc);
  std::string filename = algo_name + "." + fieldname + ".";
  
  // Supply the dynamic compiler with enough information to build a file in the
  // on-the-fly libs which will have the templated function in there
  
  cinfo = scinew SCIRun::CompileInfo(filename,base_name, algo_name, fielddesc);
  cinfo->add_namespace(name_space);
  cinfo->add_namespace(name_space2);
  cinfo->add_include(include_path);
  
  // We do not have a field so we rely on the actual name to figure out
  // what files to include

  return(cinfo);
}

bool MatlabToFieldAlgo::execute(SCIRun::FieldHandle& fieldH, matlabarray& mlarray)
{
  error("Failed to execute dynamic algorithm");
  return (false);
}

int MatlabToFieldAlgo::analyze_iscompatible(matlabarray mlarray, std::string& infotext, bool postremark)
{
  infotext = "";
  
  int ret = mlanalyze(mlarray,postremark);
  if (ret == 0) return(0);

  std::ostringstream oss;
  std::string name = mlarray.getname();
  oss << name << " ";
  if (name.length() < 20) oss << string(20-(name.length()),' '); // add some form of spacing                
  
  oss << "[" << meshtype << "<" << meshbasis << "> - ";
  if (fieldbasistype != "nodata")
  {
    std::string fieldtypestr = "Scalar";
    if (fieldtype == "Vector") fieldtypestr = "Vector";
    if (fieldtype == "Tensor") fieldtypestr = "Tensor";
    oss << fieldtypestr << "<" << fieldbasis << "> - ";
  } 
  else
  {
    oss << "NoData - ";
  }
    
  if (numnodesvec.size() > 0)
  {
    for (size_t p = 0; p < numnodesvec.size()-1; p++) oss << numnodesvec[p] << "x";
    oss << numnodesvec[numnodesvec.size()-1];
    oss << " NODES";
  }                  
  else
  {
    oss << numnodes << " NODES " << numelements << " ELEMENTS";
  }                    
  oss << "]";                                            
  infotext = oss.str();

  return (ret);
}

int MatlabToFieldAlgo::analyze_fieldtype(matlabarray mlarray, std::string& fielddesc)
{
  fielddesc = "";
  
  int ret = mlanalyze(mlarray,false);
  if (ret == 0) return(0);

  if (fieldtype == "") fieldtype = "double";
  if (fieldtype == "nodata") fieldtype = "double";  

  fielddesc = "GenericField<"+meshtype+"<"+meshbasis+"<Point> >,"+
              fieldbasis+"<"+fieldtype+">,"+fdatatype+"<"+fieldtype;

  // DEAL WITH SOME MORE SCIRUN INCONSISTENCIES 
  if (fdatatype == "FData2d") fielddesc += "," + meshtype + "<" + meshbasis + "<Point> > ";
  if (fdatatype == "FData3d") fielddesc += "," + meshtype + "<" + meshbasis + "<Point> > ";
   
  fielddesc += "> > ";
  return(1);
}

matlabarray MatlabToFieldAlgo::findfield(matlabarray mlarray,std::string fieldnames)
{
  matlabarray subarray;
  
  while (1)
  {
    size_t loc = fieldnames.find(';');
    if (loc > fieldnames.size()) break;
    std::string fieldname = fieldnames.substr(0,loc);
    fieldnames = fieldnames.substr(loc+1);
    
    int index = mlarray.getfieldnameindexCI(fieldname);
    if (index > -1) 
    {
      subarray = mlarray.getfield(0,index);
      break;
    }
  }
  
  return(subarray);
}

int MatlabToFieldAlgo::mlanalyze(matlabarray mlarray, bool postremark)
{
  int ret = 1;

  if (mlarray.isempty()) 
  {
    if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into SCIRun Field (matrix is empty)"));
    return (0);
  }
  
  if (mlarray.getnumelements() == 0)
  {
    if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into SCIRun Field (matrix is empty)"));
    return (0);
  }
  // If it is regular matrix translate it to a image or a latvol
  // The following section of code rewrites the matlab matrix into a
  // structure and then the normal routine picks up the field and translates it
  // properly.

  if (mlarray.isdense())
  {
    int numdims = mlarray.getnumdims();
    if ((numdims >0)&&(numdims < 4))
    {
      matlabarray ml;
      matlabarray dimsarray;
      std::vector<int> d = mlarray.getdims();
      if ((d[0]==1)||(d[1]==1))
      {
        if (d[0]==1) d[0] = d[1];
        int temp = d[0];
        d.resize(1);
        d[0] = temp;
      }                 
      dimsarray.createintvector(d);
      ml.createstructarray();
      ml.setfield(0,"dims",dimsarray);
      ml.setfield(0,"field",mlarray);
      ml.setname(mlarray.getname());                    
      mlarray = ml;
    }
    else if (numdims == 4)
    {
      matlabarray ml;
      matlabarray dimsarray;
      matlabarray mltype;
      std::vector<int> d = mlarray.getdims();    
    
      if ((d[0] == 1)||(d[0] == 3)||(d[0] == 6)||(d[0] == 9))
      {
        std::vector<int> dm(3);
        for (size_t p = 0; p < 3; p++) dm[p] = d[p+1];
        dimsarray.createintvector(dm);
        if (d[0] == 1) mltype.createstringarray("double");
        if (d[0] == 3) mltype.createstringarray("Vector");
        if ((d[0] == 6)||(d[0] == 9)) mltype.createstringarray("Tensor");
        ml.createstructarray();
        ml.setfield(0,"dims",dimsarray);
        ml.setfield(0,"field",mlarray);
        ml.setfield(0,"fieldtype",mltype);
        ml.setname(mlarray.getname());                    
        mlarray = ml;      
      }
      else if ((d[3] == 1)||(d[3] == 3)||(d[3] == 6)||(d[3] == 9))
      {
        std::vector<int> dm(3);
        for (size_t p = 0; p < 3; p++) dm[p] = d[p];
        dimsarray.createintvector(dm);
        if (d[3] == 1) mltype.createstringarray("double");
        if (d[3] == 3) mltype.createstringarray("Vector");
        if ((d[3] == 6)||(d[3] == 9)) mltype.createstringarray("Tensor");
        ml.createstructarray();
        ml.setfield(0,"dims",dimsarray);
        ml.setfield(0,"field",mlarray);
        ml.setfield(0,"fieldtype",mltype);
        ml.setname(mlarray.getname());                    
        mlarray = ml;              
      }      
    }

    // If the matrix is dense, we score less as we could translate it into a
    // matrix as well. This help for bundle convertions, by which we can 
    // automatically determine how to translate objects.
    ret = 0;
  }
  


  // Check whether we have a structured matrix
  if (!(mlarray.isstruct())) 
  {
    if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into SCIRun Field (matrix is not a structured matrix)"));
    return(0);
  }
  
  
  // Get all the matrices that specify the mesh
  mlnode = findfield(mlarray,"node;pts;pos;");
  mlx =    findfield(mlarray,"x;");      
  mly =    findfield(mlarray,"y;");      
  mlz =    findfield(mlarray,"z;");      
                  
  mledge = findfield(mlarray,"meshedge;edge;line;"); 
  mlface = findfield(mlarray,"meshface;face;quad;fac;tri;");
  mlcell = findfield(mlarray,"meshcell;cell;prism;hex;tet;");
  mldims = findfield(mlarray,"dims;dim;dimension;");
  mltransform = findfield(mlarray,"transform;");

  mlmeshderivatives  = findfield(mlarray,"meshderivatives;derivatives;");
  mlmeshscalefactors = findfield(mlarray,"meshscalefactors;scalefactors;");
  
  mlmeshtype =       findfield(mlarray,"elemtype;meshclass;meshtype;");
  mlmeshbasis =      findfield(mlarray,"meshbasis;basis;");
  mlmeshbasisorder = findfield(mlarray,"meshbasisorder;meshat;meshlocation;");

  // Get all the matrices that specify the field
  
  // Make it compatible with some old versions
  matlabarray mlfieldvector = findfield(mlarray,"vectorfield;vectordata;");
  if (mlfieldvector.isdense()) 
  { 
    mlfield = mlfieldvector; mlfieldtype.createstringarray("vector");
  }
  matlabarray mlfieldtensor = findfield(mlarray,"tensorfield;tensordata;");
  if (mlfieldtensor.isdense()) 
  { 
    mlfield = mlfieldtensor; mlfieldtype.createstringarray("tensor");
  }

  mlfield     = findfield(mlarray,"field;scalarfield;scalardata;potvals;data;");
  mlfieldedge = findfield(mlarray,"fieldedge;edge;line;");
  mlfieldface = findfield(mlarray,"fieldface;face;quad;fac;tri;");
  mlfieldcell = findfield(mlarray,"fieldcell;cell;prism;hex;tet;");

  mlfieldderivatives  = findfield(mlarray,"fieldderivatives;derivatives;");
  mlfieldscalefactors = findfield(mlarray,"fieldscalefactors;scalefactors;");

  mlfieldtype =       findfield(mlarray,"fieldtype;datatype;");
  mlfieldbasis =      findfield(mlarray,"fieldbasis;basis;");
  mlfieldbasisorder = findfield(mlarray,"fieldbasisorder;basisorder;fieldat;fieldlocation;dataat;");

  mlchannels        = findfield(mlarray,"channels;");

  // Figure out the basis type
  // Since we went through several systems over the time
  // the next code is all for compatibility.

  if (!(mlmeshbasisorder.isempty()))
  {   // converter table for the string in the mesh array
    if (mlmeshbasisorder.isstring())
    {
      if ((mlmeshbasisorder.compareCI("node"))||(mlmeshbasisorder.compareCI("pts"))) 
      {
        mlmeshbasisorder.createdoublescalar(1.0);
      }
      else if (mlmeshbasisorder.compareCI("none")||
               mlmeshbasisorder.compareCI("nodata")) 
      {
        mlmeshbasisorder.createdoublescalar(-1.0);
      }
      else if (mlmeshbasisorder.compareCI("egde")||
          mlmeshbasisorder.compareCI("line")||mlmeshbasisorder.compareCI("face")||
          mlmeshbasisorder.compareCI("fac")||mlmeshbasisorder.compareCI("cell")||
          mlmeshbasisorder.compareCI("tet")||mlmeshbasisorder.compareCI("hex")||
          mlmeshbasisorder.compareCI("prism")) 
      {
        mlmeshbasisorder.createdoublescalar(0.0);
      }
    }
  }

  if (mlmeshbasis.isstring())
  {
    std::string str = mlmeshbasis.getstring();
    for (size_t p = 0; p < str.size(); p++) str[p] = tolower(str[p]);
    
    if      (str.find("nodata") != std::string::npos)    meshbasistype = "nodata";
    else if (str.find("constant") != std::string::npos)  meshbasistype = "constant";
    else if (str.find("linear") != std::string::npos)    meshbasistype = "linear";
    else if (str.find("quadratic") != std::string::npos) meshbasistype = "quadratic";
    else if (str.find("cubic") != std::string::npos)     meshbasistype = "cubic";
  }

  if ((meshbasistype == "")&&(mlmeshbasisorder.isdense()))
  {
    std::vector<int> data;
    mlmeshbasisorder.getnumericarray(data); 
    
    if (data.size() > 0) 
    {
      if (data[0] == -1) meshbasistype = "nodata";
      if (data[0] ==  0) meshbasistype = "constant";
      if (data[0] ==  1) meshbasistype = "linear";
      if (data[0] ==  2) meshbasistype = "quadratic";
    }
  }

  // figure out the basis of the field

  if (!(mlfieldbasisorder.isempty()))
  {   // converter table for the string in the field array
    if (mlfieldbasisorder.isstring())
    {
      if (mlfieldbasisorder.compareCI("node")||mlfieldbasisorder.compareCI("pts")) 
      {
        mlfieldbasisorder.createdoublescalar(1.0);
      }
      else if (mlfieldbasisorder.compareCI("none")||
               mlfieldbasisorder.compareCI("nodata")) 
      {
        mlfieldbasisorder.createdoublescalar(-1.0);
      }
      else if (mlfieldbasisorder.compareCI("egde")||
          mlfieldbasisorder.compareCI("line")||mlfieldbasisorder.compareCI("face")||
          mlfieldbasisorder.compareCI("fac")||mlfieldbasisorder.compareCI("cell")||
          mlfieldbasisorder.compareCI("tet")||mlfieldbasisorder.compareCI("hex")||
          mlfieldbasisorder.compareCI("prism")) 
      {
        mlfieldbasisorder.createdoublescalar(0.0);
      }
    }
  }

  if (mlfieldbasis.isstring())
  {
    std::string str = mlfieldbasis.getstring();
    for (size_t p = 0; p < str.size(); p++) str[p] = tolower(str[p]);
        
    if (str.find("nodata") != std::string::npos)         fieldbasistype = "nodata";
    else if (str.find("constant") != std::string::npos)  fieldbasistype = "constant";
    else if (str.find("linear") != std::string::npos)    fieldbasistype = "linear";
    else if (str.find("quadratic") != std::string::npos) fieldbasistype = "quadratic";
    else if (str.find("cubic") != std::string::npos)     fieldbasistype = "cubic";
  }

  if ((fieldbasistype == "")&&(mlfieldbasisorder.isdense()))
  {
    std::vector<int> data;
    mlfieldbasisorder.getnumericarray(data); 
    if (data.size() > 0) 
    {
      if (data[0] == -1) fieldbasistype = "nodata";
      if (data[0] ==  0) fieldbasistype = "constant";
      if (data[0] ==  1) fieldbasistype = "linear";
      if (data[0] ==  2) fieldbasistype = "quadratic";
    }
  }

  // Figure out the fieldtype

  fieldtype = "";
  if (mlfield.isdense())
  {
    if (mlfieldtype.isstring())
    {
      if (mlfieldtype.compareCI("nodata"))             fieldtype = "nodata";
      if (mlfieldtype.compareCI("vector"))             fieldtype = "Vector";
      if (mlfieldtype.compareCI("tensor"))             fieldtype = "Tensor";
      if (mlfieldtype.compareCI("double"))             fieldtype = "double";
      if (mlfieldtype.compareCI("float"))              fieldtype = "float";
      if (mlfieldtype.compareCI("long long"))          fieldtype = "long long";
      if (mlfieldtype.compareCI("unsigned long long")) fieldtype = "unsigned long long";      
      if (mlfieldtype.compareCI("long"))               fieldtype = "long";
      if (mlfieldtype.compareCI("unsigned long"))      fieldtype = "unsigned long";      
      if (mlfieldtype.compareCI("int"))                fieldtype = "int";
      if (mlfieldtype.compareCI("unsigned int"))       fieldtype = "unsigned int";
      if (mlfieldtype.compareCI("short"))              fieldtype = "short";
      if (mlfieldtype.compareCI("unsigned short"))     fieldtype = "unsigned short";
      if (mlfieldtype.compareCI("char"))               fieldtype = "char";
      if (mlfieldtype.compareCI("unsigned char"))      fieldtype = "unsigned char";
    }
  } 
  
  if ((fieldtype == "nodata")||(mlfield.isempty()))
  {
    fieldbasis = "NoDataBasis";
    fieldbasistype = "nodata";
    fieldtype = "double";
  }
  
  // if no meshbasistype is supplied we need to figure it out on the fly

  // Now figure out the mesh type and check we have everything for that meshtype

  meshtype = "";
  if (mlmeshtype.isstring())
  {
    std::string str = mlmeshtype.getstring();
    for (size_t p = 0; p < str.size(); p++) str[p] = tolower(str[p]);
    
    if (str.find("pointcloud") != std::string::npos)          meshtype = "PointCloudMesh";
    else if (str.find("scanline")       != std::string::npos) meshtype = "ScanlineMesh";
    else if (str.find("image")          != std::string::npos) meshtype = "ImageMesh";
    else if (str.find("latvol")         != std::string::npos) meshtype = "LatVolMesh";
    else if (str.find("structcurve")    != std::string::npos) meshtype = "StructCurveMesh";
    else if (str.find("structquadsurf") != std::string::npos) meshtype = "StructQuadSurfMesh";
    else if (str.find("structhexvol")   != std::string::npos) meshtype = "StructHexVolMesh";    
    else if (str.find("curve")          != std::string::npos) meshtype = "CurveMesh";
    else if (str.find("trisurf")        != std::string::npos) meshtype = "TriSurfMesh";
    else if (str.find("quadsurf")       != std::string::npos) meshtype = "QuadSurfMesh";
    else if (str.find("tetvol")         != std::string::npos) meshtype = "TetVolMesh";
    else if (str.find("prismvol")       != std::string::npos) meshtype = "PrismVolMesh";
    else if (str.find("hexvol")         != std::string::npos) meshtype = "HexVolMesh";
  }

  fdatatype = "vector";
  numnodes = 0;
  numelements = 0;
  numnodesvec.clear();
  numelementsvec.clear();

  if (mltransform.isdense())
  {
    if (mltransform.getnumdims() != 2) 
    {
      if (postremark) remark(std::string("Matrix '" + mlarray.getname() +
       "' cannot be translated into a SCIRun Field (transformation matrix is not 2D)"));
      return(0);
    }
    if ((mltransform.getn() != 4)&&(mltransform.getm() != 4))
    {
      if (postremark) remark(std::string("Matrix '" + mlarray.getname() + 
      "' cannot be translated into a SCIRun Field (transformation matrix is not 4x4)"));
      return(0);
    }
  }

  if (mlx.isdense()||mly.isdense()||mly.isdense())
  {
    if (mlx.isempty()||mly.isempty()||mlz.isempty())
    {
      if (postremark) remark(std::string("Matrix '" + mlarray.getname() + 
      "' cannot be translated into a SCIRun Field (does not have a complete set of  x, y, or z coodinates"));
      return (0);
    }
  }

  // FIGURE OUT THE REGULAR MESH STUFF
  // CHECK WHETHER IT IS ONE, AND IF SO CHECK ALL THE DATA

  if (((mlnode.isempty()&&(mlx.isempty()))||
       (mltransform.isdense())||(meshtype  == "ScanlineMesh")||
       (meshtype == "ImageMesh")||(meshtype == "LatVolMesh"))&&(mldims.isempty()))
  {
    if (mlfield.isdense())
    {
      std::vector<int> dims = mlfield.getdims();

      if ((fieldtype == "")&&(dims.size() > 3))
      {
        if (dims[0] == 3) fieldtype = "Vector";
        if ((dims[0] == 6)||(dims[0] == 9)) fieldtype = "Tensor";
      }

      if ((fieldtype == "Vector")||(fieldtype == "Tensor"))
      {
        if (fieldbasistype == "quadratic")
        {
          if (dims.size() > 2) mldims.createintvector(static_cast<int>((dims.size()-2)),&(dims[1]));        
        }
        else
        {
          if (dims.size() > 1) mldims.createintvector(static_cast<int>((dims.size()-1)),&(dims[1]));
        }
      }
      else
      {
        if (fieldbasistype == "quadratic")
        {
          if (dims.size() > 1) mldims.createintvector(static_cast<int>((dims.size()-1)),&(dims[0]));        
        }
        else
        {  
          mldims.createintvector(dims);
        }
      }
    }
    
    if (fieldbasistype == "constant")
    {
      std::vector<int> dims = mlfield.getdims();
      // dimensions need to be one bigger
      for (int p = 0; p<static_cast<int>(dims.size()); p++) dims[p] = dims[p]+1;
      mldims.createintvector(dims);
    }
  }

  // CHECK WHETHER WE HAVE A REGULAR FIELD  

  if ((mldims.isempty())&&(mlx.isempty())&&(mlnode.isempty()))
  {
    if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated because no node, x, or dims field was found"));
  }

  // WE HAVE POSSIBLY A FIELD
  // NOW CHECK EVERY COMBINATION

  // HANDLE REGULAR MESHES

  numfield = 0;  
  datasize = 1;

  if (mldims.isdense())
  {
    size_t size = static_cast<size_t>(mldims.getnumelements());
                                                     
    if (!((size > 0)&&(size < 4)))
    {
      if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (number of dimensions (.dims field) needs to 1, 2, or 3)"));
      return(0);
    }
    
    if (meshtype != "")
    {   // explicitly stated type: (check whether type confirms the guessed type, otherwise someone supplied us with improper data)
      if ((meshtype == "ScanlineMesh")&&(size!=1))
      {
        if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (scanline needs only one dimension)"));
        return(0);
      }
      if ((meshtype == "ImageMesh")&&(size!=2)) 
      {
        if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (an image needs two dimensions)"));
        return(0);
      }
      if ((meshtype == "LatVolMesh")&&(size!=3))
      {
        if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (a latvolmesh needs three dimensions)"));
        return(0);
      }
    } 
    else
    {
      if (size == 1) meshtype = "ScanlineMesh";
      if (size == 2) meshtype = "ImageMesh";
      if (size == 3) meshtype = "LatVolMesh";        
    }

    // We always make this into a linear one
    if (meshbasistype == "") meshbasistype = "linear";
    
    if (meshbasistype != "linear")
    {
      if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (regular meshes cannot have higher order basis)"));
      return(0);      
    }
    
    if (meshtype == "ScanlineMesh") { meshbasis = "CrvLinearLgn";    fdatatype = "vector"; }
    if (meshtype == "ImageMesh")    { meshbasis = "QuadBilinearLgn"; fdatatype = "FData2d"; }
    if (meshtype == "LatVolMesh")   { meshbasis = "HexTrilinearLgn"; fdatatype = "FData3d"; }
    
    // compute number of elements and number of nodes
    mldims.getnumericarray(numnodesvec);
    numelementsvec = numnodesvec;
    // Number of elements is one less than the dimension in a certain direction
    for (size_t p = 0; p < numnodesvec.size(); p++) numelementsvec[p]--;
    
    // try to figure out the field basis
    if (fieldbasistype == "")
    {
      // In case no data is there
      if (mlfield.isempty())
      {
        fieldbasistype = "nodata";
        fieldtype = "";
      }
      else
      { 
        if (fieldtype == "") fieldtype = "double";
        
        std::vector<int> fdims = mlfield.getdims();
        if (fieldtype == "Vector")
        {
          if (fdims[0] != 3)
          {
            if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (first dimension of field needs to be 3)"));
            return(0);      
          }          
          std::vector<int> temp(fdims.size()-1);
          for (size_t p = 0; p < temp.size(); p++) temp[p] = fdims[p+1];
          fdims = temp;
        }
        if (fieldtype == "Tensor")
        {
          if ((fdims[0] != 6)&&(fdims[0] != 9))
          {
            if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (first dimension of field needs to be 6, or 9)"));
            return(0);      
          }          
          std::vector<int> temp(fdims.size()-1);
          for (size_t p = 0; p < temp.size(); p++) temp[p] = fdims[p+1];
          fdims = temp;
        }
        
        if ((size == 1)&&(size == fdims.size())&&(fdims[0] == numnodesvec[0])) fieldbasistype = "linear";
        if ((size == 2)&&(size == fdims.size())&&(fdims[0] == numnodesvec[0])&&(fdims[1] == numnodesvec[1])) fieldbasistype = "linear";
        if ((size == 3)&&(size == fdims.size())&&(fdims[0] == numnodesvec[0])&&(fdims[1] == numnodesvec[1])&&(fdims[2] == numnodesvec[2])) fieldbasistype = "linear";

        if ((size == 1)&&(size+1 == fdims.size())&&(fdims[0] == numnodesvec[0])&&(fdims[1] == 2)) fieldbasistype = "quadratic";
        if ((size == 2)&&(size+1 == fdims.size())&&(fdims[0] == numnodesvec[0])&&(fdims[1] == numnodesvec[1])&&(fdims[2] == 3)) fieldbasistype = "quadratic";
        if ((size == 3)&&(size+1 == fdims.size())&&(fdims[0] == numnodesvec[0])&&(fdims[1] == numnodesvec[1])&&(fdims[2] == numnodesvec[2])&&(fdims[3] == 4)) fieldbasistype = "quadratic";

        if ((size == 1)&&(size == fdims.size())&&(fdims[0] == numelementsvec[0])) fieldbasistype == "constant";
        if ((size == 2)&&(size == fdims.size())&&(fdims[0] == numelementsvec[0])&&(fdims[1] == numelementsvec[1])) fieldbasistype = "constant";
        if ((size == 3)&&(size == fdims.size())&&(fdims[0] == numelementsvec[0])&&(fdims[1] == numelementsvec[1])&&(fdims[2] == numelementsvec[2])) fieldbasistype = "constant";

        if ((mlfieldderivatives.isdense())&&(fieldbasis == "linear")) fieldbasistype = "cubic";
      }
    }
    //by now we should know what kind of basis we would like

    if (fieldbasistype == "")
    {
      if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (the dimensions field matrix do not match mesh)"));
      return(0);    
    }

    
    if (fieldbasistype == "nodata") fieldbasis = "NoDataBasis";
    if (fieldbasistype == "constant") fieldbasis = "ConstantBasis";

    if (fieldbasistype == "linear") 
    {
      std::vector<int> fdims = mlfield.getdims();
      if (fieldtype == "Vector")
      {
        if (fdims[0] != 3)
        {
          if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (first dimension of field needs to be 3"));
          return(0);      
        }
        std::vector<int> temp(fdims.size()-1);
        for (size_t p = 0; p < temp.size(); p++) temp[p] = fdims[p+1];
        fdims = temp;
        datasize = fdims[0];
      }
      if (fieldtype == "Tensor")
      {
        if ((fdims[0] != 6)&&(fdims[0] != 9))
        {
          if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (first dimension of field needs to be 6, or 9)"));
          return(0);      
        }
        std::vector<int> temp(fdims.size()-1);
        for (size_t p = 0; p < temp.size(); p++) temp[p] = fdims[p+1];
        fdims = temp;
        datasize = fdims[0];
      }      
      
      if ((!((size == 1)&&(size == fdims.size())&&(fdims[0] == numnodesvec[0]))) &&
          (!((size == 2)&&(size == fdims.size())&&(fdims[0] == numnodesvec[0])&&(fdims[1] == numnodesvec[1]))) &&
          (!((size == 3)&&(size == fdims.size())&&(fdims[0] == numnodesvec[0])&&(fdims[1] == numnodesvec[1])&&(fdims[2] == numnodesvec[2]))))
      {
        if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (dimensions of field do not match dimensions of mesh"));
        return(0);          
      }
      if (meshtype == "ScanlineMesh") { fieldbasis = "CrvLinearLgn";}
      if (meshtype == "ImageMesh")    { fieldbasis = "QuadBilinearLgn"; }
      if (meshtype == "LatVolMesh")   { fieldbasis = "HexTrilinearLgn"; }      
    }
    
    if (fieldbasistype == "quadratic") 
    {
      std::vector<int> fdims = mlfield.getdims();
      if (fieldtype == "Vector")
      {
        if (fdims[0] != 3)
        {
          if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (first dimension of field needs to be 3"));
          return(0);      
        }
        std::vector<int> temp(fdims.size()-1);
        for (size_t p = 0; p < temp.size(); p++) temp[p] = fdims[p+1];
        fdims = temp;
        datasize = fdims[0];        
      }
      if (fieldtype == "Tensor")
      {
        if ((fdims[0] != 6)&&(fdims[0] != 9))
        {
          if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (first dimension of field needs to be 6, or 9)"));
          return(0);      
        }
        std::vector<int> temp(fdims.size()-1);
        for (size_t p = 0; p < temp.size(); p++) temp[p] = fdims[p+1];
        fdims = temp;
        datasize = fdims[0];        
      }          
   
      if ((!((size == 1)&&(size+1 == fdims.size())&&(fdims[0] == numnodesvec[0])&&(fdims[1] == 2))) &&
          (!((size == 2)&&(size+1 == fdims.size())&&(fdims[0] == numnodesvec[0])&&(fdims[1] == numnodesvec[1])&&(fdims[2] == 3))) &&
          (!((size == 3)&&(size+1 == fdims.size())&&(fdims[0] == numnodesvec[0])&&(fdims[1] == numnodesvec[1])&&(fdims[2] == numnodesvec[2])&&(fdims[3] == 4))))
      {
        if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (dimensions of field do not match dimensions of mesh"));
        return(0);          
      }
    
      if (meshtype == "ScanlineMesh") { fieldbasis = "CrvQuadraticLgn";}
      if (meshtype == "ImageMesh")    { fieldbasis = "QuadBiquadraticLgn"; }
      if (meshtype == "LatVolMesh")   { fieldbasis = "HexTriquadraticLgn"; }      
    }
       
    if (fieldbasistype == "cubic")
    {
      std::vector<int> fdims = mlfield.getdims();
      if (fieldtype == "Vector")
      {
        if (fdims[0] != 3)
        {
          if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (first dimension of field needs to be 3"));
          return(0);      
        }
        std::vector<int> temp(fdims.size()-1);
        for (size_t p = 0; p < temp.size(); p++) temp[p] = fdims[p+1];
        fdims = temp;
        datasize = fdims[0];        
      }
      if (fieldtype == "Tensor")
      {
        if ((fdims[0] != 6)&&(fdims[0] != 9))
        {
          if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (first dimension of field needs to be 6, or 9)"));
          return(0);      
        }
        std::vector<int> temp(fdims.size()-1);
        for (size_t p = 0; p < temp.size(); p++) temp[p] = fdims[p+1];
        fdims = temp;
        datasize = fdims[0];        
      }     
 
      if (meshtype == "ScanlineMesh") { fieldbasis = "CrvCubicHmt";}
      if (meshtype == "ImageMesh")    { fieldbasis = "QuadBicubicLgn"; }
      if (meshtype == "LatVolMesh")   { fieldbasis = "HexTricubicLgn"; }       
    
      if (mlfieldderivatives.isdense())
      {
        std::vector<int> derivativesdims = mlfieldderivatives.getdims();
        std::vector<int> fielddims = mlfieldderivatives.getdims();
        
        if (meshtype == "ScanlineMesh")
        {
          if (derivativesdims.size() != size+2)
          {
            if (postremark) remark(std::string("Matrix '"+mlarray.getname()+"' cannot be translated into a SCIRun Field, the derivative matrix should have two more dimensions then the field matrix"));
            return (0);            
          }

          if ((derivativesdims[0] != 1)||(derivativesdims[1] != datasize)||(derivativesdims[2] != fielddims[0]))
          {
            if (postremark) remark(std::string("Matrix '"+mlarray.getname()+"' cannot be translated into a SCIRun Field, the derivative matrix and the field matrix do not match"));
            return (0);         
          }        
        }
      
        if (meshtype == "ImageMesh")
        {
          if (derivativesdims.size() != size+2)
          {
            if (postremark) remark(std::string("Matrix '"+mlarray.getname()+"' cannot be translated into a SCIRun Field, the derivative matrix should have two more dimensions then the field matrix"));
            return (0);            
          }

          if ((derivativesdims[0] != 2)||(derivativesdims[1] != datasize)||(derivativesdims[2] != fielddims[0])||(derivativesdims[3] != fielddims[1]))
          {
            if (postremark) remark(std::string("Matrix '"+mlarray.getname()+"' cannot be translated into a SCIRun Field, the derivative matrix and the field matrix do not match"));
            return (0);         
          }        
        }

        if (meshtype == "LatVolMesh")
        {
          if (derivativesdims.size() != size+2)
          {
            if (postremark) remark(std::string("Matrix '"+mlarray.getname()+"' cannot be translated into a SCIRun Field, the derivative matrix should have two more dimensions then the field matrix"));
            return (0);            
          }

          if ((derivativesdims[0] != 7)||(derivativesdims[1] != datasize)||(derivativesdims[2] != fielddims[0])||(derivativesdims[3] != fielddims[1])||(derivativesdims[4] != fielddims[2]))
          {
            if (postremark) remark(std::string("Matrix '"+mlarray.getname()+"' cannot be translated into a SCIRun Field, the derivative matrix and the field matrix do not match"));
            return (0);         
          }        
        }        
      }
      else
      {
        if (postremark) remark(std::string("Matrix '"+mlarray.getname()+"' cannot be translated into a SCIRun Field, the derivative matrix and the field matrix do not match"));
        return (0);
      }
    }    
    
    return(1+ret);                                    
  }
   
  /////////////////////////////////////////////////// 
  // DEAL WITH STRUCTURED MESHES

  if ((mlx.isdense())&&(mly.isdense())&(mlz.isdense()))
  {
      
    // TEST: The dimensions of the x, y, and z ,atrix should be equal

    int size = mlx.getnumdims();
    if (mly.getnumdims() != size) 
    {
      if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (the dimensions of the x and y matrix do not match)"));
      return(0);
    }
    if (mlz.getnumdims() != size) 
    {
      if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (the dimensions of the x and z matrix do not match)"));
      return(0);
    }
                
    std::vector<int> dimsx = mlx.getdims();
    std::vector<int> dimsy = mly.getdims();
    std::vector<int> dimsz = mlz.getdims();
                
    // Check dimension by dimension for any problems
    for (int p=0 ; p < size ; p++)
    {
      if(dimsx[p] != dimsy[p]) 
      {
        if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (the dimensions of the x and y matrix do not match)"));
        return(0);
      }
      if(dimsx[p] != dimsz[p]) 
      {
        if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (the dimensions of the x and z matrix do not match)"));
        return(0);
      }
    }
              
    // Minimum number of dimensions is in matlab is 2 and hence detect any empty dimension
    if (size == 2)
    {
      // This case will filter out the scanline objects
      // Currently SCIRun will fail/crash with an image where one of the
      // dimensions is one, hence prevent some troubles
      if ((dimsx[0] == 1)||(dimsx[1] == 1)) size = 1;
    }

    // Disregard data at odd locations. The translation function for those is not straight forward
    // Hence disregard those data locations.
    
    if ((fieldtype == "Vector")||(fieldtype == "Tensor")) size--;

    if (meshtype != "")
    {   // explicitly stated type (check whether type confirms the guessed type, otherwise someone supplied us with improper data)
      if ((meshtype == "StructCurveMesh")&&(size!=1))
      {
        if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (invalid number of dimensions for x, y, and z matrix)"));
        return(0);
      }
      if ((meshtype == "StructQuadSurfMesh")&&(size!=2)) 
      {
        if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (invalid number of dimensions for x, y, and z matrix)"));
        return(0);
      }
      if ((meshtype == "StructHexVolMesh")&&(size!=3)) 
      {
        if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (invalid number of dimensions for x, y, and z matrix)"));
        return(0);
      }
    }           
      
    if (size == 1) { meshtype = "StructCurveMesh"; }   
    if (size == 2) { meshtype = "StructQuadSurfMesh"; }   
    if (size == 3) { meshtype = "StructHexVolMesh"; }   

    std::vector<int> dims = mlx.getdims();  
    if ((fieldtype == "Vector")||(fieldtype == "Tensor"))
    {
      std::vector<int> temp(dims.size()-1);
      for (size_t p=0; p < dims.size()-1; p++) temp[p] = dims[p];
      dims = temp;
    }

    numnodesvec = dims;
    numelementsvec = dims;
    for (size_t p = 0; p < numnodesvec.size(); p++) numelementsvec[p]--;
    
    if (meshtype == "")
    {
      if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (unknown mesh type)"));
      return(0);      
    }

    // We always make this into a linear one
    if (meshbasistype == "") meshbasistype = "linear";
    
    if (meshbasistype != "linear")
    {
      if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (geometrical higher order basis for structured meshes is not yet supported)"));
      return(0);      
    }
    
    if (meshtype == "StructCurveMesh")    { meshbasis = "CrvLinearLgn";   fdatatype = "vector"; }
    if (meshtype == "StructQuadSurfMesh") { meshbasis = "QuadBilinearLgn"; fdatatype = "FData2d"; }
    if (meshtype == "StructHexVolMesh")   { meshbasis = "HexTrilinearLgn"; fdatatype = "FData3d"; }

    // We should have a meshbasis and a meshtype by now
    
    // try to figure out the field basis
    if (fieldbasistype == "")
    {
      // In case no data is there
      if (mlfield.isempty())
      {
        fieldbasistype = "nodata";
        fieldtype = "";
      }
      else
      { 
        if (fieldtype == "") fieldtype = "double";
      
        std::vector<int> fdims = mlfield.getdims();
        if (fieldtype == "Vector")
        {
          if (fdims[0] != 3)
          {
            if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (first dimension of field needs to be 3)"));
            return(0);      
          }          
          std::vector<int> temp(fdims.size()-1);
          for (size_t p = 0; p < temp.size(); p++) temp[p] = fdims[p+1];
          fdims = temp;
        }
        if (fieldtype == "Tensor")
        {
          if ((fdims[0] != 6)&&(fdims[0] != 9))
          {
            if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (first dimension of field needs to be 6, or 9)"));
            return(0);      
          }          
          std::vector<int> temp(fdims.size()-1);
          for (size_t p = 0; p < temp.size(); p++) temp[p] = fdims[p+1];
          fdims = temp;
        }
        
        if ((size == 1)&&(size == fdims.size())&&(fdims[0] == numnodesvec[0])) fieldbasistype = "linear";
        if ((size == 2)&&(size == fdims.size())&&(fdims[0] == numnodesvec[0])&&(fdims[1] == numnodesvec[1])) fieldbasistype = "linear";
        if ((size == 3)&&(size == fdims.size())&&(fdims[0] == numnodesvec[0])&&(fdims[1] == numnodesvec[1])&&(fdims[2] == numnodesvec[2])) fieldbasistype = "linear";

        if ((size == 1)&&(size+1 == fdims.size())&&(fdims[0] == numnodesvec[0])&&(fdims[1] == 2)) fieldbasistype = "quadratic";
        if ((size == 2)&&(size+1 == fdims.size())&&(fdims[0] == numnodesvec[0])&&(fdims[1] == numnodesvec[1])&&(fdims[2] == 3)) fieldbasistype = "quadratic";
        if ((size == 3)&&(size+1 == fdims.size())&&(fdims[0] == numnodesvec[0])&&(fdims[1] == numnodesvec[1])&&(fdims[2] == numnodesvec[2])&&(fdims[3] == 4)) fieldbasistype = "quadratic";

        if ((size == 1)&&(size == fdims.size())&&(fdims[0] == numelementsvec[0])) fieldbasistype = "constant";
        if ((size == 2)&&(size == fdims.size())&&(fdims[0] == numelementsvec[0])&&(fdims[1] == numelementsvec[1])) fieldbasistype = "constant";
        if ((size == 3)&&(size == fdims.size())&&(fdims[0] == numelementsvec[0])&&(fdims[1] == numelementsvec[1])&&(fdims[2] == numelementsvec[2])) fieldbasistype = "constant";

        if ((mlfieldderivatives.isdense())&&(fieldbasistype == "linear")) fieldbasistype = "cubic";
      }
    }
    //by now we should know what kind of basis we would like

    if (fieldbasistype == "")
    {
      if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (the dimensions field matrix do not match mesh)"));
      return(0);    
    }

    if (fieldbasistype == "nodata") fieldbasis = "NoDataBasis";
    if (fieldbasistype == "constant") fieldbasis = "ConstantBasis";

    if (fieldbasistype == "linear") 
    {
      std::vector<int> fdims = mlfield.getdims();
      if (fieldtype == "Vector")
      {
        if (fdims[0] != 3)
        {
          if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (first dimension of field needs to be 3"));
          return(0);      
        }
        std::vector<int> temp(fdims.size()-1);
        for (size_t p = 0; p < temp.size(); p++) temp[p] = fdims[p+1];
        fdims = temp;
        datasize = fdims[0];
      }
      if (fieldtype == "Tensor")
      {
        if ((fdims[0] != 6)&&(fdims[0] != 9))
        {
          if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (first dimension of field needs to be 6, or 9)"));
          return(0);      
        }
        std::vector<int> temp(fdims.size()-1);
        for (size_t p = 0; p < temp.size(); p++) temp[p] = fdims[p+1];
        fdims = temp;
        datasize = fdims[0];
      }      
      
      if ((!((size == 1)&&(size == fdims.size())&&(fdims[0] == numnodesvec[0]))) &&
          (!((size == 2)&&(size == fdims.size())&&(fdims[0] == numnodesvec[0])&&(fdims[1] == numnodesvec[1]))) &&
          (!((size == 3)&&(size == fdims.size())&&(fdims[0] == numnodesvec[0])&&(fdims[1] == numnodesvec[1])&&(fdims[2] == numnodesvec[2]))))
      {
        if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (dimensions of field do not match dimensions of mesh"));
        return(0);          
      }
      if (meshtype == "StructCurveMesh")    { fieldbasis = "CrvLinearLgn";}
      if (meshtype == "StructQuadSurfMesh") { fieldbasis = "QuadBilinearLgn"; }
      if (meshtype == "StructHexVolMesh")   { fieldbasis = "HexTrilinearLgn"; }      
    }
    
    if (fieldbasistype == "quadratic") 
    {
      std::vector<int> fdims = mlfield.getdims();
      if (fieldtype == "Vector")
      {
        if (fdims[0] != 3)
        {
          if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (first dimension of field needs to be 3"));
          return(0);      
        }
        std::vector<int> temp(fdims.size()-1);
        for (size_t p = 0; p < temp.size(); p++) temp[p] = fdims[p+1];
        fdims = temp;
        datasize = fdims[0];        
      }
      if (fieldtype == "Tensor")
      {
        if ((fdims[0] != 6)&&(fdims[0] != 9))
        {
          if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (first dimension of field needs to be 6, or 9)"));
          return(0);      
        }
        std::vector<int> temp(fdims.size()-1);
        for (size_t p = 0; p < temp.size(); p++) temp[p] = fdims[p+1];
        fdims = temp;
        datasize = fdims[0];        
      }          
   
      if ((!((size == 1)&&(size+1 == fdims.size())&&(fdims[0] == numnodesvec[0])&&(fdims[1] == 2))) &&
          (!((size == 2)&&(size+1 == fdims.size())&&(fdims[0] == numnodesvec[0])&&(fdims[1] == numnodesvec[1])&&(fdims[2] == 3))) &&
          (!((size == 3)&&(size+1 == fdims.size())&&(fdims[0] == numnodesvec[0])&&(fdims[1] == numnodesvec[1])&&(fdims[2] == numnodesvec[2])&&(fdims[3] == 4))))
      {
        if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (dimensions of field do not match dimensions of mesh"));
        return(0);          
      }
    
      if (meshtype == "StructCurveMesh")    { fieldbasis = "CrvQuadraticLgn";}
      if (meshtype == "StructQuadSurfMesh") { fieldbasis = "QuadBiquadraticLgn"; }
      if (meshtype == "StructHexVolMesh")   { fieldbasis = "HexTriquadraticLgn"; }      
    }
       
    if (fieldbasistype == "cubic")
    {
      std::vector<int> fdims = mlfield.getdims();
      if (fieldtype == "Vector")
      {
        if (fdims[0] != 3)
        {
          if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (first dimension of field needs to be 3"));
          return(0);      
        }
        std::vector<int> temp(fdims.size()-1);
        for (size_t p = 0; p < temp.size(); p++) temp[p] = fdims[p+1];
        fdims = temp;
        datasize = fdims[0];        
      }
      if (fieldtype == "Tensor")
      {
        if ((fdims[0] != 6)&&(fdims[0] != 9))
        {
          if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (first dimension of field needs to be 6, or 9)"));
          return(0);      
        }
        std::vector<int> temp(fdims.size()-1);
        for (size_t p = 0; p < temp.size(); p++) temp[p] = fdims[p+1];
        fdims = temp;
        datasize = fdims[0];        
      }     
 
      if (meshtype == "StructCurveMesh")    { fieldbasis = "CrvCubicHmt";}
      if (meshtype == "StructQuadSurfMesh") { fieldbasis = "QuadBicubicLgn"; }
      if (meshtype == "StructHexVolMesh")   { fieldbasis = "HexTricubicLgn"; }       
    
      if (mlfieldderivatives.isdense())
      {
        std::vector<int> derivativesdims = mlfieldderivatives.getdims();
        std::vector<int> fielddims = mlfieldderivatives.getdims();
        
        if (meshtype == "StructCurveMesh")
        {
          if (derivativesdims.size() != size+2)
          {
            if (postremark) remark(std::string("Matrix '"+mlarray.getname()+"' cannot be translated into a SCIRun Field, the derivative matrix should have two more dimensions then the field matrix"));
            return (0);            
          }

          if ((derivativesdims[0] != 1)||(derivativesdims[1] != datasize)||(derivativesdims[2] != fielddims[0]))
          {
            if (postremark) remark(std::string("Matrix '"+mlarray.getname()+"' cannot be translated into a SCIRun Field, the derivative matrix and the field matrix do not match"));
            return (0);         
          }        
        }
      
        if (meshtype == "StructQuadSurfMesh")
        {
          if (derivativesdims.size() != size+2)
          {
            if (postremark) remark(std::string("Matrix '"+mlarray.getname()+"' cannot be translated into a SCIRun Field, the derivative matrix should have two more dimensions then the field matrix"));
            return (0);            
          }

          if ((derivativesdims[0] != 2)||(derivativesdims[1] != datasize)||(derivativesdims[2] != fielddims[0])||(derivativesdims[3] != fielddims[1]))
          {
            if (postremark) remark(std::string("Matrix '"+mlarray.getname()+"' cannot be translated into a SCIRun Field, the derivative matrix and the field matrix do not match"));
            return (0);         
          }        
        }

        if (meshtype == "StructHexVolMesh")
        {
          if (derivativesdims.size() != size+2)
          {
            if (postremark) remark(std::string("Matrix '"+mlarray.getname()+"' cannot be translated into a SCIRun Field, the derivative matrix should have two more dimensions then the field matrix"));
            return (0);            
          }

          if ((derivativesdims[0] != 7)||(derivativesdims[1] != datasize)||(derivativesdims[2] != fielddims[0])||(derivativesdims[3] != fielddims[1])||(derivativesdims[4] != fielddims[2]))
          {
            if (postremark) remark(std::string("Matrix '"+mlarray.getname()+"' cannot be translated into a SCIRun Field, the derivative matrix and the field matrix do not match"));
            return (0);         
          }        
        }        
      }
      else
      {
        if (postremark) remark(std::string("Matrix '"+mlarray.getname()+"' cannot be translated into a SCIRun Field, the derivative matrix and the field matrix do not match"));
        return (0);
      }
    }    

    return(1+ret);              
  }

 ///////////////////////////////////////////////////////////////////////////////
 // CHECK FOR UNSTRUCTURED MESHES:

  // THIS ONE IS NOW KNOWN
  fdatatype = "vector";

  if (mlnode.isempty()) 
  {
    if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (no node matrix for unstructured mesh, create a .node field)"));
    return(0); // a node matrix is always required
  }
        
  if (mlnode.getnumdims() > 2) 
  {
    if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (invalid number of dimensions for node matrix)"));
    return(0); // Currently N dimensional arrays are not supported here
  }
    
  // Check the dimensions of the NODE array supplied only [3xM] or [Mx3] are supported
  int m,n;
  m = mlnode.getm();
  n = mlnode.getn();
        
  if ((n==0)||(m==0)) 
  {
    if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (node matrix is empty)"));
    return(0); //empty matrix, no nodes => no mesh => no field......
  }

  if ((n != 3)&&(m != 3)) 
  {
    if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (one of the dimensions of node matrix needs to be 3)"));
    return(0); // SCIRun is ONLY 3D data, no 2D, or 1D
  }

  numnodes = n; if ((m!=3)&&(n==3)) { numnodes = m; mlnode.transpose(); }
  numelements = 0;

  //////////////
  // IT IS GOOD TO HAVE THE NUMBER OF ELEMENTS IN THE FIELD

  numfield = 0;  
  datasize = 1;
  
  if (mlfield.isdense())
  {
    if (fieldtype == "Vector")
    {
      n = mlfield.getn(); m = mlfield.getm();
      if ((m != 3)&&(n != 3))
      {
        if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (field matrix with vectors does not have a dimension of 3)"));
        return (0);      
      }
      
      numfield = n;
      if (m!=3) { numfield = m; mlfield.transpose(); }
      datasize = 3;
    }
    else if (fieldtype == "Tensor")
    {
      n = mlfield.getn(); m = mlfield.getm();
      if (((m != 6)&&(m !=9))&&((n != 6)&&(n != 9)))
      {
        if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (field matrix with tensors does not have a dimension of 6 or 9)"));
        return (0);      
      }
      
      numfield = n;
      datasize = m;
      if ((m!=6)&&(m!=9)) { numfield = m; mlfield.transpose(); datasize = n; }      
    }
    else if (fieldtype == "")
    {
      n = mlfield.getn(); m = mlfield.getm();
      if (((m != 1)&&(n != 1))&&((m != 3)&&(n != 3))&&((m != 6)&&(n != 6))&&((m != 9)&&(n != 9)))
      {
        if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (field matrix does not have a dimension of 1, 3, 6, or 9)"));
        return (0);      
      }
        
      numfield = n;
      if (m == 1) fieldtype = "double";
      if (m == 3) fieldtype = "Vector";
      if ((m == 6)||(m == 9)) fieldtype = "Tensor";
      datasize = m;
      
      if ((m!=1)&&(m!=3)&&(m!=6)&&(m!=9)) 
      {
        numfield = m; 
        if (n == 1) fieldtype = "double";
        if (n == 3) fieldtype = "Vector";
        if ((n == 6)||(n == 9)) fieldtype = "Tensor";          
        datasize = n;
        mlfield.transpose();
      }
    }
    else
    {
      n = mlfield.getn(); m = mlfield.getm();
      if (((m != 1)&&(n != 1)))
      {
        if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (field matrix does not have a dimension of 1)"));
        return (0);      
      }
        
      numfield = n;      
      if (m!=1) { numfield = m; mlfield.transpose(); }
      datasize = 1;
    }
  }

  // FIRST COUPLE OF CHECKS ON FIELDDATA
  if (fieldbasistype == "nodata")
  {
    if (numfield != 0)
    {
      if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (nodata field basis should not have field data)"));
      return (0);      
    }
    fieldbasis = "NoDataBasis";
  }
  else if ((fieldbasistype == "linear")||(fieldbasistype == "cubic"))
  {
    if (meshbasistype != "quadratic")
    { 
      if (numfield != numnodes)
      {
        if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (number of field entries does not match number of nodes)"));
        return (0);          
      }
    }
  }
  
  if (fieldbasistype == "")
  {
    if (numfield == 0)
    {
      fieldbasis = "NoDataBasis";
      fieldbasistype = "nodata";
    }
  }
  
  if (meshbasistype == "nodata")
  {
    if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (mesh needs to have points and cannot be nodata)"));
    return (0);            
  }
  
  //// POINTCLOUD CODE ////////////
                      
  if ((mledge.isempty())&&(mlface.isempty())&&(mlcell.isempty()))
  {
    // This has no connectivity data => it must be a pointcloud ;)
    // Supported mesh/field types here:
    // PointCloudField
    
    if (meshtype == "") meshtype = "PointCloudMesh";
    
    if (meshtype != "PointCloudMesh")
    {   
      // explicitly stated type (check whether type confirms the guessed type, otherwise someone supplied us with improper data)
      if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (data has to be of the pointcloud class)"));
      return (0);
    }
    
    if (meshbasistype == "")
    {
      meshbasistype = "constant";
    }
    
    // Now pointcloud does store data at nodes as constant
    if (meshbasistype != "constant")
    {
      if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (element is a point, hence no linear/higher order interpolation is supported)"));
      return (0);      
    }
  
    meshbasis = "ConstantBasis";
    
    if (fieldbasistype == "")
    {
      if (numfield == numnodes)
      {
        fieldbasistype = "constant";
      }
      else if (numfield == 0)
      {
        fieldbasistype = "nodata";
      } 
    }
  
    if ((fieldbasistype != "nodata")&&(fieldbasistype != "constant"))
    {
      if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (element is a point, hence no linear/higher order interpolation is supported)"));
      return (0);
    }
    
    numelements = numnodes;    

    if (fieldbasistype == "nodata")    fieldbasis = "NoDataBasis";
    if (fieldbasistype == "constant")
    {
      fieldbasis = "ConstantBasis";
      if (numfield != numelements)
      {
        if (datasize == numelements)
        {
          if (numfield == 1) { fieldtype = "double"; numfield = datasize; datasize = 1; mlfield.transpose(); }
          else if (numfield == 3) { fieldtype = "Vector"; numfield = datasize; datasize = 3; mlfield.transpose(); }
          else if (numfield == 6) { fieldtype = "Tensor"; numfield = datasize; datasize = 6; mlfield.transpose(); }
          else if (numfield == 9) { fieldtype = "Tensor"; numfield = datasize; datasize = 9; mlfield.transpose(); }
        }
        if (numfield != numelements)
        {
          if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (number of elements does not match number of field entries)"));
          return (0);   
        }
      }
    }
    
    if ((mlmeshderivatives.isdense())||(mlfieldderivatives.isdense())||
      (mlfieldedge.isdense())||(mlfieldface.isdense())||(mlfieldcell.isdense()))
    {
      if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (element is a point, hence no linear/higher order interpolation is supported)"));
      return (0);      
    }
    
    return(1+ret);
  }

  if (meshbasistype == "constant")
  {
    if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (spatial distributed elements cannot have a constant mesh interpolation)"));
    return (0);            
  }

  ///// CURVEMESH ////////////////////////////////////

  if (mledge.isdense())
  {
    int n,m;
    // Edge data is provide hence it must be some line element!
    // Supported mesh/field types here:
    //  CurveField
    
    if (meshtype == "") meshtype = "CurveMesh";
    
    if (meshtype != "CurveMesh")
    {   // explicitly stated type 
      if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (edge connectivity does not macth meshtype)"));
      return(0);
    }

    // established meshtype //

    if ((mlface.isdense())||(mlcell.isdense())||
        (mlfieldface.isdense())||(mlfieldcell.isdense()))
    {   // a matrix with multiple connectivities is not yet allowed
      if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (multiple connectivity matrices defined)"));
      return(0);
    }

    // Connectivity should be 2D
    if ((mledge.getnumdims() > 2)||(mlfieldedge.getnumdims() > 2))
    {
      if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (edge connectivity matrix should be 2D)"));
      return(0);                
    } 
                              
    // Check whether the connectivity data makes any sense
    // from here on meshtype can only be  linear/cubic/quadratic

    if ((meshbasistype == "linear")||(meshbasistype == "cubic"))
    {
      m = mledge.getm(); n = mledge.getn();
      if ((n!=2)&&(m!=2))
      {
        if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (one of the dimensions of edge needs to be of size 2)"));
        return(0);                
      }     
      numelements = n; 
      if ((m!=2)&&(n==2)) { numelements = m; mledge.transpose(); }   
      if (meshbasistype == "linear") meshbasis = "CrvLinearLgn"; else meshbasis = "CrvCubicHmt";
    }
    else if (meshbasistype == "quadratic")
    {
      m = mledge.getm(); n = mledge.getn();
      if ((n!=3)&&(m!=3))
      {
        if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (one of the dimensions of edge needs to be of size 3)"));
        return(0);                
      }     
      numelements = n; 
      if ((m!=3)&&(n==3)) { numelements = m; mledge.transpose(); }     
      meshbasistype = "CrvQuadraticLgn";
    }
    else
    {
      m = mledge.getm(); n = mledge.getn();
      if (((n!=2)&&(m!=2))&&((n!=3)&&(m!=3)))
      {
        if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (one of the dimensions of edge needs to be of size 2 or 3)"));
        return(0);                
      }     
      numelements = n; 
      if (((m!=2)&&(m!=3))&&((n==2)||(n==3))) { numelements = m; m = n; mledge.transpose(); }     
      if (m == 2) { meshbasistype = "linear"; meshbasis = "CrvLinearLgn"; }
      if (m == 3) { meshbasistype = "quadratic"; meshbasis = "CrvQuadraticLgn"; }
    }
    
    // established meshbasis
    
    if ((mlmeshderivatives.isempty())&&(meshbasistype == "cubic"))
    {
      if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (no meshderatives matrix was found)"));
      return(0);               
    }

    if (meshbasistype == "cubic")
    {
      std::vector<int> ddims = mlmeshderivatives.getdims();
      if (ddims.size() != 4)
      {
        if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (meshderatives matrix has not proper dimensions)"));
        return(0);               
      }
    
      if ((ddims[0] != 1)&&(ddims[1] != 3)&&(ddims[2] != numelements)&&(ddims[3] != 2))
      {
        if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (meshderatives matrix has not proper dimensions)"));
        return(0);         
      }
    }
    
    // established and checked mesh type ///
    // CHECK THE FIELD PROPERTIES

    if (fieldbasistype == "")
    {
      if ((numfield == numelements)&&(numfield != numnodes))
      {
        fieldbasistype = "constant";
      }
      else if (numfield == numnodes)
      {
        if (meshbasistype == "quadratic")
        {
          fieldbasistype = "quadratic";
        }
        else if ((meshbasistype == "cubic")&&(mlfieldderivatives.isdense()))
        {
          fieldbasistype = "cubic";
        }
        else
        {
          fieldbasistype = "linear";
        }
      }
      else
      {
        if ((meshbasistype == "quadratic")&&(mlfieldedge.isdense()))
        {
          fieldbasistype = "linear";
        }
        else
        {
          if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (number of elements in field does not match mesh)"));
          return(0);         
        }
      }
    }

    if (fieldbasistype == "constant") 
    {
      if (numfield != numelements)
      {
        if (datasize == numelements)
        {
          if (numfield == 1) { fieldtype = "double"; numfield = datasize; datasize = 1; mlfield.transpose(); }
          else if (numfield == 3) { fieldtype = "Vector"; numfield = datasize; datasize = 3; mlfield.transpose(); }
          else if (numfield == 6) { fieldtype = "Tensor"; numfield = datasize; datasize = 6; mlfield.transpose(); }
          else if (numfield == 9) { fieldtype = "Tensor"; numfield = datasize; datasize = 9; mlfield.transpose(); }
        }
        if (numfield != numelements)
        {
          if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (number of elements does not match number of field entries)"));
          return (0);   
        }
            }
      fieldbasis = "ConstantBasis";
    }
    
    if ((fieldbasistype == "linear")||(fieldbasistype == "cubic"))
    {
      if ((meshbasistype == "quadratic")&&(mlfieldedge.isempty()))
      {
        if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (no fieldedge connectivity matrix)"));
        return(0);               
      }
      
      if (fieldbasistype == "linear") 
      {
        fieldbasis = "CrvLinearLgn";
      }
      else 
      {
        if (mlfieldderivatives.isempty())
        {
          if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (no fieldderivatives matrix)"));
          return(0);                         
        }
        fieldbasis = "CrvCubicHmt";
      }
    }
    
    if (fieldbasis == "quadratic")
    {
      if (((meshbasistype == "linear")||(meshbasistype == "cubic"))&&(mlfieldedge.isempty()))
      {
        if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (no fieldedge connectivity matrix)"));
        return(0);                               
      }
      fieldbasis = "CrvQuadraticLgn";
    }
  
    // established fieldbasis //
  
    if (mlfieldedge.isdense())
    {
      m = mlfieldedge.getm(); n = mlfieldedge.getn();    
      if (fieldbasistype == "quadratic")
      {
        if (!(((m==3)&&(n==numelements))||((m==numelements)&&(n==3))))
        {
          if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (one of the dimensions of fieldedge needs to be of size 2 or 3)"));
          return(0);                        
        }
        if (m!=3) mlfieldedge.transpose();
      }
      else
      {
        if (!(((m==2)&&(n==numelements))||((m==numelements)&&(n==2))))
        {
          if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (one of the dimensions of fieldedge needs to be of size 2 or 3)"));
          return(0);                                
        }      
        if (m!=2) mlfieldedge.transpose();
      }
    }

    if (mlfieldderivatives.isdense())
    {
      std::vector<int> ddims = mlfieldderivatives.getdims();
      if (ddims.size() != 4)
      {
        if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (fieldderatives matrix has not proper dimensions)"));
        return(0);               
      }

      if ((ddims[0] != 1)&&(ddims[1] != datasize)&&(ddims[3] != numelements)&&(ddims[2] != 2))
      {
        if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (fieldderatives matrix has not proper dimensions)"));
        return(0);         
      }
    }

    return(1+ret);
  }

  if (mlface.isdense())
  {
    int n,m;
      
    // Edge data is provide hence it must be some line element!
    // Supported mesh/field types here:
    //  CurveField
    
    if (meshtype == "")
    {
      m = mlface.getm();
      n = mlface.getn();
      
      if ((m==3)||(m==4)||(m==6)||(m==8)) n = m;
      if ((n==3)||(n==4)||(n==6)||(n==8))
      {
        if ((n==3)||(n==6)) meshtype = "TriSurfMesh";
        else meshtype = "QuadSurfMesh";            
      }
    }

    m = mlface.getm();
    n = mlface.getn();
    
    if ((meshtype != "TriSurfMesh")&&(meshtype != "QuadSurfMesh"))
    {   // explicitly stated type 
      if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (face connectivity does not match meshtype)"));
      return(0);
    }

    // established meshtype //

    if ((mledge.isdense())||(mlcell.isdense())||
        (mlfieldedge.isdense())||(mlfieldcell.isdense()))
    {   // a matrix with multiple connectivities is not yet allowed
      if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (multiple connectivity matrices defined)"));
      return(0);
    }

    // Connectivity should be 2D
    if ((mlface.getnumdims() > 2)||(mlfieldface.getnumdims() > 2))
    {
      if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (face connectivity matrix should be 2D)"));
      return(0);                
    } 
                              
    // Check whether the connectivity data makes any sense
    // from here on meshtype can only be  linear/cubic/quadratic

    if (meshtype == "TriSurfMesh")
    {
      if ((meshbasistype == "linear")||(meshbasistype == "cubic"))
      {
        m = mlface.getm(); n = mlface.getn();
        if ((n!=3)&&(m!=3))
        {
          if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (one of the dimensions of face needs to be of size 3)"));
          return(0);                
        }     
        numelements = n; 
        if ((m!=3)&&(n==3)) { numelements = m; mlface.transpose(); }  
        if (meshbasistype == "linear") meshbasis = "TriLinearLgn"; else meshbasis = "TriCubicHmt";
      }
      else if (meshbasistype == "quadratic")
      {
        m = mlface.getm(); n = mlface.getn();
        if ((n!=6)&&(m!=6))
        {
          if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (one of the dimensions of face needs to be of size 6)"));
          return(0);                
        }     
        numelements = n; 
        if ((m!=6)&&(n==6)) { numelements = m; mlface.transpose(); }
        meshbasistype = "TriQuadraticLgn";
      }
      else
      {
        m = mlface.getm(); n = mlface.getn();
        if (((n!=3)&&(m!=3))&&((n!=6)&&(m!=6)))
        {
          if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (one of the dimensions of face needs to be of size 3 or 6)"));
          return(0);                
        }     
        numelements = n; 
        if (((m!=3)&&(m!=6))&&((n==3)||(n==6))) { numelements = m; m = n; mlface.transpose(); }     
        if (m == 3) { meshbasistype = "linear"; meshbasis = "TriLinearLgn"; }
        if (m == 6) { meshbasistype = "quadratic"; meshbasis = "TriQuadraticLgn"; }
      }
    }
    else
    {
      if ((meshbasistype == "linear")||(meshbasistype == "cubic"))
      {
        m = mlface.getm(); n = mlface.getn();
        if ((n!=4)&&(m!=4))
        {
          if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (one of the dimensions of face needs to be of size 4)"));
          return(0);                
        }     
        numelements = n; 
        if ((m!=4)&&(n==4)) { numelements = m; mlface.transpose(); }
        if (meshbasistype == "linear") meshbasis = "QuadBilinearLgn"; else meshbasis = "QuadBicubicHmt";
      }
      else if (meshbasistype == "quadratic")
      {
        m = mlface.getm(); n = mlface.getn();
        if ((n!=8)&&(m!=8))
        {
          if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (one of the dimensions of face needs to be of size 8)"));
          return(0);                
        }     
        numelements = n; 
        if ((m!=8)&&(n==8)) { numelements = m; mlface.transpose(); }
        meshbasistype = "QuadBiquadraticLgn";
      }
      else
      {
        m = mlface.getm(); n = mlface.getn();
        if (((n!=4)&&(m!=4))&&((n!=8)&&(m!=8)))
        {
          if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (one of the dimensions of face needs to be of size 4 or 8)"));
          return(0);                
        }     
        numelements = n; 
        if (((m!=4)&&(m!=8))&&((n==4)||(n==8))) { numelements = m; m = n; mlface.transpose(); }     
        if (m == 4) { meshbasistype = "linear"; meshbasis = "QuadBilinearLgn"; }
        if (m == 8) { meshbasistype = "quadratic"; meshbasis = "QuadBiquadraticLgn"; }
      }    
    }
    
    // established meshbasis
    
    if ((mlmeshderivatives.isempty())&&(meshbasistype == "cubic"))
    {
      if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (no meshderatives matrix was found)"));
      return(0);               
    }

    if (meshbasistype == "cubic")
    {
      std::vector<int> ddims = mlmeshderivatives.getdims();
      if (ddims.size() != 4)
      {
        if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (meshderatives matrix has not proper dimensions)"));
        return(0);               
      }

      if (meshtype == "TriSurfMesh")
      {    
        if ((ddims[0] != 2)&&(ddims[1] != 3)&&(ddims[2] != numelements)&&(ddims[3] != 3))
        {
          if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (meshderatives matrix has not proper dimensions)"));
          return(0);         
        }
      }
      else
      {
        if ((ddims[0] != 2)&&(ddims[1] != 3)&&(ddims[2] != numelements)&&(ddims[3] != 4))
        {
          if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (meshderatives matrix has not proper dimensions)"));
          return(0);         
        }      
      }
    }

    // established and checked mesh type ///
    // CHECK THE FIELD PROPERTIES

    if (fieldbasistype == "")
    {
      if ((numfield == numelements)&&(numfield != numnodes))
      {
        fieldbasistype = "constant";
      }
      else if (numfield == numnodes)
      {
        if (meshbasistype == "quadratic")
        {
          fieldbasistype = "quadratic";
        }
        else if ((meshbasistype == "cubic")&&(mlfieldderivatives.isdense()))
        {
          fieldbasistype = "cubic";
        }
        else
        {
          fieldbasistype = "linear";
        }
      }
      else
      {
        if ((meshbasistype == "quadratic")&&(mlfieldface.isdense()))
        {
          fieldbasistype = "linear";
        }
        else
        {
          if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (number of elements in field does not match mesh)"));
          return(0);         
        }
      }
    }

    if (fieldbasistype == "constant") 
    {
      if (numfield != numelements)
      {
        if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (number of elements in field does not match mesh)"));
        return(0);                 
      }
      fieldbasis = "ConstantBasis";
    }
    
    if ((fieldbasistype == "linear")||(fieldbasistype == "cubic"))
    {
      if ((meshbasistype == "quadratic")&&(mlfieldedge.isempty()))
      {
        if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (no fieldedge connectivity matrix)"));
        return(0);               
      }
      
      if (fieldbasistype == "linear") 
      {
        if (meshtype == "TriSurfMesh") fieldbasis = "TriLinearLgn"; else fieldbasis = "QuadBilinearLgn";
      }
      else 
      {
        if (mlfieldderivatives.isempty())
        {
          if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (no fieldderivatives matrix)"));
          return(0);                         
        }
        if (meshtype == "TriSurfMesh") fieldbasis = "TriCubicHmt"; else fieldbasis = "QuadBicubicHmt";
      }
    }
    
    if (fieldbasis == "quadratic")
    {
      if (((meshbasistype == "linear")||(meshbasistype == "cubic"))&&(mlfieldface.isempty()))
      {
        if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (no fieldedge connectivity matrix)"));
        return(0);                               
      }
      if (meshtype == "TriSurfMesh") fieldbasis = "TriQuadraticLgn"; else fieldbasis = "QuadBiquadraticLgn";
    }
  
    // established fieldbasis //
  
    if (meshtype == "TriSurfMesh")
    {
      if (mlfieldface.isdense())
      {
        m = mlfieldface.getm(); n = mlfieldface.getn();    
        if (fieldbasistype == "quadratic")
        {
          if (!(((m==6)&&(n==numelements))||((m==numelements)&&(n==6))))
          {
            if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (one of the dimensions of fieldedge needs to be of size 6)"));
            return(0);                        
          }
          if (m!=6) mlfieldface.transpose();
        }
        else
        {
          if (!(((m==3)&&(n==numelements))||((m==numelements)&&(n==3))))
          {
            if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (one of the dimensions of fieldedge needs to be of size 3)"));
            return(0);                                
          }      
          if (m!=3) mlfieldface.transpose();
        }
      }
    }
    else
    {
      if (mlfieldface.isdense())
      {
        m = mlfieldface.getm(); n = mlfieldface.getn();    
        if (fieldbasistype == "quadratic")
        {
          if (!(((m==8)&&(n==numelements))||((m==numelements)&&(n==8))))
          {
            if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (one of the dimensions of fieldface needs to be of size 8)"));
            return(0);                        
          }
          if (m!=8) mlfieldface.transpose();
        }
        else
        {
          if (!(((m==4)&&(n==numelements))||((m==numelements)&&(n==4))))
          {
            if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (one of the dimensions of fieldface needs to be of size 4)"));
            return(0);                                
          }      
          if (m!=4) mlfieldface.transpose();
        }
      }    
    }
    
    if ((mlfieldderivatives.isdense())&&(fieldbasistype == "cubic"))
    {
      std::vector<int> ddims = mlfieldderivatives.getdims();
      if (ddims.size() != 4)
      {
        if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (fieldderatives matrix has not proper dimensions)"));
        return(0);               
      }

      if (meshtype == "TriSurfMesh")
      {    
        if ((ddims[0] != 2)&&(ddims[1] != datasize)&&(ddims[3] != numelements)&&(ddims[2] != 3))
        {
          if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (fieldderatives matrix has not proper dimensions)"));
          return(0);         
        }
      }
      else
      {
        if ((ddims[0] != 2)&&(ddims[1] != datasize)&&(ddims[3] != numelements)&&(ddims[2] != 4))
        {
          if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (fieldderatives matrix has not proper dimensions)"));
          return(0);         
        }      
      }
    }

    return(1+ret);
  }


  if (mlcell.isdense())
  {
    int n,m;
    
    // Edge data is provide hence it must be some line element!
    // Supported mesh/field types here:
    //  CurveField
    
    if (meshtype == "")
    {
      m = mlcell.getm();
      n = mlcell.getn();
      
      if ((m==4)||(m==6)||(m==8)||(m==10)||(m==15)||(m==20)) n = m;
      if ((n==4)||(n==6)||(n==8)||(n==10)||(n==15)||(n==20))
      {
        if ((n==4)||(n==10)) meshtype = "TetVolMesh";
        else if ((n==6)||(n==15)) meshtype = "PrismVolMesh";
        else meshtype = "HexVolMesh";            
      }
    }
  
    m = mlcell.getm();
    n = mlcell.getn();    
        
    if ((meshtype != "TetVolMesh")&&(meshtype != "PrismVolMesh")&&(meshtype != "HexVolMesh"))
    {   // explicitly stated type 
      if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (cell connectivity does not match meshtype)"));
      return(0);
    }

    // established meshtype //

    if ((mledge.isdense())||(mlface.isdense())||
        (mlfieldedge.isdense())||(mlfieldface.isdense()))
    {   // a matrix with multiple connectivities is not yet allowed
      if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (multiple connectivity matrices defined)"));
      return(0);
    }

    // Connectivity should be 2D
    if ((mlcell.getnumdims() > 2)||(mlfieldcell.getnumdims() > 2))
    {
      if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (cell connectivity matrix should be 2D)"));
      return(0);                
    } 
                              
    // Check whether the connectivity data makes any sense
    // from here on meshtype can only be  linear/cubic/quadratic

    if (meshtype == "TetVolMesh")
    {
      if ((meshbasistype == "linear")||(meshbasistype == "cubic"))
      {
        m = mlcell.getm(); n = mlcell.getn();
        if ((n!=4)&&(m!=4))
        {
          if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (one of the dimensions of cell needs to be of size 4)"));
          return(0);                
        }     
        numelements = n; 
        if ((m!=4)&&(n==4)) { numelements = m; mlcell.transpose(); }  
        if (meshbasistype == "linear") meshbasis = "TetLinearLgn"; else meshbasis = "TetCubicHmt";
      }
      else if (meshbasistype == "quadratic")
      {
        m = mlcell.getm(); n = mlcell.getn();
        if ((n!=10)&&(m!=10))
        {
          if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (one of the dimensions of cell needs to be of size 10)"));
          return(0);                
        }     
        numelements = n; 
        if ((m!=10)&&(n==10)) { numelements = m; mlcell.transpose(); }
        meshbasistype = "TetQuadraticLgn";
      }
      else
      {
        m = mlcell.getm(); n = mlcell.getn();
        if (((n!=4)&&(m!=4))&&((n!=10)&&(m!=10)))
        {
          if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (one of the dimensions of edge needs to be of size 4 or 10)"));
          return(0);                
        }     
        numelements = n; 
        if (((m!=4)&&(m!=10))&&((n==4)||(n==10))) { numelements = m; m = n; mlcell.transpose(); }
        if (m == 4) { meshbasistype = "linear"; meshbasis = "TetLinearLgn"; }
        if (m == 10) { meshbasistype = "quadratic"; meshbasis = "TetQuadraticLgn"; }
      }
    }
    else if (meshtype == "PrismVolMesh")
    {
      if ((meshbasistype == "linear")||(meshbasistype == "cubic"))
      {
        m = mlcell.getm(); n = mlcell.getn();
        if ((n!=6)&&(m!=6))
        {
          if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (one of the dimensions of cell needs to be of size 6)"));
          return(0);                
        }     
        numelements = n; 
        if ((m!=6)&&(n==6)) { numelements = m; mlcell.transpose(); }
        if (meshbasistype == "linear") meshbasis = "TetLinearLgn"; else meshbasis = "PrismCubicHmt";
      }
      else if (meshbasistype == "quadratic")
      {
        m = mlcell.getm(); n = mlcell.getn();
        if ((n!=15)&&(m!=15))
        {
          if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (one of the dimensions of cell needs to be of size 15)"));
          return(0);                
        }     
        numelements = n; 
        if ((m!=15)&&(n==15)) { numelements = m; mlcell.transpose(); }
        meshbasistype = "PrismQuadraticLgn";
      }
      else
      {
        m = mlcell.getm(); n = mlcell.getn();
        if (((n!=6)&&(m!=6))&&((n!=15)&&(m!=15)))
        {
          if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (one of the dimensions of cell needs to be of size 6 or 15)"));
          return(0);                
        }     
        numelements = n; 
        if (((m!=6)&&(m!=15))&&((n==6)||(n==15))) { numelements = m; m = n; mlcell.transpose(); }
        if (m == 6) { meshbasistype = "linear"; meshbasis = "PrismLinearLgn"; }
        if (m == 15) { meshbasistype = "quadratic"; meshbasis = "PrismQuadraticLgn"; }
      }    
    }
    else
    {
      if ((meshbasistype == "linear")||(meshbasistype == "cubic"))
      {
        m = mlcell.getm(); n = mlcell.getn();
        if ((n!=8)&&(m!=8))
        {
          if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (one of the dimensions of cell needs to be of size 8)"));
          return(0);                
        }     
        numelements = n; 
        if ((m!=8)&&(n==8)) { numelements = m; mlcell.transpose(); }
        if (meshbasistype == "linear") meshbasis = "HexTrilinearLgn"; else meshbasis = "HexTricubicHmt";
      }
      else if (meshbasistype == "quadratic")
      {
        m = mlcell.getm(); n = mlcell.getn();
        if ((n!=20)&&(m!=20))
        {
          if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (one of the dimensions of cell needs to be of size 20)"));
          return(0);                
        }     
        numelements = n; 
        if ((m!=20)&&(n==20)) { numelements = m; mlcell.transpose(); }
        meshbasistype = "HexTriquadraticLgn";
      }
      else
      {
        m = mlcell.getm(); n = mlcell.getn();
        if (((n!=8)&&(m!=8))&&((n!=20)&&(m!=20)))
        {
          if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (one of the dimensions of cell needs to be of size 8 or 20)"));
          return(0);                
        }     
        numelements = n; 
        if (((m!=8)&&(m!=20))&&((n==8)||(n==20))) { numelements = m; m = n; mlcell.transpose(); }
        if (m == 8) { meshbasistype = "linear"; meshbasis = "HexTrilinearLgn"; }
        if (m == 20) { meshbasistype = "quadratic"; meshbasis = "HexTriquadraticLgn"; }
      }    
    }
    
    // established meshbasis
    
    if ((mlmeshderivatives.isempty())&&(meshbasistype == "cubic"))
    {
      if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (no meshderatives matrix was found)"));
      return(0);               
    }

    if (meshbasistype == "cubic")
    {
      std::vector<int> ddims = mlmeshderivatives.getdims();
      if (ddims.size() != 4)
      {
        if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (meshderatives matrix has not proper dimensions)"));
        return(0);               
      }

      if (meshtype == "TetVolMesh")
      {    
        if ((ddims[0] != 3)&&(ddims[1] != 3)&&(ddims[2] != numelements)&&(ddims[3] != 4))
        {
          if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (meshderatives matrix has not proper dimensions)"));
          return(0);         
        }
      }
      else if (meshtype == "PrismVolMesh")
      {
        if ((ddims[0] != 3)&&(ddims[1] != 3)&&(ddims[2] != numelements)&&(ddims[3] != 6))
        {
          if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (meshderatives matrix has not proper dimensions)"));
          return(0);         
        }      
      }
      else
      {
        if ((ddims[0] != 7)&&(ddims[1] != 3)&&(ddims[2] != numelements)&&(ddims[3] != 8))
        {
          if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (meshderatives matrix has not proper dimensions)"));
          return(0);         
        }      
      }
    }

    // established and checked mesh type ///
    // CHECK THE FIELD PROPERTIES

    if (fieldbasistype == "")
    {
      if ((numfield == numelements)&&(numfield != numnodes))
      {
        fieldbasistype = "constant";
      }
      else if (numfield == numnodes)
      {
        if (meshbasistype == "quadratic")
        {
          fieldbasistype = "quadratic";
        }
        else if ((meshbasistype == "cubic")&&(mlfieldderivatives.isdense()))
        {
          fieldbasistype = "cubic";
        }
        else
        {
          fieldbasistype = "linear";
        }
      }
      else
      {
        if ((meshbasistype == "quadratic")&&(mlfieldface.isdense()))
        {
          fieldbasistype = "linear";
        }
        else
        {
          if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (number of elements in field does not match mesh)"));
          return(0);         
        }
      }
    }

    if (fieldbasistype == "constant") 
    {
      if (numfield != numelements)
      {
        if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (number of elements in field does not match mesh)"));
        return(0);                 
      }
      fieldbasis = "ConstantBasis";
    }
    
    if ((fieldbasistype == "linear")||(fieldbasistype == "cubic"))
    {
      if ((meshbasistype == "quadratic")&&(mlfieldedge.isempty()))
      {
        if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (no fieldedge connectivity matrix)"));
        return(0);               
      }
      
      if (fieldbasistype == "linear") 
      {
        if (meshtype == "TetVolMesh") fieldbasis = "TetLinearLgn"; 
        else if (meshtype == "PrismVolMesh") fieldbasis = "PrismLinearLgn";
        else fieldbasis = "HexTrilinearLgn";
      }
      else 
      {
        if (mlfieldderivatives.isempty())
        {
          if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (no fieldderivatives matrix)"));
          return(0);                         
        }
        if (meshtype == "TetVolMesh") fieldbasis = "TetCubicHmt";
        else if (meshtype == "PrismVolMesh") fieldbasis = "PrismCubicHmt";
        else fieldbasis = "HexTricubicHmt";
      }
    }
    
    if (fieldbasis == "quadratic")
    {
      if (((meshbasistype == "linear")||(meshbasistype == "cubic"))&&(mlfieldface.isempty()))
      {
        if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (no fieldedge connectivity matrix)"));
        return(0);                               
      }
      if (meshtype == "TetVolMesh") fieldbasis = "TetQuadraticLgn"; 
      else if (meshtype == "PrismVolMesh") fieldbasis = "PrismQuadraticLgn";
      else fieldbasis = "HexTriquadraticLgn";
    }
  
    // established fieldbasis //
  
    if (meshtype == "TetVolMesh")
    {
      if (mlfieldcell.isdense())
      {
        m = mlfieldcell.getm(); n = mlfieldcell.getn();    
        if (fieldbasistype == "quadratic")
        {
          if (!(((m==10)&&(n==numelements))||((m==numelements)&&(n==10))))
          {
            if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (one of the dimensions of fieldedge needs to be of size 10)"));
            return(0);                        
          }
          if (m!=10) mlfieldcell.transpose();
        }
        else
        {
          if (!(((m==4)&&(n==numelements))||((m==numelements)&&(n==4))))
          {
            if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (one of the dimensions of fieldedge needs to be of size 4)"));
            return(0);                                
          }      
          if (m!=4) mlfieldcell.transpose();
        }
      }
    }
    else if (meshtype == "PrismVolMesh")
    {
      if (mlfieldcell.isdense())
      {
        m = mlfieldcell.getm(); n = mlfieldcell.getn();    
        if (fieldbasistype == "quadratic")
        {
          if (!(((m==15)&&(n==numelements))||((m==numelements)&&(n==15))))
          {
            if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (one of the dimensions of fieldface needs to be of size 15)"));
            return(0);                        
          }
          if (m!=15) mlfieldcell.transpose();
        }
        else
        {
          if (!(((m==6)&&(n==numelements))||((m==numelements)&&(n==6))))
          {
            if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (one of the dimensions of fieldface needs to be of size 6)"));
            return(0);                                
          }      
          if (m!=6) mlfieldcell.transpose();
        }
      }        
    }
    else
    {
      if (mlfieldcell.isdense())
      {
        m = mlfieldcell.getm(); n = mlfieldcell.getn();    
        if (fieldbasistype == "quadratic")
        {
          if (!(((m==20)&&(n==numelements))||((m==numelements)&&(n==20))))
          {
            if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (one of the dimensions of fieldface needs to be of size 20)"));
            return(0);                        
          }
          if (m!=20) mlfieldcell.transpose();
        }
        else
        {
          if (!(((m==8)&&(n==numelements))||((m==numelements)&&(n==8))))
          {
            if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (one of the dimensions of fieldface needs to be of size 8)"));
            return(0);                                
          }      
          if (m!=8) mlfieldcell.transpose();
        }
      }    
    }
    
    if ((mlfieldderivatives.isdense())&&(fieldbasistype == "cubic"))
    {
      std::vector<int> ddims = mlfieldderivatives.getdims();
      if (ddims.size() != 4)
      {
        if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (fieldderatives matrix has not proper dimensions)"));
        return(0);               
      }

      if (meshtype == "TetVolMesh")
      {    
        if ((ddims[0] != 3)&&(ddims[1] != datasize)&&(ddims[3] != numelements)&&(ddims[2] != 4))
        {
          if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (fieldderatives matrix has not proper dimensions)"));
          return(0);         
        }
      }
      else if (meshtype == "PrismVolMesh")
      {    
        if ((ddims[0] != 3)&&(ddims[1] != datasize)&&(ddims[3] != numelements)&&(ddims[2] != 6))
        {
          if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (fieldderatives matrix has not proper dimensions)"));
          return(0);         
        }
      }
      else
      {
        if ((ddims[0] != 7)&&(ddims[1] != datasize)&&(ddims[3] != numelements)&&(ddims[2] != 8))
        {
          if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (fieldderatives matrix has not proper dimensions)"));
          return(0);         
        }      
      }
    }
    return(1+ret);
  }

  if (postremark) remark(std::string("Matrix '" + mlarray.getname() + "' cannot be translated into a SCIRun Field (cannot match the matlab structure with any of the supported mesh classes)"));
  return(0);
}        


bool MatlabToFieldAlgo::addfield(std::vector<SCIRun::Vector> &fdata)
{
  std::vector<double> fielddata;
  mlfield.getnumericarray(fielddata); // cast and copy the real part of the data
        
  unsigned int numdata = fielddata.size();
  if (numdata > (3*fdata.size())) numdata = (3*fdata.size()); // make sure we do not copy more data than there are elements
        
  unsigned int p,q;
  for (p=0,q=0; p < numdata; q++) 
  { 
    fdata[q][0] = fielddata[p++];
    fdata[q][1] = fielddata[p++];
    fdata[q][2] = fielddata[p++];
  }
  
  return(true);
}


bool MatlabToFieldAlgo::addfield(std::vector<SCIRun::Tensor> &fdata)
{
  std::vector<double> fielddata;
  mlfield.getnumericarray(fielddata); // cast and copy the real part of the data

  unsigned int numdata = fielddata.size();
  SCIRun::Tensor tensor;

  if (mlfield.getm() == 6)
  { // Compressed tensor data : xx,yy,zz,xy,xz,yz
    if (numdata > (6*fdata.size())) numdata = (6*fdata.size()); // make sure we do not copy more data than there are elements
    unsigned int p,q;
    for (p = 0, q = 0; p < numdata; p +=6, q++) { compressedtensor(fielddata,tensor,p); fdata[q] =  tensor; }
  }
  else
  {  // UnCompressed tensor data : xx,xy,xz,yx,yy,yz,zx,zy,zz 
    if (numdata > (9*fdata.size())) numdata = (9*fdata.size()); // make sure we do not copy more data than there are elements
    unsigned int p,q;
    for (p = 0, q = 0; p < numdata; p +=9, q++) { uncompressedtensor(fielddata,tensor,p); fdata[q] =  tensor; }
  }

  return(true);
}


} // end namespace
