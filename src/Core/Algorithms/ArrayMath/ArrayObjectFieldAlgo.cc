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

#include <Core/Algorithms/ArrayMath/ArrayObjectFieldAlgo.h>

namespace SCIRunAlgo {

using namespace SCIRun;

void ArrayObjectFieldDataAlgo::getnextscalar(TensorVectorMath::Scalar& scalar)
{
  scalar = 0.0;
}

void ArrayObjectFieldDataAlgo::getnextvector(TensorVectorMath::Vector& vector)
{
  vector = TensorVectorMath::Vector(0.0,0.0,0.0);
}

void ArrayObjectFieldDataAlgo::getnexttensor(TensorVectorMath::Tensor& tensor)
{
  tensor = TensorVectorMath::Tensor(0.0);
}

void ArrayObjectFieldDataAlgo::setnextscalar(TensorVectorMath::Scalar& scalar)
{
}

void ArrayObjectFieldDataAlgo::setnextvector(TensorVectorMath::Vector& vector)
{
}

void ArrayObjectFieldDataAlgo::setnexttensor(TensorVectorMath::Tensor& tensor)
{
}

void ArrayObjectFieldDataAlgo::reset()
{
}

void ArrayObjectFieldDataAlgo::reset(unsigned int idx)
{
}

int ArrayObjectFieldDataAlgo::size()
{
  return(0);
}

bool ArrayObjectFieldDataAlgo::isscalar()
{
  return(false);
}

bool ArrayObjectFieldDataAlgo::isvector()
{
  return(false);
}

bool ArrayObjectFieldDataAlgo::istensor()
{
  return(false);
}

bool ArrayObjectFieldDataAlgo::setfield(SCIRun::FieldHandle handle)
{
  return(false);
}

SCIRun::CompileInfoHandle 
    ArrayObjectFieldDataAlgo::get_compile_info(SCIRun::FieldHandle& field)
{
  const SCIRun::TypeDescription *fieldtype = field->get_type_description();
  const SCIRun::TypeDescription *locationtype = field->order_type_description();
  const SCIRun::TypeDescription *basistype = field->get_type_description(SCIRun::Field::BASIS_TD_E);
  const SCIRun::TypeDescription::td_vec *basis_subtype = basistype->get_sub_type();
  const SCIRun::TypeDescription *datatype = (*basis_subtype)[0];

  // As I use my own Tensor and Vector algorithms they need to be
  // converted when reading the data, hence separate algorithms are
  // implemented for those cases
  
  std::string algo_type = "Scalar";  
  if (datatype->get_name() == "Vector") algo_type = "Vector";
  if (datatype->get_name() == "Tensor") algo_type = "Tensor";

  std::string algo_name = "ArrayObjectFieldData" + algo_type + "AlgoT";
  std::string algo_base = "ArrayObjectFieldDataAlgo";

  std::string include_path(SCIRun::TypeDescription::cc_to_h(__FILE__));

  SCIRun::CompileInfoHandle ci = 
    scinew SCIRun::CompileInfo("ALGO"+algo_name + "." +
                       fieldtype->get_filename() + "." +
                       locationtype->get_filename() + ".",
                       algo_base, 
                       algo_name, 
                       fieldtype->get_name() + "," + locationtype->get_name());

  ci->add_data_include(include_path);
  ci->add_namespace("SCIRunAlgo");
  fieldtype->fill_compile_info(ci.get_rep());
  return(ci);
}
    
SCIRun::CompileInfoHandle 
    ArrayObjectFieldLocationAlgo::get_compile_info(SCIRun::FieldHandle& field)
{
  const SCIRun::TypeDescription *fieldtype = field->get_type_description();
  const SCIRun::TypeDescription *locationtype = field->order_type_description();
  const SCIRun::TypeDescription *meshtype = field->get_type_description(SCIRun::Field::MESH_TD_E);
  
  // As I use my own Tensor and Vector algorithms they need to be
  // converted when reading the data, hence separate algorithms are
  // implemented for those cases

  std::string mesh = meshtype->get_name();

  std::string algotype ="";
  if (field->basis_order() == 0) algotype = "Elem";

  if (((mesh.find("Scanline") != std::string::npos)||
      (mesh.find("Image") != std::string::npos)||
      (mesh.find("LatVol") != std::string::npos))&&
      (algotype == "")) algotype = "Node";

  std::string algo_name = "ArrayObjectFieldLocation"+algotype+"AlgoT";
  std::string algo_base = "ArrayObjectFieldLocationAlgo";

  std::string include_path(SCIRun::TypeDescription::cc_to_h(__FILE__));

  SCIRun::CompileInfoHandle ci = 
    scinew SCIRun::CompileInfo("ALGO"+algo_name + "." +
                       fieldtype->get_filename() + "." +
                       locationtype->get_filename() + ".",
                       algo_base, 
                       algo_name, 
                       fieldtype->get_name());

  // Add in the include path to compile this obj
  ci->add_include(include_path);
  ci->add_namespace("SCIRunAlgo");
  fieldtype->fill_compile_info(ci.get_rep());
  return(ci);
}

SCIRun::CompileInfoHandle 
    ArrayObjectFieldCreateAlgo::get_compile_info(SCIRun::FieldHandle field,std::string datatype, std::string basistype)
{
  const SCIRun::TypeDescription *basis_type = field->get_type_description(SCIRun::Field::BASIS_TD_E);
  const SCIRun::TypeDescription::td_vec *basis_subtype = basis_type->get_sub_type();
  const SCIRun::TypeDescription *data_type = (*basis_subtype)[0];
  const SCIRun::TypeDescription *meshtype = field->get_type_description(SCIRun::Field::MESH_TD_E);
  
  std::string mesh = meshtype->get_name();
  std::string basis = "";
  
  if ((basistype == "constant")||(basistype == "Constant"))
  {
    basis = "ConstantBasis";
  }
  
  if ((basistype == "linear")||(basistype == "Linear"))
  {
    if (mesh.find("Scanline") != std::string::npos) basis = "CrvLinearLgn";
    if (mesh.find("Image") != std::string::npos) basis = "QuadBilinearLgn";
    if (mesh.find("LatVol") != std::string::npos) basis = "HexTrilinearLgn";
    if (mesh.find("Curve") != std::string::npos) basis = "CrvLinearLgn";
    if (mesh.find("TriSurf") != std::string::npos) basis = "TriLinearLgn";
    if (mesh.find("QuadSurf") != std::string::npos) basis = "QuadBilinearLgn";
    if (mesh.find("TetVol") != std::string::npos) basis = "TetLinearLgn";
    if (mesh.find("PrismVol") != std::string::npos) basis = "PrismLinearLgn";
    if (mesh.find("HexVol") != std::string::npos) basis = "HexTrilinearLgn";
  }

  if ((basistype == "quadratic")||(basistype == "Quadratic"))
  {
    if (mesh.find("Scanline") != std::string::npos) basis = "CrvQuadraticLgn";
    if (mesh.find("Image") != std::string::npos) basis = "QuadBiquadraticLgn";
    if (mesh.find("LatVol") != std::string::npos) basis = "HexTriquadraticLgn";
    if (mesh.find("Curve") != std::string::npos) basis = "CrvQuadraticLgn";
    if (mesh.find("TriSurf") != std::string::npos) basis = "TriQuadraticLgn";
    if (mesh.find("QuadSurf") != std::string::npos) basis = "QuadBiquadraticLgn";
    if (mesh.find("TetVol") != std::string::npos) basis = "TetQuadraticLgn";
    if (mesh.find("PrismVol") != std::string::npos) basis = "PrismQuadraticLgn";
    if (mesh.find("HexVol") != std::string::npos) basis = "HexTriquadraticLgn";
  }

  if ((basistype == "cubic")||(basistype == "Cubic"))
  {
    if (mesh.find("Scanline") != std::string::npos) basis = "CrvCubicHmt";
    if (mesh.find("Image") != std::string::npos) basis = "QuadCubicHmt";
    if (mesh.find("LatVol") != std::string::npos) basis = "HexCubicHmt";
    if (mesh.find("Curve") != std::string::npos) basis = "CrvCubicHmt";
    if (mesh.find("TriSurf") != std::string::npos) basis = "TriCubicHmt";
    if (mesh.find("QuadSurf") != std::string::npos) basis = "QuadCubicHmt";
    if (mesh.find("TetVol") != std::string::npos) basis = "TetCubicHmt";
    if (mesh.find("PrismVol") != std::string::npos) basis = "PrismCubicHmt";
    if (mesh.find("HexVol") != std::string::npos) basis = "HexCubicHmt";
  }

  if (datatype == "Scalar") datatype = "double";
  if ((datatype == "input")||(datatype == "Same as Input"))
  {
    datatype = data_type->get_name();
  }

  if (basis != "") basis = basis + "<" + datatype +" >";
  if (basis == "") basis = basis_type->get_similar_name(datatype, 0, "<", " >");

  std::string fieldtype = field->get_type_description(SCIRun::Field::FIELD_NAME_ONLY_E)->get_name() + "<" +
              field->get_type_description(SCIRun::Field::MESH_TD_E)->get_name() + "," + basis + "," +
              field->get_type_description(SCIRun::Field::FDATA_TD_E)->get_similar_name(datatype, 0,"<", " >") + " > ";
              
  // As I use my own Tensor and Vector algorithms they need to be
  // converted when reading the data, hence separate algorithms are
  // implemented for those cases
  
  std::string algo_name = "ArrayObjectFieldCreateAlgoT";
  std::string algo_base = "ArrayObjectFieldCreateAlgo";

  std::string include_path(SCIRun::TypeDescription::cc_to_h(__FILE__));

  SCIRun::CompileInfoHandle ci = 
    scinew SCIRun::CompileInfo("ALGO"+algo_name + "." +
                       to_filename(fieldtype) + ".",
                       algo_base, 
                       algo_name, 
                       fieldtype);

  // Add in the include path to compile this obj
  ci->add_namespace("SCIRunAlgo");  
  field->get_type_description()->fill_compile_info(ci.get_rep());
  ci->add_data_include(include_path);
  return(ci);
}

void ArrayObjectFieldElemAlgo::getcenter(TensorVectorMath::Vector& node)
{
  node = TensorVectorMath::Vector(0.0,0.0,0.0);
}

void ArrayObjectFieldElemAlgo::getsize(TensorVectorMath::Scalar& size)
{
  size = 0.0;
}

void ArrayObjectFieldElemAlgo::getlength(TensorVectorMath::Scalar& length)
{
  length = 0.0;
}

void ArrayObjectFieldElemAlgo::getarea(TensorVectorMath::Scalar& area)
{
  area = 0.0;
}

void ArrayObjectFieldElemAlgo::getvolume(TensorVectorMath::Scalar& volume)
{
  volume = 0.0;
}

void ArrayObjectFieldElemAlgo::getnormal(TensorVectorMath::Vector& normal)
{
  normal = TensorVectorMath::Vector(0.0,0.0,0.0);
}


bool ArrayObjectFieldElemAlgo::ispoint()
{
  return(false);
}

bool ArrayObjectFieldElemAlgo::isline()
{
  return(false);
}

bool ArrayObjectFieldElemAlgo::issurface()
{
  return(false);
}

bool ArrayObjectFieldElemAlgo::isvolume()
{
  return(false);
}

bool ArrayObjectFieldElemAlgo::setfield(SCIRun::FieldHandle handle)
{
  return(false);
}

void ArrayObjectFieldElemAlgo::reset()
{
}

void ArrayObjectFieldElemAlgo::reset(unsigned int idx)
{
}
 
void ArrayObjectFieldElemAlgo::next()
{
}

int ArrayObjectFieldElemAlgo::size()
{
  return(0);
}

 
void ArrayObjectFieldElemAlgo::getdimension(TensorVectorMath::Scalar& dim)
{
  dim = 0.0;
}    
          
SCIRun::CompileInfoHandle ArrayObjectFieldElemAlgo::get_compile_info(SCIRun::FieldHandle field)
{
  const SCIRun::TypeDescription *fieldtype = field->get_type_description();
  const SCIRun::TypeDescription *locationtype = field->order_type_description();

  // As I use my own Tensor and Vector algorithms they need to be
  // converted when reading the data, hence separate algorithms are
  // implemented for those cases
  
  std::string algo_type = "Point";
  
  if (field->basis_order() == 0)
  {
    SCIRun::MeshHandle mesh = field->mesh().get_rep();
    int dim = mesh->dimensionality();
    if (dim == 1) algo_type = "Line";
    if (dim == 2) algo_type = "Surf";
    if (dim == 3) algo_type = "Volume";
  }
  
  std::string algo_name = "ArrayObjectFieldElem" + algo_type + "AlgoT";
  std::string algo_base = "ArrayObjectFieldElemAlgo";

  std::string include_path(SCIRun::TypeDescription::cc_to_h(__FILE__));

  

  SCIRun::CompileInfoHandle ci = 
    scinew SCIRun::CompileInfo("ALGO"+algo_name + "." +
                       fieldtype->get_filename() + ".",
                       algo_base, 
                       algo_name, 
                       fieldtype->get_name() + "," + locationtype->get_name());

  ci->add_namespace("SCIRunAlgo");
  fieldtype->fill_compile_info(ci.get_rep());
  ci->add_data_include(include_path);

  return(ci);
}

void ArrayObjectFieldElemAlgo::get_normal(SCIRun::TriSurfMesh<TriLinearLgn<Point> > *mesh,SCIRun::TriSurfMesh<TriLinearLgn<Point> >::Face::iterator& it,TensorVectorMath::Vector& vec)
{
  TriSurfMesh<TriLinearLgn<Point> >::Node::array_type nodes;
  mesh->get_nodes(nodes,*it);
  
  Point p1,p2,p3;
  if (nodes.size() >= 3)
  {
    mesh->get_center(p1,nodes[0]);
    mesh->get_center(p2,nodes[1]);
    mesh->get_center(p3,nodes[2]);
    
    TensorVectorMath::Vector vec1(p2.x()-p1.x(),p2.y()-p1.y(),p2.z()-p1.z());
    TensorVectorMath::Vector vec2(p3.x()-p1.x(),p3.y()-p1.y(),p3.z()-p1.z());
    vec = TensorVectorMath::cross(vec1,vec2);    
  }
  else
  {
    vec = TensorVectorMath::Vector(0.0,0.0,0.0);
  }
}


void ArrayObjectFieldElemAlgo::get_normal(SCIRun::QuadSurfMesh<QuadBilinearLgn<Point> > *mesh,SCIRun::QuadSurfMesh<QuadBilinearLgn<Point> >::Face::iterator& it,TensorVectorMath::Vector& vec)
{
  SCIRun::QuadSurfMesh<QuadBilinearLgn<Point> >::Node::array_type nodes;
  mesh->get_nodes(nodes,*it);
  
  SCIRun::Point p1,p2,p3;
  if (nodes.size() >= 3)
  {
    mesh->get_center(p1,nodes[0]); 
    mesh->get_center(p2,nodes[1]); 
    mesh->get_center(p3,nodes[2]); 
    
    TensorVectorMath::Vector vec1(p2.x()-p1.x(),p2.y()-p1.y(),p2.z()-p1.z());
    TensorVectorMath::Vector vec2(p3.x()-p1.x(),p3.y()-p1.y(),p3.z()-p1.z());
    vec = TensorVectorMath::cross(vec1,vec2);    
  }
  else
  {
    vec = TensorVectorMath::Vector(0.0,0.0,0.0);
  }
}

void ArrayObjectFieldElemAlgo::get_normal(SCIRun::StructQuadSurfMesh<QuadBilinearLgn<Point> > *mesh,SCIRun::StructQuadSurfMesh<QuadBilinearLgn<Point> >::Face::iterator& it,TensorVectorMath::Vector& vec)
{
  SCIRun::StructQuadSurfMesh<QuadBilinearLgn<Point> >::Node::array_type nodes;
  mesh->get_nodes(nodes,*it);
  
  Point p1,p2,p3;
  if (nodes.size() >= 3)
  {
    mesh->get_center(p1,nodes[0]);
    mesh->get_center(p2,nodes[1]);
    mesh->get_center(p3,nodes[2]);
    
    TensorVectorMath::Vector vec1(p2.x()-p1.x(),p2.y()-p1.y(),p2.z()-p1.z());
    TensorVectorMath::Vector vec2(p3.x()-p1.x(),p3.y()-p1.y(),p3.z()-p1.z());
    vec = TensorVectorMath::cross(vec1,vec2);    
  }
  else
  {
    vec = TensorVectorMath::Vector(0.0,0.0,0.0);
  }
}

void ArrayObjectFieldElemAlgo::get_normal(SCIRun::ImageMesh<QuadBilinearLgn<Point> > *mesh,SCIRun::ImageMesh<QuadBilinearLgn<Point> >::Face::iterator& it,TensorVectorMath::Vector& vec)
{
  SCIRun::ImageMesh<QuadBilinearLgn<Point> >::Node::array_type nodes;
  mesh->get_nodes(nodes,*it);
  
  Point p1,p2,p3;
  if (nodes.size() >= 3)
  {
    mesh->get_center(p1,nodes[0]);
    mesh->get_center(p2,nodes[1]);
    mesh->get_center(p3,nodes[2]);
    
    TensorVectorMath::Vector vec1(p2.x()-p1.x(),p2.y()-p1.y(),p2.z()-p1.z());
    TensorVectorMath::Vector vec2(p3.x()-p1.x(),p3.y()-p1.y(),p3.z()-p1.z());
    vec = TensorVectorMath::cross(vec1,vec2);    
  }
  else
  {
    vec = TensorVectorMath::Vector(0.0,0.0,0.0);
  }
}


void ArrayObjectFieldElemAlgo::get_normal(SCIRun::TriSurfMesh<TriLinearLgn<Point> > *mesh,SCIRun::TriSurfMesh<TriLinearLgn<Point> >::Node::iterator& it,TensorVectorMath::Vector& vec)
{
  SCIRun::Vector v;
  mesh->synchronize(SCIRun::Mesh::NORMALS_E);
  mesh->get_normal(v,*it);
  vec = TensorVectorMath::Vector(v.x(),v.y(),v.z());
}

void ArrayObjectFieldElemAlgo::get_normal(SCIRun::QuadSurfMesh<QuadBilinearLgn<Point> > *mesh,SCIRun::QuadSurfMesh<QuadBilinearLgn<Point> >::Node::iterator& it,TensorVectorMath::Vector& vec)
{
  SCIRun::Vector v;
  mesh->synchronize(SCIRun::Mesh::NORMALS_E);
  mesh->get_normal(v,*it);
  vec = TensorVectorMath::Vector(v.x(),v.y(),v.z());
}

void ArrayObjectFieldElemAlgo::get_normal(SCIRun::StructQuadSurfMesh<QuadBilinearLgn<Point> > *mesh,SCIRun::StructQuadSurfMesh<QuadBilinearLgn<Point> >::Node::iterator& it,TensorVectorMath::Vector& vec)
{
  SCIRun::Vector v;
  mesh->synchronize(SCIRun::Mesh::NORMALS_E);
  mesh->get_normal(v,*it);
  vec = TensorVectorMath::Vector(v.x(),v.y(),v.z());
}

void ArrayObjectFieldElemAlgo::get_normal(SCIRun::ImageMesh<QuadBilinearLgn<Point> > *mesh,SCIRun::ImageMesh<QuadBilinearLgn<Point> >::Node::iterator& it,TensorVectorMath::Vector& vec)
{
  SCIRun::Vector v;
  mesh->synchronize(SCIRun::Mesh::NORMALS_E);
  mesh->get_normal(v,*it);
  vec = TensorVectorMath::Vector(v.x(),v.y(),v.z());
}

} // namespace ModelCreation
