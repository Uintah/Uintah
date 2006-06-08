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

#include <Packages/ModelCreation/Core/Algorithms/TVMHelp.h>

// TVM stands for Tensor Vector Array Math

namespace TensorVectorMath {

void TVMHelp::createhelplist()
{
    add("Vector()","VECTOR = Vector(SCALAR x,SCALAR y,SCALAR z)","Create a new vector");
    add("Tensor()","TENSOR = Tensor(SCALAR xx,SCALAR xy,SCALAR xz,SCALAR yy,SCALAR yz,SCALAR zz)","Create a new tensor");
    add("Tensor()","TENSOR = Tensor(VECTOR eigvec1,VECTOR eigvec2, VECTOR eigvec3,SCALAR eigval1,SCALAR eigval2,SCALAR eigval3)","Create a new tensor");
    add("Tensor()","TENSOR = Tensor(VECTOR eigvec1,VECTOR eigvec2, SCALAR eigval1,SCALAR eigval2,SCALAR eigval3)","Create a new tensor, the third eigenvector is automatically calculated");
    add("Tensor()","TENSOR = Tensor(VECTOR eigvec1,VECTOR eigvec2, SCALAR eigval1,SCALAR eigval2)","Create a new tensor, assuming the last two eigenvalues are equal");
    add("Tensor()","TENSOR = Tensor(VECTOR eigvec1,SCALAR eigval1,SCALAR eigval2)","Create a new tensor, assuming the last two eigenvalues are equal");
    add("Tensor()","TENSOR = Tensor(SCALAR)","Create isotropic tensor with given eigen value");
    
    add("x()","SCALAR = x(VECTOR)","Get the x coordinate of a vector");
    add("y()","SCALAR = y(VECTOR)","Get the y coordinate of a vector");
    add("z()","SCALAR = z(VECTOR)","Get the z coordinate of a vector");
    add("xx()","SCALAR = xx(TENSOR)","Get the xx coordinate of a tensor");
    add("xy()","SCALAR = xy(TENSOR)","Get the xy coordinate of a tensor");
    add("xz()","SCALAR = xz(TENSOR)","Get the xz coordinate of a tensor");
    add("yy()","SCALAR = yz(TENSOR)","Get the yz coordinate of a tensor");
    add("yz()","SCALAR = yz(TENSOR)","Get the yz coordinate of a tensor");
    add("zz()","SCALAR = zz(TENSOR)","Get the zz coordinate of a tensor");

    add("length()","SCALAR = length(VECTOR)","Compute the length of a vector");
    add("norm()","SCALAR = norm(VECTOR)","Compute the norm of a vector");
    add("maxnorm()","SCALAR = maxnorm(VECTOR)","Compute the maxnorm of a vector");
    add("max()","SCALAR = max(VECTOR)","Find the maximum component of a vector");
    add("min()","SCALAR = min(VECTOR)","Find the minimum component of a vector");

    add("sin()","SCALAR = sin(SCALAR)","Compute the sine");
    add("cos()","SCALAR = cos(SCALAR)","Compute the cosine");
    add("sinh()","SCALAR = sinh(SCALAR)","Compute the hyperbolic sine");
    add("cosh()","SCALAR = cosh(SCALAR)","Compute the hyperbolic cosine ");
    add("tan()","SCALAR = tan(SCALAR)","Compute the tangent " );

    add("asin()","SCALAR = asin(SCALAR)","Compute the arcsine ");
    add("acos()","SCALAR = acos(SCALAR)","Compute the arccosine ");
    add("asinh()","SCALAR = asinh(SCALAR)","Compute the inverse hyperbolic sine ");
    add("acosh()","SCALAR = acosh(SCALAR)","Compute the inverse hyperbolic cosine ");
    add("atan()","SCALAR = atan(SCALAR)","Compute the arctangent ");

    add("log()","SCALAR = log(SCALAR)","Compute the natural logarithm ");
    add("log2()","SCALAR = log2(SCALAR)","Compute the base 2 logarithm ");
    add("log10()","SCALAR = log10(SCALAR)","Compute the base 10 logarithm ");
    
    add("abs()","SCALAR = abs(SCALAR)","Compute the absolute value ");
    add("cbrt()","SCALAR = cbrt(SCALAR)","Compute the cubical root ");
    add("sqrt()","SCALAR = sqrt(SCALAR)","Compute the square root ");
    
    add("pow()","SCALAR = pow(SCALAR,SCALAR)","Compute the nth power ");
    add("exp()","SCALAR = exp(SCALAR)","Compute the exponential ");
    add("ceil()","SCALAR = ceil(SCALAR)","Round each component towards plus infinity");
    add("floor()","SCALAR = floor(SCALAR)","Round each component towards negative infinity");
    add("round()","SCALAR = round(SCALAR)","Round each component towards nearest integer value");

    add("sin()","VECTOR = sin(VECTOR)","Compute the sine of each component");
    add("cos()","VECTOR = cos(VECTOR)","Compute the cosine of each component");
    add("sinh()","VECTOR = sinh(VECTOR)","Compute the hyperbolic sine of each component");
    add("cosh()","VECTOR = cosh(VECTOR)","Compute the hyperbolic cosine of each component");
    add("tan()","VECTOR = tan(VECTOR)","Compute the tangent of each component" );

    add("asin()","VECTOR = asin(VECTOR)","Compute the arcsine of each component");
    add("acos()","VECTOR = acos(VECTOR)","Compute the arccosine of each component");
    add("asinh()","VECTOR = asinh(VECTOR)","Compute the inverse hyperbolic sine of each component");
    add("acosh()","VECTOR = acosh(VECTOR)","Compute the inverse hyperbolic cosine of each component");
    add("atan()","VECTOR = atan(VECTOR)","Compute the arctangent of each component");

    add("log()","VECTOR = log(VECTOR)","Compute the natural logarithm of each component");
    add("log2()","VECTOR = log2(VECTOR)","Compute the base 2 logarithm of each component");
    add("log10()","VECTOR = log10(VECTOR)","Compute the base 10 logarithm of each component");
    
    add("abs()","VECTOR = abs(VECTOR)","Compute the absolute values of each component");
    add("cbrt()","VECTOR = cbrt(VECTOR)","Compute the cubical root of each component");
    add("sqrt()","VECTOR = sqrt(VECTOR)","Compute the square root of each component");
    
    add("pow()","VECTOR = pow(VECTOR,SCALAR)","Compute the nth power of each component");
    add("exp()","VECTOR = exp(VECTOR)","Compute the exponential of each component");
    add("ceil()","VECTOR = ceil(VECTOR)","Round each component towards plus infinity");
    add("floor()","VECTOR = floor(VECTOR)","Round each component towards negative infinity");
    add("round()","VECTOR = round(VECTOR)","Round each component towards nearest integer value");

    add("sin()","TENSOR = sin(TENSOR)","Compute the sine of each component");
    add("cos()","TENSOR = cos(TENSOR)","Compute the cosine of each component");
    add("sinh()","TENSOR = sinh(TENSOR)","Compute the hyperbolic sine of each component");
    add("cosh()","TENSOR = cosh(TENSOR)","Compute the hyperbolic cosine of each component");
    add("tan()","TENSOR = tan(TENSOR)","Compute the tangent of each component" );
    
    add("asin()","TENSOR = asin(TENSOR)","Compute the arcsine of each component");
    add("acos()","TENSOR = acos(TENSOR)","Compute the arccosine of each component");
    add("asinh()","TENSOR = asinh(TENSOR)","Compute the inverse hyperbolic sine of each component");
    add("acosh()","TENSOR = acosh(TENSOR)","Compute the inverse hyperbolic cosine of each component");
    add("atan()","TENSOR = atan(TENSOR)","Compute the arctangent of each component");

    add("log()","TENSOR = log(TENSOR)","Compute the natural logarithm of each component");
    add("log2()","TENSOR = log2(TENSOR)","Compute the base 2 logarithm of each component");
    add("log10()","TENSOR = log10(TENSOR)","Compute the base 10 logarithm of each component");
    
    add("abs()","TENSOR = abs(TENSOR)","Compute the absolute values of each component");
    add("cbrt()","TENSOR = cbrt(TENSOR)","Compute the cubical root of each component");
    add("sqrt()","TENSOR = sqrt(TENSOR)","Compute the square root of each component");
    
    add("pow()","TENSOR = pow(TENSOR,SCALAR)","Compute the nth power of each component");
    add("exp()","TENSOR = exp(TENSOR)","Compute the exponential of each component");
    add("ceil()","TENSOR = ceil(TENSOR)","Round each component towards plus infinity");
    add("floor()","TENSOR = floor(TENSOR)","Round each component towards negative infinity");
    add("round()","TENSOR = round(TENSOR)","Round each component towards nearest integer value");

    add("normalize()","VECTOR = normalize(VECTOR)","Normalize vector");
    add("findnormal1()","VECTOR = findnormal1(VECTOR)","Find a vector that is normal to the argument");
    add("findnormal2()","VECTOR = findnormal2(VECTOR)","Find a vector that is normal to the argument and the vector from findnormal1");

    add("isnan()","SCALAR = isnan(SCALAR)","Check whether scalar is not-a-number");
    add("isinf()","SCALAR = isinf(SCALAR)","Check whether scalar is infinite");
    add("isfinite()","SCALAR = isfinite(SCALAR)","Check whether scalar is a finite number v");
  
    add("isnan()","SCALAR = isnan(VECTOR)","Check whether one of the components is not-a-number");
    add("isinf()","SCALAR = isinf(VECTOR)","Check whether one of the components is infinite");
    add("isfinite()","SCALAR = isfinite(VECTOR)","Check whether all components are finite numbers");

    add("isnan()","SCALAR = isnan(TENSOR)","Check whether one of the components is not-a-number");
    add("isinf()","SCALAR = isinf(TENSOR)","Check whether one of the components is infinite");
    add("isfinite()","SCALAR = isfinite(TENSOR)","Check whether all components are finite numbers");
    
    add("dot()","SCALAR = dot(VECTOR,VECTOR)","Compute the dot product");
    add("cross()","VECTOR = cross(VECTOR,VECTOR)","Compute the cross product");
    
    add("boolean()","SCALAR = boolean(SCALAR)","This function returns 1.0 if the scalar is not 0.0 and otherwise 0.0");     
    add("boolean()","SCALAR = boolean(VECTOR)","This function returns 1.0 if the vector is not (0.0,0.0,0.0) and otherwise 0.0"); 
    add("boolean()","SCALAR = boolean(TENSOR)","This function returns 1.0 if the tensor is not (0.0,0.0,0.0,0.0,0.0,0.0) and otherwise 0.0"); 

    add("nan","nan","Not-a-Number constant");
    add("pi","pi","the number 3.14159265358979323846");
    add("e","e","the number 2.71828182845904523536");  

    add("eigvec1()","VECTOR = eigvec1(TENSOR)","The first eigenvector of the tensor");
    add("eigvec2()","VECTOR = eigvec2(TENSOR)","The second eigenvector of the tensor");
    add("eigvec3()","VECTOR = eigvec3(TENSOR)","The third eigenvector of the tensor");
    add("eigvals()","VECTOR = eigvals(TENSOR)","The three eigenvalues of the tensor");

    add("eigval1()","SCALAR = eigval1(TENSOR)","The first eigenvector of the tensor");
    add("eigval2()","SCALAR = eigval2(TENSOR)","The second eigenvector of the tensor");
    add("eigval3()","SCALAR = eigval3(TENSOR)","The third eigenvector of the tensor");

    add("trace()","SCALAR =  trace(TENSOR)","The trace of the tensor (sum of eigenvalues)");
    add("det()","SCALAR =  det(TENSOR)","The determinant of the tensor (product of eigenvalues)");

    add("frobenius()","SCALAR = frobenius(TENSOR)","Frobenius norm of tensor");
    add("frobenius2()","SCALAR = frobenius2(TENSOR)","Square of Frobenius norm of tensor");
    add("fracanisotropy()","SCALAR = fracanisotropy(TENSOR)","Fractional anisotropy of tensor");

    add("vec1()","VECTOR = vec1(TENSOR)","First column vector of tensor");
    add("vec2()","VECTOR = vec2(TENSOR)","Second column vector of tensor");
    add("vec3()","VECTOR = vec3(TENSOR)","Third column vector of tensor");

    add("AND()","SCALAR = AND(SCALAR/VECTOR/TENSOR,SCALAR/VECOR/TENSOR)","Boolean 'and' operation");
    add("OR()","SCALAR = OR(SCALAR/VECOR/TENSOR,SCALAR/VECOR/TENSOR)","Boolean 'or' operation");
    add("XOR()","SCALAR = XOR(SCALAR/VECOR/TENSOR,SCALAR/VECOR/TENSOR)","Boolean 'xor' operation");
    add("NOT()","SCALAR = NOT(SCALAR/VECOR/TENSOR)","Boolean 'not' operation");
  
    add("inv()","TENSOR = inv(TENSOR)","Compute the inverse of the tensor");
    add("inv()","SCALAR = inv(SCALAR)","Compute 1/SCALAR");

}


void TVMHelp::createhelplist_element()
{
    add("dimension()","SCALAR = dimension(ELEMENT)","This function returns 0 for data at a node, 1 for data at a line, 2 for data at a surface, 3 for data in a volume");
    add("center()","VECTOR = center(ELEMENT)","Compute the center of the current element");
    add("size()","SCALAR = size(ELEMENT)","Compute the size of the element. Depending on the dimension of the element it computes the length, area, or volume");
    add("length()","SCALAR = length(ELEMENT)","Compute the length of the element");
    add("area()","SCALAR = area(ELEMENT)","Compute the volume of the element");
    add("volume()","SCALAR = volume(ELEMENT)","Compute the volume of the element");
    add("normal()","VECTOR = normal(ELEMENT)","Compute the local normal for surface mesh");
}


std::string TVMHelp::gethelp(bool addelem )
{
  helplist_.clear();
 
  createhelplist();
  if (addelem)
  {
    createhelplist_element();
  }
  
  helphtml_ = std::string("<H3>Tensor/Vector/Scalar functions</H3>\n\n");

  helplist_.sort();
  
  std::list<TVMHelpEntry>::iterator it;
  std::string oldname;

  for (it = helplist_.begin(); it != helplist_.end(); it++)
  {
    std::string helptext;
    if (oldname != (*it).functionname) helptext = "\n<H4>"+(*it).functionname+"</H4>\n";
    helptext += "<p>"+(*it).functionsyntax+"<\\p>\n";
    helptext += "<p>"+(*it).description+"<\\p>\n\n";
    helphtml_ += helptext;    
    oldname = (*it).functionname;
  }  

  return(helphtml_);
}

void TVMHelp::add(std::string functionname,std::string functionsyntax, std::string description)
{
  TVMHelpEntry entry;
  entry.functionname = functionname;
  entry.functionsyntax = functionsyntax;
  entry.description = description;
  helplist_.push_back(entry);
}

TVMHelp::TVMHelp()
{
}

} //end namespace
