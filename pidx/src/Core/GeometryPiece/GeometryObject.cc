/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <Core/GeometryPiece/GeometryObject.h>
#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <iostream>

using namespace Uintah;


GeometryObject::GeometryObject(GeometryPieceP piece, ProblemSpecP& ps,
                               list<DataItem>& data) :
  d_piece(piece)
{
   for (list<DataItem>::iterator it = data.begin(); it != data.end();it++)
   {
     switch(it->type)
     {
        case Double:
        {
          double val;
          if(it->name == "volumeFraction")
          {
              ps->getWithDefault(it->name,val,-1.0);
          } else
          {
              ps->require(it->name,val);
          }
          d_double_data[it->name] = val;
          break;
        }
        case Integer:
        {
          int val;
          ps->require(it->name,val);
          d_int_data[it->name] = val;
          break;
        }
        case Vector:
        {
          Uintah::Vector val;
          if(it->name == "affineTransformation_A0")
          {
              ps->getWithDefault(it->name,val,Uintah::Vector(1.,0.,0.));

          } else if(it->name == "affineTransformation_A1")
          {
              ps->getWithDefault(it->name,val,Uintah::Vector(0.,1.,0.));

          } else if(it->name == "affineTransformation_A2")
          {
              ps->getWithDefault(it->name,val,Uintah::Vector(0.,0.,1.));

          } else if(it->name == "affineTransformation_b")
          {
              ps->getWithDefault(it->name,val,Uintah::Vector(0.,0.,0.));

          } else
          {
              ps->require(it->name,val);
          }
          d_vector_data[it->name] = val;
          break;
        }
        case IntVector:
        {
          Uintah::IntVector val;
          ps->require(it->name,val);
          d_intvector_data[it->name] = val;
          break;
        }
        case Point:
        {
          Uintah::Point val;
          ps->require(it->name,val);
          d_point_data[it->name] = val;
          break;
        }
     };
   }
}

void
GeometryObject::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP geom_obj_ps = ps->appendChild("geom_object");
  d_piece->outputProblemSpec(geom_obj_ps);
  
  for (map<string,double>::iterator it = d_double_data.begin(); 
       it != d_double_data.end(); it++) {
    if(!(it->first.compare("volumeFraction") == 0 && it->second == -1.0))
      geom_obj_ps->appendElement(it->first.c_str(),it->second);
  }
  for (map<string,Uintah::Vector>::iterator it = d_vector_data.begin(); 
       it != d_vector_data.end(); it++) {
    geom_obj_ps->appendElement(it->first.c_str(),it->second);
  }
  for (map<string,Uintah::IntVector>::iterator it = d_intvector_data.begin(); 
       it != d_intvector_data.end(); it++) {
    geom_obj_ps->appendElement(it->first.c_str(),it->second);
  }
  for (map<string,Uintah::Point>::iterator it = d_point_data.begin(); 
       it != d_point_data.end(); it++) {
    geom_obj_ps->appendElement(it->first.c_str(),it->second);
  }
}
