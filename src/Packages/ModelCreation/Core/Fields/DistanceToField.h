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

#ifndef MODELCREATION_CORE_FIELDS_DISTANCETOFIELD_H
#define MODELCREATION_CORE_FIELDS_DISTANCETOFIELD_H 1

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Util/DynamicCompilation.h>
#include <Core/Util/ProgressReporter.h>

#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/Mesh.h>

#include <sci_hash_map.h>
#include <Packages/ModelCreation/Core/Datatypes/SelectionMask.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace ModelCreation {

using namespace SCIRun;

class DistanceToFieldAlgo : public DynamicAlgoBase
{
public:

  virtual bool execute(ProgressReporter *reporter,
                              FieldHandle input,
                              FieldHandle& output,
                              FieldHandle objectfield) = 0;

  virtual bool execute_signed(ProgressReporter *reporter,
                              FieldHandle input,
                              FieldHandle& output,
                              FieldHandle objectfield) = 0;

  virtual bool execute_unsigned(ProgressReporter *reporter,
                              FieldHandle input,
                              FieldHandle& output,
                              FieldHandle objectfield) = 0;

  virtual bool execute_isinside(ProgressReporter *reporter,
                              FieldHandle input,
                              FieldHandle& output,
                              FieldHandle objectfield) = 0;
                              
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc,
                                            const TypeDescription *ofsrc);

  inline void distance(Point c, Point t1, Point t2, Point t3, Point t4, double& dist);                                            
  inline void distance(Point c, Point t1, Point t2, Point t3, double& dist);
  inline void distance(Point c, Point t1, Point t2, double& dist);
  inline void distance(Point c, Point t1, double& dist);                                            

  inline bool sdistance(Point c, Point t1, Point t2, Point t3, Point t4, double& dist);                                            
  inline bool sdistance(Point c, Point t1, Point t2, Point t3, double& dist);
  inline bool sdistance(Point c, Point t1, Point t2, double& dist);
  inline bool sdistance(Point c, Point t1, double& dist);                                            

};

template<class FIELD, class OBJECTFIELD>
class DistanceToFieldAlgoT : public DistanceToFieldAlgo
{
public:
  virtual bool execute(ProgressReporter *reporter,
                              FieldHandle input,
                              FieldHandle& output,
                              FieldHandle objectfield);                          

  virtual bool execute_signed(ProgressReporter *reporter,
                              FieldHandle input,
                              FieldHandle& output,
                              FieldHandle objectfield); 

  virtual bool execute_unsigned(ProgressReporter *reporter,
                              FieldHandle input,
                              FieldHandle& output,
                              FieldHandle objectfield);

  virtual bool execute_isinside(ProgressReporter *reporter,
                              FieldHandle input,
                              FieldHandle& output,
                              FieldHandle objectfield); 

};

template<class FIELD, class OBJECTFIELD>
bool DistanceToFieldAlgoT<FIELD,OBJECTFIELD>::execute(ProgressReporter *reporter,
                              FieldHandle input,
                              FieldHandle& output,
                              FieldHandle objectfield)
{                              
  OBJECTFIELD* tfield = dynamic_cast<OBJECTFIELD* >(objectfield.get_rep());
  if (tfield == 0)
  {
    reporter->error("DistanceToField: Object is not valid");
    return(false);
  }

  FIELD* field = dynamic_cast<FIELD* >(input.get_rep());
  if (field == 0)
  {
    reporter->error("DistanceToField: There is no input field");
    return(false);
  }

  typename FIELD::mesh_type *mesh =
    dynamic_cast<typename FIELD::mesh_type *>(field->mesh().get_rep());

  typename OBJECTFIELD::mesh_type *objectmesh =
    dynamic_cast<typename OBJECTFIELD::mesh_type *>(objectfield->mesh().get_rep());  

  if (objectmesh->dimensionality() == 3)
  {
    reporter->error("DistanceToField: This function has not been implemented for volume fields");
    return(false);
  }

  if (field->basis_order() == 0)
  {
    FIELD* ofield = scinew FIELD(mesh,0);
    ofield->resize_fdata();
    output = dynamic_cast<SCIRun::Field* >(ofield);

    typename FIELD::mesh_type::Elem::iterator bi, ei;
    mesh->begin(bi); mesh->end(ei);

    typename OBJECTFIELD::mesh_type::Elem::iterator bt, et;
    typename OBJECTFIELD::mesh_type::Node::array_type nodes;
    typename OBJECTFIELD::mesh_type::Node::array_type nodestest;
    
    Point p;
    Point p0,p1,p2,p3;    
    
    objectmesh->begin(bt);
    objectmesh->get_nodes(nodestest, *(bt));
    
    switch (nodestest.size())
    {
      case 1:
        while (bi != ei)
        {
          mesh->get_center(p,*(bi));
          objectmesh->begin(bt); objectmesh->end(et);
          
          objectmesh->get_nodes(nodes, *(bt));
          double dist = 0.0;
          double d = 0.0;

          objectmesh->get_point(p0,nodes[0]);
          distance(p,p0,dist);

          ++bt;
          while (bt != et)
          {
            objectmesh->get_nodes(nodes, *(bt));
            objectmesh->get_point(p0,nodes[0]);
            distance(p,p0,d);
            if(d < dist ) dist = d;
            ++bt;
          }
          ofield->set_value(sqrt(dist),*(bi));
          ++bi;   
        }
        break;
      case 2:
        while (bi != ei)
        {
          mesh->get_center(p,*(bi));
          objectmesh->begin(bt); objectmesh->end(et);
          
          objectmesh->get_nodes(nodes, *(bt));
          double dist = 0.0;
          double d = 0.0;

          objectmesh->get_point(p0,nodes[0]);
          objectmesh->get_point(p1,nodes[1]);
          distance(p,p0,p1,dist);

          ++bt;
          while (bt != et)
          {
            objectmesh->get_nodes(nodes, *(bt));
            objectmesh->get_point(p0,nodes[0]);
            objectmesh->get_point(p1,nodes[1]);
            distance(p,p0,p1,d);
            if(d < dist ) dist = d;
            ++bt;
          }
          ofield->set_value(sqrt(dist),*(bi));
          ++bi;   
        }
        break;
      case 3:
        while (bi != ei)
        {
          mesh->get_center(p,*(bi));
          objectmesh->begin(bt); objectmesh->end(et);
          
          objectmesh->get_nodes(nodes, *(bt));
          double dist = 0.0;
          double d = 0.0;

          objectmesh->get_point(p0,nodes[0]);
          objectmesh->get_point(p1,nodes[1]);
          objectmesh->get_point(p2,nodes[2]);
          distance(p,p0,p1,p2,dist);

          ++bt;
          while (bt != et)
          {
            objectmesh->get_nodes(nodes, *(bt));
            objectmesh->get_point(p0,nodes[0]);
            objectmesh->get_point(p1,nodes[1]);
            objectmesh->get_point(p2,nodes[2]);
            distance(p,p0,p1,p2,d);
            if(d < dist ) dist = d;
            ++bt;
          }
          ofield->set_value(sqrt(dist),*(bi));
          ++bi;   
        }
        break;
      case 4:
        while (bi != ei)
        {
          mesh->get_center(p,*(bi));
          objectmesh->begin(bt); objectmesh->end(et);
          
          objectmesh->get_nodes(nodes, *(bt));
          double dist = 0.0;
          double d = 0.0;

          objectmesh->get_point(p0,nodes[0]);
          objectmesh->get_point(p1,nodes[1]);
          objectmesh->get_point(p2,nodes[2]);
          objectmesh->get_point(p3,nodes[3]);
          distance(p,p0,p1,p2,p3,dist);

          ++bt;
          while (bt != et)
          {
            objectmesh->get_nodes(nodes, *(bt));
            objectmesh->get_point(p0,nodes[0]);
            objectmesh->get_point(p1,nodes[1]);
            objectmesh->get_point(p2,nodes[2]);
            objectmesh->get_point(p3,nodes[3]);
            distance(p,p0,p1,p2,p3,d);
            if(d < dist ) dist = d;
            ++bt;
          }
          ofield->set_value(sqrt(dist),*(bi));
          ++bi;   
        }
        break;    
      default:
        reporter->error("DistanceToField: Expected a Quadrilateral, Triangular, Line or Point element");
        return(false);              
    }
  }
  else
  {
    FIELD* ofield = scinew FIELD(mesh,1);
    ofield->resize_fdata();
    output = dynamic_cast<SCIRun::Field* >(ofield);

    typename FIELD::mesh_type::Node::iterator bi, ei;
    mesh->begin(bi); mesh->end(ei);

    typename OBJECTFIELD::mesh_type::Elem::iterator bt, et;
    typename OBJECTFIELD::mesh_type::Node::array_type nodes;
    typename OBJECTFIELD::mesh_type::Node::array_type nodestest;
    
    Point p;
    Point p0,p1,p2,p3;    
    
    objectmesh->begin(bt);
    objectmesh->get_nodes(nodestest, *(bt));
    
    switch (nodestest.size())
    {
      case 1:
        while (bi != ei)
        {
          mesh->get_center(p,*(bi));
          objectmesh->begin(bt); objectmesh->end(et);
          
          objectmesh->get_nodes(nodes, *(bt));
          double dist = 0.0;
          double d = 0.0;

          objectmesh->get_point(p0,nodes[0]);
          distance(p,p0,dist);

          ++bt;
          while (bt != et)
          {
            objectmesh->get_nodes(nodes, *(bt));
            objectmesh->get_point(p0,nodes[0]);
            distance(p,p0,d);
            if(d < dist ) dist = d;
            ++bt;
          }
          ofield->set_value(sqrt(dist),*(bi));
          ++bi;   
        }
        break;
      case 2:
        while (bi != ei)
        {
          mesh->get_center(p,*(bi));
          objectmesh->begin(bt); objectmesh->end(et);
          
          objectmesh->get_nodes(nodes, *(bt));
          double dist = 0.0;
          double d = 0.0;

          objectmesh->get_point(p0,nodes[0]);
          objectmesh->get_point(p1,nodes[1]);
          distance(p,p0,p1,dist);

          ++bt;
          while (bt != et)
          {
            objectmesh->get_nodes(nodes, *(bt));
            objectmesh->get_point(p0,nodes[0]);
            objectmesh->get_point(p1,nodes[1]);
            distance(p,p0,p1,d);
            if(d < dist ) dist = d;
            ++bt;
          }
          ofield->set_value(sqrt(dist),*(bi));
          ++bi;   
        }
        break;
      case 3:
        while (bi != ei)
        {
          mesh->get_center(p,*(bi));
          objectmesh->begin(bt); objectmesh->end(et);
          
          objectmesh->get_nodes(nodes, *(bt));
          double dist = 0.0;
          double d = 0.0;

          objectmesh->get_point(p0,nodes[0]);
          objectmesh->get_point(p1,nodes[1]);
          objectmesh->get_point(p2,nodes[2]);
          distance(p,p0,p1,p2,dist);

          ++bt;
          while (bt != et)
          {
            objectmesh->get_nodes(nodes, *(bt));
            objectmesh->get_point(p0,nodes[0]);
            objectmesh->get_point(p1,nodes[1]);
            objectmesh->get_point(p2,nodes[2]);
            distance(p,p0,p1,p2,d);
            if(d < dist ) dist = d;
            ++bt;
          }
          ofield->set_value(sqrt(dist),*(bi));
          ++bi;   
        }
        break;
      case 4:
        while (bi != ei)
        {
          mesh->get_center(p,*(bi));
          objectmesh->begin(bt); objectmesh->end(et);
          
          objectmesh->get_nodes(nodes, *(bt));
          double dist = 0.0;
          double d = 0.0;

          objectmesh->get_point(p0,nodes[0]);
          objectmesh->get_point(p1,nodes[1]);
          objectmesh->get_point(p2,nodes[2]);
          objectmesh->get_point(p3,nodes[3]);
          distance(p,p0,p1,p2,p3,dist);

          ++bt;
          while (bt != et)
          {
            objectmesh->get_nodes(nodes, *(bt));
            objectmesh->get_point(p0,nodes[0]);
            objectmesh->get_point(p1,nodes[1]);
            objectmesh->get_point(p2,nodes[2]);
            objectmesh->get_point(p3,nodes[3]);
            distance(p,p0,p1,p2,p3,d);
            if(d < dist ) dist = d;
            ++bt;
          }
          ofield->set_value(sqrt(dist),*(bi));
          ++bi;   
        }
        break;  
      default:
        reporter->error("DistanceToField: Expected a Quadrilateral, Triangular, Line or Point element");
        return(false);          
    }
  }

  return(true);
}



template<class FIELD, class OBJECTFIELD>
bool DistanceToFieldAlgoT<FIELD,OBJECTFIELD>::execute_signed(ProgressReporter *reporter,
                              FieldHandle input,
                              FieldHandle& output,
                              FieldHandle objectfield)
{              
  OBJECTFIELD* tfield = dynamic_cast<OBJECTFIELD* >(objectfield.get_rep());
  if (tfield == 0)
  {
    reporter->error("SignedDistanceToField: Object is not valid");
    return(false);
  }

  FIELD* field = dynamic_cast<FIELD* >(input.get_rep());
  if (field == 0)
  {
    reporter->error("SignedDistanceToField: There is no input field");
    return(false);
  }

  typename FIELD::mesh_type *mesh =
    dynamic_cast<typename FIELD::mesh_type *>(field->mesh().get_rep());

  typename OBJECTFIELD::mesh_type *objectmesh =
    dynamic_cast<typename OBJECTFIELD::mesh_type *>(objectfield->mesh().get_rep());  

  if (objectmesh->dimensionality() != 2)
  {
    reporter->error("SignedDistanceToField: This function has not been implemented for point clouds, line element fields, or volume fields,");
    return(false);
  }

  if (field->basis_order() == 0)
  {
    FIELD* ofield = scinew FIELD(mesh,0);
    ofield->resize_fdata();
    output = dynamic_cast<SCIRun::Field* >(ofield);

    typename FIELD::mesh_type::Elem::iterator bi, ei;
    mesh->begin(bi); mesh->end(ei);

    typename OBJECTFIELD::mesh_type::Elem::iterator bt, et;
    typename OBJECTFIELD::mesh_type::Node::array_type nodes;
    typename OBJECTFIELD::mesh_type::Node::array_type nodestest;
    
    Point p;
    Point p0,p1,p2,p3;    
    
    objectmesh->begin(bt);
    objectmesh->get_nodes(nodestest, *(bt));
    
    switch (nodestest.size())
    {
      case 3:
        while (bi != ei)
        {
          mesh->get_center(p,*(bi));
          objectmesh->begin(bt); objectmesh->end(et);
          
          objectmesh->get_nodes(nodes, *(bt));
          double dist = 0.0;
          double absdist = 0.0;
          double d = 0.0;

          while (bt != et)
          {
            objectmesh->get_point(p0,nodes[0]);
            objectmesh->get_point(p1,nodes[1]);
            objectmesh->get_point(p2,nodes[2]);
            if(sdistance(p,p0,p1,p2,dist)) 
            {
              absdist = abs(dist);
              break;
            }
            ++bt;
          }
          
          ++bt;
          while (bt != et)
          {
            objectmesh->get_nodes(nodes, *(bt));
            objectmesh->get_point(p0,nodes[0]);
            objectmesh->get_point(p1,nodes[1]);
            objectmesh->get_point(p2,nodes[2]);
            if (sdistance(p,p0,p1,p2,d))
            {
              if(abs(d) < absdist ) 
              {
                dist = d;
                absdist = abs(d);
              }
            }
            ++bt;
          }
          double s = 1.0; if (dist < 0) s = -1.0;
          ofield->set_value(s*sqrt(absdist),*(bi));
          ++bi;   
        }
        break;
      case 4:
        while (bi != ei)
        {
          mesh->get_center(p,*(bi));
          objectmesh->begin(bt); objectmesh->end(et);
          
          objectmesh->get_nodes(nodes, *(bt));
          double dist = 0.0;
          double absdist = 0.0;
          double d = 0.0;

          while (bt != et)
          {
            objectmesh->get_point(p0,nodes[0]);
            objectmesh->get_point(p1,nodes[1]);
            objectmesh->get_point(p2,nodes[2]);
            objectmesh->get_point(p3,nodes[3]);
            if(sdistance(p,p0,p1,p2,p3,dist)) 
            {
              absdist = abs(dist);
              break;
            }
            ++bt;
          }
          
          ++bt;
          while (bt != et)
          {
            objectmesh->get_nodes(nodes, *(bt));
            objectmesh->get_point(p0,nodes[0]);
            objectmesh->get_point(p1,nodes[1]);
            objectmesh->get_point(p2,nodes[2]);
            objectmesh->get_point(p3,nodes[3]);
            if (sdistance(p,p0,p1,p2,d))
            {
              if(abs(d) < absdist ) 
              {
                dist = d;
                absdist = abs(d);
              }
            }
            ++bt;
          }
          double s = 1.0; if (dist < 0) s = -1.0;
          ofield->set_value(s*sqrt(absdist),*(bi));
          ++bi;   
        }
        break;
      default:
        reporter->error("SignedDistanceToField: Expected a Quadrilateral or Triangular element");
        return(false);
    }
  }
  else
  {
    FIELD* ofield = scinew FIELD(mesh,1);
    ofield->resize_fdata();
    output = dynamic_cast<SCIRun::Field* >(ofield);

    typename FIELD::mesh_type::Node::iterator bi, ei;
    mesh->begin(bi); mesh->end(ei);

    typename OBJECTFIELD::mesh_type::Elem::iterator bt, et;
    typename OBJECTFIELD::mesh_type::Node::array_type nodes;
    typename OBJECTFIELD::mesh_type::Node::array_type nodestest;
    
    Point p;
    Point p0,p1,p2,p3;    
    
    objectmesh->begin(bt);
    objectmesh->get_nodes(nodestest, *(bt));
    
    switch (nodestest.size())
    {
      case 3:
        while (bi != ei)
        {
          mesh->get_center(p,*(bi));
          objectmesh->begin(bt); objectmesh->end(et);
          
          objectmesh->get_nodes(nodes, *(bt));
          double dist = 0.0;
          double absdist = 0.0;
          double d = 0.0;

          while (bt != et)
          {
            objectmesh->get_point(p0,nodes[0]);
            objectmesh->get_point(p1,nodes[1]);
            objectmesh->get_point(p2,nodes[2]);
            if(sdistance(p,p0,p1,p2,dist)) 
            {
              absdist = abs(dist);
              break;
            }
            ++bt;
          }
          
          ++bt;
          while (bt != et)
          {
            objectmesh->get_nodes(nodes, *(bt));
            objectmesh->get_point(p0,nodes[0]);
            objectmesh->get_point(p1,nodes[1]);
            objectmesh->get_point(p2,nodes[2]);
            if (sdistance(p,p0,p1,p2,d))
            {
              if(abs(d) < absdist ) 
              {
                dist = d;
                absdist = abs(d);
              }
            }
            ++bt;
          }
          double s = 1.0; if (dist < 0) s = -1.0;
          ofield->set_value(s*sqrt(absdist),*(bi));
          ++bi;   
        }
        break;
      case 4:
        while (bi != ei)
        {
          mesh->get_center(p,*(bi));
          objectmesh->begin(bt); objectmesh->end(et);
          
          objectmesh->get_nodes(nodes, *(bt));
          double dist = 0.0;
          double absdist = 0.0;
          double d = 0.0;

          while (bt != et)
          {
            objectmesh->get_point(p0,nodes[0]);
            objectmesh->get_point(p1,nodes[1]);
            objectmesh->get_point(p2,nodes[2]);
            objectmesh->get_point(p3,nodes[3]);
            if(sdistance(p,p0,p1,p2,p3,dist)) 
            {
              absdist = abs(dist);
              break;
            }
            ++bt;
          }
          
          ++bt;
          while (bt != et)
          {
            objectmesh->get_nodes(nodes, *(bt));
            objectmesh->get_point(p0,nodes[0]);
            objectmesh->get_point(p1,nodes[1]);
            objectmesh->get_point(p2,nodes[2]);
            objectmesh->get_point(p3,nodes[3]);
            if (sdistance(p,p0,p1,p2,d))
            {
              if(abs(d) < absdist ) 
              {
                dist = d;
                absdist = abs(d);
              }
            }
            ++bt;
          }
          double s = 1.0; if (dist < 0) s = -1.0;
          ofield->set_value(s*sqrt(absdist),*(bi));
          ++bi;   
        }
        break;
      default:
        reporter->error("SignedDistanceToField: Expected a Quadrilateral or Triangular element");
        return(false);
    }
  }

  return(true);
}



template<class FIELD, class OBJECTFIELD>
bool DistanceToFieldAlgoT<FIELD,OBJECTFIELD>::execute_unsigned(ProgressReporter *reporter,
                              FieldHandle input,
                              FieldHandle& output,
                              FieldHandle objectfield)
{              
  OBJECTFIELD* tfield = dynamic_cast<OBJECTFIELD* >(objectfield.get_rep());
  if (tfield == 0)
  {
    reporter->error("SignedDistanceToField: Object is not valid");
    return(false);
  }

  FIELD* field = dynamic_cast<FIELD* >(input.get_rep());
  if (field == 0)
  {
    reporter->error("SignedDistanceToField: There is no input field");
    return(false);
  }

  typename FIELD::mesh_type *mesh =
    dynamic_cast<typename FIELD::mesh_type *>(field->mesh().get_rep());

  typename OBJECTFIELD::mesh_type *objectmesh =
    dynamic_cast<typename OBJECTFIELD::mesh_type *>(objectfield->mesh().get_rep());  

  if (objectmesh->dimensionality() != 2)
  {
    reporter->error("SignedDistanceToField: This function has not been implemented for point clouds, line element fields, or volume fields,");
    return(false);
  }

  if (field->basis_order() == 0)
  {
    FIELD* ofield = scinew FIELD(mesh,0);
    ofield->resize_fdata();
    output = dynamic_cast<SCIRun::Field* >(ofield);

    typename FIELD::mesh_type::Elem::iterator bi, ei;
    mesh->begin(bi); mesh->end(ei);

    typename OBJECTFIELD::mesh_type::Elem::iterator bt, et;
    typename OBJECTFIELD::mesh_type::Node::array_type nodes;
    typename OBJECTFIELD::mesh_type::Node::array_type nodestest;
    
    Point p;
    Point p0,p1,p2,p3;    
    
    objectmesh->begin(bt);
    objectmesh->get_nodes(nodestest, *(bt));
    
    switch (nodestest.size())
    {
      case 3:
        while (bi != ei)
        {
          mesh->get_center(p,*(bi));
          objectmesh->begin(bt); objectmesh->end(et);
          
          objectmesh->get_nodes(nodes, *(bt));
          double dist = 0.0;
          double absdist = 0.0;
          double d = 0.0;

          while (bt != et)
          {
            objectmesh->get_point(p0,nodes[0]);
            objectmesh->get_point(p1,nodes[1]);
            objectmesh->get_point(p2,nodes[2]);
            if(sdistance(p,p0,p1,p2,dist)) 
            {
              absdist = abs(dist);
              break;
            }
            ++bt;
          }
          
          ++bt;
          while (bt != et)
          {
            objectmesh->get_nodes(nodes, *(bt));
            objectmesh->get_point(p0,nodes[0]);
            objectmesh->get_point(p1,nodes[1]);
            objectmesh->get_point(p2,nodes[2]);
            if (sdistance(p,p0,p1,p2,d))
            {
              if(abs(d) < absdist ) 
              {
                dist = d;
                absdist = abs(d);
              }
            }
            ++bt;
          }
          double s = 1.0; if (dist < 0) s = 0.0;
          ofield->set_value(s*sqrt(absdist),*(bi));
          ++bi;   
        }
        break;
      case 4:
        while (bi != ei)
        {
          mesh->get_center(p,*(bi));
          objectmesh->begin(bt); objectmesh->end(et);
          
          objectmesh->get_nodes(nodes, *(bt));
          double dist = 0.0;
          double absdist = 0.0;
          double d = 0.0;

          while (bt != et)
          {
            objectmesh->get_point(p0,nodes[0]);
            objectmesh->get_point(p1,nodes[1]);
            objectmesh->get_point(p2,nodes[2]);
            objectmesh->get_point(p3,nodes[3]);
            if(sdistance(p,p0,p1,p2,p3,dist)) 
            {
              absdist = abs(dist);
              break;
            }
            ++bt;
          }
          
          ++bt;
          while (bt != et)
          {
            objectmesh->get_nodes(nodes, *(bt));
            objectmesh->get_point(p0,nodes[0]);
            objectmesh->get_point(p1,nodes[1]);
            objectmesh->get_point(p2,nodes[2]);
            objectmesh->get_point(p3,nodes[3]);
            if (sdistance(p,p0,p1,p2,d))
            {
              if(abs(d) < absdist ) 
              {
                dist = d;
                absdist = abs(d);
              }
            }
            ++bt;
          }
          double s = 1.0; if (dist < 0) s = 0.0;
          ofield->set_value(s*sqrt(absdist),*(bi));
          ++bi;   
        }
        break;
      default:
        reporter->error("SignedDistanceToField: Expected a Quadrilateral or Triangular element");
        return(false);
    }
  }
  else
  {
    FIELD* ofield = scinew FIELD(mesh,1);
    ofield->resize_fdata();
    output = dynamic_cast<SCIRun::Field* >(ofield);

    typename FIELD::mesh_type::Node::iterator bi, ei;
    mesh->begin(bi); mesh->end(ei);

    typename OBJECTFIELD::mesh_type::Elem::iterator bt, et;
    typename OBJECTFIELD::mesh_type::Node::array_type nodes;
    typename OBJECTFIELD::mesh_type::Node::array_type nodestest;
    
    Point p;
    Point p0,p1,p2,p3;    
    
    objectmesh->begin(bt);
    objectmesh->get_nodes(nodestest, *(bt));
    
    switch (nodestest.size())
    {
      case 3:
        while (bi != ei)
        {
          mesh->get_center(p,*(bi));
          objectmesh->begin(bt); objectmesh->end(et);
          
          objectmesh->get_nodes(nodes, *(bt));
          double dist = 0.0;
          double absdist = 0.0;
          double d = 0.0;

          while (bt != et)
          {
            objectmesh->get_point(p0,nodes[0]);
            objectmesh->get_point(p1,nodes[1]);
            objectmesh->get_point(p2,nodes[2]);
            if(sdistance(p,p0,p1,p2,dist)) 
            {
              absdist = abs(dist);
              break;
            }
            ++bt;
          }
          
          ++bt;
          while (bt != et)
          {
            objectmesh->get_nodes(nodes, *(bt));
            objectmesh->get_point(p0,nodes[0]);
            objectmesh->get_point(p1,nodes[1]);
            objectmesh->get_point(p2,nodes[2]);
            if (sdistance(p,p0,p1,p2,d))
            {
              if(abs(d) < absdist ) 
              {
                dist = d;
                absdist = abs(d);
              }
            }
            ++bt;
          }
          double s = 1.0; if (dist < 0) s = 0.0;
          ofield->set_value(s*sqrt(absdist),*(bi));
          ++bi;   
        }
        break;
      case 4:
        while (bi != ei)
        {
          mesh->get_center(p,*(bi));
          objectmesh->begin(bt); objectmesh->end(et);
          
          objectmesh->get_nodes(nodes, *(bt));
          double dist = 0.0;
          double absdist = 0.0;
          double d = 0.0;

          while (bt != et)
          {
            objectmesh->get_point(p0,nodes[0]);
            objectmesh->get_point(p1,nodes[1]);
            objectmesh->get_point(p2,nodes[2]);
            objectmesh->get_point(p3,nodes[3]);
            if(sdistance(p,p0,p1,p2,p3,dist)) 
            {
              absdist = abs(dist);
              break;
            }
            ++bt;
          }
          
          ++bt;
          while (bt != et)
          {
            objectmesh->get_nodes(nodes, *(bt));
            objectmesh->get_point(p0,nodes[0]);
            objectmesh->get_point(p1,nodes[1]);
            objectmesh->get_point(p2,nodes[2]);
            objectmesh->get_point(p3,nodes[3]);
            if (sdistance(p,p0,p1,p2,d))
            {
              if(abs(d) < absdist ) 
              {
                dist = d;
                absdist = abs(d);
              }
            }
            ++bt;
          }
          double s = 1.0; if (dist < 0) s = 0.0;
          ofield->set_value(s*sqrt(absdist),*(bi));
          ++bi;   
        }
        break;
      default:
        reporter->error("SignedDistanceToField: Expected a Quadrilateral or Triangular element");
        return(false);
    }
  }

  return(true);
}


template<class FIELD, class OBJECTFIELD>
bool DistanceToFieldAlgoT<FIELD,OBJECTFIELD>::execute_isinside(ProgressReporter *reporter,
                              FieldHandle input,
                              FieldHandle& output,
                              FieldHandle objectfield)
{              
  OBJECTFIELD* tfield = dynamic_cast<OBJECTFIELD* >(objectfield.get_rep());
  if (tfield == 0)
  {
    reporter->error("SignedDistanceToField: Object is not valid");
    return(false);
  }

  FIELD* field = dynamic_cast<FIELD* >(input.get_rep());
  if (field == 0)
  {
    reporter->error("SignedDistanceToField: There is no input field");
    return(false);
  }

  typename FIELD::mesh_type *mesh =
    dynamic_cast<typename FIELD::mesh_type *>(field->mesh().get_rep());

  typename OBJECTFIELD::mesh_type *objectmesh =
    dynamic_cast<typename OBJECTFIELD::mesh_type *>(objectfield->mesh().get_rep());  

  if (objectmesh->dimensionality() != 2)
  {
    reporter->error("SignedDistanceToField: This function has not been implemented for point clouds, line element fields, or volume fields,");
    return(false);
  }

  if (field->basis_order() == 0)
  {
    FIELD* ofield = scinew FIELD(mesh,0);
    ofield->resize_fdata();
    output = dynamic_cast<SCIRun::Field* >(ofield);

    typename FIELD::mesh_type::Elem::iterator bi, ei;
    mesh->begin(bi); mesh->end(ei);

    typename OBJECTFIELD::mesh_type::Elem::iterator bt, et;
    typename OBJECTFIELD::mesh_type::Node::array_type nodes;
    typename OBJECTFIELD::mesh_type::Node::array_type nodestest;
    
    Point p;
    Point p0,p1,p2,p3;    
    
    objectmesh->begin(bt);
    objectmesh->get_nodes(nodestest, *(bt));
    
    switch (nodestest.size())
    {
      case 3:
        while (bi != ei)
        {
          mesh->get_center(p,*(bi));
          objectmesh->begin(bt); objectmesh->end(et);
          
          objectmesh->get_nodes(nodes, *(bt));
          double dist = 0.0;
          double absdist = 0.0;
          double d = 0.0;

          while (bt != et)
          {
            objectmesh->get_point(p0,nodes[0]);
            objectmesh->get_point(p1,nodes[1]);
            objectmesh->get_point(p2,nodes[2]);
            if(sdistance(p,p0,p1,p2,dist)) 
            {
              absdist = abs(dist);
              break;
            }
            ++bt;
          }
          
          ++bt;
          while (bt != et)
          {
            objectmesh->get_nodes(nodes, *(bt));
            objectmesh->get_point(p0,nodes[0]);
            objectmesh->get_point(p1,nodes[1]);
            objectmesh->get_point(p2,nodes[2]);
            if (sdistance(p,p0,p1,p2,d))
            {
              if(abs(d) < absdist ) 
              {
                dist = d;
                absdist = abs(d);
              }
            }
            ++bt;
          }
          double s = 1.0; if (dist < 0) s = 0.0;
          ofield->set_value(s,*(bi));
          ++bi;   
        }
        break;
      case 4:
        while (bi != ei)
        {
          mesh->get_center(p,*(bi));
          objectmesh->begin(bt); objectmesh->end(et);
          
          objectmesh->get_nodes(nodes, *(bt));
          double dist = 0.0;
          double absdist = 0.0;
          double d = 0.0;

          while (bt != et)
          {
            objectmesh->get_point(p0,nodes[0]);
            objectmesh->get_point(p1,nodes[1]);
            objectmesh->get_point(p2,nodes[2]);
            objectmesh->get_point(p3,nodes[3]);
            if(sdistance(p,p0,p1,p2,p3,dist)) 
            {
              absdist = abs(dist);
              break;
            }
            ++bt;
          }
          
          ++bt;
          while (bt != et)
          {
            objectmesh->get_nodes(nodes, *(bt));
            objectmesh->get_point(p0,nodes[0]);
            objectmesh->get_point(p1,nodes[1]);
            objectmesh->get_point(p2,nodes[2]);
            objectmesh->get_point(p3,nodes[3]);
            if (sdistance(p,p0,p1,p2,d))
            {
              if(abs(d) < absdist ) 
              {
                dist = d;
                absdist = abs(d);
              }
            }
            ++bt;
          }
          double s = 1.0; if (dist < 0) s = 0.0;
          ofield->set_value(s,*(bi));
          ++bi;   
        }
        break;
      default:
        reporter->error("SignedDistanceToField: Expected a Quadrilateral or Triangular element");
        return(false);
    }
  }
  else
  {
    FIELD* ofield = scinew FIELD(mesh,1);
    ofield->resize_fdata();
    output = dynamic_cast<SCIRun::Field* >(ofield);

    typename FIELD::mesh_type::Node::iterator bi, ei;
    mesh->begin(bi); mesh->end(ei);

    typename OBJECTFIELD::mesh_type::Elem::iterator bt, et;
    typename OBJECTFIELD::mesh_type::Node::array_type nodes;
    typename OBJECTFIELD::mesh_type::Node::array_type nodestest;
    
    Point p;
    Point p0,p1,p2,p3;    
    
    objectmesh->begin(bt);
    objectmesh->get_nodes(nodestest, *(bt));
    
    switch (nodestest.size())
    {
      case 3:
        while (bi != ei)
        {
          mesh->get_center(p,*(bi));
          objectmesh->begin(bt); objectmesh->end(et);
          
          objectmesh->get_nodes(nodes, *(bt));
          double dist = 0.0;
          double absdist = 0.0;
          double d = 0.0;

          while (bt != et)
          {
            objectmesh->get_point(p0,nodes[0]);
            objectmesh->get_point(p1,nodes[1]);
            objectmesh->get_point(p2,nodes[2]);
            if(sdistance(p,p0,p1,p2,dist)) 
            {
              absdist = abs(dist);
              break;
            }
            ++bt;
          }
          
          ++bt;
          while (bt != et)
          {
            objectmesh->get_nodes(nodes, *(bt));
            objectmesh->get_point(p0,nodes[0]);
            objectmesh->get_point(p1,nodes[1]);
            objectmesh->get_point(p2,nodes[2]);
            if (sdistance(p,p0,p1,p2,d))
            {
              if(abs(d) < absdist ) 
              {
                dist = d;
                absdist = abs(d);
              }
            }
            ++bt;
          }
          double s = 1.0; if (dist < 0) s = 0.0;
          ofield->set_value(s,*(bi));
          ++bi;   
        }
        break;
      case 4:
        while (bi != ei)
        {
          mesh->get_center(p,*(bi));
          objectmesh->begin(bt); objectmesh->end(et);
          
          objectmesh->get_nodes(nodes, *(bt));
          double dist = 0.0;
          double absdist = 0.0;
          double d = 0.0;

          while (bt != et)
          {
            objectmesh->get_point(p0,nodes[0]);
            objectmesh->get_point(p1,nodes[1]);
            objectmesh->get_point(p2,nodes[2]);
            objectmesh->get_point(p3,nodes[3]);
            if(sdistance(p,p0,p1,p2,p3,dist)) 
            {
              absdist = abs(dist);
              break;
            }
            ++bt;
          }
          
          ++bt;
          while (bt != et)
          {
            objectmesh->get_nodes(nodes, *(bt));
            objectmesh->get_point(p0,nodes[0]);
            objectmesh->get_point(p1,nodes[1]);
            objectmesh->get_point(p2,nodes[2]);
            objectmesh->get_point(p3,nodes[3]);
            if (sdistance(p,p0,p1,p2,d))
            {
              if(abs(d) < absdist ) 
              {
                dist = d;
                absdist = abs(d);
              }
            }
            ++bt;
          }
          double s = 1.0; if (dist < 0) s = 0.0;
          ofield->set_value(s,*(bi));
          ++bi;   
        }
        break;
      default:
        reporter->error("SignedDistanceToField: Expected a Quadrilateral or Triangular element");
        return(false);
    }
  }

  return(true);
}



// Distance functions

inline void DistanceToFieldAlgo::distance(Point c,Point t1,Point t2, Point t3, double& dist)
{
  Vector p,p1; 
  Vector n = Cross(Vector(t2-t1),Vector(t3-t2));
  
  double d1 = Dot(Vector(c-t1),Cross(n,Vector(t1-t3)));
  double d2 = Dot(Vector(c-t2),Cross(n,Vector(t2-t1)));
  double d3 = Dot(Vector(c-t3),Cross(n,Vector(t3-t2)));
  
  if (d1 >= 0)
  {
    if (d2 >= 0)
    {
      if (d3 >= 0)
      {
        p = n*(Dot(Vector(c-t1),n)/Dot(n,n));
        dist = Dot(p,p);
        return;
      }
      else
      {
        d1 = Dot(Vector(c-t3),Vector(t3-t2));
        if (d1 >= 0.0)
        {
          p = Vector(c-t3);
          dist = Dot(p,p);
          return;
        }
        d1 = Dot(Vector(c-t2),Vector(t2-t3));
        if (d1 >= 0.0)
        {
          p = Vector(c-t2);
          dist = Dot(p,p);
          return;
        }
        p1 = Vector(t2-t3);
        p = Vector(c-t3)-p1*(Dot(Vector(c-t3),p1)/Dot(p1,p1));
        dist = Dot(p,p);
        return;
      }
    } 
    else
    {
      if (d3 >= 0)
      {
        d1 = Dot(Vector(c-t2),Vector(t2-t1));
        if (d1 >= 0.0)
        {
          p = Vector(c-t2);
          dist  = Dot(p,p);
          return;
        }
        d1 = Dot(Vector(c-t1),Vector(t1-t2));
        if (d1 >= 0.0)
        {
          p = Vector(c-t1);
          dist = Dot(p,p);
          return;
        }
        p1 = Vector(t1-t2);
        p = Vector(c-t2)-p1*(Dot(Vector(c-t2),p1)/Dot(p1,p1));
        dist = Dot(p,p);
        return;      
      }
      else
      {
        p = Vector(c-t2);
        dist = Dot(p,p);
        return;
      }
    }
  }
  else
  {
    if (d2 >= 0)
    {
      if (d3 >= 0)
      {
         d1 = Dot(Vector(c-t1),Vector(t1-t3));
        if (d1 >= 0.0)
        {
          p = Vector(c-t1);
          dist = Dot(p,p);
          return;
        }
        d1 = Dot(Vector(c-t3),Vector(t3-t1));
        if (d1 >= 0.0)
        {
          p = Vector(c-t3);
          dist = Dot(p,p);
          return;
        }
        p1 = Vector(t3-t1);
        p = Vector(c-t1)-(p1*(Dot(Vector(c-t1),p1)/Dot(p1,p1)));
        dist = Dot(p,p);
        return;         
      }
      else
      {
        p = Vector(c-t3);
        dist = Dot(p,p);
        return;
      }
    }
    else
    {
      if (d3 >= 0)
      {
        p = Vector(c-t1);
        dist = Dot(p,p);
        return;
      }
      else
      {
        std::cout << "ERROR: Algorithm should not be able to get here\n"; 
      }      
    }
  }
}

inline void DistanceToFieldAlgo::distance(Point c,Point t1,Point t2, Point t3, Point t4, double& dist)
{
  double d1, d2;
  distance(c,t1,t2,t3,d1);
  distance(c,t3,t4,t1,d2);
  dist = d1<d2?d1:d2;
}

inline void DistanceToFieldAlgo::distance(Point c,Point t1,Point t2, double& dist)
{
  double d1;
  Vector p,p1;
  
  d1 = Dot(Vector(c-t2),Vector(t2-t1));
  if (d1 >= 0.0)
  {
    p = Vector(c-t2);
    dist  = Dot(p,p);
    return;
  }
  d1 = Dot(Vector(c-t1),Vector(t1-t2));
  if (d1 >= 0.0)
  {
    p = Vector(c-t1);
    dist = Dot(p,p);
    return;
  }
  p1 = Vector(t1-t2);
  p = Vector(c-t2)-p1*(Dot(Vector(c-t2),p1)/Dot(p1,p1));
  dist = Dot(p,p);
  return;    
}

inline void DistanceToFieldAlgo::distance(Point c,Point t1, double& dist)
{
  dist = (Dot(Vector(c-t1),Vector(c-t1)));
}


inline bool DistanceToFieldAlgo::sdistance(Point c,Point t1,Point t2, Point t3, double& dist)
{
  Vector p,p1; 
  Vector n = Cross(Vector(t2-t1),Vector(t3-t2));
  if (Dot(n,n) < 1e-12)
  { // Element has no size, skip this one
    return(false);
  }
  
  double s = Dot(Vector(c-t1),n);
  double d1 = Dot(Vector(c-t1),Cross(n,Vector(t1-t3)));
  double d2 = Dot(Vector(c-t2),Cross(n,Vector(t2-t1)));
  double d3 = Dot(Vector(c-t3),Cross(n,Vector(t3-t2)));
  
  if (s == 0.0)
  {
    // We are in the plane of the element
    if ((d1 > -(1e-12))&&(d2 > -(1e-12))&&(d3 > -(1e-12)))
    { // We are in the element itself
      dist = 0.0;
      return(true);
    }
    else
    {
      // We are in plane and not on the element, hence we cannot determine
      // whether we are inside or outside of the volume
      // Use an adjoining element to determine the sign
      // The adjoining element must have a different inclination, or must be
      // in plane as well.
      return(false);
    }
  }
  else
  {
    if (s > 0.0) { s = 1.0; } else  { s = -1.0; }
  }
  
  if (d1 >= 0)
  {
    if (d2 >= 0)
    {
      if (d3 >= 0)
      {
        p = n*(Dot(Vector(c-t1),n)/Dot(n,n));
        dist = s*Dot(p,p);
        return(true);
      }
      else
      {
        d1 = Dot(Vector(c-t3),Vector(t3-t2));
        if (d1 >= 0.0)
        {
          p = Vector(c-t3);
          dist = s*Dot(p,p);
          return(true);
        }
        d1 = Dot(Vector(c-t2),Vector(t2-t3));
        if (d1 >= 0.0)
        {
          p = Vector(c-t2);
          dist = s*Dot(p,p);
          return(true);
        }
        p1 = Vector(t2-t3);
        p = Vector(c-t3)-p1*(Dot(Vector(c-t3),p1)/Dot(p1,p1));
        dist = s*Dot(p,p);
        return(true);
      }
    } 
    else
    {
      if (d3 >= 0)
      {
        d1 = Dot(Vector(c-t2),Vector(t2-t1));
        if (d1 >= 0.0)
        {
          p = Vector(c-t2);
          dist  = s*Dot(p,p);
          return(true);
        }
        d1 = Dot(Vector(c-t1),Vector(t1-t2));
        if (d1 >= 0.0)
        {
          p = Vector(c-t1);
          dist = s*Dot(p,p);
          return(true);
        }
        p1 = Vector(t1-t2);
        p = Vector(c-t2)-p1*(Dot(Vector(c-t2),p1)/Dot(p1,p1));
        dist = s*Dot(p,p);
        return(true);      
      }
      else
      {
        p = Vector(c-t2);
        dist = s*Dot(p,p);
        return(true);
      }
    }
  }
  else
  {
    if (d2 >= 0)
    {
      if (d3 >= 0)
      {
         d1 = Dot(Vector(c-t1),Vector(t1-t3));
        if (d1 >= 0.0)
        {
          p = Vector(c-t1);
          dist = s*Dot(p,p);
          return(true);
        }
        d1 = Dot(Vector(c-t3),Vector(t3-t1));
        if (d1 >= 0.0)
        {
          p = Vector(c-t3);
          dist = s*Dot(p,p);
          return(true);
        }
        p1 = Vector(t3-t1);
        p = Vector(c-t1)-(p1*(Dot(Vector(c-t1),p1)/Dot(p1,p1)));
        dist = s*Dot(p,p);
        return(true);   
      }
      else
      {
        p = Vector(c-t3);
        dist = s*Dot(p,p);
        return(true);
      }
    }
    else
    {
      if (d3 >= 0)
      {
        p = Vector(c-t1);
        dist = s*Dot(p,p);
        return(true);
      }
      else
      {
        std::cout << "ERROR: Algorithm should not be able to get here\n"; 
        return(false);
      }      
    }
  }
}


inline bool DistanceToFieldAlgo::sdistance(Point c,Point t1,Point t2, Point t3, Point t4, double& dist)
{
  double d1, d2;
  if(!(sdistance(c,t1,t2,t3,d1))) return(false);
  if(!(sdistance(c,t3,t4,t1,d2))) return(false);
  dist = abs(d1)<abs(d2)?d1:d2;
  return(true);
}

inline bool DistanceToFieldAlgo::sdistance(Point c,Point t1,Point t2, double& dist)
{
  // Cannot determine inside/outside for line element in 3D space
  return(false);
}

inline bool DistanceToFieldAlgo::sdistance(Point c,Point t1, double& dist)
{
  // Cannot determine inside/outside for point element in 3D space
  return(false);
}


} // namespace ModelCreation

#endif
