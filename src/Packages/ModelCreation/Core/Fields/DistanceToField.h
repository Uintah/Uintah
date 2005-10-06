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

  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc,
                                            const TypeDescription *ofsrc);

  inline double distance(Point c,Point t1,Point t2, Point t3, Point t4);                                            
  inline double distance(Point c,Point t1,Point t2, Point t3);
  inline double distance(Point c,Point t1,Point t2);
  inline double distance(Point c,Point t1);                                            
};


template<class FIELD, class OBJECTFIELD>
class DistanceToFieldAlgoT : public DistanceToFieldAlgo
{
public:
  virtual bool execute(ProgressReporter *reporter,
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

  if (field->basis_order() == 0)
  {
    FIELD* ofield = scinew FIELD(mesh,0);
    ofield->resize_fdata();
    output = dynamic_cast<SCIRun::Field* >(ofield);

    typename FIELD::mesh_type::Elem::iterator bi, ei;
    mesh->begin(bi); mesh->end(ei);

    typename OBJECTFIELD::mesh_type::Elem::iterator bt, et;

    Point p;
    Point p0,p1,p2,p3;    
    while (bi != ei)
    {
      mesh->get_center(p,*(bi));

      objectmesh->begin(bt); objectmesh->end(et);
      typename OBJECTFIELD::mesh_type::Node::array_type nodes;
      objectmesh->get_nodes(nodes, *(bt));
      double dist = 0.0;
      switch(nodes.size())
      {
        case 1: 
          objectmesh->get_point(p0,nodes[0]);
          dist = distance(p,p0);
          break;
        case 2: 
          objectmesh->get_point(p0,nodes[0]);          
          objectmesh->get_point(p1,nodes[1]);          
          dist = distance(p,p0,p1);
          break;
        case 3: 
          objectmesh->get_point(p0,nodes[0]);          
          objectmesh->get_point(p1,nodes[1]);
          objectmesh->get_point(p2,nodes[2]);                  
          dist = distance(p,p0,p1,p2);
          break;
        case 4: 
          objectmesh->get_point(p0,nodes[0]);          
          objectmesh->get_point(p1,nodes[1]);
          objectmesh->get_point(p2,nodes[2]);                  
          objectmesh->get_point(p3,nodes[3]);          
          dist = distance(p,p0,p1,p2,p3);  
          break;
      }
      ++bt;
      double d = 0.0;
      while (bt != et)
      {
        objectmesh->get_nodes(nodes, *(bt));
        switch(nodes.size())
        {
        case 1: 
          objectmesh->get_point(p0,nodes[0]);
          d = distance(p,p0);
          break;
        case 2: 
          objectmesh->get_point(p0,nodes[0]);          
          objectmesh->get_point(p1,nodes[1]);          
          d = distance(p,p0,p1);
          break;
        case 3: 
          objectmesh->get_point(p0,nodes[0]);          
          objectmesh->get_point(p1,nodes[1]);
          objectmesh->get_point(p2,nodes[2]);                  
          d = distance(p,p0,p1,p2);
          break;
        case 4: 
          objectmesh->get_point(p0,nodes[0]);          
          objectmesh->get_point(p1,nodes[1]);
          objectmesh->get_point(p2,nodes[2]);                  
          objectmesh->get_point(p3,nodes[3]);          
          d = distance(p,p0,p1,p2,p3);  
          break; 
        }
        if(d < dist ) dist = d;
        ++bt;
      }
      ofield->set_value(sqrt(dist),*(bi));
     ++bi;   
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

    // Somehow someone did forget to program some simple functions to get access
    // to the nodes of an element .........
    Point p;
    Point p0,p1,p2,p3;

    while (bi != ei)
    {
      mesh->get_center(p,*(bi));

      objectmesh->begin(bt); objectmesh->end(et);
      typename OBJECTFIELD::mesh_type::Node::array_type nodes;
      objectmesh->get_nodes(nodes, *(bt));
      double dist = 0.0;
      switch(nodes.size())
      {
        case 1: 
          objectmesh->get_point(p0,nodes[0]);
          dist = distance(p,p0);
          break;
        case 2: 
          objectmesh->get_point(p0,nodes[0]);          
          objectmesh->get_point(p1,nodes[1]);          
          dist = distance(p,p0,p1);
          break;
        case 3: 
          objectmesh->get_point(p0,nodes[0]);          
          objectmesh->get_point(p1,nodes[1]);
          objectmesh->get_point(p2,nodes[2]);                  
          dist = distance(p,p0,p1,p2);
          break;
        case 4: 
          objectmesh->get_point(p0,nodes[0]);          
          objectmesh->get_point(p1,nodes[1]);
          objectmesh->get_point(p2,nodes[2]);                  
          objectmesh->get_point(p3,nodes[3]);          
          dist = distance(p,p0,p1,p2,p3);  
          break;
      }      
      ++bt;
      double d = 0.0;
      while (bt != et)
      {
        objectmesh->get_nodes(nodes, *(bt));
        switch(nodes.size())
        {
        case 1: 
          objectmesh->get_point(p0,nodes[0]);
          d = distance(p,p0);
          break;
        case 2: 
          objectmesh->get_point(p0,nodes[0]);          
          objectmesh->get_point(p1,nodes[1]);          
          d = distance(p,p0,p1);
          break;
        case 3: 
          objectmesh->get_point(p0,nodes[0]);          
          objectmesh->get_point(p1,nodes[1]);
          objectmesh->get_point(p2,nodes[2]);                  
          d = distance(p,p0,p1,p2);
          break;
        case 4: 
          objectmesh->get_point(p0,nodes[0]);          
          objectmesh->get_point(p1,nodes[1]);
          objectmesh->get_point(p2,nodes[2]);                  
          objectmesh->get_point(p3,nodes[3]);          
          d = distance(p,p0,p1,p2,p3);  
          break;  
        }
        if(d < dist) dist = d;
        ++bt;
      }
      ofield->set_value(sqrt(dist),*(bi));
     ++bi;   
    }
  }

  return(true);
}

// DUE TO SOME ODD REASON SOMEONE DECIDED TO HAVE A VECTOR AND POINT CLASS THAT
// ARE ALMOST EQUAL BUT NOT FULLY. HENCE AS A RESULT ONE CANNOT ADD POINTS.
// THE VECTOR CLASS AND POINT CLASS NEED TO BE MERGED, WHEN THE PIO SYSTEM IS UP
// TO THIS THESE FUNCTIONS MAY BECOME OBSOLETE!!!

inline Point add(Point p1, Point p2)
{
  Point p(p1.x()+p2.x(),p1.y()+p2.y(),p1.z()+p2.z());
  return(p);
}

inline Point sub(Point p1, Point p2)
{
  Point p(p1.x()-p2.x(),p1.y()-p2.y(),p1.z()-p2.z());
  return(p);
}

// Some one did not implement this obvious function hence we add it here
inline Point cross(Point p1,Point p2)
{
  Point p(p1.y()*p2.z()-p1.z()*p2.y(),p1.z()*p2.x()-p1.x()*p2.z(),p1.x()*p2.y()-p1.y()*p2.x());
  return(p);
}

inline double DistanceToFieldAlgo::distance(Point c,Point t1,Point t2, Point t3)
{
  Point n = cross(sub(t2,t1),sub(t3,t2));
  Point p, p1;
  double d1 = Dot(sub(c,t1),cross(n,sub(t1,t3)));
  double d2 = Dot(sub(c,t2),cross(n,sub(t2,t1)));
  double d3 = Dot(sub(c,t3),cross(n,sub(t3,t2)));

  if (d1 >= 0)
  {
    if (d2 >= 0)
    {
      if (d3 >= 0)
      {
        p = n*(Dot(sub(c,t1),n)/Dot(n,n));
        return(Dot(p,p));
      }
      else
      {
        d1 = Dot(sub(c,t3),(t3,t2));
        if (d1 >= 0.0)
        {
          p = sub(c,t3);
          return(Dot(p,p));
        }
        d1 = Dot(sub(c,t2),(t2,t3));
        if (d1 >= 0.0)
        {
          p = sub(c,t2);
          return(Dot(p,p));
        }
        p1 = sub(t2,t3);
        p = sub(sub(c,t3),p1*(Dot(sub(c,t3),p1)/Dot(p1,p1)));
        return(Dot(p,p));
      }
    } 
    else
    {
      if (d3 >= 0)
      {
        d1 = Dot(sub(c,t2),(t2,t1));
        if (d1 >= 0.0)
        {
          p = sub(c,t2);
          return(Dot(p,p));
        }
        d1 = Dot(sub(c,t1),(t1,t2));
        if (d1 >= 0.0)
        {
          p = sub(c,t1);
          return(Dot(p,p));
        }
        p1 = sub(t1,t2);
        p = sub(sub(c,t2),p1*(Dot(sub(c,t2),p1)/Dot(p1,p1)));
        return(Dot(p,p));      
      }
      else
      {
        p = sub(c,t2);
        return(Dot(p,p));
      }
    }
  }
  else
  {
    if (d2 >= 0)
    {
      if (d3 >= 0)
      {
         d1 = Dot(sub(c,t1),(t1,t3));
        if (d1 >= 0.0)
        {
          p = sub(c,t1);
          return(Dot(p,p));
        }
        d1 = Dot(sub(c,t3),(t3,t1));
        if (d1 >= 0.0)
        {
          p = sub(c,t3);
          return(Dot(p,p));
        }
        p1 = sub(t3,t1);
        p = sub(sub(c,t1),p1*(Dot(sub(c,t1),p1)/Dot(p1,p1)));
        return(Dot(p,p));         
      }
      else
      {
        p = sub(c,t3);
        return(Dot(p,p));
      }
    }
    else
    {
      if (d3 >= 0)
      {
        p = sub(c,t1);
        return(Dot(p,p));
      }
      else
      {
        std::cout << "ERROR: Algorithm should not be able to get here\n"; 
      }      
    }
  }
}

inline double DistanceToFieldAlgo::distance(Point c,Point t1,Point t2, Point t3, Point t4)
{
  double d1 = distance(c,t1,t2,t3);
  double d2 = distance(c,t3,t4,t1);
  return(d1<d2?d1:d2);
}

inline double DistanceToFieldAlgo::distance(Point c,Point t1,Point t2)
{
  double d = Dot(sub(c,t1),sub(t2,t1));
  if (d < 0.0) return(Dot(sub(c,t1),sub(c,t1)));
  if (d > 1.0) return(Dot(sub(c,t2),sub(c,t2)));
  Point p = sub(c,add(t1,sub(t2,t1)*d));
  return(Dot(p,p));
}

inline double DistanceToFieldAlgo::distance(Point c,Point t1)
{
  return(Dot(sub(c,t1),sub(c,t1)));
}

} // namespace ModelCreation

#endif
