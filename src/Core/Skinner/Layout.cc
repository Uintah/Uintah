//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2006 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//  
//    File   : Layout.cc
//    Author : McKay Davis
//    Date   : Tue Jun 27 13:00:55 2006
#include <Core/Skinner/Variables.h>
#include <Core/Skinner/Layout.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Math/MinMax.h>
#include <iostream>

namespace SCIRun {
  namespace Skinner {
    
    Anchor::Anchor(const Drawable *object, unsigned int type)
      : object_(object),
	type_(type)
    {
    }

    Anchor::~Anchor()
    {
    }

    const Drawable * 
    Anchor::object() const
    {
      return object_;
    }

    unsigned int
    Anchor::type() const
    {
      return type_;
    }

    string
    Anchor::type_name() const
    {
      switch (type_) {
      case NORTH: return "NORTH"; break;
      case SOUTH: return "SOUTH"; break;
      case EAST: return "EAST"; break;
      case WEST: return "WEST"; break;
      default: break;
      }
      return ("INVALID TYPE #" + to_string(type_));
    }

    bool
    Anchor::pair_valid(unsigned int s1, unsigned int s2) {
      bool s1Horiz = ((s1 & Link::HORIZONTAL) != 0);
      bool s2Horiz = ((s2 & Link::HORIZONTAL) != 0);
      return (s1Horiz == s2Horiz);
      
//    return (((s1 == NORTH || s1 == SOUTH) && (s2 == NORTH || s2 == SOUTH)) ||
//            ((s1 == EAST || s1 == WEST) && (s2 == EAST || s2 == WEST)));
    }

    unsigned int
    Anchor::opposite_type(unsigned int intype) {
      switch (intype) {
      case NORTH: return SOUTH; break;
      case SOUTH: return NORTH; break;
      case EAST: return WEST; break;
      case WEST: return EAST; break;
      default: break;
      }
      return 0;
    }

    unsigned int
    Anchor::opposite_type() const {
      return opposite_type(type());
    }

    Anchor
    Anchor::opposite() const {
      return Anchor(object(), opposite_type());
    }

    void
    Anchor::set_object(const Drawable * obj) {
      object_ = obj;
    }

    Link::Link() 
      : anchors_(Anchor(0,0), Anchor(0,0)),
	minmax_(SPRING_MINMAX),
        spring_count_(0)
    {
    }

    Link::Link(const Link &link) 
      : anchors_(link.anchors_.first, link.anchors_.second),
	minmax_(link.minmax_),
        spring_count_(link.spring_count_)
    {
    }

    Link::Link(const Drawable *obj1, unsigned int side1,
               const Drawable *obj2, unsigned int side2,
               double min, double max) 
      : anchors_(Anchor(obj1, side1), Anchor(obj2, side2))
    {
      if (!Anchor::pair_valid(side1, side2))
	throw "Link anchors are not compatible";
      set(make_pair(min,max));
    }


    Link::Link(const Anchor &anchor, const MinMax &minmax)
      : anchors_(anchor, anchor.opposite())
    {
      set(minmax);
    }
    
    Link::~Link() 
    {
    }
    
    Link & 
    Link::operator+=(const Link &right) {
      anchors_.second.set_object(right.anchors_.second.object());
      spring_count_ += right.spring_count_;
      minmax_.first += right.minmax_.first;
      minmax_.second += right.minmax_.second;
      return *this;
    }

    void
    Link::print() {
      cerr << "Link obj1: "  << anchors_.first.object()->get_id() 
           << " = " << anchors_.first.type_name()
           << " obj2: "  << anchors_.second.object()->get_id() 
           << " = " << anchors_.second.type_name()
           << " min: " << minmax_.first << " max: " << minmax_.second 
           << " springcount: " << spring_count_
           << std::endl;
    }

    // Untested
    Link &
    Link::Max(Link &a, Link &b) {
      return (a.minmax_.first > b.minmax_.first) ? a : b;
    }

    Anchor &
    Link::master() {
      return anchors_.second;
    }

    const Anchor &
    Link::slave() {
      return anchors_.first;
    }

    void
    Link::set(MinMax minmax)
    {
      minmax_ = minmax;
      if (minmax_.first != minmax_.second) 
        spring_count_ = 1;
      else
        spring_count_ = 0;

    }
    
    MinMax
    Link::get() const
    {
      return minmax_;
    }

    unsigned int 
    Link::type() const
    {
      bool s1Horiz = ((anchors_.first.type() & HORIZONTAL) != 0);
      bool s2Horiz = ((anchors_.second.type() & HORIZONTAL) != 0);
      if (s1Horiz != s2Horiz)
        throw "Link type() error";

      return s1Horiz ? HORIZONTAL : VERTICAL;
    }


    bool
    Link::spring_p () {
      return (Abs(minmax_.first - minmax_.second) > 0.001);
    }

    double
    Link::strut () {
      return minmax_.first;//SCIRun::Max(0.0, double(minmax_.first));
    }


    Layout::Layout(Variables *variables) :
      Parent(variables),
      objects_(),
      links_()
    {
    }

    Layout::~Layout() {
    }
    
    void
    Layout::compute_link_graph(unsigned int ltype,
                                  AnchorLinkPairMap &graph) {
      DrawableList::iterator oiter = objects_.begin();
      DrawableList::iterator olast = objects_.end();      
      for(;oiter != olast; ++oiter) {
        Drawable *obj = *oiter;
        MinMax minmax = obj->get_minmax(ltype);
        for (int pass = 0; pass < 2; ++pass) {
          unsigned int atype = 
            (ltype == Link::HORIZONTAL) ? Anchor::WEST : Anchor::SOUTH;
          if (pass) atype *= 2; // Turns West->East and South->North

          Anchor anchor(obj, atype);
          if (!pass) {
            graph[anchor] = make_pair(links_[anchor], Link(anchor, minmax));
            ASSERT(graph[anchor].first.master().object() != obj);
            ASSERT(graph[anchor].second.master().object() == obj);
          } else {
            graph[anchor] = make_pair(Link(anchor, minmax), links_[anchor]);
            ASSERT(graph[anchor].first.master().object() == obj);
            ASSERT(graph[anchor].second.master().object() != obj);
          }

          ASSERT(graph[anchor].first.slave().object() == obj);
          ASSERT(graph[anchor].second.slave().object() == obj);

        }
      }
    }

    void
    Layout::compute_spring_matrix(unsigned int ltype,
                                     DenseMatrix &matrix,
                                     DrawableDoubleMap &struts,
                                     AnchorLinkPairMap &graph) 
    {
      unsigned int atype = 
        ltype == Link::HORIZONTAL ? Anchor::WEST : Anchor::SOUTH;
      unsigned int num = objects_.size();
      vector<Drawable *> objvec;
      map<const Drawable *, int> objid;
      objvec.reserve(num);
      DrawableList::iterator oiter = objects_.begin();
      DrawableList::iterator olast = objects_.end();      
      for(;oiter != olast; ++oiter) {
        objid[*oiter] = objvec.size();
        objvec.push_back(*oiter);
      }

      struts.clear();
      matrix = DenseMatrix(num, num);
      matrix.zero();
      for (unsigned int row = 0; row < num; ++row) {
        Drawable *obj = objvec[row];
        for (unsigned int pass = 0; pass < 2; ++pass) {
          Anchor anchor = Anchor(obj, atype);
          while (anchor.object() != this) {
            Link &link = pass ? graph[anchor].second : graph[anchor].first; 
            const Drawable *linkobj = link.slave().object();
            if (link.spring_p()) {
              int col = objid[linkobj];
              matrix.add(row, col, 1.0);
            }
            struts[linkobj] = struts[linkobj]+link.strut();
            anchor = link.master();
          }
        }
      }
    }


    void
    Layout::compute_spring_lengths(double span,
                                      DenseMatrix &matrix,
                                      DrawableDoubleMap &struts,
                                      DrawableDoubleMap &springs) 
    {
      unsigned int num = matrix.ncols();
      vector<double> springval(num, span);
      unsigned int i = 0;
      DrawableList::iterator oiter = objects_.begin();
      DrawableList::iterator olast = objects_.end();      
      for(;oiter != olast; ++oiter)
        springval[i++] -= struts[*oiter];


      int valid = matrix.solve(springval, true);
      ASSERT(valid);
      
      //      for (unsigned int n = 0; n < num; ++n)
        //         cerr << n << " = " << springval[n] << std::endl;

      springs.clear();
      i = 0;
      oiter = objects_.begin();
      olast = objects_.end();      
      for(;oiter != olast; ++oiter) {
        const Drawable* obj = *oiter;
        springs[obj] = springval[i++];
      }      
    }

    double
    Layout::compute_anchor_offset(Anchor anchor,
                                     DrawableDoubleMap &springs,
                                     AnchorLinkPairMap &graph) 
    {
      double val = 0;
      while (anchor.object() != this) {
        Link &link = graph[anchor].first;
        if (link.spring_p()) 
          val += springs[link.slave().object()];
        val += link.strut();
        anchor = link.master();
      }
      return val;
    }


    // Untested
    MinMax
    Layout::get_minmax(unsigned int ltype) {
      AnchorLinkPairMap graph;
      compute_link_graph(ltype, graph);

      DenseMatrix spring_matrix;
      DrawableDoubleMap struts;
      compute_spring_matrix(ltype, spring_matrix, struts, graph);
      DrawableList::iterator oiter = objects_.begin();
      DrawableList::iterator olast = objects_.end();
      double min = AIR_NEG_INF;
      for(;oiter != olast; ++oiter)
        min = Max(min, struts[*oiter]);

      return make_pair(min, AIR_POS_INF);
      
    }

    BaseTool::propagation_state_e
    Layout::process_event(event_handle_t event) {

      AnchorLinkPairMap graph;
      compute_link_graph(Link::HORIZONTAL, graph);
      compute_link_graph(Link::VERTICAL, graph);
      
      DenseMatrix h_spring_matrix;
      DenseMatrix v_spring_matrix;
      DrawableDoubleMap h_struts;
      DrawableDoubleMap v_struts;
      
      compute_spring_matrix(Link::HORIZONTAL, h_spring_matrix, h_struts, graph);
      compute_spring_matrix(Link::VERTICAL, v_spring_matrix, v_struts, graph);
      
      DrawableDoubleMap h_springs;
      DrawableDoubleMap v_springs;
      const RectRegion &region = get_region();
      compute_spring_lengths(region.width(), h_spring_matrix, 
                             h_struts, h_springs);
      compute_spring_lengths(region.height(), v_spring_matrix, 
                             v_struts, v_springs);
      
      DrawableList::iterator oiter = objects_.begin();
      DrawableList::iterator olast = objects_.end();
      double x1 = region.x1();
      double y1 = region.y1();
      for(;oiter != olast; ++oiter) {
        Drawable *obj = *oiter;
        const RectRegion subregion
          (x1+compute_anchor_offset(Anchor(obj,Anchor::WEST),h_springs,graph),
           y1+compute_anchor_offset(Anchor(obj,Anchor::SOUTH),v_springs,graph),
           x1+compute_anchor_offset(Anchor(obj,Anchor::EAST),h_springs,graph),
           y1+compute_anchor_offset(Anchor(obj,Anchor::NORTH),v_springs,graph));
        obj->set_region(subregion);
        obj->process_event(event);
      }
      return CONTINUE_E;
    }


    void
    Layout::spring(Drawable *obj1, unsigned int type, Drawable *obj2) {
      Anchor a1(obj1, type);
      Anchor a2 = Anchor(obj2, type).opposite();
      Link &l1 = link(a1);
      Link &l2 = link(a2);
      l1.master() = a2;
      l2.master() = a1;
      l1.set(SPRING_MINMAX);
      l2.set(SPRING_MINMAX);
    }
      
    Link &
    Layout::link(Drawable *obj, unsigned int type) {
      Anchor a(obj, type);
      if (links_.find(a) == links_.end()) 
        throw "Link not found";
      return links_[a];
    }

    Link &
    Layout::link(const Anchor &a) {
      if (links_.find(a) == links_.end()) 
        throw "Link not found";
      return links_[a];
    }

    bool
    Layout::add(Drawable *obj) {
      return add_over(obj);
    }


    bool
    Layout::add_over(Drawable *obj, Drawable *under) {
      if (!under)
	objects_.push_back(obj);
      else {
	list<Drawable *>::iterator pos = objects_.begin();
	while (pos != objects_.end() && *pos != under) ++pos;
	if (pos == objects_.end())
	  return false;
	objects_.insert(pos, obj);
      }
      create_default_links(obj);
      return true;
    }


    bool
    Layout::add_under(Drawable *obj, Drawable *over) {
      if (!over)
	objects_.push_front(obj);
      else {
	list<Drawable *>::iterator pos = objects_.begin();
	while (pos != objects_.end() && *pos != over) ++pos;
	if (pos == objects_.end())
	  return false;
	objects_.insert(++pos, obj);

      }
      create_default_links(obj);
      return true;
    }

//     Drawable *
//     Layout::get_child(const string &id) {
//       for (DrawableList::iterator oiter = objects_.begin(); 
//            oiter != objects_.end(); ++oiter) {
//         if ((*oiter)->get_id() == get_id()+":"+id)
//           return *oiter;
//       }
//       return 0;
//     }

    void
    Layout::set_children(const Drawables_t &children) {
      children_ = children;
      for (Drawables_t::const_iterator iter = children.begin();
           iter != children.end(); ++iter) {
        add_over(*iter);
      }
    }


    void
    Layout::create_default_links(Drawable *obj) {
      unsigned int dirs[4] = {Anchor::SOUTH, Anchor::NORTH, 
			      Anchor::WEST, Anchor::EAST};
      for (unsigned int i = 0; i < 4; ++i) {
	Link dirlink(obj, dirs[i], this, dirs[i], 0);
	links_[dirlink.slave()] = dirlink;
      }
    }


    Drawable *
    Layout::maker(Variables *vars)
    {
      return new Layout(vars);
    }                 
  }
}
