//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2006 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  
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
//    File   : Layout.h
//    Author : McKay Davis
//    Date   : Tue Jun 27 13:00:49 2006
#ifndef SKINNER_LAYOUT_H
#define SKINNER_LAYOUT_H

#include <Core/Skinner/Parent.h>
#include <list>
#include <map>
#include <string>
using std::list;
using std::map;
using std::string;

namespace SCIRun {
  class DenseMatrix;
  namespace Skinner {
    
    class Anchor {
    public:
      enum Type_e {
	SOUTH = 1,
	NORTH = 2,
	WEST  = 4,
        EAST  = 8
      };

      Anchor(const Drawable *, unsigned int side);
      virtual ~Anchor();
      const Drawable *            object() const;
      unsigned int              type() const;
      string                    type_name() const;
      static bool               pair_valid(unsigned int, unsigned int);
      static unsigned int       opposite_type(unsigned int );
      unsigned int              opposite_type() const;
      Anchor                    opposite() const;
      
      void                      set_object(const Drawable *);
    private:
      const Drawable *            object_;
      unsigned int              type_;
    };
    
    struct lt_Anchor
    {
      bool operator()(const Anchor *c1, const Anchor *c2) const
      {
        if (c1->type() != c2->type())
          return c1->type() < c2->type();
        unsigned long ptr1 = (unsigned long)(c1->object());
        unsigned long ptr2 = (unsigned long)(c2->object());
        return ptr1 < ptr2;
      }
      bool operator()(const Anchor &c1, const Anchor &c2) const
      {
        if (c1.type() != c2.type())
          return c1.type() < c2.type();
        unsigned long ptr1 = (unsigned long)(c1.object());
        unsigned long ptr2 = (unsigned long)(c2.object());
        return ptr1 < ptr2;
      }

    };


    typedef pair<Anchor, Anchor> AnchorPair;
    class Link {
    public:
      enum State_e {
	SPRING = 0x01,
	STRUT  = 0x02,
	SPRING_AND_STRUT = 0x03,
	NATURAL = 0x06,
	NATURAL_SPRING = 0x07
      };

      enum Type_e {
	VERTICAL = 3,
	HORIZONTAL = 12,
      };

      Link();

      Link(const Link &link);

      Link(const Drawable *, unsigned int,
           const Drawable *, unsigned int, 
           double min = AIR_NEG_INF, double max = AIR_POS_INF);
      
      Link(const Anchor &anchor, const MinMax &minmax);

      virtual ~Link();

      Link                      operator+(const Link &right);
      Link &                    operator+=(const Link &right);
      static Link &             Max(Link &, Link &);

      Anchor &                  master();
      const Anchor &            slave();
      
      void                      set(MinMax minmax = SPRING_MINMAX);
      MinMax                    get() const;
      unsigned int              type() const;

      void                      print();
      
      bool                      spring_p();
      double                    strut();

    private:
      AnchorPair                anchors_;
      MinMax                    minmax_;
      int                       spring_count_;
    };


    typedef list<Link> LinkList;
    typedef list<Drawable *>                      DrawableList;
    typedef map<Anchor, Link, lt_Anchor>        AnchorLinkMap;
    typedef pair<Link, Link>                    LinkPair;
    typedef map<Anchor, LinkPair, lt_Anchor>    AnchorLinkPairMap;
    typedef map<const Drawable *, double>         DrawableDoubleMap;

    class Layout : public Parent {
    public:
      Layout(Variables *);
      virtual ~Layout();
      static string                     class_name() { return "Layout"; }
      virtual propagation_state_e       process_event(event_handle_t);
      static DrawableMakerFunc_t        maker;

      virtual MinMax                    get_minmax(unsigned int);
      
      Link &                            link(Drawable *, unsigned int);
      Link &                            link(const Anchor &);
      void                              spring(Drawable *, unsigned int, 
                                               Drawable *);
      bool                              add(Drawable *);
      bool                              add_under(Drawable *, 
                                                  Drawable *under = 0);
      bool                              add_over(Drawable *, 
                                                 Drawable *over = 0);
      //      Drawable *                        get_child(const string &id);
      void                              set_children(const Drawables_t &);

    protected:
      void                      create_default_links(Drawable *);
      void                      compute_link_graph(unsigned int,
                                                   AnchorLinkPairMap &);
      void                      compute_spring_matrix(unsigned int,
                                                      DenseMatrix &,
                                                      DrawableDoubleMap &,
                                                      AnchorLinkPairMap &);
      void                      compute_spring_lengths(double,
                                                       DenseMatrix &,
                                                       DrawableDoubleMap &,
                                                       DrawableDoubleMap &);
      double                    compute_anchor_offset(Anchor,
                                                      DrawableDoubleMap &,
                                                      AnchorLinkPairMap &);

      

      void                      solve_springs(unsigned int,
                                              AnchorLinkPairMap &,
                                              DrawableDoubleMap &);
    private:
      DrawableList		objects_;
      AnchorLinkMap             links_;
      //      DrawableDoubleMap           h_springs_;
      //      DrawableDoubleMap           v_springs_;

    };
  }
}

#endif
