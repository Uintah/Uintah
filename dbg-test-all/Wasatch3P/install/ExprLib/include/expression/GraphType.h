/**
 *  \file   GraphType.h
 *  \date   Dec 16, 2013
 *  \author "James C. Sutherland"
 *
 *
 * The MIT License
 *
 * Copyright (c) 2013 The University of Utah
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
 *
 */
#ifndef GRAPHTYPE_H_
#define GRAPHTYPE_H_

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>

#include <expression/VertexProperty.h>

namespace Expr{

  /** Boost graph related things */

  // define a directed graph with only out-edge traversal.  We could
  // get bidirectional traversal easily if needed...
  typedef boost::adjacency_list< boost::listS,
                                 boost::listS,
                                 boost::directedS,
                                 VertexProperty,
                                 boost::no_property,  // no edge properties
                                 boost::no_property   // no graph properties
                                 >  Graph;  ///< Defines the type of graph

  typedef boost::graph_traits<Graph>::vertex_descriptor Vertex;   ///< Vertex in a graph
  typedef boost::graph_traits<Graph>::vertex_iterator   VertIter; ///< Vertex iterator
  typedef boost::graph_traits<Graph>::edge_descriptor   Edge;     ///< Edge in a graph
  typedef boost::graph_traits<Graph>::edge_iterator EdgeIter; ///< Edge iterator
  typedef boost::graph_traits<Graph>::out_edge_iterator OutEdgeIter;

}


#endif /* GRAPHTYPE_H_ */
