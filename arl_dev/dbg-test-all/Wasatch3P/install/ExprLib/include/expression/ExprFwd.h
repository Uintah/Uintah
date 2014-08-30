/*
 * \file ExprFwd.h
 *
 *  Created on: Aug 2, 2012
 *      Author: "James C. Sutherland"
 *
 * Copyright (c) 2012 The University of Utah
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

#ifndef EXPRFWD_H_
#define EXPRFWD_H_

#include <map>
#include <vector>
#include <set>
#include <iosfwd>

#include <boost/shared_ptr.hpp>

#include <expression/Context.h>

namespace SpatialOps{
  class OperatorDatabase;
}

namespace Expr{

  typedef size_t FieldID;

  // forward declarations
  template<typename T> struct FieldMgrSelector;

  template<typename T> class Expression;

  class FieldManagerList;
  class FieldManagerBase;
  class FieldDeps;
  class ExprDeps;
  class ExpressionBase;
  class ExpressionBuilder;
  class ExpressionTree;
  class ExpressionFactory;
  class Tag;
  class TransportEquation;

  class Poller;
  class PollWorker;
  class NonBlockingPoller;
  typedef boost::shared_ptr<PollWorker>         PollWorkerPtr;
  typedef boost::shared_ptr<Poller>             PollerPtr;
  typedef std::set<PollerPtr>                   PollerList;
  typedef boost::shared_ptr<NonBlockingPoller>  NonBlockingPollerPtr;
  typedef std::set<NonBlockingPollerPtr>        NonBlockingPollerList;


  /**
   * \typedef FMLMap
   * \brief Provides a mechanism to support multiple FieldManagerList in the graph.
   */
  typedef std::map< int, FieldManagerList* > FMLMap;

  /**
   * \typedef OpDBMap
   * \brief Provides a mechanism to support multiple OperatorDatabase in the graph.
   */
  typedef std::map< int, const SpatialOps::OperatorDatabase* > OpDBMap;

  typedef std::vector<Tag> TagList;  ///< defines a vector of Tag objects
  typedef std::set   <Tag> TagSet;   ///< defines a set of Tag objects

  std::ostream& operator<<( std::ostream& os, const Tag& );
  std::ostream& operator<<( std::ostream& os, const TagList& );
  std::ostream& operator<<( std::ostream& os, const TagSet& );

  std::ostream& operator<<( std::ostream& out, const FieldDeps& fd );

}


#endif /* EXPRFWD_H_ */
