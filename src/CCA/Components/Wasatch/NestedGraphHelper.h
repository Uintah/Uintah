/*
 * The MIT License
 *
 * Copyright (c) 2019 The University of Utah
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

/**
 *  \file   NestedGraphHelper.h
 *  \date   Jan 24, 2019
 *  \author josh McConnell
 */

#ifndef SRC_CCA_COMPONENTS_WASATCH_EXPRESSIONS_NESTEDGRAPHHELPER_H_
#define SRC_CCA_COMPONENTS_WASATCH_EXPRESSIONS_NESTEDGRAPHHELPER_H_

#include <expression/ExprLib.h>

// forward decalrations -----
struct AllocInfo;

namespace Uintah{ class Patch; }

namespace WasatchCore{

  class NestedGraphHelper
  {
    typedef std::set<Expr::ExpressionTree*> TreeSet;

    private:
      TreeSet    treeList_;
      AllocInfo* allocInfo_;

    public:
      Expr::ExpressionFactory* const factory_;
      Expr::FieldManagerList*        fml_;

      NestedGraphHelper();
      ~NestedGraphHelper();


      /*
       * brief  creates  a new ExpressionTree* given a name for the tree and rootIds and inserts
       */
      template <typename IDType>
      Expr::ExpressionTree*
      new_tree(const std::string treeName,
                    const IDType& rootIDs);

      /*
       * \brief defines member allocInfo based info from Uintah::Patch
       */
      void
      set_alloc_info(const Uintah::Patch* const patch,
                     const bool isGPU );

      /*
       * \brief Sets the memory manager of the fields owned by treePtr to SpatialOps' memory manager
       */
      void
      set_memory_manager_to_dynamic( Expr::ExpressionTree* treePtr );

      /*
       * \brief allocates, registers and binds fields
       */
      void
      finalize();
  };
}

#endif /* SRC_CCA_COMPONENTS_WASATCH_EXPRESSIONS_NESTEDGRAPHHELPER_H_ */
