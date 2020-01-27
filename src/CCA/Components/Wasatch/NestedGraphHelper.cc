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
 *  \file   NestedGraphHelper.cc
 *  \date   Jan 24, 2019
 *  \author Josh McConnell
 */

#include <CCA/Components/Wasatch/NestedGraphHelper.h>
#include <CCA/Components/Wasatch/PatchInfo.h>
#include <CCA/Components/Wasatch/FieldAdaptor.h>

#include <expression/ManagerTypes.h>

namespace WasatchCore
{

  NestedGraphHelper::
  NestedGraphHelper()
  :
  treeList_  (),
  allocInfo_ ( nullptr                       ),
  factory_   ( new Expr::ExpressionFactory() ),
  fml_       ( new Expr::FieldManagerList()  )
  {}

//--------------------------------------------------------------------

  NestedGraphHelper::
  ~NestedGraphHelper()
  {
    for( Expr::ExpressionTree* tree : treeList_){
      if( tree ) delete tree;
    }
    if( allocInfo_ ) delete allocInfo_;

    delete fml_;
    delete factory_;
  }

//--------------------------------------------------------------------

  void
  NestedGraphHelper::
  set_alloc_info(const Uintah::Patch* const patch,
                 const bool isGPU )
  {
    // this may need to change for GPU builds
    allocInfo_ = new AllocInfo( nullptr,
                                nullptr,
                                0,
                                patch,
                                nullptr,
                                nullptr,
                                isGPU );
  }

//--------------------------------------------------------------------

  template<>
  Expr::ExpressionTree*
  NestedGraphHelper::
  new_tree(const std::string treeName, const Expr::ExpressionID& rootID)
  {
    assert(allocInfo_!=nullptr);

    Expr::ExpressionTree*
    newTree = new Expr::ExpressionTree( rootID,
                                        *factory_,
                                        allocInfo_->patch->getID(),
                                        treeName );
    treeList_.insert(newTree);

    return newTree;
}

//--------------------------------------------------------------------

  template<>
  Expr::ExpressionTree*
  NestedGraphHelper::
  new_tree(const std::string treeName, const Expr::IDSet& rootIDs)
  {
    assert(allocInfo_!=nullptr);

    Expr::ExpressionTree*
    newTree = new Expr::ExpressionTree( rootIDs,
                                        *factory_,
                                        allocInfo_->patch->getID(),
                                        treeName );
    treeList_.insert(newTree);

    return newTree;
}

//--------------------------------------------------------------------

  void
  NestedGraphHelper::
  set_memory_manager_to_dynamic( Expr::ExpressionTree* treePtr )
  {
    assert(allocInfo_ != nullptr);

    const auto fieldMap = treePtr->field_map();
    for( const auto& fieldMapPair : fieldMap ){
      const Expr::TagList tags = fieldMapPair.second->get_tags();
      for( const Expr::Tag& tag : tags){
        for( Expr::FieldManagerList::iterator ifm = fml_->begin(); ifm!=fml_->end(); ++ifm ){
          auto fm = ifm->second;
          // this also may need to change for GPU builds
          if( fm->has_field(tag) ) fm->set_field_memory_manager( tag, Expr::MEM_DYNAMIC, CPU_INDEX );
        }
      }
    }
  }

//--------------------------------------------------------------------

  void
  NestedGraphHelper::
  finalize()
  {
    for( Expr::ExpressionTree* tree : treeList_){
      tree->register_fields(*fml_);
      set_memory_manager_to_dynamic(tree);
    }

    fml_->allocate_fields( *allocInfo_ );

    for( Expr::ExpressionTree* tree : treeList_){
      tree->bind_fields( *fml_ );
    }

    if(allocInfo_->patch->getID() == 0){
      for( Expr::ExpressionTree* tree : treeList_){
        std::ofstream ofile(tree->name() + ".dot");
        tree->write_tree(ofile, false, true);
      }
    }
  }

//--------------------------------------------------------------------

}
