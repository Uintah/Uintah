/**
 * \file   FieldManagerList.h
 * \author James C. Sutherland
 *
 * Copyright (c) 2011 The University of Utah
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
#ifndef Expr_FieldManagerList_h
#define Expr_FieldManagerList_h

#include <vector>
#include <iosfwd>

#include <boost/shared_ptr.hpp>
#include <boost/ref.hpp>
#include <boost/functional/hash.hpp>

#include <expression/ExprFwd.h>
#include <expression/FieldManagerBase.h>

namespace Expr{

/**
 *  @class  FieldManagerList
 *  @author James C. Sutherland
 *  @date   March, 2008
 *
 *  Provides functionality to maintain a list of strongly typed
 *  FieldManager objects.  The FieldManager provides an interface to
 *  the computational framework that is providing memory management
 *  services.
 *
 *  This class is intended for use primarily with the ExpressionTree
 *  class as well as framework-specific Patch classes.
 */

class FieldManagerList
{
public:

  /** @brief Build a FieldManagerList */
  FieldManagerList( const std::string name ) : listName_(name) {}

  ~FieldManagerList(){}

  /** @brief Build a FieldManagerList with a default name */
  FieldManagerList() : listName_( name_counter() ) {}

  /** @brief Obtain a reference to a FieldManager of the requested
   *  field type on the given patch
   */
  template< typename FieldT >
  inline typename FieldMgrSelector<FieldT>::type&
  field_manager();

  /** @brief Obtain a const reference to a FieldManager of the
   *  requested field type on the given patch
   */
  template< typename FieldT >
  inline const typename FieldMgrSelector<FieldT>::type&
  field_manager() const;

  /**
   * @brief Obtain a field of the requested type.
   * @param tag the Tag describing the desired field.
   * @return The field corresponding to the supplied tag.
   *
   * Note: if you are requesting multiple fields of the same type, it is more
   *       efficient to first obtain the FieldManager and then the field reference:
   * \code{.cpp}
   *   const FieldMgrSelector<FieldT>::type& fm = fml.template field_manager<FieldT>();
   *   myField1_ = &fm.field_ref( f1tag_ );
   *   myField2_ = &fm.field_ref( f2tag_ );
   * \endcode
   */
  template< typename FieldT >
  inline const FieldT& field_ref( const Tag& tag ) const;

  /**
   * @brief Obtain a field of the requested type.
   * @param tag the Tag describing the desired field.
   * @return The field corresponding to the supplied tag.
   *
   * Note: if you are requesting multiple fields of the same type, it is more
   *       efficient to first obtain the FieldManager and then the field reference:
   * \code{.cpp}
   *   const FieldMgrSelector<FieldT>::type& fm = fml.template field_manager<FieldT>();
   *   myField1_ = &fm.field_ref( f1tag_ );
   *   myField2_ = &fm.field_ref( f2tag_ );
   * \endcode
   */
  template< typename FieldT >
  inline FieldT& field_ref( const Tag& );


  //@{ /** Iterators */

  typedef boost::shared_ptr<FieldManagerBase> FMPtr;
  typedef std::map<size_t,FMPtr>              FMList;
  typedef FMList::iterator                    iterator;
  typedef FMList::const_iterator              const_iterator;

  inline iterator begin(){ return fieldMgr_.begin(); }
  inline iterator   end(){ return fieldMgr_.end();   }

  inline const_iterator begin() const { return fieldMgr_.begin(); }
  inline const_iterator   end() const { return fieldMgr_.end();   }

  //}@

  /** @brief Allocate all of the fields registered on all FieldManager
   *  objects on this FieldManagerList
   */
  template< typename T >
  inline void allocate_fields( const T& );

  void deallocate_fields();

  /**
   *  @brief iterate all FieldManager objects and output all registered fields.
   */
  void dump_fields( std::ostream& os ) const;

  const std::string& name() const{ return listName_; }

  // jcs label as advanced in documentation?
  /**
   * @brief In situations where the user wants to supply a specific FieldManager
   *  to be used for a given field type, this function can be used.  In general,
   *  users should NOT call this interface, since FieldManagers will be created
   *  automatically.
   *
   * @param fm the FieldManager to use for the specified field type.  This is a
   *  shared pointer to ensure proper memory management.
   */
  template<typename FieldT>
  inline void set_field_manager( FMPtr fm ) const;

private:

  FieldManagerList( const FieldManagerList& );           // no copy constructing.
  FieldManagerList& operator=( const FieldManagerList& );// no assignment.

  std::string name_counter();

  template<typename FieldT>
  inline size_t get_key() const;

  const std::string listName_;
  mutable FMList fieldMgr_;  // mutable so that we can provide a const interface for field_manager

  typedef std::vector<std::string>  FMTypeList;
  mutable FMTypeList fieldMgrTList_;

  boost::hash<std::string> hash_;
};


/**
 * \fn FieldManagerList* extract_field_manager_list( FMLMap& fmls, const int id );
 * \param fmls the FMLMap
 * \param id the identifier for what FieldManagerList is desired
 * \return
 */
FieldManagerList*
extract_field_manager_list( FMLMap& fmls, const int id );

/**
 * \fn const FieldManagerList* extract_field_manager_list( const FMLMap& fmls, const int id );
 * \param fmls the FMLMap
 * \param id the identifier for what FieldManagerList is desired
 * \return
 */
const FieldManagerList*
extract_field_manager_list( const FMLMap& fmls, const int id );

//====================================================================


//--------------------------------------------------------------------

template<typename T>
void
FieldManagerList::allocate_fields( const T& t )
{
  for( iterator i=begin(); i!=end(); ++i ){
    i->second->allocate_fields( boost::cref(t) );
  }
}

//--------------------------------------------------------------------

template<typename FieldT>
size_t
FieldManagerList::get_key() const
{
  return hash_( typeid(FieldT).name() );
}

//--------------------------------------------------------------------

template<typename FieldT>
void
FieldManagerList::set_field_manager( FMPtr fm ) const
{
  fieldMgr_[get_key<FieldT>()] = fm;
}

//--------------------------------------------------------------------

template<typename FieldT>
typename FieldMgrSelector<FieldT>::type&
FieldManagerList::field_manager()
{
  const FieldManagerList& fml = *this;
  typedef typename FieldMgrSelector<FieldT>::type FMT;
  const FMT& fm = fml.field_manager<FieldT>();
  return const_cast<FMT&>( fm );
}

template<typename FieldT>
const typename FieldMgrSelector<FieldT>::type&
FieldManagerList::field_manager() const
{
  const size_t key = get_key<FieldT>();

  typedef typename FieldMgrSelector<FieldT>::type FM;
  FM* fm;

  const iterator i = fieldMgr_.find(key);
  if( i == fieldMgr_.end() ){
    fm = new FM();
    this->set_field_manager<FieldT>( FMPtr(fm) );
  }
  else{
    return dynamic_cast<const FM&>(*i->second);
  }
  return *fm;
}

//-------------------------------------------------------------------

template< typename FieldT >
const FieldT&
FieldManagerList::field_ref( const Tag& tag ) const
{
  return field_manager<FieldT>().field_ref(tag);
}

template< typename FieldT >
FieldT&
FieldManagerList::field_ref( const Tag& tag )
{
  return field_manager<FieldT>().field_ref(tag);
}

//====================================================================

} // namespace Expr


#endif
