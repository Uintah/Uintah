/**
 * \file FieldManagerBase.h
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

#ifndef Expr_FieldManagerBase_h
#define Expr_FieldManagerBase_h

#include <string>
#include <map>

#include <expression/ExprFwd.h>
#include <expression/ManagerTypes.h>

#include <spatialops/structured/MemoryTypes.h>

#include <boost/any.hpp>

namespace Expr{


//====================================================================


/**
 *  @class FieldManagerBase
 *  @author James C. Sutherland
 *
 *  @brief Common base class for all concrete FieldManager types so
 *  that they can be held in a container.
 */
class FieldManagerBase
{
public:

  virtual void allocate_fields( const boost::any& patch ) = 0;

  virtual void deallocate_fields() = 0;

  /**
   *  @brief Release memory allocated for a specific field ONLY IF it is a scratch field
   */
  virtual bool release_field( const Tag& ) = 0;

  /** @brief Place a hold on a field, preventing its memory from being released during execution */
  virtual bool lock_field(const Tag& tag) = 0;

  /** @brief Remove a hold on a field, allowing its memory to be released during execution */
  virtual bool unlock_field(const Tag& tag) = 0;

  /**
   *  @brief Set the method for managing field memory
   *  \param t Tag for this field.
   *  \param m the memory manager to use for this field.
   *  \param deviceIndex the device (CPU, GPU) index for this field.
   */
  virtual void set_field_memory_manager( const Tag& t,
                                         const MemoryManager m,
                                         const short int deviceIndex ) = 0;

  /**
   * @param tag Tag for this field
   * @param deviceIndex -- Location of the device type that will consume the field
   **/
  virtual void prep_field_for_consumption( const Tag& tag,
                                           const short int deviceIndex ) = 0;

  /**
   * @param tag Tag for this field
   * @param deviceIndex -- Location of the device to be validated
   **/
  virtual void validate_field_location( const Tag& tag,
                                        const short int deviceIndex ) = 0;

  /**
   * @param tag Tag for this field
   * @param deviceIndex -- Location of the device type that will be set as active
   **/
  virtual void set_active_field_location( const Tag& tag,
                                           const short int deviceIndex ) = 0;

  /** @brief Register a field given the Tag */
  virtual FieldID register_field( const Tag& tag ) = 0;

  /** @brief Register a field with the given name and context */
  FieldID register_field( const std::string& fieldName, const Context c );

  /** @brief Dump a list of registered fields */
  virtual void dump_fields(std::ostream& os) const = 0;

  /** @brief Query if the given field exists */
  virtual bool has_field( const Tag& fieldTag ) const = 0;


  // this allows individual field managers to define information and
  // place it on this map for use later on.  Because this is on the
  // base class, it allows type-specific information (from
  // FieldManager<FieldT> classes) to be pushed out to places where
  // that type information is unavailable.
  typedef std::map<std::string,boost::any> PropertyMap;
  PropertyMap& properties(){ return properties_; }
  const PropertyMap& properties() const{ return properties_; }

protected:

  FieldManagerBase(){}
  virtual ~FieldManagerBase(){}
  static int get_name_id();
  PropertyMap properties_;

private:
  FieldManagerBase( const FieldManagerBase& ); // no copying
  FieldManagerBase& operator=( const FieldManagerBase& ); // no assignment
};

//====================================================================

} // namespace Expr

#endif
