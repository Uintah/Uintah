/**
 * \file   FieldDeps.h
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
#ifndef FieldDeps_h
#define FieldDeps_h

#include <vector>
#include <stdio.h>

#include <expression/ExprFwd.h>
#include <expression/Tag.h>
#include <expression/FieldManagerList.h>
#include <expression/ManagerTypes.h>

#include <spatialops/structured/MemoryTypes.h>
#include <spatialops/SpatialOpsTools.h> // nghost();


namespace Expr{

/**
 *  \class  FieldDeps
 *  \author James C. Sutherland
 *  \date   March, 2008
 *
 *  \brief Provides functionality for an Expression to record the
 *  fields it requires.  This provides the ability to retain type
 *  information of the fields for later registration via the
 *  appropriate field manager.
 */
class FieldDeps
{
public:

  class FieldHelperBase;

  typedef std::vector<FieldHelperBase*> FldHelpers;

  /** Construct a FieldDeps object. */
  FieldDeps();

  /** copy constructor for FieldDeps */
  FieldDeps( const FieldDeps& deps );

  ~FieldDeps();

  /**
   *  \brief This is the main functionality of the FieldDeps class.
   *  \param tag the Tag for this field
   */
  template< typename FieldT >
  void requires_field( const Tag& tag )
  {
    fldHelpers_.push_back( new FieldHelper<FieldT>(tag) );
  }

  /**
   *  register all fields from this FieldDeps object onto the supplied FieldManagerList
   */
  void register_fields( FieldManagerList& fml );

  /**
   * \brief Pass through to the FieldHelper
   */
  void set_memory_manager( FieldManagerList& fml, const MemoryManager m, const short int deviceIndex = CPU_INDEX);

  /**
   * @note Pass through to the field helper
   * @param fml the FieldManagerList associated with this field.
   * @param deviceIndex -- Location of the device that will consume this field
   */
  void prep_field_for_consuption( FieldManagerList& fml, const short int deviceIndex );

  /**
   * @note Pass through to the field helper
   * @param fml the FieldManagerList associated with this field.
   * @param deviceIndex -- Location of the device that the field has to be validated
   */
  void validate_field_location( FieldManagerList& fml, const short int deviceIndex );

  /**
   * @note Pass through to the field helper
   * @param fml the FieldManagerList associated with this field.
   * @param deviceIndex -- Location of the device that the field has to be active
   */
  void set_active_field_location( FieldManagerList& fml, const short int deviceIndex );

  /**
   * @brief Pass through to the FieldHelper
   * @param fml the FieldManagerList associated with this field.
   */
  bool release_fields( FieldManagerList& fml );

  /**
   * \brief Pass through to the FieldHelper
   * \param fml the FieldManagerList associated with this field.
   */
  bool lock_fields( FieldManagerList& fml );

  /**
   * \brief Pass through to the FieldHelper
   * \param fml the FieldManagerList associated with this field.
   */
  bool unlock_fields( FieldManagerList& fml );

  const FldHelpers& field_helpers() const{ return fldHelpers_; }
  FldHelpers& field_helpers(){ return fldHelpers_; }

  class FieldHelperBase
  {
  public:
    FieldHelperBase( const Tag& tag )
      : tag_( tag )
    {}

    virtual FieldHelperBase* clone() const = 0;

    virtual void register_field( FieldManagerList& ) = 0;

    const Tag& tag() const{ return tag_; }

    virtual FieldManagerBase& field_manager( FieldManagerList& fml ) = 0;

    virtual const FieldManagerBase& field_manager( const FieldManagerList& fml ) const = 0;

    virtual ~FieldHelperBase(){}

    /**
     * @brief Pass through to the FieldHelper
     * @param fml the FieldManagerList associated with this field.
     * @param m the MemoryManager
     * @param deviceIndex which device (or CPU) this field is associated with.
     */
    virtual void set_field_memory_manager( FieldManagerList& fml, const MemoryManager m,
                                           const short int deviceIndex) {}

    /**
     * @note Passthrough to the field helper
     * @param fml the FieldManagerList associated with this field.
     * @param deviceIndex -- Location of the device that will consume this field
     */
    virtual void prep_field_for_consumption( FieldManagerList& fml, short int deviceIndex ) = 0;

    /**
     * @note Pass through to the field helper
     * @param fml the FieldManagerList associated with this field.
     * @param deviceIndex -- Location of the device that will be validated
     */
    virtual void validate_field_location( FieldManagerList& fml, short int deviceIndex ) = 0;

    /**
     * @note Pass through to the field helper
     * @param fml the FieldManagerList associated with this field.
     * @param deviceIndex -- Location of the device that will consume this field
     */
    virtual void set_active_field_location( FieldManagerList& fml, short int deviceIndex ) = 0;

    /**
     * @brief Pass through to the FieldHelper
     * @param fml the FieldManagerList associated with this field.
     */
    virtual bool release_field( FieldManagerList& fml ) { return false; }

    /**
     * @brief Pass through to the FieldHelper
     * @param fml the FieldManagerList associated with this field.
     */
    virtual bool lock_field( FieldManagerList& fml ) { return false; }

    /**
     * @brief Pass through to the FieldHelper
     * @param fml the FieldManagerList associated with this field.
     */
    virtual bool unlock_field( FieldManagerList& fml ) { return false; }

  protected:
    const Tag tag_;
  };

private:

  FldHelpers fldHelpers_;

  template< typename FieldT >
  class FieldHelper : public FieldHelperBase
  {
  public:
    FieldHelper( const Tag& tag )
      : FieldHelperBase( tag )
    {}

    FieldHelperBase* clone() const{
      return new FieldHelper( tag_ );
    }

    void register_field( FieldManagerList& fml ){
      fml.field_manager<FieldT>().register_field( tag_ );
    }

    /**
     * @brief Pass through function to the FieldManager, sets how memory is managed.
     * @param fml the FieldManagerList associated with this field.
     * @param m the MemoryManager
     * @param deviceIndex which device (or CPU) this field is associated with.
     */
    void set_field_memory_manager( FieldManagerList& fml,
                                   const MemoryManager m,
                                   const short int deviceIndex = CPU_INDEX )
    {
      fml.field_manager<FieldT>().set_field_memory_manager( tag_, m, deviceIndex );
    }

    /**
     * @note Pass through
     * @param fml the FieldManagerList associated with this field.
     * @param deviceIndex -- Location of the device that will consume this field
     */
    void prep_field_for_consumption( FieldManagerList& fml, short int deviceIndex ){
      fml.field_manager<FieldT>().prep_field_for_consumption( tag_, deviceIndex );
    }

    /**
     * @note Pass through
     * @param fml the FieldManagerList associated with this field.
     * @param deviceIndex -- Location of the device that will be validated
     */
    void validate_field_location( FieldManagerList& fml, short int deviceIndex ){
      fml.field_manager<FieldT>().validate_field_location( tag_, deviceIndex );
    }

    /**
     * @note Pass through
     * @param fml the FieldManagerList associated with this field.
     * @param deviceIndex -- Location of the device that will be active
     */
    void set_active_field_location( FieldManagerList& fml, short int deviceIndex ){
      fml.field_manager<FieldT>().set_active_field_location( tag_, deviceIndex );
    }

    /**
     * @brief Pass through to the FieldManager, attempts to release an allocated field
     * @param fml the FieldManagerList associated with this field.
     */
    bool release_field(FieldManagerList& fml){
      return fml.field_manager<FieldT>().release_field( tag_ );
    }

    /**
     * @brief Pass through to the FieldManager, attempt to lock a specific filed
     * @param fml the FieldManagerList associated with this field.
     */
    bool lock_field( FieldManagerList & fml ) {
      return fml.field_manager<FieldT>().lock_field( tag_ );
    }

    /**
     * @brief Pass through to the FieldManager, attempt to unlock a specific filed
     * @param fml the FieldManagerList associated with this field.
     */
    bool unlock_field( FieldManagerList & fml ) {
      return fml.field_manager<FieldT>().unlock_field( tag_ );
    }

    FieldManagerBase& field_manager( FieldManagerList& fml ){ return fml.field_manager<FieldT>(); }
    const FieldManagerBase& field_manager( const FieldManagerList& fml ) const{
       return fml.field_manager<FieldT>();
    }

  };

};

} // namespace Expr

#endif
