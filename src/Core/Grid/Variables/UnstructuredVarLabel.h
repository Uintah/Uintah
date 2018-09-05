/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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


#ifndef CORE_GRID_VARIABLES_UNSTRUCTURED_VARLABEL_H
#define CORE_GRID_VARIABLES_UNSTRUCTURED_VARLABEL_H


#include <Core/Util/RefCounted.h>
#include <Core/Geometry/IntVector.h>

#include <iosfwd>
#include <string>


namespace Uintah {

class UnstructuredTypeDescription;
class UnstructuredPatch;

/**************************************

  CLASS
    VarLabel

  GENERAL INFORMATION

    VarLabel.h

    Steven G. Parker
    Department of Computer Science
    University of Utah


  KEYWORDS
    VarLabel

  DESCRIPTION

****************************************/

class UnstructuredVarLabel : public RefCounted {

public:

  enum VarType {
      Normal
    , PositionVariable
  };

  // Ensure the uniqueness of UnstructuredVarLabel names (same name, same object).
  static UnstructuredVarLabel* create( const std::string       & name
                         , const UnstructuredTypeDescription   * type_description
                         , const IntVector         & boundaryLayer = IntVector(0,0,0)
                         ,       VarType             vartype       = Normal
                         );

  static bool destroy( const UnstructuredVarLabel * label );

  inline const std::string & getName() const { return m_name;  }

  std::string getFullName( int matlIndex, const UnstructuredPatch * patch ) const;

  bool isPositionVariable() const { return m_var_type == PositionVariable; }

  const UnstructuredTypeDescription * typeDescription() const { return m_td; }

  IntVector getBoundaryLayer() const { return m_boundary_layer; }

  void allowMultipleComputes();

  bool allowsMultipleComputes() const { return m_allow_multiple_computes; }

  static UnstructuredVarLabel* find( const std::string& name );

  static UnstructuredVarLabel* particlePositionLabel();

  static void setParticlePositionName(const std::string& pPosName) { s_particle_position_name = pPosName; }

  static std::string& getParticlePositionName() { return s_particle_position_name; }

  class Compare {

  public:

    inline bool operator()(const UnstructuredVarLabel* v1, const UnstructuredVarLabel* v2) const {
      // because of uniqueness, we can use pointer comparisons
      //return v1 < v2;
      // No we cannot, because we need the order to be the same on different processes
      if(v1 == v2) {
        return false;
      }
      return v1->getName() < v2->getName();
    }

  };

  bool equals(const UnstructuredVarLabel* v2) const {
    // because of uniqueness, we can use pointer comparisons
    return this == v2;
    /* old way
       if(this == v2)
       return true;
       return getName() == v2->getName();
    */
  }

  void setCompressionMode( std::string compressionMode ) { m_compression_mode = compressionMode; }

  const std::string& getCompressionMode() const {
    return (m_compression_mode == "default") ? s_default_compression_mode : m_compression_mode;
  }

  static void setDefaultCompressionMode( const std::string & compressionMode ) {
    s_default_compression_mode = compressionMode;
  }

  static void printAll(); // for debugging

  friend std::ostream & operator<<( std::ostream & out, const UnstructuredVarLabel & vl );


private:

  // You must use UnstructuredVarLabel::create.
  UnstructuredVarLabel( const std::string     &
          , const UnstructuredTypeDescription *
          , const IntVector       & boundaryLayer
          ,       VarType           vartype
          );

  // You must use destroy.
  ~UnstructuredVarLabel(){};

          std::string         m_name{""};
  const   UnstructuredTypeDescription   * m_td{nullptr};
          IntVector           m_boundary_layer{IntVector(0,0,0)};
          VarType             m_var_type{Normal};

  mutable std::string         m_compression_mode{"default"};
  static  std::string         s_default_compression_mode;
  static  std::string         s_particle_position_name;

  // Allow a variable of this label to be computed multiple times in a TaskGraph without complaining.
  bool                        m_allow_multiple_computes{false};

  // eliminate copy, assignment and move
  UnstructuredVarLabel( const UnstructuredVarLabel & )            = delete;
  UnstructuredVarLabel& operator=( const UnstructuredVarLabel & ) = delete;
  UnstructuredVarLabel( UnstructuredVarLabel && )                 = delete;
  UnstructuredVarLabel& operator=( UnstructuredVarLabel && )      = delete;

  // Static member to keep track of all labels created to prevent
  // duplicates.
  static std::map<std::string, UnstructuredVarLabel*> g_all_labels; 
};

} // End namespace Uintah

#endif // CORE_GRID_VARIABLES_VARLABEL_H
