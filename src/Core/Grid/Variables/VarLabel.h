/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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


#ifndef CORE_GRID_VARIABLES_VARLABEL_H
#define CORE_GRID_VARIABLES_VARLABEL_H


#include <Core/Util/RefCounted.h>
#include <Core/Geometry/IntVector.h>

#include <iosfwd>
#include <string>
#include <Core/Grid/Ghost.h>


namespace Uintah {

class TypeDescription;
class Patch;

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

class VarLabel : public RefCounted {

public:

  enum VarType {
      Normal
    , PositionVariable
  };

  // Ensure the uniqueness of VarLabel names (same name, same object).
  static VarLabel* create( const std::string       & name
                         , const TypeDescription   * type_description
                         , const IntVector         & boundaryLayer = IntVector(0,0,0)
                         ,       VarType             vartype       = Normal
                         );

  static bool destroy( const VarLabel * label );

  inline const std::string & getName() const { return m_name;  }

  std::string getFullName( int matlIndex, const Patch * patch ) const;

  bool isPositionVariable() const { return m_var_type == PositionVariable; }

  const TypeDescription * typeDescription() const { return m_td; }

  IntVector getBoundaryLayer() const { return m_boundary_layer; }

  void allowMultipleComputes();

  bool allowsMultipleComputes() const { return m_allow_multiple_computes; }

  static VarLabel* find( const std::string& name );

  static VarLabel* find( const std::string& name,
                         const std::string& messg );

  static VarLabel* particlePositionLabel();

  static void setParticlePositionName(const std::string& pPosName) { s_particle_position_name = pPosName; }

  static std::string& getParticlePositionName() { return s_particle_position_name; }

  class Compare {

  public:

    inline bool operator()(const VarLabel* v1, const VarLabel* v2) const {
      // because of uniqueness, we can use pointer comparisons
      //return v1 < v2;
      // No we cannot, because we need the order to be the same on different processes
      if(v1 == v2) {
        return false;
      }
      return v1->getName() < v2->getName();
    }

  };

  bool equals(const VarLabel* v2) const {
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

  friend std::ostream & operator<<( std::ostream & out, const VarLabel & vl );

  //DS 12062019: Store max ghost cell count for this variable across all GPU tasks. update it in dependencies of all gpu tasks before task graph compilation
  inline int getMaxDeviceGhost() const {return m_max_device_ghost;}
  inline void setMaxDeviceGhost(int val) const {m_max_device_ghost = val;}
  inline Ghost::GhostType getMaxDeviceGhostType() const {return m_max_device_ghost_type;}
  inline void setMaxDeviceGhostType (Ghost::GhostType val) const {m_max_device_ghost_type = val;}

private:

  // You must use VarLabel::create.
  VarLabel( const std::string     &
          , const TypeDescription *
          , const IntVector       & boundaryLayer
          ,       VarType           vartype
          );

  // You must use destroy.
  ~VarLabel(){};

          std::string         m_name{""};
  const   TypeDescription   * m_td{nullptr};
          IntVector           m_boundary_layer{IntVector(0,0,0)};
          VarType             m_var_type{Normal};

  mutable std::string         m_compression_mode{"default"};
  static  std::string         s_default_compression_mode;
  static  std::string         s_particle_position_name;

  // Allow a variable of this label to be computed multiple times in a TaskGraph without complaining.
  bool                        m_allow_multiple_computes{false};

  mutable int                 m_max_device_ghost{0};	//DS 12062019: Store max ghost cell count for this variable across all GPU tasks. update it in dependencies of all gpu tasks before task graph compilation
  mutable Ghost::GhostType    m_max_device_ghost_type{Ghost::None};

  // eliminate copy, assignment and move
  VarLabel( const VarLabel & )            = delete;
  VarLabel& operator=( const VarLabel & ) = delete;
  VarLabel( VarLabel && )                 = delete;
  VarLabel& operator=( VarLabel && )      = delete;

  // Static member to keep track of all labels created to prevent
  // duplicates.
  static std::map<std::string, VarLabel*> g_all_labels;
};

} // End namespace Uintah

#endif // CORE_GRID_VARIABLES_VARLABEL_H
