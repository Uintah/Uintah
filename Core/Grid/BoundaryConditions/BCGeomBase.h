/*
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
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

#ifndef UINTAH_GRID_BCGeomBase_H
#define UINTAH_GRID_BCGeomBase_H

#include <Core/Grid/BoundaryConditions/BCData.h>
#include <Core/Grid/Patch.h>
#include <Core/Geometry/Point.h>
#include <Core/Grid/Variables/Iterator.h>
#include <Core/Grid/Variables/BaseIterator.h>
#include <vector>
#include <typeinfo>
#include <iterator>


namespace Uintah {

  /*!

  \class BCGeomBase

  \ brief Base class for the boundary condition geometry types.
  
  \author John A. Schmidt \n
  Department of Mechanical Engineering \n
  University of Utah \n
  Center for the Simulation of Accidental Fires and Explosions (C-SAFE) \n\n

  */

  using SCIRun::Point;
   

  class BCGeomBase {
  public:

    /*  \struct ParticleBoundarySpec
     *  \author Tony Saad
     *  \date   August 2014
     *  \brief  Struct that holds information about particle boundary conditions
     */
    struct ParticleBndSpec {

      /*
       *  \brief Particle boundary type wall or inlet
       */
      enum  ParticleBndTypeEnum {
        WALL, INLET, NOTSET
      };

      /*
       *  \brief Wall boundary type: Elastic, Inelastic, etc...
       */
      enum  ParticleWallTypeEnum {
        ELASTIC, INELASTIC, PARTIALLYELASTIC
      };
      
      // constructor
      ParticleBndSpec()
      {
        ParticleBndSpec(ParticleBndSpec::NOTSET, ParticleBndSpec::ELASTIC, 0.0, 0.0);
      }

      // constructor
      ParticleBndSpec(ParticleBndTypeEnum bndTypet, ParticleWallTypeEnum wallTypet, const double restitution, const double pPerSec)
      {
        bndType = bndTypet;
        wallType = wallTypet;
        restitutionCoef = restitution;
        particlesPerSec = pPerSec;
      }
      
      // constructor
      ParticleBndSpec(const ParticleBndSpec& rhs)
      {
        bndType = rhs.bndType;
        wallType = rhs.wallType;
        restitutionCoef = rhs.restitutionCoef;
        particlesPerSec = rhs.particlesPerSec;
      }
      
      /*
       *  \brief Checks whether a particle boundary condition has been specified on this BCGeometry
       */
      const bool hasParticlesBoundary() const
      {
        return (bndType != ParticleBndSpec::NOTSET);
      }
      
      ParticleBndTypeEnum bndType;
      ParticleWallTypeEnum wallType;
      double restitutionCoef;
      double particlesPerSec;
    };
    /// Constructor
    BCGeomBase();

    /// Copy constructor
    BCGeomBase(const BCGeomBase& rhs);

    /// Assignment operator
    BCGeomBase& operator=(const BCGeomBase& rhs);

    /// Destructor
    virtual ~BCGeomBase();    

    /// Equality test
    virtual bool operator==(const BCGeomBase&) const = 0;

    /// Make a clone
    virtual BCGeomBase* clone() = 0;

    /// Get the boundary condition data
    virtual void getBCData(BCData& bc) const = 0;

    /// For old boundary conditions
    virtual void addBCData(BCData& bc)  = 0;

    /// For old boundary conditions
    virtual void addBC(BoundCondBase* bc)  = 0;

    void getCellFaceIterator(Iterator& b_ptr);

    void getNodeFaceIterator(Iterator& b_ptr);


    /// Determine if a point is inside the geometry where the boundary
    /// condition is applied.
    virtual bool inside(const Point& p) const = 0;

    /// Print out the type of boundary condition -- debugging
    virtual void print() = 0;

    /// Determine the cell centered boundary and node centered boundary
    /// iterators.
    virtual void determineIteratorLimits(Patch::FaceType face, 
                                         const Patch* patch, 
                                         std::vector<Point>& test_pts);
    
    /// Print out the iterators for the boundary.
    void printLimits() const;

    /// Get the name for this boundary specification
    std::string getBCName(){ return d_bcname; };
    void setBCName( std::string bcname ){ d_bcname = bcname; }; 

    /// Get the type for this boundary specification (type is usually associated with a user-friendly
    /// boundary type such as Wall, Inlet, Outflow...
    std::string getBndType(){ return d_bndtype; };
    void setBndType( std::string bndType ){ d_bndtype = bndType; };
    
    ParticleBndSpec getParticleBndSpec(){return d_particleBndSpec;}
    void setParticleBndSpec(const ParticleBndSpec pBndSpec){ d_particleBndSpec = pBndSpec; }
    const bool hasParticlesBoundary(){return d_particleBndSpec.hasParticlesBoundary();}

    const double surfaceArea(){return d_surfaceArea;}
    
  protected:
    Iterator          d_cells;
    Iterator          d_nodes;
    std::string       d_bcname;
    std::string       d_bndtype;
    ParticleBndSpec   d_particleBndSpec;
    double            d_surfaceArea;
  };

  template<class T> class cmp_type {
    public:
    bool operator()(const BCGeomBase* p) {
      return (typeid(T) == typeid(*p));
    }
  };

  template<class T> class not_type {
    public:
    bool operator()(const BCGeomBase* p) {
      return (typeid(T) != typeid(*p));
    }
  };

  template<typename T> class delete_object {
  public:
    void operator() (T* ptr) {
      delete ptr;
    }
  };

} // End namespace Uintah

#endif
