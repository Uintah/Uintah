/*
 * VoronoiGeometryPiece.h
 *
 *  Created on: Sep 25, 2018
 *      Author: jbhooper
 */

#ifndef SRC_CORE_GEOMETRYPIECE_VORONOIGEOMETRYPIECE_H_
#define SRC_CORE_GEOMETRYPIECE_VORONOIGEOMETRYPIECE_H_

#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/GeometryPiece/GeometryPieceFactory.h>
#include <Core/GeometryPiece/UnionGeometryPiece.h>
#include <Core/Grid/Box.h>
#include <Core/Geometry/Point.h>

// Specific types of voronoi cell geometries
#include <Core/GeometryPiece/Voronoi/FCCVoronoiSubset.h>

namespace Uintah {
  class VoronoiGeometryPiece : public GeometryPiece {
    public:
      VoronoiGeometryPiece(ProblemSpecP &);

      VoronoiGeometryPiece& operator=(const VoronoiGeometryPiece &);

      virtual ~VoronoiGeometryPiece() { }

      static const std::string TYPE_NAME;
      virtual std::string getType() const {return TYPE_NAME; }

      GeometryPieceP clone() const;

      virtual bool inside(const Point &p) const;

      virtual Box getBoundingBox() const;

    private:
      virtual void outputHelper(ProblemSpecP & ps) const;
      std::vector<GeometryPieceP> child_;
      GeometryPieceP tesselation_;
      GeometryPieceP _wholepiece;

      std::string         m_cellType;
      std::vector<Point>  m_latticePoints;
      double              m_latticeConstant;
      IntVector           m_extents;
      Point               m_startPoint;
      int                 m_subsetIndex;
  };
}

#endif /* SRC_CORE_GEOMETRYPIECE_VORONOIGEOMETRYPIECE_H_ */
