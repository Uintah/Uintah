#ifndef SCI_Wangxl_Datatypes_EdgeMap_h
#define SCI_Wangxl_Datatypes_EdgeMap_h

#include "GraphEdge.h"
#include "GraphMap.h"

namespace Wangxl {

template < class T >
class EdgeMap : public GraphMap< GraphEdge, T >
{
public:
    EdgeMap() :	GraphMap< GraphEdge, T >() {};
    explicit EdgeMap( const Graph &graph, T t = T() ) : GraphMap< GraphEdge, T >( graph, t ) {};
};

}

#endif
