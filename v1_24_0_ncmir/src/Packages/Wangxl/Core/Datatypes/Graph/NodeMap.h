#ifndef SCI_Wangxl_Datatypes_NodeMap_h
#define SCI_Wangxl_Datatypes_NodeMap_h

#include "GraphNode.h"
#include "GraphMap.h"

namespace Wangxl {

template < class T >
class NodeMap : public GraphMap< GraphNode, T >
{
public:
    NodeMap() :	GraphMap< GraphNode, T >() {};
    explicit NodeMap( const Graph &graph, T t=T() ) : GraphMap< GraphNode, T >( graph, t ) {};
};

}

#endif
