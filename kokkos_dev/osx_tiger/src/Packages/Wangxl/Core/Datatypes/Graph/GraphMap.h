#ifndef SCI_Wangxl_Datatypes_GraphMap_h
#define SCI_Wangxl_Datatypes_GraphMap_h

#include <vector>

namespace Wangxl {

using std::vector;

class Graph;

template < class Key, class Value > 
class GraphMap
{
protected:
    GraphMap();
    explicit GraphMap( const Graph &graph, Value def=Value() );
public:
    void init(const Graph &, Value def=Value());
    typedef typename vector< Value >::reference value_reference;
    typedef typename vector< Value >::const_reference const_value_reference;
    value_reference operator[]( Key key );
    const_value_reference operator[](Key key) const;
private:
    vector< Value > data;
};

template < class Key, class Value >
GraphMap< Key, Value >::GraphMap()
{
}

template < class Key, class Value >
GraphMap< Key, Value >::GraphMap( const Graph &graph, Value val ) :
    data(graph.getIdsNum(Key()), val)
{
}

template < class Key, class Value >
void GraphMap< Key,Value >::init(const Graph &graph, Value val)
{
int n;
  n = graph.getIdsNum( Key() );
  data.resize( n );
  fill_n( data.begin(), n, val );
}

template < class Key, class Value >
GraphMap< Key, Value >::value_reference GraphMap< Key, Value >::operator[](Key key)
{
  if(key.id() >= (signed)data.size())
  {
    if (key.id() >= (signed)data.capacity()) {
      data.reserve((6 * key.id()) / 5 + 1);
    }
    data.insert( data.end(), key.id() + 1 - data.size(), Value());
  }
  return data.operator[](key.id());
}

template < class Key, class Value >
GraphMap< Key, Value >::const_value_reference GraphMap< Key, Value >::operator[](Key key) const
{
    assert(key.id() < (signed)data.size());
    return data.operator[](key.id());
}

}

#endif



