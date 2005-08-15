#ifndef SCI_Wangxl_Datatypes_Mesh_CirculatorBase_h
#define SCI_Wangxl_Datatypes_Mesh_CirculatorBase_h

namespace Wangxl {

using namespace SCIRun;

//struct Bidirectional_circulator_tag
//: public std::bidirectional_iterator_tag {};

template <class T, class Dist = ptrdiff_t, class Size = size_t>
struct Bidirectional_circulator_base {
    typedef T                            value_type;
    typedef Dist                         difference_type;
    typedef Size                         size_type;
    typedef T*                           pointer;
    typedef T&                           reference;
    typedef std::bidirectional_iterator_tag iterator_category;
};

}

#endif
