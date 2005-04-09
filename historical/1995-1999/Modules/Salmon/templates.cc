
/*
 * Manual template instantiations for g++
 */

#include <Classlib/Array1.cc>
#include <Classlib/AVLTree.cc>
#include <Classlib/HashTable.cc>
#include <Classlib/String.cc>
#include <Multitask/Mailbox.cc>
#include <X11/Xlib.h>
#include <X11/Xutil.h>

class Roe;
template class Array1<Roe*>;

class RegisterRenderer;
template class AVLTree<clString, RegisterRenderer*>;
template class AVLTreeIter<clString, RegisterRenderer*>;

class Renderer;
template class HashTable<clString, Renderer*>;
class ObjTag;
template class HashTable<clString, ObjTag*>;

class TexStruct1D;
template class Array1<TexStruct1D*>;
class TexStruct2D;
template class Array1<TexStruct2D*>;
class TexStruct3D;
template class Array1<TexStruct3D*>;
class GeomSalmonItem;
template class Array1<GeomSalmonItem*>;
template class Array1<XVisualInfo*>;

template class Mailbox<int>;
