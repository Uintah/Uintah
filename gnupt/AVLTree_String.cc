
#pragma implementation "AVLTree.h"

#include <Classlib/AVLTree.cc>
#include <Classlib/String.h>

class Arg_base;
class DebugSwitch;
class Module;
class RegisterRenderer;
class GeomObj;
template<class T> class Array1;

typedef TreeLink<clString, Arg_base*> _dummy1_;
typedef TreeLink<clString, Module* (*)(const clString&)> _dummy2_;
typedef TreeLink<clString, AVLTree<clString, Module* (*)(const clString&)>*> _dummy3_;
typedef TreeLink<clString, int> _dummy4_;
typedef TreeLink<clString, RegisterRenderer*> _dummy5_;
typedef TreeLink<double, GeomObj*> _dummy6_;
typedef TreeLink<clString, Array1<DebugSwitch*>*> _dummy7_;

typedef AVLTreeIter<clString, Arg_base*> _dummy10_;
typedef AVLTree<clString, Arg_base*> _dummy11_;

typedef AVLTree<clString, Module* (*)(const clString&)> _dummy20_;
typedef AVLTreeIter<clString, Module* (*)(const clString&)> _dummy21_;

typedef AVLTree<clString, AVLTree<clString, Module* (*)(const clString&)>*> _dummy30_;
typedef AVLTreeIter<clString, AVLTree<clString, Module* (*)(const clString&)>*> _dummy31_;

typedef AVLTree<clString, int> _dummy40_;
typedef AVLTreeIter<clString, int> _dummy41_;

typedef AVLTreeIter<clString, RegisterRenderer*> _dummy50_;
typedef AVLTree<clString, RegisterRenderer*> _dummy51_;

typedef AVLTreeIter<double, GeomObj* > _dummy60_;
typedef AVLTree<double, GeomObj* > _dummy61_;

typedef AVLTreeIter<clString, Array1<DebugSwitch*>*> _dummy70_;
typedef AVLTree<clString, Array1<DebugSwitch*>*> _dummy71_;

#if 0
typedef TreeLink<clString, ModuleSubCategory*> _dummy7_;
typedef AVLTree<clString, ModuleSubCategory*> _dummy8_;
typedef AVLTreeIter<clString, ModuleSubCategory*> _dummy9_;

typedef TreeLink<clString, ModuleCategory*> _dummy10_;
typedef AVLTree<clString, ModuleCategory*> _dummy11_;
typedef AVLTreeIter<clString, ModuleCategory*> _dummy12_;
#endif

