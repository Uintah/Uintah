//  This is a generated file, DO NOT EDIT!!

#if !defined(Datatypes_DispatchMesh1_h)
#define Datatypes_DispatchMesh1_h

// uintah_dispatch_mesh1 macro follows
#define uintah_dispatch_mesh1(mesh1, callback)\
  bool mdisp_error = false;\
  string mdisp_msg;\
  string mdisp_name = mesh1->get_type_name(0);\
  if (mdisp_name == "LevelMesh") {\
    LevelMesh *f1 = 0;\
    f1 = dynamic_cast<LevelMesh*>(mesh1.get_rep());\
    if (f1) {\
      callback(f1);\
    } else {\
      mdisp_error = true; mdisp_msg = "LevelMesh::get_type_name is broken";\
    }\
  } else if (mdisp_error) {\
    cerr << "Error: " << mdisp_msg << endl;\
  }\

#endif //Datatypes_DispatchMesh1_h
