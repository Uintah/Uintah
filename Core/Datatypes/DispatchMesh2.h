//  This is a generated file, DO NOT EDIT!!

#if !defined(Datatypes_DispatchMesh2_h)
#define Datatypes_DispatchMesh2_h

// dispatch_mesh2 macro follows
#define dispatch_mesh2(mesh1, mesh2, callback)\
  bool mdisp_error = false;\
  string mdisp_msg;\
  string mdisp_name = mesh2->get_type_name(0);\
  if (mdisp_name == "TetVolMesh") {\
    TetVolMesh *f2 = 0;\
    f2 = dynamic_cast<TetVolMesh*>(mesh2.get_rep());\
    if (f2) {\
      if (mdisp_name == "TetVolMesh") {\
        TetVolMesh *f1 = 0;\
        f1 = dynamic_cast<TetVolMesh*>(mesh1.get_rep());\
        if (f1) {\
          callback(f1, f2);\
        } else {\
          mdisp_error = true; mdisp_msg = "TetVolMesh::get_type_name is broken";\
        }\
      } else if (mdisp_name == "LatVolMesh") {\
        LatVolMesh *f1 = 0;\
        f1 = dynamic_cast<LatVolMesh*>(mesh1.get_rep());\
        if (f1) {\
          callback(f1, f2);\
        } else {\
          mdisp_error = true; mdisp_msg = "LatVolMesh::get_type_name is broken";\
        }\
      } else if (mdisp_name == "ContourMesh") {\
        ContourMesh *f1 = 0;\
        f1 = dynamic_cast<ContourMesh*>(mesh1.get_rep());\
        if (f1) {\
          callback(f1, f2);\
        } else {\
          mdisp_error = true; mdisp_msg = "ContourMesh::get_type_name is broken";\
        }\
      } else if (mdisp_name == "TriSurfMesh") {\
        TriSurfMesh *f1 = 0;\
        f1 = dynamic_cast<TriSurfMesh*>(mesh1.get_rep());\
        if (f1) {\
          callback(f1, f2);\
        } else {\
          mdisp_error = true; mdisp_msg = "TriSurfMesh::get_type_name is broken";\
        }\
      } else if (mdisp_error) {\
        cerr << "Error: " << mdisp_msg << endl;\
      }\
    } else {\
      mdisp_error = true; mdisp_msg = "TetVolMesh::get_type_name is broken";\
    }\
  } else if (mdisp_name == "LatVolMesh") {\
    LatVolMesh *f2 = 0;\
    f2 = dynamic_cast<LatVolMesh*>(mesh2.get_rep());\
    if (f2) {\
      if (mdisp_name == "TetVolMesh") {\
        TetVolMesh *f1 = 0;\
        f1 = dynamic_cast<TetVolMesh*>(mesh1.get_rep());\
        if (f1) {\
          callback(f1, f2);\
        } else {\
          mdisp_error = true; mdisp_msg = "TetVolMesh::get_type_name is broken";\
        }\
      } else if (mdisp_name == "LatVolMesh") {\
        LatVolMesh *f1 = 0;\
        f1 = dynamic_cast<LatVolMesh*>(mesh1.get_rep());\
        if (f1) {\
          callback(f1, f2);\
        } else {\
          mdisp_error = true; mdisp_msg = "LatVolMesh::get_type_name is broken";\
        }\
      } else if (mdisp_name == "ContourMesh") {\
        ContourMesh *f1 = 0;\
        f1 = dynamic_cast<ContourMesh*>(mesh1.get_rep());\
        if (f1) {\
          callback(f1, f2);\
        } else {\
          mdisp_error = true; mdisp_msg = "ContourMesh::get_type_name is broken";\
        }\
      } else if (mdisp_name == "TriSurfMesh") {\
        TriSurfMesh *f1 = 0;\
        f1 = dynamic_cast<TriSurfMesh*>(mesh1.get_rep());\
        if (f1) {\
          callback(f1, f2);\
        } else {\
          mdisp_error = true; mdisp_msg = "TriSurfMesh::get_type_name is broken";\
        }\
      } else if (mdisp_error) {\
        cerr << "Error: " << mdisp_msg << endl;\
      }\
    } else {\
      mdisp_error = true; mdisp_msg = "LatVolMesh::get_type_name is broken";\
    }\
  } else if (mdisp_name == "ContourMesh") {\
    ContourMesh *f2 = 0;\
    f2 = dynamic_cast<ContourMesh*>(mesh2.get_rep());\
    if (f2) {\
      if (mdisp_name == "TetVolMesh") {\
        TetVolMesh *f1 = 0;\
        f1 = dynamic_cast<TetVolMesh*>(mesh1.get_rep());\
        if (f1) {\
          callback(f1, f2);\
        } else {\
          mdisp_error = true; mdisp_msg = "TetVolMesh::get_type_name is broken";\
        }\
      } else if (mdisp_name == "LatVolMesh") {\
        LatVolMesh *f1 = 0;\
        f1 = dynamic_cast<LatVolMesh*>(mesh1.get_rep());\
        if (f1) {\
          callback(f1, f2);\
        } else {\
          mdisp_error = true; mdisp_msg = "LatVolMesh::get_type_name is broken";\
        }\
      } else if (mdisp_name == "ContourMesh") {\
        ContourMesh *f1 = 0;\
        f1 = dynamic_cast<ContourMesh*>(mesh1.get_rep());\
        if (f1) {\
          callback(f1, f2);\
        } else {\
          mdisp_error = true; mdisp_msg = "ContourMesh::get_type_name is broken";\
        }\
      } else if (mdisp_name == "TriSurfMesh") {\
        TriSurfMesh *f1 = 0;\
        f1 = dynamic_cast<TriSurfMesh*>(mesh1.get_rep());\
        if (f1) {\
          callback(f1, f2);\
        } else {\
          mdisp_error = true; mdisp_msg = "TriSurfMesh::get_type_name is broken";\
        }\
      } else if (mdisp_error) {\
        cerr << "Error: " << mdisp_msg << endl;\
      }\
    } else {\
      mdisp_error = true; mdisp_msg = "ContourMesh::get_type_name is broken";\
    }\
  } else if (mdisp_name == "TriSurfMesh") {\
    TriSurfMesh *f2 = 0;\
    f2 = dynamic_cast<TriSurfMesh*>(mesh2.get_rep());\
    if (f2) {\
      if (mdisp_name == "TetVolMesh") {\
        TetVolMesh *f1 = 0;\
        f1 = dynamic_cast<TetVolMesh*>(mesh1.get_rep());\
        if (f1) {\
          callback(f1, f2);\
        } else {\
          mdisp_error = true; mdisp_msg = "TetVolMesh::get_type_name is broken";\
        }\
      } else if (mdisp_name == "LatVolMesh") {\
        LatVolMesh *f1 = 0;\
        f1 = dynamic_cast<LatVolMesh*>(mesh1.get_rep());\
        if (f1) {\
          callback(f1, f2);\
        } else {\
          mdisp_error = true; mdisp_msg = "LatVolMesh::get_type_name is broken";\
        }\
      } else if (mdisp_name == "ContourMesh") {\
        ContourMesh *f1 = 0;\
        f1 = dynamic_cast<ContourMesh*>(mesh1.get_rep());\
        if (f1) {\
          callback(f1, f2);\
        } else {\
          mdisp_error = true; mdisp_msg = "ContourMesh::get_type_name is broken";\
        }\
      } else if (mdisp_name == "TriSurfMesh") {\
        TriSurfMesh *f1 = 0;\
        f1 = dynamic_cast<TriSurfMesh*>(mesh1.get_rep());\
        if (f1) {\
          callback(f1, f2);\
        } else {\
          mdisp_error = true; mdisp_msg = "TriSurfMesh::get_type_name is broken";\
        }\
      } else if (mdisp_error) {\
        cerr << "Error: " << mdisp_msg << endl;\
      }\
    } else {\
      mdisp_error = true; mdisp_msg = "TriSurfMesh::get_type_name is broken";\
    }\
  } else if (mdisp_error) {\
    cerr << "Error: " << mdisp_msg << endl;\
  }\

#endif //Datatypes_DispatchMesh2_h
