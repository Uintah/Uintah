<9: //#define running_NGC_nozzle
<10: #undef  running_NGC_nozzle 

>102:     BCGeomBase* bc_geom_type = patch->getBCDataArray(face)->getChild(mat_id,child);
>103:     cmp_type<CircleBCData> nozzle;
>104:         
>105:     if(kind == "Pressure" && face == Patch::xminus && nozzle(bc_geom_type) ) {

>143:   
>144:     BCGeomBase* bc_geom_type = patch->getBCDataArray(face)->getChild(mat_id,child);
>145:     cmp_type<CircleBCData> nozzle;
>146: 
>147:     if(face == Patch:: xminus && mat_id == 1 && nozzle(bc_geom_type)){

