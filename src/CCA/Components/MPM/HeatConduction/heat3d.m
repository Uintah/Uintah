1;

function [points,elems] = make_grid(bar_dim,spacing)
  count = 1;
  x = y = z = 0;
  while (z <= bar_dim.z)
    while (y <= bar_dim.y)
      while (x <= bar_dim.x)
        points(count).x = x;
        points(count).y = y;
        points(count++).z = z;
        x += spacing.x;
      endwhile
      x = 0;
      y += spacing.y;
    endwhile
    y = 0;
    z += spacing.z;
  endwhile

  dx = bar_dim.x/spacing.x;
  dy = bar_dim.y/spacing.y;
  dz = bar_dim.z/spacing.z;

  e=1;
  for (k = 1:dz)
    for (j = 1:dy)
      for (i = 1:dx)
        elems(e,1) = i + (j-1)*(dx+1);
        elems(e,2) = i + (j-1)*(dx+1) + 1;
        elems(e,3) = i + (dx + 1)  + (j-1)*(dx+1);
        elems(e,4) = i + 1 + (dx + 1) + (j-1)*(dx+1);
        elems(e,5) = (dx+1)*(dy+1) + i + (j-1)*(dx+1);
        elems(e,6) = (dx+1)*(dy+1) + i + 1 + (j-1)*(dx+1);
        elems(e,7) = (dx+1)*(dy+1) + i + (dx + 1) +  (j-1)*(dx+1);
        elems(e,8) = (dx+1)*(dy+1) + i + 1 + (dx + 1) + (j-1)*(dx+1);
        e++;
      endfor
    endfor
  endfor

endfunction

function [K,F] = initialize_K_F(num_pts)
  K(1:num_pts,1:num_pts) = 0;
  F(1:num_pts) = 0;
endfunction

function [K,F] = form_matrix(K,F,points,elems,materials,dt,theta,T)
  [nr,nc] = size(elems);
  
  for (elem_num = 1:nr)
    for (j = 1:8)
      element(j) = elems(elem_num,j);
    endfor
    [KE,C] = element_linear(elem_num,element,points,materials);
    K = assemble(element,K,KE,C,dt,theta);
    F = source_term(element,F,KE,C,dt,theta,T);
  endfor
  K
  T
  F
endfunction

function [KE,Ca] = element_linear(i,element,points,materials)

  n = [element(1:8)];
  pts = [points(n)];
  
  [xi,eta,delta,weight] = gauss_quadrature;

  for (order=1:2)
    for (jor = 1:order)
      for (kor = 1:order)
        for (lor = 1:order)
          gauss_point.xi = xi(order,lor);
          gauss_point.eta = eta(order,kor);
          gauss_point.delta = delta(order,jor);
          shape(order,lor,kor,jor) = shape_linear(gauss_point,pts);
        endfor
      endfor
    endfor
  endfor

  kond = materials.kond(i);
  density = materials.density(i);
  specific_heat = materials.specific_heat(i);

  KE(1:8,1:8) = 0;
  Ca(1:8,1:8) = 0;

  Ka = Kalpha(2,shape,weight,kond); # (2x2x2) 8 gauss pt integration
  Ca = Capacitance(2,shape,weight,density,specific_heat); # (2x2x2) 8 gauss pt

  KE = Ka;

endfunction

function K = assemble(element,K,KE,C,dt,theta)
  #printf("Doing assemble\n");

  for (i=1:8)
    ii = element(i);
    for (j=1:8)
      jj = element(j);
      K(ii,jj) += (C(i,j)/dt + theta*KE(i,j));
    endfor
  endfor

endfunction

function [xi,eta,delta,weight] = gauss_quadrature

  xi(1,1,1) = 0;
  eta(1,1,1) = 0;
  delta(1,1,1) = 0;
  weight(1,1) = 2;

  part_loc = true;

   if (part_loc)
     xi(2,1) = -1/2;
     xi(2,2) = 1/2;
     eta(2,1) = -1/2;
     eta(2,2) = 1/2;
     delta(2,1) = -1/2;
     delta(2,2) = 1/2;
     weight(2,1) = weight(2,2) = 1;
  else
     xi(2,1) = -1/sqrt(3);
     xi(2,2) = 1/sqrt(3);
     eta(2,1) = -1/sqrt(3);
     eta(2,2) = 1/sqrt(3);
     delta(2,1) = -1/sqrt(3);
     delta(2,2) = 1/sqrt(3);
     weight(2,1) = weight(2,2) = 1;
  endif


  xi(3,1) = -sqrt(3/5);
  xi(3,2) = 0;
  xi(3,3) = sqrt(3/5);
  eta(3,1) = -sqrt(3/5);
  eta(3,2) = 0;
  eta(3,3) = sqrt(3/5);
  delta(3,1) = -sqrt(3/5);
  delta(3,2) = 0;
  delta(3,3) = sqrt(3/5);

  weight(3,1) = 5/9;
  weight(3,2) = 8/9;
  weight(3,3) = 5/9;

  xi(4,1) = -(1/35)*sqrt(525+70*sqrt(30));
  xi(4,2) = -(1/35)*sqrt(525-70*sqrt(30));
  xi(4,3) = (1/35)*sqrt(525-70*sqrt(30));
  xi(4,4) = (1/35)*sqrt(525+70*sqrt(30));
  eta(4,1) = -(1/35)*sqrt(525+70*sqrt(30));
  eta(4,2) = -(1/35)*sqrt(525-70*sqrt(30));
  eta(4,3) = (1/35)*sqrt(525-70*sqrt(30));
  eta(4,4) = (1/35)*sqrt(525+70*sqrt(30));
  delta(4,1) = -(1/35)*sqrt(525+70*sqrt(30));
  delta(4,2) = -(1/35)*sqrt(525-70*sqrt(30));
  delta(4,3) = (1/35)*sqrt(525-70*sqrt(30));
  delta(4,4) = (1/35)*sqrt(525+70*sqrt(30));

  weight(4,1) = (1/36)*(18-sqrt(30));
  weight(4,2) = (1/36)*(18+sqrt(30));
  weight(4,3) = (1/36)*(18+sqrt(30));
  weight(4,4) = (1/36)*(18-sqrt(30));

  xi(5,1) = -(1/21)*sqrt(245+14*sqrt(70));
  xi(5,2) = -(1/21)*sqrt(245-14*sqrt(70));
  xi(5,3) = 0;
  xi(5,4) = (1/21)*sqrt(245-14*sqrt(70));
  xi(5,5) = (1/21)*sqrt(245+14*sqrt(70));
  eta(5,1) = -(1/21)*sqrt(245+14*sqrt(70));
  eta(5,2) = -(1/21)*sqrt(245-14*sqrt(70));
  eta(5,3) = 0;
  eta(5,4) = (1/21)*sqrt(245-14*sqrt(70));
  eta(5,5) = (1/21)*sqrt(245+14*sqrt(70));
  delta(5,1) = -(1/21)*sqrt(245+14*sqrt(70));
  delta(5,2) = -(1/21)*sqrt(245-14*sqrt(70));
  delta(5,3) = 0;
  delta(5,4) = (1/21)*sqrt(245-14*sqrt(70));
  delta(5,5) = (1/21)*sqrt(245+14*sqrt(70));

  weight(5,1) = (1/900)*(322-13*sqrt(70));
  weight(5,2) = (1/900)*(322+13*sqrt(70));
  weight(5,3) = 128/225;
  weight(5,4) = (1/900)*(322+13*sqrt(70));
  weight(5,5) = (1/900)*(322-13*sqrt(70));



endfunction

function shape = shape_linear(gp,pts)
  xi = gp.xi;
  eta = gp.eta;
  delta = gp.delta;

  shape.phi(1) = 1/8*(1-xi)*(1-eta)*(1-delta);
  shape.phi(2) = 1/8*(1+xi)*(1-eta)*(1-delta);
  shape.phi(3) = 1/8*(1-xi)*(1+eta)*(1-delta);
  shape.phi(4) = 1/8*(1+xi)*(1+eta)*(1-delta);

  shape.phi(5) = 1/8*(1-xi)*(1-eta)*(1+delta);
  shape.phi(6) = 1/8*(1+xi)*(1-eta)*(1+delta);
  shape.phi(7) = 1/8*(1-xi)*(1+eta)*(1+delta);
  shape.phi(8) = 1/8*(1+xi)*(1+eta)*(1+delta);

  shape.dphidxi(1) = -1/8*(1-eta)*(1-delta);
  shape.dphideta(1) = -1/8*(1-xi)*(1-delta);
  shape.dphiddelta(1) = -1/8*(1-xi)*(1-eta);
  shape.dphidxi(2) = 1/8*(1-eta)*(1-delta);
  shape.dphideta(2) = -1/8*(1+xi)*(1-delta);
  shape.dphiddelta(2) = -1/8*(1+xi)*(1-eta);
  shape.dphidxi(3) = -1/8*(1+eta)*(1-delta);
  shape.dphideta(3) = 1/8*(1-xi)*(1-delta);
  shape.dphiddelta(3) = -1/8*(1-xi)*(1+eta);
  shape.dphidxi(4) = 1/8*(1+eta)*(1-delta);
  shape.dphideta(4) = 1/8*(1+xi)*(1-delta);
  shape.dphiddelta(4) = -1/8*(1+xi)*(1+eta);

  shape.dphidxi(5) = -1/8*(1-eta)*(1+delta);
  shape.dphideta(5) = -1/8*(1-xi)*(1+delta);
  shape.dphiddelta(5) = 1/8*(1-xi)*(1-eta);
  shape.dphidxi(6) = 1/8*(1-eta)*(1+delta);
  shape.dphideta(6) = -1/8*(1+xi)*(1+delta);
  shape.dphiddelta(6) = 1/8*(1+xi)*(1-eta);
  shape.dphidxi(7) = -1/8*(1+eta)*(1+delta);
  shape.dphideta(7) = 1/8*(1-xi)*(1+delta);
  shape.dphiddelta(7) = 1/8*(1-xi)*(1+eta);
  shape.dphidxi(8) = 1/8*(1+eta)*(1+delta);
  shape.dphideta(8) = 1/8*(1+xi)*(1+delta);
  shape.dphiddelta(8) = 1/8*(1+xi)*(1+eta);


  jac = compute_jacobian(pts,shape);
  shape.jac = jac;

  for (i=1:8)
    shape.dphidx(i) = jac.jinv(1,1)*shape.dphidxi(i) + jac.jinv(1,2)*shape.dphideta(i) + jac.jinv(1,3)*shape.dphiddelta(i);
    shape.dphidy(i) = jac.jinv(2,1)*shape.dphidxi(i) + jac.jinv(2,2)*shape.dphideta(i) + jac.jinv(2,3)*shape.dphiddelta(i);
    shape.dphidz(i) = jac.jinv(3,1)*shape.dphidxi(i) + jac.jinv(3,2)*shape.dphideta(i) + jac.jinv(3,3)*shape.dphiddelta(i);
  endfor

endfunction

function jacobian = compute_jacobian(pts,shape)

  jacobian.dxdxi = jacobian.dxdeta = jacobian.dxddelta = 0;
  jacobian.dydxi = jacobian.dydeta = jacobian.dyddelta = 0;
  jacobian.dzdxi = jacobian.dzdeta = jacobian.dzddelta = 0;
  for (i=1:8)
    jacobian.dxdxi += pts(i).x * shape.dphidxi(i);
    jacobian.dxdeta += pts(i).x * shape.dphideta(i);
    jacobian.dxddelta += pts(i).x * shape.dphiddelta(i);
    jacobian.dydxi += pts(i).y * shape.dphidxi(i);
    jacobian.dydeta += pts(i).y * shape.dphideta(i);
    jacobian.dyddelta += pts(i).y * shape.dphiddelta(i);
    jacobian.dzdxi += pts(i).z * shape.dphidxi(i);
    jacobian.dzdeta += pts(i).z * shape.dphideta(i);
    jacobian.dzddelta += pts(i).z * shape.dphiddelta(i);
  endfor
  
  jac = [[jacobian.dxdxi jacobian.dydxi jacobian.dzdxi];
         [jacobian.dxdeta jacobian.dydeta jacobian.dzdeta];
         [jacobian.dxddelta jacobian.dyddelta jacobian.dzddelta]];

  jacobian.mag = jacobian.dxdxi* jacobian.dydeta * jacobian.dzddelta - jacobian.dxdeta * jacobian.dydxi* jacobian.dzddelta - jacobian.dxdxi * jacobian.dyddelta* jacobian.dzdeta + jacobian.dxddelta * jacobian.dydxi* jacobian.dzdeta +  jacobian.dxdeta * jacobian.dyddelta* jacobian.dzdxi -  jacobian.dxddelta * jacobian.dydeta* jacobian.dzdxi;

  jacobian.jinv(1,1) = 1/jacobian.mag * (jacobian.dydeta*jacobian.dzddelta - jacobian.dzdeta*jacobian.dyddelta);
  jacobian.jinv(1,2) = 1/jacobian.mag * (jacobian.dzdxi*jacobian.dyddelta - jacobian.dydxi*jacobian.dzddelta);
  jacobian.jinv(1,3) = 1/jacobian.mag * (jacobian.dydxi*jacobian.dzdeta - jacobian.dzdxi*jacobian.dydeta);
  jacobian.jinv(2,1) = 1/jacobian.mag * (jacobian.dzdeta*jacobian.dxddelta-jacobian.dxdeta*jacobian.dzddelta);
  jacobian.jinv(2,2) = 1/jacobian.mag * (jacobian.dxdxi*jacobian.dzddelta - jacobian.dzdxi*jacobian.dxddelta);
  jacobian.jinv(2,3) = 1/jacobian.mag * (jacobian.dzdxi*jacobian.dxdeta - jacobian.dxdxi * jacobian.dzdeta);
  jacobian.jinv(3,1) = 1/jacobian.mag * (jacobian.dxdeta *jacobian.dyddelta - jacobian.dydeta*jacobian.dxddelta);
  jacobian.jinv(3,2) = 1/jacobian.mag * (jacobian.dydxi*jacobian.dxddelta-jacobian.dxdxi* jacobian.dyddelta);
  jacobian.jinv(3,3) = 1/jacobian.mag * (jacobian.dxdxi*jacobian.dydeta - jacobian.dydxi*jacobian.dxdeta);


endfunction


function Ka = Kalpha(order,shape,weight,kond)
#  printf("Doing Kalpha\n");
  for(i=1:8)
    for(j=1:8)
      value = 0;
      for (jor=1:order)
        for (kor=1:order)
          for (lor=1:order)
#            value += weight(order,jor)*weight(order,kor)*weight(order,lor)*((shape(order,lor,kor,jor).jac.mag)*(shape(order,lor,kor,jor).dphidx(i)*kond*shape(order,lor,kor,jor).dphidx(j) ));
            value += weight(order,jor)*weight(order,kor)*weight(order,lor)*((shape(order,lor,kor,jor).jac.mag)*(shape(order,lor,kor,jor).dphidx(i)*kond*shape(order,lor,kor,jor).dphidx(j) + shape(order,lor,kor,jor).dphidy(i)*kond*shape(order,lor,kor,jor).dphidy(j) + shape(order,lor,kor,jor).dphidz(i)*kond*shape(order,lor,kor,jor).dphidz(j)));
          endfor
        endfor
      endfor
      Ka(i,j) = value;
    endfor
  endfor
    
endfunction

function Kbeta()
  printf("Doing Kbeta\n");
endfunction

function C = Capacitance(order,shape,weight,density,specific_heat)
   # printf("Doing Capacitance\n");
    for(i=1:8)
      for(j=1:8)
        value = 0;
        for (jor=1:order)
          for (kor=1:order)
            for (lor=1:order)
              value += weight(order,jor)*weight(order,kor)*weight(order,lor)*(shape(order,lor,kor,jor).jac.mag)*shape(order,lor,kor,jor).phi(i)*density*specific_heat*shape(order,lor,kor,jor).phi(j);
            endfor
          endfor
        endfor
        C(i,j) = value;
      endfor
    endfor

endfunction

function F = source_term(element,F,KE,C,dt,theta,T)
 # printf("Doing source term\n");
  for (i=1:8)
    ii = element(i);
    for (j=1:8)
      jj = element(j);
      F(ii) += (C(i,j)/dt - (1 -theta)*KE(i,j))*T(jj);
    endfor
  endfor

endfunction

function MP = generate_material_points(points,elems)

  for (elem_num=1:nr)

    for (j=1:8)
      element(j) = elems(elem_num,j);
    endfor

    pts = [points(element(1:8))];

    order = 2;

    for (jor = 1:order)
      for (kor = 1:order)
        for (lor = 1:order)
          gauss_point.xi = xi(order,lor);
          gauss_point.eta = eta(order,kor);
          gauss_point.delta = delta(order,jor);
          shape(order,lor,kor,jor) = shape_linear(gauss_point,pts);
        endfor
      endfor
    endfor
    
    for (jor=1:order)
      for (kor=1:order)
        for (lor=1:order)
          gp.xi = xi(order,lor);
          gp.eta = eta(order,kor);
          gp.delta = delta(order,jor);
          mp.x = mp.y = mp.z = 0;
          for (i=1:8)
            mp.x += shape(order,lor,kor,jor).phi(i)*pts(i).x;
            mp.y += shape(order,lor,kor,jor).phi(i)*pts(i).y;
            mp.z += shape(order,lor,kor,jor).phi(i)*pts(i).z;
          endfor
          MP(count++) = mp;
        endfor
      endfor
    endfor

  endfor  

  
endfunction

function found_it = find_element(points,elements,tp)
  [nr,nc] = size(elements);

  for (i=1:nr)
    element = elements(i,1:8);
    pts = [points(element(1:8))];

    if (tp.x >= pts(1).x && tp.y >= pts(1).y && tp.z >= pts(1).z && tp.x <= pts(8).x && tp.y <= pts(8).y && tp.z <= pts(8).z)
      index = i;
    endif

  endfor
  found_it = elements(index,1:8);

endfunction


function [pts,shape] = findNodesAndWeights(points,elements,material_point)

  element = find_element(points,elements,material_point);

  pts = [points(element(1:8))];

  mp.xi = material_point.x;
  mp.eta = material_point.y;
  mp.delta = material_point.z;
  shape = shape_linear(mp,pts);

endfunction

function bcs = generate_bcs(bar_dim,spacing)

  xface.l.type = input("input bc type for x left face (D)irchlet or (N)eumann ","s");
  xface.l.value = input("input bc value for x left face ");
  xface.r.type = input("input bc type for x right face (D)irchlet or (N)eumann ","s");
  xface.r.value = input("input bc value for x right face ");

  yface.b.type = input("input bc type for y bottom face (D)irchlet or (N)eumann ","s");
  yface.b.value = input("input bc value for y bottom face ");
  yface.t.type = input("input bc type for y top face (D)irchlet or (N)eumann ","s");
  yface.t.value = input("input bc value for y top face ");

  zface.f.type = input("input bc type for z front face (D)irchlet or (N)eumann ","s");
  zface.f.value = input("input bc value for z front face ");
  zface.b.type = input("input bc type for z back face (D)irchlet or (N)eumann ","s");
  zface.b.value = input("input bc value for z back face ");

  dx = bar_dim.x/spacing.x;
  dy = bar_dim.y/spacing.y;
  dz = bar_dim.z/spacing.z;
  count = 1;

  #left face (x minus case)
  if (xface.l.type == "D" || xface.l.type == "d" || xface.l.type == "N" ||
      xface.l.type == "n")
    for (j = 1:dy+1)
      i = 1;
      bcs.n(count) = i + (j - 1)*(dx + 1);
      bcs.t(count) = xface.l.type;
      bcs.v(count++) = xface.l.value;
      bcs.n(count) = (dx+1)*(dy+1) + i + (j - 1)*(dx+1);
      bcs.t(count) = xface.l.type;
      bcs.v(count++) = xface.l.value;
    endfor
  endif

  #right face (x plus case)
  if (xface.r.type == "D" || xface.r.type == "d" || xface.r.type == "N" || 
      xface.r.type == "n")
    for (j = 1:dy+1)
      i = dx + 1;
      bcs.n(count) = i + (j - 1)*(dx + 1);
      bcs.t(count) = xface.r.type;
      bcs.v(count++) = xface.r.value;
      bcs.n(count) = (dx+1)*(dy+1) + i + (j - 1)*(dx + 1);
      bcs.t(count) = xface.r.type;
      bcs.v(count++) = xface.r.value;
    endfor
  endif

  #bottom face (y minus case)
  if (yface.b.type == "D" || yface.b.type == "d" || yface.b.type == "N" ||
      yface.b.type == "n")
    for (i = 1:dx+1)
      j = 1;
      bcs.n(count) = i + (j-1)*(dx+1);
      bcs.t(count) = yface.b.type;
      bcs.v(count++) = yface.b.value;
      bcs.n(count) = (dx+1)*(dy+1) + i + (j-1)*(dx+1);
      bcs.t(count) = yface.b.type;
      bcs.v(count++) = yface.b.value;
    endfor
  endif

  #top face (y plus case)
  if (yface.t.type == "D" || yface.t.type == "d" || yface.t.type == "N" ||
      yface.t.type == "n")
    for (i = 1:dx+1)
      j = dy;
      bcs.n(count) = i + (j-1)*(dx+1);
      bcs.t(count) = yface.t.type;
      bcs.v(count++) = yface.t.value;
      bcs.n(count) = (dx+1)*(dy+1) + i + (j-1)*(dx+1);
      bcs.t(count) = yface.t.type;
      bcs.v(count++) = yface.t.value;
    endfor
  endif

#FIX for z
  #front face (z minus case)
  if (zface.f.type == "D" || zface.f.type == "d" || zface.f.type == "N" ||
      zface.f.type == "n")
    for (i = 1:dx+1)
      j = 1;
      bcs.n(count) = i + (j-1)*(dx+1);
      bcs.t(count) = zface.f.type;
      bcs.v(count++) = zface.f.value;
      bcs.n(count) = (dx+1)*(dy+1)+ i + (j-1)*(dx+1);
      bcs.t(count) = zface.f.type;
      bcs.v(count++) = zface.f.value;
    endfor
  endif

  #back face (z plus case)
  if (zface.b.type == "D" || zface.b.type == "d" || zface.b.type == "N" ||
      zface.b.type == "n")
    for (i = 1:dx+1)
      j = dy;
      bcs.n(count) = i + (j-1)*(dx+1);
      bcs.t(count) = zface.b.type;
      bcs.v(count++) = zface.b.value;
      bcs.n(count) = (dx+1)*(dy+1) + i + (j-1)*(dx+1);
      bcs.t(count) = zface.b.type;
      bcs.v(count++) = zface.b.value;
    endfor
  endif

endfunction



function [K,F] = apply_bcs(K,F,bcs)
   # printf("Doing apply_bcs\n");

    [nr,nc]=size(K);

    bc_size = length(bcs.n);
    printf("BEFORE applying bcs\n");
    bcs
    K
    F

    for (bc=1:bc_size)
      node=bcs.n(bc);
      bcvalue = bcs.v(bc);
      bctype = bcs.t(bc);
      if (bctype == "N" || bctype == "n")
        F(node) += bcvalue;
      endif
      if (bctype == "D" || bctype == "d")
        for (j=1:nr)
          kvalue = K(j,node);
          F(j) -= kvalue*bcvalue;
          if (j != node)
	    K(j,node) = 0;
            K(node,j) = 0;
          else
            K(j,node) = 1;
          endif
        endfor
        F(node) = bcvalue;
      endif
    endfor
    F = F';

    printf("After applying bcs\n");
    K
    F
endfunction

function a = solve_system(K,F)
#printf("Doing solve_system\n");
  a = K \ F;
endfunction

function mat = define_materials()

  num_mat = input("input number of materials ");
  for (i=1:num_mat)
    mat.le(i).x = input("input left edge - x ");
    mat.le(i).y = input("input left edge - y ");
    mat.le(i).z = input("input left edge - z ");
    mat.re(i).x = input("input right edge - x ");
    mat.re(i).y = input("input right edge - y ");
    mat.re(i).z = input("input right edge - z ");
    mat.kond(i) = input("input thermal conductivity ");
    mat.density(i) = input("input density ");
    mat.specific_heat(i) = input("input specific heat ");
  endfor

endfunction

function materials = create_materials_element(p,e,mat)
  [nr,nc] = size(e);

  for (elem = 1:nr)
    center.x = center.y = center.z = 0;
    for (j = 1:8)
      n = e(elem,j);
      center.x += (p(n).x)/8;
      center.y += (p(n).y)/8;
      center.z += (p(n).z)/8;
    endfor
    for (i=1:length(mat.kond))
      if (mat.le(i).x <= center.x && center.x <= mat.re(i).x && mat.le(i).y <= center.y && center.y <= mat.re(i).y && mat.le(i).z <= center.z && center.z <= mat.re(i).z)
        materials.kond(elem) = mat.kond(i);
        materials.density(elem) = mat.density(i);
        materials.specific_heat(elem) = mat.specific_heat(i);
      endif
    endfor

  endfor

endfunction

function T = set_intial_condition(initial_temp,num)
  T(1:num) = initial_temp;
endfunction


function [MP,Temp_mp] = interpolate_temperature_to_gauss_points(points,elems,T)

  [xi,eta,delta,weight] = gauss_quadrature;

  [nr,nc] = size(elems);
  count = 1;

  for (elem_num=1:nr)
    for (j=1:8)
      element(j) = elems(elem_num,j);
      T_elem(j) = T(element(j));
    endfor

    pts = [points(element(1:8))];

    order = 2;

    for (jor = 1:order)
      for (kor = 1:order)
        for (lor = 1:order)
          gauss_point.xi = xi(order,lor);
          gauss_point.eta = eta(order,kor);
          gauss_point.delta = delta(order,jor);
          shape(order,lor,kor,jor) = shape_linear(gauss_point,pts);
        endfor
      endfor
    endfor
    
    for (jor=1:order)
      for (kor=1:order)
        for (lor=1:order)
          gp.xi = xi(order,lor);
          gp.eta = eta(order,kor);
          gp.delta = delta(order,jor);
          Temp = 0;
          mp.x = mp.y = mp.z = 0;
          for (i=1:8)
            Temp += shape(order,lor,kor,jor).phi(i)*T_elem(i);
            mp.x += shape(order,lor,kor,jor).phi(i)*pts(i).x;
            mp.y += shape(order,lor,kor,jor).phi(i)*pts(i).y;
            mp.z += shape(order,lor,kor,jor).phi(i)*pts(i).z;
          endfor
          Temp_mp(count)=Temp;
          MP(count++) = mp;
        endfor
      endfor
    endfor

  endfor

endfunction

function Tnodes = interpolate_MP_temperatures_to_nodes(points,elems,MP,Temp_mp)

  np = length(points);
  Tnodes(1:np) = 0;

  for (i=1:length(MP))
    element = find_element(points,elems,MP(i))
    [pts,shape] = findNodesAndWeights(points,elems,MP(i));

    for (j=1:8)
      Temp_mp(i)
      shape.phi(j)
      Tnodes(element(j)) += Temp_mp(i)*shape.phi(j)
    endfor

  endfor

endfunction

function main()

  bar.x = input("input size of bar in x dimension ");
  bar.y = input("input size of bar in y dimension ");
  bar.z = input("input size of bar in z dimension ");
  spacing.x = input("input grid spacing in x dimension ");
  spacing.y = input("input grid spacing in y dimension ");
  spacing.z = input("input grid spacing in z dimension ");

  [p,e] = make_grid(bar,spacing);
  
  bcs = generate_bcs(bar,spacing);

  initial_temp = input("input initial temperature ");

  theta = input("input theta (0 = explicit, .5 = midpoint, 1 = implicit  ");
  dt = 0;
  lump = true;
  if (theta < .5)
    dt = input("input dt ");
    lump = input("input lumping (true/false) ");
  endif
  end_time = input("input end time ");
  if (dt == 0)
    dt = input("input time step size ");
  endif
  
  mat = define_materials();

  if (theta < .5)
    kond_max = max(mat.kond);
    kond_min = min(mat.kond);
    spec_heat_max = max(mat.specific_heat);
    spec_heat_min = min(mat.specific_heat);
    density_max = max(mat.density);
    density_min = min(mat.density);
    
    lamba_max = (pi/spacing)^2 * kond_max/(spec_heat_max*density_max)
    dt_critical = 2/(1-2*theta) * 1/lamba_max
    
    if (dt > dt_critical)
      dt = .9*dt_critical;
    endif
  endif
    
  printf("Using dt = %f\n",dt);

  materials = create_materials_element(p,e,mat);
  T = set_intial_condition(initial_temp,length(p));

  
  t = 0;
  while (t <= end_time)
    [K,F] = initialize_K_F(length(p));
    [Keff,Feff] = form_matrix(K,F,p,e,materials,dt,theta,T);
    [Keff,Feff] = apply_bcs(Keff,Feff,bcs,materials);
    T = solve_system(Keff,Feff)
#    [MP, Temp_mp] = interpolate_temperature_to_gauss_points(p,e,T);
#    MP
#    Temp_mp
#    Tnodes = interpolate_MP_temperatures_to_nodes(p,e,MP,Temp_mp);
#    Tnodes
    xlabel("Bar points");
    ylabel("Temperature");
    plot_title = "Temperature at ";
    plot_time = num2str(t);
    title(strcat(plot_title,plot_time," seconds"));
    #contour(p,T)
    t += dt;
  endwhile
endfunction

