1;

function [points,elems] = make_grid(bar_dim,spacing)
  count = 1;
  x = y = 0;
  while (y <= bar_dim.y)
    while (x <= bar_dim.x)
      points(count).x = x;
      points(count++).y = y;
      x += spacing.x;
    endwhile
    x = 0;
    y += spacing.y;
  endwhile

  dx = bar_dim.x/spacing.x;
  dy = bar_dim.y/spacing.y;

  e=1;
  for (j = 1:dy)
    for (i = 1:dx)
      elems(e,1) = i + (j-1)*(dx+1);
      elems(e,2) = i + (j-1)*(dx+1) + 1;
      elems(e,3) = i + (dx + 1)  + (j-1)*(dx+1);
      elems(e,4) = i + 1 + (dx + 1) + (j-1)*(dx+1);
      e++;
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
    for (j = 1:4)
      element(j) = elems(elem_num,j);
    endfor
    [KE,C] = element_linear(elem_num,element,points,materials);
    K = assemble(element,K,KE,C,dt,theta);
    F = source_term(element,F,KE,C,dt,theta,T);
  endfor
endfunction

function [KE,Ca] = element_linear(i,element,points,materials)
  n1 = element(1);
  n2 = element(2);
  n3 = element(3);
  n4 = element(4);

  pts = [points(n1),points(n2),points(n3),points(n4)];

  [xi,eta,weight] = gauss_quadrature;

  for (order=1:2)
    for (kor = 1:order)
      for (lor = 1:order)
        gauss_point.xi = xi(order,lor);
        gauss_point.eta = eta(order,kor);
        shape(order,lor,kor) = shape_linear(gauss_point,pts);
      endfor
    endfor
  endfor
  
  kond = materials.kond(i);
  density = materials.density(i);
  specific_heat = materials.specific_heat(i);

  KE(1:4,1:4) = 0;
  Ca(1:4,1:4) = 0;

  Ka = Kalpha(2,shape,weight,kond); # (2x2) 2 gauss pt integration
  Ca = Capacitance(2,shape,weight,density,specific_heat); # (2x2) 2 gauss pt

  KE = Ka;

endfunction

function K = assemble(element,K,KE,C,dt,theta)
  #printf("Doing assemble\n");

  for (i=1:4)
    ii = element(i);
    for (j=1:4)
      jj = element(j);
      K(ii,jj) += (C(i,j)/dt + theta*KE(i,j));
    endfor
  endfor

endfunction

function [xi,eta,weight] = gauss_quadrature
  xi(1,1) = 0;
  eta(1,1) = 0;
  weight(1,1) = 2;

  xi(2,1) = -1/sqrt(3);
  xi(2,2) = 1/sqrt(3);
  eta(2,1) = -1/sqrt(3);
  eta(2,2) = 1/sqrt(3);
  weight(2,1) = weight(2,2) = 1;

  xi(3,1) = -sqrt(3/5);
  xi(3,2) = 0;
  xi(3,3) = sqrt(3/5);
  eta(3,1) = -sqrt(3/5);
  eta(3,2) = 0;
  eta(3,3) = sqrt(3/5);

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

  weight(5,1) = (1/900)*(322-13*sqrt(70));
  weight(5,2) = (1/900)*(322+13*sqrt(70));
  weight(5,3) = 128/225;
  weight(5,4) = (1/900)*(322+13*sqrt(70));
  weight(5,5) = (1/900)*(322-13*sqrt(70));


  
endfunction

function shape = shape_linear(gp,pts)
  xi = gp.xi;
  eta = gp.eta;

  shape.phi(1) = 1/4*(1-xi)*(1-eta);
  shape.phi(2) = 1/4*(1+xi)*(1-eta);
  shape.phi(3) = 1/4*(1-xi)*(1+eta);
  shape.phi(4) = 1/4*(1+xi)*(1+eta);

  shape.dphidxi(1) = 1/4*(eta - 1);
  shape.dphideta(1) = 1/4*(xi -1);
  shape.dphidxi(2) = 1/4*(1-eta);
  shape.dphideta(2) = -1/4*(1+xi);
  shape.dphidxi(3) = -1/4*(1+eta);
  shape.dphideta(3) = 1/4*(1-xi);
  shape.dphidxi(4) = 1/4*(1+eta);
  shape.dphideta(4) = 1/4*(1+xi);

  jac = compute_jacobian(pts,shape);
  shape.jac = jac;

  for (i=1:4)
    shape.dphidx(i) = jac.jinv(1,1)*shape.dphidxi(i) + jac.jinv(1,2)*shape.dphideta(i);
    shape.dphidy(i) = jac.jinv(2,1)*shape.dphidxi(i) + jac.jinv(2,2)*shape.dphideta(i);
  endfor

endfunction

function jacobian = compute_jacobian(pts,shape)
  
  jacobian.dxdxi = jacobian.dxdeta = jacobian.dydxi = jacobian.dydeta = 0;
  for (i=1:4)
    jacobian.dxdxi += pts(i).x * shape.dphidxi(i);
    jacobian.dxdeta += pts(i).x * shape.dphideta(i);
    jacobian.dydxi += pts(i).y * shape.dphidxi(i);
    jacobian.dydeta += pts(i).y * shape.dphideta(i);
  endfor

  jac = [[jacobian.dxdxi jacobian.dydxi];
         [jacobian.dxdeta jacobian.dydeta]];

  
  jacobian.mag = jacobian.dxdxi * jacobian.dydeta - jacobian.dydxi* jacobian.dxdeta;

  jacobian.jinv(1,1) = 1/jacobian.mag * jacobian.dydeta;
  jacobian.jinv(1,2) = -1/jacobian.mag * jacobian.dydxi;
  jacobian.jinv(2,1) = -1/jacobian.mag * jacobian.dxdeta;
  jacobian.jinv(2,2) = 1/jacobian.mag * jacobian.dxdxi;

endfunction


function Ka = Kalpha(order,shape,weight,kond)
#  printf("Doing Kalpha\n");
  for(i=1:4)
    for(j=1:4)
      value = 0;
      for (kor=1:order)
        for (lor=1:order)
#          value += weight(order,kor)*weight(order,lor)*((shape(order,lor,kor).jac.mag)*(shape(order,lor,kor).dphidy(i)*kond*shape(order,lor,kor).dphidy(j) ));
          value += weight(order,kor)*weight(order,lor)*((shape(order,lor,kor).jac.mag)*(shape(order,lor,kor).dphidx(i)*kond*shape(order,lor,kor).dphidx(j) + shape(order,lor,kor).dphidy(i)*kond*shape(order,lor,kor).dphidy(j)));
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

    for(i=1:4)
      for(j=1:4)
        value = 0;
        for (kor=1:order)
          for (lor=1:order)
            value += weight(order,kor)*weight(order,lor)*(shape(order,lor,kor).jac.mag)*shape(order,lor,kor).phi(i)*density*specific_heat*shape(order,lor,kor).phi(j);
          endfor
        endfor
        C(i,j) = value;
      endfor
    endfor

endfunction

function F = source_term(element,F,KE,C,dt,theta,T)
 # printf("Doing source term\n");
  for (i=1:4)
    ii = element(i);
    for (j=1:4)
      jj = element(j);
      F(ii) += (C(i,j)/dt - (1 -theta)*KE(i,j))*T(jj);
    endfor
  endfor

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

  dx = bar_dim.x/spacing.x;
  dy = bar_dim.y/spacing.y;
  count = 1;

  #left face (x minus case)
  if (xface.l.type == "D" || xface.l.type == "d" || xface.l.type == "N" ||
      xface.l.type == "n")
    for (j = 1:dy+1)
      i = 1;
      bcs.n(count) = i + (j - 1)*(dx + 1);
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
    endfor
  endif

  #top face (y top case)
  if (yface.t.type == "D" || yface.t.type == "d" || yface.t.type == "N" || 
      yface.t.type == "n")
    for (i = 1:dx+1)
      j = dy;
      bcs.n(count) = i + (j-1)*(dx+1);
      bcs.t(count) = yface.t.type;
      bcs.v(count++) = yface.t.value;
    endfor
  endif
  
endfunction



function [K,F] = apply_bcs(K,F,bcs)
   # printf("Doing apply_bcs\n");

    [nr,nc]=size(K);

    bc_size = length(bcs.n);

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
    mat.re(i).x = input("input right edge - x ");
    mat.re(i).y = input("input right edge - y ");
    mat.kond(i) = input("input thermal conductivity ");
    mat.density(i) = input("input density ");
    mat.specific_heat(i) = input("input specific heat ");
  endfor

endfunction

function materials = create_materials_element(p,e,mat)
  [nr,nc] = size(e);

  for (elem = 1:nr)
    n1 = e(elem,1);
    n2 = e(elem,2);
    n3 = e(elem,3);
    n4 = e(elem,4);
    center.x = (p(n1).x + p(n2).x + p(n3).x + p(n4).x)/4;
    center.y = (p(n1).y + p(n2).y + p(n3).y + p(n4).y)/4;
    for (i=1:length(mat.kond))
      if (mat.le(i).x <= center.x && center.x <= mat.re(i).x && mat.le(i).y <= center.y && center.y <= mat.re(i).y)
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


function main()

  bar.x = input("input size of bar in x dimension ");
  bar.y = input("input size of bar in y dimension ");
  spacing.x = input("input grid spacing in x dimension ");
  spacing.y = input("input grid spacing in y dimension ");

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
    xlabel("Bar points");
    ylabel("Temperature");
    plot_title = "Temperature at ";
    plot_time = num2str(t);
    title(strcat(plot_title,plot_time," seconds"));
    #contour(p,T)
    t += dt;
  endwhile
endfunction

