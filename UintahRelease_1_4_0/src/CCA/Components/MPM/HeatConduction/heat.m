
# Solve the 1d heat equation:
# rho*Cp*du/dt - k d^2/dx^2 (u) = 0, 
# u = temperature 
# k = thermal conductivity.

# form matrics
# apply bounday conditions
# solve system

1;

function [points, elems] = make_grid(bar_dim,resolution)
  count = 1;
  j = 0;
  while (j <= bar_dim)
    points(count++) = j;
    j += bar_dim/resolution;
  endwhile
  if (points(length(points)) != bar_dim)
    points(count) = bar_dim;
  endif

  e = 1;
  for (ro=1:length(points)-1)
    elems(ro,1) = e;
    elems(ro,2) = ++e;
  endfor
  
endfunction

function [K,F] = initialize_K_F(num_pts)
  K(1:num_pts,1:num_pts) = 0;
  F(1:num_pts) = 0;
endfunction

function [K,F] = form_matrix(K,F,points,elems,materials,dt,theta,T)
  [nr,nc] = size(elems);

  for (elem_num = 1:nr)
    element(1) = elems(elem_num,1);
    element(2) = elems(elem_num,2);
    [KE,C] = element_linear(elem_num,element,points,materials);
#    KE
#    C
    K = assemble(element,K,KE,C,dt,theta);
    F = source_term(element,F,KE,C,dt,theta,T);
  endfor
#  K
#  F

endfunction

function [KE,Ca] = element_linear(i,element,points,materials)
#  printf("Doing element linear\n");
  n1 = element(1);
  n2 = element(2);
  pts = [points(n1),points(n2)];

  [xi,weight] = gauss_quadrature;  
  # This probably only needs to loop over the highest order integration that
  # we will ultimately do. For linear elements, we use 2nd order integration
  # for the capacitive matrix term.
  for (order=1:3)
    for (or=1:order)
      gauss_point = xi(order,or);
      shape(order,or) = shape_linear(gauss_point,pts);
    endfor
  endfor


  kond = materials.kond(i);
  density = materials.density(i);
  specific_heat = materials.specific_heat(i);

  KE(1:2,1:2)=0;

  Ka = Kalpha(3,shape,weight,kond); # 1 gauss point for integration
  Ca = Capacitance(3,shape,weight,density,specific_heat); # 2 gauss pt integ

  KE = Ka;
   
endfunction

function K = assemble(element,K,KE,C,dt,theta)
  #printf("Doing assemble\n");

  for (i=1:2)
    ii = element(i);
    for (j=1:2)
      jj = element(j);
      K(ii,jj) += (C(i,j)/dt + theta*KE(i,j));
    endfor
  endfor

endfunction

function [xi,weight] = gauss_quadrature
 # printf("Doing gauss_quadrature\n");
  xi(1,1) = 0;
  weight(1,1) = 2;
  
#  xi(2,1) = -1/2;
#  xi(2,2) = -xi(2,1);
#  weight(2,1) = 1;
#  weight(2,2) = weight(2,1);

  xi(2,1) = -1/sqrt(3);
  xi(2,2) = -xi(2,1);
  weight(2,1) = 1;
  weight(2,2) = weight(2,1);
  
  
  xi(3,1) = -sqrt(3/5);
  xi(3,2) = 0;
  xi(3,3) = -xi(3,1);
  weight(3,1) = 5/9;
  weight(3,2) = 8/9;
  weight(3,3) = weight(3,1);

endfunction

function shape = shape_linear(xi,pts)
  #printf("Doing shape_linear\n");

  shape.phi(1) = 1/2 * (1-xi);
  shape.phi(2) = 1/2 * (1+xi);

  shape.dphidxi(1) = -1/2;
  shape.dphidxi(2) = 1/2;
  
  shape.jac = compute_jacobian(pts);

  shape.dphidx(1) = shape.dphidxi(1) * 1/shape.jac;
  shape.dphidx(2) = shape.dphidxi(2) * 1/shape.jac;

endfunction

function jac = compute_jacobian(pts)
 # printf("Doing compute_jacobian\n");

  jac = 1/2 *(pts(2) - pts(1));
  
endfunction

function Ka = Kalpha(order,shape,weight,kond)
#  printf("Doing Kalpha\n");
  for(i=1:2)
    for(j=1:2)
      value = 0;
      for (or=1:order)
        value += weight(order,or)*(shape(order,or).jac)*shape(order,or).dphidx(i)*kond*shape(order,or).dphidx(j);
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

    for(i=1:2)
      for(j=1:2)
        value = 0;
        for (or=1:order)
          value += weight(order,or)*(shape(order,or).jac)*shape(order,or).phi(i)*density*specific_heat*shape(order,or).phi(j);
        endfor
        C(i,j) = value;
      endfor
    endfor

endfunction

function F = source_term(element,F,KE,C,dt,theta,T)
 # printf("Doing source term\n");
  for (i=1:2)
    ii = element(i);
    for (j=1:2)
      jj = element(j);
      F(ii) += (C(i,j)/dt - (1 -theta)*KE(i,j))*T(jj);
    endfor
  endfor


endfunction

function bcs = generate_bcs(bc,p)
  #printf("Doing generate_bcs\n");
  bcs.n(1) = 1;
  bcs.v(1) = bc.left.value;
  bcs.t(1) = bc.left.type;
  bcs.n(2) = length(p);
  bcs.v(2) = bc.right.value;
  bcs.t(2) = bc.right.type;
  
endfunction

function [K,F] = apply_bcs(K,F,bcs)
   # printf("Doing apply_bcs\n");

    [nr,nc]=size(K);

    for (bc=1:2)
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
    mat.le(i) = input("input left edge ");
    mat.re(i) = input("input right edge ");
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
    center = (p(n1) + p(n2))/2;
    for (i=1:length(mat.kond))
      if (mat.le(i) <= center && center <= mat.re(i))
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

function Texact = exact_solution(x,t,n,bc,initial_temp,bar,mat)

  C = initial_temp - bc.left.value; # f(x) - U_0
  D = bc.right.value - bc.left.value;     # U_l - U_0
  c_p = mat.specific_heat(1);
  rho = mat.density(1);
  k = mat.kond(1);

  T_ic = 0;
  for (i=1:n)
    A = (2/(pi*i))*(1-cos(i*pi))*C - 2*D/(pi^2*i^2)*(sin(pi*i) - pi*i*cos(pi*i));
    T_ic += A*sin(i*pi*x/bar)*exp(-i^2 * pi^2 * t * k/(c_p*rho*bar^2));
  endfor

  T_ss = bc.left.value + D/bar * x;

  Texact = T_ss + T_ic; 

endfunction


function main()
  
  bar = input("input size of bar ");
  resolution = input("input grid resolution ");
  bc.left.value = input("input left bc value ");
  bc.left.type = input("input left bc type (D)irchlet or (N)eumann ","s");
  bc.right.value = input("input right bc value ");
  bc.right.type = input("input right bc type (D)irchlet or (N)eumann ","s");

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
    
    lamba_max = (pi/(bar/resolution))^2 * kond_max/(spec_heat_max*density_max)
    dt_critical = 2/(1-2*theta) * 1/lamba_max
    
    if (dt > dt_critical)
      dt = .9*dt_critical;
    endif
  endif
    
  printf("Using dt = %f\n",dt);

  [p,e] = make_grid(bar,resolution);
  bcs = generate_bcs(bc,p);
  materials = create_materials_element(p,e,mat);
  T = set_intial_condition(initial_temp,length(p));

  mid_pt = ceil(length(p)/2)

  t = 0;
  file_name = input("input name of data file ","s");
  data_file = fopen(file_name,"w","native");

  while (t <= end_time)
    xlabel("Bar points");
    ylabel("Temperature");
    plot_title = "Temperature at ";
    plot_time = num2str(t);
    title(strcat(plot_title,plot_time," seconds"));
    hold off;
    plot(p,T);
    hold on;
    Tex = exact_solution(p,t,30,bc,initial_temp,bar,mat);
#    plot(p,Tex,'x')
    printf("At %f, T = %f, Texact = %f\n",p(mid_pt),T(mid_pt),Tex(mid_pt));
#    T(mid_pt)
#    pl=input('hit return');
    [K,F] = initialize_K_F(length(p));
    [Keff,Feff] = form_matrix(K,F,p,e,materials,dt,theta,T);
#    Feff
    [Keff,Feff] = apply_bcs(Keff,Feff,bcs,materials);
#    Keff
#    Feff
    T = solve_system(Keff,Feff);
    t
    T
    for (i=1:length(p))
      fprintf(data_file,"%f, %f %f\n",p(i),T(i),Tex(i));
    endfor
    fprintf(data_file,"\n");
    
    
    t += dt;
  endwhile
  fclose(data_file);
endfunction

