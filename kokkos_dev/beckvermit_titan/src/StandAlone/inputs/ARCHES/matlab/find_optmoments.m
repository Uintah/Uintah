% Work on DQMOM matrix condition number

clear all

CC = 1e99;
CCgood = 1e99;
% Moment matrix definition

Nic = input('Input the number of internal coordinates: ');
Nqn = input('Input the number of quadrature nodes: '); 

Num_trys = 500; 

fprintf('(running with a default of %u total attempts)\n', Num_trys);



for ii = 1:Num_trys
  
    A = zeros((Nic+1)*Nqn,(Nic+1)*Nqn);
    Moments = zeros(1,Nic);
    for k =1:1
        Moments = cat(1,Moments, k.*eye(Nic));
    end
    %Moments = [0,0,0,0;0,0,0,1;0,0,1,0;0,1,0,0;1,0,0,0];
    Moments = cat(1,Moments, randi([0,3],[((Nqn-1)*(Nic+1)),Nic]));
    
    %Moments = randi([0,2],[(Nqn*(Nic+1)),Nic]);
       
    % Optimal Abscissas
    Weights = ones(1,Nqn);

    for kk =1:50
        X = 2.*randi([0,1],[Nic,Nqn])-ones(Nic,Nqn);
        %X = [1;1;1;1];
        Abscissas = X(1,:);
        for i =2:Nic
          Abscissas = cat(2,Abscissas,X(i,:));
        end

        models = ones((Nic+1)*Nqn,1);

        % Constructing A
        [n,m] = size(Moments);
        for k = 1:n
            % weights
            for alpha = 1:Nqn
                prefixA = 1;
                productA = 1;
                for i = 1:m
                    prefixA = prefixA - Moments(k,i);
                    base = Abscissas((i-1)*Nqn+alpha);
                    exponent = Moments(k,i);
                    productA = productA*(base^exponent);
                end
                A(k,alpha) = prefixA*productA;
            end
            % weighted abscissas
            totalsumS = 0;
            for j = 1:Nic
                prefixA = 1;
                productA =1;
                prefixS = 1;
                productS =1;
                modelsumS = 0;
                quadsumS = 0;
                for alpha = 1:Nqn
                    base = Abscissas((j-1)*Nqn+alpha);
                    exponent = Moments(k,j)-1;
                    prefixA = Moments(k,j)*(base^exponent);
                    productA =1;
                    prefixS = -Moments(k,j)*(base^exponent);
                    productS = 1;
                    for nn = 1:Nic
                        if(nn~=j)
                            base2 = Abscissas((nn-1)*Nqn+alpha);
                            %base2 = weightedAbscissas((nn-1)*Nqn+alpha)/Weights0(alpha);
                            exponent2 = Moments(k,nn);
                            productA = productA*(base2^exponent2);
                            productS = productS*(base2^exponent2);
                        end
                    end
                    modelsumS = -models((j-1)*Nqn+alpha,1);
                    col = j*Nqn+alpha;
                    A(k,col) = prefixA*productA;
                    quadsumS = quadsumS + Weights(alpha)*modelsumS*prefixS*productS;
                end
                totalsumS = totalsumS + quadsumS;
            end
            S(k,1) = totalsumS;
        end

        CondN = cond(A,2);
        if CondN < CC
            if rank(X) == min(Nic,Nqn)
                CC = CondN; 
                Xgood = X;
                CCgood = CC;
                Agood = A;
                Momentsgood = Moments;
            end
        end
    end
    %conditionA(ii) = CCgood;

end

O = []; 
for i = 1:size(Xgood,1)
    for j = 1:size(Xgood,2)
        O = [O Xgood(i,j)];
    end
end

fprintf('\n---------------------------\n'); 

fprintf('\n Condition number is: \n %f \n \n',CCgood);

P='';

for i =1:size(O,2)
    if ( i ~= size(O,2) ) 
        P=[P num2str(O(i)) ','];
    else 
        P=[P num2str(O(i))]; 
    end
end

P = ['<Optimal_abscissas>[' P ']</Optimal_abscissas>'];

fprintf(' Copy this into the <Optimization> node of your UPS file: \n %s \n \n', P);

