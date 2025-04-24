function [U,V,Z,Q,obj] = FEMSRL(X,numanchor,gt,lambda,lambda2,ind)

v = size(X,1) + size(X,2)-1; % the number of views
n = size(X{1},2); % the number of samples, n_dimension
m = numanchor; % the number of anchor samples, n_anchor_number
k = length(unique(gt)); % the number of clusters
% we set Z = U*V, s.t. U'U = I
U = cell(1,v); % m*m
V = ones(n,m)/m; % n*m
Y = cell(1,v);
Z = cell(1,v);
G = cell(1,v);
UU = cell(1,v);
VV = cell(1,v);
for i = 1:v
    Z{i} = ones(n,m)/m;
    G{i} = zeros(n,m);
    U{i} = ones(m,m)/m;
    Y{i} = zeros(n,m);
end
G_tensor = zeros(n,m, v);
P = Z;
for iv = 1:v
    C{iv} = zeros(size(Z{iv}));
end
% for k = 1:v
%     Y2{k} = zeros(n,m);
%     J{k} = zeros(n,m);
%     HS{k}= zeros(n,m);
%     Y3{k} = zeros(n,m);
% end
% mu = 1e-4;
% mu3 = 10e-5;

tot_mark = 1:n;
for i=1:v
    ind1{i}=ind(:,i);
    los_mark{i}=find(ind1{i}==0);
end
for view_mark = 1:v    
     ext_mark = setdiff(tot_mark,los_mark{view_mark});
     tq_l{view_mark} = eye(n);
     tq_l{view_mark}(:,ext_mark) = [];%单位矩阵已有样本的列删除
     l_num = length(los_mark{view_mark}); % number of missing data of each view
     e_num = n - l_num;  % number of existing data of each view
     kz_l{view_mark} = zeros(l_num,n);
     kz_l{view_mark}(:,los_mark{view_mark}) = eye(l_num);
     kz_e{view_mark} = zeros(e_num,n);
     kz_e{view_mark}(:,ext_mark) = eye(e_num);
     data_e{view_mark} = X{view_mark};
     data_e{view_mark}(:,los_mark{view_mark}) = [];
end
missingindex = constructA(ind);
for iv=1:v
xx{iv} = repmat(missingindex{iv},m,1);
end
SE = cell(1, v);
% Q = cell(1, v);
%     for nv_idx = 1 : v
%           SE{nv_idx} = diag(missingindex{nv_idx});
%           ind = find(missingindex{nv_idx}==0);
%           SE{nv_idx}(:,ind)=[];
%           Q{nv_idx} = SE{nv_idx}*SE{nv_idx}';
%     end

    for i=1:v
    [~, A{i}] = litekmeans((data_e{i})',numanchor,'MaxIter', 100,'Replicates',10);
    A{i}=(A{i})';
%       di = size(X{i},1); 
%       A{i} = zeros(di,numanchor); 
    end
quadprog_options = optimset( 'Algorithm','interior-point-convex','Display','off');

iter = 0;
Isconverg = 0;
max_iter = 30;

epson = 1e-3;

rho = 10e-5; max_rho = 10e10; pho_rho = 2;
miu = 10e-5;
alpha = ones(1,v)/v;
% max_mu = 10e10; pho_mu = 2;
% weight_vector = ones(1,v)';
% sX = [n, m, v];

%% optimization
while (Isconverg == 0)
     iter = iter + 1;
     Z_pre = Z;
     % update Xi
     for view_mark = 1:v
         X{view_mark} = A{view_mark}*Z{view_mark}'*tq_l{view_mark}*kz_l{view_mark}+data_e{view_mark}*kz_e{view_mark};
     end
    %optimize Ai
    for ia = 1:v
        part1 = X{ia} * Z{ia};
        [Unew,~,Vnew] = svd(part1,'econ');
        A{ia} = Unew*Vnew';
    end
    
    % updating Z
    Z_old = Z;
    for i = 1:v
        Z_tmp1 = A{i}'*A{i}+ 0.5*rho*eye(m)+0.5*miu*eye(m);%A{i}'*A{i}+ 0.5*rho*eye(m)+lambda2*eye(m)
        Z_tmp1 = 2*Z_tmp1;
        Z_tmp1 = (Z_tmp1'+Z_tmp1)/2;
        
        Z_i = Z{i};
        X_i = X{i};
        A_i = A{i};
        Y_i = Y{i};
        C_i = C{i};
        P_i = P{i};
        G_i = G{i};
        alpha_i = alpha(i);
%         HS_i = HS{i}; 
%         J_i = J{i};
%         Y2_i = Y2{i};
%         Y3_i = Y3{i};
        UV_i = U{i}*V';
        parfor j = 1:n%parfor
            Z_tmp2 = -2*((X_i(:,j))'*A_i)+Y_i(j,:)-rho*(UV_i(:,j))'+C_i(j,:)-miu*P_i(j,:);%+Y2_i(j,:)-mu3*HS_i(j,:)+Y3_i(j,:)-mu*J_i(j,:)
            Z_i(j,:) = quadprog(Z_tmp1,Z_tmp2',[],[],ones(1,m),1,zeros(m,1),ones(m,1),Z_i(j,:),quadprog_options);
        end
        Z{i}=Z_i; 
    end
    
     % update Pv
    Z1_tensor = cat(3, Z{:,:});
    C_tensor = cat(3, C{:,:});
    Zv = Z1_tensor(:);
    Cv = C_tensor(:);
    [Pv, objV] = wshrinkObj(Zv + 1/miu*Cv,lambda2/miu,[n,m,v],0,1);
    P_tensor = reshape(Pv, [n,m,v]);
    % -----------------------------------%
    for iv = 1:v
        P{iv} = P_tensor(:,:,iv);
        % -------- C{iv} ------%
        tmp2{iv} = C{iv}'+miu*(Z{iv}'-P{iv}');
        C{iv} = tmp2{iv}';
    end
    clear Z1_tensor C_tensor Zv Cv Pv P_tensor
%     Z_tensor = cat(3, Z{ : , : });
%     hatZ = fft(Z_tensor, [], 3);
%     if iter == 1
%         for iv = 1 : v
%             [Unum_view, Sigmanum_view, Vnum_view] = svds(hatZ( : , : , iv), k);
%             UU{iv} = Unum_view * Sigmanum_view;
%             VV{iv} = Vnum_view';
%             G_tensor( : , : , iv) = UU{iv} * VV{iv};
%         end
%     else
%         for iv = 1 : v
%             UU{iv} = hatZ( : , : , iv) * VV{iv}' * pinv(VV{iv} * VV{iv}');
%             Usq{iv} = UU{iv}' * UU{iv};
%             VV{iv} = pinv(Usq{iv}) * UU{iv}' * hatZ( : , : , iv);
%             G_tensor( : , : , iv) = UU{iv} * VV{iv};
%         end
%     end
%     G_tensor = ifft(G_tensor, [], 3);
%     for iv = 1 : v
%         G{iv} = G_tensor( : , : , iv);
%     end
    
    % optimize alpha
%     M = zeros(v,1);
%     for iv = 1:v
%         M(iv) = norm( X{iv} - A{iv} * Z{iv}','fro')^2;
%     end
%     Mfra = M.^-1;
%     QQ = 1/sum(Mfra);
%     alpha = QQ*Mfra;   


    % update V
    V_tmp1 = (lambda+0.5*rho*v)*eye(m); % V_tmp1 is defined as left form, since U'*U=I
    V_tmp1 = 2*V_tmp1;
    V_tmp1 = (V_tmp1'+V_tmp1)/2;
    for j = 1:n
        V_tmp2 = zeros(1,m);
        for i = 1:v
            V_tmp2 = V_tmp2 + 0.5*Y{i}(j,:)*U{i}+0.5*rho*Z{i}(j,:)*U{i};
         end
        V_tmp2 = -2*V_tmp2;
        V(j,:) = quadprog(V_tmp1,V_tmp2',[],[],ones(1,m),1,zeros(m,1),ones(m,1),V(j,:),quadprog_options);
    end

    % updating U
    for i = 1:v
        U_a = Z{i} + Y{i}/rho;
        U_b = V'*U_a;
        [svd_U,~,svd_V] = svd(U_b,'econ');
        U{i} = svd_V*svd_U';
    end

    % update rho and Y
    for i = 1:v
        tmp1 = Y{i}' + rho*(Z{i}'-U{i}*V');
        Y{i} = tmp1';
%         Y2{i} = Y2{i} + mu*(Z{i}-J{i});
%         Y3{i} = Y3{i} + mu3*(Z{i}-HS{i});
    end
    rho = min(rho*pho_rho, max_rho);
    miu = min(miu*pho_rho, max_rho);
%     mu = min(mu*pho_mu, max_mu);
%     mu3 = min(mu3*pho_mu, max_mu);
    diff_Z = 0;
    %Isconverg = 1;
    for iv = 1:v
        leq{iv} = Z{iv}-P{iv};
        diff_Z = max(diff_Z,max(abs(Z{iv}(:)-Z_pre{iv}(:))));     
    end
    leqm = cat(3, leq{:,:});
    leqm2 = max(abs(leqm(:)));
    clear leq leqm Rec_error_tensor Rec_error
    err = leqm2;
    obj(iter) = err;  
    for i = 1:v
        % convergence condiction: norm(Z{i}-V*U{i}',inf) < epson
        if ( norm(Z{i}-V*U{i}',inf)<epson && err < 1e-3)% && err < 1e-3
            Isconverg = 1;
        end
    end

    if (iter>=max_iter)
        Isconverg  = 1;
    end

end

Sbar = V;

[Q,~,~] = mySVD(Sbar,k); 

% rng(1234,'twister') % set random seed for re-production
% labels=litekmeans(Q, k, 'MaxIter', 100,'Replicates',10);
end