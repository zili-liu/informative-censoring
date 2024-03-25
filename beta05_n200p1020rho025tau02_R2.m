tic
format short
clc;clear;close;

%% === model parameters ===
n = 200; p = 10; n0 = 2;
N = 500;
rho = [0.25 0.5];
mu = zeros(p,1); c = (1:p);
ama = bsxfun(@minus,c,c');
rho = rho(1);
sigma = rho.^(abs(ama));

%% === 'control' parameter ===
a = 100;  % P = 60

%% === True Beta ===

% Beta = 0.3*[1.0;1.0;zeros(p-2,1)];
% PHI =  0.2*[1.0;1.0;zeros(p-2,1)];

Beta = 1.0*[1.0;0;-1.0;0;0;1.0;zeros(p-6,1)];  % % Example1 (信号较强的情形)
PHI =  1.0*[0;1.0;0;-1.0;zeros(p-4,1)];   % % 强信号

Censorrate = zeros(N,1);
index = find(Beta~=0);
index1 = find(PHI~=0);

%% ----------------------------------------
%     Monte Carlo simulation
% -----------------------------------------------

for iter = 1:N
    iter
    rng(iter)   % % 设置随机种子，进一步控制重复性
    [T,Z,Delta,T1,Z1,Delta1] = survival_data(n,Beta,PHI,mu,sigma,iter);

    [ini_beta(:,iter),ini_phi(:,iter)] =...
        initial_beta(n,Z,1e-5,Delta,a);
    %     MIC_cox(:,iter) = MIC(n,ini_beta(:,iter),Z,1e-5,Delta,a);
end

for iter = 1:N
    iter
    rng(iter)   % % 设置随机种子，进一步控制重复性
    [T,Z,Delta,T1,Z1,Delta1] = survival_data(n,Beta,PHI,mu,sigma,iter);
    Censorrate(iter) = 1-mean(Delta);

    oracle_beta(:,iter) = initial_beta1(n,Z(:,index),1e-4,Delta);
    full_beta(:,iter) = initial_beta1(n,Z,1e-5,Delta);

    %     [ini_beta(:,iter),ini_phi(:,iter),opt_lambda1(iter),opt_lambda2(iter)] =...
    %         initial_beta(n,Z,1e-5,Delta,theta1,theta2);

    [pl(:,iter)] = ista_LQA(n,ini_beta(:,iter),Z,1e-5,Delta,a);   % theta=95 for n=200
    [pl1(:,iter)] = ista_LQA(n,mean(ini_beta,2),Z,1e-5,Delta,a);
    [mpl1(:,iter),phi(:,iter),seta(:,iter),gama(:,iter),H_T(:,iter),H_C(:,iter)]=...
        MPL(n,n0,ini_beta(:,iter),mean(ini_phi,2),Z1,T1,1e-5,Delta1,a);
    [mpl(:,iter),phi(:,iter),seta(:,iter),gama(:,iter),H_T(:,iter),H_C(:,iter)]=...
        MPL(n,n0,mean(ini_beta,2),mean(ini_phi,2),Z1,T1,1e-5,Delta1,a);  % theta=200 for n=200

% 
%     if sum(mpl(index,iter)~=0)+sum(phi(index1,iter)~=0) ~= length(index)+length(index1)
%          continue  
%     end

    se_pl(iter) = (pl(:,iter)-Beta)'*sigma*(pl(:,iter)-Beta);    % % mse
    se_mpl(iter) = (mpl(:,iter)-Beta)'*sigma*(mpl(:,iter)-Beta); % % mse
    se_phi(iter) = (phi(:,iter)-PHI)'*sigma*(phi(:,iter)-PHI);   % % mse
    se_oracle(iter)= (oracle_beta(:,iter)-Beta(index))'*sigma(index,index)*(oracle_beta(:,iter)-Beta(index));   % % mse
    se_full(iter)= (full_beta(:,iter)-Beta)'*sigma*(full_beta(:,iter)-Beta);   % % mse

    [cov_beta(:,iter),cov_phi(:,iter)] = cov_matrix(mpl(:,iter),phi(:,iter),seta(:,iter)',gama(:,iter)', ...
        T1,Z1,Delta1,n,n0,index,index1);


     PL_cov_beta(:,iter) = PL_cov(n,Z,pl(:,iter),Delta,index);

end

time = toc

% cov_matrix(mean(mpl,2),mean(phi,2),0.5*ones(1,100),0.5*ones(1,100), ...
%         T1,Z1,Delta1,n,n0,index,index1)
%% ---------------------------------------------------------------------------
%      Assessment Criteria :
% ----------------------------------------------------------------------------
%corr: 选出正确模型的频率；the proportion of all active predictors are selected
%size：估计模型的大小；the number of selected variables
%MSE：模型误差；
%N_+：多选指标；the number of incorrectly selected variables (false negatives)
%N_-：少选指标.
% -------------------------------------------------------------------------------------
corr_pl = (1/N)*sum((all(pl(index,:))).*(1-any(pl(setdiff(1:1:p, index),:))));
MSE_pl = mean(se_pl);
N_plus_pl = (1/N)*sum(sum(pl(setdiff(1:1:p, index),:)~=0));
N_minus_pl = (1/N)*sum(sum(pl(index,:)==0));
Size_pl = sum(sum(pl(:,:)~=0))/N;

% -------------------------------------------------------------------------------------

corr_mpl = sum((all(mpl(index,:))).*(1-any(mpl(setdiff(1:1:p, index),:))))/N;
MSE_mpl = mean(se_mpl);
N_plus_mpl = sum(sum(mpl(setdiff(1:1:p, index),:)~=0))/N;
N_minus_mpl = sum(sum(mpl(index,:)==0))/N;
Size_mpl = sum(sum(mpl(:,:)~=0))/N;

% -------------------------------------------------------------------------------------

corr_phi = sum((all(phi(index1,:))).*(1-any(phi(setdiff(1:1:p, index1),:))))/N;
MSE_phi = mean(se_phi);
N_plus_phi = sum(sum(phi(setdiff(1:1:p, index1),:)~=0))/N;
N_minus_phi = sum(sum(phi(index1,:)==0))/N;
Size_phi = sum(sum(phi(:,:)~=0))/N;

% -------------------------------------------------------------------------------------

corr_oracle = sum((all(oracle_beta(:,:))))/N;
MSE_oracle = mean(se_oracle);
% N_plus_oracle = sum(sum(oracle_beta(setdiff(index, index),:)~=0))/N;
N_plus_oracle = 0;
N_minus_oracle = sum(sum(oracle_beta(:,:)==0))/N;
Size_oracle = sum(sum(oracle_beta(:,:)~=0))/N;

% -------------------------------------------------------------------------------------

corr_full = sum((all(full_beta(index,:))).*(1-any(full_beta(setdiff(1:1:p, index),:))))/N;
MSE_full = mean(se_full);
N_plus_full = sum(sum(full_beta(setdiff(1:1:p, index),:)~=0))/N;
N_minus_full = sum(sum(full_beta(index,:)==0))/N;
Size_full = sum(sum(full_beta(:,:)~=0))/N;


%% output
Criteria = [corr_full MSE_full N_plus_full N_minus_full Size_full;
    corr_pl MSE_pl N_plus_pl N_minus_pl Size_pl;
    corr_mpl MSE_mpl N_plus_mpl N_minus_mpl Size_mpl;
    corr_oracle MSE_oracle N_plus_oracle N_minus_oracle length(index)]

mean(Censorrate)   %% Censoring rate

[mean(mpl(index,:),2),mean(pl(index,:),2)]
BISE = [mean(mpl(index,:),2)-Beta(index) mean(pl(index,:),2)-Beta(index)]

s1=[std(pl(1,:)),std(pl(3,:)),std(pl(6,:))];
s2=[std(mpl(1,:)),std(mpl(3,:)),std(mpl(6,:))];


s3=[std(phi(2,:)),std(phi(4,:))];
[mean(phi(index1,:),2),mean(phi(index1,:)-PHI(index1),2),s3']


%MPL_SD_beta = sqrt(var(mpl(index,:),0,2))
MPL_SD_beta = s2';
a = sqrt(cov_beta);
%MPL_ASD_beta = [median(a(1,:),'all')./0.6745 median(a(2,:),'all')./0.6745 median(a(3,:),'all')./0.6745]'

MPL_ASD_beta = mean(sqrt(cov_beta),2);


PL_SD_beta = s1';
b = abs(sqrt(PL_cov_beta));
% PL_ASD_beta = [median(b(1,:),'all')./0.6745 median(b(2,:),'all')./0.6745]'
PL_ASD_beta = mean(b,2);


MPL_SD_phi = sqrt(var(phi(index1,:),0,2));
c =sqrt(cov_phi);
%MPL_ASD_phi =  [median(c(1,:),'all')./0.6745 median(c(2,:),'all')./0.6745]'
MPL_ASD_phi = mean(sqrt(cov_phi),2);


%%% confidence intervals

CP_MPL_beta = mean( and( mpl(index,:)-1.96*sqrt(cov_beta) <= repmat(Beta(index),1,N),...
    mpl(index,:)+1.96*sqrt(cov_beta) >= repmat(Beta(index),1,N) ),2);

CP_PL_beta = mean( and( pl(index,:)-1.96*abs(sqrt(PL_cov_beta)) <= repmat(Beta(index),1,N),...
    pl(index,:)+1.96*abs(sqrt(PL_cov_beta)) >= repmat(Beta(index),1,N) ),2);

% CP_MPL_phi = mean( and( repmat(mean(phi(index1,:),2),1,500)-1.96*repmat(MPL_ASD_phi,1,500) <= phi(index1,:),...
%     repmat(mean(phi(index1,:),2),1,500)+1.96*repmat(MPL_ASD_phi,1,500) >= phi(index1,:) ),2)
CP_MPL_phi = mean( and( phi(index1,:)-1.96*sqrt(cov_phi) <= repmat(PHI(index1),1,N),...
    phi(index1,:)+1.96*sqrt(cov_phi) >= repmat(PHI(index1),1,N) ),2);



Result_MPL_beta = [mean(mpl(index,:),2)-Beta(index) MPL_SD_beta MPL_ASD_beta CP_MPL_beta]

Result_MPL_phi = [mean(phi(index1,:)-PHI(index1),2) MPL_SD_phi MPL_ASD_phi CP_MPL_phi]

Result_PL_beta = [mean(pl(index,:),2)-Beta(index) PL_SD_beta PL_ASD_beta CP_PL_beta]


%% ===================================================
%                 survival_data()
% ============================================================
function  [T,Z,status,T1,Z1,status1] = survival_data(n,Beta,Phi,mu,sigma,iter)
%% Generating the survival data
p = length(Beta);
ZZ = mvnrnd(mu,sigma,n);  % % n*p 协变量
tau = 0.2;  % % Kendall's tau
alph = copulaparam('Frank',tau);  % %  the scalar parameter alpha.
u = copularnd('Frank',alph,n);
lamt = 1.4;    % tau = 0.2
% lamt = 1.6;    % tau = 0.5
Death_time = sqrt(-lamt^2*log(1-u(:,1))./exp(ZZ*Beta));  % % death time
C = -5*log(1-u(:,2))./exp(ZZ*Phi);

statu = (Death_time <= C);
TT = min(Death_time,C);     % % survial time

[T,I] = sort(TT,'descend');  % % sorting the time
% Y = bsxfun(@ge,T,T');     % % at risk process
Z = ZZ(I,:);
status = statu(I);

[T1,I1] = sort(TT,'ascend');  % % sorting the time
% Y = bsxfun(@ge,T1,T1');     % % at risk process
Z1 = ZZ(I1,:);
status1 = statu(I1);

end

%% ===================================================
%                 initial_beta1()
% ============================================================

function initial_beta= initial_beta1(n,Z,r,status)
[~,p] = size(Z);
beta=zeros(p,1);

k = 1; err = 0;  tk = 1;
while k<=1000 && err==0
    %k
    L_prime = -(sum(status.*(Z-cumsum((exp(Z*beta).*Z))./ cumsum(exp(Z*beta)))))'/n;

    beta1 =  beta - L_prime/tk;
    w = beta1-beta;
    err = norm(w,2)^2 <= r*norm(beta,2)^2;
    beta = beta1;

    k = k+1;
end

beta1 = beta1.*(abs(beta1)>=2*1e-4);

initial_beta = beta1;


end


%% ===================================================
%                 initial_beta()
% ============================================================
function [initial_beta,initial_phi]= initial_beta(n,Z,r,status,a)
[~,p] = size(Z);
beta = zeros(p,1);
phi = zeros(p,1);
opt_BIC = 1e+10;
opt_BIC2 = 1e+10;

n0 = sum(status);
lambda0 = log(n0);


% ============================================================
k = 1; err = 0; 
tk = 2; % p = 60
% tk = 70; % p = 100

while k<=1000 && err==0
    %     k
    L_prime = -(sum(status.*(Z-cumsum((exp(Z*beta).*Z))./cumsum(exp(Z*beta)))))'/n;
    beta1 =  beta - L_prime/tk;

    w = beta1-beta;
    err = norm(w,2)^2 <= r*norm(beta,2)^2;
    beta = beta1;
    k = k+1;
end

ini_beta = beta1;

k = 1;err = 0; tk = 2;
beta = ini_beta;
while k<=1000 && err==0
    %k

    L_prime = -(sum(status.*(Z-cumsum((exp(Z*beta).*Z))./ cumsum(exp(Z*beta)))))'/n;
    W1 = diag( (4*a*lambda0*exp(2*a*beta.^2))./((exp(2*a*beta.^2)+1).^2) );
    u1 = eye(p) + 2*W1/tk;
    beta_tilde = beta - L_prime/tk;
    beta1 = u1\beta_tilde;


    w = beta1-beta;
    err = norm(w,2)^2 <= r*norm(beta,2)^2;
    beta = beta1;
    k = k+1;

end

beta2 = beta1.*(abs(beta1)>=2.5*1e-4);
opt_beta = beta2;


% ============================================================
k = 1; err1 = 0;
% tk = 10; % p = 60
tk = 12; % p = 100
while k<=1000 && err1==0
    % k
    L_prime1 = -(sum((1-status).*(Z-cumsum((exp(Z*phi).*Z))./cumsum(exp(Z*phi)))))'/n;
    phi1 =  phi - L_prime1/tk;
    w1 = phi1-phi;
    err1 = norm(w1,2)^2 <= r*norm(phi,2)^2;
    phi = phi1;
    k = k+1;
end
ini_phi = phi1;

k = 1; err1 = 0; tk = 10;
phi =ini_phi;
while k<=1000 && err1==0
    % k
    L_prime1 = -(sum((1-status).*(Z-cumsum((exp(Z*phi).*Z))./ cumsum(exp(Z*phi)))))'/n;

    W2 = diag( (4*a*lambda0*exp(2*a*phi.^2))./((exp(2*a*phi.^2)+1).^2) );
    u2 = eye(p) + 2*W2/tk;
    phi_tilde = phi - L_prime1/tk;
    phi1 = u2\phi_tilde;


    w1 = phi1-phi;
    err1 = norm(w1,2)^2 <= r*norm(phi,2)^2;
    %         err1 = sqrt(abs(norm(phi1,2)^2-norm(phi,2)^2)) <= r;
    phi = phi1;
    k = k+1;

end

phi1 = phi1.*(abs(phi1)>=2.5*1e-4);
opt_phi = phi1;


initial_beta = opt_beta;
initial_phi = opt_phi;

end


%% ===================================================
%                 ista_LQA()
% ============================================================
function [opt_beta,opt_theta] = ista_LQA(n,ini_beta,Z,r,status,a)
[~,p] = size(Z);
% beta = ini_beta;
beta = zeros(p,1);
% opt_BIC = 1e+10;
n0 = sum(status);
lambda0 = log(n0);

k = 1; err = 0; tk = 4;
while k<=1000 && err==0
    % k
    L_prime = -(sum(status.*(Z-cumsum((exp(Z*beta).*Z))./cumsum(exp(Z*beta)))))'/n;
    beta1 =  beta - L_prime/tk;
    w = beta1-beta;

    err = norm(w,2)^2 <= r*norm(beta,2)^2;
    beta = beta1;
    k = k+1;
end

ini_beta1 = beta1;

k = 1;err = 0; tk = 4;
beta = ini_beta1;

while k<=1000 && err==0
    %k

    L_prime = -(sum(status.*(Z-cumsum((exp(Z*beta).*Z))./ cumsum(exp(Z*beta)))))'/n;
    W1 = diag( (4*a*lambda0*exp(2*a*beta.^2))./((exp(2*a*beta.^2)+1).^2) );
    u1 = eye(p) + 2*W1/tk;
    beta_tilde = beta - L_prime/tk;
    beta1 = u1\beta_tilde;


    w = beta1-beta;
    err = norm(w,2)^2 <= r*norm(beta,2)^2;
    beta = beta1;
    k = k+1;

end

beta2 = beta1.*(abs(beta1)>=2.5*1e-4);
beta = beta2;


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

k=1;err=0; tk = 4;
while k<=1000 && err==0
    %       k
    W1 = diag( (4*a*lambda0*exp(2*a*beta.^2))./((exp(2*a*beta.^2)+1).^2) );
    u = eye(p) + 2*W1/tk;

    L_prime = -(sum(status.*(Z-cumsum((exp(Z*beta).*Z))./ cumsum(exp(Z*beta)))))'/n;
    beta_tilde = beta - L_prime/tk;
    beta1 = u\beta_tilde;


    w = beta1-beta;
    err = norm(w,2)^2 <= r*norm(beta,2)^2;
    beta = beta1;
    k = k+1;
end

beta2 = beta.*(abs(beta)>=5*1e-4);

opt_beta = beta2;

end



%% ===================================================
%                 ista_MIC()
% ============================================================
function opt_beta = MIC(n,ini_beta,Z,r,status,a)
[~,p] = size(Z);
beta = ini_beta;
n0 = sum(status);
lambda0 = log(n0);

k = 1; err = 0; tk = 4;
while k<=1000 && err==0
    % k
    L_prime = -(sum(status.*(Z-cumsum((exp(Z*beta).*Z))./cumsum(exp(Z*beta)))))'/n;
    beta1 =  beta - L_prime/tk;
    w = beta1-beta;
    %     err = sqrt(abs(norm(beta1,2)^2-norm(beta,2)^2)) <= r;
    err = norm(w,2)^2 <= r*norm(beta,2)^2;
    beta = beta1;
    k = k+1;
end

k=1;err=0; tk = 2;
while k<=1000 && err==0
    %   k
    W1 = diag( (4*a*lambda0*exp(2*a*beta.^2))./((exp(2*a*beta.^2)+1).^2) );
    u = eye(p) + 2/tk*W1;
    L_prime = -(sum(status.*(Z-cumsum((exp(Z*beta).*Z))./ cumsum(exp(Z*beta)))))'/n;
    beta_tilde = beta - L_prime/tk;
    beta1 = u\beta_tilde;

    w = beta1-beta;
    err = norm(w,2)^2 <= r*norm(beta,2)^2;
    beta = beta1;
    k = k+1;
end

opt_beta = beta.*(abs(beta)>5*1e-4);

end

%% ===================================================
%                 MPL()
% ============================================================

function [opt_beta,opt_phi,opt_seta,opt_gama,H_T,H_C] = MPL(n,n0,ini_beta,ini_phi,Z,T,r,Delta,a)
[~,p] = size(Z);
m = n/n0;
tau = 0.2;  % % Kendall's tau
alph = copulaparam('Frank',tau);  % %  the scalar parameter alpha.

beta = ini_beta;
phi = ini_phi;
eta = [beta;phi];
lambda0 = log(sum(Delta));  % According to Su et al., 2016
% lambda1 = 0.5;
% lambda2 = 0.5;
lambda1 = 1e-1*sqrt(n);
lambda2 = 1e-1*sqrt(n);


psix = zeros(n,m);
Psi = zeros(n,m);
%binwv = zeros(1,m);
%DJ = zeros(1,m);

%% Discretizing a sample of n survival times into sub-intervals
% with no. of 'binCount' subjects in each sub-interval
[ID,binwv] = discrBinNA(T,n0);

% binid = 1:n0:n;
% binedg = T(binid)';
% binwv = binedg(2:m)-binedg(1:(m-1)); %binwv: bin widths

% for i = 1:m
%     ID( ((i-1)*n0+1):i*n0 ) = i;
% end


%% piecewise constant basis function for the non-parametric baseline hazard
for i=1:m
    psix(((i-1)*n0+1):1:i*n0,i) =1;
end
%% piecewise constant cumulative basis function for the non-parametric baseline hazard
for i = 1:n
    Psi(i,1:ID(i)) = 1*binwv(1:ID(i));
end

%% Initial estimates of theta (piecewise constant estimate of h_{0t}) based on independent censoring assumption
ini_seta = sum( repmat(Delta,1,m).*psix )./( sum(repmat(exp(Z*ini_beta),1,m).*Psi)+1e-6 );

%% Initial estimates of gamma (piecewise constant estimate of h_{0c}) based on independent censoring assumption
ini_gama = sum(repmat(1-Delta,1,m).*psix)./( sum(repmat(exp(Z*ini_phi),1,m).*Psi)+1e-6 );
seta = ini_seta+1e-6;  % 1*m
gama = ini_gama+1e-6;  % 1*m
RT = mat1(psix,T,1e-5); % First order difference -- For piecewise constant
% seta = 0.5*ones(1,m);  % baseline coefficients
% gama = 0.5*ones(1,m);  % baseline coefficients

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

k=1; err1=0; err2=0; err3=0; err4=0; 
tk1= 2; tk2= 4;
% tk1= 20; tk2= 18;  % p = 60
% tk1= 24; tk2= 16;  % p = 100

%     beta = ini_beta; phi = ini_phi;
while k<=1000 && (err1==0 ||err2==0 || err3==0 || err4==0)
   %k
    W1 = diag( (4*a*lambda0*exp(2*a*beta.^2))./((exp(2*a*beta.^2)+1).^2) );
    u1 = eye(p) + 2*W1/tk1;
    W2 = diag( (4*a*lambda0*exp(2*a*phi.^2))./((exp(2*a*phi.^2)+1).^2) );
    u2 = eye(p) + 2*W2/tk2;


    %%

    H_T = sum( exp(Z*beta).*(seta.*Psi),2 );  % Cumulative Risk - Failure
    H_C = sum( exp(Z*phi).*(gama.*Psi),2 );   % Cumulative Risk - Censoring

    S_T = exp(-H_T);  % Survival Probility - Failure
    S_C = exp(-H_C);  % Survival Probility - Censoring

    S1 = exp(alph*S_T).*exp(alph*S_C)-exp(alph*S_T)-exp(alph*S_C)+exp(alph);  % Copula: K(a,b,alph)
    S2 = exp(alph)*exp(alph*S_C)+exp(alph*S_C)-exp(2*alph*S_C)-exp(alph);
    S3 = exp(alph)*exp(alph*S_T)+exp(alph*S_T)-exp(2*alph*S_T)-exp(alph);

    LAM1 = alph*(exp(alph*S_T)./S1).*(Delta.*(S2./(exp(alph*S_T).*(exp(alph*S_C)-1)))+...
        (1-Delta).*(exp(alph)-1)./(exp(alph*S_T)-1));
    LAM2 = (alph./S1).*(Delta.*((exp(alph*S_T)*(exp(alph)-1))./(exp(alph*S_C)-1))+...
        (1-Delta).*(S3./(exp(alph*S_C)-1)));


    DL_beta =  -(sum( (Delta-Delta.*H_T-LAM1.*S_T.*H_T).*Z,1) )'/n;  % first partial derivative - beta
    DL_phi =   -(sum(((1-Delta)-(1-Delta).*H_C-LAM2.*S_C.*H_C).*Z,1))'/n;  % first partial derivative - phi
    
 
    beta_tilde = beta - DL_beta/tk1;
    beta1 = u1\beta_tilde;

    w1 = beta1-beta;
    err1 = norm(w1,2)^2 <= r*norm(beta,2)^2;

    beta = beta1;


    phi_tilde = phi - DL_phi/tk2;
    phi1 = u2\phi_tilde;

    w2 = phi1-phi;
    err2 = norm(w2,2)^2 <= r*norm(phi,2)^2;

    phi = phi1;

    %% 嵌套Armijo线性搜索法
    uu = 0.6; % 取值范围(0,1)
    gamm = 0.5; % 取值范围(0,0.5)越大越快
    sigm = 0.35; % 取值范围(0,1)越大越慢

    t = 1; %循环标志

    while (t>0)

        H_T = sum( exp(Z*beta).*(seta.*Psi),2 );  % Cumulative Risk - Failure
        H_C = sum( exp(Z*phi).*(gama.*Psi),2 );   % Cumulative Risk - Censoring


        S_T = exp(-H_T);  % Survival Probility - Failure
        S_C = exp(-H_C);  % Survival Probility - Censoring

        S1 = exp(alph*S_T).*exp(alph*S_C)-exp(alph*S_T)-exp(alph*S_C)+exp(alph);  % Copula: K(a,b,alph)
        S2 = exp(alph)*exp(alph*S_C)+exp(alph*S_C)-exp(2*alph*S_C)-exp(alph);
        S3 = exp(alph)*exp(alph*S_T)+exp(alph*S_T)-exp(2*alph*S_T)-exp(alph);

        %%  Penalty of seta_u and gama_u
%         DJ_seta(1) = -2*(seta(2) - seta(1));
%         DJ_gama(1) = -2*(gama(2) - gama(1));
%         DJ_seta(m) = 2*(seta(m) -seta(m-1));
%         DJ_gama(m) = 2*(gama(m) -gama(m-1));
% 
%         for i=2:m-1
%             DJ_seta(i) = 2*(2*seta(i) - seta(1+1) - seta(i-1));
%             DJ_gama(i) = 2*(2*gama(i) - gama(1+1) - gama(i-1));
%         end
        DL_seta =  sum( (Delta.*psix)./(sum(seta.*psix,2))- ...
            (exp(Z*beta).*(Delta+LAM1.*S_T)).*Psi,1)/n - lambda1*seta*RT;   % first partial derivative - seta

        DL_gama =  sum( ((1-Delta).*psix)./(sum(seta.*psix,2))- ...
            (exp(Z*phi).*(1-Delta+LAM2.*S_C)).*Psi,1)/n - lambda2*gama*RT;  % first partial derivative - gama

        LAM11 = alph*(exp(alph*S_T)./S1).*(Delta.*(max(S2,0)./(exp(alph*S_T).*(exp(alph*S_C)-1)))+...
            (1-Delta).*(exp(alph)-1)./(exp(alph*S_T)-1));
        LAM22 = (alph./S1).*(Delta.*((exp(alph*S_T)*(exp(alph)-1))./(exp(alph*S_C)-1))+...
            (1-Delta).*(max(S3,0)./(exp(alph*S_C)-1)));


        Ps1 = sum(( Delta.*exp(Z*beta) ).*Psi + ...
            ((LAM11.*S_T).*exp(Z*beta)).*Psi,1)+lambda1*max(seta*RT,0)+1e-5;
        Ps2 = sum(((1-Delta).*exp(Z*beta)).*Psi + ...
            ((LAM22.*S_C).*exp(Z*phi)).*Psi,1)+lambda2*max(gama*RT,0)+1e-5;

        % eta = [seta gama] + uu*[DL_seta DL_gama]*diag([seta./Ps1 gama./Ps2]);
        seta1 = seta + uu*DL_seta*diag(seta./Ps1);
        gama1 = gama + uu*DL_gama*diag(gama./Ps2);


        %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        S1 = exp(alph*S_T).*exp(alph*S_C)-exp(alph*S_T)-exp(alph*S_C)+exp(alph);
        K1 = (exp(alph*S_T).*(exp(alph*S_C)-1))./S1;
        K2 = (exp(alph*S_C).*(exp(alph*S_T)-1))./S1;
        L_T = log(sum(seta.*psix,2))+Z*beta-H_T+log(K1);
        L_C = log(sum(gama.*psix,2))+Z*phi-H_C+log(K2);

        L_0 = sum(Delta.*L_T+(1-Delta).*L_C)/n;

        %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        H_T = sum( exp(Z*beta).*(seta1.*Psi),2 );  % Cumulative Risk - Failure
        H_C = sum( exp(Z*phi).*(gama1.*Psi),2 );   % Cumulative Risk - Censoring


        S_T = exp(-H_T);  % Survival Probility - Failure
        S_C = exp(-H_C);  % Survival Probility - Censoring

        S1 = exp(alph*S_T).*exp(alph*S_C)-exp(alph*S_T)-exp(alph*S_C)+exp(alph);
        K1 = (exp(alph*S_T).*(exp(alph*S_C)-1))./S1;
        K2 = (exp(alph*S_C).*(exp(alph*S_T)-1))./S1;
        L_T = log(sum(seta1.*psix,2))+Z*beta-H_T+log(K1);
        L_C = log(sum(gama1.*psix,2))+Z*phi-H_C+log(K2);

        L_new = sum(Delta.*L_T+(1-Delta).*L_C)/n;


        %if  L_0 + gamm*uu.*[DL_seta DL_gama]*(-[DL_seta DL_gama]') <= L_new %检验条件
        if  L_0 + gamm*uu.*[seta1-seta gama1-gama]*([DL_seta DL_gama]') <= L_new
            t = 0; % 循环break
            uu_armijo = uu;
        else
            uu = uu*sigm; %缩小uu，进入下一次循环

        end


    end

    seta1 = seta + uu_armijo*DL_seta*diag(seta./Ps1);
    gama1 = gama + uu_armijo*DL_gama*diag(gama./Ps2);

    %         seta1 = seta + uu_armijo*(seta1-seta);
    %         gama1 = gama + uu_armijo*(gama1-gama);

    err3 = norm(seta1-seta,2)^2 <= 1e-7*norm(seta,2)^2;
    err4 = norm(gama1-gama,2)^2 <= 1e-7*norm(gama,2)^2;

    %         err3 = norm(seta1-seta,2) <= 1e-5;
    %         err4 = norm(gama1-gama,2) <= 1e-5;

    gama = gama1;
    seta = seta1;

    k = k+1;
end

%%
beta2 = beta1.*(abs(beta1)>=5*1e-4);
phi2 = phi1.*(abs(phi1)>=5*1e-4);

opt_beta = beta2;
opt_phi = phi2;
opt_seta = seta;
opt_gama = gama;


% H_T = sum(opt_seta.*Psi,2);
% H_C = sum(opt_gama.*Psi,2);

end



function [cov_beta, cov_phi] = cov_matrix(beta,phi,seta,gama,T,Z,Delta,n,n0,index,index1)
m = n/n0;
[~,p] = size(Z);
tau = 0.2;  % % Kendall's tau
alph = copulaparam('Frank',tau);  % %  the scalar parameter alpha.
% eta = [beta;phi];
% index = find(eta~=0);
psix = zeros(n,m);
Psi = zeros(n,m);
[ID,binwv] = discrBinNA(T,n0);

%% piecewise constant basis function for the non-parametric baseline hazard
for i=1:m
    psix(((i-1)*n0+1):1:i*n0,i) =1;
end
%% piecewise constant cumulative basis function for the non-parametric baseline hazard
for i = 1:n
    Psi(i,1:ID(i)) = 1*binwv(1:ID(i));
end

H_T = sum( exp(Z*beta).*(seta.*Psi),2 );  % Cumulative Risk - Failure
H_C = sum( exp(Z*phi).*(gama.*Psi),2 );   % Cumulative Risk - Censoring

S_T = exp(-H_T);  % Survival Probility - Failure
S_C = exp(-H_C);  % Survival Probility - Censoring


% S1 = exp(alph*S_T).*exp(alph*S_C)-exp(alph*S_T)-exp(alph*S_C)+exp(alph);  % Copula: K(a,b,alph)
% S2 = exp(alph)*exp(alph*S_C)+exp(alph*S_C)-exp(2*alph*S_C)-exp(alph);
% S3 = exp(alph)*exp(alph*S_T)+exp(alph*S_T)-exp(2*alph*S_T)-exp(alph);
%
% LAM1 = alph*(exp(alph*S_T)./S1).*(Delta.*(S2./(exp(alph*S_T).*(exp(alph*S_C)-1)))+...
%     (1-Delta).*(exp(alph)-1)./(exp(alph*S_T)-1));
% LAM2 = (alph./S1).*(Delta.*((exp(alph*S_T)*(exp(alph)-1))./(exp(alph*S_C)-1))+...
%     (1-Delta).*(S3./(exp(alph*S_C)-1)));

K1 = zeros(n,1);   K2 = zeros(n,1);    K11 = zeros(n,1);   K211 = zeros(n,1);
K22 = zeros(n,1);  K12 = zeros(n,1);   K21 = zeros(n,1);   K212 = zeros(n,1);
K111 = zeros(n,1); K112 = zeros(n,1);  K221 = zeros(n,1);  K222 = zeros(n,1);


for i=1:n

    K1(i) = ( exp(alph*S_T(i))*(exp(alph*S_C(i))-1) )/( exp(alph*S_T(i))*exp(alph*S_C(i))-...
        exp(alph*S_T(i))-exp(alph*S_C(i))+exp(alph) );


    K2(i) = ( exp(alph*S_C(i))*(exp(alph*S_T(i))-1) )/( exp(alph*S_T(i))*exp(alph*S_C(i))-...
        exp(alph*S_T(i))-exp(alph*S_C(i))+exp(alph) );


    K11(i) = ( alph*exp(alph*S_T(i))*(exp(alph*S_C(i))*exp(alph)+exp(alph*S_C(i))-exp(2*alph*S_C(i))-exp(alph)) )/...
        (exp(alph*S_T(i))*exp(alph*S_C(i))-exp(alph*S_T(i))-exp(alph*S_C(i))+exp(alph))^2;


    K22(i) = ( alph*exp(alph*S_C(i))*( exp(alph*S_T(i))*exp(alph)+exp(alph*S_T(i))-exp(2*alph*S_T(i))-exp(alph) ) )/...
        ( exp(alph*S_T(i))*exp(alph*S_C(i))-exp(alph*S_T(i))-exp(alph*S_C(i))+exp(alph) )^2;


    K12(i) = ( alph*exp(alph*S_T(i))*exp(alph*S_C(i))*( exp(alph)-1) )/( exp(alph*S_T(i))*exp(alph*S_C(i))-...
        exp(alph*S_T(i))-exp(alph*S_C(i))+exp(alph) )^2;


    K21(i) = K12(i);


    K111(i) = ( alph^2*exp(alph*S_T(i))*( exp(alph)*exp(alph*S_C(i))+exp(alph*S_C(i))-exp(2*alph*S_C(i))-exp(alph) )* ...
        (exp(alph*S_T(i))-exp(alph*S_T(i))*exp(alph*S_C(i)))-exp(alph*S_C(i))+exp(alph) )/...
        (exp(alph*S_T(i))*exp(alph*S_C(i))-exp(alph*S_T(i))-exp(alph*S_C(i))+exp(alph))^3;


    K222(i) = ( alph^2*exp(alph*S_C(i))*( exp(alph)*exp(alph*S_T(i))+exp(alph*S_T(i))-exp(2*alph*S_T(i))-exp(alph) )* ...
        (exp(alph*S_C(i))-exp(alph*S_T(i))*exp(alph*S_C(i)))-exp(alph*S_T(i))+exp(alph) )/...
        (exp(alph*S_T(i))*exp(alph*S_C(i))-exp(alph*S_T(i))-exp(alph*S_C(i))+exp(alph))^3;


    K112(i) = ( alph^2*exp(alph*S_T(i))*exp(alph*S_C(i))*( exp(alph)+1-2*exp(alph*S_C(i)) )*...
        ( exp(alph*S_T(i))*exp(alph*S_C(i))-exp(alph*S_T(i))-exp(alph*S_C(i))+exp(alph) )-2*alph^2*exp(alph*S_T(i))*exp(alph*S_C(i))*...
        ( exp(alph*S_C(i))*exp(alph)+exp(alph*S_C(i))-exp(2*alph*S_C(i))-exp(alph) )*( exp(alph*S_T(i))-1 ) )/...
        (exp(alph*S_T(i))*exp(alph*S_C(i))-exp(alph*S_T(i))-exp(alph*S_C(i))+exp(alph))^3;


    K221(i) = ( alph^2*exp(alph*S_T(i))*exp(alph*S_C(i))*( exp(alph)+1-2*exp(alph*S_T(i)) )*...
        ( exp(alph*S_T(i))*exp(alph*S_C(i))-exp(alph*S_T(i))-exp(alph*S_C(i))+exp(alph) )-2*alph^2*exp(alph*S_T(i))*exp(alph*S_C(i))*...
        ( exp(alph*S_T(i))*exp(alph)+exp(alph*S_T(i))-exp(2*alph*S_T(i))-exp(alph) )*( exp(alph*S_C(i))-1 ) )/...
        (exp(alph*S_T(i))*exp(alph*S_C(i))-exp(alph*S_T(i))-exp(alph*S_C(i))+exp(alph))^3;


    K211(i) =  ( alph^2*exp(alph*S_T(i))*exp(alph*S_C(i))*(exp(alph)-1)*( exp(alph*S_T(i))-...
        exp(alph*S_T(i))*exp(alph*S_C(i))- exp(alph*S_C(i)) +exp(alph) ) )/...
        (exp(alph*S_T(i))*exp(alph*S_C(i))-exp(alph*S_T(i))-exp(alph*S_C(i))+exp(alph))^3;

    K212(i) =  ( alph^2*exp(alph*S_T(i))*exp(alph*S_C(i))*(exp(alph)-1)*( exp(alph*S_C(i))-...
        exp(alph*S_T(i))*exp(alph*S_C(i))- exp(alph*S_T(i)) +exp(alph) ) )/...
        (exp(alph*S_T(i))*exp(alph*S_C(i))-exp(alph*S_T(i))-exp(alph*S_C(i))+exp(alph))^3;


end

LAM1 = Delta.*(K11./K1) +(1-Delta).*(K21./K2);

LAM2 = Delta.*(K12./K1) +(1-Delta).*(K22./K2);


L1 = Delta.*(K111.*K1-K11.^2)./(K1.^2)+(1-Delta).*(K112.*K2-K12.^2)./(K2.^2);

L2 = (1-Delta).*(K222.*K2-K22.^2)./(K2.^2)+Delta.*(K221.*K2-K21.^2)./(K1.^2);

L3 = Delta.*( K211.*K1-K11.*K21 )./K1.^2 + (1-Delta).*( (K212.*K2-K12.*K22) ./(K2.^2));

D1 = (Delta-L1.*H_T.*S_T.^2-LAM1.*H_T.*S_T+LAM1.*S_T).*H_T;

D2 = ((1-Delta)-L2.*H_C.*S_C.^2-LAM2.*H_C.*S_C+LAM2.*S_C).*H_C;

D3 = L3.*H_T.*H_C.*S_T.*S_T;

P1 = zeros(p,p);   P2 = zeros(p,p);  P3 = zeros(p,p);

for i=1:p

    for j=1:p

        P1(i,j) = 2*sum(D1.*Z(:,i).*Z(:,j));

        P2(i,j) = 2*sum(D2.*Z(:,i).*Z(:,j));

        P3(i,j) = 2*sum(D3.*Z(:,i).*Z(:,j));

    end

end

tt = [P1 P3; P3 P2];
% tt1 = inv(tt);
% diag(tt1(index,index))
% diag(tt1(p+index1,p+index1))

cov_beta = diag(inv(tt(index,index)));
cov_phi = diag(inv(tt(p+index1,p+index1)));

% ttt= diag(inv(tt([index p+index1],[index p+index1])));
% cov_beta = [ttt(1) ttt(2)]';
% cov_phi = [ttt(4) ttt(4)]';




% P11 = P1 + diag( 2*lambda0*opt_theta*exp(-opt_theta*beta.^2).*(1-2*opt_theta*beta.^2.*sign(beta)) );
%
% P22 = P2 + diag( 2*lambda0*opt_theta*exp(-opt_theta*phi.^2).*(1-2*opt_theta*phi.^2.*sign(phi)) );
%
% P1 = (P11\P1)/P11;
% P2 = (P22\P2)/P22;
%
% cov_beta = diag( P1(index,index) );
% cov_phi = diag( P2(index1,index1) );

% cov_beta = diag( inv(P1(index,index)) );
% cov_phi = diag( inv(P2(index1,index1)) );


end

function PL_cov_beta = PL_cov(n,Z,beta,status,index)
%% % * --- ASE:the average of estimated standard error; --- *% % %%
eta = Z*beta;
t = cumsum(exp(eta));  % % n*1
v = cumsum( (exp(eta).*Z) ); % % n*p
for i=1:n
    Cel1(1,i) = {exp(eta(i))*Z(i,:)'*Z(i,:)};
    Cel2(1,i) = {v(i,:)'*v(i,:)};
end
f1 = cumsum(cat(3,Cel1{:}),3);   % % 二阶偏导的第一项
f2 = cumsum(cat(3,Cel2{:}),3);   % % 二阶偏导的第二项
for i=1:n
    Cel3(1,i) = {f1(:,:,i)};
    Cel4(1,i) = {f2(:,:,i)};
    Cel3(1,i) =  cellfun(@(x) (status(i)/t(i)).*x, Cel3(1, i),'UniformOutput',false);
    Cel4(1,i) =  cellfun(@(x) (status(i)/t(i)^2).*x, Cel4(1,i),'UniformOutput',false);
end
% L_primeprime = -(sum(cat(3,Cel4{:}),3) - sum(cat(3,Cel3{:}),3));
L_primeprime = -2*(sum(cat(3,Cel4{:}),3) - 2*sum(cat(3,Cel3{:}),3));
%         W = diag(log(sum(Delta))*opt_theta(iter)*exp(-opt_theta(iter)*ista_lqa(index,iter).^2));
%         ESE_lqa(:,iter) = diag(( (L_primeprime(index,index)+W)\(inv( L_primeprime(index,index))) )/(L_primeprime(index,index)+W));   % % ESE

tt = diag( inv( L_primeprime(index,index) ) );

PL_cov_beta = tt;  % % 曹永秀等(2017) Cox-SELO、Zhang and Lu(2007) Adaptive-Cox



end


