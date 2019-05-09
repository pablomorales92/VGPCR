function output = VGPCR_TRAIN(X_tr,Y,options)
%% Reading data
[N R] = size(Y); %N data points, R annotators
mask = double(Y >= 0);
n_lab= sum(mask,2);

%% Initialization

% Initial q(z)
mean_qz = sum(Y .* mask,2)./n_lab;

% Parameters xi 
xi = ones(N,1);
lambda = (1./(2*xi)) .* ((1./(1+exp(-xi))) - 0.5);

% Hyperparameters
par_alpha_a = 1;
par_alpha_b = 1;
par_beta_a = 1;
par_beta_b = 1;

% Preparing the matrix XX for calculating K
XX = X_tr' * X_tr;
if strcmp(options.ker,'rbf')
    d = diag(XX);
    XX = repmat(d,1,N) - 2*XX + repmat(d',N,1);
end
    
% Parameters Omega. They are initialized to make the code more readable, 
% but these initial values do not influence in the algorithm.
gam2= 100;
l2 = 10;
mean_f = zeros(N,1);

%% Main Loop
for iter = 1:options.maxiter
    
    xi_old = xi;
    g_old = gam2;
    l2_old = l2;
    muf_old = mean_f;
    mqz_old = mean_qz;
        
    %% Calculating the Kernel parameters 
    u = 0.5*diag(1./lambda)*(mean_qz-0.5);
	par = log([gam2;l2]);
    par = minimize(par,@op_log_prob_u,-options.maxiter_par,u,XX,lambda,options);
    gam2 = exp(par(1));
    l2 = exp(par(2));
    
    if strcmp(options.ker,'rbf')
        K = gam2 * exp(-XX/(2*l2));
    elseif strcmp(options.ker,'lin')
        K = gam2 * XX;
    end
    
    %% Updating q(f)
    W = diag(sqrt(2*lambda));
    B = W*((eye(N)+W*K*W)\W);
    
    Covar_f = K - K*B*K;
    mean_f = Covar_f*(mean_qz - 0.5);
       
    %% Updating the Parameters (alpha and beta)
    alpha_1 = (par_alpha_a + sum(repmat(mean_qz,1,R).*Y.*mask));
    alpha_2 = (par_alpha_b + sum(repmat(mean_qz,1,R).*(1-Y).*mask));
    
    log_alpha = psi(alpha_1) - psi(alpha_1+alpha_2);
    log_1malpha = psi(alpha_2) - psi(alpha_1+alpha_2);
    
    beta_1 = (par_beta_a + sum(repmat(1-mean_qz,1,R).*(1-Y).*mask));
    beta_2 = (par_beta_b + sum(repmat(1-mean_qz,1,R).*Y.*mask));
    
    log_beta = psi(beta_1) - psi(beta_1+beta_2);
    log_1mbeta = psi(beta_2) - psi(beta_1+beta_2);
    
    mean_alpha = (alpha_1)./(alpha_1+alpha_2); 
    mean_beta = (beta_1)./(beta_1+beta_2);
        
    %% Updating q(z)
    un_nor_log_q1 = mean_f + sum((repmat(log_alpha,N,1).*Y+repmat(log_1malpha,N,1).*(1-Y)).*mask,2);
    un_nor_log_q0 = sum((repmat(log_1mbeta,N,1).*Y + repmat(log_beta,N,1).*(1-Y)).*mask,2);
    
    mean_qz = exp(un_nor_log_q1)./(exp(un_nor_log_q1)+exp(un_nor_log_q0));
    
    %% Updating the Parameters (xi)
    xi = sqrt(mean_f.^2 + diag(Covar_f));
    lambda = (1./(2*xi)) .* ((1./(1+exp(-xi))) - 0.5);
    
    %% Check convergence
    xi_c = norm(xi-xi_old)/norm(xi_old);
    gm_c = abs(gam2-g_old)/abs(g_old);
    l2_c = abs(l2-l2_old)/abs(l2_old);
    f_c = norm(mean_f-muf_old)/norm(muf_old);
    z_c = norm(mean_qz-mqz_old)/norm(mqz_old);
    
    fprintf('%i \t | %.7f \t %.7f \t %.7f \t %.7f \t %.7f | \t %.7f \t %.7f \t %.7f \t %.7f \t %.7f \t %.7f\n',[iter,z_c,f_c,xi_c,gm_c,l2_c,gam2,l2,min(mean_alpha),max(mean_alpha),min(mean_beta),max(mean_beta)])
    
    if (z_c < options.thr) && (f_c < options.thr) && (xi_c < options.thr) && (gm_c< options.thr)&& (l2_c< options.thr)
        fprintf('Iter |\tch_z \t\t ch_f \t\t ch_xi \t\t ch_gm2 \t\t ch_l2 | \t gm2 \t\t\t l2 \t\t min_alpha \t max_alpha \t min_beta \t max_beta\n\n');
        fprintf('Convergence reached at iteration %i. \n',iter);
        break;
    end
end

%% Outputs

%Also need the Kernel parameters.
output.N = N;

output.q_z.mean = mean_qz;
output.q_f.mean = mean_f;
output.q_f.Covar = Covar_f;

output.ker = options.ker;
output.K = K;
if strcmp(options.ker,'rbf')
    output.K_par.l2 = l2;
else
    output.K_par.l2 = 0;
end
output.K_par.gam2 = gam2;
output.B = B;

output.param.xi = xi; 
output.q_alpha.mean = mean_alpha;
output.q_alpha.par_1 = alpha_1;
output.q_alpha.par_2 = alpha_2;

output.q_beta.mean = mean_beta;
output.q_beta.par_1 = beta_1;
output.q_beta.par_2 = beta_2;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [f,df] = op_log_prob_u(parameters,u,XX,lambda,options)

N = length(u);

%% Calculating covariance matrix
gamma = exp(parameters(1));
sigma2 = exp(parameters(2));

if strcmp(options.ker,'rbf')
    K = exp(-XX/(2*sigma2));
elseif strcmp(options.ker,'lin')
    K = XX;
end
Lambdai = diag(1./(2*lambda));
C = gamma*K + Lambdai;

L = chol(C,'lower');
v = L\u;
v2 = (L')\v;
Ci = L'\(L\eye(N));

%% Calculating function
f = 0.5*(v'*v) + sum(log(diag(L))) + 0.5*N*log(2*pi);

%% Calculating gradient
Q = (Ci - v2*v2');

df(1) = 0.5*gamma*sum(dot(Q',K));

switch options.ker
    case 'lin'
        df(2) = 0;
    case 'rbf'
        D = K.*XX;
        df(2) = (gamma*0.25/sigma2) *sum(dot(Q',D));
end

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% FROM GPML PACKAGE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [X, fX, i] = minimize(X, f, length, varargin)

% Minimize a differentiable multivariate function using conjugate gradients.
%
% Usage: [X, fX, i] = minimize(X, f, length, P1, P2, P3, ... )
% 
% X       initial guess; may be of any type, including struct and cell array
% f       the name or pointer to the function to be minimized. The function
%         f must return two arguments, the value of the function, and it's
%         partial derivatives wrt the elements of X. The partial derivative  
%         must have the same type as X.
% length  length of the run; if it is positive, it gives the maximum number of
%         line searches, if negative its absolute gives the maximum allowed
%         number of function evaluations. Optionally, length can have a second
%         component, which will indicate the reduction in function value to be
%         expected in the first line-search (defaults to 1.0).
% P1, P2, ... parameters are passed to the function f.
%
% X       the returned solution
% fX      vector of function values indicating progress made
% i       number of iterations (line searches or function evaluations, 
%         depending on the sign of "length") used at termination.
%
% The function returns when either its length is up, or if no further progress
% can be made (ie, we are at a (local) minimum, or so close that due to
% numerical problems, we cannot get any closer). NOTE: If the function
% terminates within a few iterations, it could be an indication that the
% function values and derivatives are not consistent (ie, there may be a bug in
% the implementation of your "f" function).
%
% The Polack-Ribiere flavour of conjugate gradients is used to compute search
% directions, and a line search using quadratic and cubic polynomial
% approximations and the Wolfe-Powell stopping criteria is used together with
% the slope ratio method for guessing initial step sizes. Additionally a bunch
% of checks are made to make sure that exploration is taking place and that
% extrapolation will not be unboundedly large.
%
% See also: checkgrad 
%
% Copyright (C) 2001 - 2010 by Carl Edward Rasmussen, 2010-01-03

INT = 0.1;    % don't reevaluate within 0.1 of the limit of the current bracket
EXT = 3.0;                  % extrapolate maximum 3 times the current step-size
MAX = 20;                         % max 20 function evaluations per line search
RATIO = 10;                                       % maximum allowed slope ratio
SIG = 0.1; RHO = SIG/2; % SIG and RHO are the constants controlling the Wolfe-
% Powell conditions. SIG is the maximum allowed absolute ratio between
% previous and new slopes (derivatives in the search direction), thus setting
% SIG to low (positive) values forces higher precision in the line-searches.
% RHO is the minimum allowed fraction of the expected (from the slope at the
% initial point in the linesearch). Constants must satisfy 0 < RHO < SIG < 1.
% Tuning of SIG (depending on the nature of the function to be optimized) may
% speed up the minimization; it is probably not worth playing much with RHO.

% The code falls naturally into 3 parts, after the initial line search is
% started in the direction of steepest descent. 1) we first enter a while loop
% which uses point 1 (p1) and (p2) to compute an extrapolation (p3), until we
% have extrapolated far enough (Wolfe-Powell conditions). 2) if necessary, we
% enter the second loop which takes p2, p3 and p4 chooses the subinterval
% containing a (local) minimum, and interpolates it, unil an acceptable point
% is found (Wolfe-Powell conditions). Note, that points are always maintained
% in order p0 <= p1 <= p2 < p3 < p4. 3) compute a new search direction using
% conjugate gradients (Polack-Ribiere flavour), or revert to steepest if there
% was a problem in the previous line-search. Return the best value so far, if
% two consecutive line-searches fail, or whenever we run out of function
% evaluations or line-searches. During extrapolation, the "f" function may fail
% either with an error or returning Nan or Inf, and minimize should handle this
% gracefully.

if max(size(length)) == 2, red=length(2); length=length(1); else red=1; end
if length>0, S='Linesearch'; else S='Function evaluation'; end 

i = 0;                                            % zero the run length counter
ls_failed = 0;                             % no previous line search has failed
[f0 df0] = feval(f, X, varargin{:});          % get function value and gradient
Z = X; X = unwrap(X); df0 = unwrap(df0);
%fprintf('%s %6i;  Value %4.6e\r', S, i, f0);
if exist('fflush','builtin') fflush(stdout); end
fX = f0;
i = i + (length<0);                                            % count epochs?!
s = -df0; d0 = -s'*s;           % initial search direction (steepest) and slope
x3 = red/(1-d0);                                  % initial step is red/(|s|+1)

while i < abs(length)                                      % while not finished
  i = i + (length>0);                                      % count iterations?!

  X0 = X; F0 = f0; dF0 = df0;                   % make a copy of current values
  if length>0, M = MAX; else M = min(MAX, -length-i); end

  while 1                             % keep extrapolating as long as necessary
    x2 = 0; f2 = f0; d2 = d0; f3 = f0; df3 = df0;
    success = 0;
    while ~success && M > 0
      try
        M = M - 1; i = i + (length<0);                         % count epochs?!
        
        [f3 df3] = feval(f, rewrap(Z,X+x3*s), varargin{:});
        df3 = unwrap(df3);
        if isnan(f3) || isinf(f3) || any(isnan(df3)+isinf(df3)), error(' '),end
        success = 1;
      catch                                % catch any error which occured in f
        x3 = (x2+x3)/2;                                  % bisect and try again
      end
    end
    if f3 < F0, X0 = X+x3*s; F0 = f3; dF0 = df3; end         % keep best values
    d3 = df3'*s;                                                    % new slope
    if d3 > SIG*d0 || f3 > f0+x3*RHO*d0 || M == 0  % are we done extrapolating?
      break
    end
    x1 = x2; f1 = f2; d1 = d2;                        % move point 2 to point 1
    x2 = x3; f2 = f3; d2 = d3;                        % move point 3 to point 2
    A = 6*(f1-f2)+3*(d2+d1)*(x2-x1);                 % make cubic extrapolation
    B = 3*(f2-f1)-(2*d1+d2)*(x2-x1);
    x3 = x1-d1*(x2-x1)^2/(B+sqrt(B*B-A*d1*(x2-x1))); % num. error possible, ok!
    if ~isreal(x3) || isnan(x3) || isinf(x3) || x3 < 0 % num prob | wrong sign?
      x3 = x2*EXT;                                 % extrapolate maximum amount
    elseif x3 > x2*EXT                  % new point beyond extrapolation limit?
      x3 = x2*EXT;                                 % extrapolate maximum amount
    elseif x3 < x2+INT*(x2-x1)         % new point too close to previous point?
      x3 = x2+INT*(x2-x1);
    end
  end                                                       % end extrapolation

  while (abs(d3) > -SIG*d0 || f3 > f0+x3*RHO*d0) && M > 0  % keep interpolating
    if d3 > 0 || f3 > f0+x3*RHO*d0                         % choose subinterval
      x4 = x3; f4 = f3; d4 = d3;                      % move point 3 to point 4
    else
      x2 = x3; f2 = f3; d2 = d3;                      % move point 3 to point 2
    end
    if f4 > f0           
      x3 = x2-(0.5*d2*(x4-x2)^2)/(f4-f2-d2*(x4-x2));  % quadratic interpolation
    else
      A = 6*(f2-f4)/(x4-x2)+3*(d4+d2);                    % cubic interpolation
      B = 3*(f4-f2)-(2*d2+d4)*(x4-x2);
      x3 = x2+(sqrt(B*B-A*d2*(x4-x2)^2)-B)/A;        % num. error possible, ok!
    end
    if isnan(x3) || isinf(x3)
      x3 = (x2+x4)/2;               % if we had a numerical problem then bisect
    end
    x3 = max(min(x3, x4-INT*(x4-x2)),x2+INT*(x4-x2));  % don't accept too close
    [f3 df3] = feval(f, rewrap(Z,X+x3*s), varargin{:});
    df3 = unwrap(df3);
    if f3 < F0, X0 = X+x3*s; F0 = f3; dF0 = df3; end         % keep best values
    M = M - 1; i = i + (length<0);                             % count epochs?!
    d3 = df3'*s;                                                    % new slope
  end                                                       % end interpolation

  if abs(d3) < -SIG*d0 && f3 < f0+x3*RHO*d0          % if line search succeeded
    X = X+x3*s; f0 = f3; fX = [fX' f0]';                     % update variables
    %fprintf('%s %6i;  Value %4.6e\r', S, i, f0);
    if exist('fflush','builtin') fflush(stdout); end
    s = (df3'*df3-df0'*df3)/(df0'*df0)*s - df3;   % Polack-Ribiere CG direction
    df0 = df3;                                               % swap derivatives
    d3 = d0; d0 = df0'*s;
    if d0 > 0                                      % new slope must be negative
      s = -df0; d0 = -s'*s;                  % otherwise use steepest direction
    end
    x3 = x3 * min(RATIO, d3/(d0-realmin));          % slope ratio but max RATIO
    ls_failed = 0;                              % this line search did not fail
  else
    X = X0; f0 = F0; df0 = dF0;                     % restore best point so far
    if ls_failed || i > abs(length)         % line search failed twice in a row
      break;                             % or we ran out of time, so we give up
    end
    s = -df0; d0 = -s'*s;                                        % try steepest
    x3 = 1/(1-d0);                     
    ls_failed = 1;                                    % this line search failed
  end
end
X = rewrap(Z,X); 
%fprintf('\n'); 
if exist('fflush','builtin') fflush(stdout); end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function v = unwrap(s)
% Extract the numerical values from "s" into the column vector "v". The
% variable "s" can be of any type, including struct and cell array.
% Non-numerical elements are ignored. See also the reverse rewrap.m. 
v = [];   
if isnumeric(s)
  v = s(:);                        % numeric values are recast to column vector
elseif isstruct(s)
  v = unwrap(struct2cell(orderfields(s))); % alphabetize, conv to cell, recurse
elseif iscell(s)
  for i = 1:numel(s)             % cell array elements are handled sequentially
    v = [v; unwrap(s{i})];
  end
end                                                   % other types are ignored
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [s v] = rewrap(s, v)
% Map the numerical elements in the vector "v" onto the variables "s" which can
% be of any type. The number of numerical elements must match; on exit "v"
% should be empty. Non-numerical entries are just copied. See also unwrap.m.
if isnumeric(s)
  if numel(v) < numel(s)
    error('The vector for conversion contains too few elements')
  end
  s = reshape(v(1:numel(s)), size(s));            % numeric values are reshaped
  v = v(numel(s)+1:end);                        % remaining arguments passed on
elseif isstruct(s) 
  [s p] = orderfields(s); p(p) = 1:numel(p);      % alphabetize, store ordering
  [t v] = rewrap(struct2cell(s), v);                 % convert to cell, recurse
  s = orderfields(cell2struct(t,fieldnames(s),1),p);  % conv to struct, reorder
elseif iscell(s)
  for i = 1:numel(s)             % cell array elements are handled sequentially 
    [s{i} v] = rewrap(s{i}, v);
  end
end                                             % other types are not processed
end
