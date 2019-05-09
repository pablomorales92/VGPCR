function output = VGPCR_TEST(Xts,Yts,zts,Xtr,options)
%% Reading CR Labels
[Nts ~] = size(Yts); %N data points
mask = double(Yts >= 0);

%% Reading parameters
l2 = options.K_par.l2;
gam2 = options.K_par.gam2;
B = options.B;

mean_alpha = repmat(options.q_alpha.mean,Nts,1);
mean_beta = repmat(options.q_beta.mean,Nts,1);
%% Calculating Kernel functions

Kts = gam2*kernelmatrix(options.ker,Xts,Xtr,l2);

%% Calculating Probabilities
a = Kts*(options.q_z.mean-0.5) - Kts*B*options.K*(options.q_z.mean-0.5);

if strcmp(options.ker,'rbf')
    c = gam2*ones(size(Xts,2),1);
else
    c = gam2*dot(Xts,Xts)';
end

b2 = c - dot(Kts',B*Kts')';
kap = sqrt(1./(1 + 0.125*pi*b2));
sigmoid = 1./(1 + exp(-a.*kap));

un_pz1 = sigmoid.*prod(mean_alpha.^(Yts.*mask),2) .* prod((1-mean_alpha).^((1-Yts).*mask),2);
un_pz0 = (1-sigmoid).*prod((1-mean_beta).^(Yts.*mask),2) .* prod(mean_beta.^((1-Yts).*mask),2);

probabilities = un_pz1./(un_pz1+un_pz0);

%% Calculating predictions
z_predic = probabilities >= 0.5;

%% Measuring results
res = measures(zts,z_predic);
[res.roc.X,res.roc.Y,res.roc.T,res.AUC] = perfcurve(zts,probabilities,'1');
res.ML = (sum(probabilities(zts==1))+sum(1-probabilities(zts==0)))/length(zts);
%% Preparing output
output.res = res;
output.prob = probabilities;
output.z_predic = z_predic;

output.means_noCR = a;
output.variances_noCR = b2;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function K = kernelmatrix(ker,X,X2,sigma2)

switch ker
    case 'lin'
        if exist('X2','var')
            K = X' * X2;
        else
            K = X' * X;
        end

    case 'rbf'

        n1sq = sum(X.^2,1);
        n1 = size(X,2);

        if isempty(X2)
            D = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq -2*X'*X;
        else
            n2sq = sum(X2.^2,1);
            n2 = size(X2,2);
            D = (ones(n2,1)*n1sq)' + ones(n1,1)*n2sq -2*X'*X2;
        end
        K = exp(-D/(2*sigma2));
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function results = measures(y_test,y_predic)
%% Only binary case
CM = zeros(2,2);
ind = (y_test==1);

%% Confussion Matrix

    % True Positives
        CM(1,1) = sum(y_predic(ind)==1);
    % True Negatives
        CM(2,2) = sum(y_predic(not(ind))==0);
    % False Positives
        CM(1,2) = sum(y_predic(not(ind))==1);
    % False Negatives
        CM(2,1) = sum(y_predic(ind)==0);

results.CM = CM;

%% Overall Accuracy
    results.OA = 100*(sum(diag(CM)) / sum(sum(CM)));

%% Precision and Recall
    PR(1) = CM(1,1)/(CM(1,1) + CM(1,2));
    PR(2) = CM(1,1)/(CM(1,1) + CM(2,1));
    
    if isnan(PR(1)), PR(1) =0; end
    if isnan(PR(2)), PR(2) =0; end
        
    results.Pre_Rec = PR;
%% F-score
    results.Fscore = 2*PR(1)*PR(2)/sum(PR);
    if isnan(results.Fscore), results.Fscore =0; end
%% TPR and FPR

    T_F(1) = CM(1,1)/(CM(1,1) + CM(2,1));
    T_F(2) = CM(1,2)/(CM(1,2) + CM(2,2));

    results.TF_ratio = T_F;
end
